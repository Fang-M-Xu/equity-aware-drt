from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString

LOGGER = logging.getLogger(__name__)

ROAD_CLASS_COLUMNS = (
    "highway",
    "road_class",
    "fclass",
    "type",
    "class",
)


def _iter_lines(geometry) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_lines(part)


def _normalise_class(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    return str(value).strip().lower()


def _detect_road_class_column(
    roads: gpd.GeoDataFrame,
    configured_column: str | None,
) -> str | None:
    if configured_column:
        if configured_column not in roads.columns:
            raise ValueError(
                f"Configured road-class column '{configured_column}' was not found. "
                f"Available columns: {list(roads.columns)}"
            )
        return configured_column

    return next(
        (column for column in ROAD_CLASS_COLUMNS if column in roads.columns),
        None,
    )


def _filter_road_classes(
    roads: gpd.GeoDataFrame,
    class_column: str | None,
    allowed_road_classes: list[str] | None,
    excluded_road_classes: list[str] | None,
) -> gpd.GeoDataFrame:
    output = roads.copy()
    if class_column is None:
        if allowed_road_classes or excluded_road_classes:
            LOGGER.warning(
                "No road-class column was found. Road-class filtering was skipped."
            )
        output["candidate_road_class"] = "unknown"
        return output

    output["candidate_road_class"] = output[class_column].map(_normalise_class)
    allowed = {_normalise_class(value) for value in (allowed_road_classes or [])}
    excluded = {_normalise_class(value) for value in (excluded_road_classes or [])}

    if allowed:
        matched = output["candidate_road_class"].isin(allowed)
        if matched.any():
            output = output.loc[matched].copy()
        else:
            LOGGER.warning(
                "None of the configured allowed road classes matched column '%s'. "
                "The allowed-class filter was skipped. Observed examples: %s",
                class_column,
                sorted(output["candidate_road_class"].dropna().unique())[:25],
            )

    if excluded:
        output = output.loc[
            ~output["candidate_road_class"].isin(excluded)
        ].copy()

    if output.empty:
        raise ValueError("No roads remain after road-class filtering.")
    return output


def _sample_line(line: LineString, spacing_m: float) -> list[tuple[float, object]]:
    """Sample interior road points without generating both endpoints of every edge.

    Short road segments receive one midpoint. Longer segments are sampled at half-spacing,
    then every ``spacing_m`` metres. This avoids the very dense duplicate endpoint cloud
    produced when every short directed edge contributes both endpoints.
    """
    length = float(line.length)
    if not np.isfinite(length) or length <= 0:
        return []

    if length <= spacing_m:
        distances = np.array([0.5 * length], dtype=float)
    else:
        distances = np.arange(0.5 * spacing_m, length, spacing_m, dtype=float)
        if len(distances) == 0:
            distances = np.array([0.5 * length], dtype=float)

    return [
        (float(distance), line.interpolate(float(distance)))
        for distance in distances
    ]


def _deduplicate_on_grid(
    candidates: gpd.GeoDataFrame,
    grid_m: float,
) -> gpd.GeoDataFrame:
    if grid_m <= 0:
        raise ValueError("candidate_dedup_grid_m must be positive.")

    output = candidates.copy()
    # floor gives stable non-overlapping metric cells and avoids banker's rounding.
    output["_grid_x"] = np.floor(output.geometry.x / grid_m).astype(np.int64)
    output["_grid_y"] = np.floor(output.geometry.y / grid_m).astype(np.int64)
    output = output.drop_duplicates(["_grid_x", "_grid_y"], keep="first")
    return output.drop(columns=["_grid_x", "_grid_y"]).reset_index(drop=True)


def _farthest_point_indices(
    coordinates: np.ndarray,
    number_to_keep: int,
) -> np.ndarray:
    """Choose spatially distributed points deterministically within one TAZ."""
    n_points = len(coordinates)
    if number_to_keep >= n_points:
        return np.arange(n_points, dtype=int)
    if number_to_keep <= 0:
        return np.empty(0, dtype=int)

    centroid = coordinates.mean(axis=0)
    first = int(np.argmin(np.sum((coordinates - centroid) ** 2, axis=1)))
    selected = [first]
    minimum_squared_distance = np.sum(
        (coordinates - coordinates[first]) ** 2,
        axis=1,
    )

    while len(selected) < number_to_keep:
        next_index = int(np.argmax(minimum_squared_distance))
        selected.append(next_index)
        new_squared_distance = np.sum(
            (coordinates - coordinates[next_index]) ** 2,
            axis=1,
        )
        minimum_squared_distance = np.minimum(
            minimum_squared_distance,
            new_squared_distance,
        )
        minimum_squared_distance[selected] = -1.0

    return np.asarray(selected, dtype=int)


def _cap_candidates_per_zone(
    candidates: gpd.GeoDataFrame,
    zone_id_column: str,
    max_candidates_per_zone: int,
) -> gpd.GeoDataFrame:
    if max_candidates_per_zone <= 0:
        return candidates.reset_index(drop=True)

    selected_groups: list[gpd.GeoDataFrame] = []
    for _, group in candidates.groupby(zone_id_column, sort=True, dropna=False):
        group = group.reset_index(drop=True)
        coordinates = np.column_stack((group.geometry.x, group.geometry.y))
        keep = _farthest_point_indices(coordinates, max_candidates_per_zone)
        selected_groups.append(group.iloc[keep].copy())

    if not selected_groups:
        return candidates.iloc[0:0].copy()

    return gpd.GeoDataFrame(
        pd.concat(selected_groups, ignore_index=True),
        geometry="geometry",
        crs=candidates.crs,
    )


def generate_road_candidates(
    roads_path: Path,
    zones_path: Path,
    zone_id_column: str,
    metric_crs: str,
    target_crs: str,
    spacing_m: float,
    dedup_grid_m: float,
    max_candidates_per_zone: int,
    active_zone_ids: list[object] | None = None,
    allowed_road_classes: list[str] | None = None,
    excluded_road_classes: list[str] | None = None,
    road_class_column: str | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, int | float | str | None]]:
    if spacing_m <= 0:
        raise ValueError("candidate_spacing_m must be positive.")

    roads = gpd.read_file(roads_path)
    zones = gpd.read_file(zones_path)
    if roads.crs is None:
        raise ValueError("Road layer has no CRS.")
    if zones.crs is None:
        raise ValueError("TAZ layer has no CRS.")
    if zone_id_column not in zones.columns:
        raise ValueError(
            f"TAZ layer does not contain '{zone_id_column}'. "
            f"Available columns: {list(zones.columns)}"
        )

    roads = roads.loc[
        roads.geometry.notna() & ~roads.geometry.is_empty
    ].to_crs(metric_crs).copy()
    zones = zones.loc[
        zones.geometry.notna() & ~zones.geometry.is_empty,
        [zone_id_column, "geometry"],
    ].to_crs(metric_crs).copy()

    if active_zone_ids:
        active_keys = {str(value) for value in active_zone_ids if pd.notna(value)}
        zones = zones.loc[
            zones[zone_id_column].astype(str).isin(active_keys)
        ].copy()
        if zones.empty:
            raise ValueError(
                "No TAZ polygons matched the zone IDs from the 10:30 forecast."
            )

    initial_road_count = len(roads)
    class_column = _detect_road_class_column(roads, road_class_column)
    roads = _filter_road_classes(
        roads,
        class_column,
        allowed_road_classes,
        excluded_road_classes,
    )
    filtered_road_count = len(roads)

    # Clip to the experiment TAZs before sampling. This prevents points outside the
    # Jerusalem study area from entering the expensive Valhalla matrix calculation.
    try:
        study_geometry = zones.geometry.union_all()
    except AttributeError:  # GeoPandas < 1.0
        study_geometry = zones.geometry.unary_union
    roads = roads.loc[roads.geometry.intersects(study_geometry)].copy()
    roads["geometry"] = roads.geometry.intersection(study_geometry)
    roads = roads.loc[
        roads.geometry.notna() & ~roads.geometry.is_empty
    ].copy()

    records: list[dict] = []
    for road_index, row in roads.iterrows():
        for line_index, line in enumerate(_iter_lines(row.geometry)):
            for distance, point in _sample_line(line, spacing_m):
                records.append(
                    {
                        "road_index": str(road_index),
                        "line_index": int(line_index),
                        "distance_along_road_m": float(distance),
                        "road_class": row.get("candidate_road_class", "unknown"),
                        "geometry": point,
                    }
                )

    if not records:
        raise ValueError("No road candidates were generated after clipping and filtering.")

    raw_candidates = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=metric_crs,
    )
    sampled_count = len(raw_candidates)
    candidates = _deduplicate_on_grid(raw_candidates, dedup_grid_m)
    deduplicated_count = len(candidates)

    # Points are already on clipped road geometry; intersects also retains points lying
    # exactly on a TAZ boundary. Duplicate boundary matches are resolved deterministically.
    candidates["_candidate_row_id"] = np.arange(len(candidates), dtype=np.int64)
    candidates = gpd.sjoin(
        candidates,
        zones,
        how="inner",
        predicate="intersects",
    ).drop(columns=["index_right"], errors="ignore")
    candidates = candidates.sort_values(
        [zone_id_column, "road_index", "line_index", "distance_along_road_m"]
    ).drop_duplicates(
        subset=["_candidate_row_id"],
        keep="first",
    ).drop(columns=["_candidate_row_id"], errors="ignore")
    joined_count = len(candidates)

    candidates = _cap_candidates_per_zone(
        candidates,
        zone_id_column=zone_id_column,
        max_candidates_per_zone=max_candidates_per_zone,
    )
    final_count = len(candidates)
    if final_count == 0:
        raise ValueError("No waiting candidates remain after per-zone capping.")

    candidates = candidates.sort_values(
        [zone_id_column, "road_index", "line_index", "distance_along_road_m"]
    ).reset_index(drop=True)
    candidates = candidates.rename(columns={zone_id_column: "zone_id"})
    candidates["candidate_id"] = [
        f"C_{index:05d}" for index in range(final_count)
    ]
    candidates["x"] = candidates.geometry.x
    candidates["y"] = candidates.geometry.y

    wgs84 = candidates.to_crs(target_crs)
    candidates["longitude"] = wgs84.geometry.x
    candidates["latitude"] = wgs84.geometry.y

    zone_counts = candidates.groupby("zone_id").size()
    audit: dict[str, int | float | str | None] = {
        "road_class_column": class_column,
        "initial_road_records": int(initial_road_count),
        "road_records_after_class_filter": int(filtered_road_count),
        "sampled_candidates_before_dedup": int(sampled_count),
        "candidates_after_grid_dedup": int(deduplicated_count),
        "candidates_inside_active_taz": int(joined_count),
        "final_candidates": int(final_count),
        "zones_with_candidates": int(zone_counts.size),
        "minimum_candidates_per_zone": int(zone_counts.min()),
        "maximum_candidates_per_zone": int(zone_counts.max()),
        "mean_candidates_per_zone": float(zone_counts.mean()),
        "spacing_m": float(spacing_m),
        "dedup_grid_m": float(dedup_grid_m),
        "max_candidates_per_zone": int(max_candidates_per_zone),
    }
    return candidates, audit


def save_candidates(candidates: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(candidates.drop(columns="geometry"))
    frame.to_parquet(path, index=False)

    gpkg_path = path.with_suffix(".gpkg")
    if gpkg_path.exists():
        gpkg_path.unlink()
    candidates.to_file(
        gpkg_path,
        layer="waiting_candidates",
        driver="GPKG",
    )