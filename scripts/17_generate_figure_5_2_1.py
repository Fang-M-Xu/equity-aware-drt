from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pyproj import Transformer
import requests as http_requests
import yaml


METHOD_LABELS = {
    "random": "Random",
    "demand_only": "Expected demand",
    "proposed": "Gap-and-vulnerability weighted",
}

ZONE_ALIASES = [
    "origin_zone_id",
    "zone_id",
    "TAZ_1270",
    "origin_taz",
    "taz",
]
REQUEST_ID_ALIASES = ["request_id", "req_id", "id"]
PICKUP_NODE_ALIASES = ["pickup_node_id", "pickup_node", "origin_node_id"]
VEHICLE_ID_ALIASES = ["vehicle_id", "veh_id", "id"]
START_NODE_ALIASES = ["start_node_id", "vehicle_start_node_id", "start_node"]

REQUEST_LON_ALIASES = [
    "pickup_lon", "snapped_pickup_lon", "pickup_longitude",
    "origin_road_lon", "origin_snapped_lon", "origin_lon",
    "origin_longitude", "lon", "longitude",
]
REQUEST_LAT_ALIASES = [
    "pickup_lat", "snapped_pickup_lat", "pickup_latitude",
    "origin_road_lat", "origin_snapped_lat", "origin_lat",
    "origin_latitude", "lat", "latitude",
]
REQUEST_X_ALIASES = [
    "pickup_x", "snapped_pickup_x", "origin_road_x",
    "origin_snapped_x", "origin_x", "x",
]
REQUEST_Y_ALIASES = [
    "pickup_y", "snapped_pickup_y", "origin_road_y",
    "origin_snapped_y", "origin_y", "y",
]
VEHICLE_LON_ALIASES = [
    "start_lon", "vehicle_lon", "start_longitude", "lon", "longitude",
]
VEHICLE_LAT_ALIASES = [
    "start_lat", "vehicle_lat", "start_latitude", "lat", "latitude",
]
VEHICLE_X_ALIASES = ["start_x", "vehicle_x", "x"]
VEHICLE_Y_ALIASES = ["start_y", "vehicle_y", "y"]

POINT_X_ALIASES = [
    "x",
    "coord_x",
    "snapped_x",
    "road_x",
    "easting",
    "vehicle_x",
    "start_x",
]
POINT_Y_ALIASES = [
    "y",
    "coord_y",
    "snapped_y",
    "road_y",
    "northing",
    "vehicle_y",
    "start_y",
]
POINT_LON_ALIASES = [
    "lon",
    "lng",
    "longitude",
    "snapped_lon",
    "vehicle_lon",
    "start_lon",
]
POINT_LAT_ALIASES = [
    "lat",
    "latitude",
    "snapped_lat",
    "vehicle_lat",
    "start_lat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.2.1: spatial distribution of "
            "road travel time from each request origin to the "
            "nearest initially prepositioned vehicle."
        )
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_prepositioning",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "random",
            "demand_only",
            "proposed",
        ],
        default=[
            "random",
            "demand_only",
            "proposed",
        ],
    )

    parser.add_argument(
        "--minimum-zone-requests",
        type=int,
        default=5,
        help=(
            "TAZs represented by fewer than this number of "
            "request observations are hatched."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help=(
            "Fail if any requests_all observation lacks "
            "a valid Valhalla road time."
        ),
    )

    parser.add_argument(
        "--show-vehicle-points",
        action="store_true",
        help=(
            "Overlay initial vehicle locations on panels (a)-(c)."
        ),
    )

    parser.add_argument(
        "--valhalla-base-url",
        default=None,
    )

    parser.add_argument(
        "--request-coordinate-mode",
        choices=[
            "prefer-pickup",
            "building-origin",
        ],
        default="prefer-pickup",
        help=(
            "Use the snapped pickup road point when available. "
            "Otherwise use the original request origin."
        ),
    )

    return parser.parse_args()

def resolve(project_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def first_existing(columns: Iterable[object], aliases: Iterable[str]) -> str | None:
    original = {str(column): str(column) for column in columns}
    lowered = {str(column).lower(): str(column) for column in columns}
    for alias in aliases:
        if alias in original:
            return original[alias]
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def normalize_zone_id(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    nonmissing = int(text.notna().sum())
    if nonmissing and int(numeric.notna().sum()) >= max(1, int(0.8 * nonmissing)):
        return numeric.astype("Int64").astype("string")
    return text


def parse_scenario_id(path: Path) -> int:
    match = re.search(r"scenario[_-]?(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse scenario ID from {path}")
    return int(match.group(1))


def load_zones(
    project_root: Path,
    config: dict,
) -> tuple[gpd.GeoDataFrame, str, str]:
    paths = config.get("paths", {})
    project = config.get("project", {})
    zone_panel = config.get("zone_panel", {})

    zone_path = resolve(project_root, paths.get("zones"))
    if zone_path is None or not zone_path.exists():
        raise FileNotFoundError(f"TAZ layer not found: {zone_path}")

    zones = gpd.read_file(zone_path)
    zone_column = zone_panel.get("zone_id_column_in_shapefile")
    if zone_column not in zones.columns:
        zone_column = first_existing(zones.columns, ZONE_ALIASES)
    if zone_column is None:
        raise KeyError(
            f"TAZ ID column not found in {zone_path}. Columns: {zones.columns.tolist()}"
        )

    source_crs = project.get("source_crs", "EPSG:2039")
    metric_crs = project.get("metric_crs", source_crs)
    if zones.crs is None:
        zones = zones.set_crs(source_crs)
    zones = zones.to_crs(metric_crs)
    zones["_zone_key"] = normalize_zone_id(zones[zone_column])
    if zones["_zone_key"].duplicated().any():
        duplicates = zones.loc[zones["_zone_key"].duplicated(), "_zone_key"].tolist()
        raise ValueError(f"Duplicate TAZ IDs in zone layer: {duplicates[:10]}")
    return zones, str(zone_column), str(metric_crs)


def load_roads(
    project_root: Path,
    config: dict,
    target_crs: str,
) -> gpd.GeoDataFrame | None:
    roads_path = resolve(project_root, config.get("paths", {}).get("roads"))
    if roads_path is None or not roads_path.exists():
        return None
    try:
        roads = gpd.read_file(roads_path)
        if roads.crs is None:
            warnings.warn(f"Road layer has no CRS and will not be drawn: {roads_path}")
            return None
        return roads.to_crs(target_crs)
    except Exception as error:  # pragma: no cover - local GIS driver differences
        warnings.warn(f"Road layer could not be loaded: {error}")
        return None


def load_gap_zone_keys(
    project_root: Path,
    config: dict,
) -> set[str]:
    static_path = resolve(
        project_root,
        config.get("paths", {}).get(
            "zone_static_features",
            "data/processed/model_inputs/zone_static_features.parquet",
        ),
    )
    if static_path is None or not static_path.exists():
        warnings.warn("zone_static_features was not found; gap boundaries are omitted.")
        return set()

    static = pd.read_parquet(static_path)
    zone_column = first_existing(static.columns, ZONE_ALIASES)
    gap_column = first_existing(
        static.columns,
        ["zone_gap", "gap_i", "is_gap", "gap", "service_gap"],
    )
    if zone_column is None or gap_column is None:
        warnings.warn(
            "Zone/gap columns were not found in zone_static_features; "
            "gap boundaries are omitted."
        )
        return set()

    static["_zone_key"] = normalize_zone_id(static[zone_column])
    gap = pd.to_numeric(static[gap_column], errors="coerce").fillna(0)
    return set(static.loc[gap.gt(0), "_zone_key"].dropna().astype(str))


def scenario_directories(
    operating_root: Path,
    method: str,
) -> list[Path]:
    method_root = operating_root / method
    if not method_root.exists():
        raise FileNotFoundError(
            f"Operating-scenario directory not found for {method}: {method_root}"
        )
    folders = sorted(
        path
        for path in method_root.glob("scenario_*")
        if path.is_dir()
    )
    if not folders:
        raise FileNotFoundError(f"No scenario_* directories under {method_root}")
    return folders


def require_columns(
    frame: pd.DataFrame,
    aliases: list[str],
    table_name: str,
) -> str:
    column = first_existing(frame.columns, aliases)
    if column is None:
        raise KeyError(
            f"Required column {aliases} not found in {table_name}. "
            f"Available columns: {frame.columns.tolist()}"
        )
    return column


def _numeric_coordinates(
    frame: pd.DataFrame,
    lon_aliases: list[str],
    lat_aliases: list[str],
    x_aliases: list[str],
    y_aliases: list[str],
    *,
    source_crs: str,
    target_crs: str,
    table_name: str,
) -> tuple[pd.Series, pd.Series, str]:
    """Coalesce coordinates row by row in the supplied priority order."""
    lon = pd.Series(np.nan, index=frame.index, dtype=float)
    lat = pd.Series(np.nan, index=frame.index, dtype=float)
    used_sources: list[str] = []

    for lon_alias, lat_alias in zip(lon_aliases, lat_aliases):
        lon_column = first_existing(frame.columns, [lon_alias])
        lat_column = first_existing(frame.columns, [lat_alias])
        if lon_column is None or lat_column is None:
            continue
        candidate_lon = pd.to_numeric(frame[lon_column], errors="coerce")
        candidate_lat = pd.to_numeric(frame[lat_column], errors="coerce")
        fill = (
            lon.isna()
            & lat.isna()
            & candidate_lon.between(-180, 180)
            & candidate_lat.between(-90, 90)
        )
        if fill.any():
            lon.loc[fill] = candidate_lon.loc[fill]
            lat.loc[fill] = candidate_lat.loc[fill]
            used_sources.append(f"{lon_column}/{lat_column}")

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    for x_alias, y_alias in zip(x_aliases, y_aliases):
        x_column = first_existing(frame.columns, [x_alias])
        y_column = first_existing(frame.columns, [y_alias])
        if x_column is None or y_column is None:
            continue
        x = pd.to_numeric(frame[x_column], errors="coerce")
        y = pd.to_numeric(frame[y_column], errors="coerce")
        fill = lon.isna() & lat.isna() & x.notna() & y.notna()
        if not fill.any():
            continue
        transformed_lon, transformed_lat = transformer.transform(
            x.loc[fill].to_numpy(dtype=float),
            y.loc[fill].to_numpy(dtype=float),
        )
        transformed_lon = np.asarray(transformed_lon, dtype=float)
        transformed_lat = np.asarray(transformed_lat, dtype=float)
        valid_transformed = (
            np.isfinite(transformed_lon)
            & np.isfinite(transformed_lat)
            & (transformed_lon >= -180)
            & (transformed_lon <= 180)
            & (transformed_lat >= -90)
            & (transformed_lat <= 90)
        )
        fill_indices = frame.index[fill]
        valid_indices = fill_indices[valid_transformed]
        lon.loc[valid_indices] = transformed_lon[valid_transformed]
        lat.loc[valid_indices] = transformed_lat[valid_transformed]
        if valid_transformed.any():
            used_sources.append(f"{x_column}/{y_column}->{target_crs}")

    if lon.notna().sum() == 0:
        raise KeyError(
            f"No usable lon/lat or x/y coordinates were found in {table_name}. "
            f"Available columns: {frame.columns.tolist()}"
        )
    return lon, lat, " | ".join(dict.fromkeys(used_sources))


def _request_coordinate_aliases(mode: str) -> tuple[list[str], list[str], list[str], list[str]]:
    if mode == "building-origin":
        return (
            ["origin_lon", "origin_longitude"],
            ["origin_lat", "origin_latitude"],
            ["origin_x"],
            ["origin_y"],
        )
    return (
        REQUEST_LON_ALIASES,
        REQUEST_LAT_ALIASES,
        REQUEST_X_ALIASES,
        REQUEST_Y_ALIASES,
    )


def _parse_valhalla_matrix(
    payload: dict,
    n_sources: int,
    n_targets: int,
) -> np.ndarray:
    rows = payload.get("sources_to_targets")
    if not isinstance(rows, list):
        raise ValueError(
            "Valhalla response has no 'sources_to_targets' array: "
            + json.dumps(payload, ensure_ascii=False)[:1000]
        )
    matrix = np.full((n_sources, n_targets), np.inf, dtype=float)
    for source_index, row in enumerate(rows[:n_sources]):
        if not isinstance(row, list):
            continue
        for target_index, cell in enumerate(row[:n_targets]):
            if not isinstance(cell, dict):
                continue
            value = pd.to_numeric(cell.get("time"), errors="coerce")
            if pd.notna(value) and np.isfinite(float(value)) and float(value) >= 0:
                matrix[source_index, target_index] = float(value)
    return matrix


def query_valhalla_sources_to_targets(
    *,
    base_url: str,
    sources: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    costing: str,
    timeout_sec: float,
    retries: int,
    backoff_sec: float,
) -> np.ndarray:
    if not sources:
        raise ValueError("No valid vehicle source coordinates were supplied.")
    if not targets:
        raise ValueError("No valid request target coordinates were supplied.")

    endpoint = base_url.rstrip("/") + "/sources_to_targets"
    request_payload = {
        "sources": [{"lon": lon, "lat": lat} for lon, lat in sources],
        "targets": [{"lon": lon, "lat": lat} for lon, lat in targets],
        "costing": costing,
        "units": "kilometers",
    }
    last_error: Exception | None = None
    last_response_text: str | None = None
    last_status_code: int | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            response = http_requests.post(
                endpoint,
                json=request_payload,
                timeout=(10.0, timeout_sec),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
            )
            last_status_code = int(response.status_code)
            last_response_text = response.text[:2000]

            # A 4xx response is deterministic input/service-limit feedback. Retrying
            # the identical request five times only hides the useful Valhalla body.
            if 400 <= response.status_code < 500:
                raise RuntimeError(
                    "Valhalla rejected the matrix request: "
                    f"HTTP {response.status_code}; sources={len(sources)}; "
                    f"targets={len(targets)}; response={last_response_text}"
                )

            response.raise_for_status()
            return _parse_valhalla_matrix(
                response.json(), len(sources), len(targets)
            )
        except RuntimeError:
            raise
        except Exception as error:  # transient local service/network error
            last_error = error
            if attempt >= retries:
                break
            time.sleep(backoff_sec * (2**attempt))
    detail = (
        f"; last_http_status={last_status_code}; response={last_response_text}"
        if last_status_code is not None
        else ""
    )
    raise RuntimeError(
        f"Valhalla matrix request failed after {retries + 1} attempt(s): "
        f"{endpoint}{detail}"
    ) from last_error



def _valid_node_ids(values: pd.Series, matrix_size: int) -> pd.Series:
    """Return valid integer matrix-node IDs while preserving the input index."""
    numeric = pd.to_numeric(values, errors="coerce")
    rounded = numeric.round()
    valid = (
        numeric.notna()
        & np.isfinite(numeric)
        & np.isclose(numeric, rounded, atol=1e-6)
        & rounded.ge(0)
        & rounded.lt(matrix_size)
    )
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    result.loc[valid] = rounded.loc[valid].astype("int64")
    return result


def query_valhalla_in_target_batches(
    *,
    base_url: str,
    sources: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    costing: str,
    timeout_sec: float,
    retries: int,
    backoff_sec: float,
    source_batch_size: int,
    batch_size: int,
    context: str,
) -> np.ndarray:
    """Query Valhalla in both source and target batches.

    Valhalla service configurations often limit the total matrix locations in a
    request. A 25-source × 1-target request can therefore return HTTP 400 even
    when every coordinate is valid. This function keeps each request small and,
    if a batch is still rejected, falls back to 1×1 pairs. Failed pairs remain
    ``inf`` rather than aborting the entire Chapter 5.2 figure.
    """
    if not targets:
        return np.empty((len(sources), 0), dtype=float)
    if not sources:
        return np.empty((0, len(targets)), dtype=float)

    source_batch_size = max(1, int(source_batch_size))
    target_batch_size = max(1, int(batch_size))
    result = np.full((len(sources), len(targets)), np.inf, dtype=float)

    for source_start in range(0, len(sources), source_batch_size):
        source_stop = min(source_start + source_batch_size, len(sources))
        source_chunk = sources[source_start:source_stop]

        for target_start in range(0, len(targets), target_batch_size):
            target_stop = min(target_start + target_batch_size, len(targets))
            target_chunk = targets[target_start:target_stop]
            print(
                f"[{context}] Valhalla supplement sources "
                f"{source_start + 1}-{source_stop}/{len(sources)}; targets "
                f"{target_start + 1}-{target_stop}/{len(targets)}"
            )

            try:
                block = query_valhalla_sources_to_targets(
                    base_url=base_url,
                    sources=source_chunk,
                    targets=target_chunk,
                    costing=costing,
                    timeout_sec=timeout_sec,
                    retries=retries,
                    backoff_sec=backoff_sec,
                )
                result[
                    source_start:source_stop,
                    target_start:target_stop,
                ] = block
                continue
            except RuntimeError as batch_error:
                warnings.warn(
                    f"Valhalla rejected a {len(source_chunk)}×{len(target_chunk)} "
                    f"block in {context}; retrying as 1×1 pairs. Detail: "
                    f"{batch_error}"
                )

            # Pairwise fallback distinguishes service-limit errors from a truly
            # unroutable source/target. A failed pair remains infinity; any valid
            # vehicle-to-request pair is still retained for the minimum-time metric.
            for source_offset, source in enumerate(source_chunk):
                for target_offset, target in enumerate(target_chunk):
                    global_source = source_start + source_offset
                    global_target = target_start + target_offset
                    try:
                        pair = query_valhalla_sources_to_targets(
                            base_url=base_url,
                            sources=[source],
                            targets=[target],
                            costing=costing,
                            timeout_sec=timeout_sec,
                            retries=retries,
                            backoff_sec=backoff_sec,
                        )
                        result[global_source, global_target] = pair[0, 0]
                    except RuntimeError as pair_error:
                        source_lon, source_lat = source
                        target_lon, target_lat = target
                        warnings.warn(
                            f"Unroutable Valhalla pair in {context}: "
                            f"source_index={global_source}, "
                            f"source_lon={source_lon:.8f}, "
                            f"source_lat={source_lat:.8f}, "
                            f"target_index={global_target}, "
                            f"target_lon={target_lon:.8f}, "
                            f"target_lat={target_lat:.8f}; detail={pair_error}"
                        )

    return result


def evaluate_one_scenario(
    scenario_dir: Path,
    method: str,
    config: dict,
    valhalla_base_url_override: str | None,
    request_coordinate_mode: str,
) -> tuple[pd.DataFrame, dict]:
    """Use the saved matrix first and query Valhalla only for missing requests."""
    required_files = {
        "all": scenario_dir / "requests_all.parquet",
        "feasible": scenario_dir / "requests_feasible.parquet",
        "vehicles": scenario_dir / "vehicles.parquet",
    }
    matrix_path = scenario_dir / "travel_matrices.npz"
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Scenario {scenario_dir} is incomplete. Missing:\n" + "\n".join(missing)
        )

    scenario_id = parse_scenario_id(scenario_dir)
    requests_all = pd.read_parquet(required_files["all"])
    feasible = pd.read_parquet(required_files["feasible"])
    vehicles = pd.read_parquet(required_files["vehicles"])

    request_column = require_columns(
        requests_all, REQUEST_ID_ALIASES, str(required_files["all"])
    )
    zone_column = require_columns(
        requests_all, ZONE_ALIASES, str(required_files["all"])
    )
    vehicle_id_column = require_columns(
        vehicles, VEHICLE_ID_ALIASES, str(required_files["vehicles"])
    )

    project = config.get("project", {})
    routing = config.get("routing", {})
    source_crs = str(project.get("metric_crs", project.get("source_crs", "EPSG:2039")))
    target_crs = str(project.get("target_crs", "EPSG:4326"))
    valhalla_base_url = str(
        valhalla_base_url_override
        or routing.get("valhalla_base_url", "http://localhost:8002")
    )
    costing = str(routing.get("costing_auto", "auto"))
    timeout_sec = float(routing.get("timeout_sec", 90))
    retries = int(routing.get("max_retries", 4))
    backoff_sec = float(routing.get("retry_backoff_sec", 2.0))
    target_batch_size = int(routing.get("matrix_target_batch_size", 5))
    source_batch_size = int(routing.get("matrix_source_batch_size", 5))

    request_lon_aliases, request_lat_aliases, request_x_aliases, request_y_aliases = (
        _request_coordinate_aliases(request_coordinate_mode)
    )
    request_lon, request_lat, request_coordinate_source = _numeric_coordinates(
        requests_all,
        request_lon_aliases,
        request_lat_aliases,
        request_x_aliases,
        request_y_aliases,
        source_crs=source_crs,
        target_crs=target_crs,
        table_name=str(required_files["all"]),
    )
    vehicle_lon, vehicle_lat, vehicle_coordinate_source = _numeric_coordinates(
        vehicles,
        VEHICLE_LON_ALIASES,
        VEHICLE_LAT_ALIASES,
        VEHICLE_X_ALIASES,
        VEHICLE_Y_ALIASES,
        source_crs=source_crs,
        target_crs=target_crs,
        table_name=str(required_files["vehicles"]),
    )

    requests = requests_all[[request_column, zone_column]].copy()
    requests = requests.rename(
        columns={request_column: "request_id", zone_column: "origin_zone_id"}
    )
    requests["request_id"] = requests["request_id"].astype(str)
    requests["_zone_key"] = normalize_zone_id(requests["origin_zone_id"])
    requests["request_lon"] = request_lon.to_numpy()
    requests["request_lat"] = request_lat.to_numpy()
    if requests["request_id"].duplicated().any():
        raise ValueError(f"Duplicate request IDs in {required_files['all']}")

    feasible_request_column = first_existing(feasible.columns, REQUEST_ID_ALIASES)
    feasible_ids: set[str] = set()
    if feasible_request_column is not None:
        feasible_ids = set(feasible[feasible_request_column].astype(str))
    requests["directly_feasible"] = requests["request_id"].isin(feasible_ids)

    # Recover pickup-node IDs for the precomputed matrix whenever possible.
    pickup_column_all = first_existing(requests_all.columns, PICKUP_NODE_ALIASES)
    pickup_column_feasible = first_existing(feasible.columns, PICKUP_NODE_ALIASES)
    if pickup_column_all is not None:
        requests["pickup_node_id"] = requests_all[pickup_column_all].to_numpy()
    elif pickup_column_feasible is not None and feasible_request_column is not None:
        pickup_lookup = feasible[
            [feasible_request_column, pickup_column_feasible]
        ].copy()
        pickup_lookup.columns = ["request_id", "pickup_node_id"]
        pickup_lookup["request_id"] = pickup_lookup["request_id"].astype(str)
        pickup_lookup = pickup_lookup.drop_duplicates("request_id")
        requests = requests.merge(
            pickup_lookup,
            on="request_id",
            how="left",
            validate="one_to_one",
        )
    else:
        requests["pickup_node_id"] = pd.NA

    vehicle_table = pd.DataFrame(
        {
            "vehicle_id": vehicles[vehicle_id_column].astype(str),
            "vehicle_lon": vehicle_lon,
            "vehicle_lat": vehicle_lat,
        }
    )
    vehicle_valid = (
        vehicle_table["vehicle_lon"].between(-180, 180)
        & vehicle_table["vehicle_lat"].between(-90, 90)
    )
    vehicle_table = vehicle_table.loc[vehicle_valid].reset_index(drop=True)
    if vehicle_table.empty:
        raise ValueError(f"No valid vehicle coordinates in {required_files['vehicles']}")

    request_valid = (
        requests["request_lon"].between(-180, 180)
        & requests["request_lat"].between(-90, 90)
    )
    requests["nearest_vehicle_time_sec"] = np.nan
    requests["nearest_vehicle_time_min"] = np.nan
    requests["nearest_vehicle_id"] = pd.Series(
        pd.NA, index=requests.index, dtype="string"
    )
    requests["matrix_covered"] = False
    requests["road_time_source"] = pd.Series(
        pd.NA, index=requests.index, dtype="string"
    )

    precomputed_covered = 0
    # First reuse the formal experiment's saved Valhalla matrix. This avoids
    # re-querying 1,758 observations that are already available.
    if matrix_path.exists():
        try:
            matrix_data = np.load(matrix_path)
            if "time_sec" not in matrix_data.files:
                raise KeyError(f"time_sec is absent; arrays={matrix_data.files}")
            time_matrix = np.asarray(matrix_data["time_sec"], dtype=float)
            if time_matrix.ndim != 2 or time_matrix.shape[0] != time_matrix.shape[1]:
                raise ValueError(f"matrix is not square: {time_matrix.shape}")
            matrix_size = int(time_matrix.shape[0])

            start_node_column = first_existing(vehicles.columns, START_NODE_ALIASES)
            if start_node_column is not None:
                matrix_vehicle = pd.DataFrame(
                    {
                        "vehicle_id": vehicles[vehicle_id_column].astype(str),
                        "start_node_id": _valid_node_ids(
                            vehicles[start_node_column], matrix_size
                        ),
                    }
                ).dropna(subset=["start_node_id"])
                matrix_vehicle["start_node_id"] = matrix_vehicle["start_node_id"].astype(int)
                requests["pickup_node_id"] = _valid_node_ids(
                    requests["pickup_node_id"], matrix_size
                )

                start_nodes = matrix_vehicle["start_node_id"].to_numpy(dtype=int)
                matrix_vehicle_ids = matrix_vehicle["vehicle_id"].to_numpy(dtype=str)
                for row_index, pickup_node in requests["pickup_node_id"].items():
                    if pd.isna(pickup_node) or len(start_nodes) == 0:
                        continue
                    times = time_matrix[start_nodes, int(pickup_node)]
                    finite = np.isfinite(times) & (times >= 0)
                    if not finite.any():
                        continue
                    finite_indices = np.flatnonzero(finite)
                    chosen_source = int(finite_indices[int(np.argmin(times[finite]))])
                    nearest_time = float(times[chosen_source])
                    requests.at[row_index, "nearest_vehicle_time_sec"] = nearest_time
                    requests.at[row_index, "nearest_vehicle_time_min"] = nearest_time / 60.0
                    requests.at[row_index, "nearest_vehicle_id"] = matrix_vehicle_ids[chosen_source]
                    requests.at[row_index, "matrix_covered"] = True
                    requests.at[row_index, "road_time_source"] = "travel_matrices.npz"
                precomputed_covered = int(requests["matrix_covered"].sum())
        except Exception as error:
            warnings.warn(
                f"Could not reuse {matrix_path}; all valid requests will be queried "
                f"from Valhalla. Reason: {error}"
            )

    # Query Valhalla only for rows absent from the saved matrix (42 observations
    # in the user's current formal outputs), using small batches for stability.
    supplement_indices = requests.index[
        (~requests["matrix_covered"]) & request_valid
    ].tolist()
    supplemented = 0
    if supplement_indices:
        source_coordinates = list(
            zip(
                vehicle_table["vehicle_lon"].astype(float),
                vehicle_table["vehicle_lat"].astype(float),
            )
        )
        target_coordinates = [
            (
                float(requests.at[index, "request_lon"]),
                float(requests.at[index, "request_lat"]),
            )
            for index in supplement_indices
        ]
        context = f"{method} scenario_{scenario_id:03d}"
        road_matrix = query_valhalla_in_target_batches(
            base_url=valhalla_base_url,
            sources=source_coordinates,
            targets=target_coordinates,
            costing=costing,
            timeout_sec=timeout_sec,
            retries=retries,
            backoff_sec=backoff_sec,
            source_batch_size=source_batch_size,
            batch_size=target_batch_size,
            context=context,
        )
        vehicle_ids = vehicle_table["vehicle_id"].to_numpy(dtype=str)
        for target_position, row_index in enumerate(supplement_indices):
            times = road_matrix[:, target_position]
            finite = np.isfinite(times) & (times >= 0)
            if not finite.any():
                continue
            finite_indices = np.flatnonzero(finite)
            chosen_source = int(finite_indices[int(np.argmin(times[finite]))])
            nearest_time = float(times[chosen_source])
            requests.at[row_index, "nearest_vehicle_time_sec"] = nearest_time
            requests.at[row_index, "nearest_vehicle_time_min"] = nearest_time / 60.0
            requests.at[row_index, "nearest_vehicle_id"] = vehicle_ids[chosen_source]
            requests.at[row_index, "matrix_covered"] = True
            requests.at[row_index, "road_time_source"] = "valhalla_supplement"
            supplemented += 1

    requests["method"] = method
    requests["method_label"] = METHOD_LABELS[method]
    requests["scenario_id"] = scenario_id
    requests["source_directory"] = str(scenario_dir)

    covered = int(requests["matrix_covered"].sum())
    report = {
        "method": method,
        "scenario_id": scenario_id,
        "requests_all": int(len(requests)),
        "requests_feasible": int(len(feasible)),
        "directly_infeasible_requests": int(len(requests) - len(feasible_ids)),
        "valid_request_coordinates": int(request_valid.sum()),
        "valid_vehicle_starts": int(len(vehicle_table)),
        "precomputed_matrix_covered_requests": precomputed_covered,
        "valhalla_supplemented_requests": supplemented,
        "road_time_covered_requests": covered,
        "road_time_uncovered_requests": int(len(requests) - covered),
        "coverage_rate": float(covered / len(requests)) if len(requests) else None,
        "valhalla_base_url": valhalla_base_url,
        "costing": costing,
        "valhalla_source_batch_size": source_batch_size,
        "valhalla_target_batch_size": target_batch_size,
        "request_coordinate_mode": request_coordinate_mode,
        "request_coordinate_source": request_coordinate_source,
        "vehicle_coordinate_source": vehicle_coordinate_source,
        "scenario_directory": str(scenario_dir),
    }
    print(
        f"[{method} scenario_{scenario_id:03d}] covered {covered}/{len(requests)} "
        f"(saved matrix={precomputed_covered}, Valhalla supplement={supplemented})"
    )
    return requests, report

def load_all_request_times(
    operating_root: Path,
    methods: list[str],
    config: dict,
    valhalla_base_url_override: str | None,
    request_coordinate_mode: str,
) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    reports: list[dict] = []
    for method in methods:
        for scenario_dir in scenario_directories(operating_root, method):
            frame, report = evaluate_one_scenario(
                scenario_dir,
                method,
                config,
                valhalla_base_url_override,
                request_coordinate_mode,
            )
            frames.append(frame)
            reports.append(report)
    if not frames:
        raise RuntimeError("No request-level road-time observations were loaded.")
    return pd.concat(frames, ignore_index=True), reports


def verify_paired_requests(request_level: pd.DataFrame, methods: list[str]) -> None:
    base_method = methods[0]
    base = (
        request_level.loc[
            request_level["method"].eq(base_method),
            ["scenario_id", "request_id", "_zone_key"],
        ]
        .drop_duplicates()
        .sort_values(["scenario_id", "request_id"])
        .reset_index(drop=True)
    )
    for method in methods[1:]:
        current = (
            request_level.loc[
                request_level["method"].eq(method),
                ["scenario_id", "request_id", "_zone_key"],
            ]
            .drop_duplicates()
            .sort_values(["scenario_id", "request_id"])
            .reset_index(drop=True)
        )
        if not base.equals(current):
            raise ValueError(
                "The methods do not contain identical scenario/request/origin-TAZ "
                f"keys: {base_method} versus {method}. A paired spatial comparison "
                "would not be valid."
            )


def quantile_90(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(0.90)) if not clean.empty else np.nan


def summarize_by_taz(
    request_level: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zone_summary = (
        request_level.groupby(
            ["method", "method_label", "_zone_key"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_requests=("request_id", "size"),
            n_scenarios=("scenario_id", "nunique"),
            n_with_road_time=("matrix_covered", "sum"),
            mean_nearest_vehicle_time_sec=("nearest_vehicle_time_sec", "mean"),
            median_nearest_vehicle_time_sec=("nearest_vehicle_time_sec", "median"),
            p90_nearest_vehicle_time_sec=("nearest_vehicle_time_sec", quantile_90),
        )
    )
    zone_summary["coverage_rate"] = (
        zone_summary["n_with_road_time"] / zone_summary["n_requests"]
    )
    for statistic in ["mean", "median", "p90"]:
        zone_summary[f"{statistic}_nearest_vehicle_time_min"] = (
            zone_summary[f"{statistic}_nearest_vehicle_time_sec"] / 60.0
        )

    wide = request_level.pivot(
        index=["scenario_id", "request_id", "_zone_key"],
        columns="method",
        values="nearest_vehicle_time_sec",
    ).reset_index()

    paired_summary = pd.DataFrame(columns=[
        "_zone_key",
        "n_paired_requests",
        "mean_proposed_minus_expected_sec",
        "median_proposed_minus_expected_sec",
        "p90_proposed_minus_expected_sec",
        "mean_proposed_minus_expected_min",
        "median_proposed_minus_expected_min",
        "p90_proposed_minus_expected_min",
    ])
    if {"proposed", "demand_only"}.issubset(wide.columns):
        paired = wide.dropna(subset=["proposed", "demand_only"]).copy()
        paired["proposed_minus_expected_sec"] = (
            paired["proposed"] - paired["demand_only"]
        )
        paired_summary = (
            paired.groupby("_zone_key", as_index=False)
            .agg(
                n_paired_requests=("request_id", "size"),
                mean_proposed_minus_expected_sec=(
                    "proposed_minus_expected_sec",
                    "mean",
                ),
                median_proposed_minus_expected_sec=(
                    "proposed_minus_expected_sec",
                    "median",
                ),
                p90_proposed_minus_expected_sec=(
                    "proposed_minus_expected_sec",
                    quantile_90,
                ),
            )
        )
        for statistic in ["mean", "median", "p90"]:
            paired_summary[f"{statistic}_proposed_minus_expected_min"] = (
                paired_summary[f"{statistic}_proposed_minus_expected_sec"] / 60.0
            )

    return zone_summary, paired_summary


def overall_summary(request_level: pd.DataFrame) -> pd.DataFrame:
    summary = (
        request_level.groupby(["method", "method_label"], as_index=False)
        .agg(
            scenarios=("scenario_id", "nunique"),
            requests=("request_id", "size"),
            requests_with_road_time=("matrix_covered", "sum"),
            mean_nearest_vehicle_time_sec=("nearest_vehicle_time_sec", "mean"),
            median_nearest_vehicle_time_sec=("nearest_vehicle_time_sec", "median"),
            p90_nearest_vehicle_time_sec=("nearest_vehicle_time_sec", quantile_90),
        )
    )
    summary["coverage_rate"] = (
        summary["requests_with_road_time"] / summary["requests"]
    )
    for statistic in ["mean", "median", "p90"]:
        summary[f"{statistic}_nearest_vehicle_time_min"] = (
            summary[f"{statistic}_nearest_vehicle_time_sec"] / 60.0
        )
    return summary


def load_vehicle_points(
    project_root: Path,
    config: dict,
    method: str,
    metric_crs: str,
) -> gpd.GeoDataFrame | None:
    preposition_root = resolve(
        project_root,
        config.get("paths", {}).get("preposition_dir", "outputs/prepositioning"),
    )
    if preposition_root is None:
        return None
    candidates = [
        preposition_root / method / "vehicle_initial_positions.parquet",
        preposition_root / method / "vehicle_initial_positions.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    x_column = first_existing(frame.columns, POINT_X_ALIASES)
    y_column = first_existing(frame.columns, POINT_Y_ALIASES)
    lon_column = first_existing(frame.columns, POINT_LON_ALIASES)
    lat_column = first_existing(frame.columns, POINT_LAT_ALIASES)

    if x_column and y_column:
        points = gpd.GeoDataFrame(
            frame.copy(),
            geometry=gpd.points_from_xy(frame[x_column], frame[y_column]),
            crs=config.get("project", {}).get("metric_crs", metric_crs),
        )
        return points.to_crs(metric_crs)
    if lon_column and lat_column:
        points = gpd.GeoDataFrame(
            frame.copy(),
            geometry=gpd.points_from_xy(frame[lon_column], frame[lat_column]),
            crs=config.get("project", {}).get("target_crs", "EPSG:4326"),
        )
        return points.to_crs(metric_crs)
    warnings.warn(f"Vehicle point coordinates were not found in {path}")
    return None


def draw_base(
    ax: plt.Axes,
    zones: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame | None,
    gap_zone_keys: set[str],
) -> None:
    # The Valhalla road network is used for travel-time calculation only.
    # It is intentionally omitted from the manuscript map to keep the TAZ-level
    # result patterns and service-gap boundaries visually clear.
    del roads
    zones.boundary.plot(ax=ax, linewidth=0.35, alpha=0.85, zorder=3)
    if gap_zone_keys:
        gap = zones.loc[zones["_zone_key"].astype(str).isin(gap_zone_keys)]
        if not gap.empty:
            gap.boundary.plot(
                ax=ax,
                linewidth=1.0,
                linestyle=(0, (4, 2)),
                zorder=6,
            )
    ax.set_axis_off()
    ax.set_aspect("equal")


def add_gap_legend(
    ax: plt.Axes,
    show_vehicle: bool = False,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            linewidth=1.2,
            linestyle=(0, (4, 2)),
            label="Service-gap TAZ boundary",
        )
    ]

    if show_vehicle:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="C0",
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=5.5,
                label="Initial vehicle location",
            )
        )

    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        fontsize=8,
    )


def hatch_low_sample(
    ax: plt.Axes,
    frame: gpd.GeoDataFrame,
    count_column: str,
    threshold: int,
) -> None:
    count = pd.to_numeric(frame[count_column], errors="coerce").fillna(0)
    low = frame.loc[count.lt(threshold)]
    if not low.empty:
        low.plot(
            ax=ax,
            facecolor="none",
            linewidth=0,
            hatch="////",
            zorder=7,
        )


def save_figure(figure: plt.Figure, output_base: Path, dpi: int) -> None:
    """Save only publication-ready PNG and PDF files."""
    output_base.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


def add_panel_mean_annotation(
    ax: plt.Axes,
    mean_value: float,
) -> None:
    if not np.isfinite(mean_value):
        return

    ax.text(
        0.98,
        0.97,
        f"Mean = {mean_value:.2f} min",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
        zorder=20,
    )


def add_difference_summary_box(
    ax: plt.Axes,
    paired_summary: pd.DataFrame,
) -> None:
    if paired_summary.empty or "mean_proposed_minus_expected_min" not in paired_summary.columns:
        return

    values = pd.to_numeric(
        paired_summary["mean_proposed_minus_expected_min"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return

    tol = 1e-12
    reduced = int((values < -tol).sum())
    increased = int((values > tol).sum())
    unchanged = int((values.abs() <= tol).sum())

    ax.text(
        0.98,
        0.03,
        (
            f"Reduced: {reduced}\n"
            f"Increased: {increased}\n"
            f"No change: {unchanged}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.94,
        },
        zorder=20,
    )


def plot_map(
    zones: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame | None,
    gap_zone_keys: set[str],
    zone_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    vehicle_points: dict[str, gpd.GeoDataFrame | None],
    minimum_zone_requests: int,
    output_base: Path,
    dpi: int,
) -> None:
    method_frames: dict[str, gpd.GeoDataFrame] = {}
    for method in METHOD_LABELS:
        subset = zone_summary.loc[zone_summary["method"].eq(method)].copy()
        method_frames[method] = zones.merge(subset, on="_zone_key", how="left")

    values = pd.concat(
        [
            pd.to_numeric(
                method_frames[method]["mean_nearest_vehicle_time_min"],
                errors="coerce",
            )
            for method in METHOD_LABELS
        ],
        ignore_index=True,
    ).dropna()
    common_max = float(values.max()) if not values.empty else 1.0
    common_max = max(common_max, 0.1)

    difference_frame = zones.merge(paired_summary, on="_zone_key", how="left")
    differences = pd.to_numeric(
        difference_frame.get("mean_proposed_minus_expected_min"),
        errors="coerce",
    ).dropna()
    difference_abs = float(differences.abs().max()) if not differences.empty else 0.1
    difference_abs = max(difference_abs, 0.1)

    figure, axes = plt.subplots(2, 2, figsize=(11.8, 10.4))
    axes = axes.ravel()

    panel_methods = ["random", "demand_only", "proposed"]

    # Overall request-weighted mean nearest-vehicle road time for each method.
    method_means: dict[str, float] = {}
    for method in panel_methods:
        subset = zone_summary.loc[zone_summary["method"].eq(method)].copy()
        valid = subset[
            subset["mean_nearest_vehicle_time_min"].notna()
            & subset["n_with_road_time"].gt(0)
        ]
        if valid.empty:
            method_means[method] = np.nan
        else:
            method_means[method] = float(
                np.average(
                    valid["mean_nearest_vehicle_time_min"].to_numpy(dtype=float),
                    weights=valid["n_with_road_time"].to_numpy(dtype=float),
                )
            )

    for panel_index, method in enumerate(panel_methods):
        ax = axes[panel_index]
        frame = method_frames[method]
        frame.plot(
            ax=ax,
            column="mean_nearest_vehicle_time_min",
            vmin=0,
            vmax=common_max,
            legend=True,
            linewidth=0,
            missing_kwds={"label": "No request observation"},
            legend_kwds={
                "label": "Mean time to nearest initial vehicle (min)",
                "shrink": 0.72,
            },
            zorder=1,
        )
        draw_base(ax, zones, roads, gap_zone_keys)
        hatch_low_sample(
            ax,
            frame,
            "n_requests",
            minimum_zone_requests,
        )
        points = vehicle_points.get(method)
        if points is not None and not points.empty:
            points.plot(
                ax=ax,
                marker="o",
                edgecolor="white",
                linewidth=0.45,
                markersize=24,
                zorder=8,
            )
        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            f"({panel_letter}) {METHOD_LABELS[method]}",
            loc="left",
            fontsize=11.2,
            fontweight="semibold",
        )
        add_panel_mean_annotation(
            ax,
            float(method_means.get(method, np.nan)),
        )
        add_gap_legend(
            ax,
            show_vehicle=(points is not None and not points.empty),
        )

    ax = axes[3]
    difference_frame.plot(
        ax=ax,
        column="mean_proposed_minus_expected_min",
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-difference_abs,
            vcenter=0.0,
            vmax=difference_abs,
        ),
        legend=True,
        linewidth=0,
        missing_kwds={"label": "No paired request observation"},
        legend_kwds={
            "label": "Proposed − expected-demand road time (min)",
            "shrink": 0.72,
        },
        zorder=1,
    )
    draw_base(ax, zones, roads, gap_zone_keys)
    hatch_low_sample(
        ax,
        difference_frame,
        "n_paired_requests",
        minimum_zone_requests,
    )
    ax.set_title(
        "(d) Proposed − expected-demand difference",
        loc="left",
        fontsize=11.2,
        fontweight="semibold",
    )
    add_difference_summary_box(
        ax,
        paired_summary,
    )
    add_gap_legend(
        ax,
        show_vehicle=False,
    )

    save_figure(figure, output_base, dpi)


def main() -> None:
    args = parse_args()
    required_methods = set(METHOD_LABELS)
    if set(args.methods) != required_methods:
        raise ValueError(
            "The four-panel Chapter 5.2 figure requires all three methods: "
            "random, demand_only, and proposed."
        )
    project_root = Path(args.project_root).expanduser().resolve()
    config_path = resolve(project_root, args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = load_yaml(config_path)

    output_dir = resolve(project_root, args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    operating_root = resolve(
        project_root,
        config.get("paths", {}).get(
            "operating_dir",
            "outputs/operating_scenarios",
        ),
    )
    if operating_root is None or not operating_root.exists():
        raise FileNotFoundError(f"Operating-scenario root not found: {operating_root}")

    zones, _, metric_crs = load_zones(project_root, config)

    # Do not load or draw the road layer. Valhalla is still used above for the
    # request-origin-to-vehicle road-time calculation.
    roads = None

    gap_zone_keys = load_gap_zone_keys(project_root, config)

    request_level, _scenario_reports = load_all_request_times(
        operating_root,
        args.methods,
        config,
        args.valhalla_base_url,
        args.request_coordinate_mode,
    )
    verify_paired_requests(request_level, args.methods)

    missing_times = int((~request_level["matrix_covered"]).sum())
    if missing_times:
        message = (
            f"{missing_times} of {len(request_level)} method-request observations "
            "lack a valid Valhalla road time from an initial vehicle to the saved "
            "request-origin coordinate. Check coordinate completeness and Valhalla "
            "network connectivity; no Euclidean fallback is applied."
        )
        if args.require_complete_coverage:
            raise RuntimeError(message)
        warnings.warn(message)

    zone_summary, paired_summary = summarize_by_taz(request_level)
    method_summary = overall_summary(request_level)

    # --------------------------------------------------------
    # Save only essential publication outputs
    # --------------------------------------------------------

    output_base = output_dir / "Figure_5_2_1_nearest_vehicle_road_time"

    taz_summary_path = output_dir / "Figure_5_2_1_TAZ_summary.csv"
    paired_difference_path = output_dir / "Figure_5_2_1_paired_difference.csv"
    method_summary_path = output_dir / "Figure_5_2_1_method_summary.csv"

    zone_summary.to_csv(
        taz_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    paired_summary.to_csv(
        paired_difference_path,
        index=False,
        encoding="utf-8-sig",
    )

    method_summary.to_csv(
        method_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    points: dict[str, gpd.GeoDataFrame | None] = {
        method: None for method in METHOD_LABELS
    }
    if args.show_vehicle_points:
        points = {
            method: load_vehicle_points(
                project_root,
                config,
                method,
                metric_crs,
            )
            for method in METHOD_LABELS
        }

    plot_map(
        zones=zones,
        roads=roads,
        gap_zone_keys=gap_zone_keys,
        zone_summary=zone_summary,
        paired_summary=paired_summary,
        vehicle_points=points,
        minimum_zone_requests=args.minimum_zone_requests,
        output_base=output_base,
        dpi=args.dpi,
    )

    print()
    print("=" * 78)
    print("Figure 5.2.1 generated successfully")
    print("=" * 78)
    print(f"Output directory : {output_dir}")
    print(f"Figure PNG       : {output_base.with_suffix('.png')}")
    print(f"Figure PDF       : {output_base.with_suffix('.pdf')}")
    print(f"TAZ summary      : {taz_summary_path}")
    print(f"Paired difference: {paired_difference_path}")
    print(f"Method summary   : {method_summary_path}")
    print()
    print(method_summary.to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
