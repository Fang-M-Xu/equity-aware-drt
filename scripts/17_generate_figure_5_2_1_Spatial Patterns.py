from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"

from config import load_config, path_from_config


METHODS = {
    "random": "(a) Random",
    "demand_only": "(b) Expected-demand",
    "proposed": "(c) Equity-weighted",
}


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

GAP_ALIASES = [
    "gap_i",
    "is_gap",
    "gap",
    "zone_gap",
    "service_gap",
]

ZONE_ID_ALIASES = [
    "TAZ_1270",
    "taz_1270",
    "zone_id",
    "origin_zone_id",
    "taz_id",
]


def first_existing(
    columns: Iterable[object],
    aliases: Iterable[str],
) -> str | None:

    original = {
        str(c): str(c)
        for c in columns
    }

    lowered = {
        str(c).lower(): str(c)
        for c in columns
    }

    for alias in aliases:
        if alias in original:
            return original[alias]

        if alias.lower() in lowered:
            return lowered[alias.lower()]

    return None


def normalize_zone_id(series: pd.Series) -> pd.Series:

    text = (
        series
        .astype("string")
        .str.strip()
    )

    numeric = pd.to_numeric(
        text,
        errors="coerce",
    )

    if numeric.notna().sum() >= max(
        1,
        int(0.8 * text.notna().sum()),
    ):
        return (
            numeric
            .astype("Int64")
            .astype("string")
        )

    return text


def load_vehicle_points(
    path: Path,
    metric_crs: str,
    source_metric_crs: str,
    geographic_crs: str,
) -> gpd.GeoDataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Vehicle prepositioning output not found:\n{path}"
        )

    frame = pd.read_parquet(path)

    x_col = first_existing(
        frame.columns,
        POINT_X_ALIASES,
    )

    y_col = first_existing(
        frame.columns,
        POINT_Y_ALIASES,
    )

    lon_col = first_existing(
        frame.columns,
        POINT_LON_ALIASES,
    )

    lat_col = first_existing(
        frame.columns,
        POINT_LAT_ALIASES,
    )

    if x_col and y_col:

        frame[x_col] = pd.to_numeric(
            frame[x_col],
            errors="coerce",
        )

        frame[y_col] = pd.to_numeric(
            frame[y_col],
            errors="coerce",
        )

        frame = frame.dropna(
            subset=[x_col, y_col]
        ).copy()

        points = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(
                frame[x_col],
                frame[y_col],
            ),
            crs=source_metric_crs,
        )

    elif lon_col and lat_col:

        frame[lon_col] = pd.to_numeric(
            frame[lon_col],
            errors="coerce",
        )

        frame[lat_col] = pd.to_numeric(
            frame[lat_col],
            errors="coerce",
        )

        frame = frame.dropna(
            subset=[lon_col, lat_col]
        ).copy()

        points = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(
                frame[lon_col],
                frame[lat_col],
            ),
            crs=geographic_crs,
        )

    else:

        raise KeyError(
            f"No usable vehicle coordinates found in:\n"
            f"{path}\n\n"
            f"Available columns:\n"
            f"{frame.columns.tolist()}"
        )

    return points.to_crs(metric_crs)


def aggregate_vehicle_locations(
    vehicles: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Merge vehicles occupying exactly the same prepositioning point.

    Marker size in the final map represents the number of vehicles
    assigned to that location.
    """

    vehicles = vehicles.copy()

    vehicles["_x"] = vehicles.geometry.x.round(3)
    vehicles["_y"] = vehicles.geometry.y.round(3)

    counts = (
        vehicles
        .groupby(
            ["_x", "_y"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "vehicle_count",
            }
        )
    )

    return gpd.GeoDataFrame(
        counts,
        geometry=gpd.points_from_xy(
            counts["_x"],
            counts["_y"],
        ),
        crs=vehicles.crs,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.2.1: spatial distribution "
            "of vehicles under alternative prepositioning strategies."
        )
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
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # ========================================================
    # 1. Formal spatial inputs
    # ========================================================

    zones_path = path_from_config(
        config,
        "zones",
    )

    preposition_dir = path_from_config(
        config,
        "preposition_dir",
    )

    zone_static_path = path_from_config(
        config,
        "zone_static_features",
    )

    metric_crs = str(
        config["project"]["metric_crs"]
    )

    geographic_crs = str(
        config["project"].get(
            "target_crs",
            "EPSG:4326",
        )
    )

    zone_id_config = str(
        config["zone_panel"][
            "zone_id_column_in_shapefile"
        ]
    )

    # ========================================================
    # 2. Load Jerusalem TAZ
    # ========================================================

    zones = gpd.read_file(
        zones_path
    )

    if zones.crs is None:
        raise ValueError(
            f"TAZ shapefile has no CRS: {zones_path}"
        )

    zones = zones.to_crs(
        metric_crs
    )

    if zone_id_config in zones.columns:
        zone_id_col = zone_id_config
    else:
        zone_id_col = first_existing(
            zones.columns,
            ZONE_ID_ALIASES,
        )

    if zone_id_col is None:
        raise KeyError(
            "Could not identify TAZ ID column.\n"
            f"Available columns:\n{zones.columns.tolist()}"
        )

    zones["_zone_key"] = normalize_zone_id(
        zones[zone_id_col]
    )

    # ========================================================
    # 3. Identify the 17 service-gap TAZ
    #
    # Prefer zone_static_features because gap_i is part of the
    # formal experimental input.
    # ========================================================

    zone_static = pd.read_parquet(
        zone_static_path
    )

    gap_col = first_existing(
        zone_static.columns,
        GAP_ALIASES,
    )

    static_zone_col = first_existing(
        zone_static.columns,
        [
            "origin_zone_id",
            "zone_id",
            "TAZ_1270",
            "taz_1270",
            "taz_id",
        ],
    )

    if gap_col is None:
        raise KeyError(
            "Could not find service-gap field in "
            f"{zone_static_path}\n"
            f"Available columns:\n"
            f"{zone_static.columns.tolist()}"
        )

    if static_zone_col is None:
        raise KeyError(
            "Could not find TAZ identifier in "
            f"{zone_static_path}"
        )

    zone_static["_zone_key"] = normalize_zone_id(
        zone_static[static_zone_col]
    )

    gap_values = pd.to_numeric(
        zone_static[gap_col],
        errors="coerce",
    ).fillna(0)

    gap_zone_keys = set(
        zone_static.loc[
            gap_values.eq(1),
            "_zone_key",
        ].dropna()
    )

    print(
        f"Service-gap TAZs detected: "
        f"{len(gap_zone_keys)}"
    )

    # ========================================================
    # 4. Load all three formal prepositioning outputs
    # ========================================================

    vehicle_data = {}

    for method in METHODS:

        path = (
            preposition_dir
            / method
            / "vehicle_initial_positions.parquet"
        )

        points = load_vehicle_points(
            path=path,
            metric_crs=metric_crs,
            source_metric_crs=metric_crs,
            geographic_crs=geographic_crs,
        )

        vehicle_data[method] = (
            aggregate_vehicle_locations(
                points
            )
        )

        print(
            f"{method:<12}: "
            f"{len(points)} vehicles, "
            f"{len(vehicle_data[method])} "
            f"unique locations"
        )

    # ========================================================
    # 5. Consistent Jerusalem map extent
    # ========================================================

    minx, miny, maxx, maxy = (
        zones.total_bounds
    )

    x_margin = (
        maxx - minx
    ) * 0.025

    y_margin = (
        maxy - miny
    ) * 0.025

    # ========================================================
    # 6. Create three-panel publication figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 7.2),
    )

    for ax, method in zip(
        axes,
        METHODS,
    ):

        # ----------------------------------------------------
        # All 90 Jerusalem TAZ
        # ----------------------------------------------------

        zones.plot(
            ax=ax,
            facecolor="0.96",
            edgecolor="0.55",
            linewidth=0.45,
            zorder=1,
        )

        # ----------------------------------------------------
        # Service-gap TAZ
        # ----------------------------------------------------

        gap_zones = zones[
            zones["_zone_key"].isin(
                gap_zone_keys
            )
        ]

        if not gap_zones.empty:

            gap_zones.plot(
                ax=ax,
                facecolor="0.82",
                edgecolor="0.20",
                linewidth=0.9,
                hatch="///",
                zorder=2,
            )

        # ----------------------------------------------------
        # Vehicle locations
        # ----------------------------------------------------

        points = vehicle_data[
            method
        ]

        # marker area increases with number of co-located cars
        marker_sizes = (
            32
            + 32
            * (
                points[
                    "vehicle_count"
                ] - 1
            )
        )

        points.plot(
            ax=ax,
            markersize=marker_sizes,
            marker="o",
            edgecolor="black",
            linewidth=0.55,
            alpha=0.90,
            zorder=5,
        )

        # ----------------------------------------------------
        # Add number when >1 vehicle occupies same point
        # ----------------------------------------------------

        multiple = points[
            points[
                "vehicle_count"
            ] > 1
        ]

        for _, row in multiple.iterrows():

            ax.text(
                row.geometry.x,
                row.geometry.y,
                str(
                    int(
                        row[
                            "vehicle_count"
                        ]
                    )
                ),
                ha="center",
                va="center",
                fontsize=7,
                zorder=6,
            )

        # ----------------------------------------------------
        # Panel title
        # ----------------------------------------------------

        ax.set_title(
            METHODS[method],
            fontsize=12,
            pad=8,
        )

        # ----------------------------------------------------
        # Same geographic extent for all panels
        # ----------------------------------------------------

        ax.set_xlim(
            minx - x_margin,
            maxx + x_margin,
        )

        ax.set_ylim(
            miny - y_margin,
            maxy + y_margin,
        )

        ax.set_aspect(
            "equal"
        )

        ax.set_axis_off()

    # ========================================================
    # 7. Shared legend
    # ========================================================

    legend_handles = [
        Patch(
            facecolor="0.96",
            edgecolor="0.55",
            label="Jerusalem TAZ",
        ),
        Patch(
            facecolor="0.82",
            edgecolor="0.20",
            hatch="///",
            label="Service-gap TAZ",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markeredgecolor="black",
            markersize=7,
            label="Vehicle location",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(
            0.5,
            0.015,
        ),
    )

    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.95,
        bottom=0.09,
        wspace=0.04,
    )

    # ========================================================
    # 8. Output
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / "Figure_5_2_1_vehicle_prepositioning_spatial_distribution.png"
    )

    pdf_path = (
        output_dir
        / "Figure_5_2_1_vehicle_prepositioning_spatial_distribution.pdf"
    )

    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # 9. Save exact plotting data for reproducibility
    # ========================================================

    rows = []

    for method, points in vehicle_data.items():

        for _, row in points.iterrows():

            rows.append(
                {
                    "method": method,
                    "method_label": METHODS[
                        method
                    ],
                    "x": row.geometry.x,
                    "y": row.geometry.y,
                    "vehicle_count": int(
                        row[
                            "vehicle_count"
                        ]
                    ),
                }
            )

    plot_data = pd.DataFrame(
        rows
    )

    plot_data_path = (
        output_dir
        / "Figure_5_2_1_vehicle_locations.csv"
    )

    plot_data.to_csv(
        plot_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 10. Audit
    # ========================================================

    print()
    print("=" * 72)
    print(
        "Figure 5.2.1 — Spatial distribution of vehicles "
        "under alternative prepositioning strategies"
    )
    print("=" * 72)

    print(
        f"TAZ file       : {zones_path}"
    )

    print(
        f"TAZ count      : {len(zones)}"
    )

    print(
        f"Gap TAZ count  : {len(gap_zone_keys)}"
    )

    for method in METHODS:

        points = vehicle_data[
            method
        ]

        print(
            f"{METHODS[method]:<25} "
            f"vehicles={points['vehicle_count'].sum():.0f}, "
            f"locations={len(points)}"
        )

    print()
    print(
        f"PNG saved      : {png_path}"
    )

    print(
        f"PDF saved      : {pdf_path}"
    )

    print(
        f"Plot data      : {plot_data_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()