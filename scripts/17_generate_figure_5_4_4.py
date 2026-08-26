from __future__ import annotations

import argparse
import re

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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


METHOD_DIR = "proposed_full"


# ============================================================
# Helpers
# ============================================================

def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(series, errors="coerce")
            .fillna(0)
            .ne(0)
        )

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "1",
                "true",
                "yes",
                "y",
            }
        )
    )


def extract_scenario_id(path: Path) -> int:
    match = re.search(
        r"scenario[_-]?0*(\d+)",
        str(path),
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot extract scenario_id from:\n{path}"
        )

    return int(match.group(1))


def extract_solver_seed(path: Path) -> int:
    match = re.search(
        r"seed[_-]?(\d+)",
        str(path),
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot extract solver seed from:\n{path}"
        )

    return int(match.group(1))


def normalize_zone_id(series: pd.Series) -> pd.Series:
    """
    Normalize zone identifiers so that values such as:
        12
        12.0
        "12"
    all become the same key.
    """

    raw = (
        series
        .astype("string")
        .str.strip()
    )

    numeric = pd.to_numeric(
        raw,
        errors="coerce",
    )

    valid_raw = raw.notna().sum()
    valid_numeric = numeric.notna().sum()

    if (
        valid_raw > 0
        and valid_numeric >= 0.8 * valid_raw
    ):
        return (
            numeric
            .round()
            .astype("Int64")
            .astype("string")
        )

    return raw


# ============================================================
# Benchmark loader
# ============================================================

def load_benchmarks(
    benchmark_dir: Path,
) -> pd.DataFrame:

    files = sorted(
        benchmark_dir.glob(
            "scenario_*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No benchmark scenario files found in:\n"
            f"{benchmark_dir}"
        )

    frames = []

    for path in files:
        frame = pd.read_parquet(
            path
        ).copy()

        frame["scenario_id"] = (
            extract_scenario_id(path)
        )

        frames.append(frame)

    data = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required = [
        "scenario_id",
        "request_id",
        "walking_within_15min",
        "transit_within_15min",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "Benchmark files are missing columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{data.columns.tolist()}"
        )

    data["walking_within_15min"] = to_bool(
        data["walking_within_15min"]
    )

    data["transit_within_15min"] = to_bool(
        data["transit_within_15min"]
    )

    duplicated = data.duplicated(
        subset=[
            "scenario_id",
            "request_id",
        ],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            "Benchmark results contain duplicated "
            "scenario_id + request_id pairs."
        )

    return data


# ============================================================
# Proposed-full DRT loader
# ============================================================

def load_drt(
    alns_dir: Path,
) -> pd.DataFrame:

    method_dir = (
        alns_dir
        / METHOD_DIR
    )

    files = sorted(
        method_dir.glob(
            "scenario_*/seed_*/request_results.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No Proposed-full request_results.parquet found under:\n"
            f"{method_dir}"
        )

    frames = []

    for path in files:

        frame = pd.read_parquet(
            path
        ).copy()

        frame["scenario_id"] = (
            extract_scenario_id(path)
        )

        frame["solver_seed"] = (
            extract_solver_seed(path)
        )

        frames.append(frame)

    data = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required = [
        "scenario_id",
        "solver_seed",
        "request_id",
        "origin_zone_id",
        "served",
        "within_15min",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "DRT request results are missing columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{data.columns.tolist()}"
        )

    data["served"] = to_bool(
        data["served"]
    )

    data["within_15min"] = to_bool(
        data["within_15min"]
    )

    # DRT 15-min accessibility
    data["drt_accessible_15min"] = (
        data["served"]
        & data["within_15min"]
    )

    data["_zone_key"] = normalize_zone_id(
        data["origin_zone_id"]
    )

    duplicated = data.duplicated(
        subset=[
            "scenario_id",
            "solver_seed",
            "request_id",
        ],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            "DRT results contain duplicated "
            "scenario_id + solver_seed + request_id."
        )

    return data


# ============================================================
# Create request-level multimodal base
# ============================================================

def build_multimodal_request_data(
    benchmark: pd.DataFrame,
    drt: pd.DataFrame,
) -> pd.DataFrame:
    """
    The DRT result contains repeated solver seeds.

    First calculate the mean DRT accessibility of each unique
    scenario + request across solver seeds.

    A value such as 0.8 means that this request was DRT-accessible
    within 15 minutes in 80% of solver-seed runs.
    """

    # --------------------------------------------------------
    # Check origin zone consistency across solver seeds
    # --------------------------------------------------------

    zone_check = (
        drt
        .groupby(
            [
                "scenario_id",
                "request_id",
            ]
        )["_zone_key"]
        .nunique()
    )

    if zone_check.gt(1).any():
        raise ValueError(
            "origin_zone_id changes across solver seeds "
            "for the same scenario + request."
        )

    # --------------------------------------------------------
    # One row per unique realized request
    # --------------------------------------------------------

    drt_request = (
        drt
        .groupby(
            [
                "scenario_id",
                "request_id",
            ],
            as_index=False,
        )
        .agg(
            origin_zone_id=(
                "origin_zone_id",
                "first",
            ),
            _zone_key=(
                "_zone_key",
                "first",
            ),
            drt_accessibility_rate=(
                "drt_accessible_15min",
                "mean",
            ),
            n_solver_seeds=(
                "solver_seed",
                "nunique",
            ),
        )
    )

    merged = drt_request.merge(
        benchmark[
            [
                "scenario_id",
                "request_id",
                "walking_within_15min",
                "transit_within_15min",
            ]
        ],
        on=[
            "scenario_id",
            "request_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No request-level matches between "
            "benchmark and Proposed-full DRT data."
        )

    return merged


# ============================================================
# Aggregate to origin TAZ
# ============================================================

def aggregate_by_zone(
    request_data: pd.DataFrame,
) -> pd.DataFrame:

    zone = (
        request_data
        .groupby(
            "_zone_key",
            as_index=False,
        )
        .agg(
            origin_zone_id=(
                "origin_zone_id",
                "first",
            ),

            n_requests=(
                "request_id",
                "size",
            ),

            n_scenarios=(
                "scenario_id",
                "nunique",
            ),

            walking_accessibility_rate=(
                "walking_within_15min",
                "mean",
            ),

            transit_accessibility_rate=(
                "transit_within_15min",
                "mean",
            ),

            drt_accessibility_rate=(
                "drt_accessibility_rate",
                "mean",
            ),

            mean_solver_seeds=(
                "n_solver_seeds",
                "mean",
            ),
        )
    )

    # --------------------------------------------------------
    # Accessibility improvements
    # --------------------------------------------------------

    zone[
        "drt_minus_walking"
    ] = (
        zone[
            "drt_accessibility_rate"
        ]
        - zone[
            "walking_accessibility_rate"
        ]
    )

    zone[
        "drt_minus_transit"
    ] = (
        zone[
            "drt_accessibility_rate"
        ]
        - zone[
            "transit_accessibility_rate"
        ]
    )

    # Percentage-point versions for plotting
    zone[
        "drt_minus_walking_pp"
    ] = (
        zone[
            "drt_minus_walking"
        ]
        * 100.0
    )

    zone[
        "drt_minus_transit_pp"
    ] = (
        zone[
            "drt_minus_transit"
        ]
        * 100.0
    )

    return zone


# ============================================================
# Plot one map
# ============================================================

def plot_accessibility_map(
    ax,
    map_data: gpd.GeoDataFrame,
    column: str,
    title: str,
    vlimit: float,
):

    norm = TwoSlopeNorm(
        vmin=-vlimit,
        vcenter=0.0,
        vmax=vlimit,
    )

    map_data.plot(
        column=column,
        ax=ax,
        cmap="RdBu",
        norm=norm,
        linewidth=0.45,
        edgecolor="white",
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "label": "No evaluated requests",
        },
    )

    ax.set_title(
        title,
        fontsize=11,
        pad=8,
    )

    ax.set_axis_off()

    return norm


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.4.4 — Spatial distribution "
            "of accessibility improvements."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/base_competitive.yaml",
    )

    parser.add_argument(
        "--benchmark-dir",
        default="outputs/benchmarks_competitive",
    )

    parser.add_argument(
        "--alns-dir",
        default="outputs/alns_competitive",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/paper_results/"
            "chapter5_multimodal"
        ),
    )

    parser.add_argument(
        "--two-panels",
        action="store_true",
        help=(
            "Plot both DRT-Walking and DRT-Public Transit. "
            "Default is DRT-Public Transit only."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Configuration and GIS
    # ========================================================

    config = load_config(
        args.config
    )

    zones_path = path_from_config(
        config,
        "zones",
    )

    zone_id_col = str(
        config[
            "zone_panel"
        ][
            "zone_id_column_in_shapefile"
        ]
    )

    zones = gpd.read_file(
        zones_path
    )

    if zone_id_col not in zones.columns:
        raise KeyError(
            f"Zone ID column '{zone_id_col}' "
            f"not found in:\n{zones_path}\n\n"
            f"Available columns:\n"
            f"{zones.columns.tolist()}"
        )

    zones["_zone_key"] = normalize_zone_id(
        zones[zone_id_col]
    )

    # ========================================================
    # 2. Load benchmark + DRT
    # ========================================================

    benchmark = load_benchmarks(
        Path(args.benchmark_dir)
    )

    drt = load_drt(
        Path(args.alns_dir)
    )

    # ========================================================
    # 3. Build matched request-level multimodal dataset
    # ========================================================

    request_data = (
        build_multimodal_request_data(
            benchmark,
            drt,
        )
    )

    # ========================================================
    # 4. Aggregate requests by origin TAZ
    # ========================================================

    zone_metrics = (
        aggregate_by_zone(
            request_data
        )
    )

    # ========================================================
    # 5. Join to Jerusalem 90 TAZ polygons
    # ========================================================

    map_data = zones.merge(
        zone_metrics,
        on="_zone_key",
        how="left",
        validate="one_to_one",
    )

    matched_zones = int(
        map_data[
            "n_requests"
        ].notna()
        .sum()
    )

    print(
        f"TAZs with evaluated requests: "
        f"{matched_zones}/{len(map_data)}"
    )

    if matched_zones == 0:
        raise ValueError(
            "No TAZ polygons matched origin_zone_id."
        )

    # ========================================================
    # 6. Common symmetric colour range
    #
    # Important:
    # zero must always have the same neutral colour.
    # Positive = DRT better.
    # Negative = benchmark better.
    # ========================================================

    if args.two_panels:

        all_values = np.concatenate(
            [
                map_data[
                    "drt_minus_walking_pp"
                ]
                .dropna()
                .to_numpy(dtype=float),

                map_data[
                    "drt_minus_transit_pp"
                ]
                .dropna()
                .to_numpy(dtype=float),
            ]
        )

    else:

        all_values = (
            map_data[
                "drt_minus_transit_pp"
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

    if len(all_values) == 0:
        raise ValueError(
            "No valid accessibility improvement values."
        )

    vlimit = float(
        np.nanmax(
            np.abs(
                all_values
            )
        )
    )

    if vlimit <= 0:
        vlimit = 1.0

    # ========================================================
    # 7. Plot
    # ========================================================

    if args.two_panels:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.2, 6.6),
        )

        norm = plot_accessibility_map(
            axes[0],
            map_data,
            "drt_minus_walking_pp",
            "(a) DRT − Walking",
            vlimit,
        )

        plot_accessibility_map(
            axes[1],
            map_data,
            "drt_minus_transit_pp",
            "(b) DRT − Public transit",
            vlimit,
        )

        sm = plt.cm.ScalarMappable(
            cmap="RdBu",
            norm=norm,
        )

        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=axes,
            fraction=0.03,
            pad=0.025,
        )

    else:

        fig, ax = plt.subplots(
            figsize=(6.8, 7.2)
        )

        norm = plot_accessibility_map(
            ax,
            map_data,
            "drt_minus_transit_pp",
            "DRT − Public transit",
            vlimit,
        )

        sm = plt.cm.ScalarMappable(
            cmap="RdBu",
            norm=norm,
        )

        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            fraction=0.038,
            pad=0.025,
        )

    cbar.set_label(
        "Change in 15-minute accessibility (percentage points)",
        fontsize=9.5,
    )

    fig.tight_layout()

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

    if args.two_panels:

        suffix = (
            "walking_and_transit"
        )

    else:

        suffix = (
            "drt_minus_transit"
        )

    png_path = (
        output_dir
        / (
            "Figure_5_4_4_spatial_accessibility_"
            f"improvements_{suffix}.png"
        )
    )

    pdf_path = (
        output_dir
        / (
            "Figure_5_4_4_spatial_accessibility_"
            f"improvements_{suffix}.pdf"
        )
    )

    zone_data_path = (
        output_dir
        / "Figure_5_4_4_zone_accessibility_data.csv"
    )

    request_data_path = (
        output_dir
        / "Figure_5_4_4_request_level_multimodal_data.csv"
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
    # 9. Save reproducible aggregated data
    # ========================================================

    zone_metrics.to_csv(
        zone_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    request_data.to_csv(
        request_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 10. Console audit
    # ========================================================

    print()
    print("=" * 94)

    print(
        "Figure 5.4.4 — Spatial distribution "
        "of accessibility improvements"
    )

    print("=" * 94)

    print(
        f"Benchmark source      : "
        f"{args.benchmark_dir}"
    )

    print(
        f"DRT source            : "
        f"{Path(args.alns_dir) / METHOD_DIR}"
    )

    print(
        f"TAZ source            : "
        f"{zones_path}"
    )

    print(
        f"TAZ ID field          : "
        f"{zone_id_col}"
    )

    print(
        f"Matched requests      : "
        f"{len(request_data):,}"
    )

    print(
        f"TAZs with requests    : "
        f"{matched_zones}"
    )

    print()

    print(
        "DRT − PT accessibility improvement:"
    )

    print(
        f"  Mean across observed TAZs : "
        f"{zone_metrics['drt_minus_transit_pp'].mean():+.2f} pp"
    )

    print(
        f"  Minimum                   : "
        f"{zone_metrics['drt_minus_transit_pp'].min():+.2f} pp"
    )

    print(
        f"  Maximum                   : "
        f"{zone_metrics['drt_minus_transit_pp'].max():+.2f} pp"
    )

    print()

    print(
        f"PNG saved             : {png_path}"
    )

    print(
        f"PDF saved             : {pdf_path}"
    )

    print(
        f"Zone plot data         : {zone_data_path}"
    )

    print(
        f"Request-level data     : {request_data_path}"
    )

    print("=" * 94)


if __name__ == "__main__":
    main()