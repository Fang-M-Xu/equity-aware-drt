from __future__ import annotations

import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.1.2: Spatial distribution of "
            "observed and predicted demand."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Experiment configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # =========================================================
    # 1. Formal experiment paths
    # =========================================================
    metrics_dir = path_from_config(config, "metrics_dir")
    
    prediction_path = (
        metrics_dir / "lightgbm_test_predictions.parquet"
    )

    paper_results_dir = path_from_config(config, "paper_results_dir")

    zones_path = path_from_config(config, "zones")

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Cannot find LightGBM held-out predictions:\n"
            f"{prediction_path}"
        )

    if not zones_path.exists():
        raise FileNotFoundError(
            f"Cannot find Jerusalem TAZ shapefile:\n"
            f"{zones_path}"
        )

    # =========================================================
    # 2. Load formal held-out predictions
    # =========================================================
    predictions = pd.read_parquet(prediction_path)

    target_col = str(
        config["zone_panel"]["target_column"]
    )

    prediction_col = "lightgbm"
    prediction_zone_col = "origin_zone_id"

    required = [
        prediction_zone_col,
        target_col,
        prediction_col,
    ]

    missing = [
        c for c in required
        if c not in predictions.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns in {prediction_path}: "
            f"{missing}\n"
            f"Available columns: "
            f"{predictions.columns.tolist()}"
        )

    # =========================================================
    # 3. Convert values to numeric and remove invalid rows
    # =========================================================
    predictions[target_col] = pd.to_numeric(
        predictions[target_col],
        errors="coerce",
    )

    predictions[prediction_col] = pd.to_numeric(
        predictions[prediction_col],
        errors="coerce",
    )

    predictions = predictions.dropna(
        subset=[
            prediction_zone_col,
            target_col,
            prediction_col,
        ]
    ).copy()

    if predictions.empty:
        raise ValueError(
            "No valid held-out prediction rows were found."
        )

    # =========================================================
    # 4. Aggregate the full held-out test panel by TAZ
    #
    # Each map represents the mean demand level of each TAZ
    # across all held-out day × time-bin observations.
    # =========================================================
    zone_demand = (
        predictions
        .groupby(
            prediction_zone_col,
            as_index=False,
        )
        .agg(
            observed_demand=(
                target_col,
                "mean",
            ),
            predicted_demand=(
                prediction_col,
                "mean",
            ),
            n_test_cells=(
                target_col,
                "size",
            ),
        )
    )

    # Use string keys to avoid int/string mismatch
    zone_demand["_zone_key"] = (
        zone_demand[prediction_zone_col]
        .astype(str)
        .str.strip()
    )

    # =========================================================
    # 5. Load Jerusalem 90-TAZ polygons
    # =========================================================
    zones = gpd.read_file(zones_path)

    zone_id_col = str(
        config["zone_panel"][
            "zone_id_column_in_shapefile"
        ]
    )

    if zone_id_col not in zones.columns:
        raise KeyError(
            f"TAZ ID column '{zone_id_col}' "
            f"not found in {zones_path}.\n"
            f"Available columns: {zones.columns.tolist()}"
        )

    zones["_zone_key"] = (
        zones[zone_id_col]
        .astype(str)
        .str.strip()
    )

    # =========================================================
    # 6. Join model results to polygons
    # =========================================================
    map_data = zones.merge(
        zone_demand[
            [
                "_zone_key",
                "observed_demand",
                "predicted_demand",
                "n_test_cells",
            ]
        ],
        on="_zone_key",
        how="left",
        validate="one_to_one",
    )

    matched = int(
        map_data["observed_demand"].notna().sum()
    )

    print(
        f"TAZ polygons matched to held-out predictions: "
        f"{matched}/{len(map_data)}"
    )

    if matched == 0:
        raise ValueError(
            "No TAZ polygons matched origin_zone_id. "
            "Check the zone identifier formats."
        )

    # =========================================================
    # 7. Common colour scale
    #
    # IMPORTANT:
    # Observed and predicted maps must use the same scale.
    # =========================================================
    all_values = np.concatenate(
        [
            map_data["observed_demand"]
            .dropna()
            .to_numpy(dtype=float),

            map_data["predicted_demand"]
            .dropna()
            .to_numpy(dtype=float),
        ]
    )

    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    # =========================================================
    # 8. Publication output directory
    # =========================================================
    output_dir = paper_results_dir / "chapter5_1_model"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # 9. Figure (a): Observed demand
    # =========================================================
    fig_a, ax_a = plt.subplots(
        figsize=(6.2, 7.2)
    )

    map_data.plot(
        column="observed_demand",
        ax=ax_a,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.45,
        edgecolor="white",
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
        },
    )

    # Invisible mappable only for common colorbar
    sm_a = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    sm_a.set_array([])

    cbar_a = fig_a.colorbar(
        sm_a,
        ax=ax_a,
        fraction=0.035,
        pad=0.02,
    )

    cbar_a.set_label(
        "Mean observed demand",
        fontsize=10,
    )

    ax_a.set_title(
        "(a) Observed demand",
        fontsize=12,
        pad=10,
    )

    ax_a.set_axis_off()

    fig_a.tight_layout()

    observed_png = (
        output_dir
        / "Figure_5_1_2a_observed_demand.png"
    )

    observed_pdf = (
        output_dir
        / "Figure_5_1_2a_observed_demand.pdf"
    )

    fig_a.savefig(
        observed_png,
        dpi=600,
        bbox_inches="tight",
    )

    fig_a.savefig(
        observed_pdf,
        bbox_inches="tight",
    )

    plt.close(fig_a)

    # =========================================================
    # 10. Figure (b): Predicted demand
    # =========================================================
    fig_b, ax_b = plt.subplots(
        figsize=(6.2, 7.2)
    )

    map_data.plot(
        column="predicted_demand",
        ax=ax_b,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.45,
        edgecolor="white",
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
        },
    )

    sm_b = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    sm_b.set_array([])

    cbar_b = fig_b.colorbar(
        sm_b,
        ax=ax_b,
        fraction=0.035,
        pad=0.02,
    )

    cbar_b.set_label(
        "Mean predicted demand",
        fontsize=10,
    )

    ax_b.set_title(
        "(b) Predicted demand",
        fontsize=12,
        pad=10,
    )

    ax_b.set_axis_off()

    fig_b.tight_layout()

    predicted_png = (
        output_dir
        / "Figure_5_1_2b_predicted_demand.png"
    )

    predicted_pdf = (
        output_dir
        / "Figure_5_1_2b_predicted_demand.pdf"
    )

    fig_b.savefig(
        predicted_png,
        dpi=600,
        bbox_inches="tight",
    )

    fig_b.savefig(
        predicted_pdf,
        bbox_inches="tight",
    )

    plt.close(fig_b)

    # =========================================================
    # 11. Save plotting data for reproducibility
    # =========================================================
    plot_data_path = (
        output_dir
        / "Figure_5_1_2_spatial_demand_plot_data.csv"
    )

    zone_demand[
        [
            prediction_zone_col,
            "observed_demand",
            "predicted_demand",
            "n_test_cells",
        ]
    ].to_csv(
        plot_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================
    # 12. Console audit
    # =========================================================
    print()
    print("=" * 72)
    print(
        "Figure 5.1.2 — Spatial distribution "
        "of observed and predicted demand"
    )
    print("=" * 72)

    print(f"Prediction source : {prediction_path}")
    print(f"TAZ source        : {zones_path}")
    print(f"TAZ ID field      : {zone_id_col}")
    print(f"Test rows         : {len(predictions):,}")
    print(f"Predicted TAZs    : {len(zone_demand):,}")
    print(f"Matched TAZs      : {matched}/{len(map_data)}")

    print(
        f"Common scale      : "
        f"{vmin:.3f} – {vmax:.3f}"
    )

    print()
    print(f"Observed PNG      : {observed_png}")
    print(f"Predicted PNG     : {predicted_png}")
    print(f"Plot data         : {plot_data_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()