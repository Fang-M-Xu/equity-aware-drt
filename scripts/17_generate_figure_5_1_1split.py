from __future__ import annotations

import argparse
import json
import textwrap

import matplotlib
matplotlib.use("Agg")

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


# ============================================================
# Editorial palette
# ============================================================

NAVY = "#1F3A5F"
BLUE = "#4C78A8"
PURPLE = "#7A5195"
CORAL = "#E45756"
ORANGE = "#F28E2B"
INK = "#263238"
MID = "#7A8793"
LIGHT = "#DCE3EA"
PALE = "#F7F8FA"
WHITE = "#FFFFFF"


# ============================================================
# Candidate column names
# ============================================================

ZONE_ID_CANDIDATES = [
    "origin_zone_id",
    "TAZ_1270",
    "zone_id",
    "taz_id",
    "taz",
    "zone",
]

ZONE_NAME_CANDIDATES = [
    "english_name",
    "zone_name_en",
    "zone_name_eng",
    "name_en",
    "name_eng",
    "eng_name",
    "taz_name_en",
    "taz_name",
    "NAME_EN",
    "NAME_ENG",
    "ENG_NAME",
    "NameEN",
    "NameEng",
    "Name_Eng",
    "Name_En",
]


# ============================================================
# Utilities
# ============================================================

def normalize_zone_id(value: object) -> str:
    """Normalize numeric and string zone identifiers."""

    if pd.isna(value):
        return ""

    text_value = str(value).strip()

    if text_value.endswith(".0"):
        text_value = text_value[:-2]

    return text_value


def detect_column(
    columns: list[str] | pd.Index,
    candidates: list[str],
) -> str | None:
    """Detect a column using exact or case-insensitive matching."""

    columns_as_text = [str(column) for column in columns]

    exact = set(columns_as_text)

    lower_lookup = {
        column.lower(): column
        for column in columns_as_text
    }

    for candidate in candidates:

        if candidate in exact:
            return candidate

        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    return None


def read_zone_name_table(path: Path) -> pd.DataFrame:
    """
    Read zone ID / English-name mapping.

    Supported formats:
        CSV
        TXT
        Excel
        Parquet
        Shapefile
        GeoPackage
        GeoJSON
    """

    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".shp", ".gpkg", ".geojson"}:

        try:
            import geopandas as gpd

        except ImportError as error:

            raise ImportError(
                "Reading a spatial zone-name file requires geopandas.\n"
                "Install it with:\n"
                "pip install geopandas pyogrio"
            ) from error

        gdf = gpd.read_file(path)

        return pd.DataFrame(
            gdf.drop(
                columns="geometry",
                errors="ignore",
            )
        )

    raise ValueError(
        f"Unsupported zone-name mapping format: {path}"
    )


def load_zone_name_mapping(
    path: Path | None,
    requested_id_column: str | None,
    requested_name_column: str | None,
) -> dict[str, str]:
    """Load origin-zone ID to English-name mapping."""

    if path is None:
        return {}

    if not path.exists():
        raise FileNotFoundError(
            f"Zone-name file does not exist:\n{path}"
        )

    frame = read_zone_name_table(path)

    # --------------------------------------------------------
    # Zone ID
    # --------------------------------------------------------

    id_column = requested_id_column

    if id_column and id_column not in frame.columns:
        id_column = detect_column(
            frame.columns,
            [id_column],
        )

    if id_column is None:
        id_column = detect_column(
            frame.columns,
            ZONE_ID_CANDIDATES,
        )

    # --------------------------------------------------------
    # English zone name
    # --------------------------------------------------------

    name_column = requested_name_column

    if name_column and name_column not in frame.columns:
        name_column = detect_column(
            frame.columns,
            [name_column],
        )

    if name_column is None:
        name_column = detect_column(
            frame.columns,
            ZONE_NAME_CANDIDATES,
        )

    if id_column is None or name_column is None:

        raise KeyError(
            "Could not identify zone ID and English-name columns in:\n"
            f"{path}\n\n"
            f"Available columns:\n{frame.columns.tolist()}\n\n"
            "Use --zone-id-column and --zone-name-column "
            "to specify them manually."
        )

    mapping_frame = (
        frame[[id_column, name_column]]
        .dropna()
        .copy()
    )

    mapping_frame["_zone_key"] = (
        mapping_frame[id_column]
        .map(normalize_zone_id)
    )

    mapping_frame["_english_name"] = (
        mapping_frame[name_column]
        .astype(str)
        .str.strip()
    )

    mapping_frame = mapping_frame.loc[
        mapping_frame["_zone_key"].ne("")
        & mapping_frame["_english_name"].ne("")
    ]

    mapping_frame = mapping_frame.drop_duplicates(
        "_zone_key",
        keep="first",
    )

    mapping = dict(
        zip(
            mapping_frame["_zone_key"],
            mapping_frame["_english_name"],
        )
    )

    print(
        f"Loaded {len(mapping)} zone-name mappings "
        f"from {path}"
    )

    return mapping


def build_zone_axis_labels(
    frame: pd.DataFrame,
    zone_name_mapping: dict[str, str],
    label_mode: str,
) -> list[str]:
    """
    Build labels for Figure 5.1.1(b).

    Modes:
        name
        name_id
        id
    """

    zone_id_column = detect_column(
        frame.columns,
        ZONE_ID_CANDIDATES,
    )

    if zone_id_column is None:

        raise KeyError(
            "Figure 5.1.1(b) requires a zone ID column.\n"
            f"Expected one of:\n{ZONE_ID_CANDIDATES}\n\n"
            f"Available columns:\n{frame.columns.tolist()}"
        )

    labels: list[str] = []

    for raw_id in frame[zone_id_column]:

        zone_id = normalize_zone_id(raw_id)

        english_name = zone_name_mapping.get(zone_id)

        if label_mode == "id":

            label = zone_id

        elif (
            label_mode == "name_id"
            and english_name
        ):

            label = (
                f"{english_name} "
                f"({zone_id})"
            )

        elif english_name:

            label = english_name

        else:

            label = zone_id

        labels.append(
            textwrap.fill(
                label,
                width=27,
            )
        )

    return labels


# ============================================================
# Plot style
# ============================================================

def configure_style() -> None:

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,

            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "axes.titleweight": "semibold",

            "axes.edgecolor": "#AAB5BD",
            "axes.linewidth": 0.8,
            "axes.facecolor": WHITE,

            "figure.facecolor": WHITE,

            "xtick.color": INK,
            "ytick.color": INK,

            "axes.labelcolor": INK,
            "text.color": INK,

            "grid.color": "#D9E1E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.72,

            "legend.frameon": False,

            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
        }
    )


def clean_axes(
    ax,
    grid_axis: str = "x",
) -> None:

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        True,
        axis=grid_axis,
    )

    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure,
    output: Path,
    dpi: int,
    use_tight_layout: bool = True,
) -> None:

    if use_tight_layout:
        fig.tight_layout()

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=WHITE,
    )

    fig.savefig(
        output.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor=WHITE,
    )

    plt.close(fig)


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.1.1(a) and Figure 5.1.1(b): "
            "held-out zone demand prediction and "
            "high-demand zone identification."
        )
    )

    parser.add_argument(
        "--results-root",
        default="outputs/paper_results/chapter5_1_model/demand_validation",
        help=(
            "Root directory containing "
            "zone_prediction_comparison.csv and "
            "paper_table_demand_model.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_1_model",
        help="Output directory.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="PNG resolution.",
    )

    parser.add_argument(
        "--figure",
        default="5.1.1",
        choices=[
            "5.1.1",
            "5.1.1a",
            "5.1.1b",
        ],
        help=(
            "5.1.1 generates both panels; "
            "5.1.1a or 5.1.1b generates one panel."
        ),
    )

    parser.add_argument(
        "--zone-name-file",
        default=None,
        help=(
            "Optional CSV, Excel, Parquet, Shapefile, "
            "GeoPackage, or GeoJSON containing "
            "zone IDs and English zone names."
        ),
    )

    parser.add_argument(
        "--zone-id-column",
        default=None,
        help=(
            "Zone ID column in --zone-name-file. "
            "Auto-detected if omitted."
        ),
    )

    parser.add_argument(
        "--zone-name-column",
        default=None,
        help=(
            "English-name column in --zone-name-file. "
            "Auto-detected if omitted."
        ),
    )

    parser.add_argument(
        "--zone-label-mode",
        default="name",
        choices=[
            "name",
            "name_id",
            "id",
        ],
        help=(
            "Figure 5.1.1(b) y-axis labels: "
            "English name, English name + ID, or ID only."
        ),
    )

    return parser.parse_args()


# ============================================================
# Input file search
# ============================================================

def find_unique(
    root: Path,
    filename: str,
) -> Path:
    """
    Recursively locate one required input file.

    Preference is given to demand-validation folders.
    """

    matches = sorted(
        root.rglob(filename)
    )

    if not matches:

        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"    {filename}\n"
            f"under:\n"
            f"    {root}\n"
        )

    preferred = [
        path
        for path in matches
        if "demand_validation"
        in str(path).lower()
    ]

    selected = (
        preferred[0]
        if preferred
        else matches[0]
    )

    if len(matches) > 1:

        print(
            f"[Warning] Multiple copies of {filename} found."
        )

        print(
            f"          Using: {selected}"
        )

    return selected


# ============================================================
# Figure 5.1.1(a)
# ============================================================

def fig_5_1_1a(
    zone: pd.DataFrame,
    metrics: dict[str, float],
    output: Path,
    dpi: int,
) -> None:

    frame = zone.copy()

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required = [
        "observed",
        "predicted",
        "observed_high",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise KeyError(
            f"Missing required columns in "
            f"zone_prediction_comparison.csv:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{frame.columns.tolist()}"
        )

    frame["observed"] = pd.to_numeric(
        frame["observed"],
        errors="coerce",
    )

    frame["predicted"] = pd.to_numeric(
        frame["predicted"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "observed",
            "predicted",
        ]
    )

    # More reliable conversion than astype(bool)
    if frame["observed_high"].dtype != bool:

        frame["observed_high"] = (
            frame["observed_high"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.6, 6.2)
    )

    low = frame.loc[
        ~frame["observed_high"]
    ]

    high = frame.loc[
        frame["observed_high"]
    ]

    ax.scatter(
        low["observed"],
        low["predicted"],
        s=38,
        alpha=0.58,
        color=BLUE,
        edgecolors=WHITE,
        linewidths=0.55,
        label="Other zones",
    )

    ax.scatter(
        high["observed"],
        high["predicted"],
        s=76,
        alpha=0.95,
        color=CORAL,
        marker="^",
        edgecolors=WHITE,
        linewidths=0.7,
        label="Observed high-demand zones",
        zorder=3,
    )

    # --------------------------------------------------------
    # Perfect agreement line
    # --------------------------------------------------------

    upper = float(
        max(
            frame["observed"].max(),
            frame["predicted"].max(),
        )
    )

    ax.plot(
        [0, upper],
        [0, upper],
        color=MID,
        linestyle=(0, (3, 3)),
        linewidth=1.1,
        label="Perfect agreement",
    )

    # --------------------------------------------------------
    # Log-scale fitted trend
    # --------------------------------------------------------

    log_x = np.log1p(
        frame["observed"].to_numpy()
    )

    log_y = np.log1p(
        frame["predicted"].to_numpy()
    )

    slope, intercept = np.polyfit(
        log_x,
        log_y,
        1,
    )

    x_line = np.geomspace(
        1,
        max(1.1, upper),
        250,
    )

    y_line = np.expm1(
        intercept
        + slope
        * np.log1p(x_line)
    )

    ax.plot(
        x_line,
        y_line,
        color=NAVY,
        linewidth=2.0,
        label="Log-scale fitted trend",
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xscale(
        "symlog",
        linthresh=20,
    )

    ax.set_yscale(
        "symlog",
        linthresh=20,
    )

    ax.set_xlabel(
        "Observed aggregated demand"
    )

    ax.set_ylabel(
        "Predicted aggregated demand"
    )

    ax.set_title(
        "Calibration across held-out origin zones",
        loc="left",
        pad=12,
    )

    clean_axes(
        ax,
        "both",
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metric_text = (
        f"RMSE  {metrics['rmse']:.3f}\n"
        f"$R^2$  {metrics['r2']:.3f}\n"
        f"Spearman $\\rho$  "
        f"{metrics['spearman_rho']:.3f}\n"
        f"High-demand F1  "
        f"{metrics['high_f1']:.3f}"
    )

    ax.text(
        0.035,
        0.965,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.0,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=PALE,
            edgecolor=LIGHT,
        ),
    )

    ax.legend(
        loc="lower right",
        fontsize=8.8,
    )

    save_figure(
        fig,
        output,
        dpi,
    )


# ============================================================
# Figure 5.1.1(b)
# ============================================================

def fig_5_1_1b(
    zone: pd.DataFrame,
    zone_name_mapping: dict[str, str],
    zone_label_mode: str,
    output: Path,
    dpi: int,
) -> None:

    frame = zone.copy()

    required = [
        "observed",
        "predicted",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise KeyError(
            f"Missing required columns in "
            f"zone_prediction_comparison.csv:\n"
            f"{missing}"
        )

    frame["observed"] = pd.to_numeric(
        frame["observed"],
        errors="coerce",
    )

    frame["predicted"] = pd.to_numeric(
        frame["predicted"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "observed",
            "predicted",
        ]
    )

    # --------------------------------------------------------
    # Select 14 highest observed-demand zones
    # --------------------------------------------------------

    top = (
        frame
        .nlargest(
            14,
            "observed",
        )
        .sort_values(
            "observed"
        )
    )

    labels = build_zone_axis_labels(
        top,
        zone_name_mapping=
            zone_name_mapping,
        label_mode=
            zone_label_mode,
    )

    y = np.arange(
        len(top)
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10.2, 7.0)
    )

    fig.subplots_adjust(
        left=0.36,
        right=0.96,
        top=0.90,
        bottom=0.12,
    )

    # Connecting lines
    for index, (_, row) in enumerate(
        top.iterrows()
    ):

        ax.plot(
            [
                row["observed"],
                row["predicted"],
            ],
            [
                index,
                index,
            ],
            color=LIGHT,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=1,
        )

    # Observed
    ax.scatter(
        top["observed"],
        y,
        color=NAVY,
        s=58,
        edgecolors=WHITE,
        linewidths=0.7,
        label="Observed",
        zorder=3,
    )
    # Predicted
    ax.scatter(
        top["predicted"],
        y,
        color=ORANGE,
        s=58,
        edgecolors=WHITE,
        linewidths=0.7,
        label="Predicted",
        zorder=3,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels,
        fontsize=9.0,
    )

    ax.set_xlabel(
        "Aggregated held-out demand"
    )

    ax.set_ylabel(
        "TAZ-zone name"
        if zone_label_mode != "id"
        else "Origin-zone ID"
    )

    ax.set_title(
        "Observed versus predicted demand "
        "in the 14 highest-demand zones",
        loc="left",
        pad=12,
    )

    clean_axes(
        ax,
        "x",
    )

    ax.legend(
        loc="lower right",
        fontsize=9.0,
    )

    # Do not use tight_layout here because
    # long TAZ labels need the manually reserved margin.
    save_figure(
        fig,
        output,
        dpi,
        use_tight_layout=False,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    configure_style()

    args = parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    root = (
        Path(args.results_root)
        .expanduser()
        .resolve()
    )

    output = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "Figure 5.1.1 — Held-out zone demand prediction "
        "and high-demand zone identification"
    )

    print("=" * 72)

    print(
        f"Results root:\n"
        f"  {root}"
    )

    print(
        f"Output directory:\n"
        f"  {output}"
    )

    # --------------------------------------------------------
    # Only load files required by Figure 5.1.1
    # --------------------------------------------------------

    zone_file = find_unique(
        root,
        "zone_prediction_comparison.csv",
    )

    demand_file = find_unique(
        root,
        "paper_table_demand_model.csv",
    )

    print(
        "\nInput files:"
    )

    print(
        f"  Zone predictions: {zone_file}"
    )

    print(
        f"  Model metrics:    {demand_file}"
    )

    zone = pd.read_csv(
        zone_file
    )

    demand_table = pd.read_csv(
        demand_file
    )

    if demand_table.empty:

        raise ValueError(
            "paper_table_demand_model.csv is empty."
        )

    demand = demand_table.iloc[0]

    # --------------------------------------------------------
    # Required metric fields
    # --------------------------------------------------------

    metric_columns = [
        "rmse",
        "r2",
        "spearman_rho",
        "high_zone_f1",
    ]

    missing_metrics = [
        column
        for column in metric_columns
        if column not in demand_table.columns
    ]

    if missing_metrics:

        raise KeyError(
            "paper_table_demand_model.csv is missing:\n"
            f"{missing_metrics}\n\n"
            f"Available columns:\n"
            f"{demand_table.columns.tolist()}"
        )

    metrics = {
        "rmse":
            float(demand["rmse"]),

        "r2":
            float(demand["r2"]),

        "spearman_rho":
            float(demand["spearman_rho"]),

        "high_f1":
            float(demand["high_zone_f1"]),
    }

    print(
        "\nMetrics used in Figure 5.1.1(a):"
    )

    print(
        f"  RMSE            = "
        f"{metrics['rmse']:.3f}"
    )

    print(
        f"  R²              = "
        f"{metrics['r2']:.3f}"
    )

    print(
        f"  Spearman rho    = "
        f"{metrics['spearman_rho']:.3f}"
    )

    print(
        f"  High-demand F1  = "
        f"{metrics['high_f1']:.3f}"
    )

    # --------------------------------------------------------
    # Optional zone-name mapping
    # --------------------------------------------------------

    zone_name_path = (
        Path(args.zone_name_file)
        .expanduser()
        .resolve()

        if args.zone_name_file

        else None
    )

    zone_name_mapping = (
        load_zone_name_mapping(
            zone_name_path,
            requested_id_column=
                args.zone_id_column,
            requested_name_column=
                args.zone_name_column,
        )
    )

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    generated_files = []

    if args.figure in {
        "5.1.1",
        "5.1.1a",
    }:

        output_a = (
            output
            / "Figure_5_1_1a_Editorial_Demand_Calibration.png"
        )

        fig_5_1_1a(
            zone,
            metrics,
            output_a,
            args.dpi,
        )

        generated_files.extend(
            [
                output_a,
                output_a.with_suffix(".svg"),
            ]
        )

        print(
            "\nGenerated Figure 5.1.1(a)"
        )

    if args.figure in {
        "5.1.1",
        "5.1.1b",
    }:

        output_b = (
            output
            / "Figure_5_1_1b_Editorial_High_Demand_Zones.png"
        )

        fig_5_1_1b(
            zone,
            zone_name_mapping,
            args.zone_label_mode,
            output_b,
            args.dpi,
        )

        generated_files.extend(
            [
                output_b,
                output_b.with_suffix(".svg"),
            ]
        )

        print(
            "Generated Figure 5.1.1(b)"
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "figure":
            args.figure,

        "results_root":
            str(root),

        "input_zone_prediction":
            str(zone_file),

        "input_demand_metrics":
            str(demand_file),

        "zone_name_file":
            (
                str(zone_name_path)
                if zone_name_path
                else None
            ),

        "zone_label_mode":
            args.zone_label_mode,

        "dpi":
            args.dpi,

        "outputs": [
            str(path)
            for path in generated_files
        ],
    }

    manifest_path = (
        output
        / "Figure_5_1_1_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print(
        "\nOutput files:"
    )

    for path in generated_files:

        print(
            f"  {path}"
        )

    print(
        f"  {manifest_path}"
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()