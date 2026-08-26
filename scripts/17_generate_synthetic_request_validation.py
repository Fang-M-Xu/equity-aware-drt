from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "4.3-distributional-validation-nameen"

from config import load_config, path_from_config


# ============================================================
# Validation dimensions
# ============================================================

DIMENSIONS = {
    "Destination-zone distribution": {
        "survey_aliases": [
            "destination_zone_id",
        ],
        "synthetic_aliases": [
            "destination_zone_id",
        ],
        "binary": False,
    },

    "Activity distribution": {
        "survey_aliases": [
            "destination_activity_type",
        ],
        "synthetic_aliases": [
            "destination_activity_type",
        ],
        "binary": False,
    },

    "Sector distribution": {
        "survey_aliases": [
            "sector_group",
        ],
        "synthetic_aliases": [
            "sector_group",
        ],
        "binary": False,
    },

    "Low-income status": {
        "survey_aliases": [
            "is_low_income",
            "low_income",
        ],
        "synthetic_aliases": [
            "is_low_income",
            "low_income",
        ],
        "binary": True,
    },

    "Car availability": {
        "survey_aliases": [
            "is_no_car",
            "no_car",
        ],
        "synthetic_aliases": [
            "is_no_car",
            "no_car",
        ],
        "binary": True,
    },

    "Older-traveller status": {
        "survey_aliases": [
            "is_older",
            "older",
        ],
        "synthetic_aliases": [
            "is_older",
            "older",
        ],
        "binary": True,
    },

    "Student status": {
        "survey_aliases": [
            "is_student",
            "student",
        ],
        "synthetic_aliases": [
            "is_student",
            "student",
        ],
        "binary": True,
    },

    "Service-gap-origin status": {
        "survey_aliases": [
            "origin_gap",
            "is_gap",
            "service_gap",
            "gap_i",
            "origin_service_gap",
        ],
        "synthetic_aliases": [
            "origin_gap",
            "is_gap",
            "service_gap",
            "gap_i",
            "origin_service_gap",
        ],
        "binary": True,
    },
}


# ============================================================
# Utilities
# ============================================================

def load_candidate_pool(directory: Path) -> pd.DataFrame:
    """
    Load the natural synthetic candidate pool.

    IMPORTANT:
    Do not use competitive scenarios here because competitive
    sampling intentionally changes the request composition.
    """

    files = sorted(
        directory.glob("forecast_scenario_*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            "No forecast_scenario_*.parquet files found in:\n"
            f"{directory}"
        )

    frames = []

    for scenario_id, path in enumerate(files):

        frame = pd.read_parquet(path).copy()

        frame["_source_file"] = path.name
        frame["_candidate_scenario_id"] = scenario_id

        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def find_column(
    frame: pd.DataFrame,
    aliases: list[str],
) -> str | None:
    """
    Find the first matching column from possible aliases.
    """

    lookup = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }

    for alias in aliases:

        key = (
            str(alias)
            .strip()
            .lower()
        )

        if key in lookup:
            return lookup[key]

    return None


def normalize_zone_id(value) -> str:
    """
    Normalise zone IDs so values such as:
        100513
        100513.0
        "100513"
    all match the same TAZ.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    try:
        number = float(text)

        if number.is_integer():
            return str(int(number))

    except (TypeError, ValueError):
        pass

    return text


def wrap_label(
    value: str,
    width: int = 15,
) -> str:
    """
    Wrap long English TAZ names for publication figures.
    """

    text = str(value).strip()

    if not text:
        return ""

    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def clean_category(
    series: pd.Series,
) -> pd.Series:
    """
    Standardise ordinary categorical values.
    """

    return (
        series
        .fillna("Missing")
        .astype(str)
        .str.strip()
        .replace(
            {
                "": "Missing",
                "nan": "Missing",
                "None": "Missing",
                "<NA>": "Missing",
            }
        )
    )


def clean_binary(
    series: pd.Series,
) -> pd.Series:
    """
    Standardise binary values.

    Output:
        "0"
        "1"
        "Missing"
    """

    output = pd.Series(
        index=series.index,
        dtype="object",
    )

    missing = series.isna()

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_valid = (
        numeric.notna()
        & ~missing
    )

    output.loc[
        numeric_valid
    ] = (
        numeric.loc[
            numeric_valid
        ]
        .ne(0)
        .astype(int)
        .astype(str)
    )

    remaining = (
        ~numeric_valid
        & ~missing
    )

    text = (
        series.loc[
            remaining
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "yes",
        "y",
        "1",
        "gap",
    }

    false_values = {
        "false",
        "no",
        "n",
        "0",
        "non-gap",
        "nongap",
    }

    true_index = text[
        text.isin(true_values)
    ].index

    false_index = text[
        text.isin(false_values)
    ].index

    output.loc[
        true_index
    ] = "1"

    output.loc[
        false_index
    ] = "0"

    unresolved = (
        output.isna()
        & ~missing
    )

    if unresolved.any():

        output.loc[
            unresolved
        ] = (
            series.loc[
                unresolved
            ]
            .astype(str)
            .str.strip()
        )

    output.loc[
        missing
    ] = "Missing"

    return output


def prepare_series(
    series: pd.Series,
    binary: bool,
) -> pd.Series:

    if binary:
        return clean_binary(series)

    return clean_category(series)


def probability_distribution(
    series: pd.Series,
    categories: list[str],
) -> np.ndarray:
    """
    Return category probabilities using a common category set.
    """

    counts = (
        series
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0.0,
        )
    )

    return counts.to_numpy(
        dtype=float
    )


def js_divergence(
    survey: pd.Series,
    synthetic: pd.Series,
) -> float:
    """
    Jensen-Shannon divergence using natural logarithms.

    Interpretation:
        0 = identical distributions

    With natural logarithms:
        0 <= JSD <= ln(2)
    """

    categories = sorted(
        set(survey.unique())
        | set(synthetic.unique())
    )

    p = probability_distribution(
        survey,
        categories,
    )

    q = probability_distribution(
        synthetic,
        categories,
    )

    m = 0.5 * (
        p + q
    )

    def kl_divergence(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        mask = a > 0

        return float(
            np.sum(
                a[mask]
                * np.log(
                    a[mask] / b[mask]
                )
            )
        )

    return float(
        0.5 * kl_divergence(p, m)
        +
        0.5 * kl_divergence(q, m)
    )


def distribution_table(
    survey: pd.Series,
    synthetic: pd.Series,
    dimension: str,
) -> pd.DataFrame:
    """
    Category-level shares used for plotting and audit.
    """

    categories = sorted(
        set(survey.unique())
        | set(synthetic.unique())
    )

    survey_share = (
        survey
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0.0,
        )
    )

    synthetic_share = (
        synthetic
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0.0,
        )
    )

    return pd.DataFrame(
        {
            "dimension":
                dimension,

            "category":
                categories,

            "held_out_survey_share":
                survey_share.values,

            "synthetic_share":
                synthetic_share.values,
        }
    )


def positive_share(
    series: pd.Series,
) -> float:
    """
    Share belonging to positive category (=1).
    """

    return float(
        series.eq("1").mean()
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 5.1.2 and Figure 5.1.3 "
            "for distributional validation of synthetic requests."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Formal experiment configuration.",
    )

    parser.add_argument(
        "--candidate-dir",
        default="outputs/forecast_scenarios",
        help=(
            "Natural synthetic request scenarios generated by "
            "08_generate_1030_scenarios.py using base.yaml."
        ),
    )

    parser.add_argument(
        "--taz-lookup",
        default="data/raw/Jerusalem_TAZ_Accessibility.xlsx",
        help=(
            "TAZ lookup Excel file containing "
            "TAZ_1270 and NameEN."
        ),
    )

    parser.add_argument(
        "--top-activity-n",
        type=int,
        default=10,
        help=(
            "Number of most common held-out survey activity "
            "categories shown separately in Figure 5.1.3. "
            "Remaining categories are merged into Other."
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    # ========================================================
    # 1. Paths
    # ========================================================

    paper_results_dir = path_from_config(
        config,
        "paper_results_dir",
    )

    heldout_path = path_from_config(
        config,
        "trips_test_routed",
    )

    candidate_dir = Path(
        args.candidate_dir
    )

    if not candidate_dir.is_absolute():
        candidate_dir = (
            PROJECT_ROOT
            / candidate_dir
        )

    taz_lookup_path = Path(
        args.taz_lookup
    )

    if not taz_lookup_path.is_absolute():
        taz_lookup_path = (
            PROJECT_ROOT
            / taz_lookup_path
        )

    if not heldout_path.exists():
        raise FileNotFoundError(
            "Cannot find held-out survey data:\n"
            f"{heldout_path}"
        )

    if not candidate_dir.exists():
        raise FileNotFoundError(
            "Cannot find natural candidate pool:\n"
            f"{candidate_dir}"
        )

    if not taz_lookup_path.exists():
        raise FileNotFoundError(
            "Cannot find TAZ lookup file:\n"
            f"{taz_lookup_path}"
        )

    # ========================================================
    # 2. Load data
    # ========================================================

    survey = pd.read_parquet(
        heldout_path
    )

    synthetic = load_candidate_pool(
        candidate_dir
    )

    # ========================================================
    # 3. Load TAZ NameEN lookup
    # ========================================================

    taz_lookup = pd.read_excel(
        taz_lookup_path
    )

    required_taz_columns = [
        "TAZ_1270",
        "NameEN",
    ]

    missing_taz_columns = [
        column
        for column in required_taz_columns
        if column not in taz_lookup.columns
    ]

    if missing_taz_columns:

        raise KeyError(
            "Missing required TAZ lookup columns:\n"
            f"{missing_taz_columns}\n\n"
            "Available columns:\n"
            f"{taz_lookup.columns.tolist()}"
        )

    taz_lookup = (
        taz_lookup[
            [
                "TAZ_1270",
                "NameEN",
            ]
        ]
        .dropna(
            subset=[
                "TAZ_1270",
            ]
        )
        .copy()
    )

    taz_lookup[
        "_zone_key"
    ] = (
        taz_lookup[
            "TAZ_1270"
        ]
        .map(
            normalize_zone_id
        )
    )

    taz_lookup[
        "NameEN"
    ] = (
        taz_lookup[
            "NameEN"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    zone_name_map = (
        taz_lookup
        .drop_duplicates(
            subset=[
                "_zone_key"
            ]
        )
        .set_index(
            "_zone_key"
        )[
            "NameEN"
        ]
        .to_dict()
    )

    # ========================================================
    # 4. Resolve validation columns
    # ========================================================

    resolved = {}
    skipped_dimensions = []

    for dimension, specification in DIMENSIONS.items():

        survey_column = find_column(
            survey,
            specification[
                "survey_aliases"
            ],
        )

        synthetic_column = find_column(
            synthetic,
            specification[
                "synthetic_aliases"
            ],
        )

        if (
            survey_column is None
            or synthetic_column is None
        ):

            skipped_dimensions.append(
                {
                    "dimension":
                        dimension,

                    "survey_column":
                        survey_column,

                    "synthetic_column":
                        synthetic_column,
                }
            )

            continue

        resolved[
            dimension
        ] = {
            "survey_column":
                survey_column,

            "synthetic_column":
                synthetic_column,

            "binary":
                bool(
                    specification[
                        "binary"
                    ]
                ),
        }

    essential = [
        "Destination-zone distribution",
        "Activity distribution",
    ]

    missing_essential = [
        dimension
        for dimension in essential
        if dimension not in resolved
    ]

    if missing_essential:

        raise KeyError(
            "Missing essential validation dimensions:\n"
            f"{missing_essential}\n\n"
            "Held-out survey columns:\n"
            f"{survey.columns.tolist()}\n\n"
            "Synthetic-request columns:\n"
            f"{synthetic.columns.tolist()}"
        )

    if len(resolved) < 7:

        raise KeyError(
            "Fewer than 7 validation dimensions could be resolved.\n"
            f"Resolved dimensions: {list(resolved.keys())}\n\n"
            f"Skipped dimensions: {skipped_dimensions}\n\n"
            "Table 5.1.2 is designed to contain 7–8 meaningful "
            "validation dimensions."
        )

    # ========================================================
    # 5. Compute distributions and JSD
    # ========================================================

    table_rows = []
    distribution_frames = []
    prepared = {}

    for dimension, specification in resolved.items():

        survey_series = prepare_series(
            survey[
                specification[
                    "survey_column"
                ]
            ],
            binary=specification[
                "binary"
            ],
        )

        synthetic_series = prepare_series(
            synthetic[
                specification[
                    "synthetic_column"
                ]
            ],
            binary=specification[
                "binary"
            ],
        )

        prepared[
            dimension
        ] = {
            "survey":
                survey_series,

            "synthetic":
                synthetic_series,

            "binary":
                specification[
                    "binary"
                ],
        }

        jsd = js_divergence(
            survey_series,
            synthetic_series,
        )

        table_rows.append(
            {
                "Validation dimension":
                    dimension,

                "Metric":
                    "Jensen-Shannon divergence",

                "Result":
                    jsd,
            }
        )

        distribution_frames.append(
            distribution_table(
                survey_series,
                synthetic_series,
                dimension,
            )
        )

    # ========================================================
    # 6. Table 5.1.2
    # ========================================================

    table = pd.DataFrame(
        table_rows
    )

    order = {
        dimension: index
        for index, dimension
        in enumerate(
            DIMENSIONS.keys()
        )
    }

    table[
        "_order"
    ] = (
        table[
            "Validation dimension"
        ]
        .map(order)
    )

    table = (
        table
        .sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )

    output_dir = (
        paper_results_dir
        / "chapter5_1_model"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = (
        output_dir
        / "Table_5_1_2_distributional_validation_of_synthetic_requests.csv"
    )

    table.to_csv(
        table_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    # ========================================================
    # 7. Save complete category-level comparison data
    # ========================================================

    plot_data = pd.concat(
        distribution_frames,
        ignore_index=True,
    )

    plot_data_path = (
        output_dir
        / "Figure_5_1_3_distribution_plot_data.csv"
    )

    plot_data.to_csv(
        plot_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 8. Prepare Figure 5.1.3
    # ========================================================

    # --------------------------------------------------------
    # (a) Destination-zone distribution
    #
    # JSD uses ALL destination zones.
    # The figure shows the 15 most common survey destinations.
    # X-axis uses NameEN instead of TAZ IDs.
    # --------------------------------------------------------

    destination_dimension = (
        "Destination-zone distribution"
    )

    destination_data = (
        plot_data[
            plot_data[
                "dimension"
            ].eq(
                destination_dimension
            )
        ]
        .copy()
    )

    top_destination_zones = (
        destination_data
        .sort_values(
            "held_out_survey_share",
            ascending=False,
        )
        .head(10)[
            "category"
        ]
        .tolist()
    )

    destination_plot = (
        destination_data[
            destination_data[
                "category"
            ].isin(
                top_destination_zones
            )
        ]
        .copy()
    )

    destination_plot[
        "category"
    ] = pd.Categorical(
        destination_plot[
            "category"
        ],
        categories=
            top_destination_zones,
        ordered=True,
    )

    destination_plot = (
        destination_plot
        .sort_values(
            "category"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Map destination zone ID -> NameEN
    # --------------------------------------------------------

    destination_plot[
        "_zone_key"
    ] = (
        destination_plot[
            "category"
        ]
        .astype(str)
        .map(
            normalize_zone_id
        )
    )

    destination_plot[
        "NameEN"
    ] = (
        destination_plot[
            "_zone_key"
        ]
        .map(
            zone_name_map
        )
    )

    missing_name = (
        destination_plot[
            "NameEN"
        ].isna()
        |
        destination_plot[
            "NameEN"
        ].astype(str).str.strip().eq("")
    )

    if missing_name.any():

        missing_zone_ids = (
            destination_plot.loc[
                missing_name,
                "_zone_key",
            ]
            .tolist()
        )

        raise KeyError(
            "Some destination zones in Figure 5.1.3 "
            "could not be mapped to NameEN:\n"
            f"{missing_zone_ids}\n\n"
            "Check whether destination_zone_id uses the same "
            "TAZ coding as Jerusalem_TAZ_Accessibility.xlsx::TAZ_1270."
        )

    destination_plot[
        "display_name"
    ] = (
        destination_plot[
            "NameEN"
        ]
        .map(
            lambda value:
                wrap_label(
                    value,
                    width=15,
                )
        )
    )

    # --------------------------------------------------------
    # (b) Activity distribution
    #
    # IMPORTANT:
    # JSD in Table 5.1.2 still uses ALL activity categories.
    #
    # Figure:
    # Top N categories + Other
    # --------------------------------------------------------

    activity_dimension = (
        "Activity distribution"
    )

    activity_all = (
        plot_data[
            plot_data[
                "dimension"
            ].eq(
                activity_dimension
            )
        ]
        .copy()
        .sort_values(
            "held_out_survey_share",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    top_activity_n = max(
        1,
        int(
            args.top_activity_n
        ),
    )

    activity_top = (
        activity_all
        .head(
            top_activity_n
        )
        .copy()
    )

    remaining_activity = (
        activity_all
        .iloc[
            top_activity_n:
        ]
        .copy()
    )

    if not remaining_activity.empty:

        activity_other = pd.DataFrame(
            {
                "dimension": [
                    activity_dimension
                ],

                "category": [
                    "Other"
                ],

                "held_out_survey_share": [
                    remaining_activity[
                        "held_out_survey_share"
                    ].sum()
                ],

                "synthetic_share": [
                    remaining_activity[
                        "synthetic_share"
                    ].sum()
                ],
            }
        )

        activity_plot = pd.concat(
            [
                activity_top,
                activity_other,
            ],
            ignore_index=True,
        )

    else:

        activity_plot = (
            activity_top.copy()
        )

    # --------------------------------------------------------
    # (c) Key population-group 
    # --------------------------------------------------------

    equity_dimensions = [
        (
            "Low-income status",
            "Low-income",
        ),
        (
            "Car availability",
            "No car",
        ),
        (
            "Older-traveller status",
            "Older",
        ),
        (
            "Student status",
            "Student",
        ),
        (
            "Service-gap-origin status",
            "Gap origin",
        ),
    ]

    equity_rows = []

    for dimension, label in equity_dimensions:

        if dimension not in prepared:
            continue

        equity_rows.append(
            {
                "group":
                    label,

                "held_out_survey_share":
                    positive_share(
                        prepared[
                            dimension
                        ][
                            "survey"
                        ]
                    ),

                "synthetic_share":
                    positive_share(
                        prepared[
                            dimension
                        ][
                            "synthetic"
                        ]
                    ),
            }
        )

    equity_plot = pd.DataFrame(
        equity_rows
    )

    # ========================================================
    # 9. Figure 5.1.3
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 5.4),
    )

    width = 0.38

    # --------------------------------------------------------
    # Panel (a): Destination zones
    # --------------------------------------------------------

    ax = axes[0]

    x = np.arange(
        len(
            destination_plot
        )
    )

    ax.bar(
        x - width / 2,
        destination_plot[
            "held_out_survey_share"
        ] * 100,
        width,
        label="Held-out survey",
    )

    ax.bar(
        x + width / 2,
        destination_plot[
            "synthetic_share"
        ] * 100,
        width,
        label="Synthetic requests",
    )

    ax.set_title(
        "(a) Destination-zone distribution"
    )

    ax.set_ylabel(
        "Percentage of requests (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        destination_plot[
            "display_name"
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    # --------------------------------------------------------
    # Panel (b): Activity distribution
    # --------------------------------------------------------

    ax = axes[1]

    x = np.arange(
        len(
            activity_plot
        )
    )

    ax.bar(
        x - width / 2,
        activity_plot[
            "held_out_survey_share"
        ] * 100,
        width,
        label="Held-out survey",
    )

    ax.bar(
        x + width / 2,
        activity_plot[
            "synthetic_share"
        ] * 100,
        width,
        label="Synthetic requests",
    )

    ax.set_title(
        "(b) Activity distribution"
    )

    ax.set_ylabel(
        "Percentage of requests (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        activity_plot[
            "category"
        ],
        rotation=40,
        ha="right",
        fontsize=8,
    )

    # --------------------------------------------------------
    # Panel (c): Key population-group 
    # --------------------------------------------------------

    ax = axes[2]

    x = np.arange(
        len(
            equity_plot
        )
    )

    ax.bar(
        x - width / 2,
        equity_plot[
            "held_out_survey_share"
        ] * 100,
        width,
        label="Held-out survey",
    )

    ax.bar(
        x + width / 2,
        equity_plot[
            "synthetic_share"
        ] * 100,
        width,
        label="Synthetic requests",
    )

    ax.set_title(
        "(c) Key population-group"
    )

    ax.set_ylabel(
        "Percentage of requests (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        equity_plot[
            "group"
        ],
        rotation=30,
        ha="right",
        fontsize=8,
    )

    # ========================================================
    # 10. Formatting
    # ========================================================

    for ax in axes:

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        ax.grid(
            axis="y",
            alpha=0.20,
            linewidth=0.6,
        )

        ax.set_axisbelow(
            True
        )

    fig.tight_layout()

    figure_png = (
        output_dir
        / "Figure_5_1_3_representative_distributional_comparison.png"
    )

    figure_pdf = (
        output_dir
        / "Figure_5_1_3_representative_distributional_comparison.pdf"
    )

    fig.savefig(
        figure_png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        figure_pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ========================================================
    # 11. Console audit information
    # ========================================================

    print()
    print("=" * 88)

    print(
        "Table 5.1.2 — Distributional validation "
        "of synthetic travel requests"
    )

    print("=" * 88)

    print(
        f"Runner version        : {RUNNER_VERSION}"
    )

    print(
        f"Held-out survey       : {heldout_path}"
    )

    print(
        f"Natural synthetic pool: {candidate_dir}"
    )

    print(
        f"TAZ lookup            : {taz_lookup_path}"
    )

    print(
        f"Survey rows           : {len(survey):,}"
    )

    print(
        f"Synthetic rows        : {len(synthetic):,}"
    )

    print(
        f"Validated dimensions  : {len(resolved)}"
    )

    print(
        f"Activity display      : Top {top_activity_n} + Other"
    )

    print(
        "Destination labels    : NameEN"
    )

    print()

    print(
        "Resolved columns:"
    )

    for dimension, specification in resolved.items():

        print(
            f"  {dimension}: "
            f"survey={specification['survey_column']} | "
            f"synthetic={specification['synthetic_column']}"
        )

    if skipped_dimensions:

        print()
        print(
            "Skipped optional dimensions:"
        )

        for item in skipped_dimensions:

            print(
                "  "
                f"{item['dimension']} | "
                f"survey={item['survey_column']} | "
                f"synthetic={item['synthetic_column']}"
            )

    print()

    print(
        table.to_string(
            index=False
        )
    )

    print()

    print(
        "Destination-zone labels used in Figure 5.1.3:"
    )

    for _, row in destination_plot.iterrows():

        print(
            f"  {row['_zone_key']} -> {row['NameEN']}"
        )

    print()

    print(
        f"Table saved           : {table_path}"
    )

    print(
        f"Figure saved          : {figure_png}"
    )

    print(
        f"Figure PDF saved      : {figure_pdf}"
    )

    print(
        f"Plot data saved       : {plot_data_path}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()