from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"

GROUP_SPECS = [
    {
        "label": "Arab sector",
        "variables": ["sector_group", "sector", "sectr"],
        "values": ["Arab", "Arab sector", "1"],
    },
    {
        "label": "Gap-origin",
        "variables": ["is_gap", "gap_i", "gap_origin", "service_gap"],
        "values": ["1", "True", "Gap-origin"],
    },
    {
        "label": "Low income",
        "variables": ["is_low_income", "low_income", "income_group"],
        "values": ["1", "True", "Low income"],
    },
    {
        "label": "Older person",
        "variables": ["is_older", "older", "older_person", "age_65_plus"],
        "values": ["1", "True", "Older person"],
    },
    {
        "label": "Student",
        "variables": ["is_student", "student", "student_status"],
        "values": ["1", "True", "Student"],
    },
    {
        "label": "Secular sector",
        "variables": ["sector_group", "sector", "sectr"],
        "values": ["Secular", "Secular sector"],
    },
    {
        "label": "No car",
        "variables": ["is_no_car", "no_car", "car_ownership"],
        "values": ["1", "True", "No car"],
    },
    {
        "label": "Ultra-Orthodox sector",
        "variables": ["sector_group", "sector", "sectr"],
        "values": [
            "Ultra-Orthodox",
            "Ultra Orthodox",
            "Ultra-Orthodox sector",
            "Haredi",
        ],
    },
]


COLUMN_ALIASES = {
    "group_variable": [
        "group_variable",
        "group_attribute",
        "attribute",
        "group_name",
    ],
    "group_value": [
        "group_value",
        "value",
        "group",
        "group_label",
    ],
    "n_requests": [
        "n_requests",
        "requests",
        "group_requests",
        "n",
        "count",
    ],
    "drt_rate": [
        "drt_expected_access_rate",
        "drt_access_rate",
        "drt_within_threshold_rate",
        "drt_within_15_rate",
        "drt_rate",
    ],
    "walking_rate": [
        "walking_access_rate",
        "walk_access_rate",
        "walking_within_threshold_rate",
        "walking_within_15_rate",
        "walking_rate",
    ],
    "transit_rate": [
        "transit_access_rate",
        "public_transit_access_rate",
        "gtfs_access_rate",
        "transit_within_threshold_rate",
        "transit_within_15_rate",
        "transit_rate",
    ],
}


def canonical(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"1.0", "true", "yes", "y"}:
        text = "1"
    return "".join(character for character in text if character.isalnum())


def first_existing(
    columns: Iterable[object],
    aliases: Iterable[str],
) -> str | None:
    original = {str(column): str(column) for column in columns}
    lowered = {str(column).lower(): str(column) for column in columns}

    for alias in aliases:
        if alias in original:
            return original[alias]
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def load_group_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[
        :,
        ~frame.columns.astype(str).str.startswith("Unnamed"),
    ].copy()

    rename: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        found = first_existing(frame.columns, aliases)
        if found is not None:
            rename[found] = target

    frame = frame.rename(columns=rename)

    required = {
        "group_variable",
        "group_value",
        "drt_rate",
        "walking_rate",
        "transit_rate",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(
            f"Missing required columns in {path}: {missing}\n"
            f"Available columns: {list(frame.columns)}"
        )

    for column in [
        "drt_rate",
        "walking_rate",
        "transit_rate",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if "n_requests" in frame.columns:
        frame["n_requests"] = pd.to_numeric(
            frame["n_requests"],
            errors="coerce",
        )
    else:
        frame["n_requests"] = np.nan

    frame["_variable_key"] = frame["group_variable"].map(canonical)
    frame["_value_key"] = frame["group_value"].map(canonical)

    return frame


def find_group_row(
    frame: pd.DataFrame,
    spec: dict[str, object],
) -> pd.Series:
    variable_keys = {
        canonical(value)
        for value in spec["variables"]
    }
    value_keys = {
        canonical(value)
        for value in spec["values"]
    }

    subset = frame.loc[
        frame["_variable_key"].isin(variable_keys)
        & frame["_value_key"].isin(value_keys)
    ]

    if subset.empty:
        available = (
            frame[["group_variable", "group_value"]]
            .drop_duplicates()
            .sort_values(["group_variable", "group_value"])
            .to_dict("records")
        )
        raise ValueError(
            f"Could not find the requested group: {spec['label']}.\n"
            f"Available group definitions: {available}"
        )

    if len(subset) > 1:
        # Prefer the row with the largest reported request count when duplicate
        # summaries exist.
        subset = subset.sort_values(
            "n_requests",
            ascending=False,
            na_position="last",
        )

    return subset.iloc[0]


def build_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for spec in GROUP_SPECS:
        row = find_group_row(frame, spec)

        drt = float(row["drt_rate"])
        walking = float(row["walking_rate"])
        transit = float(row["transit_rate"])

        requests = row.get("n_requests", np.nan)
        requests_value = (
            int(round(float(requests)))
            if pd.notna(requests)
            else pd.NA
        )

        rows.append(
            {
                "Group": spec["label"],
                "Requests": requests_value,
                "DRT within-15-minute rate": drt,
                "Walking within-15-minute rate": walking,
                "GTFS transit within-15-minute rate": transit,
                "DRT - walking (pp)": 100.0 * (drt - walking),
                "DRT - transit (pp)": 100.0 * (drt - transit),
            }
        )

    return pd.DataFrame(rows)


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.4.3: group-level multimodal accessibility."
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "outputs/metrics/multimodal_evaluation/"
            "group_drt_vs_benchmarks.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_multimodal",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--show-values",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def plot_figure(
    table: pd.DataFrame,
    output: Path,
    dpi: int,
    show_values: bool,
) -> None:
    groups = table["Group"].tolist()
    y = np.arange(len(groups), dtype=float)

    modes = [
        (
            "Walking",
            "Walking within-15-minute rate",
            "o",
            0.18,
        ),
        (
            "GTFS public transit",
            "GTFS transit within-15-minute rate",
            "s",
            0.00,
        ),
        (
            "Proposed DRT",
            "DRT within-15-minute rate",
            "D",
            -0.18,
        ),
    ]

    default_colors = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.6))

    # A light range line emphasizes the gap between the lowest and highest
    # mode for each audited group.
    for row_index, (_, row) in enumerate(table.iterrows()):
        values = [
            float(row["Walking within-15-minute rate"]),
            float(row["GTFS transit within-15-minute rate"]),
            float(row["DRT within-15-minute rate"]),
        ]
        ax.plot(
            [min(values), max(values)],
            [y[row_index], y[row_index]],
            linewidth=2.5,
            alpha=0.16,
            solid_capstyle="round",
            zorder=1,
        )

    for mode_index, (
        label,
        column,
        marker,
        vertical_offset,
    ) in enumerate(modes):
        values = table[column].to_numpy(dtype=float)
        mode_y = y + vertical_offset

        ax.scatter(
            values,
            mode_y,
            s=58,
            marker=marker,
            label=label,
            color=default_colors[
                mode_index % len(default_colors)
            ],
            zorder=3,
        )

        if show_values:
            for value, y_value in zip(values, mode_y):
                # Keep the text readable near the right edge.
                if value >= 0.965:
                    horizontal_alignment = "right"
                    x_offset = -7
                else:
                    horizontal_alignment = "left"
                    x_offset = 7

                ax.annotate(
                    f"{value:.1%}",
                    (value, y_value),
                    xytext=(x_offset, 0),
                    textcoords="offset points",
                    va="center",
                    ha=horizontal_alignment,
                    fontsize=8.0,
                )

    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()

    all_rates = table[
        [
            "DRT within-15-minute rate",
            "Walking within-15-minute rate",
            "GTFS transit within-15-minute rate",
        ]
    ].to_numpy(dtype=float)

    lower = max(
        0.0,
        float(np.nanmin(all_rates)) - 0.06,
    )
    upper = min(
        1.04,
        max(1.0, float(np.nanmax(all_rates)) + 0.04),
    )

    ax.set_xlim(lower, upper)
    ax.set_xlabel("Within-15-minute accessibility rate")
    ax.set_ylabel("Equity-audit group")
    ax.set_title(
        (
            "Group-level multimodal accessibility under DRT, "
            "walking, and GTFS transit"
        ),
        loc="left",
        pad=14,
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.25,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=True,
    )

    # The explanatory note is provided in the paper caption rather than
    # inside the figure.
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}\n"
            "Generate the formal multimodal group output first."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_group_table(
        load_group_results(input_path)
    )

    plot_data_path = (
        output_dir
        / "Figure_5_4_3_group_multimodal_plot_data.csv"
    )
    output = (
        output_dir
        / "Figure_5_4_3_group_multimodal_accessibility.png"
    )

    table.to_csv(plot_data_path, index=False)
    plot_figure(
        table,
        output,
        args.dpi,
        args.show_values,
    )

    print(table.to_string(index=False))
    print(f"Saved: {plot_data_path}")
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.pdf')}")
    print(f"Saved: {output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
