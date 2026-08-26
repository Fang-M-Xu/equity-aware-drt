from __future__ import annotations

"""Generate Table 5.4.1: overall multimodal accessibility and travel time.

Default input:
    outputs/multimodal_evaluation/paper_table_overall_modes.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODE_ORDER = [
    ("DRT proposed full", "Proposed DRT"),
    ("Walking", "Direct walking"),
    ("Public transit", "GTFS public transit"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/multimodal_evaluation/paper_table_overall_modes.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_rq3",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")].copy()
    if "index" in frame.columns:
        frame = frame.drop(columns="index")

    required = {
        "mode",
        "n_requests",
        "availability_or_access_rate",
        "within_threshold_rate",
        "mean_time_sec",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(
            f"Missing columns in {path}: {missing}\n"
            f"Available columns: {list(frame.columns)}"
        )
    return frame


def build_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_mode, label in MODE_ORDER:
        subset = frame.loc[frame["mode"].astype(str).str.strip().eq(raw_mode)]
        if subset.empty:
            raise ValueError(f"Mode not found in input: {raw_mode}")
        row = subset.iloc[0]
        rows.append(
            {
                "Mode": label,
                "Requests": int(row["n_requests"]),
                "Accepted/valid itinerary rate": float(
                    row["availability_or_access_rate"]
                ),
                "Within-15-minute rate": float(row["within_threshold_rate"]),
                "Mean door-to-door time (min)": float(row["mean_time_sec"]) / 60.0,
            }
        )
    return pd.DataFrame(rows)


def save_latex(table: pd.DataFrame, path: Path) -> None:
    latex = table.to_latex(
        index=False,
        escape=True,
        column_format="lrrrr",
        caption="Overall multimodal accessibility and travel-time comparison.",
        label="tab:rq3_overall_multimodal",
        formatters={
            "Accepted/valid itinerary rate": lambda x: f"{x:.3f}",
            "Within-15-minute rate": lambda x: f"{x:.3f}",
            "Mean door-to-door time (min)": lambda x: f"{x:.2f}",
        },
    )
    note = (
        "\n% Note: for DRT, the accepted/valid rate is the expected service "
        "probability across solver seeds and the mean time is calculated for "
        "observed served journeys. Walking and transit means are calculated "
        "from their valid benchmark itineraries.\n"
    )
    path.write_text(latex + note, encoding="utf-8")


def save_image(table: pd.DataFrame, path: Path, dpi: int) -> None:
    display = table.copy()
    display["Accepted/valid itinerary rate"] = display[
        "Accepted/valid itinerary rate"
    ].map(lambda x: f"{x:.3f}")
    display["Within-15-minute rate"] = display[
        "Within-15-minute rate"
    ].map(lambda x: f"{x:.3f}")
    display["Mean door-to-door time (min)"] = display[
        "Mean door-to-door time (min)"
    ].map(lambda x: f"{x:.2f}")

    fig, ax = plt.subplots(figsize=(11.8, 3.0))
    ax.axis("off")
    mpl_table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9.2)
    mpl_table.scale(1.0, 1.55)

    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_text_props(ha="left")
        if row in (0, len(display)):
            cell.set_linewidth(0.8)

    fig.text(
        0.02,
        0.035,
        "Note: DRT mean time is based on served journeys; walking and transit "
        "means are based on valid benchmark itineraries.",
        fontsize=8.1,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}\n"
            "Generate it with scripts/17_analyze_drt_vs_benchmarks.py."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(load_table(input_path))
    csv_path = output_dir / "Table_5_4_1_overall_multimodal.csv"
    tex_path = output_dir / "Table_5_4_1_overall_multimodal.tex"
    png_path = output_dir / "Table_5_4_1_overall_multimodal.png"

    table.to_csv(csv_path, index=False)
    save_latex(table, tex_path)
    save_image(table, png_path, args.dpi)

    print(table.to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {png_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
