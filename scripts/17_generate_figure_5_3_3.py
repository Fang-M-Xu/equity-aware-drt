from __future__ import annotations

import argparse
from pathlib import Path
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

RUNNER_VERSION = "3.1-group-order-and-colors"

GROUP_SPECS = [
    ("is_gap", "1", "Gap-origin"),
    ("is_no_car", "1", "No car"),
    ("is_low_income", "1", "Low income"),
    ("is_older", "1", "Older person"),
    ("is_student", "1", "Student"),
    ("sector_group", "Arab", "Arab sector"),
    ("sector_group", "Secular", "Secular sector"),
    ("sector_group", "Ultra-Orthodox", "Ultra-Orthodox sector"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 5.3.3: service rates across population groups under Proposed-full."
    )
    parser.add_argument(
        "--group-metrics",
        default="outputs/paper_results/chapter5_routing/group_service_metrics.csv",
    )
    parser.add_argument(
        "--scenario-metrics",
        default="outputs/paper_results/chapter5_routing/routing_scenario_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_routing",
    )
    parser.add_argument("--method", default="proposed_full")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def normalize_group_value(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def bootstrap_mean_ci(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_group_summary(group: pd.DataFrame, method: str, repetitions: int, seed: int) -> pd.DataFrame:
    required = {"method", "scenario_id", "group_attribute", "group_value", "service_rate"}
    missing = sorted(required - set(group.columns))
    if missing:
        raise KeyError(f"group_service_metrics.csv is missing required columns: {missing}")

    frame = group.loc[group["method"].astype(str).eq(method)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for method {method!r}.")

    frame["scenario_id"] = pd.to_numeric(frame["scenario_id"], errors="coerce")
    frame["service_rate"] = pd.to_numeric(frame["service_rate"], errors="coerce")
    frame["group_attribute"] = frame["group_attribute"].astype(str)
    frame["group_value_norm"] = frame["group_value"].map(normalize_group_value)
    frame = frame.dropna(subset=["scenario_id", "service_rate"]).copy()

    rng = np.random.default_rng(seed)
    rows = []
    for attribute, value, label in GROUP_SPECS:
        subset = frame.loc[
            frame["group_attribute"].eq(attribute)
            & frame["group_value_norm"].eq(value)
        ].copy()
        if subset.empty:
            continue

        scenario_values = (
            subset.groupby("scenario_id", as_index=False)["service_rate"]
            .mean()["service_rate"]
            .to_numpy(dtype=float)
        )
        mean, lower, upper = bootstrap_mean_ci(scenario_values, repetitions, rng)
        rows.append(
            {
                "group": label,
                "mean": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_scenarios": int(len(scenario_values)),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("None of the requested population groups were found.")
    return result


def overall_service_rate(scenario: pd.DataFrame, method: str) -> float:
    required = {"method", "scenario_id", "service_rate"}
    missing = sorted(required - set(scenario.columns))
    if missing:
        raise KeyError(f"routing_scenario_metrics.csv is missing required columns: {missing}")

    subset = scenario.loc[scenario["method"].astype(str).eq(method)].copy()
    if subset.empty:
        raise ValueError(f"No scenario metrics found for method {method!r}.")
    values = pd.to_numeric(subset["service_rate"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("No valid overall service-rate values were found.")
    return float(values.mean())


def plot_figure(summary: pd.DataFrame, overall: float, output_base: Path, dpi: int) -> None:
    order = [
        "Gap-origin",
        "No car",
        "Low income",
        "Older person",
        "Student",
        "Ultra-Orthodox sector",
        "Arab sector",
        "Secular sector",
    ]
    order_map = {label: i for i, label in enumerate(order)}
    frame = summary.copy()
    frame["_order"] = frame["group"].map(order_map)
    frame = frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    y = np.arange(len(frame))
    means = frame["mean"].to_numpy(dtype=float)
    lower = frame["ci95_lower"].to_numpy(dtype=float)
    upper = frame["ci95_upper"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    higher_color = "#7B4EA3"
    lower_color = "#F58518"
    overall_color = "#1F4E79"

    ax.axvline(
        overall,
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        color=overall_color,
        label=f"Overall service rate = {overall:.3f}",
    )

    for i in range(len(frame)):
        color = higher_color if means[i] >= overall else lower_color
        ax.errorbar(
            means[i],
            y[i],
            xerr=np.array([
                [means[i] - lower[i]],
                [upper[i] - means[i]],
            ]),
            fmt="o",
            markersize=6.5,
            linewidth=1.6,
            capsize=0,
            color=color,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )

    for yy, mean, hi in zip(y, means, upper):
        ax.annotate(
            f"{mean:.3f}",
            xy=(hi, yy),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(frame["group"])
    ax.invert_yaxis()
    ax.set_xlabel("Service rate")
    ax.set_ylabel("Population group")
    ax.set_title("Service rates across population groups under Proposed-full", pad=12)
    ax.grid(axis="x", alpha=0.20, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color=higher_color, markerfacecolor=higher_color,
               markeredgecolor=higher_color, linewidth=1.6, label="Above overall service rate"),
        Line2D([0], [0], marker="o", color=lower_color, markerfacecolor=lower_color,
               markeredgecolor=lower_color, linewidth=1.6, label="Below overall service rate"),
        Line2D([0], [0], color=overall_color, linestyle=(0, (4, 3)), linewidth=1.4,
               label=f"Overall service rate = {overall:.3f}"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    finite = np.concatenate([lower[np.isfinite(lower)], upper[np.isfinite(upper)]])
    if finite.size:
        xmin = max(0.0, float(finite.min()) - 0.05)
        xmax = min(1.0, float(finite.max()) + 0.09)
        if xmax <= xmin:
            xmax = min(1.0, xmin + 0.2)
        ax.set_xlim(xmin, xmax)

    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    group_path = resolve(args.group_metrics)
    scenario_path = resolve(args.scenario_metrics)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not group_path.exists():
        raise FileNotFoundError(f"Group metrics not found: {group_path}\nRun 17_5_evaluate_routing_equity.py first.")
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario metrics not found: {scenario_path}\nRun 17_5_evaluate_routing_equity.py first.")

    group = pd.read_csv(group_path)
    scenario = pd.read_csv(scenario_path)
    summary = build_group_summary(group, args.method, args.bootstrap_repetitions, args.seed)
    overall = overall_service_rate(scenario, args.method)

    data_path = output_dir / "Figure_5_3_3_group_service_rates_data.csv"
    summary.to_csv(data_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    output_base = output_dir / "Figure_5_3_3_group_service_rates"
    plot_figure(summary, overall, output_base, args.dpi)

    print(f"Saved: {output_base.with_suffix('.png')}")
    print(f"Saved: {output_base.with_suffix('.pdf')}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
