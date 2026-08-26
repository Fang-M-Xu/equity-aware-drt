from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT = Path(
    r"outputs/alns_competitive/proposed_full_35"
)

OUTPUT = Path(
    r"outputs/metrics_competitive/proposed_full_35_precheck.csv"
)


# ============================================================
# Main
# ============================================================

def main() -> None:

    files = sorted(
        ROOT.glob(
            "scenario_*/seed_*/request_results.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No request_results.parquet found under:\n{ROOT}"
        )

    rows = []

    for path in files:

        df = pd.read_parquet(path)

        required = [
            "served",
            "within_15min",
            "is_gap",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise KeyError(
                f"Missing columns in {path}:\n{missing}"
            )

        served = (
            df["served"]
            .fillna(False)
            .astype(bool)
        )

        within = (
            df["within_15min"]
            .fillna(False)
            .astype(bool)
        )

        gap = (
            df["is_gap"]
            .fillna(False)
            .astype(bool)
        )

        drt_accessible = (
            served & within
        )

        total_requests = len(df)

        served_requests = int(
            served.sum()
        )

        gap_requests = int(
            gap.sum()
        )

        service_rate = (
            served.mean()
            if total_requests > 0
            else float("nan")
        )

        within_15min_among_served = (
            within[served].mean()
            if served_requests > 0
            else float("nan")
        )

        overall_drt_accessibility = (
            drt_accessible.mean()
            if total_requests > 0
            else float("nan")
        )

        gap_drt_accessibility = (
            drt_accessible[gap].mean()
            if gap_requests > 0
            else float("nan")
        )

        scenario_name = path.parents[1].name
        seed_name = path.parent.name

        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed_name,
                "total_requests": total_requests,
                "served_requests": served_requests,
                "gap_requests": gap_requests,
                "service_rate": service_rate,
                "within_15min_among_served":
                    within_15min_among_served,
                "overall_drt_accessibility":
                    overall_drt_accessibility,
                "gap_origin_drt_accessibility":
                    gap_drt_accessibility,
            }
        )

    result = pd.DataFrame(rows)

    # ========================================================
    # Save scenario-level results
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 90)
    print(
        "Proposed-full 35-vehicle pre-check"
    )
    print("=" * 90)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("-" * 90)
    print("AVERAGES ACROSS SCENARIOS")
    print("-" * 90)

    print(
        f"n_scenarios = {len(result)}"
    )

    print(
        "mean_service_rate = "
        f"{result['service_rate'].mean():.4f}"
    )

    print(
        "mean_within_15min_among_served = "
        f"{result['within_15min_among_served'].mean():.4f}"
    )

    print(
        "mean_overall_drt_accessibility = "
        f"{result['overall_drt_accessibility'].mean():.4f}"
    )

    print(
        "mean_gap_origin_drt_accessibility = "
        f"{result['gap_origin_drt_accessibility'].mean():.4f}"
    )

    print()
    print(
        f"Scenario-level CSV saved to:\n{OUTPUT}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()