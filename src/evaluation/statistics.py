from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def paired_wilcoxon_table(
    scenario_summary: pd.DataFrame,
    proposed_method: str,
    baseline_methods: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in metrics:
        proposed = (
            scenario_summary[scenario_summary["method"].eq(proposed_method)]
            .set_index("scenario_id")[metric]
            .sort_index()
        )
        for baseline in baseline_methods:
            baseline_values = (
                scenario_summary[scenario_summary["method"].eq(baseline)]
                .set_index("scenario_id")[metric]
                .sort_index()
            )
            common = proposed.index.intersection(baseline_values.index)
            differences = proposed.loc[common] - baseline_values.loc[common]
            differences = differences.dropna()
            if len(differences) < 2 or np.allclose(differences, 0):
                statistic, p_value = np.nan, 1.0
            else:
                statistic, p_value = wilcoxon(differences)
            rows.append(
                {
                    "metric": metric,
                    "proposed": proposed_method,
                    "baseline": baseline,
                    "pairs": len(differences),
                    "mean_difference": float(differences.mean()) if len(differences) else np.nan,
                    "median_difference": float(differences.median()) if len(differences) else np.nan,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = multipletests(result["p_value"], method="holm")[1]
    return result


def bootstrap_mean_ci(values: pd.Series, seed: int, iterations: int = 5000) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        means[index] = rng.choice(clean, size=len(clean), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
