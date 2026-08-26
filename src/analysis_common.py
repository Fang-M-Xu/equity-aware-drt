from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import friedmanchisquare, wilcoxon


TRUE_VALUES = {"1", "true", "t", "yes", "y", "served", "accepted", "available", "feasible"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "unserved", "rejected", "unavailable", "infeasible", ""}


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def find_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    required: bool = False,
    explicit: str | None = None,
) -> str | None:
    lookup = {normalize_name(column): str(column) for column in frame.columns}
    if explicit:
        key = normalize_name(explicit)
        if key in lookup:
            return lookup[key]
        if required:
            raise ValueError(f"Column '{explicit}' not found. Available: {list(frame.columns)}")
        return None
    for alias in aliases:
        key = normalize_name(alias)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(f"Could not detect any of {list(aliases)}. Available: {list(frame.columns)}")
    return None


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    text = series.fillna("").astype(str).str.strip().str.lower()
    result = text.isin(TRUE_VALUES)
    unknown = ~(text.isin(TRUE_VALUES | FALSE_VALUES))
    if unknown.any():
        numeric = pd.to_numeric(text.where(unknown), errors="coerce")
        result.loc[unknown & numeric.notna()] = numeric.loc[unknown & numeric.notna()].ne(0)
    return result.astype(bool)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def canonical_id(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        text = str(value).strip()
        return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text
    return series.map(convert)


def parse_scenario_from_path(path: Path) -> int | None:
    for pattern in [r"scenario[_=-]?0*(\d+)", r"scenario=(\d+)"]:
        match = re.search(pattern, str(path), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def parse_seed_from_path(path: Path) -> int | None:
    for pattern in [r"seed[_=-]?(\d+)", r"rep_\d+_seed_(\d+)"]:
        match = re.search(pattern, str(path), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator and denominator > 0 else float("nan")


def finite_values(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_mean_ci(
    values: Iterable[Any],
    repetitions: int = 5000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    arr = finite_values(values)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    estimate = float(arr.mean())
    if n == 1:
        return estimate, estimate, estimate, 1
    rng = np.random.default_rng(seed)
    sampled = rng.choice(arr, size=(repetitions, n), replace=True).mean(axis=1)
    return (
        estimate,
        float(np.quantile(sampled, alpha / 2)),
        float(np.quantile(sampled, 1 - alpha / 2)),
        n,
    )


def paired_bootstrap_difference_ci(
    left: Iterable[Any],
    right: Iterable[Any],
    repetitions: int = 5000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    pair = pd.DataFrame({"left": list(left), "right": list(right)}).apply(pd.to_numeric, errors="coerce").dropna()
    if pair.empty:
        return float("nan"), float("nan"), float("nan"), 0
    diff = (pair["left"] - pair["right"]).to_numpy(dtype=float)
    n = len(diff)
    estimate = float(diff.mean())
    if n == 1:
        return estimate, estimate, estimate, 1
    rng = np.random.default_rng(seed)
    sampled = rng.choice(diff, size=(repetitions, n), replace=True).mean(axis=1)
    return estimate, float(np.quantile(sampled, alpha / 2)), float(np.quantile(sampled, 1 - alpha / 2)), n


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.where(np.isfinite(values))[0]
    if len(valid) == 0:
        return result.tolist()
    ordered = valid[np.argsort(values[valid])]
    running = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        running = max(running, (m - rank) * values[index])
        result[index] = min(1.0, running)
    return result.tolist()


def cliffs_delta(left: Iterable[Any], right: Iterable[Any]) -> float:
    x = finite_values(left)
    y = finite_values(right)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    total = 0
    # Chunking avoids constructing a very large full pairwise matrix.
    for start in range(0, len(x), 256):
        chunk = x[start:start + 256, None]
        total += int((chunk > y[None, :]).sum() - (chunk < y[None, :]).sum())
    return total / float(len(x) * len(y))


def paired_wilcoxon(left: Iterable[Any], right: Iterable[Any]) -> dict[str, float | int]:
    pair = pd.DataFrame({"left": list(left), "right": list(right)}).apply(pd.to_numeric, errors="coerce").dropna()
    if pair.empty:
        return {"pairs": 0, "statistic": float("nan"), "p_value": float("nan")}
    difference = pair["left"] - pair["right"]
    if np.allclose(difference.to_numpy(dtype=float), 0.0):
        return {"pairs": len(pair), "statistic": float("nan"), "p_value": 1.0}
    try:
        result = wilcoxon(pair["left"], pair["right"], zero_method="pratt", alternative="two-sided")
        return {"pairs": len(pair), "statistic": float(result.statistic), "p_value": float(result.pvalue)}
    except ValueError:
        return {"pairs": len(pair), "statistic": float("nan"), "p_value": 1.0}


def friedman_test(
    frame: pd.DataFrame,
    *,
    subject_column: str,
    method_column: str,
    value_column: str,
) -> dict[str, float | int]:
    pivot = frame.pivot_table(index=subject_column, columns=method_column, values=value_column, aggfunc="mean").dropna()
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {"subjects": int(pivot.shape[0]), "methods": int(pivot.shape[1]), "statistic": float("nan"), "p_value": float("nan")}
    samples = [pivot[column].to_numpy(dtype=float) for column in pivot.columns]
    result = friedmanchisquare(*samples)
    return {
        "subjects": int(pivot.shape[0]),
        "methods": int(pivot.shape[1]),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def categorical_distribution(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("__MISSING__").astype(str)
    counts = cleaned.value_counts(dropna=False)
    return counts / counts.sum() if counts.sum() else counts.astype(float)


def aligned_probabilities(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    categories = sorted(set(left.index.astype(str)) | set(right.index.astype(str)))
    p = left.reindex(categories, fill_value=0.0).to_numpy(dtype=float)
    q = right.reindex(categories, fill_value=0.0).to_numpy(dtype=float)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    return p, q, categories


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()))


def jain_index(values: Iterable[Any]) -> float:
    arr = finite_values(values)
    arr = arr[arr >= 0]
    if len(arr) == 0 or np.allclose(arr, 0):
        return float("nan")
    return float(arr.sum() ** 2 / (len(arr) * np.square(arr).sum()))


def theil_index(values: Iterable[Any]) -> float:
    arr = finite_values(values)
    arr = arr[arr >= 0]
    if len(arr) == 0 or arr.mean() <= 0:
        return float("nan")
    ratio = arr / arr.mean()
    positive = ratio > 0
    return float(np.mean(ratio[positive] * np.log(ratio[positive]))) if positive.any() else 0.0


def cvar(values: Iterable[Any], tail_fraction: float = 0.10, lower_is_bad: bool = False) -> float:
    arr = np.sort(finite_values(values))
    if len(arr) == 0:
        return float("nan")
    count = max(1, int(math.ceil(len(arr) * tail_fraction)))
    selected = arr[:count] if lower_is_bad else arr[-count:]
    return float(selected.mean())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_excel(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        used: set[str] = set()
        for raw_name, frame in tables.items():
            base = re.sub(r"[\[\]:*?/\\]", "_", raw_name)[:31] or "Sheet"
            name = base
            suffix = 1
            while name in used:
                ending = f"_{suffix}"
                name = (base[:31 - len(ending)] + ending)
                suffix += 1
            used.add(name)
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column_cells in sheet.columns:
                width = min(42, max(10, max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells) + 2))
                sheet.column_dimensions[column_cells[0].column_letter].width = width
