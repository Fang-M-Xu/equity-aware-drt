from __future__ import annotations

"""Generate Figure 5.4.2 without representative A/B/C/D annotations.

The four panels report:
(a) OTP transit within-15-minute accessibility;
(b) proposed DRT within-15-minute accessibility;
(c) DRT minus OTP accessibility, in percentage points;
(d) DRT conversion among transit-disadvantaged requests.

Representative TAZ selection
----------------------------
A and B:
    Two sufficiently sampled TAZs with the largest positive
    DRT-minus-transit accessibility differences.

C:
    The sufficiently sampled TAZ with the most negative
    DRT-minus-transit accessibility difference.

D:
    The service-gap TAZ with the highest DRT conversion rate among
    transit-disadvantaged requests, subject to the minimum sample threshold.

Default inputs
--------------
outputs/metrics_competitive/multimodal_evaluation/request_consensus_across_seeds.csv
outputs/forecast_scenarios_competitive/scenario_*.parquet
data/raw/spatial/Jerusalem_90_zones.shp
data/processed/model_inputs/zone_static_features.parquet

Default outputs
---------------
outputs/paper_results/chapter5_multimodal/
    Figure_5_4_2_spatial_accessibility_gains.png
    Figure_5_4_2_spatial_accessibility_gains.pdf
    Figure_5_4_2_zone_summary.csv
    Figure_5_4_2_representative_taz.csv
"""

import argparse
import re
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"


ZONE_ALIASES = [
    "origin_zone_id",
    "zone_id",
    "TAZ_1270",
    "taz_1270",
    "taz_id",
    "origin_taz",
    "taz",
]

REQUEST_ALIASES = [
    "request_id",
    "req_id",
    "trip_id",
    "id",
]

SCENARIO_ALIASES = [
    "scenario_id",
    "scenario",
    "scenario_idx",
]

GAP_ALIASES = [
    "zone_gap",
    "gap_i",
    "is_gap",
    "gap",
]

ZONE_NAME_ALIASES = [
    "zone_name_en",
    "zone_name",
    "english_name",
    "englishname",
    "name_en",
    "name_eng",
    "nameenglish",
    "taz_name_en",
    "taz_name",
    "zone_en",
    "eng_name",
    "engname",
    "area_name_en",
    "area_name",
    "neighborhood_en",
    "neighbourhood_en",
    "neighborhood",
    "neighbourhood",
    "quarter_en",
    "quarter",
    "district_en",
    "district_name",
    "shem_eng",
    "shem_engl",
    "name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--project-root", default=".")

    parser.add_argument(
        "--consensus",
        default=(
            "outputs/metrics/multimodal_evaluation/"
            "request_consensus_across_seeds.csv"
        ),
    )

    parser.add_argument(
        "--scenario-dir",
        default="outputs/forecast_scenarios",
    )

    parser.add_argument(
        "--zones",
        default="data/raw/spatial/Jerusalem_90_zones.shp",
    )

    parser.add_argument(
        "--zone-static",
        default=(
            "data/processed/model_inputs/"
            "zone_static_features.parquet"
        ),
    )

    parser.add_argument(
        "--zone-name-map",
        default="",
        help=(
            "Optional CSV/XLSX/Parquet/SHP/GPKG file containing a TAZ ID "
            "column and a zone-name column. When omitted, the script first "
            "checks the zone shapefile and then automatically searches likely "
            "mapping files under the spatial-data directories."
        ),
    )

    parser.add_argument(
        "--zone-name-id-column",
        default="",
        help=(
            "Optional exact TAZ-ID column in --zone-name-map. Use this when "
            "automatic column detection selects the wrong field."
        ),
    )

    parser.add_argument(
        "--zone-name-column",
        default="",
        help=(
            "Optional exact zone-name column in --zone-name-map. English-name "
            "columns are preferred automatically, but this argument overrides "
            "automatic detection."
        ),
    )

    parser.add_argument(
        "--zone-name-search-dir",
        action="append",
        default=[],
        help=(
            "Additional directory to search recursively for a zone-name "
            "mapping file. This option may be supplied more than once."
        ),
    )

    parser.add_argument(
        "--representative-label-mode",
        choices=["letter", "letter-and-zone", "letter-and-name"],
        default="letter",
        help=(
            "Text displayed for representative TAZs. Names are always written "
            "to the output CSV even when the map displays letters only."
        ),
    )

    parser.add_argument(
        "--roads",
        default="data/raw/network/jerusalem_roads.geojson",
        help="Optional road layer. It is ignored when the file does not exist.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_multimodal",
    )

    parser.add_argument(
        "--minimum-zone-requests",
        type=int,
        default=5,
        help="Minimum total requests for panels (a)-(c).",
    )

    parser.add_argument(
        "--minimum-transit-disadvantaged",
        type=int,
        default=5,
        help="Minimum transit-disadvantaged requests for panel (d) and TAZ D.",
    )

    parser.add_argument(
        "--annotate-representatives",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--dpi", type=int, default=400)

    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def first_existing(
    columns: Iterable[object],
    aliases: Iterable[str],
) -> str | None:
    originals = {str(column): str(column) for column in columns}
    lowered = {str(column).lower(): str(column) for column in columns}

    for alias in aliases:
        if alias in originals:
            return originals[alias]
        if alias.lower() in lowered:
            return lowered[alias.lower()]

    return None


def normalize_zone(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")

    nonmissing = int(text.notna().sum())
    numeric_count = int(numeric.notna().sum())

    if nonmissing and numeric_count >= max(1, int(0.8 * nonmissing)):
        return numeric.astype("Int64").astype("string")

    return text


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).ne(0)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported table format: {path}")

def infer_scenario_id(path: Path) -> int | None:
    """Infer scenario ID from either forecast_scenario_000 or scenario_000."""
    patterns = [
        r"forecast_scenario[_-]?0*(\d+)",
        r"(?:^|[\\/])scenario[_-]?0*(\d+)(?:\.[^.]+)?$",
        r"scenario[_-]?0*(\d+)",
    ]
    text = str(path)
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def load_origins(scenario_dir: Path) -> pd.DataFrame:
    patterns = [
        "scenario_*.parquet",
        "scenario_*.csv",
        "forecast_scenario_*.parquet",
        "forecast_scenario_*.csv",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(scenario_dir.glob(pattern)))

    # Deduplicate while preserving deterministic order.
    paths = list(dict.fromkeys(paths))

    if not paths:
        raise FileNotFoundError(
            "No scenario_*.parquet/csv or forecast_scenario_*.parquet/csv "
            f"files found under {scenario_dir}"
        )

    frames: list[pd.DataFrame] = []

    for path in paths:
        frame = read_table(path)

        request_col = first_existing(frame.columns, REQUEST_ALIASES)
        zone_col = first_existing(frame.columns, ZONE_ALIASES)
        scenario_col = first_existing(frame.columns, SCENARIO_ALIASES)

        if request_col is None or zone_col is None:
            continue

        if scenario_col is None:
            scenario_id = infer_scenario_id(path)
            if scenario_id is None:
                continue
            frame = frame.copy()
            frame["scenario_id"] = scenario_id
            scenario_col = "scenario_id"

        selected = frame[[scenario_col, request_col, zone_col]].copy()
        selected.columns = [
            "scenario_id",
            "request_id",
            "origin_zone_id",
        ]

        selected["scenario_id"] = pd.to_numeric(
            selected["scenario_id"],
            errors="coerce",
        ).astype("Int64")

        selected["request_id"] = selected["request_id"].astype(str)
        selected["_zone_key"] = normalize_zone(
            selected["origin_zone_id"]
        )

        frames.append(selected)

    if not frames:
        raise ValueError(
            "Scenario files were found, but scenario ID, request ID, and "
            "origin-zone columns could not be detected."
        )

    origins = pd.concat(frames, ignore_index=True)

    origins = origins.dropna(
        subset=["scenario_id", "request_id", "_zone_key"]
    )

    origins["scenario_id"] = origins["scenario_id"].astype(int)

    return origins.drop_duplicates(
        ["scenario_id", "request_id"]
    )


def load_consensus(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    frame = frame.loc[
        :,
        ~frame.columns.astype(str).str.startswith("Unnamed"),
    ].copy()

    scenario_col = first_existing(
        frame.columns,
        SCENARIO_ALIASES,
    )

    request_col = first_existing(
        frame.columns,
        REQUEST_ALIASES,
    )

    required = {
        "transit_within_threshold",
        "drt_access_probability",
    }

    missing = sorted(required.difference(frame.columns))

    if scenario_col is None or request_col is None or missing:
        raise KeyError(
            "Consensus file is missing required keys or metrics. "
            f"Missing metrics: {missing}; "
            f"available columns: {list(frame.columns)}"
        )

    frame = frame.rename(
        columns={
            scenario_col: "scenario_id",
            request_col: "request_id",
        }
    )

    frame["scenario_id"] = pd.to_numeric(
        frame["scenario_id"],
        errors="raise",
    ).astype(int)

    frame["request_id"] = frame["request_id"].astype(str)

    frame["transit_within_threshold"] = to_bool(
        frame["transit_within_threshold"]
    )

    frame["drt_access_probability"] = pd.to_numeric(
        frame["drt_access_probability"],
        errors="coerce",
    )

    return frame



def normalize_column_token(value: object) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )


def resolve_requested_column(
    columns: Iterable[object],
    requested: str,
) -> str | None:
    if not requested:
        return None

    direct = {str(column): str(column) for column in columns}
    lowered = {str(column).lower(): str(column) for column in columns}

    if requested in direct:
        return direct[requested]

    if requested.lower() in lowered:
        return lowered[requested.lower()]

    raise KeyError(
        f"Requested column '{requested}' was not found. "
        f"Available columns: {list(columns)}"
    )


def valid_name_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()

    return (
        text.notna()
        & text.ne("")
        & ~text.str.lower().isin(
            {
                "nan",
                "none",
                "null",
                "<na>",
                "unknown",
                "unnamed",
            }
        )
    )


def infer_zone_id_column(
    frame: pd.DataFrame,
    target_zone_keys: set[str],
    requested: str = "",
) -> tuple[str, int]:
    explicit = resolve_requested_column(
        frame.columns,
        requested,
    )

    if explicit is not None:
        overlap = len(
            set(
                normalize_zone(frame[explicit])
                .dropna()
                .astype(str)
            )
            & target_zone_keys
        )
        return explicit, overlap

    alias_match = first_existing(
        frame.columns,
        ZONE_ALIASES,
    )

    candidates: list[tuple[int, int, str]] = []

    for column in frame.columns:
        if str(column).lower() == "geometry":
            continue

        keys = set(
            normalize_zone(frame[column])
            .dropna()
            .astype(str)
        )
        overlap = len(keys & target_zone_keys)

        token = normalize_column_token(column)
        semantic_score = 0

        if token in {
            normalize_column_token(alias)
            for alias in ZONE_ALIASES
        }:
            semantic_score += 80
        if "taz" in token:
            semantic_score += 50
        if "zone" in token:
            semantic_score += 35
        if "1270" in token:
            semantic_score += 30
        if "id" in token or "code" in token:
            semantic_score += 15

        if alias_match is not None and str(column) == alias_match:
            semantic_score += 70

        candidates.append(
            (
                overlap,
                semantic_score,
                str(column),
            )
        )

    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    if not candidates or candidates[0][0] == 0:
        raise KeyError(
            "Could not identify a TAZ-ID column with values matching the "
            "zone layer. "
            f"Available columns: {list(frame.columns)}"
        )

    overlap, _, column = candidates[0]
    return column, overlap


def infer_zone_name_column(
    frame: pd.DataFrame,
    requested: str = "",
) -> str:
    explicit = resolve_requested_column(
        frame.columns,
        requested,
    )

    if explicit is not None:
        return explicit

    exact_alias = first_existing(
        frame.columns,
        ZONE_NAME_ALIASES,
    )

    candidates: list[tuple[float, str]] = []

    english_markers = {
        "eng",
        "english",
        "nameen",
        "nameeng",
        "zoneen",
        "shemeng",
        "shemengl",
    }

    name_markers = {
        "name",
        "shem",
        "neighborhood",
        "neighbourhood",
        "quarter",
        "district",
        "area",
        "locality",
    }

    for column in frame.columns:
        column_name = str(column)

        if column_name.lower() == "geometry":
            continue

        series = frame[column]
        mask = valid_name_mask(series)

        if not mask.any():
            continue

        text = series.loc[mask].astype(str).str.strip()
        numeric_share = pd.to_numeric(
            text,
            errors="coerce",
        ).notna().mean()

        # A zone name should usually be textual rather than almost entirely
        # numeric.
        if numeric_share > 0.80:
            continue

        token = normalize_column_token(column_name)
        score = 0.0

        if exact_alias is not None and column_name == exact_alias:
            score += 160.0

        if any(marker in token for marker in english_markers):
            score += 120.0

        if any(marker in token for marker in name_markers):
            score += 65.0

        if "eng" in token or "english" in token:
            score += 70.0

        if any(
            marker in token
            for marker in {
                "id",
                "code",
                "number",
                "num",
                "index",
                "geometry",
            }
        ):
            score -= 90.0

        unique_count = int(text.nunique())
        average_length = float(text.str.len().mean())

        score += min(unique_count, 100) * 0.20
        score += min(average_length, 30.0) * 0.50
        score += (1.0 - float(numeric_share)) * 20.0

        candidates.append((score, column_name))

    if not candidates:
        raise KeyError(
            "Could not identify a textual zone-name column. "
            f"Available columns: {list(frame.columns)}"
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )
    return candidates[0][1]


def read_zone_name_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".shp", ".gpkg", ".geojson"}:
        return gpd.read_file(path)

    return read_table(path)


def build_name_mapping(
    frame: pd.DataFrame,
    target_zone_keys: set[str],
    *,
    requested_id_column: str = "",
    requested_name_column: str = "",
    source_label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    zone_col, overlap = infer_zone_id_column(
        frame,
        target_zone_keys,
        requested=requested_id_column,
    )

    name_col = infer_zone_name_column(
        frame,
        requested=requested_name_column,
    )

    mapping = frame[[zone_col, name_col]].copy()
    mapping["_zone_key"] = normalize_zone(
        mapping[zone_col]
    )
    mapping["_external_zone_name"] = (
        mapping[name_col]
        .astype("string")
        .str.strip()
    )

    mapping = mapping.loc[
        mapping["_zone_key"].notna()
        & valid_name_mask(mapping["_external_zone_name"])
    ].copy()

    # If a source contains duplicate rows per TAZ, retain the first non-empty
    # name. Conflicting duplicates are reported through the diagnostics.
    conflicts = (
        mapping.groupby("_zone_key")[
            "_external_zone_name"
        ]
        .nunique()
        .gt(1)
    )
    conflict_count = int(conflicts.sum())

    mapping = mapping.drop_duplicates(
        "_zone_key",
        keep="first",
    )

    matching = mapping.loc[
        mapping["_zone_key"]
        .astype(str)
        .isin(target_zone_keys)
    ]

    diagnostics = {
        "source": source_label,
        "zone_id_column": zone_col,
        "zone_name_column": name_col,
        "overlapping_zone_ids": overlap,
        "matched_nonempty_names": int(len(matching)),
        "unique_names": int(
            matching["_external_zone_name"].nunique()
        ),
        "conflicting_zone_ids": conflict_count,
    }

    return (
        mapping[
            [
                "_zone_key",
                "_external_zone_name",
            ]
        ],
        diagnostics,
    )


def candidate_zone_name_files(
    root: Path,
    zones_path: Path,
    additional_dirs: list[str],
) -> list[Path]:
    directories: list[Path] = [
        zones_path.parent,
        root / "data" / "raw" / "spatial",
        root / "data" / "processed",
    ]

    directories.extend(
        resolve(root, value)
        for value in additional_dirs
    )

    extensions = {
        ".csv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".shp",
        ".gpkg",
        ".geojson",
    }

    candidates: set[Path] = set()

    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            continue

        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in extensions
                and path.resolve() != zones_path.resolve()
            ):
                candidates.add(path.resolve())

    return sorted(candidates)


def discover_zone_name_mapping(
    *,
    root: Path,
    zones_path: Path,
    zones: gpd.GeoDataFrame,
    explicit_path: Path | None,
    requested_id_column: str,
    requested_name_column: str,
    additional_dirs: list[str],
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    target_zone_keys = set(
        zones["_zone_key"]
        .dropna()
        .astype(str)
    )

    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"Zone-name mapping file not found: {explicit_path}"
            )

        mapping, diagnostics = build_name_mapping(
            read_zone_name_source(explicit_path),
            target_zone_keys,
            requested_id_column=requested_id_column,
            requested_name_column=requested_name_column,
            source_label=str(explicit_path),
        )

        if diagnostics["matched_nonempty_names"] == 0:
            raise ValueError(
                "The supplied zone-name mapping did not match any TAZ IDs. "
                f"Diagnostics: {diagnostics}"
            )

        return mapping, diagnostics

    ranked: list[
        tuple[
            float,
            pd.DataFrame,
            dict[str, object],
        ]
    ] = []

    for path in candidate_zone_name_files(
        root,
        zones_path,
        additional_dirs,
    ):
        try:
            frame = read_zone_name_source(path)
            mapping, diagnostics = build_name_mapping(
                frame,
                target_zone_keys,
                source_label=str(path),
            )
        except Exception:
            continue

        matched = int(
            diagnostics["matched_nonempty_names"]
        )
        overlap = int(
            diagnostics["overlapping_zone_ids"]
        )
        unique_names = int(
            diagnostics["unique_names"]
        )

        if matched == 0:
            continue

        filename = normalize_column_token(path.stem)
        filename_bonus = 0.0

        if "name" in filename:
            filename_bonus += 35.0
        if "eng" in filename or "english" in filename:
            filename_bonus += 30.0
        if "taz" in filename or "zone" in filename:
            filename_bonus += 20.0

        # The matched-name count receives the highest weight. This prevents a
        # small but semantically named file from outranking a nearly complete
        # mapping.
        score = (
            matched * 100.0
            + overlap * 10.0
            + unique_names
            + filename_bonus
        )

        ranked.append(
            (
                score,
                mapping,
                diagnostics,
            )
        )

    if not ranked:
        return None, None

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, mapping, diagnostics = ranked[0]
    return mapping, diagnostics


def load_zones(
    path: Path,
) -> tuple[gpd.GeoDataFrame, str, str | None]:
    zones = gpd.read_file(path)

    zone_col = first_existing(
        zones.columns,
        ZONE_ALIASES,
    )

    if zone_col is None:
        # Fall back to a self-overlap-free semantic search.
        semantic_candidates = [
            str(column)
            for column in zones.columns
            if (
                "taz" in normalize_column_token(column)
                or "zone" in normalize_column_token(column)
            )
        ]
        if semantic_candidates:
            zone_col = semantic_candidates[0]

    if zone_col is None:
        raise KeyError(
            f"Zone ID column not found in {path}. "
            f"Available columns: {list(zones.columns)}"
        )

    zones = zones.copy()
    zones["_zone_key"] = normalize_zone(
        zones[zone_col]
    )

    name_col: str | None = None

    try:
        name_col = infer_zone_name_column(zones)
    except KeyError:
        name_col = None

    if name_col is not None:
        zones["_zone_name"] = (
            zones[name_col]
            .astype("string")
            .str.strip()
        )
        zones["_zone_name_source"] = (
            f"{path.name}:{name_col}"
        )
    else:
        zones["_zone_name"] = pd.NA
        zones["_zone_name_source"] = pd.NA

    return zones, zone_col, name_col


def add_external_zone_names(
    zones: gpd.GeoDataFrame,
    *,
    root: Path,
    zones_path: Path,
    explicit_path: Path | None,
    requested_id_column: str,
    requested_name_column: str,
    additional_dirs: list[str],
) -> gpd.GeoDataFrame:
    existing_count = int(
        valid_name_mask(zones["_zone_name"]).sum()
    )

    mapping, diagnostics = discover_zone_name_mapping(
        root=root,
        zones_path=zones_path,
        zones=zones,
        explicit_path=explicit_path,
        requested_id_column=requested_id_column,
        requested_name_column=requested_name_column,
        additional_dirs=additional_dirs,
    )

    if mapping is None or diagnostics is None:
        print(
            "Warning: no usable external TAZ-name mapping was found. "
            "Use --zone-name-map together with --zone-name-id-column and "
            "--zone-name-column when necessary."
        )
        print(
            f"Names already available in the zone layer: "
            f"{existing_count}/{len(zones)}"
        )
        return zones

    output = zones.merge(
        mapping,
        on="_zone_key",
        how="left",
        validate="one_to_one",
    )

    external_mask = valid_name_mask(
        output["_external_zone_name"]
    )

    existing_mask = valid_name_mask(
        output["_zone_name"]
    )

    # Explicit or automatically discovered external English names take
    # precedence. Existing shapefile names fill any remaining gaps.
    output["_zone_name"] = (
        output["_external_zone_name"]
        .where(external_mask)
        .combine_first(
            output["_zone_name"].where(existing_mask)
        )
    )

    output["_zone_name_source"] = (
        pd.Series(pd.NA, index=output.index, dtype="string")
        .where(~external_mask, str(diagnostics["source"]))
        .combine_first(output["_zone_name_source"])
    )

    output = output.drop(
        columns=["_external_zone_name"],
        errors="ignore",
    )

    final_count = int(
        valid_name_mask(output["_zone_name"]).sum()
    )

    print("\nZone-name mapping selected:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")
    print(
        f"  final named TAZs: {final_count}/{len(output)}"
    )

    return output


def load_gap_keys(path: Path) -> set[str]:
    if not path.exists():
        print(
            f"Warning: zone-static file not found: {path}. "
            "Service-gap boundaries will not be drawn."
        )
        return set()

    static = read_table(path)

    zone_col = first_existing(
        static.columns,
        ZONE_ALIASES,
    )

    gap_col = first_existing(
        static.columns,
        GAP_ALIASES,
    )

    if zone_col is None or gap_col is None:
        print(
            "Warning: zone ID or gap column was not detected in "
            f"{path}. Service-gap boundaries will not be drawn."
        )
        return set()

    static = static.copy()
    static["_zone_key"] = normalize_zone(static[zone_col])

    return set(
        static.loc[
            to_bool(static[gap_col]),
            "_zone_key",
        ]
        .dropna()
        .astype(str)
    )


def build_zone_summary(
    consensus: pd.DataFrame,
    origins: pd.DataFrame,
) -> pd.DataFrame:
    merged = consensus.merge(
        origins[
            [
                "scenario_id",
                "request_id",
                "_zone_key",
            ]
        ],
        on=["scenario_id", "request_id"],
        how="left",
        validate="one_to_one",
    )

    missing = int(merged["_zone_key"].isna().sum())

    if missing:
        raise ValueError(
            f"{missing} consensus request rows could not be linked "
            "to an origin TAZ."
        )

    merged["transit_disadvantaged"] = (
        ~merged["transit_within_threshold"]
    )

    summary = (
        merged.groupby(
            "_zone_key",
            as_index=False,
        )
        .agg(
            n_requests=("request_id", "size"),
            transit_within_15_rate=(
                "transit_within_threshold",
                "mean",
            ),
            drt_within_15_rate=(
                "drt_access_probability",
                "mean",
            ),
            n_transit_disadvantaged=(
                "transit_disadvantaged",
                "sum",
            ),
        )
    )

    conversion = (
        merged.loc[
            merged["transit_disadvantaged"]
        ]
        .groupby(
            "_zone_key",
            as_index=False,
        )["drt_access_probability"]
        .mean()
        .rename(
            columns={
                "drt_access_probability": (
                    "transit_disadvantage_conversion_rate"
                )
            }
        )
    )

    summary = summary.merge(
        conversion,
        on="_zone_key",
        how="left",
    )

    summary["drt_minus_transit_pp"] = (
        summary["drt_within_15_rate"]
        - summary["transit_within_15_rate"]
    ) * 100.0

    return summary


def select_representative_taz(
    mapped: gpd.GeoDataFrame,
    gap_keys: set[str],
    minimum_zone_requests: int,
    minimum_transit_disadvantaged: int,
) -> gpd.GeoDataFrame:
    data = mapped.copy()

    data["is_service_gap"] = (
        data["_zone_key"]
        .astype(str)
        .isin(gap_keys)
    )

    rows: list[pd.Series] = []
    used_zone_keys: set[str] = set()

    sufficiently_sampled = data.loc[
        pd.to_numeric(
            data["n_requests"],
            errors="coerce",
        )
        .fillna(0)
        .ge(minimum_zone_requests)
        & data["drt_minus_transit_pp"].notna()
    ].copy()

    positive = (
        sufficiently_sampled.loc[
            sufficiently_sampled[
                "drt_minus_transit_pp"
            ] > 0
        ]
        .sort_values(
            "drt_minus_transit_pp",
            ascending=False,
        )
        .head(2)
    )

    for label, (_, row) in zip(
        ["A", "B"],
        positive.iterrows(),
    ):
        item = row.copy()
        item["representative_label"] = label
        item["selection_reason"] = (
            "Largest positive DRT-minus-transit accessibility gain"
        )
        rows.append(item)
        used_zone_keys.add(str(item["_zone_key"]))

    negative = (
        sufficiently_sampled.loc[
            sufficiently_sampled[
                "drt_minus_transit_pp"
            ] < 0
        ]
        .sort_values(
            "drt_minus_transit_pp",
            ascending=True,
        )
        .head(1)
    )

    if not negative.empty:
        item = negative.iloc[0].copy()
        item["representative_label"] = "C"
        item["selection_reason"] = (
            "Largest negative DRT-minus-transit accessibility difference"
        )
        rows.append(item)
        used_zone_keys.add(str(item["_zone_key"]))
    else:
        print(
            "No sufficiently sampled TAZ had a negative "
            "DRT-minus-transit accessibility difference; "
            "representative C was not created."
        )

    gap_conversion = data.loc[
        data["is_service_gap"]
        & pd.to_numeric(
            data["n_transit_disadvantaged"],
            errors="coerce",
        )
        .fillna(0)
        .ge(minimum_transit_disadvantaged)
        & data[
            "transit_disadvantage_conversion_rate"
        ].notna()
    ].copy()

    # Prefer a TAZ that is different from A, B, and C.
    distinct_gap_conversion = gap_conversion.loc[
        ~gap_conversion["_zone_key"]
        .astype(str)
        .isin(used_zone_keys)
    ]

    d_pool = (
        distinct_gap_conversion
        if not distinct_gap_conversion.empty
        else gap_conversion
    )

    d_candidate = (
        d_pool.sort_values(
            "transit_disadvantage_conversion_rate",
            ascending=False,
        )
        .head(1)
    )

    if not d_candidate.empty:
        item = d_candidate.iloc[0].copy()
        item["representative_label"] = "D"
        item["selection_reason"] = (
            "Highest transit-disadvantage conversion rate "
            "among sufficiently sampled service-gap TAZs"
        )
        rows.append(item)
    else:
        print(
            "No service-gap TAZ met the minimum transit-disadvantaged "
            "request threshold; representative D was not created."
        )

    if not rows:
        return gpd.GeoDataFrame(
            columns=[
                *data.columns,
                "representative_label",
                "selection_reason",
            ],
            geometry="geometry",
            crs=data.crs,
        )

    representatives = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=data.crs,
    )

    label_order = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }

    representatives["_label_order"] = (
        representatives["representative_label"]
        .map(label_order)
        .fillna(99)
    )

    return (
        representatives.sort_values("_label_order")
        .drop(columns="_label_order")
    )


def hatch_low_sample(
    ax: plt.Axes,
    frame: gpd.GeoDataFrame,
    count_column: str,
    threshold: int,
) -> None:
    count = pd.to_numeric(
        frame[count_column],
        errors="coerce",
    ).fillna(0)

    low = frame.loc[count.lt(threshold)]

    if not low.empty:
        low.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=0.15,
            hatch="////",
            zorder=7,
        )


def draw_gap_boundary(
    ax: plt.Axes,
    frame: gpd.GeoDataFrame,
    gap_keys: set[str],
) -> None:
    if not gap_keys:
        return

    gap = frame.loc[
        frame["_zone_key"]
        .astype(str)
        .isin(gap_keys)
    ]

    if not gap.empty:
        gap.boundary.plot(
            ax=ax,
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            zorder=8,
        )


def representative_label_text(
    row: pd.Series,
    mode: str,
) -> str:
    letter = str(row["representative_label"])
    zone_key = str(row["_zone_key"])

    raw_name = row.get("_zone_name", pd.NA)
    has_name = bool(
        valid_name_mask(pd.Series([raw_name])).iloc[0]
    )
    zone_name = str(raw_name).strip() if has_name else ""

    if mode == "letter-and-zone":
        return f"{letter}\nTAZ {zone_key}"

    if mode == "letter-and-name":
        display_name = zone_name if zone_name else f"TAZ {zone_key}"
        if len(display_name) > 28:
            display_name = display_name[:25].rstrip() + "..."
        return f"{letter}: {display_name}"

    return letter


def annotate_representatives(
    ax: plt.Axes,
    representatives: gpd.GeoDataFrame,
    labels: set[str],
    label_mode: str,
) -> None:
    if representatives.empty:
        return

    selected = representatives.loc[
        representatives[
            "representative_label"
        ].isin(labels)
    ]

    for _, row in selected.iterrows():
        point = row.geometry.representative_point()
        label = representative_label_text(
            row,
            label_mode,
        )

        boxstyle = (
            "circle,pad=0.28"
            if label_mode == "letter"
            else "round,pad=0.25"
        )

        ax.annotate(
            label,
            xy=(point.x, point.y),
            xytext=(8, 8),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.5 if label_mode != "letter" else 9,
            fontweight="bold",
            bbox={
                "boxstyle": boxstyle,
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.9,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "black",
                "linewidth": 0.7,
            },
            zorder=20,
            annotation_clip=False,
        )


def plot_panel(
    ax: plt.Axes,
    frame: gpd.GeoDataFrame,
    column: str,
    title: str,
    count_column: str,
    minimum_count: int,
    gap_keys: set[str],
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
) -> None:
    frame.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        linewidth=0.35,
        edgecolor="white",
        legend=True,
        legend_kwds={"shrink": 0.72},
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "label": "No observations",
        },
    )

    frame.boundary.plot(
        ax=ax,
        linewidth=0.25,
        alpha=0.7,
        zorder=5,
    )

    hatch_low_sample(
        ax,
        frame,
        count_column,
        minimum_count,
    )

    draw_gap_boundary(
        ax,
        frame,
        gap_keys,
    )

    ax.set_title(
        title,
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
    )

    ax.set_axis_off()
    ax.set_aspect("equal")


def export_zone_summary(
    mapped: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    columns = [
        "_zone_key",
        "_zone_name",
        "_zone_name_source",
        "is_service_gap",
        "n_requests",
        "transit_within_15_rate",
        "drt_within_15_rate",
        "drt_minus_transit_pp",
        "n_transit_disadvantaged",
        "transit_disadvantage_conversion_rate",
    ]

    available = [
        column
        for column in columns
        if column in mapped.columns
    ]

    mapped[available].to_csv(
        output_path,
        index=False,
    )


def export_representatives(
    representatives: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    columns = [
        "representative_label",
        "_zone_key",
        "_zone_name",
        "_zone_name_source",
        "selection_reason",
        "is_service_gap",
        "n_requests",
        "transit_within_15_rate",
        "drt_within_15_rate",
        "drt_minus_transit_pp",
        "n_transit_disadvantaged",
        "transit_disadvantage_conversion_rate",
    ]

    available = [
        column
        for column in columns
        if column in representatives.columns
    ]

    representatives[available].to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    args = parse_args()

    root = Path(args.project_root).resolve()

    consensus_path = resolve(
        root,
        args.consensus,
    )

    scenario_dir = resolve(
        root,
        args.scenario_dir,
    )

    zones_path = resolve(
        root,
        args.zones,
    )

    static_path = resolve(
        root,
        args.zone_static,
    )

    roads_path = resolve(
        root,
        args.roads,
    )

    name_map_path = (
        resolve(root, args.zone_name_map)
        if args.zone_name_map
        else None
    )

    output_dir = resolve(
        root,
        args.output_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        consensus_path,
        scenario_dir,
        zones_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    consensus = load_consensus(
        consensus_path
    )

    origins = load_origins(
        scenario_dir
    )

    zones, _, _ = load_zones(
        zones_path
    )

    zones = add_external_zone_names(
        zones,
        root=root,
        zones_path=zones_path,
        explicit_path=name_map_path,
        requested_id_column=args.zone_name_id_column,
        requested_name_column=args.zone_name_column,
        additional_dirs=args.zone_name_search_dir,
    )

    gap_keys = load_gap_keys(
        static_path
    )

    summary = build_zone_summary(
        consensus,
        origins,
    )

    mapped = zones.merge(
        summary,
        on="_zone_key",
        how="left",
    )

    mapped["is_service_gap"] = (
        mapped["_zone_key"]
        .astype(str)
        .isin(gap_keys)
    )

    representatives = gpd.GeoDataFrame(
        columns=[
            *mapped.columns,
            "representative_label",
            "selection_reason",
        ],
        geometry="geometry",
        crs=mapped.crs,
    )

    roads = None

    if roads_path.exists():
        try:
            roads = gpd.read_file(
                roads_path
            )

            if (
                zones.crs is not None
                and roads.crs is not None
            ):
                roads = roads.to_crs(
                    zones.crs
                )

        except Exception as error:
            print(
                f"Warning: road layer was not drawn: {error}"
            )
            roads = None

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 10.0),
    )

    for ax in axes.flat:
        if roads is not None:
            roads.plot(
                ax=ax,
                linewidth=0.15,
                alpha=0.25,
                zorder=0,
            )

    plot_panel(
        axes[0, 0],
        mapped,
        "transit_within_15_rate",
        "(a) GTFS fixed-route transit within 15 minutes",
        "n_requests",
        args.minimum_zone_requests,
        gap_keys,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_panel(
        axes[0, 1],
        mapped,
        "drt_within_15_rate",
        "(b) DRT within 15 minutes",
        "n_requests",
        args.minimum_zone_requests,
        gap_keys,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    differences = pd.to_numeric(
        mapped["drt_minus_transit_pp"],
        errors="coerce",
    ).dropna()

    max_abs = (
        max(
            10.0,
            float(differences.abs().max()),
        )
        if len(differences)
        else 100.0
    )

    plot_panel(
        axes[1, 0],
        mapped,
        "drt_minus_transit_pp",
        "(c) DRT − GTFS accessibility difference (percentage points)",
        "n_requests",
        args.minimum_zone_requests,
        gap_keys,
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0,
            vmax=max_abs,
        ),
    )

    plot_panel(
        axes[1, 1],
        mapped,
        "transit_disadvantage_conversion_rate",
        "(d) DRT conversion of transit-disadvantaged trips",
        "n_transit_disadvantaged",
        args.minimum_transit_disadvantaged,
        gap_keys,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    # Representative labels A/B/C/D are intentionally not drawn.

    legend_handles = [
        Line2D(
            [0],
            [0],
            linewidth=1.2,
            linestyle=(0, (4, 2)),
            label="Service-gap TAZ boundary",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="////",
            label="Below minimum request threshold",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
    )


    fig.tight_layout(
        rect=(0, 0.065, 1, 0.97)
    )

    output_base = (
        output_dir
        / "Figure_5_4_2_spatial_accessibility_gains"
    )

    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )


    plt.close(fig)

    zone_summary_path = (
        output_dir
        / "Figure_5_4_2_zone_summary.csv"
    )

    representative_path = (
        output_dir
        / "Figure_5_4_2_representative_taz.csv"
    )

    export_zone_summary(
        mapped,
        zone_summary_path,
    )

    export_representatives(
        representatives,
        representative_path,
    )

    print("\nRepresentative TAZs:")
    if representatives.empty:
        print("No representative TAZ met the configured thresholds.")
    else:
        print(
            representatives[
                [
                    "representative_label",
                    "_zone_key",
                    "_zone_name",
                    "_zone_name_source",
                    "n_requests",
                    "transit_within_15_rate",
                    "drt_within_15_rate",
                    "drt_minus_transit_pp",
                    "n_transit_disadvantaged",
                    "transit_disadvantage_conversion_rate",
                    "is_service_gap",
                    "selection_reason",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved: {output_base.with_suffix('.png')}")
    print(f"Saved: {output_base.with_suffix('.pdf')}")
    print(f"Saved: {zone_summary_path}")
    print(f"Saved: {representative_path}")


if __name__ == "__main__":
    main()