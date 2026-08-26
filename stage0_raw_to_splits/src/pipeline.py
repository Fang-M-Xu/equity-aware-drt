from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn
import shapely
import yaml
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("sidre_data_pipeline")
PARQUET_ENGINE_AVAILABLE = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)
_PARQUET_WARNING_EMITTED = False


BUILDING_USE_MAP = {
    1: "Residential",
    2: "Mixed_residential",
    3: "Commercial",
    4: "Public_community_service",
    5: "Industrial_other",
    6: "Senior_sheltered_residence",
}

PERSON_COLUMNS = [
    "pid",
    "sampn",
    "perno",
    "sectr",
    "gend",
    "age",
    "age_r",
    "pagecat",
    "imig",
    "disab",
    "clic",
    "mlic",
    "emply",
    "jobs",
    "wksta",
    "tpass",
    "educ",
    "stud",
    "schol",
    "hhtaz",
    "wutaz",
    "wulocno",
    "sutaz",
    "sulocno",
    "loop",
    "aloop",
]

HOUSEHOLD_COLUMNS = [
    "sampn",
    "hhtaz",
    "hxcrd",
    "hycrd",
    "sectr",
    "hhsize",
    "hheduc",
    "hwork_f",
    "hwork_p",
    "huniv",
    "hnwork",
    "hretire",
    "hschdriv",
    "hschpred",
    "hpresch",
    "hadults18",
    "h0005",
    "h0611",
    "h1215",
    "h1617",
    "h1824",
    "h2534",
    "h3549",
    "h5064",
    "h6579",
    "h80up",
    "hadnwst",
    "hadwpst",
    "hadkids",
    "hhveh",
    "hinccat1",
    "hinccat2",
    "hownrent",
    "htypdwel",
    "lang",
    "dow",
]


@dataclass
class PipelineArtifacts:
    buildings: pd.DataFrame
    homes: pd.DataFrame
    events: pd.DataFrame
    trips_all: pd.DataFrame
    trips_model_ready: pd.DataFrame
    split_manifest: pd.DataFrame


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    config["__config_path__"] = str(path.resolve())
    return config


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = [str(column).strip().lower() for column in output.columns]
    return output


def normalize_pid(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"^p", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def mode_or_first(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    modes = non_null.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else non_null.iloc[0]


def nullable_binary(
    series: pd.Series,
    yes_codes: Iterable[int],
    no_codes: Iterable[int],
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series(pd.NA, index=series.index, dtype="Int64")
    output.loc[numeric.isin(list(yes_codes))] = 1
    output.loc[numeric.isin(list(no_codes))] = 0
    return output


def safe_map(series: pd.Series, mapping: dict[Any, Any], missing_label: str) -> pd.Series:
    normalized_mapping: dict[Any, Any] = {}
    for key, value in mapping.items():
        try:
            normalized_mapping[int(key)] = value
        except (TypeError, ValueError):
            normalized_mapping[key] = value
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.map(normalized_mapping).fillna(missing_label).astype("string")


def sanitize_token(value: Any) -> str:
    text = "unknown" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return text or "unknown"


def load_raw_tables(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stops = normalize_columns(pd.read_csv(paths["stops"], low_memory=False))
    persons = normalize_columns(pd.read_csv(paths["persons"], low_memory=False))
    households = normalize_columns(pd.read_csv(paths["households"], low_memory=False))

    stops = stops.drop(columns=["unnamed: 0"], errors="ignore")

    validate_columns(
        stops,
        [
            "pid",
            "stopnum",
            "stoptime",
            "x",
            "y",
            "fromhome",
            "ishome",
            "uniq_id",
            "usg_code",
            "xcoord",
            "ycoord",
            "distance",
        ],
        "stops",
    )
    validate_columns(persons, ["pid", "sampn", "perno"], "persons")
    validate_columns(households, ["sampn", "hxcrd", "hycrd"], "households")

    stops["person_id"] = normalize_pid(stops["pid"])
    persons["person_id"] = normalize_pid(persons["pid"])
    households["household_id"] = pd.to_numeric(households["sampn"], errors="coerce").astype("Int64")
    persons["household_id"] = pd.to_numeric(persons["sampn"], errors="coerce").astype("Int64")

    if persons["person_id"].duplicated().any():
        duplicated = persons.loc[persons["person_id"].duplicated(False), "person_id"].head(10).tolist()
        raise ValueError(f"Person IDs are not unique. Examples: {duplicated}")
    if households["household_id"].duplicated().any():
        duplicated = households.loc[
            households["household_id"].duplicated(False), "household_id"
        ].head(10).tolist()
        raise ValueError(f"Household IDs are not unique. Examples: {duplicated}")
    if stops.duplicated(["person_id", "stopnum", "stoptime"]).any():
        raise ValueError("Raw stops contain duplicate person-stop-time records.")

    return stops, persons, households


def prepare_people_and_households(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    person_keep = [column for column in PERSON_COLUMNS if column in persons.columns]
    household_keep = [column for column in HOUSEHOLD_COLUMNS if column in households.columns]

    people = persons[person_keep + ["person_id", "household_id"]].copy()
    homes = households[household_keep + ["household_id"]].copy()

    people = people.rename(
        columns={
            "perno": "person_number",
            "sectr": "person_sector_code",
            "gend": "gender_code",
            "age": "age_reported",
            "age_r": "age_revised",
            "pagecat": "age_category_code",
            "imig": "immigration_code",
            "disab": "disability_code",
            "clic": "car_license_code",
            "mlic": "motorcycle_license_code",
            "emply": "employment_code",
            "jobs": "number_of_jobs",
            "wksta": "work_status_code",
            "tpass": "transit_pass_code",
            "educ": "education_code",
            "stud": "student_code",
            "schol": "school_level_code",
            "hhtaz": "person_home_taz_original",
            "wutaz": "work_taz_original",
            "wulocno": "work_location_number",
            "sutaz": "school_taz_original",
            "sulocno": "school_location_number",
            "loop": "loop_code",
            "aloop": "activity_loop_code",
        }
    )

    homes = homes.rename(
        columns={
            "hhtaz": "household_home_taz_original",
            "hxcrd": "household_x",
            "hycrd": "household_y",
            "sectr": "household_sector_code",
            "hhsize": "household_size",
            "hheduc": "household_education_code",
            "hwork_f": "household_full_time_workers",
            "hwork_p": "household_part_time_workers",
            "huniv": "household_university_students",
            "hnwork": "household_non_workers",
            "hretire": "household_retirees",
            "hschdriv": "household_driving_school_children",
            "hschpred": "household_predriving_school_children",
            "hpresch": "household_preschool_children",
            "hadults18": "household_adults_18plus",
            "h0005": "household_age_0_5",
            "h0611": "household_age_6_11",
            "h1215": "household_age_12_15",
            "h1617": "household_age_16_17",
            "h1824": "household_age_18_24",
            "h2534": "household_age_25_34",
            "h3549": "household_age_35_49",
            "h5064": "household_age_50_64",
            "h6579": "household_age_65_79",
            "h80up": "household_age_80plus",
            "hadnwst": "household_adult_nonworking_students",
            "hadwpst": "household_adult_parttime_working_students",
            "hadkids": "household_adult_children",
            "hhveh": "household_vehicles",
            "hinccat1": "income_category_code",
            "hinccat2": "income_category_original_code",
            "hownrent": "home_tenure_code",
            "htypdwel": "dwelling_type_code",
            "lang": "household_language_code",
            "dow": "survey_day_of_week_code",
        }
    )

    mappings = config["value_mappings"]
    people["sector_group"] = safe_map(
        people["person_sector_code"], mappings["sector"], "Missing"
    )
    people["gender_group"] = safe_map(
        people["gender_code"], mappings["gender"], "Missing"
    )

    age_revised = pd.to_numeric(people.get("age_revised"), errors="coerce")
    age_reported = pd.to_numeric(people.get("age_reported"), errors="coerce")
    people["age_final"] = age_revised.where(age_revised.between(0, 110), age_reported)
    people.loc[~people["age_final"].between(0, 110), "age_final"] = np.nan
    people["age_group"] = pd.cut(
        people["age_final"],
        bins=[-0.1, 17, 34, 49, 64, 79, 120],
        labels=["Under_18", "18_34", "35_49", "50_64", "65_79", "80_plus"],
        include_lowest=True,
    ).astype("string")
    people["is_older"] = pd.Series(
        np.where(people["age_final"].notna(), people["age_final"] >= 65, np.nan),
        index=people.index,
    ).astype("Int64")
    people["is_80plus"] = pd.Series(
        np.where(people["age_final"].notna(), people["age_final"] >= 80, np.nan),
        index=people.index,
    ).astype("Int64")

    people["is_disabled"] = nullable_binary(
        people["disability_code"],
        mappings["disability_yes_codes"],
        mappings["disability_no_codes"],
    )
    people["is_student"] = nullable_binary(
        people["student_code"],
        mappings["student_yes_codes"],
        mappings["student_no_codes"],
    )
    people["is_employed"] = nullable_binary(
        people["employment_code"],
        mappings["employed_yes_codes"],
        mappings["employed_no_codes"],
    )
    people["has_car_license"] = nullable_binary(
        people["car_license_code"],
        mappings["car_license_yes_codes"],
        mappings["car_license_no_codes"],
    )
    people["has_transit_pass"] = nullable_binary(
        people["transit_pass_code"],
        mappings["transit_pass_yes_codes"],
        mappings["transit_pass_no_codes"],
    )
    people["is_unlicensed_adult"] = pd.Series(pd.NA, index=people.index, dtype="Int64")
    adult_valid = people["age_final"].ge(18) & people["has_car_license"].notna()
    people.loc[adult_valid, "is_unlicensed_adult"] = (
        people.loc[adult_valid, "has_car_license"] == 0
    ).astype(int)

    homes["household_sector_group"] = safe_map(
        homes["household_sector_code"], mappings["sector"], "Missing"
    )
    homes["income_group"] = safe_map(
        homes["income_category_code"], mappings["income"], "Missing_or_refused"
    )
    income_code = pd.to_numeric(homes["income_category_code"], errors="coerce")
    homes["is_low_income"] = pd.Series(pd.NA, index=homes.index, dtype="Int64")
    homes.loc[income_code.isin([1, 2]), "is_low_income"] = 1
    homes.loc[income_code.isin([3, 4, 5]), "is_low_income"] = 0

    vehicles = pd.to_numeric(homes["household_vehicles"], errors="coerce")
    homes["is_no_car"] = pd.Series(
        np.where(vehicles.notna(), vehicles.eq(0), np.nan), index=homes.index
    ).astype("Int64")
    homes["vehicle_group"] = pd.cut(
        vehicles,
        bins=[-0.1, 0, 1, np.inf],
        labels=["No_vehicle", "One_vehicle", "Two_or_more"],
    ).astype("string")

    child_columns = [
        "household_age_0_5",
        "household_age_6_11",
        "household_age_12_15",
        "household_age_16_17",
    ]
    older_columns = ["household_age_65_79", "household_age_80plus"]
    for column in child_columns + older_columns:
        homes[column] = pd.to_numeric(homes[column], errors="coerce")
    homes["household_children"] = homes[child_columns].fillna(0).sum(axis=1)
    homes["has_children"] = (homes["household_children"] > 0).astype("Int64")
    homes["household_older_members"] = homes[older_columns].fillna(0).sum(axis=1)
    homes["has_older_household_member"] = (
        homes["household_older_members"] > 0
    ).astype("Int64")
    homes["household_dependency_count"] = (
        homes["household_children"] + homes["household_older_members"]
    )

    return people, homes


def build_building_master(stops: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "uniq_id",
        "xcoord",
        "ycoord",
        "usg_code",
        "bldg_ht",
        "height",
        "ht_land",
        "floorspace",
        "hh_assets",
        "distance",
    ]:
        if column in stops.columns:
            stops[column] = pd.to_numeric(stops[column], errors="coerce")

    valid = stops.dropna(subset=["uniq_id", "xcoord", "ycoord"]).copy()
    valid["building_id"] = valid["uniq_id"].astype("Int64")

    buildings = (
        valid.groupby("building_id", as_index=False)
        .agg(
            building_x=("xcoord", "median"),
            building_y=("ycoord", "median"),
            usg_code=("usg_code", mode_or_first),
            building_height=("bldg_ht", "median"),
            elevation=("height", "median"),
            land_elevation=("ht_land", "median"),
            floorspace=("floorspace", "median"),
            building_household_assets=("hh_assets", "median"),
            stop_event_count=("person_id", "size"),
            mean_observation_to_building_distance_m=("distance", "mean"),
            max_observation_to_building_distance_m=("distance", "max"),
        )
    )
    buildings["building_use_group"] = (
        pd.to_numeric(buildings["usg_code"], errors="coerce")
        .astype("Int64")
        .map(BUILDING_USE_MAP)
        .fillna("Unknown")
    )
    return buildings


def _load_zone_polygons(path: Path, zone_column: str, source_crs: str) -> gpd.GeoDataFrame:
    zones = gpd.read_file(path)
    zones = normalize_columns(zones)
    zone_column_lower = zone_column.lower()
    if zone_column_lower not in zones.columns:
        raise ValueError(
            f"Zone polygon field {zone_column!r} is missing. Available: {list(zones.columns)}"
        )
    if zones.crs is None:
        LOGGER.warning("Zone polygons have no CRS; assigning %s from configuration.", source_crs)
        zones = zones.set_crs(source_crs)
    zones = zones.rename(columns={zone_column_lower: "zone_id"})
    zones["zone_id"] = pd.to_numeric(zones["zone_id"], errors="coerce").astype("Int64")
    zones = zones.dropna(subset=["zone_id", "geometry"]).copy()
    zones = zones[["zone_id", "geometry"]].drop_duplicates("zone_id")
    return zones


def assign_zone_to_points(
    points: pd.DataFrame,
    x_column: str,
    y_column: str,
    id_column: str,
    zones: gpd.GeoDataFrame,
    source_crs: str,
    nearest_fallback_max_distance_m: float,
    zone_output_column: str,
) -> pd.DataFrame:
    base = points.copy()
    base["__row_order"] = np.arange(len(base))
    valid = base[x_column].notna() & base[y_column].notna()
    point_gdf = gpd.GeoDataFrame(
        base.loc[valid].copy(),
        geometry=gpd.points_from_xy(base.loc[valid, x_column], base.loc[valid, y_column]),
        crs=source_crs,
    )
    zones_work = zones.to_crs(source_crs)

    joined = gpd.sjoin(
        point_gdf,
        zones_work[["zone_id", "geometry"]],
        how="left",
        predicate="within",
    )
    if joined.index.duplicated().any():
        joined = joined.sort_values(["__row_order", "zone_id"]).loc[
            ~joined.index.duplicated(keep="first")
        ]
    joined["zone_assignment_method"] = np.where(
        joined["zone_id"].notna(), "within", "unmatched"
    )
    joined["zone_assignment_distance_m"] = np.where(
        joined["zone_id"].notna(), 0.0, np.nan
    )

    unmatched = joined[joined["zone_id"].isna()].copy()
    if not unmatched.empty and nearest_fallback_max_distance_m > 0:
        unmatched = unmatched.drop(columns=["index_right", "zone_id"], errors="ignore")
        nearest = gpd.sjoin_nearest(
            unmatched,
            zones_work[["zone_id", "geometry"]],
            how="left",
            max_distance=nearest_fallback_max_distance_m,
            distance_col="__nearest_distance_m",
        )
        if nearest.index.duplicated().any():
            nearest = nearest.sort_values(["__nearest_distance_m", "zone_id"]).loc[
                ~nearest.index.duplicated(keep="first")
            ]
        joined.loc[nearest.index, "zone_id"] = nearest["zone_id"]
        matched_nearest = nearest["zone_id"].notna()
        joined.loc[nearest.index[matched_nearest], "zone_assignment_method"] = "nearest"
        joined.loc[
            nearest.index[matched_nearest], "zone_assignment_distance_m"
        ] = nearest.loc[matched_nearest, "__nearest_distance_m"]

    result = base.copy()
    result[zone_output_column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result[f"{zone_output_column}_assignment_method"] = "invalid_coordinate"
    result[f"{zone_output_column}_assignment_distance_m"] = np.nan
    result.loc[joined.index, zone_output_column] = joined["zone_id"].astype("Int64")
    result.loc[joined.index, f"{zone_output_column}_assignment_method"] = joined[
        "zone_assignment_method"
    ]
    result.loc[joined.index, f"{zone_output_column}_assignment_distance_m"] = joined[
        "zone_assignment_distance_m"
    ]
    return result.drop(columns=["__row_order"])


def assign_building_zones(
    buildings: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, gpd.GeoDataFrame | None]:
    zone_cfg = config["zone_assignment"]
    source_crs = config["project"]["source_crs"]
    mode = str(zone_cfg.get("mode", "polygon")).lower()

    if mode == "polygon":
        polygon_path = paths["taz_polygons"]
        if not polygon_path.exists():
            raise FileNotFoundError(
                "Official polygon zone assignment requires the TAZ shapefile. "
                f"File not found: {polygon_path}"
            )
        zones = _load_zone_polygons(
            polygon_path,
            zone_cfg["polygon_zone_column"],
            source_crs,
        )
        assigned = assign_zone_to_points(
            buildings,
            "building_x",
            "building_y",
            "building_id",
            zones,
            source_crs,
            float(zone_cfg["nearest_fallback_max_distance_m"]),
            "zone_id",
        )
        return assigned, zones

    if mode == "crosswalk":
        crosswalk_raw = zone_cfg.get("crosswalk_path")
        if not crosswalk_raw:
            raise ValueError("crosswalk_path must be supplied when zone_assignment.mode=crosswalk")
        config_dir = Path(config["__config_path__"]).parent
        base_dir = resolve_path(config.get("project", {}).get("base_dir", "."), config_dir)
        crosswalk_path = resolve_path(crosswalk_raw, base_dir)
        crosswalk = normalize_columns(pd.read_csv(crosswalk_path, low_memory=False))
        bcol = zone_cfg["crosswalk_building_column"].lower()
        zcol = zone_cfg["crosswalk_zone_column"].lower()
        validate_columns(crosswalk, [bcol, zcol], "zone crosswalk")
        crosswalk = crosswalk[[bcol, zcol]].drop_duplicates(bcol)
        crosswalk = crosswalk.rename(columns={bcol: "building_id", zcol: "zone_id"})
        crosswalk["building_id"] = pd.to_numeric(
            crosswalk["building_id"], errors="coerce"
        ).astype("Int64")
        crosswalk["zone_id"] = pd.to_numeric(crosswalk["zone_id"], errors="coerce").astype(
            "Int64"
        )
        assigned = buildings.merge(crosswalk, on="building_id", how="left", validate="1:1")
        assigned["zone_id_assignment_method"] = np.where(
            assigned["zone_id"].notna(), "crosswalk", "unmatched"
        )
        assigned["zone_id_assignment_distance_m"] = np.nan
        return assigned, None

    raise ValueError(f"Unsupported zone_assignment.mode: {mode}")


def merge_accessibility(
    buildings: pd.DataFrame,
    accessibility_path: Path,
    zone_column: str,
) -> pd.DataFrame:
    accessibility = normalize_columns(pd.read_excel(accessibility_path))
    zone_col = zone_column.lower()
    validate_columns(accessibility, [zone_col, "ai", "gap_i"], "TAZ accessibility")
    accessibility = accessibility[[zone_col, "ai", "gap_i"]].copy()
    accessibility = accessibility.rename(columns={zone_col: "zone_id"})
    accessibility["zone_id"] = pd.to_numeric(
        accessibility["zone_id"], errors="coerce"
    ).astype("Int64")
    accessibility["ai"] = pd.to_numeric(accessibility["ai"], errors="coerce")
    accessibility["gap_i"] = pd.to_numeric(accessibility["gap_i"], errors="coerce").astype(
        "Int64"
    )
    if accessibility["zone_id"].duplicated().any():
        duplicates = accessibility.loc[
            accessibility["zone_id"].duplicated(False), "zone_id"
        ].tolist()
        raise ValueError(f"TAZ accessibility has duplicate zone IDs: {duplicates[:10]}")
    output = buildings.merge(accessibility, on="zone_id", how="left", validate="m:1")
    output["gap_label_available"] = output["gap_i"].notna().astype("Int64")
    return output


def attach_poi_attributes(
    buildings: pd.DataFrame,
    poi_path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    source_crs = config["project"]["source_crs"]
    poi_cfg = config["poi"]
    poi = normalize_columns(pd.read_excel(poi_path))
    validate_columns(poi, ["osm_id", "fclass", "name", "type", "point_x", "point_y"], "building POI")
    poi["point_x"] = pd.to_numeric(poi["point_x"], errors="coerce")
    poi["point_y"] = pd.to_numeric(poi["point_y"], errors="coerce")
    poi = poi[
        poi["point_x"].between(-180, 180)
        & poi["point_y"].between(-90, 90)
    ].copy()
    poi["poi_type_clean"] = poi["type"].astype("string").str.strip().replace({"": pd.NA})
    poi["poi_name_clean"] = poi["name"].astype("string").str.strip().replace({"": pd.NA})

    building_gdf = gpd.GeoDataFrame(
        buildings.copy(),
        geometry=gpd.points_from_xy(buildings["building_x"], buildings["building_y"]),
        crs=source_crs,
    )
    poi_gdf = gpd.GeoDataFrame(
        poi.copy(),
        geometry=gpd.points_from_xy(poi["point_x"], poi["point_y"]),
        crs=config["project"]["geographic_crs"],
    ).to_crs(source_crs)

    building_gdf["__building_order"] = np.arange(len(building_gdf))
    nearest = gpd.sjoin_nearest(
        building_gdf,
        poi_gdf[
            ["osm_id", "code", "fclass", "name", "type", "poi_type_clean", "poi_name_clean", "geometry"]
        ],
        how="left",
        max_distance=float(poi_cfg["primary_poi_max_distance_m"]),
        distance_col="primary_poi_distance_m",
    )
    nearest["__has_specific_type"] = nearest["poi_type_clean"].notna().astype(int)
    nearest["__has_name"] = nearest["poi_name_clean"].notna().astype(int)
    nearest = nearest.sort_values(
        [
            "__building_order",
            "primary_poi_distance_m",
            "__has_specific_type",
            "__has_name",
            "osm_id",
        ],
        ascending=[True, True, False, False, True],
        na_position="last",
    )
    nearest = nearest.drop_duplicates("building_id", keep="first")
    nearest = nearest.rename(
        columns={
            "osm_id": "primary_poi_osm_id",
            "code": "primary_poi_code",
            "fclass": "primary_poi_fclass",
            "name": "primary_poi_name",
            "type": "primary_poi_type",
        }
    )

    keep = [
        "building_id",
        "primary_poi_osm_id",
        "primary_poi_code",
        "primary_poi_fclass",
        "primary_poi_name",
        "primary_poi_type",
        "primary_poi_distance_m",
    ]
    output = buildings.merge(pd.DataFrame(nearest[keep]), on="building_id", how="left", validate="1:1")

    if bool(poi_cfg.get("compute_nearby_poi_summary", True)):
        radius = float(poi_cfg["nearby_poi_radius_m"])
        buffers = building_gdf[["building_id", "geometry"]].copy()
        buffers["geometry"] = buffers.geometry.buffer(radius)
        nearby = gpd.sjoin(
            poi_gdf[["osm_id", "type", "geometry"]],
            buffers,
            how="inner",
            predicate="within",
        )
        if nearby.empty:
            summary = pd.DataFrame(
                columns=["building_id", "nearby_poi_count", "nearby_poi_type_count", "nearby_poi_types"]
            )
        else:
            nearby["poi_type_for_summary"] = nearby["type"].fillna("unknown").map(sanitize_token)
            summary = (
                nearby.groupby("building_id", as_index=False)
                .agg(
                    nearby_poi_count=("osm_id", "nunique"),
                    nearby_poi_type_count=("poi_type_for_summary", "nunique"),
                    nearby_poi_types=(
                        "poi_type_for_summary",
                        lambda values: "|".join(sorted(set(values))),
                    ),
                )
            )
        output = output.merge(summary, on="building_id", how="left", validate="1:1")
        output["nearby_poi_count"] = output["nearby_poi_count"].fillna(0).astype(int)
        output["nearby_poi_type_count"] = output["nearby_poi_type_count"].fillna(0).astype(int)
        output["nearby_poi_types"] = output["nearby_poi_types"].fillna("")

    primary_type = output["primary_poi_type"].astype("string").str.strip().replace({"": pd.NA})
    output["activity_type"] = primary_type.fillna(output["building_use_group"]).fillna("Unknown")
    return output


def build_enriched_events(
    stops: pd.DataFrame,
    people: pd.DataFrame,
    households: pd.DataFrame,
    buildings: pd.DataFrame,
    zones: gpd.GeoDataFrame | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    events = stops.copy()
    events["household_id"] = pd.to_numeric(
        events["person_id"].map(people.set_index("person_id")["household_id"]),
        errors="coerce",
    ).astype("Int64")
    events["stop_seq"] = pd.to_numeric(events["stopnum"], errors="coerce").astype("Int64")
    events["arrival_time"] = pd.to_datetime(events["stoptime"], errors="coerce")
    events["building_id"] = pd.to_numeric(events["uniq_id"], errors="coerce").astype("Int64")
    events["observed_x"] = pd.to_numeric(events["x"], errors="coerce")
    events["observed_y"] = pd.to_numeric(events["y"], errors="coerce")
    events["observation_to_building_distance_m"] = pd.to_numeric(
        events["distance"], errors="coerce"
    )
    events["from_home"] = pd.to_numeric(events["fromhome"], errors="coerce").astype("Int64")
    events["is_home"] = pd.to_numeric(events["ishome"], errors="coerce").astype("Int64")

    event_columns = [
        "person_id",
        "household_id",
        "stop_seq",
        "arrival_time",
        "building_id",
        "observed_x",
        "observed_y",
        "observation_to_building_distance_m",
        "from_home",
        "is_home",
    ]
    events = events[event_columns].copy()
    events = events.dropna(
        subset=["person_id", "household_id", "stop_seq", "arrival_time", "building_id"]
    )

    events = events.merge(buildings, on="building_id", how="left", validate="m:1")
    events = events.merge(people, on=["person_id", "household_id"], how="left", validate="m:1")
    events = events.merge(households, on="household_id", how="left", validate="m:1")

    if events["person_number"].isna().any():
        raise RuntimeError(f"{events['person_number'].isna().sum()} events failed to match persons.")
    if events["household_size"].isna().any():
        LOGGER.warning(
            "%d events failed to match a populated household record.",
            int(events["household_size"].isna().sum()),
        )

    events["date"] = events["arrival_time"].dt.date.astype("string")
    events["survey_day_id"] = events["person_id"] + "_" + events["date"]
    events["day_of_week"] = events["arrival_time"].dt.day_name().str.lower()
    # Israeli workweek: Sunday-Thursday; weekend: Friday-Saturday.
    events["day_type"] = np.where(
        events["arrival_time"].dt.dayofweek.isin([4, 5]),
        "weekend",
        "weekday",
    )
    events["arrival_hour"] = events["arrival_time"].dt.hour.astype("Int64")
    events["arrival_minute_of_day"] = (
        events["arrival_time"].dt.hour * 60 + events["arrival_time"].dt.minute
    ).astype("Int64")
    events["arrival_time_bin_30min"] = (
        (events["arrival_minute_of_day"] // 30) * 30
    ).astype("Int64")
    events["event_id"] = (
        events["survey_day_id"]
        + "_stop_"
        + events["stop_seq"].astype("string")
    )

    max_match_distance = float(config["quality"]["max_observation_to_building_distance_m"])
    events["building_match_ok"] = (
        events["observation_to_building_distance_m"].le(max_match_distance)
    ).astype("Int64")

    if zones is not None:
        observed_assignment = assign_zone_to_points(
            events[["event_id", "observed_x", "observed_y"]],
            "observed_x",
            "observed_y",
            "event_id",
            zones,
            config["project"]["source_crs"],
            float(config["zone_assignment"]["nearest_fallback_max_distance_m"]),
            "observed_zone_id",
        )
        events = events.merge(
            observed_assignment[
                [
                    "event_id",
                    "observed_zone_id",
                    "observed_zone_id_assignment_method",
                    "observed_zone_id_assignment_distance_m",
                ]
            ],
            on="event_id",
            how="left",
            validate="1:1",
        )
        events["observed_and_building_zone_match"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
        comparable = events["observed_zone_id"].notna() & events["zone_id"].notna()
        events.loc[comparable, "observed_and_building_zone_match"] = (
            events.loc[comparable, "observed_zone_id"] == events.loc[comparable, "zone_id"]
        ).astype(int)
    else:
        events["observed_zone_id"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
        events["observed_and_building_zone_match"] = pd.Series(
            pd.NA, index=events.index, dtype="Int64"
        )

    events = events.sort_values(
        ["survey_day_id", "stop_seq", "arrival_time", "event_id"]
    ).reset_index(drop=True)
    grouped = events.groupby("survey_day_id", sort=False)
    for source, target in [
        ("event_id", "previous_event_id"),
        ("stop_seq", "previous_stop_seq"),
        ("arrival_time", "previous_arrival_time"),
        ("building_id", "previous_building_id"),
        ("building_x", "previous_building_x"),
        ("building_y", "previous_building_y"),
        ("zone_id", "previous_zone_id"),
        ("gap_i", "previous_gap_i"),
        ("ai", "previous_ai"),
        ("activity_type", "previous_activity_type"),
        ("building_match_ok", "previous_building_match_ok"),
    ]:
        events[target] = grouped[source].shift(1)

    events["sequence_is_consecutive"] = (
        events["stop_seq"] == events["previous_stop_seq"] + 1
    ).astype("Int64")
    events["minutes_since_previous_arrival"] = (
        events["arrival_time"] - events["previous_arrival_time"]
    ).dt.total_seconds() / 60.0
    return events


def build_household_home_master(
    events: pd.DataFrame,
    households: pd.DataFrame,
    buildings: pd.DataFrame,
    zones: gpd.GeoDataFrame | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    home_events = events[events["is_home"].eq(1) & events["building_id"].notna()].copy()
    if home_events.empty:
        event_anchors = pd.DataFrame(columns=["household_id", "home_building_id"])
    else:
        candidate = (
            home_events.groupby(["household_id", "building_id"], as_index=False)
            .agg(
                home_event_count=("event_id", "size"),
                median_home_match_distance_m=(
                    "observation_to_building_distance_m",
                    "median",
                ),
            )
            .sort_values(
                [
                    "household_id",
                    "home_event_count",
                    "median_home_match_distance_m",
                    "building_id",
                ],
                ascending=[True, False, True, True],
            )
        )
        event_anchors = candidate.drop_duplicates("household_id", keep="first").rename(
            columns={"building_id": "home_building_id"}
        )

    homes = households.copy()
    homes = homes.merge(event_anchors, on="household_id", how="left", validate="1:1")
    building_cols = [
        "building_id",
        "building_x",
        "building_y",
        "zone_id",
        "ai",
        "gap_i",
        "activity_type",
    ]
    homes = homes.merge(
        buildings[building_cols].rename(
            columns={
                "building_id": "home_building_id",
                "building_x": "event_home_x",
                "building_y": "event_home_y",
                "zone_id": "event_home_zone_id",
                "ai": "event_home_ai",
                "gap_i": "event_home_gap_i",
                "activity_type": "event_home_activity_type",
            }
        ),
        on="home_building_id",
        how="left",
        validate="m:1",
    )

    homes["raw_home_coordinate_valid"] = (
        pd.to_numeric(homes["household_x"], errors="coerce").notna()
        & pd.to_numeric(homes["household_y"], errors="coerce").notna()
        & pd.to_numeric(homes["household_x"], errors="coerce").ne(0)
        & pd.to_numeric(homes["household_y"], errors="coerce").ne(0)
    )
    homes["home_anchor_source"] = np.select(
        [homes["home_building_id"].notna(), homes["raw_home_coordinate_valid"]],
        ["observed_home_building", "household_coordinate"],
        default="unavailable",
    )
    homes["home_x"] = np.where(
        homes["home_building_id"].notna(), homes["event_home_x"], homes["household_x"]
    )
    homes["home_y"] = np.where(
        homes["home_building_id"].notna(), homes["event_home_y"], homes["household_y"]
    )
    homes["home_location_id"] = np.where(
        homes["home_building_id"].notna(),
        "BUILDING_" + homes["home_building_id"].astype("Int64").astype("string"),
        "HOME_" + homes["household_id"].astype("string"),
    )

    coordinate_only = homes["home_building_id"].isna() & homes["raw_home_coordinate_valid"]
    homes["home_zone_id"] = homes["event_home_zone_id"].astype("Int64")
    homes["home_ai"] = homes["event_home_ai"]
    homes["home_gap_i"] = homes["event_home_gap_i"].astype("Int64")

    if coordinate_only.any() and zones is not None:
        coordinate_assignment = assign_zone_to_points(
            homes.loc[coordinate_only, ["household_id", "home_x", "home_y"]],
            "home_x",
            "home_y",
            "household_id",
            zones,
            config["project"]["source_crs"],
            float(config["zone_assignment"]["nearest_fallback_max_distance_m"]),
            "raw_home_zone_id",
        )
        zone_lookup = coordinate_assignment.set_index("household_id")["raw_home_zone_id"]
        homes.loc[coordinate_only, "home_zone_id"] = homes.loc[
            coordinate_only, "household_id"
        ].map(zone_lookup).astype("Int64")

        accessibility_lookup = buildings[["zone_id", "ai", "gap_i"]].drop_duplicates("zone_id")
        ai_lookup = accessibility_lookup.set_index("zone_id")["ai"]
        gap_lookup = accessibility_lookup.set_index("zone_id")["gap_i"]
        homes.loc[coordinate_only, "home_ai"] = homes.loc[coordinate_only, "home_zone_id"].map(
            ai_lookup
        )
        homes.loc[coordinate_only, "home_gap_i"] = homes.loc[
            coordinate_only, "home_zone_id"
        ].map(gap_lookup).astype("Int64")

    homes["home_activity_type"] = "home"
    homes["home_anchor_available"] = homes["home_anchor_source"].ne("unavailable").astype(
        "Int64"
    )
    homes["home_gap_label_available"] = homes["home_gap_i"].notna().astype("Int64")

    keep = [
        "household_id",
        "home_location_id",
        "home_building_id",
        "home_anchor_source",
        "home_anchor_available",
        "home_x",
        "home_y",
        "home_zone_id",
        "home_ai",
        "home_gap_i",
        "home_gap_label_available",
        "home_activity_type",
    ]
    return homes[keep].copy()


def add_welfare_score(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = df.copy()
    weights = config["welfare"]
    components = {
        "is_low_income": float(weights["low_income_weight"]),
        "is_no_car": float(weights["no_car_weight"]),
        "is_disabled": float(weights["disability_weight"]),
        "is_older": float(weights["older_weight"]),
        "is_student": float(weights["student_weight"]),
    }
    max_score = sum(components.values())
    weighted = pd.Series(0.0, index=output.index)
    available_weight = pd.Series(0.0, index=output.index)
    missing_count = pd.Series(0, index=output.index, dtype=int)
    for column, weight in components.items():
        values = pd.to_numeric(output[column], errors="coerce")
        weighted += values.fillna(0) * weight
        available_weight += values.notna().astype(float) * weight
        missing_count += values.isna().astype(int)
    output["social_welfare_points"] = weighted
    output["social_welfare_score"] = weighted / max_score
    output["social_welfare_score_available_normalized"] = np.where(
        available_weight > 0, weighted / available_weight, np.nan
    )
    output["equity_missing_component_count"] = missing_count
    output["equity_core_complete"] = missing_count.eq(0).astype("Int64")
    return output


def build_od_trips(
    events: pd.DataFrame,
    home_master: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    homes = home_master.set_index("household_id")
    trips = events.copy()
    trips["home_anchor_available"] = trips["household_id"].map(homes["home_anchor_available"])
    for source in [
        "home_location_id",
        "home_building_id",
        "home_x",
        "home_y",
        "home_zone_id",
        "home_ai",
        "home_gap_i",
        "home_activity_type",
    ]:
        trips[source] = trips["household_id"].map(homes[source])

    from_home = trips["from_home"].eq(1) & trips["home_anchor_available"].eq(1)
    previous_valid = (
        trips["sequence_is_consecutive"].eq(1)
        & trips["previous_building_id"].notna()
        & trips["minutes_since_previous_arrival"].gt(0)
    )
    use_previous = ~from_home & previous_valid

    trips["origin_source"] = np.select(
        [from_home, use_previous],
        ["home_anchor", "previous_building"],
        default="unavailable",
    )
    trips["origin_location_id"] = np.where(
        from_home,
        trips["home_location_id"],
        np.where(
            use_previous,
            "BUILDING_" + trips["previous_building_id"].astype("Int64").astype("string"),
            pd.NA,
        ),
    )
    trips["origin_building_id"] = pd.Series(pd.NA, index=trips.index, dtype="Int64")
    trips.loc[from_home, "origin_building_id"] = trips.loc[from_home, "home_building_id"].astype(
        "Int64"
    )
    trips.loc[use_previous, "origin_building_id"] = trips.loc[
        use_previous, "previous_building_id"
    ].astype("Int64")
    trips["origin_x"] = np.where(
        from_home, trips["home_x"], np.where(use_previous, trips["previous_building_x"], np.nan)
    )
    trips["origin_y"] = np.where(
        from_home, trips["home_y"], np.where(use_previous, trips["previous_building_y"], np.nan)
    )
    trips["origin_zone_id"] = pd.Series(pd.NA, index=trips.index, dtype="Int64")
    trips.loc[from_home, "origin_zone_id"] = trips.loc[from_home, "home_zone_id"].astype("Int64")
    trips.loc[use_previous, "origin_zone_id"] = trips.loc[
        use_previous, "previous_zone_id"
    ].astype("Int64")
    trips["origin_ai"] = np.where(
        from_home, trips["home_ai"], np.where(use_previous, trips["previous_ai"], np.nan)
    )
    trips["origin_gap"] = pd.Series(pd.NA, index=trips.index, dtype="Int64")
    trips.loc[from_home, "origin_gap"] = trips.loc[from_home, "home_gap_i"].astype("Int64")
    trips.loc[use_previous, "origin_gap"] = trips.loc[use_previous, "previous_gap_i"].astype(
        "Int64"
    )
    trips["origin_activity_type"] = np.where(
        from_home,
        "home",
        np.where(use_previous, trips["previous_activity_type"], pd.NA),
    )

    to_home = trips["is_home"].eq(1) & trips["home_anchor_available"].eq(1)
    trips["destination_source"] = np.where(to_home, "home_anchor", "current_building")
    trips["destination_location_id"] = np.where(
        to_home,
        trips["home_location_id"],
        "BUILDING_" + trips["building_id"].astype("Int64").astype("string"),
    )
    trips["destination_building_id"] = pd.Series(pd.NA, index=trips.index, dtype="Int64")
    trips.loc[to_home, "destination_building_id"] = trips.loc[to_home, "home_building_id"].astype(
        "Int64"
    )
    trips.loc[~to_home, "destination_building_id"] = trips.loc[~to_home, "building_id"].astype(
        "Int64"
    )
    trips["destination_x"] = np.where(to_home, trips["home_x"], trips["building_x"])
    trips["destination_y"] = np.where(to_home, trips["home_y"], trips["building_y"])
    trips["destination_zone_id"] = pd.Series(pd.NA, index=trips.index, dtype="Int64")
    trips.loc[to_home, "destination_zone_id"] = trips.loc[to_home, "home_zone_id"].astype(
        "Int64"
    )
    trips.loc[~to_home, "destination_zone_id"] = trips.loc[~to_home, "zone_id"].astype("Int64")
    trips["destination_ai"] = np.where(to_home, trips["home_ai"], trips["ai"])
    trips["destination_gap"] = pd.Series(pd.NA, index=trips.index, dtype="Int64")
    trips.loc[to_home, "destination_gap"] = trips.loc[to_home, "home_gap_i"].astype("Int64")
    trips.loc[~to_home, "destination_gap"] = trips.loc[~to_home, "gap_i"].astype("Int64")
    trips["destination_activity_type"] = np.where(to_home, "home", trips["activity_type"])

    trips["destination_arrival_time"] = trips["arrival_time"]
    trips["trip_time_reference"] = trips["destination_arrival_time"]
    trips["straight_line_distance_m"] = np.hypot(
        trips["destination_x"] - trips["origin_x"],
        trips["destination_y"] - trips["origin_y"],
    )
    trips["is_gap"] = trips["origin_gap"].astype("Int64")
    trips["touches_gap"] = pd.Series(pd.NA, index=trips.index, dtype="Int64")
    gap_known = trips["origin_gap"].notna() & trips["destination_gap"].notna()
    trips.loc[gap_known, "touches_gap"] = (
        trips.loc[gap_known, "origin_gap"].eq(1)
        | trips.loc[gap_known, "destination_gap"].eq(1)
    ).astype(int)

    trips["trip_id"] = (
        trips["survey_day_id"]
        + "_to_stop_"
        + trips["stop_seq"].astype("string")
    )
    trips["reveal_time_sec"] = 0.0

    trips["invalid_origin"] = trips["origin_source"].eq("unavailable")
    trips["invalid_distance"] = trips["straight_line_distance_m"].lt(
        float(config["quality"]["minimum_od_distance_m"])
    ) | trips["straight_line_distance_m"].isna()
    trips["invalid_zone"] = trips["origin_zone_id"].isna() | trips[
        "destination_zone_id"
    ].isna()
    trips["invalid_gap_label"] = trips["origin_gap"].isna() | trips[
        "destination_gap"
    ].isna()
    origin_previous_bad = use_previous & trips["previous_building_match_ok"].ne(1)
    trips["invalid_building_match"] = trips["building_match_ok"].ne(1) | origin_previous_bad

    reason_columns = {
        "origin_unavailable": "invalid_origin",
        "distance_below_threshold_or_missing": "invalid_distance",
        "zone_missing": "invalid_zone",
        "gap_label_missing": "invalid_gap_label",
        "destination_building_match_over_threshold": "invalid_building_match",
    }
    def reasons_for_row(row: pd.Series) -> str:
        return "|".join(reason for reason, column in reason_columns.items() if bool(row[column]))

    trips["invalid_reason"] = trips[list(reason_columns.values())].apply(reasons_for_row, axis=1)
    trips["od_complete"] = (~trips["invalid_origin"] & ~trips["invalid_distance"]).astype("Int64")

    model_ready = ~trips["invalid_origin"] & ~trips["invalid_distance"]
    quality_cfg = config["quality"]
    if bool(quality_cfg["require_zone_for_model_ready"]):
        model_ready &= ~trips["invalid_zone"]
    if bool(quality_cfg["require_gap_label_for_model_ready"]):
        model_ready &= ~trips["invalid_gap_label"]
    if bool(quality_cfg["require_building_match_quality_for_model_ready"]):
        model_ready &= ~trips["invalid_building_match"]
    trips["model_ready"] = model_ready.astype("Int64")

    trips = add_welfare_score(trips, config)

    selected_columns = [
        "trip_id",
        "person_id",
        "household_id",
        "survey_day_id",
        "stop_seq",
        "destination_arrival_time",
        "trip_time_reference",
        "reveal_time_sec",
        "origin_source",
        "origin_location_id",
        "origin_building_id",
        "origin_x",
        "origin_y",
        "origin_zone_id",
        "origin_ai",
        "origin_gap",
        "origin_activity_type",
        "destination_source",
        "destination_location_id",
        "destination_building_id",
        "destination_x",
        "destination_y",
        "destination_zone_id",
        "destination_ai",
        "destination_gap",
        "destination_activity_type",
        "straight_line_distance_m",
        "is_gap",
        "touches_gap",
        "sector_group",
        "gender_group",
        "age_final",
        "age_group",
        "is_older",
        "is_80plus",
        "disability_code",
        "is_disabled",
        "student_code",
        "is_student",
        "employment_code",
        "is_employed",
        "car_license_code",
        "has_car_license",
        "is_unlicensed_adult",
        "transit_pass_code",
        "has_transit_pass",
        "education_code",
        "immigration_code",
        "household_sector_group",
        "income_category_code",
        "income_group",
        "is_low_income",
        "household_vehicles",
        "vehicle_group",
        "is_no_car",
        "household_size",
        "household_children",
        "has_children",
        "household_older_members",
        "has_older_household_member",
        "household_dependency_count",
        "social_welfare_points",
        "social_welfare_score",
        "social_welfare_score_available_normalized",
        "equity_missing_component_count",
        "equity_core_complete",
        "building_match_ok",
        "sequence_is_consecutive",
        "minutes_since_previous_arrival",
        "od_complete",
        "model_ready",
        "invalid_reason",
    ]
    selected_columns = [column for column in selected_columns if column in trips.columns]
    trips_all = trips[selected_columns].copy().sort_values(
        ["household_id", "person_id", "destination_arrival_time", "stop_seq"]
    )
    trips_ready = trips_all[trips_all["model_ready"].eq(1)].copy()
    return trips_all.reset_index(drop=True), trips_ready.reset_index(drop=True)


def _safe_stratified_split(
    frame: pd.DataFrame,
    train_size: float,
    seed: int,
    stratify_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = frame[stratify_column].astype("string")
    counts = stratify.value_counts(dropna=False)
    rare = counts[counts < 2].index
    stratify = stratify.where(~stratify.isin(rare), "RARE")
    try:
        left, right = train_test_split(
            frame,
            train_size=train_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as error:
        LOGGER.warning("Stratified split failed (%s); using deterministic random split.", error)
        left, right = train_test_split(
            frame,
            train_size=train_size,
            random_state=seed,
        )
    return left.copy(), right.copy()


def split_by_household(
    events: pd.DataFrame,
    trips_model_ready: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    split_cfg = config["split"]
    train_ratio = float(split_cfg["train_ratio"])
    validation_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Train/validation/test ratios must sum to 1.")

    household_base = (
        events.groupby("household_id", as_index=False)
        .agg(
            sector_group=("sector_group", mode_or_first),
            is_low_income=("is_low_income", mode_or_first),
            is_no_car=("is_no_car", mode_or_first),
            event_count=("event_id", "size"),
            person_count=("person_id", "nunique"),
        )
    )
    gap_households = (
        trips_model_ready.groupby("household_id", as_index=False)
        .agg(
            has_gap_trip=("is_gap", lambda values: int(pd.Series(values).eq(1).any())),
            model_ready_trip_count=("trip_id", "size"),
        )
    )
    household_base = household_base.merge(
        gap_households, on="household_id", how="left", validate="1:1"
    )
    household_base["has_gap_trip"] = household_base["has_gap_trip"].fillna(0).astype(int)
    household_base["model_ready_trip_count"] = (
        household_base["model_ready_trip_count"].fillna(0).astype(int)
    )
    household_base["split_stratum"] = (
        household_base["sector_group"].fillna("Missing").astype(str)
        + "__gap_"
        + household_base["has_gap_trip"].astype(str)
    )

    seed = int(config["project"]["seed"])
    if bool(split_cfg.get("stratify_by_sector_and_gap", True)):
        train_households, temporary = _safe_stratified_split(
            household_base,
            train_size=train_ratio,
            seed=seed,
            stratify_column="split_stratum",
        )
        validation_share = validation_ratio / (validation_ratio + test_ratio)
        validation_households, test_households = _safe_stratified_split(
            temporary,
            train_size=validation_share,
            seed=seed + 1,
            stratify_column="split_stratum",
        )
    else:
        train_households, temporary = train_test_split(
            household_base, train_size=train_ratio, random_state=seed
        )
        validation_share = validation_ratio / (validation_ratio + test_ratio)
        validation_households, test_households = train_test_split(
            temporary, train_size=validation_share, random_state=seed + 1
        )

    train_households["split"] = "train"
    validation_households["split"] = "validation"
    test_households["split"] = "test"
    manifest = pd.concat(
        [train_households, validation_households, test_households], ignore_index=True
    ).sort_values("household_id")

    split_sets = {
        name: set(manifest.loc[manifest["split"].eq(name), "household_id"].tolist())
        for name in ["train", "validation", "test"]
    }
    if split_sets["train"] & split_sets["validation"]:
        raise RuntimeError("Household leakage between train and validation.")
    if split_sets["train"] & split_sets["test"]:
        raise RuntimeError("Household leakage between train and test.")
    if split_sets["validation"] & split_sets["test"]:
        raise RuntimeError("Household leakage between validation and test.")
    if len(manifest) != manifest["household_id"].nunique():
        raise RuntimeError("A household appears more than once in the split manifest.")
    return manifest.reset_index(drop=True)


def write_table(df: pd.DataFrame, path_without_suffix: Path, write_csv: bool) -> None:
    global _PARQUET_WARNING_EMITTED
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    if PARQUET_ENGINE_AVAILABLE:
        df.to_parquet(path_without_suffix.with_suffix(".parquet"), index=False)
    elif not _PARQUET_WARNING_EMITTED:
        LOGGER.warning(
            "Parquet outputs are being skipped because pyarrow/fastparquet is unavailable. "
            "CSV outputs will still be written. Install pyarrow in the official Python 3.11 environment."
        )
        _PARQUET_WARNING_EMITTED = True
    if write_csv:
        df.to_csv(path_without_suffix.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def write_geopackage(
    df: pd.DataFrame,
    path: Path,
    layer: str,
    x_column: str,
    y_column: str,
    crs: str,
) -> None:
    valid = df[x_column].notna() & df[y_column].notna()
    gdf = gpd.GeoDataFrame(
        df.loc[valid].copy(),
        geometry=gpd.points_from_xy(df.loc[valid, x_column], df.loc[valid, y_column]),
        crs=crs,
    )
    gdf.to_file(path, layer=layer, driver="GPKG")


def build_equity_dictionary() -> pd.DataFrame:
    rows = [
        ("sector_group", "Person", "Ethno-religious sector: Arab, Ultra-Orthodox, Secular, or other/unknown.", "Audit", False),
        ("gender_group", "Person", "Gender group from GEND; used for outcome auditing, not preferential scoring.", "Audit", False),
        ("age_group", "Person", "Age group based on AGE_R with AGE fallback.", "Audit", False),
        ("is_older", "Person", "Person aged 65 years or older.", "Core vulnerability", True),
        ("is_80plus", "Person", "Person aged 80 years or older.", "Audit", False),
        ("is_disabled", "Person", "Travel-limiting disability or circumstance.", "Core vulnerability", True),
        ("is_student", "Person", "Currently enrolled in school, technical school, daycare, or university.", "Core/secondary vulnerability", True),
        ("is_employed", "Person", "Employment indicator derived from configured survey codes.", "Audit", False),
        ("is_unlicensed_adult", "Person", "Adult without a valid car driving licence.", "Secondary transport disadvantage", False),
        ("has_transit_pass", "Person", "Transit pass ownership indicator from configured codes.", "Audit", False),
        ("income_group", "Household", "Household income category.", "Audit", False),
        ("is_low_income", "Household", "Income category below 50K (categories 1 or 2).", "Core vulnerability", True),
        ("vehicle_group", "Household", "No vehicle, one vehicle, or two or more vehicles.", "Audit", False),
        ("is_no_car", "Household", "Household owns zero vehicles.", "Core vulnerability", True),
        ("has_children", "Household", "Household contains at least one member younger than 18.", "Household structure audit", False),
        ("has_older_household_member", "Household", "Household contains at least one member aged 65 or older.", "Household structure audit", False),
        ("household_dependency_count", "Household", "Count of household members aged under 18 or 65 and older.", "Household structure audit", False),
        ("is_gap", "Spatial", "Origin TAZ has gap_i=1.", "Spatial equity objective", False),
        ("social_welfare_score", "Derived", "Weighted vulnerability score; sector and gender are deliberately excluded.", "Optimization", True),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "field",
            "level",
            "definition",
            "recommended_role",
            "included_in_social_welfare_score",
        ],
    )


def produce_audits(
    stops: pd.DataFrame,
    persons: pd.DataFrame,
    households: pd.DataFrame,
    artifacts: PipelineArtifacts,
    output_dir: Path,
) -> None:
    events = artifacts.events
    trips_all = artifacts.trips_all
    trips_ready = artifacts.trips_model_ready
    manifest = artifacts.split_manifest

    audit_rows = [
        ("raw_stop_rows", len(stops)),
        ("raw_unique_person_ids_in_stops", stops["person_id"].nunique()),
        ("raw_person_rows", len(persons)),
        ("raw_household_rows", len(households)),
        ("unique_buildings", len(artifacts.buildings)),
        ("enriched_event_rows", len(events)),
        ("event_households", events["household_id"].nunique()),
        ("event_persons", events["person_id"].nunique()),
        ("events_building_match_over_threshold", int(events["building_match_ok"].ne(1).sum())),
        ("events_zone_missing", int(events["zone_id"].isna().sum())),
        ("events_gap_missing", int(events["gap_i"].isna().sum())),
        ("events_observed_building_zone_mismatch", int(events["observed_and_building_zone_match"].eq(0).sum())),
        ("trip_candidates", len(trips_all)),
        ("trips_origin_unavailable", int(trips_all["invalid_reason"].str.contains("origin_unavailable", na=False).sum())),
        ("trips_model_ready", len(trips_ready)),
        ("split_households", len(manifest)),
    ]
    pd.DataFrame(audit_rows, columns=["metric", "value"]).to_csv(
        output_dir / "audit_counts.csv", index=False, encoding="utf-8-sig"
    )

    invalid_summary = (
        trips_all.assign(
            reason=trips_all["invalid_reason"].replace("", "valid_or_other")
        )
        .groupby("reason", as_index=False)
        .agg(count=("trip_id", "size"))
        .sort_values("count", ascending=False)
    )
    invalid_summary.to_csv(
        output_dir / "trip_invalid_reason_summary.csv", index=False, encoding="utf-8-sig"
    )

    split_summary = (
        manifest.groupby("split", as_index=False)
        .agg(
            households=("household_id", "nunique"),
            events=("event_count", "sum"),
            persons=("person_count", "sum"),
            model_ready_trips=("model_ready_trip_count", "sum"),
            gap_households=("has_gap_trip", "sum"),
        )
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    equity_fields = [
        "sector_group",
        "gender_group",
        "age_group",
        "is_older",
        "is_disabled",
        "is_student",
        "income_group",
        "is_low_income",
        "vehicle_group",
        "is_no_car",
        "is_gap",
    ]
    if "split" in trips_ready.columns:
        ready_with_split = trips_ready.copy()
    else:
        ready_with_split = trips_ready.merge(
            manifest[["household_id", "split"]],
            on="household_id",
            how="left",
            validate="m:1",
        )
    equity_rows: list[pd.DataFrame] = []
    for field in equity_fields:
        if field not in ready_with_split.columns:
            continue
        table = (
            ready_with_split.groupby(["split", field], dropna=False, as_index=False)
            .agg(requests=("trip_id", "size"), households=("household_id", "nunique"))
        )
        table.insert(1, "attribute", field)
        table = table.rename(columns={field: "group"})
        equity_rows.append(table)
    if equity_rows:
        pd.concat(equity_rows, ignore_index=True).to_csv(
            output_dir / "split_equity_distribution.csv", index=False, encoding="utf-8-sig"
        )

    build_equity_dictionary().to_csv(
        output_dir / "equity_attribute_dictionary.csv", index=False, encoding="utf-8-sig"
    )


def write_selected_source_dictionary(paths: dict[str, Path], output_dir: Path) -> None:
    selected_person = {
        "SECTR", "GEND", "AGE", "AGE_R", "IMIG", "DISAB", "CLIC",
        "MLIC", "EMPLY", "JOBS", "WKSTA", "TPASS", "EDUC", "STUD", "SCHOL"
    }
    selected_household = {
        "SECTR", "HHSIZE", "HHEDUC", "HWORK_F", "HWORK_P", "HUNIV",
        "HNWORK", "HRETIRE", "HSCHDRIV", "HSCHPRED", "HPRESCH",
        "HADULTS18", "H0005", "H0611", "H1215", "H1617", "H1824",
        "H2534", "H3549", "H5064", "H6579", "H80UP", "HADNWST",
        "HADWPST", "HADKIDS", "HHVEH", "HINCCAT1", "HINCCAT2",
        "HOWNRENT", "HTYPDWEL", "LANG"
    }
    frames: list[pd.DataFrame] = []
    for level, key, selected in [
        ("person", "person_dictionary", selected_person),
        ("household", "household_dictionary", selected_household),
    ]:
        path = paths.get(key)
        if path is None or not path.exists():
            continue
        dictionary = normalize_columns(pd.read_csv(path, low_memory=False))
        validate_columns(dictionary, ["field_name", "field_desc"], f"{level} dictionary")
        dictionary["field_name_upper"] = dictionary["field_name"].astype(str).str.upper()
        subset = dictionary[dictionary["field_name_upper"].isin(selected)].copy()
        subset.insert(0, "source_level", level)
        keep = [
            column for column in [
                "source_level", "field_name", "field_desc", "field_val", "field_use"
            ] if column in subset.columns
        ]
        frames.append(subset[keep])
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            output_dir / "selected_source_dictionary_fields.csv",
            index=False,
            encoding="utf-8-sig",
        )


def write_metadata(
    config: dict[str, Any], paths: dict[str, Path], output_dir: Path
) -> None:
    input_keys = [
        "stops",
        "persons",
        "households",
        "person_dictionary",
        "household_dictionary",
        "building_poi",
        "taz_accessibility",
        "taz_polygons",
    ]
    inputs: dict[str, Any] = {}
    for key in input_keys:
        path = paths.get(key)
        if path is None:
            continue
        inputs[key] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config["__config_path__"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "geopandas": gpd.__version__,
        "shapely": shapely.__version__,
        "scikit_learn": sklearn.__version__,
        "inputs": inputs,
        "configuration": {key: value for key, value in config.items() if key != "__config_path__"},
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)


def run_pipeline(config_path: Path) -> PipelineArtifacts:
    config = load_config(config_path.resolve())
    config_dir = config_path.resolve().parent
    base_dir = resolve_path(config.get("project", {}).get("base_dir", "."), config_dir)
    paths = {
        key: resolve_path(value, base_dir)
        for key, value in config["paths"].items()
        if key != "output_dir"
    }
    output_dir = resolve_path(config["paths"]["output_dir"], base_dir)
    _configure_logging(output_dir)
    LOGGER.info("Starting experiment data pipeline.")
    LOGGER.info("Output directory: %s", output_dir)

    for required in [
        "stops",
        "persons",
        "households",
        "building_poi",
        "taz_accessibility",
    ]:
        if not paths[required].exists():
            raise FileNotFoundError(f"Required input file not found: {paths[required]}")

    stops, persons_raw, households_raw = load_raw_tables(paths)
    LOGGER.info(
        "Loaded %d stops, %d persons, and %d households.",
        len(stops),
        len(persons_raw),
        len(households_raw),
    )

    people, household_features = prepare_people_and_households(
        persons_raw, households_raw, config
    )

    buildings = build_building_master(stops)
    buildings, zones = assign_building_zones(buildings, config, paths)
    buildings = merge_accessibility(
        buildings,
        paths["taz_accessibility"],
        config["zone_assignment"]["accessibility_zone_column"],
    )
    buildings = attach_poi_attributes(buildings, paths["building_poi"], config)
    LOGGER.info("Built master table for %d unique buildings.", len(buildings))

    events = build_enriched_events(
        stops, people, household_features, buildings, zones, config
    )
    LOGGER.info("Built %d enriched dwell events.", len(events))

    home_master = build_household_home_master(
        events, household_features, buildings, zones, config
    )
    LOGGER.info(
        "Built %d household home anchors; %d are available.",
        len(home_master),
        int(home_master["home_anchor_available"].sum()),
    )

    trips_all, trips_ready = build_od_trips(events, home_master, config)
    LOGGER.info(
        "Built %d OD candidates; %d are model-ready.", len(trips_all), len(trips_ready)
    )

    split_manifest = split_by_household(events, trips_ready, config)
    split_lookup = split_manifest.set_index("household_id")["split"]
    events["split"] = events["household_id"].map(split_lookup)
    trips_all["split"] = trips_all["household_id"].map(split_lookup)
    trips_ready["split"] = trips_ready["household_id"].map(split_lookup)
    home_master["split"] = home_master["household_id"].map(split_lookup)

    write_csv = bool(config["outputs"].get("write_csv", True))
    write_table(buildings, output_dir / "building_master", write_csv)
    write_table(home_master, output_dir / "household_home_master", write_csv)
    write_table(events, output_dir / "dwell_events_master", write_csv)
    write_table(trips_all, output_dir / "survey_trips_all", write_csv)
    write_table(trips_ready, output_dir / "survey_trips_model_ready", write_csv)
    write_table(split_manifest, output_dir / "household_split_manifest", True)

    for split_name in ["train", "validation", "test"]:
        write_table(
            events[events["split"].eq(split_name)].copy(),
            output_dir / f"dwell_events_{split_name}",
            write_csv,
        )
        write_table(
            trips_ready[trips_ready["split"].eq(split_name)].copy(),
            output_dir / f"survey_trips_{split_name}",
            write_csv,
        )

    if bool(config["outputs"].get("write_gpkg", True)):
        try:
            building_gpkg = output_dir / "building_master.gpkg"
            home_gpkg = output_dir / "household_home_master.gpkg"
            for gpkg_path in [building_gpkg, home_gpkg]:
                if gpkg_path.exists():
                    gpkg_path.unlink()
            write_geopackage(
                buildings,
                building_gpkg,
                "buildings",
                "building_x",
                "building_y",
                config["project"]["source_crs"],
            )
            write_geopackage(
                home_master,
                home_gpkg,
                "household_homes",
                "home_x",
                "home_y",
                config["project"]["source_crs"],
            )
        except Exception as error:  # pragma: no cover - driver-dependent
            LOGGER.warning("Could not write GeoPackage: %s", error)

    artifacts = PipelineArtifacts(
        buildings=buildings,
        homes=home_master,
        events=events,
        trips_all=trips_all,
        trips_model_ready=trips_ready,
        split_manifest=split_manifest,
    )
    produce_audits(stops, persons_raw, households_raw, artifacts, output_dir)
    write_selected_source_dictionary(paths, output_dir)
    write_metadata(config, paths, output_dir)

    LOGGER.info("Pipeline completed successfully.")
    return artifacts
