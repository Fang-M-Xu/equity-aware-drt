from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def _natural_sort_key(value: Any) -> tuple[int, float | str]:
    """Sort numeric identifiers numerically and all other tokens lexicographically."""
    token = str(value)
    try:
        return (0, float(token))
    except (TypeError, ValueError):
        return (1, token)


class Vocabulary:
    def __init__(self, values: list[str] | None = None) -> None:
        self.token_to_index = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self._index_to_token: dict[int, str] | None = None
        if values is not None:
            for value in values:
                self.add(value)

    def _invalidate_reverse(self) -> None:
        self._index_to_token = None

    def add(self, value: Any) -> int:
        token = str(value)
        if token not in self.token_to_index:
            self.token_to_index[token] = len(self.token_to_index)
            self._invalidate_reverse()
        return self.token_to_index[token]

    def encode(self, value: Any) -> int:
        return self.token_to_index.get(str(value), self.token_to_index[UNK_TOKEN])

    def decode(self, index: int) -> str:
        if self._index_to_token is None:
            self._index_to_token = {
                int(value): str(key) for key, value in self.token_to_index.items()
            }
        return self._index_to_token.get(int(index), UNK_TOKEN)

    def __len__(self) -> int:
        return len(self.token_to_index)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_index)

    @classmethod
    def from_dict(cls, values: dict[str, int]) -> "Vocabulary":
        vocabulary = cls()
        vocabulary.token_to_index = {
            str(key): int(value) for key, value in values.items()
        }
        vocabulary._invalidate_reverse()
        return vocabulary


@dataclass
class VocabularyBundle:
    zone: Vocabulary
    activity: Vocabulary
    departure: Vocabulary
    duration: Vocabulary

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "zone": self.zone.to_dict(),
                    "activity": self.activity.to_dict(),
                    "departure": self.departure.to_dict(),
                    "duration": self.duration.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "VocabularyBundle":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            zone=Vocabulary.from_dict(payload["zone"]),
            activity=Vocabulary.from_dict(payload["activity"]),
            departure=Vocabulary.from_dict(payload["departure"]),
            duration=Vocabulary.from_dict(payload["duration"]),
        )


PROFILE_BASE_COLUMNS = [
    "is_low_income",
    "is_no_car",
    "is_disabled",
    "is_older",
    "is_student",
    "social_welfare_score",
    "household_size",
    "household_dependency_count",
]
PROFILE_CATEGORICAL_COLUMNS = ["sector_group", "gender_group", "age_group"]


def build_profile_matrix(
    frame: pd.DataFrame,
    fit_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    numeric = pd.DataFrame(index=frame.index)
    for column in PROFILE_BASE_COLUMNS:
        source = frame[column] if column in frame.columns else 0.0
        numeric[column] = pd.to_numeric(source, errors="coerce").fillna(0.0)

    categorical_source = (
        frame.reindex(columns=PROFILE_CATEGORICAL_COLUMNS)
        .fillna("Missing")
        .astype(str)
    )
    categorical = pd.get_dummies(
        categorical_source,
        prefix=PROFILE_CATEGORICAL_COLUMNS,
        dtype=float,
    )
    profile = pd.concat([numeric, categorical], axis=1)
    columns = sorted(profile.columns.tolist()) if fit_columns is None else fit_columns
    profile = profile.reindex(columns=columns, fill_value=0.0).astype(float)
    return profile, columns


def duration_token(seconds: float, bin_minutes: int, max_minutes: int) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    if bin_minutes <= 0:
        raise ValueError("duration_bin_minutes must be positive.")
    minutes = max(0, min(max_minutes, int(np.ceil(seconds / 60.0))))
    lower = minutes // bin_minutes * bin_minutes
    return str(lower)


def build_vocabularies(
    train: pd.DataFrame,
    duration_bin_minutes: int,
    max_duration_minutes: int,
) -> VocabularyBundle:
    zones = sorted(
        set(train["origin_zone_id"].dropna().astype(str))
        | set(train["destination_zone_id"].dropna().astype(str)),
        key=_natural_sort_key,
    )
    activities = sorted(
        set(train["origin_activity_type"].fillna("unknown").astype(str))
        | set(train["destination_activity_type"].fillna("unknown").astype(str)),
        key=_natural_sort_key,
    )
    departures = sorted(
        train["estimated_departure_bin"]
        .dropna()
        .astype(int)
        .astype(str)
        .unique()
        .tolist(),
        key=_natural_sort_key,
    )
    durations = sorted(
        train["estimated_direct_drive_sec"]
        .map(
            lambda value: duration_token(
                float(value), duration_bin_minutes, max_duration_minutes
            )
        )
        .astype(str)
        .unique()
        .tolist(),
        key=_natural_sort_key,
    )
    return VocabularyBundle(
        zone=Vocabulary(zones),
        activity=Vocabulary(activities),
        departure=Vocabulary(departures),
        duration=Vocabulary(durations),
    )


class TripSequenceDataset(Dataset):
    def __init__(
        self,
        trips: pd.DataFrame,
        vocabularies: VocabularyBundle,
        profile_columns: list[str],
        max_sequence_length: int,
        duration_bin_minutes: int,
        max_duration_minutes: int,
    ) -> None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")

        self.max_sequence_length = max_sequence_length
        self.vocabularies = vocabularies
        self.duration_bin_minutes = int(duration_bin_minutes)
        self.max_duration_minutes = int(max_duration_minutes)

        required_columns = [
            "survey_day_id",
            "stop_seq",
            "origin_zone_id",
            "destination_zone_id",
            "estimated_departure_bin",
            "estimated_direct_drive_sec",
        ]
        missing = [column for column in required_columns if column not in trips.columns]
        if missing:
            raise ValueError(f"Transformer trip data missing columns: {missing}")

        frame = trips.dropna(
            subset=[
                "survey_day_id",
                "origin_zone_id",
                "destination_zone_id",
                "estimated_departure_bin",
                "estimated_direct_drive_sec",
            ]
        ).copy()
        if "origin_activity_type" not in frame.columns:
            frame["origin_activity_type"] = "unknown"
        else:
            frame["origin_activity_type"] = frame["origin_activity_type"].fillna("unknown")
        if "destination_activity_type" not in frame.columns:
            frame["destination_activity_type"] = "unknown"
        else:
            frame["destination_activity_type"] = frame["destination_activity_type"].fillna("unknown")
        frame = frame.sort_values(["survey_day_id", "stop_seq"]).reset_index(drop=True)

        profile, _ = build_profile_matrix(frame, profile_columns)
        profile = profile.reset_index(drop=True)
        self.samples: list[dict[str, Any]] = []

        for survey_day_id, group in frame.groupby("survey_day_id", sort=False):
            indices = group.index.tolist()
            for local_position, row_index in enumerate(indices):
                start_position = max(0, local_position - max_sequence_length + 1)
                prefix_indices = indices[start_position : local_position + 1]
                row = frame.loc[row_index]

                origin_zones = [
                    vocabularies.zone.encode(frame.loc[index, "origin_zone_id"])
                    for index in prefix_indices
                ]
                origin_activities = [
                    vocabularies.activity.encode(
                        frame.loc[index, "origin_activity_type"]
                    )
                    for index in prefix_indices
                ]
                departures = [
                    vocabularies.departure.encode(
                        int(frame.loc[index, "estimated_departure_bin"])
                    )
                    for index in prefix_indices
                ]

                trip_id = row.get("trip_id")
                if pd.isna(trip_id):
                    trip_id = f"{survey_day_id}_stop_{row['stop_seq']}"

                self.samples.append(
                    {
                        "sample_id": str(trip_id),
                        "survey_day_id": str(survey_day_id),
                        "person_id": str(row.get("person_id", "")),
                        "household_id": str(row.get("household_id", "")),
                        "stop_seq": int(row["stop_seq"]),
                        "origin_zones": origin_zones,
                        "origin_activities": origin_activities,
                        "departures": departures,
                        "profile": profile.loc[row_index].to_numpy(dtype=np.float32),
                        "destination_zone": vocabularies.zone.encode(
                            row["destination_zone_id"]
                        ),
                        "destination_activity": vocabularies.activity.encode(
                            row["destination_activity_type"]
                        ),
                        "duration": vocabularies.duration.encode(
                            duration_token(
                                float(row["estimated_direct_drive_sec"]),
                                duration_bin_minutes,
                                max_duration_minutes,
                            )
                        ),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def collate_trip_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(len(item["origin_zones"]) for item in batch)
    batch_size = len(batch)
    zones = torch.zeros((batch_size, max_length), dtype=torch.long)
    activities = torch.zeros_like(zones)
    departures = torch.zeros_like(zones)
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
    profiles = torch.tensor(
        np.stack([item["profile"] for item in batch]), dtype=torch.float32
    )

    for row, item in enumerate(batch):
        length = len(item["origin_zones"])
        zones[row, :length] = torch.tensor(item["origin_zones"], dtype=torch.long)
        activities[row, :length] = torch.tensor(
            item["origin_activities"], dtype=torch.long
        )
        departures[row, :length] = torch.tensor(item["departures"], dtype=torch.long)
        padding_mask[row, :length] = False

    return {
        "sample_id": [item["sample_id"] for item in batch],
        "survey_day_id": [item["survey_day_id"] for item in batch],
        "person_id": [item["person_id"] for item in batch],
        "household_id": [item["household_id"] for item in batch],
        "stop_seq": torch.tensor([item["stop_seq"] for item in batch], dtype=torch.long),
        "origin_zones": zones,
        "origin_activities": activities,
        "departures": departures,
        "last_origin_zone": torch.tensor(
            [item["origin_zones"][-1] for item in batch], dtype=torch.long
        ),
        "last_origin_activity": torch.tensor(
            [item["origin_activities"][-1] for item in batch], dtype=torch.long
        ),
        "last_departure": torch.tensor(
            [item["departures"][-1] for item in batch], dtype=torch.long
        ),
        "sequence_length": torch.tensor(
            [len(item["origin_zones"]) for item in batch], dtype=torch.long
        ),
        "padding_mask": padding_mask,
        "profiles": profiles,
        "destination_zone": torch.tensor(
            [item["destination_zone"] for item in batch], dtype=torch.long
        ),
        "destination_activity": torch.tensor(
            [item["destination_activity"] for item in batch], dtype=torch.long
        ),
        "duration": torch.tensor(
            [item["duration"] for item in batch], dtype=torch.long
        ),
    }