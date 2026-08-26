from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import requests

from .cache import SQLiteJSONCache

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteResult:
    time_sec: float
    distance_m: float


class ValhallaHTTPError(RuntimeError):
    def __init__(self, action: str, response: requests.Response) -> None:
        self.action = action
        self.status_code = response.status_code
        self.response_body = response.text
        super().__init__(
            f"Valhalla {action} returned HTTP {response.status_code}: {response.text}"
        )


def _stable_key(action: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{action}:{canonical}".encode("utf-8")).hexdigest()


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


class ValhallaClient:
    def __init__(
        self,
        base_url: str,
        timeout_sec: int = 90,
        max_retries: int = 4,
        retry_backoff_sec: float = 2.0,
        cache: SQLiteJSONCache | None = None,
        allow_fallback_speed: bool = False,
        fallback_drive_speed_kph: float = 28.0,
        fallback_walk_speed_kph: float = 4.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.cache = cache
        self.allow_fallback_speed = allow_fallback_speed
        self.fallback_drive_speed_kph = fallback_drive_speed_kph
        self.fallback_walk_speed_kph = fallback_walk_speed_kph
        self.session = requests.Session()

    def _post(self, action: str, payload: dict[str, Any]) -> Any:
        key = _stable_key(action, payload)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/{action}", json=payload, timeout=self.timeout_sec
                )
                if 400 <= response.status_code < 500:
                    raise ValhallaHTTPError(action, response)
                response.raise_for_status()
                data = response.json()
                if self.cache is not None:
                    self.cache.set(key, data)
                return data
            except ValhallaHTTPError:
                # Retrying a request rejected by Valhalla cannot change the result.
                raise
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_sec * (2**attempt))
        raise RuntimeError(f"Valhalla {action} failed after {self.max_retries} attempts") from last_error

    def status(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/status", timeout=self.timeout_sec)
        response.raise_for_status()
        return response.json()

    def route(
        self,
        origin_lon: float,
        origin_lat: float,
        destination_lon: float,
        destination_lat: float,
        costing: str,
    ) -> RouteResult:
        payload = {
            "locations": [
                {"lon": round(float(origin_lon), 6), "lat": round(float(origin_lat), 6)},
                {"lon": round(float(destination_lon), 6), "lat": round(float(destination_lat), 6)},
            ],
            "costing": costing,
            "directions_options": {"units": "kilometers"},
        }
        try:
            data = self._post("route", payload)
            summary = data["trip"]["summary"]
            return RouteResult(time_sec=float(summary["time"]), distance_m=float(summary["length"]) * 1000.0)
        except Exception:
            if not self.allow_fallback_speed:
                raise
            distance = haversine_m(origin_lon, origin_lat, destination_lon, destination_lat)
            speed_kph = self.fallback_walk_speed_kph if costing == "pedestrian" else self.fallback_drive_speed_kph
            return RouteResult(time_sec=distance / (speed_kph * 1000 / 3600), distance_m=distance)

    def locate(self, lon: float, lat: float, costing: str = "auto") -> Any:
        payload = {
            "verbose": True,
            "locations": [{"lon": round(float(lon), 6), "lat": round(float(lat), 6)}],
            "costing": costing,
        }
        return self._post("locate", payload)

    def matrix(
        self,
        sources: Iterable[tuple[float, float]],
        targets: Iterable[tuple[float, float]],
        costing: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_list = [
            {"lon": round(float(lon), 6), "lat": round(float(lat), 6)} for lon, lat in sources
        ]
        target_list = [
            {"lon": round(float(lon), 6), "lat": round(float(lat), 6)} for lon, lat in targets
        ]
        payload = {
            "sources": source_list,
            "targets": target_list,
            "costing": costing,
            "units": "kilometers",
        }
        try:
            data = self._post("sources_to_targets", payload)
            rows = data.get("sources_to_targets") or data.get("matrix")
            if rows is None:
                raise KeyError("Valhalla matrix response has no sources_to_targets field")
            times = np.full((len(source_list), len(target_list)), np.inf, dtype=float)
            distances = np.full_like(times, np.inf)
            for i, row in enumerate(rows):
                for j, item in enumerate(row):
                    if item is None:
                        continue
                    if item.get("time") is not None:
                        times[i, j] = float(item["time"])
                    if item.get("distance") is not None:
                        distances[i, j] = float(item["distance"]) * 1000.0
            return times, distances
        except ValhallaHTTPError as error:
            no_path = error.status_code == 400 and "No path could be found" in error.response_body
            matrix_too_large = (
                error.status_code == 400
                and "Exceeded max locations" in error.response_body
            )
            if (no_path or matrix_too_large) and (
                len(source_list) > 1 or len(target_list) > 1
            ):
                # Valhalla rejects the whole matrix if one location cannot be
                # connected or if sources * targets exceeds the server limit.
                # Split along the larger dimension until the request is valid
                # or only the unreachable OD pair remains.
                if len(target_list) >= len(source_list) and len(target_list) > 1:
                    midpoint = len(target_list) // 2
                    left_time, left_distance = self.matrix(
                        [(item["lon"], item["lat"]) for item in source_list],
                        [(item["lon"], item["lat"]) for item in target_list[:midpoint]],
                        costing,
                    )
                    right_time, right_distance = self.matrix(
                        [(item["lon"], item["lat"]) for item in source_list],
                        [(item["lon"], item["lat"]) for item in target_list[midpoint:]],
                        costing,
                    )
                    return (
                        np.concatenate([left_time, right_time], axis=1),
                        np.concatenate([left_distance, right_distance], axis=1),
                    )
                midpoint = len(source_list) // 2
                top_time, top_distance = self.matrix(
                    [(item["lon"], item["lat"]) for item in source_list[:midpoint]],
                    [(item["lon"], item["lat"]) for item in target_list],
                    costing,
                )
                bottom_time, bottom_distance = self.matrix(
                    [(item["lon"], item["lat"]) for item in source_list[midpoint:]],
                    [(item["lon"], item["lat"]) for item in target_list],
                    costing,
                )
                return (
                    np.concatenate([top_time, bottom_time], axis=0),
                    np.concatenate([top_distance, bottom_distance], axis=0),
                )
            if no_path and not self.allow_fallback_speed:
                LOGGER.warning(
                    "No Valhalla path for source %s and target %s; marking OD pair unavailable",
                    source_list[0],
                    target_list[0],
                )
                unavailable = np.full((1, 1), np.inf, dtype=float)
                return unavailable, unavailable.copy()
            if not self.allow_fallback_speed:
                raise
            return self._fallback_matrix(source_list, target_list, costing)
        except Exception:
            if not self.allow_fallback_speed:
                raise
            return self._fallback_matrix(source_list, target_list, costing)

    def _fallback_matrix(
        self,
        source_list: list[dict[str, float]],
        target_list: list[dict[str, float]],
        costing: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        times = np.zeros((len(source_list), len(target_list)), dtype=float)
        distances = np.zeros_like(times)
        speed_kph = self.fallback_walk_speed_kph if costing == "pedestrian" else self.fallback_drive_speed_kph
        for i, source in enumerate(source_list):
            for j, target in enumerate(target_list):
                distance = haversine_m(
                    source["lon"], source["lat"], target["lon"], target["lat"]
                )
                distances[i, j] = distance
                times[i, j] = distance / (speed_kph * 1000 / 3600)
        return times, distances

    def matrix_chunked(
        self,
        sources: list[tuple[float, float]],
        targets: list[tuple[float, float]],
        costing: str,
        max_sources: int,
        max_targets: int,
        max_pairs: int = 2500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a Valhalla source-to-target matrix in safe blocks.

        Valhalla limits a matrix request by the total number of
        source-target pairs:

            number_of_sources * number_of_targets <= max_pairs

        For example, when max_pairs=2500:
            50 sources × 50 targets = 2500 pairs
            25 sources × 100 targets = 2500 pairs

        The function dynamically reduces the target chunk size so that
        every request remains within the Valhalla server limit.
        """

        if max_sources <= 0:
            raise ValueError(
                "max_sources must be greater than zero."
            )

        if max_targets <= 0:
            raise ValueError(
                "max_targets must be greater than zero."
            )

        if max_pairs <= 0:
            raise ValueError(
                "max_pairs must be greater than zero."
            )

        source_count = len(sources)
        target_count = len(targets)

        times = np.full(
            (source_count, target_count),
            np.inf,
            dtype=float,
        )

        distances = np.full_like(
            times,
            np.inf,
        )

        if source_count == 0 or target_count == 0:
            return times, distances

        # One source chunk cannot contain more locations than the
        # complete pair limit, because at least one target is required.
        source_chunk_size = min(
            int(max_sources),
            int(max_pairs),
        )

        configured_pairs = (
            int(max_sources) * int(max_targets)
        )

        if configured_pairs > max_pairs:
            LOGGER.warning(
                "Configured Valhalla matrix block %d x %d = %d pairs "
                "exceeds max_pairs=%d. Target chunks will be reduced "
                "automatically.",
                max_sources,
                max_targets,
                configured_pairs,
                max_pairs,
            )

        completed_blocks = 0

        for source_start in range(
            0,
            source_count,
            source_chunk_size,
        ):
            source_end = min(
                source_start + source_chunk_size,
                source_count,
            )

            source_chunk = sources[
                source_start:source_end
            ]

            current_source_count = len(source_chunk)

            # Dynamically calculate the largest legal target chunk.
            legal_target_count = max(
                1,
                max_pairs // current_source_count,
            )

            target_chunk_size = min(
                int(max_targets),
                int(legal_target_count),
            )

            for target_start in range(
                0,
                target_count,
                target_chunk_size,
            ):
                target_end = min(
                    target_start + target_chunk_size,
                    target_count,
                )

                target_chunk = targets[
                    target_start:target_end
                ]

                pair_count = (
                    len(source_chunk)
                    * len(target_chunk)
                )

                if pair_count > max_pairs:
                    raise RuntimeError(
                        "Internal matrix chunking error: "
                        f"{len(source_chunk)} sources × "
                        f"{len(target_chunk)} targets = "
                        f"{pair_count} pairs, which exceeds "
                        f"max_pairs={max_pairs}."
                    )

                LOGGER.info(
                    "Valhalla matrix block: "
                    "sources [%d:%d], targets [%d:%d], "
                    "pairs=%d",
                    source_start,
                    source_end,
                    target_start,
                    target_end,
                    pair_count,
                )

                block_time, block_distance = self.matrix(
                    source_chunk,
                    target_chunk,
                    costing,
                )

                expected_shape = (
                    len(source_chunk),
                    len(target_chunk),
                )

                if block_time.shape != expected_shape:
                    raise RuntimeError(
                        "Unexpected Valhalla time-matrix shape: "
                        f"expected {expected_shape}, "
                        f"received {block_time.shape}."
                    )

                if block_distance.shape != expected_shape:
                    raise RuntimeError(
                        "Unexpected Valhalla distance-matrix shape: "
                        f"expected {expected_shape}, "
                        f"received {block_distance.shape}."
                    )

                times[
                    source_start:source_end,
                    target_start:target_end,
                ] = block_time

                distances[
                    source_start:source_end,
                    target_start:target_end,
                ] = block_distance

                completed_blocks += 1

        LOGGER.info(
            "Completed Valhalla chunked matrix: "
            "%d sources × %d targets in %d blocks.",
            source_count,
            target_count,
            completed_blocks,
        )

        return times, distances
