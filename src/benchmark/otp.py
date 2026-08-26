from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests


PLAN_QUERY = """
query Plan($originLat: CoordinateValue!, $originLon: CoordinateValue!,
           $destinationLat: CoordinateValue!, $destinationLon: CoordinateValue!,
           $departure: OffsetDateTime!, $first: Int!) {
  planConnection(
    origin: {location: {coordinate: {latitude: $originLat, longitude: $originLon}}}
    destination: {location: {coordinate: {latitude: $destinationLat, longitude: $destinationLon}}}
    dateTime: {earliestDeparture: $departure}
    first: $first
    modes: {direct: [WALK], transit: {transit: [{mode: BUS}, {mode: RAIL}]}}
  ) {
    edges {
      node {
        start
        end
        legs {
          mode
          from { lat lon departure { scheduledTime } }
          to { lat lon arrival { scheduledTime } }
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class TransitResult:
    available: bool
    total_time_sec: float | None = None
    transfers: int | None = None
    walk_legs: int | None = None
    transit_legs: int | None = None


class OTPClient:
    def __init__(self, endpoint: str, timeout_sec: int = 90) -> None:
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def plan(
        self,
        origin_lon: float,
        origin_lat: float,
        destination_lon: float,
        destination_lat: float,
        departure_datetime: str,
        first_itineraries: int = 5,
    ) -> TransitResult:
        variables = {
            "originLat": float(origin_lat),
            "originLon": float(origin_lon),
            "destinationLat": float(destination_lat),
            "destinationLon": float(destination_lon),
            "departure": departure_datetime,
            "first": int(first_itineraries),
        }
        response = self.session.post(
            self.endpoint,
            json={"query": PLAN_QUERY, "variables": variables},
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        if payload.get("errors"):
            raise RuntimeError(payload["errors"])
        edges = payload.get("data", {}).get("planConnection", {}).get("edges", [])
        if not edges:
            return TransitResult(False)
        itineraries = []
        for edge in edges:
            node = edge.get("node") or {}
            start = node.get("start")
            end = node.get("end")
            if not start or not end:
                continue
            start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
            legs = node.get("legs") or []
            transit_legs = sum(1 for leg in legs if str(leg.get("mode", "")).upper() not in {"WALK", "BICYCLE", "CAR"})
            walk_legs = sum(1 for leg in legs if str(leg.get("mode", "")).upper() == "WALK")
            itineraries.append(
                TransitResult(
                    True,
                    total_time_sec=(end_time - start_time).total_seconds(),
                    transfers=max(0, transit_legs - 1),
                    walk_legs=walk_legs,
                    transit_legs=transit_legs,
                )
            )
        return min(
            itineraries,
            key=lambda result: result.total_time_sec if result.total_time_sec is not None else float("inf"),
            default=TransitResult(False),
        )
