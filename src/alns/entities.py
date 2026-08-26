from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: int
    pickup_node: int
    dropoff_node: int
    access_walk_sec: float
    egress_walk_sec: float
    direct_drive_sec: float
    service_gap: int
    social_welfare_score: float
    reveal_time_sec: float = 0.0


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    start_node: int
    capacity: int


@dataclass(frozen=True)
class Stop:
    request_id: int
    node_id: int
    kind: str

    @staticmethod
    def pickup(request: Request) -> "Stop":
        return Stop(request.request_id, request.pickup_node, "pickup")

    @staticmethod
    def dropoff(request: Request) -> "Stop":
        return Stop(request.request_id, request.dropoff_node, "dropoff")
