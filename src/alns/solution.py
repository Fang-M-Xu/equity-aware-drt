from __future__ import annotations

from dataclasses import dataclass, field

from .entities import Stop


@dataclass
class Route:
    vehicle_id: int
    stops: list[Stop] = field(default_factory=list)

    def clone(self) -> "Route":
        return Route(self.vehicle_id, list(self.stops))

    def request_ids(self) -> set[int]:
        return {stop.request_id for stop in self.stops}

    def remove_request(self, request_id: int) -> None:
        self.stops = [stop for stop in self.stops if stop.request_id != request_id]


@dataclass
class Solution:
    routes: dict[int, Route]
    unserved: set[int]

    def clone(self) -> "Solution":
        return Solution(
            routes={vehicle_id: route.clone() for vehicle_id, route in self.routes.items()},
            unserved=set(self.unserved),
        )

    def served_request_ids(self) -> set[int]:
        served: set[int] = set()
        for route in self.routes.values():
            served |= route.request_ids()
        return served

    def remove_requests(self, request_ids: set[int]) -> None:
        for route in self.routes.values():
            for request_id in request_ids:
                route.remove_request(request_id)
        self.unserved |= request_ids
