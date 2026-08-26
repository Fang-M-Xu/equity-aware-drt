from __future__ import annotations

from config import path_from_config

from .cache import SQLiteJSONCache
from .valhalla import ValhallaClient


def build_valhalla_client(config: dict) -> ValhallaClient:
    routing = config["routing"]
    return ValhallaClient(
        base_url=routing["valhalla_base_url"],
        timeout_sec=int(routing["timeout_sec"]),
        max_retries=int(routing["max_retries"]),
        retry_backoff_sec=float(routing["retry_backoff_sec"]),
        cache=SQLiteJSONCache(path_from_config(config, "cache_db")),
        allow_fallback_speed=bool(routing["allow_fallback_speed"]),
        fallback_drive_speed_kph=float(routing["fallback_drive_speed_kph"]),
        fallback_walk_speed_kph=float(routing["fallback_walk_speed_kph"]),
    )
