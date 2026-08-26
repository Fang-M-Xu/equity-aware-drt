from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points


def load_roads_metric(roads_path: Path, metric_crs: str) -> gpd.GeoDataFrame:
    roads = gpd.read_file(roads_path)
    if roads.crs is None:
        raise ValueError("Road layer has no CRS")
    roads = roads.to_crs(metric_crs)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    return roads


def snap_xy_to_roads(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    id_column: str,
    roads: gpd.GeoDataFrame,
    metric_crs: str,
    target_crs: str,
    max_distance_m: float,
) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        frame[[id_column, x_column, y_column]].copy(),
        geometry=gpd.points_from_xy(frame[x_column], frame[y_column]),
        crs=metric_crs,
    )
    nearest = gpd.sjoin_nearest(
        points,
        roads[["geometry"]],
        how="left",
        distance_col="snap_distance_m",
    )
    nearest = nearest.sort_values([id_column, "snap_distance_m"]).drop_duplicates(id_column)
    snapped_geometries = []
    for row in nearest.itertuples():
        if pd.isna(row.index_right):
            snapped_geometries.append(None)
            continue
        road_geometry = roads.loc[row.index_right, "geometry"]
        snapped_geometries.append(nearest_points(row.geometry, road_geometry)[1])
    nearest["snapped_geometry"] = snapped_geometries
    nearest["snap_valid"] = (
        nearest["snapped_geometry"].notna()
        & nearest["snap_distance_m"].le(max_distance_m)
    )
    snapped = gpd.GeoDataFrame(
        nearest[[id_column, "snap_distance_m", "snap_valid"]].copy(),
        geometry=nearest["snapped_geometry"],
        crs=metric_crs,
    )
    snapped["snapped_x"] = snapped.geometry.x
    snapped["snapped_y"] = snapped.geometry.y
    wgs84 = snapped.to_crs(target_crs)
    snapped["snapped_lon"] = wgs84.geometry.x
    snapped["snapped_lat"] = wgs84.geometry.y
    return pd.DataFrame(snapped.drop(columns="geometry"))
