"""
This file will clean the data 

"""

import geopandas as gpd
import pandas as pd
import numpy as np

from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
from shapely.ops import unary_union
from .config import WGS84 



def polygonal(geometry):
    """Convert a geometry to a polygonal geometry"""

    if geometry is None or geometry.is_empty:
        return Polygon()

    geometry = make_valid(geometry)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry

    if isinstance(geometry, GeometryCollection):
        return unary_union([polygonal(g) for g in geometry.geoms])
    return Polygon()

def empty_layer():
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=WGS84)


def clean_polygons(frame):
    if frame.crs is None:
        raise ValueError("Data has no CRS defined, declare in the src config")

    result = frame.to_crs(WGS84).copy()
    result["geometry"] = result["geometry"].map(polygonal)
    result = result.loc[~result["geometry"].is_empty].reset_index(drop=True)
    if len(result) == 0:
        xmin, ymin, xmax, ymax = result.total_bounds
        if not np.isfinite([xmin, ymin, xmax, ymax]).all() or xmin < -100 or ymin < -90 or xmax > 180 or ymax > 90:
            raise ValueError("Data has no valid geometry")
    return result


def clean_ookla(frame):
    required = ["quadkey", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices"]
    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = clean_polygons(frame)

    if result.quadkey.isna().any():
        raise ValueError("Missing quadkey values")
    result["quadkey"] = result["quadkey"].astype(str)

    if result.quadkey.duplicated().any():
        raise ValueError("Duplicate quadkey values")

    for column in ["tests", "devices"]:
        result.loc[result[column] % 1 != 0, column] = np.nan

    result.loc[result.devices > result.tests, ["tests", "devices"]] = np.nan
    result["download_mbps"] = result["avg_d_kbps"] / 1000
    result["upload_mbps"] = result["avg_u_kbps"] / 1000
    result["latency_ms"] = result["avg_lat_ms"]

    return result