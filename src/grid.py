"""
This fille will create the H3 hex grids and handle grid data

"""

import geopandas as gpd
import h3

from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

from .config import METRIC_CRS, WGS84
from .clean import polygonal

def build_grid(county, resolution=8):

    boundary = unary_union(county.to_crs(WGS84).geometry)

    ids = h3.geo_to_cells(mapping(boundary.envelope), resolution)
    expanded = set()

    for cell in ids:
        expanded.update(h3.grid_disk(cell, 1))
    if not expanded:
        point = boundary.representative_point()
        expanded.update(h3.grid_disk(h3.latlng_to_cell(point.y, point.x, resolution), 2))
    rows = []

    for cell in sorted(expanded):
        polygon = Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)]) 
        if polygon.intersects(boundary):
            rows.append({"h3_id": cell, "geometry": polygon})
    
    full = gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84).to_crs(METRIC_CRS)
    metric_boundary = unary_union(county.to_crs(METRIC_CRS).geometry)
    full["geometry"] = full.geometry.map(lambda g: polygonal(g.intersection(metric_boundary)))
    full = full.loc[full.geometry.area > 0].reset_index(drop=True)
    full["area_km2"] = full.geometry.area / 1e6

    residential = metric_boundary.symmetric_difference(unary_union(full.geometry)).area
    if residential > max(1.0, metric_boundary.area * 1e-6):
        raise ValueError(f"Grid does not cover the county, missing area: {residential:.2f} m^2")
    return full

def candidate_sites(grid):
    result = grid[["h3_id", "geometry"]].copy()
    result["geometry"] = result.geometry.representative_point()
    return result.rename(columns={"h3_id": "candidate_id"})