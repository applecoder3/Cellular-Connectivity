"""
This file will load tower data from OpencelliD

"""

from datetime import datetime, date, timezone

from pathlib import Path

import json
import uuid

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
from shapely import from_wkt
from shapely.ops import unary_union

from .clean import clean_ookla, clean_polygons, empty_layer
from .config import ROOT, METRIC_CRS, WGS84, resolve_path
from .grid import build_grid

def vector_path(path):
    path = Path(path).resolve()
    return "zip://" + path.as_posix() if path.suffix.lower() == ".zip" else str(path)

def read_vector(path, source, region=None):

    kwargs = {}

    if source.get("layer"):
        kwargs["layer"] = source["layer"] 
    url = vector_path(path)
    if region is not None:
        info = pyogrio.read_info(url, layer=source.get("layer"))
        source_crs = info["crs"] or source.get("crs")
        if source_crs is None:
            raise ValueError(f"Source {source} has no CRS defined, declare in the src config")

        kwargs["bbox"] = tuple(region.to_crs(source_crs).total_bounds)
    frame = gpd.read_file(url, engine="pyogrio", **kwargs)
    if frame.crs is None and source.get("crs"):
        frame = frame.set_crs(source["crs"])

    for field, allowed in source.get("filters", {}).items():
        if field not in frame:
            raise ValueError(f"Source {source} has no field {field} for filtering")
        allowed = allowed if isinstance(allowed, list) else [allowed]
        if all(isinstance(v, (int, float)) for v in allowed):
            keep = pd.to_numeric(frame[field], errors="coerce").isin(allowed)
        else:
            keep = frame[field].astype(str).isin([str(v) for v in allowed])
        frame = frame.loc[keep]
    return clean_polygons(frame)


def read_county(path, geoid="37069"):
    frame = gpd.read_file(vector_path(path), where=f"GEOID = '{geoid}'")
    if "GEOID" not in frame:
        raise ValueError("County source must contain GEOID.")
    frame = frame.loc[frame.GEOID.astype(str) == geoid]
    frame = clean_polygons(frame)
    if len(frame) != 1:
        raise ValueError(f"Expected one county for {geoid}, got {len(frame)}.")
    return frame[["GEOID", "NAME", "geometry"]]

def read_ookla(path, source, county):
    quarter = source.get("quarter", "")
    if source.get("service_type") != "mobile":
        raise ValueError("Ookla service_type must explicitly be 'mobile'.")
    quarter_end(quarter)
    boundary = unary_union(county.to_crs(WGS84).geometry)
    if Path(path).suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(path)
        if b"geo" in (parquet.schema_arrow.metadata or {}):
            frame = gpd.read_parquet(path)
        else:
            columns = parquet.schema_arrow.names
            geometry_field = next((c for c in ["tile", "tiles"] if c in columns), None)
            if geometry_field is None:
                raise ValueError("Ookla Parquet requires a WKT 'tile' or 'tiles' column.")
            required = ["quadkey", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices"]
            if not set(required).issubset(columns):
                raise ValueError("Ookla Parquet is missing required measurement columns.")
            parts = []
            for batch in parquet.iter_batches(batch_size=65536, columns=required + [geometry_field]):
                part = batch.to_pandas()
                geometry = from_wkt(part.pop(geometry_field).to_numpy())
                part = gpd.GeoDataFrame(part, geometry=geometry, crs=WGS84)
                part = part.loc[part.geometry.intersects(boundary)]
                if len(part):
                    parts.append(part)
            if not parts:
                return clean_ookla(gpd.GeoDataFrame({c: [] for c in required}, geometry=[], crs=WGS84))
            frame = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=WGS84)
    else:
        frame = read_vector(path, source)
    frame = clean_ookla(frame)
    # Keep native tile geometry to preserve denominator for overlap weighting.
    return frame.loc[frame.geometry.intersects(boundary)].reset_index(drop=True)

CELL_COLUMNS = ["radio", "mcc", "net", "area", "cell", "unit", "lon", "lat",
                "range", "samples", "changeable", "created", "updated", "averageSignal"]

def read_cells(path, source, county):
    area = unary_union(county.to_crs(METRIC_CRS).geometry).buffer(5000)
    bounds = gpd.GeoSeries([area], crs=METRIC_CRS).to_crs(WGS84).total_bounds
    kwargs = {"chunksize": 100000, "compression": "infer", "dtype": str}
    if source.get("headerless", False):
        kwargs.update(header=None, names=CELL_COLUMNS)
    parts = []
    for chunk in pd.read_csv(path, **kwargs):
        if not {"lon", "lat", "mcc", "net", "area", "cell", "radio"}.issubset(chunk.columns):
            raise ValueError("OpenCelliD CSV missing coordinates or network identifiers.")
        for col in ["lon", "lat"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.loc[chunk.lon.between(bounds[0], bounds[2]) & chunk.lat.between(bounds[1], bounds[3])]
        if chunk.empty:
            continue
        frame = gpd.GeoDataFrame(chunk, geometry=gpd.points_from_xy(chunk.lon, chunk.lat), crs=WGS84)
        frame = frame.loc[frame.to_crs(METRIC_CRS).geometry.intersects(area)]
        parts.append(frame)
    if not parts:
        return empty_layer()
    result = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=WGS84)
    return deduplicate_cells(result)

def deduplicate_cells(frame):
    """Keep the newest record per network cell, including across input files."""
    result = frame.copy()
    if result.empty:
        return result.reset_index(drop=True)
    if "updated" in result:
        result = result.assign(_updated_sort=pd.to_numeric(result.updated, errors="coerce"))
        result = result.sort_values("_updated_sort", kind="stable", na_position="first").drop(columns="_updated_sort")
    return result.drop_duplicates(["radio", "mcc", "net", "area", "cell"], keep="last").reset_index(drop=True)

def quarter_end(value):
    import re
    import calendar
    match = re.fullmatch(r"(\d{4})-Q([1-4])", value)
    if not match:
        raise ValueError("Ookla quarter must look like 2025-Q4.")
    year, quarter = map(int, match.groups())
    month = quarter * 3
    return date(year, month, calendar.monthrange(year, month)[1])

def fcc_completeness(source, loaded_ids):
    expected = [str(p) for p in source.get("expected_provider_ids", [])]
    reasons = []
    if source.get("inventory_verified") is not True or not source.get("inventory_source_url"):
        reasons.append("Provider inventory has not been verified against the FCC release.")
    if not expected or len(expected) != len(set(expected)):
        reasons.append("Expected provider inventory is missing or contains duplicates.")
    missing = sorted(set(expected) - set(loaded_ids))
    extra = sorted(set(loaded_ids) - set(expected))
    if missing:
        reasons.append("Missing providers: " + ", ".join(missing))
    if extra:
        reasons.append("Providers outside inventory: " + ", ".join(extra))
    if not source.get("reference_date"):
        reasons.append("FCC reference date is missing.")
    return not reasons, reasons