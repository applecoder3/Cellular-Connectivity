"""
This file will handle the scoring rules of a region 
"""

from dataclasses import dataclass

import geopandas as gpd

import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .config import METRIC_CRS, Scenario
from .clean import polygonal


@dataclass
class Analysis:

    summaries: gpd.GeoDataFrame
    need: gpd.GeoDataFrame
    tiles: gpd.GeoDataFrame
    claims_union: gpd.GeoDataFrame
    fcc_complete: bool

def classify_tiles(tiles, scenario):

    result = tiles.to_crs(METRIC_CRS).copy()
    if result.empty:
        result["sufficient"] = False
        result["low_speed"] = False
        result["evidence"] = ""
        return result
    result["sufficient"] = (
        result.tests.ge(scenario.min_tests)
        & result.devices.ge(scenario.min_devices)
        & np.isfinite(result.download_mbps)
    )

    result["low_speed"] = result.sufficient & result.download_mbps.lt(scenario.download_mbps)
    result["evidence"] = np.select(
        [result.low_speed, result.sufficient], ["Observed low speed", "Measured (not low)"], default="Insufficient measurements",
    )

    return result

def analyze(layers, fcc_complete=False, scenario=Scenario()):

    grid = layers["grid"].to_crs(METRIC_CRS)
    county = unary_union(layers["county"].to_crs(METRIC_CRS).geometry)
    tiles = classify_tiles(layers["ookla"], scenario)
    claims = layers["claims"].to_crs(METRIC_CRS)

    union_claims = polygonal(unary_union(claims.geometry).intersection(county))
    gaps = polygonal(county.difference(union_claims)) if fcc_complete else Polygon()
    slow = polygonal(unary_union(tiles.loc[tiles.low_speed].geometry).intersection(county))
    eligible = polygonal(slow.union(gaps))

    measured = unary_union(tiles.geometry)
    reliable = unary_union(tiles.loc[tiles.sufficient].geometry)
    index = tiles.sindex if len(tiles) else None
    summaries, needs = [], []


    for row in grid.itertuples():

        geometry, area = row.geometry, row.geometry.area
        slow_area = geometry.intersection(slow).area
        gap_area = geometry.intersection(gaps).area

        observed_area = geometry.intersection(measured).area
        reliable_area = geometry.intersection(reliable).area
        metrics = {"download_mbps": np.nan, "upload_mbps": np.nan, "latency_ms": np.nan}
        effective_tests = 0.0

        if index is not None:
            subset = tiles.iloc[index.query(geometry, predicate="intersects")]
            overlap = subset.geometry.intersection(geometry).area
            weights = overlap / subset.geometry.area * subset.tests.fillna(0)
            effective_tests = float(weights.sum())
            for column in metrics:
                valid = subset[column].notna() & np.isfinite(subset[column]) & weights.gt(0)
                if valid.any():
                    metrics[column] = float(np.average(subset.loc[valid, column], weights=weights.loc[valid]))

        if slow_area > 1e-6:
            evidence = "Observed low speed"
        elif reliable_area > 1e-6:
            evidence = "Measured (not low)"
        elif observed_area > 1e-6:
            evidence = "Insufficient measurements"
        else:
            evidence = "Unmeasured"

        demand = polygonal(geometry.intersection(eligible))
        common = {
            "h3_id": row.h3_id, "area_km2": area / 1e6,
            "slow_km2": slow_area / 1e6,
            "gap_km2": gap_area / 1e6 if fcc_complete else np.nan,
            "need_km2": demand.area / 1e6,
            "observed_fraction": min(1.0, observed_area / area),
            "reliable_fraction": min(1.0, reliable_area / area),
            "loaded_claim_fraction": min(1.0, geometry.intersection(union_claims).area / area),
            "effective_tests": effective_tests,
            "evidence": evidence,
        }

        summaries.append({**common, **metrics, "geometry": geometry})
        if demand.area > 1e-6:
            needs.append({**common, "geometry": demand})

    summary_frame = gpd.GeoDataFrame(summaries, geometry="geometry", crs=METRIC_CRS)
    need_frame = gpd.GeoDataFrame(needs, geometry="geometry", crs=METRIC_CRS) if needs else gpd.GeoDataFrame(
        {"h3_id": [], "need_km2": [], "geometry": []}, geometry="geometry", crs=METRIC_CRS)
    union_frame = gpd.GeoDataFrame({"geometry": [union_claims]}, geometry="geometry", crs=METRIC_CRS)
    
    return Analysis(summary_frame, need_frame, tiles, union_frame, fcc_complete)