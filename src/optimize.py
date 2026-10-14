"""Deterministic greedy maximum-area selection using marginal spatial gain."""


import math
import geopandas as gpd
import numpy as np
from shapely import STRtree
from .config import METRIC_CRS, WGS84

OUTPUT_COLUMNS = [
        "rank", "candidate_id", "longitude", "latitude", "incremental_km2",
        "cumulative_km2", "eligible_fraction", "radius_km", "geometry"
        ]

def empty_result():
    return gpd.GeoDataFrame({c: [] for c in OUTPUT_COLUMNS}, geometry="geometry", crs=METRIC_CRS)

def select_sites(candidates, need, site_count=3, radius_km=2.0):

    if not isinstance(site_count, (int, np.integer)) or site_count < 0:
        raise ValueError("site_count must be a nonnegative integer.")
    
    if not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("radius_km must be finite and positive.")
    
    if candidates.crs is None or need.crs is None:
        raise ValueError("Candidates and need require an explicit CRS.")
    
    if not site_count or need.empty or candidates.empty:
        return empty_result()
    
    candidates = candidates.to_crs(METRIC_CRS).copy()
    need = need.to_crs(METRIC_CRS)
    candidates["candidate_id"] = candidates.candidate_id.astype(str)

    if candidates.candidate_id.duplicated().any():
        raise ValueError("Candidate IDs must be unique.")
    
    if not candidates.geometry.geom_type.eq("Point").all() or candidates.geometry.is_empty.any():
        raise ValueError("Candidates must be nonempty points.")
    
    candidates = candidates.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    remaining = list(need.geometry)
    total = sum(g.area for g in remaining)

    if total <= 1e-6:
        return empty_result()
    
    tree = STRtree(remaining)
    footprints = [p.buffer(radius_km * 1000, quad_segs=32) for p in candidates.geometry]
    neighbors = [tree.query(g, predicate="intersects") for g in footprints]
    chosen, result, cumulative = set(), [], 0.0

    for rank in range(1, min(site_count, len(candidates)) + 1):
        best, best_gain = None, 0.0
        for i, footprint in enumerate(footprints):
            if i in chosen:
                continue
            gain = sum(remaining[int(j)].intersection(footprint).area for j in neighbors[i]
                if not remaining[int(j)].is_empty)
            # IDs are sorted: retain the first candidate when gains are tied.
            if gain > best_gain + 1e-6:
                best, best_gain = i, gain
        if best is None:
            break

        chosen.add(best)

        for j in neighbors[best]:
            remaining[int(j)] = remaining[int(j)].difference(footprints[best])
        cumulative += best_gain

        point = candidates.geometry.iloc[best]
        geographic = gpd.GeoSeries([point], crs=METRIC_CRS).to_crs(WGS84).iloc[0]

        result.append({
            "rank": rank, "candidate_id": candidates.candidate_id.iloc[best],
            "longitude": geographic.x, "latitude": geographic.y,
            "incremental_km2": best_gain / 1e6, "cumulative_km2": cumulative / 1e6,
            "eligible_fraction": min(1.0, cumulative / total),
            "radius_km": radius_km, "geometry": point,
        })
        
    return gpd.GeoDataFrame(result, geometry="geometry", crs=METRIC_CRS) if result else empty_result()
