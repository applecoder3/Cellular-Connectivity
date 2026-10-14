from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import json
import uuid

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from .clean import clean_polygons, empty_layer
from .config import ATTRIBUTIONS, ROOT, METRIC_CRS, WGS84, resolve_path
from .grid import build_grid
from .ingest import (
    read_county, read_ookla, read_cells, read_vector, deduplicate_cells,
    quarter_end, fcc_completeness,
)

def file_record(config, source, key):
    path = resolve_path(config, source["path"]).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
        "source_url": source.get("source_url", ""),
        "acquired_date": source.get("acquired_date", ""),
        "filters": source.get("filters", {}),
        "attribution": ATTRIBUTIONS[key],
    }

def prepare(config, output=ROOT / "data" / "processed"):
    census = config["census"]
    county = read_county(resolve_path(config, census["path"]))
    grid = build_grid(county)
    boundary = unary_union(county.geometry)
    provenance = {"census": file_record(config, census, "census")}
    ookla_source = config.get("ookla") or {}
    ookla = empty_layer()
    if ookla_source.get("path"):
        ookla = read_ookla(resolve_path(config, ookla_source["path"]), ookla_source, county)
        provenance["ookla"] = file_record(config, ookla_source, "ookla")
        provenance["ookla"].update(quarter=ookla_source["quarter"], service_type="mobile")
    cell_source = config.get("opencellid") or {}
    cells = empty_layer()
    if cell_source.get("path") and "paths" in cell_source:
        raise ValueError("OpenCelliD config must use either 'path' or 'paths', not both.")
    cell_paths = cell_source.get("paths", [cell_source["path"]] if cell_source.get("path") else [])
    if not isinstance(cell_paths, list) or any(not isinstance(p, str) or not p for p in cell_paths):
        raise ValueError("OpenCelliD 'paths' must be a list of nonempty file paths.")
    cell_parts, cell_records = [], []
    for cell_path in cell_paths:
        source = {**cell_source, "path": cell_path}
        part = read_cells(resolve_path(config, cell_path), source, county)
        if not part.empty:
            cell_parts.append(part)
        record = file_record(config, source, "opencellid")
        record["filtered_cells"] = len(part)
        cell_records.append(record)
    if cell_parts:
        cells = deduplicate_cells(gpd.GeoDataFrame(pd.concat(cell_parts, ignore_index=True), crs=WGS84))
    if cell_records:
        # Preserve the original single-file manifest format for existing configs.
        provenance["opencellid"] = cell_records[0] if cell_source.get("path") else {
            "files": cell_records, "attribution": ATTRIBUTIONS["opencellid"],
        }
    fcc_source = config.get("fcc") or {}
    claims_parts, loaded_ids, fcc_records = [], set(), []
    for source in fcc_source.get("layers", []):
        profile = (source.get("technology"), source.get("minimum_download_mbps"),
                   source.get("minimum_upload_mbps"), source.get("environment"))
        if profile != ("LTE", 5, 1, "outdoor_stationary"):
            raise ValueError("Each FCC layer must declare LTE, 5/1 Mbps, outdoor_stationary. Filter mixed files before use.")
        provider_id = str(source.get("provider_id", "")).strip()
        if not provider_id:
            raise ValueError("Each FCC layer requires provider_id.")
        frame = read_vector(resolve_path(config, source["path"]), source, region=county)
        frame["geometry"] = frame.geometry.intersection(boundary)
        frame = clean_polygons(frame)
        frame["provider_id"] = provider_id
        frame["provider_name"] = str(source.get("provider_name", provider_id))
        claims_parts.append(frame[["provider_id", "provider_name", "geometry"]])
        loaded_ids.add(provider_id)  # A verified source with zero county coverage is valid.
        record = file_record(config, source, "fcc")
        record.update(provider_id=provider_id, provider_name=source.get("provider_name", provider_id),
                      technology="LTE", minimum_download_mbps=5, minimum_upload_mbps=1,
                      environment="outdoor_stationary")
        fcc_records.append(record)
    claims = gpd.GeoDataFrame(pd.concat(claims_parts, ignore_index=True), crs=WGS84) if claims_parts else empty_layer()
    complete, reasons = fcc_completeness(fcc_source, loaded_ids)
    fcc_date = fcc_source.get("reference_date")
    if fcc_date:
        date.fromisoformat(fcc_date)
    mismatch = None
    if ookla_source.get("path") and fcc_date:
        end = quarter_end(ookla_source["quarter"])
        reference = date.fromisoformat(fcc_date)
        if end > reference:
            raise ValueError("Ookla quarter must end on or before the FCC reference date.")
        if end != reference:
            mismatch = f"Ookla quarter ends {end}; FCC claims are dated {reference}."
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    snapshot = output / version
    snapshot.mkdir()
    layers = {"county": county, "grid": grid, "ookla": ookla, "cells": cells, "claims": claims}
    for name, layer in layers.items():
        layer.to_parquet(snapshot / f"{name}.parquet", index=False)
    manifest = {
        "version": version, "prepared_at": datetime.now(timezone.utc).isoformat(),
        "county_geoid": "37069", "county_name": "Franklin County, NC",
        "h3_resolution": 8, "metric_crs": METRIC_CRS,
        "sources": provenance, "fcc_layers": fcc_records,
        "fcc": {"reference_date": fcc_date, "complete": complete, "reasons": reasons,
                "expected_provider_ids": fcc_source.get("expected_provider_ids", []),
                "loaded_provider_ids": sorted(loaded_ids),
                "inventory_source_url": fcc_source.get("inventory_source_url", "")},
        "temporal_mismatch": mismatch, "attributions": ATTRIBUTIONS,
        "counts": {name: len(layer) for name, layer in layers.items()},
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pointer = output / ("current-" + uuid.uuid4().hex + ".tmp")
    pointer.write_text(json.dumps({"version": version}), encoding="utf-8")
    pointer.replace(output / "current.json")  # Readers never see a partially prepared dataset.
    return manifest

def load_prepared(output=ROOT / "data" / "processed", version=None):
    output = Path(output)
    if version is None:
        version = json.loads((output / "current.json").read_text(encoding="utf-8"))["version"]
    if Path(version).name != version or ".." in version:
        raise ValueError("Invalid dataset version.")
    snapshot = output / version
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    layers = {name: gpd.read_parquet(snapshot / f"{name}.parquet")
              for name in ["county", "grid", "ookla", "cells", "claims"]}
    return layers, manifest
