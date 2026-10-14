"""Self-contained site exports with evidence, scenario and provenance."""
import json
from .config import LIMITATION, WGS84

def export_sites(sites, manifest, scenario):
    attribution = " | ".join(manifest["attributions"].values())
    frame = sites.to_crs(WGS84).copy()
    frame["dataset_version"] = manifest["version"]
    frame["fcc_gap_enabled"] = manifest["fcc"]["complete"]
    frame["fcc_reference_date"] = manifest["fcc"]["reference_date"] or ""
    frame["ookla_quarter"] = manifest["sources"].get("ookla", {}).get("quarter", "")
    frame["download_threshold_mbps"] = scenario.download_mbps
    frame["min_tests"] = scenario.min_tests
    frame["min_devices"] = scenario.min_devices
    frame["limitation"] = LIMITATION
    frame["attribution"] = attribution
    csv = frame.drop(columns="geometry").to_csv(index=False).encode("utf-8-sig")
    collection = json.loads(frame.to_json(drop_id=True, na="null"))
    collection["metadata"] = {
        "manifest": manifest, "scenario": scenario.__dict__, "limitation": LIMITATION,
    }
    return csv, json.dumps(collection, ensure_ascii=False, allow_nan=False).encode("utf-8")

def export_evidence(summaries, manifest, scenario):
    frame = summaries.drop(columns="geometry").copy()
    frame["dataset_version"] = manifest["version"]
    frame["fcc_gap_enabled"] = manifest["fcc"]["complete"]
    frame["fcc_reference_date"] = manifest["fcc"]["reference_date"] or ""
    frame["ookla_quarter"] = manifest["sources"].get("ookla", {}).get("quarter", "")
    frame["download_threshold_mbps"] = scenario.download_mbps
    frame["min_tests"] = scenario.min_tests
    frame["min_devices"] = scenario.min_devices
    frame["aggregation"] = "Approximate native tile mean weighted by tests times overlap fraction"
    frame["attribution"] = " | ".join(manifest["attributions"].values())
    return frame.to_csv(index=False).encode("utf-8-sig")
