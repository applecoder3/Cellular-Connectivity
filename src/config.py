"""This file will control the region tested, hex size, and thresholds"""

from dataclasses import dataclass
from pathlib import Path

import json
import math


ROOT = Path(__file__).resolve().parent[1]
WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:32617"

 


@dataclass(frozen=True)
class Scenario:
    download_mbps: float = 5.0
    min_tests: int = 10
    min_devices: int = 3
    site_count: int = 3
    radius_km: float = 2.0


    def __post_init__(self):
        pass

def load_config(config_path: Path):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_base"] = str(path.parent)
    
    return config

def resolve_config(config, path):
    result = Path(path)
    return result if result.is_absolute() else Path(config["_base"]) / result