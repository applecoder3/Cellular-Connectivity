"""Independent fallback UI: python -m streamlit run proto.py.

Only prepared data is required for the base map. Optional backend contracts:
score.analyze(layers, fcc_complete, scenario) -> object with summaries, need, tiles
optimize.select_sites(candidates, need, site_count, radius_km) -> GeoDataFrame
No imports from app.py or src/app.py: either frontend can run independently.
"""
import html
import json
import os
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from src.pipeline import load_prepared

ROOT = Path(__file__).resolve().parent
WGS84 = "EPSG:4326"
EVIDENCE_COLORS = {
    "Observed low speed": "#d94832",
    "Measured (not low)": "#238b71",
    "Insufficient measurements": "#d5a037",
    "Unmeasured": "#a5afb9",
}


@st.cache_data(show_spinner="Loading prepared data…")
def load_data(directory, version):
    return load_prepared(Path(directory), version)


@st.cache_data(show_spinner="Scoring evidence…")
def score_data(directory, version, threshold, tests, devices):
    # Lazy imports allow the base map to run while these modules are unfinished.
    from src.config import Scenario
    from src.score import analyze

    layers, manifest = load_data(directory, version)
    scenario = Scenario(download_mbps=threshold, min_tests=tests, min_devices=devices)
    analysis = analyze(layers, manifest.get("fcc", {}).get("complete", False), scenario)
    if not all(hasattr(analysis, field) for field in ["summaries", "need", "tiles"]):
        raise ValueError("analyze() must return summaries, need and tiles; see proto.py's module docstring.")
    for field in ["summaries", "need", "tiles"]:
        frame = getattr(analysis, field)
        if not hasattr(frame, "to_crs") or frame.crs is None:
            raise ValueError(f"Analysis.{field} must be a GeoDataFrame with a coordinate system.")
    if not analysis.tiles.empty and "sufficient" not in analysis.tiles.columns:
        raise ValueError("Analysis.tiles must include the sample-sufficiency flag.")
    return analysis


@st.cache_data(show_spinner="Selecting hypothetical sites…")
def optimize_data(directory, version, threshold, tests, devices, count, radius):
    from src.grid import candidate_sites
    from src.optimize import select_sites

    layers, _ = load_data(directory, version)
    analysis = score_data(directory, version, threshold, tests, devices)
    sites = select_sites(candidate_sites(layers["grid"]), analysis.need, count, radius)
    required = {"geometry", "rank", "radius_km", "incremental_km2", "cumulative_km2"}
    if not hasattr(sites, "to_crs") or not required.issubset(sites.columns):
        raise ValueError("select_sites() must return the site GeoDataFrame described by proto.py.")
    if sites.crs is None or not sites.geometry.geom_type.eq("Point").all():
        raise ValueError("Selected sites must be points with a coordinate system.")
    return sites


def add_polygons(view, frame, name, color, fields=(), show=True, style=None):
    """Render only relevant properties; do not modify analysis geometry."""
    if frame.empty:
        return
    fields = [field for field in fields if field in frame.columns]
    display = frame[fields + ["geometry"]].to_crs(WGS84).copy()
    for field in fields:
        if pd.api.types.is_object_dtype(display[field]):
            display[field] = display[field].map(lambda value: html.escape(str(value)) if value is not None else None)
    folium.GeoJson(
        json.loads(display.to_json(drop_id=True, na="null")), name=name, show=show,
        style_function=style or (lambda _: {"color": color, "weight": 0.8, "fillOpacity": 0.15}),
        tooltip=folium.GeoJsonTooltip(fields=fields, localize=True) if fields else None,
    ).add_to(view)


def build_map(layers, provider_ids, analysis=None, sites=None):
    county = layers["county"].to_crs(WGS84)
    center = county.geometry.iloc[0].representative_point()
    view = folium.Map([center.y, center.x], zoom_start=10, tiles="OpenStreetMap", control_scale=True)
    add_polygons(view, county, "County boundary", "#182c3a", ["NAME", "GEOID"],
                 style=lambda _: {"color": "#182c3a", "weight": 2, "fillOpacity": 0})
    add_polygons(view, layers["grid"], "H3 grid", "#7a8b9d", ["h3_id", "area_km2"], show=False)
    claims = layers["claims"]
    for provider_id in provider_ids:
        subset = claims.loc[claims.provider_id == provider_id]
        label = html.escape(str(subset.provider_name.iloc[0]))
        add_polygons(view, subset, f"FCC LTE claims: {label}", "#446aba", ["provider_name", "provider_id"])
    scale = LinearColormap(["#d94832", "#eac35e", "#238b71"], vmin=0, vmax=100)
    scale.caption = "Observed download Mbps; color capped at 100 (gray = missing value)"
    def speed_style(feature):
        speed = feature["properties"].get("download_mbps")
        return {"color": "#ffffff", "weight": 0.5, "fillOpacity": 0.75,
                "fillColor": "#a5afb9" if speed is None else scale(min(speed, 100))}
    add_polygons(view, layers["ookla"], "Native Ookla speed tiles", "#238b71",
                 ["quadkey", "download_mbps", "upload_mbps", "latency_ms", "tests", "devices"], style=speed_style)
    if not layers["ookla"].empty:
        scale.add_to(view)
    cells = folium.FeatureGroup(name="Estimated cell positions", show=True)
    for row in layers["cells"].to_crs(WGS84).itertuples():
        label = " · ".join(f"{field}: {html.escape(str(getattr(row, field, 'unknown')))}"
                           for field in ["radio", "mcc", "net", "cell"])
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=4, color="#23384a",
                            fill=True, fill_opacity=0.8, tooltip=label).add_to(cells)
    cells.add_to(view)
    if analysis is not None:
        add_polygons(view, analysis.summaries, "Scored H3 evidence", "#a5afb9",
                     ["h3_id", "evidence", "download_mbps", "need_km2"],
                     style=lambda feature: {"color": "#ffffff", "weight": 0.3, "fillOpacity": 0.45,
                         "fillColor": EVIDENCE_COLORS.get(feature["properties"].get("evidence"), "#a5afb9")})
        add_polygons(view, analysis.need, "Eligible need", "#d94832", ["h3_id", "need_km2"])
    if sites is not None and not sites.empty:
        footprints = sites[["geometry", "rank", "radius_km"]].to_crs("EPSG:32617").copy()
        footprints["geometry"] = [row.geometry.buffer(row.radius_km * 1000, quad_segs=32)
                                  for row in footprints.itertuples()]
        add_polygons(view, footprints, "Hypothetical footprints", "#842bb6", ["rank", "radius_km"])
        for row in sites.to_crs(WGS84).itertuples():
            folium.Marker([row.geometry.y, row.geometry.x], tooltip=f"Site {row.rank}: +{row.incremental_km2:.2f} km²",
                          icon=folium.Icon(color="purple", icon="signal")).add_to(view)
    xmin, ymin, xmax, ymax = county.total_bounds
    view.fit_bounds([[ymin, xmin], [ymax, xmax]])
    folium.LayerControl().add_to(view)
    return view


def run():
    st.set_page_config(page_title="Cellular coverage prototype", page_icon="📡", layout="wide")
    st.title("Franklin County — prototype")
    st.caption("Prepared source layers with optional evidence scoring and hypothetical site selection.")
    directory = str(Path(os.environ.get("CELLULAR_DATA_DIR", str(ROOT / "data" / "processed"))).resolve())
    try:
        version = json.loads((Path(directory) / "current.json").read_text(encoding="utf-8"))["version"]
        layers, manifest = load_data(directory, version)
    except (OSError, ValueError, KeyError) as error:
        st.info("Prepare your local dataset before opening the prototype.")
        st.code("python main.py prepare --config config.local.json")
        st.caption(str(error))
        return

    with st.sidebar:
        st.header("Map layers")
        claims = layers["claims"]
        names = dict(zip(claims.provider_id, claims.provider_name)) if not claims.empty else {}
        provider_ids = st.multiselect("Display FCC providers", sorted(names), format_func=lambda key: names[key])
        st.caption("Provider display does not filter the combined-network Ookla measurements.")
        scoring_enabled = st.checkbox("Enable scoring", value=False)
        optimization_enabled = st.checkbox("Enable site selection", value=False, disabled=not scoring_enabled)
        threshold = st.number_input("Low download threshold (Mbps)", min_value=0.1, value=5.0, disabled=not scoring_enabled)
        tests = st.number_input("Minimum tests per tile", min_value=1, value=10, disabled=not scoring_enabled)
        devices = st.number_input("Minimum devices per tile", min_value=1, value=3, disabled=not scoring_enabled)
        count = st.slider("Maximum sites", 0, 10, 3, disabled=not optimization_enabled or not scoring_enabled)
        radius = st.select_slider("Radius (km)", [1.0, 2.0, 5.0], value=2.0, disabled=not optimization_enabled or not scoring_enabled)

    columns = st.columns(4)
    for column, key, label in zip(columns, ["grid", "cells", "claims", "ookla"],
                                  ["Grid cells", "Nearby cell records", "FCC polygons", "Ookla tiles"]):
        column.metric(label, len(layers[key]))
    fcc_complete = manifest.get("fcc", {}).get("complete", False)
    if not fcc_complete:
        st.warning("Combined FCC coverage-gap analysis is disabled: release date or complete provider inventory is unverified.")
    st.caption("FCC polygons are coverage claims; cell positions are estimates. No speed tile means unmeasured, not a confirmed dead zone.")
    st.caption("Hypothetical circles do not predict signal strength or speed gains. Terrain, congestion and construction feasibility are not modeled.")

    analysis, sites = None, None
    if scoring_enabled:
        try:
            analysis = score_data(directory, version, float(threshold), int(tests), int(devices))
            if not analysis.tiles.empty and not analysis.tiles.sufficient.any():
                st.warning("No tiles meet the current sample thresholds; no low-speed need qualifies from these observations.")
            if tests < 10 or devices < 3:
                st.warning("Exploratory thresholds use weaker evidence than the defaults of 10 tests and 3 devices.")
        except Exception as error:
            # This boundary intentionally protects the fallback map from work-in-progress backends.
            analysis = None
            st.warning(f"Scoring unavailable; source map remains usable. {type(error).__name__}: {error}")
    if analysis is not None and optimization_enabled:
        try:
            sites = optimize_data(directory, version, float(threshold), int(tests), int(devices), int(count), float(radius))
        except Exception as error:
            sites = None
            st.warning(f"Site selection unavailable; source map and scoring remain usable. {type(error).__name__}: {error}")

    map_tab, data_tab = st.tabs(["Map", "Tables and results"])
    with map_tab:
        st_folium(build_map(layers, provider_ids, analysis, sites), height=600,
                  use_container_width=True, returned_objects=[], key="prototype-map")
        st.caption("Use the sidebar to choose FCC providers and the map layer control to toggle overlays. Map background tiles require internet access.")
    with data_tab:
        if sites is not None:
            if sites.empty:
                st.info("No candidate adds eligible area under this scenario.")
            else:
                st.write("Selected hypothetical sites")
                st.dataframe(sites.drop(columns="geometry"), hide_index=True, width="stretch")
                st.line_chart(sites.set_index("rank")[["cumulative_km2"]])
        if analysis is not None:
            st.write("Scored H3 evidence")
            st.dataframe(analysis.summaries.drop(columns="geometry"), hide_index=True, width="stretch")
        st.write("Native Ookla observations")
        st.dataframe(layers["ookla"].drop(columns="geometry"), hide_index=True, width="stretch")
        st.write("Estimated cell records")
        st.dataframe(layers["cells"].drop(columns="geometry"), hide_index=True, width="stretch")
        st.download_button("Download source manifest", json.dumps(manifest, indent=2),
                           "prototype-manifest.json", "application/json")
    with st.expander("Data credits and backend connections"):
        for attribution in manifest.get("attributions", {}).values():
            st.write(attribution)
        st.code("score.analyze(layers, fcc_complete, scenario)\noptimize.select_sites(candidates, need, site_count, radius_km)")
        st.caption(f"Dataset version: {version}. Scoring and selection start disabled; enable them when your modules are ready.")


if __name__ == "__main__":
    run()