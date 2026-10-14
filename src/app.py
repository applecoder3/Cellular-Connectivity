import html
import json
import os
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from .config import ROOT, WGS84, METRIC_CRS, Scenario, LIMITATION
from .export import export_sites, export_evidence
from .grid import candidate_sites
from .pipeline import load_prepared
from .optimize import select_sites
from .score import analyze

COLORS = {
    "Observed low speed": "#dc4b35",
    "Measured (not low)": "#258879",
    "Insufficient measurements": "#d8a642",
    "Unmeasured": "#a6b0bc",
}

@st.cache_data(show_spinner="Loading prepared county layers…")
def cached_layers(directory, version):
    return load_prepared(Path(directory), version)

@st.cache_data(show_spinner="Combining evidence…")
def cached_analysis(directory, version, threshold, tests, devices):
    layers, manifest = cached_layers(directory, version)
    scenario = Scenario(download_mbps=threshold, min_tests=tests, min_devices=devices)
    return analyze(layers, manifest["fcc"]["complete"], scenario)

@st.cache_data(show_spinner="Selecting sites by additional area reached…")
def cached_selection(directory, version, threshold, tests, devices, count, radius):
    layers, _ = cached_layers(directory, version)
    analysis = cached_analysis(directory, version, threshold, tests, devices)
    return select_sites(candidate_sites(layers["grid"]), analysis.need, count, radius)

def geojson(frame, fields=None):
    result = frame.to_crs(WGS84).copy()
    if fields is not None:
        result = result[fields + ["geometry"]]
    # Escape textual properties displayed by Leaflet tooltips.
    for column in result.columns:
        if column != "geometry" and pd.api.types.is_object_dtype(result[column]):
            result[column] = result[column].map(lambda v: html.escape(str(v)) if v is not None else None)
    return json.loads(result.to_json(drop_id=True, na="null"))

def polygon_layer(frame, name, style, fields=None, show=False):
    return folium.GeoJson(
        geojson(frame, fields), name=name, show=show,
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=fields, localize=True) if fields else None,
    )

def make_map(layers, analysis, sites, providers, manifest, radius_km):
    county = layers["county"].to_crs(WGS84)
    center = county.geometry.iloc[0].representative_point()
    view = folium.Map(location=[center.y, center.x], tiles="OpenStreetMap", zoom_start=10, control_scale=True)
    xmin, ymin, xmax, ymax = county.total_bounds
    view.fit_bounds([[ymin, xmin], [ymax, xmax]])
    polygon_layer(county, "Franklin County", lambda _: {
        "color": "#22384a", "weight": 2, "fillOpacity": 0,
    }, ["GEOID", "NAME"], True).add_to(view)
    polygon_layer(analysis.summaries, "Measurement evidence (H3)", lambda feature: {
        "color": "#ffffff", "weight": 0.3, "fillOpacity": 0.55,
        "fillColor": COLORS[feature["properties"]["evidence"]],
    }, ["h3_id", "evidence", "download_mbps", "observed_fraction",
        "reliable_fraction", "loaded_claim_fraction", "slow_km2", "gap_km2"], True).add_to(view)
    scale = LinearColormap(["#dc4b35", "#efba53", "#258879", "#174f77"], vmin=0, vmax=100)
    scale.caption = "Observed download Mbps (colors capped at 100; missing = gray)"
    def speed_style(feature):
        speed = feature["properties"].get("download_mbps")
        return {"color": "#ffffff", "weight": 0.3, "fillOpacity": 0.7,
                "fillColor": "#a6b0bc" if speed is None else scale(min(speed, 100))}
    polygon_layer(analysis.summaries, "Approximate H3 download speed", speed_style,
                  ["h3_id", "download_mbps", "upload_mbps", "latency_ms", "effective_tests"]).add_to(view)
    if not analysis.tiles.empty:
        polygon_layer(analysis.tiles, "Native Ookla speed tiles", speed_style,
                      ["quadkey", "download_mbps", "upload_mbps", "latency_ms", "tests", "devices", "evidence"]).add_to(view)
    if not analysis.claims_union.geometry.iloc[0].is_empty:
        name = "Combined LTE claims" if manifest["fcc"]["complete"] else "Loaded LTE claims (inventory incomplete)"
        polygon_layer(analysis.claims_union, name, lambda _: {
            "color": "#446aba", "weight": 1, "fillColor": "#446aba", "fillOpacity": 0.15,
        }).add_to(view)
    for provider in providers:
        claims = layers["claims"]
        subset = claims.loc[claims.provider_id == provider]
        if not subset.empty:
            provider_name = html.escape(str(subset.provider_name.iloc[0]))
            polygon_layer(subset, f"FCC LTE: {provider_name}", lambda _: {
                "color": "#446aba", "weight": 1, "fillOpacity": 0.20,
            }, ["provider_id", "provider_name"], True).add_to(view)
    if not analysis.need.empty:
        polygon_layer(analysis.need, "Eligible need: slow observations + enabled claim gaps",
                      lambda _: {"color": "#b13825", "weight": 1, "fillColor": "#ef8354", "fillOpacity": 0.4}).add_to(view)
    cells = folium.FeatureGroup(name="Estimated cell positions (not physical tower inventory)", show=False)
    for row in layers["cells"].to_crs(WGS84).itertuples():
        description = " · ".join(f"{key}: {html.escape(str(getattr(row, key, 'unknown')))}"
                                 for key in ["radio", "mcc", "net", "area", "cell", "samples", "updated"])
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=3, color="#40546c",
                            fill=True, tooltip=description).add_to(cells)
    cells.add_to(view)
    if not sites.empty:
        footprints = sites[["geometry"]].copy()
        footprints["geometry"] = footprints.geometry.buffer(radius_km * 1000, quad_segs=32)
        polygon_layer(footprints, "Hypothetical site footprints", lambda _: {
            "color": "#842bb6", "weight": 2, "fillColor": "#842bb6", "fillOpacity": 0.12,
        }, show=True).add_to(view)
        markers = folium.FeatureGroup(name="Selected sites", show=True)
        for row in sites.to_crs(WGS84).itertuples():
            folium.Marker([row.geometry.y, row.geometry.x],
                          tooltip=f"Site {row.rank}: +{row.incremental_km2:.2f} km²",
                          icon=folium.Icon(color="purple", icon="signal")).add_to(markers)
        markers.add_to(view)
    scale.add_to(view)
    folium.LayerControl(collapsed=False).add_to(view)
    return view

def run():
    st.set_page_config(page_title="Franklin County | Cellular coverage", page_icon="📡", layout="wide")
    st.title("Franklin County cellular coverage")
    st.write("Compare reported LTE coverage with observed mobile speeds, then explore hypothetical new sites.")
    directory = Path(os.environ.get("CELLULAR_DATA_DIR", str(ROOT / "data" / "processed"))).resolve()
    try:
        pointer = json.loads((directory / "current.json").read_text(encoding="utf-8"))
        version = pointer["version"]
        layers, manifest = cached_layers(str(directory), version)
    except (OSError, ValueError, KeyError) as exc:
        st.info("Prepare your county dataset before opening the explorer.")
        st.code("python main.py prepare --config config.example.json")
        st.caption(f"Dataset unavailable: {exc}")
        return

    with st.sidebar:
        st.header("Scenario")
        threshold = st.number_input("Low download threshold (Mbps)", min_value=0.1, max_value=1000.0, value=5.0, step=1.0)
        tests = st.number_input("Minimum tests per native tile", min_value=1, value=10, step=1)
        devices = st.number_input("Minimum devices per native tile", min_value=1, value=3, step=1)
        count = st.slider("Maximum new sites", 0, 10, 3)
        radius = st.select_slider("Illustrative radius (km)", options=[1.0, 2.0, 5.0], value=2.0)
        claims = layers["claims"]
        names = dict(zip(claims.provider_id, claims.provider_name)) if not claims.empty else {}
        providers = st.multiselect("Display provider claims", options=sorted(names),
                                   format_func=lambda key: names[key])
        st.caption("Provider display changes only the map. Ookla measurements and site selection remain county-wide.")
        st.caption("Sample thresholds describe evidence availability, not statistical certainty.")
    scenario = Scenario(float(threshold), int(tests), int(devices), int(count), float(radius))
    analysis = cached_analysis(str(directory), version, scenario.download_mbps, scenario.min_tests, scenario.min_devices)
    sites = cached_selection(str(directory), version, scenario.download_mbps, scenario.min_tests,
                             scenario.min_devices, scenario.site_count, scenario.radius_km)

    if not manifest["fcc"]["complete"]:
        st.warning("Combined FCC claim-gap analysis is disabled. " + " ".join(manifest["fcc"]["reasons"]))
    if "ookla" not in manifest["sources"]:
        st.info("Ookla mobile data has not been supplied. All measurement areas are unmeasured.")
    elif layers["ookla"].empty:
        st.info("The supplied Ookla quarter has no tiles intersecting Franklin County.")
    elif not analysis.tiles.sufficient.any():
        st.warning(
            f"None of the {len(analysis.tiles)} Ookla tiles meets the current sample thresholds "
            f"({scenario.min_tests} tests and {scenario.min_devices} devices per tile). "
            "Speeds remain visible, but these tiles do not qualify for low-speed site selection. "
            "Lower thresholds in the sidebar only for exploratory analysis with weaker evidence."
        )
    elif scenario.min_tests < 10 or scenario.min_devices < 3:
        st.warning("Exploratory sample thresholds are below the defaults of 10 tests and 3 devices. Site results use weaker measurement evidence.")
    if "ookla" in manifest["sources"] and not manifest["fcc"]["reference_date"]:
        st.info("The FCC availability reference date is unconfirmed, so alignment with the Ookla quarter has not been verified.")
    if manifest["temporal_mismatch"]:
        st.warning(manifest["temporal_mismatch"])
    st.caption("Missing measurements do not identify dead zones. FCC polygons are claims; OpenCelliD points are estimated cell locations.")

    eligible_area = float(analysis.need.geometry.area.sum() / 1e6)
    reached = float(sites.cumulative_km2.iloc[-1]) if not sites.empty else 0.0
    columns = st.columns(4)
    columns[0].metric("County area", f"{layers['grid'].area_km2.sum():,.1f} km²")
    columns[1].metric("Eligible need", f"{eligible_area:,.2f} km²")
    columns[2].metric("Area reached in scenario", f"{reached:,.2f} km²")
    columns[3].metric("Selected sites", len(sites))

    map_tab, sites_tab, evidence_tab, source_tab = st.tabs(["Coverage map", "Site scenarios", "Evidence details", "Data & assumptions"])
    with map_tab:
        st_folium(make_map(layers, analysis, sites, providers, manifest, radius),
                height=620, use_container_width=True, returned_objects=[], key="coverage-map")
        legend = st.columns(4)
        for column, (label, color) in zip(legend, COLORS.items()):
            column.markdown(f":{ {'#dc4b35':'red','#258879':'green','#d8a642':'orange','#a6b0bc':'gray'}[color]}[{label}]")
        st.caption("Hexagons can contain mixed evidence; hover for observed and reliable area fractions. Native tiles retain source averages.")
    with sites_tab:
        st.write(LIMITATION)
        if sites.empty:
            st.info("No sites selected: there is no eligible need, the site budget is zero, or no candidate adds area.")
        else:
            table = sites.drop(columns="geometry")
            st.dataframe(table, hide_index=True, width="stretch")
            chart = pd.concat([pd.DataFrame({"rank": [0], "cumulative_km2": [0.0]}),
                        table[["rank", "cumulative_km2"]]], ignore_index=True)
            st.plotly_chart(px.line(chart, x="rank", y="cumulative_km2", markers=True,
                        labels={"rank": "Sites selected", "cumulative_km2": "Eligible area reached (km²)"}),
                        width="stretch")
        csv, geo = export_sites(sites, manifest, scenario)
        left, right = st.columns(2)
        left.download_button("Download site CSV", csv, "franklin-sites.csv", "text/csv")
        right.download_button("Download site GeoJSON + provenance", geo, "franklin-sites.geojson", "application/geo+json")
        if st.button("Compare 1, 2 and 5 km radius scenarios"):
            comparisons = []
            for candidate_radius in [1.0, 2.0, 5.0]:
                selected = cached_selection(str(directory), version, threshold, int(tests), int(devices), int(count), candidate_radius)
                comparisons.append({"radius_km": candidate_radius, "sites": len(selected),
                                    "reached_km2": float(selected.cumulative_km2.iloc[-1]) if len(selected) else 0.0})
            st.dataframe(pd.DataFrame(comparisons), hide_index=True, width="stretch")
    with evidence_tab:
        st.write("H3 means are approximate: native tile averages are weighted by tests × spatial overlap fraction. Effective test counts are fractional estimates; device counts are not combined across tiles.")
        st.dataframe(analysis.summaries.drop(columns="geometry"), hide_index=True, width="stretch")
        st.download_button("Download H3 evidence CSV", export_evidence(analysis.summaries, manifest, scenario),
                           "franklin-evidence.csv", "text/csv")
        if not analysis.tiles.empty:
            st.write("Native mobile tile measurements")
            st.dataframe(analysis.tiles.drop(columns="geometry"), hide_index=True, width="stretch")
    with source_tab:
        st.write(LIMITATION)
        st.write("Selection maximizes the union of eligible area inside circular footprints. After each site, overlapping benefit is removed before choosing the next site.")
        st.write("Default need: adequately sampled download below 5 Mbps, plus claim gaps only when the complete FCC LTE inventory is verified.")
        st.caption(f"Pinned dataset: {version}")
        for name, attribution in manifest["attributions"].items():
            st.write(f"{name}: {attribution}")
        if "ookla" in manifest["sources"]:
            source = manifest["sources"]["ookla"]
            st.write(f"Speedtest® by Ookla® maps accessed on {source['acquired_date'] or 'date not recorded'}; analysis period {source['quarter']}.")
        st.download_button("Download dataset manifest", json.dumps(manifest, indent=2),
                           "franklin-manifest.json", "application/json")
        with st.expander("Dataset manifest"):
            st.json(manifest)
