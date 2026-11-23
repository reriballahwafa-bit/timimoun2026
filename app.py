"""
app.py - Streamlit user interface.      Run:  streamlit run app.py

Lets the user: pick a site or enter latitude/longitude, choose the temporal resolution and community,
select parameters (availability is checked live against the POWER parameter-manager API), define the period,
start the acquisition, view the charts/tables and download datasets and reports.
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from timimoun_climate import config as C
from timimoun_climate.nasa_power_client import PowerAPIError, PowerClient
from timimoun_climate.pipeline import run_pipeline

st.set_page_config(page_title="NASA POWER - Bioclimatic analysis", layout="wide", page_icon="☀️")
st.title("☀️ NASA POWER climate acquisition & bioclimatic analysis")
st.caption("Timimoun, Algeria - hot-arid climate | data: NASA POWER (MERRA-2 / CERES) | no API key required")

with st.sidebar:
    st.header("1 · Location")
    preset = st.selectbox("Site", list(C.SITE_PRESETS))
    plat, plon = C.SITE_PRESETS[preset]
    lat = st.number_input("Latitude (°N)", -90.0, 90.0, plat if plat is not None else C.DEFAULT_SITE["lat"], 0.0001, format="%.4f")
    lon = st.number_input("Longitude (°E)", -180.0, 180.0, plon if plon is not None else C.DEFAULT_SITE["lon"], 0.0001, format="%.4f")
    name = st.text_input("Site name", preset.split(" (")[0] if plat is not None else "Custom site")

    st.header("2 · Data service")
    temporal = st.selectbox("Temporal resolution", C.TEMPORALS, index=2)
    community = st.selectbox("Community", C.COMMUNITIES, help="RE renewable energy · AG agroclimatology · SB sustainable buildings")
    fmt = st.selectbox("Raw format", C.FORMATS, help="NetCDF is read with xarray")
    tstd = st.selectbox("Hourly time standard", ["LST", "UTC"]) if temporal == "hourly" else "LST"

    st.header("3 · Period")
    start = end = ""
    if temporal in ("daily", "hourly"):
        first = date(C.FIRST_YEAR[temporal], 1, 1)
        d0 = st.date_input("Start", date(2015, 1, 1) if temporal == "daily" else date(2022, 1, 1), min_value=first)
        d1 = st.date_input("End", date.today() - timedelta(days=C.NRT_LAG_DAYS))
        start, end = d0.strftime("%Y%m%d"), d1.strftime("%Y%m%d")
    elif temporal == "monthly":
        y0 = st.number_input("Start year", C.FIRST_YEAR["monthly"], date.today().year, 2000)
        y1 = st.number_input("End year", C.FIRST_YEAR["monthly"], date.today().year, date.today().year - 1)
        start, end = str(int(y0)), str(int(y1))
    else:
        st.info("Climatology returns long-term monthly means (no period needed).")

    st.header("4 · Parameters")
    try:
        avail, miss = PowerClient(community=community).check_parameters(temporal, list(C.PARAMETERS))
    except Exception as exc:  # offline
        avail, miss = list(C.PARAMETERS), []
        st.warning(f"Availability check failed: {exc}")
    choices = st.multiselect("Parameters", avail, default=[p for p in C.DEFAULT_PARAMS if p in avail],
                             format_func=lambda c: f"{c} - {C.PARAMETERS[c].label}")
    if miss:
        st.caption(f"Not available for {temporal}/{community}: {', '.join(miss)}")
    update = st.checkbox("Incremental update of existing dataset")
    run = st.button("▶ Start acquisition & analysis", type="primary", use_container_width=True)

if run:
    cfg = C.RunConfig(name=name, lat=lat, lon=lon, temporal=temporal, community=community, parameters=choices,
                      start=start, end=end, time_standard=tstd, raw_format=fmt, data_dir=Path("data"))
    bar = st.progress(0.0, "Contacting NASA POWER ...")
    try:
        st.session_state["out"] = run_pipeline(cfg, progress=lambda f, m: bar.progress(f, m), update=update)
        bar.empty()
    except (PowerAPIError, ValueError) as exc:
        bar.empty()
        st.error(f"Acquisition failed: {exc}")

out = st.session_state.get("out")
if out:
    cd, res, bio = out["climate"], out["analysis"], out["bioclimatic"]
    ov = res["overall"]
    c = st.columns(5)
    c[0].metric("Mean T", f"{ov['T_annual_mean']:.1f} °C")
    c[1].metric("Abs. max", f"{ov['Tmax_abs']:.1f} °C")
    c[2].metric("GHI", f"{ov['GHI_mean_kWh_m2_day']:.2f} kWh/m²/d")
    c[3].metric("Mean RH", f"{ov['RH_mean']:.0f} %")
    c[4].metric("Precip.", f"{res['precipitation']['annual_mm'] or 0:.1f} mm/yr")
    tabs = st.tabs(["Charts", "Statistics", "Bioclimatic", "Recommendations", "Quality & limits", "Downloads"])
    with tabs[0]:
        for k, fig in out["figures"].items():
            st.pyplot(fig)
    with tabs[1]:
        st.subheader("Monthly [NASA stats / CALC]")
        st.dataframe(res["monthly"].round(2))
        st.subheader("Seasonal")
        st.dataframe(res["seasonal"].round(2))
        if res["thermal_stress"]["summary"] is not None:
            st.subheader("Thermal stress [CALC]")
            st.dataframe(res["thermal_stress"]["summary"].round(1))
        st.subheader("Data (head)")
        st.dataframe(out["result"].data.head(500))
    with tabs[2]:
        st.subheader("Adaptive comfort [CALC]")
        st.dataframe(bio["comfort"].round(1))
        st.subheader("Surface exposure kWh/m²/day [CALC]")
        st.dataframe(bio["orientation_summary"].round(2))
        st.subheader("Strategy matrix [INTERP]")
        st.dataframe(bio["strategies"].astype(int))
    with tabs[3]:
        for r in bio["recommendations"]:
            st.markdown(f"**{r['topic']}** · _{r['priority']}_  \n{r['recommendation']}  \n<small>Evidence: {r['evidence']}</small>", unsafe_allow_html=True)
    with tabs[4]:
        st.dataframe(out["validation"])
        for i in out["issues"] + cd.notes:
            st.warning(i)
        from timimoun_climate.reporting import LIMITATIONS
        st.markdown("\n".join(f"- {l}" for l in LIMITATIONS))
    with tabs[5]:
        for key in ("pdf", "xlsx"):
            if key in out:
                st.download_button(f"Download report ({key.upper()})", Path(out[key]).read_bytes(), Path(out[key]).name)
        for p in out["files"]:
            st.download_button(f"Download {Path(p).name}", Path(p).read_bytes(), Path(p).name, key=str(p))
        st.caption(f"Raw responses preserved in: {out['config'].raw_dir}")
else:
    st.info("Choose the options in the sidebar and press **Start acquisition & analysis**.")
