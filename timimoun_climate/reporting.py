"""
reporting.py - structured analytical report (PDF via ReportLab) and Excel workbook (openpyxl).

Report outline (as specified in the project brief)
    1 Geographic and climatic context      5 Seasonal analysis
    2 Data sources and acquisition period  6 Bioclimatic interpretation
    3 Climate statistics                   7 Architectural design recommendations
    4 Climate diagrams                     8 Data limitations and reliability
Evidence layers are labelled in every section: [NASA] retrieved, [CALC] calculated, [INTERP] interpretation.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

LIMITATIONS = [
    "NASA POWER is a model/satellite-assimilation product (MERRA-2 meteorology on a 0.5° x 0.625° grid; CERES SYN1deg "
    "radiation on 1° x 1°), NOT a ground-station record. Values describe a ~50 km grid cell, not the oasis, ksar or street scale.",
    "Local effects - urban heat island of the ksar, palm-grove (palmeraie) evapotranspiration, sebkha surfaces, courtyard "
    "microclimates - are not resolved. Validate against the nearest ONM (Office National de la Météorologie) station before design decisions.",
    "T2M_MAX / T2M_MIN are extremes of hourly values. In the monthly and climatology services they are monthly extremes, "
    "not mean daily maxima/minima. Mean diurnal range needs daily or hourly data.",
    "Wind speed/direction at 2 m are reanalysis values and smooth out dust-devils, gusts and local topographic channelling; "
    "use only for prevailing-direction and order-of-magnitude assessments.",
    "PRECTOTCORR is a bias-corrected reanalysis product; rare convective storms in the Sahara are poorly captured. "
    "Annual totals of a few mm should be read as 'hyper-arid', not as exact values.",
    "Hourly data are given in Local Solar Time (15° longitude bands) unless UTC was requested; this differs from Algerian legal time (UTC+1).",
    "Calculated indicators (heat index, humidex, wet-bulb, façade irradiation, PET) are empirical approximations; daily-mean-based "
    "heat-stress indices under-estimate afternoon peaks. Façade irradiation uses an isotropic-sky model and distributes daily "
    "energy by extraterrestrial shape (no clear-sky/cloud timing).",
    "The ASHRAE 55 adaptive model was calibrated for naturally ventilated buildings with T_rm between 10 and 33.5 °C; here it is "
    "clamped at those limits. Treat the comfort band as a reference, not as a prediction of occupant votes in Timimoun.",
    "Strategy flags and recommendations are rule-based interpretations with thresholds listed in config.Assumptions; they "
    "support, but do not replace, dynamic thermal simulation (e.g. EnergyPlus/DesignBuilder) and fieldwork on heritage buildings.",
    "Results depend on the retrieved period; use >= 10 years of daily/hourly data for stable statistics.",
]

CONTEXT = ("Timimoun is the capital of the Gourara region in the Algerian Sahara (Adrar Province), set on the edge of the Great "
           "Western Erg at the foot of the Gourara plateau and a sebkha. The oasis is irrigated by the traditional foggara "
           "system and is famed for its red earthen (toub) ksour. The climate is hot desert (Köppen BWh) with extremely high "
           "insolation, very low humidity, rare rain and a large diurnal range.")


def _f(v, n=1):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def write_excel(path: Path, cd, res: dict, bio: dict, validation: pd.DataFrame, issues: list[str],
                max_hourly_rows: int = 250_000) -> Path:
    from openpyxl.styles import Alignment, Font, PatternFill
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets: dict[str, tuple[str, pd.DataFrame]] = {}

    def add(name, layer, df, index=True):
        if df is not None and len(df):
            sheets[name] = (layer, df.reset_index() if index else df)

    meta = cd.meta
    info = pd.DataFrame({"item": ["Generated (UTC)", "Latitude", "Longitude", "Elevation (m, MERRA-2 cell mean)", "Temporal service",
                                  "Time standard", "Period", "Community", "Parameters retrieved", "Unavailable parameters",
                                  "Sources", "API version(s)", "Fill value -> NaN"],
                         "value": [datetime.utcnow().isoformat(timespec="seconds"), cd.lat, cd.lon, meta.get("elevation_m"),
                                   cd.temporal, meta.get("time_standard"), cd.period, meta.get("community", ""),
                                   ", ".join(meta.get("parameters", [])), ", ".join(meta.get("unavailable_parameters", [])) or "none",
                                   ", ".join(meta.get("sources", [])), ", ".join(meta.get("api_versions", [])), meta.get("fill_value")]})
    add("Metadata", "[NASA] provenance", info, False)
    add("Units", "[NASA]", pd.DataFrame({"parameter": list(meta.get("units", {})), "unit": list(meta.get("units", {}).values()),
                                          "long_name": [meta.get("long_names", {}).get(k) for k in meta.get("units", {})]}), False)
    if cd.daily is not None:
        add("Data_daily", "[NASA] + [CALC] derived columns", cd.daily)
    if cd.hourly is not None and len(cd.hourly) <= max_hourly_rows:
        add("Data_hourly", "[NASA]", cd.hourly)
    if cd.monthly is not None:
        add("Data_monthly", "[NASA]", cd.monthly)
    if cd.climatology is not None:
        add("Data_climatology", "[NASA]", cd.climatology)
    add("Monthly_stats", "[CALC]", res["monthly"])
    add("Seasonal_stats", "[CALC]", res["seasonal"])
    add("Annual_stats", "[CALC]", res["annual"])
    ts = res["thermal_stress"]
    add("Thermal_stress_summary", "[CALC]", ts["summary"])
    add("Thermal_stress_yearly", "[CALC]", ts["yearly"])
    add("Hot_hours_yearly", "[CALC]", ts["hourly_summary"])
    if res["wind"]["rose"] is not None:
        add("Wind_rose_%", "[CALC]", res["wind"]["rose"])
        add("Wind_summary", "[CALC]", pd.DataFrame(list(res["wind"]["summary"].items()), columns=["metric", "value"]), False)
    pr = res["precipitation"]
    add("Precip_aridity", "[CALC]", pd.DataFrame({"metric": ["Annual P (mm)", "Max daily P (mm)", "Wet days/yr", "De Martonne",
                                                              "UNEP aridity index P/PET", "Class"],
                                                  "value": [pr["annual_mm"], pr["max_daily_mm"], pr["wet_days"], pr["de_martonne"],
                                                            pr["aridity_index"], pr["class"]]}), False)
    for k, df in (res["diurnal"] or {}).items():
        if k != "by_season":
            add(f"Diurnal_{k}"[:31], "[CALC]", df)
    add("Facade_irradiation", "[CALC]", bio["orientation"])
    add("Facade_summary", "[CALC]", bio["orientation_summary"])
    add("Shading_design", "[CALC]", bio["shading"])
    add("Adaptive_comfort", "[CALC]", bio["comfort"])
    add("Strategy_metrics", "[CALC]", bio["strategy_metrics"])
    add("Strategy_matrix", "[INTERP]", bio["strategies"].astype(int))
    add("Recommendations", "[INTERP]", pd.DataFrame(bio["recommendations"]), False)
    add("Validation", "[CALC] QA", validation, False)
    add("Issues_Limitations", "QA", pd.DataFrame({"note": issues + cd.notes + LIMITATIONS}), False)
    readme = pd.DataFrame({"sheet": list(sheets), "evidence layer": [v[0] for v in sheets.values()]})
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for name, (_, df) in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        for ws in xw.book.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="7C2D12")
                cell.alignment = Alignment(horizontal="center", wrap_text=True)
            ws.freeze_panes = "A2"
            for col in ws.columns:
                w = max((len(str(c.value)) for c in list(col)[:60] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 70)
    return path


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def write_pdf(path: Path, cd, res: dict, bio: dict, figure_paths: dict, validation: pd.DataFrame, issues: list[str],
              site_name: str = "Timimoun") -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#7c2d12"), spaceBefore=10)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#9a3412"))
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12.5)
    S = ParagraphStyle("S", parent=B, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#4b5563"))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title=f"Climate & bioclimatic analysis - {site_name}")
    W = A4[0] - 3.2 * cm
    story = []
    P = lambda t, st=B: story.append(Paragraph(t, st))

    def table(df: pd.DataFrame, col_w=None, fs=7):
        data = [[str(c) for c in df.columns]] + [[(_f(v, 2) if isinstance(v, (float, np.floating)) else str(v)) for v in r] for r in df.to_numpy()]
        t = Table(data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c2d12")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff7ed")]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(t)
        story.append(Spacer(1, 6))

    def para_table(rows, widths, header):
        data = [[Paragraph(f"<b>{h}</b>", S) for h in header]] + [[Paragraph(str(c), S) for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fed7aa")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t)

    def fig(name, width=W, caption=None):
        if name in figure_paths:
            from PIL import Image as PILImage
            w, h = PILImage.open(figure_paths[name]).size
            story.append(Image(str(figure_paths[name]), width=width, height=width * h / w))
            if caption:
                P(caption, S)
            story.append(Spacer(1, 6))

    mt, ov, meta = res["monthly"], res["overall"], cd.meta
    # ---- title
    P(f"Climate Data Acquisition &amp; Bioclimatic Analysis - {site_name}, Algeria", ParagraphStyle("T", parent=H1, fontSize=20, leading=24))
    P(f"Automated report generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC from NASA POWER ({cd.temporal} service). "
      "Evidence layers: <b>[NASA]</b> retrieved variable - <b>[CALC]</b> calculated indicator - <b>[INTERP]</b> architectural interpretation.", S)
    story.append(Spacer(1, 8))

    # ---- 1
    P("1. Geographic and climatic context", H1)
    P(CONTEXT)
    pr = res["precipitation"]
    table(pd.DataFrame({"Item": ["Latitude / longitude", "Elevation (MERRA-2 cell mean)", "Annual mean temperature [NASA stat]",
                                 "Hottest / coldest month [CALC]", "Absolute extremes [NASA stat]", "Mean relative humidity",
                                 "Global horizontal irradiation [NASA stat]", "Annual precipitation / aridity [CALC]"],
                        "Value": [f"{cd.lat:.4f}°, {cd.lon:.4f}°", f"{_f(meta.get('elevation_m'), 0)} m", f"{ov['T_annual_mean']:.1f} °C",
                                  f"{C.MONTHS[ov['hottest_month'] - 1]} ({mt['T_mean'].max():.1f} °C) / {C.MONTHS[ov['coldest_month'] - 1]} ({mt['T_mean'].min():.1f} °C)",
                                  f"{_f(ov['Tmax_abs'])} °C / {_f(ov['Tmin_abs'])} °C", f"{_f(ov['RH_mean'], 0)} %",
                                  f"{_f(ov['GHI_mean_kWh_m2_day'], 2)} kWh/m²/day  ({_f(ov['GHI_annual_kWh_m2'], 0)} kWh/m²/yr)",
                                  f"{_f(pr['annual_mm'], 1)} mm; P/PET = {_f(pr['aridity_index'], 3)} ({pr['class'] or 'n/a'})"]}),
          [6 * cm, W - 6 * cm], 8)

    # ---- 2
    P("2. Data sources and acquisition period", H1)
    P(f"Source: NASA Prediction Of Worldwide Energy Resources (POWER), <i>{', '.join(meta.get('sources', [])) or 'MERRA-2, CERES SYN1deg'}</i>; "
      f"API version(s) {', '.join(meta.get('api_versions', [])) or 'n/a'}; community {meta.get('community', 'RE')}; "
      f"time standard {meta.get('time_standard') or 'LST'}; no API key required. Period: <b>{cd.period}</b>. "
      f"Retrieved: {meta.get('retrieved_utc', '')[:19]} UTC. Raw responses are preserved unchanged under <i>data/raw/</i> with request sidecar files (URL, SHA-256).")
    rows = [[k, (meta.get('long_names') or {}).get(k) or "", (meta.get('units') or {}).get(k, "")] for k in meta.get("parameters", [])]
    para_table(rows, [4 * cm, W - 6.5 * cm, 2.5 * cm], ["Parameter", "Name", "Unit"])
    if meta.get("unavailable_parameters"):
        P(f"Not offered by the {cd.temporal} service (checked via the POWER parameter-manager API): {', '.join(meta['unavailable_parameters'])}.", S)
    story.append(Spacer(1, 4))
    vt = validation[["parameter", "unit", "n", "missing", "completeness_%", "min", "mean", "max", "out_of_range"]]
    P("Data validation [CALC]", H2)
    table(vt, None, 6.5)
    for i in issues + cd.notes:
        P("- " + i, S)

    # ---- 3
    story.append(PageBreak())
    P("3. Climate statistics", H1)
    P("Long-term monthly statistics [NASA stat / CALC]", H2)
    cols = [c for c in ["T_mean", "Tmax_mean", "Tmin_mean", "Tmax_abs", "Tmin_abs", "DTR", "RH_mean", "GHI", "WS_mean", "P_month_mm"] if c in mt]
    t = mt[cols].round(1).copy()
    t.insert(0, "Month", C.MONTHS)
    table(t, None, 7)
    P("T_mean °C; Tmax/Tmin_mean = mean daily extremes; Tmax/Tmin_abs = absolute extremes; DTR = mean diurnal range (K); RH %; GHI kWh/m²/day; WS m/s at 2 m; P mm/month.", S)
    ts = res["thermal_stress"]
    if ts["summary"] is not None:
        P("Thermal-stress indicators [CALC] (days or degree-days per year)", H2)
        s = ts["summary"].round(1).reset_index().rename(columns={"index": "indicator"})
        table(s, [8 * cm] + [(W - 8 * cm) / 3] * 3, 7.5)
    if ts["hourly_summary"] is not None:
        table(ts["hourly_summary"].round(0).reset_index().astype(str), None, 7.5)

    # ---- 4
    story.append(PageBreak())
    P("4. Climate diagrams", H1)
    for i, (k, cap) in enumerate([("01_monthly_temperature", "Fig. 1 Monthly temperature regime"),
                                  ("02_temperature_distribution", "Fig. 2 Temperature distribution"),
                                  ("03_solar_radiation", "Fig. 3 Solar radiation and surface exposure"),
                                  ("04_relative_humidity", "Fig. 4 Relative humidity"),
                                  ("05_wind_rose", "Fig. 5 Wind roses"), ("06_precipitation", "Fig. 6 Precipitation and aridity"),
                                  ("07_diurnal_profiles", "Fig. 7 Diurnal profiles"), ("08_thermal_carpet", "Fig. 8 Thermal carpet")]):
        fig(k, W, cap)
    # ---- 5
    P("5. Seasonal analysis", H1)
    sea = res["seasonal"][[c for c in ["T_mean", "Tmax_mean", "Tmin_mean", "DTR", "RH_mean", "GHI", "WS_mean", "P_season_mm"] if c in res["seasonal"]]].round(1)
    table(sea.reset_index().rename(columns={"index": "Season"}), None, 7.5)
    fig("09_seasonal_comparison", W, "Fig. 9 Seasonal comparison (distribution of daily values)")
    # ---- 6
    story.append(PageBreak())
    P("6. Bioclimatic interpretation", H1)
    P("Adaptive comfort [CALC] - ASHRAE 55-2020, 80 % acceptability; thermal state shares [CALC]", H2)
    cf = bio["comfort"]
    cc = cf[[c for c in ["Trm", "T_comfort", "T_lower", "T_upper", "day_hot_%", "day_comfort_%", "day_cold_%", "night_hot_%", "night_comfort_%", "night_cold_%"] if c in cf]].round(1)
    cc.insert(0, "Month", C.MONTHS)
    table(cc, None, 6.5)
    fig("10_adaptive_comfort", W, "Fig. 10 Adaptive comfort band and thermal state")
    P("Solar exposure of building surfaces [CALC] (kWh/m²/day)", H2)
    os_ = bio["orientation_summary"].copy()
    if len(os_):
        os_ = os_.round(2).reset_index().rename(columns={"index": "Surface"})
        table(os_, None, 7)
    P("Shading geometry [CALC]", H2)
    table(bio["shading"].round(2).reset_index(), None, 7.5)
    fig("12_sun_path", 9 * cm, "Fig. 11 Sun path")
    P("Passive strategy matrix [INTERP]", H2)
    fig("11_strategy_matrix", W, "Fig. 12 Rule-based passive strategy matrix")
    fig("13_bioclimatic_dashboard", W, "Fig. 13 Bioclimatic dashboard")
    # ---- 7
    story.append(PageBreak())
    P("7. Architectural design recommendations [INTERP]", H1)
    rows = [[r["priority"], r["topic"], r["recommendation"], r["evidence"]] for r in bio["recommendations"]]
    para_table(rows, [1.4 * cm, 2.6 * cm, W * 0.42, W - 4 * cm - W * 0.42], ["Priority", "Topic", "Recommendation", "Evidence (calculated)"])
    # ---- 8
    P("8. Data limitations and reliability", H1)
    for l in LIMITATIONS:
        P("- " + l, B)
    P("References: NASA POWER documentation (power.larc.nasa.gov/docs); ASHRAE 55-2020; de Dear &amp; Brager (1998); Duffie &amp; Beckman (2013); "
      "Hargreaves &amp; Samani (1985); Stull (2011); Rothfusz (1990); Masterton &amp; Richardson (1979); UNEP (1997) aridity index.", S)
    doc.build(story)
    return path
