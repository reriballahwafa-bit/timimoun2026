"""
bioclimatic.py - architectural interpretation of the climate data.

Layers (kept strictly separate, see processing.py header):
    [NASA]   inputs   - retrieved POWER variables
    [CALC]   outputs  - comfort band, façade irradiation, shading ratios, % of time hot/cold ...
    [INTERP] outputs  - strategy matrix and design recommendations (rule-based; thresholds in config.Assumptions)

Adaptive comfort model: ASHRAE Standard 55-2020 (de Dear & Brager 1998):
    T_comfort = 0.31 * T_rm + 17.8  (°C),  80 % acceptability limits  ± 3.5 K
    T_rm = running-mean outdoor temperature, alpha = 0.8:   T_rm(d) = (1-a) * sum_i a^i * T(d-1-i)
The model is formally valid for 10 <= T_rm <= 33.5 °C; Timimoun exceeds this in summer, so the report flags it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import solar

A = C.ASSUMPTIONS
MONTHS = C.MONTHS


def month_ranges(months: list[int]) -> str:
    """[6,7,8,9] -> 'Jun-Sep' ; handles wrap-around (e.g. Nov-Feb)."""
    ms = sorted(set(int(m) for m in months))
    if not ms:
        return "none"
    if len(ms) == 12:
        return "all year"
    runs, start = [], ms[0]
    for a, b in zip(ms, ms[1:] + [None]):
        if b is None or b != a + 1:
            runs.append((start, a))
            start = b
    if len(runs) > 1 and runs[0][0] == 1 and runs[-1][1] == 12:
        runs[0] = (runs[-1][0], runs[0][1])
        runs.pop()
    return ", ".join(MONTHS[a - 1] if a == b else f"{MONTHS[a - 1]}-{MONTHS[b - 1]}" for a, b in runs)


# --------------------------------------------------------------------------- #
# [CALC] solar exposure of building surfaces
# --------------------------------------------------------------------------- #

def orientation_table(lat: float, mt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Monthly daily-irradiation (kWh/m²/day) on a horizontal roof and on 8 vertical façades.
    Returns (monthly table, summary table, diffuse_assumed)."""
    if "GHI" not in mt:
        return pd.DataFrame(), pd.DataFrame(), False
    diffuse_assumed = "DIFF" not in mt or mt["DIFF"].isna().all()
    surf = {"Roof": (0, 0)} | {k: (90, az) for k, az in C.ORIENTATIONS.items()}
    tab = pd.DataFrame(index=mt.index, columns=list(surf), dtype=float)
    for k in mt.index:
        n = solar.MID_MONTH_DOY[k - 1]
        ghi = mt.loc[k, "GHI"]
        dhi = np.nan if diffuse_assumed else mt.loc[k, "DIFF"]
        for name, (tilt, az) in surf.items():
            tab.loc[k, name] = solar.surface_irradiance_day(lat, n, ghi, dhi, tilt, az, A.albedo)
    summer, winter = [6, 7, 8], [12, 1, 2]
    if lat < 0:
        summer, winter = winter, summer
    sm = pd.DataFrame({"annual_mean": tab.mean(), "summer_mean": tab.loc[summer].mean(), "winter_mean": tab.loc[winter].mean(),
                       "peak_month": tab.idxmax().map(lambda m: MONTHS[m - 1]), "peak_value": tab.max()})
    sm["summer/winter"] = sm["summer_mean"] / sm["winter_mean"]
    return tab, sm, diffuse_assumed


def shading_design(lat: float) -> pd.DataFrame:
    """Noon solar altitude and fully-shading overhang ratio (P/H) for the equator-facing wall."""
    rows = []
    for name, n in {"Winter solstice (21 Dec)": 355, "Equinox (21 Mar)": 80, "Summer solstice (21 Jun)": 172}.items():
        _, alt, az = solar.sun_position(lat, n, [12.0])
        rows.append({"date": name, "noon_altitude_deg": float(alt[0]),
                     "overhang_P/H_for_full_noon_shade": solar.overhang_ratio(lat, n)})
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- #
# [CALC] adaptive comfort
# --------------------------------------------------------------------------- #

def running_mean(t_daily: pd.Series, alpha: float = A.runmean_alpha) -> pd.Series:
    """ASHRAE 55 exponentially-weighted running-mean outdoor temperature."""
    return t_daily.shift(1).ewm(alpha=1 - alpha, adjust=False).mean().fillna(t_daily.mean())


def comfort_band(trm):
    """Comfort temperature and 80 % limits.  T_rm is held at the validity limits of the model (10-33.5 °C) so the
    linear relation is never extrapolated; the share of days outside validity is reported separately."""
    tc = A.comfort_slope * np.clip(np.asarray(trm, float), *A.runmean_valid) + A.comfort_intercept
    return tc, tc - A.comfort_halfwidth, tc + A.comfort_halfwidth


def comfort_analysis(cd, mt: pd.DataFrame) -> pd.DataFrame:
    """Per-month comfort band and percentage of time cold / comfortable / hot (day and night)."""
    tab = pd.DataFrame(index=mt.index)
    basis = "monthly means only"
    if cd.daily is not None and "T2M" in cd.daily:
        d = cd.daily
        trm = running_mean(d["T2M"])
        tc, lo, hi = comfort_band(trm)
        d = d.assign(Trm=trm, Tc=tc, lo=lo, hi=hi)
        grp = d.groupby(d.index.month)
        tab["Trm"], tab["T_comfort"], tab["T_lower"], tab["T_upper"] = grp["Trm"].mean(), grp["Tc"].mean(), grp["lo"].mean(), grp["hi"].mean()
        tab["Trm_in_validity_%"] = grp["Trm"].apply(lambda s: 100 * s.between(*A.runmean_valid).mean())
        if cd.hourly is not None:
            h = cd.hourly
            day_idx = h.index.floor("D")
            lo_h, hi_h = d["lo"].reindex(day_idx).to_numpy(), d["hi"].reindex(day_idx).to_numpy()
            state = np.where(h["T2M"] > hi_h, "hot", np.where(h["T2M"] < lo_h, "cold", "comfort"))
            isday = (h.index.hour >= A.day_hours[0]) & (h.index.hour < A.day_hours[1])
            basis = "hourly T2M (day = 06-18 LST)"
        elif "T2M_MAX" in d:
            hi_d, lo_d = d["hi"], d["lo"]
            st_day = np.where(d["T2M_MAX"] > hi_d, "hot", np.where(d["T2M_MAX"] < lo_d, "cold", "comfort"))
            st_night = np.where(d["T2M_MIN"] > hi_d, "hot", np.where(d["T2M_MIN"] < lo_d, "cold", "comfort"))
            basis = "daily T2M_MAX (day) and T2M_MIN (night)"
        else:
            st_day = st_night = None
        for part in ("day", "night"):
            for s in ("hot", "comfort", "cold"):
                col = f"{part}_{s}_%"
                if cd.hourly is not None:
                    sel = isday if part == "day" else ~isday
                    ser = pd.Series((state == s)[sel], index=h.index[sel])
                    tab[col] = 100 * ser.groupby(ser.index.month).mean()
                elif st_day is not None:
                    arr = st_day if part == "day" else st_night
                    ser = pd.Series(arr == s, index=d.index)
                    tab[col] = 100 * ser.groupby(ser.index.month).mean()
    else:
        trm = mt["T_mean"]
        tc, lo, hi = comfort_band(trm)
        tab["Trm"], tab["T_comfort"], tab["T_lower"], tab["T_upper"] = trm, tc, lo, hi
        tab["Trm_in_validity_%"] = 100.0 * trm.between(*A.runmean_valid)
    tab.attrs["basis"] = basis
    return tab


# --------------------------------------------------------------------------- #
# [INTERP] passive strategy matrix
# --------------------------------------------------------------------------- #
STRATEGIES = ["Solar protection (shading)", "Thermal mass + night flush", "Evaporative cooling", "Daytime ventilation",
              "Night ventilation", "Passive solar heating", "Wind/dust protection", "Reflective roof + insulation"]


def strategy_matrix(cd, mt: pd.DataFrame, comfort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-by-month passive-design strategy flags (True = recommended) and the metrics behind them."""
    m = pd.DataFrame(index=mt.index)
    tday = mt["Tmax_mean"] if "Tmax_mean" in mt and mt["Tmax_mean"].notna().any() else mt["T_mean"]
    tnight = mt["Tmin_mean"] if "Tmin_mean" in mt and mt["Tmin_mean"].notna().any() else mt["T_mean"]
    upper, lower = comfort["T_upper"], comfort["T_lower"]
    m["T_day"], m["T_night"], m["T_upper"], m["T_lower"] = tday, tnight, upper, lower
    m["cooling_needed_day"] = tday > upper
    m["heating_needed_night"] = tnight < lower
    m["DTR"] = mt["DTR"] if "DTR" in mt else np.nan
    m["GHI"] = mt["GHI"] if "GHI" in mt else np.nan
    m["WB_depression"] = np.nan
    if "T_mean" in mt and "RH_mean" in mt:
        from .processing import wet_bulb_stull
        m["WB_depression"] = mt["T_mean"] - wet_bulb_stull(mt["T_mean"], mt["RH_mean"])
    # night-time cooling potential: share of nights whose minimum is below the comfort (target) temperature,
    # i.e. the night air can actually pull the structure below the temperature we want indoors
    if cd.daily is not None and "T2M_MIN" in cd.daily:
        d = cd.daily
        tc_d = comfort_band(running_mean(d["T2M"]))[0]
        m["night_below_comfort_%"] = 100 * pd.Series(d["T2M_MIN"].to_numpy() <= tc_d, index=d.index).groupby(d.index.month).mean()
    else:
        m["night_below_comfort_%"] = np.where(tnight <= comfort["T_comfort"], 100.0, 0.0)
    # daytime ventilation potential: hours (or days) in the comfort band with a usable breeze
    if cd.hourly is not None and "WS2M" in cd.hourly:
        h = cd.hourly
        up_h = comfort["T_upper"].reindex(h.index.month).to_numpy()
        lo_h = comfort["T_lower"].reindex(h.index.month).to_numpy()
        isday = (h.index.hour >= A.day_hours[0]) & (h.index.hour < A.day_hours[1])
        ok = (h["T2M"].between(lo_h, up_h + 1.0)) & (h["WS2M"] >= A.wind_min_vent) & isday
        m["day_vent_%"] = 100 * ok[isday].groupby(h.index[isday].month).mean()
    else:
        m["day_vent_%"] = np.where((tday <= upper + 1.0) & (tday >= lower), 100.0, 0.0)
    if cd.daily is not None and "WS2M" in cd.daily:
        m["WS_p90"] = cd.daily["WS2M"].groupby(cd.daily.index.month).quantile(0.9)
    else:
        m["WS_p90"] = mt["WS_mean"] if "WS_mean" in mt else np.nan

    s = pd.DataFrame(index=mt.index)
    s[STRATEGIES[0]] = (m["cooling_needed_day"] | (m["GHI"] >= A.ghi_high)).fillna(False)
    s[STRATEGIES[1]] = (m["cooling_needed_day"] & (m["DTR"] >= A.dtr_mass) & (m["night_below_comfort_%"] >= 50)).fillna(False)
    s[STRATEGIES[2]] = (m["cooling_needed_day"] & (m["WB_depression"] >= A.wb_depression_evap)).fillna(False)
    s[STRATEGIES[3]] = (m["day_vent_%"] >= 30).fillna(False)
    s[STRATEGIES[4]] = (m["cooling_needed_day"] & (m["night_below_comfort_%"] >= 50)).fillna(False)
    s[STRATEGIES[5]] = (m["heating_needed_night"] & (m["GHI"] >= A.ghi_heating_gain)).fillna(False)
    s[STRATEGIES[6]] = (m["WS_p90"] >= A.wind_dust).fillna(False)
    s[STRATEGIES[7]] = (m["GHI"] >= A.ghi_high).fillna(False)
    return s, m


# --------------------------------------------------------------------------- #
# [INTERP] recommendations
# --------------------------------------------------------------------------- #

def recommendations(cd, res: dict, comfort: pd.DataFrame, strat: pd.DataFrame, metrics: pd.DataFrame,
                    orient_sum: pd.DataFrame, shade: pd.DataFrame) -> list[dict]:
    """Evidence-based design statements.  Each cites the numbers on which it rests."""
    mt, ov = res["monthly"], res["overall"]
    out: list[dict] = []

    def add(topic, text, evidence, priority="High"):
        out.append({"topic": topic, "recommendation": text, "evidence": evidence, "priority": priority, "basis": "[INTERP]"})

    hot = month_ranges(list(metrics.index[metrics["cooling_needed_day"]]))
    cold = month_ranges(list(metrics.index[metrics["heating_needed_night"]]))
    ai, cls = res["precipitation"]["aridity_index"], res["precipitation"]["class"]
    add("Climate type",
        "Design for a hot-arid climate: avoid solar and heat gain in the long hot season, exploit large diurnal swings, "
        "and keep winter solar access for the cold nights.",
        f"Annual mean T = {ov['T_annual_mean']:.1f} °C; hottest month {MONTHS[ov['hottest_month'] - 1]} "
        f"({mt['T_mean'].max():.1f} °C); annual precipitation = {res['precipitation']['annual_mm'] or float('nan'):.0f} mm"
        + (f"; aridity index P/PET = {ai:.3f} ({cls})." if ai is not None else "."))

    if not orient_sum.empty:
        s = orient_sum["summer_mean"].sort_values(ascending=False)
        w = orient_sum["winter_mean"]
        add("Orientation & form",
            "Roof and west/east façades are the critical summer exposures; make the main façades face north/south "
            "(elongate the plan along the east-west axis), keep E/W walls blind or minimal, and use compact, courtyard "
            "or terraced forms that self-shade.",
            "Summer (JJA) irradiation, kWh/m²/day: " + ", ".join(f"{k} {v:.2f}" for k, v in s.items())
            + f". Winter south façade {w['S']:.2f} vs summer {orient_sum.loc['S', 'summer_mean']:.2f} (south is the only façade that gains more in winter).")
    nz = shade["noon_altitude_deg"]
    add("Shading",
        "Equator-facing openings: a fixed horizontal overhang is efficient because the summer noon sun is almost vertical, "
        "while allowing winter sun in. East/west openings cannot be shaded by overhangs (low sun) - use vertical fins, deep "
        "reveals, mashrabiya-type screens, or avoid openings. Shade roofs and courtyards (pergolas, vegetation, double roof).",
        f"Noon altitude: winter {nz.iloc[0]:.0f}°, equinox {nz.iloc[1]:.0f}°, summer {nz.iloc[2]:.0f}°. Overhang P/H for full noon shade "
        f"at summer solstice = {shade.iloc[2, 1]:.2f}; at winter solstice = {shade.iloc[0, 1]:.2f}.")
    dtr = ov["DTR_mean"]
    add("Envelope & thermal mass",
        "Use heavy, high-inertia envelopes (earth/toub, stone, thick hollow masonry) with exterior insulation or reflective, "
        "light-coloured finishes; avoid thin uninsulated concrete/lightweight roofs. Thermal lag of 8-12 h shifts the afternoon "
        "heat to the night, when ventilation can remove it.",
        (f"Mean diurnal range = {dtr:.1f} K; months with cooling needed in daytime: {hot}." if np.isfinite(dtr)
         else f"Months with cooling needed in daytime: {hot}. (Diurnal range requires daily/hourly resolution.)"))
    nv = month_ranges(list(strat.index[strat[STRATEGIES[4]]]))
    add("Night ventilation",
        "Provide secure, operable night-time openings (courtyard, roof-terrace, stack/wind-catcher) to flush stored heat; close "
        "and shade the building during the day.",
        f"Night flush effective in {nv}; share of nights with minimum below the comfort temperature (%): "
        + ", ".join(f"{MONTHS[k - 1]} {v:.0f}" for k, v in metrics["night_below_comfort_%"].items()) + ".", "High")
    ev = month_ranges(list(strat.index[strat[STRATEGIES[2]]]))
    add("Evaporative cooling",
        "Evaporative strategies (shaded planted courtyard, water basin, indirect evaporative wind-catcher) are effective in dry "
        "months but water is scarce: prefer indirect or passive systems and reuse/foggara-fed water.",
        f"Wet-bulb depression >= {A.wb_depression_evap:g} K with daytime cooling need in {ev}; mean RH = {ov['RH_mean']:.0f} %."
        if np.isfinite(ov["RH_mean"]) else f"Effective in {ev}.", "Medium")
    sh = month_ranges(list(strat.index[strat[STRATEGIES[5]]]))
    add("Winter heating",
        "Admit low winter sun through shaded-in-summer south openings onto massive floors/walls (direct gain); protect from "
        "cold night air with compact volumes and insulated roofs.",
        f"Cold nights (T_night below the comfort band) in {cold}; useful winter radiation in {sh} "
        f"(GHI >= {A.ghi_heating_gain:g} kWh/m²/day).", "Medium")
    wd = res["wind"]["summary"]
    if wd:
        add("Wind & dust",
            "Do NOT ventilate during hot afternoons; orient wind-catchers/night openings to the prevailing direction, and "
            "protect openings from dust-laden spring winds with porches, screens and compact street patterns.",
            f"Prevailing wind from {wd['prevailing_sector']} ({wd['prevailing_pct']:.1f} % of records, {res['wind']['basis']}); "
            f"mean speed {wd['mean_speed']:.1f} m/s, 95th percentile {wd['p95_speed']:.1f} m/s at 2 m.", "Medium")
    pr = res["precipitation"]
    if pr["annual_mm"] is not None:
        add("Rain & earthen heritage",
            "Rain is rare, but isolated storms can seriously damage earthen (toub) heritage walls: protect parapets and wall "
            "tops, keep plinths dry, and provide roof drainage/spouts and lime or clay renders for maintenance.",
            f"Annual precipitation = {pr['annual_mm']:.0f} mm"
            + (f"; maximum daily value = {pr['max_daily_mm']:.1f} mm." if pr['max_daily_mm'] is not None else "."), "Low")
    add("Heritage vs contemporary buildings",
        "Heritage ksar fabric (thick earthen walls, narrow shaded streets, courtyards, small high openings) matches the "
        "strategies above; contemporary interventions should reinterpret these principles rather than import lightweight "
        "glazed envelopes that require mechanical cooling.",
        "Qualitative synthesis of the calculated indicators above and published literature on Saharan vernacular architecture "
        "(not derived from POWER data).", "Medium")
    return out


# --------------------------------------------------------------------------- #
def run_bioclimatic(cd, res: dict) -> dict:
    mt = res["monthly"]
    orient, orient_sum, diffuse_assumed = orientation_table(cd.lat, mt)
    shade = shading_design(cd.lat)
    comfort = comfort_analysis(cd, mt)
    strat, metrics = strategy_matrix(cd, mt, comfort)
    recs = recommendations(cd, res, comfort, strat, metrics, orient_sum, shade)
    return {"orientation": orient, "orientation_summary": orient_sum, "diffuse_assumed": diffuse_assumed,
            "shading": shade, "comfort": comfort, "strategies": strat, "strategy_metrics": metrics, "recommendations": recs}
