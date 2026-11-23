"""
analysis.py - climate statistics  (every output is [NASA]-derived statistics or [CALC] indicators).

Entry point: run_analysis(ClimateData) -> dict of pandas objects.
Which analyses are possible depends on the temporal resolution that was retrieved:

    resolution     monthly/seasonal   distributions   diurnal profile   wind rose     thermal stress
    climatology    yes (12 months)    no              no                no            extremes only
    monthly        yes                monthly values  no                mean dir only extremes only
    daily          yes                yes             no                yes           yes
    hourly         yes                yes             yes               yes           yes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import processing as P
from . import solar

A = C.ASSUMPTIONS


@dataclass
class ClimateData:
    temporal: str
    lat: float
    lon: float
    meta: dict
    hourly: pd.DataFrame | None = None
    daily: pd.DataFrame | None = None        # includes derived indicators
    monthly: pd.DataFrame | None = None      # POWER monthly service
    climatology: pd.DataFrame | None = None  # POWER climatology service (index 1..12)
    notes: list = field(default_factory=list)

    @property
    def period(self) -> str:
        ref = self.hourly if self.hourly is not None else self.daily if self.daily is not None else self.monthly
        if ref is None:
            return "POWER climatology (multi-year monthly means)"
        return f"{ref.index.min():%Y-%m-%d} to {ref.index.max():%Y-%m-%d}"


def build_climate_data(result, lat: float, lon: float) -> ClimateData:
    """Create the analysis container from a PowerResult (see nasa_power_client)."""
    t, df = result.temporal, result.data
    cd = ClimateData(temporal=t, lat=lat, lon=lon, meta=result.meta)
    if t == "hourly":
        cd.hourly = df
        cd.daily = P.add_derived(P.hourly_to_daily(df), daily=True)
        cd.notes.append("Daily series aggregated from hourly data (T2M_MAX/MIN = extremes of hourly T2M).")
    elif t == "daily":
        cd.daily = P.add_derived(df, daily=True)
    elif t == "monthly":
        cd.monthly = df
        cd.notes.append("Monthly service: T2M_MAX / T2M_MIN are the *extreme* hourly values of each month, "
                        "not mean daily maxima/minima; mean diurnal range is therefore unavailable.")
    else:
        cd.climatology = df
        cd.notes.append("Climatology service: T2M_MAX / T2M_MIN are monthly extremes over the climatological record; "
                        "no distributions, diurnal profiles or wind roses can be derived.")
    return cd


# --------------------------------------------------------------------------- #
# monthly climatology table
# --------------------------------------------------------------------------- #
COLS = ["T_mean", "Tmax_mean", "Tmin_mean", "Tmax_abs", "Tmin_abs", "DTR", "Tdew", "RH_mean", "RH_min_daily", "QV2M",
        "GHI", "DIFF", "DIFFUSE_FRACTION", "LW_down", "WS_mean", "WD_prevailing", "P_month_mm", "wet_days",
        "HEAT_INDEX", "HUMIDEX"]


def monthly_table(cd: ClimateData) -> pd.DataFrame:
    """Twelve-row table (index 1..12) of long-term monthly statistics."""
    t = pd.DataFrame(index=pd.RangeIndex(1, 13, name="month"), columns=COLS, dtype=float)
    if cd.daily is not None:
        d = cd.daily
        g = d.groupby(d.index.month)
        m = lambda c, f="mean": getattr(g[c], f)() if c in d else np.nan
        t["T_mean"], t["Tdew"], t["RH_mean"], t["QV2M"] = m("T2M"), m("T2MDEW"), m("RH2M"), m("QV2M")
        t["Tmax_mean"], t["Tmin_mean"] = m("T2M_MAX"), m("T2M_MIN")
        t["Tmax_abs"], t["Tmin_abs"] = m("T2M_MAX", "max"), m("T2M_MIN", "min")
        t["DTR"], t["RH_min_daily"] = m("DTR"), m("RH2M", "min")
        t["GHI"], t["DIFF"], t["DIFFUSE_FRACTION"] = m("ALLSKY_SFC_SW_DWN"), m("ALLSKY_SFC_SW_DIFF"), m("DIFFUSE_FRACTION")
        t["LW_down"], t["WS_mean"] = m("ALLSKY_SFC_LW_DWN"), m("WS2M")
        t["HEAT_INDEX"], t["HUMIDEX"] = m("HEAT_INDEX"), m("HUMIDEX")
        if "WD2M" in d:
            t["WD_prevailing"] = [P.circular_mean_deg(d.loc[d.index.month == k, "WD2M"],
                                                      d.loc[d.index.month == k, "WS2M"] if "WS2M" in d else None)
                                  for k in range(1, 13)]
        if "PRECTOTCORR" in d:
            ym = d["PRECTOTCORR"].groupby([d.index.year, d.index.month])
            t["P_month_mm"] = ym.sum(min_count=15).groupby(level=1).mean()
            wet = (d["PRECTOTCORR"] >= A.wet_day_mm).groupby([d.index.year, d.index.month]).sum()
            t["wet_days"] = wet.groupby(level=1).mean()
    else:
        src = cd.monthly if cd.monthly is not None else cd.climatology
        mo = src.index.month if cd.monthly is not None else src.index
        g = src.groupby(mo)
        m = lambda c, f="mean": getattr(g[c], f)() if c in src else np.nan
        t["T_mean"], t["Tdew"], t["RH_mean"], t["QV2M"] = m("T2M"), m("T2MDEW"), m("RH2M"), m("QV2M")
        t["Tmax_abs"], t["Tmin_abs"] = m("T2M_MAX", "max"), m("T2M_MIN", "min")
        t["GHI"], t["DIFF"], t["LW_down"], t["WS_mean"] = m("ALLSKY_SFC_SW_DWN"), m("ALLSKY_SFC_SW_DIFF"), m("ALLSKY_SFC_LW_DWN"), m("WS2M")
        t["WD_prevailing"] = m("WD2M")
        t["DIFFUSE_FRACTION"] = t["DIFF"] / t["GHI"]
        if "PRECTOTCORR" in src:
            t["P_month_mm"] = m("PRECTOTCORR") * [solar.days_in_month_avg(k) for k in range(1, 13)]
        if {"T2M", "RH2M"} <= set(src.columns):
            t["HEAT_INDEX"] = P.heat_index(t["T_mean"], t["RH_mean"])
    return t.dropna(axis=1, how="all").reindex(columns=[c for c in COLS if c in t.dropna(axis=1, how="all")])


def annual_table(cd: ClimateData) -> pd.DataFrame | None:
    """Yearly statistics (daily/hourly) or POWER annual values (monthly)."""
    if cd.daily is not None:
        d = cd.daily
        g = d.groupby(d.index.year)
        a = pd.DataFrame({"T_mean": g["T2M"].mean()})
        if "T2M_MAX" in d:
            a["Tmax_abs"], a["Tmin_abs"] = g["T2M_MAX"].max(), g["T2M_MIN"].min()
            a["DTR"] = g["DTR"].mean()
        for k, c in {"RH_mean": "RH2M", "GHI_kWh_m2_day": "ALLSKY_SFC_SW_DWN", "WS_mean": "WS2M"}.items():
            if c in d:
                a[k] = g[c].mean()
        if "PRECTOTCORR" in d:
            a["P_year_mm"] = g["PRECTOTCORR"].sum(min_count=330)
        a["days_covered"] = g.size()
        a.index.name = "year"
        return a
    if cd.monthly is not None:
        mm = cd.monthly
        return mm.groupby(mm.index.year).mean().rename(columns={"T2M": "T_mean"})
    return None


def seasonal_table(mt: pd.DataFrame) -> pd.DataFrame:
    """Meteorological seasons (N-hemisphere: DJF/MAM/JJA/SON; swapped for lat<0 in plotting layer)."""
    rows = {}
    for name, months in C.SEASONS.items():
        sub = mt.loc[list(months)]
        r = sub.mean(numeric_only=True)
        for c in ("Tmax_abs", "HEAT_INDEX"):
            if c in sub:
                r[c] = sub[c].max()
        if "Tmin_abs" in sub:
            r["Tmin_abs"] = sub["Tmin_abs"].min()
        if "P_month_mm" in sub:
            r["P_season_mm"] = sub["P_month_mm"].sum()
        rows[name] = r
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# thermal stress
# --------------------------------------------------------------------------- #

def thermal_stress(cd: ClimateData) -> dict:
    """Return {'yearly': DataFrame, 'summary': DataFrame, 'hourly_summary': DataFrame|None}."""
    out = {"yearly": None, "summary": None, "hourly_summary": None}
    d = cd.daily
    if d is None:
        return out
    y = d.index.year
    rec = {}
    if "T2M_MAX" in d:
        for th in A.hot_day_thresholds:
            rec[f"days_Tmax>={th:g}C"] = (d["T2M_MAX"] >= th).groupby(y).sum()
        rec["days_Tmin<=0C(frost)"] = (d["T2M_MIN"] <= 0).groupby(y).sum()
        rec[f"tropical_nights_Tmin>={A.tropical_night:g}C"] = (d["T2M_MIN"] >= A.tropical_night).groupby(y).sum()
        rec[f"warm_nights_Tmin>={A.warm_night:g}C"] = (d["T2M_MIN"] >= A.warm_night).groupby(y).sum()
    if "HEAT_INDEX" in d:
        rec["days_HeatIndex(daily mean)>=32C"] = (d["HEAT_INDEX"] >= 32).groupby(y).sum()
    if "HUMIDEX" in d:
        rec["days_Humidex(daily mean)>=40C"] = (d["HUMIDEX"] >= 40).groupby(y).sum()
    if "CDD" in d:
        rec[f"CDD_base{A.cdd_base:g}"] = d["CDD"].groupby(y).sum()
        rec[f"HDD_base{A.hdd_base:g}"] = d["HDD"].groupby(y).sum()
    yearly = pd.DataFrame(rec)
    yearly.index.name = "year"
    full = d.groupby(y).size() >= 360          # use complete years only (partial years would bias the counts)
    if full.any():
        yearly = yearly[full[yearly.index]]
    out["yearly"] = yearly
    out["summary"] = pd.DataFrame({"mean_per_year": yearly.mean(), "max_year": yearly.max(), "min_year": yearly.min()})
    if cd.hourly is not None and {"T2M", "RH2M"} <= set(cd.hourly.columns):
        h = cd.hourly
        hi = P.heat_index(h["T2M"], h["RH2M"])
        yr = h.index.year
        out["hourly_summary"] = pd.DataFrame({
            "hours_T>=40C": (h["T2M"] >= 40).groupby(yr).sum(),
            "hours_T>=45C": (h["T2M"] >= 45).groupby(yr).sum(),
            "hours_HeatIndex>=41C(danger)": pd.Series(hi >= 41, index=h.index).groupby(yr).sum(),
        })
    return out


# --------------------------------------------------------------------------- #
# wind
# --------------------------------------------------------------------------- #
SPEED_BINS = [0, 2, 4, 6, 8, np.inf]
SPEED_LABELS = ["0-2", "2-4", "4-6", "6-8", ">8"]


def wind_rose(ws: pd.Series, wd: pd.Series) -> pd.DataFrame:
    """16-sector x 5-speed-class frequency table (%) of wind coming FROM each sector."""
    df = pd.DataFrame({"ws": ws, "wd": wd}).dropna()
    if df.empty:
        return pd.DataFrame()
    sector = (((df["wd"] + 11.25) % 360) // 22.5).astype(int)
    cls = pd.cut(df["ws"], SPEED_BINS, labels=SPEED_LABELS, right=False)
    tab = pd.crosstab(sector, cls, normalize="all") * 100
    tab = tab.reindex(index=range(16), columns=SPEED_LABELS, fill_value=0.0)
    tab.index = C.SECTORS16
    return tab


def wind_analysis(cd: ClimateData) -> dict:
    src = cd.hourly if cd.hourly is not None else cd.daily
    out = {"rose": None, "rose_by_season": {}, "basis": None, "summary": {}}
    if src is None or not {"WS2M", "WD2M"} <= set(src.columns):
        return out
    out["basis"] = "hourly" if cd.hourly is not None else "daily means"
    out["rose"] = wind_rose(src["WS2M"], src["WD2M"])
    for s, months in C.SEASONS.items():
        sub = src[src.index.month.isin(months)]
        out["rose_by_season"][s] = wind_rose(sub["WS2M"], sub["WD2M"])
    r = out["rose"].sum(axis=1)
    out["summary"] = {"prevailing_sector": r.idxmax(), "prevailing_pct": float(r.max()),
                      "resultant_direction_deg": P.circular_mean_deg(src["WD2M"], src["WS2M"]),
                      "mean_speed": float(src["WS2M"].mean()), "p95_speed": float(src["WS2M"].quantile(0.95)),
                      "calm_pct(<1m/s)": float((src["WS2M"] < 1).mean() * 100)}
    return out


# --------------------------------------------------------------------------- #
# precipitation and aridity
# --------------------------------------------------------------------------- #

def precipitation_analysis(cd: ClimateData, mt: pd.DataFrame) -> dict:
    out = {"annual_mm": None, "monthly_mm": mt["P_month_mm"] if "P_month_mm" in mt else None, "wet_days": None,
           "max_daily_mm": None, "de_martonne": None, "aridity_index": None, "pet_mm": None, "class": None}
    if out["monthly_mm"] is None:
        return out
    p_ann = float(out["monthly_mm"].sum())
    out["annual_mm"] = p_ann
    if cd.daily is not None and "PRECTOTCORR" in cd.daily:
        out["max_daily_mm"] = float(cd.daily["PRECTOTCORR"].max())
        out["wet_days"] = float(mt["wet_days"].sum())
    t_ann = float(mt["T_mean"].mean())
    out["de_martonne"] = p_ann / (t_ann + 10.0)
    # Hargreaves-Samani PET (needs mean daily Tmax/Tmin)
    if {"Tmax_mean", "Tmin_mean"} <= set(mt.columns):
        pet = []
        for k in range(1, 13):
            ra = solar.extraterrestrial_daily(cd.lat, solar.MID_MONTH_DOY[k - 1]) * 3.6 / 2.45  # kWh -> MJ -> mm/day
            dt = max(mt.loc[k, "Tmax_mean"] - mt.loc[k, "Tmin_mean"], 0)
            pet.append(0.0023 * ra * (mt.loc[k, "T_mean"] + 17.8) * np.sqrt(dt) * solar.days_in_month_avg(k))
        out["pet_mm"] = pd.Series(pet, index=mt.index)
        ai = p_ann / out["pet_mm"].sum()
        out["aridity_index"] = ai
        out["class"] = ("Hyper-arid" if ai < 0.05 else "Arid" if ai < 0.2 else "Semi-arid" if ai < 0.5
                        else "Dry sub-humid" if ai < 0.65 else "Humid")
    return out


# --------------------------------------------------------------------------- #
# diurnal profiles (hourly only)
# --------------------------------------------------------------------------- #

def diurnal_profiles(cd: ClimateData) -> dict:
    h = cd.hourly
    if h is None:
        return {}
    out = {}
    for c in ("T2M", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M", "T2MDEW"):
        if c in h:
            out[c] = h[c].groupby([h.index.hour, h.index.month]).mean().unstack()
            out[c].index.name = "hour"
    season = {}
    for s, months in C.SEASONS.items():
        sub = h[h.index.month.isin(months)]
        season[s] = sub.groupby(sub.index.hour).mean(numeric_only=True)
    out["by_season"] = season
    return out


# --------------------------------------------------------------------------- #
# master function
# --------------------------------------------------------------------------- #

def run_analysis(cd: ClimateData) -> dict:
    mt = monthly_table(cd)
    res = {"monthly": mt, "annual": annual_table(cd), "seasonal": seasonal_table(mt),
           "thermal_stress": thermal_stress(cd), "wind": wind_analysis(cd),
           "precipitation": precipitation_analysis(cd, mt), "diurnal": diurnal_profiles(cd)}
    res["overall"] = {
        "T_annual_mean": float(mt["T_mean"].mean()),
        "hottest_month": int(mt["T_mean"].idxmax()), "coldest_month": int(mt["T_mean"].idxmin()),
        "Tmax_abs": float(mt["Tmax_abs"].max()) if "Tmax_abs" in mt else np.nan,
        "Tmin_abs": float(mt["Tmin_abs"].min()) if "Tmin_abs" in mt else np.nan,
        "DTR_mean": float(mt["DTR"].mean()) if "DTR" in mt else np.nan,
        "RH_mean": float(mt["RH_mean"].mean()) if "RH_mean" in mt else np.nan,
        "GHI_annual_kWh_m2": float((mt["GHI"] * [solar.days_in_month_avg(k) for k in range(1, 13)]).sum()) if "GHI" in mt else np.nan,
        "GHI_mean_kWh_m2_day": float(mt["GHI"].mean()) if "GHI" in mt else np.nan,
    }
    return res
