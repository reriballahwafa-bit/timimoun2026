"""
processing.py - cleaning, validation, aggregation, derived indicators and export.

Convention used throughout the project (and in the report):
    [NASA]  variable retrieved from NASA POWER (MERRA-2 re-analysis / CERES-SYN1deg satellite-model product)
    [CALC]  indicator calculated here from [NASA] variables (formula documented in the docstring)
    [INTERP] architectural interpretation (rule-based, thresholds in config.Assumptions)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------- #
# psychrometrics / comfort indices  [CALC]
# --------------------------------------------------------------------------- #

def dew_point(t, rh):
    """Magnus formula (Alduchov & Eskridge 1996), °C."""
    a, b = 17.625, 243.04
    g = np.log(np.clip(rh, 0.1, 100) / 100.0) + a * t / (b + t)
    return b * g / (a - g)


def wet_bulb_stull(t, rh):
    """Wet-bulb temperature (°C) - Stull (2011) empirical fit, valid 5-99 % RH, -20..50 °C."""
    rh = np.clip(rh, 5, 99)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def heat_index(t, rh):
    """NWS heat index (Rothfusz regression + Steadman simple form + NWS adjustments), returns °C."""
    t = np.asarray(t, float)
    rh = np.asarray(rh, float)
    tf = t * 9 / 5 + 32
    simple = 0.5 * (tf + 61.0 + (tf - 68.0) * 1.2 + rh * 0.094)
    hi = (-42.379 + 2.04901523 * tf + 10.14333127 * rh - 0.22475541 * tf * rh - 6.83783e-3 * tf ** 2
          - 5.481717e-2 * rh ** 2 + 1.22874e-3 * tf ** 2 * rh + 8.5282e-4 * tf * rh ** 2 - 1.99e-6 * tf ** 2 * rh ** 2)
    low_rh = (rh < 13) & (tf >= 80) & (tf <= 112)
    hi = np.where(low_rh, hi - ((13 - rh) / 4) * np.sqrt(np.abs(17 - np.abs(tf - 95)) / 17), hi)
    hi = np.where((rh > 85) & (tf >= 80) & (tf <= 87), hi + ((rh - 85) / 10) * ((87 - tf) / 5), hi)
    out = np.where(simple >= 80, hi, simple)
    return (out - 32) * 5 / 9


def humidex(t, td):
    """Canadian humidex (Masterton & Richardson 1979) from air and dew-point temperature, °C-equivalent."""
    e = 6.11 * np.exp(5417.7530 * (1 / 273.16 - 1 / (273.15 + td)))
    return t + 0.5555 * (e - 10.0)


def circular_mean_deg(direction, weights=None):
    """Vector-mean direction (degrees from which the wind blows); weights = wind speed -> resultant wind."""
    d = np.deg2rad(np.asarray(direction, float))
    w = np.ones_like(d) if weights is None else np.asarray(weights, float)
    m = ~(np.isnan(d) | np.isnan(w))
    if not m.any():
        return np.nan
    return float(np.rad2deg(np.arctan2((w[m] * np.sin(d[m])).sum(), (w[m] * np.cos(d[m])).sum())) % 360)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
FREQ = {"hourly": "h", "daily": "D", "monthly": "MS"}


def validate(df: pd.DataFrame, temporal: str, meta: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Completeness, range and time-continuity checks.  Returns (table, list of human readable issues)."""
    meta = meta or {}
    rows, issues = [], []
    for col in df.columns:
        s = df[col]
        lo, hi = C.VALID_RANGES.get(col, (-np.inf, np.inf))
        out = int(((s < lo) | (s > hi)).sum())
        rows.append({"parameter": col, "unit": (meta.get("units") or {}).get(col, ""), "n": int(s.notna().sum()),
                     "missing": int(s.isna().sum()), "completeness_%": round(100 * s.notna().mean(), 2),
                     "min": s.min(), "mean": s.mean(), "max": s.max(), "out_of_range": out})
        if out:
            issues.append(f"{col}: {out} values outside the plausible range [{lo}, {hi}].")
        if s.notna().mean() < 0.95:
            issues.append(f"{col}: only {100 * s.notna().mean():.1f} % complete.")
    if temporal in FREQ and len(df) > 1:
        full = pd.date_range(df.index.min(), df.index.max(), freq=FREQ[temporal])
        gaps = len(full.difference(df.index))
        if gaps:
            issues.append(f"{gaps} timestamps missing from the continuous {temporal} series.")
        if df.index.has_duplicates:
            issues.append("Duplicated timestamps detected.")
    return pd.DataFrame(rows), issues


# --------------------------------------------------------------------------- #
# aggregation hourly -> daily   [CALC]
# --------------------------------------------------------------------------- #
SUM_KWH = ("ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN", "ALLSKY_SFC_SW_DIFF", "ALLSKY_SFC_SW_DNI", "CLRSKY_SFC_SW_DWN")


def hourly_to_daily(h: pd.DataFrame) -> pd.DataFrame:
    """Daily statistics from hourly POWER data.
    * means for T2M, T2MDEW, RH2M, QV2M, WS*, PS ...   * T2M_MAX / T2M_MIN = daily extremes of hourly T2M
    * radiation: Wh/m² per hour summed -> kWh/m²/day     * precipitation: mm/hour summed -> mm/day
    * wind direction: speed-weighted vector mean.  Days with < 20 valid hours are set to NaN."""
    g = h.groupby(h.index.floor("D"))
    out = {}
    nvalid = g.count()
    for col in h.columns:
        if col in SUM_KWH:
            out[col] = g[col].sum(min_count=24) / 1000.0
        elif col == "PRECTOTCORR":
            out[col] = g[col].sum(min_count=24)
        elif col.startswith("WD"):
            ws = "WS" + col[2:]
            out[col] = g.apply(lambda x, c=col, w=ws: circular_mean_deg(x[c], x[w] if w in x else None))
        else:
            out[col] = g[col].mean().where(nvalid[col] >= 20)
    if "T2M" in h:
        out["T2M_MAX"] = g["T2M"].max().where(nvalid["T2M"] >= 20)
        out["T2M_MIN"] = g["T2M"].min().where(nvalid["T2M"] >= 20)
    d = pd.DataFrame(out)
    d.index.name = "time"
    return d


# --------------------------------------------------------------------------- #
# derived variables [CALC]
# --------------------------------------------------------------------------- #

def add_derived(d: pd.DataFrame, a: C.Assumptions = C.ASSUMPTIONS, daily: bool = True) -> pd.DataFrame:
    """Append calculated indicators to a daily (daily=True) or hourly frame."""
    d = d.copy()
    if {"T2M_MAX", "T2M_MIN"} <= set(d.columns):
        d["DTR"] = d["T2M_MAX"] - d["T2M_MIN"]
    if {"T2M", "RH2M"} <= set(d.columns):
        d["T_WETBULB_CALC"] = wet_bulb_stull(d["T2M"], d["RH2M"])
        d["HEAT_INDEX"] = heat_index(d["T2M"], d["RH2M"])
        td = d["T2MDEW"] if "T2MDEW" in d else dew_point(d["T2M"], d["RH2M"])
        d["HUMIDEX"] = humidex(d["T2M"], td)
    if "T2M" in d and daily:
        d["HDD"] = (a.hdd_base - d["T2M"]).clip(lower=0)
        d["CDD"] = (d["T2M"] - a.cdd_base).clip(lower=0)
    if {"ALLSKY_SFC_SW_DIFF", "ALLSKY_SFC_SW_DWN"} <= set(d.columns):
        d["DIFFUSE_FRACTION"] = (d["ALLSKY_SFC_SW_DIFF"] / d["ALLSKY_SFC_SW_DWN"]).where(d["ALLSKY_SFC_SW_DWN"] > 0.05)
    return d


# --------------------------------------------------------------------------- #
# xarray / export
# --------------------------------------------------------------------------- #

def to_xarray(df: pd.DataFrame, meta: dict, lat: float, lon: float, title: str = ""):
    """Convert a time-indexed frame to an xarray Dataset with CF-style attributes."""
    import xarray as xr
    ds = xr.Dataset({c: (df.index.name or "time", df[c].to_numpy(dtype="float32"),
                         {"units": (meta.get("units") or {}).get(c, ""), "long_name": (meta.get("long_names") or {}).get(c) or c})
                     for c in df.columns}, coords={df.index.name or "time": df.index})
    ds.attrs.update(title=title or "NASA POWER climate data", source="NASA POWER (https://power.larc.nasa.gov)",
                    latitude=lat, longitude=lon, elevation_m=meta.get("elevation_m") or np.nan,
                    temporal=meta.get("temporal", ""), time_standard=str(meta.get("time_standard")),
                    history="Processed with timimoun_climate", conventions="CF-1.8")
    return ds


def export_dataset(df: pd.DataFrame, meta: dict, out_dir: Path, stem: str, lat: float, lon: float,
                   formats=("csv", "xlsx", "nc")) -> list[Path]:
    """Write processed data to CSV / Excel / NetCDF.  Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if "csv" in formats:
        p = out_dir / f"{stem}.csv"
        df.to_csv(p, float_format="%.4f")
        written.append(p)
    if "xlsx" in formats and len(df) < 1_000_000:
        p = out_dir / f"{stem}.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="data")
            pd.DataFrame({"parameter": list(meta.get("units", {})), "unit": list(meta.get("units", {}).values()),
                          "long_name": [meta.get("long_names", {}).get(k) for k in meta.get("units", {})]}
                         ).to_excel(xw, sheet_name="units", index=False)
        written.append(p)
    if "nc" in formats:
        p = out_dir / f"{stem}.nc"
        ds = to_xarray(df, meta, lat, lon, stem)
        try:
            ds.to_netcdf(p, engine="netcdf4")
        except Exception:
            ds.to_netcdf(p, engine="h5netcdf")
        written.append(p)
    return written
