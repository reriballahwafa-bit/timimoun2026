"""
visualization.py - publication-style figures (Matplotlib) and an interactive dashboard (Plotly).

Every figure function accepts the objects produced by analysis.run_analysis / bioclimatic.run_bioclimatic,
returns a matplotlib Figure (or None when the selected temporal resolution cannot support it)
and `save_all()` writes PNG files for the report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config as C
from . import solar
from .analysis import SPEED_LABELS

MON = C.MONTHS
PAL = {"mean": "#c2410c", "max": "#dc2626", "min": "#2563eb", "ghi": "#f59e0b", "diff": "#0ea5e9", "rh": "#0891b2",
       "wind": "#475569", "rain": "#1d4ed8", "comfort": "#16a34a", "grid": "#e5e7eb"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": PAL["grid"], "grid.linewidth": 0.6, "figure.dpi": 110, "savefig.dpi": 200,
                     "axes.titleweight": "bold", "axes.titlesize": 10})


def _title(ax, t, sub=None):
    ax.set_title(t, loc="left", pad=15 if sub else 6)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7, color="#6b7280", va="bottom")


def _xmonth(ax):
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MON)


def _src(fig, cd):
    fig.text(0.995, 0.005, f"Source: NASA POWER ({cd.temporal}) | {cd.lat:.3f}°, {cd.lon:.3f}° | {cd.period}",
             ha="right", va="bottom", fontsize=6, color="#6b7280")


# ------------------------------------------------------------------ individual figures
def fig_monthly_temperature(cd, res):
    mt = res["monthly"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if "Tmax_mean" in mt:
        ax.fill_between(mt.index, mt["Tmin_mean"], mt["Tmax_mean"], color=PAL["mean"], alpha=0.15, label="Mean daily range")
        ax.plot(mt.index, mt["Tmax_mean"], color=PAL["max"], marker="o", label="Mean daily maximum")
        ax.plot(mt.index, mt["Tmin_mean"], color=PAL["min"], marker="o", label="Mean daily minimum")
    ax.plot(mt.index, mt["T_mean"], color=PAL["mean"], lw=2.4, marker="o", label="Mean temperature (T2M)")
    if "Tmax_abs" in mt:
        ax.plot(mt.index, mt["Tmax_abs"], color=PAL["max"], ls="--", lw=1, label="Absolute maximum")
        ax.plot(mt.index, mt["Tmin_abs"], color=PAL["min"], ls="--", lw=1, label="Absolute minimum")
    _xmonth(ax)
    ax.set_ylabel("°C")
    _title(ax, "Monthly temperature regime", "Measured series (NASA POWER); statistics computed from the selected period")
    ax.legend(ncol=3, fontsize=7, loc="upper left", frameon=False)
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_temperature_distribution(cd, res):
    src = cd.hourly["T2M"] if cd.hourly is not None else cd.daily["T2M"] if cd.daily is not None else None
    if src is None:
        return None
    from scipy.stats import gaussian_kde
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    series = {"T2M (all values)": (src.dropna(), PAL["mean"])}
    if cd.daily is not None and "T2M_MAX" in cd.daily:
        series["Daily max"] = (cd.daily["T2M_MAX"].dropna(), PAL["max"])
        series["Daily min"] = (cd.daily["T2M_MIN"].dropna(), PAL["min"])
    for lab, (s, col) in series.items():
        x = np.linspace(s.min() - 3, s.max() + 3, 300)
        axs[0].plot(x, gaussian_kde(s.sample(min(len(s), 6000), random_state=1))(x), color=col, label=lab)
        axs[0].fill_between(x, 0, gaussian_kde(s.sample(min(len(s), 6000), random_state=1))(x), color=col, alpha=0.12)
    axs[0].set_xlabel("°C")
    axs[0].set_ylabel("Density")
    axs[0].legend(frameon=False, fontsize=7)
    _title(axs[0], "Temperature distribution (kernel density)")
    for s, months in C.SEASONS.items():
        axs[1].hist(src[src.index.month.isin(months)].dropna(), bins=30, histtype="step", density=True, lw=1.6, label=s)
    axs[1].set_xlabel("°C")
    axs[1].legend(frameon=False, fontsize=7)
    _title(axs[1], "Seasonal temperature histograms")
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_solar(cd, res, bio):
    mt = res["monthly"]
    if "GHI" not in mt:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax = axs[0]
    diff = mt["DIFF"] if "DIFF" in mt else pd.Series(0, index=mt.index)
    ax.bar(mt.index, mt["GHI"] - diff, color=PAL["ghi"], label="Direct (horizontal) = GHI - DIFF [calc]")
    ax.bar(mt.index, diff, bottom=mt["GHI"] - diff, color=PAL["diff"], label="Diffuse (ALLSKY_SFC_SW_DIFF)")
    _xmonth(ax)
    ax.set_ylabel("kWh/m²/day")
    ax.legend(frameon=False, fontsize=7, loc="lower center")
    _title(ax, "Global horizontal irradiation", "ALLSKY_SFC_SW_DWN")
    ori = bio["orientation"]
    ax = axs[1]
    if not ori.empty:
        cols = {"Roof": "#111827", "N": "#2563eb", "E": "#16a34a", "S": "#dc2626", "W": "#9333ea", "NE": "#93c5fd", "SE": "#fca5a5"}
        for c in ["Roof", "N", "E", "S", "W"]:
            ax.plot(ori.index, ori[c], marker="o", ms=3, color=cols[c], label=c)
        _xmonth(ax)
        ax.set_ylabel("kWh/m²/day")
        ax.legend(frameon=False, ncol=5, fontsize=7)
        _title(ax, "Solar exposure of building surfaces", "Calculated: isotropic-sky transposition, albedo %.2f" % C.ASSUMPTIONS.albedo)
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_humidity(cd, res):
    mt = res["monthly"]
    if "RH_mean" not in mt:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axs[0]
    if cd.daily is not None and "RH2M" in cd.daily:
        d = cd.daily
        data = [d.loc[d.index.month == k, "RH2M"].dropna() for k in range(1, 13)]
        ax.boxplot(data, positions=range(1, 13), widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#cffafe", color=PAL["rh"]), medianprops=dict(color="#0e7490"))
    ax.plot(mt.index, mt["RH_mean"], color=PAL["rh"], marker="o", label="Monthly mean RH2M")
    _xmonth(ax)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7)
    _title(ax, "Relative humidity", "Boxes: distribution of daily means")
    src = cd.hourly["RH2M"] if cd.hourly is not None else cd.daily["RH2M"] if cd.daily is not None else None
    if src is not None:
        axs[1].hist(src.dropna(), bins=40, color=PAL["rh"], alpha=0.8, weights=np.ones(src.notna().sum()) * 100 / src.notna().sum())
        axs[1].set_xlabel("RH2M (%)")
        axs[1].set_ylabel("Frequency (%)")
        _title(axs[1], "RH distribution")
    else:
        axs[1].axis("off")
    _src(fig, cd)
    fig.tight_layout()
    return fig


def _rose(ax, tab, title=None):
    theta = np.deg2rad(np.arange(16) * 22.5)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bottom = np.zeros(16)
    cols = plt.cm.YlOrRd(np.linspace(0.25, 0.95, len(SPEED_LABELS)))
    for lab, col in zip(SPEED_LABELS, cols):
        ax.bar(theta, tab[lab].to_numpy(), width=np.deg2rad(20), bottom=bottom, color=col, edgecolor="white", lw=0.3, label=f"{lab} m/s")
        bottom += tab[lab].to_numpy()
    ax.set_xticks(theta[::2])
    ax.set_xticklabels(C.SECTORS16[::2], fontsize=7)
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title, fontsize=9, pad=12)


def fig_wind_rose(cd, res):
    w = res["wind"]
    if w["rose"] is None or w["rose"].empty:
        return None
    fig = plt.figure(figsize=(9.5, 5.2))
    ax = fig.add_axes([0.02, 0.1, 0.4, 0.8], projection="polar")
    _rose(ax, w["rose"], f"Annual wind rose ({w['basis']}), WS2M / WD2M")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.12), ncol=5, fontsize=6, frameon=False)
    for i, (s, tab) in enumerate(w["rose_by_season"].items()):
        a = fig.add_axes([0.48 + (i % 2) * 0.27, 0.55 - (i // 2) * 0.46, 0.22, 0.32], projection="polar")
        if not tab.empty:
            _rose(a, tab, s)
    fig.text(0.46, 0.96, "Direction = where the wind comes FROM; % of all records", fontsize=7, color="#6b7280")
    _src(fig, cd)
    return fig


def fig_precipitation(cd, res):
    pr = res["precipitation"]
    if pr["monthly_mm"] is None:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axs[0]
    ax.bar(pr["monthly_mm"].index, pr["monthly_mm"], color=PAL["rain"], label="Precipitation (PRECTOTCORR)")
    if pr["pet_mm"] is not None:
        ax.plot(pr["pet_mm"].index, pr["pet_mm"], color=PAL["max"], marker="o", ms=3, label="PET Hargreaves [calc]")
        ax.set_yscale("symlog", linthresh=1)
    _xmonth(ax)
    ax.set_ylabel("mm / month")
    ax.legend(frameon=False, fontsize=7)
    _title(ax, "Precipitation vs. potential evapotranspiration", f"Annual P = {pr['annual_mm']:.1f} mm" +
           (f" | P/PET = {pr['aridity_index']:.3f} ({pr['class']})" if pr["aridity_index"] is not None else ""))
    a = res["annual"]
    if a is not None and "P_year_mm" in a:
        axs[1].bar(a.index.astype(str), a["P_year_mm"], color=PAL["rain"])
        axs[1].set_ylabel("mm / year")
        axs[1].tick_params(axis="x", rotation=60)
        _title(axs[1], "Annual totals")
    else:
        axs[1].axis("off")
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_diurnal(cd, res):
    dn = res["diurnal"]
    if not dn:
        return None
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    cols = dict(zip(C.SEASONS, ["#2563eb", "#16a34a", "#dc2626", "#d97706"]))
    for ax, (c, lab) in zip(axs, [("T2M", "°C"), ("RH2M", "%"), ("ALLSKY_SFC_SW_DWN", "W/m²")]):
        for s, df in dn["by_season"].items():
            if c in df:
                ax.plot(df.index, df[c], color=cols[s], label=s)
        ax.set_xlabel("Local solar hour")
        ax.set_ylabel(lab)
        ax.set_xticks(range(0, 24, 3))
        _title(ax, {"T2M": "Diurnal temperature", "RH2M": "Diurnal relative humidity", "ALLSKY_SFC_SW_DWN": "Diurnal solar irradiance"}[c])
    axs[0].legend(frameon=False, fontsize=7)
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_heatmap(cd, res):
    t = res["diurnal"].get("T2M") if res["diurnal"] else None
    if t is None:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cmap = LinearSegmentedColormap.from_list("t", ["#1e3a8a", "#38bdf8", "#fef08a", "#f97316", "#7f1d1d"])
    im = ax.pcolormesh(range(1, 14), range(0, 25), t.to_numpy(), cmap=cmap, shading="flat")
    ax.set_xticks(np.arange(1, 13) + 0.5)
    ax.set_xticklabels(MON)
    ax.set_ylabel("Local solar hour")
    ax.grid(False)
    fig.colorbar(im, label="°C")
    _title(ax, "Mean temperature by hour and month (thermal 'carpet')")
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_seasonal(cd, res):
    if cd.daily is None:
        return None
    d = cd.daily
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    names = list(C.SEASONS)
    for ax, (c, lab, ttl) in zip(axs, [("T2M", "°C", "Daily mean temperature"), ("RH2M", "%", "Relative humidity"),
                                       ("ALLSKY_SFC_SW_DWN", "kWh/m²/day", "Solar irradiation")]):
        if c not in d:
            ax.axis("off")
            continue
        data = [d.loc[d.index.month.isin(m), c].dropna() for m in C.SEASONS.values()]
        bp = ax.boxplot(data, tick_labels=[n.split()[0] for n in names], patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], ["#bfdbfe", "#bbf7d0", "#fecaca", "#fde68a"]):
            patch.set_facecolor(col)
        ax.set_ylabel(lab)
        _title(ax, ttl)
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_comfort(cd, res, bio):
    cf, mt = bio["comfort"], res["monthly"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axs[0]
    ax.fill_between(cf.index, cf["T_lower"], cf["T_upper"], color=PAL["comfort"], alpha=0.25, label="Adaptive comfort band (80 %)")
    ax.plot(cf.index, cf["T_comfort"], color=PAL["comfort"], lw=1)
    if "Tmax_mean" in mt:
        ax.plot(mt.index, mt["Tmax_mean"], color=PAL["max"], marker="o", ms=3, label="Mean daily max")
        ax.plot(mt.index, mt["Tmin_mean"], color=PAL["min"], marker="o", ms=3, label="Mean daily min")
    ax.plot(mt.index, mt["T_mean"], color=PAL["mean"], marker="o", ms=3, label="Mean T")
    _xmonth(ax)
    ax.set_ylabel("°C")
    ax.legend(frameon=False, fontsize=7)
    _title(ax, "Adaptive comfort (ASHRAE 55) vs. outdoor temperature", "Calculated; T_rm clamped to 10-33.5 °C validity range")
    ax = axs[1]
    if "day_hot_%" in cf and cf["day_hot_%"].notna().any():
        x = np.arange(12)
        for off, part in [(-0.2, "day"), (0.2, "night")]:
            b = np.zeros(12)
            for s, col in [("cold", "#60a5fa"), ("comfort", "#4ade80"), ("hot", "#f87171")]:
                v = cf[f"{part}_{s}_%"].to_numpy()
                ax.bar(x + off, v, 0.38, bottom=b, color=col, label=s if part == "day" else None)
                b += v
        ax.set_xticks(x)
        ax.set_xticklabels(MON)
        ax.set_ylabel("% of time")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center")
        _title(ax, "Thermal state: day (left bar) / night (right bar)", cf.attrs.get("basis", ""))
    else:
        ax.axis("off")
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_strategies(cd, res, bio):
    s = bio["strategies"].astype(float)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.imshow(s.T.to_numpy(), cmap=LinearSegmentedColormap.from_list("s", ["#f3f4f6", "#16a34a"]), aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MON)
    ax.set_yticks(range(len(s.columns)))
    ax.set_yticklabels(s.columns)
    ax.grid(False)
    _title(ax, "Passive design strategy matrix (rule-based interpretation)", "Green = recommended; thresholds in config.Assumptions")
    _src(fig, cd)
    fig.tight_layout()
    return fig


def fig_sun_path(cd):
    fig = plt.figure(figsize=(5.8, 5.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    hrs = np.arange(4, 20.01, 0.1)
    for name, n, col in [("21 Dec", 355, "#2563eb"), ("21 Mar / 23 Sep", 80, "#16a34a"), ("21 Jun", 172, "#dc2626")]:
        _, alt, az = solar.sun_position(cd.lat, n, hrs)
        m = alt > 0
        ax.plot(np.deg2rad(az[m]), 90 - alt[m], color=col, label=name)
    ax.set_rmax(90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(["90°", "60°", "30°", "0°"], fontsize=6)
    ax.set_title(f"Sun path at {cd.lat:.2f}°N (altitude rings)", fontsize=9, pad=14)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), fontsize=7, frameon=False)
    return fig


def fig_dashboard(cd, res, bio):
    """Single-page bioclimatic dashboard."""
    mt = res["monthly"]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.75)
    ax = fig.add_subplot(gs[0, 0])
    if "Tmax_mean" in mt:
        ax.fill_between(mt.index, mt["Tmin_mean"], mt["Tmax_mean"], color=PAL["mean"], alpha=0.2)
    ax.plot(mt.index, mt["T_mean"], color=PAL["mean"], marker="o", ms=3)
    cf = bio["comfort"]
    ax.fill_between(cf.index, cf["T_lower"], cf["T_upper"], color=PAL["comfort"], alpha=0.25)
    _xmonth(ax)
    ax.tick_params(axis="x", labelsize=6)
    _title(ax, "Temperature vs comfort band (°C)")
    ax = fig.add_subplot(gs[0, 1])
    if "GHI" in mt:
        ax.bar(mt.index, mt["GHI"], color=PAL["ghi"])
    _xmonth(ax)
    ax.tick_params(axis="x", labelsize=6)
    _title(ax, "Solar irradiation (kWh/m²/day)")
    ax = fig.add_subplot(gs[0, 2])
    if "RH_mean" in mt:
        ax.plot(mt.index, mt["RH_mean"], color=PAL["rh"], marker="o", ms=3)
    ax.set_ylim(0, 100)
    _xmonth(ax)
    ax.tick_params(axis="x", labelsize=6)
    _title(ax, "Relative humidity (%)")
    ori = bio["orientation_summary"]
    ax = fig.add_subplot(gs[1, 0])
    if not ori.empty:
        x = np.arange(len(ori))
        ax.bar(x - 0.2, ori["summer_mean"], 0.4, color="#dc2626", label="Summer")
        ax.bar(x + 0.2, ori["winter_mean"], 0.4, color="#2563eb", label="Winter")
        ax.set_xticks(x)
        ax.set_xticklabels(ori.index, fontsize=7)
        ax.legend(frameon=False, fontsize=7)
    _title(ax, "Surface exposure (kWh/m²/day)")
    w = res["wind"]
    if w["rose"] is not None:
        _rose(fig.add_subplot(gs[1, 1], projection="polar"), w["rose"], "Wind rose (WS2M)")
    ax = fig.add_subplot(gs[1, 2])
    s = bio["strategies"].astype(float)
    ax.imshow(s.T.to_numpy(), cmap=LinearSegmentedColormap.from_list("s", ["#f3f4f6", "#16a34a"]), aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels([m[0] for m in MON], fontsize=6)
    ax.set_yticks(range(len(s.columns)))
    ax.set_yticklabels(s.columns, fontsize=6)
    ax.grid(False)
    _title(ax, "Passive strategies")
    fig.suptitle(f"Bioclimatic dashboard - {cd.lat:.3f}°N, {cd.lon:.3f}°E", fontweight="bold", x=0.01, ha="left")
    _src(fig, cd)
    return fig


# ------------------------------------------------------------------ batch
def build_figures(cd, res, bio) -> dict:
    figs = {"01_monthly_temperature": fig_monthly_temperature(cd, res),
            "02_temperature_distribution": fig_temperature_distribution(cd, res),
            "03_solar_radiation": fig_solar(cd, res, bio),
            "04_relative_humidity": fig_humidity(cd, res),
            "05_wind_rose": fig_wind_rose(cd, res),
            "06_precipitation": fig_precipitation(cd, res),
            "07_diurnal_profiles": fig_diurnal(cd, res),
            "08_thermal_carpet": fig_heatmap(cd, res),
            "09_seasonal_comparison": fig_seasonal(cd, res),
            "10_adaptive_comfort": fig_comfort(cd, res, bio),
            "11_strategy_matrix": fig_strategies(cd, res, bio),
            "12_sun_path": fig_sun_path(cd),
            "13_bioclimatic_dashboard": fig_dashboard(cd, res, bio)}
    return {k: v for k, v in figs.items() if v is not None}


def save_all(figs: dict, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fig in figs.items():
        p = out_dir / f"{name}.png"
        fig.savefig(p, bbox_inches="tight")
        paths[name] = p
    return paths


def plotly_dashboard(cd, res, bio, out_html: str | Path) -> Path | None:
    """Interactive Plotly dashboard (HTML) - monthly temperature, thermal carpet, wind rose, façade exposure."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None
    mt = res["monthly"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Monthly temperature (°C)", "Mean temperature by hour × month (°C)",
                                                        "Wind rose (WS2M, % of records)", "Façade/roof irradiation (kWh/m²/day)"),
                        specs=[[{}, {}], [{"type": "polar"}, {}]])
    x = MON
    if "Tmax_mean" in mt:
        fig.add_scatter(x=x, y=mt["Tmax_mean"], name="Mean max", line=dict(color="#dc2626"), row=1, col=1)
        fig.add_scatter(x=x, y=mt["Tmin_mean"], name="Mean min", line=dict(color="#2563eb"), fill="tonexty", row=1, col=1)
    fig.add_scatter(x=x, y=mt["T_mean"], name="Mean T2M", line=dict(color="#c2410c", width=3), row=1, col=1)
    t = res["diurnal"].get("T2M") if res["diurnal"] else None
    if t is not None:
        fig.add_heatmap(z=t.to_numpy(), x=x, y=list(t.index), colorscale="Turbo", colorbar=dict(len=0.4, y=0.8), row=1, col=2)
    if res["wind"]["rose"] is not None:
        for lab in SPEED_LABELS:
            fig.add_barpolar(r=res["wind"]["rose"][lab], theta=np.arange(16) * 22.5, name=f"{lab} m/s", row=2, col=1)
        fig.update_polars(angularaxis=dict(direction="clockwise", rotation=90), row=2, col=1)
    ori = bio["orientation"]
    for c in ori.columns:
        fig.add_scatter(x=x, y=ori[c], name=c, mode="lines+markers", row=2, col=2)
    fig.update_layout(height=900, title=f"Bioclimatic dashboard - {cd.lat:.3f}°, {cd.lon:.3f}° ({cd.temporal}, {cd.period})",
                      template="plotly_white")
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html
