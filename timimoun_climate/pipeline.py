"""
pipeline.py - orchestrates acquisition -> validation -> analysis -> bioclimatic interpretation -> figures -> reports.

    from timimoun_climate.pipeline import run_pipeline
    from timimoun_climate.config import RunConfig
    out = run_pipeline(RunConfig(lat=29.2639, lon=0.2306, temporal="daily", start="20150101", end="20241231"))
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import analysis, bioclimatic, processing, reporting, visualization
from . import config as C
from .nasa_power_client import PowerClient, PowerResult

log = logging.getLogger(__name__)


def _processed_path(cfg: C.RunConfig) -> Path:
    return cfg.processed_dir / f"{cfg.temporal}_data.csv"


def acquire(cfg: C.RunConfig, progress=None, update: bool = False) -> PowerResult:
    """Download (or incrementally update) the dataset described by `cfg`."""
    client = PowerClient(community=cfg.community, time_standard=cfg.time_standard, raw_dir=cfg.raw_dir, use_cache=cfg.use_cache)
    start, end, previous = cfg.start, cfg.end, None
    if update and cfg.temporal in ("daily", "hourly", "monthly") and _processed_path(cfg).exists():
        previous = pd.read_csv(_processed_path(cfg), index_col=0, parse_dates=True)
        last = previous.index.max()
        if cfg.temporal == "monthly":
            start, end = str(last.year), str(date.today().year)
        else:
            start = last.strftime("%Y%m%d")
            end = (date.today() - timedelta(days=C.NRT_LAG_DAYS)).strftime("%Y%m%d")
        log.info("Auto-update: existing data end %s; requesting %s-%s", last, start, end)
        client.use_cache = False  # never reuse a cached tail
        if start > end:
            log.info("Dataset already up to date.")
            return PowerResult(data=previous, temporal=cfg.temporal, meta={"units": {}, "parameters": list(previous.columns)})
    result = client.fetch(cfg.temporal, cfg.parameters, cfg.lat, cfg.lon, start, end, cfg.raw_format, progress)
    result.meta["community"] = cfg.community
    if previous is not None:
        merged = pd.concat([previous[previous.index < result.data.index.min()], result.data])
        result.data = merged[~merged.index.duplicated(keep="last")].sort_index()
    return result


def run_pipeline(cfg: C.RunConfig, progress=None, update: bool = False, make_report: bool = True,
                 make_figures: bool = True) -> dict:
    """Execute the complete workflow and return all objects + paths of the written files."""
    for d in (cfg.raw_dir, cfg.processed_dir, cfg.figures_dir, cfg.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    result = acquire(cfg, progress, update)
    validation, issues = processing.validate(result.data, cfg.temporal, result.meta)

    # --- save processed data (CSV/Excel/NetCDF) -------------------------------------------------
    files = processing.export_dataset(result.data, result.meta, cfg.processed_dir, f"{cfg.temporal}_data", cfg.lat, cfg.lon)
    (cfg.processed_dir / "metadata.json").write_text(json.dumps(result.meta, indent=2, default=str))

    cd = analysis.build_climate_data(result, cfg.lat, cfg.lon)
    cd.meta["community"] = cfg.community
    if cd.daily is not None and cfg.temporal == "hourly":
        files += processing.export_dataset(cd.daily, result.meta, cfg.processed_dir, "daily_from_hourly_with_indicators", cfg.lat, cfg.lon, ("csv",))
    elif cd.daily is not None:
        files += processing.export_dataset(cd.daily, result.meta, cfg.processed_dir, "daily_with_indicators", cfg.lat, cfg.lon, ("csv",))
    res = analysis.run_analysis(cd)
    bio = bioclimatic.run_bioclimatic(cd, res)
    for name in ("monthly", "seasonal"):
        res[name].to_csv(cfg.processed_dir / f"stats_{name}.csv")
    out = {"config": cfg, "result": result, "climate": cd, "analysis": res, "bioclimatic": bio,
           "validation": validation, "issues": issues, "files": files, "figures": {}, "figure_paths": {}}

    if make_figures:
        figs = visualization.build_figures(cd, res, bio)
        out["figures"] = figs
        out["figure_paths"] = visualization.save_all(figs, cfg.figures_dir)
        html = visualization.plotly_dashboard(cd, res, bio, cfg.figures_dir / "interactive_dashboard.html")
        if html:
            out["files"].append(html)
    if make_report:
        stamp = datetime.now().strftime("%Y%m%d")
        out["pdf"] = reporting.write_pdf(cfg.reports_dir / f"{cfg.slug}_{cfg.temporal}_report_{stamp}.pdf", cd, res, bio,
                                         out["figure_paths"], validation, issues, cfg.name)
        out["xlsx"] = reporting.write_excel(cfg.reports_dir / f"{cfg.slug}_{cfg.temporal}_report_{stamp}.xlsx", cd, res, bio, validation, issues)
    return out
