#!/usr/bin/env python
"""
Command-line entry point.

Examples
    python main.py                                             # Timimoun, daily 2015-2024, all default parameters
    python main.py --temporal hourly --start 20230101 --end 20241231 --format NETCDF
    python main.py --lat 27.87 --lon -0.29 --name Adrar --temporal monthly --start 2000 --end 2024
    python main.py --temporal climatology
    python main.py --update                                    # append only new days to an existing dataset
    python main.py --update --watch 24                         # keep updating every 24 h (or use cron / Task Scheduler)
"""
import argparse
import logging
import time
from pathlib import Path

from timimoun_climate import config as C
from timimoun_climate.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="NASA POWER climate acquisition & bioclimatic analysis")
    ap.add_argument("--name", default=C.DEFAULT_SITE["name"])
    ap.add_argument("--lat", type=float, default=C.DEFAULT_SITE["lat"])
    ap.add_argument("--lon", type=float, default=C.DEFAULT_SITE["lon"])
    ap.add_argument("--temporal", choices=C.TEMPORALS, default="daily")
    ap.add_argument("--community", choices=C.COMMUNITIES, default="RE")
    ap.add_argument("--start", default="20150101", help="YYYYMMDD (daily/hourly) or YYYY (monthly)")
    ap.add_argument("--end", default="20241231")
    ap.add_argument("--parameters", default=",".join(C.DEFAULT_PARAMS), help="comma separated POWER codes")
    ap.add_argument("--format", default="JSON", choices=C.FORMATS, help="raw format requested from POWER")
    ap.add_argument("--time-standard", default="LST", choices=["LST", "UTC"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--update", action="store_true", help="incremental update of an existing dataset")
    ap.add_argument("--watch", type=float, default=0, help="repeat every N hours (automatic updates)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = C.RunConfig(name=a.name, lat=a.lat, lon=a.lon, temporal=a.temporal, community=a.community,
                      parameters=[p.strip().upper() for p in a.parameters.split(",") if p.strip()],
                      start=a.start, end=a.end, time_standard=a.time_standard, raw_format=a.format,
                      data_dir=Path(a.data_dir), use_cache=not a.no_cache)
    first = True
    while True:
        out = run_pipeline(cfg, progress=lambda f, m: print(f"[{f:5.0%}] {m}"), update=a.update and not first or a.update,
                           make_report=not a.no_report)
        print("\nProcessed files:", *[str(p) for p in out["files"]], sep="\n  ")
        if not a.no_report:
            print("\nReports:\n ", out["pdf"], "\n ", out["xlsx"])
        for i in out["issues"]:
            print("QA:", i)
        first = False
        if a.watch <= 0:
            break
        time.sleep(a.watch * 3600)


if __name__ == "__main__":
    main()
