"""
nasa_power_client.py - Client for the official NASA POWER point API.

Endpoints (https://power.larc.nasa.gov/docs/services/api/):
    /api/temporal/hourly/point
    /api/temporal/daily/point
    /api/temporal/monthly/point
    /api/temporal/climatology/point
    /api/system/manager/parameters            (parameter availability per temporal service / community)

No API key is required.  The client provides
    * automatic retries with exponential back-off (HTTP 429/5xx, time-outs)
    * parameter-availability checking *before* any data request
    * transparent splitting of long periods / long parameter lists into legal requests
    * response validation (HTTP status, POWER error messages, empty / all-fill responses)
    * preservation of every original raw response (JSON, CSV or NetCDF) + a request sidecar file
    * parsing of JSON, CSV and NetCDF (xarray) responses into pandas DataFrames
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config as C

log = logging.getLogger(__name__)

MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


class PowerAPIError(RuntimeError):
    """Raised when NASA POWER rejects a request or returns unusable content."""


@dataclass
class PowerResult:
    """Container for a retrieved dataset."""
    data: pd.DataFrame                       # time-indexed (hourly/daily/monthly) or month-number-indexed (climatology)
    temporal: str
    annual: pd.DataFrame | None = None       # monthly: annual rows (month 13); climatology: ANN column as 1-row frame
    meta: dict = field(default_factory=dict)  # units, long names, elevation, sources, urls ...
    raw_files: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _year_of(s: str) -> int:
    return int(str(s)[:4])


def split_period(temporal: str, start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into blocks that respect CHUNK_YEARS for the temporal service."""
    if temporal == "climatology":
        return [("", "")]
    step = C.CHUNK_YEARS[temporal]
    if temporal == "monthly":
        y0, y1 = _year_of(start), _year_of(end)
        return [(str(y), str(min(y + step - 1, y1))) for y in range(y0, y1 + 1, step)]
    d0 = datetime.strptime(str(start), "%Y%m%d").date()
    d1 = datetime.strptime(str(end), "%Y%m%d").date()
    blocks, cur = [], d0
    while cur <= d1:
        nxt = date(cur.year + step, cur.month, cur.day) if not (cur.month == 2 and cur.day == 29) else date(cur.year + step, 3, 1)
        blk_end = min(nxt - timedelta(days=1), d1)
        blocks.append((cur.strftime("%Y%m%d"), blk_end.strftime("%Y%m%d")))
        cur = blk_end + timedelta(days=1)
    return blocks


def chunk_list(items: list, n: int) -> list[list]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def validate_period(temporal: str, start: str, end: str) -> None:
    """Raise ValueError for impossible periods (before the first available year / reversed / future)."""
    if temporal == "climatology":
        return
    fmt = "%Y" if temporal == "monthly" else "%Y%m%d"
    try:
        d0, d1 = datetime.strptime(str(start), fmt), datetime.strptime(str(end), fmt)
    except ValueError as exc:
        raise ValueError(f"{temporal}: dates must be {'YYYY' if temporal == 'monthly' else 'YYYYMMDD'} ({exc})") from exc
    if d0 > d1:
        raise ValueError("Start date is after end date.")
    if d0.year < C.FIRST_YEAR[temporal]:
        raise ValueError(f"{temporal} data start in {C.FIRST_YEAR[temporal]}.")
    if d1 > datetime.now() + timedelta(days=1):
        raise ValueError("End date lies in the future.")


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
class PowerClient:
    """Thin, robust wrapper around the NASA POWER point API."""

    def __init__(self, community: str = "RE", time_standard: str = "LST", timeout: int = 180,
                 retries: int = 5, backoff: float = 1.5, pause: float = 0.4,
                 raw_dir: str | Path | None = None, use_cache: bool = True,
                 session: requests.Session | None = None):
        if community not in C.COMMUNITIES:
            raise ValueError(f"community must be one of {C.COMMUNITIES}")
        self.community, self.time_standard = community, time_standard
        self.timeout, self.pause, self.use_cache = timeout, pause, use_cache
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self._avail_cache: dict[str, dict] = {}
        self.session = session or requests.Session()
        retry = Retry(total=retries, connect=retries, read=retries, status=retries, backoff_factor=backoff,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
                      respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"User-Agent": "timimoun-bioclimatic-research/1.0 (academic use)"})

    # ------------------------------------------------------------------ #
    # low level
    # ------------------------------------------------------------------ #
    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:  # after urllib3 retries are exhausted
            raise PowerAPIError(f"Network error contacting NASA POWER: {exc}") from exc
        if resp.status_code != 200:
            detail = ""
            try:
                body = resp.json()
                detail = "; ".join(body.get("messages", [])) or body.get("header", "")
            except ValueError:
                detail = resp.text[:300]
            raise PowerAPIError(f"HTTP {resp.status_code} from NASA POWER: {detail}")
        return resp

    # ------------------------------------------------------------------ #
    # availability
    # ------------------------------------------------------------------ #
    def available_parameters(self, temporal: str) -> dict:
        """Live list of parameters offered for a temporal service + community (parameter-manager API)."""
        if temporal not in self._avail_cache:
            url = f"{C.BASE_URL}/system/manager/parameters"
            resp = self._get(url, {"community": self.community.lower(), "temporal": temporal})
            self._avail_cache[temporal] = resp.json()
        return self._avail_cache[temporal]

    def check_parameters(self, temporal: str, parameters: list[str]) -> tuple[list[str], list[str]]:
        """Return (available, unavailable) for the requested parameter codes."""
        try:
            catalogue = self.available_parameters(temporal)
        except PowerAPIError as exc:  # metadata service down: fall back to the documented rules
            log.warning("Parameter manager unreachable (%s); using built-in rules.", exc)
            no_hourly = {"T2M_MAX", "T2M_MIN", "FROST_DAYS", "T2M_RANGE", "WS2M_MAX", "WS2M_MIN"}
            ok = [p for p in parameters if not (temporal == "hourly" and p in no_hourly)]
            return ok, [p for p in parameters if p not in ok]
        ok = [p for p in parameters if p in catalogue]
        return ok, [p for p in parameters if p not in catalogue]

    # ------------------------------------------------------------------ #
    # request construction
    # ------------------------------------------------------------------ #
    def build_request(self, temporal: str, parameters: list[str], lat: float, lon: float,
                      start: str, end: str, fmt: str = "JSON") -> tuple[str, dict]:
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("Latitude must be in [-90, 90] and longitude in [-180, 180].")
        url = f"{C.BASE_URL}/temporal/{temporal}/point"
        params = {"parameters": ",".join(parameters), "community": self.community,
                  "latitude": round(lat, 4), "longitude": round(lon, 4), "format": fmt.upper(), "header": "true"}
        if temporal != "climatology":
            params.update(start=start, end=end)
        if temporal == "hourly":
            params["time-standard"] = self.time_standard
        return url, params

    # ------------------------------------------------------------------ #
    # fetch one legal request, preserve the raw answer
    # ------------------------------------------------------------------ #
    def _fetch_block(self, temporal, parameters, lat, lon, start, end, fmt) -> tuple[pd.DataFrame, pd.DataFrame | None, dict, Path | None]:
        url, params = self.build_request(temporal, parameters, lat, lon, start, end, fmt)
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
        ext = {"JSON": "json", "CSV": "csv", "NETCDF": "nc"}[fmt.upper()]
        raw_path = None
        content = None
        if self.raw_dir:
            raw_path = self.raw_dir / temporal / fmt.lower() / f"{temporal}_{self.community}_{start or 'clim'}_{end or 'clim'}_{key}.{ext}"
            if self.use_cache and raw_path.exists():
                content = raw_path.read_bytes()
                log.info("cache hit: %s", raw_path.name)
        if content is None:
            log.info("GET %s %s", url, params)
            resp = self._get(url, params)
            content = resp.content
            if fmt.upper() == "JSON":
                self._validate_json(resp.json())
            if raw_path:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(content)  # original, untouched
                sidecar = {"url": resp.url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                           "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "params": params}
                raw_path.with_suffix(raw_path.suffix + ".request.json").write_text(json.dumps(sidecar, indent=2))
            time.sleep(self.pause)  # be polite: stay well below the POWER rate limits
        meta = {"urls": [f"{url}?{requests.compat.urlencode(params)}"]}
        if fmt.upper() == "JSON":
            df, ann, m = self.parse_json(json.loads(content), temporal)
        elif fmt.upper() == "CSV":
            df, ann, m = self.parse_csv(content.decode("utf-8", "replace"), temporal)
        else:
            df, ann, m = self.parse_netcdf(content, temporal)
        meta.update(m)
        return df, ann, meta, raw_path

    @staticmethod
    def _validate_json(body: dict) -> None:
        if "messages" in body and "properties" not in body:
            raise PowerAPIError("; ".join(body["messages"]))
        if "properties" not in body or "parameter" not in body["properties"]:
            raise PowerAPIError("Unexpected NASA POWER response structure (no properties.parameter).")
        if not body["properties"]["parameter"]:
            raise PowerAPIError("NASA POWER returned an empty parameter block.")

    # ------------------------------------------------------------------ #
    # public: complete retrieval
    # ------------------------------------------------------------------ #
    def fetch(self, temporal: str, parameters: list[str], lat: float, lon: float,
              start: str = "", end: str = "", fmt: str = "JSON", progress=None) -> PowerResult:
        """Retrieve (and merge) all blocks needed for the requested period/parameters."""
        if temporal not in C.TEMPORALS:
            raise ValueError(f"temporal must be one of {C.TEMPORALS}")
        validate_period(temporal, start, end)
        ok, missing = self.check_parameters(temporal, parameters)
        if missing:
            log.warning("Parameters not offered by the %s API for community %s: %s", temporal, self.community, missing)
        if not ok:
            raise PowerAPIError(f"None of the requested parameters exist for {temporal}/{self.community}.")
        periods = split_period(temporal, start, end)
        groups = chunk_list(ok, C.MAX_PARAMS[temporal])
        total = len(periods) * len(groups)
        frames, annuals, metas, raws, done = [], [], [], [], 0
        for p0, p1 in periods:
            row_frames, row_annuals = [], []
            for grp in groups:
                df, ann, meta, raw = self._fetch_block(temporal, grp, lat, lon, p0, p1, fmt)
                row_frames.append(df)
                if ann is not None:
                    row_annuals.append(ann)
                metas.append(meta)
                if raw:
                    raws.append(raw)
                done += 1
                if progress:
                    progress(done / total, f"{temporal} {p0}-{p1}  ({done}/{total})")
            frames.append(pd.concat(row_frames, axis=1))
            if row_annuals:
                annuals.append(pd.concat(row_annuals, axis=1))
        data = pd.concat(frames).sort_index()
        data = data[~data.index.duplicated(keep="last")]
        annual = pd.concat(annuals).sort_index() if annuals else None
        meta = self._merge_meta(metas, temporal, lat, lon, ok, missing)
        return PowerResult(data=data, temporal=temporal, annual=annual, meta=meta, raw_files=raws)

    @staticmethod
    def _merge_meta(metas: list[dict], temporal, lat, lon, ok, missing) -> dict:
        out = {"temporal": temporal, "lat": lat, "lon": lon, "parameters": ok, "unavailable_parameters": missing,
               "units": {}, "long_names": {}, "urls": [], "sources": set(), "api_versions": set(),
               "elevation_m": None, "time_standard": None, "fill_value": C.FILL_VALUE,
               "retrieved_utc": datetime.now(timezone.utc).isoformat()}
        for m in metas:
            out["units"].update(m.get("units", {}))
            out["long_names"].update(m.get("long_names", {}))
            out["urls"] += m.get("urls", [])
            out["sources"] |= set(m.get("sources", []))
            if m.get("api_version"):
                out["api_versions"].add(m["api_version"])
            out["elevation_m"] = m.get("elevation_m", out["elevation_m"])
            out["time_standard"] = m.get("time_standard", out["time_standard"])
        out["sources"], out["api_versions"] = sorted(out["sources"]), sorted(out["api_versions"])
        return out

    # ------------------------------------------------------------------ #
    # parsers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.mask(df <= C.FILL_VALUE + 0.5)  # -999 -> NaN

    def parse_json(self, body: dict, temporal: str):
        self._validate_json(body)
        par = body["properties"]["parameter"]
        hdr = body.get("header", {})
        meta = {"units": {k: v.get("units") for k, v in body.get("parameters", {}).items()},
                "long_names": {k: v.get("longname") for k, v in body.get("parameters", {}).items()},
                "sources": hdr.get("sources", []), "api_version": (hdr.get("api") or {}).get("version"),
                "time_standard": hdr.get("time_standard"),
                "elevation_m": (body.get("geometry", {}).get("coordinates") or [None, None, None])[2]}
        df = self._clean(pd.DataFrame(par))
        annual = None
        if temporal == "hourly":
            df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
        elif temporal == "daily":
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
        elif temporal == "monthly":  # keys YYYYMM, month 13 = annual mean
            is_ann = df.index.str[4:] == "13"
            annual = df[is_ann].copy()
            annual.index = annual.index.str[:4].astype(int)
            df = df[~is_ann]
            df.index = pd.to_datetime(df.index, format="%Y%m")
        else:  # climatology: JAN..DEC, ANN
            annual = df.loc[["ANN"]] if "ANN" in df.index else None
            df = df.loc[MONTH_ABBR]
            df.index = range(1, 13)
        df.index.name = "time" if temporal != "climatology" else "month"
        return df, annual, meta

    def parse_csv(self, text: str, temporal: str):
        head, _, body = text.partition("-END HEADER-")
        units, longn = {}, {}
        for line in head.splitlines():
            m = re.match(r"^([A-Z0-9_]+)\s+(?:MERRA-2|CERES|POWER|SYN1deg|GEOS)?.*?\(([^)]*)\)\s*$", line.strip())
            if m:
                units[m.group(1)] = m.group(2)
        elev = re.search(r"elevation.*?=\s*([\d.\-]+)\s*meters", head)
        meta = {"units": units, "long_names": longn, "sources": [], "api_version": None,
                "time_standard": "UTC" if "in UTC" in head else "LST",
                "elevation_m": float(elev.group(1)) if elev else None}
        raw = pd.read_csv(io.StringIO(body.strip()))
        annual = None
        if temporal == "climatology":
            raw = raw.set_index("PARAMETER").T
            raw = self._clean(raw)
            annual = raw.loc[["ANN"]] if "ANN" in raw.index else None
            df = raw.loc[MONTH_ABBR]
            df.index = range(1, 13)
            df.index.name = "month"
            return df, annual, meta
        if temporal == "monthly":  # layout: PARAMETER,YEAR,JAN..DEC,ANN
            long = raw.melt(id_vars=["PARAMETER", "YEAR"], var_name="MON", value_name="v")
            wide = long.pivot_table(index=["YEAR", "MON"], columns="PARAMETER", values="v", aggfunc="first")
            wide = self._clean(wide)
            annual = wide.xs("ANN", level="MON")
            annual.index.name = "year"
            df = wide.drop("ANN", level="MON")
            df.index = pd.to_datetime([f"{y}-{MONTH_ABBR.index(m) + 1:02d}-01" for y, m in df.index])
            df = df.sort_index()
            df.index.name = "time"
            return df, annual, meta
        cols = ["YEAR", "MO", "DY", "HR"][: {"hourly": 4, "daily": 3}[temporal]]
        idx = pd.to_datetime(raw[cols].rename(columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}))
        df = self._clean(raw.drop(columns=cols))
        df.index = idx
        df.index.name = "time"
        return df, annual, meta

    def parse_netcdf(self, content: bytes, temporal: str):
        import xarray as xr  # local import: only needed for NetCDF
        try:
            ds = xr.open_dataset(io.BytesIO(content), engine="h5netcdf")
        except Exception:  # fall back to the netCDF4 engine (needs a file on disk)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(content)
            ds = xr.open_dataset(tmp.name, engine="netcdf4")
        ds = ds.load()
        dims = {d for d in ("lat", "lon", "latitude", "longitude") if d in ds.dims}
        ds = ds.squeeze(list(dims), drop=True) if dims else ds
        meta = {"units": {v: ds[v].attrs.get("units") for v in ds.data_vars},
                "long_names": {v: ds[v].attrs.get("long_name") for v in ds.data_vars},
                "sources": [], "api_version": ds.attrs.get("version"), "time_standard": None,
                "elevation_m": float(ds.attrs["elevation"]) if "elevation" in ds.attrs else None}
        df = ds.to_dataframe()[list(ds.data_vars)]
        df = self._clean(df)
        annual = None
        tdim = [d for d in ds.dims][0]
        if temporal == "climatology":
            df = df.reset_index(drop=True)
            if len(df) == 13:
                annual, df = df.iloc[[12]], df.iloc[:12]
            df.index = range(1, 13)
            df.index.name = "month"
        elif temporal == "monthly":  # time axis holds integers YYYYMM; month 13 = annual mean
            keys = pd.Index(df.index.astype(str))
            is_ann = keys.str[4:] == "13"
            annual = df[is_ann].copy()
            annual.index = keys[is_ann].str[:4].astype(int)
            annual.index.name = "year"
            df = df[~is_ann]
            df.index = pd.to_datetime(keys[~is_ann], format="%Y%m")
            df.index.name = "time"
        else:
            df.index = pd.to_datetime(df.index)
            df.index.name = tdim
        return df, annual, meta
