# Timimoun climate · NASA POWER acquisition & bioclimatic analysis

Automated Python pipeline that connects the **official NASA POWER API** (https://power.larc.nasa.gov/api/ - no API key)
with architectural environmental analysis for Timimoun, Algeria (29.2639 °N, 0.2306 °E; configurable).

## Install (Anaconda)
```bash
conda env create -f environment.yml
conda activate timimoun-climate
```

## Run
```bash
streamlit run app.py                         # graphical interface
python main.py                               # CLI: Timimoun, daily 2015-2024
python main.py --temporal hourly --start 20230101 --end 20241231 --format NETCDF
python main.py --lat 27.87 --lon -0.29 --name Adrar --temporal monthly --start 2000 --end 2024
python main.py --temporal climatology
python main.py --update --watch 24           # incremental automatic updates every 24 h
```

## Verified API facts (checked live against power.larc.nasa.gov)
| Service | Endpoint | Period format | Params/request | Notes |
|---|---|---|---|---|
| Hourly | `/api/temporal/hourly/point` | YYYYMMDD, from 2001 | 15 | LST (default) or UTC via `time-standard`; **no T2M_MAX/T2M_MIN**; radiation in Wh/m²; precipitation mm/hour |
| Daily | `/api/temporal/daily/point` | YYYYMMDD, from 1981 | 20 | radiation kWh/m²/day |
| Monthly | `/api/temporal/monthly/point` | YYYY | 20 | key `YYYY13` = annual; **T2M_MAX/MIN are monthly extremes** |
| Climatology | `/api/temporal/climatology/point` | - | 20 | keys JAN..DEC, ANN |
| Availability | `/api/system/manager/parameters?community=re&temporal=daily` | | | used for the pre-request check |

Formats supported by the client: JSON, CSV, NETCDF (point requests; NetCDF is read with xarray). Fill value `-999` -> NaN.
Communities: RE, AG, SB. CORS is open, errors are returned as `{"messages": [...]}` and surfaced as `PowerAPIError`.

## Modules
| file | role |
|---|---|
| `config.py` | parameter catalogue, API limits, **all scientific assumptions** |
| `nasa_power_client.py` | requests session, retries/back-off, availability check, chunking, raw preservation, JSON/CSV/NetCDF parsers |
| `processing.py` | validation, hourly->daily aggregation, derived indices (wet-bulb, heat index, humidex), xarray + CSV/Excel/NetCDF export |
| `solar.py` | solar geometry, façade/roof irradiation transposition, overhang geometry |
| `analysis.py` | monthly/seasonal/annual statistics, thermal stress, wind rose, precipitation & aridity, diurnal profiles |
| `bioclimatic.py` | adaptive comfort (ASHRAE 55), passive strategy matrix, recommendations |
| `visualization.py` | Matplotlib figures + Plotly interactive dashboard |
| `reporting.py` | PDF (ReportLab) and Excel (openpyxl) reports |
| `pipeline.py`, `main.py`, `app.py` | orchestration, CLI, Streamlit UI |

## Output layout
```
data/raw/<site>_<lat>_<lon>/<temporal>/<format>/...   original responses + .request.json (URL, SHA-256)
data/processed/<site>.../                              CSV, XLSX, NetCDF, metadata.json, statistics
data/figures/<site>.../                                PNG + interactive_dashboard.html
data/reports/<site>.../                                PDF and Excel reports
```

## Evidence layers (used in every output)
`[NASA]` retrieved variable · `[CALC]` calculated indicator · `[INTERP]` architectural interpretation.
POWER values are re-analysis/satellite products (MERRA-2, CERES), **not station measurements**; see report section 8.
