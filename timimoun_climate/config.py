"""
config.py - Central configuration for the NASA POWER / Timimoun bioclimatic pipeline.

Everything that is a *scientific assumption* (thresholds, comfort-model constants,
albedo ...) lives here so that the study is transparent and reproducible.

Official documentation used to verify endpoints and parameters
    https://power.larc.nasa.gov/docs/services/api/
    https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
    https://power.larc.nasa.gov/docs/services/api/temporal/daily/
    https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
    https://power.larc.nasa.gov/docs/services/api/temporal/climatology/
    Parameter metadata (live):  GET /api/system/manager/parameters?community=re&temporal=daily
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
BASE_URL = "https://power.larc.nasa.gov/api"          # NASA POWER - no API key needed
TEMPORALS = ("climatology", "monthly", "daily", "hourly")
COMMUNITIES = ("RE", "AG", "SB")                       # Renewable Energy / Agroclimatology / Sustainable Buildings
FORMATS = ("JSON", "CSV", "NETCDF")                    # point-API formats supported by this client
FILL_VALUE = -999.0                                    # POWER missing-data flag

# Maximum parameters per request (official docs: hourly 15, daily 20, monthly 20, climatology 20)
MAX_PARAMS = {"hourly": 15, "daily": 20, "monthly": 20, "climatology": 20}
# First year available per temporal service
FIRST_YEAR = {"hourly": 2001, "daily": 1981, "monthly": 1981, "climatology": 1981}
# Largest period requested in one HTTP call (client splits longer periods automatically)
CHUNK_YEARS = {"hourly": 1, "daily": 10, "monthly": 40, "climatology": 1}
# NASA POWER near-real-time lag (days) - used when computing automatic updates
NRT_LAG_DAYS = 4

# --------------------------------------------------------------------------- #
# Default study site
# --------------------------------------------------------------------------- #
DEFAULT_SITE = {"name": "Timimoun", "country": "Algeria", "lat": 29.2639, "lon": 0.2306}
SITE_PRESETS = {
    "Timimoun (Gourara, Algeria)": (29.2639, 0.2306),
    "Adrar (Algeria)": (27.8742, -0.2939),
    "Ghardaia (M'Zab, Algeria)": (32.4909, 3.6735),
    "Ouargla (Algeria)": (31.9493, 5.3250),
    "Tamanrasset (Algeria)": (22.7850, 5.5228),
    "Custom coordinates": (None, None),
}

# --------------------------------------------------------------------------- #
# Parameter catalogue  (names verified against the POWER parameter-manager API)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Param:
    code: str
    label: str
    group: str
    default: bool = True


PARAMETERS: dict[str, Param] = {p.code: p for p in [
    # Temperature
    Param("T2M", "Air temperature at 2 m (°C)", "Temperature"),
    Param("T2M_MAX", "Maximum temperature at 2 m (°C)", "Temperature"),
    Param("T2M_MIN", "Minimum temperature at 2 m (°C)", "Temperature"),
    Param("T2MDEW", "Dew/frost point at 2 m (°C)", "Temperature"),
    Param("T2MWET", "Wet-bulb temperature at 2 m (°C)", "Temperature", False),
    Param("TS", "Earth skin temperature (°C)", "Temperature", False),
    # Solar radiation
    Param("ALLSKY_SFC_SW_DWN", "All-sky surface shortwave downward irradiance (global horizontal)", "Solar radiation"),
    Param("ALLSKY_SFC_LW_DWN", "All-sky surface longwave downward irradiance", "Solar radiation"),
    Param("ALLSKY_SFC_SW_DIFF", "All-sky surface shortwave diffuse irradiance", "Solar radiation"),
    Param("ALLSKY_SFC_SW_DNI", "All-sky surface shortwave direct normal irradiance", "Solar radiation", False),
    Param("CLRSKY_SFC_SW_DWN", "Clear-sky surface shortwave downward irradiance", "Solar radiation", False),
    Param("ALLSKY_KT", "All-sky insolation clearness index", "Solar radiation", False),
    Param("CLOUD_AMT", "Cloud amount (%)", "Solar radiation", False),
    # Humidity
    Param("RH2M", "Relative humidity at 2 m (%)", "Humidity"),
    Param("QV2M", "Specific humidity at 2 m (g/kg)", "Humidity"),
    # Wind
    Param("WS2M", "Wind speed at 2 m (m/s)", "Wind"),
    Param("WD2M", "Wind direction at 2 m (°)", "Wind"),
    Param("WS10M", "Wind speed at 10 m (m/s)", "Wind", False),
    Param("WD10M", "Wind direction at 10 m (°)", "Wind", False),
    # Precipitation / pressure
    Param("PRECTOTCORR", "Precipitation corrected (mm)", "Precipitation"),
    Param("PS", "Surface pressure (kPa)", "Pressure", False),
]}
DEFAULT_PARAMS = [c for c, p in PARAMETERS.items() if p.default]

# Physically plausible ranges used by the validation step (POWER valid_min/valid_max are wider)
VALID_RANGES = {
    "T2M": (-60, 65), "T2M_MAX": (-60, 70), "T2M_MIN": (-60, 65), "T2MDEW": (-80, 40), "T2MWET": (-60, 45),
    "TS": (-70, 90), "RH2M": (0, 100), "QV2M": (0, 40), "WS2M": (0, 60), "WS10M": (0, 80),
    "WD2M": (0, 360), "WD10M": (0, 360), "PRECTOTCORR": (0, 1000), "PS": (50, 110),
    "ALLSKY_SFC_SW_DWN": (0, 1500), "ALLSKY_SFC_SW_DIFF": (0, 1500), "ALLSKY_SFC_SW_DNI": (0, 1500),
    "CLRSKY_SFC_SW_DWN": (0, 1500), "ALLSKY_SFC_LW_DWN": (0, 700), "ALLSKY_KT": (0, 1.5), "CLOUD_AMT": (0, 100),
}

# --------------------------------------------------------------------------- #
# Scientific assumptions (documented in the report)
# --------------------------------------------------------------------------- #
@dataclass
class Assumptions:
    # Adaptive thermal comfort - ASHRAE 55-2020 / de Dear & Brager (1998), 80 % acceptability
    comfort_slope: float = 0.31
    comfort_intercept: float = 17.8
    comfort_halfwidth: float = 3.5          # K (80 % acceptability); 2.5 K for 90 %
    runmean_alpha: float = 0.8              # ASHRAE 55 running-mean outdoor temperature weight
    runmean_valid: tuple = (10.0, 33.5)     # validity range of the adaptive model (°C)
    # Degree days
    hdd_base: float = 18.0
    cdd_base: float = 24.0
    # Thermal stress thresholds
    hot_day_thresholds: tuple = (35.0, 40.0, 45.0)   # daily Tmax (°C)
    tropical_night: float = 25.0                      # daily Tmin (°C)
    warm_night: float = 20.0
    # Day / night definition (local solar hours)
    day_hours: tuple = (6, 18)
    # Ground albedo for façade/roof irradiance (desert sand / light soil)
    albedo: float = 0.30
    # Strategy thresholds
    dtr_mass: float = 10.0                  # K - diurnal range allowing thermal-mass + night flush
    wb_depression_evap: float = 10.0        # K - wet-bulb depression for effective direct evaporative cooling
    wind_min_vent: float = 1.0              # m/s at 2 m for useful cross ventilation
    wind_dust: float = 5.0                  # m/s at 2 m - sand/dust-drift concern (monthly mean daily max proxy)
    ghi_high: float = 5.5                   # kWh/m²/day - high insolation
    ghi_heating_gain: float = 3.0           # kWh/m²/day - useful winter gain
    wet_day_mm: float = 1.0                 # mm/day threshold for a wet day


ASSUMPTIONS = Assumptions()

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = {"Winter (DJF)": (12, 1, 2), "Spring (MAM)": (3, 4, 5), "Summer (JJA)": (6, 7, 8), "Autumn (SON)": (9, 10, 11)}
SECTORS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

# Orientation azimuths (0 = North, 90 = East, 180 = South, 270 = West)
ORIENTATIONS = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


# --------------------------------------------------------------------------- #
# Project paths / run configuration
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    name: str = DEFAULT_SITE["name"]
    lat: float = DEFAULT_SITE["lat"]
    lon: float = DEFAULT_SITE["lon"]
    temporal: str = "daily"
    community: str = "RE"
    parameters: list = field(default_factory=lambda: list(DEFAULT_PARAMS))
    start: str = "20150101"            # YYYYMMDD (daily/hourly) or YYYY (monthly)
    end: str = "20241231"
    time_standard: str = "LST"         # hourly only: LST or UTC
    raw_format: str = "JSON"           # JSON | CSV | NETCDF
    data_dir: Path = Path("data")
    use_cache: bool = True

    @property
    def slug(self) -> str:
        return f"{self.name.lower().replace(' ', '_')}_{self.lat:.4f}_{self.lon:.4f}"

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw" / self.slug

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_dir) / "processed" / self.slug

    @property
    def figures_dir(self) -> Path:
        return Path(self.data_dir) / "figures" / self.slug

    @property
    def reports_dir(self) -> Path:
        return Path(self.data_dir) / "reports" / self.slug
