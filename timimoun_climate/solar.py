"""
solar.py - solar geometry and surface-irradiance transposition  [CALC]

Equations: Duffie & Beckman, "Solar Engineering of Thermal Processes" (4th ed.), ch. 1-2.
All times are *local solar time*, which is also the POWER default time standard (LST).
Azimuth convention: 0° = North, 90° = East, 180° = South, 270° = West.
"""
from __future__ import annotations

import calendar

import numpy as np

GSC = 1367.0  # W/m² solar constant (Duffie & Beckman)
MID_MONTH_DOY = [17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]  # Klein (1977) representative days


def declination(n):
    """Solar declination (deg) - Cooper (1969)."""
    return 23.45 * np.sin(np.deg2rad(360.0 * (284 + np.asarray(n)) / 365.0))


def g0n(n):
    """Extraterrestrial normal irradiance (W/m²) incl. Earth-Sun distance correction."""
    return GSC * (1 + 0.033 * np.cos(np.deg2rad(360.0 * np.asarray(n) / 365.0)))


def sun_position(lat: float, n: int, hours):
    """Return (cos_zenith, altitude_deg, azimuth_deg) for local solar time `hours` (array) on day-of-year n."""
    h = np.asarray(hours, float)
    phi, dec = np.deg2rad(lat), np.deg2rad(declination(n))
    w = np.deg2rad(15.0 * (h - 12.0))
    cosz = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(w)
    cosz_c = np.clip(cosz, -1, 1)
    alt = 90.0 - np.rad2deg(np.arccos(cosz_c))
    # azimuth from North, clockwise
    sinz = np.sqrt(np.clip(1 - cosz_c ** 2, 1e-12, None))
    cosaz = (np.sin(dec) - cosz_c * np.sin(phi)) / (sinz * np.cos(phi) + 1e-12)
    az = np.rad2deg(np.arccos(np.clip(cosaz, -1, 1)))
    az = np.where(h > 12, 360.0 - az, az)
    return cosz, alt, az


def extraterrestrial_daily(lat: float, n: int) -> float:
    """Daily extraterrestrial horizontal radiation H0 in kWh/m²/day."""
    hrs = np.arange(0.05, 24, 0.1)
    cosz, _, _ = sun_position(lat, n, hrs)
    return float(g0n(n) * np.clip(cosz, 0, None).sum() * 0.1 / 1000.0)


def solstice_noon_altitude(lat: float) -> dict:
    """Solar altitude at solar noon on the solstices/equinox (deg)."""
    out = {}
    for name, n in {"Winter solstice (21 Dec)": 355, "Equinox (21 Mar)": 80, "Summer solstice (21 Jun)": 172}.items():
        _, alt, _ = sun_position(lat, n, [12.0])
        out[name] = float(alt[0])
    return out


def surface_irradiance_day(lat: float, n: int, ghi_kwh: float, dhi_kwh: float, tilt: float, azimuth: float,
                           albedo: float = 0.3) -> float:
    """Daily irradiation (kWh/m²/day) on a plane of given tilt/azimuth for a representative day.

    Method (transparent, isotropic-sky Liu-Jordan transposition):
      1. distribute the daily global (GHI) and diffuse (DHI) energy over the day proportionally to the
         extraterrestrial horizontal irradiance  I0(h) = G0n * cos(zenith)
      2. beam-horizontal = GHI - DHI ; DNI = beam-horizontal / cos(zenith)   (cos z >= 0.1)
      3. I_plane = DNI*cos(AOI) + DHI*(1+cos beta)/2 + GHI*albedo*(1-cos beta)/2
    """
    if not np.isfinite(ghi_kwh) or ghi_kwh <= 0:
        return float("nan")
    dhi_kwh = min(max(dhi_kwh if np.isfinite(dhi_kwh) else 0.25 * ghi_kwh, 0), ghi_kwh)
    hrs = np.arange(0.5, 24, 1.0)
    cosz, alt, az = sun_position(lat, n, hrs)
    i0 = np.clip(g0n(n) * cosz, 0, None)
    if i0.sum() <= 0:
        return 0.0
    share = i0 / i0.sum()
    ghi_h, dhi_h = ghi_kwh * 1000 * share, dhi_kwh * 1000 * share      # W/m² (1-h steps)
    beam_h = np.clip(ghi_h - dhi_h, 0, None)
    dni = np.where(cosz >= 0.1, beam_h / np.clip(cosz, 0.1, None), 0.0)
    b, g = np.deg2rad(tilt), np.deg2rad(azimuth)
    sinz = np.sqrt(np.clip(1 - np.clip(cosz, -1, 1) ** 2, 0, None))
    cos_aoi = np.clip(cosz * np.cos(b) + sinz * np.sin(b) * np.cos(np.deg2rad(az) - g), 0, None)
    plane = dni * cos_aoi + dhi_h * (1 + np.cos(b)) / 2 + ghi_h * albedo * (1 - np.cos(b)) / 2
    return float(plane.sum() / 1000.0)


def days_in_month_avg(month: int) -> float:
    return calendar.monthrange(2001, month)[1] if month != 2 else 28.25


def overhang_ratio(lat: float, n: int) -> float:
    """Projection/height ratio of a horizontal overhang on a south wall that fully shades the window at solar
    noon of day n:  P/H = 1 / tan(altitude_noon)  (valid for lat > declination, i.e. sun south of zenith)."""
    _, alt, _ = sun_position(lat, n, [12.0])
    a = float(alt[0])
    return float(1.0 / np.tan(np.deg2rad(a))) if 0 < a < 90 else float("inf")
