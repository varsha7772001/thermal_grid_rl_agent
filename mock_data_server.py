"""
Mock Data Server for Thermal Grid RL Agent Hackathon Demo
=========================================================
Loads the 3 real-world CSV datasets and exposes REST endpoints
that the RL environment can call instead of using hardcoded values.

Endpoints
---------
GET /ambient-temp?hour=<0-23>
GET /carbon-intensity?month=<1-12>
GET /pue-benchmark?region=india
GET /energy-price?hour=<0-23>
GET /off-peak?hour=<0-23>
GET /signals?hour=<0-23>&month=<1-12>   ← all-in-one

Run:  uvicorn mock_data_server:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FastAPI (optional – falls back to stdlib SimpleHTTPServer if not installed)
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, Query
    from fastapi.responses import JSONResponse
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = None  # type: ignore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
WEATHER_CSV     = DATA_DIR / "india_2000_2024_daily_weather.csv"
CARBON_CSV      = DATA_DIR / "india_monthly_full_release.csv"
REGION_CSV      = DATA_DIR / "region-metadata.csv"

# ---------------------------------------------------------------------------
# Lazy Data Loading
# ---------------------------------------------------------------------------
_DATA_LOADED = False
_monthly_temp: dict[int, float] = {}
_monthly_carbon: dict[int, float] = {}
INDIA_BENCHMARK_PUE: float = 1.58

def load_data():
    """Load and pre-process CSV data (once)."""
    global _DATA_LOADED, _monthly_temp, _monthly_carbon, INDIA_BENCHMARK_PUE
    if _DATA_LOADED:
        return
    
    try:
        print("[MockServer] Loading weather data …")
        if not WEATHER_CSV.exists():
            raise FileNotFoundError(f"Weather CSV not found: {WEATHER_CSV}")
        _weather_df = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
        # Focus on Delhi as representative Indian DC location
        _delhi_weather = _weather_df[_weather_df["city"] == "Delhi"].copy()
        _delhi_weather["month"] = _delhi_weather["date"].dt.month
        
        # Build hourly ambient temp by month: use mean of max+min
        for month in range(1, 13):
            sub = _delhi_weather[_delhi_weather["month"] == month]
            if len(sub):
                _monthly_temp[month] = float(
                    (sub["temperature_2m_max"].mean() + sub["temperature_2m_min"].mean()) / 2
                )
            else:
                _monthly_temp[month] = 25.0

        print("[MockServer] Loading carbon intensity data …")
        if not CARBON_CSV.exists():
            raise FileNotFoundError(f"Carbon CSV not found: {CARBON_CSV}")
        # Pre-filtered: only load CO2 intensity rows (headers: Area, Country code, Date, ..., Variable, Unit, Value, ...)
        _carbon_df = pd.read_csv(CARBON_CSV, usecols=["Country code", "Variable", "Unit", "Date", "Value"])
        _india_carbon = _carbon_df[
            (_carbon_df["Variable"] == "CO2 intensity") &
            (_carbon_df["Unit"] == "gCO2/kWh")
        ].copy()
        _india_carbon["Date"] = pd.to_datetime(_india_carbon["Date"])
        _india_carbon["month"] = _india_carbon["Date"].dt.month

        # Monthly average carbon intensity for India
        for month in range(1, 13):
            sub = _india_carbon[_india_carbon["month"] == month]
            if len(sub) and sub["Value"].notna().any():
                _monthly_carbon[month] = float(sub["Value"].mean())
            else:
                _monthly_carbon[month] = 720.0  # India mean ~720 gCO2/kWh

        print("[MockServer] Loading region benchmarks …")
        if not REGION_CSV.exists():
            raise FileNotFoundError(f"Region CSV not found: {REGION_CSV}")
        _region_df = pd.read_csv(REGION_CSV)
        # India rows
        _india_regions = _region_df[_region_df["location"].str.contains("India|Mumbai|Delhi|Hyderabad", na=False)]
        _india_pue_values = _india_regions["power-usage-efficiency"].dropna().astype(float)
        INDIA_BENCHMARK_PUE = float(_india_pue_values.mean()) if len(_india_pue_values) else 1.58
        
        print(f"[MockServer] India benchmark PUE = {INDIA_BENCHMARK_PUE:.3f}")
        _DATA_LOADED = True
        print("[MockServer] All data loaded successfully ✓")
    except Exception as e:
        print(f"[MockServer] ERROR: Failed to load data: {e}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Indian electricity tariff – time-of-use schedule (₹/kWh → $/kWh @84 INR/$)
# ---------------------------------------------------------------------------
_INR_TO_USD = 1.0 / 84.0
_TOU_INR: list[float] = [
    5.5,  5.5,  5.5,  5.5,  5.5,  6.0,   # 00-05 off-peak
    7.0,  8.5,  9.5, 10.0, 10.0,  9.5,   # 06-11 morning peak
    9.0,  8.5,  9.0, 10.0, 11.0, 12.0,   # 12-17 afternoon/evening peak
   12.0, 11.5, 10.0,  8.5,  7.0,  6.0,   # 18-23 evening taper
]
_TOU_USD = [p * _INR_TO_USD for p in _TOU_INR]

# Off-peak: midnight to 6 AM
_OFF_PEAK_HOURS = set(range(0, 6))

# Diurnal temperature adjustment (how much hotter/cooler relative to monthly mean)
_HOUR_TEMP_DELTA: list[float] = [
   -4.0, -4.5, -5.0, -5.0, -4.5, -3.0,  # 00-05 cool night
   -2.0,  0.0,  2.0,  4.0,  5.0,  5.5,  # 06-11 morning rise
    5.5,  5.0,  4.5,  4.0,  3.0,  1.5,  # 12-17 peak then fall
    0.0, -1.0, -2.0, -2.5, -3.0, -3.5,  # 18-23 evening drop
]

# Solar ramp factor for carbon intensity (lower at midday due to solar)
_SOLAR_FACTOR: list[float] = [
    1.00, 1.00, 1.00, 1.00, 1.00, 0.98,  # 00-05
    0.95, 0.88, 0.80, 0.72, 0.68, 0.65,  # 06-11
    0.63, 0.65, 0.68, 0.75, 0.82, 0.90,  # 12-17
    0.95, 0.98, 1.00, 1.00, 1.00, 1.00,  # 18-23
]

# ---------------------------------------------------------------------------
# Helper functions (used both by FastAPI routes and HybridSignalGenerator)
# ---------------------------------------------------------------------------

def _current_month() -> int:
    return datetime.now().month


def get_ambient_temp(hour: int, month: Optional[int] = None) -> float:
    """Return ambient temperature in °C for given hour and month."""
    load_data()
    month = month or _current_month()
    month = max(1, min(12, month))
    hour  = hour % 24
    base  = _monthly_temp.get(month, 25.0)
    noise = random.gauss(0, 0.5)
    return round(base + _HOUR_TEMP_DELTA[hour] + noise, 2)


def get_carbon_intensity(month: Optional[int] = None, hour: int = 12) -> float:
    """Return grid carbon intensity in gCO₂/kWh."""
    load_data()
    month = month or _current_month()
    month = max(1, min(12, month))
    hour  = hour % 24
    base  = _monthly_carbon.get(month, 720.0)
    noise = random.gauss(0, base * 0.03)
    return round(float(np.clip(base * _SOLAR_FACTOR[hour] + noise, 300, 900)), 2)


def get_energy_price(hour: int) -> float:
    """Return energy price in $/kWh."""
    hour = hour % 24
    noise = random.gauss(0, 0.002)
    return round(float(np.clip(_TOU_USD[hour] + noise, 0.04, 0.20)), 4)


def get_off_peak(hour: int) -> int:
    """Return 1 if off-peak, else 0."""
    return int((hour % 24) in _OFF_PEAK_HOURS)


def get_pue_benchmark(region: str = "india") -> float:
    """Return real-world PUE benchmark for specified region."""
    load_data()
    return round(INDIA_BENCHMARK_PUE, 3)


def get_all_signals(hour: int, month: Optional[int] = None) -> dict:
    """Return all signals in a single dictionary."""
    month = month or _current_month()
    amb   = get_ambient_temp(hour, month)
    ci    = get_carbon_intensity(month, hour)
    price = get_energy_price(hour)
    offpk = get_off_peak(hour)
    dr    = 1 if (hour in {17, 18, 19} and random.random() < 0.25) else 0
    pue_b = get_pue_benchmark()
    return {
        "hour": hour,
        "month": month,
        "ambient_temp_c": amb,
        "energy_price_per_kwh": price,
        "grid_carbon_intensity_g_per_kwh": ci,
        "off_peak_window": offpk,
        "demand_response_signal": dr,
        "pue_benchmark": pue_b,
    }


# ---------------------------------------------------------------------------
# FastAPI app (only if fastapi is available)
# ---------------------------------------------------------------------------
if _HAS_FASTAPI:
    app = FastAPI(
        title="Thermal Grid RL – Mock Data Server",
        description="Serves real India weather, carbon, and grid data for the RL environment.",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        load_data()
        return {"status": "ok", "india_benchmark_pue": INDIA_BENCHMARK_PUE}

    @app.get("/ambient-temp")
    def ambient_temp(
        hour: int = Query(12, ge=0, le=23),
        month: Optional[int] = Query(None, ge=1, le=12),
    ):
        return {"ambient_temp_c": get_ambient_temp(hour, month)}

    @app.get("/carbon-intensity")
    def carbon_intensity(
        hour: int = Query(12, ge=0, le=23),
        month: Optional[int] = Query(None, ge=1, le=12),
    ):
        return {"grid_carbon_intensity_g_per_kwh": get_carbon_intensity(month, hour)}

    @app.get("/energy-price")
    def energy_price(hour: int = Query(12, ge=0, le=23)):
        return {"energy_price_per_kwh": get_energy_price(hour)}

    @app.get("/off-peak")
    def off_peak(hour: int = Query(12, ge=0, le=23)):
        return {"off_peak_window": get_off_peak(hour), "is_off_peak": bool(get_off_peak(hour))}

    @app.get("/pue-benchmark")
    def pue_benchmark(region: str = Query("india")):
        return {"pue_benchmark": get_pue_benchmark(region), "region": region}

    @app.get("/signals")
    def signals(
        hour: int = Query(12, ge=0, le=23),
        month: Optional[int] = Query(None, ge=1, le=12),
    ):
        return get_all_signals(hour, month)

else:
    print("[MockServer] WARNING: FastAPI not installed. "
          "Install with:  pip install fastapi uvicorn")
    print("[MockServer] The HybridSignalGenerator can still run in 'direct' mode.")
    app = None  # type: ignore


# ---------------------------------------------------------------------------
# Fallback: direct-call mode (no HTTP, used by HybridSignalGenerator)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("mock_data_server:app", host="0.0.0.0", port=8001, reload=True)
