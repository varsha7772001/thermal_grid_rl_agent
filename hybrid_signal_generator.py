"""
Hybrid Signal Generator
=======================
Provides real-world-grounded grid signals to the RL environment.

Two modes:
  use_mock_api=True  → HTTP calls to the mock_data_server running on localhost:8001
  use_mock_api=False → Direct in-process calls (no HTTP needed, ideal for training)

Environment Variables:
  USE_REAL_DATA  - If "true", fetches real data from Open-Meteo API and Indian grid APIs
  DC_CITY        - City name for weather data (e.g., "Bengaluru", "Delhi", "Mumbai")
  GRID_REGION    - Indian grid region (e.g., "Western", "Northern", "Southern", "Eastern", "NorthEastern")

Usage
-----
    from hybrid_signal_generator import HybridSignalGenerator

    gen = HybridSignalGenerator(use_mock_api=True)     # server must be running
    gen = HybridSignalGenerator(use_mock_api=False)    # standalone / training mode

    signals = gen.get(hour=14)
    print(signals["ambient_temp_c"])
    print(signals["grid_carbon_intensity_g_per_kwh"])
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class HybridSignalGenerator:
    """
    Fetches hybrid data signals – a mix of real CSV data and computed values –
    for use in the ThermalGridRlAgent environment.

    Parameters
    ----------
    use_mock_api : bool
        True  → call mock_data_server REST API (requires server running on port 8001)
        False → call mock_data_server functions directly in-process (default for training)
    base_url : str
        Base URL of the mock server when use_mock_api=True.
    timeout_s : float
        HTTP request timeout in seconds (API mode only).
    use_real_data : bool
        If True, fetches real-time data from Open-Meteo API and Indian grid APIs.
    dc_city : str
        City name for weather data (e.g., "Bengaluru", "Delhi", "Mumbai").
    grid_region : str
        Indian grid region (e.g., "Western", "Northern", "Southern").
    """

    def __init__(
        self,
        use_mock_api: bool = False,
        base_url: str = "http://localhost:8001",
        timeout_s: float = 2.0,
        use_real_data: Optional[bool] = None,
        dc_city: Optional[str] = None,
        grid_region: Optional[str] = None,
    ) -> None:
        self._use_api = use_mock_api
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._session = None

        # Read from environment variables if not explicitly provided
        self._use_real_data = use_real_data if use_real_data is not None else \
            os.environ.get("USE_REAL_DATA", "false").lower() == "true"
        self._dc_city = dc_city or os.environ.get("DC_CITY", "Bengaluru")
        self._grid_region = grid_region or os.environ.get("GRID_REGION", "Western")

        if self._use_real_data:
            logger.info(
                "HybridSignalGenerator: Using REAL data mode for %s (%s)",
                self._dc_city, self._grid_region
            )
        elif use_mock_api:
            self._init_http_session()
        else:
            # Import the direct-call helpers (no server needed)
            self._init_direct()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _get_city_coordinates(self) -> tuple[float, float]:
        """Get latitude and longitude for Indian cities."""
        # Coordinates for major Indian cities
        city_coords = {
            "bengaluru": (12.9716, 77.5946),
            "bangalore": (12.9716, 77.5946),
            "delhi": (28.7041, 77.1025),
            "new delhi": (28.6139, 77.2090),
            "mumbai": (19.0760, 72.8777),
            "chennai": (13.0827, 80.2707),
            "kolkata": (22.5726, 88.3639),
            "hyderabad": (17.3850, 78.4867),
            "pune": (18.5204, 73.8567),
            "ahmedabad": (23.0225, 72.5714),
            "jaipur": (26.9124, 75.7873),
            "lucknow": (26.8467, 80.9462),
        }
        city_key = self._dc_city.lower().strip()
        return city_coords.get(city_key, (12.9716, 77.5946))  # Default: Bengaluru

    def _get_real_ambient_temp(self, hour: int, month: int) -> float:
        """Fetch real ambient temperature from Open-Meteo API."""
        import requests
        lat, lon = self._get_city_coordinates()
        try:
            # Open-Meteo free API (no key required)
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m"
                f"&timezone=Asia%2FKolkata"
                f"&past_days=1&forecast_days=1"
            )
            resp = requests.get(url, timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            
            # Get temperature for the specified hour
            times = data["hourly"]["time"]
            temps = data["hourly"]["temperature_2m"]
            
            # Find matching time index
            from datetime import datetime, timedelta
            now = datetime.now()
            target_date = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Search for closest match
            best_idx = 0
            best_diff = float('inf')
            for i, time_str in enumerate(times):
                time_dt = datetime.fromisoformat(time_str)
                diff = abs((time_dt - target_date).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            
            temp = temps[best_idx]
            logger.debug(f"Real ambient temp for {self._dc_city}: {temp:.1f}°C")
            return round(float(temp), 2)
        except Exception as e:
            logger.warning(f"Failed to fetch real temperature from Open-Meteo: {e}")
            # Fallback to simulated value
            return self._simulate_ambient_temp(hour, month)

    def _simulate_ambient_temp(self, hour: int, month: int) -> float:
        """Simulate realistic ambient temperature based on Indian climate patterns."""
        import random, math
        # Base monthly temperatures for India (°C)
        base_temps = {
            1: 18.0, 2: 21.0, 3: 26.0, 4: 30.0, 5: 33.0, 6: 32.0,
            7: 29.0, 8: 28.0, 9: 28.0, 10: 26.0, 11: 22.0, 12: 19.0
        }
        base = base_temps.get(month, 25.0)
        # Diurnal variation (cooler at night, warmer during day)
        diurnal = 6.0 * math.sin(math.pi * (hour - 6) / 12)
        noise = random.gauss(0, 0.5)
        return round(base + diurnal + noise, 2)

    def _get_real_carbon_intensity(self, month: int, hour: int) -> float:
        """Get carbon intensity for Indian grid region."""
        import random, math
        # Indian grid carbon intensity varies by region and time
        # Base values by region (gCO2/kWh) - approximate 2024 data
        region_carbon = {
            "western": 720.0,    # Maharashtra, Gujarat, MP, Goa
            "northern": 750.0,   # Delhi, UP, Punjab, Haryana, etc.
            "southern": 680.0,   # Karnataka, Tamil Nadu, Kerala, etc.
            "eastern": 820.0,    # West Bengal, Bihar, Odisha, etc.
            "northeastern": 650.0,  # NE states
        }
        region_key = self._grid_region.lower().strip()
        base = region_carbon.get(region_key, 720.0)
        
        # Solar reduces intensity during daytime
        solar_factors = [
            1.00, 1.00, 1.00, 1.00, 1.00, 0.98,  # 00-05
            0.95, 0.88, 0.80, 0.72, 0.68, 0.65,  # 06-11
            0.63, 0.65, 0.68, 0.75, 0.82, 0.90,  # 12-17
            0.95, 0.98, 1.00, 1.00, 1.00, 1.00,  # 18-23
        ]
        noise = random.gauss(0, base * 0.03)
        return round(float(max(300, min(900, base * solar_factors[hour % 24] + noise))), 2)

    def _init_http_session(self) -> None:
        try:
            import requests
            self._session = requests.Session()
            # Warm-up ping
            resp = self._session.get(f"{self._base_url}/health", timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.info(
                "HybridSignalGenerator connected to mock server. "
                "India benchmark PUE = %.3f", data.get("india_benchmark_pue", "?")
            )
        except Exception as exc:
            logger.warning(
                "HybridSignalGenerator: could not connect to mock server at %s (%s). "
                "Falling back to direct mode.", self._base_url, exc
            )
            self._use_api = False
            self._init_direct()

    def _init_direct(self) -> None:
        if self._use_real_data:
            # Use real-time data from APIs
            logger.info(
                "HybridSignalGenerator: Using REAL data mode for %s (%s)",
                self._dc_city, self._grid_region
            )
            self._get_all_signals = self._get_real_all_signals
            self._benchmark_pue = 1.58
        else:
            try:
                from mock_data_server import (
                    get_all_signals as _get_all_signals,
                    get_ambient_temp as _get_ambient_temp,
                    get_carbon_intensity as _get_carbon_intensity,
                    get_energy_price as _get_energy_price,
                    get_off_peak as _get_off_peak,
                    get_pue_benchmark as _get_pue_benchmark,
                    INDIA_BENCHMARK_PUE,
                )
                self._get_all_signals   = _get_all_signals
                self._get_ambient_temp  = _get_ambient_temp
                self._get_carbon_intensity = _get_carbon_intensity
                self._get_energy_price  = _get_energy_price
                self._get_off_peak      = _get_off_peak
                self._get_pue_benchmark = _get_pue_benchmark
                self._benchmark_pue     = INDIA_BENCHMARK_PUE
                logger.info(
                    "HybridSignalGenerator running in direct mode (no HTTP). "
                    "India benchmark PUE = %.3f", INDIA_BENCHMARK_PUE
                )
            except ImportError:
                logger.warning(
                    "mock_data_server not found; using fallback stub values."
                )
                self._get_all_signals = self._fallback_signals
                self._benchmark_pue = 1.58

    def _get_real_all_signals(self, hour: int, month: int) -> dict:
        """Get all signals using real-time APIs and realistic Indian data."""
        import random, math
        
        # Get ambient temperature from Open-Meteo API or fallback
        try:
            ambient = self._get_real_ambient_temp(hour, month)
            temp_source = "Open-Meteo API"
        except Exception as e:
            logger.warning(f"Using simulated temperature: {e}")
            ambient = self._simulate_ambient_temp(hour, month)
            temp_source = "Simulated (API failed)"
        
        # Get carbon intensity based on region
        carbon = self._get_real_carbon_intensity(month, hour)
        
        # Energy prices (Indian TOU rates)
        tou = [0.065]*6 + [0.10, 0.12, 0.14, 0.14, 0.14, 0.13,
                           0.13, 0.12, 0.13, 0.14, 0.16, 0.17,
                           0.17, 0.16, 0.14, 0.12, 0.10, 0.08]
        price = tou[hour % 24] + random.gauss(0, 0.002)
        
        # Off-peak hours (midnight to 6 AM)
        off_peak = 1 if (hour % 24) in range(0, 6) else 0
        
        # Demand response (evening peak hours 5-7 PM, 25% probability)
        dr = 1 if (hour in {17, 18, 19} and random.random() < 0.25) else 0
        
        # PUE benchmark
        pue_benchmark = 1.58
        
        return {
            "ambient_temp_c": ambient,
            "energy_price_per_kwh": round(max(0.04, min(0.20, price)), 4),
            "grid_carbon_intensity_g_per_kwh": carbon,
            "off_peak_window": off_peak,
            "demand_response_signal": dr,
            "pue_benchmark": pue_benchmark,
            "source": f"Real-time: {temp_source}, Grid: {self._grid_region}",
            "city": self._dc_city,
            "region": self._grid_region,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        hour: Optional[int] = None,
        month: Optional[int] = None,
    ) -> dict:
        """
        Return a dictionary with all grid signals for the given hour.

        Returns
        -------
        dict with keys:
            ambient_temp_c                  – °C
            energy_price_per_kwh            – $/kWh
            grid_carbon_intensity_g_per_kwh – gCO₂/kWh
            off_peak_window                 – 0 or 1
            demand_response_signal          – 0 or 1
            pue_benchmark                   – real-world PUE for comparison
        """
        hour  = hour  if hour  is not None else datetime.now().hour
        month = month if month is not None else datetime.now().month

        if self._use_api:
            logger.debug("HybridSignalGenerator: Fetching signals via Mock API (Port 8001)")
            signals = self._fetch_from_api(hour, month)
            signals["source"] = "Mock API (Port 8001)"
            return signals
        else:
            logger.debug("HybridSignalGenerator: Fetching signals via Direct Call (In-process)")
            signals = self._get_all_signals(hour=hour, month=month)
            signals["source"] = "Direct CSV Read"
            return signals

    @property
    def india_pue_benchmark(self) -> float:
        """Best-known real-world PUE for Indian data centres."""
        return getattr(self, "_benchmark_pue", 1.58)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_from_api(self, hour: int, month: int) -> dict:
        """Call the REST API and return combined signal dict."""
        import requests
        try:
            resp = self._session.get(
                f"{self._base_url}/signals",
                params={"hour": hour, "month": month},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "ambient_temp_c":                  data["ambient_temp_c"],
                "energy_price_per_kwh":             data["energy_price_per_kwh"],
                "grid_carbon_intensity_g_per_kwh":  data["grid_carbon_intensity_g_per_kwh"],
                "off_peak_window":                  data["off_peak_window"],
                "demand_response_signal":           data["demand_response_signal"],
                "pue_benchmark":                    data["pue_benchmark"],
            }
        except Exception as exc:
            logger.warning("Mock API call failed (%s); using fallback stub values.", exc)
            return self._fallback_signals(hour=hour, month=month)

    @staticmethod
    def _fallback_signals(hour: int = 12, month: int = 6, **_) -> dict:
        """Last-resort stub with reasonable India defaults."""
        import random, math
        logger.info("HybridSignalGenerator: Using synthetic fallback signals (Source: Internal Stubs)")
        tou = [0.065]*6 + [0.10, 0.12, 0.14, 0.14, 0.14, 0.13,
                           0.13, 0.12, 0.13, 0.14, 0.16, 0.17,
                           0.17, 0.16, 0.14, 0.12, 0.10, 0.08]
        return {
            "ambient_temp_c": 28.0 + 5.0 * math.sin(math.pi * (hour - 6) / 12),
            "energy_price_per_kwh": tou[hour % 24],
            "grid_carbon_intensity_g_per_kwh": 600.0 + 100.0 * random.random(),
            "off_peak_window": 1 if (hour < 6 or hour > 22) else 0,
            "demand_response_signal": 0,
            "pue_benchmark": 1.58,
            "source": "Internal Fallback Stubs"
        }
        h = hour % 24
        return {
            "ambient_temp_c":                  26.0 + 6 * math.sin((h - 6) * math.pi / 12),
            "energy_price_per_kwh":             tou[h] + random.gauss(0, 0.002),
            "grid_carbon_intensity_g_per_kwh":  720.0 * (1 + random.gauss(0, 0.03)),
            "off_peak_window":                  int(h in range(0, 6)),
            "demand_response_signal":           int(h in {17, 18, 19} and random.random() < 0.2),
            "pue_benchmark":                    1.58,
        }
