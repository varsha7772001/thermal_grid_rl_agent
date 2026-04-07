"""
Hybrid Signal Generator
=======================
Provides real-world-grounded grid signals to the RL environment.

Two modes:
  use_mock_api=True  → HTTP calls to the mock_data_server running on localhost:8001
  use_mock_api=False → Direct in-process calls (no HTTP needed, ideal for training)

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
    """

    def __init__(
        self,
        use_mock_api: bool = False,
        base_url: str = "http://localhost:8001",
        timeout_s: float = 2.0,
    ) -> None:
        self._use_api = use_mock_api
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._session = None

        if use_mock_api:
            self._init_http_session()
        else:
            # Import the direct-call helpers (no server needed)
            self._init_direct()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

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
