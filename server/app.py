# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Thermal Grid Rl Agent Environment.

This module creates an HTTP server that exposes the ThermalGridRlAgentEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment and return initial thermal state
    - POST /step: Execute a control action (cooling + load distribution)
    - GET /state: Get current environment state (episode_id, step_count)
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Action fields (POST /step):
    Cooling (continuous):
        crac_setpoint_c        — CRAC supply-air setpoint in °C (12–27)
        fan_speeds_pct         — Per-rack VFD fan speed 0–100 % (Power ∝ Speed³)
        num_active_chillers    — Chiller units to run (efficient at 75–85 % load)
    Load distribution (discrete):
        region_traffic_weights — GLB traffic fractions per region (sums to 1.0)
        batch_job_schedule     — Indices of batch jobs to run this step
        workload_matrix        — Per-server utilisation [num_racks][servers_per_rack]
        power_caps_w           — Per-server power cap in Watts

Observation fields (response):
    Thermal: inlet_temps_c, mean/max_cpu_temps_c, thermal_mass_lag_c_per_min
    IT load: rack_powers_w, rack_utilisation, live_traffic_load_w,
             deferred_batch_load_w, pending_batch_jobs
    Cooling: pue, total_it/facility_power_w, crac_power_w, chiller_power_w,
             num_active_chillers, chiller_load_pct, crac_supply_temp_c,
             avg_fan_speed_pct
    Grid:    energy_price_per_kwh, grid_carbon_intensity_g_per_kwh,
             demand_response_signal, off_peak_window

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import sys as _sys
import os as _os
# Ensure the project root (parent of this server/ directory) is always on
# sys.path so that `models` is importable regardless of CWD / launch method.
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_server_dir = _os.path.dirname(_os.path.abspath(__file__))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)
# Also add server/ so the bare-name fallback import works in Docker
if _server_dir not in _sys.path:
    _sys.path.insert(0, _server_dir)

from fastapi.middleware.cors import CORSMiddleware
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation
    from .thermal_grid_rl_agent_environment import ThermalGridRlAgentEnvironment
except (ModuleNotFoundError, ImportError):
    from models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation
    from thermal_grid_rl_agent_environment import ThermalGridRlAgentEnvironment


# Create the app with web interface and README integration.
# SUPPORTS_CONCURRENT_SESSIONS = True on the environment, so increasing
# max_concurrent_envs allows multiple RL agents to train in parallel,
# each with an isolated simulator instance (thermal state, batch queue,
# grid signals) via factory mode.
app = create_app(
    ThermalGridRlAgentEnvironment,
    ThermalGridRlAgentAction,
    ThermalGridRlAgentObservation,
    env_name="thermal_grid_rl_agent",
    max_concurrent_envs=10,  # increase further for large-scale parallel training
)

# Add CORS middleware for Windows-WSL and remote communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8000
        python -m thermal_grid_rl_agent.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn thermal_grid_rl_agent.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()