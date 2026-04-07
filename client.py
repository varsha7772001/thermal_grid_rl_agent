# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thermal Grid Rl Agent Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation
except (ImportError, ValueError):
    from models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation


class ThermalGridRlAgentEnv(
    EnvClient[ThermalGridRlAgentAction, ThermalGridRlAgentObservation, State]
):
    """
    Client for the Thermal Grid Rl Agent Environment.
    Each client instance has its own dedicated environment session on the server.
    """

    async def reset(self, **kwargs) -> StepResult[ThermalGridRlAgentObservation]:
        """Reset the environment with optional parameters (e.g. task_id)."""
        return await super().reset(**kwargs)

    def _step_payload(self, action: ThermalGridRlAgentAction) -> Dict:
        """Convert action object to JSON payload."""
        return {
            "crac_setpoint_c": action.crac_setpoint_c,
            "fan_speeds_pct": action.fan_speeds_pct,
            "num_active_chillers": action.num_active_chillers,
            "region_traffic_weights": action.region_traffic_weights,
            "batch_job_schedule": action.batch_job_schedule,
            "workload_matrix": action.workload_matrix,
            "power_caps_w": action.power_caps_w,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ThermalGridRlAgentObservation]:
        """Parse server response into Observation object with GPU/Ambient support."""
        obs_data = payload.get("observation", {})

        observation = ThermalGridRlAgentObservation(
            step_summary=obs_data.get("step_summary", ""),

            # 1. Thermal state
            inlet_temps_c=obs_data.get("inlet_temps_c", []),
            mean_cpu_temps_c=obs_data.get("mean_cpu_temps_c", []),
            max_cpu_temps_c=obs_data.get("max_cpu_temps_c", []),
            max_gpu_temps_c=obs_data.get("max_gpu_temps_c", []),
            thermal_mass_lag_c_per_min=obs_data.get("thermal_mass_lag_c_per_min", 0.0),

            # 2. IT load
            rack_powers_w=obs_data.get("rack_powers_w", []),
            rack_utilisation=obs_data.get("rack_utilisation", []),
            live_traffic_load_w=obs_data.get("live_traffic_load_w", 0.0),
            deferred_batch_load_w=obs_data.get("deferred_batch_load_w", 0.0),
            pending_batch_jobs=obs_data.get("pending_batch_jobs", 0),

            # 3. Cooling state
            pue=obs_data.get("pue", 0.0),
            total_it_power_w=obs_data.get("total_it_power_w", 0.0),
            total_facility_power_w=obs_data.get("total_facility_power_w", 0.0),
            crac_power_w=obs_data.get("crac_power_w", 0.0),
            chiller_power_w=obs_data.get("chiller_power_w", 0.0),
            num_active_chillers=obs_data.get("num_active_chillers", 0),
            chiller_load_pct=obs_data.get("chiller_load_pct", 0.0),
            crac_supply_temp_c=obs_data.get("crac_supply_temp_c", 0.0),
            avg_fan_speed_pct=obs_data.get("avg_fan_speed_pct", 0.0),

            # 4. External grid
            ambient_temp_c=obs_data.get("ambient_temp_c", 25.0),
            energy_price_per_kwh=obs_data.get("energy_price_per_kwh", 0.0),
            grid_carbon_intensity_g_per_kwh=obs_data.get("grid_carbon_intensity_g_per_kwh", 0.0),
            demand_response_signal=obs_data.get("demand_response_signal", 0),
            off_peak_window=obs_data.get("off_peak_window", 0),
            safety_override_triggered=obs_data.get("safety_override_triggered", False),

            # Episode
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )