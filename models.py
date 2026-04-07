# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Thermal Grid Rl Agent Environment.

The agent interacts with a high-dimensional datacenter + power-grid environment.
It observes thermal state, IT load, and external grid signals, then controls:
  A. Cooling (continuous) — CRAC setpoints, fan speeds (VFD), chiller stacking
  B. Load distribution (discrete/structural) — geographical load balancing (GLB)
     and batch-job scheduling
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ThermalGridRlAgentAction(Action):
    """
    Control action for the Thermal Grid RL Agent.

    Two distinct levers that interact: redistributing load changes the thermal
    profile, which in turn requires adjustments to cooling.

    A. Cooling Control (Continuous)
    --------------------------------
    crac_setpoint_c   : CRAC/CRAH supply-air temperature setpoint (°C).
    fan_speeds_pct    : Per-rack Variable Frequency Drive fan speed 0–100 %.
                        Power scales as Speed³ (Affinity Laws), so small
                        reductions yield large savings.
    num_active_chillers: How many chiller units to run. A chiller at 20 %
                        load is far less efficient than one at 80 % load.

    B. Load Distribution (Discrete / Structural)
    ---------------------------------------------
    region_traffic_weights : Fraction of incoming user traffic routed to each
                             region (must sum to 1.0). Enables GLB — shift
                             traffic away from hot/expensive regions.
    batch_job_schedule     : Per-batch-job flag — 1 = run now, 0 = defer to
                             off-peak. Enables temporal load shifting.
    workload_matrix        : Per-server utilisation after placement (0–1),
                             shape [num_racks][servers_per_rack].
    power_caps_w           : Per-server power cap in Watts.
    """

    # ---- A. Cooling (continuous) -----------------------------------------
    crac_setpoint_c: float = Field(
        default=18.0,
        description="CRAC/CRAH supply-air temperature setpoint in °C (12.0–27.0)",
    )
    fan_speeds_pct: list[float] = Field(
        default_factory=lambda: [60.0] * 10,
        description=(
            "Per-rack fan speed (0–100 %). Power ∝ Speed³ (Affinity Laws). "
            "One value per rack."
        ),
    )
    num_active_chillers: int = Field(
        default=2,
        description=(
            "Number of chiller units to activate (1–max_chillers). "
            "Fewer chillers at higher load is more efficient than many at low load."
        ),
    )

    # ---- B. Load distribution (discrete / structural) --------------------
    region_traffic_weights: list[float] = Field(
        default_factory=lambda: [1.0],
        description=(
            "Fraction of live user traffic routed to each region (sums to 1.0). "
            "Geographical Load Balancing (GLB) lever."
        ),
    )
    batch_job_schedule: list[int] = Field(
        default_factory=list,
        description=(
            "Per-batch-job scheduling flag: 1 = run this step, 0 = defer to "
            "off-peak. Enables temporal load shifting."
        ),
    )
    workload_matrix: list[list[float]] = Field(
        default_factory=lambda: [[0.5] * 8] * 10,
        description=(
            "Per-server utilisation after placement decisions (0–1). "
            "Shape [num_racks][servers_per_rack]."
        ),
    )
    power_caps_w: list[list[float]] = Field(
        default_factory=lambda: [[300.0] * 8] * 10,
        description="Per-server power cap in Watts. Shape [num_racks][servers_per_rack].",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ThermalGridRlAgentObservation(Observation):
    """
    Observation returned by the Thermal Grid RL Agent environment.

    Groups into four sections mirroring the environment's key components:
      1. Thermal state   — rack inlet/CPU temps, thermal mass lag indicator
      2. IT load         — per-rack power, utilisation, live vs batch split
      3. Cooling state   — PUE, CRAC/chiller status, fan speeds
      4. External grid   — energy price, carbon intensity, demand-response flag
    """

    step_summary: str = Field(default="", description="Step summary string")

    # ---- 1. Thermal state ------------------------------------------------
    inlet_temps_c: list[float] = Field(
        default_factory=list,
        description="Per-rack cold-aisle inlet temperature in °C",
    )
    mean_cpu_temps_c: list[float] = Field(
        default_factory=list,
        description="Per-rack mean CPU junction temperature in °C",
    )
    max_cpu_temps_c: list[float] = Field(
        default_factory=list,
        description="Per-rack maximum CPU junction temperature in °C",
    )
    max_gpu_temps_c: list[float] = Field(
        default_factory=list,
        description="Per-rack maximum GPU junction temperature in °C",
    )
    thermal_mass_lag_c_per_min: float = Field(
        default=0.0,
        description=(
            "Estimated rate of temperature change (°C/min) due to thermal mass. "
            "Positive = room heating up; negative = cooling down. "
            "Enables predictive rather than purely reactive control."
        ),
    )

    # ---- 2. IT load ------------------------------------------------------
    rack_powers_w: list[float] = Field(
        default_factory=list,
        description="Per-rack total IT power draw in Watts",
    )
    rack_utilisation: list[float] = Field(
        default_factory=list,
        description="Per-rack mean server utilisation (0–1)",
    )
    live_traffic_load_w: float = Field(
        default=0.0,
        description="Power attributable to live user-facing traffic (Watts)",
    )
    deferred_batch_load_w: float = Field(
        default=0.0,
        description="Power attributable to batch jobs running this step (Watts)",
    )
    pending_batch_jobs: int = Field(
        default=0,
        description="Number of batch jobs currently queued / deferred",
    )

    # ---- 3. Cooling state ------------------------------------------------
    pue: float = Field(
        default=0.0,
        description="Power Usage Effectiveness = total facility power / IT power",
    )
    total_it_power_w: float = Field(default=0.0, description="Total IT power draw in Watts")
    # ---- Episode metadata ------------------------------------------------
    done: bool = Field(default=False, description="1 if the episode has ended")
    reward: float = Field(default=0.0, description="Reward from the environment")
    safety_override_triggered: bool = Field(
        default=False,
        description="True if the Hard Safety Layer overrode agent actions to prevent damage",
    )
    metadata: dict = Field(default_factory=dict, description="Metadata dictionary")
    total_facility_power_w: float = Field(default=0.0, description="Total facility power in Watts")
    crac_power_w: float = Field(default=0.0, description="CRAC/CRAH unit power draw in Watts")
    chiller_power_w: float = Field(default=0.0, description="Active chiller unit power draw in Watts")
    num_active_chillers: int = Field(
        default=0,
        description="Number of chiller units currently running",
    )
    chiller_load_pct: float = Field(
        default=0.0,
        description=(
            "Mean load fraction across active chillers (0–100 %). "
            "Low values (< 40 %) indicate inefficient part-load operation."
        ),
    )
    crac_supply_temp_c: float = Field(
        default=0.0,
        description="Current CRAC supply-air temperature setpoint in °C",
    )
    avg_fan_speed_pct: float = Field(
        default=0.0,
        description="Average VFD fan speed across all racks (0–100 %)",
    )

    # ---- 4. External grid ------------------------------------------------
    ambient_temp_c: float = Field(
        default=25.0,
        description="Current ambient air temperature in °C. Affects chiller and CRAC efficiency.",
    )
    energy_price_per_kwh: float = Field(
        default=0.0,
        description=(
            "Current electricity price ($/kWh). Changes hourly (TOU) or every "
            "5 minutes (Real-Time Pricing). Key signal for temporal load shifting."
        ),
    )
    grid_carbon_intensity_g_per_kwh: float = Field(
        default=0.0,
        description=(
            "Grid carbon intensity (gCO₂/kWh). Low during solar/wind peaks, "
            "high during fossil-fuel peaks."
        ),
    )
    demand_response_signal: int = Field(
        default=0,
        description=(
            "Binary grid demand-response request: 1 = grid requests load shed "
            "to prevent instability, 0 = normal operation."
        ),
    )
    off_peak_window: int = Field(
        default=0,
        description=(
            "1 if current time falls within off-peak pricing window "
            "(e.g. 02:00–05:00), 0 otherwise. Used for batch job scheduling."
        ),
    )