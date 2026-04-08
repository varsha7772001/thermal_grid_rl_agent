# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Thermal Grid Rl Agent Environment Implementation.

Models a high-dimensional, non-linear datacenter + power-grid environment:

Key environment components
--------------------------
* IT Load (Servers)     — primary heat source; variable live traffic + batch jobs.
                          Agent can decide *where* and *when* load runs, not if.
* Cooling Infrastructure — CRACs, chillers (stacked), VFD fans, pumps.
                          Cooling = 30–50 % of total energy bill.
* Thermal Dynamics      — thermal mass introduces lag: temp changes slowly.
                          Requires predictive, not purely reactive, control.
* External Grid         — dynamic energy prices (TOU / 5-min RTP), carbon
                          intensity, and demand-response shed signals.

Agent action levers
-------------------
A. Cooling (continuous)
   • CRAC/CRAH setpoint adjustment (fraction of a degree)
   • Fan VFD speed — Power ∝ Speed³ (Affinity Laws)
   • Chiller stacking — more efficient at 80 % load than 20 %

B. Load distribution (discrete / structural)
   • Geographical Load Balancing (GLB) — route traffic to cheaper/cooler regions
   • Batch-job scheduling — defer non-latency-sensitive jobs to off-peak hours

Reward
------
Weighted combination of:
  • Energy efficiency  — low PUE, low $/kWh, low gCO₂/kWh
  • Thermal safety     — no CPU / inlet temperature overages
  • Grid cooperation   — shed load during demand-response events
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum
from typing import Dict, Optional

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation
except ImportError:
    from models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation

# ---------------------------------------------------------------------------
# Hybrid Signal Generator (real CSV data integration)
# ---------------------------------------------------------------------------
try:
    import sys, os
    # Allow importing from project root when running from server/ sub-directory
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from hybrid_signal_generator import HybridSignalGenerator
    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False
    HybridSignalGenerator = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Configuration
# ---------------------------------------------------------------------------

class ThermalGridTaskID(str, Enum):
    BASELINE = "baseline"
    LOAD_SHIFT = "load_shift"
    GRID_STRESS = "grid_stress"


@dataclass
class ThermalGridTaskConfig:
    """Configuration for a specific operational scenario."""
    id: ThermalGridTaskID
    name: str
    ambient_temp_c: float
    price_volatility: float          # Std dev of price scaling
    dr_event_probability: float      # Prob of demand-response signal
    batch_job_arrival_rate: float    # Jobs per 10 steps
    pue_target: float
    # Reward weights (α, β, γ) for the Joint Cost Function:
    # Total Cost = α × (Price × Energy) + β × (Cost of Inefficiency) + γ × (Thermal Risk)
    alpha: float  # Energy cost weight
    beta: float   # Inefficiency (PUE) weight
    gamma: float  # Safety/Thermal risk weight


TASKS: Dict[ThermalGridTaskID, ThermalGridTaskConfig] = {
    ThermalGridTaskID.BASELINE: ThermalGridTaskConfig(
        id=ThermalGridTaskID.BASELINE,
        name="Level 1: Baseline Efficiency (Easy)",
        ambient_temp_c=22.0,
        price_volatility=0.02,
        dr_event_probability=0.02,
        batch_job_arrival_rate=1.0,
        pue_target=1.25,
        alpha=0.4, beta=0.3, gamma=0.3,
    ),
    ThermalGridTaskID.LOAD_SHIFT: ThermalGridTaskConfig(
        id=ThermalGridTaskID.LOAD_SHIFT,
        name="Level 2: Temporal Load Shifting (Medium)",
        ambient_temp_c=25.0,
        price_volatility=0.15,       # High price swings
        dr_event_probability=0.05,
        batch_job_arrival_rate=5.0,  # More batch jobs to shift
        pue_target=1.20,
        alpha=0.6, beta=0.2, gamma=0.2, # Focus on cost (alpha)
    ),
    ThermalGridTaskID.GRID_STRESS: ThermalGridTaskConfig(
        id=ThermalGridTaskID.GRID_STRESS,
        name="Level 3: Grid Resilience & Safety (Hard)",
        ambient_temp_c=38.0,         # Heat wave!
        price_volatility=0.25,
        dr_event_probability=0.40,   # Frequent DR signals
        batch_job_arrival_rate=2.0,
        pue_target=1.15,
        alpha=0.2, beta=0.2, gamma=0.6, # Focus on safety/reliability (gamma)
    ),
}


# ---------------------------------------------------------------------------
# Internal simulator data classes
# ---------------------------------------------------------------------------

@dataclass
class ThermalGridRackState:
    rack_id: str
    inlet_temp_c: float          # cold-aisle inlet temperature
    cpu_temps_c: list[float]     # per-server CPU junction temps
    gpu_temps_c: list[float]     # per-server GPU temps
    power_w: float               # total rack IT power
    utilisation: float           # mean server utilisation 0-1
    live_power_w: float          # portion from live traffic
    batch_power_w: float         # portion from batch jobs


@dataclass
class ThermalGridChillerState:
    num_active: int              # how many chiller units are running
    load_pct: float              # mean load across active units (0-100)
    power_w: float               # total chiller power draw


@dataclass
class ThermalGridGridState:
    energy_price_per_kwh: float          # $/kWh (TOU or 5-min RTP)
    carbon_intensity_g_per_kwh: float    # gCO₂/kWh
    demand_response_signal: int          # 1 = shed load requested
    off_peak_window: int                 # 1 = currently off-peak hours
    hour_of_day: int                     # 0-23


@dataclass
class ThermalGridDataCenterState:
    timestamp_s: float
    racks: list[ThermalGridRackState]
    chiller: ThermalGridChillerState
    grid: ThermalGridGridState
    crac_supply_temp_c: float
    crac_power_w: float
    avg_fan_speed_pct: float
    total_it_power_w: float
    total_facility_power_w: float
    pue: float
    safety_override_triggered: bool      # True if failsafe cooling was activated
    prev_avg_inlet_temp_c: float         # for thermal-mass lag calculation


# ---------------------------------------------------------------------------
# External grid signal generator (stub — replace with real grid API / data)
# ---------------------------------------------------------------------------

class ThermalGridSignalGenerator:
    """
    Generates realistic grid signals: dynamic pricing, carbon intensity,
    demand-response events, and off-peak windows.

    Replace with a real grid api or historical price-trace reader.
    """

    # Hourly TOU price profile ($/kWh) — 24 values, index = hour of day
    _TOU_PRICES: list[float] = [
        0.06, 0.06, 0.06, 0.06, 0.06, 0.07,  # 00-05
        0.09, 0.12, 0.15, 0.16, 0.16, 0.15,  # 06-11
        0.14, 0.14, 0.15, 0.16, 0.18, 0.20,  # 12-17 (peak)
        0.20, 0.19, 0.17, 0.13, 0.09, 0.07,  # 18-23
    ]
    # Hourly carbon intensity (gCO₂/kWh) — lower midday due to solar
    _CARBON_PROFILE: list[float] = [
        420, 415, 410, 408, 405, 400,  # 00-05
        390, 370, 340, 310, 280, 260,  # 06-11 (solar ramp)
        240, 230, 235, 250, 290, 340,  # 12-17
        380, 400, 415, 420, 425, 422,  # 18-23
    ]
    _OFF_PEAK_HOURS: set[int] = {0, 1, 2, 3, 4, 5}
    _DR_EVENT_HOURS: set[int] = {17, 18, 19}  # evening peak

    def __init__(self, seed: int = 42, config: Optional[ThermalGridTaskConfig] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._config = config or TASKS[ThermalGridTaskID.BASELINE]

    def get(
        self,
        timestamp_s: float,
        hybrid_override: Optional[dict] = None,
    ) -> ThermalGridGridState:
        """
        Generate grid state.
        If hybrid_override is provided (from HybridSignalGenerator), the real-data
        values replace the synthetic profile values for price, carbon, DR, and off-peak.
        """
        hour = int((timestamp_s / 3600) % 24)

        if hybrid_override:
            # ---- Use real CSV-derived values --------------------------------
            price     = float(hybrid_override.get("energy_price_per_kwh", self._TOU_PRICES[hour]))
            carbon    = float(hybrid_override.get("grid_carbon_intensity_g_per_kwh",
                                                   self._CARBON_PROFILE[hour]))
            dr_signal = int(hybrid_override.get("demand_response_signal", 0))
            off_peak  = int(hybrid_override.get("off_peak_window", int(hour in self._OFF_PEAK_HOURS)))
            # Still apply task-level price volatility on top of the real base
            price_noise = self._rng.normal(0, self._config.price_volatility * 0.5)
            price = float(np.clip(price * (1 + price_noise), 0.03, 0.40))
            carbon = float(np.clip(carbon * (1 + self._rng.normal(0, 0.02)), 100, 900))
        else:
            # ---- Fallback: synthetic profile ---------------------------------
            price_scale = self._rng.normal(0, self._config.price_volatility)
            price = self._TOU_PRICES[hour] * (1 + price_scale)
            carbon = self._CARBON_PROFILE[hour] * (1 + self._rng.normal(0, 0.03))
            dr_prob   = self._config.dr_event_probability if hour in self._DR_EVENT_HOURS else 0.01
            dr_signal = int(self._rng.random() < dr_prob)
            off_peak  = int(hour in self._OFF_PEAK_HOURS)
            price  = float(np.clip(price,  0.03, 0.40))
            carbon = float(np.clip(carbon, 100,  600))

        return ThermalGridGridState(
            energy_price_per_kwh=price,
            carbon_intensity_g_per_kwh=carbon,
            demand_response_signal=dr_signal,
            off_peak_window=off_peak,
            hour_of_day=hour,
        )


# ---------------------------------------------------------------------------
# Batch job queue (stub)
# ---------------------------------------------------------------------------

@dataclass
class ThermalGridJobQueue:
    """Tracks deferrable batch jobs and their power profiles."""
    _pending: list[float] = field(default_factory=list)  # power (W) per job

    def enqueue(self, jobs_power_w: list[float]) -> None:
        self._pending.extend(jobs_power_w)

    def run(self, indices: list[int]) -> float:
        """Run selected jobs (by index), remove from queue, return power used."""
        indices_set = set(i for i in indices if 0 <= i < len(self._pending))
        power = sum(self._pending[i] for i in indices_set)
        self._pending = [p for i, p in enumerate(self._pending) if i not in indices_set]
        return power

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()


# ---------------------------------------------------------------------------
# Datacenter thermal simulator (DCcluster-Opt shim)
# ---------------------------------------------------------------------------

class ThermalGridSimulator:
    """
    Thin wrapper around the DCcluster-Opt thermal simulator.

    Implements:
      • IT load model: live traffic + batch jobs, with per-server power caps
      • Thermal dynamics with thermal mass (lag): temperature changes gradually
      • Cooling model: CRAC setpoint, VFD fans (Power ∝ Speed³), chiller stacking
      • Chiller efficiency curve: optimal around 75-85 % load

    To wire in the real library:
      1. Uncomment the two TODO lines in __init__
      2. Replace _compute_state() body with self._sim.get_state()
    """

    CPU_TEMP_MAX_C: float = 85.0
    INLET_TEMP_MAX_C: float = 27.0
    CRAC_SETPOINT_MIN_C: float = 12.0
    CRAC_SETPOINT_MAX_C: float = 27.0
    FAN_SPEED_MIN_PCT: float = 20.0
    FAN_SPEED_MAX_PCT: float = 100.0
    MAX_CHILLERS: int = 4
    # Thermal constants
    THERMAL_MASS_TAU = 12.0  # Time constant for inlet temperature lag (steps)
    SAFETY_THRESHOLD_CPU_C = 80.0  # Buffer before 85°C critical limit
    CRAC_FAILSAFE_SETPOINT_C = 15.0  # Emergency CRAC setpoint

    def __init__(
        self,
        num_racks: int = 10,
        servers_per_rack: int = 8,
        num_regions: int = 1,
        config_path: str | None = None,
        seed: int = 42,
        config: Optional[ThermalGridTaskConfig] = None,
    ) -> None:
        self._num_racks = num_racks
        self._servers_per_rack = servers_per_rack
        self._num_regions = num_regions
        self._rng = np.random.default_rng(seed)
        self._config = config or TASKS[ThermalGridTaskID.BASELINE]
        self._time_s: float = 0.0
        self._dt_s: float = 60.0  # 1-minute steps

        # Control state
        # Initial temp set to ambient from task config
        self._crac_setpoint_c: float = self._config.ambient_temp_c - 4.0
        self._fan_speeds_pct: np.ndarray = np.full(num_racks, 60.0)
        self._power_caps_w: np.ndarray = np.full((num_racks, servers_per_rack), 300.0)
        self._workload: np.ndarray = np.zeros((num_racks, servers_per_rack))
        self._num_active_chillers: int = 2

        # Thermal mass: track previous inlet temps for lag model
        self._prev_inlet_temps: np.ndarray = np.full(num_racks, 20.0)
        self._prev_state: ThermalGridDataCenterState | None = None
        self._safety_triggered: bool = False

        # Live traffic baseline per rack (W) — varies with user demand
        self._live_load_w: np.ndarray = np.full(num_racks, 800.0)

        # TODO: uncomment when dccluster_opt is installed
        # from dccluster_opt import DataCenterSimulator
        # self._sim = DataCenterSimulator(config_path=config_path)

    def reset(self) -> ThermalGridDataCenterState:
        self._time_s = 0.0
        # Start at a reasonable setpoint relative to task ambient
        # Phase 1: Dynamic setpoint drift
        self._crac_setpoint_c = max(18.0, self._config.ambient_temp_c - 10.0) + self._rng.uniform(-1.0, 1.0)
        logger.info("[DynamicSim] Phase 1 enabled: Injecting real-world fluctuations.")
        self._fan_speeds_pct = np.full(self._num_racks, 60.0)
        self._power_caps_w = np.full((self._num_racks, self._servers_per_rack), 300.0)
        self._num_active_chillers = 2
        # Randomized initial workloads
        self._workload = self._rng.uniform(0.1, 0.7, (self._num_racks, self._servers_per_rack))
        self._live_load_w = self._rng.uniform(400, 1200, self._num_racks)
        # Initial room temp drift
        self._prev_inlet_temps = np.full(self._num_racks, self._crac_setpoint_c + self._rng.uniform(1.0, 4.0))
        self._prev_state = None
        return self._compute_state(batch_power_per_rack=np.zeros(self._num_racks))

    def step(
        self,
        crac_setpoint_c: float,
        fan_speeds_pct: np.ndarray,
        power_caps_w: np.ndarray,
        workload_matrix: np.ndarray,
        num_active_chillers: int,
        batch_power_per_rack: np.ndarray,
        region_traffic_weights: np.ndarray,
        ambient_temp_c: Optional[float] = None,
    ) -> ThermalGridDataCenterState:
        """
        Advance simulator by one timestep, applying all control inputs.

        The two levers interact: redistributing load via region_traffic_weights
        changes per-rack thermal profiles, requiring cooling re-adjustment.
        """
        # ------------------------------------------------------------------
        # HARD SAFETY LAYER (Failsafe Cooling)
        # ------------------------------------------------------------------
        self._safety_triggered = False
        
        # Check previous state for any thermal risks
        if self._prev_state is not None:
            for rack in self._prev_state.racks:
                rack_idx = int(rack.rack_id.split("_")[1])
                max_cpu = max(rack.cpu_temps_c)
                
                # If rack is overheating, override its fan to 100%
                if max_cpu > self.SAFETY_THRESHOLD_CPU_C:
                    fan_speeds_pct[rack_idx] = 100.0
                    self._safety_triggered = True
            
            # If global max is very high, force CRAC setpoint down
            all_cpus = [t for r in self._prev_state.racks for t in r.cpu_temps_c]
            if max(all_cpus) > self.SAFETY_THRESHOLD_CPU_C + 2.0:
                crac_setpoint_c = self.CRAC_FAILSAFE_SETPOINT_C
                self._safety_triggered = True

        # Clamp continuous cooling inputs
        self._crac_setpoint_c = float(
            np.clip(crac_setpoint_c, self.CRAC_SETPOINT_MIN_C, self.CRAC_SETPOINT_MAX_C)
        )
        self._fan_speeds_pct = np.clip(fan_speeds_pct, self.FAN_SPEED_MIN_PCT, self.FAN_SPEED_MAX_PCT)
        self._power_caps_w = np.clip(power_caps_w, 50.0, 500.0)
        self._num_active_chillers = int(np.clip(num_active_chillers, 1, self.MAX_CHILLERS))

        # Apply geographical load balancing: redistribute live traffic across racks
        weights = np.clip(region_traffic_weights, 0.0, 1.0)
        weights = weights / (weights.sum() + 1e-9)
        total_live = self._live_load_w.sum()
        # Distribute live load proportionally across racks via GLB weights
        rack_glb_factor = np.interp(
            np.arange(self._num_racks),
            np.linspace(0, self._num_racks - 1, len(weights)),
            weights,
        )
        rack_glb_factor = rack_glb_factor / (rack_glb_factor.sum() + 1e-9)
        self._live_load_w = rack_glb_factor * total_live

        # Simulate natural traffic variation (Boosted to ±15% for Phase 1)
        self._live_load_w *= 1.0 + self._rng.uniform(-0.15, 0.15, self._num_racks)
        self._live_load_w = np.clip(self._live_load_w, 200.0, 2000.0)

        self._workload = np.clip(workload_matrix, 0.0, 1.0)
        self._time_s += self._dt_s

        return self._compute_state(
            batch_power_per_rack=batch_power_per_rack,
            ambient_temp_c=ambient_temp_c,
        )

    def _compute_state(
        self,
        batch_power_per_rack: np.ndarray,
        ambient_temp_c: Optional[float] = None,
    ) -> ThermalGridDataCenterState:
        # Use provided ambient or fallback to task config
        amb_c = ambient_temp_c if ambient_temp_c is not None else self._config.ambient_temp_c

        # TODO: replace with self._sim.get_state() when dccluster_opt is installed
        racks: list[ThermalGridRackState] = []
        total_it_power = 0.0
        new_inlet_temps = np.zeros(self._num_racks)

        for r in range(self._num_racks):
            fan = self._fan_speeds_pct[r]
            # VFD fan: cooling effectiveness rises non-linearly with speed
            cooling_effectiveness = 0.4 + 0.6 * (fan / 100.0) ** 0.8

            server_cpu_temps: list[float] = []
            server_gpu_temps: list[float] = []
            rack_power = 0.0
            batch_power = float(batch_power_per_rack[r]) if r < len(batch_power_per_rack) else 0.0
            live_power = float(self._live_load_w[r])
            extra_heat_per_server = (live_power + batch_power) / self._servers_per_rack

            for s in range(self._servers_per_rack):
                util = self._workload[r, s]
                cap = self._power_caps_w[r, s]
                server_power = util * cap
                rack_power += server_power

                # Thermal resistance model with jitter (W/°C)
                k = 12.0 + self._rng.normal(0, 0.4) 
                cpu_temp_instant = (
                    self._crac_setpoint_c
                    + (server_power + extra_heat_per_server) / (cooling_effectiveness * k)
                    + self._rng.normal(0, 1.2)
                )
                server_cpu_temps.append(float(np.clip(cpu_temp_instant, 20.0, 105.0)))
                # GPUs are intrinsically hotter and more volatile
                gpu_jitter = self._rng.normal(0, 1.8)
                server_gpu_temps.append(float(np.clip(cpu_temp_instant + 5.0 + gpu_jitter, 20.0, 115.0)))

            rack_power += live_power
            rack_power += batch_power
            total_it_power += rack_power

            # Thermal mass lag: inlet temp moves toward equilibrium slowly
            # T_eq = setpoint + rack_heat_load / (airflow * cooling_effectiveness)
            t_eq = self._crac_setpoint_c + (rack_power / (1000.0 * cooling_effectiveness))
            t_lag = (
                self._prev_inlet_temps[r]
                + (t_eq - self._prev_inlet_temps[r]) / self.THERMAL_MASS_TAU
            )
            new_inlet_temps[r] = float(np.clip(t_lag, 15.0, 50.0))

            racks.append(ThermalGridRackState(
                rack_id=f"rack_{r:02d}",
                inlet_temp_c=float(new_inlet_temps[r]),
                cpu_temps_c=server_cpu_temps,
                gpu_temps_c=server_gpu_temps,
                power_w=float(rack_power),
                utilisation=float(np.mean(self._workload[r])),
                live_power_w=live_power,
                batch_power_w=batch_power,
            ))

        self._prev_inlet_temps = new_inlet_temps

        # CRAC power: rises when setpoint is lower, rises with fan speed.
        # Also increases with ambient temperature (cooling is harder).
        avg_fan = float(np.mean(self._fan_speeds_pct))
        amb_impact = 1.0 + max(0, (amb_c - 25.0) * 0.02)  # +2% power per °C above 25°C
        crac_power = (
            total_it_power * 0.3
            * (1.0 - (self._crac_setpoint_c - 12.0) / 20.0)
            * (avg_fan / 100.0) ** 1.5
            * amb_impact
        )

        # Chiller stacking: efficiency curve peaks at ~80 % load
        chiller_capacity_w = total_it_power * 0.4  # total cooling load
        chiller_load_pct = float(
            np.clip(
                (chiller_capacity_w / max(self._num_active_chillers, 1)) / 5000.0 * 100.0,
                0.0, 100.0,
            )
        )
        # Chiller COP degrades below 40 % and above 95 % load.
        # Also degrades as ambient temperature rises (condenser temperature).
        chiller_cop_base = 3.5 * np.exp(-0.5 * ((chiller_load_pct - 80.0) / 40.0) ** 2) + 1.5
        amb_cop_penalty = max(0, (amb_c - 25.0) * 0.05)  # lose 0.05 COP per °C above 25°C
        chiller_cop = max(1.1, chiller_cop_base - amb_cop_penalty)
        chiller_power = float(chiller_capacity_w / max(chiller_cop, 0.5))

        total_facility_power = total_it_power + crac_power + chiller_power + self._rng.normal(0, 75.0)
        pue = total_facility_power / max(total_it_power, 1.0)

        # Store for next step failsafe check
        state = ThermalGridDataCenterState(
            timestamp_s=self._time_s,
            racks=racks,
            chiller=ThermalGridChillerState(
                num_active=self._num_active_chillers,
                load_pct=chiller_load_pct,
                power_w=chiller_power,
            ),
            grid=ThermalGridGridState(0.0, 0.0, 0, 0, 0),  # filled by environment
            crac_supply_temp_c=self._crac_setpoint_c,
            crac_power_w=float(crac_power),
            avg_fan_speed_pct=avg_fan,
            total_it_power_w=float(total_it_power),
            total_facility_power_w=float(total_facility_power),
            pue=float(pue),
            safety_override_triggered=self._safety_triggered,
            prev_avg_inlet_temp_c=float(np.mean(self._prev_inlet_temps)),
        )
        self._prev_state = state
        return state


# ---------------------------------------------------------------------------
# Reward
def _compute_thermal_grid_reward(
    dc_state: ThermalGridDataCenterState,
    config: ThermalGridTaskConfig,
) -> float:
    """
    Implements the Joint Cost Function specified by the user:
    Total Cost = α × (Electricity Price × Total Energy Use) + β × (Cost of Poor PUE) + γ × (Thermal Risk Penalty)

    The RL agent maximizes reward (Reward = -Total Cost).
    We normalize components to ensure a useful reward signal.
    """
    g = dc_state.grid
    dt_hours = 60.0 / 3600.0  # 1-minute step

    # ---- 1. Direct Electricity Cost (α component) -----------------------
    # Energy in kWh: Power (W) * time (h) / 1000
    energy_kwh = (dc_state.total_facility_power_w * dt_hours) / 1000.0
    direct_cost = g.energy_price_per_kwh * energy_kwh
    # Normalize direct cost: typical rack power 10kW * 10 racks = 100kW facility.
    # 100kW * (1/60)h = 1.66 kWh. Price $0.10. Cost ~$0.17.
    # We'll scale this to ~1.0 for the reward component.
    alpha_cost = direct_cost * 5.0

    # ---- 2. Cost of Poor PUE (β component) ------------------------------
    # Measure of inefficiency relative to target
    pue_inefficiency = max(0.0, dc_state.pue - config.pue_target)
    # Cost scales with IT load — it's more expensive to be inefficient at high load
    beta_cost = pue_inefficiency * (dc_state.total_it_power_w / 1000.0) * dt_hours

    # ---- 3. Thermal Risk Penalty (γ component) ---------------------------
    # Large penalty for any violation of safe operation limits
    cpu_violation = max(0.0, max(max(r.cpu_temps_c) for r in dc_state.racks) - ThermalGridSimulator.CPU_TEMP_MAX_C)
    inlet_violation = max(0.0, max(r.inlet_temp_c for r in dc_state.racks) - ThermalGridSimulator.INLET_TEMP_MAX_C)
    # Binary penalty + degree of violation
    thermal_risk = 0.0
    if cpu_violation > 0 or inlet_violation > 0:
        thermal_risk = 1.0 + 0.1 * cpu_violation + 0.2 * inlet_violation

    # ---- Aggregate Joint Cost Function ----------------------------------
    # Total Cost = α*(Price*Energy) + β*(Inefficiency) + γ*(Thermal Risk)
    # We normalize components so that "optimal" is ~0.0 cost.
    total_cost = (
        config.alpha * (alpha_cost * 1.0)
        + config.beta * (beta_cost * 5.0)  # scale β to be comparable to α
        + config.gamma * (thermal_risk * 0.5) # scale γ so violations are dominant but not explosive
    )

    # Reward = -Total Cost.
    # We clip the final reward to [-50, 50] to ensure stable gradients
    return float(np.clip(1.0 - total_cost, -50.0, 50.0))


# ---------------------------------------------------------------------------
# Environment
# Environment
# ---------------------------------------------------------------------------

class ThermalGridRlAgentEnvironment(Environment):
    """
    RL environment modelling a full datacenter + power-grid system.

    Environment components
    ----------------------
    * IT Load (Servers)      — variable live traffic + deferrable batch jobs.
                               Agent controls placement, not whether a job runs.
    * Cooling Infrastructure — CRACs, VFD fans, chiller stacking.
    * Thermal Dynamics       — thermal mass introduces temperature lag; rewards
                               predictive control over reactive control.
    * External Grid          — dynamic pricing, carbon intensity, demand response.

    Action levers
    -------------
    A. Cooling (continuous): crac_setpoint_c, fan_speeds_pct, num_active_chillers
    B. Load (discrete):      region_traffic_weights (GLB), batch_job_schedule

    Example:
        >>> env = ThermalGridRlAgentEnvironment()
        >>> obs = env.reset()
        >>> print(obs.step_summary)  # "Thermal Grid Rl Agent environment ready!"
        >>>
        >>> obs = env.step(ThermalGridRlAgentAction(
        ...     crac_setpoint_c=19.0,
        ...     fan_speeds_pct=[70.0] * 10,
        ...     num_active_chillers=2,
        ...     region_traffic_weights=[0.6, 0.4],
        ...     batch_job_schedule=[1, 0, 1],
        ...     workload_matrix=[[0.5] * 8] * 10,
        ...     power_caps_w=[[300.0] * 8] * 10,
        ... ))
        >>> print(obs.pue)
        >>> print(obs.energy_price_per_kwh)
        >>> print(obs.demand_response_signal)
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        num_racks: int = 10,
        servers_per_rack: int = 8,
        num_regions: int = 2,
        config_path: str | None = None,
        simulator_seed: int = 42,
        max_steps: int = 1000,
        task_id: str = "baseline",
        use_hybrid_signals: bool = True,
        use_mock_api: bool = False,
    ):
        """Initialize the thermal_grid_rl_agent environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._max_steps = max_steps
        self._num_regions = num_regions

        try:
            tid = ThermalGridTaskID(task_id)
        except ValueError:
            logger.warning("Unknown task_id '%s', defaulting to baseline", task_id)
            tid = ThermalGridTaskID.BASELINE

        self._task_config = TASKS[tid]

        self._simulator = ThermalGridSimulator(
            num_racks=num_racks,
            servers_per_rack=servers_per_rack,
            num_regions=num_regions,
            config_path=config_path,
            seed=simulator_seed,
            config=self._task_config,
        )
        self._grid = ThermalGridSignalGenerator(seed=simulator_seed, config=self._task_config)
        self._batch_queue = ThermalGridJobQueue()

        # ---- Hybrid Signal Generator (real CSV data) ----------------------
        self._hybrid_gen: Optional[HybridSignalGenerator] = None
        if use_hybrid_signals and _HYBRID_AVAILABLE:
            try:
                self._hybrid_gen = HybridSignalGenerator(
                    use_mock_api=use_mock_api,
                    base_url="http://localhost:8001",
                )
                logger.info(
                    "HybridSignalGenerator active (API=%s, real_data=%s). "
                    "India benchmark PUE = %.3f",
                    use_mock_api, 
                    getattr(self._hybrid_gen, '_use_real_data', False),
                    self._hybrid_gen.india_pue_benchmark,
                )
            except Exception as exc:
                logger.warning("Could not init HybridSignalGenerator: %s", exc)
                self._hybrid_gen = None
        elif use_hybrid_signals:
            logger.warning(
                "use_hybrid_signals=True but HybridSignalGenerator not found. "
                "Falling back to synthetic signals."
            )

        # Accumulated PUE for episode-end benchmark comparison
        self._episode_pue_sum: float = 0.0
        self._episode_pue_count: int = 0

    def reset(self, task_id: Optional[str] = None) -> ThermalGridRlAgentObservation:
        """
        Reset the environment.

        Args:
            task_id: Optional task_id to switch scenarios during reset.
                     If None, continues with the current task.

        Returns:
            ThermalGridRlAgentObservation with a ready message
        """
        if task_id is not None:
            try:
                tid = ThermalGridTaskID(task_id)
                self._task_config = TASKS[tid]
                # Re-initialize task-dependent components
                self._simulator = ThermalGridSimulator(
                    num_racks=self._simulator._num_racks,
                    servers_per_rack=self._simulator._servers_per_rack,
                    num_regions=self._num_regions,
                    seed=int(self._simulator._rng.integers(0, 1000000)),
                    config=self._task_config,
                )
                self._grid = ThermalGridSignalGenerator(
                    seed=int(self._simulator._rng.integers(0, 1000000)),
                    config=self._task_config
                )
                logger.info("Environment switched to task: %s", task_id)
            except ValueError:
                logger.warning("Unknown task_id '%s', continuing with current task", task_id)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._batch_queue.reset()
        # Seed the batch queue with initial jobs
        self._batch_queue.enqueue(self._simulator._rng.uniform(200, 800, 20).tolist())

        dc_state = self._simulator.reset()

        # Fetch hybrid signals for step-0
        _hybrid_signals: Optional[dict] = None
        if self._hybrid_gen is not None:
            current_hour = int((dc_state.timestamp_s / 3600) % 24)
            _hybrid_signals = self._hybrid_gen.get(hour=current_hour)

        grid_state = self._grid.get(dc_state.timestamp_s, hybrid_override=_hybrid_signals)
        dc_state.grid = grid_state

        # Reset episode PUE accumulator
        self._episode_pue_sum   = 0.0
        self._episode_pue_count = 0

        logger.info(
            "Episode %s started. Initial PUE=%.3f, Price=%.3f $/kWh",
            self._state.episode_id, dc_state.pue, grid_state.energy_price_per_kwh,
        )

        return self._build_thermal_grid_observation(
            dc_state=dc_state,
            step_summary="Thermal Grid Rl Agent environment ready!",
            reward=0.0,
            done=False,
        )

    def step(self, action: ThermalGridRlAgentAction) -> ThermalGridRlAgentObservation:  # type: ignore[override]
        """
        Execute a step in the environment using the thermal simulator.

        Args:
            action: ThermalGridRlAgentAction with both cooling and load levers:
                    - crac_setpoint_c, fan_speeds_pct, num_active_chillers (cooling)
                    - region_traffic_weights, batch_job_schedule (load distribution)
                    - workload_matrix, power_caps_w (placement fine-control)

        Returns:
            ThermalGridRlAgentObservation with full datacenter + grid state and reward
        """
        self._state.step_count += 1
        n_racks = self._simulator._num_racks
        n_srv = self._simulator._servers_per_rack

        # ---- Parse & validate cooling inputs ----------------------------
        fan_speeds = np.array(action.fan_speeds_pct, dtype=float)
        if fan_speeds.shape != (n_racks,):
            fan_speeds = np.full(n_racks, 60.0)

        power_caps = np.array(action.power_caps_w, dtype=float)
        if power_caps.shape != (n_racks, n_srv):
            power_caps = np.full((n_racks, n_srv), 300.0)

        workload = np.array(action.workload_matrix, dtype=float)
        if workload.shape != (n_racks, n_srv):
            workload = np.full((n_racks, n_srv), 0.5)

        # ---- Parse load distribution inputs -----------------------------
        region_weights = np.array(
            action.region_traffic_weights if action.region_traffic_weights else [1.0],
            dtype=float,
        )

        # Batch job scheduling: run selected jobs, accumulate power per rack
        batch_power_total = self._batch_queue.run(action.batch_job_schedule)
        batch_power_per_rack = np.full(n_racks, batch_power_total / max(n_racks, 1))

        # Enqueue new batch jobs occasionally (simulates incoming batch workload)
        if self._state.step_count % 10 == 0:
            new_jobs = self._simulator._rng.uniform(200, 600, 5).tolist()
            self._batch_queue.enqueue(new_jobs)

        # ---- Advance simulator -------------------------------------------
        # ---- Attach grid state (with optional real-data hybrid override) ----
        _hybrid_signals: Optional[dict] = None
        if self._hybrid_gen is not None:
            current_hour = int((self._simulator._time_s / 3600) % 24)
            _hybrid_signals = self._hybrid_gen.get(hour=current_hour)
        
        current_amb = _hybrid_signals.get("ambient_temp_c") if _hybrid_signals else None

        dc_state = self._simulator.step(
            crac_setpoint_c=action.crac_setpoint_c,
            fan_speeds_pct=fan_speeds,
            power_caps_w=power_caps,
            workload_matrix=workload,
            num_active_chillers=action.num_active_chillers,
            batch_power_per_rack=batch_power_per_rack,
            region_traffic_weights=region_weights,
            ambient_temp_c=current_amb,
        )

        grid_state = self._grid.get(dc_state.timestamp_s, hybrid_override=_hybrid_signals)
        dc_state.grid = grid_state

        # ---- Reward & termination ---------------------------------------
        reward = float(_compute_thermal_grid_reward(dc_state, self._task_config))
        done = self._state.step_count >= self._max_steps

        # Accumulate PUE for episode-end comparison
        self._episode_pue_sum   += dc_state.pue
        self._episode_pue_count += 1

        if done:
            avg_pue = (
                self._episode_pue_sum / self._episode_pue_count
                if self._episode_pue_count else float("nan")
            )
            benchmark_pue = (
                self._hybrid_gen.india_pue_benchmark
                if self._hybrid_gen else 1.58
            )
            logger.info(
                "Episode %s ended after %d steps. "
                "Achieved PUE: %.3f | India Benchmark: %.3f | Delta: %+.3f",
                self._state.episode_id,
                self._state.step_count,
                avg_pue,
                benchmark_pue,
                avg_pue - benchmark_pue,
            )
            print(
                f"\n[ThermalGrid] Episode Summary "
                f"(ep={self._state.episode_id[:8]})\n"
                f"  Achieved PUE : {avg_pue:.3f}\n"
                f"  India Benchmark PUE : {benchmark_pue:.3f}\n"
                f"  Delta        : {avg_pue - benchmark_pue:+.3f} "
                f"({'better' if avg_pue < benchmark_pue else 'worse'} than benchmark)\n"
            )

        summary = (
            f"Step {self._state.step_count}: "
            f"PUE={dc_state.pue:.3f}, "
            f"IT={dc_state.total_it_power_w / 1000:.1f} kW, "
            f"MaxCPU={max(max(r.cpu_temps_c) for r in dc_state.racks):.1f}°C, "
            f"Price={grid_state.energy_price_per_kwh:.3f}$/kWh, "
            f"DR={grid_state.demand_response_signal}, "
            f"Reward={reward:.4f}"
        )

        return self._build_thermal_grid_observation(
            dc_state=dc_state,
            step_summary=summary,
            reward=reward,
            done=done,
            _hybrid_signals=_hybrid_signals,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _build_thermal_grid_observation(
        self,
        dc_state: ThermalGridDataCenterState,
        step_summary: str,
        reward: float,
        done: bool,
        _hybrid_signals: Optional[dict] = None,
    ) -> ThermalGridRlAgentObservation:
        g = dc_state.grid

        # Thermal mass lag: rate of inlet temp change (°C/min)
        current_avg_inlet = float(np.mean([r.inlet_temp_c for r in dc_state.racks]))
        thermal_lag = (current_avg_inlet - dc_state.prev_avg_inlet_temp_c) / (
            self._simulator._dt_s / 60.0
        )

        return ThermalGridRlAgentObservation(
            # Summary
            step_summary=step_summary,
            # 1. Thermal state
            inlet_temps_c=[r.inlet_temp_c for r in dc_state.racks],
            mean_cpu_temps_c=[float(np.mean(r.cpu_temps_c)) for r in dc_state.racks],
            max_cpu_temps_c=[float(np.max(r.cpu_temps_c)) for r in dc_state.racks],
            max_gpu_temps_c=[float(np.max(r.gpu_temps_c)) for r in dc_state.racks],
            thermal_mass_lag_c_per_min=float(thermal_lag),
            # 2. IT load
            rack_powers_w=[r.power_w for r in dc_state.racks],
            rack_utilisation=[r.utilisation for r in dc_state.racks],
            live_traffic_load_w=sum(r.live_power_w for r in dc_state.racks),
            deferred_batch_load_w=sum(r.batch_power_w for r in dc_state.racks),
            pending_batch_jobs=self._batch_queue.pending_count,
            # 3. Cooling state
            pue=dc_state.pue,
            total_it_power_w=dc_state.total_it_power_w,
            total_facility_power_w=dc_state.total_facility_power_w,
            crac_power_w=dc_state.crac_power_w,
            chiller_power_w=dc_state.chiller.power_w,
            num_active_chillers=dc_state.chiller.num_active,
            chiller_load_pct=dc_state.chiller.load_pct,
            crac_supply_temp_c=dc_state.crac_supply_temp_c,
            avg_fan_speed_pct=dc_state.avg_fan_speed_pct,
            ambient_temp_c=float(_hybrid_signals.get("ambient_temp_c", self._simulator._config.ambient_temp_c)) if _hybrid_signals else self._simulator._config.ambient_temp_c,
            # 4. External grid
            energy_price_per_kwh=g.energy_price_per_kwh,
            grid_carbon_intensity_g_per_kwh=g.carbon_intensity_g_per_kwh,
            demand_response_signal=g.demand_response_signal,
            off_peak_window=g.off_peak_window,
            # Episode metadata
            done=done,
            reward=reward,
            safety_override_triggered=dc_state.safety_override_triggered,
            metadata={
                "step": self._state.step_count,
                "episode_id": self._state.episode_id,
                "hour_of_day": g.hour_of_day,
                "data_source": _hybrid_signals.get("source", "Synthetic (Legacy)") if _hybrid_signals else "Simulator Default",
                "sim_phase": "1: High Volatility (Visible Intelligence)",
            },
        )