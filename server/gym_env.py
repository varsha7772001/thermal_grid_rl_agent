import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

try:
    from .thermal_grid_rl_agent_environment import ThermalGridRlAgentEnvironment, ThermalGridTaskID
    from ..models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation
except (ImportError, ValueError):
    try:
        from thermal_grid_rl_agent_environment import ThermalGridRlAgentEnvironment, ThermalGridTaskID
        from models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation
    except ImportError:
        from server.thermal_grid_rl_agent_environment import ThermalGridRlAgentEnvironment, ThermalGridTaskID
        from thermal_grid_rl_agent.models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation

class ThermalGridGymEnv(gym.Env):
    """
    Gymnasium wrapper for ThermalGridRlAgentEnvironment.
    Simplifies action/observation spaces for efficient RL training.
    """
    def __init__(
        self,
        task_id: str = "baseline",
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        use_hybrid_signals: bool = True,
        use_mock_api: bool = False,
    ):
        super().__init__()
        self.env = ThermalGridRlAgentEnvironment(
            task_id=task_id,
            max_steps=max_steps,
            use_hybrid_signals=use_hybrid_signals,
            use_mock_api=use_mock_api,
        )
        
        # Observation Space: 67-dimensional vector
        # [inlet_temps(10), mean_cpu(10), max_cpu(10), max_gpu(10), lag(1), 
        #  rack_powers(10), rack_util(10), live_load(1), pue(1), amb_temp(1), price(1), dr(1), off_peak(1)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(67,), dtype=np.float32
        )
        
        # Action Space: 5-dimensional continuous vector [0, 1]
        # [crac_setpoint, fan_speed, chillers, batch_fraction, power_cap]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

    def _get_obs_vector(self, obs: ThermalGridRlAgentObservation) -> np.ndarray:
        return np.concatenate([
            np.array(obs.inlet_temps_c, dtype=np.float32),
            np.array(obs.mean_cpu_temps_c, dtype=np.float32),
            np.array(obs.max_cpu_temps_c, dtype=np.float32),
            np.array(obs.max_gpu_temps_c, dtype=np.float32),
            np.array([obs.thermal_mass_lag_c_per_min], dtype=np.float32),
            np.array(obs.rack_powers_w, dtype=np.float32) / 10000.0,  # Normalize W to 10kW units
            np.array(obs.rack_utilisation, dtype=np.float32),
            np.array([obs.live_traffic_load_w], dtype=np.float32) / 10000.0,
            np.array([obs.pue], dtype=np.float32),
            np.array([obs.ambient_temp_c], dtype=np.float32),
            np.array([obs.energy_price_per_kwh], dtype=np.float32),
            np.array([obs.demand_response_signal], dtype=np.float32),
            np.array([obs.off_peak_window], dtype=np.float32),
        ]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.env._simulator._rng = np.random.default_rng(seed)
        
        # Allow switching tasks via options
        task_id = options.get("task_id") if options else None
        obs_pydantic = self.env.reset(task_id=task_id)
        
        return self._get_obs_vector(obs_pydantic), {"summary": obs_pydantic.step_summary}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map [0, 1] action vector back to ThermalGridRlAgentAction
        crac_setpoint = 12.0 + action[0] * 15.0  # [12, 27]
        fan_speed = 20.0 + action[1] * 80.0     # [20, 100]
        num_chillers = int(1 + action[2] * 3)   # [1, 4]
        
        # Batch jobs: run fraction of pending
        pending_count = self.env._batch_queue.pending_count
        num_to_run = int(action[3] * pending_count)
        batch_indices = list(range(num_to_run))
        
        power_cap = 50.0 + action[4] * 450.0     # [50, 500]
        
        # Build pydantic action
        tg_action = ThermalGridRlAgentAction(
            crac_setpoint_c=crac_setpoint,
            fan_speeds_pct=[fan_speed] * 10,
            num_active_chillers=num_chillers,
            batch_job_schedule=batch_indices,
            workload_matrix=[[0.5] * 8] * 10,  # Fixed util for now
            power_caps_w=[[power_cap] * 8] * 10,
            region_traffic_weights=[0.5, 0.5]
        )
        
        # Advance environment
        obs_pydantic = self.env.step(tg_action)
        
        obs_vec = self._get_obs_vector(obs_pydantic)
        reward = float(obs_pydantic.reward if hasattr(obs_pydantic, "reward") else 0.0)
        
        # Gymnasium 0.26+ expects (obs, reward, terminated, truncated, info)
        # Terminated: The Episode is actually finished (e.g., failure or goal reached)
        # Truncated: The Time limit was reached
        terminated = False # Our simulator doesn't have a "fail" state that ends the ep early yet
        truncated = obs_pydantic.done if hasattr(obs_pydantic, "done") else False
        
        return obs_vec, reward, terminated, truncated, {"summary": obs_pydantic.step_summary}

    def render(self):
        pass

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
