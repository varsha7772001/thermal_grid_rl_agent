# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import logging
from typing import List, Dict
import numpy as np
try:
    from .thermal_grid_rl_agent_environment import ThermalGridTaskID, TASKS
except (ImportError, ValueError):
    try:
        from thermal_grid_rl_agent_environment import ThermalGridTaskID, TASKS
    except ImportError:
        from server.thermal_grid_rl_agent_environment import ThermalGridTaskID, TASKS

logger = logging.getLogger(__name__)

class ThermalGridGrader:
    """
    Programmatic grader for the Thermal Grid RL environment.
    Evaluates episode performance and returns a normalized score [0.0, 1.0].
    """

    def __init__(self, task_id: str):
        try:
            self.task_id = ThermalGridTaskID(task_id)
        except ValueError:
            self.task_id = ThermalGridTaskID.BASELINE
        
        self.config = TASKS[self.task_id]
        self.history: List[Dict] = []

    def log_thermal_grid_step(self, observation, reward: float):
        """Record a step's data for final grading."""
        self.history.append({
            "pue": observation.pue,
            "reward": reward,
            "max_cpu_temp": max(observation.max_cpu_temps_c) if observation.max_cpu_temps_c else 0.0,
            "energy_price": observation.energy_price_per_kwh,
            "dr_signal": observation.demand_response_signal,
            "it_power": observation.total_it_power_w,
            "facility_power": observation.total_facility_power_w,
        })

    def get_thermal_grid_score(self) -> float:
        """
        Calculate final normalized score [0.0, 1.0] based on task-specific criteria.
        """
        if not self.history:
            return 0.0

        n_steps = len(self.history)
        avg_pue = np.mean([s["pue"] for s in self.history])
        avg_reward = np.mean([s["reward"] for s in self.history])
        max_temp_ever = max([s["max_cpu_temp"] for s in self.history])
        
        # 1. Safety Score (Common to all tasks)
        # 85C is max. We penalize heavily if exceeded.
        safety_score = max(0.0, 1.0 - max(0.0, max_temp_ever - 80.0) / 10.0)

        # 2. Efficiency Score (PUE focus)
        # Target 1.25 -> 1.0 score. 1.50 -> 0.0 score.
        pue_score = max(0.0, 1.0 - (avg_pue - 1.1) / 0.4)

        if self.task_id == ThermalGridTaskID.BASELINE:
            # Easy: Just stay safe and reasonably efficient
            return float(np.clip(0.6 * pue_score + 0.4 * safety_score, 0.0, 1.0))

        elif self.task_id == ThermalGridTaskID.LOAD_SHIFT:
            # Medium: Focus on reward (which includes cost) and PUE
            # We normalize reward to [0, 1] range roughly.
            # Base reward is 1.0 - cost. 1.0 is great, 0.0 is okay, negative is bad.
            reward_score = max(0.0, avg_reward) 
            return float(np.clip(0.5 * reward_score + 0.3 * pue_score + 0.2 * safety_score, 0.0, 1.0))

        elif self.task_id == ThermalGridTaskID.GRID_STRESS:
            # Hard: Focus on and safety under extreme heat
            # A reward score of 1.0 here is extremely hard.
            reward_score = max(0.0, avg_reward)
            # Extra penalty for safety violations in hard task
            hard_safety = safety_score if max_temp_ever < 85.0 else 0.0
            return float(np.clip(0.7 * reward_score + 0.3 * hard_safety, 0.0, 1.0))


def _run_grading_episodes(env, agent, task_id: str, num_episodes: int = 3, max_steps: int = 24) -> float:
    """Helper to run multiple evaluation episodes and return an average score in [0, 1]."""
    grader = ThermalGridGrader(task_id=task_id)
    for _ in range(num_episodes):
        observation = env.reset(task_id=task_id)
        for _ in range(max_steps):
            # Convert Pydantic observation to vectorized observation (as gym_env does)
            obs_vec = np.concatenate([
                np.array(observation.inlet_temps_c, dtype=np.float32),
                np.array(observation.mean_cpu_temps_c, dtype=np.float32),
                np.array(observation.max_cpu_temps_c, dtype=np.float32),
                np.array(observation.max_gpu_temps_c, dtype=np.float32),  # Added GPU monitoring (67 dims total)
                np.array([observation.thermal_mass_lag_c_per_min], dtype=np.float32),
                np.array(observation.rack_powers_w, dtype=np.float32) / 10000.0,
                np.array(observation.rack_utilisation, dtype=np.float32),
                np.array([observation.live_traffic_load_w], dtype=np.float32) / 10000.0,
                np.array([observation.pue], dtype=np.float32),
                np.array([observation.ambient_temp_c], dtype=np.float32),
                np.array([observation.energy_price_per_kwh], dtype=np.float32),
                np.array([observation.demand_response_signal], dtype=np.float32),
                np.array([observation.off_peak_window], dtype=np.float32),
            ]).astype(np.float32)

            # agent.predict is typical for stable-baselines3
            action_data = agent.predict(obs_vec)
            
            # Handle (action, state) or action return
            action_vec = action_data[0] if isinstance(action_data, tuple) else action_data
                
            # Convert vector action back to Pydantic for the env
            try:
                from ..models import ThermalGridRlAgentAction
            except (ImportError, ValueError):
                try:
                    from models import ThermalGridRlAgentAction
                except ImportError:
                    from thermal_grid_rl_agent.models import ThermalGridRlAgentAction
            tg_action = ThermalGridRlAgentAction(
                crac_setpoint_c=12.0 + action_vec[0] * 15.0,
                fan_speeds_pct=[20.0 + action_vec[1] * 80.0] * 10,
                num_active_chillers=int(1 + action_vec[2] * 3),
                batch_job_schedule=list(range(int(action_vec[3] * env._batch_queue.pending_count))),
                workload_matrix=[[0.5] * 8] * 10,
                power_caps_w=[[50.0 + action_vec[4] * 450.0] * 8] * 10,
                region_traffic_weights=[0.5, 0.5]
            )

            observation = env.step(tg_action)
            grader.log_thermal_grid_step(observation, getattr(observation, 'reward', 0.0))
            
            if getattr(observation, 'done', False):
                break
    return grader.get_thermal_grid_score()

def grade_baseline(env, agent) -> float:
    return _run_grading_episodes(env, agent, "baseline")

def grade_load_shift(env, agent) -> float:
    return _run_grading_episodes(env, agent, "load_shift")

def grade_grid_stress(env, agent) -> float:
    return _run_grading_episodes(env, agent, "grid_stress")
