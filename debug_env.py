"""Debug script to see what the environment actually returns."""
import asyncio
import json
from client import ThermalGridRlAgentEnv
from models import ThermalGridRlAgentAction
from server.thermal_grid_rl_agent_environment import ThermalGridTaskID

ENV_URL = "http://localhost:8000"

async def debug_task(task_id: ThermalGridTaskID, steps: int = 5):
    print(f"\n{'='*80}")
    print(f"DEBUGGING TASK: {task_id.value}")
    print(f"{'='*80}")
    
    env = ThermalGridRlAgentEnv(base_url=ENV_URL)
    
    try:
        reset_result = await env.reset(task_id=task_id.value)
        obs = reset_result.observation
        
        print(f"\n--- INITIAL OBSERVATION ---")
        print(f"  max_cpu_temps_c: {obs.max_cpu_temps_c}")
        print(f"  max_gpu_temps_c: {obs.max_gpu_temps_c}")
        print(f"  mean_cpu_temps_c: {obs.mean_cpu_temps_c}")
        print(f"  inlet_temps_c: {obs.inlet_temps_c}")
        print(f"  pue: {obs.pue}")
        print(f"  crac_supply_temp_c: {obs.crac_supply_temp_c}")
        print(f"  avg_fan_speed_pct: {obs.avg_fan_speed_pct}")
        print(f"  num_active_chillers: {obs.num_active_chillers}")
        print(f"  energy_price_per_kwh: {obs.energy_price_per_kwh}")
        print(f"  demand_response_signal: {obs.demand_response_signal}")
        print(f"  thermal_mass_lag_c_per_min: {obs.thermal_mass_lag_c_per_min}")
        print(f"  total_it_power_w: {obs.total_it_power_w}")
        print(f"  total_facility_power_w: {obs.total_facility_power_w}")
        print(f"  pending_batch_jobs: {obs.pending_batch_jobs}")
        print(f"  ambient_temp_c: {obs.ambient_temp_c}")
        
        for step in range(1, steps + 1):
            action = ThermalGridRlAgentAction(
                crac_setpoint_c=18.0,
                fan_speeds_pct=[70.0] * 10,
                num_active_chillers=2,
            )
            
            result = await env.step(action)
            obs = result.observation
            
            print(f"\n--- STEP {step} ---")
            print(f"  max_cpu: {max(obs.max_cpu_temps_c):.2f}°C")
            print(f"  max_gpu: {max(obs.max_gpu_temps_c):.2f}°C")
            print(f"  pue: {obs.pue:.3f}")
            print(f"  price: ${obs.energy_price_per_kwh:.3f}/kWh")
            print(f"  DR signal: {obs.demand_response_signal}")
            print(f"  thermal_lag: {obs.thermal_mass_lag_c_per_min:.3f} °C/min")
            print(f"  reward: {result.reward:.4f}")
            print(f"  done: {result.done}")
            if step < steps:
                print(f"  [Continuing...]")
        
    finally:
        await env.close()

async def main():
    for task_id in [ThermalGridTaskID.BASELINE, ThermalGridTaskID.LOAD_SHIFT, ThermalGridTaskID.GRID_STRESS]:
        await debug_task(task_id, steps=8)

if __name__ == "__main__":
    asyncio.run(main())
