"""
Inference Script for Thermal Grid RL Agent Environment
========================================================
Runs an intelligent adaptive agent that uses LLM reasoning with rule-based safety fallback.

The agent:
1. Calls LLM with full observation state for intelligent decisions
2. Falls back to adaptive rule-based control if LLM fails
3. Adapts actions every step based on real environment values
4. Optimizes for task-specific grading criteria

Environment Variables:
    API_BASE_URL   - The API endpoint for the LLM (provided by OpenEnv validator)
    API_KEY        - API key for the LLM proxy (provided by OpenEnv validator)
    HF_TOKEN       - Your Hugging Face / API key (fallback if API_KEY not provided)
    MODEL_NAME     - The model identifier to use for inference
    ENV_URL        - URL of the thermal grid environment server (default: http://localhost:8000)
"""

import os
import re
import json
import asyncio
import logging
from typing import List, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

from client import ThermalGridRlAgentEnv
from models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation
from server.thermal_grid_rl_agent_environment import ThermalGridTaskID
from server.grader import ThermalGridGrader

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Validator injects API_BASE_URL and API_KEY — use them for LiteLLM proxy
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Use validator's API_KEY if available, otherwise fallback to HF_TOKEN
API_KEY = os.getenv("API_KEY") or HF_TOKEN

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK = "thermal_grid_rl_agent"
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.1

# All 3 tasks to evaluate
TASKS = [
    ThermalGridTaskID.BASELINE,
    ThermalGridTaskID.LOAD_SHIFT,
    ThermalGridTaskID.GRID_STRESS,
]

if API_BASE_URL and API_KEY:
    # Initialize OpenAI client with LiteLLM proxy (validator requirement)
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    logger.info(f"OpenAI client initialized with API_BASE_URL: {API_BASE_URL}")
else:
    client = None
    logger.warning("No API_BASE_URL or API_KEY/HF_TOKEN found. Agent will use rule-based control only.")


# ============================================================================
# LLM-BASED INTELLIGENT AGENT
# ============================================================================

SYSTEM_PROMPT = """You are an expert datacenter cooling control agent. Your goal is to maximize the reward by dynamically adjusting cooling based on current conditions.

You must respond with a JSON object containing EXACTLY these keys:
{"crac_setpoint_c": 16.0, "fan_speeds_pct": [75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0], "num_active_chillers": 3}

Rules:
- crac_setpoint_c: 12-27°C (lower = more cooling, higher = save energy)
- fan_speeds_pct: 20-100% (exactly 10 values, one per rack). Power scales as speed³, so small reductions save huge energy.
- num_active_chillers: 1-4 (fewer chillers at higher load is more efficient than many at low load)

Decision Guidelines:
- If max CPU > 60°C: INCREASE cooling (lower crac, raise fans, add chillers)
- If max CPU < 50°C and PUE > 1.20: REDUCE cooling to improve PUE (raise crac, lower fans)
- If demand_response (DR) = 1: Reduce energy consumption while keeping CPU < 85°C
- If energy price > $0.08/kWh: Favor energy-efficient settings (higher crac, lower fans, fewer chillers)
- If ambient > 35°C: You MUST use aggressive cooling (crac ≤ 15°C, fans ≥ 80%, 3+ chillers)
- If temperatures are moderate (50-58°C): Use balanced settings (crac 16-18°C, fans 70-75%, 2-3 chillers)

CRITICAL: Do NOT repeat the same action if reward dropped or is below 0.90. Adjust based on what the environment tells you.
Adapt every step — the environment state changes, and your action should too.

Respond with ONLY the JSON object. No markdown, no explanation."""


def build_user_message(obs: ThermalGridRlAgentObservation, step: int, task: str) -> str:
    """Build informative user message with current state."""
    max_cpu = max(obs.max_cpu_temps_c) if obs.max_cpu_temps_c else 0.0
    max_gpu = max(obs.max_gpu_temps_c) if obs.max_gpu_temps_c else 0.0
    avg_cpu = sum(obs.mean_cpu_temps_c) / len(obs.mean_cpu_temps_c) if obs.mean_cpu_temps_c else 0.0
    
    return f"""STEP {step}/{MAX_STEPS} | Task: {task}

CURRENT STATE:
- Thermal: max_cpu={max_cpu:.1f}°C, avg_cpu={avg_cpu:.1f}°C, max_gpu={max_gpu:.1f}°C
- Cooling: PUE={obs.pue:.3f}, setpoint={obs.crac_supply_temp_c:.1f}°C, fans={obs.avg_fan_speed_pct:.0f}%, chillers={obs.num_active_chillers}
- Power: IT={obs.total_it_power_w/1000:.1f}kW, Facility={obs.total_facility_power_w/1000:.1f}kW
- Grid: price=${obs.energy_price_per_kwh:.3f}/kWh, DR={obs.demand_response_signal}, ambient={obs.ambient_temp_c:.1f}°C
- Batch: {obs.pending_batch_jobs} jobs pending, off_peak={obs.off_peak_window}

TASK GRADING CRITERIA:
- baseline: 60% PUE score + 40% safety (keep temps < 80°C, PUE < 1.25)
- load_shift: 50% reward + 30% PUE + 20% safety (optimize cost with price signals)
- grid_stress: 70% reward + 30% safety (safety critical under 38°C ambient heat)

Respond with JSON containing your action."""


def _extract_json(raw: str) -> dict:
    """Try json.loads first, then regex extraction as fallback."""
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


async def get_llm_action(obs: ThermalGridRlAgentObservation, step: int, task_id: ThermalGridTaskID) -> Optional[ThermalGridRlAgentAction]:
    """Get action from LLM. Returns None if LLM fails."""
    if client is None:
        return None
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(obs, step, task_id.value)},
            ],
            max_tokens=128,
            temperature=0.1,
        )
        
        raw_content = response.choices[0].message.content or "{}"
        data = _extract_json(raw_content)
        
        if not data:
            logger.warning(f"LLM returned empty response at step {step}")
            return None
        
        # Extract action fields with validation
        crac = data.get("crac_setpoint_c", 18.0)
        fans = data.get("fan_speeds_pct", [70.0] * 10)
        chillers = data.get("num_active_chillers", 2)
        
        # Validate ranges
        crac = max(12.0, min(27.0, float(crac)))
        fans = [max(20.0, min(100.0, float(f))) for f in fans]
        chillers = max(1, min(4, int(chillers)))
        
        # Ensure 10 fan speeds
        if len(fans) != 10:
            fans = [fans[0] if fans else 70.0] * 10
        
        return ThermalGridRlAgentAction(
            crac_setpoint_c=crac,
            fan_speeds_pct=fans,
            num_active_chillers=chillers,
        )
        
    except Exception as e:
        logger.warning(f"LLM call failed at step {step}: {e}")
        return None


# ============================================================================
# ADAPTIVE RULE-BASED AGENT (Fallback + No-LLM Mode)
# ============================================================================

class RuleBasedAgent:
    """
    Adaptive rule-based agent that responds to ACTUAL environment values.
    Uses realistic thresholds based on observed environment behavior.
    """
    
    # Observed environment ranges:
    # CPU: 48-66°C, PUE: 1.15-1.21, Price: $0.06-0.08, Ambient: 22-38°C
    
    def __init__(self, task_id: ThermalGridTaskID):
        self.task_id = task_id
    
    def decide(self, obs: ThermalGridRlAgentObservation, step: int) -> ThermalGridRlAgentAction:
        max_cpu = max(obs.max_cpu_temps_c) if obs.max_cpu_temps_c else 50.0
        pue = obs.pue if obs.pue > 1.0 else 1.20
        price = obs.energy_price_per_kwh
        dr = obs.demand_response_signal
        ambient = obs.ambient_temp_c
        
        # === TASK-SPECIFIC STRATEGIES ===
        
        if self.task_id == ThermalGridTaskID.GRID_STRESS:
            # 38°C ambient, gamma=0.6 safety weight
            # Initial CPU is 64°C (high), need proactive cooling
            if max_cpu > 62.0 or step <= 3:
                # High initial temps from 38°C ambient - cool aggressively
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=14.0,
                    fan_speeds_pct=[85.0] * 10,
                    num_active_chillers=3,
                )
            elif dr == 1:
                # Demand response: balance safety with load reduction
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=15.0,
                    fan_speeds_pct=[80.0] * 10,
                    num_active_chillers=3,
                )
            elif max_cpu > 55.0:
                # Moderate cooling to maintain safety
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=15.0,
                    fan_speeds_pct=[75.0] * 10,
                    num_active_chillers=3,
                )
            else:
                # Steady state: efficient but safe
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=16.0,
                    fan_speeds_pct=[72.0] * 10,
                    num_active_chillers=3,
                )
        
        elif self.task_id == ThermalGridTaskID.LOAD_SHIFT:
            # 25°C ambient, alpha=0.6 cost weight
            if price > 0.07:
                # High price: reduce cooling consumption
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=18.0,
                    fan_speeds_pct=[68.0] * 10,
                    num_active_chillers=2,
                )
            elif price < 0.06:
                # Low price: can afford more cooling
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=16.0,
                    fan_speeds_pct=[75.0] * 10,
                    num_active_chillers=2,
                )
            elif max_cpu > 52.0:
                # CPU warming: increase cooling
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=16.0,
                    fan_speeds_pct=[76.0] * 10,
                    num_active_chillers=2,
                )
            else:
                # Moderate: balance cost and cooling
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=17.0,
                    fan_speeds_pct=[72.0] * 10,
                    num_active_chillers=2,
                )
        
        else:  # BASELINE
            # 22°C ambient, 60% PUE + 40% safety
            if pue > 1.20:
                # PUE too high: reduce cooling overhead
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=19.0,
                    fan_speeds_pct=[65.0] * 10,
                    num_active_chillers=2,
                )
            elif max_cpu > 52.0:
                # CPU getting warm: increase cooling slightly
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=16.0,
                    fan_speeds_pct=[74.0] * 10,
                    num_active_chillers=2,
                )
            elif pue < 1.18 and max_cpu < 50.0:
                # Very efficient: maintain current settings
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=18.0,
                    fan_speeds_pct=[70.0] * 10,
                    num_active_chillers=2,
                )
            else:
                # Default: balanced operation
                return ThermalGridRlAgentAction(
                    crac_setpoint_c=17.5,
                    fan_speeds_pct=[71.0] * 10,
                    num_active_chillers=2,
                )


# ============================================================================
# HYBRID AGENT: LLM + Rule-Based Fallback
# ============================================================================

async def get_adaptive_action(obs: ThermalGridRlAgentObservation, step: int, task_id: ThermalGridTaskID) -> Tuple[ThermalGridRlAgentAction, str, str]:
    """
    Get action using LLM with rule-based fallback.
    Returns (action, action_json, reasoning).
    """
    # Try LLM first
    llm_action = await get_llm_action(obs, step, task_id)
    
    if llm_action is not None:
        # LLM succeeded
        action = llm_action
        reasoning = "LLM decision"
    else:
        # Fallback to rule-based agent
        rule_agent = RuleBasedAgent(task_id)
        action = rule_agent.decide(obs, step)
        reasoning = "Rule-based decision (LLM unavailable)"
    
    action_str = json.dumps({
        "crac_setpoint_c": action.crac_setpoint_c,
        "fan_speeds_pct": action.fan_speeds_pct,
        "num_active_chillers": action.num_active_chillers,
    }, separators=(",", ":"))
    
    return action, action_str, reasoning


# ============================================================================
# STDOUT LOGGING FUNCTIONS
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ============================================================================
# MAIN TASK EXECUTION
# ============================================================================

async def run_thermal_grid_task(task_id: ThermalGridTaskID):
    """Run a single task through the environment with intelligent agent."""
    
    log_start(task=task_id.value, env=BENCHMARK, model=MODEL_NAME)

    env    = ThermalGridRlAgentEnv(base_url=ENV_URL)
    grader = ThermalGridGrader(task_id=task_id.value)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    error = None

    try:
        reset_result = await env.reset(task_id=task_id.value)
        observation  = reset_result.observation

        for step_idx in range(1, MAX_STEPS + 1):
            try:
                # Get adaptive action (LLM + rule-based fallback)
                action, action_str, reasoning = await get_adaptive_action(observation, step_idx, task_id)

                step_result = await env.step(action)

                reward = float(step_result.reward or 0.0)
                done   = step_result.done
                grader.log_thermal_grid_step(step_result.observation, reward)
                rewards.append(reward)
                steps_taken = step_idx

                log_step(step=step_idx, action=action_str, reward=reward, done=done, error=None)

                observation = step_result.observation

                if done:
                    break

            except Exception as e:
                error = str(e)
                log_step(step=step_idx, action="{}", reward=0.0, done=False, error=error)
                logger.error(f"Step {step_idx} failed: {e}")
                break

        score = grader.get_thermal_grid_score()
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as e:
        error = str(e)
        score = 0.0
        success = False
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        logger.error(f"Task {task_id.value} failed: {e}")
        
    finally:
        try:
            await env.close()
        except Exception as e:
            logger.warning(f"env.close() error: {e}")


async def main():
    """Run all 3 tasks sequentially."""
    print(f"Starting Thermal Grid RL Agent Inference", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Tasks: {[t.value for t in TASKS]}", flush=True)
    print("=" * 60, flush=True)

    for task_id in TASKS:
        print(f"\nRunning task: {task_id.value}", flush=True)
        await run_thermal_grid_task(task_id)
        print("-" * 60, flush=True)

    print("\nAll tasks completed.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
