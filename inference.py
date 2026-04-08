"""
Inference Script for Thermal Grid RL Agent Environment
========================================================
MANDATORY
- API_BASE_URL, MODEL_NAME, HF_TOKEN must be set in environment / .env
- Use OpenAI client for all LLM calls
- Emit [START], [STEP], [END] to stdout exactly as specified

Environment Variables:
    HF_TOKEN       - Hugging Face / API key (checked first)
    API_KEY        - Alternative API key (fallback)
    API_BASE_URL   - The API endpoint for the LLM
    MODEL_NAME     - The model identifier to use for inference
    ENV_URL        - URL of the thermal grid environment server

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import re
import json
import logging
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from client import ThermalGridRlAgentEnv
from models import ThermalGridRlAgentAction, ThermalGridRlAgentObservation
from server.thermal_grid_rl_agent_environment import ThermalGridTaskID
from server.grader import ThermalGridGrader

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Mandatory config (HF_TOKEN checked first — matches validator injection) ---
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")  or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK               = "thermal_grid_rl_agent"
MAX_STEPS               = 20
SUCCESS_SCORE_THRESHOLD = 0.1

TASKS = [
    ThermalGridTaskID.BASELINE,
    ThermalGridTaskID.LOAD_SHIFT,
    ThermalGridTaskID.GRID_STRESS,
]


# ============================================================================
# STDOUT LOGGING
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are an expert datacenter cooling control agent. Your goal is to maximize reward by dynamically adjusting cooling based on current conditions.

Respond with a JSON object containing EXACTLY these keys:
{"crac_setpoint_c": 16.0, "fan_speeds_pct": [75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0], "num_active_chillers": 3}

Rules:
- crac_setpoint_c: 12-27°C (lower = more cooling, higher = save energy)
- fan_speeds_pct: 20-100% (exactly 10 values, one per rack). Power scales as speed³.
- num_active_chillers: 1-4

Decision Guidelines:
- max CPU > 60°C  → INCREASE cooling (lower crac, raise fans, add chillers)
- max CPU < 50°C and PUE > 1.20 → REDUCE cooling (raise crac, lower fans)
- demand_response = 1 → reduce energy while keeping CPU < 85°C
- energy price > $0.08/kWh → favour efficiency (higher crac, lower fans, fewer chillers)
- ambient > 35°C → aggressive cooling (crac ≤ 15°C, fans ≥ 80%, 3+ chillers)
- moderate temps (50-58°C) → balanced (crac 16-18°C, fans 70-75%, 2-3 chillers)

CRITICAL: Adapt every step — do NOT repeat the same action if reward dropped.

Respond with ONLY the JSON object. No markdown, no explanation."""


def build_user_message(obs: ThermalGridRlAgentObservation, step: int, task: str) -> str:
    max_cpu = max(obs.max_cpu_temps_c) if obs.max_cpu_temps_c else 0.0
    max_gpu = max(obs.max_gpu_temps_c) if obs.max_gpu_temps_c else 0.0
    avg_cpu = sum(obs.mean_cpu_temps_c) / len(obs.mean_cpu_temps_c) if obs.mean_cpu_temps_c else 0.0

    return f"""STEP {step}/{MAX_STEPS} | Task: {task}

CURRENT STATE:
- Thermal : max_cpu={max_cpu:.1f}°C  avg_cpu={avg_cpu:.1f}°C  max_gpu={max_gpu:.1f}°C
- Cooling : PUE={obs.pue:.3f}  setpoint={obs.crac_supply_temp_c:.1f}°C  fans={obs.avg_fan_speed_pct:.0f}%  chillers={obs.num_active_chillers}
- Power   : IT={obs.total_it_power_w/1000:.1f}kW  Facility={obs.total_facility_power_w/1000:.1f}kW
- Grid    : price=${obs.energy_price_per_kwh:.3f}/kWh  DR={obs.demand_response_signal}  ambient={obs.ambient_temp_c:.1f}°C
- Batch   : {obs.pending_batch_jobs} jobs pending  off_peak={obs.off_peak_window}

TASK GRADING:
- baseline   : 60% PUE + 40% safety  (temps < 80°C, PUE < 1.25)
- load_shift : 50% cost + 30% PUE + 20% safety
- grid_stress: 70% reward + 30% safety  (38°C ambient, safety-critical)

Respond with JSON only."""


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


# ============================================================================
# LLM ACTION  (client passed in from main — initialized once)
# ============================================================================

def get_llm_action(
    client: OpenAI,
    obs: ThermalGridRlAgentObservation,
    step: int,
    task_id: ThermalGridTaskID,
) -> ThermalGridRlAgentAction:
    """Get action from LLM via validator-injected proxy."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(obs, step, task_id.value)},
        ],
        max_tokens=128,
        temperature=0.1,
    )

    raw  = response.choices[0].message.content or "{}"
    data = _extract_json(raw)

    if not data:
        raise ValueError(f"Unparseable LLM response at step {step}: {raw!r}")

    crac     = max(12.0, min(27.0, float(data.get("crac_setpoint_c", 18.0))))
    fans     = [max(20.0, min(100.0, float(f))) for f in data.get("fan_speeds_pct", [70.0] * 10)]
    chillers = max(1, min(4, int(data.get("num_active_chillers", 2))))

    if len(fans) != 10:
        fans = [fans[0] if fans else 70.0] * 10

    return ThermalGridRlAgentAction(
        crac_setpoint_c=crac,
        fan_speeds_pct=fans,
        num_active_chillers=chillers,
    )


# ============================================================================
# TASK RUNNER
# ============================================================================

async def run_task(client: OpenAI, task_id: ThermalGridTaskID) -> None:
    """Run a single task end-to-end."""
    log_start(task=task_id.value, env=BENCHMARK, model=MODEL_NAME)

    # _ws_url: https → wss  so HF Spaces WebSocket handshake succeeds
    env    = ThermalGridRlAgentEnv(base_url=ENV_URL)
    grader = ThermalGridGrader(task_id=task_id.value)
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    try:
        try:
            reset_result = await env.reset(task_id=task_id.value)
        except Exception as e:
            raise RuntimeError(
                f"env.reset() failed — is the server at {ENV_URL} reachable? ({e})"
            ) from e

        observation = reset_result.observation

        for step_idx in range(1, MAX_STEPS + 1):
            try:
                action = get_llm_action(client, observation, step_idx, task_id)

                action_str = json.dumps({
                    "crac_setpoint_c":     action.crac_setpoint_c,
                    "fan_speeds_pct":      action.fan_speeds_pct,
                    "num_active_chillers": action.num_active_chillers,
                }, separators=(",", ":"))

                step_result = await env.step(action)
                reward      = float(step_result.reward or 0.0)
                done        = step_result.done

                grader.log_thermal_grid_step(step_result.observation, reward)
                rewards.append(reward)
                steps_taken = step_idx
                observation = step_result.observation

                log_step(step=step_idx, action=action_str, reward=reward, done=done, error=None)

                if done:
                    break

            except Exception as e:
                log_step(step=step_idx, action="{}", reward=0.0, done=False, error=str(e))
                logger.error(f"Step {step_idx} failed: {e}")
                break

        score   = min(max(grader.get_thermal_grid_score(), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            logger.warning(f"env.close() error: {e}")

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ============================================================================
# MAIN
# ============================================================================

async def main() -> None:
    # Client initialised once — matches mandatory OpenEnv sample pattern
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASKS:
        await run_task(client, task_id)


if __name__ == "__main__":
    asyncio.run(main())