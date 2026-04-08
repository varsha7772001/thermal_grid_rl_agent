---
title: Thermal Grid RL Agent Environment
emoji: 🏭
sdk: docker
base_path: /docs
app_port: 8000 
author: varsha komati
tags:
  - openenv
  - reinforcement-learning
  - datacenter
  - energy-efficiency
---

# 🏭 Thermal Grid RL Agent Environment

**A high-fidelity datacenter thermal simulator integrated with real-time power grid signals.** Designed for training and evaluating AI agents to optimize energy efficiency, thermal safety, and grid cooperation in modern AI datacenters.

> **Real-World Problem:** Datacenters consume ~1-2% of global electricity. Cooling accounts for 30-50% of that energy. Poor thermal management leads to equipment failures, while overcooling wastes millions. This environment lets AI agents learn optimal control strategies.

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Environment Type** | Continuous control + discrete scheduling |
| **Action Space** | 3 cooling controls + 4 load distribution levers |
| **Observation Space** | 67-dimensional vector (thermal, IT, cooling, grid) |
| **Tasks** | 3 (Easy → Medium → Hard) |
| **Simulator** | 10 racks × 8 servers/rack, 1-minute timesteps |
| **Real-World Data** | India weather (2000-2024) + CEA energy tariffs |
| **Baseline Scores** | 0.844 (baseline), 0.885 (load_shift), 0.905 (grid_stress) |

---

## 🎯 Motivation & Description

### The Problem
Modern AI datacenters face three competing objectives:
1. **Keep servers cool** — CPU/GPU failures above 85°C cause outages
2. **Minimize energy waste** — Overcooling increases PUE (Power Usage Effectiveness)
3. **Cooperate with the grid** — Demand-response events require rapid load reduction

### Why This Matters
- **Google reduced cooling costs by 40%** using DeepMind's RL agent (2016)
- **India's datacenter market** is projected to reach $5B by 2025, with energy as the #1 operational cost
- **Grid integration** is becoming critical as renewable energy penetration increases

### What This Environment Models
- Physics-based thermal dynamics with thermal mass lag (τ=12 steps)
- CRAC units, VFD fans, chiller stacking with real efficiency curves (COP peaks at 80% load)
- Time-of-use energy pricing from Indian grid data (₹0.06–0.20/kWh)
- Carbon intensity variations (solar/wind peaks: 230–420 gCO₂/kWh)
- Demand-response events during peak hours (17:00–19:00)
- Batch job scheduling for temporal load shifting
- 10 racks × 8 servers per rack simulation with per-server power caps (50–500W)

---

## 🎮 Tasks (Easy → Medium → Hard)

| Task ID | Difficulty | Goal | Reward Weights (α, β, γ) | Conditions |
|---------|------------|------|-------------------------|------------|
| `baseline` | Easy | Keep PUE < 1.25, CPU < 85°C | α=0.4, β=0.3, γ=0.3 | 22°C ambient, ±2% price volatility, 2% DR probability |
| `load_shift` | Medium | Shift batch jobs to off-peak hours to cut cost | α=0.6, β=0.2, γ=0.2 | 25°C ambient, ±15% price volatility, 5% DR probability |
| `grid_stress` | Hard | Survive heatwave + frequent DR requests | α=0.2, β=0.2, γ=0.6 | 38°C ambient, ±25% price volatility, 40% DR probability |

**Reward weights explanation:**
- **α (Energy Cost):** Higher in `load_shift` — cost optimization matters most
- **β (PUE Inefficiency):** Moderate across all tasks (penalizes deviation from target PUE)
- **γ (Thermal Risk):** Highest in `grid_stress` — safety is critical under heat stress

**Scoring (Grader):**
- All tasks: Safety score based on max CPU temp ever seen (80°C buffer, 85°C hard limit)
- `baseline`: 60% PUE score + 40% safety
- `load_shift`: 50% avg reward + 30% PUE + 20% safety
- `grid_stress`: 70% avg reward + 30% safety (zero tolerance for >85°C)

---

## 🕹️ Action Space

The agent controls **two distinct levers** that interact:

### A. Cooling Control (Continuous)
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `crac_setpoint_c` | float | 12.0 – 27.0 | CRAC supply-air temperature setpoint (°C) |
| `fan_speeds_pct` | list[float] | 20.0 – 100.0 | Per-rack VFD fan speed (%) — power ∝ speed³ (Affinity Laws) |
| `num_active_chillers` | int | 1 – 4 | Number of chiller units to run |

### B. Load Distribution (Discrete/Structural)
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `region_traffic_weights` | list[float] | each 0.0–1.0, sum=1.0 | Fraction of live traffic to each region (GLB) |
| `batch_job_schedule` | list[int] | 0 or 1 | Run (1) or defer (0) each queued batch job |
| `workload_matrix` | list[list[float]] | 0.0 – 1.0 | Per-server utilisation [10 racks × 8 servers] |
| `power_caps_w` | list[list[float]] | 50 – 500 | Per-server power cap (Watts) |

**Key Interaction:** Redistributing load changes the thermal profile, which requires cooling re-adjustment. This creates a non-linear, multi-objective optimization problem.

---

## 👁️ Observation Space (67-Dimensional)

| Category | Fields | Dimensions | Description |
|----------|--------|------------|-------------|
| **Thermal State** | `inlet_temps_c` (10), `mean_cpu_temps_c` (10), `max_cpu_temps_c` (10), `max_gpu_temps_c` (10), `thermal_mass_lag_c_per_min` (1) | 41 | Rack temperatures + thermal inertia indicator |
| **IT Load** | `rack_powers_w` (10), `rack_utilisation` (10), `live_traffic_load_w` (1), `pue` (1) | 22 | Power usage and workload distribution |
| **External Grid** | `ambient_temp_c` (1), `energy_price_per_kwh` (1), `demand_response_signal` (1), `off_peak_window` (1) | 4 | Real-world grid signals |
| **Total** | | **67** | Normalized vector for RL training |

**Full observation fields (Pydantic):**
- Thermal: `inlet_temps_c`, `mean_cpu_temps_c`, `max_cpu_temps_c`, `max_gpu_temps_c`, `thermal_mass_lag_c_per_min`
- IT load: `rack_powers_w`, `rack_utilisation`, `live_traffic_load_w`, `deferred_batch_load_w`, `pending_batch_jobs`
- Cooling: `pue`, `total_it_power_w`, `total_facility_power_w`, `crac_power_w`, `chiller_power_w`, `num_active_chillers`, `chiller_load_pct`, `crac_supply_temp_c`, `avg_fan_speed_pct`, `safety_override_triggered`
- External: `ambient_temp_c`, `energy_price_per_kwh`, `grid_carbon_intensity_g_per_kwh`, `demand_response_signal`, `off_peak_window`

---

## 📈 Reward Function

**Joint Cost Function:**
```
Total Cost = α × (Price × Energy) + β × (PUE inefficiency) + γ × (Thermal risk)
Reward = clamp(1.0 - Total Cost, 0.0, 1.0)
```

**Components:**
1. **Energy Cost (α):** Direct electricity cost (Price × Total Facility Power × dt), normalized to ~$1000 baseline
2. **PUE Inefficiency (β):** Penalizes deviation from task target PUE, scaled by IT load
3. **Thermal Risk (γ):** Heavy penalty for CPU >85°C or inlet >27°C — scales quadratically above threshold

**Design Choice:** Reward is computed every step (dense feedback), allowing partial progress signals rather than sparse end-of-episode rewards.

**Safety Layer:** Hard safety override activates when CPU >80°C — forces fans to 100% and CRAC setpoint to 15°C. Tracked in `safety_override_triggered` flag.

---

## 🏆 Baseline Scores

Results from running `inference.py` with **Qwen/Qwen2.5-72B-Instruct**:

| Task | Score (0.0–1.0) | Avg Reward | Status |
|------|-----------------|------------|--------|
| `baseline` | **0.844** | 0.95 | ✅ Strong performance |
| `load_shift` | **0.885** | 0.92 | ✅ Good cost optimization |
| `grid_stress` | **0.905** | 0.88 | ✅ Handles extreme conditions |

**Agent Architecture:** Hybrid LLM + rule-based fallback
- **LLM path:** Qwen/Qwen2.5-72B-Instruct via Hugging Face router
- **Fallback:** Adaptive rule-based agent with task-specific strategies
- **LLM failure recovery:** Graceful degradation to rule-based control if API unavailable

*Scores are reproducible with the same model and environment setup. Run `python inference.py` to verify.*

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Hugging Face API token (for LLM inference)
- OpenEnv core: `pip install openenv-core`

### 1. Clone the Repository
```bash
git clone https://github.com/varsha7772001/openenv_demo.git
cd openenv_demo
```

### 2. Install Dependencies
```bash
# Using uv (recommended, faster)
uv sync

# Or using pip
pip install -e .
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_URL=http://localhost:8000
```

### 4. Start the Environment Servers

**Option A: Using the start script (recommended)**
```bash
bash start.sh
```

**Option B: Using Docker**
```bash
docker build -t thermal-grid-env:latest .
docker run -p 8000:8000 -p 8001:8001 thermal-grid-env:latest
```

**Option C: Direct Python**
```bash
python -m thermal_grid_rl_agent.server.app
# or
uv run --project . server
```

Servers will be available at:
- **Environment Server:** `http://localhost:8000`
- **Mock Data Server:** `http://localhost:8001`

### 5. Run Baseline Inference
```bash
python inference.py
```

This runs all 3 tasks sequentially (10 steps each) and outputs structured logs in `[START]` / `[STEP]` / `[END]` format.

### 6. Train RL Agent (Optional)
```bash
# Train PPO agent using Stable-Baselines3
python train_rl.py --task baseline --steps 10000

# Evaluate trained model
python inference_rl.py --task baseline --model models/ppo_baseline_final
```

### 7. Validate with OpenEnv
```bash
pip install openenv-core
openenv validate
```

---

## 🐳 Docker Deployment

### Build
```bash
docker build -t thermal-grid-env:latest .
```

### Run Locally
```bash
docker run -p 8000:8000 -p 8001:8001 thermal-grid-env:latest
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Deploy to Hugging Face Spaces
1. Create a new Space at https://huggingface.co/new-space
2. Select **Docker** as the SDK
3. Push your code to the Space repository
4. Add environment variables in Space settings: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`

---

## 🤖 Agent Scripts

| Script | Purpose |
|--------|---------|
| `inference.py` | **Required** — Baseline LLM agent for evaluation (hybrid LLM + rule-based fallback) |
| `train_rl.py` | Optional — Train a PPO agent using Stable-Baselines3 |
| `inference_rl.py` | Optional — Evaluate a trained RL model |
| `server/gym_env.py` | Gymnasium wrapper for RL training (67-dim obs, 5-dim action) |

---

## 📁 Project Structure

```
thermal_grid_rl_agent/
├── inference.py                          # Baseline LLM agent (required)
├── client.py                             # HTTP/WebSocket client for environment
├── models.py                             # Pydantic Action/Observation schemas
├── hybrid_signal_generator.py            # Real-world grid data integration
├── mock_data_server.py                   # Mock API for Indian energy data
├── train_rl.py                           # Optional: Train PPO agent
├── inference_rl.py                       # Optional: Evaluate trained RL model
├── openenv.yaml                          # OpenEnv manifest (spec v1)
├── pyproject.toml                        # Dependencies + project config
├── Dockerfile                            # Containerized deployment (Python 3.11)
├── start.sh                              # Server startup script
├── validate-submission.sh                # Pre-submission validation script
├── data/                                 # CSV data for India weather & energy
│   ├── india_2000_2024_daily_weather.csv # 24 years of daily weather data
│   ├── india_monthly_full_release.csv    # Monthly energy price data
│   └── region-metadata.csv               # Regional metadata for GLB
├── server/
│   ├── thermal_grid_rl_agent_environment.py  # Core physics simulator (992 lines)
│   ├── app.py                        # FastAPI server with OpenEnv interface
│   ├── gym_env.py                    # Gymnasium wrapper for RL training
│   └── grader.py                     # Task-specific graders (0.0-1.0 scores)
└── README.md                         # This file
```

---

## 🔬 Technical Details

### Physics Simulation
- **Thermal Model:** Resistance-capacitance network with thermal mass lag (τ=12 steps)
  - Inlet temp equilibrium: `T_eq = setpoint + rack_power / (1000 × cooling_effectiveness)`
  - Lag model: `T_t = T_prev + (T_eq - T_prev) / τ`
- **Cooling Efficiency:** Chiller COP peaks at 80% load, degrades with ambient temperature
  - Base COP: `3.5 × exp(-0.5 × ((load% - 80) / 40)²) + 1.5`
  - Ambient penalty: -0.05 COP per °C above 25°C
- **Fan Power:** Scales as speed³ (Affinity Laws) — small reductions yield large savings
  - Cooling effectiveness: `0.4 + 0.6 × (speed/100)^0.8`
- **CRAC Power:** Increases at lower setpoints and higher fan speeds
  - Formula: `IT_power × 0.3 × (1 - (setpoint - 12) / 20) × (fan_speed/100)^1.5`
  - Ambient impact: +2% power per °C above 25°C

### Simulator Configuration
- **Racks:** 10 (each with 8 servers)
- **Regions:** 1 (configurable)
- **Timestep:** 60 seconds (1 minute)
- **Chillers:** Max 4 units, 5000W capacity each
- **Server power caps:** 50–500W per server
- **CPU/GPU temp limits:** 85°C / 115°C (hard limits)
- **Safety threshold:** 80°C (triggers failsafe cooling)

### Real-World Data Integration
- **Ambient Temperature:** Open-Meteo API for Indian cities (Bengaluru default)
  - Fallback: Synthetic profiles based on `data/india_2000_2024_daily_weather.csv`
- **Energy Pricing:** Central Electricity Authority of India tariffs
  - TOU profile: $0.06–0.20/kWh across 24 hours
  - Peak hours: 12:00–19:00 (up to $0.20/kWh)
  - Off-peak: 00:00–05:00 (down to $0.06/kWh)
- **Carbon Intensity:** Grid emission factors by region
  - Solar peak (12:00): 230 gCO₂/kWh
  - Fossil peak (22:00): 425 gCO₂/kWh
- **Demand Response:** Triggered during evening peak (17:00–19:00)
- **Hybrid Mode:** Real CSV data overrides synthetic profiles when available

### Safety Mechanisms
- **Hard Safety Layer:** Overrides agent actions if CPU >80°C
  - Per-rack: Forces fan to 100% if any CPU exceeds threshold
  - Global: Forces CRAC setpoint to 15°C if max CPU >82°C
- **Failsafe Cooling:** Activates when `safety_override_triggered = True`
- **Graceful Degradation:** Uses safe defaults if LLM calls fail
- **Input Clamping:** All continuous actions clamped to valid ranges automatically

### Grid Signal Generator
- **TOU Price Profile:** 24-hour cycle with realistic Indian grid patterns
- **Carbon Profile:** Solar ramp during midday (230–340 gCO₂/kWh)
- **DR Events:** Probabilistic, concentrated in evening peak hours
- **Price Volatility:** Task-dependent (±2% to ±25%)
- **Off-Peak Window:** 00:00–05:00 for batch job scheduling

---

## 📚 References & Inspiration

- **Google DeepMind:** "Data Center Cooling with Deep Reinforcement Learning" (2016) — Reduced cooling costs by 40%
- **DCCluster-Opt:** Digital twin thermal simulator for datacenter cooling optimization
- **Affinity Laws:** Fan power ∝ speed³, pump power ∝ speed³ — fundamental HVAC engineering
- **Central Electricity Authority of India:** Real tariff data for energy pricing
- **Open-Meteo API:** Historical weather data for Indian cities

---

*Built by [varsha komati] — A self-driven, single-author project.*
