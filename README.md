---
title: Thermal Grid RL Agent Environment
emoji: 🏭
sdk: docker
base_path: /
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
| **Real-World Data** | India energy prices, carbon intensity, ambient temps |
| **Baseline Scores** | 0.84 (baseline), 0.89 (load_shift), 0.91 (grid_stress) |

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
- Physics-based thermal dynamics with thermal mass lag
- CRAC units, VFD fans, chiller stacking with real efficiency curves
- Time-of-use energy pricing from Indian grid data
- Carbon intensity variations (solar/wind peaks)
- Demand-response events during peak hours
- Batch job scheduling for temporal load shifting

---

## 🎮 Tasks (Easy → Medium → Hard)

| Task ID | Difficulty | Goal | Reward Weights (α,β,γ) | Conditions |
|---------|------------|------|------------------------|------------|
| `baseline` | Easy | Keep PUE <1.25, CPU <85°C | 0.4, 0.3, 0.3 | 22°C ambient, low price volatility, rare DR |
| `load_shift` | Medium | Shift batch jobs to off-peak hours to cut cost | 0.6, 0.2, 0.2 | 25°C ambient, high price volatility, many batch jobs |
| `grid_stress` | Hard | Survive heatwave + frequent DR requests | 0.2, 0.2, 0.6 | 38°C ambient, extreme price swings, 40% DR probability |

**Reward weights explanation:**
- **α (Energy Cost):** Higher in `load_shift` — cost optimization matters most
- **β (PUE Inefficiency):** Moderate across all tasks
- **γ (Thermal Risk):** Highest in `grid_stress` — safety is critical under heat stress

---

## 🕹️ Action Space

The agent controls **two distinct levers** that interact:

### A. Cooling Control (Continuous)
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `crac_setpoint_c` | float | 12.0 – 27.0 | CRAC supply-air temperature setpoint (°C) |
| `fan_speeds_pct` | list[float] (len=10) | 20.0 – 100.0 | Per-rack VFD fan speed (%) — power ∝ speed³ |
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

## 👁️ Observation Space

| Category | Fields | Description |
|----------|--------|-------------|
| **Thermal State** | `inlet_temps_c` (10), `mean_cpu_temps_c` (10), `max_cpu_temps_c` (10), `max_gpu_temps_c` (10), `thermal_mass_lag_c_per_min` | Rack temperatures + thermal inertia indicator |
| **IT Load** | `rack_powers_w` (10), `rack_utilisation` (10), `live_traffic_load_w`, `deferred_batch_load_w`, `pending_batch_jobs` | Power usage and workload distribution |
| **Cooling State** | `pue`, `total_it_power_w`, `total_facility_power_w`, `crac_power_w`, `chiller_power_w`, `num_active_chillers`, `chiller_load_pct`, `crac_supply_temp_c`, `avg_fan_speed_pct` | Efficiency metrics and equipment status |
| **External Grid** | `ambient_temp_c`, `energy_price_per_kwh`, `grid_carbon_intensity_g_per_kwh`, `demand_response_signal`, `off_peak_window` | Real-world grid signals |

---

## 📈 Reward Function

**Joint Cost Function:**
```
Total Cost = α × (Price × Energy) + β × (PUE inefficiency) + γ × (Thermal risk)
Reward = clamp(1.0 - Total Cost, 0.0, 1.0)
```

**Components:**
1. **Energy Cost (α):** Direct electricity cost normalized to typical datacenter consumption
2. **PUE Inefficiency (β):** Penalizes deviation from target PUE, scaled by IT load
3. **Thermal Risk (γ):** Heavy penalty for CPU >85°C or inlet >27°C

**Design Choice:** Reward is computed every step (dense feedback), allowing partial progress signals rather than sparse end-of-episode rewards.

---

## 🏆 Baseline Scores

Results from running `inference.py` with **Qwen/Qwen2.5-72B-Instruct**:

| Task | Score (0.0–1.0) | Avg Reward | Status |
|------|-----------------|------------|--------|
| `baseline` | **0.844** | 0.95 | ✅ Strong performance |
| `load_shift` | **0.885** | 0.92 | ✅ Good cost optimization |
| `grid_stress` | **0.905** | 0.88 | ✅ Handles extreme conditions |

*Scores are reproducible with the same model and environment setup. Run `python inference.py` to verify.*

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Hugging Face API token (for LLM inference)

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

Servers will be available at:
- **Environment Server:** `http://localhost:8000`
- **Mock Data Server:** `http://localhost:8001`

### 5. Run Baseline Inference
```bash
python inference.py
```

This runs all 3 tasks sequentially and outputs structured logs in `[START]` / `[STEP]` / `[END]` format.

### 6. Validate with OpenEnv
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
| `inference.py` | **Required** — Baseline LLM agent for evaluation |
| `train_rl.py` | Optional — Train a PPO agent using Stable-Baselines3 |
| `inference_rl.py` | Optional — Evaluate a trained RL model |
| `server/gym_env.py` | Gymnasium wrapper for RL training |

---

## 📁 Project Structure

```
thermal_grid_rl_agent/
├── inference.py                      # Baseline LLM agent (required)
├── client.py                         # HTTP/WebSocket client for environment
├── models.py                         # Pydantic Action/Observation schemas
├── hybrid_signal_generator.py        # Real-world grid data integration
├── mock_data_server.py               # Mock API for Indian energy data
├── train_rl.py                       # Optional: Train PPO agent
├── inference_rl.py                   # Optional: Evaluate trained RL model
├── openenv.yaml                      # OpenEnv manifest
├── pyproject.toml                    # Dependencies
├── Dockerfile                        # Containerized deployment
├── start.sh                          # Server startup script
├── data/                             # CSV data for India weather & energy
│   ├── india_2000_2024_daily_weather.csv
│   ├── india_monthly_full_release.csv
│   └── region-metadata.csv
├── server/
│   ├── thermal_grid_rl_agent_environment.py  # Core physics simulator
│   ├── app.py                        # FastAPI server with OpenEnv interface
│   ├── gym_env.py                    # Gymnasium wrapper for RL training
│   └── grader.py                     # Task-specific graders (0.0-1.0 scores)
└── README.md                         # This file
```

---

## 🔬 Technical Details

### Physics Simulation
- **Thermal Model:** Resistance-capacitance network with thermal mass lag (τ=12 steps)
- **Cooling Efficiency:** Chiller COP peaks at 80% load, degrades with ambient temperature
- **Fan Power:** Scales as speed³ (Affinity Laws) — small reductions yield large savings
- **CRAC Power:** Increases at lower setpoints and higher fan speeds

### Real-World Data Integration
- **Ambient Temperature:** Open-Meteo API for Indian cities (Bengaluru default)
- **Energy Pricing:** Central Electricity Authority of India tariffs
- **Carbon Intensity:** Grid emission factors by region
- **Fallback:** Synthetic profiles with realistic patterns if APIs unavailable

### Safety Mechanisms
- **Hard Safety Layer:** Overrides agent actions if CPU >80°C
- **Failsafe Cooling:** Forces fans to 100% and lowers CRAC setpoint on overheating
- **Graceful Degradation:** Uses safe defaults if LLM calls fail

---

## 📚 References & Inspiration

- **Google DeepMind:** "Data Center Cooling with Deep Reinforcement Learning" (2016)
- **DCCluster-Opt:** Digital twin thermal simulator for datacenter cooling optimization

---

## 📄 License

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
