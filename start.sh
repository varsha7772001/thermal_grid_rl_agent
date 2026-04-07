#!/bin/bash
# start.sh — OpenEnv Thermal Grid RL Agent

PYTHON=$(command -v python3 || command -v python)

# Allow port overrides via environment variables (useful for Docker/HF)
MOCK_PORT=${MOCK_PORT:-8001}
ENV_PORT=${ENV_PORT:-8000}

echo "[HF-Space] Starting Mock Data Server on port $MOCK_PORT..."
$PYTHON -m uvicorn mock_data_server:app --host 0.0.0.0 --port $MOCK_PORT &
MOCK_SERVER_PID=$!

echo "[HF-Space] Starting Environment Server on port $ENV_PORT..."
$PYTHON -m uvicorn server.app:app --host 0.0.0.0 --port $ENV_PORT &
ENV_SERVER_PID=$!

wait $MOCK_SERVER_PID $ENV_SERVER_PID