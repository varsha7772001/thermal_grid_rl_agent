#!/bin/bash
# start.sh — OpenEnv Thermal Grid RL Agent (HF Spaces Compatible)

PYTHON=$(command -v python3 || command -v python)

# Main server port (default 8000 for local development)
MAIN_PORT=${PORT:-8000}
MOCK_PORT=${MOCK_PORT:-8001}

echo "===== Application Startup at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
echo "[HF-Space] Port Configuration:"
echo "  - Main App (exposed): $MAIN_PORT"
echo "  - Mock Data Server: $MOCK_PORT (internal only)"

# Set PORT env var so server.app uses the correct port
export PORT=$MAIN_PORT

# Start Mock Data Server in background (internal, not exposed)
echo "[HF-Space] Starting Mock Data Server on port $MOCK_PORT..."
$PYTHON -m uvicorn mock_data_server:app --host 0.0.0.0 --port $MOCK_PORT &
MOCK_SERVER_PID=$!

# Wait for mock server to be ready
echo "[HF-Space] Waiting for Mock Server to be ready..."
for i in $(seq 1 10); do
    if curl -s http://localhost:$MOCK_PORT/health > /dev/null 2>&1; then
        echo "[HF-Space] Mock Server is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "[HF-Space] ERROR: Mock Server failed to start after 10 attempts"
        kill $MOCK_SERVER_PID 2>/dev/null
        exit 1
    fi
    echo "[HF-Space] Waiting for Mock Server... (attempt $i/10)"
    sleep 2
done

# Start Environment Server on PORT (this is what HF Spaces monitors)
echo "[HF-Space] Starting Environment Server on port $MAIN_PORT..."
$PYTHON -m server.app &
ENV_SERVER_PID=$!

echo "[HF-Space] All services started!"
echo "[HF-Space] Main App: http://localhost:$MAIN_PORT"
echo "[HF-Space] API Docs: http://localhost:$MAIN_PORT/docs"
echo "[HF-Space] Health: http://localhost:$MAIN_PORT/health"

# Wait for env server to be ready
sleep 3
for i in $(seq 1 10); do
    if curl -s http://localhost:$MAIN_PORT/health > /dev/null 2>&1; then
        echo "[HF-Space] Environment Server is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "[HF-Space] ERROR: Environment Server failed to start after 10 attempts"
        kill $MOCK_SERVER_PID $ENV_SERVER_PID 2>/dev/null
        exit 1
    fi
    echo "[HF-Space] Waiting for Environment Server... (attempt $i/10)"
    sleep 2
done

# Wait for both processes
wait $MOCK_SERVER_PID $ENV_SERVER_PID
exit_code=$?
echo "[HF-Space] Services exited with code: $exit_code"
exit $exit_code
