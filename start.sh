#!/bin/bash
set -e

PYTHON=$(command -v python3 || command -v python)
MAIN_PORT=${PORT:-8000}
MOCK_PORT=${MOCK_PORT:-8001}

echo "===== Startup $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
echo "Main: $MAIN_PORT | Mock: $MOCK_PORT"

export PORT=$MAIN_PORT

# Start mock server
$PYTHON -m uvicorn mock_data_server:app --host 0.0.0.0 --port $MOCK_PORT &
MOCK_SERVER_PID=$!

# Wait for mock server
for i in $(seq 1 15); do
    curl -sf http://localhost:$MOCK_PORT/health > /dev/null && echo "Mock ready" && break
    [ $i -eq 15 ] && echo "Mock failed" && exit 1
    echo "Mock not ready ($i/15)..."; sleep 2
done

# Start env server — must bind to $MAIN_PORT
$PYTHON -m uvicorn server.app:app --host 0.0.0.0 --port $MAIN_PORT &
ENV_SERVER_PID=$!

# Wait for env server (give it more time)
sleep 8
for i in $(seq 1 20); do
    curl -sf http://localhost:$MAIN_PORT/health > /dev/null && echo "Env server ready" && break
    [ $i -eq 20 ] && echo "Env server failed" && exit 1
    echo "Env not ready ($i/20)..."; sleep 3
done

echo "All services up!"

# Keep alive — exit only if env server dies
while kill -0 $ENV_SERVER_PID 2>/dev/null; do sleep 5; done
echo "Env server exited"; exit 1