# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git and curl are available
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Build arguments
ARG BUILD_MODE=standalone
ARG ENV_NAME=thermal_grid_rl_agent

# Copy entire project to /app/env (root of project = root of build context)
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies using uv sync
# If uv.lock exists use it for reproducibility, otherwise resolve on the fly
ENV UV_HTTP_TIMEOUT=300

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ─── Final runtime stage ────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

# Set WORKDIR to project root so that:
#   - mock_data_server.py finds data/ via relative path
#   - server/ is importable as a package
#   - inference.py runs without PYTHONPATH tricks
WORKDIR /app/env

# Copy virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy entire project from builder
COPY --from=builder /app/env /app/env

# Use the venv binaries
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so `server` package and root modules are always importable
# regardless of how uvicorn or inference.py is invoked
ENV PYTHONPATH="/app/env:${PYTHONPATH}"

# Make startup script executable
RUN chmod +x /app/env/start.sh

# HF Spaces will provide $PORT environment variable (default 7860)
# The start.sh script reads this and starts the app on the correct port
ENV MOCK_PORT=8001

EXPOSE 7860

# Health check on the main exposed port
# start-period=40s gives mock server time to load CSVs before env server starts
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Launch both servers via start.sh
CMD ["/app/env/start.sh"]