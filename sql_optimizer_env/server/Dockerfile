# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ─────────────────────────────────────────────────────────────────────────────
# SQL Query Optimizer Environment
#
# This container runs the FastAPI environment server ONLY.
# It does NOT run a database.
#
# The environment connects to the USER's own Postgres database at runtime
# when they call reset(query=..., db_url=...).
#
# REQUIREMENT ON THE USER'S DATABASE:
#   pg_hint_plan must be installed on the user's Postgres instance.
#   Without it, index/join hint actions will have no effect.
#   Installation: https://github.com/ossc-db/pg_hint_plan
#
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is required for installing any VCS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=standalone
ARG ENV_NAME=sql_optimizer_env

# Copy environment source into the build context
COPY . /app/env
WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies — use lockfile if present for reproducibility
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

# ─────────────────────────────────────────────────────────────────────────────
# Runtime stage — lean final image
# ─────────────────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env       /app/env

# Virtual environment takes priority in PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# ── Server tuning ─────────────────────────────────────────────────────────────
# These can all be overridden at docker run time with -e FLAG=value

# Number of uvicorn worker processes
# Rule of thumb: 2 x CPU cores + 1
# Default 4 works for most single-node deployments
ENV WORKERS=4

# Maximum concurrent WebSocket sessions per worker
# Each session = one active optimization episode
ENV MAX_CONCURRENT_ENVS=100

# Server bind config — do not change unless you know why
ENV HOST=0.0.0.0
ENV PORT=8000

# ── Optional query execution config ──────────────────────────────────────────

# Hard limit on how long a single query is allowed to run (milliseconds)
# Prevents runaway queries from blocking the worker
# User's query must complete within this time for EXPLAIN ANALYZE to work
ENV QUERY_TIMEOUT_MS=30000

# Maximum number of rewrite steps per episode
# Agent must submit within this many steps
ENV MAX_STEPS=10

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
# cd into /app/env so relative imports resolve correctly
CMD ["sh", "-c", \
    "cd /app/env && uvicorn server.app:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS"]