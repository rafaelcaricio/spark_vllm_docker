#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
WORKER_HOST="${WORKER_HOST:-192.168.250.13}"
WORKER_REPO_DIR="${WORKER_REPO_DIR:-$REPO_DIR}"
ENV_FILE="${ENV_FILE:-.env.unholy-fusion}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
UNHOLY_COMPOSE_FILE="${UNHOLY_COMPOSE_FILE:-compose/docker-compose.unholy.yml}"
ENTRYPOINT_FILE="${ENTRYPOINT_FILE:-entrypoints/entrypoint.unholy.sh}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_WAIT_SECONDS="${HEALTH_WAIT_SECONDS:-900}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"

cd "$REPO_DIR"

echo "Syncing unholy-fusion stack files to ${WORKER_HOST}:${WORKER_REPO_DIR}"
ssh -o BatchMode=yes "$WORKER_HOST" \
  "mkdir -p '$WORKER_REPO_DIR/compose' '$WORKER_REPO_DIR/entrypoints'"
scp "$ENV_FILE" "$COMPOSE_FILE" "${WORKER_HOST}:${WORKER_REPO_DIR}/"
scp "$UNHOLY_COMPOSE_FILE" "${WORKER_HOST}:${WORKER_REPO_DIR}/${UNHOLY_COMPOSE_FILE}"
scp "$ENTRYPOINT_FILE" "${WORKER_HOST}:${WORKER_REPO_DIR}/${ENTRYPOINT_FILE}"

echo "Starting worker on ${WORKER_HOST}"
ssh -o BatchMode=yes "$WORKER_HOST" \
  "cd '$WORKER_REPO_DIR' && docker compose --env-file '$ENV_FILE' -f '$COMPOSE_FILE' -f '$UNHOLY_COMPOSE_FILE' --profile worker up -d"

echo "Starting head locally"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -f "$UNHOLY_COMPOSE_FILE" --profile head up -d

echo "Waiting for ${HEALTH_URL}"
max_attempts=$((HEALTH_WAIT_SECONDS / HEALTH_POLL_SECONDS))
if [ "$max_attempts" -lt 1 ]; then
  max_attempts=1
fi

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null; then
    echo "DeepSeek V4 Flash is ready: ${HEALTH_URL}"
    exit 0
  fi

  if ((attempt % 12 == 0)); then
    echo "Still waiting after $((attempt * HEALTH_POLL_SECONDS))s..."
  fi
  sleep "$HEALTH_POLL_SECONDS"
done

echo "Timed out waiting for ${HEALTH_URL}; containers were left running for inspection." >&2
exit 1
