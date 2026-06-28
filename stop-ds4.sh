#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
WORKER_HOST="${WORKER_HOST:-192.168.250.13}"
WORKER_REPO_DIR="${WORKER_REPO_DIR:-$REPO_DIR}"
ENV_FILE="${ENV_FILE:-.env.unholy-fusion}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
UNHOLY_COMPOSE_FILE="${UNHOLY_COMPOSE_FILE:-compose/docker-compose.unholy.yml}"

cd "$REPO_DIR"

rc=0

echo "Stopping head locally"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -f "$UNHOLY_COMPOSE_FILE" --profile head down || rc=$?

echo "Stopping worker on ${WORKER_HOST}"
ssh -o BatchMode=yes "$WORKER_HOST" \
  "cd '$WORKER_REPO_DIR' && docker compose --env-file '$ENV_FILE' -f '$COMPOSE_FILE' -f '$UNHOLY_COMPOSE_FILE' --profile worker down" || rc=$?

exit "$rc"
