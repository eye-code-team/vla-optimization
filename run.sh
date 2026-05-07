#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

ACTION="${1:-bash}"

ensure_env() {
  if [[ ! -f .env ]]; then
    echo "Missing .env. Run ./setup.sh first."
    exit 1
  fi
}

ensure_env

case "${ACTION}" in
  up)
    docker compose up -d
    ;;
  bash)
    docker compose up -d
    docker compose exec smolvla_dev bash
    ;;
  down)
    docker compose down
    ;;
  logs)
    docker compose logs -f --tail=200
    ;;
  rebuild)
    docker compose build --no-cache
    ;;
  *)
    echo "Usage: ./run.sh [up|bash|down|logs|rebuild]"
    exit 1
    ;;
esac
