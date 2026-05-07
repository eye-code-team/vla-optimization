#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f .env ]]; then
  echo "Missing .env. Run ./setup.sh first."
  exit 1
fi

docker compose up -d

docker compose exec smolvla_dev python finetune_dynamic_lora_prunning_snapflow.py "$@"
