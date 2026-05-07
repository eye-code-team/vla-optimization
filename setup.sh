#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p data/datasets data/checkpoints data/hf_cache outputs

if [[ ! -f .env ]]; then
  cp .env.template .env
  echo "Created .env from .env.template"
else
  echo ".env already exists"
fi

if command -v id >/dev/null 2>&1; then
  UID_VAL="$(id -u)"
  GID_VAL="$(id -g)"
  if ! grep -q '^UID=' .env; then
    echo "UID=${UID_VAL}" >> .env
  fi
  if ! grep -q '^GID=' .env; then
    echo "GID=${GID_VAL}" >> .env
  fi
fi

echo "Setup complete"
echo "Next: ./quick_start.sh"
