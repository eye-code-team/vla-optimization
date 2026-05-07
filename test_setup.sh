#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f .env ]]; then
  echo "Missing .env. Run ./setup.sh first."
  exit 1
fi

docker compose up -d

echo "[1/4] GPU visibility"
docker compose exec -T smolvla_dev nvidia-smi

echo "[2/4] Python imports"
docker compose exec -T smolvla_dev python -c "import torch, cv2, matplotlib; print('imports_ok')"

echo "[3/4] SmolVLA base model load test"
docker compose exec -T smolvla_dev python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; p=SmolVLAPolicy.from_pretrained('lerobot/smolvla_base'); print(type(p).__name__)"

echo "[4/4] Output write test"
docker compose exec -T smolvla_dev python integration_test.py

echo "All setup checks passed"
