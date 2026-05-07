#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH"
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose plugin is not available"
  exit 1
fi

./setup.sh

docker compose build

docker compose up -d

echo "Container started. Running GPU and Python checks..."

if ! docker compose exec -T smolvla_dev nvidia-smi >/dev/null 2>&1; then
  echo "Warning: GPU check failed. Verify NVIDIA Container Toolkit on host."
else
  echo "GPU check passed"
fi

docker compose exec -T smolvla_dev python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"

echo "Quick start complete"
echo "Use ./run.sh bash to enter container"
