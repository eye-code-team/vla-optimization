#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

DATASET_REPO="${1:-lerobot/svla_so100_pickplace}"
TARGET_DIR="${2:-/datasets/${DATASET_REPO//\//_}}"

if [[ ! -f .env ]]; then
  echo "Missing .env. Run ./setup.sh first."
  exit 1
fi

docker compose up -d

docker compose exec smolvla_dev bash -lc "python - <<'PY'
from huggingface_hub import snapshot_download
repo = '${DATASET_REPO}'
target = '${TARGET_DIR}'
path = snapshot_download(repo_id=repo, repo_type='dataset', local_dir=target, local_dir_use_symlinks=False)
print(f'Downloaded to: {path}')
PY"
