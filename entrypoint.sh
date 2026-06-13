#!/usr/bin/env bash
# Entrypoint: print env summary, optionally download the smolvla_libero model,
# then exec the requested command (default: bash).
set -e

echo "=================================================================="
echo "  SmolVLA-LIBERO container"
echo "  python : $(python --version 2>&1)"
echo "  torch  : $(python -c 'import torch;print(torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())' 2>/dev/null || echo 'NOT READY')"
echo "  MUJOCO_GL=${MUJOCO_GL}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "=================================================================="

# Pre-cache the base policy so the first train/eval is fast (idempotent).
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  python - <<'PY' || echo "  [warn] model pre-download skipped/failed (will lazy-download on first run)"
from huggingface_hub import snapshot_download
import os
for repo in ["lerobot/smolvla_libero"]:
    try:
        snapshot_download(repo_id=repo, cache_dir=os.environ.get("HF_HOME"))
        print("  [ok] cached", repo)
    except Exception as e:
        print("  [warn] could not cache", repo, "->", e)
PY
fi

exec "$@"
