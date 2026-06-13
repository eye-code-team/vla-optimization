#!/usr/bin/env bash
# run_pipeline.sh — sequential: convert → split → train
# Run with: nohup bash run_pipeline.sh > project/outputs/log_pipeline.txt 2>&1 &
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo " PIPELINE START  $(date)"
echo "=========================================="

# ── STEP 1: Convert HDF5 → LeRobot ──────────────────────────────────────────
echo ""
echo "[STEP 1] Convert LIBERO-10 HDF5 → LeRobot format  $(date)"
docker compose run --rm -T smolvla bash -c \
  "rm -rf data/datasets/libero_10_full && \
   python convert_libero_hdf5_to_lerobot.py --no_download"
echo "[STEP 1] Done  $(date)"

# ── STEP 2: Build 450 train / 50 test splits ────────────────────────────────
echo ""
echo "[STEP 2] Build episode splits  $(date)"
printf 'y\n/workspace/project/data/libero_datasets\ny\n' | \
  docker compose run -T --rm smolvla \
  python build_libero10_splits.py --suite libero_10 --use_full --test_per_task 5
echo "[STEP 2] Done  $(date)"

# ── STEP 3: Train Phase 1 — 30 000 steps ─────────────────────────────────────
echo ""
echo "[STEP 3] Training Phase 1 — 30000 steps  $(date)"
CUDA_VISIBLE_DEVICES=0 docker compose run --rm smolvla \
  python finetune_hierarchical_action_aware.py \
    --dataset libero_10_full \
    --start_phase 1 --end_phase 1 \
    --phase1_steps 30000 \
    --micro_batch 16 \
    --grad_accum 1 \
    --task_priority \
    --output_name hier_p1_full
echo "[STEP 3] Done  $(date)"

echo ""
echo "=========================================="
echo " PIPELINE COMPLETE  $(date)"
echo "=========================================="
