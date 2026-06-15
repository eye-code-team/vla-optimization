# VLA Optimization — SmolVLA × LIBERO-10

Fine-tuning [SmolVLA](https://huggingface.co/lerobot/smolvla_libero) on the full LIBERO-10 benchmark with a hierarchical action-aware optimization stack.

## Architecture

The training pipeline wraps the SmolVLA base model with three complementary components:

| Component | Role |
|-----------|------|
| **CascadeSTARRouter** (Dynamic Layer Skipping) | Per-layer binary gates that learn to skip redundant transformer layers, reducing inference compute while preserving task accuracy |
| **HierarchicalADPPruner** (Dynamic Token Pruning) | Adaptive pruning of visual tokens in the VLM backbone, retaining only action-relevant patches |
| **CogKD Teacher** | Frozen teacher (smolvla_base) used for cognitive knowledge distillation during Phase 1 |

Training runs in phases:
- **Phase 1** (30 000 steps): Full model — task loss + gate regularization + temporal entropy + KD
- **Phase 2** (10 000 steps): Router specialization
- **Phase 3** (10 000 steps): Joint fine-tuning with learned skip schedule

## Dataset

LIBERO-10: 10 long-horizon manipulation tasks, 50 expert demos per task (500 total).

| Split | Episodes | Per task |
|-------|----------|----------|
| Train | 450 | 45 |
| Eval  | 50  | 5 (demos 45–49) |

Source: [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) — original LIBERO HDF5 at 20 Hz, converted to LeRobot v3.0 parquet format.

> **Image orientation note:** Raw HDF5 images are stored in MuJoCo native orientation (upside-down). The converter applies a 180° rotation so stored images match the pretrained SmolVLA base model's expected orientation (same as HuggingFaceVLA).

## Quickstart

### 1. Build the Docker image

```bash
docker compose build
```

### 2. Run the full pipeline (convert → split → train)

```bash
bash run_pipeline.sh
```

The pipeline runs three Docker containers sequentially:
1. **Convert** — downloads LIBERO-10 HDF5 and converts to LeRobot format
2. **Split** — builds `libero_10_full_train_episodes.json` (450 eps) and `libero_10_full_eval_episodes.json` (50 eps)
3. **Train** — Phase 1 fine-tuning for 30 000 steps, saves `outputs/hier_p1_full/phase1_complete.pt`

### 3. Evaluate

```bash
nohup bash -c '
printf "y\n/workspace/project/data/libero_datasets\ny\n" | \
  docker compose run -T --rm \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  smolvla bash -c "
    sed -i \"s/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g\" \
      /usr/local/lib/python3.10/dist-packages/libero/libero/benchmark/__init__.py && \
    python eval_libero_rollout.py \
      --checkpoint outputs/hier_p1_full/phase1_complete.pt \
      --suite libero_10 \
      --n_rollouts 5 \
      --skip_baseline \
      --save_video \
      --no_flip \
      --no_skip \
      --device cuda:1 \
      --output_dir outputs/eval_phase1_complete
  "
' > project/outputs/log_eval.txt 2>&1 &
```

Key eval flags:

| Flag | Meaning |
|------|---------|
| `--no_flip` | HDF5 images are stored in raw MuJoCo orientation (upside-down); eval env renders the same — no flip needed to match training |
| `--no_skip` | Training task loss is computed on the full model (skip gates are not applied during training forward pass); use full model at eval to match |
| `--n_rollouts 20` | LIBERO official protocol: 20 rollouts per task |

The eval script automatically:
- Loads stats from `data/datasets/libero_10_full/meta/stats.json` (matches training)
- Uses LIBERO's pre-sampled init states (`suite.get_task_init_states()`) — these are **independent** from HDF5 demo init states, so there is no train/test contamination concern

### LIBERO evaluation protocol (from source)

Based on reading `libero/benchmark/__init__.py`, `libero/lifelong/metric.py`, and `libero/envs/env_wrapper.py`:

| Parameter | Official value | Our script |
|-----------|---------------|------------|
| Init states | 50 pre-sampled per task (`.pruned_init` files, independent from demo HDF5) | ✓ `suite.get_task_init_states()` |
| Rollouts / task | 20 | ✓ `--n_rollouts 20` |
| Max horizon | 600 steps | ✓ `--horizon 600` |
| Settle steps | 5 zero-action steps after `set_init_state()` | ✓ `--settle_steps 5` |
| Image size | 128×128 (official policies); 256×256 for our fine-tuned model (trained on 256×256 HDF5) | ✓ `--img_size 256` |
| Image normalization | uint8 → float32 ÷ 255 | ✓ |
| Camera obs keys | `agentview_image`, `robot0_eye_in_hand_image` | ✓ |
| Action space | 7-DOF delta EEF (OSC\_POSE): Δxyz + Δaxis\_angle + gripper | ✓ |
| Success check | `env.check_success()` → BDDL predicate conjunction (all sub-goals must hold simultaneously) | ✓ |
| Success metric | `n_success / n_rollouts` per task → mean across all tasks | ✓ |

## File Overview

```
.
├── convert_libero_hdf5_to_lerobot.py   # HDF5 → LeRobot v3.0 parquet + stats
├── build_libero10_splits.py            # 45/5 train/eval split per task
├── download_libero.py                  # Dataset download helpers
├── finetune_config.py                  # Hyperparameters and architecture config
├── finetune_hierarchical_action_aware.py  # Main training loop (Phase 1/2/3)
├── eval_libero_rollout.py              # Live rollout evaluation on LIBERO suite
├── Dockerfile                          # CUDA 12.4 + PyTorch 2.6 + LIBERO env
├── docker-compose.yml                  # Service definition
├── requirements.txt                    # Python dependencies
└── run_pipeline.sh                     # End-to-end pipeline script
```

## Hardware

Tested on a single NVIDIA RTX 4090 (24 GB VRAM).

- Training: GPU 0, ~22 GB VRAM, ~47 hours for 30 000 steps (micro_batch=16)
- Evaluation: GPU 1 recommended to avoid OOM conflicts with training

## Results

Phase 1 training curve saved to `outputs/hier_p1_full/training_curves.png`.

Rollout evaluation results (JSON) saved to `outputs/eval_phase1_complete/results.json` after running the eval command above.
