"""
convert_libero_hdf5_to_lerobot.py
==================================
Download LIBERO-10 HDF5 (50 demos/task × 10 tasks = 500 episodes) from
yifengzhu-hf/LIBERO-datasets and convert to LeRobot v3.0 format compatible
with the existing training pipeline.

Output: data/datasets/libero_10_full/  (identical schema to HuggingFaceVLA/libero)

Usage (inside Docker or local):
  # Download + convert (default)
  python convert_libero_hdf5_to_lerobot.py

  # Skip download (HDF5 already cached at data/datasets/libero_10_hdf5/)
  python convert_libero_hdf5_to_lerobot.py --no_download

  # Custom paths
  python convert_libero_hdf5_to_lerobot.py \
      --hdf5_dir data/datasets/libero_10_hdf5 \
      --out_dir  data/datasets/libero_10_full

Format details:
  - Images: PNG bytes inline in parquet {'bytes': ..., 'path': 'frame_XXXXXX.png'}
  - observation.state: [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)] = 8-D float32
  - action: 7-D float32
  - FPS: 20 Hz (native LIBERO rate; images at 256×256)
  - ~3 episodes per parquet file (matches HuggingFaceVLA/libero density)
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO_ID    = "yifengzhu-hf/LIBERO-datasets"
SUITE_SUBDIR  = "libero_10"
IMAGE_H       = 256
IMAGE_W       = 256
FPS           = 20.0
EPISODES_PER_FILE = 3   # group N episodes per parquet file (mirrors existing dataset)

# ── Argument parsing ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--hdf5_dir",    default="data/datasets/libero_10_hdf5",
                help="Directory where HDF5 files are cached (or will be downloaded)")
ap.add_argument("--out_dir",     default="data/datasets/libero_10_full",
                help="Output directory for the LeRobot dataset")
ap.add_argument("--no_download", action="store_true",
                help="Skip HuggingFace download; use --hdf5_dir as-is")
ap.add_argument("--fps",         type=float, default=FPS,
                help=f"Target FPS (default: {FPS}). Set to 10 to halve frame rate.")
args = ap.parse_args()

HDF5_DIR = Path(args.hdf5_dir)
OUT_DIR  = Path(args.out_dir)
FPS      = args.fps


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download HDF5 files from yifengzhu-hf/LIBERO-datasets
# ══════════════════════════════════════════════════════════════════════════════

def download_hdf5():
    """Download libero_10 subfolder from yifengzhu-hf/LIBERO-datasets."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        sys.exit(1)

    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Downloading {HF_REPO_ID}/{SUITE_SUBDIR} → {HDF5_DIR} …")
    print("      (This is the ORIGINAL LIBERO dataset: 50 demos/task × 10 tasks)")

    local = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(HDF5_DIR),
        allow_patterns=[f"{SUITE_SUBDIR}/*", f"{SUITE_SUBDIR}/**"],
    )
    print(f"      Downloaded to: {local}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Locate and optionally unzip HDF5 files
# ══════════════════════════════════════════════════════════════════════════════

def locate_hdf5_files() -> list[Path]:
    """Return sorted list of .hdf5 files (unzip if needed)."""
    search_root = HDF5_DIR / SUITE_SUBDIR if (HDF5_DIR / SUITE_SUBDIR).exists() else HDF5_DIR

    # Unzip any .zip files first
    for zp in list(search_root.glob("*.zip")):
        print(f"  Unzipping {zp.name} …")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(search_root)
        zp.unlink()

    hdf5_files = sorted(search_root.rglob("*.hdf5"))
    if not hdf5_files:
        print(f"ERROR: no .hdf5 files found under {search_root}")
        sys.exit(1)
    print(f"[2/4] Found {len(hdf5_files)} HDF5 file(s):")
    for f in hdf5_files:
        print(f"       {f.name}")
    return hdf5_files


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Helper: encode image as PNG bytes
# ══════════════════════════════════════════════════════════════════════════════

def _encode_png(arr: np.ndarray) -> bytes:
    """Convert HxWxC uint8 ndarray to PNG bytes, resizing to IMAGE_H×IMAGE_W.

    Raw MuJoCo/robosuite renders are 180° rotated vs human-viewable orientation.
    We rotate here so stored images match HuggingFaceVLA orientation, keeping
    the dataset consistent with the pretrained SmolVLA base model's expectations.
    """
    # 180° rotation = flip both axes; matches HuggingFaceVLA / pretrained base
    arr = arr[::-1, ::-1].copy()
    img = Image.fromarray(arr.astype(np.uint8))
    if img.size != (IMAGE_W, IMAGE_H):
        img = img.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()



# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Convert HDF5 files → LeRobot parquet
# ══════════════════════════════════════════════════════════════════════════════

def convert(hdf5_files: list[Path]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = OUT_DIR / "data" / "chunk-000"
    meta_dir = OUT_DIR / "meta"
    ep_meta_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    ep_meta_dir.mkdir(parents=True, exist_ok=True)

    all_episodes = []   # list of episode-level dicts (for episodes.parquet)
    task_index_map: dict[str, int] = {}  # instruction → task_index

    episode_index   = 0   # global episode counter
    global_frame_idx = 0  # global frame counter
    file_index      = 0   # parquet file counter within chunk-000

    pending_rows: list[dict] = []   # accumulates rows until we flush a file
    pending_ep_start = 0            # episode_index of first ep in current file

    # Online stats accumulators for state (8D) and action (7D)
    n_frames   = 0
    state_sum  = np.zeros(8,  dtype=np.float64)
    state_sq   = np.zeros(8,  dtype=np.float64)
    state_min  = np.full(8,   np.inf)
    state_max  = np.full(8,  -np.inf)
    action_sum = np.zeros(7,  dtype=np.float64)
    action_sq  = np.zeros(7,  dtype=np.float64)
    action_min = np.full(7,   np.inf)
    action_max = np.full(7,  -np.inf)

    def _flush_parquet():
        nonlocal file_index, pending_rows, pending_ep_start
        if not pending_rows:
            return
        df = pd.DataFrame(pending_rows)
        path = data_dir / f"file-{file_index:03d}.parquet"
        df.to_parquet(path, index=False)
        # Update all_episodes with the correct file_index for these episodes
        for ep in all_episodes:
            if pending_ep_start <= ep["episode_index"] < pending_ep_start + EPISODES_PER_FILE:
                ep["data/file_index"] = file_index
        file_index += 1
        pending_rows.clear()
        pending_ep_start = episode_index

    print(f"[3/4] Converting {len(hdf5_files)} HDF5 files …")
    for hdf5_path in tqdm(hdf5_files, desc="Tasks"):
        with h5py.File(hdf5_path, "r") as f:
            # — language instruction (stored per-file in data.attrs["problem_info"])
            problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
            lang = (problem_info.get("language_instruction")
                    or problem_info.get("problem_name", "")
                    or hdf5_path.stem.replace("_demo", "").replace("_", " ").lower())

            if lang not in task_index_map:
                task_index_map[lang] = len(task_index_map)
            t_idx = task_index_map[lang]

            # — sorted demo list
            demos = sorted(
                f["data"].keys(),
                key=lambda k: int(k.split("_")[-1]) if "_" in k else 0
            )
            tqdm.write(f"  task {t_idx:2d}: {len(demos)} demos | {lang[:60]}")

            for demo_key in demos:
                demo = f["data"][demo_key]

                # ── Read arrays ──────────────────────────────────────────────
                try:
                    img1    = demo["obs"]["agentview_rgb"][:]    # (T, 128, 128, 3) uint8
                    img2    = demo["obs"]["eye_in_hand_rgb"][:]  # (T, 128, 128, 3) uint8
                    ee_pos  = demo["obs"]["ee_pos"][:]           # (T, 3) float64
                    ee_ori  = demo["obs"]["ee_ori"][:]           # (T, 3) axis-angle float64
                    gripper = demo["obs"]["gripper_states"][:]   # (T, 2) float64
                    actions = demo["actions"][:]                  # (T, 7) float64
                except KeyError as e:
                    tqdm.write(f"    SKIP {demo_key}: missing key {e}")
                    continue

                T = len(actions)
                if T == 0:
                    continue

                # ── Optional frame-rate halving (20 Hz → 10 Hz) ─────────────
                if args.fps < FPS - 0.5:
                    stride = max(1, round(FPS / args.fps))
                    idx = np.arange(0, T, stride)
                    img1    = img1[idx]
                    img2    = img2[idx]
                    ee_pos  = ee_pos[idx]
                    ee_ori  = ee_ori[idx]
                    gripper = gripper[idx]
                    actions = actions[idx]
                    T = len(idx)

                # ── Compute 8-D state: ee_pos(3) + ee_ori_axis_angle(3) + gripper(2)
                states = np.concatenate([
                    ee_pos.astype(np.float32),    # (T, 3)
                    ee_ori.astype(np.float32),    # (T, 3) already axis-angle
                    gripper.astype(np.float32),   # (T, 2)
                ], axis=1)  # (T, 8)

                # ── Accumulate stats ─────────────────────────────────────────
                n_frames   += T
                state_sum  += states.sum(axis=0).astype(np.float64)
                state_sq   += (states ** 2).sum(axis=0).astype(np.float64)
                state_min   = np.minimum(state_min, states.min(axis=0))
                state_max   = np.maximum(state_max, states.max(axis=0))
                act_f       = actions[: T].astype(np.float64) if isinstance(actions, np.ndarray) else actions[:T].astype(np.float64)
                action_sum += act_f.sum(axis=0)
                action_sq  += (act_f ** 2).sum(axis=0)
                action_min  = np.minimum(action_min, act_f.min(axis=0))
                action_max  = np.maximum(action_max, act_f.max(axis=0))

                # ── Track episode metadata ───────────────────────────────────
                ep_from = global_frame_idx
                ep_to   = global_frame_idx + T  # exclusive upper bound

                # ── Write per-frame rows ─────────────────────────────────────
                for t in range(T):
                    png1 = _encode_png(img1[t])
                    png2 = _encode_png(img2[t])
                    pending_rows.append({
                        "observation.images.image":  {"bytes": png1, "path": f"frame_{global_frame_idx:06d}.png"},
                        "observation.images.image2": {"bytes": png2, "path": f"frame_{global_frame_idx:06d}.png"},
                        "observation.state": states[t],
                        "action": actions[t].astype(np.float32),
                        "timestamp": float(t) / args.fps,
                        "frame_index": t,
                        "episode_index": episode_index,
                        "index": global_frame_idx,
                        "task_index": t_idx,
                    })
                    global_frame_idx += 1

                # ── Episode summary (file_index filled in during _flush) ─────
                all_episodes.append({
                    "episode_index": episode_index,
                    "task_index": t_idx,             # needed by build_libero10_splits.py
                    "data/chunk_index": 0,
                    "data/file_index": -1,           # placeholder; filled on flush
                    "dataset_from_index": ep_from,
                    "dataset_to_index": ep_to,
                    "tasks": np.array([lang], dtype=object),
                    "length": T,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                })
                episode_index   += 1

                # Flush every EPISODES_PER_FILE
                if episode_index % EPISODES_PER_FILE == 0:
                    _flush_parquet()

    # Flush remaining
    _flush_parquet()

    total_episodes = episode_index
    total_frames   = global_frame_idx
    print(f"\n  Processed: {total_episodes} episodes, {total_frames} frames")

    # ── Fix any episodes whose file_index was never back-patched ────────────
    for ep in all_episodes:
        if ep["data/file_index"] == -1:
            ep["data/file_index"] = file_index - 1

    # ══════════════════════════════════════════════════════════════════════════
    # Write metadata
    # ══════════════════════════════════════════════════════════════════════════
    print("[4/4] Writing metadata …")

    # ── meta/tasks.parquet ───────────────────────────────────────────────────
    tasks_df = pd.DataFrame(
        {"task_index": list(task_index_map.values())},
        index=pd.Index(list(task_index_map.keys()), name=None),
    )
    tasks_df.to_parquet(meta_dir / "tasks.parquet")
    print(f"  tasks.parquet: {len(tasks_df)} tasks")

    # ── meta/episodes/chunk-000/file-000.parquet ─────────────────────────────
    ep_df = pd.DataFrame(all_episodes)
    ep_df.to_parquet(ep_meta_dir / "file-000.parquet", index=False)
    print(f"  episodes.parquet: {len(ep_df)} episodes")

    # ── meta/info.json ───────────────────────────────────────────────────────
    info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_index_map),
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.image": {
                "dtype": "image",
                "shape": [IMAGE_H, IMAGE_W, 3],
                "names": ["height", "width", "channel"],
                "fps": args.fps,
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": [IMAGE_H, IMAGE_W, 3],
                "names": ["height", "width", "channel"],
                "fps": args.fps,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["state"],
                "fps": args.fps,
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["actions"],
                "fps": args.fps,
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "fps": args.fps},
            "frame_index":   {"dtype": "int64",   "shape": [1], "fps": args.fps},
            "episode_index": {"dtype": "int64",   "shape": [1], "fps": args.fps},
            "index":         {"dtype": "int64",   "shape": [1], "fps": args.fps},
            "task_index":    {"dtype": "int64",   "shape": [1], "fps": args.fps},
        },
    }
    with open(meta_dir / "info.json", "w") as fh:
        json.dump(info, fh, indent=2)

    # ── meta/stats.json ──────────────────────────────────────────────────────
    # Computed from all frames in one pass (online mean/std/min/max).
    N = float(n_frames)
    state_mean = state_sum / N
    state_std  = np.sqrt(np.maximum(state_sq / N - state_mean ** 2, 1e-8))
    act_mean   = action_sum / N
    act_std    = np.sqrt(np.maximum(action_sq / N - act_mean ** 2, 1e-8))

    stats = {
        "observation.state": {
            "mean": state_mean.tolist(),
            "std":  state_std.tolist(),
            "min":  state_min.tolist(),
            "max":  state_max.tolist(),
        },
        "action": {
            "mean": act_mean.tolist(),
            "std":  act_std.tolist(),
            "min":  action_min.tolist(),
            "max":  action_max.tolist(),
        },
    }
    with open(meta_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  stats.json: state mean={[f'{v:.3f}' for v in state_mean.tolist()]}")

    print(f"\n  Output dataset: {OUT_DIR}")
    print(f"  Total episodes : {total_episodes}  (expected 500 = 50 × 10 tasks)")
    print(f"  Total frames   : {total_frames}")
    print(f"  Total tasks    : {len(task_index_map)}")
    print(f"\n  Next step: run build_libero10_splits.py --dataset_root {OUT_DIR}")
    print( "            then: python finetune_hierarchical_action_aware.py --dataset libero_10_full")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not args.no_download:
        download_hdf5()
    else:
        print("[1/4] Skipping download (--no_download)")

    hdf5_files = locate_hdf5_files()
    convert(hdf5_files)
    print("\nDONE.")
