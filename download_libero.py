"""
Download and inspect the HuggingFaceVLA/libero dataset (LeRobot format).
Filters to `libero_spatial` suite (10 tasks × 50 demos = ~500 episodes).
Saves episode list to data/datasets/libero_spatial_episodes.json.

Usage:
    python download_libero.py [--suite libero_spatial]

Suites available: libero_spatial, libero_object, libero_goal, libero_10, libero_90
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Parse args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_spatial",
                    choices=["libero_spatial", "libero_object", "libero_goal",
                             "libero_10", "libero_90"],
                    help="LIBERO suite to download (default: libero_spatial)")
parser.add_argument("--root", default="data/datasets/HuggingFaceVLA/libero",
                    help="Local directory to store the dataset")
parser.add_argument("--no-download", action="store_true",
                    help="Skip downloading; inspect already-cached dataset")
args = parser.parse_args()

REPO_ID = "HuggingFaceVLA/libero"
LOCAL_DIR = Path(args.root).resolve()
SUITE = args.suite
OUT_JSON = Path("data/datasets") / f"{SUITE}_episodes.json"

print(f"{'='*60}")
print(f"  LIBERO Dataset Downloader")
print(f"  Suite   : {SUITE}")
print(f"  Repo    : {REPO_ID}")
print(f"  LocalDir: {LOCAL_DIR}")
print(f"{'='*60}\n")

# ── Step 1: Download ─────────────────────────────────────────────────────────
if not args.no_download:
    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Downloading metadata + data files...")
    print("  (This downloads parquet + video files for ALL suites — may take a while)")
    print("  Tip: Use --no-download if already cached.\n")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(LOCAL_DIR),
            ignore_patterns=["*.git", "*.gitattributes"],
        )
        print("  Download complete.\n")
    except Exception as e:
        print(f"\nERROR during download: {e}")
        print("\nIf you see a 403 / unauthorized error, run:")
        print("  huggingface-cli login")
        print("  (or set HF_TOKEN environment variable)")
        sys.exit(1)
else:
    print("Step 1: Skipping download (--no-download flag set)\n")

# ── Step 2: Inspect metadata ─────────────────────────────────────────────────
print("Step 2: Inspecting dataset metadata...")

import pandas as pd

meta_dir = LOCAL_DIR / "meta"
tasks_path = meta_dir / "tasks.parquet"
episodes_path = meta_dir / "episodes.parquet"

if not tasks_path.exists():
    # Some dataset versions store meta under a different path
    alt_tasks = list(LOCAL_DIR.rglob("tasks.parquet"))
    if alt_tasks:
        tasks_path = alt_tasks[0]
        meta_dir = tasks_path.parent
        episodes_path = meta_dir / "episodes.parquet"
        print(f"  Found tasks.parquet at: {tasks_path}")
    else:
        print("ERROR: meta/tasks.parquet not found. Did the download complete?")
        sys.exit(1)

tasks_df = pd.read_parquet(tasks_path)
print(f"\n  tasks.parquet columns: {list(tasks_df.columns)}")
print(f"  Total tasks: {len(tasks_df)}")
print(f"\n  First 5 tasks:")
print(tasks_df.head())

# ── Step 3: Identify suite task indices ──────────────────────────────────────
print(f"\nStep 3: Identifying {SUITE} task indices...")

# Column names vary; find the description column
desc_col = None
for col in ["task", "task_description", "description", "language_instruction"]:
    if col in tasks_df.columns:
        desc_col = col
        break

if desc_col is None:
    print(f"  WARNING: Could not find task description column. Columns: {list(tasks_df.columns)}")
    print("  Falling back to task_index column if available.")

# Check if tasks have a suite label or if we need to detect from description
suite_task_indices = []

# Try direct suite column
if "suite" in tasks_df.columns:
    suite_tasks = tasks_df[tasks_df["suite"] == SUITE]
    suite_task_indices = suite_tasks.index.tolist()
    print(f"  Found 'suite' column — {len(suite_task_indices)} tasks for {SUITE}")

# Try task_index column with known ordering
elif "task_index" in tasks_df.columns:
    # LIBERO ordering in HuggingFaceVLA/libero:
    #   libero_spatial : task_index  0..9
    #   libero_object  : task_index 10..19
    #   libero_goal    : task_index 20..29
    #   libero_10      : task_index 30..39
    #   libero_90      : task_index 40..129
    SUITE_TASK_RANGES = {
        "libero_spatial": (0, 10),
        "libero_object":  (10, 20),
        "libero_goal":    (20, 30),
        "libero_10":      (30, 40),
        "libero_90":      (40, 130),
    }
    lo, hi = SUITE_TASK_RANGES[SUITE]
    total_tasks = len(tasks_df)
    # Clamp to actual task count
    lo = min(lo, total_tasks)
    hi = min(hi, total_tasks)
    suite_task_indices = list(range(lo, hi))
    print(f"  Using task_index range [{lo}, {hi}) for {SUITE} ({len(suite_task_indices)} tasks)")

# Try description-based detection
elif desc_col:
    # Match suite name keyword in task description
    keyword = SUITE.replace("_", " ").replace("libero ", "").lower()
    mask = tasks_df[desc_col].str.lower().str.contains(keyword, na=False)
    suite_task_indices = tasks_df[mask].index.tolist()
    if not suite_task_indices:
        # Fall back: assume 10 tasks starting at index 0
        print(f"  WARNING: No keyword match for '{keyword}'. Using first 10 tasks as fallback.")
        suite_task_indices = list(range(min(10, len(tasks_df))))
    else:
        print(f"  Keyword match found {len(suite_task_indices)} tasks for {SUITE}")

print(f"  Suite task indices: {suite_task_indices}")
if desc_col and suite_task_indices:
    print(f"  Task descriptions:")
    for i in suite_task_indices:
        if i < len(tasks_df):
            print(f"    [{i}] {tasks_df.iloc[i][desc_col]}")

# ── Step 4: Map task indices → episode IDs ───────────────────────────────────
print(f"\nStep 4: Mapping task indices to episode IDs...")

if not episodes_path.exists():
    alt_eps = list(LOCAL_DIR.rglob("episodes.parquet"))
    if alt_eps:
        episodes_path = alt_eps[0]
    else:
        print("WARNING: meta/episodes.parquet not found. Will use all episodes.")
        episodes_path = None

suite_episode_ids = []

if episodes_path and episodes_path.exists():
    episodes_df = pd.read_parquet(episodes_path)
    print(f"  episodes.parquet columns: {list(episodes_df.columns)}")
    print(f"  Total episodes: {len(episodes_df)}")

    # Find episode_id and task_index columns
    ep_id_col = next((c for c in ["episode_index", "episode_id", "index"] 
                      if c in episodes_df.columns), None)
    ep_task_col = next((c for c in ["task_index", "task_id"] 
                        if c in episodes_df.columns), None)

    if ep_id_col and ep_task_col:
        mask = episodes_df[ep_task_col].isin(suite_task_indices)
        suite_episode_ids = episodes_df[mask][ep_id_col].tolist()
        print(f"  {SUITE} episodes: {len(suite_episode_ids)} "
              f"(from ep {min(suite_episode_ids)} to ep {max(suite_episode_ids)})")
    else:
        print(f"  WARNING: Could not find episode_id/task_index columns: {list(episodes_df.columns)}")
        # Fallback: assume first N episodes (50 demos × 10 tasks)
        n_demos = 50
        n_tasks = len(suite_task_indices)
        suite_episode_ids = list(range(n_demos * n_tasks))
        print(f"  Fallback: using first {len(suite_episode_ids)} episodes")
else:
    # No episodes.parquet — assume 50 demos × 10 tasks at start
    n_demos = 50
    n_tasks = len(suite_task_indices) if suite_task_indices else 10
    suite_episode_ids = list(range(n_demos * n_tasks))
    print(f"  No episodes.parquet found. Fallback: using first {len(suite_episode_ids)} episodes.")

# ── Step 5: Save episode list ─────────────────────────────────────────────────
print(f"\nStep 5: Saving episode list...")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

result = {
    "repo_id": REPO_ID,
    "suite": SUITE,
    "task_indices": suite_task_indices,
    "episode_ids": suite_episode_ids,
    "num_episodes": len(suite_episode_ids),
    "local_dir": str(LOCAL_DIR),
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"  Saved: {OUT_JSON}  ({len(suite_episode_ids)} episodes)")

# ── Step 6: Verify dataset loads ─────────────────────────────────────────────
print(f"\nStep 6: Verifying dataset loads with LeRobotDataset...")

try:
    sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load just first 3 episodes to verify
    probe_eps = suite_episode_ids[:3]
    print(f"  Loading probe episodes: {probe_eps} ...")
    ds = LeRobotDataset(REPO_ID, root=str(LOCAL_DIR), episodes=probe_eps)
    sample = ds[0]
    all_keys = list(sample.keys())

    image_keys = [k for k in all_keys if "image" in k.lower()]
    state_key = next((k for k in all_keys if "state" in k.lower()), None)
    action_dim = sample["action"].shape[-1] if "action" in sample else "N/A"
    state_dim = sample[state_key].shape[-1] if state_key else "N/A"

    print(f"\n  ✓ Dataset loaded successfully!")
    print(f"  Episodes (probe): {ds.num_episodes}, Frames: {len(ds)}, FPS: {ds.fps}")
    print(f"  Image keys : {image_keys}")
    print(f"  State key  : {state_key} ({state_dim}D)")
    print(f"  Action dim : {action_dim}D")
    print(f"  All keys   : {all_keys}")

    if action_dim != 7:
        print(f"\n  WARNING: Expected ACTION_DIM=7 for LIBERO, got {action_dim}")
    if state_dim != 8:
        print(f"\n  WARNING: Expected STATE_DIM=8 for LIBERO, got {state_dim}")
    if len(image_keys) < 2:
        print(f"\n  WARNING: Expected 2 image keys for LIBERO, got {len(image_keys)}")

except ImportError as e:
    print(f"  WARNING: Could not import LeRobotDataset — {e}")
    print("  Run this script from the EyetechCode directory with the plantgpu env.")
except Exception as e:
    print(f"  WARNING: Dataset load check failed — {e}")
    print("  The episode list was still saved; you can proceed with training config.")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  Suite        : {SUITE}")
print(f"  Episodes     : {len(suite_episode_ids)}")
if suite_episode_ids:
    print(f"  Episode range: {min(suite_episode_ids)} .. {max(suite_episode_ids)}")
print(f"  Episode JSON : {OUT_JSON}")
print(f"\n  Add this to finetune_config.py DATASETS dict:")
print(f"""
  "libero_{SUITE.replace('libero_', '')}": {{
      "repo_id": "{REPO_ID}",
      "root": "{str(LOCAL_DIR).replace(chr(92), '/')}",
      "episodes": <list from {OUT_JSON}>,
      "task_instruction": "Perform the manipulation task.",
      "train_episodes": {int(len(suite_episode_ids)*0.8)},
      "eval_episodes":  {int(len(suite_episode_ids)*0.2)},
  }},
""")
