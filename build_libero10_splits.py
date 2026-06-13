"""
build_libero10_splits.py
Builds train/eval episode splits for LIBERO-10 — robustly.

IMPORTANT (fixed 2026-06-04):
  The HuggingFaceVLA/libero dataset's task_index ordering does NOT match the
  positional assumption (old code assumed libero_10 == task_index 30-39, which
  is actually libero_SPATIAL). The libero_10 long-horizon tasks live at
  task_index 0-9, but in a DIFFERENT order than suite.get_task(i).

  We therefore resolve the correct dataset task_index for each suite task by
  MATCHING THE INSTRUCTION STRING (suite.get_task(i).language ==
  tasks.parquet instruction). This is version-independent and order-independent.

LIBERO-10: 10 long-horizon tasks, ~29–49 demos each = 379 episodes total.
  • Train : first (N-TEST_PER_TASK) episodes per task  (mentor: 45/task if N=50)
  • Test  : last TEST_PER_TASK episodes per task        (mentor: 5/task)
  NOTE: HuggingFace data has 29–49 eps/task (not 50), so some tasks yield
        fewer than 45 train eps. Split still respects last-5-as-test per task.

Outputs
-------
  data/datasets/libero_10_train_episodes.json
  data/datasets/libero_10_eval_episodes.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
sys.path.insert(0, "lerobot/src")
OUT_DIR = Path("data/datasets")

_ap = argparse.ArgumentParser(description="Build LIBERO train/eval episode splits "
                                          "(resolved by instruction-string match).")
_ap.add_argument("--suite", default="libero_10",
                 choices=["libero_10", "libero_spatial", "libero_object",
                          "libero_goal", "libero_90"],
                 help="LIBERO suite to build splits for")
_ap.add_argument("--test_per_task", type=int, default=5,
                 help="Number of episodes per task held out for test (last N). Default: 5")
_ap.add_argument("--dataset_root", default=None,
                 help="Path to LeRobot dataset root. "
                      "Defaults to data/datasets/libero_10_full for libero_10_full, "
                      "or data/datasets/HuggingFaceVLA/libero for libero_10.")
_ap.add_argument("--use_full", action="store_true",
                 help="Shortcut: use data/datasets/libero_10_full (50 demos/task)")
_args = _ap.parse_args()
SUITE = _args.suite
TEST_PER_TASK = _args.test_per_task

# Resolve dataset root
if _args.dataset_root:
    LIBERO_ROOT = Path(_args.dataset_root)
elif _args.use_full:
    LIBERO_ROOT = Path("data/datasets/libero_10_full")
else:
    LIBERO_ROOT = Path("data/datasets/HuggingFaceVLA/libero")

TASKS_PARQUET = LIBERO_ROOT / "meta" / "tasks.parquet"
EPISODES_PARQUET_CANDIDATES = [
    LIBERO_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    LIBERO_ROOT / "meta" / "episodes.parquet",
]


def _norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def _to_int(val):
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val).ravel()
        return int(arr[0]) if arr.size else -1
    return int(val)


# ── Resolve suite instruction → dataset task_index (string match) ────────────
from libero.libero import benchmark

suite = benchmark.get_benchmark_dict()[SUITE]()
n_tasks = suite.get_num_tasks()

tasks_df = pd.read_parquet(TASKS_PARQUET).reset_index()
str_col = next(c for c in tasks_df.columns if tasks_df[c].dtype == object)
idx_col = next(c for c in tasks_df.columns if c != str_col)
instr_to_dsidx = {_norm(r[str_col]): int(r[idx_col]) for _, r in tasks_df.iterrows()}
print(f"  tasks.parquet: {len(instr_to_dsidx)} tasks  (str col={str_col!r}, idx col={idx_col!r})")

# suite_local_index -> dataset_task_index, and reverse map for language alignment
suite_to_ds = {}
ds_to_suite = {}
print(f"\n  Resolving {SUITE} tasks by instruction match:")
for li in range(n_tasks):
    t = suite.get_task(li)
    lang = _norm(getattr(t, "language", getattr(t, "language_instruction", t.name)))
    ds_idx = instr_to_dsidx.get(lang)
    if ds_idx is None:
        print(f"  ERROR: suite task {li} instruction not found in dataset:\n    {lang!r}")
        sys.exit(1)
    suite_to_ds[li] = ds_idx
    ds_to_suite[ds_idx] = li
    print(f"    suite {li:2d}  ->  dataset task_index {ds_idx:2d}  | {lang}")

ds_task_ids = sorted(suite_to_ds.values())
print(f"\n  Matched dataset task_indices for {SUITE}: {ds_task_ids}")

# ── Locate episodes.parquet ──────────────────────────────────────────────────
episodes_path = next((c for c in EPISODES_PARQUET_CANDIDATES if c.exists()), None)
if episodes_path is None:
    print("ERROR: episodes.parquet not found in:", EPISODES_PARQUET_CANDIDATES)
    sys.exit(1)
eps_df = pd.read_parquet(episodes_path)

id_col = next((c for c in ["episode_index", "episode_id", "index"] if c in eps_df.columns), None)
task_col = next(
    (c for c in ["task_index", "task_id", "stats/task_index/min"] if c in eps_df.columns), None
)
if id_col is None or task_col is None:
    print(f"ERROR: cannot find id/task columns. Available: {list(eps_df.columns)}")
    sys.exit(1)
eps_df["_t"] = eps_df[task_col].apply(_to_int)
print(f"\n  episodes.parquet: {len(eps_df)} episodes  (id col={id_col!r}, task col={task_col!r})")

# ── Build splits: last TEST_PER_TASK → test, rest → train ────────────────────
train_ids, test_ids = [], []
task_summary = {}
print(f"\n  Split: last {TEST_PER_TASK} eps/task → test, rest → train")
for ds_idx in ds_task_ids:
    ep_ids = sorted(int(x) for x in eps_df[eps_df["_t"] == ds_idx][id_col].tolist())
    n = len(ep_ids)
    n_test = min(TEST_PER_TASK, n)
    n_train = n - n_test
    t_ids = ep_ids[:n_train]
    e_ids = ep_ids[n_train:]
    train_ids.extend(t_ids)
    test_ids.extend(e_ids)
    task_summary[ds_idx] = {
        "suite_local_index": ds_to_suite[ds_idx],
        "num_episodes": n,
        "num_train": n_train,
        "num_test": n_test,
        "ep_range": [ep_ids[0], ep_ids[-1]] if ep_ids else [],
    }
    print(f"    ds task {ds_idx:2d} (suite {ds_to_suite[ds_idx]:2d}): "
          f"{n:2d} eps  →  {n_train:2d} train + {n_test} test  "
          f"(ep {ep_ids[0]}–{ep_ids[-1]})")

train_ids = sorted(train_ids)
test_ids  = sorted(test_ids)
print(f"\n  Train total : {len(train_ids)} episodes across {len(task_summary)} tasks")
print(f"  Test  total : {len(test_ids)}  episodes across {len(task_summary)} tasks")

missing = [d for d, s in task_summary.items() if s["num_episodes"] == 0]
print("  [OK] All tasks present" if not missing else f"  WARNING: empty tasks {missing}")

# ── Save ─────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
base_payload = {
    "suite": SUITE,
    "num_tasks": len(task_summary),
    "test_per_task": TEST_PER_TASK,
    "suite_to_dataset_task_index": suite_to_ds,
    "dataset_task_summary": task_summary,
    "note": "Resolved by instruction-string match; libero_10 == dataset task_index 0-9. "
            "Split: last TEST_PER_TASK eps/task → test, rest → train.",
}

splits = {
    "train": train_ids,
    "eval":  test_ids,
}
# Use _full suffix when using the full 500-episode dataset
_suffix = "_full" if (_args.use_full or (_args.dataset_root and "libero_10_full" in str(_args.dataset_root))) else ""
_out_key = f"{SUITE}{_suffix}"

for split, ids in splits.items():
    out = OUT_DIR / f"{_out_key}_{split}_episodes.json"
    with open(out, "w") as f:
        json.dump({**base_payload, "split": split, "dataset_root": str(LIBERO_ROOT),
                   "num_episodes": len(ids), "episode_ids": ids}, f, indent=2)
    print(f"  Saved: {out}  ({len(ids)} eps)")

_dataset_arg = f"libero_10{_suffix}"
print(f"\n  DONE — train: python finetune_hierarchical_action_aware.py --dataset {_dataset_arg}")
print( "         eval : python eval_libero_rollout.py --suite libero_10")
