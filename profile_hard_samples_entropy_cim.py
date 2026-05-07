"""
Profile hard samples for Entropy-CIM semi-finetuning.

This script reads per-timestep CSV metrics (for example ep40/ep44), builds a
proxy error signal, applies a hybrid threshold schedule (fixed warm-up then
rolling percentile), and exports:
  - hard_samples_manifest.json
  - hard_samples_summary.csv

The manifest is designed to be consumed by finetune_semifinetune_entropy_cim.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class EpisodeRows:
    episode_id: int
    rows: List[Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-sample profiler for Entropy-CIM")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="d:/EyetechCode/results/dynamic_lora_pruning_snapflow",
        help="Directory containing dynamic_ep*_timestep_metrics.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="d:/EyetechCode/results/dynamic_lora_pruning_snapflow",
        help="Directory where manifest/summary are written.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="40,44",
        help="Comma-separated episode IDs to include (for example: 40,44).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=80,
        help="Warm-up steps using fixed tau before adaptive percentile.",
    )
    parser.add_argument(
        "--fixed_tau",
        type=float,
        default=0.62,
        help="Fixed threshold used during warm-up on proxy MSE in [0,1].",
    )
    parser.add_argument(
        "--adaptive_percentile",
        type=float,
        default=85.0,
        help="Percentile for adaptive threshold after warm-up.",
    )
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=50,
        help="Rolling window size for adaptive threshold computation.",
    )
    parser.add_argument(
        "--weight_scale",
        type=float,
        default=2.5,
        help="Maximum curriculum amplification for severe hard samples.",
    )
    parser.add_argument(
        "--min_severity",
        type=float,
        default=0.05,
        help="Minimum severity to keep a sample in the manifest.",
    )
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_episode_ids(episodes_text: str) -> List[int]:
    parts = [p.strip() for p in episodes_text.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return sorted(set(out))


def _csv_path(input_dir: Path, episode_id: int) -> Path:
    return input_dir / f"dynamic_ep{episode_id}_timestep_metrics.csv"


def _load_episode_rows(input_dir: Path, episode_id: int) -> EpisodeRows | None:
    csv_path = _csv_path(input_dir, episode_id)
    if not csv_path.exists():
        return None

    rows: List[Dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "step": _safe_float(row.get("step"), 0.0),
                    "skip_ratio": _safe_float(row.get("skip_ratio"), 0.0),
                    "token_keep_ratio": _safe_float(row.get("token_keep_ratio"), 1.0),
                    "visual_entropy": _safe_float(row.get("visual_entropy"), 0.0),
                    "ee_velocity": _safe_float(row.get("ee_velocity"), 0.0),
                    "policy_step_ms": _safe_float(
                        row.get("policy_step_ms", row.get("latency_model_ms", 0.0)),
                        0.0,
                    ),
                }
            )

    return EpisodeRows(episode_id=episode_id, rows=rows)


def _global_minmax(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    return vmin, vmax


def _norm(x: float, vmin: float, vmax: float) -> float:
    return float(np.clip((x - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0))


def _phase_name(step: int, total_steps: int) -> str:
    if total_steps <= 1:
        return "interaction"
    ratio = step / float(max(total_steps - 1, 1))
    if ratio < 0.33:
        return "approach"
    if ratio < 0.66:
        return "interaction"
    return "retreat"


def build_proxy_scores(episodes: List[EpisodeRows]) -> List[EpisodeRows]:
    all_entropy: List[float] = []
    all_velocity: List[float] = []
    all_policy_ms: List[float] = []

    for ep in episodes:
        for r in ep.rows:
            all_entropy.append(r["visual_entropy"])
            all_velocity.append(r["ee_velocity"])
            all_policy_ms.append(r["policy_step_ms"])

    ent_min, ent_max = _global_minmax(all_entropy)
    vel_min, vel_max = _global_minmax(all_velocity)
    pol_min, pol_max = _global_minmax(all_policy_ms)

    out: List[EpisodeRows] = []
    for ep in episodes:
        scored_rows: List[Dict[str, float]] = []
        for r in ep.rows:
            e = _norm(r["visual_entropy"], ent_min, ent_max)
            v = _norm(r["ee_velocity"], vel_min, vel_max)
            p = _norm(r["policy_step_ms"], pol_min, pol_max)
            k = float(np.clip(r["token_keep_ratio"], 0.0, 1.0))

            # Proxy "MSE-like" hardness signal in [0,1].
            proxy = (
                0.40 * e
                + 0.30 * v
                + 0.15 * p
                + 0.10 * (1.0 - k)
                + 0.05 * float(np.clip(r["skip_ratio"], 0.0, 1.0))
            )
            proxy = float(np.clip(proxy, 0.0, 1.0))

            row = dict(r)
            row["mse_proxy"] = proxy
            scored_rows.append(row)

        out.append(EpisodeRows(episode_id=ep.episode_id, rows=scored_rows))

    return out


def build_manifest(
    episodes: List[EpisodeRows],
    warmup_steps: int,
    fixed_tau: float,
    adaptive_percentile: float,
    rolling_window: int,
    weight_scale: float,
    min_severity: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    manifest_items: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for ep in episodes:
        total = len(ep.rows)
        if total == 0:
            continue

        proxies = np.asarray([r["mse_proxy"] for r in ep.rows], dtype=np.float64)
        taus = np.zeros_like(proxies)

        for i in range(total):
            if i < warmup_steps:
                taus[i] = fixed_tau
            else:
                start = max(0, i - rolling_window + 1)
                window = proxies[start : i + 1]
                taus[i] = float(np.percentile(window, adaptive_percentile))

        spikes = proxies > taus
        for i, row in enumerate(ep.rows):
            if not spikes[i]:
                continue

            tau = float(taus[i])
            proxy = float(proxies[i])
            severity = float(np.clip((proxy - tau) / (1.0 - tau + 1e-8), 0.0, 1.0))
            if severity < min_severity:
                continue

            curriculum_weight = float(1.0 + severity * (weight_scale - 1.0))
            manifest_items.append(
                {
                    "episode_id": ep.episode_id,
                    "step": int(row["step"]),
                    "phase": _phase_name(i, total),
                    "mse_proxy": proxy,
                    "tau_eff": tau,
                    "severity": severity,
                    "curriculum_weight": curriculum_weight,
                    "signals": {
                        "visual_entropy": float(row["visual_entropy"]),
                        "ee_velocity": float(row["ee_velocity"]),
                        "policy_step_ms": float(row["policy_step_ms"]),
                        "skip_ratio": float(row["skip_ratio"]),
                        "token_keep_ratio": float(row["token_keep_ratio"]),
                    },
                }
            )

        hard_count = int(np.sum(spikes))
        summary_rows.append(
            {
                "episode_id": ep.episode_id,
                "total_steps": total,
                "hard_steps": hard_count,
                "hard_ratio": float(hard_count / max(total, 1)),
                "mse_proxy_mean": float(np.mean(proxies)),
                "mse_proxy_p95": float(np.percentile(proxies, 95)),
                "mse_proxy_p99": float(np.percentile(proxies, 99)),
                "tau_mean": float(np.mean(taus)),
                "tau_p95": float(np.percentile(taus, 95)),
            }
        )

    manifest_items.sort(key=lambda x: (-float(x["severity"]), int(x["episode_id"]), int(x["step"])))

    payload: Dict[str, object] = {
        "schema_version": "1.0",
        "generator": "profile_hard_samples_entropy_cim.py",
        "strategy": {
            "warmup_steps": warmup_steps,
            "fixed_tau": fixed_tau,
            "adaptive_percentile": adaptive_percentile,
            "rolling_window": rolling_window,
            "weight_scale": weight_scale,
            "min_severity": min_severity,
        },
        "items": manifest_items,
    }

    return payload, summary_rows


def write_summary_csv(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "episode_id",
        "total_steps",
        "hard_steps",
        "hard_ratio",
        "mse_proxy_mean",
        "mse_proxy_p95",
        "mse_proxy_p99",
        "tau_mean",
        "tau_p95",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _parse_episode_ids(args.episodes)
    if not episodes:
        raise ValueError("No valid episode IDs were parsed from --episodes")

    loaded: List[EpisodeRows] = []
    missing: List[int] = []
    for ep_id in episodes:
        ep_rows = _load_episode_rows(input_dir, ep_id)
        if ep_rows is None:
            missing.append(ep_id)
        else:
            loaded.append(ep_rows)

    if not loaded:
        raise FileNotFoundError(
            f"No timestep CSV files found in {input_dir} for episodes={episodes}."
        )

    if missing:
        print(f"[warn] Missing CSV for episodes: {missing}")

    scored = build_proxy_scores(loaded)
    manifest, summary_rows = build_manifest(
        episodes=scored,
        warmup_steps=args.warmup_steps,
        fixed_tau=args.fixed_tau,
        adaptive_percentile=args.adaptive_percentile,
        rolling_window=args.rolling_window,
        weight_scale=args.weight_scale,
        min_severity=args.min_severity,
    )

    manifest_path = output_dir / "hard_samples_manifest.json"
    summary_path = output_dir / "hard_samples_summary.csv"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    write_summary_csv(summary_path, summary_rows)

    print("=" * 70)
    print("Hard-sample profiling complete")
    print("=" * 70)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Loaded episodes: {[ep.episode_id for ep in scored]}")
    print(f"Manifest items: {len(manifest['items'])}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
