"""
Evaluate Dynamic SmolVLA from final_model.pt — No Retraining
=============================================================
Loads the saved checkpoint from vlaiap_coral_snapflow
and produces:
  - dynamic_ep40.mp4, dynamic_ep41.mp4, dynamic_ep42.mp4
  - dynamic_comparison_plots.png
  - dynamic_report.json
"""
import os, sys, time, json, types, gc, math, csv, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# =====================================================================
#  PATCH: Fix lerobot 0.4.4 import chain crash on Windows/Python 3.10
# =====================================================================
import lerobot
_pkg = lerobot.__path__[0]
import importlib.util

_robots_mod = types.ModuleType('lerobot.robots')
_robots_mod.__path__ = [os.path.join(_pkg, 'robots')]
_robots_mod.__package__ = 'lerobot.robots'
_spec = importlib.util.spec_from_file_location('lerobot.robots.config', os.path.join(_pkg, 'robots', 'config.py'))
_cfg_mod = importlib.util.module_from_spec(_spec)
sys.modules['lerobot.robots.config'] = _cfg_mod
_spec.loader.exec_module(_cfg_mod)
_robots_mod.RobotConfig = _cfg_mod.RobotConfig
sys.modules['lerobot.robots'] = _robots_mod

_proc_mod = types.ModuleType('lerobot.processor')
_proc_mod.__path__ = [os.path.join(_pkg, 'processor')]
_proc_mod.__package__ = 'lerobot.processor'
_proc_mod.RobotAction = dict
_proc_mod.RobotObservation = dict
_proc_mod.PolicyAction = dict
sys.modules['lerobot.processor'] = _proc_mod

_policies_mod = types.ModuleType('lerobot.policies')
_policies_mod.__path__ = [os.path.join(_pkg, 'policies')]
_policies_mod.__package__ = 'lerobot.policies'
sys.modules['lerobot.policies'] = _policies_mod

# =====================================================================
#  LOAD CONFIG
# =====================================================================
from finetune_config import DATASETS, TRAINING, EVAL

DATASET_KEY = "svla_so100_pickplace"
ds_cfg = DATASETS[DATASET_KEY]

# Single place to configure input checkpoint/output folder/pipeline.
# Switch only these fields for either pipeline without touching logic below.
RUN_CONFIG = {
    # Supported: "dynamic_lora_pruning_snapflow", "vlaiap_coral_snapflow", "lyapunov_tokenbottleneck_snapflow02", "semifinetune_entropy_cim"
    "pipeline": "dynamic_lora_pruning_snapflow",
    # Input checkpoint (set None to use profile default final_model.pt)
    "checkpoint_path": "d:/EyetechCode/results/dynamic_layerskip_snapflow_only/final_model.pt",
    # Optional extra dirs to search for fallback checkpoints.
    # Useful when one result folder contains corrupted/incomplete files.
    "checkpoint_search_dirs": [],
    # Output folder for artifacts of this run
    "output_dir": "d:/EyetechCode/results/dynamic_layerskip_snapflow_only",
}

PIPELINE_PROFILES = {
    "dynamic_lora_pruning_snapflow": {
        "training_key": "dynamic_lora_pruning_snapflow",
        "default_ckpt_dir": "d:/EyetechCode/results/dynamic_lora_pruning_snapflow",
        "router_kind": "simple",
        "pruner_kind": "adp",
        "use_coral": False,
        "pipeline_name": "finetune_dynamic_lora_prunning_snapflow",
        "run_label": "Dynamic SmolVLA",
        "report_label": "SmolVLA + DySL + LoRA-SP + ADP + SnapFlow",
        "plot_title": "Dynamic SmolVLA Evaluation — DySL + LoRA-SP + ADP + SnapFlow",
    },
    "vlaiap_coral_snapflow": {
        "training_key": "vlaiap_coral_snapflow",
        "default_ckpt_dir": "d:/EyetechCode/results/vlaiap_coral_snapflow",
        "router_kind": "hierarchical",
        "pruner_kind": "iap",
        "use_coral": True,
        "pipeline_name": "finetune_dynamicNEW_VLAIAP_CoralExpert_Snapflow",
        "run_label": "VLA-IAP + CORAL SmolVLA",
        "report_label": "SmolVLA + Hierarchical STAR + LoRA-SP + VLA-IAP + CORAL + SnapFlow",
        "plot_title": "Dynamic SmolVLA Evaluation — Hierarchical STAR + LoRA-SP + VLA-IAP + CORAL + SnapFlow",
    },
    "lyapunov_tokenbottleneck_snapflow02": {
        "training_key": "lyapunov_tokenbottleneck_snapflow02",
        "default_ckpt_dir": "d:/EyetechCode/results/lyapunov_tokenbottleneck_snapflow02",
        "router_kind": "lyapunov",
        "pruner_kind": "vtb",
        "use_coral": False,
        "pipeline_name": "finetune_lyapunov_tokenbottleneck_snapflow02",
        "run_label": "Lyapunov+VTB SmolVLA",
        "report_label": "SmolVLA + Lyapunov STAR + VTB + LoRA-SP + SnapFlow2 + CogKD",
        "plot_title": "Dynamic SmolVLA Evaluation — Lyapunov STAR + VTB + LoRA-SP + SnapFlow2",
    },
    "semifinetune_entropy_cim": {
        "training_key": "semifinetune_entropy_cim",
        "default_ckpt_dir": "d:/EyetechCode/results/semifinetune_entropy_cim",
        "router_kind": "hierarchical",
        "pruner_kind": "cim",
        "use_coral": True,
        "pipeline_name": "finetune_semifinetune_entropy_cim",
        "run_label": "Entropy-CIM SmolVLA",
        "report_label": "SmolVLA + EGSF + CIM + LoRA-SP + CORAL + SnapFlow",
        "plot_title": "Dynamic SmolVLA Evaluation — EGSF + CIM + LoRA-SP + CORAL + SnapFlow",
    },
}

pipeline_key = RUN_CONFIG["pipeline"]
if pipeline_key not in PIPELINE_PROFILES:
    raise ValueError(f"Unsupported pipeline '{pipeline_key}'. Choose one of {list(PIPELINE_PROFILES.keys())}")

profile = PIPELINE_PROFILES[pipeline_key]
dyn_cfg = TRAINING[profile["training_key"]]
RUN_LABEL = profile["run_label"]
REPORT_LABEL = profile["report_label"]
PLOT_TITLE = profile["plot_title"]


def build_technique_lines(cfg, profile_cfg):
    lines = []
    if profile_cfg["router_kind"] == "hierarchical":
        lines.append("Hierarchical STAR Routing")
    elif profile_cfg["router_kind"] == "lyapunov":
        lines.append("Lyapunov-Stable STAR Routing")
    else:
        lines.append("DySL STAR Routing")
    lines.append(f"LoRA-SP (r={cfg['lora_max_rank']})")
    if profile_cfg["pruner_kind"] == "cim":
        lines.append("Contextual Interaction Masking (CIM)")
        lines.append("Hybrid Tau Safeguard (lambda_cost on/off)")
    elif profile_cfg["pruner_kind"] == "iap":
        lines.append("VLA-IAP Token Pruning")
    elif profile_cfg["pruner_kind"] == "vtb":
        lines.append("Variational Token Bottleneck")
    else:
        lines.append("ADP Token Pruning")
    if profile_cfg["use_coral"]:
        lines.append("CORAL Expert Routing")
    if profile_cfg["router_kind"] == "lyapunov":
        lines.append("SnapFlow2 Curvature-Aware Distillation")
    else:
        lines.append("SnapFlow 1-NFE Denoising")
    return lines


TECHNIQUE_LINES = build_technique_lines(dyn_cfg, profile)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(RUN_CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if RUN_CONFIG.get("checkpoint_path"):
    CHECKPOINT_PATH = Path(RUN_CONFIG["checkpoint_path"])
else:
    CHECKPOINT_PATH = Path(profile["default_ckpt_dir"]) / "final_model.pt"


def _step_id(path_obj):
    m = re.search(r"_step(\d+)\.pt$", path_obj.name)
    return int(m.group(1)) if m else -1


def _zip_integrity_reason(path_obj):
    try:
        size = path_obj.stat().st_size
        if size < 1024:
            return f"file too small ({size} bytes)"

        with open(path_obj, "rb") as f:
            head = f.read(4)
            if len(head) < 4:
                return "cannot read header"

            # PyTorch new serialization uses ZIP container.
            if head[:2] != b"PK":
                return None

            # ZIP EOCD must appear near file tail; absence usually means truncated write/copy.
            tail_len = min(size, 131072)
            f.seek(-tail_len, os.SEEK_END)
            tail = f.read(tail_len)
            if b"PK\x05\x06" not in tail:
                return "missing ZIP end-of-central-directory (file likely truncated/corrupted)"
        return None
    except Exception as e:
        return f"integrity probe failed: {e}"


def _collect_checkpoint_candidates(primary_path):
    primary_path = Path(primary_path)
    primary_dir = primary_path.parent
    search_dirs = [primary_dir]

    for extra_dir in RUN_CONFIG.get("checkpoint_search_dirs", []):
        try:
            p = Path(extra_dir)
        except Exception:
            continue
        if p not in search_dirs:
            search_dirs.append(p)

    candidates = [primary_path]

    # Prefer explicit final_model.pt in configured dirs.
    for d in search_dirs:
        candidates.append(d / "final_model.pt")

    fallback_names = [
        "phase3_complete.pt",
        "phase2_complete.pt",
        "phase1_complete.pt",
    ]
    for d in search_dirs:
        # Step checkpoints are often safer than "complete" files if save was interrupted.
        for phase in [3, 2, 1]:
            phase_steps = sorted(d.glob(f"phase{phase}_step*.pt"), key=_step_id, reverse=True)
            candidates.extend(phase_steps)
        for name in fallback_names:
            candidates.append(d / name)

    seen = set()
    dedup_candidates = []
    for c in candidates:
        c_str = str(c)
        if c_str not in seen:
            seen.add(c_str)
            dedup_candidates.append(c)
    return dedup_candidates, search_dirs


def preflight_checkpoint_candidates(primary_path):
    candidates, search_dirs = _collect_checkpoint_candidates(primary_path)
    existing = [p for p in candidates if p.exists()]
    valid = []
    issues = []

    for ckpt_path in existing:
        issue = _zip_integrity_reason(ckpt_path)
        if issue is None:
            valid.append(ckpt_path)
        else:
            issues.append(f"{ckpt_path.name}: {issue}")

    return {
        "search_dirs": search_dirs,
        "existing": existing,
        "valid": valid,
        "issues": issues,
    }


def load_checkpoint_with_fallback(primary_path, device):
    dedup_candidates, search_dirs = _collect_checkpoint_candidates(primary_path)

    load_errors = []
    for ckpt_path in dedup_candidates:
        if not ckpt_path.exists():
            continue
        integrity_issue = _zip_integrity_reason(ckpt_path)
        if integrity_issue:
            load_errors.append(f"{ckpt_path.name}: {integrity_issue}")
            continue
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            return ckpt, ckpt_path
        except Exception as e:
            load_errors.append(f"{ckpt_path.name}: {e}")

    detail = "\n  - ".join(load_errors) if load_errors else "No candidate checkpoint file found."
    raise RuntimeError(
        "Failed to load checkpoint from all candidates.\n"
        f"Tried directories: {[str(p) for p in search_dirs]}\n"
        "Hint: if every file reports missing ZIP end-of-central-directory, those .pt files are incomplete/corrupted and need to be recopied or re-saved.\n"
        f"  - {detail}"
    )

print("=" * 70)
print(f"  Evaluate {profile['run_label']} from final_model.pt")
print("=" * 70)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, "total_mem", getattr(props, "total_memory", 0))
    print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM: {total_mem/1024**3:.1f} GB")
else:
    print(f"  Device: {DEVICE} (CPU)")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Pipeline profile: {pipeline_key}")
print()

# =====================================================================
#  CUSTOM MODULES (same as training script — needed for state_dict load)
# =====================================================================

class STARRouter(nn.Module):
    """Hierarchical STAR router (VLA-IAP/CORAL variant)."""

    def __init__(self, hidden_dim, num_spatial_layers=4, num_action_layers=4):
        super().__init__()
        self.num_spatial_layers = num_spatial_layers
        self.num_action_layers = num_action_layers
        self.total_skippable = num_spatial_layers + num_action_layers

        input_dim = hidden_dim + 4
        self.spatial_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_spatial_layers),
        )
        self.action_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_action_layers),
        )
        nn.init.constant_(self.spatial_gate[-1].bias, 2.0)
        nn.init.constant_(self.action_gate[-1].bias, 2.0)

        self.snap_mse_proj = nn.Linear(1, hidden_dim // 4)
        self.feedback_gate = nn.Linear(hidden_dim // 4, self.total_skippable)
        nn.init.zeros_(self.feedback_gate.weight)
        nn.init.zeros_(self.feedback_gate.bias)

    def forward(self, hidden_pooled, e_view, delta_s_norm, accel_norm, action_entropy, tau=1.0, hard=True, snap_mse=None):
        x = torch.cat([hidden_pooled, e_view, delta_s_norm, accel_norm, action_entropy], dim=-1)
        spatial_logits = self.spatial_gate(x)
        action_logits = self.action_gate(x)
        all_logits = torch.cat([spatial_logits, action_logits], dim=-1)

        if snap_mse is not None:
            feedback = F.silu(self.snap_mse_proj(snap_mse))
            all_logits = all_logits + self.feedback_gate(feedback)

        entropy_boost = 1.0 + action_entropy * 0.5
        all_logits = all_logits * entropy_boost

        if hard or not self.training:
            gates = (torch.sigmoid(all_logits) > 0.5).float()
        else:
            logits_2class = torch.stack([torch.zeros_like(all_logits), all_logits], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_2class, tau=tau, hard=False, dim=-1)
            gates = gumbel_out[..., 1]
        gate_loss = gates.mean()
        diversity_loss = torch.tensor(0.0, device=gates.device)
        return gates, gate_loss, diversity_loss


class STARRouterSimple(nn.Module):
    """Original STAR router used by dynamic_lora_pruning_snapflow."""

    def __init__(self, hidden_dim, num_skippable_layers=8):
        super().__init__()
        self.num_skippable_layers = num_skippable_layers
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_skippable_layers),
        )
        nn.init.constant_(self.gate_net[-1].bias, 2.0)

    def forward(self, hidden_pooled, e_view, delta_s_norm, tau=1.0, hard=False):
        x = torch.cat([hidden_pooled, e_view, delta_s_norm], dim=-1)
        logits = self.gate_net(x)
        if hard or not self.training:
            gates = (torch.sigmoid(logits) > 0.5).float()
        else:
            logits_2class = torch.stack([torch.zeros_like(logits), logits], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_2class, tau=tau, hard=False, dim=-1)
            gates = gumbel_out[..., 1]
        return gates, gates.mean()


class LyapunovSTARRouter(nn.Module):
    """Lyapunov STAR router matching finetune_lyapunov_tokenbottleneck_snapflow02."""

    def __init__(self, hidden_dim, num_skippable_layers=8):
        super().__init__()
        self.num_skippable_layers = num_skippable_layers
        self.feature_dim = hidden_dim + 4

        self.gate_net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_skippable_layers),
        )
        self.energy_net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(16, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.feedback_gate = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_skippable_layers),
        )
        nn.init.constant_(self.gate_net[-1].bias, 2.0)

    def _project_state(self, state):
        if state.shape[-1] < 16:
            state_in = F.pad(state, (0, 16 - state.shape[-1]))
        else:
            state_in = state[:, :16]
        return self.state_proj(state_in)


class VariationalTokenBottleneckPruner(nn.Module):
    """VTB pruner matching finetune_lyapunov_tokenbottleneck_snapflow02."""

    def __init__(
        self,
        conservative_ratio=0.8,
        aggressive_ratio=0.3,
        entropy_lock_threshold=1.6,
        temporal_gamma=0.7,
        semantic_weight=0.65,
        structural_weight=0.35,
    ):
        super().__init__()
        self.conservative_ratio = conservative_ratio
        self.aggressive_ratio = aggressive_ratio
        self.entropy_lock_threshold = entropy_lock_threshold
        self.temporal_gamma = temporal_gamma
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight

        self.threshold_adjust = nn.Parameter(torch.tensor(0.0))
        self.beta_ib = nn.Parameter(torch.tensor(0.5))
        self.var_mu = nn.Linear(128, 64)
        self.var_logvar = nn.Linear(128, 64)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._ema_scores = None

    def reset_temporal(self):
        self._ema_scores = None

    def _project_tokens(self, token_embeddings):
        if token_embeddings.shape[-1] == 128:
            return token_embeddings
        return F.adaptive_avg_pool1d(token_embeddings, 128)

    def forward(self, token_embeddings, token_mask, state, lang_tokens):
        B, N, D = token_embeddings.shape
        tok_128 = self._project_tokens(token_embeddings)

        lang = lang_tokens.float()
        if lang.shape[-1] != 128:
            lang = F.adaptive_avg_pool1d(lang.unsqueeze(1), 128).squeeze(1)
        tok_norm = F.normalize(tok_128, dim=-1)
        lang_norm = F.normalize(lang, dim=-1)
        semantic = torch.einsum("bnd,bd->bn", tok_norm, lang_norm)

        diff = token_embeddings[:, 1:] - token_embeddings[:, :-1]
        structural = F.pad(diff.abs().mean(dim=-1), (1, 0), value=0.0)
        raw_score = self.semantic_weight * semantic + self.structural_weight * structural

        if self._ema_scores is not None and self._ema_scores.shape == raw_score.shape:
            scores = self.temporal_gamma * self._ema_scores + (1.0 - self.temporal_gamma) * raw_score
        else:
            scores = raw_score
        self._ema_scores = scores.detach()

        probs = torch.softmax(semantic, dim=-1)
        interaction_entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        eff_tau = self.entropy_lock_threshold + torch.tanh(self.threshold_adjust) * 0.2
        aggressive = (interaction_entropy < eff_tau).float()
        keep_ratio = self.conservative_ratio * (1.0 - aggressive) + self.aggressive_ratio * aggressive

        if state.shape[-1] < 16:
            state_in = F.pad(state, (0, 16 - state.shape[-1]))
        else:
            state_in = state[:, :16]
        uncertainty = self.uncertainty_head(state_in)
        keep_ratio = torch.clamp(keep_ratio + uncertainty * 0.25, min=self.aggressive_ratio, max=1.0)

        K_per_sample = (keep_ratio * N).long().clamp(min=1, max=N)
        K = max(int(K_per_sample.max().item()), 1)
        _, top_indices = scores.topk(K, dim=-1, sorted=False)
        top_indices_sorted, _ = top_indices.sort(dim=-1)

        pruned_embeddings = torch.gather(
            token_embeddings,
            1,
            top_indices_sorted.unsqueeze(-1).expand(-1, -1, D),
        )
        pruned_mask = torch.gather(token_mask, 1, top_indices_sorted)

        pooled = tok_128.mean(dim=1)
        mu = self.var_mu(pooled)
        logvar = self.var_logvar(pooled)
        kl = 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
        ib_loss = (keep_ratio.mean() - self.aggressive_ratio).abs() + torch.sigmoid(self.beta_ib) * kl

        info = {
            'interaction_entropy': interaction_entropy.mean().item(),
            'aggressive_ratio': aggressive.mean().item(),
            'uncertainty': uncertainty.mean().item(),
            'ib_loss': ib_loss.item(),
        }

        return pruned_embeddings, pruned_mask, keep_ratio.mean(), keep_ratio, info


class ActionAwareTokenPruner(nn.Module):
    """VLA-IAP token pruner (state/interation aligned)."""

    def __init__(self, conservative_ratio=0.8, aggressive_ratio=0.3,
                 iou_threshold=0.5, temporal_momentum=0.7,
                 geometric_anchor_ratio=0.1,
                 entropy_weight=0.4, iou_weight=0.35, grad_weight=0.25,
                 interaction_entropy_tau=1.6, iou_proxy_tau=0.5):
        super().__init__()
        self.conservative_ratio = conservative_ratio
        self.aggressive_ratio = aggressive_ratio
        self.iou_threshold = iou_threshold
        self.temporal_momentum = temporal_momentum
        self.geometric_anchor_ratio = geometric_anchor_ratio
        self.entropy_weight = entropy_weight
        self.iou_weight = iou_weight
        self.grad_weight = grad_weight
        self.interaction_entropy_tau = interaction_entropy_tau
        self.iou_proxy_tau = iou_proxy_tau

        self.threshold_adjust = nn.Parameter(torch.tensor(0.0))
        self.iou_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.anchor_scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.action_classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._prev_importance = None

    def reset_temporal(self):
        self._prev_importance = None

    def forward(self, token_embeddings, token_mask, state, attention_scores=None, v_ee=None):
        B, N, D = token_embeddings.shape
        token_pool = token_embeddings.mean(dim=1)
        if token_pool.shape[-1] != 128:
            token_pool = F.adaptive_avg_pool1d(token_pool.unsqueeze(1), 128).squeeze(1)
        iou_proxy = self.iou_estimator(token_pool)

        effective_threshold = self.iou_threshold + torch.tanh(self.threshold_adjust) * 0.1
        is_aggressive = (iou_proxy > effective_threshold).float()
        keep_ratio = (1.0 - is_aggressive) * self.conservative_ratio + is_aggressive * self.aggressive_ratio

        visual_entropy = token_embeddings.var(dim=1).mean(dim=-1, keepdim=True)
        entropy_norm = torch.sigmoid(visual_entropy)
        if v_ee is None:
            grad_proxy = state.abs().mean(dim=-1, keepdim=True)
        else:
            grad_proxy = v_ee
        grad_norm = torch.tanh(grad_proxy)
        complexity_score = (
            self.entropy_weight * entropy_norm
            + self.iou_weight * iou_proxy
            + self.grad_weight * grad_norm
        ).clamp(0.0, 1.0)
        cim_keep_ratio = self.aggressive_ratio + (self.conservative_ratio - self.aggressive_ratio) * complexity_score
        keep_ratio = torch.maximum(keep_ratio, cim_keep_ratio)

        interaction_lock = (
            (visual_entropy > self.interaction_entropy_tau).float()
            * (iou_proxy > self.iou_proxy_tau).float()
        )
        keep_ratio = torch.where(
            interaction_lock > 0.5,
            torch.maximum(keep_ratio, torch.ones_like(keep_ratio) * 0.90),
            keep_ratio,
        )

        if state.shape[-1] > 16:
            state_trunc = state[:, :16]
        else:
            state_trunc = F.pad(state, (0, 16 - state.shape[-1]))
        action_fine = self.action_classifier(state_trunc)
        keep_ratio = torch.where(action_fine > 0.7, torch.ones_like(keep_ratio) * 0.95, keep_ratio)

        if attention_scores is None:
            positions = torch.arange(N, device=token_embeddings.device).float()
            center = N / 2.0
            attention_scores = torch.exp(-((positions - center) ** 2) / (2 * (N / 4.0) ** 2))
            attention_scores = attention_scores.unsqueeze(0).expand(B, -1)

        if D != 128:
            tokens_proj = F.adaptive_avg_pool1d(token_embeddings, 128)
        else:
            tokens_proj = token_embeddings
        anchor_scores = self.anchor_scorer(tokens_proj).squeeze(-1)
        num_anchors = max(int(N * self.geometric_anchor_ratio), 1)
        _, anchor_indices = anchor_scores.topk(num_anchors, dim=-1)
        anchor_boost = torch.zeros_like(attention_scores)
        anchor_boost.scatter_(1, anchor_indices, 1.0)
        importance = attention_scores + anchor_boost

        if self._prev_importance is not None and self._prev_importance.shape == importance.shape:
            importance = self.temporal_momentum * self._prev_importance + (1.0 - self.temporal_momentum) * importance
        self._prev_importance = importance.detach().clone()

        K_per_sample = (keep_ratio * N).long().clamp(min=1, max=N)
        K = max(K_per_sample.max().item(), 1)
        _, top_indices = importance.topk(K, dim=-1, sorted=False)
        top_indices_sorted, _ = top_indices.sort(dim=-1)

        pruned_embeddings = torch.gather(
            token_embeddings, 1,
            top_indices_sorted.unsqueeze(-1).expand(-1, -1, D)
        )
        pruned_mask = torch.gather(token_mask, 1, top_indices_sorted)
        avg_keep_ratio = keep_ratio.mean()
        iap_info = {
            'iou_proxy': iou_proxy.mean().item(),
            'is_aggressive_ratio': is_aggressive.mean().item(),
            'action_fine_prob': action_fine.mean().item(),
            'keep_ratio': keep_ratio.mean().item(),
            'complexity_score': complexity_score.mean().item(),
            'interaction_lock_ratio': interaction_lock.mean().item(),
            'visual_entropy': visual_entropy.mean().item(),
        }
        return pruned_embeddings, pruned_mask, avg_keep_ratio, keep_ratio, iap_info


class ADPTokenPruner(nn.Module):
    """Original ADP token pruner used by dynamic_lora_pruning_snapflow."""

    def __init__(self, v_threshold=0.15, min_keep_ratio=0.3):
        super().__init__()
        self.v_threshold = v_threshold
        self.min_keep_ratio = min_keep_ratio
        self.threshold_adjust = nn.Parameter(torch.tensor(0.0))

    def forward(self, token_embeddings, token_mask, v_ee, attention_scores=None):
        B, N, D = token_embeddings.shape
        effective_threshold = self.v_threshold + torch.tanh(self.threshold_adjust) * 0.05
        keep_ratio_per_sample = torch.where(
            v_ee > effective_threshold,
            torch.clamp(
                self.min_keep_ratio + (1.0 - self.min_keep_ratio) * (effective_threshold / (v_ee + 1e-8)),
                min=self.min_keep_ratio,
                max=1.0,
            ),
            torch.ones_like(v_ee),
        )
        K = max(int(N * self.min_keep_ratio), 1)
        if attention_scores is None:
            attention_scores = torch.arange(N, 0, -1, device=token_embeddings.device).float()
            attention_scores = attention_scores.unsqueeze(0).expand(B, -1)
        _, top_indices = attention_scores.topk(K, dim=-1, sorted=False)
        top_indices_sorted, _ = top_indices.sort(dim=-1)
        pruned_embeddings = torch.gather(
            token_embeddings, 1,
            top_indices_sorted.unsqueeze(-1).expand(-1, -1, D)
        )
        pruned_mask = torch.gather(token_mask, 1, top_indices_sorted)
        avg_keep_ratio = keep_ratio_per_sample.mean()
        return pruned_embeddings, pruned_mask, avg_keep_ratio, keep_ratio_per_sample


class LoRASPAdapter(nn.Module):
    """LoRA-SP: LoRA with Select-Prune for dynamic rank adaptation."""

    def __init__(self, in_features, out_features, max_rank=128,
                 energy_threshold=0.9, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.energy_threshold = energy_threshold
        self.alpha = alpha
        self.U = nn.Parameter(torch.randn(out_features, max_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)
        self.router = nn.Sequential(
            nn.Linear(in_features, max_rank),
            nn.Sigmoid(),
        )
        self.scaling = alpha / max_rank

    def forward(self, x, return_spec_loss=False):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        scores = self.router(x_flat.detach())
        Vx = F.linear(x_flat, self.V)
        SVx = Vx * scores * self.scaling
        delta = F.linear(SVx, self.U)
        delta = delta.reshape(*orig_shape[:-1], self.out_features)
        if return_spec_loss:
            scores_sq = scores ** 2
            total_energy = scores_sq.sum(dim=-1, keepdim=True) + 1e-8
            prob = scores_sq / total_energy
            spec_loss = -(prob * (prob + 1e-8).log()).sum(dim=-1).mean()
            return delta, spec_loss
        return delta


class CORALExpertManager(nn.Module):
    """CORAL language-routed expert manager."""

    def __init__(self, hidden_dim, num_experts=4, expert_rank=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_rank = expert_rank
        self.expert_A = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, expert_rank) * 0.01)
            for _ in range(num_experts)
        ])
        self.expert_B = nn.ParameterList([
            nn.Parameter(torch.randn(expert_rank, hidden_dim) * 0.01)
            for _ in range(num_experts)
        ])
        self.lang_projector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts),
        )
        self.expert_scaling = 1.0 / expert_rank
        self._active_expert_id = 0

    def route_from_language(self, lang_embeds):
        lang_pooled = lang_embeds.mean(dim=1)
        if lang_pooled.shape[-1] != self.hidden_dim:
            lang_pooled = F.adaptive_avg_pool1d(lang_pooled.unsqueeze(1), self.hidden_dim).squeeze(1)
        # Align input dtype/device with routing MLP to avoid bf16/float32 matmul mismatch.
        proj_param = next(self.lang_projector.parameters())
        lang_pooled = lang_pooled.to(device=proj_param.device, dtype=proj_param.dtype)
        routing_logits = self.lang_projector(lang_pooled)
        routing_probs = F.softmax(routing_logits, dim=-1)
        expert_id = routing_probs.argmax(dim=-1).mode().values.item()
        self._active_expert_id = expert_id
        return expert_id, routing_probs


# =====================================================================
#  PHASE 1: Load Dataset
# =====================================================================
print("=" * 70)
print("  PHASE 1: Load Dataset")
print("=" * 70)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(ds_cfg["repo_id"])

sample = dataset[0]
all_keys = list(sample.keys())
image_keys = [k for k in all_keys if 'image' in k.lower() and isinstance(sample[k], torch.Tensor)]
state_key = next((k for k in all_keys if 'state' in k.lower() and isinstance(sample[k], torch.Tensor)), None)
action_key = 'action'
meta_keys = {'episode_index', 'frame_index', 'timestamp', 'index', 'task_index'}
string_keys = [k for k in all_keys if isinstance(sample[k], str)]
ACTION_DIM = sample[action_key].shape[-1]
STATE_DIM = sample[state_key].shape[-1] if state_key else 0
IMG_C, IMG_H, IMG_W = sample[image_keys[0]].shape if image_keys else (3, 256, 256)

print(f"  Episodes: {dataset.num_episodes}, Frames: {len(dataset)}, FPS: {dataset.fps}")
print(f"  Images: {image_keys}, State: {state_key} ({STATE_DIM}D), Action: {ACTION_DIM}D")

# Fast episode indexing
ep_col = dataset.hf_dataset['episode_index']
episode_indices = {}
for idx, ep in enumerate(ep_col):
    ep_int = ep.item() if isinstance(ep, torch.Tensor) else int(ep)
    if ep_int not in episode_indices: episode_indices[ep_int] = []
    episode_indices[ep_int].append(idx)

all_eps = sorted(episode_indices.keys())
n_train = ds_cfg["train_episodes"]
n_eval = ds_cfg["eval_episodes"]
train_eps = all_eps[:n_train]
eval_eps = all_eps[n_train:n_train + n_eval]

print(f"  Train: {len(train_eps)} eps")
print(f"  Eval: {len(eval_eps)} eps ({eval_eps})")

# Build a robust eval episode list for any dataset split.
requested_eps = EVAL.get("eval_episodes") if isinstance(EVAL, dict) else None
num_eval_target = int(EVAL.get("num_eval_episodes", 3)) if isinstance(EVAL, dict) else 3
if requested_eps:
    eval_ep_list = [int(ep) for ep in requested_eps if int(ep) in episode_indices]
else:
    eval_ep_list = list(eval_eps[:num_eval_target])

# Fallback if dataset split produced no eval episodes.
if not eval_ep_list:
    eval_ep_list = list(all_eps[-max(1, num_eval_target):])
    print(f"  WARNING: eval split is empty. Fallback to last episodes: {eval_ep_list}")

# Fail fast on checkpoint integrity before loading HF base model.
ckpt_preflight = preflight_checkpoint_candidates(CHECKPOINT_PATH)
if not ckpt_preflight["valid"]:
    issue_text = "\n  - ".join(ckpt_preflight["issues"]) if ckpt_preflight["issues"] else "No checkpoint file exists in configured search dirs."
    raise RuntimeError(
        "Checkpoint preflight failed: no loadable checkpoint candidate found.\n"
        f"Pipeline: {pipeline_key}\n"
        f"Configured checkpoint: {CHECKPOINT_PATH}\n"
        f"Search dirs: {[str(p) for p in ckpt_preflight['search_dirs']]}\n"
        "Set RUN_CONFIG['checkpoint_path'] to a valid .pt or add backup dirs in RUN_CONFIG['checkpoint_search_dirs'].\n"
        f"  - {issue_text}"
    )

print(f"  Preflight checkpoint candidates OK: {len(ckpt_preflight['valid'])} valid file(s) detected.")

# =====================================================================
#  PHASE 2: Load SmolVLA + Restore Checkpoint
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 2: Load SmolVLA + Restore final_model.pt")
print("=" * 70)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("  Loading pretrained SmolVLA base...")
smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# Key remapping
smolvla_img_keys = list(smolvla.config.image_features.keys())
KEY_REMAP_S = {}
for i, dk in enumerate(image_keys):
    if i < len(smolvla_img_keys):
        KEY_REMAP_S[dk] = smolvla_img_keys[i]
print(f"  Key remap: {KEY_REMAP_S}")

# Language tokens
tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
instruction = ds_cfg["task_instruction"]
_tok = tokenizer(instruction, return_tensors="pt", padding="max_length", max_length=64)
LANG_IDS = _tok['input_ids']
LANG_MASK = _tok['attention_mask'].bool()
print(f"  Instruction: '{instruction}'")

CHUNK_SIZE_S = smolvla.config.chunk_size

smolvla.to(DEVICE)

# Build dynamic modules (needed to load state_dict)
flow_model = smolvla.model
vlm_hidden = flow_model.vlm_with_expert.config.text_config.hidden_size
num_vlm_layers = flow_model.vlm_with_expert.num_vlm_layers

if profile["router_kind"] == "hierarchical":
    star_router = STARRouter(
        hidden_dim=vlm_hidden,
        num_spatial_layers=dyn_cfg["num_spatial_layers"],
        num_action_layers=dyn_cfg["num_action_layers"],
    ).to(DEVICE)
elif profile["router_kind"] == "lyapunov":
    star_router = LyapunovSTARRouter(
        hidden_dim=vlm_hidden,
        num_skippable_layers=dyn_cfg["num_skippable_layers"],
    ).to(DEVICE)
else:
    star_router = STARRouterSimple(
        hidden_dim=vlm_hidden,
        num_skippable_layers=dyn_cfg["num_skippable_layers"],
    ).to(DEVICE)

if profile["pruner_kind"] == "cim":
    token_pruner = ActionAwareTokenPruner(
        conservative_ratio=dyn_cfg.get("cim_conservative_ratio", dyn_cfg.get("iap_conservative_ratio", 0.8)),
        aggressive_ratio=dyn_cfg.get("cim_aggressive_ratio", dyn_cfg.get("iap_aggressive_ratio", 0.3)),
        iou_threshold=dyn_cfg.get("cim_iou_proxy_tau", dyn_cfg.get("iap_iou_threshold", 0.5)),
        temporal_momentum=dyn_cfg.get("cim_temporal_gamma", dyn_cfg.get("iap_temporal_momentum", 0.7)),
        geometric_anchor_ratio=dyn_cfg.get("iap_geometric_anchor_ratio", 0.1),
        entropy_weight=dyn_cfg.get("complexity_alpha_entropy", 0.4),
        iou_weight=dyn_cfg.get("complexity_alpha_iou_proxy", 0.35),
        grad_weight=dyn_cfg.get("complexity_alpha_grad_proxy", 0.25),
        interaction_entropy_tau=dyn_cfg.get("cim_interaction_entropy_tau", 1.6),
        iou_proxy_tau=dyn_cfg.get("cim_iou_proxy_tau", 0.5),
    ).to(DEVICE)
elif profile["pruner_kind"] == "iap":
    token_pruner = ActionAwareTokenPruner(
        conservative_ratio=dyn_cfg["iap_conservative_ratio"],
        aggressive_ratio=dyn_cfg["iap_aggressive_ratio"],
        iou_threshold=dyn_cfg["iap_iou_threshold"],
        temporal_momentum=dyn_cfg["iap_temporal_momentum"],
        geometric_anchor_ratio=dyn_cfg["iap_geometric_anchor_ratio"],
    ).to(DEVICE)
elif profile["pruner_kind"] == "vtb":
    token_pruner = VariationalTokenBottleneckPruner(
        conservative_ratio=dyn_cfg["vtb_conservative_ratio"],
        aggressive_ratio=dyn_cfg["vtb_aggressive_ratio"],
        entropy_lock_threshold=dyn_cfg["interaction_lock_entropy_tau"],
        temporal_gamma=dyn_cfg["vtb_temporal_gamma"],
        semantic_weight=dyn_cfg["vtb_semantic_weight"],
        structural_weight=dyn_cfg["vtb_structural_weight"],
    ).to(DEVICE)
else:
    adp_threshold = dyn_cfg.get("adp_v_threshold", dyn_cfg.get("adp_velocity_threshold", 0.15))
    token_pruner = ADPTokenPruner(
        v_threshold=adp_threshold,
        min_keep_ratio=dyn_cfg["adp_min_keep_ratio"],
    ).to(DEVICE)

coral_manager = None
if profile["use_coral"]:
    coral_manager = CORALExpertManager(
        hidden_dim=vlm_hidden,
        num_experts=dyn_cfg["coral_num_experts"],
        expert_rank=dyn_cfg["coral_expert_rank"],
    ).to(DEVICE)

lora_adapters = nn.ModuleDict()
vlm_layers = flow_model.vlm_with_expert.get_vlm_model().text_model.layers
for layer_idx in range(dyn_cfg["num_fixed_layers"], num_vlm_layers):
    layer = vlm_layers[layer_idx]
    layer_key = f"layer_{layer_idx}"
    attn = layer.self_attn
    q_in, q_out = attn.q_proj.in_features, attn.q_proj.out_features
    k_in, k_out = attn.k_proj.in_features, attn.k_proj.out_features
    v_in, v_out = attn.v_proj.in_features, attn.v_proj.out_features
    o_in, o_out = attn.o_proj.in_features, attn.o_proj.out_features
    lora_adapters[f"{layer_key}_q"] = LoRASPAdapter(q_in, q_out, max_rank=dyn_cfg["lora_max_rank"], energy_threshold=dyn_cfg["lora_energy_threshold"])
    lora_adapters[f"{layer_key}_k"] = LoRASPAdapter(k_in, k_out, max_rank=dyn_cfg["lora_max_rank"], energy_threshold=dyn_cfg["lora_energy_threshold"])
    lora_adapters[f"{layer_key}_v"] = LoRASPAdapter(v_in, v_out, max_rank=dyn_cfg["lora_max_rank"], energy_threshold=dyn_cfg["lora_energy_threshold"])
    lora_adapters[f"{layer_key}_o"] = LoRASPAdapter(o_in, o_out, max_rank=dyn_cfg["lora_max_rank"], energy_threshold=dyn_cfg["lora_energy_threshold"])
lora_adapters = lora_adapters.to(DEVICE)

# ── Load checkpoint ──
print(f"\n  Loading checkpoint: {CHECKPOINT_PATH}")
ckpt, loaded_ckpt_path = load_checkpoint_with_fallback(CHECKPOINT_PATH, DEVICE)
if loaded_ckpt_path != CHECKPOINT_PATH:
    print(f"  WARNING: primary checkpoint failed. Fallback loaded: {loaded_ckpt_path}")
CHECKPOINT_PATH = loaded_ckpt_path

# Restore SmolVLA weights
smolvla_state = ckpt.get('smolvla', None)
if smolvla_state is not None:
    smolvla.load_state_dict(smolvla_state)
    print("  ✓ SmolVLA weights restored")
else:
    print("  ✗ No 'smolvla' key in checkpoint!")

# Restore dynamic modules
if 'star_router' in ckpt:
    star_router.load_state_dict(ckpt['star_router'])
    print("  ✓ STAR Router weights restored")
if 'token_pruner' in ckpt:
    token_pruner.load_state_dict(ckpt['token_pruner'])
    print("  ✓ Token Pruner weights restored")
if 'lora_adapters' in ckpt:
    lora_adapters.load_state_dict(ckpt['lora_adapters'])
    print("  ✓ LoRA-SP Adapters weights restored")
if coral_manager is not None and 'coral_manager' in ckpt:
    coral_manager.load_state_dict(ckpt['coral_manager'])
    print("  ✓ CORAL Manager weights restored")

del ckpt
gc.collect()
torch.cuda.empty_cache()

smolvla_total = sum(p.numel() for p in smolvla.parameters())
wrapper_params = sum(p.numel() for p in star_router.parameters())
wrapper_params += sum(p.numel() for p in token_pruner.parameters())
wrapper_params += sum(p.numel() for p in lora_adapters.parameters())
if coral_manager is not None:
    wrapper_params += sum(p.numel() for p in coral_manager.parameters())
print(f"\n  SmolVLA params: {smolvla_total/1e6:.1f}M")
print(f"  Dynamic params: {wrapper_params/1e6:.2f}M")
MODEL_ACTION_DIM = int(smolvla.config.action_feature.shape[0]) if smolvla.config.action_feature is not None else ACTION_DIM
print(f"  Model action dim: {MODEL_ACTION_DIM}, Dataset action dim: {ACTION_DIM}")

# =====================================================================
#  PHASE 3: Evaluate
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 3: Evaluate Fine-Tuned Dynamic Model")
print("=" * 70)

smolvla.eval()
star_router.eval()
token_pruner.eval()
lora_adapters.eval()
if coral_manager is not None:
    coral_manager.eval()


def build_eval_batch(sample_dict, device):
    batch = {}
    for k in sample_dict:
        if k in meta_keys or k in string_keys:
            continue
        v = sample_dict[k]
        if not isinstance(v, torch.Tensor):
            continue
        out_key = KEY_REMAP_S.get(k, k)
        batch[out_key] = v.unsqueeze(0).to(device)
    batch['observation.language.tokens'] = LANG_IDS.to(device)
    batch['observation.language.attention_mask'] = LANG_MASK.to(device)
    return batch


def infer_dynamic_signals(batch, prev_state):
    num_skip = dyn_cfg["num_skippable_layers"]
    num_fixed = dyn_cfg["num_fixed_layers"]
    default_gate = np.ones(num_skip, dtype=np.float32)
    default_probs = np.ones(num_skip, dtype=np.float32)

    state = batch.get('observation.state')
    if state is None:
        state = torch.zeros(1, 6, device=DEVICE)
    else:
        state = state.float()

    if prev_state is None:
        delta_s = torch.zeros(1, 1, device=DEVICE)
        ee_velocity = torch.zeros(1, 1, device=DEVICE)
        accel_norm = torch.zeros(1, 1, device=DEVICE)
    else:
        delta_s = (state - prev_state).norm(dim=-1, keepdim=True)
        ee_dims = min(state.shape[-1], prev_state.shape[-1], 6)
        ee_velocity = (state[:, :ee_dims] - prev_state[:, :ee_dims]).norm(dim=-1, keepdim=True)
        accel_norm = delta_s

    if state.shape[-1] > 0:
        action_abs = state.abs()
        action_prob = action_abs / (action_abs.sum(dim=-1, keepdim=True) + 1e-8)
        action_entropy = -(action_prob * (action_prob + 1e-8).log()).sum(dim=-1, keepdim=True)
    else:
        action_entropy = torch.zeros(1, 1, device=DEVICE)

    with torch.no_grad():
        images, _ = smolvla.prepare_images(batch)
        img_embs = [flow_model.vlm_with_expert.embed_image(img).float() for img in images]
        all_img_emb = torch.cat(img_embs, dim=1) if img_embs else None

    if all_img_emb is None:
        return {
            'skip_ratio': 0.0,
            'active_layers': int(num_fixed + num_skip),
            'token_keep_ratio': 1.0,
            'visual_entropy': 0.0,
            'ee_velocity': float(ee_velocity.item()),
            'lora_effective_rank': float(dyn_cfg['lora_max_rank']),
            'token_count_original': 0,
            'token_count_pruned': 0,
            'iou_proxy': 0.0,
            'action_fine_prob': 0.0,
            'complexity_score': 0.0,
            'interaction_lock_ratio': 0.0,
            'coral_expert_id': 0,
            'gate_binary': default_gate,
            'gate_probs': default_probs,
        }, state.detach().clone()

    visual_entropy = all_img_emb.var(dim=1).mean(dim=-1, keepdim=True)
    hidden_pooled = all_img_emb.mean(dim=1)
    if profile["router_kind"] == "hierarchical":
        expected_dim = star_router.spatial_gate[0].in_features - 4
    elif profile["router_kind"] == "lyapunov":
        expected_dim = star_router.feature_dim - 4
    else:
        expected_dim = star_router.gate_net[0].in_features - 2
    if hidden_pooled.shape[-1] != expected_dim:
        hidden_pooled = F.adaptive_avg_pool1d(hidden_pooled.unsqueeze(1), expected_dim).squeeze(1)

    with torch.no_grad():
        action_entropy_router = action_entropy
        if profile["pruner_kind"] == "cim" and hasattr(token_pruner, "iou_estimator"):
            token_pool = all_img_emb.mean(dim=1)
            if token_pool.shape[-1] != 128:
                token_pool = F.adaptive_avg_pool1d(token_pool.unsqueeze(1), 128).squeeze(1)
            cim_iou_hint = token_pruner.iou_estimator(token_pool)
            entropy_norm = torch.sigmoid(visual_entropy)
            grad_norm = torch.tanh(delta_s)
            action_entropy_router = (
                dyn_cfg.get("complexity_alpha_entropy", 0.4) * entropy_norm
                + dyn_cfg.get("complexity_alpha_iou_proxy", 0.35) * cim_iou_hint
                + dyn_cfg.get("complexity_alpha_grad_proxy", 0.25) * grad_norm
            ).clamp(0.0, 1.0)

        if profile["router_kind"] == "hierarchical":
            router_in = torch.cat(
                [hidden_pooled.float(), visual_entropy.float(), delta_s.float(), accel_norm.float(), action_entropy_router.float()],
                dim=-1,
            )
            spatial_logits = star_router.spatial_gate(router_in)
            action_logits = star_router.action_gate(router_in)
            logits = torch.cat([spatial_logits, action_logits], dim=-1)
            entropy_boost = 1.0 + action_entropy_router * 0.5
            logits = logits * entropy_boost
        elif profile["router_kind"] == "lyapunov":
            state_scalar = star_router._project_state(state.float())
            router_in = torch.cat(
                [hidden_pooled.float(), visual_entropy.float(), delta_s.float(), accel_norm.float(), state_scalar.float()],
                dim=-1,
            )
            logits = star_router.gate_net(router_in)
        else:
            router_in = torch.cat([hidden_pooled.float(), visual_entropy.float(), delta_s.float()], dim=-1)
            logits = star_router.gate_net(router_in)

        gate_probs = torch.sigmoid(logits)
        gate_binary = (gate_probs > 0.5).float()

        token_mask = torch.ones(1, all_img_emb.shape[1], dtype=torch.bool, device=DEVICE)
        if profile["pruner_kind"] in {"iap", "cim"}:
            pruned_emb, _, keep_ratio, _, iap_info = token_pruner(
                all_img_emb,
                token_mask,
                state.float(),
                attention_scores=None,
                v_ee=ee_velocity.float(),
            )
        elif profile["pruner_kind"] == "vtb":
            pruned_emb, _, keep_ratio, _, vtb_info = token_pruner(
                all_img_emb,
                token_mask,
                state.float(),
                batch['observation.language.tokens'].float(),
            )
            iap_info = {
                'iou_proxy': vtb_info.get('interaction_entropy', 0.0),
                'action_fine_prob': vtb_info.get('aggressive_ratio', 0.0),
            }
        else:
            pruned_emb, _, keep_ratio, _ = token_pruner(
                all_img_emb,
                token_mask,
                ee_velocity.float(),
                attention_scores=None,
            )
            iap_info = {}

        if coral_manager is not None and hasattr(flow_model.vlm_with_expert.get_vlm_model().text_model, 'embed_tokens'):
            lang_embs = flow_model.vlm_with_expert.get_vlm_model().text_model.embed_tokens(batch['observation.language.tokens'])
            expert_id, _ = coral_manager.route_from_language(lang_embs)
        else:
            expert_id = 0

        adapter = next(iter(lora_adapters.values()), None)
        if adapter is None:
            eff_rank = float(dyn_cfg['lora_max_rank'])
        else:
            lora_input = all_img_emb.mean(dim=1, keepdim=True)
            if lora_input.shape[-1] != adapter.in_features:
                lora_input = F.adaptive_avg_pool1d(
                    lora_input.permute(0, 2, 1), adapter.in_features
                ).permute(0, 2, 1)
            x_flat = lora_input.reshape(-1, adapter.in_features)
            scores = adapter.router(x_flat)
            scores_sorted = scores.sort(dim=-1, descending=True).values
            cum_energy = (scores_sorted ** 2).cumsum(dim=-1)
            total_energy = (scores ** 2).sum(dim=-1, keepdim=True) + 1e-8
            energy_ratio = cum_energy / total_energy
            eff_rank = float((energy_ratio < dyn_cfg['lora_energy_threshold']).sum().item() + 1)

    gate_binary_np = gate_binary.detach().cpu().numpy()[0].astype(np.float32)
    gate_probs_np = gate_probs.detach().cpu().numpy()[0].astype(np.float32)

    return {
        'skip_ratio': float(1.0 - gate_binary_np.mean()),
        'active_layers': int(gate_binary_np.sum() + num_fixed),
        'token_keep_ratio': float(keep_ratio.item()),
        'visual_entropy': float(visual_entropy.item()),
        'ee_velocity': float(ee_velocity.item()),
        'lora_effective_rank': eff_rank,
        'token_count_original': int(all_img_emb.shape[1]),
        'token_count_pruned': int(pruned_emb.shape[1]),
        'iou_proxy': float(iap_info.get('iou_proxy', 0.0)),
        'action_fine_prob': float(iap_info.get('action_fine_prob', 0.0)),
        'complexity_score': float(iap_info.get('complexity_score', 0.0)),
        'interaction_lock_ratio': float(iap_info.get('interaction_lock_ratio', 0.0)),
        'coral_expert_id': int(expert_id),
        'gate_binary': gate_binary_np,
        'gate_probs': gate_probs_np,
    }, state.detach().clone()


def evaluate_dynamic_model(model, eval_episodes, fps_value):
    results = {
        'mse_per_episode': [],
        'latency_ms': [],
        'preprocess_ms': [],
        'postprocess_ms': [],
        'policy_step_ms': [],
        'analysis_ms': [],
        'predictions': {}, 'ground_truth': {},
        'timesteps': {},
        'episode_stats': {},
    }
    eval_action_dim = None
    fps_eff = float(fps_value if fps_value else 15.0)
    control_period_s = 1.0 / max(fps_eff, 1e-6)

    for ep_idx in eval_episodes:
        ep_preds, ep_gts = [], []
        indices = episode_indices[ep_idx]
        model.reset()
        if hasattr(token_pruner, 'reset_temporal'):
            token_pruner.reset_temporal()
        prev_state = None

        ep_latency_ms = []
        ep_pre_ms = []
        ep_post_ms = []
        ep_policy_ms = []
        ep_analysis_ms = []
        ep_skip_ratio = []
        ep_active_layers = []
        ep_token_keep = []
        ep_visual_entropy = []
        ep_ee_velocity = []
        ep_lora_rank = []
        ep_token_count_original = []
        ep_token_count_pruned = []
        ep_iou_proxy = []
        ep_action_fine_prob = []
        ep_complexity_score = []
        ep_interaction_lock_ratio = []
        ep_coral_expert_id = []
        ep_gate_binary = []
        ep_gate_probs = []

        ep_t0 = time.perf_counter()

        for step_idx in tqdm(indices, desc=f"  Ep{ep_idx}", ncols=85, leave=False):
            s = dataset[step_idx]
            gt_full = s[action_key].numpy()

            t_pre0 = time.perf_counter()
            batch = build_eval_batch(s, DEVICE)
            t_pre1 = time.perf_counter()

            t_model0 = time.perf_counter()
            with torch.no_grad():
                pred = model.select_action(batch)
            t_model1 = time.perf_counter()

            t_post0 = time.perf_counter()

            pred_np = np.asarray(pred.squeeze().cpu().numpy())
            if pred_np.ndim > 1:
                pred_np = pred_np[0]
            pred_np = pred_np.reshape(-1)
            t_post1 = time.perf_counter()

            step_dim = min(pred_np.shape[0], gt_full.shape[-1], MODEL_ACTION_DIM)
            if step_dim <= 0:
                continue

            if eval_action_dim is None:
                eval_action_dim = step_dim

            # Keep a fixed action dimension across all samples for stable metrics.
            if step_dim < eval_action_dim:
                continue

            pred_np = pred_np[:eval_action_dim]
            gt = gt_full[:eval_action_dim]

            t_dyn0 = time.perf_counter()
            dyn_step, prev_state = infer_dynamic_signals(batch, prev_state)
            t_dyn1 = time.perf_counter()

            latency_ms = (t_model1 - t_model0) * 1000
            pre_ms = (t_pre1 - t_pre0) * 1000
            post_ms = (t_post1 - t_post0) * 1000
            policy_ms = (t_post1 - t_pre0) * 1000
            analysis_ms = (t_dyn1 - t_dyn0) * 1000

            results['latency_ms'].append(latency_ms)
            results['preprocess_ms'].append(pre_ms)
            results['postprocess_ms'].append(post_ms)
            results['policy_step_ms'].append(policy_ms)
            results['analysis_ms'].append(analysis_ms)

            ep_latency_ms.append(latency_ms)
            ep_pre_ms.append(pre_ms)
            ep_post_ms.append(post_ms)
            ep_policy_ms.append(policy_ms)
            ep_analysis_ms.append(analysis_ms)
            ep_skip_ratio.append(dyn_step['skip_ratio'])
            ep_active_layers.append(dyn_step['active_layers'])
            ep_token_keep.append(dyn_step['token_keep_ratio'])
            ep_visual_entropy.append(dyn_step['visual_entropy'])
            ep_ee_velocity.append(dyn_step['ee_velocity'])
            ep_lora_rank.append(dyn_step['lora_effective_rank'])
            ep_token_count_original.append(dyn_step['token_count_original'])
            ep_token_count_pruned.append(dyn_step['token_count_pruned'])
            ep_iou_proxy.append(dyn_step['iou_proxy'])
            ep_action_fine_prob.append(dyn_step['action_fine_prob'])
            ep_complexity_score.append(dyn_step['complexity_score'])
            ep_interaction_lock_ratio.append(dyn_step['interaction_lock_ratio'])
            ep_coral_expert_id.append(dyn_step['coral_expert_id'])
            ep_gate_binary.append(dyn_step['gate_binary'].tolist())
            ep_gate_probs.append(dyn_step['gate_probs'].tolist())

            ep_preds.append(pred_np)
            ep_gts.append(gt)

        ep_t1 = time.perf_counter()

        if not ep_preds:
            print(f"    Ep{ep_idx}: skipped (no valid aligned samples)")
            continue

        ep_preds = np.array(ep_preds)
        ep_gts = np.array(ep_gts)
        mse = float(np.mean((ep_preds - ep_gts)**2))
        results['mse_per_episode'].append(mse)
        results['predictions'][ep_idx] = ep_preds
        results['ground_truth'][ep_idx] = ep_gts

        n_valid = len(ep_preds)
        control_bound_s = n_valid / fps_eff
        compute_aware_s = float(np.sum(np.maximum(np.array(ep_policy_ms) / 1000.0, control_period_s)))
        wall_s = ep_t1 - ep_t0

        results['episode_stats'][ep_idx] = {
            'num_frames': int(n_valid),
            'dataset_fps': fps_eff,
            'episode_wall_s': float(wall_s),
            'control_bound_s': float(control_bound_s),
            'compute_aware_s': float(compute_aware_s),
            'latency_mean_ms': float(np.mean(ep_latency_ms)),
            'latency_p95_ms': float(np.percentile(ep_latency_ms, 95)),
            'policy_step_mean_ms': float(np.mean(ep_policy_ms)),
            'preprocess_mean_ms': float(np.mean(ep_pre_ms)),
            'postprocess_mean_ms': float(np.mean(ep_post_ms)),
            'analysis_mean_ms': float(np.mean(ep_analysis_ms)),
            'skip_ratio_mean': float(np.mean(ep_skip_ratio)),
            'token_keep_mean': float(np.mean(ep_token_keep)),
            'active_layers_mean': float(np.mean(ep_active_layers)),
            'iou_proxy_mean': float(np.mean(ep_iou_proxy)),
            'action_fine_prob_mean': float(np.mean(ep_action_fine_prob)),
            'complexity_score_mean': float(np.mean(ep_complexity_score)),
            'interaction_lock_ratio_mean': float(np.mean(ep_interaction_lock_ratio)),
        }

        results['timesteps'][ep_idx] = {
            'latency_model_ms': ep_latency_ms,
            'preprocess_ms': ep_pre_ms,
            'postprocess_ms': ep_post_ms,
            'policy_step_ms': ep_policy_ms,
            'analysis_ms': ep_analysis_ms,
            'skip_ratio': ep_skip_ratio,
            'active_layers': ep_active_layers,
            'token_keep_ratio': ep_token_keep,
            'visual_entropy': ep_visual_entropy,
            'ee_velocity': ep_ee_velocity,
            'lora_effective_rank': ep_lora_rank,
            'token_count_original': ep_token_count_original,
            'token_count_pruned': ep_token_count_pruned,
            'iou_proxy': ep_iou_proxy,
            'action_fine_prob': ep_action_fine_prob,
            'complexity_score': ep_complexity_score,
            'interaction_lock_ratio': ep_interaction_lock_ratio,
            'coral_expert_id': ep_coral_expert_id,
            'gate_binary': ep_gate_binary,
            'gate_probs': ep_gate_probs,
        }

        print(
            f"    Ep{ep_idx}: MSE={mse:.4f} ({n_valid} frames) | "
            f"policy={np.mean(ep_policy_ms):.2f}ms | "
            f"task(control)={control_bound_s:.2f}s | "
            f"task(compute-aware)={compute_aware_s:.2f}s"
        )

    if not results['predictions']:
        raise RuntimeError("No valid evaluation samples after dimension alignment.")

    all_p = np.concatenate(list(results['predictions'].values()))
    all_g = np.concatenate(list(results['ground_truth'].values()))
    results['mse_total'] = float(np.mean((all_p - all_g)**2))
    results['mse_per_joint'] = np.mean((all_p - all_g)**2, axis=0).tolist()
    results['latency_mean_ms'] = float(np.mean(results['latency_ms']))
    results['latency_p95_ms'] = float(np.percentile(results['latency_ms'], 95))
    results['policy_step_mean_ms'] = float(np.mean(results['policy_step_ms']))
    results['preprocess_mean_ms'] = float(np.mean(results['preprocess_ms']))
    results['postprocess_mean_ms'] = float(np.mean(results['postprocess_ms']))
    results['analysis_mean_ms'] = float(np.mean(results['analysis_ms']))

    all_skip = []
    all_token_keep = []
    all_active_layers = []
    all_iou_proxy = []
    all_action_fine_prob = []
    all_complexity_score = []
    all_interaction_lock_ratio = []
    all_coral_expert_id = []
    for ep_data in results['timesteps'].values():
        all_skip.extend(ep_data['skip_ratio'])
        all_token_keep.extend(ep_data['token_keep_ratio'])
        all_active_layers.extend(ep_data['active_layers'])
        all_iou_proxy.extend(ep_data['iou_proxy'])
        all_action_fine_prob.extend(ep_data['action_fine_prob'])
        all_complexity_score.extend(ep_data['complexity_score'])
        all_interaction_lock_ratio.extend(ep_data['interaction_lock_ratio'])
        all_coral_expert_id.extend(ep_data['coral_expert_id'])
    results['skip_ratio_mean'] = float(np.mean(all_skip)) if all_skip else 0.0
    results['token_keep_mean'] = float(np.mean(all_token_keep)) if all_token_keep else 1.0
    results['active_layers_mean'] = float(np.mean(all_active_layers)) if all_active_layers else float(num_vlm_layers)
    results['iou_proxy_mean'] = float(np.mean(all_iou_proxy)) if all_iou_proxy else 0.0
    results['action_fine_prob_mean'] = float(np.mean(all_action_fine_prob)) if all_action_fine_prob else 0.0
    results['complexity_score_mean'] = float(np.mean(all_complexity_score)) if all_complexity_score else 0.0
    results['interaction_lock_ratio_mean'] = float(np.mean(all_interaction_lock_ratio)) if all_interaction_lock_ratio else 0.0
    results['coral_expert_id_mean'] = float(np.mean(all_coral_expert_id)) if all_coral_expert_id else 0.0

    ep_stats = list(results['episode_stats'].values())
    results['task_time_control_mean_s'] = float(np.mean([x['control_bound_s'] for x in ep_stats]))
    results['task_time_compute_aware_mean_s'] = float(np.mean([x['compute_aware_s'] for x in ep_stats]))
    results['task_time_wall_mean_s'] = float(np.mean([x['episode_wall_s'] for x in ep_stats]))
    results['eval_action_dim'] = int(all_p.shape[-1])

    return results


def evaluate_baseline_model(eval_episodes, eval_action_dim):
    baseline = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    baseline.to(DEVICE)
    baseline.eval()

    base_results = {
        'mse_per_episode': [],
        'latency_ms': [],
        'predictions': {},
        'ground_truth': {},
        'mse_total': float('nan'),
        'latency_mean_ms': float('nan'),
    }

    for ep_idx in eval_episodes:
        ep_preds, ep_gts = [], []
        baseline.reset()
        for step_idx in tqdm(episode_indices[ep_idx], desc=f"  Baseline Ep{ep_idx}", ncols=85, leave=False):
            s = dataset[step_idx]
            gt_full = s[action_key].numpy()
            batch = build_eval_batch(s, DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                pred = baseline.select_action(batch)
            t1 = time.perf_counter()

            pred_np = np.asarray(pred.squeeze().cpu().numpy()).reshape(-1)
            step_dim = min(pred_np.shape[0], gt_full.shape[-1], eval_action_dim)
            if step_dim < eval_action_dim:
                continue

            pred_np = pred_np[:eval_action_dim]
            gt_np = gt_full[:eval_action_dim]
            ep_preds.append(pred_np)
            ep_gts.append(gt_np)
            base_results['latency_ms'].append((t1 - t0) * 1000)

        if not ep_preds:
            continue

        ep_preds = np.array(ep_preds)
        ep_gts = np.array(ep_gts)
        base_results['predictions'][ep_idx] = ep_preds
        base_results['ground_truth'][ep_idx] = ep_gts
        base_results['mse_per_episode'].append(float(np.mean((ep_preds - ep_gts) ** 2)))

    if base_results['predictions']:
        all_bp = np.concatenate(list(base_results['predictions'].values()))
        all_bg = np.concatenate(list(base_results['ground_truth'].values()))
        base_results['mse_total'] = float(np.mean((all_bp - all_bg) ** 2))
        base_results['mse_per_joint'] = np.mean((all_bp - all_bg) ** 2, axis=0).tolist()
        base_results['latency_mean_ms'] = float(np.mean(base_results['latency_ms']))

    return base_results


def export_timestep_csv(results, output_dir):
    for ep_idx, ep_data in results['timesteps'].items():
        csv_path = output_dir / f'dynamic_ep{ep_idx}_timestep_metrics.csv'
        fieldnames = [
            'step', 'latency_model_ms', 'preprocess_ms', 'postprocess_ms', 'policy_step_ms', 'analysis_ms',
            'skip_ratio', 'active_layers', 'token_keep_ratio',
            'visual_entropy', 'ee_velocity', 'lora_effective_rank',
            'token_count_original', 'token_count_pruned',
            'iou_proxy', 'action_fine_prob', 'complexity_score', 'interaction_lock_ratio', 'coral_expert_id',
        ] + [f'gate_binary_{i}' for i in range(dyn_cfg['num_skippable_layers'])] + [
            f'gate_prob_{i}' for i in range(dyn_cfg['num_skippable_layers'])
        ]

        n_steps = len(ep_data['latency_model_ms'])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for step in range(n_steps):
                row = {
                    'step': step,
                    'latency_model_ms': ep_data['latency_model_ms'][step],
                    'preprocess_ms': ep_data['preprocess_ms'][step],
                    'postprocess_ms': ep_data['postprocess_ms'][step],
                    'policy_step_ms': ep_data['policy_step_ms'][step],
                    'analysis_ms': ep_data['analysis_ms'][step],
                    'skip_ratio': ep_data['skip_ratio'][step],
                    'active_layers': ep_data['active_layers'][step],
                    'token_keep_ratio': ep_data['token_keep_ratio'][step],
                    'visual_entropy': ep_data['visual_entropy'][step],
                    'ee_velocity': ep_data['ee_velocity'][step],
                    'lora_effective_rank': ep_data['lora_effective_rank'][step],
                    'token_count_original': ep_data['token_count_original'][step],
                    'token_count_pruned': ep_data['token_count_pruned'][step],
                    'iou_proxy': ep_data['iou_proxy'][step],
                    'action_fine_prob': ep_data['action_fine_prob'][step],
                    'complexity_score': ep_data['complexity_score'][step],
                    'interaction_lock_ratio': ep_data['interaction_lock_ratio'][step],
                    'coral_expert_id': ep_data['coral_expert_id'][step],
                }
                for i in range(dyn_cfg['num_skippable_layers']):
                    row[f'gate_binary_{i}'] = ep_data['gate_binary'][step][i]
                    row[f'gate_prob_{i}'] = ep_data['gate_probs'][step][i]
                writer.writerow(row)


# Evaluate on selected episodes from split/config.
print(f"  Eval episodes: {eval_ep_list}")

if not eval_ep_list:
    print("  ERROR: No valid eval episodes found for this dataset/config.")
    sys.exit(1)

fps_eval = dataset.fps if hasattr(dataset, 'fps') and dataset.fps else 15

print(f"\n  Evaluating {RUN_LABEL} (from final_model.pt)...")
dyn_results = evaluate_dynamic_model(smolvla, eval_ep_list, fps_eval)
print(f"\n  Overall MSE: {dyn_results['mse_total']:.4f}")
print(f"  Avg Latency (model): {dyn_results['latency_mean_ms']:.1f}ms (p95={dyn_results['latency_p95_ms']:.1f}ms)")
print(f"  Avg Policy Step (pre+model+post): {dyn_results['policy_step_mean_ms']:.1f}ms")
print(f"  Avg Task Time (control-bound): {dyn_results['task_time_control_mean_s']:.2f}s")
print(f"  Avg Task Time (compute-aware): {dyn_results['task_time_compute_aware_mean_s']:.2f}s")
print(f"  Per-joint MSE: {[f'{m:.2f}' for m in dyn_results['mse_per_joint']]}")

baseline_results = None
print("\n  Evaluating Baseline SmolVLA (for side-by-side video/report)...")
try:
    baseline_results = evaluate_baseline_model(eval_ep_list, dyn_results['eval_action_dim'])
    if baseline_results['predictions']:
        print(f"  Baseline MSE: {baseline_results['mse_total']:.4f}")
        print(f"  Baseline Avg Latency: {baseline_results['latency_mean_ms']:.1f}ms")
    else:
        baseline_results = None
        print("  Baseline evaluation skipped (no valid aligned samples).")
except Exception as e:
    baseline_results = None
    print(f"  WARNING: baseline evaluation failed: {e}")

export_timestep_csv(dyn_results, OUTPUT_DIR)
print("  Timestep CSV exported.")

# =====================================================================
#  PHASE 4: Comparison Plots
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 4: Comparison Plots")
print("=" * 70)

n_eval_eps = len(eval_ep_list)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
blue = '#2196F3'

# [0,0] MSE per episode
ax = axes[0,0]
x = np.arange(n_eval_eps)
ax.bar(x, dyn_results['mse_per_episode'], 0.6, label=RUN_LABEL, color=blue)
ax.set_xlabel('Episode'); ax.set_ylabel('MSE'); ax.set_title('MSE per Episode')
ax.set_xticks(x); ax.set_xticklabels([f'Ep{e}' for e in eval_ep_list])
ax.legend(); ax.grid(True, alpha=0.3)

# [0,1] MSE per joint
ax = axes[0,1]
x_j = np.arange(dyn_results['eval_action_dim'])
ax.bar(x_j, dyn_results['mse_per_joint'], 0.6, label=RUN_LABEL, color=blue)
ax.set_xlabel('Joint'); ax.set_ylabel('MSE'); ax.set_title('MSE per Joint')
ax.set_xticks(x_j); ax.legend(); ax.grid(True, alpha=0.3)

# [0,2] Latency distribution
ax = axes[0,2]
ax.hist(dyn_results['latency_ms'], bins=40, alpha=0.7, color=blue, label='Dynamic')
if baseline_results is not None:
    ax.hist(baseline_results['latency_ms'], bins=40, alpha=0.5, color='#ef5350', label='Baseline')
ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Count'); ax.set_title('Inference Latency')
ax.legend()
ax.grid(True, alpha=0.3)

# [1,0] Trajectory Ep40
ax = axes[1,0]
ep0 = eval_ep_list[0]
gt0 = dyn_results['ground_truth'][ep0]
dp0 = dyn_results['predictions'][ep0]
nj = min(3, dyn_results['eval_action_dim'])
t_ax = np.arange(len(gt0))
for j in range(nj):
    ax.plot(t_ax, gt0[:,j], '-', color=f'C{j}', lw=2, label=f'GT J{j}')
    ax.plot(t_ax, dp0[:,j], '--', color=f'C{j}', alpha=0.6, label=f'Pred J{j}')
    if baseline_results is not None and ep0 in baseline_results['predictions']:
        bp0 = baseline_results['predictions'][ep0]
        ax.plot(t_ax[:len(bp0)], bp0[:len(t_ax),j], ':', color=f'C{j}', alpha=0.9, label=f'Base J{j}')
ax.set_xlabel('Step'); ax.set_ylabel('Action'); ax.set_title(f'Trajectory Ep{ep0}')
ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)

# [1,1] Trajectory Ep41 (if available)
ax = axes[1,1]
if len(eval_ep_list) > 1:
    ep1 = eval_ep_list[1]
    gt1 = dyn_results['ground_truth'][ep1]
    dp1 = dyn_results['predictions'][ep1]
    t_ax1 = np.arange(len(gt1))
    for j in range(nj):
        ax.plot(t_ax1, gt1[:,j], '-', color=f'C{j}', lw=2, label=f'GT J{j}')
        ax.plot(t_ax1, dp1[:,j], '--', color=f'C{j}', alpha=0.6, label=f'Pred J{j}')
        if baseline_results is not None and ep1 in baseline_results['predictions']:
            bp1 = baseline_results['predictions'][ep1]
            ax.plot(t_ax1[:len(bp1)], bp1[:len(t_ax1),j], ':', color=f'C{j}', alpha=0.9, label=f'Base J{j}')
    ax.set_xlabel('Step'); ax.set_ylabel('Action'); ax.set_title(f'Trajectory Ep{ep1}')
    ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)
else:
    ax.axis('off')

# [1,2] Summary text
ax = axes[1,2]
ax.axis('off')
summary = (
    f"{RUN_LABEL.upper()} EVALUATION\n"
    f"(from final_model.pt)\n\n"
    f"Dataset: {ds_cfg['repo_id']}\n"
    f"Eval episodes: {eval_ep_list}\n\n"
    f"Total MSE:   {dyn_results['mse_total']:.4f}\n"
    f"Avg Latency: {dyn_results['latency_mean_ms']:.1f}ms (p95={dyn_results['latency_p95_ms']:.1f})\n"
    f"Policy Step: {dyn_results['policy_step_mean_ms']:.1f}ms\n"
    f"Task Time(control): {dyn_results['task_time_control_mean_s']:.2f}s\n"
    f"Task Time(compute-aware): {dyn_results['task_time_compute_aware_mean_s']:.2f}s\n"
    f"Skip Ratio(mean): {dyn_results['skip_ratio_mean']:.2%}\n"
    f"Token Keep(mean): {dyn_results['token_keep_mean']:.2%}\n\n"
    f"Per-episode MSE:\n"
)
for i, ep in enumerate(eval_ep_list):
    ep_stats = dyn_results['episode_stats'].get(ep, {})
    summary += (
        f"  Ep{ep}: {dyn_results['mse_per_episode'][i]:.4f} | "
        f"time={ep_stats.get('compute_aware_s', float('nan')):.2f}s\n"
    )
if baseline_results is not None:
    summary += (
        f"\nBaseline MSE: {baseline_results['mse_total']:.4f}\n"
        f"Baseline Latency: {baseline_results['latency_mean_ms']:.1f}ms\n"
    )
summary += "\nTechniques:\n"
for tech in TECHNIQUE_LINES:
    summary += f"  - {tech}\n"
ax.text(0.05, 0.5, summary, fontsize=10, fontfamily='monospace',
        va='center', ha='left', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle(PLOT_TITLE,
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dynamic_comparison_plots.png', dpi=150)
plt.close()
print(f"  Plots saved: {OUTPUT_DIR / 'dynamic_comparison_plots.png'}")

def draw_metric_bar(frame, x, y, label, value, vmin, vmax, color):
    bar_w, bar_h = 180, 14
    cv2.putText(frame, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (55, 55, 55), -1)
    span = max(vmax - vmin, 1e-8)
    ratio = float(np.clip((value - vmin) / span, 0.0, 1.0))
    fill_w = int(bar_w * ratio)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), color, -1)
    cv2.putText(frame, f"{value:.3f}", (x + bar_w + 8, y + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)


def draw_recent_heatmap(frame, gate_binary, t_idx, x, y, w, h):
    if not gate_binary:
        return
    st = max(0, t_idx - 59)
    recent = np.array(gate_binary[st:t_idx + 1], dtype=np.float32)
    if recent.size == 0:
        return

    total_layers = dyn_cfg['num_fixed_layers'] + dyn_cfg['num_skippable_layers']
    full = np.ones((recent.shape[0], total_layers), dtype=np.float32)
    full[:, dyn_cfg['num_fixed_layers']:] = recent

    heat = np.zeros((total_layers, recent.shape[0], 3), dtype=np.uint8)
    active = full.T > 0.5
    heat[active] = (70, 210, 70)
    heat[~active] = (60, 60, 220)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = heat
    cv2.rectangle(frame, (x, y), (x + w, y + h), (140, 140, 140), 1)


# =====================================================================
#  PHASE 5: Generate Videos with Dynamic Overlay + Baseline Comparison
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 5: Generate Videos — overlay + heatmap + baseline comparison")
print("=" * 70)

for ep_idx in eval_ep_list:
    if ep_idx not in dyn_results['timesteps']:
        continue

    gt = dyn_results['ground_truth'][ep_idx]
    dp = dyn_results['predictions'][ep_idx]
    step_data = dyn_results['timesteps'][ep_idx]
    bp = baseline_results['predictions'].get(ep_idx) if baseline_results is not None else None

    indices = episode_indices[ep_idx][:len(gt)]
    n = min(len(gt), len(dp), len(step_data['skip_ratio']), len(indices))
    if bp is not None:
        n = min(n, len(bp))

    vpath = OUTPUT_DIR / f'dynamic_ep{ep_idx}.mp4'
    fw, fh = 1280, 550
    fps_out = dataset.fps if hasattr(dataset, 'fps') and dataset.fps else 15
    writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*'mp4v'), fps_out, (fw, fh))

    for t in tqdm(range(n), desc=f"  Video Ep{ep_idx}", ncols=85, leave=False):
        frame = np.ones((fh, fw, 3), dtype=np.uint8) * 25
        s = dataset[indices[t]]
        if image_keys:
            img = s[image_keys[0]].numpy()
            if img.shape[0] <= 4:
                img = np.transpose(img, (1, 2, 0))
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (320, 240))
            frame[10:250, 10:330] = img

        cv2.putText(frame, f'Ep {ep_idx} | Step {t}/{n} | {RUN_LABEL} Signals + Trajectory', (340, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)

        draw_recent_heatmap(frame, step_data['gate_binary'], t, 340, 38, 260, 185)
        cv2.putText(frame, 'Layer Activity (green=keep, red=skip)', (340, 234),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1)

        skip_ratio = step_data['skip_ratio'][t]
        active_layers = step_data['active_layers'][t]
        token_keep = step_data['token_keep_ratio'][t]
        latency_model = step_data['latency_model_ms'][t]
        policy_step = step_data['policy_step_ms'][t]

        draw_metric_bar(frame, 340, 260, 'Skip Ratio', skip_ratio, 0.0, 1.0, (90, 190, 255))
        draw_metric_bar(frame, 340, 288, 'Token Keep', token_keep, 0.0, 1.0, (90, 255, 150))
        draw_metric_bar(frame, 340, 316, 'Model Latency (ms)', latency_model, 0.0, max(20.0, dyn_results['latency_p95_ms'] * 1.2), (255, 180, 90))
        draw_metric_bar(frame, 340, 344, 'Policy Step (ms)', policy_step, 0.0, max(30.0, dyn_results['policy_step_mean_ms'] * 2.0), (255, 140, 140))
        cv2.putText(frame, f'Active Layers: {active_layers}/{dyn_cfg["num_fixed_layers"] + dyn_cfg["num_skippable_layers"]}',
                    (340, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        nj_v = min(dyn_results['eval_action_dim'], 6)
        px0 = 620
        pw = fw - px0 - 15
        jh = max(45, (fh - 80) // nj_v - 6)

        for j in range(nj_v):
            y0 = 35 + j * (jh + 4)
            cv2.putText(frame, f'J{j}', (px0, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            cv2.rectangle(frame, (px0 + 22, y0), (px0 + pw, y0 + jh), (40, 40, 40), -1)

            win_size = 40
            st = max(0, t - win_size)
            vals = [gt[st:t + 1, j], dp[st:t + 1, j]]
            if bp is not None:
                vals.append(bp[st:t + 1, j])
            vmin = min(v.min() for v in vals) - 0.05
            vmax = max(v.max() for v in vals) + 0.05
            if vmax - vmin < 0.01:
                vmax = vmin + 0.01

            def _px(tt, _st=st, _pw=pw, _px0=px0):
                return _px0 + 22 + int((tt - _st) / max(win_size, 1) * (_pw - 25))

            def _py(v, _y0=y0, _jh=jh, _vmin=vmin, _vmax=vmax):
                return _y0 + _jh - int((v - _vmin) / (_vmax - _vmin) * _jh)

            for tt in range(st, min(t, n - 1)):
                x1, x2 = _px(tt), _px(tt + 1)
                cv2.line(frame, (x1, _py(gt[tt, j])), (x2, _py(gt[tt + 1, j])), (0, 220, 0), 2)
                cv2.line(frame, (x1, _py(dp[tt, j])), (x2, _py(dp[tt + 1, j])), (255, 160, 0), 1)
                if bp is not None:
                    cv2.line(frame, (x1, _py(bp[tt, j])), (x2, _py(bp[tt + 1, j])), (80, 120, 255), 1)

        yb = fh - 25
        cv2.putText(frame, 'GT', (15, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(frame, RUN_LABEL, (55, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
        if bp is not None:
            cv2.putText(frame, 'Baseline', (150, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 120, 255), 1)

        mse_dyn = float(np.mean((dp[t] - gt[t]) ** 2))
        cv2.putText(frame, f'Dynamic MSE:{mse_dyn:.4f}', (620, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 160, 0), 1)
        if bp is not None:
            mse_base = float(np.mean((bp[t] - gt[t]) ** 2))
            cv2.putText(frame, f'Baseline MSE:{mse_base:.4f}', (840, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 120, 255), 1)

        writer.write(frame)

    writer.release()
    print(f"  ✓ Video: {vpath}")

# =====================================================================
#  PHASE 6: Save Report
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 6: Save Report")
print("=" * 70)

config_block = {
    'lora_max_rank': dyn_cfg['lora_max_rank'],
    'lora_energy_threshold': dyn_cfg['lora_energy_threshold'],
    'num_fixed_layers': dyn_cfg['num_fixed_layers'],
    'num_skippable_layers': dyn_cfg['num_skippable_layers'],
}
if profile["router_kind"] == "hierarchical":
    config_block['num_spatial_layers'] = dyn_cfg['num_spatial_layers']
    config_block['num_action_layers'] = dyn_cfg['num_action_layers']
elif profile["router_kind"] == "lyapunov":
    config_block.update({
        'interaction_lock_entropy_tau': dyn_cfg['interaction_lock_entropy_tau'],
        'snap_curvature_delta': dyn_cfg['snap_curvature_delta'],
        'cogkd_temperature': dyn_cfg['cogkd_temperature'],
    })
if profile["pruner_kind"] == "cim":
    config_block.update({
        'cim_conservative_ratio': dyn_cfg.get('cim_conservative_ratio', dyn_cfg.get('iap_conservative_ratio', 0.8)),
        'cim_aggressive_ratio': dyn_cfg.get('cim_aggressive_ratio', dyn_cfg.get('iap_aggressive_ratio', 0.3)),
        'cim_iou_proxy_tau': dyn_cfg.get('cim_iou_proxy_tau', dyn_cfg.get('iap_iou_threshold', 0.5)),
        'cim_temporal_gamma': dyn_cfg.get('cim_temporal_gamma', dyn_cfg.get('iap_temporal_momentum', 0.7)),
        'cim_interaction_entropy_tau': dyn_cfg.get('cim_interaction_entropy_tau', 1.6),
        'complexity_alpha_entropy': dyn_cfg.get('complexity_alpha_entropy', 0.40),
        'complexity_alpha_iou_proxy': dyn_cfg.get('complexity_alpha_iou_proxy', 0.35),
        'complexity_alpha_grad_proxy': dyn_cfg.get('complexity_alpha_grad_proxy', 0.25),
        'hybrid_tau_warmup_steps': dyn_cfg.get('hybrid_tau_warmup_steps', 0),
        'hybrid_tau_fixed': dyn_cfg.get('hybrid_tau_fixed', 1000.0),
        'hybrid_tau_percentile': dyn_cfg.get('hybrid_tau_percentile', 85.0),
        'hybrid_tau_window': dyn_cfg.get('hybrid_tau_window', 128),
        'hard_sample_manifest_path': dyn_cfg.get('hard_sample_manifest_path', ''),
    })
elif profile["pruner_kind"] == "iap":
    config_block.update({
        'iap_conservative_ratio': dyn_cfg['iap_conservative_ratio'],
        'iap_aggressive_ratio': dyn_cfg['iap_aggressive_ratio'],
        'iap_iou_threshold': dyn_cfg['iap_iou_threshold'],
        'iap_temporal_momentum': dyn_cfg['iap_temporal_momentum'],
        'iap_geometric_anchor_ratio': dyn_cfg['iap_geometric_anchor_ratio'],
    })
elif profile["pruner_kind"] == "vtb":
    config_block.update({
        'vtb_conservative_ratio': dyn_cfg['vtb_conservative_ratio'],
        'vtb_aggressive_ratio': dyn_cfg['vtb_aggressive_ratio'],
        'vtb_temporal_gamma': dyn_cfg['vtb_temporal_gamma'],
        'vtb_semantic_weight': dyn_cfg['vtb_semantic_weight'],
        'vtb_structural_weight': dyn_cfg['vtb_structural_weight'],
    })
else:
    adp_threshold = dyn_cfg.get('adp_v_threshold', dyn_cfg.get('adp_velocity_threshold', 0.15))
    config_block.update({
        'adp_v_threshold': adp_threshold,
        'adp_min_keep_ratio': dyn_cfg['adp_min_keep_ratio'],
    })
if profile["use_coral"]:
    config_block.update({
        'coral_num_experts': dyn_cfg['coral_num_experts'],
        'coral_expert_rank': dyn_cfg['coral_expert_rank'],
    })

techniques_block = {
    'Router': (
        'Hierarchical STAR Router with spatial/action sub-routing'
        if profile['router_kind'] == 'hierarchical'
        else 'Lyapunov-Stable STAR Router'
        if profile['router_kind'] == 'lyapunov'
        else 'DySL STAR Router'
    ),
    'LoRA-SP': f'Select-Prune LoRA (max_rank={dyn_cfg["lora_max_rank"]}, eta={dyn_cfg["lora_energy_threshold"]})',
    'SnapFlow': (
        f'Second-order curvature-aware distillation (teacher_steps={dyn_cfg["snap_teacher_steps"]}, delta={dyn_cfg.get("snap_curvature_delta", 0.08)})'
        if profile['router_kind'] == 'lyapunov'
        else f'Self-distillation for 1-NFE denoising (teacher_steps={dyn_cfg["snap_teacher_steps"]})'
    ),
}
if profile["pruner_kind"] == "cim":
    techniques_block['Pruner'] = (
        f'Contextual Interaction Masking (conservative={dyn_cfg.get("cim_conservative_ratio", dyn_cfg.get("iap_conservative_ratio", 0.8))}, '
        f'aggressive={dyn_cfg.get("cim_aggressive_ratio", dyn_cfg.get("iap_aggressive_ratio", 0.3))}, iou_tau={dyn_cfg.get("cim_iou_proxy_tau", dyn_cfg.get("iap_iou_threshold", 0.5))})'
    )
    techniques_block['EGSF'] = (
        f'Entropy-guided complexity (w_entropy={dyn_cfg.get("complexity_alpha_entropy", 0.4)}, '
        f'w_iou={dyn_cfg.get("complexity_alpha_iou_proxy", 0.35)}, w_grad={dyn_cfg.get("complexity_alpha_grad_proxy", 0.25)})'
    )
    techniques_block['HybridTauGuard'] = (
        f'lambda_cost disabled when mse_signal > tau_eff (warmup={dyn_cfg.get("hybrid_tau_warmup_steps", 0)}, p={dyn_cfg.get("hybrid_tau_percentile", 85.0)})'
    )
elif profile["pruner_kind"] == "iap":
    techniques_block['Pruner'] = (
        f'Interaction-Aligned Pruning (conservative={dyn_cfg["iap_conservative_ratio"]}, '
        f'aggressive={dyn_cfg["iap_aggressive_ratio"]}, iou_threshold={dyn_cfg["iap_iou_threshold"]})'
    )
elif profile["pruner_kind"] == "vtb":
    techniques_block['Pruner'] = (
        f'Variational Token Bottleneck (conservative={dyn_cfg["vtb_conservative_ratio"]}, '
        f'aggressive={dyn_cfg["vtb_aggressive_ratio"]}, gamma={dyn_cfg["vtb_temporal_gamma"]})'
    )
else:
    adp_threshold = dyn_cfg.get('adp_v_threshold', dyn_cfg.get('adp_velocity_threshold', 0.15))
    techniques_block['Pruner'] = (
        f'Adaptive Dynamic Pruning (v_threshold={adp_threshold}, '
        f'min_keep_ratio={dyn_cfg["adp_min_keep_ratio"]})'
    )
if profile["use_coral"]:
    techniques_block['CORAL'] = (
        f'Language-routed LoRA experts ({dyn_cfg["coral_num_experts"]} experts, rank={dyn_cfg["coral_expert_rank"]})'
    )

report = {
    'pipeline': profile['pipeline_name'],
    'checkpoint': str(CHECKPOINT_PATH),
    'dataset': ds_cfg['repo_id'],
    'dataset_key': DATASET_KEY,
    'dataset_fps': float(fps_eval),
    'split': {
        'train_episodes': list(train_eps),
        'eval_episodes_from_config': list(eval_eps),
        'eval_episodes_used': list(eval_ep_list),
        'eval_episode_frame_counts': {
            f'ep{ep}': int(dyn_results['episode_stats'][ep]['num_frames'])
            for ep in eval_ep_list if ep in dyn_results['episode_stats']
        },
    },
    'eval_episodes': list(eval_ep_list),
    'config': config_block,
    'results': {
        'label': REPORT_LABEL,
        'params_total_M': round(smolvla_total/1e6, 1),
        'dynamic_params_M': round(wrapper_params/1e6, 2),
        'total_mse': dyn_results['mse_total'],
        'per_episode_mse': {f'ep{ep}': mse for ep, mse in zip(eval_ep_list, dyn_results['mse_per_episode'])},
        'per_joint_mse': dyn_results['mse_per_joint'],
        'avg_latency_ms': dyn_results['latency_mean_ms'],
        'p95_latency_ms': dyn_results['latency_p95_ms'],
        'avg_policy_step_ms': dyn_results['policy_step_mean_ms'],
        'avg_preprocess_ms': dyn_results['preprocess_mean_ms'],
        'avg_postprocess_ms': dyn_results['postprocess_mean_ms'],
        'avg_analysis_ms': dyn_results['analysis_mean_ms'],
        'avg_skip_ratio': dyn_results['skip_ratio_mean'],
        'avg_token_keep_ratio': dyn_results['token_keep_mean'],
        'avg_active_layers': dyn_results['active_layers_mean'],
        'avg_iou_proxy': dyn_results['iou_proxy_mean'],
        'avg_action_fine_prob': dyn_results['action_fine_prob_mean'],
        'avg_complexity_score': dyn_results['complexity_score_mean'],
        'avg_interaction_lock_ratio': dyn_results['interaction_lock_ratio_mean'],
        'avg_coral_expert_id': dyn_results['coral_expert_id_mean'],
        'task_time_control_mean_s': dyn_results['task_time_control_mean_s'],
        'task_time_compute_aware_mean_s': dyn_results['task_time_compute_aware_mean_s'],
        'task_time_wall_mean_s': dyn_results['task_time_wall_mean_s'],
        'per_episode_timing': {
            f'ep{ep}': dyn_results['episode_stats'][ep]
            for ep in eval_ep_list if ep in dyn_results['episode_stats']
        },
    },
    'baseline': (
        {
            'label': 'SmolVLA Base (Hub)',
            'total_mse': baseline_results['mse_total'],
            'avg_latency_ms': baseline_results['latency_mean_ms'],
            'per_episode_mse': {
                f'ep{ep}': mse for ep, mse in zip(eval_ep_list, baseline_results['mse_per_episode'])
            },
            'per_joint_mse': baseline_results.get('mse_per_joint', []),
        }
        if baseline_results is not None else None
    ),
    'techniques': techniques_block,
    'outputs': {
        'videos': [f'dynamic_ep{ep}.mp4' for ep in eval_ep_list],
        'timestep_csv': [f'dynamic_ep{ep}_timestep_metrics.csv' for ep in eval_ep_list],
        'plots': 'dynamic_comparison_plots.png',
        'report': 'dynamic_report.json',
    },
}
with open(OUTPUT_DIR / 'dynamic_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f"  Report: {OUTPUT_DIR / 'dynamic_report.json'}")

# =====================================================================
#  DONE
# =====================================================================
print("\n" + "=" * 70)
print("  EVALUATION COMPLETE")
print("=" * 70)
print(f"\n  Total MSE: {dyn_results['mse_total']:.4f}")
print(f"  Avg Latency: {dyn_results['latency_mean_ms']:.1f}ms (p95={dyn_results['latency_p95_ms']:.1f}ms)")
print(f"  Avg Policy Step: {dyn_results['policy_step_mean_ms']:.1f}ms")
print(f"  Avg Task Time (compute-aware): {dyn_results['task_time_compute_aware_mean_s']:.2f}s")
if baseline_results is not None:
    print(f"  Baseline MSE: {baseline_results['mse_total']:.4f}")
    print(f"  Baseline Avg Latency: {baseline_results['latency_mean_ms']:.1f}ms")
for i, ep in enumerate(eval_ep_list):
    print(f"  Ep{ep} MSE: {dyn_results['mse_per_episode'][i]:.4f}")
print(f"\n  Output directory: {OUTPUT_DIR}")
print(f"  Videos: {[f'dynamic_ep{ep}.mp4' for ep in eval_ep_list]}")
print(f"  Plots:  dynamic_comparison_plots.png")
print(f"  Report: dynamic_report.json")
print("\n  DONE!")
