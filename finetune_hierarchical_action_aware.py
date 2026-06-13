"""
Fine-Tune SmolVLA — Hierarchical Action-Aware Optimization
===========================================================
Pipeline: finetune_hierarchical_action_aware
RTX 4070 SUPER (12 GB) | fp16/bf16 | lerobot 0.4.4

Key Contribution: TRUE CASCADE — DTP output (R_keep) directly
conditions the DLS router (CascadeSTARRouter), creating a
coupled budget signal that drives FLOPs reduction from both
the horizontal (token) and vertical (layer) dimension jointly.

Architecture Enhancements:
  1. HierarchicalADPPruner     — velocity-driven token pruning,
                                 wave-aligned for GPU tail elimination
  2. CascadeSTARRouter         — layer skip router conditioned on R_keep
  3. LoRASPAdapter             — dynamic rank adaptation (same as base)
  4. ToIAwareCogKDHelper       — CogKD with ToI masking + frozen teacher
  5. SnapFlowTrainer           — 1-NFE self-distillation (same as base)

Training Schedule:
  Phase 1 (Steps 0-5K):      CascadeSTAR + ADP initialisation
  Phase 2 (Steps 5K-15K):    Joint — add LoRA-SP + budget coupling
  Phase 3 (Steps 15K-20K):   CogKD (ToI-masked) + SnapFlow distillation

Unified Loss:
  L = L_task + alpha * L_budget + beta * L_gate
      + gamma * L_distill + eta * L_spec
"""
import argparse
import os, sys, time, json, types, gc, copy, math
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
_spec = importlib.util.spec_from_file_location(
    'lerobot.robots.config', os.path.join(_pkg, 'robots', 'config.py'))
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
#  ARGUMENT PARSING
# =====================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Hierarchical Action-Aware VLA Fine-Tune")
    p.add_argument("--dataset", default="libero_spatial",
                   help="Key from DATASETS in finetune_config.py")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_root", default="./outputs")
    p.add_argument("--output_name", default="hierarchical_action_aware")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: run 10 steps per phase then exit")
    p.add_argument("--start_phase", type=int, default=1, choices=[1, 2, 3],
                   help="Start from this phase (1=default, 3=skip to SnapFlow). "
                        "Requires a checkpoint saved by a previous run.")
    p.add_argument("--end_phase", type=int, default=3, choices=[1, 2, 3],
                   help="Stop AFTER this phase (default 3). "
                        "Use --end_phase 1 to train Phase 1 only and stop.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt checkpoint to resume from (auto-detected if omitted). "
                        "For --start_phase 3 defaults to <output_dir>/phase1_complete.pt")
    p.add_argument("--phase1_steps", type=int, default=None, help="Override cfg phase1_steps")
    p.add_argument("--phase2_steps", type=int, default=None, help="Override cfg phase2_steps")
    p.add_argument("--phase3_steps", type=int, default=None, help="Override cfg phase3_steps")
    p.add_argument("--task_priority", action="store_true",
                   help="Scale DOWN the compression/anti-collapse lambdas so the task "
                        "loss dominates — preserves base policy accuracy when finetuning "
                        "from a competent base (smolvla_libero). Recommended.")
    p.add_argument("--micro_batch", type=int, default=None,
                   help="Override cfg micro_batch (samples per grad-accum step). "
                        "Default from config (8 for RTX 4090). Use 4 if OOM.")
    p.add_argument("--grad_accum", type=int, default=None,
                   help="Override cfg grad_accum. Effective batch = micro_batch × grad_accum.")
    p.add_argument("--eval_every", type=int, default=None,
                   help="Compute validation loss on test split every N steps. "
                        "Default: same as log_every.")
    return p.parse_args()


# =====================================================================
#  LOAD CONFIG
# =====================================================================
from finetune_config import DATASETS, TRAINING, EVAL

args = _parse_args()

DATASET_KEY = args.dataset
ds_cfg = DATASETS[DATASET_KEY]
cfg = dict(TRAINING["hierarchical_action_aware"])   # copy so CLI overrides never mutate the global
SMOKE = args.smoke
DEVICE = args.device
OUTPUT_DIR = Path(args.output_root) / args.output_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Smoke overrides
if SMOKE:
    cfg["phase1_steps"] = 10
    cfg["phase2_steps"] = 10
    cfg["phase3_steps"] = 10
    cfg["save_every"] = 10
    cfg["log_every"] = 1

# Per-phase step overrides from CLI (take precedence over config / smoke)
if args.phase1_steps is not None:
    cfg["phase1_steps"] = args.phase1_steps
if args.phase2_steps is not None:
    cfg["phase2_steps"] = args.phase2_steps
if args.phase3_steps is not None:
    cfg["phase3_steps"] = args.phase3_steps
if args.micro_batch is not None:
    cfg["micro_batch"] = args.micro_batch
if args.grad_accum is not None:
    cfg["grad_accum"] = args.grad_accum
EVAL_EVERY = args.eval_every if args.eval_every is not None else cfg["log_every"]

# Task-priority: gentle compression lambdas so finetuning a competent base
# (smolvla_libero) does not destroy its task accuracy. The original aggressive
# values (min_layer_on=8, skip_ceiling=6, gate_diversity=7, …) overwhelmed the
# task loss and dropped success 36.7% → 6-10%.
if args.task_priority:
    cfg.update({
        "lambda_min_layer_on":    0.5,
        "lambda_skip_ceiling":    0.5,
        "lambda_gate_diversity":  0.2,
        "lambda_temporal_entropy":0.2,
        "lambda_visual_coupling": 0.2,
        "lambda_rho_spread":      0.1,
        "lambda_rho_supervision": 0.3,
        "lambda_always_on":       0.1,
        "lambda_gate":            0.05,
        "lambda_budget":          0.3,
        "lambda_budget_start":    0.1,
        "lambda_budget_end":      0.5,
        "lambda_gate_flip":       0.1,
    })
    print("  [TASK-PRIORITY] compression lambdas scaled down — task loss dominates.")

print("=" * 70)
print("  SmolVLA — Hierarchical Action-Aware Optimization")
print("  DTP → DLS Cascade  |  CogKD (ToI)  |  SnapFlow 1-NFE")
print("=" * 70)
if DEVICE == "cuda":
    print(f"  Device  : {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  Device  : cpu")
print(f"  Dataset : {ds_cfg['repo_id']}")
print(f"  Output  : {OUTPUT_DIR}")
if SMOKE:
    print("  [SMOKE TEST — 10 steps per phase]")
print()

# =====================================================================
#  CUSTOM MODULES
# =====================================================================

class HierarchicalADPPruner(nn.Module):
    """Hierarchical Action-Aware Dynamic Token Pruner.

    Derives keep-ratio from EE velocity using the formula from the report:
        R_keep = clamp(R_min + (1-R_min) * v_thresh / v_ee, R_min, 1.0)

    Includes GPU-wave alignment: snaps K to the nearest multiple of
    wave_size so the pruned sequence never spans a partial GPU wave,
    eliminating the "GPU tail" latency overhead.

    Returns R_keep_per_sample so the CascadeSTARRouter can condition
    its budget target on it directly.
    """

    def __init__(self, v_threshold: float = 0.15,
                 min_keep_ratio: float = 0.30,
                 wave_size: int = 32):
        super().__init__()
        self.v_threshold = v_threshold
        self.min_keep_ratio = min_keep_ratio
        self.wave_size = wave_size
        # Small learnable offset so the threshold adapts from data
        self.threshold_offset = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------
    def compute_ee_velocity(self, state_t, state_prev):
        """||state_t - state_{t-1}||_2  per batch item.

        Returns:
            v_ee: (B, 1)
        """
        if state_prev is None:
            return torch.zeros(state_t.shape[0], 1, device=state_t.device)
        return (state_t - state_prev).norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _wave_align(k: int, wave_size: int, n_min: int) -> int:
        """Round K up to next wave_size multiple, capped at n_min floor."""
        if wave_size <= 1:
            return k
        aligned = math.ceil(k / wave_size) * wave_size
        return max(aligned, n_min)

    # ------------------------------------------------------------------
    def forward(self, token_embeddings, token_mask, v_ee,
                attention_scores=None):
        """
        Args:
            token_embeddings : (B, N, D)
            token_mask       : (B, N) bool
            v_ee             : (B, 1) end-effector velocity
            attention_scores : (B, N) optional importance scores

        Returns:
            pruned_emb       : (B, K, D)   wave-aligned K
            pruned_mask      : (B, K) bool
            avg_keep_ratio   : scalar tensor
            r_keep_per_sample: (B, 1) per-sample ratio in [R_min, 1]
        """
        B, N, D = token_embeddings.shape
        eff_thresh = self.v_threshold + torch.tanh(self.threshold_offset) * 0.05

        # R_keep per sample
        r_keep_per_sample = torch.where(
            v_ee > eff_thresh,
            torch.clamp(
                self.min_keep_ratio
                + (1.0 - self.min_keep_ratio) * (eff_thresh / (v_ee + 1e-8)),
                min=self.min_keep_ratio, max=1.0),
            torch.ones_like(v_ee),
        )  # (B, 1)

        # Global K: use batch-max keep ratio (conservative but batchable)
        K_raw = max(int(N * self.min_keep_ratio), 1)
        K = self._wave_align(K_raw, self.wave_size, n_min=self.wave_size)
        K = min(K, N)

        if attention_scores is None:
            # Positional fallback: prefer earlier tokens
            attention_scores = torch.arange(N, 0, -1,
                                            device=token_embeddings.device,
                                            dtype=torch.float32)
            attention_scores = attention_scores.unsqueeze(0).expand(B, -1)

        _, top_idx = attention_scores.topk(K, dim=-1, sorted=False)
        top_idx_sorted, _ = top_idx.sort(dim=-1)

        pruned_emb = torch.gather(
            token_embeddings, 1,
            top_idx_sorted.unsqueeze(-1).expand(-1, -1, D))
        pruned_mask = torch.gather(token_mask, 1, top_idx_sorted)

        avg_keep_ratio = r_keep_per_sample.mean()
        return pruned_emb, pruned_mask, avg_keep_ratio, r_keep_per_sample


# ----------------------------------------------------------------------

class ActionAwarePRouter(nn.Module):
    """Action-aware router: predicts dynamic rho (layer budget) + per-layer gates.

    Ported from finetune_dynamic_P_Loss_CogKD.py — proven to produce actual skipping.

    Differences from CascadeSTARRouter:
    - Gate input: [hidden (D), e_view (1), delta_s (1), s_t (1)] = D+3
      r_keep is NOT in the gate net; instead it multiplies rho_target AFTER
      the forward pass (explicit DTP → DLS cascade coupling in the wrapper).
    - Returns theta_t as 4th value (used for logging / future per-layer analysis).

    DTP → DLS Cascade Relationship (proved externally in wrapper):
        rho_coupled = rho_net(kin) × r_keep
        → few tokens kept (low r_keep) → layer budget scales down proportionally
        → model must also skip more layers to maintain the compute budget
    """

    def __init__(self, hidden_dim: int, num_skippable_layers: int = 8,
                 rho_min: float = 0.0, rho_max: float = 0.5,
                 init_gain: float = 1.0, gate_bias_init: float = -2.0):
        super().__init__()
        self.num_skippable_layers = num_skippable_layers
        self.rho_min = rho_min
        self.rho_max = rho_max

        # Gate net: D+3 → hidden//2 → L   (e_view + delta_s + s_t, no r_keep)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_skippable_layers),
        )
        # rho_net: kinematic features (3) → scalar budget target in [rho_min, rho_max]
        self.rho_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # theta_net: per-layer adaptive threshold — each layer learns its own sensitivity
        self.theta_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_skippable_layers),
        )

        # Apply init_gain to all output heads
        for head in (self.gate_net[-1], self.rho_net[-1], self.theta_net[-1]):
            with torch.no_grad():
                head.weight.mul_(init_gain)

        # gate_bias_init=-2.0 → σ(-2)≈0.12, gates start mostly OFF
        nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)

        # theta_net biases spread [−1, +1]: each layer starts with a different threshold
        with torch.no_grad():
            bias_spread = torch.linspace(-1.0, 1.0, num_skippable_layers)
            self.theta_net[-1].bias.copy_(bias_spread)

    # ------------------------------------------------------------------
    def forward(self, hidden_pooled, e_view, delta_s_norm,
                kin_features, s_t,
                tau: float = 1.0, hard: bool = False):
        """
        Args:
            hidden_pooled : (B, D)
            e_view        : (B, 1) visual entropy proxy
            delta_s_norm  : (B, 1) state-change magnitude
            kin_features  : (B, 3) [m_norm, j_norm, s_t]
            s_t           : (B, 1) fused kinematic sensitivity
            tau           : Gumbel-Softmax temperature
            hard          : use hard threshold (inference)

        Returns:
            gates      : (B, num_skippable_layers)
            gate_loss  : scalar mean gate activation (cost penalty)
            rho_target : (B, 1) raw budget from rho_net (before cascade coupling)
            theta_t    : (B, L) per-layer thresholds (for logging)
        """
        # Dynamic budget target from kinematic features only
        rho_logits = self.rho_net(kin_features)           # (B, 1)
        rho_target = (self.rho_min
                      + (self.rho_max - self.rho_min) * torch.sigmoid(rho_logits))

        # Per-layer kinematic thresholds
        theta_t = torch.sigmoid(self.theta_net(kin_features))  # (B, L)

        # Gate logits: net output minus per-layer threshold
        x = torch.cat([hidden_pooled, e_view, delta_s_norm, s_t], dim=-1)
        logits = self.gate_net(x) - theta_t

        if hard or not self.training:
            gates = (torch.sigmoid(logits / max(tau, 1e-4)) > 0.5).float()
        else:
            logits_2c = torch.stack(
                [torch.zeros_like(logits), logits], dim=-1)  # (B, L, 2)
            gumbel_out = F.gumbel_softmax(logits_2c, tau=tau,
                                          hard=False, dim=-1)
            gates = gumbel_out[..., 1]

        # gate_probs: smooth sigmoid probabilities (no Gumbel noise).
        # Used for constraint losses so gradients flow even when logits are
        # extreme negative (Gumbel gates ≈ 0 with near-zero grad at low tau).
        gate_probs = torch.sigmoid(logits)
        gate_loss = gates.mean()
        return gates, gate_loss, rho_target, theta_t, gate_probs


# ----------------------------------------------------------------------

class ToIAwareCogKDHelper:
    """CogKD with Tokens-of-Interest masking.

    Teacher: frozen smolvla_base loaded at script start (never mutated).
    The same teacher is used throughout all 3 phases; only the KD loss
    weight activates in Phase 3.

    ToI mask: top-toi_ratio tokens by cosine similarity to mean embedding.
    Only ToI tokens contribute to MSE and KL terms, avoiding waste on
    background regions.

    Loss:
        L_CogKD = (1-lambda)*MSE(h_S*M, h_T*M) + lambda*KL(P_S||P_T)
    """

    def __init__(self, temperature: float = 2.0,
                 toi_ratio: float = 0.30,
                 toi_min_tokens: int = 4):
        self.temperature = temperature
        self.toi_ratio = toi_ratio
        self.toi_min_tokens = toi_min_tokens
        self.teacher_model = None   # set via set_teacher()

    def set_teacher(self, smolvla_policy):
        """Freeze a policy as permanent teacher."""
        self.teacher_model = smolvla_policy
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def compute_toi_mask(self, teacher_prefix: torch.Tensor) -> torch.Tensor:
        """Binary (B, L, 1) mask — 1 for the top-toi_ratio tokens."""
        B, L, _ = teacher_prefix.shape
        cog = teacher_prefix.mean(dim=1, keepdim=True)          # (B, 1, D)
        sims = F.cosine_similarity(teacher_prefix, cog, dim=-1) # (B, L)
        k = max(self.toi_min_tokens, int(L * self.toi_ratio))
        k = min(k, L)
        topk_idx = sims.topk(k, dim=-1).indices
        mask = torch.zeros(B, L, device=teacher_prefix.device)
        mask.scatter_(1, topk_idx, 1.0)
        return mask.unsqueeze(-1)  # (B, L, 1)

    # ------------------------------------------------------------------
    def compute_loss(self, student_prefix: torch.Tensor,
                     teacher_prefix: torch.Tensor,
                     cogkd_lambda: float = 0.30) -> tuple:
        """
        Args:
            student_prefix: (B, L_s, D)
            teacher_prefix: (B, L_t, D)

        Returns:
            loss, mse_term, kl_term
        """
        L = min(student_prefix.shape[1], teacher_prefix.shape[1])
        sp = student_prefix[:, :L]
        tp = teacher_prefix[:, :L]
        mask = self.compute_toi_mask(tp)          # (B, L, 1)

        mse = F.mse_loss(sp * mask, tp * mask)
        T = self.temperature
        sl = F.log_softmax(sp / T, dim=-1)
        tp_ = F.softmax(tp / T, dim=-1)
        kl = F.kl_div(sl, tp_, reduction="batchmean") * (T ** 2)
        loss = (1.0 - cogkd_lambda) * mse + cogkd_lambda * kl
        return loss, mse, kl


# ----------------------------------------------------------------------

class LoRASPAdapter(nn.Module):
    """LoRA-SP: Select-Prune dynamic rank adaptation.

    deltaW(x) = U * diag(s(x)) * V
    Spectral concentration loss encourages sparse effective rank.
    """

    def __init__(self, in_features: int, out_features: int,
                 max_rank: int = 128, energy_threshold: float = 0.9,
                 alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.energy_threshold = energy_threshold
        self.scaling = alpha / max_rank

        self.U = nn.Parameter(torch.randn(out_features, max_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)
        self.router = nn.Sequential(
            nn.Linear(in_features, max_rank),
            nn.Sigmoid(),
        )

    def compute_spectral_loss(self, scores: torch.Tensor) -> torch.Tensor:
        sq = scores ** 2
        prob = sq / (sq.sum(dim=-1, keepdim=True) + 1e-8)
        return -(prob * (prob + 1e-8).log()).sum(dim=-1).mean()  # neg-entropy

    def forward(self, x: torch.Tensor,
                return_spec_loss: bool = False):
        orig = x.shape
        xf = x.reshape(-1, self.in_features)
        scores = self.router(xf.detach())
        Vx = F.linear(xf, self.V)
        SVx = Vx * scores * self.scaling
        delta = F.linear(SVx, self.U)
        delta = delta.reshape(*orig[:-1], self.out_features)
        if return_spec_loss:
            return delta, self.compute_spectral_loss(scores)
        return delta


# ----------------------------------------------------------------------

class SnapFlowTrainer:
    """SnapFlow 1-NFE self-distillation (identical to base script)."""

    def __init__(self, num_teacher_steps: int = 10):
        self.num_teacher_steps = num_teacher_steps
        self.teacher_model = None

    def create_teacher(self, model):
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute_teacher_target(self, teacher_flow_model,
                                prefix_embs, prefix_pad_masks,
                                prefix_att_masks, noise,
                                chunk_size, action_out_proj):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        bsize = noise.shape[0]
        device = noise.device
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, pkv = teacher_flow_model.vlm_with_expert.forward(
            attention_mask=att_2d, position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True, fill_kv_cache=True,
        )
        dt = -1.0 / self.num_teacher_steps
        x_t = noise.clone()
        for step in range(self.num_teacher_steps):
            t = 1.0 + step * dt
            t_ten = torch.tensor(t, dtype=torch.float32,
                                 device=device).expand(bsize)
            v_t = teacher_flow_model.denoise_step(
                x_t=x_t, prefix_pad_masks=prefix_pad_masks,
                past_key_values=pkv, timestep=t_ten,
            )
            x_t = x_t + dt * v_t
        return x_t

    def compute_snapflow_loss(self, student_velocity, x_t,
                               teacher_x0, t):
        t_exp = t[:, None, None]
        u_shortcut = (x_t - teacher_x0) / (t_exp + 1e-6)
        return F.mse_loss(student_velocity, u_shortcut)


# =====================================================================
#  MAIN WRAPPER
# =====================================================================

class HierarchicalSmolVLAWrapper(nn.Module):
    """Orchestrates the DTP → DLS cascade + CogKD + SnapFlow.

    Forward order:
        1. Embed images  → all_img_emb
        2. HierarchicalADPPruner  → R_keep (wave-aligned)
        3. CascadeSTARRouter      → gates conditioned on R_keep
        4. Task loss via flow_model.forward()  (no structural change)
        5. LoRA-SP spectral loss  (Phase 2+)
        6. Budget loss:  ||mean(gates) - rho_target||  (Phase 2+)
        7. CogKD loss with ToI mask             (Phase 3)
        8. SnapFlow loss                        (Phase 3)
    """

    def __init__(self, smolvla_policy, dyn_cfg: dict):
        super().__init__()
        self.policy = smolvla_policy
        self.flow_model = smolvla_policy.model
        self.cfg = dyn_cfg

        vlm = self.flow_model.vlm_with_expert
        vlm_hidden = vlm.config.text_config.hidden_size
        num_vlm_layers = vlm.num_vlm_layers

        # Module 1 — HierarchicalADPPruner
        self.adp = HierarchicalADPPruner(
            v_threshold=dyn_cfg["adp_velocity_threshold"],
            min_keep_ratio=dyn_cfg["adp_min_keep_ratio"],
            wave_size=dyn_cfg.get("gpu_wave_size", 32),
        )

        # Module 2 — ActionAwarePRouter (ported from finetune_dynamic_P_Loss_CogKD, proven working)
        # r_keep from ADP is NOT an input to gate_net; instead it multiplies rho_target
        # after the forward pass to create explicit DTP → DLS cascade coupling.
        self.action_router = ActionAwarePRouter(
            hidden_dim=vlm_hidden,
            num_skippable_layers=dyn_cfg["num_skippable_layers"],
            rho_min=dyn_cfg.get("rho_target_min", 0.0),
            rho_max=dyn_cfg.get("rho_target_max", 0.5),
            init_gain=dyn_cfg.get("router_init_gain", 1.0),
            gate_bias_init=dyn_cfg.get("router_gate_bias_init", -2.0),
        )

        # Module 3 — LoRA-SP adapters on skippable VLM layers
        self.lora_adapters = nn.ModuleDict()
        vlm_layers = vlm.get_vlm_model().text_model.layers
        for li in range(dyn_cfg["num_fixed_layers"], num_vlm_layers):
            attn = vlm_layers[li].self_attn
            key = f"layer_{li}"
            for proj_name, proj in [("q", attn.q_proj), ("k", attn.k_proj),
                                     ("v", attn.v_proj), ("o", attn.o_proj)]:
                self.lora_adapters[f"{key}_{proj_name}"] = LoRASPAdapter(
                    proj.in_features, proj.out_features,
                    max_rank=dyn_cfg["lora_max_rank"],
                    energy_threshold=dyn_cfg["lora_energy_threshold"],
                )

        # Module 4 — ToI-aware CogKD helper (teacher set externally)
        self.cogkd = ToIAwareCogKDHelper(
            temperature=dyn_cfg["cogkd_temperature"],
            toi_ratio=dyn_cfg["toi_ratio"],
            toi_min_tokens=dyn_cfg.get("toi_min_tokens", 4),
        )

        # Module 5 — SnapFlow
        self.snap = SnapFlowTrainer(
            num_teacher_steps=dyn_cfg["snap_teacher_steps"],
        )

        # Kinematic state tracking (current, prev, prev2 for jerk; prev_gates for flip loss)
        self._prev_state:  torch.Tensor | None = None
        self._prev_state2: torch.Tensor | None = None
        self._prev_gates:  torch.Tensor | None = None
        self.kinematic_state_dim: int = dyn_cfg.get("kinematic_state_dim", 16)

        # Training telemetry
        self.skip_stats: list[float] = []
        self.token_stats: list[float] = []

    # ------------------------------------------------------------------
    def reset_episode(self):
        self._prev_state  = None
        self._prev_state2 = None
        self._prev_gates  = None
        self.policy.reset()

    # ------------------------------------------------------------------
    def _compute_kinematics(self, state: torch.Tensor,
                             state_prev: torch.Tensor,
                             state_prev2: torch.Tensor,
                             all_img_emb: torch.Tensor | None):
        """Compute kinematic features for CascadeSTAR.

        Returns:
            e_view       : (B, 1) visual entropy
            delta_s_norm : (B, 1) velocity magnitude
            s_t          : (B, 1) fused kinematic sensitivity
            kin_features : (B, 3) [m_norm, j_norm, s_t]
        """
        B, device = state.shape[0], state.device

        e_view = (all_img_emb.var(dim=1).mean(dim=-1, keepdim=True)
                  if all_img_emb is not None
                  else torch.zeros(B, 1, device=device))

        kin_dim = min(state.shape[-1], self.kinematic_state_dim)
        s_cur  = state[:, :kin_dim]
        s_prv  = state_prev[:, :kin_dim]
        s_prv2 = state_prev2[:, :kin_dim]

        delta        = s_cur - s_prv
        delta_s_norm = delta.norm(dim=-1, keepdim=True)           # (B, 1)

        # Inverse velocity (m) and jerk (j)
        m_raw = 1.0 / (delta_s_norm + 1e-6)
        j_raw = (s_cur - 2.0 * s_prv + s_prv2).norm(dim=-1, keepdim=True)

        # log1p + tanh normalization: bounded [0,1), smooth, no batch-size dependency.
        # Replaces batch-norm which degenerates with static robot state or batch=1.
        scale_m = self.cfg.get("kinematic_scale_m", 5.0)
        scale_j = self.cfg.get("kinematic_scale_j", 10.0)
        m_norm = torch.tanh(torch.log1p(m_raw * scale_m))   # slow=high, fast=low
        j_norm = torch.tanh(torch.log1p(j_raw * scale_j))   # high jerk=high

        k_lambda = self.cfg.get("kinematic_lambda", 0.6)
        s_t = torch.relu(k_lambda * m_norm + (1.0 - k_lambda) * j_norm)

        kin_features = torch.cat([m_norm, j_norm, s_t], dim=-1)   # (B, 3)
        return e_view, delta_s_norm, s_t, kin_features

    # ------------------------------------------------------------------
    def _register_soft_skip_hooks(self, gates):
        """Differentiable soft layer-skip applied DURING the task forward.

        For each skippable VLM layer j (global index num_fixed+j):
            out = g·layer_out + (1-g)·layer_in,   g = gates[:, j]  (per-sample)

        - Training: gates are soft (Gumbel) → the action expert learns to predict
          under skipping AND gradient flows to the router from the task loss.
        - Eval: gates are hard 0/1 → g=0 is a true identity skip (real FLOPs cut),
          g=1 is the full layer; both are endpoints the model trained on.

        This is what makes DLS a TRAINED mechanism instead of an eval-only side-car.
        Returns hook handles to remove right after flow_model.forward().
        """
        vlm_layers = self.flow_model.vlm_with_expert.get_vlm_model().text_model.layers
        num_fixed = self.cfg["num_fixed_layers"]
        handles = []
        for j in range(self.cfg["num_skippable_layers"]):
            li = num_fixed + j
            if li >= len(vlm_layers):
                break

            def _make(gj):
                def _hook(module, inp, out):
                    h_in = inp[0]
                    h_out = out[0] if isinstance(out, tuple) else out
                    # only blend when shapes line up with the per-sample gate
                    if (not torch.is_tensor(h_out) or not torch.is_tensor(h_in)
                            or h_out.shape != h_in.shape
                            or h_out.shape[0] != gj.shape[0]):
                        return out
                    g = gj.to(h_out.dtype).view(-1, *([1] * (h_out.dim() - 1)))
                    blended = g * h_out + (1.0 - g) * h_in
                    if isinstance(out, tuple):
                        return (blended,) + tuple(out[1:])
                    return blended
                return _hook

            handles.append(vlm_layers[li].register_forward_hook(_make(gates[:, j])))
        return handles

    # ------------------------------------------------------------------
    def _patch_adp_token_mask(self, r_keep_per_sample):
        """ADP (DTP) token pruning applied DURING the forward.

        Monkeypatches vlm_with_expert.embed_image so only the top-K image tokens
        are kept (K = round(keep_ratio · N), positional importance — matches the
        ADP module fallback); the rest are zeroed. This makes the action expert
        learn to act under token pruning, realising the horizontal (token) half of
        the DTP→DLS cascade. Train: soft via keep_ratio; eval mirrors it (hard).
        Returns a restore() callable to undo the patch after the forward.
        """
        vlm = self.flow_model.vlm_with_expert
        orig = vlm.embed_image
        keep = float(r_keep_per_sample.mean().clamp(0.05, 1.0).item())

        def patched(img):
            emb = orig(img)                       # (B, N, D)
            N = emb.shape[1]
            K = max(int(round(keep * N)), 1)
            if K < N:
                m = torch.zeros(1, N, 1, device=emb.device, dtype=emb.dtype)
                m[:, :K, :] = 1.0
                emb = emb * m
            return emb

        vlm.embed_image = patched
        self.token_stats.append(keep)
        return lambda: setattr(vlm, "embed_image", orig)

    # ------------------------------------------------------------------
    def forward_with_skip(self, batch: dict, tau: float = 1.0,
                           enable_skip: bool = True,
                           enable_adp: bool = False,
                           enable_lora: bool = False,
                           enable_cogkd: bool = False,
                           enable_snap: bool = False,
                           noise=None, time_val=None,
                           lambda_budget_override=None,
                           lambda_skip_floor_override=None):
        """Hierarchical forward pass with phase-gated module activation.

        Returns:
            total_loss : scalar tensor
            loss_dict  : dict with all component losses for logging
        """
        flow_model = self.flow_model
        device = next(self.parameters()).device

        # ---- Prepare inputs ------------------------------------------
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        B = state.shape[0]
        original_action_dim = self.policy.config.action_feature.shape[0]

        # ---- Visual embeddings (no grad — used for routing only) -----
        with torch.no_grad():
            img_emb_list = [flow_model.vlm_with_expert.embed_image(img)
                            for img in images]
        all_img_emb = (torch.cat(img_emb_list, dim=1)
                       if img_emb_list else None)

        # ---- State prev/prev2 from batch or tracked fallback ---------
        # When the dataset is loaded with delta_timestamps, observation.state has
        # shape [B, 3, state_dim] ordered [-2dt, -dt, 0] (ascending timestamps).
        # prepare_state already extracts the current state via [:, -1, :].
        # We extract t-1 and t-2 here for true temporal velocity computation.
        _obs_state_raw = batch.get("observation.state")
        state_prev  = batch.get("observation.state_prev")
        state_prev2 = batch.get("observation.state_prev2")
        if _obs_state_raw is not None and _obs_state_raw.dim() == 3:
            # delta_timestamps mode: index 0=t-2, 1=t-1, 2=current (ascending)
            state_prev  = _obs_state_raw[:, 1, :]   # t-1
            state_prev2 = _obs_state_raw[:, 0, :]   # t-2
        if state_prev is None:
            tracked = self._prev_state
            state_prev = (tracked.to(device)
                          if tracked is not None and tracked.shape[0] == B
                          else torch.zeros_like(state))
        if state_prev2 is None:
            tracked2 = self._prev_state2
            state_prev2 = (tracked2.to(device)
                           if tracked2 is not None and tracked2.shape[0] == B
                           else torch.zeros_like(state))

        # Pad / trim to same length as state
        def _match_len(t, ref):
            if t.shape[-1] == ref.shape[-1]:
                return t
            if t.shape[-1] < ref.shape[-1]:
                return F.pad(t, (0, ref.shape[-1] - t.shape[-1]))
            return t[:, :ref.shape[-1]]
        state_prev  = _match_len(state_prev,  state)
        state_prev2 = _match_len(state_prev2, state)

        # Update tracked states for next call (fallback path)
        self._prev_state2 = (self._prev_state.detach().clone()
                             if self._prev_state is not None
                             else state_prev.detach().clone())
        self._prev_state = state.detach().clone()

        # ---- Context + kinematic features ----------------------------
        e_view, delta_s_norm, s_t, kin_features = self._compute_kinematics(
            state, state_prev, state_prev2, all_img_emb)

        # ---- Step 1: ADP — compute R_keep (wave-aligned) -------------
        v_ee = delta_s_norm   # velocity magnitude already computed above (B,1)

        avg_keep_ratio = torch.tensor(1.0, device=device)
        r_keep_per_sample = torch.ones(B, 1, device=device)
        if enable_adp and all_img_emb is not None:
            _, _, avg_keep_ratio, r_keep_per_sample = self.adp(
                all_img_emb,
                torch.ones(B, all_img_emb.shape[1],
                           dtype=torch.bool, device=device),
                v_ee,
            )
        self.token_stats.append(avg_keep_ratio.item()
                                 if isinstance(avg_keep_ratio, torch.Tensor)
                                 else float(avg_keep_ratio))

        # ---- Step 2: ActionAwarePRouter — gates from kinematics, rho coupled to DTP ---
        gates = torch.ones(B, self.cfg["num_skippable_layers"], device=device)
        gate_loss = torch.tensor(0.0, device=device)
        rho_target = torch.zeros(B, 1, device=device)
        gate_probs = torch.ones(B, self.cfg["num_skippable_layers"], device=device)
        if enable_skip:
            expected_D = self.action_router.gate_net[0].in_features - 3
            hidden_pooled = (all_img_emb.mean(dim=1) if all_img_emb is not None
                             else torch.zeros(B, expected_D, device=device))
            if hidden_pooled.shape[-1] != expected_D:
                hidden_pooled = F.adaptive_avg_pool1d(
                    hidden_pooled.unsqueeze(1), expected_D).squeeze(1)
            gates, gate_loss, rho_target, _theta_t, gate_probs = self.action_router(
                hidden_pooled, e_view, delta_s_norm,
                kin_features, s_t, tau=tau,
            )
            # ---- DTP → DLS CASCADE COUPLING ----
            # rho_target from rho_net reflects kinematic complexity alone.
            # Multiplying by r_keep_per_sample makes the layer budget proportional
            # to the token budget: pruning 70% of tokens → also reduce active layers.
            #   r_keep=1.0 → rho_coupled = rho_target (no change, full token stream)
            #   r_keep=0.3 → rho_coupled = 0.3*rho_target (must skip proportionally more)
            if enable_adp:
                rho_target = rho_target * r_keep_per_sample
            self.skip_stats.append(gate_probs.mean().item())  # gate_probs = sigmoid(logits), smooth

        # ---- Step 3: Task loss via flow_model — WITH differentiable soft layer-skip ----
        # Register soft-skip hooks so the task forward actually runs under the gates'
        # decisions (gate·layer_out + (1-gate)·layer_in). This trains the action
        # expert to predict under skipping and lets the task loss supervise the gates.
        skip_handles = self._register_soft_skip_hooks(gates) if enable_skip else []
        adp_restore = (self._patch_adp_token_mask(r_keep_per_sample)
                       if enable_adp else None)
        try:
            losses = flow_model.forward(
                images, img_masks, lang_tokens, lang_masks, state, actions,
                noise=noise, time=time_val,
            )
        finally:
            for _h in skip_handles:
                _h.remove()
            if adp_restore is not None:
                adp_restore()
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        task_loss = losses.mean()

        # ---- Step 4: Budget loss — ||mean(gates) - rho_target||_2 ---
        # NOTE: rho_target is NOT detached here so rho_net receives gradients from
        # budget_loss. To prevent the trivial collapse (rho_net just mirrors gate_ratio),
        # we add an explicit kinematic supervision loss (rho_supervision_loss below)
        # that pins rho_net to the kinematic signal.
        budget_loss       = torch.tensor(0.0, device=device)
        rho_supervision_loss = torch.tensor(0.0, device=device)
        if enable_skip:
            gate_ratio = gates.mean(dim=-1, keepdim=True)  # (B, 1)
            budget_loss = F.mse_loss(gate_ratio, rho_target)  # gradient flows to rho_net

            # L_rho_supervision: explicitly train rho_net to be kinematic-sensitive.
            # Target: rho_min when s_t≈0 (fast/easy), rho_max when s_t≈1 (slow/hard).
            # This is the primary signal that makes layer skipping STATE-DEPENDENT.
            _rho_min = self.cfg.get("rho_target_min", 0.15)
            _rho_max = self.cfg.get("rho_target_max", 0.55)
            rho_ideal = _rho_min + (_rho_max - _rho_min) * s_t  # (B, 1)
            rho_supervision_loss = F.mse_loss(rho_target, rho_ideal.detach())

        # ---- Step 4b: Diversity + skip-forcing losses ---------------
        temporal_entropy_loss = torch.tensor(0.0, device=device)
        rho_spread_loss       = torch.tensor(0.0, device=device)
        gate_flip_loss        = torch.tensor(0.0, device=device)
        always_on_loss        = torch.tensor(0.0, device=device)
        skip_floor_loss       = torch.tensor(0.0, device=device)
        if enable_skip:
            # Use gate_probs (smooth sigmoid) for ALL constraint losses so gradients
            # remain non-zero even when Gumbel gates collapse to near-binary at low tau.
            # gates (Gumbel sample) is kept only for gate_loss (cost penalty) and
            # budget_loss (dynamic target tracking).
            layer_mean_on = gate_probs.mean(dim=0)   # (L,)  per-layer ON rate across batch

            # L_temporal_entropy: maximise per-layer H → each layer toggles ~50%
            eps = 1e-6
            layer_H = -(layer_mean_on * (layer_mean_on + eps).log()
                        + (1.0 - layer_mean_on) * (1.0 - layer_mean_on + eps).log())
            temporal_entropy_loss = -layer_H.mean()   # minimise → maximise H

            # L_always_on (PRIMARY FORCE): per-layer ON ceiling
            # Any layer whose mean gate exceeds max_layer_on_rate is penalised quadratically.
            # This is the dominant force that breaks the all-ON equilibrium.
            max_on = self.cfg.get("max_layer_on_rate", 0.60)
            always_on_loss = F.relu(layer_mean_on - max_on).pow(2).mean()

            # L_skip_floor: global skip floor
            # relu → zero gradient once actual skip ≥ target (no over-pushing)
            skip_target = self.cfg.get("skip_target_ratio", 0.40)
            # Use gate_probs (not Gumbel-sampled gates) for smooth gradients
            actual_skip = 1.0 - gate_probs.mean()
            skip_floor_loss = F.relu(
                torch.tensor(skip_target, device=device, dtype=gates.dtype)
                - actual_skip
            ).pow(2)

            # L_rho_spread: rho_target must vary across batch
            rho_std = rho_target.std(dim=0).mean()
            min_rho_std_val = self.cfg.get("min_rho_std", 0.05)
            rho_spread_loss = torch.relu(
                torch.tensor(min_rho_std_val, device=device) - rho_std)

            # L_gate_flip: reward gate state changes vs previous optimizer step
            if (self._prev_gates is not None
                    and self._prev_gates.shape[0] == gates.shape[0]):
                gate_flip_loss = -((gates - self._prev_gates).abs().mean())
            self._prev_gates = gates.detach()

            # ---- NEW: Losses to prevent the all-OFF trivial solution ----

            # L_skip_ceiling: model may NOT skip more than max_skip_ratio.
            # Closes the loophole where skip=1.0 satisfies always_on AND
            # skip_floor with zero penalty (both losses are zero when all OFF).
            max_skip = self.cfg.get("max_skip_ratio", 0.60)
            skip_ceiling_loss = F.relu(actual_skip - max_skip).pow(2)

            # L_layer_min_on: each skippable layer MUST be active >= min_layer_on_rate.
            # Prevents individual layers from permanently switching off.
            min_on_rate = self.cfg.get("min_layer_on_rate", 0.30)
            layer_min_on_loss = F.relu(
                torch.tensor(min_on_rate, device=device, dtype=gate_probs.dtype)
                - layer_mean_on
            ).pow(2).mean()

            # L_gate_diversity: different inputs in the same batch MUST produce
            # different gate decisions. Penalises low per-layer variance across batch.
            # With B=2 this directly forces the 2 samples to disagree on ≥ 1 layer.
            min_gate_var = self.cfg.get("min_gate_var", 0.10)
            gate_var_per_layer = gate_probs.var(dim=0)      # (L,) smooth gradient
            gate_diversity_loss = F.relu(
                torch.tensor(min_gate_var, device=device, dtype=gate_probs.dtype)
                - gate_var_per_layer
            ).mean()

            # L_visual_coupling: visual complexity (e_view) should drive gate rate.
            # High visual entropy → more layers active (complex scene → more compute).
            # This makes gate decisions action/scene-aware rather than static.
            e_view_mu  = e_view.mean()
            e_view_std = e_view.std().clamp(min=1e-4)
            vis_norm   = (e_view - e_view_mu) / e_view_std       # (B, 1) z-score
            target_gate_rate = torch.sigmoid(vis_norm)                 # → (0, 1)
            actual_gate_rate = gate_probs.mean(dim=-1, keepdim=True)   # (B, 1) smooth gradient
            visual_coupling_loss = F.mse_loss(actual_gate_rate, target_gate_rate.detach())
        else:
            skip_ceiling_loss    = torch.tensor(0.0, device=device)
            layer_min_on_loss    = torch.tensor(0.0, device=device)
            gate_diversity_loss  = torch.tensor(0.0, device=device)
            visual_coupling_loss = torch.tensor(0.0, device=device)

        # ---- Step 5: LoRA-SP spectral loss ---------------------------
        spec_loss = torch.tensor(0.0, device=device)
        lora_n = 0
        if enable_lora and all_img_emb is not None:
            lora_input = all_img_emb.detach()
            for adapter in self.lora_adapters.values():
                inp = (F.adaptive_avg_pool1d(
                           lora_input.permute(0, 2, 1),
                           adapter.in_features).permute(0, 2, 1)
                       if lora_input.shape[-1] != adapter.in_features
                       else lora_input)
                _, s = adapter(inp, return_spec_loss=True)
                spec_loss = spec_loss + s
                lora_n += 1
            if lora_n:
                spec_loss = spec_loss / lora_n

        # ---- Step 6: CogKD (ToI) -------------------------------------
        cogkd_loss = torch.tensor(0.0, device=device)
        if enable_cogkd and self.cogkd.teacher_model is not None:
            student_prefix, _, _ = flow_model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state)
            with torch.no_grad():
                teacher_prefix, _, _ = self.cogkd.teacher_model.model.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks, state=state)
            cogkd_loss, _, _ = self.cogkd.compute_loss(
                student_prefix, teacher_prefix,
                cogkd_lambda=self.cfg["cogkd_lambda"],
            )

        # ---- Step 7: SnapFlow ----------------------------------------
        snap_loss = torch.tensor(0.0, device=device)
        if enable_snap and self.snap.teacher_model is not None:
            prefix_embs, pfx_pad, pfx_att = flow_model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state)
            if noise is None:
                noise_sf = flow_model.sample_noise(actions.shape, actions.device)
            else:
                noise_sf = noise
            teacher_x0 = self.snap.compute_teacher_target(
                self.snap.teacher_model,
                prefix_embs.detach(), pfx_pad.detach(), pfx_att.detach(),
                noise_sf, flow_model.config.chunk_size,
                flow_model.action_out_proj,
            )
            teacher_x0 = teacher_x0[:, :, :original_action_dim]

            t_val = (time_val if time_val is not None
                     else flow_model.sample_time(B, actions.device))
            t_exp = t_val[:, None, None]
            a_clean = actions[:, :, :original_action_dim]
            n_slice = noise_sf[:, :, :original_action_dim]
            x_t = t_exp * n_slice + (1 - t_exp) * a_clean
            student_v = n_slice - a_clean  # flow matching target
            snap_loss = self.snap.compute_snapflow_loss(
                student_v, x_t, teacher_x0, t_val)

        # ---- Combine losses  -----------------------------------------
        λ_gate      = self.cfg["lambda_gate"]
        λ_budget    = (lambda_budget_override if lambda_budget_override is not None
                       else self.cfg["lambda_budget"])
        λ_spec      = self.cfg["lambda_spec"]
        λ_distill   = self.cfg["lambda_distill"]
        λ_snap      = self.cfg["lambda_snap"]
        λ_cost      = self.cfg["lambda_cost"]
        λ_tent      = self.cfg.get("lambda_temporal_entropy", 0.5)
        λ_rspread   = self.cfg.get("lambda_rho_spread", 0.10)
        λ_flip      = self.cfg.get("lambda_gate_flip", 0.15)
        λ_always_on = self.cfg.get("lambda_always_on", 2.0)
        λ_skip_floor= (lambda_skip_floor_override if lambda_skip_floor_override is not None
                       else self.cfg.get("lambda_skip_floor", 1.0))
        # New constraint lambdas (fix all-OFF trivial solution + action-awareness)
        λ_skip_ceil = self.cfg.get("lambda_skip_ceiling",   3.0)
        λ_min_on    = self.cfg.get("lambda_min_layer_on",   3.0)
        λ_diversity = self.cfg.get("lambda_gate_diversity", 2.0)
        λ_vis_cpl   = self.cfg.get("lambda_visual_coupling", 1.0)
        λ_rho_sup   = self.cfg.get("lambda_rho_supervision", 3.0)

        total = task_loss
        if enable_skip:
            total = total + λ_gate       * gate_loss          # penalise fraction ON
            total = total + λ_always_on  * always_on_loss     # per-layer ceiling
            total = total + λ_skip_floor * skip_floor_loss    # global skip floor
            total = total + λ_skip_ceil  * skip_ceiling_loss  # global skip ceiling
            total = total + λ_min_on     * layer_min_on_loss  # per-layer min ON
            total = total + λ_diversity  * gate_diversity_loss  # batch diversity
            total = total + λ_vis_cpl    * visual_coupling_loss # visual coupling
            total = total + λ_tent       * temporal_entropy_loss
            total = total + λ_rspread    * rho_spread_loss
            total = total + λ_flip       * gate_flip_loss
            total = total + λ_rho_sup    * rho_supervision_loss  # rho_net kinematic pin
        if enable_skip and enable_adp:
            total = total + λ_budget * budget_loss
        if enable_adp:
            total = total + λ_cost * avg_keep_ratio
        if enable_lora and lora_n:
            total = total + λ_spec * spec_loss
        if enable_cogkd:
            total = total + λ_distill * cogkd_loss
        if enable_snap:
            total = total + λ_snap * snap_loss

        num_vlm   = self.flow_model.vlm_with_expert.num_vlm_layers
        num_fixed = self.cfg["num_fixed_layers"]
        loss_dict = {
            "total_loss":   total.item(),
            "task_loss":    task_loss.item(),
            "gate_loss":    gate_loss.item(),
            "budget_loss":  budget_loss.item(),
            "spec_loss":    spec_loss.item() if isinstance(spec_loss, torch.Tensor) else spec_loss,
            "cogkd_loss":   cogkd_loss.item(),
            "snap_loss":    snap_loss.item(),
            "temporal_entropy_loss": temporal_entropy_loss.item(),
            "rho_spread_loss":       rho_spread_loss.item(),
            "gate_flip_loss":        gate_flip_loss.item(),
            "always_on_loss":        always_on_loss.item(),
            "skip_floor_loss":       skip_floor_loss.item(),
            "skip_ceiling_loss":     skip_ceiling_loss.item(),
            "layer_min_on_loss":     layer_min_on_loss.item(),
            "gate_diversity_loss":   gate_diversity_loss.item(),
            "visual_coupling_loss":  visual_coupling_loss.item(),
            "rho_supervision_loss":  rho_supervision_loss.item(),
            "token_keep_ratio": avg_keep_ratio.item() if isinstance(avg_keep_ratio, torch.Tensor) else avg_keep_ratio,
            "avg_skip_ratio":   1.0 - gate_probs.mean().item() if enable_skip else 0.0,
            "active_layers":    (gates.sum(dim=-1).mean().item() + num_fixed) if enable_skip else num_vlm,
            "rho_target":       rho_target.mean().item() if enable_skip else 0.0,
            "rho_std":          rho_target.std().item() if enable_skip else 0.0,
            "layer_entropy_mean": (-temporal_entropy_loss.item()) if enable_skip else 0.0,
            "layer_on_max":     gates.mean(dim=0).max().item() if enable_skip else 1.0,
        }
        return total, loss_dict


# =====================================================================
#  DATASET LOADING
# =====================================================================
print("=" * 70)
print("  Loading Dataset")
print("=" * 70)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load train and test episode IDs from JSON split files.
# Prefer *_train_episodes.json / *_eval_episodes.json (per-task split built by
# build_libero10_splits.py: last 5 eps/task → test, rest → train).
import json as _json

_train_episode_ids: list[int] | None = ds_cfg.get("episodes", None)
_test_episode_ids: list[int] = []

if _train_episode_ids is None:
    _train_json = Path("data/datasets") / f"{DATASET_KEY}_train_episodes.json"
    _eval_json  = Path("data/datasets") / f"{DATASET_KEY}_eval_episodes.json"
    _fallback_json = Path("data/datasets") / f"{DATASET_KEY}_episodes.json"
    if _train_json.exists():
        with open(_train_json) as _f:
            _train_episode_ids = _json.load(_f).get("episode_ids", [])
        print(f"  Auto-loaded {len(_train_episode_ids)} TRAIN episodes from {_train_json}")
    elif _fallback_json.exists():
        with open(_fallback_json) as _f:
            _train_episode_ids = _json.load(_f).get("episode_ids", [])
        print(f"  Auto-loaded {len(_train_episode_ids)} episodes from {_fallback_json}")
    if _eval_json.exists():
        with open(_eval_json) as _f:
            _test_episode_ids = _json.load(_f).get("episode_ids", [])
        if _test_episode_ids:
            print(f"  Auto-loaded {len(_test_episode_ids)}  TEST  episodes from {_eval_json}")

# Load dataset with ALL episodes (train + test) so frame normalization is consistent.
_test_set = set(_test_episode_ids)
_all_episode_ids = list(dict.fromkeys((_train_episode_ids or []) + _test_episode_ids))
_episodes = _all_episode_ids or None

dataset = LeRobotDataset(
    ds_cfg["repo_id"],
    root=ds_cfg.get("root", None),
    episodes=_episodes,
)

# Reload with delta_timestamps so each sample carries its own t-1 and t-2 states.
# Without this, forward_with_skip falls back to self._prev_state (a cross-batch
# state from a random episode), producing artificially large "velocities" (0.5-2.0)
# that do not reflect real robot motion — making ADP thresholds meaningless in eval.
_fps = getattr(dataset, "fps", 10.0)
_dt  = 1.0 / _fps
dataset = LeRobotDataset(
    ds_cfg["repo_id"],
    root=ds_cfg.get("root", None),
    episodes=_episodes,
    delta_timestamps={"observation.state": [-2.0 * _dt, -_dt, 0.0]},
)

sample = dataset[0]
all_keys = list(sample.keys())
image_keys = [k for k in all_keys
              if 'image' in k.lower()
              and isinstance(sample[k], torch.Tensor)]
state_key = next((k for k in all_keys
                  if 'state' in k.lower()
                  and isinstance(sample[k], torch.Tensor)), None)
action_key = 'action'
meta_keys = {'episode_index', 'frame_index', 'timestamp', 'index', 'task_index'}
task_key = 'task_index' if 'task_index' in all_keys else None
string_keys = [k for k in all_keys if isinstance(sample[k], str)]

ACTION_DIM = sample[action_key].shape[-1]
STATE_DIM = sample[state_key].shape[-1] if state_key else 0

print(f"  Episodes: {dataset.num_episodes}, Frames: {len(dataset)}, FPS: {dataset.fps}")
print(f"  Images: {image_keys}")
print(f"  State : {state_key} ({STATE_DIM}D), Action: {ACTION_DIM}D")
if task_key is not None:
    print(f"  Task key: {task_key} (task-specific language enabled)")

# Build per-episode frame index maps; split train vs test by episode ID.
ep_col = dataset.hf_dataset['episode_index']
episode_indices: dict[int, list[int]] = {}
for idx, ep in enumerate(ep_col):
    ep_int = ep.item() if isinstance(ep, torch.Tensor) else int(ep)
    episode_indices.setdefault(ep_int, []).append(idx)

all_eps  = sorted(episode_indices.keys())
train_eps = [ep for ep in all_eps if ep not in _test_set]
eval_eps  = [ep for ep in all_eps if ep in _test_set]
train_idx = [i for ep in train_eps for i in episode_indices[ep]]
eval_idx  = [i for ep in eval_eps  for i in episode_indices[ep]]

print(f"  Train : {len(train_eps)} eps ({len(train_idx)} frames)")
print(f"  Test  : {len(eval_eps)}  eps ({len(eval_idx)}  frames)")

# =====================================================================
#  LOAD SMOLVLA
# =====================================================================
print("\n" + "=" * 70)
print("  Loading SmolVLA Base")
print("=" * 70)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Base = lerobot/smolvla_libero (ALREADY fine-tuned on LIBERO: action_feature=7,
# knows the tasks) — NOT smolvla_base (generic, never saw LIBERO → would need huge
# training and never matches the LIBERO rollout env). This is the policy the
# hierarchical action-aware optimisation is applied ON TOP OF.
BASE_MODEL = os.environ.get("SMOLVLA_BASE", "lerobot/smolvla_libero")
print(f"  Base model: {BASE_MODEL}")
smolvla = SmolVLAPolicy.from_pretrained(BASE_MODEL)

smolvla_img_keys = list(smolvla.config.image_features.keys())
KEY_REMAP: dict[str, str] = {dk: smolvla_img_keys[i]
                              for i, dk in enumerate(image_keys)
                              if i < len(smolvla_img_keys)}
print(f"  Key remap: {KEY_REMAP}")

tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
instruction = ds_cfg["task_instruction"]
_tok = tokenizer(instruction, return_tensors="pt",
                 padding="max_length", max_length=64)
LANG_IDS = _tok['input_ids']
LANG_MASK = _tok['attention_mask'].bool()
print(f"  Instruction: '{instruction}'")

# Per-sample language from the dataset's own `task` instruction string.
# This is the ROBUST fix for the task/language alignment bug: the dataset
# provides the exact instruction for each demonstration in s['task'], so we
# tokenize that directly instead of mapping task_index → suite.get_task(i)
# positionally (which was scrambled — dataset task_index order ≠ suite order).
_LANG_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _lang_for_instruction(instr: str):
    """Tokenize an instruction string, cached by string."""
    if not isinstance(instr, str) or not instr:
        instr = instruction  # ds_cfg fallback
    cached = _LANG_CACHE.get(instr)
    if cached is None:
        tok = tokenizer(instr, return_tensors="pt", padding="max_length",
                        max_length=64, truncation=True)
        cached = (tok['input_ids'], tok['attention_mask'].bool())
        _LANG_CACHE[instr] = cached
    return cached


# Sanity check: confirm dataset exposes the per-frame instruction string.
if 'task' in all_keys and isinstance(sample.get('task'), str):
    print(f"  Per-sample language ENABLED via s['task'] (e.g. {sample['task']!r})")
else:
    print("  WARN: dataset sample has no string 'task' field — "
          "falling back to single ds_cfg instruction for all frames")

CHUNK_SIZE_S = smolvla.config.chunk_size

# ---------------------------------------------------------------------
# Reload dataset adding the ACTION chunk via delta_timestamps so each sample
# carries the real future action sequence [a_t, a_{t+1}, …, a_{t+chunk-1}]
# instead of a single action repeated. SmolVLA is an action-chunking flow
# policy; training on a constant chunk (the previous `actions.unsqueeze(1)
# .expand(...)`) taught degenerate dynamics → poor closed-loop rollout.
# Episodes/order are unchanged, so episode_indices/train_eps stay valid.
print(f"\n  Reloading dataset with action horizon = {CHUNK_SIZE_S} frames "
      f"(delta_timestamps on action) …")
dataset = LeRobotDataset(
    ds_cfg["repo_id"],
    root=ds_cfg.get("root", None),
    episodes=_episodes,
    delta_timestamps={
        "observation.state": [-2.0 * _dt, -_dt, 0.0],
        "action": [i * _dt for i in range(CHUNK_SIZE_S)],
    },
)
_chk = dataset[0]
print(f"  action sample shape now: {tuple(_chk['action'].shape)}  "
      f"(expect [{CHUNK_SIZE_S}, {ACTION_DIM}]);  "
      f"action_is_pad present: {'action_is_pad' in _chk}")

# Freeze VLM backbone
if cfg["freeze_vlm"]:
    frozen = trainable_base = 0
    for name, param in smolvla.named_parameters():
        if "vlm_with_expert.vlm" in name:
            param.requires_grad_(False)
            frozen += param.numel()
        else:
            param.requires_grad_(True)
            trainable_base += param.numel()
    print(f"  Frozen (VLM)             : {frozen/1e6:.1f} M")
    print(f"  Trainable (action expert): {trainable_base/1e6:.1f} M")

smolvla.to(DEVICE)

# Build wrapper
print("\n  Creating HierarchicalSmolVLAWrapper …")
wrapper = HierarchicalSmolVLAWrapper(smolvla, cfg).to(DEVICE)

print(f"  ADP pruner params  : {sum(p.numel() for p in wrapper.adp.parameters())/1e6:.4f} M")
print(f"  ActionAwarePRouter : {sum(p.numel() for p in wrapper.action_router.parameters())/1e6:.3f} M")
print(f"  LoRA-SP params     : {sum(p.numel() for p in wrapper.lora_adapters.parameters())/1e6:.2f} M")

# =====================================================================
#  TEACHER MODEL FOR CogKD (loaded once, frozen, permanent)
# =====================================================================
print("\n  Loading frozen CogKD teacher (smolvla_base) …")
teacher_policy = SmolVLAPolicy.from_pretrained(BASE_MODEL)
teacher_policy.to(DEVICE)
wrapper.cogkd.set_teacher(teacher_policy)
print("  CogKD teacher loaded and frozen.")

# =====================================================================
#  BATCH BUILDER
# =====================================================================
MAX_ACT_DIM = smolvla.config.max_action_dim
ENV_ACTION_DIM = ds_cfg.get("env_action_dim", 7 if DATASET_KEY == "libero_10" else ACTION_DIM)
if ENV_ACTION_DIM > MAX_ACT_DIM:
    raise ValueError(
        f"env_action_dim={ENV_ACTION_DIM} exceeds model max_action_dim={MAX_ACT_DIM}")
print(f"  Action target dim (env-aligned): {ENV_ACTION_DIM}  | model max_action_dim: {MAX_ACT_DIM}")

# ── MEAN_STD normalization (REQUIRED: smolvla_libero trains in normalized space) ──
# Verified by open-loop probe: smolvla_libero reproduces dataset actions only when
# state is MEAN_STD-normalized on input and actions are normalized as the target.
# We normalize state + action here so the hierarchical finetune is consistent with
# the base policy AND with the rollout eval (which normalizes the same way).
NORMALIZE = os.environ.get("NO_NORMALIZE", "0") != "1"
_S_MEAN = _S_STD = _A_MEAN = _A_STD = None
if NORMALIZE:
    import json as _json
    _stats_path = Path(ds_cfg.get("root", "data/datasets/HuggingFaceVLA/libero")) / "meta" / "stats.json"
    with open(_stats_path) as _f:
        _stt = _json.load(_f)
    _dev = DEVICE
    _S_MEAN = torch.tensor(np.array(_stt["observation.state"]["mean"], np.float32).ravel(), device=_dev)
    _S_STD  = torch.tensor(np.array(_stt["observation.state"]["std"],  np.float32).ravel(), device=_dev)
    _A_MEAN = torch.tensor(np.array(_stt["action"]["mean"], np.float32).ravel(), device=_dev)
    _A_STD  = torch.tensor(np.array(_stt["action"]["std"],  np.float32).ravel(), device=_dev)
    print(f"  [NORMALIZE] MEAN_STD state(in)/action(target) from {_stats_path}")


def _norm_state(t: torch.Tensor) -> torch.Tensor:
    """Normalize the last-dim state (…, 8) with MEAN_STD. No-op if disabled."""
    if not NORMALIZE:
        return t
    n = min(t.shape[-1], _S_MEAN.shape[0])
    out = t.clone()
    out[..., :n] = (t[..., :n] - _S_MEAN[:n]) / (_S_STD[:n] + 1e-8)
    return out


def _norm_action(t: torch.Tensor) -> torch.Tensor:
    """Normalize the last-dim action (…, 7) with MEAN_STD. No-op if disabled."""
    if not NORMALIZE:
        return t
    n = min(t.shape[-1], _A_MEAN.shape[0])
    out = t.clone()
    out[..., :n] = (t[..., :n] - _A_MEAN[:n]) / (_A_STD[:n] + 1e-8)
    return out

# CRITICAL: smolvla_base config declares action_feature.shape=[6], so
# original_action_dim=6 and the model would DROP the 7th action dim (gripper).
# LIBERO needs 7D (6 eef delta + gripper); without this override the gripper is
# never supervised in training nor emitted at inference → robot can never grasp.
_af = smolvla.config.action_feature
if _af is not None and _af.shape[0] != ENV_ACTION_DIM:
    _old_shape = tuple(_af.shape)
    _af.shape = (ENV_ACTION_DIM,)
    print(f"  [FIX] action_feature.shape {_old_shape} -> ({ENV_ACTION_DIM},) "
          f"(include gripper; was dropping dim {_old_shape[0]}..{ENV_ACTION_DIM-1})")


def _fit_action_chunk(acts: torch.Tensor) -> torch.Tensor:
    """(B, chunk, raw_dim) → (B, chunk, MAX_ACT_DIM): trim/pad env dim then pad to max.

    Also accepts (B, raw_dim) for back-compat (single action) and expands to a
    constant chunk as a last-resort fallback.
    """
    if acts.dim() == 2:                       # (B, raw) fallback → constant chunk
        acts = acts.unsqueeze(1).expand(-1, CHUNK_SIZE_S, -1)
    raw = acts.shape[-1]
    if raw > ENV_ACTION_DIM:
        acts = acts[..., :ENV_ACTION_DIM]
    elif raw < ENV_ACTION_DIM:
        pad = torch.zeros(*acts.shape[:-1], ENV_ACTION_DIM - raw, device=acts.device)
        acts = torch.cat([acts, pad], dim=-1)
    if ENV_ACTION_DIM < MAX_ACT_DIM:
        pad = torch.zeros(*acts.shape[:-1], MAX_ACT_DIM - ENV_ACTION_DIM, device=acts.device)
        acts = torch.cat([acts, pad], dim=-1)
    return acts                                # (B, chunk, MAX_ACT_DIM)


def build_train_batch(indices: list[int]) -> dict:
    batch_imgs = {v: [] for v in KEY_REMAP.values()}
    batch_states, batch_actions, batch_act_pads = [], [], []
    batch_lang_ids, batch_lang_masks = [], []
    for idx in indices:
        s = dataset[idx]
        for dk, sk in KEY_REMAP.items():
            batch_imgs[sk].append(s[dk])
        if state_key:
            batch_states.append(s[state_key])
        batch_actions.append(s[action_key])            # (chunk, action_dim)
        ap = s.get('action_is_pad')
        batch_act_pads.append(ap if ap is not None
                              else torch.zeros(CHUNK_SIZE_S, dtype=torch.bool))
        _ids, _mask = _lang_for_instruction(s.get('task'))
        batch_lang_ids.append(_ids)
        batch_lang_masks.append(_mask)

    batch: dict = {}
    for sk, imgs in batch_imgs.items():
        batch[sk] = torch.stack(imgs).to(DEVICE)
    if batch_states:
        batch['observation.state'] = _norm_state(torch.stack(batch_states).to(DEVICE))

    actions = torch.stack(batch_actions).to(DEVICE)     # (B, chunk, action_dim)
    batch['action'] = _fit_action_chunk(_norm_action(actions))  # normalize → (B, chunk, MAX)
    batch['action_is_pad'] = torch.stack(batch_act_pads).to(DEVICE).bool()  # (B, chunk)
    batch['observation.language.tokens'] = torch.cat(batch_lang_ids, dim=0).to(DEVICE)
    batch['observation.language.attention_mask'] = torch.cat(batch_lang_masks, dim=0).to(DEVICE)
    return batch


def build_sequential_batch(window: int) -> dict:
    """Sample `window` consecutive frames from one train episode.

    Provides observation.state_prev and observation.state_prev2 so the
    CascadeSTARRouter receives meaningful kinematic features (velocity /
    jerk) instead of white-noise state differences from random batches.
    """
    eligible = [ep for ep in train_eps if len(episode_indices[ep]) >= window + 2]
    if not eligible:
        eligible = train_eps
    ep = int(np.random.choice(eligible))
    ep_list = episode_indices[ep]

    # Start >= 2 so we always have 2 lookback frames in the same episode
    lo = 2
    hi = max(lo + 1, len(ep_list) - window + 1)
    start = int(np.random.randint(lo, hi))
    indices = ep_list[start: start + window]

    # Pre-load states for lookback: positions (start-2) … (start+window-1)
    # states_ext[0] = t-2, states_ext[1] = t-1, states_ext[2+k] = batch frame k
    states_ext = []
    for i in range(start - 2, start + window):
        if 0 <= i < len(ep_list):
            s = dataset[ep_list[i]]
            st = s[state_key] if state_key else torch.zeros(STATE_DIM)
        else:
            st = torch.zeros(STATE_DIM)
        states_ext.append(st)

    batch_imgs = {v: [] for v in KEY_REMAP.values()}
    batch_states, batch_actions, batch_act_pads = [], [], []
    state_prev_list, state_prev2_list = [], []
    batch_lang_ids, batch_lang_masks = [], []

    for k, idx in enumerate(indices):
        s = dataset[idx]
        for dk, sk in KEY_REMAP.items():
            batch_imgs[sk].append(s[dk])
        batch_states.append(states_ext[2 + k])
        batch_actions.append(s[action_key])          # (chunk, action_dim)
        ap = s.get('action_is_pad')
        batch_act_pads.append(ap if ap is not None
                              else torch.zeros(CHUNK_SIZE_S, dtype=torch.bool))
        state_prev_list.append(states_ext[1 + k])   # t-1
        state_prev2_list.append(states_ext[0 + k])  # t-2
        _ids, _mask = _lang_for_instruction(s.get('task'))
        batch_lang_ids.append(_ids)
        batch_lang_masks.append(_mask)

    batch: dict = {}
    for sk, imgs in batch_imgs.items():
        batch[sk] = torch.stack(imgs).to(DEVICE)
    if batch_states:
        batch['observation.state']       = _norm_state(torch.stack(batch_states).to(DEVICE))
        batch['observation.state_prev']  = _norm_state(torch.stack(state_prev_list).to(DEVICE))
        batch['observation.state_prev2'] = _norm_state(torch.stack(state_prev2_list).to(DEVICE))

    actions = torch.stack(batch_actions).to(DEVICE)     # (B, chunk, action_dim)
    batch['action'] = _fit_action_chunk(_norm_action(actions))  # normalize → (B, chunk, MAX)
    batch['action_is_pad'] = torch.stack(batch_act_pads).to(DEVICE).bool()  # (B, chunk)
    batch['observation.language.tokens'] = torch.cat(batch_lang_ids, dim=0).to(DEVICE)
    batch['observation.language.attention_mask'] = torch.cat(batch_lang_masks, dim=0).to(DEVICE)
    return batch


# =====================================================================
#  TRAINING HELPERS
# =====================================================================

import queue, threading, datetime

class _BatchPrefetcher:
    """Pre-fetches train batches in a background thread to hide I/O latency.

    Usage:
        pf = _BatchPrefetcher(micro_bs)
        pf.start()
        batch = pf.get()   # blocks only if the queue is empty
        pf.stop()
    """
    def __init__(self, micro_bs: int, depth: int = 4):
        self._micro_bs = micro_bs
        self._q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                bi = np.random.choice(train_idx, self._micro_bs, replace=True).tolist()
                batch = build_train_batch(bi)
                self._q.put(batch, timeout=2.0)
            except Exception:
                pass

    def get(self) -> dict:
        return self._q.get(timeout=30.0)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)


def build_eval_batch(micro_bs: int) -> dict:
    """Sample a random mini-batch from the test split (for validation loss)."""
    bi = np.random.choice(eval_idx, micro_bs, replace=True).tolist()
    return build_train_batch(bi)


def _anneal_tau(step: int, total: int) -> float:
    p = step / max(total - 1, 1)
    return cfg["gumbel_tau_start"] * (1 - p) + cfg["gumbel_tau_end"] * p


def _run_phase(phase_name: str, total_steps: int, optimizer,
               scheduler=None, enable_skip=True, enable_adp=False,
               enable_lora=False, enable_cogkd=False, enable_snap=False,
               anneal_tau=True, fixed_tau=None,
               save_keys: list | None = None, save_tag: str = "phase",
               use_sequential: bool = False,
               lambda_budget_ramp: tuple | None = None,
               lambda_skip_floor_ramp: tuple | None = None):
    """Generic phase runner. Returns list of per-step losses.

    Args:
        use_sequential     : if True, use build_sequential_batch (meaningful kinematics)
        lambda_budget_ramp : (start, end) tuple for linear ramp; None = use cfg default
    """
    micro_bs   = cfg["micro_batch"]
    grad_accum = cfg["grad_accum"]
    log_every  = cfg.get("log_every", 100)
    save_every = cfg["save_every"]
    seq_window = cfg.get("sequential_window", 8)

    losses: list[float] = []
    val_losses: list[float] = []
    _step_times: list[float] = []
    _phase_start = time.perf_counter()

    # Start prefetch worker (background I/O for non-sequential batches)
    _prefetcher = _BatchPrefetcher(micro_bs)
    if not use_sequential:
        _prefetcher.start()

    pbar = tqdm(range(total_steps), desc=f"  {phase_name}", ncols=100)

    for step in pbar:
        _t0 = time.perf_counter()
        optimizer.zero_grad()
        tau = (_anneal_tau(step, total_steps) if anneal_tau
               else (fixed_tau if fixed_tau is not None
                     else cfg["gumbel_tau_end"]))

        # Lambda skip_floor ramp: p-ratio gradually enforced toward 0.70
        if lambda_skip_floor_ramp is not None:
            lam_sf_start, lam_sf_end = lambda_skip_floor_ramp
            lam_sf = lam_sf_start + (lam_sf_end - lam_sf_start) * step / max(total_steps - 1, 1)
        else:
            lam_sf = None   # forward_with_skip uses cfg default

        # Lambda budget ramp: task dominates early, budget strong at end
        if lambda_budget_ramp is not None:
            lam_b_start, lam_b_end = lambda_budget_ramp
            lam_b = lam_b_start + (lam_b_end - lam_b_start) * step / max(total_steps - 1, 1)
        else:
            lam_b = None   # forward_with_skip uses cfg default

        accum_loss = 0.0
        accum_dict: dict[str, float] = {}

        for _ in range(grad_accum):
            if use_sequential:
                batch = build_sequential_batch(seq_window)
            else:
                batch = _prefetcher.get()

            if cfg["fp16"]:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss, ld = wrapper.forward_with_skip(
                        batch, tau=tau,
                        enable_skip=enable_skip, enable_adp=enable_adp,
                        enable_lora=enable_lora, enable_cogkd=enable_cogkd,
                        enable_snap=enable_snap,
                        lambda_budget_override=lam_b,
                        lambda_skip_floor_override=lam_sf,
                    )
                    loss = loss / grad_accum
                loss.backward()
            else:
                loss, ld = wrapper.forward_with_skip(
                    batch, tau=tau,
                    enable_skip=enable_skip, enable_adp=enable_adp,
                    enable_lora=enable_lora, enable_cogkd=enable_cogkd,
                    enable_snap=enable_snap,
                    lambda_budget_override=lam_b,
                    lambda_skip_floor_override=lam_sf,
                )
                loss = loss / grad_accum
                loss.backward()

            accum_loss += loss.item()
            for k, v in ld.items():
                accum_dict[k] = accum_dict.get(k, 0.0) + v / grad_accum

        torch.nn.utils.clip_grad_norm_(
            [p for pg in optimizer.param_groups for p in pg['params']],
            cfg["max_grad_norm"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(accum_loss)

        # ETA — rolling average over last 50 steps
        _step_times.append(time.perf_counter() - _t0)
        if len(_step_times) > 50:
            _step_times.pop(0)

        # Validation loss on test split (skip if no test data)
        if eval_idx and (step + 1) % EVAL_EVERY == 0:
            wrapper.eval()
            with torch.no_grad():
                vb = build_eval_batch(micro_bs)
                if cfg["fp16"]:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        v_loss, _ = wrapper.forward_with_skip(
                            vb, tau=tau,
                            enable_skip=enable_skip, enable_adp=enable_adp,
                            enable_lora=enable_lora, enable_cogkd=enable_cogkd,
                            enable_snap=enable_snap,
                        )
                else:
                    v_loss, _ = wrapper.forward_with_skip(
                        vb, tau=tau,
                        enable_skip=enable_skip, enable_adp=enable_adp,
                        enable_lora=enable_lora, enable_cogkd=enable_cogkd,
                        enable_snap=enable_snap,
                    )
            val_losses.append(v_loss.item())
            wrapper.train()

        if step % log_every == 0:
            avg = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
            postfix: dict = {"loss": f"{avg:.4f}"}
            if enable_skip:
                postfix["skip"] = f"{accum_dict.get('avg_skip_ratio', 0):.2f}"
                postfix["aon"]  = f"{accum_dict.get('layer_on_max', 1.0):.2f}"
                postfix["rho"]  = f"{accum_dict.get('rho_target', 0):.2f}"
                postfix["H"]    = f"{accum_dict.get('layer_entropy_mean', 0):.2f}"
                postfix["ceil"] = f"{accum_dict.get('skip_ceiling_loss', 0):.3f}"
                postfix["div"]  = f"{accum_dict.get('gate_diversity_loss', 0):.3f}"
                postfix["ρσ"]   = f"{accum_dict.get('rho_std', 0):.3f}"
            if enable_adp:
                postfix["tkr"]  = f"{accum_dict.get('token_keep_ratio', 1):.2f}"
            if enable_lora:
                postfix["spec"] = f"{accum_dict.get('spec_loss', 0):.3f}"
            if enable_cogkd:
                postfix["kd"]   = f"{accum_dict.get('cogkd_loss', 0):.4f}"
            if enable_snap:
                postfix["snap"] = f"{accum_dict.get('snap_loss', 0):.4f}"
            if lambda_budget_ramp is not None:
                postfix["λ_b"]  = f"{lam_b:.4f}"
            if lambda_skip_floor_ramp is not None:
                postfix["λ_sf"] = f"{lam_sf:.3f}"  # p-ratio floor lambda ramp
            if val_losses:
                postfix["val"]  = f"{val_losses[-1]:.4f}"
            # ETA
            if _step_times:
                _avg_s = sum(_step_times) / len(_step_times)
                _rem   = (total_steps - step - 1) * _avg_s
                postfix["eta"] = str(datetime.timedelta(seconds=int(_rem)))
            pbar.set_postfix(postfix)

        if (step + 1) % save_every == 0:
            ckpt = {"step": step + 1,
                    "smolvla": smolvla.state_dict(),
                    "action_router": wrapper.action_router.state_dict(),
                    "adp": wrapper.adp.state_dict()}
            if enable_lora:
                ckpt["lora_adapters"] = wrapper.lora_adapters.state_dict()
            torch.save(ckpt, OUTPUT_DIR / f"{save_tag}_step{step+1}.pt")

    _prefetcher.stop()

    elapsed = datetime.timedelta(seconds=int(time.perf_counter() - _phase_start))
    final_avg = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
    val_summary = (f"  val loss: {np.mean(val_losses[-10:]):.6f}" if val_losses else "")
    print(f"\n  {phase_name} complete — avg loss: {final_avg:.6f}{val_summary}  "
          f"[elapsed: {elapsed}]")
    return losses


# =====================================================================
#  RESUME FROM CHECKPOINT (when --start_phase > 1)
# =====================================================================
START_PHASE = args.start_phase
END_PHASE = args.end_phase
if END_PHASE < START_PHASE:
    raise ValueError(f"--end_phase ({END_PHASE}) must be >= --start_phase ({START_PHASE})")
print("\n" + "=" * 70)
print(f"  PHASE PLAN: run phases {START_PHASE}..{END_PHASE}  "
      f"(P1={cfg['phase1_steps']}  P2={cfg['phase2_steps']}  P3={cfg['phase3_steps']} steps)")
if START_PHASE == 1 and END_PHASE == 1:
    print("  → Phase 1 ONLY: saves phase1_complete.pt then stops (no P2/P3).")
if START_PHASE == 3:
    print("  → Phase 3 from phase1_complete.pt: saves final_model.pt.")
print("=" * 70)

if START_PHASE > 1:
    _auto = {
        2: OUTPUT_DIR / "phase1_complete.pt",
        3: OUTPUT_DIR / "phase1_complete.pt",
    }
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _auto[START_PHASE]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run with --start_phase 1 first, or pass --checkpoint <path>")
    print(f"\n  Resuming from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    smolvla.load_state_dict(ckpt["smolvla"])
    wrapper.action_router.load_state_dict(ckpt["action_router"])
    wrapper.adp.load_state_dict(ckpt["adp"])
    if "lora_adapters" in ckpt:
        wrapper.lora_adapters.load_state_dict(ckpt["lora_adapters"])
        print("  LoRA-SP adapters loaded from checkpoint.")
    else:
        print("  LoRA-SP adapters: starting from random init (not in checkpoint).")
    print(f"  Model weights restored — starting at Phase {START_PHASE}.")
    del ckpt; gc.collect(); torch.cuda.empty_cache()

# Placeholder loss lists for phases that are skipped (used in training curves)
losses_p1: list = []
losses_p2: list = []
losses_p3: list = []

# =====================================================================
#  PHASE 1: CascadeSTAR + ADP initialisation
# =====================================================================
#  PHASE 1: CascadeSTAR + ADP initialisation
# =====================================================================
if START_PHASE <= 1:
    print("\n" + "=" * 70)
    print("  PHASE 1: ActionAwarePRouter + ADP Initialisation")
    print("=" * 70)

    p1_params = (list(p for p in smolvla.parameters() if p.requires_grad)
                 + list(wrapper.action_router.parameters())
                 + list(wrapper.adp.parameters()))

    opt_p1 = torch.optim.AdamW(p1_params, lr=cfg["phase1_lr"],
                                weight_decay=cfg["weight_decay"])

    smolvla.train()
    wrapper.train()

    print(f"  Steps: {cfg['phase1_steps']}  micro_batch={cfg['micro_batch']}  grad_accum={cfg['grad_accum']}")
    print(f"  Gumbel tau: {cfg['gumbel_tau_start']} → {cfg['gumbel_tau_end']}")
    print(f"  ADP v_thresh={cfg['adp_velocity_threshold']}  R_min={cfg['adp_min_keep_ratio']}  wave={cfg.get('gpu_wave_size',32)}")
    print(f"  gate_bias_init={cfg.get('router_gate_bias_init', 0.0)}  init_gain={cfg.get('router_init_gain', 3.0)}")
    print(f"  lambda_budget ramp: {cfg.get('lambda_budget_start', 0.005)} → {cfg.get('lambda_budget_end', 0.10)}")
    print(f"  sequential_window={cfg.get('sequential_window', 8)}  λ_tent={cfg.get('lambda_temporal_entropy', 0.1)}  λ_rspread={cfg.get('lambda_rho_spread', 0.05)}  λ_flip={cfg.get('lambda_gate_flip', 0.1)}")
    print(f"  [Phase 1 loss: task + gate + L_temporal_entropy + L_rho_spread + L_gate_flip + budget(ramp)]")

    losses_p1 = _run_phase(
        "Phase1 CascadeSTAR+ADP", cfg["phase1_steps"], opt_p1,
        enable_skip=True, enable_adp=True,
        enable_lora=False, enable_cogkd=False, enable_snap=False,
        anneal_tau=True, save_tag="p1",
        use_sequential=True,
        lambda_budget_ramp=(cfg.get("lambda_budget_start", 0.005),
                            cfg.get("lambda_budget_end", 0.10)),
        lambda_skip_floor_ramp=(0.0, 0.5),
    )

    torch.save({
        "smolvla": smolvla.state_dict(),
        "action_router": wrapper.action_router.state_dict(),
        "adp": wrapper.adp.state_dict(),
    }, OUTPUT_DIR / "phase1_complete.pt")

    gc.collect(); torch.cuda.empty_cache()

# =====================================================================
#  PHASE 2: Joint — LoRA-SP + budget coupling
# =====================================================================
if START_PHASE <= 2 <= END_PHASE:
    print("\n" + "=" * 70)
    print("  PHASE 2: Joint Optimisation — LoRA-SP + Budget Coupling")
    print("=" * 70)

    p2_params = (list(p for p in smolvla.parameters() if p.requires_grad)
                 + list(wrapper.action_router.parameters())
                 + list(wrapper.adp.parameters())
                 + list(wrapper.lora_adapters.parameters()))

    opt_p2 = torch.optim.AdamW(p2_params, lr=cfg["phase2_lr"],
                                weight_decay=cfg["weight_decay"])

    print(f"  Steps: {cfg['phase2_steps']}")
    print(f"  LoRA max_rank={cfg['lora_max_rank']}  energy_thresh={cfg['lora_energy_threshold']}")
    print(f"  [Phase 2 loss: task + gate + budget + spec + cost]")

    losses_p2 = _run_phase(
        "Phase2 Joint", cfg["phase2_steps"], opt_p2,
        enable_skip=True, enable_adp=True,
        enable_lora=True, enable_cogkd=False, enable_snap=False,
        anneal_tau=False, save_tag="p2",
        lambda_skip_floor_ramp=(0.5, 2.0),
    )

    torch.save({
        "smolvla": smolvla.state_dict(),
        "action_router": wrapper.action_router.state_dict(),
        "adp": wrapper.adp.state_dict(),
        "lora_adapters": wrapper.lora_adapters.state_dict(),
    }, OUTPUT_DIR / "phase2_complete.pt")

    gc.collect(); torch.cuda.empty_cache()

# =====================================================================
#  PHASE 3: CogKD (ToI) + SnapFlow
# =====================================================================
if START_PHASE <= 3 <= END_PHASE:
    print("\n" + "=" * 70)
    print("  PHASE 3: CogKD (ToI-Masked) + SnapFlow Self-Distillation")
    print("=" * 70)

    # SnapFlow teacher: frozen copy of current student
    print("  Creating SnapFlow teacher from current student weights …")
    wrapper.snap.create_teacher(smolvla.model)
    print(f"  SnapFlow teacher steps: {cfg['snap_teacher_steps']}")
    print(f"  CogKD teacher: frozen smolvla_base (already loaded)")
    print(f"  [Phase 3 loss: task + gate + budget + spec + cost + CogKD + SnapFlow]")

    p3_params = (list(p for p in smolvla.parameters() if p.requires_grad)
                 + list(wrapper.action_router.parameters())
                 + list(wrapper.adp.parameters())
                 + list(wrapper.lora_adapters.parameters()))

    opt_p3 = torch.optim.AdamW(p3_params, lr=cfg["phase3_lr"],
                                 weight_decay=cfg["weight_decay"])
    sched_p3 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_p3, T_max=cfg["phase3_steps"], eta_min=1e-6)

    losses_p3 = _run_phase(
        "Phase3 CogKD+Snap", cfg["phase3_steps"], opt_p3, scheduler=sched_p3,
        enable_skip=True, enable_adp=True,
        enable_lora=True, enable_cogkd=True, enable_snap=True,
        anneal_tau=False, save_tag="p3",
        # Phase 3: enforce p-ratio floor strongly toward 0.70 target
        lambda_skip_floor_ramp=(2.0, 3.0),
    )

    # Save name: final_model.pt when phase 3 was the end, else a phase-tagged name.
    _p3_name = "final_model.pt" if END_PHASE == 3 else "phase3_complete.pt"
    torch.save({
        "smolvla": smolvla.state_dict(),
        "action_router": wrapper.action_router.state_dict(),
        "adp": wrapper.adp.state_dict(),
        "lora_adapters": wrapper.lora_adapters.state_dict(),
    }, OUTPUT_DIR / _p3_name)
    print(f"\n  Phase 3 complete. Model saved → {OUTPUT_DIR / _p3_name}")
else:
    print("\n  [Phase 3 SKIPPED — END_PHASE < 3]")

# Free teachers
del wrapper.snap.teacher_model
wrapper.snap.teacher_model = None
del teacher_policy
gc.collect(); torch.cuda.empty_cache()

# =====================================================================
#  TRAINING CURVES
# =====================================================================
print("\n  Saving training curves …")
# Only plot phases that actually ran (non-empty loss lists)
active_phases = [(ls, lbl, col) for ls, lbl, col in [
    (losses_p1, "Phase 1: CascadeSTAR + ADP", "#2196F3"),
    (losses_p2, "Phase 2: Joint (LoRA-SP + Budget)", "#4CAF50"),
    (losses_p3, "Phase 3: CogKD + SnapFlow", "#FF5722"),
] if ls]
n_plots = max(len(active_phases), 1)
W = 50
fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
if n_plots == 1:
    axes = [axes]
for ax, (ls, title, color) in zip(axes, active_phases):
    if len(ls) > W:
        ax.plot(np.convolve(ls, np.ones(W) / W, 'valid'),
                color=color, lw=1.5)
    else:
        ax.plot(ls, color=color, lw=1.5)
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
plt.suptitle("Hierarchical Action-Aware VLA — Training Curves",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150)
plt.close()

# =====================================================================
#  EVALUATION
# =====================================================================
print("\n" + "=" * 70)
print("  Evaluation")
print("=" * 70)

smolvla.eval()
wrapper.eval()


def _eval_batch(sample_dict: dict) -> dict:
    batch: dict = {}
    for k, v in sample_dict.items():
        if k in meta_keys or k in string_keys:
            continue
        if not isinstance(v, torch.Tensor):
            continue
        out_key = KEY_REMAP.get(k, k)
        batch[out_key] = v.unsqueeze(0).to(DEVICE)
    batch['observation.language.tokens'] = LANG_IDS.to(DEVICE)
    batch['observation.language.attention_mask'] = LANG_MASK.to(DEVICE)
    return batch


results: dict = {
    "mse_per_episode": [], "latency_ms": [],
    "predictions": {}, "ground_truth": {},
    "avg_skip_ratio": 0.0, "avg_token_keep_ratio": 1.0,
}

eval_ep_list = eval_eps[:EVAL["num_eval_episodes"]]
print(f"  Eval episodes: {eval_ep_list}")

for ep_idx in eval_ep_list:
    preds, gts = [], []
    smolvla.reset()
    wrapper.reset_episode()
    for si in tqdm(episode_indices[ep_idx],
                   desc=f"  Ep{ep_idx}", ncols=85, leave=False):
        s = dataset[si]
        gt = s[action_key].numpy()
        batch = _eval_batch(s)
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = smolvla.select_action(batch)
        t1 = time.perf_counter()
        pnp = pred.squeeze().cpu().numpy()
        if pnp.ndim > 1:
            pnp = pnp[0]
        pnp = pnp[:ACTION_DIM]           # clip to dataset action dim (may be < ACTION_DIM if model has smaller head)
        gt  = gt[:len(pnp)]              # align ground truth to actual pred length to avoid broadcast errors
        results["latency_ms"].append((t1 - t0) * 1000)
        preds.append(pnp); gts.append(gt)

    preds_np = np.array(preds); gts_np = np.array(gts)
    mse = float(np.mean((preds_np - gts_np) ** 2))
    results["mse_per_episode"].append(mse)
    results["predictions"][ep_idx] = preds_np
    results["ground_truth"][ep_idx] = gts_np
    print(f"    Ep{ep_idx}: MSE={mse:.4f}  ({len(episode_indices[ep_idx])} frames)")

if results["predictions"]:
    all_p = np.concatenate(list(results["predictions"].values()))
    all_g = np.concatenate(list(results["ground_truth"].values()))
    results["mse_total"] = float(np.mean((all_p - all_g) ** 2))
    results["mse_per_joint"] = np.mean((all_p - all_g) ** 2, axis=0).tolist()
else:
    all_p = all_g = np.array([])
    results["mse_total"] = float("nan")
    results["mse_per_joint"] = []
    print("  [INFO] No eval episodes — skipping MSE computation (use eval_dynamic_model.py)")
results["latency_mean_ms"] = float(np.mean(results["latency_ms"])) if results["latency_ms"] else 0.0
results["avg_skip_ratio"] = float(np.mean(wrapper.skip_stats[-500:])) if wrapper.skip_stats else 0.0
results["avg_token_keep_ratio"] = float(np.mean(wrapper.token_stats[-500:])) if wrapper.token_stats else 1.0
# Per-step gate active ratio (fraction of skippable layers ON) — used to verify dynamic variation
results["skip_stats_per_step"] = [round(float(v), 4) for v in wrapper.skip_stats]
if wrapper.skip_stats:
    ss = np.array(wrapper.skip_stats)
    results["skip_stats_std"]  = float(ss.std())
    results["skip_stats_min"]  = float(ss.min())
    results["skip_stats_max"]  = float(ss.max())
    results["skip_stats_range"] = float(ss.max() - ss.min())
    print(f"  Skip std   : {ss.std():.4f}  (want >=0.10 for dynamic behaviour)")
    print(f"  Skip range : {ss.min():.3f} – {ss.max():.3f}")

print(f"\n  MSE total  : {results['mse_total']:.4f}" if not (isinstance(results['mse_total'], float) and np.isnan(results['mse_total'])) else "\n  MSE total  : N/A (no eval eps)")
print(f"  Latency    : {results['latency_mean_ms']:.1f} ms")
print(f"  Skip ratio : {results['avg_skip_ratio']:.3f}  (training tail)")
print(f"  Token keep : {results['avg_token_keep_ratio']:.3f}  (training tail)")

# =====================================================================
#  SUMMARY PLOTS
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
_has_eval = len(eval_ep_list) > 0

ax = axes[0, 0]
if _has_eval:
    ax.bar(np.arange(len(eval_ep_list)), results["mse_per_episode"], color='#2196F3')
    ax.set_xticks(np.arange(len(eval_ep_list)))
    ax.set_xticklabels([f"Ep{e}" for e in eval_ep_list])
else:
    ax.text(0.5, 0.5, "No eval episodes", ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel("Episode"); ax.set_ylabel("MSE")
ax.set_title("MSE per Episode"); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
n_joints = len(results["mse_per_joint"])
if n_joints:
    ax.bar(np.arange(n_joints), results["mse_per_joint"], color='#4CAF50')
else:
    ax.text(0.5, 0.5, "No eval episodes", ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel("Joint"); ax.set_ylabel("MSE")
ax.set_title("MSE per Joint"); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
if results["latency_ms"]:
    ax.hist(results["latency_ms"], bins=max(1, min(40, len(results["latency_ms"]))), color='#FF5722', alpha=0.8)
else:
    ax.text(0.5, 0.5, "No latency data", ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count")
ax.set_title("Inference Latency"); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
all_ls = losses_p1 + losses_p2 + losses_p3
if len(all_ls) > W:
    ax.plot(np.convolve(all_ls, np.ones(W) / W, 'valid'),
            color='#2196F3', lw=1.0)
    ax.axvline(len(losses_p1), color='red', ls='--', alpha=0.5,
               label='P1→P2')
    ax.axvline(len(losses_p1) + len(losses_p2), color='green',
               ls='--', alpha=0.5, label='P2→P3')
elif all_ls:
    ax.plot(all_ls, color='#2196F3', lw=1.0)
ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.set_title("Combined Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if _has_eval:
    ep0 = eval_ep_list[0]
    gt0 = results["ground_truth"][ep0]
    pd0 = results["predictions"][ep0]
    nj = min(3, ACTION_DIM)
    t_ax = np.arange(len(gt0))
    for j in range(nj):
        ax.plot(t_ax, gt0[:, j], '-', color=f'C{j}', lw=2, label=f'GT J{j}')
        ax.plot(t_ax, pd0[:, j], '--', color=f'C{j}', alpha=0.6, label=f'Pred J{j}')
    ax.set_xlabel("Step"); ax.set_ylabel("Action")
    ax.set_title(f"Trajectory Ep{ep0}")
    ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, "No eval episodes\n(run eval_dynamic_model.py)", ha='center', va='center', transform=ax.transAxes)
    ax.set_title("Trajectory")

ax = axes[1, 2]
ax.axis('off')
mse_str = f"{results['mse_total']:.4f}" if not (isinstance(results['mse_total'], float) and (results['mse_total'] != results['mse_total'])) else "N/A"
summary_lines = [
    f"Pipeline: hierarchical_action_aware",
    f"Dataset : {ds_cfg['repo_id']}",
    f"",
    f"MSE total   : {mse_str}",
    f"Latency     : {results['latency_mean_ms']:.1f} ms",
    f"",
    f"Skip ratio  : {results['avg_skip_ratio']:.3f}",
    f"Token keep  : {results['avg_token_keep_ratio']:.3f}",
    f"",
    f"Phase 1 steps: {cfg['phase1_steps']}",
    f"Phase 2 steps: {cfg['phase2_steps']}",
    f"Phase 3 steps: {cfg['phase3_steps']}",
    f"",
    f"v_thresh : {cfg['adp_velocity_threshold']}",
    f"R_min    : {cfg['adp_min_keep_ratio']}",
    f"wave_size: {cfg.get('gpu_wave_size', 32)}",
    f"α_couple : {cfg.get('budget_coupling_alpha', 0.70)}",
    f"ToI ratio: {cfg['toi_ratio']}",
]
ax.text(0.05, 0.95, '\n'.join(summary_lines),
        transform=ax.transAxes,
        verticalalignment='top', fontfamily='monospace', fontsize=9)

plt.suptitle("Hierarchical Action-Aware VLA — Summary",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'summary.png', dpi=150)
plt.close()

# =====================================================================
#  JSON RESULTS
# =====================================================================
results_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in results.items()
                if not isinstance(v, dict)}
with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results_save, f, indent=2)

print(f"\n  Results  → {OUTPUT_DIR / 'results.json'}")
print(f"  Summary  → {OUTPUT_DIR / 'summary.png'}")
print(f"  Curves   → {OUTPUT_DIR / 'training_curves.png'}")
print("\n  Done.")
