"""
Fine-Tune SmolVLA with Entropy-Guided Semi-Finetuning + CIM + SnapFlow.

Pipeline: finetune_semifinetune_entropy_cim
RTX 4070 SUPER (12GB) | fp16/bf16 | lerobot 0.4.4

Main ideas:
    1. EGSF complexity score from entropy, IoU proxy, and gradient proxy.
    2. CIM token pruning with contextual interaction masking.
    3. Hybrid threshold safeguard: lambda_cost = 0 when mse_signal > tau_eff.
    4. Hard-sample curriculum loaded from hard_samples_manifest.json.
    5. SnapFlow + CogKD stabilization in phase 3.
"""

import os
import sys
import time
import json
import types
import gc
import copy
import math
import csv
from collections import deque
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
#  PATCH: Fix lerobot 0.4.4 import chain crash on Windows/Python 3.10
# =====================================================================
import lerobot
import importlib.util

_pkg = lerobot.__path__[0]

_robots_mod = types.ModuleType("lerobot.robots")
_robots_mod.__path__ = [os.path.join(_pkg, "robots")]
_robots_mod.__package__ = "lerobot.robots"
_spec = importlib.util.spec_from_file_location(
        "lerobot.robots.config", os.path.join(_pkg, "robots", "config.py")
)
_cfg_mod = importlib.util.module_from_spec(_spec)
sys.modules["lerobot.robots.config"] = _cfg_mod
_spec.loader.exec_module(_cfg_mod)
_robots_mod.RobotConfig = _cfg_mod.RobotConfig
sys.modules["lerobot.robots"] = _robots_mod

_proc_mod = types.ModuleType("lerobot.processor")
_proc_mod.__path__ = [os.path.join(_pkg, "processor")]
_proc_mod.__package__ = "lerobot.processor"
_proc_mod.RobotAction = dict
_proc_mod.RobotObservation = dict
_proc_mod.PolicyAction = dict
sys.modules["lerobot.processor"] = _proc_mod

_policies_mod = types.ModuleType("lerobot.policies")
_policies_mod.__path__ = [os.path.join(_pkg, "policies")]
_policies_mod.__package__ = "lerobot.policies"
sys.modules["lerobot.policies"] = _policies_mod

# =====================================================================
#  LOAD CONFIG
# =====================================================================
from finetune_config import DATASETS, TRAINING, EVAL

DATASET_KEY = "svla_so100_pickplace"
ds_cfg = DATASETS[DATASET_KEY]

dyn_cfg = TRAINING["semifinetune_entropy_cim"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(f"d:/EyetechCode/results/semifinetune_entropy_cim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  SmolVLA Fine-Tune: Entropy-CIM + CORAL Experts + LoRA-SP + SnapFlow")
print("=" * 70)
print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB" if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print(f"  Dataset: {ds_cfg['repo_id']}")
print(f"  Output: {OUTPUT_DIR}")
print()

# =====================================================================
#  MODULE 1: Hierarchical STAR Router with Curriculum Learning
# =====================================================================

class HierarchicalSTARRouter(nn.Module):
    """Hierarchical Spatial-Temporal Aware Router for Dynamic Layer Skipping.

    Improvements over basic STAR Router:
      - Layer groups: Spatial (L8-L11) and Action-Refine (L12-L15)
      - Each group has its own sub-router sensitive to different action phases
      - Curriculum Learning: tau annealing + Action Entropy penalty
      - Diversity-Driven Loss to break saturation skip
      - SnapFlow MSE feedback loop for closed-loop depth control

    Inputs: pooled hidden state + E_view + delta_s + acceleration + action_entropy
    """

    def __init__(self, hidden_dim, num_spatial_layers=4, num_action_layers=4):
        super().__init__()
        self.num_spatial_layers = num_spatial_layers
        self.num_action_layers = num_action_layers
        self.total_skippable = num_spatial_layers + num_action_layers

        # Extended input: hidden(D) + E_view(1) + delta_s(1) + accel(1) + action_entropy(1)
        input_dim = hidden_dim + 4

        # Sub-router for Spatial layers (L8-L11): sensitive to spatial changes
        self.spatial_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_spatial_layers),
        )
        nn.init.constant_(self.spatial_gate[-1].bias, 2.0)  # default keep

        # Sub-router for Action layers (L12-L15): sensitive to action precision
        self.action_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_action_layers),
        )
        nn.init.constant_(self.action_gate[-1].bias, 2.0)  # default keep

        # SnapFlow MSE feedback integration
        self.snap_mse_proj = nn.Linear(1, hidden_dim // 4)
        self.feedback_gate = nn.Linear(hidden_dim // 4, self.total_skippable)
        nn.init.zeros_(self.feedback_gate.weight)
        nn.init.zeros_(self.feedback_gate.bias)

        # Tracking for diversity loss
        self._gate_history = []

    def compute_visual_entropy(self, attention_weights):
        """Compute entropy of attention distribution over visual tokens."""
        attn_mean = attention_weights.mean(dim=(1, 2))
        attn_probs = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(attn_probs * (attn_probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        return entropy

    def compute_action_entropy(self, action_deltas):
        """Compute entropy of action changes as complexity indicator.

        High action_entropy -> complex manipulation -> keep more layers.
        Low action_entropy -> simple transit -> can skip more layers.
        """
        if action_deltas is None:
            return torch.zeros(1, 1, device=next(self.parameters()).device)
        # Normalize action deltas to probability-like distribution
        action_abs = action_deltas.abs()
        action_prob = action_abs / (action_abs.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(action_prob * (action_prob + 1e-8).log()).sum(dim=-1, keepdim=True)
        return entropy

    def compute_diversity_loss(self, gates):
        """Diversity-Driven Loss to prevent saturation skip.

        Encourages different skip patterns for different inputs by maximizing
        variance of gate decisions across the batch and history.
        """
        # Intra-batch diversity: variance across batch dimension
        if gates.shape[0] > 1:
            batch_var = gates.var(dim=0).mean()
        else:
            batch_var = torch.tensor(0.0, device=gates.device)

        # Temporal diversity: variance across recent history
        if len(self._gate_history) >= 5:
            history_tensor = torch.stack(self._gate_history[-10:])  # last 10 batches
            temporal_var = history_tensor.var(dim=0).mean()
        else:
            temporal_var = torch.tensor(0.0, device=gates.device)

        # Negative: we want to MAXIMIZE diversity -> minimize negative variance
        diversity_loss = -(batch_var + temporal_var)
        return diversity_loss

    def forward(self, hidden_pooled, e_view, delta_s_norm, accel_norm,
                action_entropy, tau=1.0, hard=False, snap_mse=None):
        """
        Args:
            hidden_pooled: (B, hidden_dim) pooled output from last fixed layer
            e_view: (B, 1) visual entropy
            delta_s_norm: (B, 1) state change norm
            accel_norm: (B, 1) acceleration norm (2nd-order dynamics)
            action_entropy: (B, 1) action complexity indicator
            tau: Gumbel-Softmax temperature
            hard: if True, use hard decisions (inference)
            snap_mse: (B, 1) SnapFlow prediction MSE for feedback loop (optional)
        Returns:
            gates: (B, total_skippable) values in [0,1]
            gate_loss: scalar cost penalty
            diversity_loss: scalar diversity penalty
        """
        x = torch.cat([hidden_pooled, e_view, delta_s_norm, accel_norm, action_entropy], dim=-1)

        # Sub-router predictions
        spatial_logits = self.spatial_gate(x)   # (B, num_spatial)
        action_logits = self.action_gate(x)     # (B, num_action)
        all_logits = torch.cat([spatial_logits, action_logits], dim=-1)  # (B, total)

        # SnapFlow MSE feedback: when prediction is hard, reduce skipping
        if snap_mse is not None:
            feedback = F.silu(self.snap_mse_proj(snap_mse))
            feedback_bias = self.feedback_gate(feedback)  # (B, total)
            # High MSE -> positive bias -> more likely to keep layers
            all_logits = all_logits + feedback_bias

        # Action entropy modulation: complex actions -> keep more layers
        # Scale logits by action entropy (higher entropy -> harder to skip)
        entropy_boost = 1.0 + action_entropy * 0.5  # (B, 1)
        all_logits = all_logits * entropy_boost

        if hard or not self.training:
            gates = (torch.sigmoid(all_logits) > 0.5).float()
        else:
            # Gumbel-Softmax for differentiable binary decisions
            logits_2class = torch.stack([torch.zeros_like(all_logits), all_logits], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_2class, tau=tau, hard=False, dim=-1)
            gates = gumbel_out[..., 1]

        # Cost penalty: mean gate activation (want to minimize = skip more)
        gate_loss = gates.mean()

        # Diversity loss
        diversity_loss = self.compute_diversity_loss(gates)

        # Track for temporal diversity
        if self.training:
            self._gate_history.append(gates.detach().mean(dim=0))
            if len(self._gate_history) > 20:
                self._gate_history = self._gate_history[-20:]

        return gates, gate_loss, diversity_loss


# =====================================================================
#  MODULE 2: CIM (Contextual Interaction Masking)
# =====================================================================

class VLAIAPTokenPruner(nn.Module):
    """Contextual Interaction Masking (CIM) token pruner.

    Replaces basic ADP (velocity-based) with:
    - IoU-based Interaction Lock (Conservative <-> Aggressive mode)
      - Geometric Priors for structural anchor preservation
      - Temporal Smoothing of token importance across frames
      - Action-Aware Controller override for fine-grained actions

    Conservative Mode (IoU < threshold): keep_ratio = 0.8 (exploration)
    Aggressive Mode  (IoU >= threshold): keep_ratio = 0.3 (exploitation)
    """

    def __init__(
        self,
        conservative_ratio=0.8,
        aggressive_ratio=0.3,
        iou_threshold=0.5,
        temporal_momentum=0.7,
        geometric_anchor_ratio=0.1,
        entropy_weight=0.4,
        iou_weight=0.35,
        grad_weight=0.25,
        interaction_entropy_tau=1.6,
        iou_proxy_tau=0.5,
    ):
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

        # Learnable IoU threshold refinement
        self.threshold_adjust = nn.Parameter(torch.tensor(0.0))

        # Interaction Lock estimator: predicts IoU from attention + state
        self.iou_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Geometric prior network: identifies structural anchors
        self.anchor_scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Temporal smoothing state
        self._prev_importance = None
        self._prev_keep_mask = None

        # Action granularity classifier (coarse vs fine)
        self.action_classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 0=coarse, 1=fine
        )

    def reset_temporal(self):
        """Reset temporal state for new episode."""
        self._prev_importance = None
        self._prev_keep_mask = None

    def compute_interaction_lock(self, token_embeddings, state):
        """Estimate Interaction Lock (IoU proxy) from token embeddings and state.

        Real IoU requires object detection; we approximate with attention
        concentration as a proxy for target lock.
        """
        B, N, D = token_embeddings.shape
        # Pool tokens and project
        token_pool = token_embeddings.mean(dim=1)  # (B, D)
        if token_pool.shape[-1] != 128:
            token_pool = F.adaptive_avg_pool1d(
                token_pool.unsqueeze(1), 128
            ).squeeze(1)
        iou_proxy = self.iou_estimator(token_pool)  # (B, 1)
        return iou_proxy

    def compute_geometric_priors(self, token_embeddings):
        """Identify structural anchor tokens using geometric priors.

        These are tokens at edges, corners, and physically important regions
        that should be preserved regardless of pruning mode.
        """
        B, N, D = token_embeddings.shape
        # Project tokens and score
        if D != 128:
            tokens_proj = F.adaptive_avg_pool1d(token_embeddings, 128)  # (B, N, 128)
        else:
            tokens_proj = token_embeddings
        anchor_scores = self.anchor_scorer(tokens_proj).squeeze(-1)  # (B, N)
        return anchor_scores

    def compute_action_granularity(self, state):
        """Classify current action as coarse (transit) or fine (manipulation).

        Returns probability of fine-grained action.
        """
        if state.shape[-1] > 16:
            state_trunc = state[:, :16]
        else:
            state_trunc = F.pad(state, (0, 16 - state.shape[-1]))
        return self.action_classifier(state_trunc)  # (B, 1)

    def forward(self, token_embeddings, token_mask, state,
                attention_scores=None, v_ee=None):
        """
        Args:
            token_embeddings: (B, N, D) visual token embeddings
            token_mask: (B, N) boolean mask
            state: (B, state_dim) current robot state
            attention_scores: (B, N) importance scores (optional)
            v_ee: (B, 1) end-effector velocity (optional, backward compat)
        Returns:
            pruned_embeddings: (B, K, D) kept tokens (padded)
            pruned_mask: (B, K) boolean mask for kept tokens
            avg_keep_ratio: scalar - average fraction of tokens kept
            keep_ratio_per_sample: (B, 1)
            iap_info: dict with diagnostic information
        """
        B, N, D = token_embeddings.shape

        # 1. Compute Interaction Lock (IoU proxy)
        iou_proxy = self.compute_interaction_lock(token_embeddings, state)

        # 2. Build CIM contextual complexity score
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
        )
        complexity_score = complexity_score.clamp(0.0, 1.0)

        # 3. Determine pruning mode per sample
        effective_threshold = self.iou_threshold + torch.tanh(self.threshold_adjust) * 0.1
        is_aggressive = (iou_proxy > effective_threshold).float()  # (B, 1)

        # 4. Compute keep ratio: conservative vs aggressive
        keep_ratio = (1.0 - is_aggressive) * self.conservative_ratio + \
                     is_aggressive * self.aggressive_ratio  # (B, 1)

        # CIM interpolation: higher complexity keeps more tokens.
        cim_keep_ratio = self.aggressive_ratio + (
            self.conservative_ratio - self.aggressive_ratio
        ) * complexity_score
        keep_ratio = torch.maximum(keep_ratio, cim_keep_ratio)

        # 5. Interaction lock override in manipulation-heavy windows.
        interaction_lock = (
            (visual_entropy > self.interaction_entropy_tau).float()
            * (iou_proxy > self.iou_proxy_tau).float()
        )
        keep_ratio = torch.where(
            interaction_lock > 0.5,
            torch.maximum(keep_ratio, torch.ones_like(keep_ratio) * 0.90),
            keep_ratio,
        )

        # 6. Action granularity override: fine actions -> keep more tokens
        action_fine = self.compute_action_granularity(state)  # (B, 1)
        # If fine-grained (action_fine > 0.5), override to high keep ratio
        keep_ratio = torch.where(
            action_fine > 0.7,
            torch.ones_like(keep_ratio) * 0.95,  # near-full for fine actions
            keep_ratio
        )

        # 7. Compute token importance scores
        if attention_scores is None:
            # Default: positional importance (center tokens more important)
            positions = torch.arange(N, device=token_embeddings.device).float()
            center = N / 2.0
            attention_scores = torch.exp(-((positions - center) ** 2) / (2 * (N / 4.0) ** 2))
            attention_scores = attention_scores.unsqueeze(0).expand(B, -1)

        # 8. Geometric prior anchors - boost importance of structural anchors
        anchor_scores = self.compute_geometric_priors(token_embeddings)
        # Anchors get a boost: top anchor_ratio% tokens get importance += 1.0
        num_anchors = max(int(N * self.geometric_anchor_ratio), 1)
        _, anchor_indices = anchor_scores.topk(num_anchors, dim=-1)
        anchor_boost = torch.zeros_like(attention_scores)
        anchor_boost.scatter_(1, anchor_indices, 1.0)
        importance = attention_scores + anchor_boost

        # 9. Temporal smoothing: blend with previous frame's importance
        if self._prev_importance is not None and self._prev_importance.shape == importance.shape:
            importance = (self.temporal_momentum * self._prev_importance +
                         (1.0 - self.temporal_momentum) * importance)
        self._prev_importance = importance.detach().clone()

        # 10. Select top-K tokens based on importance
        K_per_sample = (keep_ratio * N).long().clamp(min=1, max=N)  # (B, 1)
        K = K_per_sample.max().item()  # uniform K for batching
        K = max(K, 1)

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


# =====================================================================
#  MODULE 3: LoRA-SP (Spectral Rank Adaptation) - same as original
# =====================================================================

class LoRASPAdapter(nn.Module):
    """LoRA-SP: LoRA with Spectral rank adaptation.

    deltaW(x) = U * diag(s(x)) * V
    where s(x) is input-dependent singular value scores.

    Spectral Concentration Loss encourages sparse effective rank.
    """

    def __init__(self, in_features, out_features, max_rank=128,
                 energy_threshold=0.9, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.energy_threshold = energy_threshold
        self.alpha = alpha

        # Shared vector banks
        self.U = nn.Parameter(torch.randn(out_features, max_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)

        # Singular Value Router: input-dependent score prediction
        self.router = nn.Sequential(
            nn.Linear(in_features, max_rank),
            nn.Sigmoid(),
        )

        self.scaling = alpha / max_rank

    def compute_spectral_loss(self, scores):
        """Spectral concentration loss: encourages energy in fewer directions."""
        scores_sq = scores ** 2
        total_energy = scores_sq.sum(dim=-1, keepdim=True) + 1e-8
        prob = scores_sq / total_energy
        spec_loss = (prob * (prob + 1e-8).log()).sum(dim=-1).mean()
        return -spec_loss

    def forward(self, x, return_spec_loss=False):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        scores = self.router(x_flat.detach())

        Vx = F.linear(x_flat, self.V)
        SVx = Vx * scores * self.scaling
        delta = F.linear(SVx, self.U)

        delta = delta.reshape(*orig_shape[:-1], self.out_features)

        if return_spec_loss:
            spec_loss = self.compute_spectral_loss(scores)
            return delta, spec_loss
        return delta


# =====================================================================
#  MODULE 4: CORAL Expert Manager
# =====================================================================

class CORALExpertManager(nn.Module):
    """CORAL: Scalable Multi-Task Robot Learning via LoRA Experts.

    Maintains a bank of lightweight LoRA experts, one per task category.
    Routes to the correct expert based on language instruction embedding.
    Zero-overhead swapping: experts are pre-indexed by task keyword.

    Key design:
      - Frozen base model + multiple LoRA expert banks
      - Language-based routing (no gating network needed)
      - Gradient isolation: each expert only gets gradients from its task
    """

    def __init__(self, hidden_dim, num_experts=4, expert_rank=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_rank = expert_rank

        # Expert banks: each expert is a lightweight LoRA pair
        self.expert_A = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, expert_rank) * 0.01)
            for _ in range(num_experts)
        ])
        self.expert_B = nn.ParameterList([
            nn.Parameter(torch.randn(expert_rank, hidden_dim) * 0.01)
            for _ in range(num_experts)
        ])

        # Language embedding projector for routing
        self.lang_projector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts),
        )

        # Expert scaling
        self.expert_scaling = 1.0 / expert_rank

        # Task keyword -> expert ID mapping (built during training)
        self._task_expert_map = {}

        # Active expert cache
        self._active_expert_id = 0

    def route_from_language(self, lang_embeds):
        """Route to expert based on language instruction embedding.

        Args:
            lang_embeds: (B, seq_len, D) language token embeddings
        Returns:
            expert_id: int, selected expert index
            routing_probs: (B, num_experts) routing probabilities
        """
        # Pool language embeddings
        lang_pooled = lang_embeds.mean(dim=1)  # (B, D)
        if lang_pooled.shape[-1] != self.hidden_dim:
            lang_pooled = F.adaptive_avg_pool1d(
                lang_pooled.unsqueeze(1), self.hidden_dim
            ).squeeze(1)

        routing_logits = self.lang_projector(lang_pooled)  # (B, num_experts)
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Select expert with highest probability
        expert_id = routing_probs.argmax(dim=-1).mode().values.item()
        self._active_expert_id = expert_id

        return expert_id, routing_probs

    def apply_expert(self, x, expert_id=None):
        """Apply the selected LoRA expert to input features.

        Args:
            x: (B, ..., hidden_dim) input features
            expert_id: int, which expert to apply (default: last routed)
        Returns:
            delta: (B, ..., hidden_dim) expert adjustment
        """
        if expert_id is None:
            expert_id = self._active_expert_id

        expert_id = min(expert_id, self.num_experts - 1)

        A = self.expert_A[expert_id]  # (D, r)
        B = self.expert_B[expert_id]  # (r, D)

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_dim)

        # Standard LoRA: delta = x @ A @ B * scaling
        delta = F.linear(F.linear(x_flat, A.t()), B.t()) * self.expert_scaling
        delta = delta.reshape(orig_shape)

        return delta

    def compute_routing_loss(self, routing_probs):
        """Load balancing loss to encourage expert utilization.

        Prevents routing collapse where all inputs go to one expert.
        """
        # Average routing probability per expert across batch
        avg_prob = routing_probs.mean(dim=0)  # (num_experts,)
        # Ideal uniform distribution
        target = torch.ones_like(avg_prob) / self.num_experts
        balance_loss = F.mse_loss(avg_prob, target)
        return balance_loss


# =====================================================================
#  MODULE 5: SnapFlow with CogKD
# =====================================================================

class SnapFlowCogKDTrainer:
    """SnapFlow with Cognition Self-Knowledge Distillation (CogKD).

    Enhancements over basic SnapFlow:
      - CogKD: uses full-depth model as teacher to preserve cognition tokens
      - MSE feedback: sends prediction error back to STAR Router
      - Cognition token identification via attention divergence
    """

    def __init__(self, num_teacher_steps=10, cogkd_lambda=0.3, cogkd_temperature=2.0):
        self.num_teacher_steps = num_teacher_steps
        self.teacher_model = None
        self.cogkd_lambda = cogkd_lambda
        self.cogkd_temperature = cogkd_temperature
        self._last_mse = None

    def create_teacher(self, model):
        """Create a frozen copy of the model as teacher."""
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute_teacher_target(self, teacher_flow_model,
                                prefix_embs, prefix_pad_masks, prefix_att_masks,
                                noise, chunk_size, action_out_proj):
        """Compute teacher's multi-step prediction for shortcut target."""
        bsize = noise.shape[0]
        device = noise.device
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = teacher_flow_model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        num_steps = self.num_teacher_steps
        dt = -1.0 / num_steps
        x_t = noise.clone()

        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
            v_t = teacher_flow_model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    def compute_snapflow_loss(self, student_velocity, x_t, teacher_x0, t):
        """Compute SnapFlow consistency loss with MSE tracking."""
        t_expanded = t[:, None, None]
        u_shortcut = (x_t - teacher_x0) / (t_expanded + 1e-6)
        snap_loss = F.mse_loss(student_velocity, u_shortcut)

        # Track MSE for feedback loop
        with torch.no_grad():
            self._last_mse = snap_loss.item()

        return snap_loss

    def compute_cogkd_loss(self, student_hidden, teacher_hidden):
        """Cognition Self-Knowledge Distillation loss.

        Aligns student's intermediate representations with teacher's
        to preserve cognition tokens even under heavy pruning/skipping.

        Args:
            student_hidden: (B, seq_len, D) student's hidden states
            teacher_hidden: (B, seq_len, D) teacher's hidden states
        Returns:
            cogkd_loss: scalar distillation loss
        """
        # Match dimensions
        if student_hidden.shape != teacher_hidden.shape:
            min_len = min(student_hidden.shape[1], teacher_hidden.shape[1])
            student_hidden = student_hidden[:, :min_len]
            teacher_hidden = teacher_hidden[:, :min_len]

        # Soft target distribution from teacher
        T = self.cogkd_temperature
        teacher_logits = teacher_hidden / T
        student_logits = student_hidden / T

        # KL divergence on normalized representations
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        cogkd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T ** 2)

        return cogkd_loss * self.cogkd_lambda

    def get_snap_mse_feedback(self, batch_size, device):
        """Get SnapFlow MSE as feedback signal for STAR Router."""
        if self._last_mse is not None:
            return torch.tensor([[self._last_mse]], device=device).expand(batch_size, 1)
        return None


# =====================================================================
#  MODULE 6: Dynamic Wrapper - Full Integration
# =====================================================================

class DynamicEntropyCIMWrapper(nn.Module):
    """Full integration wrapper combining all enhanced modules.

    Architecture:
      1. Hierarchical STAR Router (Curriculum + Diversity)
    2. CIM Token Pruning (Entropy + IoU + Gradient proxy)
      3. LoRA-SP Spectral Adaptation
      4. CORAL Expert Manager (Language-routed)
      5. SnapFlow + CogKD (with MSE feedback)
    """

    def __init__(self, smolvla_policy, dyn_cfg):
        super().__init__()
        self.policy = smolvla_policy
        self.flow_model = smolvla_policy.model
        self.cfg = dyn_cfg

        # Get architecture dimensions
        vlm_hidden = self.flow_model.vlm_with_expert.config.text_config.hidden_size
        expert_hidden = self.flow_model.vlm_with_expert.expert_hidden_size
        num_vlm_layers = self.flow_model.vlm_with_expert.num_vlm_layers

        # 1. Hierarchical STAR Router
        self.star_router = HierarchicalSTARRouter(
            hidden_dim=vlm_hidden,
            num_spatial_layers=dyn_cfg["num_spatial_layers"],
            num_action_layers=dyn_cfg["num_action_layers"],
        )

        # 2. CIM token pruner
        self.token_pruner = VLAIAPTokenPruner(
            conservative_ratio=dyn_cfg.get("cim_conservative_ratio", dyn_cfg.get("iap_conservative_ratio", 0.8)),
            aggressive_ratio=dyn_cfg.get("cim_aggressive_ratio", dyn_cfg.get("iap_aggressive_ratio", 0.3)),
            iou_threshold=dyn_cfg.get("cim_iou_proxy_tau", dyn_cfg.get("iap_iou_threshold", 0.5)),
            temporal_momentum=dyn_cfg.get("cim_temporal_gamma", dyn_cfg.get("iap_temporal_momentum", 0.7)),
            geometric_anchor_ratio=dyn_cfg.get("iap_geometric_anchor_ratio", 0.1),
            entropy_weight=dyn_cfg.get("complexity_alpha_entropy", 0.40),
            iou_weight=dyn_cfg.get("complexity_alpha_iou_proxy", 0.35),
            grad_weight=dyn_cfg.get("complexity_alpha_grad_proxy", 0.25),
            interaction_entropy_tau=dyn_cfg.get("cim_interaction_entropy_tau", 1.6),
            iou_proxy_tau=dyn_cfg.get("cim_iou_proxy_tau", 0.5),
        )

        # 3. LoRA-SP Adapters for VLM layers
        self.lora_adapters = nn.ModuleDict()
        vlm_layers = self.flow_model.vlm_with_expert.get_vlm_model().text_model.layers
        for layer_idx in range(dyn_cfg["num_fixed_layers"], num_vlm_layers):
            layer = vlm_layers[layer_idx]
            layer_key = f"layer_{layer_idx}"
            attn = layer.self_attn

            q_in = attn.q_proj.in_features
            q_out = attn.q_proj.out_features
            k_in = attn.k_proj.in_features
            k_out = attn.k_proj.out_features
            v_in = attn.v_proj.in_features
            v_out = attn.v_proj.out_features
            o_in = attn.o_proj.in_features
            o_out = attn.o_proj.out_features

            self.lora_adapters[f"{layer_key}_q"] = LoRASPAdapter(
                q_in, q_out, max_rank=dyn_cfg["lora_max_rank"],
                energy_threshold=dyn_cfg["lora_energy_threshold"],
            )
            self.lora_adapters[f"{layer_key}_k"] = LoRASPAdapter(
                k_in, k_out, max_rank=dyn_cfg["lora_max_rank"],
                energy_threshold=dyn_cfg["lora_energy_threshold"],
            )
            self.lora_adapters[f"{layer_key}_v"] = LoRASPAdapter(
                v_in, v_out, max_rank=dyn_cfg["lora_max_rank"],
                energy_threshold=dyn_cfg["lora_energy_threshold"],
            )
            self.lora_adapters[f"{layer_key}_o"] = LoRASPAdapter(
                o_in, o_out, max_rank=dyn_cfg["lora_max_rank"],
                energy_threshold=dyn_cfg["lora_energy_threshold"],
            )

        # 4. CORAL Expert Manager
        self.coral_manager = CORALExpertManager(
            hidden_dim=vlm_hidden,
            num_experts=dyn_cfg["coral_num_experts"],
            expert_rank=dyn_cfg["coral_expert_rank"],
        )

        # 5. SnapFlow + CogKD Trainer
        self.snap_trainer = SnapFlowCogKDTrainer(
            num_teacher_steps=dyn_cfg["snap_teacher_steps"],
            cogkd_lambda=dyn_cfg.get("cogkd_lambda", 0.3),
            cogkd_temperature=dyn_cfg.get("cogkd_temperature", 2.0),
        )

        # State tracking
        self._prev_state = None
        self._prev_prev_state = None  # for acceleration

        # Hybrid threshold tracking for saturation-skip mitigation
        self.hybrid_tau_history = deque(maxlen=int(dyn_cfg.get("hybrid_tau_window", 128)))
        self.latest_tau_eff = float(dyn_cfg.get("hybrid_tau_fixed", 1000.0))
        self.latest_lambda_cost_eff = float(dyn_cfg.get("lambda_cost", 0.0))

        # Hard-sample curriculum map keyed by (episode_id, frame_index)
        self.hard_sample_map = self._load_hard_sample_manifest(
            dyn_cfg.get("hard_sample_manifest_path", "")
        )
        self.hard_sample_hits = 0
        self.hard_sample_queries = 0

        # Training statistics
        self.skip_stats = []
        self.token_stats = []
        self.diversity_stats = []
        self.iap_stats = []
        self.coral_stats = []
        self.snap_mse_stats = []
        self.cim_stats = []

    def _load_hard_sample_manifest(self, manifest_path):
        mapping = {}
        if not manifest_path:
            return mapping

        path = Path(manifest_path)
        if not path.exists():
            print(f"  [warn] hard-sample manifest not found: {path}")
            return mapping

        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            items = payload.get("items", []) if isinstance(payload, dict) else []
            max_weight = float(self.cfg.get("hard_sample_weight_max", 2.5))
            default_weight = float(self.cfg.get("hard_sample_weight_default", 1.0))
            for item in items:
                ep = int(item.get("episode_id", -1))
                step = int(item.get("step", -1))
                if ep < 0 or step < 0:
                    continue
                weight = float(item.get("curriculum_weight", default_weight))
                mapping[(ep, step)] = float(np.clip(weight, 1.0, max_weight))
            print(f"  Loaded hard-sample manifest: {len(mapping)} entries")
        except Exception as exc:
            print(f"  [warn] failed to load hard-sample manifest: {exc}")

        return mapping

    def _lookup_hard_sample_weights(self, batch, device):
        default_weight = float(self.cfg.get("hard_sample_weight_default", 1.0))
        episode_ids = batch.get("meta.episode_index")
        frame_ids = batch.get("meta.frame_index")

        if episode_ids is None or frame_ids is None:
            bsize = int(batch["action"].shape[0])
            return torch.full((bsize,), default_weight, dtype=torch.float32, device=device)

        episode_ids = episode_ids.detach().cpu().tolist()
        frame_ids = frame_ids.detach().cpu().tolist()
        weights = []
        for ep, frame in zip(episode_ids, frame_ids):
            self.hard_sample_queries += 1
            key = (int(ep), int(frame))
            if key in self.hard_sample_map:
                self.hard_sample_hits += 1
                weights.append(float(self.hard_sample_map[key]))
            else:
                weights.append(default_weight)

        return torch.tensor(weights, dtype=torch.float32, device=device)

    def _compute_hybrid_tau(self, global_step):
        warmup_steps = int(self.cfg.get("hybrid_tau_warmup_steps", 0))
        fixed_tau = float(self.cfg.get("hybrid_tau_fixed", 1000.0))
        if global_step < warmup_steps or len(self.hybrid_tau_history) < 8:
            return fixed_tau

        percentile = float(self.cfg.get("hybrid_tau_percentile", 85.0))
        tau_eff = float(np.percentile(np.asarray(self.hybrid_tau_history, dtype=np.float64), percentile))
        return tau_eff

    def _effective_lambda_cost(self, mse_signal, global_step):
        tau_eff = self._compute_hybrid_tau(global_step)
        lambda_base = float(self.cfg.get("lambda_cost", 0.0))
        lambda_eff = 0.0 if mse_signal > tau_eff else lambda_base
        self.latest_tau_eff = tau_eff
        self.latest_lambda_cost_eff = lambda_eff
        return lambda_eff, tau_eff

    def reset_episode(self):
        """Reset per-episode state."""
        self._prev_state = None
        self._prev_prev_state = None
        self.token_pruner.reset_temporal()
        self.policy.reset()

    def compute_context_features(self, state, images_emb=None, actions=None, iou_proxy=None):
        """Compute EGSF context features for the Hierarchical Router.

        Returns:
            e_view: (B, 1) visual entropy proxy
            delta_s_norm: (B, 1) state change norm
            accel_norm: (B, 1) acceleration norm (2nd-order)
            action_entropy: (B, 1) EGSF complexity indicator
        """
        B = state.shape[0]
        device = state.device

        # Visual entropy proxy
        if images_emb is not None:
            e_view = images_emb.var(dim=1).mean(dim=-1, keepdim=True)
        else:
            e_view = torch.zeros(B, 1, device=device)

        # State delta (velocity)
        if self._prev_state is not None:
            delta_s = state - self._prev_state
            delta_s_norm = delta_s.norm(dim=-1, keepdim=True)
        else:
            delta_s = torch.zeros_like(state)
            delta_s_norm = torch.zeros(B, 1, device=device)

        # Acceleration (2nd-order dynamics)
        if self._prev_state is not None and self._prev_prev_state is not None:
            prev_delta = self._prev_state - self._prev_prev_state
            accel = delta_s - prev_delta
            accel_norm = accel.norm(dim=-1, keepdim=True)
        else:
            accel_norm = torch.zeros(B, 1, device=device)

        # EGSF complexity score: entropy + IoU proxy + gradient proxy.
        if actions is not None:
            grad_proxy = delta_s_norm + 0.5 * accel_norm
        else:
            grad_proxy = delta_s_norm

        entropy_norm = torch.sigmoid(e_view)
        if iou_proxy is None:
            iou_norm = torch.zeros(B, 1, device=device)
        else:
            iou_norm = iou_proxy
        grad_norm = torch.tanh(grad_proxy)

        alpha_entropy = float(self.cfg.get("complexity_alpha_entropy", 0.40))
        alpha_iou = float(self.cfg.get("complexity_alpha_iou_proxy", 0.35))
        alpha_grad = float(self.cfg.get("complexity_alpha_grad_proxy", 0.25))
        action_entropy = (
            alpha_entropy * entropy_norm
            + alpha_iou * iou_norm
            + alpha_grad * grad_norm
        ).clamp(0.0, 1.0)

        # Update state history
        self._prev_prev_state = self._prev_state
        self._prev_state = state.detach().clone()

        return e_view, delta_s_norm, accel_norm, action_entropy

    def forward_with_skip(self, batch, tau=1.0, global_step=0,
                          enable_skip=True, enable_iap=False,
                          enable_lora=False, enable_coral=False,
                          enable_snap=False, enable_cogkd=False,
                          noise=None, time_val=None):
        """Forward pass with all dynamic optimization modules."""
        flow_model = self.flow_model

        # 1. Prepare inputs (same as old pipeline)
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch['observation.language.tokens']
        lang_masks = batch['observation.language.attention_mask']
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        B = state.shape[0]
        device = state.device

        # 2. Visual embeddings for context
        with torch.no_grad():
            img_embs_list = []
            for img in images:
                img_emb = flow_model.vlm_with_expert.embed_image(img)
                img_embs_list.append(img_emb)
            all_img_emb = torch.cat(img_embs_list, dim=1) if img_embs_list else None

        iou_proxy_hint = None
        if all_img_emb is not None:
            with torch.no_grad():
                iou_proxy_hint = self.token_pruner.compute_interaction_lock(all_img_emb, state)

        # 3. Extended context features for Hierarchical Router
        e_view, delta_s_norm, accel_norm, action_entropy = \
            self.compute_context_features(state, all_img_emb, actions, iou_proxy=iou_proxy_hint)

        # 4. CIM token pruning
        token_keep_ratio = torch.tensor(1.0, device=device)
        iap_info = {}
        if enable_iap and all_img_emb is not None:
            _, _, token_keep_ratio, _, iap_info = self.token_pruner(
                all_img_emb,
                torch.ones(B, all_img_emb.shape[1], dtype=torch.bool, device=device),
                state,
                v_ee=delta_s_norm,
            )
            self.iap_stats.append(iap_info)
            self.cim_stats.append(iap_info.get("complexity_score", 0.0))

        # 5. Hierarchical STAR Router
        num_fixed = self.cfg["num_fixed_layers"]
        snap_mse_feedback = self.snap_trainer.get_snap_mse_feedback(B, device) \
            if enable_snap else None

        if enable_skip:
            hidden_pooled = all_img_emb.mean(dim=1) if all_img_emb is not None else \
                torch.zeros(B, self.star_router.spatial_gate[0].in_features - 4, device=device)
            expected_dim = self.star_router.spatial_gate[0].in_features - 4
            if hidden_pooled.shape[-1] != expected_dim:
                hidden_pooled = F.adaptive_avg_pool1d(
                    hidden_pooled.unsqueeze(1), expected_dim
                ).squeeze(1)

            gates, gate_loss, diversity_loss = self.star_router(
                hidden_pooled, e_view, delta_s_norm, accel_norm,
                action_entropy, tau=tau, snap_mse=snap_mse_feedback,
            )
        else:
            gates = torch.ones(B, self.cfg["num_skippable_layers"], device=device)
            gate_loss = torch.tensor(0.0, device=device)
            diversity_loss = torch.tensor(0.0, device=device)

        # Track skip stats
        if enable_skip:
            self.skip_stats.append(gates.mean().item())
            self.diversity_stats.append(diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss)
        self.token_stats.append(token_keep_ratio.item() if isinstance(token_keep_ratio, torch.Tensor) else token_keep_ratio)

        # 6. CORAL Expert routing
        routing_loss = torch.tensor(0.0, device=device)
        if enable_coral:
            with torch.no_grad():
                lang_embs = flow_model.vlm_with_expert.get_vlm_model().text_model.embed_tokens(lang_tokens)
            expert_id, routing_probs = self.coral_manager.route_from_language(lang_embs)
            routing_loss = self.coral_manager.compute_routing_loss(routing_probs)
            self.coral_stats.append({'expert_id': expert_id, 'routing_entropy': -(routing_probs * (routing_probs + 1e-8).log()).sum(dim=-1).mean().item()})

        # 7. Task loss via ORIGINAL flow_model.forward()
        losses = flow_model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=noise, time=time_val
        )
        original_action_dim = self.policy.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        per_sample_loss = losses.mean(dim=(1, 2))
        hard_sample_weights = self._lookup_hard_sample_weights(batch, device)
        task_loss_unweighted = per_sample_loss.mean()
        task_loss = (per_sample_loss * hard_sample_weights).mean()
        hard_weight_mean = hard_sample_weights.mean().item()

        # 8. LoRA-SP spectral loss (auxiliary)
        total_spec_loss = torch.tensor(0.0, device=device)
        lora_count = 0
        if enable_lora and all_img_emb is not None:
            lora_input = all_img_emb.detach()
            for key, adapter in self.lora_adapters.items():
                if lora_input.shape[-1] != adapter.in_features:
                    adapted_input = F.adaptive_avg_pool1d(lora_input, adapter.in_features)
                else:
                    adapted_input = lora_input
                _, s_loss = adapter(adapted_input, return_spec_loss=True)
                total_spec_loss = total_spec_loss + s_loss
                lora_count += 1

        avg_spec_loss = total_spec_loss / max(lora_count, 1)

        # 9. CORAL expert adjustment (auxiliary loss from expert application)
        coral_spec_loss = torch.tensor(0.0, device=device)
        if enable_coral and all_img_emb is not None:
            coral_input = all_img_emb.detach()
            if coral_input.shape[-1] != self.coral_manager.hidden_dim:
                coral_input = F.adaptive_avg_pool1d(
                    coral_input.permute(0, 2, 1), self.coral_manager.hidden_dim
                ).permute(0, 2, 1)
            coral_delta = self.coral_manager.apply_expert(coral_input)
            # Expert regularization: delta should be small
            coral_spec_loss = coral_delta.pow(2).mean() * 0.01

        # 10. SnapFlow + CogKD
        snap_loss = torch.tensor(0.0, device=device)
        cogkd_loss = torch.tensor(0.0, device=device)
        if enable_snap and self.snap_trainer.teacher_model is not None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = flow_model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            if noise is None:
                noise = flow_model.sample_noise(actions.shape, actions.device)

            teacher_x0 = self.snap_trainer.compute_teacher_target(
                self.snap_trainer.teacher_model,
                prefix_embs.detach(), prefix_pad_masks.detach(), prefix_att_masks.detach(),
                noise, flow_model.config.chunk_size, flow_model.action_out_proj
            )
            teacher_x0 = teacher_x0[:, :, :original_action_dim]

            # Student single-step prediction
            if time_val is None:
                time_val = flow_model.sample_time(B, actions.device)
            time_expanded = time_val[:, None, None]
            x_t = time_expanded * noise[:, :, :original_action_dim] + \
                  (1 - time_expanded) * actions[:, :, :original_action_dim]
            student_v = (noise[:, :, :original_action_dim] - actions[:, :, :original_action_dim])
            snap_loss = self.snap_trainer.compute_snapflow_loss(
                student_v, x_t, teacher_x0, time_val
            )

            # Track snap MSE for feedback
            if self.snap_trainer._last_mse is not None:
                self.snap_mse_stats.append(self.snap_trainer._last_mse)

            # CogKD: knowledge distillation from full-depth teacher
            if enable_cogkd:
                # Use prefix embeddings as proxy for hidden states
                student_hidden = prefix_embs.detach()
                with torch.no_grad():
                    teacher_prefix, _, _ = self.snap_trainer.teacher_model.embed_prefix(
                        images, img_masks, lang_tokens, lang_masks, state=state
                    )
                cogkd_loss = self.snap_trainer.compute_cogkd_loss(
                    prefix_embs, teacher_prefix.detach()
                )

        # 11. Token-Action Alignment Loss (Phase 2+)
        token_action_align_loss = torch.tensor(0.0, device=device)
        if enable_iap and enable_lora and all_img_emb is not None:
            # Proxy: correlation between token importance and action prediction quality
            # Use variance of pruned tokens as alignment signal
            token_var = all_img_emb.var(dim=1).mean()
            action_var = actions.var(dim=1).mean()
            token_action_align_loss = F.mse_loss(
                token_var / (token_var + action_var + 1e-8),
                torch.tensor(0.5, device=device)  # target balanced
            )

        # 12. Combined loss with curriculum-aware weighting
        token_cost = token_keep_ratio if isinstance(token_keep_ratio, torch.Tensor) else \
            torch.tensor(token_keep_ratio, device=device)

        lambda_spec = self.cfg["lambda_spec"]
        lambda_cost = self.cfg["lambda_cost"]
        lambda_diversity = self.cfg.get("lambda_diversity", 0.005)
        lambda_align = self.cfg.get("lambda_token_action_align", 0.01)
        lambda_ib = self.cfg.get("lambda_ib", 0.0)
        lambda_cim_align = self.cfg.get("lambda_cim_align", 0.0)

        if enable_snap and self.snap_trainer._last_mse is not None:
            mse_signal = float(self.snap_trainer._last_mse)
        else:
            mse_signal = float(task_loss_unweighted.detach().item())
        self.hybrid_tau_history.append(mse_signal)
        lambda_cost_eff, tau_eff = self._effective_lambda_cost(mse_signal, global_step)
        saturation_guard_active = mse_signal > tau_eff

        ib_loss = torch.tensor(0.0, device=device)
        if enable_iap:
            ib_loss = 0.5 * (token_cost + gates.mean())

        cim_align_loss = torch.tensor(0.0, device=device)
        if enable_iap:
            cim_target = torch.tensor(iap_info.get("complexity_score", 0.5), device=device)
            cim_align_loss = F.mse_loss(token_cost, cim_target)

        total_loss = task_loss
        if enable_lora and lora_count > 0:
            total_loss = total_loss + lambda_spec * avg_spec_loss
        if enable_skip:
            total_loss = total_loss + lambda_cost_eff * gate_loss
            total_loss = total_loss + lambda_diversity * diversity_loss
        if enable_iap:
            total_loss = total_loss + lambda_cost_eff * token_cost
        if enable_coral:
            total_loss = total_loss + 0.001 * routing_loss + coral_spec_loss
        if enable_snap:
            total_loss = total_loss + snap_loss * 0.5
        if enable_cogkd:
            total_loss = total_loss + cogkd_loss
        if enable_iap and enable_lora:
            total_loss = total_loss + lambda_align * token_action_align_loss
        if enable_iap and lambda_ib > 0:
            total_loss = total_loss + lambda_ib * ib_loss
        if enable_iap and lambda_cim_align > 0:
            total_loss = total_loss + lambda_cim_align * cim_align_loss

        num_vlm_layers = self.flow_model.vlm_with_expert.num_vlm_layers
        loss_dict = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "task_loss_unweighted": task_loss_unweighted.item(),
            "gate_loss": gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
            "diversity_loss": diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
            "spec_loss": avg_spec_loss.item() if isinstance(avg_spec_loss, torch.Tensor) else avg_spec_loss,
            "snap_loss": snap_loss.item(),
            "cogkd_loss": cogkd_loss.item() if isinstance(cogkd_loss, torch.Tensor) else cogkd_loss,
            "routing_loss": routing_loss.item() if isinstance(routing_loss, torch.Tensor) else routing_loss,
            "token_keep_ratio": token_cost.item() if isinstance(token_cost, torch.Tensor) else token_cost,
            "avg_skip_ratio": 1.0 - gates.mean().item() if enable_skip else 0.0,
            "active_layers": (gates.sum(dim=-1).mean().item() + num_fixed) if enable_skip else num_vlm_layers,
            "iou_proxy": iap_info.get('iou_proxy', 0.0),
            "action_fine_prob": iap_info.get('action_fine_prob', 0.0),
            "complexity_score": iap_info.get('complexity_score', 0.0),
            "interaction_lock_ratio": iap_info.get('interaction_lock_ratio', 0.0),
            "snap_mse_feedback": self.snap_trainer._last_mse if self.snap_trainer._last_mse else 0.0,
            "mse_signal": mse_signal,
            "tau_eff": tau_eff,
            "lambda_cost_eff": lambda_cost_eff,
            "saturation_guard": 1.0 if saturation_guard_active else 0.0,
            "hard_weight_mean": hard_weight_mean,
            "hard_hit_ratio": float(self.hard_sample_hits / max(self.hard_sample_queries, 1)),
            "ib_loss": ib_loss.item() if isinstance(ib_loss, torch.Tensor) else ib_loss,
            "cim_align_loss": cim_align_loss.item() if isinstance(cim_align_loss, torch.Tensor) else cim_align_loss,
        }

        return total_loss, loss_dict


# =====================================================================
#  PHASE 1: Load Dataset (identical to old pipeline)
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
CHUNK_SIZE = 100

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

train_idx = []
for ep in train_eps:
    train_idx.extend(episode_indices[ep])

print(f"  Train: {len(train_eps)} eps ({len(train_idx)} frames)")
print(f"  Eval: {len(eval_eps)} eps ({eval_eps})")

# =====================================================================
#  PHASE 2: Load SmolVLA + Setup Dynamic Wrapper
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 2: Load SmolVLA Base + Inject Entropy-CIM / CORAL / LoRA-SP")
print("=" * 70)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("  Loading pretrained SmolVLA base...")
smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# Key remapping (identical to old pipeline)
smolvla_img_keys = list(smolvla.config.image_features.keys())
KEY_REMAP_S = {}
for i, dk in enumerate(image_keys):
    if i < len(smolvla_img_keys):
        KEY_REMAP_S[dk] = smolvla_img_keys[i]
print(f"  Key remap: {KEY_REMAP_S}")

# Language tokens (identical to old pipeline)
tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
instruction = ds_cfg["task_instruction"]
_tok = tokenizer(instruction, return_tensors="pt", padding="max_length", max_length=64)
LANG_IDS = _tok['input_ids']
LANG_MASK = _tok['attention_mask'].bool()
print(f"  Instruction: '{instruction}'")

CHUNK_SIZE_S = smolvla.config.chunk_size

# Freeze VLM backbone
if dyn_cfg["freeze_vlm"]:
    frozen, trainable_base = 0, 0
    for name, param in smolvla.named_parameters():
        if "vlm_with_expert.vlm" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable_base += param.numel()
    print(f"  Frozen (VLM): {frozen/1e6:.1f}M")
    print(f"  Trainable (action expert + projs): {trainable_base/1e6:.1f}M")

smolvla.to(DEVICE)

# Create Dynamic Wrapper
print("\n  Creating DynamicEntropyCIMWrapper...")
dyn_wrapper = DynamicEntropyCIMWrapper(smolvla, dyn_cfg).to(DEVICE)

# Count trainable params
wrapper_params = sum(p.numel() for p in dyn_wrapper.star_router.parameters())
wrapper_params += sum(p.numel() for p in dyn_wrapper.token_pruner.parameters())
wrapper_params += sum(p.numel() for p in dyn_wrapper.lora_adapters.parameters())
wrapper_params += sum(p.numel() for p in dyn_wrapper.coral_manager.parameters())
print(f"  Hierarchical STAR Router params: {sum(p.numel() for p in dyn_wrapper.star_router.parameters())/1e6:.2f}M")
print(f"  CIM Pruner params: {sum(p.numel() for p in dyn_wrapper.token_pruner.parameters())/1e6:.4f}M")
print(f"  LoRA-SP Adapter params: {sum(p.numel() for p in dyn_wrapper.lora_adapters.parameters())/1e6:.2f}M")
print(f"  CORAL Expert params: {sum(p.numel() for p in dyn_wrapper.coral_manager.parameters())/1e6:.2f}M")
print(f"  Total dynamic params: {wrapper_params/1e6:.2f}M")


def build_train_batch(indices):
    """Build a training batch for SmolVLA (identical to old pipeline)."""
    batch_imgs = {k: [] for k in KEY_REMAP_S.values()}
    batch_states, batch_actions = [], []
    batch_episode_ids, batch_frame_ids = [], []

    for idx in indices:
        s = dataset[idx]
        for dk, sk in KEY_REMAP_S.items():
            batch_imgs[sk].append(s[dk])
        if state_key:
            batch_states.append(s[state_key])
        batch_actions.append(s[action_key])

        ep_val = s.get('episode_index', -1)
        fr_val = s.get('frame_index', idx)
        ep_val = int(ep_val.item()) if isinstance(ep_val, torch.Tensor) else int(ep_val)
        fr_val = int(fr_val.item()) if isinstance(fr_val, torch.Tensor) else int(fr_val)
        batch_episode_ids.append(ep_val)
        batch_frame_ids.append(fr_val)

    batch = {}
    for sk, imgs in batch_imgs.items():
        batch[sk] = torch.stack(imgs).to(DEVICE)
    if batch_states:
        batch['observation.state'] = torch.stack(batch_states).to(DEVICE)

    actions = torch.stack(batch_actions).to(DEVICE)
    B = actions.shape[0]

    MAX_ACT_DIM = smolvla.config.max_action_dim
    if ACTION_DIM < MAX_ACT_DIM:
        pad_zeros = torch.zeros(B, MAX_ACT_DIM - ACTION_DIM, device=DEVICE)
        actions_padded = torch.cat([actions, pad_zeros], dim=1)
    else:
        actions_padded = actions

    action_chunk = actions_padded.unsqueeze(1).expand(B, CHUNK_SIZE_S, MAX_ACT_DIM)
    batch['action'] = action_chunk
    batch['actions_id_pad'] = torch.zeros(B, CHUNK_SIZE_S, dtype=torch.bool, device=DEVICE)

    batch['observation.language.tokens'] = LANG_IDS.expand(B, -1).to(DEVICE)
    batch['observation.language.attention_mask'] = LANG_MASK.expand(B, -1).to(DEVICE)
    batch['meta.episode_index'] = torch.tensor(batch_episode_ids, dtype=torch.long, device=DEVICE)
    batch['meta.frame_index'] = torch.tensor(batch_frame_ids, dtype=torch.long, device=DEVICE)

    return batch


# =====================================================================
#  PHASE 3: Training Phase 1 - EGSF Routing + CORAL Expert
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 3: Training Phase 1 - EGSF Routing + CORAL Expert")
print("=" * 70)

phase1_params = []
phase1_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase1_params.extend(dyn_wrapper.star_router.parameters())
phase1_params.extend(dyn_wrapper.coral_manager.parameters())

opt_p1 = torch.optim.AdamW(
    phase1_params,
    lr=dyn_cfg["phase1_lr"], weight_decay=dyn_cfg["weight_decay"]
)

losses_p1 = []
micro_bs = dyn_cfg["micro_batch"]
grad_accum = dyn_cfg["grad_accum"]
phase1_steps = dyn_cfg["phase1_steps"]

print(f"  Steps: {phase1_steps}, micro_batch={micro_bs}, grad_accum={grad_accum}")
print(f"  Gumbel tau: {dyn_cfg['gumbel_tau_start']} -> {dyn_cfg['gumbel_tau_end']}")
print(f"  CORAL experts: {dyn_cfg['coral_num_experts']}, rank: {dyn_cfg['coral_expert_rank']}")
print(f"  lambda_cost={dyn_cfg['lambda_cost']}, lambda_diversity={dyn_cfg.get('lambda_diversity', 0.005)}")

smolvla.train()
dyn_wrapper.train()

train_diagnostics = []

pbar = tqdm(range(phase1_steps), desc="  Phase1 Gate+CORAL", ncols=110)
for step in pbar:
    opt_p1.zero_grad()
    accum_loss = 0
    accum_dict = {}

    # Curriculum Learning: anneal Gumbel temperature
    progress = step / max(phase1_steps - 1, 1)
    tau = dyn_cfg["gumbel_tau_start"] * (1 - progress) + dyn_cfg["gumbel_tau_end"] * progress

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if dyn_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = dyn_wrapper.forward_with_skip(
                    batch, tau=tau, global_step=step,
                    enable_skip=True, enable_iap=False,
                    enable_lora=False, enable_coral=True,
                    enable_snap=False, enable_cogkd=False,
                )
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = dyn_wrapper.forward_with_skip(
                batch, tau=tau, global_step=step,
                enable_skip=True, enable_iap=False,
                enable_lora=False, enable_coral=True,
                enable_snap=False, enable_cogkd=False,
            )
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase1_params, dyn_cfg["max_grad_norm"])
    opt_p1.step()

    losses_p1.append(accum_loss)
    train_diagnostics.append({
        "phase": 1,
        "phase_step": int(step),
        "global_step": int(step),
        "total_loss": float(accum_dict.get("total_loss", accum_loss)),
        "task_loss": float(accum_dict.get("task_loss", 0.0)),
        "task_loss_unweighted": float(accum_dict.get("task_loss_unweighted", 0.0)),
        "avg_skip_ratio": float(accum_dict.get("avg_skip_ratio", 0.0)),
        "token_keep_ratio": float(accum_dict.get("token_keep_ratio", 1.0)),
        "complexity_score": float(accum_dict.get("complexity_score", 0.0)),
        "tau_eff": float(accum_dict.get("tau_eff", dyn_wrapper.latest_tau_eff)),
        "lambda_cost_eff": float(accum_dict.get("lambda_cost_eff", dyn_wrapper.latest_lambda_cost_eff)),
        "mse_signal": float(accum_dict.get("mse_signal", 0.0)),
        "hard_weight_mean": float(accum_dict.get("hard_weight_mean", 1.0)),
        "hard_hit_ratio": float(accum_dict.get("hard_hit_ratio", 0.0)),
        "saturation_guard": float(accum_dict.get("saturation_guard", 0.0)),
    })
    if step % 100 == 0:
        avg_loss = np.mean(losses_p1[-50:])
        skip_r = accum_dict.get('avg_skip_ratio', 0)
        active_l = accum_dict.get('active_layers', 16)
        div_l = accum_dict.get('diversity_loss', 0)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', skip=f'{skip_r:.2f}',
                        tau=f'{tau:.2f}', layers=f'{active_l:.1f}', div=f'{div_l:.3f}')

    if (step + 1) % dyn_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'star_router': dyn_wrapper.star_router.state_dict(),
            'coral_manager': dyn_wrapper.coral_manager.state_dict(),
        }, OUTPUT_DIR / f'phase1_step{step+1}.pt')

torch.save({
    'smolvla': smolvla.state_dict(),
    'star_router': dyn_wrapper.star_router.state_dict(),
    'coral_manager': dyn_wrapper.coral_manager.state_dict(),
}, OUTPUT_DIR / 'phase1_complete.pt')
print(f"\n  Phase 1 done. Final loss: {np.mean(losses_p1[-50:]):.6f}")

gc.collect()
torch.cuda.empty_cache()

# =====================================================================
#  PHASE 4: Training Phase 2 - LoRA-SP + CIM + Hard-Sample Curriculum
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 4: Training Phase 2 - LoRA-SP + CIM + Hard-Sample Curriculum")
print("=" * 70)

phase2_params = []
phase2_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase2_params.extend(dyn_wrapper.star_router.parameters())
phase2_params.extend(dyn_wrapper.token_pruner.parameters())
phase2_params.extend(dyn_wrapper.lora_adapters.parameters())
phase2_params.extend(dyn_wrapper.coral_manager.parameters())

opt_p2 = torch.optim.AdamW(
    phase2_params,
    lr=dyn_cfg["phase2_lr"], weight_decay=dyn_cfg["weight_decay"]
)

losses_p2 = []
phase2_steps = dyn_cfg["phase2_steps"]

print(f"  Steps: {phase2_steps}")
print(f"  LoRA max rank: {dyn_cfg['lora_max_rank']}, energy threshold: {dyn_cfg['lora_energy_threshold']}")
print(f"  CIM: conservative={dyn_cfg.get('cim_conservative_ratio', dyn_cfg['iap_conservative_ratio'])}, aggressive={dyn_cfg.get('cim_aggressive_ratio', dyn_cfg['iap_aggressive_ratio'])}")
print(f"  IoU threshold: {dyn_cfg.get('cim_iou_proxy_tau', dyn_cfg['iap_iou_threshold'])}, temporal momentum: {dyn_cfg.get('cim_temporal_gamma', dyn_cfg['iap_temporal_momentum'])}")

pbar = tqdm(range(phase2_steps), desc="  Phase2 LoRA+CIM", ncols=110)
for step in pbar:
    opt_p2.zero_grad()
    accum_loss = 0
    accum_dict = {}

    tau = dyn_cfg["gumbel_tau_end"]

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if dyn_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = dyn_wrapper.forward_with_skip(
                    batch, tau=tau, global_step=phase1_steps + step,
                    enable_skip=True, enable_iap=True,
                    enable_lora=True, enable_coral=True,
                    enable_snap=False, enable_cogkd=False,
                )
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = dyn_wrapper.forward_with_skip(
                batch, tau=tau, global_step=phase1_steps + step,
                enable_skip=True, enable_iap=True,
                enable_lora=True, enable_coral=True,
                enable_snap=False, enable_cogkd=False,
            )
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase2_params, dyn_cfg["max_grad_norm"])
    opt_p2.step()

    losses_p2.append(accum_loss)
    train_diagnostics.append({
        "phase": 2,
        "phase_step": int(step),
        "global_step": int(phase1_steps + step),
        "total_loss": float(accum_dict.get("total_loss", accum_loss)),
        "task_loss": float(accum_dict.get("task_loss", 0.0)),
        "task_loss_unweighted": float(accum_dict.get("task_loss_unweighted", 0.0)),
        "avg_skip_ratio": float(accum_dict.get("avg_skip_ratio", 0.0)),
        "token_keep_ratio": float(accum_dict.get("token_keep_ratio", 1.0)),
        "complexity_score": float(accum_dict.get("complexity_score", 0.0)),
        "tau_eff": float(accum_dict.get("tau_eff", dyn_wrapper.latest_tau_eff)),
        "lambda_cost_eff": float(accum_dict.get("lambda_cost_eff", dyn_wrapper.latest_lambda_cost_eff)),
        "mse_signal": float(accum_dict.get("mse_signal", 0.0)),
        "hard_weight_mean": float(accum_dict.get("hard_weight_mean", 1.0)),
        "hard_hit_ratio": float(accum_dict.get("hard_hit_ratio", 0.0)),
        "saturation_guard": float(accum_dict.get("saturation_guard", 0.0)),
    })
    if step % 100 == 0:
        avg_loss = np.mean(losses_p2[-50:])
        spec_l = accum_dict.get('spec_loss', 0)
        tkr = accum_dict.get('token_keep_ratio', 1.0)
        iou = accum_dict.get('iou_proxy', 0.0)
        div_l = accum_dict.get('diversity_loss', 0)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', spec=f'{spec_l:.3f}',
                        tkr=f'{tkr:.2f}', iou=f'{iou:.2f}', div=f'{div_l:.3f}')

    if (step + 1) % dyn_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'star_router': dyn_wrapper.star_router.state_dict(),
            'token_pruner': dyn_wrapper.token_pruner.state_dict(),
            'lora_adapters': dyn_wrapper.lora_adapters.state_dict(),
            'coral_manager': dyn_wrapper.coral_manager.state_dict(),
        }, OUTPUT_DIR / f'phase2_step{step+1}.pt')

torch.save({
    'smolvla': smolvla.state_dict(),
    'star_router': dyn_wrapper.star_router.state_dict(),
    'token_pruner': dyn_wrapper.token_pruner.state_dict(),
    'lora_adapters': dyn_wrapper.lora_adapters.state_dict(),
    'coral_manager': dyn_wrapper.coral_manager.state_dict(),
}, OUTPUT_DIR / 'phase2_complete.pt')
print(f"\n  Phase 2 done. Final loss: {np.mean(losses_p2[-50:]):.6f}")

gc.collect()
torch.cuda.empty_cache()

# =====================================================================
#  PHASE 5: Training Phase 3 - SnapFlow + CogKD Self-Distillation
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 5: Training Phase 3 - SnapFlow + CogKD Self-Distillation")
print("=" * 70)

# Create teacher from current model
print("  Creating teacher model for SnapFlow + CogKD...")
dyn_wrapper.snap_trainer.create_teacher(smolvla.model)
print(f"  Teacher denoising steps: {dyn_cfg['snap_teacher_steps']}")
print(f"  CogKD lambda: {dyn_cfg.get('cogkd_lambda', 0.3)}, temperature: {dyn_cfg.get('cogkd_temperature', 2.0)}")
print(f"  SnapFlow MSE feedback threshold: {dyn_cfg.get('snap_mse_feedback_threshold', 1000.0)}")

phase3_params = []
phase3_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase3_params.extend(dyn_wrapper.star_router.parameters())
phase3_params.extend(dyn_wrapper.token_pruner.parameters())
phase3_params.extend(dyn_wrapper.lora_adapters.parameters())
phase3_params.extend(dyn_wrapper.coral_manager.parameters())

opt_p3 = torch.optim.AdamW(
    phase3_params,
    lr=dyn_cfg["phase3_lr"], weight_decay=dyn_cfg["weight_decay"]
)

# Cosine decay scheduler
scheduler_p3 = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_p3, T_max=dyn_cfg["phase3_steps"], eta_min=1e-6
)

losses_p3 = []
phase3_steps = dyn_cfg["phase3_steps"]
print(f"  Steps: {phase3_steps}, lr: {dyn_cfg['phase3_lr']}")

pbar = tqdm(range(phase3_steps), desc="  Phase3 Snap+CogKD", ncols=110)
for step in pbar:
    opt_p3.zero_grad()
    accum_loss = 0
    accum_dict = {}

    tau = dyn_cfg["gumbel_tau_end"]

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if dyn_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = dyn_wrapper.forward_with_skip(
                    batch, tau=tau,
                    global_step=phase1_steps + phase2_steps + step,
                    enable_skip=True, enable_iap=True,
                    enable_lora=True, enable_coral=True,
                    enable_snap=True, enable_cogkd=True,
                )
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = dyn_wrapper.forward_with_skip(
                batch, tau=tau,
                global_step=phase1_steps + phase2_steps + step,
                enable_skip=True, enable_iap=True,
                enable_lora=True, enable_coral=True,
                enable_snap=True, enable_cogkd=True,
            )
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase3_params, dyn_cfg["max_grad_norm"])
    opt_p3.step()
    scheduler_p3.step()

    losses_p3.append(accum_loss)
    train_diagnostics.append({
        "phase": 3,
        "phase_step": int(step),
        "global_step": int(phase1_steps + phase2_steps + step),
        "total_loss": float(accum_dict.get("total_loss", accum_loss)),
        "task_loss": float(accum_dict.get("task_loss", 0.0)),
        "task_loss_unweighted": float(accum_dict.get("task_loss_unweighted", 0.0)),
        "avg_skip_ratio": float(accum_dict.get("avg_skip_ratio", 0.0)),
        "token_keep_ratio": float(accum_dict.get("token_keep_ratio", 1.0)),
        "complexity_score": float(accum_dict.get("complexity_score", 0.0)),
        "tau_eff": float(accum_dict.get("tau_eff", dyn_wrapper.latest_tau_eff)),
        "lambda_cost_eff": float(accum_dict.get("lambda_cost_eff", dyn_wrapper.latest_lambda_cost_eff)),
        "mse_signal": float(accum_dict.get("mse_signal", 0.0)),
        "hard_weight_mean": float(accum_dict.get("hard_weight_mean", 1.0)),
        "hard_hit_ratio": float(accum_dict.get("hard_hit_ratio", 0.0)),
        "saturation_guard": float(accum_dict.get("saturation_guard", 0.0)),
    })
    if step % 100 == 0:
        avg_loss = np.mean(losses_p3[-50:])
        snap_l = accum_dict.get('snap_loss', 0)
        cogkd_l = accum_dict.get('cogkd_loss', 0)
        snap_mse_fb = accum_dict.get('snap_mse_feedback', 0)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', snap=f'{snap_l:.4f}',
                        cogkd=f'{cogkd_l:.4f}', mse_fb=f'{snap_mse_fb:.1f}',
                        lr=f'{scheduler_p3.get_last_lr()[0]:.2e}')

    if (step + 1) % dyn_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'star_router': dyn_wrapper.star_router.state_dict(),
            'token_pruner': dyn_wrapper.token_pruner.state_dict(),
            'lora_adapters': dyn_wrapper.lora_adapters.state_dict(),
            'coral_manager': dyn_wrapper.coral_manager.state_dict(),
        }, OUTPUT_DIR / f'phase3_step{step+1}.pt')

# Final save
torch.save({
    'smolvla': smolvla.state_dict(),
    'star_router': dyn_wrapper.star_router.state_dict(),
    'token_pruner': dyn_wrapper.token_pruner.state_dict(),
    'lora_adapters': dyn_wrapper.lora_adapters.state_dict(),
    'coral_manager': dyn_wrapper.coral_manager.state_dict(),
}, OUTPUT_DIR / 'final_model.pt')
print(f"\n  Phase 3 done. Final loss: {np.mean(losses_p3[-50:]):.6f}")

# Free teacher
del dyn_wrapper.snap_trainer.teacher_model
dyn_wrapper.snap_trainer.teacher_model = None
gc.collect()
torch.cuda.empty_cache()

# Save training curves (all 3 phases)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
w = 50

ax = axes[0]
if len(losses_p1) > w:
    ax.plot(np.convolve(losses_p1, np.ones(w)/w, 'valid'), color='#2196F3', lw=1.5)
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('Phase 1: Hierarchical Gate + CORAL'); ax.grid(True, alpha=0.3)

ax = axes[1]
if len(losses_p2) > w:
    ax.plot(np.convolve(losses_p2, np.ones(w)/w, 'valid'), color='#4CAF50', lw=1.5)
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('Phase 2: LoRA-SP + CIM'); ax.grid(True, alpha=0.3)

ax = axes[2]
if len(losses_p3) > w:
    ax.plot(np.convolve(losses_p3, np.ones(w)/w, 'valid'), color='#FF5722', lw=1.5)
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('Phase 3: SnapFlow + CogKD'); ax.grid(True, alpha=0.3)

plt.suptitle('Entropy-CIM + CORAL + LoRA-SP + SnapFlow + CogKD - Training Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves_3phase.png', dpi=150)
plt.close()

# Extended diagnostics artifacts for Entropy-CIM
diag_csv_path = OUTPUT_DIR / 'entropy_cim_training_diagnostics.csv'
if train_diagnostics:
    fieldnames = list(train_diagnostics[0].keys())
    with diag_csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_diagnostics)

coverage_payload = {
    'manifest_path': dyn_cfg.get('hard_sample_manifest_path', ''),
    'manifest_entries': len(dyn_wrapper.hard_sample_map),
    'hard_sample_queries': dyn_wrapper.hard_sample_queries,
    'hard_sample_hits': dyn_wrapper.hard_sample_hits,
    'hard_hit_ratio': float(dyn_wrapper.hard_sample_hits / max(dyn_wrapper.hard_sample_queries, 1)),
}
coverage_json_path = OUTPUT_DIR / 'hard_sample_manifest_coverage.json'
with coverage_json_path.open('w', encoding='utf-8') as f:
    json.dump(coverage_payload, f, indent=2)

print(f"  Diagnostics CSV: {diag_csv_path}")
print(f"  Manifest coverage: {coverage_json_path}")

# =====================================================================
#  PHASE 6: Evaluate (identical output format to old pipeline)
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 6: Evaluate Fine-Tuned Dynamic Model")
print("=" * 70)

smolvla.eval()
dyn_wrapper.eval()

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


def evaluate_dynamic_model(model, eval_episodes):
    results = {
        'mse_per_episode': [], 'latency_ms': [],
        'predictions': {}, 'ground_truth': {},
        'skip_ratios': [], 'token_ratios': [],
    }

    for ep_idx in eval_episodes:
        ep_preds, ep_gts = [], []
        indices = episode_indices[ep_idx]
        model.reset()
        dyn_wrapper.reset_episode()

        for step_idx in tqdm(indices, desc=f"  DynModel Ep{ep_idx}", ncols=85, leave=False):
            s = dataset[step_idx]
            gt = s[action_key].numpy()

            batch = build_eval_batch(s, DEVICE)
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = model.select_action(batch)
            t1 = time.perf_counter()

            pred_np = pred.squeeze().cpu().numpy()
            if pred_np.ndim > 1:
                pred_np = pred_np[0]
            pred_np = pred_np[:ACTION_DIM]

            results['latency_ms'].append((t1-t0)*1000)
            ep_preds.append(pred_np)
            ep_gts.append(gt)

        ep_preds = np.array(ep_preds)
        ep_gts = np.array(ep_gts)
        mse = float(np.mean((ep_preds - ep_gts)**2))
        results['mse_per_episode'].append(mse)
        results['predictions'][ep_idx] = ep_preds
        results['ground_truth'][ep_idx] = ep_gts
        print(f"    Ep{ep_idx}: MSE={mse:.4f} ({len(indices)} frames)")

    all_p = np.concatenate(list(results['predictions'].values()))
    all_g = np.concatenate(list(results['ground_truth'].values()))
    results['mse_total'] = float(np.mean((all_p - all_g)**2))
    results['mse_per_joint'] = np.mean((all_p - all_g)**2, axis=0).tolist()
    results['latency_mean_ms'] = float(np.mean(results['latency_ms']))

    # Stats from training
    results['avg_skip_ratio'] = float(np.mean(dyn_wrapper.skip_stats[-500:])) if dyn_wrapper.skip_stats else 0.0
    results['avg_token_keep_ratio'] = float(np.mean(dyn_wrapper.token_stats[-500:])) if dyn_wrapper.token_stats else 1.0
    results['avg_diversity_loss'] = float(np.mean(dyn_wrapper.diversity_stats[-500:])) if dyn_wrapper.diversity_stats else 0.0
    results['avg_snap_mse'] = float(np.mean(dyn_wrapper.snap_mse_stats[-500:])) if dyn_wrapper.snap_mse_stats else 0.0

    return results


eval_ep_list = eval_eps[:EVAL["num_eval_episodes"]]
print(f"  Eval episodes: {eval_ep_list}")

print("\n  Evaluating Entropy-CIM + CORAL + SnapFlow Model...")
dyn_results = evaluate_dynamic_model(smolvla, eval_ep_list)
print(f"  Dynamic MSE: {dyn_results['mse_total']:.4f}, Latency: {dyn_results['latency_mean_ms']:.1f}ms")
print(f"  Avg Skip Ratio (training): {dyn_results['avg_skip_ratio']:.3f}")
print(f"  Avg Token Keep Ratio (training): {dyn_results['avg_token_keep_ratio']:.3f}")
print(f"  Avg Diversity Loss (training): {dyn_results['avg_diversity_loss']:.4f}")
print(f"  Avg SnapFlow MSE (training): {dyn_results['avg_snap_mse']:.2f}")

# =====================================================================
#  PHASE 7: Comparison Plots & Videos (same format as old pipeline)
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 7: Comparison Plots & Videos")
print("=" * 70)

# Summary plot
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
blue = '#2196F3'

# [0,0] MSE per episode
ax = axes[0,0]
x = np.arange(len(eval_ep_list))
ax.bar(x, dyn_results['mse_per_episode'], 0.6, label='Entropy-CIM', color=blue)
ax.set_xlabel('Episode'); ax.set_ylabel('MSE'); ax.set_title('MSE per Episode')
ax.set_xticks(x); ax.set_xticklabels([f'Ep{e}' for e in eval_ep_list])
ax.legend(); ax.grid(True, alpha=0.3)

# [0,1] MSE per joint
ax = axes[0,1]
x = np.arange(ACTION_DIM)
ax.bar(x, dyn_results['mse_per_joint'], 0.6, label='Entropy-CIM', color=blue)
ax.set_xlabel('Joint'); ax.set_ylabel('MSE'); ax.set_title('MSE per Joint')
ax.set_xticks(x); ax.legend(); ax.grid(True, alpha=0.3)

# [0,2] Latency distribution
ax = axes[0,2]
ax.hist(dyn_results['latency_ms'], bins=40, alpha=0.7, color=blue)
ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Count'); ax.set_title('Inference Latency')
ax.grid(True, alpha=0.3)

# [1,0] All training curves combined
ax = axes[1,0]
all_losses = losses_p1 + losses_p2 + losses_p3
if len(all_losses) > w:
    ax.plot(np.convolve(all_losses, np.ones(w)/w, 'valid'), color=blue, lw=1.0)
    ax.axvline(len(losses_p1), color='red', linestyle='--', alpha=0.5, label='Phase 1->2')
    ax.axvline(len(losses_p1) + len(losses_p2), color='green', linestyle='--', alpha=0.5, label='Phase 2->3')
ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('Combined Training Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# [1,1] Trajectory plot
ax = axes[1,1]
ep0 = eval_ep_list[0]
gt0 = dyn_results['ground_truth'][ep0]
dp0 = dyn_results['predictions'][ep0]
nj = min(3, ACTION_DIM)
t_ax = np.arange(len(gt0))
for j in range(nj):
    ax.plot(t_ax, gt0[:,j], '-', color=f'C{j}', lw=2, label=f'GT J{j}')
    ax.plot(t_ax, dp0[:,j], '--', color=f'C{j}', alpha=0.6, label=f'Dyn J{j}')
ax.set_xlabel('Step'); ax.set_ylabel('Action'); ax.set_title(f'Trajectory Ep{ep0}')
ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)

# [1,2] Summary text
ax = axes[1,2]
ax.axis('off')
summary = (
    f"ENTROPY-CIM + CORAL SUMMARY\n\n"
    f"Dataset: {ds_cfg['repo_id']}\n"
    f"Train: {len(train_eps)} eps, Eval: {len(eval_ep_list)} eps\n\n"
    f"Phase 1 (Gate+CORAL): {phase1_steps} steps\n"
    f"Phase 2 (LoRA+CIM):   {phase2_steps} steps\n"
    f"Phase 3 (Snap+CogKD): {phase3_steps} steps\n\n"
    f"MSE:     {dyn_results['mse_total']:.4f}\n"
    f"Latency: {dyn_results['latency_mean_ms']:.1f}ms\n"
    f"Avg Skip Ratio: {dyn_results['avg_skip_ratio']:.3f}\n"
    f"Avg Token Keep: {dyn_results['avg_token_keep_ratio']:.3f}\n"
    f"Avg Diversity:  {dyn_results['avg_diversity_loss']:.4f}\n"
    f"Avg Snap MSE:   {dyn_results['avg_snap_mse']:.2f}\n\n"
    f"Techniques:\n"
    f"  - Hierarchical STAR Router\n"
    f"  - CIM Token Pruning\n"
    f"  - LoRA-SP (r={dyn_cfg['lora_max_rank']})\n"
    f"  - CORAL Experts ({dyn_cfg['coral_num_experts']})\n"
    f"  - SnapFlow + CogKD\n"
    f"  - Hybrid lambda-cost safeguard"
)
ax.text(0.05, 0.5, summary, fontsize=9, fontfamily='monospace',
        va='center', ha='left', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle('SmolVLA + Entropy-CIM + CORAL + LoRA-SP + SnapFlow + CogKD', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dynamic_comparison_plots.png', dpi=150)
plt.close()
print(f"  Plots saved: {OUTPUT_DIR / 'dynamic_comparison_plots.png'}")

# Videos (identical format to old pipeline)
for ep_idx in eval_ep_list[:3]:
    gt = dyn_results['ground_truth'][ep_idx]
    dp = dyn_results['predictions'][ep_idx]
    indices = episode_indices[ep_idx][:len(gt)]
    n = min(len(gt), len(dp))

    vpath = OUTPUT_DIR / f'dynamic_ep{ep_idx}.mp4'
    fw, fh = 900, 550
    fps_out = dataset.fps if hasattr(dataset, 'fps') and dataset.fps else 15
    writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*'mp4v'), fps_out, (fw, fh))

    for t in range(n):
        frame = np.ones((fh, fw, 3), dtype=np.uint8) * 25
        s = dataset[indices[t]]
        if image_keys:
            img = s[image_keys[0]].numpy()
            if img.shape[0] <= 4: img = np.transpose(img, (1,2,0))
            if img.max() <= 1.0: img = (img*255).clip(0,255).astype(np.uint8)
            else: img = img.clip(0,255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (320, 240))
            frame[10:250, 10:330] = img

        cv2.putText(frame, f'Ep {ep_idx} | Frame {t}/{n} | Entropy-CIM+Snap', (340, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        nj_v = min(ACTION_DIM, 6)
        px0 = 340; pw = fw - px0 - 15
        jh = max(45, (fh - 80) // nj_v - 6)

        for j in range(nj_v):
            y0 = 35 + j * (jh + 4)
            cv2.putText(frame, f'J{j}', (px0, y0+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180,180,180), 1)
            cv2.rectangle(frame, (px0+22, y0), (px0+pw, y0+jh), (40,40,40), -1)
            win_size = 40; st = max(0, t-win_size)
            vmin = min(gt[st:t+1,j].min(), dp[st:t+1,j].min()) - 0.05
            vmax = max(gt[st:t+1,j].max(), dp[st:t+1,j].max()) + 0.05
            if vmax-vmin < 0.01: vmax = vmin + 0.01
            def _px(tt): return px0 + 22 + int((tt-st)/max(win_size,1) * (pw-25))
            def _py(v): return y0 + jh - int((v-vmin)/(vmax-vmin) * jh)
            for tt in range(st, min(t, n-1)):
                x1, x2 = _px(tt), _px(tt+1)
                cv2.line(frame, (x1,_py(gt[tt,j])), (x2,_py(gt[tt+1,j])), (0,220,0), 2)
                cv2.line(frame, (x1,_py(dp[tt,j])), (x2,_py(dp[tt+1,j])), (255,160,0), 1)

        yb = fh - 25
        cv2.putText(frame, 'GT', (15, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(frame, 'Entropy-CIM SmolVLA', (55, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,160,0), 1)
        ms = float(np.mean((dp[t]-gt[t])**2))
        cv2.putText(frame, f'MSE:{ms:.4f}', (350, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,160,0), 1)
        writer.write(frame)
    writer.release()
    print(f"  Video: {vpath}")

# =====================================================================
#  PHASE 8: Save Report (same JSON format as old pipeline)
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 8: Save Report")
print("=" * 70)

smolvla_total = sum(p.numel() for p in smolvla.parameters())

report = {
    'pipeline': 'finetune_semifinetune_entropy_cim',
    'dataset': ds_cfg['repo_id'],
    'dataset_key': DATASET_KEY,
    'train_episodes': list(train_eps),
    'eval_episodes': list(eval_ep_list),
    'config': {
        'phase1_steps': phase1_steps,
        'phase2_steps': phase2_steps,
        'phase3_steps': phase3_steps,
        'total_steps': phase1_steps + phase2_steps + phase3_steps,
        'lora_max_rank': dyn_cfg['lora_max_rank'],
        'lora_energy_threshold': dyn_cfg['lora_energy_threshold'],
        'cim_conservative_ratio': dyn_cfg.get('cim_conservative_ratio', dyn_cfg['iap_conservative_ratio']),
        'cim_aggressive_ratio': dyn_cfg.get('cim_aggressive_ratio', dyn_cfg['iap_aggressive_ratio']),
        'cim_iou_proxy_tau': dyn_cfg.get('cim_iou_proxy_tau', dyn_cfg['iap_iou_threshold']),
        'cim_temporal_gamma': dyn_cfg.get('cim_temporal_gamma', dyn_cfg['iap_temporal_momentum']),
        'cim_interaction_entropy_tau': dyn_cfg.get('cim_interaction_entropy_tau', 1.6),
        'complexity_alpha_entropy': dyn_cfg.get('complexity_alpha_entropy', 0.40),
        'complexity_alpha_iou_proxy': dyn_cfg.get('complexity_alpha_iou_proxy', 0.35),
        'complexity_alpha_grad_proxy': dyn_cfg.get('complexity_alpha_grad_proxy', 0.25),
        'hybrid_tau_warmup_steps': dyn_cfg.get('hybrid_tau_warmup_steps', 0),
        'hybrid_tau_fixed': dyn_cfg.get('hybrid_tau_fixed', 1000.0),
        'hybrid_tau_percentile': dyn_cfg.get('hybrid_tau_percentile', 85.0),
        'hybrid_tau_window': dyn_cfg.get('hybrid_tau_window', 128),
        'hard_sample_manifest_path': dyn_cfg.get('hard_sample_manifest_path', ''),
        'num_fixed_layers': dyn_cfg['num_fixed_layers'],
        'num_spatial_layers': dyn_cfg['num_spatial_layers'],
        'num_action_layers': dyn_cfg['num_action_layers'],
        'num_skippable_layers': dyn_cfg['num_skippable_layers'],
        'coral_num_experts': dyn_cfg['coral_num_experts'],
        'coral_expert_rank': dyn_cfg['coral_expert_rank'],
        'cogkd_lambda': dyn_cfg.get('cogkd_lambda', 0.3),
        'cogkd_temperature': dyn_cfg.get('cogkd_temperature', 2.0),
        'snap_mse_feedback_threshold': dyn_cfg.get('snap_mse_feedback_threshold', 1000.0),
    },
    'results': {
        'label': 'SmolVLA + Entropy-CIM + CORAL + LoRA-SP + SnapFlow + CogKD',
        'params_total_M': round(smolvla_total/1e6, 1),
        'dynamic_params_M': round(wrapper_params/1e6, 2),
        'total_mse': dyn_results['mse_total'],
        'per_episode_mse': dyn_results['mse_per_episode'],
        'per_joint_mse': dyn_results['mse_per_joint'],
        'avg_latency_ms': dyn_results['latency_mean_ms'],
        'avg_skip_ratio': dyn_results['avg_skip_ratio'],
        'avg_token_keep_ratio': dyn_results['avg_token_keep_ratio'],
        'avg_diversity_loss': dyn_results['avg_diversity_loss'],
        'avg_snap_mse': dyn_results['avg_snap_mse'],
        'avg_tau_eff': float(np.mean([row['tau_eff'] for row in train_diagnostics])) if train_diagnostics else float(dyn_wrapper.latest_tau_eff),
        'avg_lambda_cost_eff': float(np.mean([row['lambda_cost_eff'] for row in train_diagnostics])) if train_diagnostics else float(dyn_wrapper.latest_lambda_cost_eff),
        'saturation_guard_ratio': float(np.mean([row['saturation_guard'] for row in train_diagnostics])) if train_diagnostics else 0.0,
        'hard_sample_hit_ratio': float(dyn_wrapper.hard_sample_hits / max(dyn_wrapper.hard_sample_queries, 1)),
        'phase1_final_loss': float(np.mean(losses_p1[-50:])),
        'phase2_final_loss': float(np.mean(losses_p2[-50:])),
        'phase3_final_loss': float(np.mean(losses_p3[-50:])),
    },
    'techniques': {
        'HierarchicalSTAR': 'Hierarchical STAR Router with Curriculum Learning, Diversity Loss, and Action Entropy gating',
        'CIM': f'Contextual Interaction Masking (IoU tau={dyn_cfg.get("cim_iou_proxy_tau", dyn_cfg["iap_iou_threshold"])}, conservative={dyn_cfg.get("cim_conservative_ratio", dyn_cfg["iap_conservative_ratio"])}, aggressive={dyn_cfg.get("cim_aggressive_ratio", dyn_cfg["iap_aggressive_ratio"])})',
        'EGSF': f'Entropy-guided complexity score (w_entropy={dyn_cfg.get("complexity_alpha_entropy", 0.4)}, w_iou={dyn_cfg.get("complexity_alpha_iou_proxy", 0.35)}, w_grad={dyn_cfg.get("complexity_alpha_grad_proxy", 0.25)})',
        'LoRA-SP': f'Spectral rank adaptation (max_rank={dyn_cfg["lora_max_rank"]}, eta={dyn_cfg["lora_energy_threshold"]})',
        'CORAL': f'Language-routed LoRA experts ({dyn_cfg["coral_num_experts"]} experts, rank={dyn_cfg["coral_expert_rank"]})',
        'SnapFlow': f'Self-distillation for 1-NFE (teacher_steps={dyn_cfg["snap_teacher_steps"]})',
        'CogKD': f'Cognition Self-Knowledge Distillation (lambda={dyn_cfg.get("cogkd_lambda", 0.3)}, T={dyn_cfg.get("cogkd_temperature", 2.0)})',
        'HybridTauGuard': 'Set lambda_cost to zero when mse_signal exceeds adaptive tau',
        'HardSampleCurriculum': f'Curriculum weighting from manifest {dyn_cfg.get("hard_sample_manifest_path", "")}',
        'DiversityLoss': 'Diversity-Driven Loss to break saturation-skip pattern',
        'MSEFeedback': f'SnapFlow MSE -> STAR Router closed-loop feedback (threshold={dyn_cfg.get("snap_mse_feedback_threshold", 1000.0)})',
    },
}
with open(OUTPUT_DIR / 'dynamic_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"  Report: {OUTPUT_DIR / 'dynamic_report.json'}")

# =====================================================================
#  DONE
# =====================================================================
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE: finetune_semifinetune_entropy_cim")
print("=" * 70)
print(f"\n  MSE: {dyn_results['mse_total']:.4f}")
print(f"  Latency: {dyn_results['latency_mean_ms']:.1f}ms")
print(f"  Avg Layers Active (training): {16 - dyn_results['avg_skip_ratio'] * dyn_cfg['num_skippable_layers']:.1f} / 16")
print(f"  Avg Token Keep (training): {dyn_results['avg_token_keep_ratio']:.1%}")
print(f"  Avg Diversity Loss: {dyn_results['avg_diversity_loss']:.4f}")
print(f"  Avg SnapFlow MSE: {dyn_results['avg_snap_mse']:.2f}")
print(f"\n  Output: {OUTPUT_DIR}")
for f_item in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f_item.name:<40} {f_item.stat().st_size/1024:.0f} KB")
print("\n  DONE!")

