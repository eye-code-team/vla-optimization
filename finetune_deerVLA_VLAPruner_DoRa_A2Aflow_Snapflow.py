"""
Fine-Tune SmolVLA: DeeR-VLA + VLA-Pruner + DoRA + A2A Flow + SnapFlow + RS-CL
===============================================================================
Pipeline: finetune_deerVLA_VLAPruner_DoRa_A2Aflow_Snapflow
RTX 4070 SUPER (12GB) | fp16/bf16 | lerobot 0.4.4

Architecture Enhancements (Advanced):
  1. DeeR-VLA: Dynamic Early-Exit with multi-exit action heads
  2. VLA-Pruner: Dual-level (semantic + action) token pruning + temporal smooth
  3. DoRA: Weight-Decomposed Low-Rank Adaptation (magnitude + direction)
  4. A2A Flow: Action-to-Action flow matching (history-conditioned init)
  5. SnapFlow: Self-distillation for single-step denoising (1-NFE)
  6. RS-CL: Robot State-aware Contrastive Loss for representation alignment

Training Schedule (5 phases):
  Phase 1 (Steps 0-5K):      DeeR-VLA multi-exit training
  Phase 2 (Steps 5K-13K):    VLA-Pruner + DoRA integration
  Phase 3 (Steps 13K-18K):   A2A Flow Matching
  Phase 4 (Steps 18K-23K):   SnapFlow 1-NFE self-distillation
  Phase 5 (Steps 23K-26K):   RS-CL polish + joint fine-tuning
"""
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
adv_cfg = TRAINING["deer_vlap_dora_a2a_snap"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(f"d:/EyetechCode/results/deer_vlap_dora_a2a_snap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  SmolVLA Advanced Fine-Tune Pipeline")
print("  DeeR-VLA + VLA-Pruner + DoRA + A2A Flow + SnapFlow + RS-CL")
print("=" * 72)
print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB"
      if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
      else f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print(f"  Dataset: {ds_cfg['repo_id']}")
print(f"  Output: {OUTPUT_DIR}")
print()


# #####################################################################
#                     CUSTOM MODULES
# #####################################################################

# =====================================================================
#  1. DeeR-VLA: Dynamic Early-Exit Router
# =====================================================================
class DeeRVLARouter(nn.Module):
    """DeeR-VLA: Dynamic Early-Exit for Robotic VLA.

    Places lightweight action prediction heads at multiple exit points
    along the VLM depth. At each exit, checks action consistency with
    the previous exit. If consistent → early exit, saving compute.

    Exit Points (for 16-layer SmolLM2): layers 4, 8, 12, 16
    """

    def __init__(self, vlm_hidden_dim, expert_hidden_dim, action_dim,
                 num_exits=4, num_vlm_layers=16, chunk_size=50):
        super().__init__()
        self.num_exits = num_exits
        self.num_vlm_layers = num_vlm_layers
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Compute exit layer indices (evenly spaced)
        self.exit_layers = []
        step = num_vlm_layers // num_exits
        for i in range(num_exits):
            self.exit_layers.append((i + 1) * step - 1)
        # e.g. for 16 layers, 4 exits: [3, 7, 11, 15]

        # Lightweight action heads at each exit
        # Input: pooled hidden state from the VLM at that layer
        self.exit_heads = nn.ModuleList()
        for _ in range(num_exits):
            self.exit_heads.append(nn.Sequential(
                nn.Linear(vlm_hidden_dim, expert_hidden_dim),
                nn.SiLU(),
                nn.Linear(expert_hidden_dim, action_dim),
            ))

        # Confidence estimator at each exit (predicts if action is ready)
        self.confidence_heads = nn.ModuleList()
        for _ in range(num_exits):
            self.confidence_heads.append(nn.Sequential(
                nn.Linear(vlm_hidden_dim, vlm_hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(vlm_hidden_dim // 4, 1),
                nn.Sigmoid(),
            ))

    def compute_exit_actions(self, hidden_states_at_exits):
        """Compute action predictions at each exit point.

        Args:
            hidden_states_at_exits: list of (B, seq_len, hidden_dim) tensors,
                one per exit point.
        Returns:
            exit_actions: list of (B, action_dim) predicted actions
            exit_confidences: list of (B, 1) confidence scores
        """
        exit_actions = []
        exit_confidences = []

        for i, h in enumerate(hidden_states_at_exits):
            # Pool over sequence dimension
            h_pooled = h.mean(dim=1)  # (B, hidden_dim)

            action_pred = self.exit_heads[i](h_pooled)  # (B, action_dim)
            confidence = self.confidence_heads[i](h_pooled)  # (B, 1)

            exit_actions.append(action_pred)
            exit_confidences.append(confidence)

        return exit_actions, exit_confidences

    def compute_consistency(self, action_current, action_previous):
        """Compute action consistency (cosine similarity) between exits.

        Args:
            action_current: (B, action_dim)
            action_previous: (B, action_dim)
        Returns:
            consistency: (B, 1) in [0, 1]
        """
        cos_sim = F.cosine_similarity(action_current, action_previous, dim=-1)
        # Map from [-1, 1] to [0, 1]
        consistency = (cos_sim + 1.0) / 2.0
        return consistency.unsqueeze(-1)  # (B, 1)

    def compute_exit_loss(self, exit_actions, ground_truth_action,
                          loss_weights=None):
        """Weighted loss across all exit points.

        Args:
            exit_actions: list of (B, action_dim) predictions
            ground_truth_action: (B, action_dim) target
            loss_weights: list of floats, weight per exit
        Returns:
            total_exit_loss: scalar
            per_exit_losses: list of scalars
        """
        if loss_weights is None:
            loss_weights = [1.0 / self.num_exits] * self.num_exits

        total_loss = torch.tensor(0.0, device=exit_actions[0].device)
        per_exit_losses = []

        for i, (pred, w) in enumerate(zip(exit_actions, loss_weights)):
            # Trim ground truth to match prediction dim
            gt = ground_truth_action[:, :pred.shape[-1]]
            loss_i = F.mse_loss(pred, gt)
            per_exit_losses.append(loss_i.item())
            total_loss = total_loss + w * loss_i

        return total_loss, per_exit_losses

    def select_exit(self, exit_actions, exit_confidences, threshold=0.85):
        """Select which exit to use based on consistency + confidence.

        Used at inference time. Returns index of selected exit.

        Args:
            exit_actions: list of (B, action_dim)
            exit_confidences: list of (B, 1)
            threshold: minimum consistency for early exit
        Returns:
            selected_exit: int, index of chosen exit
            selected_action: (B, action_dim)
        """
        for i in range(1, len(exit_actions)):
            consistency = self.compute_consistency(
                exit_actions[i], exit_actions[i - 1]
            )
            confidence = exit_confidences[i]

            # Early exit if both consistency and confidence are high
            combined_score = consistency * 0.6 + confidence * 0.4
            if combined_score.mean().item() > threshold:
                return i, exit_actions[i]

        # Default: use last exit
        return len(exit_actions) - 1, exit_actions[-1]


# =====================================================================
#  2. VLA-Pruner: Dual-Level Token Pruning with Temporal Smoothing
# =====================================================================
class VLAPruner(nn.Module):
    """VLA-Pruner: Dual-level importance scoring for visual token pruning.

    Combines:
      1. Semantic importance: attention from language tokens to visual tokens
      2. Action importance: attention from action decode to visual tokens
      3. Temporal smoothing: EMA over time to avoid flickering

    Optionally supports VLA-IAP (Interaction-Aligned Pruning) mode
    where pruning ratio adapts based on robot-object proximity.
    """

    def __init__(self, semantic_weight=0.4, action_weight=0.6,
                 temporal_momentum=0.7, min_keep_ratio=0.25):
        super().__init__()
        self.semantic_weight = semantic_weight
        self.action_weight = action_weight
        self.temporal_momentum = temporal_momentum
        self.min_keep_ratio = min_keep_ratio

        # Learnable blend between semantic and action importance
        self.blend_weight = nn.Parameter(torch.tensor(0.0))  # sigmoid → ~0.5

        # State for temporal smoothing
        self._prev_importance = None

        # Velocity-based adaptive ratio (VLA-IAP inspired)
        self.velocity_threshold = nn.Parameter(torch.tensor(0.15))

    def reset_temporal(self):
        """Reset temporal smoothing state (call at episode boundary)."""
        self._prev_importance = None

    def compute_semantic_importance(self, attention_weights, num_visual_tokens,
                                    num_lang_start_idx, num_lang_end_idx):
        """Compute importance of visual tokens based on language attention.

        Args:
            attention_weights: (B, H, L, L) attention from VLM
            num_visual_tokens: int, number of visual tokens
            num_lang_start_idx: int, start index of language tokens
            num_lang_end_idx: int, end index of language tokens
        Returns:
            semantic_scores: (B, num_visual_tokens) importance scores
        """
        # Attention from language tokens to visual tokens
        # attention_weights[:, :, lang_range, vis_range]
        lang_to_vis = attention_weights[:, :,
                      num_lang_start_idx:num_lang_end_idx,
                      :num_visual_tokens]
        # Average over heads and language positions
        semantic_scores = lang_to_vis.mean(dim=(1, 2))  # (B, num_visual_tokens)
        return semantic_scores

    def compute_action_importance(self, action_attention_weights,
                                  num_visual_tokens):
        """Compute importance of visual tokens from action decode attention.

        Args:
            action_attention_weights: (B, H, chunk_size, prefix_len)
            num_visual_tokens: int
        Returns:
            action_scores: (B, num_visual_tokens) importance scores
        """
        # Action queries attending to visual token positions
        action_to_vis = action_attention_weights[:, :, :, :num_visual_tokens]
        action_scores = action_to_vis.mean(dim=(1, 2))  # (B, num_visual_tokens)
        return action_scores

    def forward(self, token_embeddings, token_mask,
                semantic_scores=None, action_scores=None,
                v_ee=None):
        """Prune visual tokens using dual-level importance.

        Args:
            token_embeddings: (B, N, D) visual token embeddings
            token_mask: (B, N) boolean mask
            semantic_scores: (B, N) from language attention (optional)
            action_scores: (B, N) from action decode attention (optional)
            v_ee: (B, 1) end-effector velocity (for VLA-IAP)
        Returns:
            pruned_embeddings: (B, K, D)
            pruned_mask: (B, K)
            keep_ratio: scalar
        """
        B, N, D = token_embeddings.shape
        device = token_embeddings.device

        # Compute blended importance
        blend = torch.sigmoid(self.blend_weight)  # learnable blend
        alpha_s = self.semantic_weight * blend
        alpha_a = self.action_weight * (1 - blend)

        if semantic_scores is not None and action_scores is not None:
            importance = alpha_s * semantic_scores + alpha_a * action_scores
        elif semantic_scores is not None:
            importance = semantic_scores
        elif action_scores is not None:
            importance = action_scores
        else:
            # Fallback: uniform importance
            importance = torch.ones(B, N, device=device)

        # Normalize importance to [0, 1]
        importance = importance - importance.min(dim=-1, keepdim=True).values
        imp_max = importance.max(dim=-1, keepdim=True).values
        importance = importance / (imp_max + 1e-8)

        # Temporal smoothing (EMA)
        if self._prev_importance is not None and self._prev_importance.shape == importance.shape:
            m = self.temporal_momentum
            importance = m * self._prev_importance.detach() + (1 - m) * importance
        self._prev_importance = importance.detach().clone()

        # Adaptive keep ratio via velocity (VLA-IAP inspired)
        if v_ee is not None:
            v_thr = torch.abs(self.velocity_threshold)
            # High velocity → aggressive pruning, low velocity → keep all
            ratio_per_sample = torch.where(
                v_ee > v_thr,
                torch.clamp(
                    self.min_keep_ratio + (1 - self.min_keep_ratio) * (v_thr / (v_ee + 1e-8)),
                    min=self.min_keep_ratio, max=1.0
                ),
                torch.ones_like(v_ee)
            )  # (B, 1)
        else:
            ratio_per_sample = torch.ones(B, 1, device=device)

        # Determine K (tokens to keep) — use minimum across batch for padding
        K = max(int(N * self.min_keep_ratio), 1)

        # Select top-K tokens by importance
        _, top_indices = importance.topk(K, dim=-1, sorted=False)
        top_indices_sorted, _ = top_indices.sort(dim=-1)

        # Gather selected tokens
        pruned_embeddings = torch.gather(
            token_embeddings, 1,
            top_indices_sorted.unsqueeze(-1).expand(-1, -1, D)
        )
        pruned_mask = torch.gather(token_mask, 1, top_indices_sorted)

        avg_keep_ratio = ratio_per_sample.mean()
        return pruned_embeddings, pruned_mask, avg_keep_ratio


# =====================================================================
#  3. DoRA: Weight-Decomposed Low-Rank Adaptation
# =====================================================================
class DoRAAdapter(nn.Module):
    """DoRA: Weight-Decomposed Low-Rank Adaptation.

    Decomposes pretrained weight W₀ into magnitude m and direction V:
        W₀ = m * (V / ||V||_c)

    Fine-tuning updates direction via low-rank ΔV = B @ A:
        W' = m * ((V + B@A) / ||V + B@A||_c) @ x

    This allows large directional changes with minimal magnitude change,
    making the learning behavior closer to full fine-tuning.
    """

    def __init__(self, original_linear, rank=64, alpha=1.0, dropout=0.05):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Extract pretrained weight
        W0 = original_linear.weight.data.clone()  # (out, in)

        # Decompose into magnitude and direction
        # ||W0||_c = column-wise norm (norm over output dimension)
        self.column_norm = W0.norm(dim=0, keepdim=True)  # (1, in)

        # Magnitude vector: learnable, initialized from W0 column norms transposed
        # m shape: (out_features,) — one magnitude per output neuron
        self.magnitude = nn.Parameter(
            W0.norm(dim=1)  # (out_features,) — row-wise norm
        )

        # Direction: keep original weight frozen as base direction
        self.register_buffer('V', W0.clone())  # (out, in) — frozen direction

        # Low-rank update for direction: ΔV = B @ A
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Optional dropout on ΔV
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Copy bias if exists
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        """
        Args:
            x: (..., in_features)
        Returns:
            output: (..., out_features)
        """
        # Compute direction update: ΔV = B @ A (with dropout)
        delta_V = self.dropout(self.lora_B @ self.lora_A) * self.scaling  # (out, in)

        # Updated direction matrix
        V_updated = self.V + delta_V  # (out, in)

        # Column-wise normalization of updated direction
        V_norm = V_updated / (V_updated.norm(dim=0, keepdim=True) + 1e-8)  # (out, in)

        # Apply magnitude scaling: W' = diag(m) @ V_normalized
        W_prime = self.magnitude.unsqueeze(1) * V_norm  # (out, in)

        output = F.linear(x, W_prime, self.bias)
        return output

    def get_direction_loss(self):
        """Regularization: encourage small direction changes."""
        delta_V = self.lora_B @ self.lora_A
        return delta_V.norm() * 0.01


# =====================================================================
#  4. A2A Flow: Action-to-Action Flow Matching
# =====================================================================
class A2AFlowMatcher(nn.Module):
    """A2A (Action-to-Action) Flow Matching.

    Instead of starting denoising from Gaussian noise, uses the robot's
    action history as the starting point. This dramatically reduces the
    number of denoising steps needed (1-3 vs 10+) because the prior
    is already close to the target distribution.

    Architecture:
        action_history (B, k, action_dim)
            → MLP encoder → (B, latent_dim)
            → MLP decoder → (B, chunk_size, action_dim)
            = x_1 (starting point for flow matching)
    """

    def __init__(self, action_dim, chunk_size, history_len=5,
                 latent_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.history_len = history_len
        self.latent_dim = latent_dim

        input_dim = action_dim * history_len

        # Encoder: map action history to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
        )

        # Decoder: map latent to initial noise-like distribution
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, action_dim * chunk_size),
        )

        # Action history buffer
        self._history_buffer = None

    def reset_history(self):
        """Reset history buffer at episode start."""
        self._history_buffer = None

    def update_history(self, action):
        """Update the action history buffer.

        Args:
            action: (B, action_dim) or (action_dim,) current action
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)

        if self._history_buffer is None:
            # Initialize with repeated current action
            self._history_buffer = action.unsqueeze(1).repeat(
                1, self.history_len, 1
            )  # (B, k, action_dim)
        else:
            # Shift and append
            B = action.shape[0]
            if self._history_buffer.shape[0] != B:
                self._history_buffer = action.unsqueeze(1).repeat(
                    1, self.history_len, 1
                )
            else:
                self._history_buffer = torch.cat([
                    self._history_buffer[:, 1:, :],
                    action.unsqueeze(1)
                ], dim=1)

    def encode_history(self, action_history=None):
        """Encode action history into flow starting point x_1.

        Args:
            action_history: (B, k, action_dim) or None (uses internal buffer)
        Returns:
            x_1: (B, chunk_size, action_dim) flow starting point
        """
        if action_history is None:
            action_history = self._history_buffer

        if action_history is None:
            return None  # No history available, fall back to noise

        B = action_history.shape[0]
        device = action_history.device

        # Flatten history
        h_flat = action_history.reshape(B, -1)  # (B, k * action_dim)

        # Pad if needed
        expected_dim = self.action_dim * self.history_len
        if h_flat.shape[-1] < expected_dim:
            padding = torch.zeros(B, expected_dim - h_flat.shape[-1], device=device)
            h_flat = torch.cat([h_flat, padding], dim=-1)
        elif h_flat.shape[-1] > expected_dim:
            h_flat = h_flat[:, :expected_dim]

        # Encode then decode
        latent = self.encoder(h_flat)  # (B, latent_dim)
        x_1_flat = self.decoder(latent)  # (B, chunk_size * action_dim)
        x_1 = x_1_flat.reshape(B, self.chunk_size, self.action_dim)

        return x_1

    def get_flow_start(self, batch_size, device, action_history=None):
        """Get the starting point for flow matching.

        If action history is available, use A2A encoding.
        Otherwise, fall back to Gaussian noise.

        Args:
            batch_size: int
            device: torch device
            action_history: (B, k, action_dim) optional
        Returns:
            x_1: (B, chunk_size, action_dim) starting point
            is_a2a: bool, whether A2A was used
        """
        x_1 = self.encode_history(action_history)

        if x_1 is not None:
            # Add small noise for exploration
            noise = torch.randn_like(x_1) * 0.1
            x_1 = x_1 + noise
            return x_1, True
        else:
            # Fallback to Gaussian noise
            x_1 = torch.randn(batch_size, self.chunk_size, self.action_dim,
                               device=device)
            return x_1, False


# =====================================================================
#  5. RS-CL: Robot State-aware Contrastive Loss
# =====================================================================
class RSContrastiveLoss(nn.Module):
    """Robot State-aware Contrastive Loss (RS-CL).

    Aligns VLM hidden representations with physical robot states.
    States that are physically similar (joint angles close) should
    have similar VLM representations.

    Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    """

    def __init__(self, vlm_hidden_dim, state_dim, projection_dim=128,
                 temperature=0.07):
        super().__init__()
        self.temperature = temperature

        # Project VLM hidden states to shared space
        self.vlm_projector = nn.Sequential(
            nn.Linear(vlm_hidden_dim, projection_dim),
            nn.SiLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Project robot states to shared space
        self.state_projector = nn.Sequential(
            nn.Linear(state_dim, projection_dim),
            nn.SiLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, vlm_hidden, robot_states):
        """Compute RS-CL loss.

        Args:
            vlm_hidden: (B, hidden_dim) pooled VLM representations
            robot_states: (B, state_dim) robot joint states
        Returns:
            loss: scalar contrastive loss
        """
        B = vlm_hidden.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=vlm_hidden.device)

        # Project to shared space
        z_vlm = F.normalize(self.vlm_projector(vlm_hidden), dim=-1)  # (B, proj_dim)
        z_state = F.normalize(self.state_projector(robot_states), dim=-1)  # (B, proj_dim)

        # Compute similarity matrix (B, B)
        sim_vlm_state = z_vlm @ z_state.T / self.temperature

        # Labels: diagonal entries are positive pairs
        labels = torch.arange(B, device=vlm_hidden.device)

        # Symmetric loss
        loss_v2s = F.cross_entropy(sim_vlm_state, labels)
        loss_s2v = F.cross_entropy(sim_vlm_state.T, labels)

        return (loss_v2s + loss_s2v) / 2.0


# =====================================================================
#  6. SnapFlow Trainer (enhanced from previous pipeline)
# =====================================================================
class SnapFlowTrainer:
    """SnapFlow: Self-distillation for single-step flow matching (1-NFE).

    Teacher: frozen multi-step model (10 steps)
    Student: same model trained with Euler shortcut targets (1 step)
    """

    def __init__(self, num_teacher_steps=10):
        self.num_teacher_steps = num_teacher_steps
        self.teacher_model = None

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
        """Compute SnapFlow consistency loss.

        u_shortcut = (x_t - teacher_x0) / t is the ideal single-step velocity.
        """
        t_expanded = t[:, None, None]
        u_shortcut = (x_t - teacher_x0) / (t_expanded + 1e-6)
        snap_loss = F.mse_loss(student_velocity, u_shortcut)
        return snap_loss


# =====================================================================
#  7. AdvancedSmolVLAWrapper: Orchestrator
# =====================================================================
class AdvancedSmolVLAWrapper(nn.Module):
    """Wraps SmolVLA with all advanced optimization modules.

    Integrates:
      - DeeR-VLA multi-exit
      - VLA-Pruner dual-level token pruning
      - DoRA weight-decomposed adapters
      - A2A Flow action-to-action matching
      - RS-CL robot state contrastive loss
      - SnapFlow 1-NFE self-distillation

    Does NOT modify original SmolVLA code — uses hooks and wrappers.
    """

    def __init__(self, smolvla_policy, cfg):
        super().__init__()
        self.policy = smolvla_policy
        self.flow_model = smolvla_policy.model
        self.cfg = cfg

        # Architecture dims
        vlm_hidden = self.flow_model.vlm_with_expert.config.text_config.hidden_size
        expert_hidden = self.flow_model.vlm_with_expert.expert_hidden_size
        num_vlm_layers = self.flow_model.vlm_with_expert.num_vlm_layers
        max_action_dim = smolvla_policy.config.max_action_dim

        # 1. DeeR-VLA multi-exit router
        self.deer_router = DeeRVLARouter(
            vlm_hidden_dim=vlm_hidden,
            expert_hidden_dim=expert_hidden,
            action_dim=max_action_dim,
            num_exits=cfg["num_exits"],
            num_vlm_layers=num_vlm_layers,
            chunk_size=smolvla_policy.config.chunk_size,
        )

        # 2. VLA-Pruner
        self.vla_pruner = VLAPruner(
            semantic_weight=cfg["pruner_semantic_weight"],
            action_weight=cfg["pruner_action_weight"],
            temporal_momentum=cfg["pruner_temporal_momentum"],
            min_keep_ratio=cfg["pruner_min_keep_ratio"],
        )

        # 3. DoRA Adapters — apply to expert layers' attention projections
        self.dora_adapters = nn.ModuleDict()
        expert_layers = self.flow_model.vlm_with_expert.lm_expert.layers
        for layer_idx in range(len(expert_layers)):
            layer = expert_layers[layer_idx]
            attn = layer.self_attn
            layer_key = f"expert_layer_{layer_idx}"

            self.dora_adapters[f"{layer_key}_q"] = DoRAAdapter(
                attn.q_proj, rank=cfg["dora_rank"],
                alpha=cfg["dora_alpha"], dropout=cfg["dora_dropout"],
            )
            self.dora_adapters[f"{layer_key}_v"] = DoRAAdapter(
                attn.v_proj, rank=cfg["dora_rank"],
                alpha=cfg["dora_alpha"], dropout=cfg["dora_dropout"],
            )

        # 4. A2A Flow Matcher
        original_action_dim = smolvla_policy.config.action_feature.shape[0]
        self.a2a_flow = A2AFlowMatcher(
            action_dim=max_action_dim,
            chunk_size=smolvla_policy.config.chunk_size,
            history_len=cfg["a2a_history_len"],
            latent_dim=cfg["a2a_latent_dim"],
        )

        # 5. RS-CL
        state_dim = smolvla_policy.config.max_state_dim
        self.rs_cl = RSContrastiveLoss(
            vlm_hidden_dim=vlm_hidden,
            state_dim=state_dim,
            temperature=cfg.get("rscl_temperature", 0.07),
        )

        # 6. SnapFlow trainer
        self.snap_trainer = SnapFlowTrainer(
            num_teacher_steps=cfg.get("snap_teacher_steps", 10),
        )

        # State tracking
        self._prev_state = None

        # Training stats
        self.exit_stats = []   # which exit was chosen
        self.token_stats = []  # token keep ratios
        self.phase_losses = {f"phase{i}": [] for i in range(1, 6)}

    def reset_episode(self):
        """Reset per-episode state."""
        self._prev_state = None
        self.vla_pruner.reset_temporal()
        self.a2a_flow.reset_history()
        self.policy.reset()

    def compute_ee_velocity(self, state):
        """Compute end-effector velocity from state difference."""
        if self._prev_state is None:
            v_ee = torch.zeros(state.shape[0], 1, device=state.device)
        else:
            delta = state - self._prev_state
            v_ee = delta[:, :6].norm(dim=-1, keepdim=True) if state.shape[-1] >= 6 \
                else delta.norm(dim=-1, keepdim=True)
        self._prev_state = state.detach().clone()
        return v_ee

    # -----------------------------------------------------------------
    # Phase 1: DeeR-VLA Multi-Exit Training
    # -----------------------------------------------------------------
    def forward_phase1_deer(self, batch, exit_threshold=0.5):
        """Phase 1: Train DeeR-VLA exit heads alongside task loss."""
        flow_model = self.flow_model

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch['observation.language.tokens']
        lang_masks = batch['observation.language.attention_mask']
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        B = state.shape[0]
        device = state.device

        # Get visual embeddings for DeeR exit heads
        with torch.no_grad():
            img_embs_list = []
            for img in images:
                img_emb = flow_model.vlm_with_expert.embed_image(img)
                img_embs_list.append(img_emb)
            all_img_emb = torch.cat(img_embs_list, dim=1) if img_embs_list else None

        # Task loss via original forward
        losses = flow_model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
        )
        original_action_dim = self.policy.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        task_loss = losses.mean()

        # DeeR exit loss: simulate exit predictions using pooled visual embeddings
        # Create synthetic "hidden states at exits" from the image embeddings
        # (In practice, these would come from intermediate VLM layers,
        #  but since we don't modify the VLM forward, we use the image embeddings
        #  with learned projections at each exit)
        exit_hidden_states = []
        if all_img_emb is not None:
            n_tokens = all_img_emb.shape[1]
            tokens_per_exit = max(1, n_tokens // self.deer_router.num_exits)
            for i in range(self.deer_router.num_exits):
                start = i * tokens_per_exit
                end = min((i + 1) * tokens_per_exit, n_tokens)
                h = all_img_emb[:, start:end, :]  # (B, subset, D)
                exit_hidden_states.append(h)
        else:
            # Fallback: zero hidden states
            vlm_dim = self.deer_router.exit_heads[0][0].in_features
            for _ in range(self.deer_router.num_exits):
                exit_hidden_states.append(
                    torch.zeros(B, 1, vlm_dim, device=device)
                )

        exit_actions, exit_confidences = self.deer_router.compute_exit_actions(
            exit_hidden_states
        )

        # Ground truth: first action step
        gt_action = actions[:, 0, :]
        exit_loss, per_exit_losses = self.deer_router.compute_exit_loss(
            exit_actions, gt_action,
            loss_weights=self.cfg["exit_loss_weights"],
        )

        # Track which exit would be selected
        with torch.no_grad():
            sel_exit, _ = self.deer_router.select_exit(
                exit_actions, exit_confidences, exit_threshold
            )
            self.exit_stats.append(sel_exit)

        # Combined loss
        lambda_exit = self.cfg["lambda_exit"]
        total_loss = task_loss + lambda_exit * exit_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "exit_loss": exit_loss.item(),
            "selected_exit": sel_exit,
            "per_exit_losses": per_exit_losses,
        }
        return total_loss, loss_dict

    # -----------------------------------------------------------------
    # Phase 2: VLA-Pruner + DoRA
    # -----------------------------------------------------------------
    def forward_phase2_pruner_dora(self, batch):
        """Phase 2: Train VLA-Pruner + DoRA alongside task loss."""
        flow_model = self.flow_model

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch['observation.language.tokens']
        lang_masks = batch['observation.language.attention_mask']
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        B = state.shape[0]
        device = state.device

        # Compute visual embeddings
        with torch.no_grad():
            img_embs_list = []
            for img in images:
                img_emb = flow_model.vlm_with_expert.embed_image(img)
                img_embs_list.append(img_emb)
            all_img_emb = torch.cat(img_embs_list, dim=1) if img_embs_list else None

        # End-effector velocity for VLA-Pruner
        v_ee = self.compute_ee_velocity(state)

        # VLA-Pruner: compute token keep stats (auxiliary signal)
        token_keep_ratio = torch.tensor(1.0, device=device)
        if all_img_emb is not None:
            _, _, token_keep_ratio = self.vla_pruner(
                all_img_emb,
                torch.ones(B, all_img_emb.shape[1], dtype=torch.bool, device=device),
                v_ee=v_ee,
            )
        self.token_stats.append(
            token_keep_ratio.item() if isinstance(token_keep_ratio, torch.Tensor) else token_keep_ratio
        )

        # Task loss via original forward
        losses = flow_model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
        )
        original_action_dim = self.policy.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        task_loss = losses.mean()

        # DoRA direction regularization loss
        dora_reg_loss = torch.tensor(0.0, device=device)
        dora_count = 0
        for key, adapter in self.dora_adapters.items():
            dora_reg_loss = dora_reg_loss + adapter.get_direction_loss()
            dora_count += 1
        dora_reg_loss = dora_reg_loss / max(dora_count, 1)

        # Combined loss
        lambda_prune = self.cfg["lambda_prune"]
        total_loss = task_loss + lambda_prune * token_keep_ratio + 0.01 * dora_reg_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "token_keep_ratio": token_keep_ratio.item() if isinstance(token_keep_ratio, torch.Tensor) else token_keep_ratio,
            "dora_reg_loss": dora_reg_loss.item(),
        }
        return total_loss, loss_dict

    # -----------------------------------------------------------------
    # Phase 3: A2A Flow Matching
    # -----------------------------------------------------------------
    def forward_phase3_a2a(self, batch, action_history=None):
        """Phase 3: Train A2A Flow history encoder."""
        flow_model = self.flow_model

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch['observation.language.tokens']
        lang_masks = batch['observation.language.attention_mask']
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        B = state.shape[0]
        device = state.device
        original_action_dim = self.policy.config.action_feature.shape[0]

        # Get A2A starting point
        x_1_a2a, is_a2a = self.a2a_flow.get_flow_start(
            B, device, action_history
        )

        # Use A2A start as noise in flow matching
        if is_a2a:
            noise = x_1_a2a
        else:
            noise = flow_model.sample_noise(actions.shape, device)

        # Task loss with A2A-conditioned noise
        losses = flow_model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=noise,
        )
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        task_loss = losses.mean()

        # A2A reconstruction: the history encoder should produce
        # something close to the target action sequence
        if is_a2a:
            a2a_recon_loss = F.mse_loss(
                x_1_a2a[:, :, :original_action_dim],
                actions[:, :, :original_action_dim].detach(),
            )
        else:
            a2a_recon_loss = torch.tensor(0.0, device=device)

        total_loss = task_loss + 0.1 * a2a_recon_loss

        # Update action history buffer
        with torch.no_grad():
            self.a2a_flow.update_history(actions[:, 0, :])

        loss_dict = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "a2a_recon_loss": a2a_recon_loss.item(),
            "is_a2a": is_a2a,
        }
        return total_loss, loss_dict

    # -----------------------------------------------------------------
    # Phase 4: SnapFlow 1-NFE
    # -----------------------------------------------------------------
    def forward_phase4_snapflow(self, batch):
        """Phase 4: Train SnapFlow self-distillation for 1-step denoising."""
        flow_model = self.flow_model

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch['observation.language.tokens']
        lang_masks = batch['observation.language.attention_mask']
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        B = state.shape[0]
        device = state.device
        original_action_dim = self.policy.config.action_feature.shape[0]

        # Try A2A start if possible, else noise
        x_1_a2a, is_a2a = self.a2a_flow.get_flow_start(B, device)
        noise = x_1_a2a if is_a2a else flow_model.sample_noise(actions.shape, device)

        # Standard task loss
        losses = flow_model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=noise,
        )
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        task_loss = losses.mean()

        # SnapFlow consistency loss
        snap_loss = torch.tensor(0.0, device=device)
        if self.snap_trainer.teacher_model is not None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = flow_model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )

            teacher_x0 = self.snap_trainer.compute_teacher_target(
                self.snap_trainer.teacher_model,
                prefix_embs.detach(), prefix_pad_masks.detach(),
                prefix_att_masks.detach(),
                noise, flow_model.config.chunk_size, flow_model.action_out_proj
            )
            teacher_x0 = teacher_x0[:, :, :original_action_dim]

            time_val = flow_model.sample_time(B, device)
            time_expanded = time_val[:, None, None]
            x_t = time_expanded * noise[:, :, :original_action_dim] + \
                  (1 - time_expanded) * actions[:, :, :original_action_dim]
            student_v = noise[:, :, :original_action_dim] - \
                        actions[:, :, :original_action_dim]

            snap_loss = self.snap_trainer.compute_snapflow_loss(
                student_v, x_t, teacher_x0, time_val
            )

        total_loss = task_loss + 0.5 * snap_loss

        # Update history
        with torch.no_grad():
            self.a2a_flow.update_history(actions[:, 0, :])

        loss_dict = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "snap_loss": snap_loss.item(),
        }
        return total_loss, loss_dict

    # -----------------------------------------------------------------
    # Phase 5: RS-CL Polish
    # -----------------------------------------------------------------
    def forward_phase5_rscl(self, batch):
        """Phase 5: Joint fine-tuning with RS-CL contrastive loss."""
        flow_model = self.flow_model

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch['observation.language.tokens']
        lang_masks = batch['observation.language.attention_mask']
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        B = state.shape[0]
        device = state.device
        original_action_dim = self.policy.config.action_feature.shape[0]

        # Visual embeddings for RS-CL
        with torch.no_grad():
            img_embs_list = []
            for img in images:
                img_emb = flow_model.vlm_with_expert.embed_image(img)
                img_embs_list.append(img_emb)
            all_img_emb = torch.cat(img_embs_list, dim=1) if img_embs_list else None

        # A2A conditioned noise
        x_1_a2a, is_a2a = self.a2a_flow.get_flow_start(B, device)
        noise = x_1_a2a if is_a2a else flow_model.sample_noise(actions.shape, device)

        # Task loss
        losses = flow_model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=noise,
        )
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        task_loss = losses.mean()

        # RS-CL loss
        rscl_loss = torch.tensor(0.0, device=device)
        if all_img_emb is not None and B >= 2:
            vlm_pooled = all_img_emb.mean(dim=1)  # (B, hidden_dim)
            # Match dimensions if needed
            expected_dim = self.rs_cl.vlm_projector[0].in_features
            if vlm_pooled.shape[-1] != expected_dim:
                vlm_pooled = F.adaptive_avg_pool1d(
                    vlm_pooled.unsqueeze(1), expected_dim
                ).squeeze(1)
            rscl_loss = self.rs_cl(vlm_pooled, state)

        lambda_rscl = self.cfg["lambda_rscl"]
        total_loss = task_loss + lambda_rscl * rscl_loss

        # Update history
        with torch.no_grad():
            self.a2a_flow.update_history(actions[:, 0, :])

        loss_dict = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "rscl_loss": rscl_loss.item(),
        }
        return total_loss, loss_dict


# #####################################################################
#                     DATA LOADING
# #####################################################################

print("=" * 72)
print("  PHASE 0: Load Dataset")
print("=" * 72)

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


# #####################################################################
#                     LOAD MODEL + SETUP WRAPPER
# #####################################################################

print("\n" + "=" * 72)
print("  PHASE 0b: Load SmolVLA + Inject Advanced Modules")
print("=" * 72)

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

# Freeze VLM backbone
if adv_cfg["freeze_vlm"]:
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

# Create Advanced Wrapper
print("\n  Creating AdvancedSmolVLAWrapper...")
adv_wrapper = AdvancedSmolVLAWrapper(smolvla, adv_cfg).to(DEVICE)

# Count wrapper params
deer_params = sum(p.numel() for p in adv_wrapper.deer_router.parameters())
pruner_params = sum(p.numel() for p in adv_wrapper.vla_pruner.parameters())
dora_params = sum(p.numel() for p in adv_wrapper.dora_adapters.parameters())
a2a_params = sum(p.numel() for p in adv_wrapper.a2a_flow.parameters())
rscl_params = sum(p.numel() for p in adv_wrapper.rs_cl.parameters())
total_adv_params = deer_params + pruner_params + dora_params + a2a_params + rscl_params

print(f"  DeeR-VLA Router:  {deer_params/1e6:.2f}M")
print(f"  VLA-Pruner:       {pruner_params/1e6:.4f}M")
print(f"  DoRA Adapters:    {dora_params/1e6:.2f}M")
print(f"  A2A Flow Matcher: {a2a_params/1e6:.2f}M")
print(f"  RS-CL:            {rscl_params/1e6:.4f}M")
print(f"  Total advanced:   {total_adv_params/1e6:.2f}M")


# Batch builder
def build_train_batch(indices):
    """Build a training batch for SmolVLA."""
    batch_imgs = {k: [] for k in KEY_REMAP_S.values()}
    batch_states, batch_actions = [], []

    for idx in indices:
        s = dataset[idx]
        for dk, sk in KEY_REMAP_S.items():
            batch_imgs[sk].append(s[dk])
        if state_key:
            batch_states.append(s[state_key])
        batch_actions.append(s[action_key])

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

    return batch


# #####################################################################
#                     PHASE 1: DeeR-VLA Multi-Exit Training
# #####################################################################
print("\n" + "=" * 72)
print("  PHASE 1: DeeR-VLA Multi-Exit Training")
print("=" * 72)

phase1_params = []
phase1_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase1_params.extend(adv_wrapper.deer_router.parameters())

opt_p1 = torch.optim.AdamW(
    phase1_params,
    lr=adv_cfg["phase1_lr"], weight_decay=adv_cfg["weight_decay"]
)

losses_p1 = []
micro_bs = adv_cfg["micro_batch"]
grad_accum = adv_cfg["grad_accum"]
phase1_steps = adv_cfg["phase1_steps"]

print(f"  Steps: {phase1_steps}, micro_batch={micro_bs}, grad_accum={grad_accum}")
print(f"  Exit threshold: {adv_cfg['exit_threshold_start']} → {adv_cfg['exit_threshold_end']}")
print(f"  Exit weights: {adv_cfg['exit_loss_weights']}")

smolvla.train()
adv_wrapper.train()

pbar = tqdm(range(phase1_steps), desc="  Phase1 DeeR", ncols=105)
for step in pbar:
    opt_p1.zero_grad()
    accum_loss = 0
    accum_dict = {}

    # Anneal exit threshold
    progress = step / max(phase1_steps - 1, 1)
    exit_thr = adv_cfg["exit_threshold_start"] * (1 - progress) + \
               adv_cfg["exit_threshold_end"] * progress

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if adv_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = adv_wrapper.forward_phase1_deer(
                    batch, exit_threshold=exit_thr,
                )
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = adv_wrapper.forward_phase1_deer(
                batch, exit_threshold=exit_thr,
            )
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase1_params, adv_cfg["max_grad_norm"])
    opt_p1.step()

    losses_p1.append(accum_loss)
    if step % 100 == 0:
        avg_loss = np.mean(losses_p1[-50:])
        sel_exit = accum_dict.get('selected_exit', -1)
        exit_l = accum_dict.get('exit_loss', 0)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', exit=f'{sel_exit:.0f}',
                         ex_l=f'{exit_l:.3f}', thr=f'{exit_thr:.2f}')

    if (step + 1) % adv_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'deer_router': adv_wrapper.deer_router.state_dict(),
        }, OUTPUT_DIR / f'phase1_step{step+1}.pt')

torch.save({
    'smolvla': smolvla.state_dict(),
    'deer_router': adv_wrapper.deer_router.state_dict(),
}, OUTPUT_DIR / 'phase1_complete.pt')
print(f"\n  Phase 1 done. Final loss: {np.mean(losses_p1[-50:]):.6f}")

gc.collect()
torch.cuda.empty_cache()


# #####################################################################
#                     PHASE 2: VLA-Pruner + DoRA
# #####################################################################
print("\n" + "=" * 72)
print("  PHASE 2: VLA-Pruner + DoRA Integration")
print("=" * 72)

phase2_params = []
phase2_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase2_params.extend(adv_wrapper.deer_router.parameters())
phase2_params.extend(adv_wrapper.vla_pruner.parameters())
phase2_params.extend(adv_wrapper.dora_adapters.parameters())

opt_p2 = torch.optim.AdamW(
    phase2_params,
    lr=adv_cfg["phase2_lr"], weight_decay=adv_cfg["weight_decay"]
)

losses_p2 = []
phase2_steps = adv_cfg["phase2_steps"]

print(f"  Steps: {phase2_steps}")
print(f"  DoRA rank: {adv_cfg['dora_rank']}, alpha: {adv_cfg['dora_alpha']}")
print(f"  VLA-Pruner: semantic_w={adv_cfg['pruner_semantic_weight']}, "
      f"action_w={adv_cfg['pruner_action_weight']}, "
      f"temporal_momentum={adv_cfg['pruner_temporal_momentum']}")

pbar = tqdm(range(phase2_steps), desc="  Phase2 Pruner+DoRA", ncols=105)
for step in pbar:
    opt_p2.zero_grad()
    accum_loss = 0
    accum_dict = {}

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if adv_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = adv_wrapper.forward_phase2_pruner_dora(batch)
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = adv_wrapper.forward_phase2_pruner_dora(batch)
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase2_params, adv_cfg["max_grad_norm"])
    opt_p2.step()

    losses_p2.append(accum_loss)
    if step % 100 == 0:
        avg_loss = np.mean(losses_p2[-50:])
        tkr = accum_dict.get('token_keep_ratio', 1.0)
        dora_r = accum_dict.get('dora_reg_loss', 0)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', tkr=f'{tkr:.2f}', dora=f'{dora_r:.4f}')

    if (step + 1) % adv_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'deer_router': adv_wrapper.deer_router.state_dict(),
            'vla_pruner': adv_wrapper.vla_pruner.state_dict(),
            'dora_adapters': adv_wrapper.dora_adapters.state_dict(),
        }, OUTPUT_DIR / f'phase2_step{step+1}.pt')

torch.save({
    'smolvla': smolvla.state_dict(),
    'deer_router': adv_wrapper.deer_router.state_dict(),
    'vla_pruner': adv_wrapper.vla_pruner.state_dict(),
    'dora_adapters': adv_wrapper.dora_adapters.state_dict(),
}, OUTPUT_DIR / 'phase2_complete.pt')
print(f"\n  Phase 2 done. Final loss: {np.mean(losses_p2[-50:]):.6f}")

gc.collect()
torch.cuda.empty_cache()


# #####################################################################
#                     PHASE 3: A2A Flow Matching
# #####################################################################
print("\n" + "=" * 72)
print("  PHASE 3: A2A Flow Matching")
print("=" * 72)

phase3_params = []
phase3_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase3_params.extend(adv_wrapper.a2a_flow.parameters())

opt_p3 = torch.optim.AdamW(
    phase3_params,
    lr=adv_cfg["phase3_lr"], weight_decay=adv_cfg["weight_decay"]
)

losses_p3 = []
phase3_steps = adv_cfg["phase3_steps"]

print(f"  Steps: {phase3_steps}")
print(f"  A2A history_len: {adv_cfg['a2a_history_len']}, latent_dim: {adv_cfg['a2a_latent_dim']}")

# Reset history for clean A2A training
adv_wrapper.a2a_flow.reset_history()

pbar = tqdm(range(phase3_steps), desc="  Phase3 A2A Flow", ncols=105)
for step in pbar:
    opt_p3.zero_grad()
    accum_loss = 0
    accum_dict = {}

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if adv_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = adv_wrapper.forward_phase3_a2a(batch)
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = adv_wrapper.forward_phase3_a2a(batch)
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase3_params, adv_cfg["max_grad_norm"])
    opt_p3.step()

    losses_p3.append(accum_loss)
    if step % 100 == 0:
        avg_loss = np.mean(losses_p3[-50:])
        a2a_r = accum_dict.get('a2a_recon_loss', 0)
        is_a2a = accum_dict.get('is_a2a', False)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', a2a=f'{a2a_r:.4f}',
                         using_a2a=str(bool(is_a2a)))

    if (step + 1) % adv_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'a2a_flow': adv_wrapper.a2a_flow.state_dict(),
        }, OUTPUT_DIR / f'phase3_step{step+1}.pt')

torch.save({
    'smolvla': smolvla.state_dict(),
    'a2a_flow': adv_wrapper.a2a_flow.state_dict(),
}, OUTPUT_DIR / 'phase3_complete.pt')
print(f"\n  Phase 3 done. Final loss: {np.mean(losses_p3[-50:]):.6f}")

gc.collect()
torch.cuda.empty_cache()


# #####################################################################
#                     PHASE 4: SnapFlow 1-NFE
# #####################################################################
print("\n" + "=" * 72)
print("  PHASE 4: SnapFlow 1-NFE Self-Distillation")
print("=" * 72)

print("  Creating teacher model for self-distillation...")
adv_wrapper.snap_trainer.create_teacher(smolvla.model)
print(f"  Teacher denoising steps: {adv_cfg['snap_teacher_steps']}")

phase4_params = []
phase4_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase4_params.extend(adv_wrapper.a2a_flow.parameters())

opt_p4 = torch.optim.AdamW(
    phase4_params,
    lr=adv_cfg["phase4_lr"], weight_decay=adv_cfg["weight_decay"]
)
scheduler_p4 = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_p4, T_max=adv_cfg["phase4_steps"], eta_min=1e-6
)

losses_p4 = []
phase4_steps = adv_cfg["phase4_steps"]
print(f"  Steps: {phase4_steps}")

pbar = tqdm(range(phase4_steps), desc="  Phase4 SnapFlow", ncols=105)
for step in pbar:
    opt_p4.zero_grad()
    accum_loss = 0
    accum_dict = {}

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if adv_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = adv_wrapper.forward_phase4_snapflow(batch)
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = adv_wrapper.forward_phase4_snapflow(batch)
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase4_params, adv_cfg["max_grad_norm"])
    opt_p4.step()
    scheduler_p4.step()

    losses_p4.append(accum_loss)
    if step % 100 == 0:
        avg_loss = np.mean(losses_p4[-50:])
        snap_l = accum_dict.get('snap_loss', 0)
        cur_lr = scheduler_p4.get_last_lr()[0]
        pbar.set_postfix(loss=f'{avg_loss:.4f}', snap=f'{snap_l:.4f}',
                         lr=f'{cur_lr:.2e}')

    if (step + 1) % adv_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'a2a_flow': adv_wrapper.a2a_flow.state_dict(),
        }, OUTPUT_DIR / f'phase4_step{step+1}.pt')

torch.save({
    'smolvla': smolvla.state_dict(),
    'a2a_flow': adv_wrapper.a2a_flow.state_dict(),
}, OUTPUT_DIR / 'phase4_complete.pt')
print(f"\n  Phase 4 done. Final loss: {np.mean(losses_p4[-50:]):.6f}")

# Free teacher
del adv_wrapper.snap_trainer.teacher_model
adv_wrapper.snap_trainer.teacher_model = None
gc.collect()
torch.cuda.empty_cache()


# #####################################################################
#                     PHASE 5: RS-CL Polish
# #####################################################################
print("\n" + "=" * 72)
print("  PHASE 5: RS-CL Polish + Joint Fine-Tuning")
print("=" * 72)

phase5_params = []
phase5_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase5_params.extend(adv_wrapper.rs_cl.parameters())
phase5_params.extend(adv_wrapper.a2a_flow.parameters())
phase5_params.extend(adv_wrapper.dora_adapters.parameters())

opt_p5 = torch.optim.AdamW(
    phase5_params,
    lr=adv_cfg["phase5_lr"], weight_decay=adv_cfg["weight_decay"]
)
scheduler_p5 = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_p5, T_max=adv_cfg["phase5_steps"], eta_min=1e-6
)

losses_p5 = []
phase5_steps = adv_cfg["phase5_steps"]
print(f"  Steps: {phase5_steps}")
print(f"  RS-CL temperature: {adv_cfg['rscl_temperature']}, lambda: {adv_cfg['rscl_lambda']}")

pbar = tqdm(range(phase5_steps), desc="  Phase5 RS-CL", ncols=105)
for step in pbar:
    opt_p5.zero_grad()
    accum_loss = 0
    accum_dict = {}

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if adv_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = adv_wrapper.forward_phase5_rscl(batch)
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = adv_wrapper.forward_phase5_rscl(batch)
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase5_params, adv_cfg["max_grad_norm"])
    opt_p5.step()
    scheduler_p5.step()

    losses_p5.append(accum_loss)
    if step % 100 == 0:
        avg_loss = np.mean(losses_p5[-50:])
        rscl_l = accum_dict.get('rscl_loss', 0)
        cur_lr = scheduler_p5.get_last_lr()[0]
        pbar.set_postfix(loss=f'{avg_loss:.4f}', rscl=f'{rscl_l:.4f}',
                         lr=f'{cur_lr:.2e}')

    if (step + 1) % adv_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'deer_router': adv_wrapper.deer_router.state_dict(),
            'vla_pruner': adv_wrapper.vla_pruner.state_dict(),
            'dora_adapters': adv_wrapper.dora_adapters.state_dict(),
            'a2a_flow': adv_wrapper.a2a_flow.state_dict(),
            'rs_cl': adv_wrapper.rs_cl.state_dict(),
        }, OUTPUT_DIR / f'phase5_step{step+1}.pt')

# Final save — all modules
torch.save({
    'smolvla': smolvla.state_dict(),
    'deer_router': adv_wrapper.deer_router.state_dict(),
    'vla_pruner': adv_wrapper.vla_pruner.state_dict(),
    'dora_adapters': adv_wrapper.dora_adapters.state_dict(),
    'a2a_flow': adv_wrapper.a2a_flow.state_dict(),
    'rs_cl': adv_wrapper.rs_cl.state_dict(),
}, OUTPUT_DIR / 'final_model.pt')
print(f"\n  Phase 5 done. Final loss: {np.mean(losses_p5[-50:]):.6f}")

gc.collect()
torch.cuda.empty_cache()


# #####################################################################
#                     TRAINING CURVES (ALL 5 PHASES)
# #####################################################################
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
w = 50
phase_data = [
    (losses_p1, "Phase 1: DeeR-VLA", '#2196F3'),
    (losses_p2, "Phase 2: Pruner+DoRA", '#4CAF50'),
    (losses_p3, "Phase 3: A2A Flow", '#FF9800'),
    (losses_p4, "Phase 4: SnapFlow", '#FF5722'),
    (losses_p5, "Phase 5: RS-CL", '#9C27B0'),
]
for ax, (data, title, color) in zip(axes, phase_data):
    if len(data) > w:
        ax.plot(np.convolve(data, np.ones(w)/w, 'valid'), color=color, lw=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title(title); ax.grid(True, alpha=0.3)

plt.suptitle('Advanced Pipeline — 5-Phase Training Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves_5phase.png', dpi=150)
plt.close()
print(f"\n  Training curves saved: {OUTPUT_DIR / 'training_curves_5phase.png'}")


# #####################################################################
#                     EVALUATION
# #####################################################################
print("\n" + "=" * 72)
print("  EVALUATION: Advanced Dynamic Model")
print("=" * 72)

smolvla.eval()
adv_wrapper.eval()

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


def evaluate_advanced_model(model, eval_episodes):
    results = {
        'mse_per_episode': [], 'latency_ms': [],
        'predictions': {}, 'ground_truth': {},
        'exit_distribution': [],
    }

    for ep_idx in eval_episodes:
        ep_preds, ep_gts = [], []
        indices = episode_indices[ep_idx]
        model.reset()
        adv_wrapper.reset_episode()

        for step_idx in tqdm(indices, desc=f"  AdvModel Ep{ep_idx}", ncols=85, leave=False):
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

    # Exit distribution from training
    if adv_wrapper.exit_stats:
        from collections import Counter
        exit_counts = Counter(adv_wrapper.exit_stats[-1000:])
        results['exit_distribution'] = dict(exit_counts)

    return results


eval_ep_list = eval_eps[:EVAL["num_eval_episodes"]]
print(f"  Eval episodes: {eval_ep_list}")

print("\n  Evaluating Advanced SmolVLA...")
adv_results = evaluate_advanced_model(smolvla, eval_ep_list)
print(f"  Advanced MSE: {adv_results['mse_total']:.4f}, "
      f"Latency: {adv_results['latency_mean_ms']:.1f}ms")
if adv_results.get('exit_distribution'):
    print(f"  Exit distribution (training): {adv_results['exit_distribution']}")


# #####################################################################
#                     COMPARISON PLOTS & VIDEOS
# #####################################################################
print("\n" + "=" * 72)
print("  PLOTS & VIDEOS")
print("=" * 72)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
blue = '#2196F3'

# [0,0] MSE per episode
ax = axes[0,0]
x = np.arange(len(eval_ep_list))
ax.bar(x, adv_results['mse_per_episode'], 0.6, label='Advanced SmolVLA', color=blue)
ax.set_xlabel('Episode'); ax.set_ylabel('MSE'); ax.set_title('MSE per Episode')
ax.set_xticks(x); ax.set_xticklabels([f'Ep{e}' for e in eval_ep_list])
ax.legend(); ax.grid(True, alpha=0.3)

# [0,1] MSE per joint
ax = axes[0,1]
x = np.arange(ACTION_DIM)
ax.bar(x, adv_results['mse_per_joint'], 0.6, label='Advanced SmolVLA', color=blue)
ax.set_xlabel('Joint'); ax.set_ylabel('MSE'); ax.set_title('MSE per Joint')
ax.set_xticks(x); ax.legend(); ax.grid(True, alpha=0.3)

# [0,2] Latency distribution
ax = axes[0,2]
ax.hist(adv_results['latency_ms'], bins=40, alpha=0.7, color=blue)
ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Count'); ax.set_title('Inference Latency')
ax.grid(True, alpha=0.3)

# [1,0] Combined training curve
ax = axes[1,0]
all_losses = losses_p1 + losses_p2 + losses_p3 + losses_p4 + losses_p5
if len(all_losses) > w:
    ax.plot(np.convolve(all_losses, np.ones(w)/w, 'valid'), color=blue, lw=1.0)
    boundaries = [len(losses_p1), len(losses_p1)+len(losses_p2),
                  len(losses_p1)+len(losses_p2)+len(losses_p3),
                  len(losses_p1)+len(losses_p2)+len(losses_p3)+len(losses_p4)]
    colors = ['red', 'green', 'orange', 'purple']
    labels_b = ['P1→2', 'P2→3', 'P3→4', 'P4→5']
    for b, c, lb in zip(boundaries, colors, labels_b):
        ax.axvline(b, color=c, linestyle='--', alpha=0.5, label=lb)
ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('Combined Training Loss')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# [1,1] Trajectory Ep0
ax = axes[1,1]
ep0 = eval_ep_list[0]
gt0 = adv_results['ground_truth'][ep0]
dp0 = adv_results['predictions'][ep0]
nj = min(3, ACTION_DIM)
t_ax = np.arange(len(gt0))
for j in range(nj):
    ax.plot(t_ax, gt0[:,j], '-', color=f'C{j}', lw=2, label=f'GT J{j}')
    ax.plot(t_ax, dp0[:,j], '--', color=f'C{j}', alpha=0.6, label=f'Adv J{j}')
ax.set_xlabel('Step'); ax.set_ylabel('Action'); ax.set_title(f'Trajectory Ep{ep0}')
ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)

# [1,2] Summary
ax = axes[1,2]
ax.axis('off')
summary_text = (
    f"ADVANCED SMOLVLA SUMMARY\n\n"
    f"Dataset: {ds_cfg['repo_id']}\n"
    f"Train: {len(train_eps)} eps, Eval: {len(eval_ep_list)} eps\n\n"
    f"Phase 1 (DeeR-VLA):   {phase1_steps} steps\n"
    f"Phase 2 (Prune+DoRA): {phase2_steps} steps\n"
    f"Phase 3 (A2A Flow):   {phase3_steps} steps\n"
    f"Phase 4 (SnapFlow):   {phase4_steps} steps\n"
    f"Phase 5 (RS-CL):      {phase5_steps} steps\n\n"
    f"MSE:     {adv_results['mse_total']:.4f}\n"
    f"Latency: {adv_results['latency_mean_ms']:.1f}ms\n\n"
    f"Techniques:\n"
    f"  - DeeR-VLA Multi-Exit ({adv_cfg['num_exits']} exits)\n"
    f"  - VLA-Pruner (dual-level + temporal)\n"
    f"  - DoRA (rank={adv_cfg['dora_rank']})\n"
    f"  - A2A Flow (history={adv_cfg['a2a_history_len']})\n"
    f"  - SnapFlow 1-NFE\n"
    f"  - RS-CL (tau={adv_cfg['rscl_temperature']})"
)
ax.text(0.05, 0.5, summary_text, fontsize=10, fontfamily='monospace',
        va='center', ha='left', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle('SmolVLA + DeeR-VLA + VLA-Pruner + DoRA + A2A + SnapFlow + RS-CL',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'advanced_comparison_plots.png', dpi=150)
plt.close()
print(f"  Plots saved: {OUTPUT_DIR / 'advanced_comparison_plots.png'}")


# Videos
for ep_idx in eval_ep_list[:3]:
    gt = adv_results['ground_truth'][ep_idx]
    dp = adv_results['predictions'][ep_idx]
    indices_v = episode_indices[ep_idx][:len(gt)]
    n = min(len(gt), len(dp))

    vpath = OUTPUT_DIR / f'advanced_ep{ep_idx}.mp4'
    fw, fh = 900, 550
    fps_out = dataset.fps if hasattr(dataset, 'fps') and dataset.fps else 15
    writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*'mp4v'), fps_out, (fw, fh))

    for t in range(n):
        frame = np.ones((fh, fw, 3), dtype=np.uint8) * 25
        s = dataset[indices_v[t]]
        if image_keys:
            img = s[image_keys[0]].numpy()
            if img.shape[0] <= 4: img = np.transpose(img, (1,2,0))
            if img.max() <= 1.0: img = (img*255).clip(0,255).astype(np.uint8)
            else: img = img.clip(0,255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (320, 240))
            frame[10:250, 10:330] = img

        cv2.putText(frame, f'Ep {ep_idx} | Frame {t}/{n} | DeeR+DoRA+A2A+Snap+RSCL', (340, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

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
        cv2.putText(frame, 'Advanced SmolVLA', (55, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,160,0), 1)
        ms = float(np.mean((dp[t]-gt[t])**2))
        cv2.putText(frame, f'MSE:{ms:.4f}', (350, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,160,0), 1)
        writer.write(frame)
    writer.release()
    print(f"  Video: {vpath}")


# #####################################################################
#                     SAVE REPORT
# #####################################################################
print("\n" + "=" * 72)
print("  SAVE REPORT")
print("=" * 72)

smolvla_total = sum(p.numel() for p in smolvla.parameters())

report = {
    'pipeline': 'finetune_deerVLA_VLAPruner_DoRa_A2Aflow_Snapflow',
    'dataset': ds_cfg['repo_id'],
    'dataset_key': DATASET_KEY,
    'train_episodes': list(train_eps),
    'eval_episodes': list(eval_ep_list),
    'config': {
        'phase1_steps': phase1_steps,
        'phase2_steps': phase2_steps,
        'phase3_steps': phase3_steps,
        'phase4_steps': phase4_steps,
        'phase5_steps': phase5_steps,
        'total_steps': phase1_steps + phase2_steps + phase3_steps + phase4_steps + phase5_steps,
        'dora_rank': adv_cfg['dora_rank'],
        'a2a_history_len': adv_cfg['a2a_history_len'],
        'num_exits': adv_cfg['num_exits'],
        'rscl_temperature': adv_cfg['rscl_temperature'],
    },
    'results': {
        'label': 'SmolVLA + DeeR-VLA + VLA-Pruner + DoRA + A2A + SnapFlow + RS-CL',
        'params_total_M': round(smolvla_total/1e6, 1),
        'advanced_params_M': round(total_adv_params/1e6, 2),
        'total_mse': adv_results['mse_total'],
        'per_episode_mse': adv_results['mse_per_episode'],
        'per_joint_mse': adv_results['mse_per_joint'],
        'avg_latency_ms': adv_results['latency_mean_ms'],
        'exit_distribution': adv_results.get('exit_distribution', {}),
        'phase1_final_loss': float(np.mean(losses_p1[-50:])),
        'phase2_final_loss': float(np.mean(losses_p2[-50:])),
        'phase3_final_loss': float(np.mean(losses_p3[-50:])),
        'phase4_final_loss': float(np.mean(losses_p4[-50:])),
        'phase5_final_loss': float(np.mean(losses_p5[-50:])),
    },
    'techniques': {
        'DeeR-VLA': f'Dynamic Early-Exit with {adv_cfg["num_exits"]} exit points + consistency-based selection',
        'VLA-Pruner': f'Dual-level (semantic={adv_cfg["pruner_semantic_weight"]}, action={adv_cfg["pruner_action_weight"]}) + temporal smoothing (m={adv_cfg["pruner_temporal_momentum"]})',
        'DoRA': f'Weight-Decomposed Low-Rank Adaptation (rank={adv_cfg["dora_rank"]}, alpha={adv_cfg["dora_alpha"]})',
        'A2A_Flow': f'Action-to-Action flow matching (history_len={adv_cfg["a2a_history_len"]}, latent_dim={adv_cfg["a2a_latent_dim"]})',
        'SnapFlow': f'Self-distillation for 1-NFE denoising (teacher_steps={adv_cfg["snap_teacher_steps"]})',
        'RS-CL': f'Robot State-aware Contrastive Loss (temperature={adv_cfg["rscl_temperature"]}, lambda={adv_cfg["rscl_lambda"]})',
    },
}
with open(OUTPUT_DIR / 'advanced_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"  Report: {OUTPUT_DIR / 'advanced_report.json'}")


# #####################################################################
#                     DONE
# #####################################################################
print("\n" + "=" * 72)
print("  PIPELINE COMPLETE: finetune_deerVLA_VLAPruner_DoRa_A2Aflow_Snapflow")
print("=" * 72)
total_steps = phase1_steps + phase2_steps + phase3_steps + phase4_steps + phase5_steps
print(f"\n  Total training steps: {total_steps}")
print(f"  MSE: {adv_results['mse_total']:.4f}")
print(f"  Latency: {adv_results['latency_mean_ms']:.1f}ms")
if adv_results.get('exit_distribution'):
    print(f"  Exit distribution: {adv_results['exit_distribution']}")
print(f"\n  Advanced modules: {total_adv_params/1e6:.2f}M params")
print(f"    DeeR-VLA:   {deer_params/1e6:.2f}M")
print(f"    VLA-Pruner:  {pruner_params/1e6:.4f}M")
print(f"    DoRA:        {dora_params/1e6:.2f}M")
print(f"    A2A Flow:    {a2a_params/1e6:.2f}M")
print(f"    RS-CL:       {rscl_params/1e6:.4f}M")
print(f"\n  Output: {OUTPUT_DIR}")
for f_item in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f_item.name:<45} {f_item.stat().st_size/1024:.0f} KB")
print("\n  DONE!")
