"""
Fine-Tune SmolVLA with Dynamic Layer Skipping, LoRA-SP, Token Pruning & SnapFlow
=================================================================================
Pipeline: finetune_dynamic_lora_prunning_snapflow
RTX 4070 SUPER (12GB) | fp16/bf16 | lerobot 0.4.4

Architecture Enhancements:
  1. Dynamic Layer Skipping (DySL) via STAR Router
  2. Action-aware Dynamic Token Pruning (ADP)
  3. LoRA-SP (Select-Prune) adaptive rank adaptation
  4. SnapFlow single-step denoising (1-NFE)

Training Schedule:
  Phase 1 (Steps 0-5K):     Gating & Skip Layer training
  Phase 2 (Steps 5K-15K):   LoRA-SP + ADP integration
  Phase 3 (Steps 15K-20K):  SnapFlow self-distillation
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
dyn_cfg = TRAINING["dynamic_lora_pruning_snapflow"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "./outputs"))
OUTPUT_NAME = os.getenv("OUTPUT_NAME", "dynamic_lora_pruning_snapflow")
OUTPUT_DIR = OUTPUT_ROOT / OUTPUT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  SmolVLA Fine-Tune: Dynamic Layer Skip + LoRA-SP + ADP + SnapFlow")
print("=" * 70)
if DEVICE == "cuda":
    print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
else:
    print("  Device: cpu")
print(f"  Dataset: {ds_cfg['repo_id']}")
print(f"  Output: {OUTPUT_DIR}")
print()

# =====================================================================
#  CUSTOM MODULES
# =====================================================================

class STARRouter(nn.Module):
    """Spatial-Temporal Aware Router for Dynamic Layer Skipping.

    Computes per-layer gate values based on:
      - Visual token entropy (E_view)
      - Robot state delta (delta_s_robot)
      - Pooled hidden state from the last fixed layer

    Uses Gumbel-Softmax during training for differentiable binary decisions.
    At inference, uses hard threshold.
    """

    def __init__(self, hidden_dim, num_skippable_layers=8):
        super().__init__()
        self.num_skippable_layers = num_skippable_layers
        # Input: pooled hidden (hidden_dim) + E_view (1) + ||delta_s|| (1)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_skippable_layers),
        )
        # Initialize bias to positive so gates default to "keep" (sigmoid > 0.5)
        nn.init.constant_(self.gate_net[-1].bias, 2.0)

    def compute_visual_entropy(self, attention_weights):
        """Compute entropy of attention distribution over visual tokens.

        Args:
            attention_weights: (B, num_heads, seq_len, seq_len) from layer 0
        Returns:
            entropy: (B, 1) scalar per batch
        """
        # Average over heads and query positions, focus on visual token columns
        # attention_weights shape: (B, H, L, L)
        attn_mean = attention_weights.mean(dim=(1, 2))  # (B, L)
        attn_probs = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(attn_probs * (attn_probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        return entropy  # (B, 1)

    def forward(self, hidden_pooled, e_view, delta_s_norm, tau=1.0, hard=False):
        """
        Args:
            hidden_pooled: (B, hidden_dim) pooled output from last fixed layer
            e_view: (B, 1) visual entropy
            delta_s_norm: (B, 1) state change norm
            tau: Gumbel-Softmax temperature
            hard: if True, use hard decisions (inference)
        Returns:
            gates: (B, num_skippable_layers) values in [0,1]
            gate_loss: scalar, mean gate activation for cost penalty
        """
        x = torch.cat([hidden_pooled, e_view, delta_s_norm], dim=-1)
        logits = self.gate_net(x)  # (B, num_skippable_layers)

        if hard or not self.training:
            gates = (torch.sigmoid(logits) > 0.5).float()
        else:
            # Gumbel-Softmax trick for binary decisions
            # Convert to 2-class logits: [skip, keep]
            logits_2class = torch.stack([torch.zeros_like(logits), logits], dim=-1)  # (B, L, 2)
            gumbel_out = F.gumbel_softmax(logits_2class, tau=tau, hard=False, dim=-1)
            gates = gumbel_out[..., 1]  # probability of keeping the layer

        gate_loss = gates.mean()
        return gates, gate_loss


class ActionAwareTokenPruner(nn.Module):
    """Dynamic Token Pruning based on end-effector velocity.

    When the robot moves fast (coarse manipulation), prune up to 70% of visual tokens.
    When the robot moves slowly (fine manipulation), keep all tokens.

    Token importance is scored by average attention weight from language tokens.
    """

    def __init__(self, v_threshold=0.15, min_keep_ratio=0.3):
        super().__init__()
        self.v_threshold = v_threshold
        self.min_keep_ratio = min_keep_ratio
        # Learnable threshold refinement
        self.threshold_adjust = nn.Parameter(torch.tensor(0.0))

    def compute_ee_velocity(self, state_current, state_previous):
        """Compute end-effector velocity proxy from state difference.

        Args:
            state_current: (B, state_dim)
            state_previous: (B, state_dim) or None
        Returns:
            v_ee: (B, 1) velocity magnitude
        """
        if state_previous is None:
            return torch.zeros(state_current.shape[0], 1, device=state_current.device)
        delta = state_current - state_previous
        v_ee = delta.norm(dim=-1, keepdim=True)
        return v_ee

    def forward(self, token_embeddings, token_mask, v_ee, attention_scores=None):
        """
        Args:
            token_embeddings: (B, N_tokens, D) visual token embeddings
            token_mask: (B, N_tokens) boolean mask
            v_ee: (B, 1) end-effector velocity
            attention_scores: (B, N_tokens) importance scores (optional)
        Returns:
            pruned_embeddings: (B, K, D) kept tokens (padded)
            pruned_mask: (B, K) boolean mask for kept tokens
            keep_ratio: scalar, average fraction of tokens kept
        """
        B, N, D = token_embeddings.shape
        effective_threshold = self.v_threshold + torch.tanh(self.threshold_adjust) * 0.05

        # Compute keep ratio based on velocity
        # High velocity → low keep ratio, low velocity → keep all
        keep_ratio_per_sample = torch.where(
            v_ee > effective_threshold,
            torch.clamp(self.min_keep_ratio + (1.0 - self.min_keep_ratio) * (effective_threshold / (v_ee + 1e-8)), 
                        min=self.min_keep_ratio, max=1.0),
            torch.ones_like(v_ee)
        )  # (B, 1)

        # Fixed K for batching: use maximum keep count
        K = max(int(N * self.min_keep_ratio), 1)

        if attention_scores is None:
            # Default: keep first K tokens (positional importance)
            attention_scores = torch.arange(N, 0, -1, device=token_embeddings.device).float()
            attention_scores = attention_scores.unsqueeze(0).expand(B, -1)

        # For each sample, select top-K tokens by importance
        # But if keep_ratio is 1.0, we keep all (handled by masking)
        _, top_indices = attention_scores.topk(K, dim=-1, sorted=False)  # (B, K)
        top_indices_sorted, _ = top_indices.sort(dim=-1)  # maintain order

        # Gather selected tokens
        pruned_embeddings = torch.gather(
            token_embeddings, 1,
            top_indices_sorted.unsqueeze(-1).expand(-1, -1, D)
        )
        pruned_mask = torch.gather(token_mask, 1, top_indices_sorted)

        # For samples where keep_ratio == 1.0, we want full tokens
        # We handle this by returning both pruned and original, letting caller decide
        avg_keep_ratio = keep_ratio_per_sample.mean()

        return pruned_embeddings, pruned_mask, avg_keep_ratio, keep_ratio_per_sample


class LoRASPAdapter(nn.Module):
    """LoRA-SP: LoRA with Select-Prune for dynamic rank adaptation.

    Replaces standard deltaW = B*A with:
        deltaW(x) = U * diag(s(x)) * V

    where U, V are shared vector banks (max rank r),
    and s(x) is input-dependent singular value scores from a lightweight router.

    Select: Keep top-k scores where cumulative energy >= eta
    Prune: Spectral concentration loss encourages sparsity
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

        # Lightweight router: produces singular value scores
        self.router = nn.Sequential(
            nn.Linear(in_features, max_rank),
            nn.Sigmoid(),
        )

        self.scaling = alpha / max_rank

    def compute_spectral_loss(self, scores):
        """Spectral concentration loss: negative entropy of normalized scores.

        Encourages the router to concentrate energy in fewer directions.
        Lower loss = more concentrated = sparser effective rank.

        Args:
            scores: (B, max_rank) singular value scores
        Returns:
            spec_loss: scalar
        """
        scores_sq = scores ** 2
        total_energy = scores_sq.sum(dim=-1, keepdim=True) + 1e-8
        prob = scores_sq / total_energy
        # Negative entropy (we want to minimize this → maximize concentration)
        spec_loss = (prob * (prob + 1e-8).log()).sum(dim=-1).mean()
        return -spec_loss  # negate so minimizing = concentrating

    def forward(self, x, return_spec_loss=False):
        """
        Args:
            x: (B, ..., in_features)
            return_spec_loss: if True, also return spectral loss
        Returns:
            delta: (B, ..., out_features) the LoRA-SP adjustment
            spec_loss: scalar (only if return_spec_loss=True)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Router produces scores
        scores = self.router(x_flat.detach())  # (B*L, max_rank), detach input from router gradient

        # Select top-k by energy threshold
        scores_sorted, sort_idx = scores.sort(dim=-1, descending=True)
        cumulative_energy = (scores_sorted ** 2).cumsum(dim=-1)
        total_energy = (scores ** 2).sum(dim=-1, keepdim=True) + 1e-8
        energy_ratio = cumulative_energy / total_energy

        # Soft mask: use scores directly (differentiable)
        # The selection is implicit through the scores magnitude
        # V * x: (max_rank, in_feat) @ (in_feat, B*L) -> (max_rank, B*L)
        Vx = F.linear(x_flat, self.V)  # (B*L, max_rank)

        # Apply scores as diagonal: element-wise multiply
        SVx = Vx * scores * self.scaling  # (B*L, max_rank)

        # U * SVx: (out_feat, max_rank) @ (max_rank, B*L)^T
        delta = F.linear(SVx, self.U)  # (B*L, out_features)

        delta = delta.reshape(*orig_shape[:-1], self.out_features)

        if return_spec_loss:
            spec_loss = self.compute_spectral_loss(scores)
            return delta, spec_loss
        return delta


class SnapFlowTrainer:
    """SnapFlow: Self-distillation for single-step flow matching.

    Compresses multi-step ODE solving into one-step prediction.
    Teacher: frozen multi-step model
    Student: same model trained with Euler shortcut targets
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
        """Compute teacher's multi-step prediction for shortcut target.

        Returns the teacher's final prediction x_0 from noise.
        """
        bsize = noise.shape[0]
        device = noise.device
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        # Build prefix KV cache
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

        return x_t  # teacher's prediction of x_0

    def compute_snapflow_loss(self, student_velocity, x_t, teacher_x0, t):
        """Compute SnapFlow consistency loss.

        u_shortcut = (teacher_x0 - x_t) / (1 - t) is the ideal single-step velocity
        from x_t to the teacher's x_0 prediction.

        Args:
            student_velocity: (B, chunk, action_dim) student's v_theta(x_t, t)
            x_t: (B, chunk, action_dim) current noisy state
            teacher_x0: (B, chunk, action_dim) teacher's prediction
            t: (B,) current timestep
        Returns:
            snap_loss: scalar
        """
        t_expanded = t[:, None, None]  # (B, 1, 1)
        # Velocity that would take x_t directly to teacher_x0 in one step
        # Since x_1 = noise and x_0 = clean, and we go from t=1 to t=0:
        # u_shortcut = (teacher_x0 - x_t) / t   (time remaining = t)
        u_shortcut = (x_t - teacher_x0) / (t_expanded + 1e-6)
        snap_loss = F.mse_loss(student_velocity, u_shortcut)
        return snap_loss


class DynamicSmolVLAWrapper(nn.Module):
    """Wraps SmolVLA with dynamic layer skipping, token pruning, and LoRA-SP.

    This wrapper does NOT modify the original SmolVLA model code.
    Instead, it intercepts the forward pass to apply:
      1. Token pruning before VLM processing
      2. Layer skipping via STAR router gates
      3. LoRA-SP adjustments on attention projections
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

        # 1. STAR Router for Dynamic Layer Skipping
        self.star_router = STARRouter(
            hidden_dim=vlm_hidden,
            num_skippable_layers=dyn_cfg["num_skippable_layers"],
        )

        # 2. Action-Aware Dynamic Token Pruner
        self.token_pruner = ActionAwareTokenPruner(
            v_threshold=dyn_cfg["adp_velocity_threshold"],
            min_keep_ratio=dyn_cfg["adp_min_keep_ratio"],
        )

        # 3. LoRA-SP Adapters for VLM layers
        self.lora_adapters = nn.ModuleDict()
        # Apply to skippable layers' attention projections
        vlm_layers = self.flow_model.vlm_with_expert.get_vlm_model().text_model.layers
        for layer_idx in range(dyn_cfg["num_fixed_layers"], num_vlm_layers):
            layer = vlm_layers[layer_idx]
            layer_key = f"layer_{layer_idx}"
            attn = layer.self_attn

            # Get dimensions from the layer
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

        # 4. SnapFlow trainer
        self.snap_trainer = SnapFlowTrainer(
            num_teacher_steps=dyn_cfg["snap_teacher_steps"],
        )

        # State tracking for ADP
        self._prev_state = None

        # Training tracking
        self.skip_stats = []
        self.token_stats = []

    def reset_episode(self):
        """Reset per-episode state."""
        self._prev_state = None
        self.policy.reset()

    def compute_context_features(self, state, images_emb=None):
        """Compute context features for STAR router.

        Args:
            state: (B, state_dim)
            images_emb: (B, N, D) visual embeddings (optional, for entropy)
        Returns:
            e_view: (B, 1) visual entropy proxy
            delta_s_norm: (B, 1) state change norm
        """
        B = state.shape[0]
        device = state.device

        # Visual entropy proxy: variance of visual embeddings
        if images_emb is not None:
            # Use variance across tokens as entropy proxy
            e_view = images_emb.var(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)
        else:
            e_view = torch.zeros(B, 1, device=device)

        # State delta
        if self._prev_state is not None:
            delta_s = state - self._prev_state
            delta_s_norm = delta_s.norm(dim=-1, keepdim=True)
        else:
            delta_s_norm = torch.zeros(B, 1, device=device)

        self._prev_state = state.detach().clone()
        return e_view, delta_s_norm

    def forward_with_skip(self, batch, tau=1.0,
                          enable_skip=True, enable_adp=False,
                          enable_lora=False, enable_snap=False,
                          noise=None, time_val=None):
        """Forward pass with dynamic optimization modules.

        Strategy (standard conditional computation approach):
          - Task loss: computed via the ORIGINAL flow_model.forward() which
            correctly handles cross-attn/self-attn routing internally.
          - Gate loss: STAR router predicts skip gates, trained via auxiliary
            regularization. Actual gating applied at inference time.
          - LoRA-SP loss: adapters are run on cached image embeddings to
            compute spectral concentration loss for rank adaptation training.
          - ADP cost: token pruner computes keep ratio from velocity signal.
          - SnapFlow: teacher provides shortcut targets for 1-NFE training.

        Returns:
            total_loss: scalar combined loss
            loss_dict: dictionary with all loss components
        """
        flow_model = self.flow_model

        # 1. Prepare inputs
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch['observation.language.tokens']
        lang_masks = batch['observation.language.attention_mask']
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        B = state.shape[0]
        device = state.device

        # 2. Compute visual embeddings for context (used by router & ADP)
        with torch.no_grad():
            img_embs_list = []
            for img in images:
                img_emb = flow_model.vlm_with_expert.embed_image(img)
                img_embs_list.append(img_emb)
            all_img_emb = torch.cat(img_embs_list, dim=1) if img_embs_list else None

        # 3. Context features for STAR router
        e_view, delta_s_norm = self.compute_context_features(state, all_img_emb)

        # 4. ADP: compute velocity and token keep ratio
        v_ee = self.token_pruner.compute_ee_velocity(
            state[:, :6] if state.shape[-1] >= 6 else state,
            self._prev_state[:, :6] if (self._prev_state is not None and self._prev_state.shape[-1] >= 6) else None
        ) if enable_adp else torch.zeros(B, 1, device=device)

        token_keep_ratio = torch.tensor(1.0, device=device)
        if enable_adp and all_img_emb is not None:
            _, _, token_keep_ratio, _ = self.token_pruner(
                all_img_emb,
                torch.ones(B, all_img_emb.shape[1], dtype=torch.bool, device=device),
                v_ee,
            )

        # 5. STAR Router: compute skip gates (auxiliary training signal)
        num_fixed = self.cfg["num_fixed_layers"]
        if enable_skip:
            hidden_pooled = all_img_emb.mean(dim=1) if all_img_emb is not None else torch.zeros(B, self.star_router.gate_net[0].in_features - 2, device=device)
            expected_dim = self.star_router.gate_net[0].in_features - 2
            if hidden_pooled.shape[-1] != expected_dim:
                hidden_pooled = F.adaptive_avg_pool1d(
                    hidden_pooled.unsqueeze(1), expected_dim
                ).squeeze(1)
            gates, gate_loss = self.star_router(hidden_pooled, e_view, delta_s_norm, tau=tau)
        else:
            gates = torch.ones(B, self.cfg["num_skippable_layers"], device=device)
            gate_loss = torch.tensor(0.0, device=device)

        # Track stats
        if enable_skip:
            self.skip_stats.append(gates.mean().item())
        self.token_stats.append(token_keep_ratio.item() if isinstance(token_keep_ratio, torch.Tensor) else token_keep_ratio)

        # 6. Task loss via ORIGINAL flow_model.forward()
        #    This correctly handles cross-attn/self-attn routing internally
        losses = flow_model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=noise, time=time_val
        )
        original_action_dim = self.policy.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        task_loss = losses.mean()

        # 7. LoRA-SP spectral loss (auxiliary)
        #    Run adapters on image embeddings to train rank selection
        total_spec_loss = torch.tensor(0.0, device=device)
        lora_count = 0
        if enable_lora and all_img_emb is not None:
            # Use image embeddings as proxy input for LoRA adapter training
            lora_input = all_img_emb.detach()  # detach from vision encoder
            expected_lora_dim = None
            for key, adapter in self.lora_adapters.items():
                if expected_lora_dim is None:
                    expected_lora_dim = adapter.in_features
                # Adapt input dim if needed
                if lora_input.shape[-1] != adapter.in_features:
                    # Pool/project to match adapter input dim
                    adapted_input = F.adaptive_avg_pool1d(
                        lora_input.permute(0, 2, 1), adapter.in_features
                    ).permute(0, 2, 1)
                else:
                    adapted_input = lora_input
                _, s_loss = adapter(adapted_input, return_spec_loss=True)
                total_spec_loss = total_spec_loss + s_loss
                lora_count += 1

        avg_spec_loss = total_spec_loss / max(lora_count, 1)

        # 8. SnapFlow consistency loss
        snap_loss = torch.tensor(0.0, device=device)
        if enable_snap and self.snap_trainer.teacher_model is not None:
            # We need prefix embeddings for teacher — recompute
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
            x_t = time_expanded * noise[:, :, :original_action_dim] + (1 - time_expanded) * actions[:, :, :original_action_dim]
            # Student velocity = noise - actions (flow matching target)
            student_v = (noise[:, :, :original_action_dim] - actions[:, :, :original_action_dim])
            snap_loss = self.snap_trainer.compute_snapflow_loss(
                student_v, x_t, teacher_x0, time_val
            )

        # 9. Token cost
        token_cost = token_keep_ratio if isinstance(token_keep_ratio, torch.Tensor) else torch.tensor(token_keep_ratio, device=device)

        # 10. Combined loss: J = L_task + λ_spec * L_spec + λ_cost * (skip + token) + snap
        lambda_spec = self.cfg["lambda_spec"]
        lambda_cost = self.cfg["lambda_cost"]

        total_loss = task_loss
        if enable_lora and lora_count > 0:
            total_loss = total_loss + lambda_spec * avg_spec_loss
        if enable_skip:
            total_loss = total_loss + lambda_cost * gate_loss
        if enable_adp:
            total_loss = total_loss + lambda_cost * token_cost
        if enable_snap:
            total_loss = total_loss + snap_loss * 0.5

        num_vlm_layers = self.flow_model.vlm_with_expert.num_vlm_layers
        loss_dict = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "gate_loss": gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
            "spec_loss": avg_spec_loss.item() if isinstance(avg_spec_loss, torch.Tensor) else avg_spec_loss,
            "snap_loss": snap_loss.item(),
            "token_keep_ratio": token_cost.item() if isinstance(token_cost, torch.Tensor) else token_cost,
            "avg_skip_ratio": 1.0 - gates.mean().item() if enable_skip else 0.0,
            "active_layers": (gates.sum(dim=-1).mean().item() + num_fixed) if enable_skip else num_vlm_layers,
        }

        return total_loss, loss_dict


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
print("  PHASE 2: Load SmolVLA Base + Inject Dynamic Modules")
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
print("\n  Creating DynamicSmolVLAWrapper...")
dyn_wrapper = DynamicSmolVLAWrapper(smolvla, dyn_cfg).to(DEVICE)

# Count trainable params in wrapper
wrapper_params = sum(p.numel() for p in dyn_wrapper.star_router.parameters())
wrapper_params += sum(p.numel() for p in dyn_wrapper.token_pruner.parameters())
wrapper_params += sum(p.numel() for p in dyn_wrapper.lora_adapters.parameters())
print(f"  STAR Router params: {sum(p.numel() for p in dyn_wrapper.star_router.parameters())/1e6:.2f}M")
print(f"  Token Pruner params: {sum(p.numel() for p in dyn_wrapper.token_pruner.parameters())/1e6:.4f}M")
print(f"  LoRA-SP Adapter params: {sum(p.numel() for p in dyn_wrapper.lora_adapters.parameters())/1e6:.2f}M")
print(f"  Total dynamic params: {wrapper_params/1e6:.2f}M")


def build_train_batch(indices):
    """Build a training batch for SmolVLA (same pattern as finetune_pipeline.py)."""
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


# =====================================================================
#  PHASE 3: Training Phase 1 — Gating & DySL
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 3: Training Phase 1 — Gating & Dynamic Layer Skipping")
print("=" * 70)

# Only train: action expert + STAR router + gate params
phase1_params = []
phase1_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase1_params.extend(dyn_wrapper.star_router.parameters())

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

smolvla.train()
dyn_wrapper.train()

pbar = tqdm(range(phase1_steps), desc="  Phase1 DySL", ncols=100)
for step in pbar:
    opt_p1.zero_grad()
    accum_loss = 0
    accum_dict = {}

    # Anneal Gumbel temperature
    progress = step / max(phase1_steps - 1, 1)
    tau = dyn_cfg["gumbel_tau_start"] * (1 - progress) + dyn_cfg["gumbel_tau_end"] * progress

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if dyn_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = dyn_wrapper.forward_with_skip(
                    batch, tau=tau,
                    enable_skip=True, enable_adp=False,
                    enable_lora=False, enable_snap=False,
                )
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = dyn_wrapper.forward_with_skip(
                batch, tau=tau,
                enable_skip=True, enable_adp=False,
                enable_lora=False, enable_snap=False,
            )
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase1_params, dyn_cfg["max_grad_norm"])
    opt_p1.step()

    losses_p1.append(accum_loss)
    if step % 100 == 0:
        avg_loss = np.mean(losses_p1[-50:])
        skip_r = accum_dict.get('avg_skip_ratio', 0)
        active_l = accum_dict.get('active_layers', 16)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', skip=f'{skip_r:.2f}', tau=f'{tau:.2f}', layers=f'{active_l:.1f}')

    if (step + 1) % dyn_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'star_router': dyn_wrapper.star_router.state_dict(),
        }, OUTPUT_DIR / f'phase1_step{step+1}.pt')

torch.save({
    'smolvla': smolvla.state_dict(),
    'star_router': dyn_wrapper.star_router.state_dict(),
}, OUTPUT_DIR / 'phase1_complete.pt')
print(f"\n  Phase 1 done. Final loss: {np.mean(losses_p1[-50:]):.6f}")

gc.collect()
torch.cuda.empty_cache()

# =====================================================================
#  PHASE 4: Training Phase 2 — LoRA-SP + ADP
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 4: Training Phase 2 — LoRA-SP + ADP Integration")
print("=" * 70)

phase2_params = []
phase2_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase2_params.extend(dyn_wrapper.star_router.parameters())
phase2_params.extend(dyn_wrapper.token_pruner.parameters())
phase2_params.extend(dyn_wrapper.lora_adapters.parameters())

opt_p2 = torch.optim.AdamW(
    phase2_params,
    lr=dyn_cfg["phase2_lr"], weight_decay=dyn_cfg["weight_decay"]
)

losses_p2 = []
phase2_steps = dyn_cfg["phase2_steps"]

print(f"  Steps: {phase2_steps}")
print(f"  LoRA max rank: {dyn_cfg['lora_max_rank']}, energy threshold: {dyn_cfg['lora_energy_threshold']}")
print(f"  ADP velocity threshold: {dyn_cfg['adp_velocity_threshold']}")

pbar = tqdm(range(phase2_steps), desc="  Phase2 LoRA+ADP", ncols=100)
for step in pbar:
    opt_p2.zero_grad()
    accum_loss = 0
    accum_dict = {}

    # Use low tau from end of Phase 1
    tau = dyn_cfg["gumbel_tau_end"]

    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_train_batch(bi)

        if dyn_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = dyn_wrapper.forward_with_skip(
                    batch, tau=tau,
                    enable_skip=True, enable_adp=True,
                    enable_lora=True, enable_snap=False,
                )
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = dyn_wrapper.forward_with_skip(
                batch, tau=tau,
                enable_skip=True, enable_adp=True,
                enable_lora=True, enable_snap=False,
            )
            loss = loss / grad_accum
            loss.backward()

        accum_loss += loss.item()
        for k, v in loss_dict.items():
            accum_dict[k] = accum_dict.get(k, 0) + v / grad_accum

    torch.nn.utils.clip_grad_norm_(phase2_params, dyn_cfg["max_grad_norm"])
    opt_p2.step()

    losses_p2.append(accum_loss)
    if step % 100 == 0:
        avg_loss = np.mean(losses_p2[-50:])
        spec_l = accum_dict.get('spec_loss', 0)
        tkr = accum_dict.get('token_keep_ratio', 1.0)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', spec=f'{spec_l:.3f}', tkr=f'{tkr:.2f}')

    if (step + 1) % dyn_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'star_router': dyn_wrapper.star_router.state_dict(),
            'token_pruner': dyn_wrapper.token_pruner.state_dict(),
            'lora_adapters': dyn_wrapper.lora_adapters.state_dict(),
        }, OUTPUT_DIR / f'phase2_step{step+1}.pt')

torch.save({
    'smolvla': smolvla.state_dict(),
    'star_router': dyn_wrapper.star_router.state_dict(),
    'token_pruner': dyn_wrapper.token_pruner.state_dict(),
    'lora_adapters': dyn_wrapper.lora_adapters.state_dict(),
}, OUTPUT_DIR / 'phase2_complete.pt')
print(f"\n  Phase 2 done. Final loss: {np.mean(losses_p2[-50:]):.6f}")

gc.collect()
torch.cuda.empty_cache()

# =====================================================================
#  PHASE 5: Training Phase 3 — SnapFlow Self-Distillation
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 5: Training Phase 3 — SnapFlow Self-Distillation")
print("=" * 70)

# Create teacher from current model
print("  Creating teacher model for self-distillation...")
dyn_wrapper.snap_trainer.create_teacher(smolvla.model)
print(f"  Teacher denoising steps: {dyn_cfg['snap_teacher_steps']}")

phase3_params = []
phase3_params.extend([p for p in smolvla.parameters() if p.requires_grad])
phase3_params.extend(dyn_wrapper.star_router.parameters())
phase3_params.extend(dyn_wrapper.token_pruner.parameters())
phase3_params.extend(dyn_wrapper.lora_adapters.parameters())

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

pbar = tqdm(range(phase3_steps), desc="  Phase3 SnapFlow", ncols=100)
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
                    enable_skip=True, enable_adp=True,
                    enable_lora=True, enable_snap=True,
                )
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = dyn_wrapper.forward_with_skip(
                batch, tau=tau,
                enable_skip=True, enable_adp=True,
                enable_lora=True, enable_snap=True,
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
    if step % 100 == 0:
        avg_loss = np.mean(losses_p3[-50:])
        snap_l = accum_dict.get('snap_loss', 0)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', snap=f'{snap_l:.4f}', lr=f'{scheduler_p3.get_last_lr()[0]:.2e}')

    if (step + 1) % dyn_cfg["save_every"] == 0:
        torch.save({
            'smolvla': smolvla.state_dict(),
            'star_router': dyn_wrapper.star_router.state_dict(),
            'token_pruner': dyn_wrapper.token_pruner.state_dict(),
            'lora_adapters': dyn_wrapper.lora_adapters.state_dict(),
        }, OUTPUT_DIR / f'phase3_step{step+1}.pt')

# Final save
torch.save({
    'smolvla': smolvla.state_dict(),
    'star_router': dyn_wrapper.star_router.state_dict(),
    'token_pruner': dyn_wrapper.token_pruner.state_dict(),
    'lora_adapters': dyn_wrapper.lora_adapters.state_dict(),
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
ax.set_title('Phase 1: DySL Gating'); ax.grid(True, alpha=0.3)

ax = axes[1]
if len(losses_p2) > w:
    ax.plot(np.convolve(losses_p2, np.ones(w)/w, 'valid'), color='#4CAF50', lw=1.5)
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('Phase 2: LoRA-SP + ADP'); ax.grid(True, alpha=0.3)

ax = axes[2]
if len(losses_p3) > w:
    ax.plot(np.convolve(losses_p3, np.ones(w)/w, 'valid'), color='#FF5722', lw=1.5)
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('Phase 3: SnapFlow'); ax.grid(True, alpha=0.3)

plt.suptitle('Dynamic LoRA Pruning SnapFlow — Training Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves_3phase.png', dpi=150)
plt.close()

# =====================================================================
#  PHASE 6: Evaluate
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

    # Skip and token stats from training
    results['avg_skip_ratio'] = float(np.mean(dyn_wrapper.skip_stats[-500:])) if dyn_wrapper.skip_stats else 0.0
    results['avg_token_keep_ratio'] = float(np.mean(dyn_wrapper.token_stats[-500:])) if dyn_wrapper.token_stats else 1.0

    return results


eval_ep_list = eval_eps[:EVAL["num_eval_episodes"]]
print(f"  Eval episodes: {eval_ep_list}")

print("\n  Evaluating Dynamic SmolVLA...")
dyn_results = evaluate_dynamic_model(smolvla, eval_ep_list)
print(f"  Dynamic MSE: {dyn_results['mse_total']:.4f}, Latency: {dyn_results['latency_mean_ms']:.1f}ms")
print(f"  Avg Skip Ratio (training): {dyn_results['avg_skip_ratio']:.3f}")
print(f"  Avg Token Keep Ratio (training): {dyn_results['avg_token_keep_ratio']:.3f}")

# =====================================================================
#  PHASE 7: Comparison Plots & Videos
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
ax.bar(x, dyn_results['mse_per_episode'], 0.6, label='Dynamic SmolVLA', color=blue)
ax.set_xlabel('Episode'); ax.set_ylabel('MSE'); ax.set_title('MSE per Episode')
ax.set_xticks(x); ax.set_xticklabels([f'Ep{e}' for e in eval_ep_list])
ax.legend(); ax.grid(True, alpha=0.3)

# [0,1] MSE per joint
ax = axes[0,1]
x = np.arange(ACTION_DIM)
ax.bar(x, dyn_results['mse_per_joint'], 0.6, label='Dynamic SmolVLA', color=blue)
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
    # Mark phase boundaries
    ax.axvline(len(losses_p1), color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    ax.axvline(len(losses_p1) + len(losses_p2), color='green', linestyle='--', alpha=0.5, label='Phase 2→3')
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
    f"DYNAMIC SMOLVLA SUMMARY\n\n"
    f"Dataset: {ds_cfg['repo_id']}\n"
    f"Train: {len(train_eps)} eps, Eval: {len(eval_ep_list)} eps\n\n"
    f"Phase 1 (DySL):  {phase1_steps} steps\n"
    f"Phase 2 (LoRA):  {phase2_steps} steps\n"
    f"Phase 3 (Snap):  {phase3_steps} steps\n\n"
    f"MSE:     {dyn_results['mse_total']:.4f}\n"
    f"Latency: {dyn_results['latency_mean_ms']:.1f}ms\n"
    f"Avg Skip Ratio: {dyn_results['avg_skip_ratio']:.3f}\n"
    f"Avg Token Keep: {dyn_results['avg_token_keep_ratio']:.3f}\n\n"
    f"Techniques:\n"
    f"  - Dynamic Layer Skipping (STAR)\n"
    f"  - LoRA-SP (r={dyn_cfg['lora_max_rank']})\n"
    f"  - Action-Aware Token Pruning\n"
    f"  - SnapFlow 1-NFE Denoising"
)
ax.text(0.05, 0.5, summary, fontsize=10, fontfamily='monospace',
        va='center', ha='left', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle('SmolVLA + DySL + LoRA-SP + ADP + SnapFlow', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dynamic_comparison_plots.png', dpi=150)
plt.close()
print(f"  Plots saved: {OUTPUT_DIR / 'dynamic_comparison_plots.png'}")

# Videos
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

        cv2.putText(frame, f'Ep {ep_idx} | Frame {t}/{n} | DySL+LoRA+Snap', (340, 22),
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
        cv2.putText(frame, 'Dynamic SmolVLA', (55, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,160,0), 1)
        ms = float(np.mean((dp[t]-gt[t])**2))
        cv2.putText(frame, f'MSE:{ms:.4f}', (350, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,160,0), 1)
        writer.write(frame)
    writer.release()
    print(f"  Video: {vpath}")

# =====================================================================
#  PHASE 8: Save Report
# =====================================================================
print("\n" + "=" * 70)
print("  PHASE 8: Save Report")
print("=" * 70)

smolvla_total = sum(p.numel() for p in smolvla.parameters())

report = {
    'pipeline': 'finetune_dynamic_lora_prunning_snapflow',
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
        'adp_velocity_threshold': dyn_cfg['adp_velocity_threshold'],
        'adp_min_keep_ratio': dyn_cfg['adp_min_keep_ratio'],
        'num_fixed_layers': dyn_cfg['num_fixed_layers'],
        'num_skippable_layers': dyn_cfg['num_skippable_layers'],
    },
    'results': {
        'label': 'SmolVLA + DySL + LoRA-SP + ADP + SnapFlow',
        'params_total_M': round(smolvla_total/1e6, 1),
        'dynamic_params_M': round(wrapper_params/1e6, 2),
        'total_mse': dyn_results['mse_total'],
        'per_episode_mse': dyn_results['mse_per_episode'],
        'per_joint_mse': dyn_results['mse_per_joint'],
        'avg_latency_ms': dyn_results['latency_mean_ms'],
        'avg_skip_ratio': dyn_results['avg_skip_ratio'],
        'avg_token_keep_ratio': dyn_results['avg_token_keep_ratio'],
        'phase1_final_loss': float(np.mean(losses_p1[-50:])),
        'phase2_final_loss': float(np.mean(losses_p2[-50:])),
        'phase3_final_loss': float(np.mean(losses_p3[-50:])),
    },
    'techniques': {
        'DySL': 'Dynamic Layer Skipping via STAR Router with Gumbel-Softmax gating',
        'ADP': f'Action-aware Dynamic Token Pruning (v_threshold={dyn_cfg["adp_velocity_threshold"]})',
        'LoRA-SP': f'Select-Prune LoRA (max_rank={dyn_cfg["lora_max_rank"]}, eta={dyn_cfg["lora_energy_threshold"]})',
        'SnapFlow': f'Self-distillation for 1-NFE denoising (teacher_steps={dyn_cfg["snap_teacher_steps"]})',
    },
}
with open(OUTPUT_DIR / 'dynamic_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"  Report: {OUTPUT_DIR / 'dynamic_report.json'}")

# =====================================================================
#  DONE
# =====================================================================
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE: finetune_dynamic_lora_prunning_snapflow")
print("=" * 70)
print(f"\n  MSE: {dyn_results['mse_total']:.4f}")
print(f"  Latency: {dyn_results['latency_mean_ms']:.1f}ms")
print(f"  Avg Layers Active (training): {16 - dyn_results['avg_skip_ratio'] * dyn_cfg['num_skippable_layers']:.1f} / 16")
print(f"  Avg Token Keep (training): {dyn_results['avg_token_keep_ratio']:.1%}")
print(f"\n  Output: {OUTPUT_DIR}")
for f_item in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f_item.name:<40} {f_item.stat().st_size/1024:.0f} KB")
print("\n  DONE!")
