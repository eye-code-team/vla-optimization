"""
Fine-Tune SmolVLA with Dynamic Layer Skipping + SnapFlow Only
==============================================================
This script intentionally implements only:
  1) Dynamic layer skipping via STAR router
  2) SnapFlow self-distillation (teacher multi-step -> student shortcut)

Out of scope by design:
  - LoRA-SP
  - Token pruning
  - CogKD / CORAL / Lyapunov variants

Important: true layer skipping is implemented at runtime through a monkey-patch
on SmolVLMWithExpert.forward. Core source files are not modified.
"""

import argparse
import contextlib
import copy
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =====================================================================
#  PATCH: Fix lerobot import chain on Windows/Python 3.10
# =====================================================================
import lerobot

_pkg = lerobot.__path__[0]
import importlib.util

_robots_mod = types.ModuleType("lerobot.robots")
_robots_mod.__path__ = [os.path.join(_pkg, "robots")]
_robots_mod.__package__ = "lerobot.robots"
_spec = importlib.util.spec_from_file_location("lerobot.robots.config", os.path.join(_pkg, "robots", "config.py"))
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


from finetune_config import DATASETS, EVAL, TRAINING
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
#  RUNTIME MONKEY-PATCH FOR TRUE LAYER SKIPPING
# =====================================================================
def _forward_with_runtime_skip(
    self,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.FloatTensor] | None = None,
    inputs_embeds: list[torch.FloatTensor] = None,
    use_cache: bool | None = None,
    fill_kv_cache: bool | None = None,
):
    keep_mask = getattr(self, "_dynskip_keep_mask", None)
    if keep_mask is None:
        outputs_embeds, past_key_values = self._dynskip_original_forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        self._dynskip_last_active_layers = self.num_vlm_layers
        return outputs_embeds, past_key_values

    models = [self.get_vlm_model().text_model, self.lm_expert]
    model_layers = self.get_model_layers(models)

    batch_size = None
    for hidden_states in inputs_embeds:
        if hidden_states is not None:
            batch_size = hidden_states.shape[0]
            break
    if batch_size is None:
        raise ValueError("No valid embeddings in inputs_embeds.")

    num_layers = self.num_vlm_layers
    head_dim = self.vlm.config.text_config.head_dim
    num_fixed = int(getattr(self, "_dynskip_num_fixed_layers", 0))
    active_layers = 0

    for layer_idx in range(num_layers):
        should_skip = False
        if layer_idx >= num_fixed:
            relative_idx = layer_idx - num_fixed
            if relative_idx < len(keep_mask):
                should_skip = not bool(keep_mask[relative_idx])

        if should_skip:
            continue

        active_layers += 1
        if (
            fill_kv_cache
            or "cross" not in self.attention_mode
            or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
        ):
            att_outputs, past_key_values = self.forward_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )
        else:
            att_outputs, past_key_values = self.forward_cross_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )

        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]

            if hidden_states is None:
                outputs_embeds.append(None)
                continue

            if layer is None:
                outputs_embeds.append(hidden_states)
                continue

            end = start + hidden_states.shape[1]
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            att_out = att_output[:, start:end]

            out_emb = layer.self_attn.o_proj(att_out)
            out_emb += hidden_states
            after_first_residual = out_emb.clone()

            out_emb = layer.post_attention_layernorm(out_emb)
            out_emb = layer.mlp(out_emb)
            out_emb += after_first_residual

            outputs_embeds.append(out_emb)
            start = end if len(att_outputs) == 1 else 0

        inputs_embeds = outputs_embeds

    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is not None:
            outputs_embeds.append(models[i].norm(hidden_states))
        else:
            outputs_embeds.append(None)

    self._dynskip_last_active_layers = active_layers
    return outputs_embeds, past_key_values


def install_runtime_layer_skip_patch(vlm_with_expert):
    if getattr(vlm_with_expert, "_dynskip_patch_installed", False):
        return

    vlm_with_expert._dynskip_original_forward = vlm_with_expert.forward
    vlm_with_expert.forward = types.MethodType(_forward_with_runtime_skip, vlm_with_expert)
    vlm_with_expert._dynskip_patch_installed = True
    vlm_with_expert._dynskip_keep_mask = None
    vlm_with_expert._dynskip_num_fixed_layers = 0
    vlm_with_expert._dynskip_last_active_layers = vlm_with_expert.num_vlm_layers


def set_runtime_keep_mask(vlm_with_expert, keep_mask, num_fixed_layers):
    if isinstance(keep_mask, torch.Tensor):
        keep_mask = keep_mask.detach().flatten().tolist()
    vlm_with_expert._dynskip_keep_mask = [bool(x) for x in keep_mask]
    vlm_with_expert._dynskip_num_fixed_layers = int(num_fixed_layers)


def clear_runtime_keep_mask(vlm_with_expert):
    vlm_with_expert._dynskip_keep_mask = None
    vlm_with_expert._dynskip_num_fixed_layers = 0


# =====================================================================
#  MODULES: STAR + SNAPFLOW
# =====================================================================
class STARRouter(nn.Module):
    """Spatial-Temporal Aware Router for dynamic layer selection."""

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

        gate_loss = gates.mean()
        return gates, gate_loss


class SnapFlowTrainer:
    """Basic SnapFlow trainer: teacher multi-step, student shortcut."""

    def __init__(self, num_teacher_steps=10):
        self.num_teacher_steps = num_teacher_steps
        self.teacher_model = None

    def create_teacher(self, model):
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        install_runtime_layer_skip_patch(self.teacher_model.vlm_with_expert)
        clear_runtime_keep_mask(self.teacher_model.vlm_with_expert)

    @torch.no_grad()
    def compute_teacher_target(
        self,
        teacher_flow_model,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        noise,
    ):
        bsize = noise.shape[0]
        device = noise.device

        clear_runtime_keep_mask(teacher_flow_model.vlm_with_expert)

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

        dt = -1.0 / self.num_teacher_steps
        x_t = noise.clone()
        for step in range(self.num_teacher_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
            v_t = teacher_flow_model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=t_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    def compute_snapflow_loss(self, student_velocity, x_t, teacher_x0, t):
        t_expanded = t[:, None, None]
        shortcut_v = (x_t - teacher_x0) / (t_expanded + 1e-6)
        return F.mse_loss(student_velocity, shortcut_v)


class DynamicLayerSkipSnapFlowWrapper(nn.Module):
    def __init__(self, smolvla_policy, dyn_cfg):
        super().__init__()
        self.policy = smolvla_policy
        self.flow_model = smolvla_policy.model
        self.cfg = dyn_cfg

        install_runtime_layer_skip_patch(self.flow_model.vlm_with_expert)

        vlm_hidden = self.flow_model.vlm_with_expert.config.text_config.hidden_size
        self.star_router = STARRouter(vlm_hidden, dyn_cfg["num_skippable_layers"])
        self.snap_trainer = SnapFlowTrainer(dyn_cfg["snap_teacher_steps"])

        self._prev_state = None
        self.skip_stats = []
        self.active_layer_stats = []

    def reset_episode(self):
        self._prev_state = None
        self.policy.reset()

    def compute_context_features(self, state, images_emb=None):
        bsize = state.shape[0]
        device = state.device

        if images_emb is not None:
            e_view = images_emb.var(dim=1).mean(dim=-1, keepdim=True)
        else:
            e_view = torch.zeros(bsize, 1, device=device)

        if self._prev_state is not None:
            delta_s = state - self._prev_state
            delta_s_norm = delta_s.norm(dim=-1, keepdim=True)
        else:
            delta_s_norm = torch.zeros(bsize, 1, device=device)

        self._prev_state = state.detach().clone()
        return e_view, delta_s_norm

    def compute_batch_keep_mask(self, gates):
        num_skip = self.cfg["num_skippable_layers"]
        keep_ratio = float(self.cfg["skip_keep_ratio"])
        keep_count = int(round(num_skip * keep_ratio))
        keep_count = max(1, min(num_skip, keep_count))

        gate_scores = gates.mean(dim=0)
        _, top_idx = torch.topk(gate_scores, k=keep_count, dim=-1)
        keep_mask = torch.zeros(num_skip, dtype=torch.bool, device=gates.device)
        keep_mask[top_idx] = True
        return keep_mask, gate_scores

    def forward_with_skip(self, batch, tau=1.0, enable_skip=True, enable_snap=False, noise=None, time_val=None):
        flow_model = self.flow_model

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        actions = self.policy.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        bsize = state.shape[0]
        device = state.device

        with torch.no_grad():
            img_embs_list = []
            for img in images:
                img_emb = flow_model.vlm_with_expert.embed_image(img)
                img_embs_list.append(img_emb)
            all_img_emb = torch.cat(img_embs_list, dim=1) if img_embs_list else None

        e_view, delta_s_norm = self.compute_context_features(state, all_img_emb)

        if enable_skip:
            expected_dim = self.star_router.gate_net[0].in_features - 2
            if all_img_emb is None:
                hidden_pooled = torch.zeros(bsize, expected_dim, device=device)
            else:
                hidden_pooled = all_img_emb.mean(dim=1)
                if hidden_pooled.shape[-1] != expected_dim:
                    hidden_pooled = F.adaptive_avg_pool1d(hidden_pooled.unsqueeze(1), expected_dim).squeeze(1)

            gates, gate_loss = self.star_router(hidden_pooled, e_view, delta_s_norm, tau=tau)
            keep_mask, gate_scores = self.compute_batch_keep_mask(gates)
        else:
            gates = torch.ones(bsize, self.cfg["num_skippable_layers"], device=device)
            gate_loss = torch.tensor(0.0, device=device)
            keep_mask = torch.ones(self.cfg["num_skippable_layers"], dtype=torch.bool, device=device)
            gate_scores = keep_mask.float()

        if enable_skip:
            self.skip_stats.append(1.0 - keep_mask.float().mean().item())

        if enable_skip:
            set_runtime_keep_mask(
                flow_model.vlm_with_expert,
                keep_mask,
                self.cfg["num_fixed_layers"],
            )
        else:
            clear_runtime_keep_mask(flow_model.vlm_with_expert)

        try:
            losses = flow_model.forward(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                actions,
                noise=noise,
                time=time_val,
            )
            active_layers_runtime = int(getattr(flow_model.vlm_with_expert, "_dynskip_last_active_layers", 0))
        finally:
            clear_runtime_keep_mask(flow_model.vlm_with_expert)

        if self.policy.config.action_feature is not None:
            original_action_dim = int(self.policy.config.action_feature.shape[0])
        else:
            original_action_dim = actions.shape[-1]

        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
        task_loss = losses.mean()

        snap_loss = torch.tensor(0.0, device=device)
        if enable_snap and self.snap_trainer.teacher_model is not None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = flow_model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
            )

            if noise is None:
                noise = flow_model.sample_noise(actions.shape, actions.device)

            teacher_x0 = self.snap_trainer.compute_teacher_target(
                self.snap_trainer.teacher_model,
                prefix_embs.detach(),
                prefix_pad_masks.detach(),
                prefix_att_masks.detach(),
                noise,
            )
            teacher_x0 = teacher_x0[:, :, :original_action_dim]

            if time_val is None:
                time_val = flow_model.sample_time(bsize, actions.device)

            t_expanded = time_val[:, None, None]
            # Denoising must run in the model's full action space.
            x_t_full = t_expanded * noise + (1 - t_expanded) * actions
            x_t = x_t_full[:, :, :original_action_dim]

            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            if enable_skip:
                set_runtime_keep_mask(
                    flow_model.vlm_with_expert,
                    keep_mask,
                    self.cfg["num_fixed_layers"],
                )

            try:
                _, past_key_values = flow_model.vlm_with_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                    fill_kv_cache=True,
                )

                student_v = flow_model.denoise_step(
                    x_t=x_t_full,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_val,
                )
                student_v = student_v[:, :, :original_action_dim]
            finally:
                clear_runtime_keep_mask(flow_model.vlm_with_expert)

            snap_loss = self.snap_trainer.compute_snapflow_loss(student_v, x_t, teacher_x0, time_val)

        total_loss = task_loss
        if enable_skip:
            total_loss = total_loss + float(self.cfg["lambda_gate"]) * gate_loss
        if enable_snap:
            total_loss = total_loss + float(self.cfg["lambda_snap"]) * snap_loss

        num_fixed = int(self.cfg["num_fixed_layers"])
        active_layers_target = num_fixed + int(keep_mask.sum().item())
        self.active_layer_stats.append(float(active_layers_runtime))

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "task_loss": float(task_loss.item()),
            "gate_loss": float(gate_loss.item()),
            "snap_loss": float(snap_loss.item()),
            "avg_skip_ratio": float(1.0 - keep_mask.float().mean().item()),
            "active_layers_target": float(active_layers_target),
            "active_layers_runtime": float(active_layers_runtime),
            "gate_score_mean": float(gate_scores.mean().item()),
        }
        return total_loss, loss_dict


# =====================================================================
#  TRAINING ENTRYPOINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", type=str, default="svla_so100_pickplace")
    parser.add_argument("--config_key", type=str, default="dynamic_layerskip_snapflow_only")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--phase1_steps", type=int, default=-1)
    parser.add_argument("--phase2_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset_key not in DATASETS:
        raise KeyError(f"Unknown dataset_key: {args.dataset_key}")
    if args.config_key not in TRAINING:
        raise KeyError(f"Unknown config_key: {args.config_key}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds_cfg = DATASETS[args.dataset_key]
    dyn_cfg = dict(TRAINING[args.config_key])
    if args.phase1_steps > 0:
        dyn_cfg["phase1_steps"] = args.phase1_steps
    if args.phase2_steps > 0:
        dyn_cfg["phase2_steps"] = args.phase2_steps

    output_dir = Path(args.output_dir) if args.output_dir else Path("d:/EyetechCode/results/dynamic_layerskip_snapflow_only")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SmolVLA Fine-Tune: Dynamic Layer Skip + SnapFlow (Only)")
    print("=" * 70)
    if DEVICE == "cuda":
        print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  Device: {DEVICE}")
    print(f"  Dataset: {ds_cfg['repo_id']}")
    print(f"  Output: {output_dir}")
    print()

    print("=" * 70)
    print("  PHASE 1: Load Dataset")
    print("=" * 70)
    dataset = LeRobotDataset(ds_cfg["repo_id"])

    sample = dataset[0]
    all_keys = list(sample.keys())
    image_keys = [k for k in all_keys if "image" in k.lower() and isinstance(sample[k], torch.Tensor)]
    state_key = next((k for k in all_keys if "state" in k.lower() and isinstance(sample[k], torch.Tensor)), None)
    action_key = "action"
    action_dim = sample[action_key].shape[-1]

    print(f"  Episodes: {dataset.num_episodes}, Frames: {len(dataset)}, FPS: {dataset.fps}")
    print(f"  Image keys: {image_keys}")
    print(f"  State key: {state_key}")
    print(f"  Action dim: {action_dim}")

    ep_col = dataset.hf_dataset["episode_index"]
    episode_indices = {}
    for idx, ep in enumerate(ep_col):
        ep_int = ep.item() if isinstance(ep, torch.Tensor) else int(ep)
        if ep_int not in episode_indices:
            episode_indices[ep_int] = []
        episode_indices[ep_int].append(idx)

    all_eps = sorted(episode_indices.keys())
    train_eps = all_eps[: ds_cfg["train_episodes"]]
    eval_eps = all_eps[ds_cfg["train_episodes"] : ds_cfg["train_episodes"] + ds_cfg["eval_episodes"]]

    train_idx = []
    for ep in train_eps:
        train_idx.extend(episode_indices[ep])

    print(f"  Train: {len(train_eps)} eps ({len(train_idx)} frames)")
    print(f"  Eval: {len(eval_eps)} eps ({eval_eps})")

    print("\n" + "=" * 70)
    print("  PHASE 2: Load SmolVLA Base + Dynamic Wrapper")
    print("=" * 70)

    smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

    smolvla_img_keys = list(smolvla.config.image_features.keys())
    key_remap = {}
    for i, dk in enumerate(image_keys):
        if i < len(smolvla_img_keys):
            key_remap[dk] = smolvla_img_keys[i]
    print(f"  Key remap: {key_remap}")

    tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
    tok = tokenizer(ds_cfg["task_instruction"], return_tensors="pt", padding="max_length", max_length=64)
    lang_ids = tok["input_ids"]
    lang_mask = tok["attention_mask"].bool()

    if dyn_cfg["freeze_vlm"]:
        frozen = 0
        trainable_base = 0
        for name, param in smolvla.named_parameters():
            if "vlm_with_expert.vlm" in name:
                param.requires_grad = False
                frozen += param.numel()
            else:
                param.requires_grad = True
                trainable_base += param.numel()
        print(f"  Frozen (VLM): {frozen / 1e6:.1f}M")
        print(f"  Trainable (non-VLM): {trainable_base / 1e6:.1f}M")

    smolvla.to(DEVICE)
    wrapper = DynamicLayerSkipSnapFlowWrapper(smolvla, dyn_cfg).to(DEVICE)

    star_params = sum(p.numel() for p in wrapper.star_router.parameters())
    print(f"  STAR router params: {star_params / 1e6:.3f}M")

    chunk_size = smolvla.config.chunk_size

    def build_train_batch(indices):
        batch_imgs = {k: [] for k in key_remap.values()}
        batch_states = []
        batch_actions = []

        for idx in indices:
            s = dataset[idx]
            for dk, sk in key_remap.items():
                batch_imgs[sk].append(s[dk])
            if state_key:
                batch_states.append(s[state_key])
            batch_actions.append(s[action_key])

        batch = {}
        for sk, imgs in batch_imgs.items():
            batch[sk] = torch.stack(imgs).to(DEVICE)

        if batch_states:
            batch["observation.state"] = torch.stack(batch_states).to(DEVICE)

        actions = torch.stack(batch_actions).to(DEVICE)
        bsize = actions.shape[0]
        max_act_dim = smolvla.config.max_action_dim
        if actions.shape[-1] < max_act_dim:
            pad = torch.zeros(bsize, max_act_dim - actions.shape[-1], device=DEVICE)
            actions = torch.cat([actions, pad], dim=-1)

        action_chunk = actions.unsqueeze(1).expand(bsize, chunk_size, max_act_dim)
        batch["action"] = action_chunk
        action_is_pad = torch.zeros(bsize, chunk_size, dtype=torch.bool, device=DEVICE)
        batch["action_is_pad"] = action_is_pad
        batch["actions_id_pad"] = action_is_pad

        batch["observation.language.tokens"] = lang_ids.expand(bsize, -1).to(DEVICE)
        batch["observation.language.attention_mask"] = lang_mask.expand(bsize, -1).to(DEVICE)
        return batch

    micro_bs = dyn_cfg["micro_batch"]
    grad_accum = dyn_cfg["grad_accum"]

    def amp_ctx():
        if dyn_cfg["fp16"] and DEVICE == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    print("\n" + "=" * 70)
    print("  PHASE 3: Train Dynamic Layer Skipping")
    print("=" * 70)

    phase1_params = [p for p in smolvla.parameters() if p.requires_grad]
    phase1_params.extend(wrapper.star_router.parameters())
    opt_p1 = torch.optim.AdamW(
        phase1_params,
        lr=dyn_cfg["phase1_lr"],
        weight_decay=dyn_cfg["weight_decay"],
    )

    phase1_losses = []
    smolvla.train()
    wrapper.train()

    pbar = tqdm(range(dyn_cfg["phase1_steps"]), desc="  Phase1 DySL", ncols=100)
    for step in pbar:
        opt_p1.zero_grad()
        accum_loss = 0.0
        accum_dict = {}

        progress = step / max(dyn_cfg["phase1_steps"] - 1, 1)
        tau = dyn_cfg["gumbel_tau_start"] * (1 - progress) + dyn_cfg["gumbel_tau_end"] * progress

        for _ in range(grad_accum):
            bi = np.random.choice(train_idx, micro_bs, replace=True)
            batch = build_train_batch(bi)

            with amp_ctx():
                loss, loss_dict = wrapper.forward_with_skip(
                    batch,
                    tau=tau,
                    enable_skip=True,
                    enable_snap=False,
                )
                loss = loss / grad_accum

            loss.backward()
            accum_loss += float(loss.item())
            for k, v in loss_dict.items():
                accum_dict[k] = accum_dict.get(k, 0.0) + float(v) / grad_accum

        torch.nn.utils.clip_grad_norm_(phase1_params, dyn_cfg["max_grad_norm"])
        opt_p1.step()
        phase1_losses.append(accum_loss)

        if step % dyn_cfg.get("log_every", 100) == 0:
            avg_loss = float(np.mean(phase1_losses[-50:]))
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                skip=f"{accum_dict.get('avg_skip_ratio', 0):.2f}",
                tau=f"{tau:.2f}",
                active=f"{accum_dict.get('active_layers_runtime', 0):.1f}",
            )

        if (step + 1) % dyn_cfg["save_every"] == 0:
            torch.save(
                {
                    "smolvla": smolvla.state_dict(),
                    "star_router": wrapper.star_router.state_dict(),
                },
                output_dir / f"phase1_step{step + 1}.pt",
            )

    torch.save(
        {
            "smolvla": smolvla.state_dict(),
            "star_router": wrapper.star_router.state_dict(),
        },
        output_dir / "phase1_complete.pt",
    )
    print(f"  Phase 1 done. Final loss: {np.mean(phase1_losses[-50:]):.6f}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  PHASE 4: Train SnapFlow with Dynamic Skipping")
    print("=" * 70)

    wrapper.snap_trainer.create_teacher(smolvla.model)
    phase2_params = [p for p in smolvla.parameters() if p.requires_grad]
    phase2_params.extend(wrapper.star_router.parameters())
    opt_p2 = torch.optim.AdamW(
        phase2_params,
        lr=dyn_cfg["phase2_lr"],
        weight_decay=dyn_cfg["weight_decay"],
    )
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_p2,
        T_max=dyn_cfg["phase2_steps"],
        eta_min=1e-6,
    )

    phase2_losses = []
    pbar = tqdm(range(dyn_cfg["phase2_steps"]), desc="  Phase2 SnapFlow", ncols=100)
    for step in pbar:
        opt_p2.zero_grad()
        accum_loss = 0.0
        accum_dict = {}

        tau = dyn_cfg["gumbel_tau_end"]
        for _ in range(grad_accum):
            bi = np.random.choice(train_idx, micro_bs, replace=True)
            batch = build_train_batch(bi)

            with amp_ctx():
                loss, loss_dict = wrapper.forward_with_skip(
                    batch,
                    tau=tau,
                    enable_skip=True,
                    enable_snap=True,
                )
                loss = loss / grad_accum

            loss.backward()
            accum_loss += float(loss.item())
            for k, v in loss_dict.items():
                accum_dict[k] = accum_dict.get(k, 0.0) + float(v) / grad_accum

        torch.nn.utils.clip_grad_norm_(phase2_params, dyn_cfg["max_grad_norm"])
        opt_p2.step()
        scheduler_p2.step()
        phase2_losses.append(accum_loss)

        if step % dyn_cfg.get("log_every", 100) == 0:
            avg_loss = float(np.mean(phase2_losses[-50:]))
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                snap=f"{accum_dict.get('snap_loss', 0):.4f}",
                active=f"{accum_dict.get('active_layers_runtime', 0):.1f}",
                lr=f"{scheduler_p2.get_last_lr()[0]:.2e}",
            )

        if (step + 1) % dyn_cfg["save_every"] == 0:
            torch.save(
                {
                    "smolvla": smolvla.state_dict(),
                    "star_router": wrapper.star_router.state_dict(),
                },
                output_dir / f"phase2_step{step + 1}.pt",
            )

    torch.save(
        {
            "smolvla": smolvla.state_dict(),
            "star_router": wrapper.star_router.state_dict(),
        },
        output_dir / "final_model.pt",
    )
    print(f"  Phase 2 done. Final loss: {np.mean(phase2_losses[-50:]):.6f}")

    if wrapper.snap_trainer.teacher_model is not None:
        del wrapper.snap_trainer.teacher_model
        wrapper.snap_trainer.teacher_model = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  PHASE 5: Save Report")
    print("=" * 70)

    avg_skip = float(np.mean(wrapper.skip_stats[-500:])) if wrapper.skip_stats else 0.0
    avg_active_runtime = float(np.mean(wrapper.active_layer_stats[-500:])) if wrapper.active_layer_stats else 0.0

    report = {
        "pipeline": "finetune_dynamic_layerskip_snapflow_only",
        "dataset": ds_cfg["repo_id"],
        "dataset_key": args.dataset_key,
        "train_episodes": list(train_eps),
        "eval_episodes": list(eval_eps[: EVAL["num_eval_episodes"]]),
        "config": {
            "phase1_steps": dyn_cfg["phase1_steps"],
            "phase2_steps": dyn_cfg["phase2_steps"],
            "phase1_lr": dyn_cfg["phase1_lr"],
            "phase2_lr": dyn_cfg["phase2_lr"],
            "num_fixed_layers": dyn_cfg["num_fixed_layers"],
            "num_skippable_layers": dyn_cfg["num_skippable_layers"],
            "skip_keep_ratio": dyn_cfg["skip_keep_ratio"],
            "snap_teacher_steps": dyn_cfg["snap_teacher_steps"],
            "lambda_gate": dyn_cfg["lambda_gate"],
            "lambda_snap": dyn_cfg["lambda_snap"],
        },
        "results": {
            "phase1_final_loss": float(np.mean(phase1_losses[-50:])),
            "phase2_final_loss": float(np.mean(phase2_losses[-50:])),
            "avg_skip_ratio": avg_skip,
            "avg_active_layers_runtime": avg_active_runtime,
        },
    }
    with open(output_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE: finetune_dynamic_layerskip_snapflow_only")
    print("=" * 70)
    print(f"  Avg skip ratio (last 500): {avg_skip:.3f}")
    print(f"  Avg active layers runtime (last 500): {avg_active_runtime:.2f}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
