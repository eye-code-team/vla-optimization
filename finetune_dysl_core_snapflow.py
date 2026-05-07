"""
Fine-Tune SmolVLA with DySL Core + SnapFlow
===========================================
This script intentionally implements:
    1) Dynamic-static layer skipping with informative static layers
    2) Prior-post skipping guidance for robust skip decisions
    3) SnapFlow self-distillation (teacher multi-step -> student shortcut)

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
import math
import os
import sys
import time
import types
from collections import deque
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
    execute_mask = getattr(self, "_dynskip_execute_mask", None)
    collect_similarity = bool(getattr(self, "_dynskip_collect_similarity", False))
    if execute_mask is None and not collect_similarity:
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
    skip_targets = getattr(self, "_dynskip_skip_targets", {}) or {}
    adapter_map = getattr(self, "_dynskip_adapter_map", {}) or {}
    head_dim = self.vlm.config.text_config.head_dim
    active_layers = 0
    skip_event_layer = -1

    if collect_similarity:
        if getattr(self, "_dynskip_similarity_sums", None) is None:
            self._dynskip_similarity_sums = [0.0] * num_layers
            self._dynskip_similarity_counts = [0] * num_layers

    layer_idx = 0
    while layer_idx < num_layers:
        execute_layer = True
        if execute_mask is not None and layer_idx < len(execute_mask):
            execute_layer = bool(execute_mask[layer_idx])

        if not execute_layer:
            if layer_idx in skip_targets:
                adapter = adapter_map.get(layer_idx)
                if adapter is not None:
                    adapted = []
                    for hidden_states in inputs_embeds:
                        if hidden_states is None:
                            adapted.append(None)
                        else:
                            adapted.append(adapter(hidden_states))
                    inputs_embeds = adapted
                if skip_event_layer < 0:
                    skip_event_layer = layer_idx
                layer_idx = int(skip_targets[layer_idx])
                continue
            layer_idx += 1
            continue

        active_layers += 1
        previous_embeds = inputs_embeds
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

        if collect_similarity and previous_embeds and inputs_embeds:
            old_h = previous_embeds[0]
            new_h = inputs_embeds[0]
            if old_h is not None and new_h is not None:
                old_flat = old_h.detach().float().reshape(old_h.shape[0], -1)
                new_flat = new_h.detach().float().reshape(new_h.shape[0], -1)
                cos = F.cosine_similarity(old_flat, new_flat, dim=-1).mean().item()
                self._dynskip_similarity_sums[layer_idx] += float(cos)
                self._dynskip_similarity_counts[layer_idx] += 1

        layer_idx += 1

    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is not None:
            outputs_embeds.append(models[i].norm(hidden_states))
        else:
            outputs_embeds.append(None)

    self._dynskip_last_active_layers = active_layers
    self._dynskip_last_skip_start_layer = skip_event_layer
    return outputs_embeds, past_key_values


def install_runtime_layer_skip_patch(vlm_with_expert):
    if getattr(vlm_with_expert, "_dynskip_patch_installed", False):
        return

    vlm_with_expert._dynskip_original_forward = vlm_with_expert.forward
    vlm_with_expert.forward = types.MethodType(_forward_with_runtime_skip, vlm_with_expert)
    vlm_with_expert._dynskip_patch_installed = True
    vlm_with_expert._dynskip_execute_mask = None
    vlm_with_expert._dynskip_skip_targets = None
    vlm_with_expert._dynskip_adapter_map = None
    vlm_with_expert._dynskip_last_active_layers = vlm_with_expert.num_vlm_layers
    vlm_with_expert._dynskip_last_skip_start_layer = -1
    vlm_with_expert._dynskip_collect_similarity = False
    vlm_with_expert._dynskip_similarity_sums = None
    vlm_with_expert._dynskip_similarity_counts = None


def set_runtime_skip_plan(vlm_with_expert, execute_mask, skip_targets=None, adapter_map=None):
    if isinstance(execute_mask, torch.Tensor):
        execute_mask = execute_mask.detach().flatten().tolist()
    vlm_with_expert._dynskip_execute_mask = [bool(x) for x in execute_mask]
    vlm_with_expert._dynskip_skip_targets = dict(skip_targets or {})
    vlm_with_expert._dynskip_adapter_map = dict(adapter_map or {})


def clear_runtime_skip_plan(vlm_with_expert):
    vlm_with_expert._dynskip_execute_mask = None
    vlm_with_expert._dynskip_skip_targets = None
    vlm_with_expert._dynskip_adapter_map = None


def reset_runtime_similarity_stats(vlm_with_expert):
    vlm_with_expert._dynskip_similarity_sums = [0.0] * vlm_with_expert.num_vlm_layers
    vlm_with_expert._dynskip_similarity_counts = [0] * vlm_with_expert.num_vlm_layers


def set_runtime_similarity_collection(vlm_with_expert, enabled):
    vlm_with_expert._dynskip_collect_similarity = bool(enabled)


# =====================================================================
#  MODULES: STAR + SNAPFLOW
# =====================================================================
class LayerSkipAdapter(nn.Module):
    """Lightweight adapter to bridge skipped dynamic layers."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        bottleneck = max(128, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_dim),
        )

    def forward(self, x):
        if x.shape[-1] != self.hidden_dim:
            return x
        return x + self.net(x)


class STARRouter(nn.Module):
    """Spatial-temporal router predicting skip propensity for all VLM layers."""

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = int(num_layers)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, self.num_layers),
        )
        nn.init.constant_(self.gate_net[-1].bias, 1.5)

    def forward(self, hidden_pooled, e_view, delta_s_norm, tau=1.0, hard=False):
        x = torch.cat([hidden_pooled, e_view, delta_s_norm], dim=-1)
        logits = self.gate_net(x)

        if hard or not self.training:
            gates = (torch.sigmoid(logits) > 0.5).float()
        else:
            logits_2class = torch.stack([torch.zeros_like(logits), logits], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_2class, tau=tau, hard=False, dim=-1)
            gates = gumbel_out[..., 1]

        return gates


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
        clear_runtime_skip_plan(self.teacher_model.vlm_with_expert)

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

        clear_runtime_skip_plan(teacher_flow_model.vlm_with_expert)

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

        self.num_vlm_layers = int(self.flow_model.vlm_with_expert.num_vlm_layers)
        vlm_hidden = self.flow_model.vlm_with_expert.config.text_config.hidden_size
        self.star_router = STARRouter(vlm_hidden, self.num_vlm_layers)
        self.snap_trainer = SnapFlowTrainer(dyn_cfg["snap_teacher_steps"])

        self.skip_adapters = nn.ModuleDict(
            {str(i): LayerSkipAdapter(vlm_hidden) for i in range(self.num_vlm_layers - 1)}
        )

        self._prev_state = None
        self._prev_action = None
        self.delta_history = deque(maxlen=max(int(self.cfg.get("continuity_k", 5)) * 2, 16))
        self.continuity_history = deque(maxlen=3)
        self.forward_step = 0
        self.last_post_verify_step = -10**9

        self.skip_stats = []
        self.active_layer_stats = []
        self.skip_trigger_layers = []
        self.post_verify_count = 0

        self.static_layer_ids = []
        self.dynamic_layer_ids = []
        self.segment_info = []
        self.allow_points = {}
        self.informative_scores = [0.0] * self.num_vlm_layers
        self._initialize_default_static_layers()

    def _initialize_default_static_layers(self):
        if self.cfg.get("manual_static_layer_ids"):
            static_ids = sorted({int(i) for i in self.cfg["manual_static_layer_ids"] if 0 <= int(i) < self.num_vlm_layers})
        else:
            anchor_front = int(self.cfg.get("num_anchor_front_layers", 2))
            anchor_front = max(1, min(anchor_front, self.num_vlm_layers))
            static_ids = list(range(anchor_front))
            static_ids.append(self.num_vlm_layers - 1)
        self._set_static_layers(static_ids)

    def _set_static_layers(self, static_ids):
        static_set = {int(i) for i in static_ids if 0 <= int(i) < self.num_vlm_layers}
        static_set.add(0)
        static_set.add(self.num_vlm_layers - 1)
        self.static_layer_ids = sorted(static_set)
        dynamic_set = set(range(self.num_vlm_layers)) - static_set
        self.dynamic_layer_ids = sorted(dynamic_set)
        self._build_segments()

    def _build_segments(self):
        self.segment_info = []
        self.allow_points = {}
        for idx in range(len(self.static_layer_ids) - 1):
            left = self.static_layer_ids[idx]
            right = self.static_layer_ids[idx + 1]
            dynamic_layers = list(range(left + 1, right))
            if not dynamic_layers:
                continue
            self.segment_info.append(
                {
                    "left_static": left,
                    "right_static": right,
                    "dynamic_layers": dynamic_layers,
                }
            )
            self.allow_points[len(self.segment_info) - 1] = dynamic_layers[0]

    @torch.no_grad()
    def calibrate_static_layers(self, batch_builder, train_idx):
        calib_batches = int(self.cfg.get("calibration_batches", 12))
        calib_batch_size = int(self.cfg.get("calibration_batch", 2))
        if calib_batches <= 0:
            return

        vlm = self.flow_model.vlm_with_expert
        reset_runtime_similarity_stats(vlm)
        set_runtime_similarity_collection(vlm, True)

        was_training = self.policy.training
        self.policy.eval()

        for _ in range(calib_batches):
            bi = np.random.choice(train_idx, calib_batch_size, replace=True)
            batch = batch_builder(bi)
            images, img_masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]
            actions = self.policy.prepare_action(batch)
            _ = self.flow_model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)

        set_runtime_similarity_collection(vlm, False)
        if was_training:
            self.policy.train()

        sums = getattr(vlm, "_dynskip_similarity_sums", [0.0] * self.num_vlm_layers)
        counts = getattr(vlm, "_dynskip_similarity_counts", [1] * self.num_vlm_layers)
        avg_cos = [float(sums[i]) / max(int(counts[i]), 1) for i in range(self.num_vlm_layers)]
        informative_scores = [1.0 - c for c in avg_cos]
        self.informative_scores = informative_scores

        target_static = int(round(float(self.cfg.get("static_layer_ratio", 0.25)) * self.num_vlm_layers))
        target_static = max(2, min(self.num_vlm_layers, target_static))

        anchor_front = int(self.cfg.get("num_anchor_front_layers", 2))
        anchor_front = max(1, min(anchor_front, self.num_vlm_layers))
        static_set = set(range(anchor_front))
        static_set.add(self.num_vlm_layers - 1)

        remaining = [i for i in range(self.num_vlm_layers) if i not in static_set]
        ranked = sorted(remaining, key=lambda i: informative_scores[i], reverse=True)
        needed = max(0, target_static - len(static_set))
        static_set.update(ranked[:needed])

        self._set_static_layers(sorted(static_set))

    def reset_episode(self):
        self._prev_state = None
        self._prev_action = None
        self.delta_history.clear()
        self.continuity_history.clear()
        self.policy.reset()
        for idx, seg in enumerate(self.segment_info):
            self.allow_points[idx] = seg["dynamic_layers"][0]

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

    def _compute_continuity(self, actions):
        current_action = actions[:, 0, :].detach()
        if self._prev_action is None:
            delta_norm = 0.0
        else:
            delta = current_action - self._prev_action
            delta_norm = float(delta.norm(dim=-1).mean().item())

        self._prev_action = current_action
        self.delta_history.append(delta_norm)

        k = max(1, int(self.cfg.get("continuity_k", 5)))
        recent = list(self.delta_history)[-k:]
        continuity = -float(np.mean(recent)) if recent else 0.0

        prev_cont = self.continuity_history[-1] if self.continuity_history else continuity
        d_cont = continuity - prev_cont
        prev_d_cont = 0.0
        if len(self.continuity_history) >= 2:
            prev_d_cont = self.continuity_history[-1] - self.continuity_history[-2]

        self.continuity_history.append(continuity)
        return continuity, d_cont, prev_d_cont

    def _update_allow_points(self, d_cont):
        eta = float(self.cfg.get("continuity_eta", 1e-3))
        if eta <= 0:
            return

        for idx, seg in enumerate(self.segment_info):
            allow = int(self.allow_points[idx])
            dyn_start = seg["dynamic_layers"][0]
            dyn_end = seg["dynamic_layers"][-1]
            if d_cont < -eta and allow < dyn_end:
                stride = int(math.ceil(abs(d_cont) / eta))
                stride = max(1, stride)
                allow = min(dyn_end, allow + stride)
            elif d_cont > eta and allow > dyn_start:
                allow = max(dyn_start, allow - 1)
            self.allow_points[idx] = allow

    def _build_skip_plan(self, gates, enable_skip):
        execute_mask = [True] * self.num_vlm_layers
        skip_targets = {}
        adapter_map = {}

        if not enable_skip or not self.dynamic_layer_ids:
            gate_loss = torch.tensor(0.0, device=gates.device)
            return execute_mask, skip_targets, adapter_map, gate_loss

        gate_scores = gates.mean(dim=0)
        dynamic_tensor = gates[:, self.dynamic_layer_ids]
        gate_loss = dynamic_tensor.mean()
        threshold = float(self.cfg.get("gate_threshold", 0.5))

        for seg_idx, seg in enumerate(self.segment_info):
            dyn_layers = seg["dynamic_layers"]
            allow_point = int(self.allow_points.get(seg_idx, dyn_layers[0]))
            candidates = [l for l in dyn_layers if l >= allow_point]
            if not candidates:
                continue

            skip_start = None
            for layer_idx in candidates:
                if float(gate_scores[layer_idx].item()) > threshold:
                    skip_start = layer_idx
                    break

            if skip_start is None:
                continue

            for layer_idx in dyn_layers:
                if layer_idx >= skip_start:
                    execute_mask[layer_idx] = False

            next_static = int(seg["right_static"])
            skip_targets[skip_start] = next_static
            adapter_map[skip_start] = self.skip_adapters[str(skip_start)]

        return execute_mask, skip_targets, adapter_map, gate_loss

    def _should_post_verify(self, enable_skip, d_cont, prev_d_cont):
        if not enable_skip or not bool(self.cfg.get("post_verify_enabled", True)):
            return False

        eta1 = float(self.cfg.get("post_verify_eta", 5e-4))
        cooldown = int(self.cfg.get("post_verify_cooldown", 2))
        if (self.forward_step - self.last_post_verify_step) <= cooldown:
            return False

        return d_cont < -eta1 and prev_d_cont > eta1

    def forward_with_skip(self, batch, tau=1.0, enable_skip=True, enable_snap=False, noise=None, time_val=None):
        self.forward_step += 1
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
        continuity, d_cont, prev_d_cont = self._compute_continuity(actions)
        if enable_skip:
            self._update_allow_points(d_cont)

        if enable_skip:
            expected_dim = self.star_router.gate_net[0].in_features - 2
            if all_img_emb is None:
                hidden_pooled = torch.zeros(bsize, expected_dim, device=device)
            else:
                hidden_pooled = all_img_emb.mean(dim=1)
                if hidden_pooled.shape[-1] != expected_dim:
                    hidden_pooled = F.adaptive_avg_pool1d(hidden_pooled.unsqueeze(1), expected_dim).squeeze(1)
            gates = self.star_router(hidden_pooled, e_view, delta_s_norm, tau=tau)
        else:
            gates = torch.zeros(bsize, self.num_vlm_layers, device=device)

        execute_mask, skip_targets, adapter_map, gate_loss = self._build_skip_plan(gates, enable_skip)
        avg_skip_ratio = 0.0
        if self.dynamic_layer_ids:
            skipped_dyn = sum(1 for l in self.dynamic_layer_ids if not execute_mask[l])
            avg_skip_ratio = float(skipped_dyn) / float(len(self.dynamic_layer_ids))

        if enable_skip:
            set_runtime_skip_plan(flow_model.vlm_with_expert, execute_mask, skip_targets, adapter_map)
        else:
            clear_runtime_skip_plan(flow_model.vlm_with_expert)

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
            skip_start_layer = int(getattr(flow_model.vlm_with_expert, "_dynskip_last_skip_start_layer", -1))
        finally:
            clear_runtime_skip_plan(flow_model.vlm_with_expert)

        post_verified = 0.0
        if self._should_post_verify(enable_skip, d_cont, prev_d_cont):
            clear_runtime_skip_plan(flow_model.vlm_with_expert)
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
            skip_start_layer = -1
            post_verified = 1.0
            self.post_verify_count += 1
            self.last_post_verify_step = self.forward_step

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
        run_snap_with_skip = enable_skip and (post_verified < 0.5)
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
            x_t_full = t_expanded * noise + (1 - t_expanded) * actions
            x_t = x_t_full[:, :, :original_action_dim]

            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            if run_snap_with_skip:
                set_runtime_skip_plan(flow_model.vlm_with_expert, execute_mask, skip_targets, adapter_map)

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
                clear_runtime_skip_plan(flow_model.vlm_with_expert)

            snap_loss = self.snap_trainer.compute_snapflow_loss(student_v, x_t, teacher_x0, time_val)

        total_loss = task_loss
        if enable_skip:
            total_loss = total_loss + float(self.cfg["lambda_gate"]) * gate_loss
        if enable_snap:
            total_loss = total_loss + float(self.cfg["lambda_snap"]) * snap_loss

        active_layers_target = float(sum(1 for x in execute_mask if x))
        self.skip_stats.append(avg_skip_ratio)
        self.active_layer_stats.append(float(active_layers_runtime))
        self.skip_trigger_layers.append(float(skip_start_layer))

        gate_score_mean = float(gates[:, self.dynamic_layer_ids].mean().item()) if self.dynamic_layer_ids else 0.0
        allow_mean = (
            float(np.mean([self.allow_points[idx] for idx in self.allow_points]))
            if self.allow_points
            else 0.0
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "task_loss": float(task_loss.item()),
            "gate_loss": float(gate_loss.item()),
            "snap_loss": float(snap_loss.item()),
            "avg_skip_ratio": float(avg_skip_ratio),
            "active_layers_target": float(active_layers_target),
            "active_layers_runtime": float(active_layers_runtime),
            "gate_score_mean": gate_score_mean,
            "continuity": float(continuity),
            "continuity_delta": float(d_cont),
            "allow_point_mean": allow_mean,
            "post_verified": float(post_verified),
            "skip_start_layer": float(skip_start_layer),
            "num_static_layers": float(len(self.static_layer_ids)),
        }
        return total_loss, loss_dict


# =====================================================================
#  TRAINING ENTRYPOINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", type=str, default="svla_so100_pickplace")
    parser.add_argument("--config_key", type=str, default="dysl_core_snapflow")
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

    output_dir = Path(args.output_dir) if args.output_dir else Path("d:/EyetechCode/results/dysl_core_snapflow")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SmolVLA Fine-Tune: DySL Core + SnapFlow")
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
    adapter_params = sum(p.numel() for p in wrapper.skip_adapters.parameters())
    print(f"  STAR router params: {star_params / 1e6:.3f}M")
    print(f"  Skip adapter params: {adapter_params / 1e6:.3f}M")

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
    print("  PHASE 2.5: Calibrate Informative Static Layers")
    print("=" * 70)
    wrapper.calibrate_static_layers(build_train_batch, train_idx)
    print(f"  Static layers: {wrapper.static_layer_ids}")
    print(f"  Dynamic layers: {wrapper.dynamic_layer_ids}")
    print(f"  Dynamic segments: {len(wrapper.segment_info)}")

    print("\n" + "=" * 70)
    print("  PHASE 3: Train Dynamic Layer Skipping")
    print("=" * 70)

    phase1_params = [p for p in wrapper.parameters() if p.requires_grad]
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
                    "skip_adapters": wrapper.skip_adapters.state_dict(),
                    "static_layer_ids": wrapper.static_layer_ids,
                    "dynamic_layer_ids": wrapper.dynamic_layer_ids,
                },
                output_dir / f"phase1_step{step + 1}.pt",
            )

    torch.save(
        {
            "smolvla": smolvla.state_dict(),
            "star_router": wrapper.star_router.state_dict(),
            "skip_adapters": wrapper.skip_adapters.state_dict(),
            "static_layer_ids": wrapper.static_layer_ids,
            "dynamic_layer_ids": wrapper.dynamic_layer_ids,
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
    phase2_params = [p for p in wrapper.parameters() if p.requires_grad]
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
                    "skip_adapters": wrapper.skip_adapters.state_dict(),
                    "static_layer_ids": wrapper.static_layer_ids,
                    "dynamic_layer_ids": wrapper.dynamic_layer_ids,
                },
                output_dir / f"phase2_step{step + 1}.pt",
            )

    torch.save(
        {
            "smolvla": smolvla.state_dict(),
            "star_router": wrapper.star_router.state_dict(),
            "skip_adapters": wrapper.skip_adapters.state_dict(),
            "static_layer_ids": wrapper.static_layer_ids,
            "dynamic_layer_ids": wrapper.dynamic_layer_ids,
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
    valid_skip_start = [x for x in wrapper.skip_trigger_layers if x >= 0]
    avg_skip_start = float(np.mean(valid_skip_start)) if valid_skip_start else -1.0

    report = {
        "pipeline": "finetune_dysl_core_snapflow",
        "dataset": ds_cfg["repo_id"],
        "dataset_key": args.dataset_key,
        "train_episodes": list(train_eps),
        "eval_episodes": list(eval_eps[: EVAL["num_eval_episodes"]]),
        "config": {
            "phase1_steps": dyn_cfg["phase1_steps"],
            "phase2_steps": dyn_cfg["phase2_steps"],
            "phase1_lr": dyn_cfg["phase1_lr"],
            "phase2_lr": dyn_cfg["phase2_lr"],
            "static_layer_ratio": dyn_cfg.get("static_layer_ratio"),
            "continuity_k": dyn_cfg.get("continuity_k"),
            "continuity_eta": dyn_cfg.get("continuity_eta"),
            "post_verify_eta": dyn_cfg.get("post_verify_eta"),
            "snap_teacher_steps": dyn_cfg["snap_teacher_steps"],
            "lambda_gate": dyn_cfg["lambda_gate"],
            "lambda_snap": dyn_cfg["lambda_snap"],
        },
        "results": {
            "phase1_final_loss": float(np.mean(phase1_losses[-50:])),
            "phase2_final_loss": float(np.mean(phase2_losses[-50:])),
            "avg_skip_ratio": avg_skip,
            "avg_active_layers_runtime": avg_active_runtime,
            "avg_skip_start_layer": avg_skip_start,
            "post_verify_count": int(wrapper.post_verify_count),
            "num_static_layers": len(wrapper.static_layer_ids),
            "num_dynamic_layers": len(wrapper.dynamic_layer_ids),
            "static_layer_ids": list(wrapper.static_layer_ids),
            "dynamic_layer_ids": list(wrapper.dynamic_layer_ids),
        },
    }
    with open(output_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE: finetune_dysl_core_snapflow")
    print("=" * 70)
    print(f"  Avg skip ratio (last 500): {avg_skip:.3f}")
    print(f"  Avg active layers runtime (last 500): {avg_active_runtime:.2f}")
    print(f"  Post-verify triggers: {wrapper.post_verify_count}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
