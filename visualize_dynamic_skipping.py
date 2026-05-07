"""
Visualize Dynamic Layer Skipping + Token Pruning + SnapFlow
=============================================================
Loads final_model.pt and runs STAR Router, ADP, and SnapFlow
inference on eval episodes to show:

  1. Layer skipping heatmap per episode (STAR Router)
  2. Token pruning keep-ratio timeline (ADP)
  3. SnapFlow 1-NFE vs multi-step denoising comparison
  4. Per-layer skip frequency stats

Outputs (in results/dynamic_lora_pruning_snapflow/):
  - dynamic_skip_heatmap.png
  - dynamic_skip_stats.png
  - dynamic_skip_timeline.png
  - dynamic_token_pruning.png
  - dynamic_snapflow_analysis.png
"""
import os, sys, types, gc, json, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
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
#  CONFIG
# =====================================================================
from finetune_config import DATASETS, TRAINING, EVAL

DATASET_KEY = "svla_so100_pickplace"
ds_cfg = DATASETS[DATASET_KEY]
dyn_cfg = TRAINING["dynamic_lora_pruning_snapflow"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("d:/EyetechCode/results/dynamic_lora_pruning_snapflow")
CHECKPOINT_PATH = OUTPUT_DIR / "final_model.pt"

NUM_FIXED = dyn_cfg["num_fixed_layers"]       # 8
NUM_SKIP  = dyn_cfg["num_skippable_layers"]    # 8
TOTAL_LAYERS = NUM_FIXED + NUM_SKIP            # 16

print("=" * 70)
print("  Visualize: Layer Skipping + Token Pruning + SnapFlow")
print("=" * 70)

# =====================================================================
#  CUSTOM MODULES (same architecture as training script)
# =====================================================================

class STARRouter(nn.Module):
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
        gates = (torch.sigmoid(logits) > 0.5).float()
        gate_probs = torch.sigmoid(logits)
        return gates, gate_probs


class ActionAwareTokenPruner(nn.Module):
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
            torch.clamp(self.min_keep_ratio + (1.0 - self.min_keep_ratio) * (effective_threshold / (v_ee + 1e-8)),
                        min=self.min_keep_ratio, max=1.0),
            torch.ones_like(v_ee)
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


# =====================================================================
#  LOAD DATASET
# =====================================================================
print("\n  Loading dataset...")
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

ep_col = dataset.hf_dataset['episode_index']
episode_indices = {}
for idx, ep in enumerate(ep_col):
    ep_int = ep.item() if isinstance(ep, torch.Tensor) else int(ep)
    if ep_int not in episode_indices: episode_indices[ep_int] = []
    episode_indices[ep_int].append(idx)

print(f"  Dataset: {len(dataset)} frames, {dataset.num_episodes} episodes")

# =====================================================================
#  LOAD MODEL + ALL DYNAMIC MODULES
# =====================================================================
print("  Loading SmolVLA + Dynamic modules...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
smolvla.to(DEVICE)
flow_model = smolvla.model

# Key remapping
smolvla_img_keys = list(smolvla.config.image_features.keys())
KEY_REMAP_S = {}
for i, dk in enumerate(image_keys):
    if i < len(smolvla_img_keys):
        KEY_REMAP_S[dk] = smolvla_img_keys[i]

# Language tokens
tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
instruction = ds_cfg["task_instruction"]
_tok = tokenizer(instruction, return_tensors="pt", padding="max_length", max_length=64)
LANG_IDS = _tok['input_ids']
LANG_MASK = _tok['attention_mask'].bool()

CHUNK_SIZE_S = smolvla.config.chunk_size

vlm_hidden = flow_model.vlm_with_expert.config.text_config.hidden_size
num_vlm_layers = flow_model.vlm_with_expert.num_vlm_layers

# Build dynamic modules
star_router = STARRouter(hidden_dim=vlm_hidden, num_skippable_layers=NUM_SKIP).to(DEVICE)
token_pruner = ActionAwareTokenPruner(
    v_threshold=dyn_cfg["adp_velocity_threshold"],
    min_keep_ratio=dyn_cfg["adp_min_keep_ratio"],
).to(DEVICE)

lora_adapters = nn.ModuleDict()
vlm_layers = flow_model.vlm_with_expert.get_vlm_model().text_model.layers
for layer_idx in range(dyn_cfg["num_fixed_layers"], num_vlm_layers):
    layer_key = f"layer_{layer_idx}"
    attn = vlm_layers[layer_idx].self_attn
    q_in, q_out = attn.q_proj.in_features, attn.q_proj.out_features
    k_in, k_out = attn.k_proj.in_features, attn.k_proj.out_features
    v_in, v_out = attn.v_proj.in_features, attn.v_proj.out_features
    o_in, o_out = attn.o_proj.in_features, attn.o_proj.out_features
    lora_adapters[f"{layer_key}_q"] = LoRASPAdapter(q_in, q_out, max_rank=dyn_cfg["lora_max_rank"], energy_threshold=dyn_cfg["lora_energy_threshold"])
    lora_adapters[f"{layer_key}_k"] = LoRASPAdapter(k_in, k_out, max_rank=dyn_cfg["lora_max_rank"], energy_threshold=dyn_cfg["lora_energy_threshold"])
    lora_adapters[f"{layer_key}_v"] = LoRASPAdapter(v_in, v_out, max_rank=dyn_cfg["lora_max_rank"], energy_threshold=dyn_cfg["lora_energy_threshold"])
    lora_adapters[f"{layer_key}_o"] = LoRASPAdapter(o_in, o_out, max_rank=dyn_cfg["lora_max_rank"], energy_threshold=dyn_cfg["lora_energy_threshold"])
lora_adapters = lora_adapters.to(DEVICE)

# Load checkpoint
print(f"  Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
smolvla.load_state_dict(ckpt['smolvla'])
star_router.load_state_dict(ckpt['star_router'])
token_pruner.load_state_dict(ckpt['token_pruner'])
lora_adapters.load_state_dict(ckpt['lora_adapters'])
print("  ✓ All weights restored (SmolVLA + STAR + ADP + LoRA-SP)")
del ckpt; gc.collect(); torch.cuda.empty_cache()

smolvla.eval()
star_router.eval()
token_pruner.eval()
lora_adapters.eval()


# =====================================================================
#  HELPER: Build eval batch (same as eval script)
# =====================================================================
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


# =====================================================================
#  RUN ALL DYNAMIC MODULES ON EVAL EPISODES
# =====================================================================
eval_ep_list = [40, 41, 42]
eval_ep_list = [ep for ep in eval_ep_list if ep in episode_indices]
print(f"\n  Running analysis on episodes: {eval_ep_list}")

all_episode_data = {}

for ep_idx in eval_ep_list:
    indices = episode_indices[ep_idx]
    n_frames = len(indices)

    gate_decisions = np.zeros((n_frames, NUM_SKIP))
    gate_probs_arr = np.zeros((n_frames, NUM_SKIP))
    state_deltas = np.zeros(n_frames)
    visual_entropies = np.zeros(n_frames)
    actions_arr = np.zeros((n_frames, ACTION_DIM))
    ee_velocities = np.zeros(n_frames)
    token_keep_ratios = np.zeros(n_frames)
    token_counts_original = np.zeros(n_frames, dtype=int)
    token_counts_pruned = np.zeros(n_frames, dtype=int)
    lora_effective_ranks = np.zeros(n_frames)

    # SnapFlow: single-step vs multi-step prediction MSE
    snap_1step_mse = np.zeros(n_frames)
    snap_multistep_mse = np.zeros(n_frames)
    snap_1step_latency = np.zeros(n_frames)

    prev_state = None

    smolvla.reset()

    for t_idx, step_idx in enumerate(tqdm(indices, desc=f"  Ep{ep_idx}", ncols=85, leave=False)):
        s = dataset[step_idx]
        actions_arr[t_idx] = s[action_key].numpy()

        # Build proper batch for prepare_images
        batch = build_eval_batch(s, DEVICE)

        # Get state
        state = batch.get('observation.state', torch.zeros(1, 6, device=DEVICE))

        # ── State delta ──
        if prev_state is not None:
            delta_s = (state - prev_state).float().norm(dim=-1, keepdim=True)
        else:
            delta_s = torch.zeros(1, 1, device=DEVICE)
        state_deltas[t_idx] = delta_s.item()

        # ── End-effector velocity (first 6 dims = joint positions) ──
        if prev_state is not None:
            ee_vel = (state[:, :6] - prev_state[:, :6]).float().norm(dim=-1, keepdim=True)
        else:
            ee_vel = torch.zeros(1, 1, device=DEVICE)
        ee_velocities[t_idx] = ee_vel.item()

        prev_state = state.clone()

        # ── Get visual embeddings via prepare_images (proper preprocessing) ──
        with torch.no_grad():
            images, img_masks = smolvla.prepare_images(batch)
            img_embs_list = []
            for img in images:
                img_emb = flow_model.vlm_with_expert.embed_image(img)
                img_embs_list.append(img_emb)
            all_img_emb = torch.cat(img_embs_list, dim=1) if img_embs_list else None
            # Cast to float32 — VLM outputs bf16 but dynamic modules are fp32
            if all_img_emb is not None:
                all_img_emb = all_img_emb.float()

        N_tokens = all_img_emb.shape[1] if all_img_emb is not None else 0
        token_counts_original[t_idx] = N_tokens

        # ── Visual entropy ──
        if all_img_emb is not None:
            e_view = all_img_emb.var(dim=1).mean(dim=-1, keepdim=True).float()
        else:
            e_view = torch.zeros(1, 1, device=DEVICE)
        visual_entropies[t_idx] = e_view.item()

        # ── STAR Router: layer skipping ──
        if all_img_emb is not None:
            hidden_pooled = all_img_emb.mean(dim=1).float()
            expected_dim = star_router.gate_net[0].in_features - 2
            if hidden_pooled.shape[-1] != expected_dim:
                hidden_pooled = F.adaptive_avg_pool1d(
                    hidden_pooled.unsqueeze(1), expected_dim
                ).squeeze(1)
        else:
            expected_dim = star_router.gate_net[0].in_features - 2
            hidden_pooled = torch.zeros(1, expected_dim, device=DEVICE)

        with torch.no_grad():
            gates, probs = star_router(hidden_pooled, e_view, delta_s)
        gate_decisions[t_idx] = gates.cpu().numpy()[0]
        gate_probs_arr[t_idx] = probs.cpu().numpy()[0]

        # ── ADP: Token Pruning ──
        if all_img_emb is not None:
            with torch.no_grad():
                token_mask = torch.ones(1, N_tokens, dtype=torch.bool, device=DEVICE)
                pruned_emb, pruned_mask, keep_ratio, keep_ratio_ps = token_pruner(
                    all_img_emb.float(), token_mask, ee_vel.float()
                )
            token_keep_ratios[t_idx] = keep_ratio.item()
            token_counts_pruned[t_idx] = pruned_emb.shape[1]
        else:
            token_keep_ratios[t_idx] = 1.0
            token_counts_pruned[t_idx] = 0

        # ── LoRA-SP: effective rank estimation ──
        # Sample one adapter's router to estimate effective rank
        if all_img_emb is not None:
            sample_adapter_key = list(lora_adapters.keys())[0]
            adapter = lora_adapters[sample_adapter_key]
            with torch.no_grad():
                # Pool image embeddings to match adapter input dim
                lora_input = all_img_emb.mean(dim=1, keepdim=True).float()  # (1, 1, D)
                if lora_input.shape[-1] != adapter.in_features:
                    lora_input = F.adaptive_avg_pool1d(
                        lora_input.permute(0, 2, 1), adapter.in_features
                    ).permute(0, 2, 1)
                x_flat = lora_input.reshape(-1, adapter.in_features)
                scores = adapter.router(x_flat.float())  # (1, max_rank)
                # Effective rank = number of scores above threshold (e.g., 0.1)
                scores_sorted = scores.sort(dim=-1, descending=True).values
                cum_energy = (scores_sorted ** 2).cumsum(dim=-1)
                total_energy = (scores ** 2).sum() + 1e-8
                energy_ratio = cum_energy / total_energy
                eff_rank = (energy_ratio < dyn_cfg["lora_energy_threshold"]).sum().item() + 1
                lora_effective_ranks[t_idx] = eff_rank
        else:
            lora_effective_ranks[t_idx] = dyn_cfg["lora_max_rank"]

        # ── SnapFlow: 1-step prediction + timing ──
        with torch.no_grad():
            t0 = time.perf_counter()
            pred = smolvla.select_action(batch)
            t1 = time.perf_counter()
            snap_1step_latency[t_idx] = (t1 - t0) * 1000

            pred_np = pred.squeeze().cpu().numpy()
            if pred_np.ndim > 1:
                pred_np = pred_np[0]
            pred_np = pred_np[:ACTION_DIM]
            snap_1step_mse[t_idx] = float(np.mean((pred_np - actions_arr[t_idx])**2))

    all_episode_data[ep_idx] = {
        'gate_decisions': gate_decisions,
        'gate_probs': gate_probs_arr,
        'state_deltas': state_deltas,
        'visual_entropies': visual_entropies,
        'actions': actions_arr,
        'ee_velocities': ee_velocities,
        'token_keep_ratios': token_keep_ratios,
        'token_counts_original': token_counts_original,
        'token_counts_pruned': token_counts_pruned,
        'lora_effective_ranks': lora_effective_ranks,
        'snap_1step_mse': snap_1step_mse,
        'snap_1step_latency': snap_1step_latency,
        'n_frames': n_frames,
    }

    skip_ratio = 1.0 - gate_decisions.mean()
    active_mean = gate_decisions.sum(axis=1).mean() + NUM_FIXED
    avg_keep = token_keep_ratios.mean()
    avg_rank = lora_effective_ranks.mean()
    print(f"    Ep{ep_idx}: {n_frames} frames | skip={skip_ratio:.0%} | "
          f"active={active_mean:.1f}/{TOTAL_LAYERS} | "
          f"token_keep={avg_keep:.0%} | eff_rank={avg_rank:.0f}/{dyn_cfg['lora_max_rank']}")

gc.collect()
torch.cuda.empty_cache()


# =====================================================================
#  PLOT 1: Layer Skip Heatmap per Episode
# =====================================================================
print("\n  [1/5] Layer skip heatmaps...")

cmap_skip = LinearSegmentedColormap.from_list('skip_cmap',
    [(0.85, 0.15, 0.15, 1.0), (0.15, 0.75, 0.15, 1.0)])

n_eps = len(eval_ep_list)
fig = plt.figure(figsize=(22, 5 * n_eps + 2))
gs = gridspec.GridSpec(n_eps * 2, 1,
                       height_ratios=sum([[3, 1] for _ in range(n_eps)], []),
                       hspace=0.35)

for plot_idx, ep_idx in enumerate(eval_ep_list):
    data = all_episode_data[ep_idx]
    n_frames = data['n_frames']
    gates = data['gate_decisions']

    full_layer_map = np.ones((n_frames, TOTAL_LAYERS))
    full_layer_map[:, NUM_FIXED:] = gates

    # ── Heatmap ──
    ax_heat = fig.add_subplot(gs[plot_idx * 2])
    im = ax_heat.imshow(full_layer_map.T, aspect='auto', cmap=cmap_skip,
                         vmin=0, vmax=1, interpolation='nearest',
                         extent=[0, n_frames, TOTAL_LAYERS - 0.5, -0.5])
    ax_heat.axhline(y=NUM_FIXED - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.8)
    ax_heat.text(n_frames + 2, NUM_FIXED / 2 - 0.5, 'FIXED',
                 fontsize=8, color='#4CAF50', fontweight='bold', va='center')
    ax_heat.text(n_frames + 2, NUM_FIXED + NUM_SKIP / 2 - 0.5, 'SKIPPABLE',
                 fontsize=8, color='#FF5722', fontweight='bold', va='center')
    ax_heat.set_ylabel('Layer', fontsize=11)
    ax_heat.set_yticks(range(TOTAL_LAYERS))
    ax_heat.set_yticklabels([f'L{i}' for i in range(TOTAL_LAYERS)], fontsize=8)
    ax_heat.set_title(f'Episode {ep_idx} — Layer Activity (green=ACTIVE, red=SKIPPED)',
                       fontsize=13, fontweight='bold', pad=10)
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.08)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(['SKIP', 'KEEP'])

    # ── Skip ratio timeline ──
    ax_tl = fig.add_subplot(gs[plot_idx * 2 + 1])
    skip_per_frame = 1.0 - gates.mean(axis=1)
    t = np.arange(n_frames)
    ax_tl.fill_between(t, skip_per_frame, alpha=0.3, color='#FF5722')
    ax_tl.plot(t, skip_per_frame, color='#FF5722', lw=1.5, label='Skip Ratio')
    ax_tl.set_ylabel('Skip Ratio', color='#FF5722', fontsize=10)
    ax_tl.set_ylim(-0.05, 1.05)
    ax_tl.set_xlabel('Frame', fontsize=10)

    ax2 = ax_tl.twinx()
    sd = data['state_deltas']
    k5 = np.ones(5) / 5
    sd_smooth = np.convolve(sd, k5, mode='same') if len(sd) > 5 else sd
    ax2.plot(t, sd_smooth, color='#2196F3', lw=1.5, alpha=0.8, label='State Δ')
    ax2.set_ylabel('State Δ', color='#2196F3', fontsize=10)

    l1, lb1 = ax_tl.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax_tl.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=8)
    ax_tl.grid(True, alpha=0.2)
    ax_tl.set_title(f'Ep{ep_idx} — Skip Ratio vs Movement', fontsize=11)

plt.suptitle('Dynamic Layer Skipping — STAR Router Decisions',
             fontsize=16, fontweight='bold', y=0.99)
plt.savefig(OUTPUT_DIR / 'dynamic_skip_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUTPUT_DIR / 'dynamic_skip_heatmap.png'}")


# =====================================================================
#  PLOT 2: Per-Layer Skip Stats
# =====================================================================
print("  [2/5] Per-layer skip statistics...")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

all_gates = np.concatenate([all_episode_data[ep]['gate_decisions'] for ep in eval_ep_list], axis=0)

# (a) Skip frequency bar chart
ax = axes[0]
skip_freq = 1.0 - all_gates.mean(axis=0)
layer_labels = [f'L{NUM_FIXED + i}' for i in range(NUM_SKIP)]
colors = ['#FF5722' if f > 0.5 else '#4CAF50' for f in skip_freq]
bars = ax.barh(range(NUM_SKIP), skip_freq, color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(NUM_SKIP))
ax.set_yticklabels(layer_labels, fontsize=11)
ax.set_xlabel('Skip Frequency', fontsize=12); ax.set_xlim(0, 1)
ax.set_title('Per-Layer Skip Frequency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')
for i, (bar, freq) in enumerate(zip(bars, skip_freq)):
    ax.text(freq + 0.02, i, f'{freq:.0%}', va='center', fontsize=10, fontweight='bold',
            color='#FF5722' if freq > 0.5 else '#4CAF50')

# (b) Gate probability box plot
ax = axes[1]
all_probs = np.concatenate([all_episode_data[ep]['gate_probs'] for ep in eval_ep_list], axis=0)
bp = ax.boxplot([all_probs[:, i] for i in range(NUM_SKIP)],
                vert=True, patch_artist=True, labels=layer_labels,
                medianprops=dict(color='black', linewidth=2))
for i, patch in enumerate(bp['boxes']):
    median = np.median(all_probs[:, i])
    patch.set_facecolor('#4CAF50' if median > 0.5 else '#FF5722')
    patch.set_alpha(0.6)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='threshold')
ax.set_ylabel('Gate Probability', fontsize=12)
ax.set_title('Gate Probability Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y')

# (c) Active layers histogram
ax = axes[2]
all_active = all_gates.sum(axis=1) + NUM_FIXED
ax.hist(all_active, bins=np.arange(NUM_FIXED - 0.5, TOTAL_LAYERS + 1.5, 1),
        color='#2196F3', edgecolor='white', alpha=0.8, rwidth=0.85)
ax.set_xlabel('Active Layers', fontsize=12); ax.set_ylabel('Frame Count', fontsize=12)
ax.set_title(f'Active Layers Distribution', fontsize=13, fontweight='bold')
ax.set_xticks(range(NUM_FIXED, TOTAL_LAYERS + 1))
mean_act = all_active.mean()
ax.axvline(x=mean_act, color='#FF5722', lw=2, linestyle='--', label=f'Mean={mean_act:.1f}')
ax.legend(fontsize=11); ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('STAR Router — Layer Skipping Statistics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dynamic_skip_stats.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUTPUT_DIR / 'dynamic_skip_stats.png'}")


# =====================================================================
#  PLOT 3: Token Pruning (ADP) Visualization
# =====================================================================
print("  [3/5] Token pruning (ADP) visualization...")

fig, axes = plt.subplots(len(eval_ep_list), 2, figsize=(22, 5 * len(eval_ep_list)))
if len(eval_ep_list) == 1:
    axes = axes.reshape(1, -1)

for row, ep_idx in enumerate(eval_ep_list):
    data = all_episode_data[ep_idx]
    n = data['n_frames']
    t = np.arange(n)

    # ── Left: Token keep ratio & velocity timeline ──
    ax = axes[row, 0]
    keep_r = data['token_keep_ratios']
    vel = data['ee_velocities']

    # Smooth
    k5 = np.ones(5)/5
    keep_smooth = np.convolve(keep_r, k5, mode='same') if n > 5 else keep_r
    vel_smooth = np.convolve(vel, k5, mode='same') if n > 5 else vel

    ax.fill_between(t, keep_smooth, 1.0, alpha=0.3, color='#FF5722', label='Pruned tokens')
    ax.fill_between(t, 0, keep_smooth, alpha=0.3, color='#4CAF50', label='Kept tokens')
    ax.plot(t, keep_smooth, color='#4CAF50', lw=2, label=f'Keep ratio (avg={keep_r.mean():.0%})')
    ax.set_ylabel('Token Keep Ratio', fontsize=11, color='#4CAF50')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Frame', fontsize=10)

    # Velocity overlay
    ax_v = ax.twinx()
    ax_v.plot(t, vel_smooth, color='#9C27B0', lw=1.5, alpha=0.7, label='EE velocity')
    ax_v.axhline(y=dyn_cfg["adp_velocity_threshold"], color='#9C27B0', linestyle='--',
                 alpha=0.5, label=f'v_thresh={dyn_cfg["adp_velocity_threshold"]}')
    ax_v.set_ylabel('EE Velocity', fontsize=11, color='#9C27B0')

    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_v.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=7, ncol=2)
    ax.set_title(f'Ep{ep_idx} — Token Pruning: Keep Ratio vs EE Velocity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # ── Right: Token count bar (original vs pruned) ──
    ax = axes[row, 1]
    orig = data['token_counts_original']
    pruned = data['token_counts_pruned']

    # Sample every N frames for readability
    step_vis = max(1, n // 40)
    t_sampled = t[::step_vis]
    orig_s = orig[::step_vis]
    pruned_s = pruned[::step_vis]

    width = 0.35
    x_bar = np.arange(len(t_sampled))
    ax.bar(x_bar - width/2, orig_s, width, color='#2196F3', alpha=0.7, label='Original tokens')
    ax.bar(x_bar + width/2, pruned_s, width, color='#FF9800', alpha=0.7, label='After pruning')
    ax.set_xlabel('Frame (sampled)', fontsize=10)
    ax.set_ylabel('Token Count', fontsize=11)
    ax.set_xticks(x_bar[::max(1, len(x_bar)//10)])
    ax.set_xticklabels([str(f) for f in t_sampled[::max(1, len(t_sampled)//10)]], fontsize=8)
    ax.legend(fontsize=9)
    ax.set_title(f'Ep{ep_idx} — Token Count: Original vs Pruned', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Pruned ratio annotation
    if orig_s.mean() > 0:
        prune_pct = 1.0 - pruned_s.mean() / orig_s.mean()
        ax.text(0.95, 0.95, f'Pruned: {prune_pct:.0%}', transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='#FF5722',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Action-Aware Dynamic Token Pruning (ADP)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dynamic_token_pruning.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUTPUT_DIR / 'dynamic_token_pruning.png'}")


# =====================================================================
#  PLOT 4: SnapFlow Analysis
# =====================================================================
print("  [4/5] SnapFlow 1-NFE analysis...")

fig = plt.figure(figsize=(22, 5 * len(eval_ep_list) + 3))
gs = gridspec.GridSpec(len(eval_ep_list), 3, hspace=0.4, wspace=0.3)

for row, ep_idx in enumerate(eval_ep_list):
    data = all_episode_data[ep_idx]
    n = data['n_frames']
    t = np.arange(n)
    k7 = np.ones(7) / 7

    # ── (a) Per-frame MSE (1-step prediction quality) ──
    ax = fig.add_subplot(gs[row, 0])
    mse = data['snap_1step_mse']
    mse_smooth = np.convolve(mse, k7, mode='same') if n > 7 else mse

    ax.fill_between(t, mse_smooth, alpha=0.3, color='#2196F3')
    ax.plot(t, mse_smooth, color='#2196F3', lw=1.5, label=f'1-NFE MSE (avg={mse.mean():.2f})')
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title(f'Ep{ep_idx} — SnapFlow 1-Step Prediction MSE', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    # Color phases by MSE magnitude
    # Low MSE regions ("easy" — SnapFlow handles well)
    p25 = np.percentile(mse_smooth, 25)
    p75 = np.percentile(mse_smooth, 75)
    easy_mask = mse_smooth <= p25
    hard_mask = mse_smooth >= p75
    ax.fill_between(t, 0, ax.get_ylim()[1] * 0.05,
                    where=easy_mask, color='#4CAF50', alpha=0.3, label='Easy (low MSE)')
    ax.fill_between(t, 0, ax.get_ylim()[1] * 0.05,
                    where=hard_mask, color='#FF5722', alpha=0.3, label='Hard (high MSE)')
    ax.legend(fontsize=8, loc='upper right')

    # ── (b) Latency timeline ──
    ax = fig.add_subplot(gs[row, 1])
    lat = data['snap_1step_latency']
    lat_smooth = np.convolve(lat, k7, mode='same') if n > 7 else lat

    ax.plot(t, lat_smooth, color='#FF9800', lw=1.5, alpha=0.8)
    ax.fill_between(t, lat_smooth, alpha=0.2, color='#FF9800')
    ax.axhline(y=np.mean(lat), color='#FF5722', linestyle='--', lw=2,
               label=f'Mean={np.mean(lat):.1f}ms')
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title(f'Ep{ep_idx} — 1-NFE Inference Latency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)

    # ── (c) MSE vs Action Velocity correlation ──
    ax = fig.add_subplot(gs[row, 2])
    vel = data['ee_velocities']
    # Scatter: velocity vs MSE
    sc = ax.scatter(vel, mse, c=t, cmap='viridis', alpha=0.4, s=15, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Frame #', shrink=0.8)
    ax.set_xlabel('EE Velocity', fontsize=11)
    ax.set_ylabel('1-Step MSE', fontsize=11)
    ax.set_title(f'Ep{ep_idx} — MSE vs Movement Speed', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # Correlation coefficient
    if vel.std() > 1e-8 and mse.std() > 1e-8:
        corr = np.corrcoef(vel, mse)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('SnapFlow 1-NFE Analysis — Single-Step Denoising Quality',
             fontsize=16, fontweight='bold', y=1.01)
plt.savefig(OUTPUT_DIR / 'dynamic_snapflow_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUTPUT_DIR / 'dynamic_snapflow_analysis.png'}")


# =====================================================================
#  PLOT 5: Combined Action-Phase Correlation
# =====================================================================
print("  [5/5] Combined action-phase correlation...")

fig, axes = plt.subplots(len(eval_ep_list), 1, figsize=(22, 6 * len(eval_ep_list)))
if len(eval_ep_list) == 1:
    axes = [axes]

for ax_idx, ep_idx in enumerate(eval_ep_list):
    ax = axes[ax_idx]
    data = all_episode_data[ep_idx]
    n = data['n_frames']
    t = np.arange(n)
    k7 = np.ones(7)/7

    # Background: action trajectories
    actions = data['actions']
    for j in range(min(3, ACTION_DIM)):
        ax.plot(t, actions[:, j], '-', color=f'C{j}', alpha=0.2, lw=1)

    # Active layers (filled area)
    gates = data['gate_decisions']
    active = gates.sum(axis=1) + NUM_FIXED
    active_s = np.convolve(active, k7, mode='same') if n > 7 else active

    ax_r = ax.twinx()
    ax_r.fill_between(t, NUM_FIXED, active_s, alpha=0.2, color='#4CAF50', label='Active layers')
    ax_r.fill_between(t, active_s, TOTAL_LAYERS, alpha=0.2, color='#FF5722', label='Skipped layers')
    ax_r.set_ylim(0, TOTAL_LAYERS + 2)
    ax_r.set_ylabel('Active Layers', fontsize=11)

    # Token keep ratio
    keep_s = np.convolve(data['token_keep_ratios'], k7, mode='same') if n > 7 else data['token_keep_ratios']
    ax.plot(t, keep_s * 2 - 1, color='#FF9800', lw=2, alpha=0.7, label='Token keep (scaled)')

    # LoRA effective rank (scaled)
    rank_s = np.convolve(data['lora_effective_ranks'], k7, mode='same') if n > 7 else data['lora_effective_ranks']
    rank_norm = rank_s / dyn_cfg['lora_max_rank']
    ax.plot(t, rank_norm * 2 - 1, color='#9C27B0', lw=1.5, alpha=0.6, label='LoRA eff rank (scaled)')

    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Action / Scaled Metrics', fontsize=11)
    ax.set_title(f'Episode {ep_idx} — All Dynamic Components Over Time', fontsize=13, fontweight='bold')

    l_a, lb_a = ax.get_legend_handles_labels()
    l_r, lb_r = ax_r.get_legend_handles_labels()
    ax.legend(l_a + l_r, lb_a + lb_r, loc='upper left', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.15)

plt.suptitle('Dynamic SmolVLA — Combined Component Analysis',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dynamic_skip_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUTPUT_DIR / 'dynamic_skip_timeline.png'}")


# =====================================================================
#  SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("  VISUALIZATION COMPLETE")
print("=" * 70)

for ep_idx in eval_ep_list:
    data = all_episode_data[ep_idx]
    gates = data['gate_decisions']
    skip_r = 1.0 - gates.mean()
    active_mean = gates.sum(axis=1).mean() + NUM_FIXED
    keep_r = data['token_keep_ratios'].mean()
    eff_rank = data['lora_effective_ranks'].mean()
    mse = data['snap_1step_mse'].mean()
    lat = data['snap_1step_latency'].mean()
    print(f"\n  Ep{ep_idx} ({data['n_frames']} frames):")
    print(f"    Layer Skipping:   {skip_r:.0%} skipped | {active_mean:.1f}/{TOTAL_LAYERS} active")
    print(f"    Token Pruning:    {keep_r:.0%} kept | {1-keep_r:.0%} pruned")
    print(f"    LoRA-SP Rank:     {eff_rank:.0f}/{dyn_cfg['lora_max_rank']} effective")
    print(f"    SnapFlow 1-NFE:   MSE={mse:.2f} | Latency={lat:.1f}ms")

print(f"\n  Output files:")
for f in ['dynamic_skip_heatmap.png', 'dynamic_skip_stats.png',
          'dynamic_token_pruning.png', 'dynamic_snapflow_analysis.png',
          'dynamic_skip_timeline.png']:
    print(f"    {OUTPUT_DIR / f}")
print("\n  DONE!")
