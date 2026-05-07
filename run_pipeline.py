"""
SmolVLA (Real Pretrained 450M) vs ACT Baseline: Full Comparison
================================================================
RTX 4070 SUPER (12GB) | PyTorch 2.6.0+cu124 | lerobot 0.4.4
"""
import os, sys, time, json, types
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

# 1) Stub lerobot.robots (skip Robot class that triggers processor->transformers crash)
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

# 2) Stub lerobot.processor (avoid tokenizer_processor -> transformers crash)
_proc_mod = types.ModuleType('lerobot.processor')
_proc_mod.__path__ = [os.path.join(_pkg, 'processor')]
_proc_mod.__package__ = 'lerobot.processor'
_proc_mod.RobotAction = dict
_proc_mod.RobotObservation = dict
_proc_mod.PolicyAction = dict
sys.modules['lerobot.processor'] = _proc_mod

# 3) Stub lerobot.policies.__init__ (skip GR00T dataclass crash)
_policies_mod = types.ModuleType('lerobot.policies')
_policies_mod.__path__ = [os.path.join(_pkg, 'policies')]
_policies_mod.__package__ = 'lerobot.policies'
sys.modules['lerobot.policies'] = _policies_mod

# =====================================================================
#  CONFIG
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("d:/EyetechCode/results")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_REPO = "lerobot/svla_so100_pickplace"
NUM_EVAL_EPISODES = 5
ACT_TRAIN_STEPS = 5000

print("=" * 65)
print("  SmolVLA (450M Pretrained) vs ACT Baseline")
print("=" * 65)
print(f"  Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  VRAM: {mem:.1f} GB")
print(f"  Dataset: {DATASET_REPO}")
print(f"  Output: {OUTPUT_DIR}")
print()

# =====================================================================
#  PHASE 1: Load Dataset
# =====================================================================
print("=" * 65)
print("  PHASE 1: Load Dataset")
print("=" * 65)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

print(f"  Loading {DATASET_REPO}...")
dataset = LeRobotDataset(DATASET_REPO)
print(f"  Episodes: {dataset.num_episodes}")
print(f"  Frames: {len(dataset)}")
print(f"  FPS: {dataset.fps}")

sample = dataset[0]
all_keys = list(sample.keys())
print(f"  Keys: {all_keys}")

# Identify keys
image_keys = [k for k in all_keys if 'image' in k.lower()]
state_key = next((k for k in all_keys if 'state' in k.lower()), None)
action_key = 'action'
meta_keys = {'episode_index', 'frame_index', 'timestamp', 'index', 'task_index'}

# Identify which keys are tensors vs strings
tensor_keys = [k for k in all_keys if isinstance(sample[k], torch.Tensor)]
string_keys = [k for k in all_keys if isinstance(sample[k], str)]
print(f"  Tensor keys: {tensor_keys}")
print(f"  String keys: {string_keys}")

print(f"  Images: {image_keys}")
print(f"  State: {state_key} -> {sample[state_key].shape if state_key else 'N/A'}")
print(f"  Action: {sample[action_key].shape}")

ACTION_DIM = sample[action_key].shape[-1]
STATE_DIM = sample[state_key].shape[-1] if state_key else 0
if image_keys:
    IMG_C, IMG_H, IMG_W = sample[image_keys[0]].shape

# Build episode index FAST using hf_dataset column (instant, no iteration)
print("  Building episode index from hf_dataset column...")
ep_col = dataset.hf_dataset['episode_index']
episode_indices = {}
for idx, ep in enumerate(ep_col):
    ep_int = ep.item() if isinstance(ep, torch.Tensor) else int(ep)
    if ep_int not in episode_indices: episode_indices[ep_int] = []
    episode_indices[ep_int].append(idx)
print(f"  Done! ({len(episode_indices)} episodes indexed instantly)")

all_eps = sorted(episode_indices.keys())
eval_eps = all_eps[:NUM_EVAL_EPISODES]
train_eps = all_eps[NUM_EVAL_EPISODES:] if len(all_eps) > NUM_EVAL_EPISODES else all_eps

print(f"  Total eps: {len(all_eps)}, Eval: {eval_eps}, Train: {len(train_eps)} eps")
for ep in eval_eps:
    print(f"    Ep {ep}: {len(episode_indices[ep])} frames")

train_idx_list = []
for ep in train_eps:
    train_idx_list.extend(episode_indices[ep])
print(f"  Train frames: {len(train_idx_list)}")

# =====================================================================
#  PHASE 2: Load SmolVLA Pretrained (450M)
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 2: Load SmolVLA Pretrained (450M)")
print("=" * 65)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("  Loading lerobot/smolvla_base...")
smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
smolvla_params = sum(p.numel() for p in smolvla.parameters())
print(f"  SmolVLA loaded! Params: {smolvla_params / 1e6:.1f}M")
print(f"  Device: {smolvla.config.device}")

# Check what features SmolVLA expects
print(f"  Expected features: {list(smolvla.config.image_features.keys()) if hasattr(smolvla.config, 'image_features') else 'check config'}")
print(f"  Action dim: {smolvla.config.action_feature.shape if hasattr(smolvla.config, 'action_feature') else 'check config'}")

# =====================================================================
#  PHASE 2b: SmolVLA Inference
# =====================================================================
print("\n  Running SmolVLA inference...")

smolvla_results = {
    'mse_per_episode': [], 'latency_ms': [],
    'predictions': {}, 'ground_truth': {},
}

# Check if dataset keys match what SmolVLA expects
# SmolVLA expects: observation.images.camera1/2/3, observation.state, language tokens
# Dataset has:     observation.images.top, observation.images.wrist
smolvla_img_keys = list(smolvla.config.image_features.keys())
dataset_img_keys = image_keys
print(f"  SmolVLA expects images: {smolvla_img_keys}")
print(f"  Dataset provides:       {dataset_img_keys}")

# Build key remap: dataset keys -> SmolVLA expected keys
KEY_REMAP = {}
for i, dk in enumerate(dataset_img_keys):
    if i < len(smolvla_img_keys):
        KEY_REMAP[dk] = smolvla_img_keys[i]
print(f"  Key remap: {KEY_REMAP}")
print(f"  (camera3 will be auto-padded by SmolVLA's empty_cameras)")

def build_smolvla_batch(sample_dict, device):
    """Build a batch dict for SmolVLA from a dataset sample, with key remapping."""
    batch = {}
    for k in sample_dict:
        if k in meta_keys or k in string_keys:
            continue
        v = sample_dict[k]
        if not isinstance(v, torch.Tensor):
            continue
        # Remap key names to match SmolVLA config
        out_key = KEY_REMAP.get(k, k)
        batch[out_key] = v.unsqueeze(0).to(device)
    # Add language tokens (SmolVLA expects observation.language.tokens with dots)
    batch['observation.language.tokens'] = lang_tokens_ids.to(device)
    batch['observation.language.attention_mask'] = lang_tokens_mask.bool().to(device)
    return batch

# Prepare language tokens once (SmolVLA needs them, dataset doesn't have them)
print("  Preparing language tokens...")
tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
default_instruction = "Pick up the cube and place it in the bin."
_tok = tokenizer(default_instruction, return_tensors="pt", padding="max_length", max_length=64)
lang_tokens_ids = _tok['input_ids']
lang_tokens_mask = _tok['attention_mask']
print(f"  Language instruction: '{default_instruction}'")

# Try inference with first sample
print("\n  Testing SmolVLA on first sample...")
smolvla.reset()

try:
    test_sample = dataset[0]
    batch = build_smolvla_batch(test_sample, smolvla.config.device)
    print(f"  Batch keys: {list(batch.keys())}")
    
    with torch.no_grad():
        t0 = time.perf_counter()
        pred_action = smolvla.select_action(batch)
        t1 = time.perf_counter()
    
    print(f"  Test inference OK! Action shape: {pred_action.shape}, Time: {(t1-t0)*1000:.1f}ms")
    SMOLVLA_OK = True
    
except Exception as e:
    print(f"  SmolVLA inference failed: {e}")
    import traceback
    traceback.print_exc()
    SMOLVLA_OK = False

if SMOLVLA_OK:
    print(f"\n  Evaluating SmolVLA on {len(eval_eps)} episodes...")
    
    for ep_idx in eval_eps:
        ep_preds, ep_gts = [], []
        indices = episode_indices[ep_idx]
        smolvla.reset()
        
        for i, step_idx in enumerate(tqdm(indices, desc=f"  Ep{ep_idx}", ncols=80, leave=False)):
            s = dataset[step_idx]
            gt = s[action_key].numpy()
            
            batch = build_smolvla_batch(s, smolvla.config.device)
            
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = smolvla.select_action(batch)
            t1 = time.perf_counter()
            
            pred_np = pred.squeeze().cpu().numpy()
            if pred_np.ndim > 1:
                pred_np = pred_np[0]  # Take first action if chunk
            pred_np = pred_np[:ACTION_DIM]  # Match dims
            
            smolvla_results['latency_ms'].append((t1 - t0) * 1000)
            ep_preds.append(pred_np)
            ep_gts.append(gt)
        
        ep_preds = np.array(ep_preds)
        ep_gts = np.array(ep_gts)
        mse = float(np.mean((ep_preds - ep_gts) ** 2))
        smolvla_results['mse_per_episode'].append(mse)
        smolvla_results['predictions'][ep_idx] = ep_preds
        smolvla_results['ground_truth'][ep_idx] = ep_gts
        print(f"    Ep {ep_idx}: MSE={mse:.6f} ({len(indices)} frames)")
    
    all_p = np.concatenate(list(smolvla_results['predictions'].values()))
    all_g = np.concatenate(list(smolvla_results['ground_truth'].values()))
    smolvla_results['mse_total'] = float(np.mean((all_p - all_g) ** 2))
    smolvla_results['mse_per_joint'] = np.mean((all_p - all_g) ** 2, axis=0).tolist()
    smolvla_results['latency_mean_ms'] = float(np.mean(smolvla_results['latency_ms']))
    
    print(f"\n  SmolVLA (450M Pretrained) Results:")
    print(f"    Total MSE: {smolvla_results['mse_total']:.6f}")
    print(f"    Avg Latency: {smolvla_results['latency_mean_ms']:.2f} ms")
else:
    print("  [SKIP] SmolVLA inference not available")
    sys.exit(1)

# =====================================================================
#  PHASE 3: ACT Baseline (Pretrained from HuggingFace)
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 3: ACT Baseline (Pretrained)")
print("=" * 65)

from lerobot.policies.act.modeling_act import ACTPolicy

# Try loading community ACT model trained on SO-100
ACT_FALLBACK_IDS = [
    "cadene/act_so100_5_lego_test_080000",
    "cadene/act_so100_5_lego_test_020000",
    "masato-ka/act_so100_pickandplace_block_circle",
    "satvikahuja/act_so100_test",
    "pepijn223/act_so100",
]

act_loaded = False
for model_id in ACT_FALLBACK_IDS:
    try:
        print(f"  Trying to load ACT from '{model_id}'...")
        act_policy = ACTPolicy.from_pretrained(model_id)
        act_params = sum(p.numel() for p in act_policy.parameters())
        print(f"  ACT loaded! Params: {act_params / 1e6:.1f}M")
        print(f"  Device: {act_policy.config.device}")
        act_loaded = True
        break
    except Exception as e:
        print(f"  Failed: {e}")

if not act_loaded:
    print("\n  No pretrained ACT found. Training lightweight ACT from scratch (1000 steps)...")
    
    class ACTBaseline(nn.Module):
        """Lightweight ACT-style baseline for comparison."""
        def __init__(self, state_dim, action_dim, img_channels=3,
                     hidden=256, heads=4, layers=3, chunk=10):
            super().__init__()
            self.chunk = chunk
            self.vision = nn.Sequential(
                nn.Conv2d(img_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(256*16, hidden))
            self.state_enc = nn.Linear(state_dim, hidden) if state_dim > 0 else None
            ci = hidden + (hidden if state_dim > 0 else 0)
            self.combine = nn.Linear(ci, hidden)
            dl = nn.TransformerDecoderLayer(hidden, heads, hidden*4, batch_first=True, dropout=0.1)
            self.decoder = nn.TransformerDecoder(dl, layers)
            self.queries = nn.Parameter(torch.randn(1, chunk, hidden) * 0.02)
            self.head = nn.Linear(hidden, action_dim)

        def forward(self, images=None, states=None):
            fs = []
            B = 1
            if images is not None:
                if images.dim() == 3: images = images.unsqueeze(0)
                B = images.shape[0]
                images = F.interpolate(images, (64, 64), mode='bilinear')
                fs.append(self.vision(images))
            if states is not None and self.state_enc:
                if states.dim() == 1: states = states.unsqueeze(0)
                B = states.shape[0]
                fs.append(self.state_enc(states))
            c = self.combine(torch.cat(fs, -1)).unsqueeze(1)
            q = self.queries.expand(B, -1, -1)
            d = self.decoder(q, c)
            return self.head(d)[:, 0, :]

    act_model = ACTBaseline(STATE_DIM, ACTION_DIM, IMG_C, hidden=256, chunk=10).to(DEVICE)
    act_params = sum(p.numel() for p in act_model.parameters())
    print(f"  ACT params: {act_params / 1e6:.2f}M")
    
    QUICK_STEPS = 1000
    print(f"  Quick training ({QUICK_STEPS} steps)...")
    act_model.train()
    opt = torch.optim.AdamW(act_model.parameters(), lr=1e-4, weight_decay=0.01)
    for step in tqdm(range(QUICK_STEPS), desc="  ACT Train", ncols=80):
        bi = np.random.choice(train_idx_list, min(16, len(train_idx_list)), replace=True)
        ba, bim, bs = [], [], []
        for idx in bi:
            s = dataset[idx]
            ba.append(s[action_key])
            if image_keys: bim.append(s[image_keys[0]])
            if state_key: bs.append(s[state_key])
        gt = torch.stack(ba).to(DEVICE)
        imgs = torch.stack(bim).to(DEVICE) if bim else None
        sts = torch.stack(bs).to(DEVICE) if bs else None
        pred = act_model(imgs, sts)
        loss = F.l1_loss(pred, gt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(act_model.parameters(), 1.0)
        opt.step()
    act_model.eval()
    print("  ACT quick-training done.")
    USE_LEROBOT_ACT = False
else:
    USE_LEROBOT_ACT = True
    # Check ACT expected features
    act_img_keys_expected = list(act_policy.config.image_features.keys()) if hasattr(act_policy.config, 'image_features') else []
    print(f"  ACT expects images: {act_img_keys_expected}")
    
    ACT_KEY_REMAP = {}
    for i, dk in enumerate(dataset_img_keys):
        if i < len(act_img_keys_expected):
            ACT_KEY_REMAP[dk] = act_img_keys_expected[i]
    print(f"  ACT Key remap: {ACT_KEY_REMAP}")

def build_act_batch(sample_dict, device):
    """Build a batch dict for ACT from a dataset sample."""
    batch = {}
    remap = ACT_KEY_REMAP if USE_LEROBOT_ACT else {}
    for k in sample_dict:
        if k in meta_keys or k in string_keys:
            continue
        v = sample_dict[k]
        if not isinstance(v, torch.Tensor):
            continue
        out_key = remap.get(k, k)
        batch[out_key] = v.unsqueeze(0).to(device)
    return batch

# ACT Eval
print(f"\n  Evaluating ACT on {len(eval_eps)} episodes...")
act_results = {
    'mse_per_episode': [], 'latency_ms': [],
    'predictions': {}, 'ground_truth': {},
}

for ep_idx in eval_eps:
    ep_preds, ep_gts = [], []
    indices = episode_indices[ep_idx]
    
    if USE_LEROBOT_ACT:
        act_policy.reset()
    
    for step_idx in tqdm(indices, desc=f"  Ep{ep_idx}", ncols=80, leave=False):
        s = dataset[step_idx]
        gt = s[action_key].numpy()
        t0 = time.perf_counter()
        with torch.no_grad():
            if USE_LEROBOT_ACT:
                batch = build_act_batch(s, act_policy.config.device)
                pred = act_policy.select_action(batch)
                pred_np = pred.squeeze().cpu().numpy()
            else:
                imgs = s[image_keys[0]].to(DEVICE).unsqueeze(0) if image_keys else None
                sts = s[state_key].to(DEVICE).unsqueeze(0) if state_key else None
                pred_np = act_model(imgs, sts).squeeze().cpu().numpy()
        t1 = time.perf_counter()
        
        if pred_np.ndim > 1:
            pred_np = pred_np[0]
        pred_np = pred_np[:ACTION_DIM]
        
        act_results['latency_ms'].append((t1-t0)*1000)
        ep_preds.append(pred_np)
        ep_gts.append(gt)
    
    ep_preds = np.array(ep_preds)
    ep_gts = np.array(ep_gts)
    mse = float(np.mean((ep_preds - ep_gts)**2))
    act_results['mse_per_episode'].append(mse)
    act_results['predictions'][ep_idx] = ep_preds
    act_results['ground_truth'][ep_idx] = ep_gts
    print(f"    Ep {ep_idx}: MSE={mse:.6f} ({len(indices)} frames)")

all_pa = np.concatenate(list(act_results['predictions'].values()))
all_ga = np.concatenate(list(act_results['ground_truth'].values()))
act_results['mse_total'] = float(np.mean((all_pa - all_ga)**2))
act_results['mse_per_joint'] = np.mean((all_pa - all_ga)**2, axis=0).tolist()
act_results['latency_mean_ms'] = float(np.mean(act_results['latency_ms']))

print(f"\n  ACT Results:")
print(f"    Total MSE: {act_results['mse_total']:.6f}")
print(f"    Avg Latency: {act_results['latency_mean_ms']:.2f} ms")

# =====================================================================
#  PHASE 4: Comparison
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 4: Comparison Report")
print("=" * 65)

print(f"\n  {'='*65}")
print(f"  {'METRIC':<25} {'SmolVLA (450M)':<22} {'ACT (L1)':<18}")
print(f"  {'='*65}")
print(f"  {'Parameters (M)':<25} {smolvla_params/1e6:<22.1f} {act_params/1e6:<18.2f}")
print(f"  {'Total MSE':<25} {smolvla_results['mse_total']:<22.6f} {act_results['mse_total']:<18.6f}")
print(f"  {'Avg Latency (ms)':<25} {smolvla_results['latency_mean_ms']:<22.2f} {act_results['latency_mean_ms']:<18.2f}")
for j in range(ACTION_DIM):
    sj = smolvla_results['mse_per_joint'][j]
    aj = act_results['mse_per_joint'][j]
    w = " <-- SmolVLA wins" if sj < aj else ""
    print(f"  {'Joint '+str(j)+' MSE':<25} {sj:<22.6f} {aj:<18.6f}{w}")
print(f"  {'='*65}")

sw = sum(1 for s, a in zip(smolvla_results['mse_per_joint'], act_results['mse_per_joint']) if s < a)
print(f"\n  SmolVLA wins: {sw}/{ACTION_DIM} joints")
print(f"  Overall MSE: {'SmolVLA' if smolvla_results['mse_total'] < act_results['mse_total'] else 'ACT'} wins")
print(f"  Per-episode:")
for i, ep in enumerate(eval_eps):
    sm = smolvla_results['mse_per_episode'][i]
    am = act_results['mse_per_episode'][i]
    print(f"    Ep {ep}: SmolVLA={sm:.6f} ACT={am:.6f} -> {'SmolVLA' if sm < am else 'ACT'}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
blue, orange = '#2196F3', '#FF5722'
w = 0.35

ax = axes[0,0]
x = np.arange(len(eval_eps))
ax.bar(x-w/2, smolvla_results['mse_per_episode'], w, label='SmolVLA (450M)', color=blue)
ax.bar(x+w/2, act_results['mse_per_episode'], w, label='ACT', color=orange)
ax.set_xlabel('Episode'); ax.set_ylabel('MSE'); ax.set_title('MSE per Episode')
ax.set_xticks(x); ax.set_xticklabels([f'Ep{e}' for e in eval_eps])
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0,1]
x = np.arange(ACTION_DIM)
ax.bar(x-w/2, smolvla_results['mse_per_joint'], w, label='SmolVLA (450M)', color=blue)
ax.bar(x+w/2, act_results['mse_per_joint'], w, label='ACT', color=orange)
ax.set_xlabel('Joint'); ax.set_ylabel('MSE'); ax.set_title('MSE per Joint')
ax.set_xticks(x); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,0]
ax.hist(smolvla_results['latency_ms'], bins=40, alpha=0.7, label='SmolVLA', color=blue)
ax.hist(act_results['latency_ms'], bins=40, alpha=0.7, label='ACT', color=orange)
ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Count'); ax.set_title('Inference Latency')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ep0 = eval_eps[0]
gt0 = smolvla_results['ground_truth'][ep0]
sp0 = smolvla_results['predictions'][ep0]
ap0 = act_results['predictions'][ep0]
nj = min(3, ACTION_DIM)
t_ax = np.arange(len(gt0))
for j in range(nj):
    ax.plot(t_ax, gt0[:,j], '-', color=f'C{j}', lw=2, label=f'GT J{j}')
    ax.plot(t_ax, sp0[:,j], '--', color=f'C{j}', alpha=0.6, label=f'SmolVLA J{j}')
    ax.plot(t_ax, ap0[:,j], ':', color=f'C{j}', alpha=0.6, label=f'ACT J{j}')
ax.set_xlabel('Step'); ax.set_ylabel('Action'); ax.set_title(f'Trajectory Ep{ep0}')
ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)

plt.suptitle('SmolVLA (450M Pretrained) vs ACT Baseline', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'comparison_plots.png', dpi=150)
plt.close()
print(f"\n  Plots: {OUTPUT_DIR / 'comparison_plots.png'}")

# =====================================================================
#  PHASE 5: Videos
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 5: Videos")
print("=" * 65)

for ep_idx in eval_eps[:3]:
    gt = smolvla_results['ground_truth'][ep_idx]
    sp = smolvla_results['predictions'][ep_idx]
    ap = act_results['predictions'][ep_idx]
    indices = episode_indices[ep_idx][:len(gt)]
    n = min(len(gt), len(sp), len(ap))

    vpath = OUTPUT_DIR / f'comparison_ep{ep_idx}.mp4'
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

        cv2.putText(frame, f'Ep {ep_idx} | Frame {t}/{n}', (340, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        nj = min(ACTION_DIM, 6)
        px0 = 340; pw = fw - px0 - 15
        jh = max(45, (fh - 80) // nj - 6)

        for j in range(nj):
            y0 = 35 + j * (jh + 4)
            cv2.putText(frame, f'J{j}', (px0, y0+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180,180,180), 1)
            cv2.rectangle(frame, (px0+22, y0), (px0+pw, y0+jh), (40,40,40), -1)

            win = 40; st = max(0, t-win)
            vmin = min(gt[st:t+1,j].min(), sp[st:t+1,j].min(), ap[st:t+1,j].min()) - 0.05
            vmax = max(gt[st:t+1,j].max(), sp[st:t+1,j].max(), ap[st:t+1,j].max()) + 0.05
            if vmax-vmin < 0.01: vmax = vmin + 0.01

            def _px(tt): return px0 + 22 + int((tt-st)/max(win,1) * (pw-25))
            def _py(v): return y0 + jh - int((v-vmin)/(vmax-vmin) * jh)

            for tt in range(st, min(t, n-1)):
                x1, x2 = _px(tt), _px(tt+1)
                cv2.line(frame, (x1,_py(gt[tt,j])), (x2,_py(gt[tt+1,j])), (0,220,0), 2)    # GT green
                cv2.line(frame, (x1,_py(sp[tt,j])), (x2,_py(sp[tt+1,j])), (255,160,0), 1)   # SmolVLA blue
                cv2.line(frame, (x1,_py(ap[tt,j])), (x2,_py(ap[tt+1,j])), (50,80,255), 1)   # ACT red

        yb = fh - 25
        cv2.putText(frame, 'GT', (15, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(frame, 'SmolVLA', (55, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,160,0), 1)
        cv2.putText(frame, 'ACT', (180, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,80,255), 1)

        ms = float(np.mean((sp[t]-gt[t])**2))
        ma = float(np.mean((ap[t]-gt[t])**2))
        cv2.putText(frame, f'SmolVLA MSE:{ms:.4f}', (350, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,160,0), 1)
        cv2.putText(frame, f'ACT MSE:{ma:.4f}', (600, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,80,255), 1)

        writer.write(frame)
    writer.release()
    print(f"  Video: {vpath}")

# =====================================================================
#  Save JSON Report
# =====================================================================
report = {
    'dataset': DATASET_REPO,
    'eval_episodes': eval_eps,
    'smolvla': {
        'label': 'SmolVLA (450M Pretrained)',
        'params_M': round(smolvla_params/1e6, 1),
        'model_loaded_from_hub': True,
        'total_mse': smolvla_results['mse_total'],
        'per_episode_mse': smolvla_results['mse_per_episode'],
        'per_joint_mse': smolvla_results['mse_per_joint'],
        'avg_latency_ms': smolvla_results['latency_mean_ms'],
        'architecture': 'SigLIP + SmolLM2(16/32 layers) + Flow Matching Expert + Interleaved CA/SA',
    },
    'act': {
        'label': 'ACT (L1 Regression)',
        'params_M': round(act_params/1e6, 2),
        'total_mse': act_results['mse_total'],
        'per_episode_mse': act_results['mse_per_episode'],
        'per_joint_mse': act_results['mse_per_joint'],
        'avg_latency_ms': act_results['latency_mean_ms'],
        'architecture': 'ResNet CNN + Transformer CVAE + L1 Regression (no language)',
    },
}
with open(OUTPUT_DIR / 'comparison_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n  Report: {OUTPUT_DIR / 'comparison_report.json'}")

# =====================================================================
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE")
print("=" * 65)
print(f"\n  SmolVLA (450M): MSE={smolvla_results['mse_total']:.6f} Latency={smolvla_results['latency_mean_ms']:.1f}ms")
print(f"  ACT Baseline:   MSE={act_results['mse_total']:.6f} Latency={act_results['latency_mean_ms']:.1f}ms")
mse_winner = 'SmolVLA' if smolvla_results['mse_total'] < act_results['mse_total'] else 'ACT'
lat_winner = 'ACT' if act_results['latency_mean_ms'] < smolvla_results['latency_mean_ms'] else 'SmolVLA'
print(f"\n  MSE Winner: {mse_winner}")
print(f"  Latency Winner: {lat_winner}")
print(f"\n  Output: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f.name:<35} {f.stat().st_size/1024:.0f} KB")
print("\n  DONE!")
