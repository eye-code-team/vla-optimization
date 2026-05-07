"""
Fine-Tune SmolVLA (450M) + ACT Baseline on Same Dataset
========================================================
RTX 4070 SUPER (12GB) | fp16 | lerobot 0.4.4
SmolVLA: freeze VLM, train action expert only (~50M trainable)
ACT: full fine-tune (51.6M)
"""
import os, sys, time, json, types, gc
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
smolvla_cfg = TRAINING["smolvla"]
act_cfg = TRAINING["act"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(f"d:/EyetechCode/results/{DATASET_KEY}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  Fine-Tune SmolVLA + ACT on Same Dataset")
print("=" * 65)
print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print(f"  Dataset: {ds_cfg['repo_id']}")
print(f"  Output: {OUTPUT_DIR}")
print()

# =====================================================================
#  PHASE 1: Load Dataset
# =====================================================================
print("=" * 65)
print("  PHASE 1: Load Dataset")
print("=" * 65)

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
CHUNK_SIZE = 100  # default action chunk

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
#  PHASE 2: Fine-Tune SmolVLA
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 2: Fine-Tune SmolVLA (450M, freeze VLM)")
print("=" * 65)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("  Loading pretrained SmolVLA base...")
smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# Key remapping: dataset -> SmolVLA expected keys
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

# Freeze VLM backbone, only train action expert
if smolvla_cfg["freeze_vlm"]:
    frozen, trainable = 0, 0
    for name, param in smolvla.named_parameters():
        # model.vlm_with_expert consists of .vlm (to be frozen) and .lm_expert (action expert, to be trained)
        # Other projection layers (action_in_proj, etc.) should also be trained
        if "vlm_with_expert.vlm" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    print(f"  Frozen (VLM): {frozen/1e6:.1f}M")
    print(f"  Trainable (action expert + projs): {trainable/1e6:.1f}M")
else:
    trainable = sum(p.numel() for p in smolvla.parameters())
    print(f"  Full fine-tune: {trainable/1e6:.1f}M trainable")

smolvla.to(DEVICE)
smolvla.train()

def build_smolvla_train_batch(indices):
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
    
    # Actions need chunk_size padding and padding to max_action_dim
    actions = torch.stack(batch_actions).to(DEVICE)  # (B, action_dim)
    B = actions.shape[0]
    
    # Pad from ACTION_DIM to max_action_dim
    MAX_ACT_DIM = smolvla.config.max_action_dim
    if ACTION_DIM < MAX_ACT_DIM:
        pad_zeros = torch.zeros(B, MAX_ACT_DIM - ACTION_DIM, device=DEVICE)
        actions_padded = torch.cat([actions, pad_zeros], dim=1)
    else:
        actions_padded = actions
        
    # Expand single-step action to chunk format (B, chunk_size, max_action_dim)
    action_chunk = actions_padded.unsqueeze(1).expand(B, CHUNK_SIZE_S, MAX_ACT_DIM)
    batch['action'] = action_chunk
    
    # Needs boolean padding mask for actions format in loss
    batch['actions_id_pad'] = torch.zeros(B, CHUNK_SIZE_S, dtype=torch.bool, device=DEVICE)
    
    batch['observation.language.tokens'] = LANG_IDS.expand(B, -1).to(DEVICE)
    batch['observation.language.attention_mask'] = LANG_MASK.expand(B, -1).to(DEVICE)
    
    return batch

# Training loop
opt_s = torch.optim.AdamW(
    [p for p in smolvla.parameters() if p.requires_grad],
    lr=smolvla_cfg["lr"], weight_decay=smolvla_cfg["weight_decay"]
)
# No GradScaler needed for bfloat16
losses_s = []
micro_bs = smolvla_cfg["micro_batch"]
grad_accum = smolvla_cfg["grad_accum"]

print(f"\n  Training: {smolvla_cfg['steps']} steps, micro_batch={micro_bs}, "
      f"grad_accum={grad_accum}, lr={smolvla_cfg['lr']}")

pbar = tqdm(range(smolvla_cfg["steps"]), desc="  SmolVLA FT", ncols=90)
for step in pbar:
    opt_s.zero_grad()
    accum_loss = 0
    
    for _ in range(grad_accum):
        bi = np.random.choice(train_idx, micro_bs, replace=True)
        batch = build_smolvla_train_batch(bi)
        
        if smolvla_cfg["fp16"]:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = smolvla.forward(batch)
                loss = loss / grad_accum
            loss.backward()
        else:
            loss, loss_dict = smolvla.forward(batch)
            loss = loss / grad_accum
            loss.backward()
        
        accum_loss += loss.item()
    
    torch.nn.utils.clip_grad_norm_(smolvla.parameters(), smolvla_cfg["max_grad_norm"])
    opt_s.step()
    
    losses_s.append(accum_loss)
    if step % 100 == 0:
        pbar.set_postfix(loss=f'{np.mean(losses_s[-50:]):.4f}')
    
    if (step + 1) % smolvla_cfg["save_every"] == 0:
        torch.save(smolvla.state_dict(), OUTPUT_DIR / f'smolvla_step{step+1}.pt')

torch.save(smolvla.state_dict(), OUTPUT_DIR / 'smolvla_finetuned.pt')
smolvla.eval()
print(f"\n  SmolVLA final loss: {np.mean(losses_s[-50:]):.6f}")

# Free some VRAM
gc.collect()
torch.cuda.empty_cache()

# =====================================================================
#  PHASE 3: Fine-Tune ACT
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 3: Fine-Tune ACT (51.6M, full)")
print("=" * 65)

from lerobot.policies.act.modeling_act import ACTPolicy

# Try loading pretrained ACT model
ACT_MODELS = [
    "cadene/act_so100_5_lego_test_080000",
    "cadene/act_so100_5_lego_test_020000",
    "masato-ka/act_so100_pickandplace_block_circle",
    "pepijn223/act_so100",
]
act_policy = None
for mid in ACT_MODELS:
    try:
        print(f"  Loading '{mid}'...")
        act_policy = ACTPolicy.from_pretrained(mid)
        print(f"  ACT loaded! Params: {sum(p.numel() for p in act_policy.parameters())/1e6:.1f}M")
        break
    except Exception as e:
        print(f"  Failed: {e}")

if act_policy is None:
    print("  ERROR: Could not load any ACT model!")
    sys.exit(1)

# Key remapping for ACT
act_img_keys = list(act_policy.config.image_features.keys()) if hasattr(act_policy.config, 'image_features') else []
KEY_REMAP_A = {}
for i, dk in enumerate(image_keys):
    if i < len(act_img_keys):
        KEY_REMAP_A[dk] = act_img_keys[i]
print(f"  ACT Key remap: {KEY_REMAP_A}")
print(f"  ACT chunk_size: {act_policy.config.chunk_size}")

act_policy.to(DEVICE)
act_policy.train()
act_params = sum(p.numel() for p in act_policy.parameters())
trainable_act = sum(p.numel() for p in act_policy.parameters() if p.requires_grad)
print(f"  ACT trainable: {trainable_act/1e6:.1f}M")

def build_act_train_batch(indices):
    """Build a training batch for ACT."""
    batch = {}
    imgs_by_key = {ak: [] for ak in KEY_REMAP_A.values()}
    states, actions = [], []
    
    for idx in indices:
        s = dataset[idx]
        for dk, ak in KEY_REMAP_A.items():
            imgs_by_key[ak].append(s[dk])
        if state_key:
            states.append(s[state_key])
        actions.append(s[action_key])
    
    for ak, imgs in imgs_by_key.items():
        batch[ak] = torch.stack(imgs).to(DEVICE)
    
    if states:
        batch['observation.state'] = torch.stack(states).to(DEVICE)
    
    # ACT needs action chunks + padding mask
    B = len(indices)
    chunk = act_policy.config.chunk_size
    action_t = torch.stack(actions).to(DEVICE)  # (B, action_dim)
    action_chunk = action_t.unsqueeze(1).expand(B, chunk, ACTION_DIM).clone()
    batch['action'] = action_chunk
    batch['action_is_pad'] = torch.zeros(B, chunk, dtype=torch.bool, device=DEVICE)
    
    return batch

opt_a = torch.optim.AdamW(
    act_policy.get_optim_params(),
    lr=act_cfg["lr"], weight_decay=act_cfg["weight_decay"]
)
scaler_a = torch.amp.GradScaler() if act_cfg["fp16"] else None
losses_a = []
micro_bs_a = act_cfg["micro_batch"]
grad_accum_a = act_cfg["grad_accum"]

print(f"\n  Training: {act_cfg['steps']} steps, micro_batch={micro_bs_a}, "
      f"grad_accum={grad_accum_a}, lr={act_cfg['lr']}")

pbar = tqdm(range(act_cfg["steps"]), desc="  ACT FT", ncols=90)
for step in pbar:
    opt_a.zero_grad()
    accum_loss = 0
    
    for _ in range(grad_accum_a):
        bi = np.random.choice(train_idx, micro_bs_a, replace=True)
        batch = build_act_train_batch(bi)
        
        if act_cfg["fp16"]:
            with torch.amp.autocast('cuda'):
                loss, loss_dict = act_policy.forward(batch)
                loss = loss / grad_accum_a
            scaler_a.scale(loss).backward()
        else:
            loss, loss_dict = act_policy.forward(batch)
            loss = loss / grad_accum_a
            loss.backward()
        
        accum_loss += loss.item()
    
    if act_cfg["fp16"]:
        scaler_a.unscale_(opt_a)
        torch.nn.utils.clip_grad_norm_(act_policy.parameters(), act_cfg["max_grad_norm"])
        scaler_a.step(opt_a)
        scaler_a.update()
    else:
        torch.nn.utils.clip_grad_norm_(act_policy.parameters(), act_cfg["max_grad_norm"])
        opt_a.step()
    
    losses_a.append(accum_loss)
    if step % 100 == 0:
        pbar.set_postfix(loss=f'{np.mean(losses_a[-50:]):.4f}')
    
    if (step + 1) % act_cfg["save_every"] == 0:
        torch.save(act_policy.state_dict(), OUTPUT_DIR / f'act_step{step+1}.pt')

torch.save(act_policy.state_dict(), OUTPUT_DIR / 'act_finetuned.pt')
act_policy.eval()
print(f"\n  ACT final loss: {np.mean(losses_a[-50:]):.6f}")

# Save training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
w = 50
if len(losses_s) > w:
    ax1.plot(np.convolve(losses_s, np.ones(w)/w, 'valid'), color='#2196F3', lw=1.5)
ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')
ax1.set_title('SmolVLA Fine-Tune Loss'); ax1.grid(True, alpha=0.3)

if len(losses_a) > w:
    ax2.plot(np.convolve(losses_a, np.ones(w)/w, 'valid'), color='#FF5722', lw=1.5)
ax2.set_xlabel('Step'); ax2.set_ylabel('Loss')
ax2.set_title('ACT Fine-Tune Loss'); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150)
plt.close()

# =====================================================================
#  PHASE 4: Evaluate Both
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 4: Evaluate Fine-Tuned Models")
print("=" * 65)

def build_smolvla_eval_batch(sample_dict, device):
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

def build_act_eval_batch(sample_dict, device):
    batch = {}
    for k in sample_dict:
        if k in meta_keys or k in string_keys:
            continue
        v = sample_dict[k]
        if not isinstance(v, torch.Tensor):
            continue
        out_key = KEY_REMAP_A.get(k, k)
        batch[out_key] = v.unsqueeze(0).to(device)
    return batch

def evaluate_model(model, batch_fn, device, model_name, eval_episodes):
    results = {
        'mse_per_episode': [], 'latency_ms': [],
        'predictions': {}, 'ground_truth': {},
    }
    
    for ep_idx in eval_episodes:
        ep_preds, ep_gts = [], []
        indices = episode_indices[ep_idx]
        model.reset()
        
        for step_idx in tqdm(indices, desc=f"  {model_name} Ep{ep_idx}", ncols=85, leave=False):
            s = dataset[step_idx]
            gt = s[action_key].numpy()
            
            batch = batch_fn(s, device)
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
        print(f"    {model_name} Ep{ep_idx}: MSE={mse:.4f} ({len(indices)} frames)")
    
    all_p = np.concatenate(list(results['predictions'].values()))
    all_g = np.concatenate(list(results['ground_truth'].values()))
    results['mse_total'] = float(np.mean((all_p - all_g)**2))
    results['mse_per_joint'] = np.mean((all_p - all_g)**2, axis=0).tolist()
    results['latency_mean_ms'] = float(np.mean(results['latency_ms']))
    
    return results

eval_ep_list = eval_eps[:EVAL["num_eval_episodes"]]
print(f"  Eval episodes: {eval_ep_list}")

print("\n  Evaluating SmolVLA (Fine-Tuned)...")
smolvla_results = evaluate_model(
    smolvla, build_smolvla_eval_batch, smolvla.config.device, "SmolVLA", eval_ep_list
)
print(f"  SmolVLA MSE: {smolvla_results['mse_total']:.4f}, Latency: {smolvla_results['latency_mean_ms']:.1f}ms")

print("\n  Evaluating ACT (Fine-Tuned)...")
act_results = evaluate_model(
    act_policy, build_act_eval_batch, act_policy.config.device, "ACT", eval_ep_list
)
print(f"  ACT MSE: {act_results['mse_total']:.4f}, Latency: {act_results['latency_mean_ms']:.1f}ms")

# =====================================================================
#  PHASE 5: Comparison
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 5: Comparison Report")
print("=" * 65)

smolvla_total = sum(p.numel() for p in smolvla.parameters())
act_total = sum(p.numel() for p in act_policy.parameters())

print(f"\n  {'='*65}")
print(f"  {'METRIC':<25} {'SmolVLA (FT)':<22} {'ACT (FT)':<18}")
print(f"  {'='*65}")
print(f"  {'Parameters (M)':<25} {smolvla_total/1e6:<22.1f} {act_total/1e6:<18.1f}")
print(f"  {'Total MSE':<25} {smolvla_results['mse_total']:<22.4f} {act_results['mse_total']:<18.4f}")
print(f"  {'Avg Latency (ms)':<25} {smolvla_results['latency_mean_ms']:<22.1f} {act_results['latency_mean_ms']:<18.1f}")
for j in range(ACTION_DIM):
    sj = smolvla_results['mse_per_joint'][j]
    aj = act_results['mse_per_joint'][j]
    tag = " <-- SmolVLA" if sj < aj else ""
    print(f"  {'J'+str(j)+' MSE':<25} {sj:<22.4f} {aj:<18.4f}{tag}")
print(f"  {'='*65}")

sw = sum(1 for s, a in zip(smolvla_results['mse_per_joint'], act_results['mse_per_joint']) if s < a)
print(f"\n  SmolVLA wins: {sw}/{ACTION_DIM} joints")
print(f"  Overall: {'SmolVLA' if smolvla_results['mse_total'] < act_results['mse_total'] else 'ACT'} wins MSE")

# Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
blue, orange = '#2196F3', '#FF5722'
bw = 0.35

# [0,0] MSE per episode
ax = axes[0,0]
x = np.arange(len(eval_ep_list))
ax.bar(x-bw/2, smolvla_results['mse_per_episode'], bw, label='SmolVLA (FT)', color=blue)
ax.bar(x+bw/2, act_results['mse_per_episode'], bw, label='ACT (FT)', color=orange)
ax.set_xlabel('Episode'); ax.set_ylabel('MSE'); ax.set_title('MSE per Episode')
ax.set_xticks(x); ax.set_xticklabels([f'Ep{e}' for e in eval_ep_list])
ax.legend(); ax.grid(True, alpha=0.3)

# [0,1] MSE per joint
ax = axes[0,1]
x = np.arange(ACTION_DIM)
ax.bar(x-bw/2, smolvla_results['mse_per_joint'], bw, label='SmolVLA (FT)', color=blue)
ax.bar(x+bw/2, act_results['mse_per_joint'], bw, label='ACT (FT)', color=orange)
ax.set_xlabel('Joint'); ax.set_ylabel('MSE'); ax.set_title('MSE per Joint')
ax.set_xticks(x); ax.legend(); ax.grid(True, alpha=0.3)

# [0,2] Latency
ax = axes[0,2]
ax.hist(smolvla_results['latency_ms'], bins=40, alpha=0.7, label='SmolVLA', color=blue)
ax.hist(act_results['latency_ms'], bins=40, alpha=0.7, label='ACT', color=orange)
ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Count'); ax.set_title('Inference Latency')
ax.legend(); ax.grid(True, alpha=0.3)

# [1,0] Training curves
ax = axes[1,0]
w = 50
if len(losses_s) > w:
    ax.plot(np.convolve(losses_s, np.ones(w)/w, 'valid'), color=blue, label='SmolVLA', lw=1.5)
if len(losses_a) > w:
    ax.plot(np.convolve(losses_a, np.ones(w)/w, 'valid'), color=orange, label='ACT', lw=1.5)
ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('Training Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# [1,1] Trajectory Ep0
ax = axes[1,1]
ep0 = eval_ep_list[0]
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
ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)

# [1,2] Summary table as text
ax = axes[1,2]
ax.axis('off')
summary = (
    f"SUMMARY\n\n"
    f"Dataset: {ds_cfg['repo_id']}\n"
    f"Train: {len(train_eps)} eps, Eval: {len(eval_ep_list)} eps\n\n"
    f"SmolVLA (450M, FT Expert)\n"
    f"  MSE: {smolvla_results['mse_total']:.4f}\n"
    f"  Latency: {smolvla_results['latency_mean_ms']:.1f}ms\n\n"
    f"ACT ({act_total/1e6:.1f}M, Full FT)\n"
    f"  MSE: {act_results['mse_total']:.4f}\n"
    f"  Latency: {act_results['latency_mean_ms']:.1f}ms\n\n"
    f"Winner (MSE): {'SmolVLA' if smolvla_results['mse_total'] < act_results['mse_total'] else 'ACT'}\n"
    f"SmolVLA wins {sw}/{ACTION_DIM} joints"
)
ax.text(0.1, 0.5, summary, fontsize=11, fontfamily='monospace',
        va='center', ha='left', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle('SmolVLA (Fine-Tuned) vs ACT (Fine-Tuned)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'comparison_plots.png', dpi=150)
plt.close()

# =====================================================================
#  PHASE 6: Videos
# =====================================================================
print("\n" + "=" * 65)
print("  PHASE 6: Videos")
print("=" * 65)

for ep_idx in eval_ep_list[:3]:
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
        nj_v = min(ACTION_DIM, 6)
        px0 = 340; pw = fw - px0 - 15
        jh = max(45, (fh - 80) // nj_v - 6)
        for j in range(nj_v):
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
                cv2.line(frame, (x1,_py(gt[tt,j])), (x2,_py(gt[tt+1,j])), (0,220,0), 2)
                cv2.line(frame, (x1,_py(sp[tt,j])), (x2,_py(sp[tt+1,j])), (255,160,0), 1)
                cv2.line(frame, (x1,_py(ap[tt,j])), (x2,_py(ap[tt+1,j])), (50,80,255), 1)

        yb = fh - 25
        cv2.putText(frame, 'GT', (15, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(frame, 'SmolVLA', (55, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,160,0), 1)
        cv2.putText(frame, 'ACT', (180, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,80,255), 1)
        ms = float(np.mean((sp[t]-gt[t])**2))
        ma = float(np.mean((ap[t]-gt[t])**2))
        cv2.putText(frame, f'SmolVLA:{ms:.2f}', (350, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,160,0), 1)
        cv2.putText(frame, f'ACT:{ma:.2f}', (600, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,80,255), 1)
        writer.write(frame)
    writer.release()
    print(f"  Video: {vpath}")

# =====================================================================
#  Save Report
# =====================================================================
report = {
    'dataset': ds_cfg['repo_id'],
    'dataset_key': DATASET_KEY,
    'train_episodes': list(train_eps),
    'eval_episodes': list(eval_ep_list),
    'smolvla': {
        'label': 'SmolVLA (450M, Fine-Tuned Expert)',
        'params_total_M': round(smolvla_total/1e6, 1),
        'params_trainable_M': round(trainable/1e6, 1),
        'training_steps': smolvla_cfg['steps'],
        'final_loss': float(np.mean(losses_s[-50:])),
        'total_mse': smolvla_results['mse_total'],
        'per_episode_mse': smolvla_results['mse_per_episode'],
        'per_joint_mse': smolvla_results['mse_per_joint'],
        'avg_latency_ms': smolvla_results['latency_mean_ms'],
    },
    'act': {
        'label': 'ACT (Full Fine-Tune)',
        'params_total_M': round(act_total/1e6, 1),
        'training_steps': act_cfg['steps'],
        'final_loss': float(np.mean(losses_a[-50:])),
        'total_mse': act_results['mse_total'],
        'per_episode_mse': act_results['mse_per_episode'],
        'per_joint_mse': act_results['mse_per_joint'],
        'avg_latency_ms': act_results['latency_mean_ms'],
    },
}
with open(OUTPUT_DIR / 'comparison_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n  Report: {OUTPUT_DIR / 'comparison_report.json'}")

print("\n" + "=" * 65)
print("  PIPELINE COMPLETE")
print("=" * 65)
print(f"\n  SmolVLA FT: MSE={smolvla_results['mse_total']:.4f} Latency={smolvla_results['latency_mean_ms']:.1f}ms")
print(f"  ACT FT:     MSE={act_results['mse_total']:.4f} Latency={act_results['latency_mean_ms']:.1f}ms")
winner = 'SmolVLA' if smolvla_results['mse_total'] < act_results['mse_total'] else 'ACT'
print(f"\n  Winner: {winner}")
print(f"  Output: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f.name:<35} {f.stat().st_size/1024:.0f} KB")
print("\n  DONE!")
