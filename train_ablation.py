"""
Ablation Study: Evaluate Finetuning Strategies Independently
============================================================
This script isolates the 3 primary architectural enhancements:
  1. Dynamic Layer Skipping Only
  2. Token Pruning (ADP) Only
  3. SnapFlow (1-NFE Distillation) Only

Usage:
  python train_ablation.py --method [layer_skip|token_prune|snapflow]
"""
import os, sys, argparse, time, copy, gc
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

# Patch lerobot for local setup
import lerobot
_pkg = lerobot.__path__[0]
import importlib.util
import types

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

from finetune_config import DATASETS, TRAINING
from finetune_dynamic_lora_prunning_snapflow import (
    DynamicSmolVLAWrapper, STARRouter, ActionAwareTokenPruner, SnapFlowTrainer
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Hardware setup mapping to MPS
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def setup_ablation_model(method, smolvla, dyn_cfg):
    """Initializes the wrapper and activates ONLY the specified component."""
    wrapper = DynamicSmolVLAWrapper(smolvla, dyn_cfg)
    
    # Freeze base parameters based on method logic
    for p in smolvla.parameters():
        p.requires_grad = False
        
    wrapper.star_router.requires_grad_(False)
    wrapper.token_pruner.requires_grad_(False)
    for adapter in wrapper.lora_adapters.values():
         adapter.requires_grad_(False)

    params_to_train = []
    
    if method == "layer_skip":
        # Unfreeze router + gate logic
        wrapper.star_router.requires_grad_(True)
        params_to_train.extend(wrapper.star_router.parameters())
        print(">> Activated ONLY Dynamic Layer Skipping (STAR Router)")
        
    elif method == "token_prune":
        # Unfreeze token pruner parameters
        wrapper.token_pruner.requires_grad_(True)
        params_to_train.extend(wrapper.token_pruner.parameters())
        print(">> Activated ONLY Action-Aware Token Pruning")
        
    elif method == "snapflow":
        # Unfreeze action experts for Flow Matching
        for name, p in smolvla.named_parameters():
             if "action_expert" in name or "proj" in name:
                  p.requires_grad = True
                  params_to_train.append(p)
        print(">> Activated ONLY SnapFlow (Flow Matching)")

    return wrapper, params_to_train


def build_train_batch(indices, dataset, smolvla, device, image_keys, state_key, action_key, lang_ids, lang_mask, chunk_size):
    # Get SmolVLA key remapping
    smolvla_img_keys = list(smolvla.config.image_features.keys())
    # Safely match lengths to avoid out of bounds
    key_remap_s = {dk: smolvla_img_keys[i] for i, dk in enumerate(image_keys) if i < len(smolvla_img_keys)}

    batch_imgs = {sk: [] for sk in key_remap_s.values()}
    batch_states, batch_actions = [], []

    for idx in indices:
        s = dataset[idx]
        for dk, sk in key_remap_s.items():
            batch_imgs[sk].append(s[dk])
        if state_key:
            batch_states.append(s[state_key])
        batch_actions.append(s[action_key])

    batch = {}
    for sk, imgs in batch_imgs.items():
        batch[sk] = torch.stack(imgs).to(device)
    if batch_states:
        batch['observation.state'] = torch.stack(batch_states).to(device)

    actions = torch.stack(batch_actions).to(device)
    B = actions.shape[0]

    MAX_ACT_DIM = smolvla.config.max_action_dim
    ACTION_DIM = actions.shape[-1]
    if ACTION_DIM < MAX_ACT_DIM:
        pad_zeros = torch.zeros(B, MAX_ACT_DIM - ACTION_DIM, device=device)
        actions_padded = torch.cat([actions, pad_zeros], dim=1)
    else:
        actions_padded = actions

    action_chunk = actions_padded.unsqueeze(1).expand(B, chunk_size, MAX_ACT_DIM)
    batch['action'] = action_chunk
    batch['actions_id_pad'] = torch.zeros(B, chunk_size, dtype=torch.bool, device=device)

    batch['observation.language.tokens'] = lang_ids.expand(B, -1).to(device)
    batch['observation.language.attention_mask'] = lang_mask.expand(B, -1).to(device)

    return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, 
                        choices=["layer_skip", "token_prune", "snapflow"],
                        help="Which ablation feature to train independently")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    # Load configs
    ds_cfg = DATASETS["svla_so100_pickplace"]
    dyn_cfg = TRAINING["dynamic_lora_pruning_snapflow"]
    
    # Fix output dir to Windows path 
    OUTPUT_DIR = Path(f"d:/EyetechCode/results/ablation_{args.method}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Dataset: {ds_cfg['repo_id']}")
    dataset = LeRobotDataset(ds_cfg["repo_id"])
    
    # Preparation for batching
    sample = dataset[0]
    all_keys = list(sample.keys())
    image_keys = [k for k in all_keys if 'image' in k.lower() and isinstance(sample[k], torch.Tensor)]
    state_key = next((k for k in all_keys if 'state' in k.lower() and isinstance(sample[k], torch.Tensor)), None)
    action_key = 'action'
    
    # Fast episode indexing
    ep_col = dataset.hf_dataset['episode_index']
    episode_indices = {}
    for idx, ep in enumerate(ep_col):
        ep_int = ep.item() if isinstance(ep, torch.Tensor) else int(ep)
        if ep_int not in episode_indices: episode_indices[ep_int] = []
        episode_indices[ep_int].append(idx)
        
    train_eps = sorted(episode_indices.keys())[:ds_cfg["train_episodes"]]
    train_idx = []
    for ep in train_eps:
        train_idx.extend(episode_indices[ep])

    print("Loading Base SmolVLA...")
    smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(DEVICE)
    
    # Text tokenizer setup
    tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
    _tok = tokenizer(ds_cfg["task_instruction"], return_tensors="pt", padding="max_length", max_length=64)
    LANG_IDS = _tok['input_ids']
    LANG_MASK = _tok['attention_mask'].bool()
    CHUNK_SIZE_S = smolvla.config.chunk_size
    
    # Prepare ablation environment
    wrapper, trainable_params = setup_ablation_model(args.method, smolvla, dyn_cfg)
    wrapper.to(DEVICE)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=dyn_cfg.get("phase1_lr", 1e-4))
    
    print(f"Starting Ablation Training for **{args.method}**...")
    wrapper.train()
    
    enable_skip = (args.method == "layer_skip")
    enable_adp = (args.method == "token_prune")
    enable_snap = (args.method == "snapflow")

    import contextlib
    micro_bs = dyn_cfg.get("micro_batch", 2)
    grad_accum = dyn_cfg.get("grad_accum", 8)
    
    pbar = tqdm(range(args.steps), desc=f"Train {args.method}", ncols=100)
    import numpy as np
    
    for step in pbar:
        optimizer.zero_grad()
        accum_loss = 0
        
        # Anneal Gumbel temperature if doing layer skip
        progress = step / max(args.steps - 1, 1)
        tau = dyn_cfg.get("gumbel_tau_start", 1.0) * (1 - progress) + dyn_cfg.get("gumbel_tau_end", 0.1) * progress

        for _ in range(grad_accum):
            bi = np.random.choice(train_idx, micro_bs, replace=True)
            batch = build_train_batch(bi, dataset, smolvla, DEVICE, image_keys, state_key, action_key, LANG_IDS, LANG_MASK, CHUNK_SIZE_S)

            ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if dyn_cfg.get("fp16", False) and DEVICE == "cuda" else contextlib.nullcontext()
            with ctx:
                loss, loss_dict = wrapper.forward_with_skip(
                    batch, tau=tau,
                    enable_skip=enable_skip,
                    enable_adp=enable_adp,
                    enable_lora=False, # LoRA disabled in ablations
                    enable_snap=enable_snap,
                )
                loss = loss / grad_accum

            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, dyn_cfg.get("max_grad_norm", 1.0))
        optimizer.step()
        
        pbar.set_postfix({"Loss": f"{accum_loss:.4f}"})
        
    print(f"Finished {args.method} ablation phase!")
    # Save ablation model checkpoint
    torch.save({
        'smolvla': smolvla.state_dict(),
        'star_router': wrapper.star_router.state_dict() if enable_skip else None,
        'token_pruner': wrapper.token_pruner.state_dict() if enable_adp else None,
    }, OUTPUT_DIR / f'ablation_{args.method}_final.pt')
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
