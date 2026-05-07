import os, sys, time, json, types, gc, argparse
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

# Patch lerobot for local setup
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

from finetune_config import DATASETS, TRAINING
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# We must redefine the custom modules to load state dict safely (same as train)
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
        gates = (torch.sigmoid(logits) > 0.5).float() if hard or not self.training else F.gumbel_softmax(
            torch.stack([torch.zeros_like(logits), logits], dim=-1), tau=tau, hard=False, dim=-1)[..., 1]
        return gates, gates.mean()

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
            attention_scores = torch.arange(N, 0, -1, device=token_embeddings.device).float().unsqueeze(0).expand(B, -1)
        _, top_indices = attention_scores.topk(K, dim=-1, sorted=False)
        top_indices_sorted, _ = top_indices.sort(dim=-1)
        pruned_embeddings = torch.gather(token_embeddings, 1, top_indices_sorted.unsqueeze(-1).expand(-1, -1, D))
        pruned_mask = torch.gather(token_mask, 1, top_indices_sorted)
        return pruned_embeddings, pruned_mask, keep_ratio_per_sample.mean(), keep_ratio_per_sample

class LoRASPAdapter(nn.Module):
    def __init__(self, in_features, out_features, max_rank=128, energy_threshold=0.9, alpha=1.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.max_rank, self.scaling = max_rank, alpha / max_rank
        self.U = nn.Parameter(torch.randn(out_features, max_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)
        self.router = nn.Sequential(nn.Linear(in_features, max_rank), nn.Sigmoid())

    def forward(self, x, return_spec_loss=False):
        x_flat = x.reshape(-1, self.in_features)
        scores = self.router(x_flat.detach())
        SVx = F.linear(x_flat, self.V) * scores * self.scaling
        delta = F.linear(SVx, self.U).reshape(*x.shape[:-1], self.out_features)
        return (delta, 0.0) if return_spec_loss else delta


def build_eval_batch(sample_dict, device, image_keys, lang_ids, lang_mask):
    batch = {}
    for k in sample_dict:
        if k in {'episode_index', 'frame_index', 'timestamp', 'index', 'task_index'} or isinstance(sample_dict[k], str):
            continue
        v = sample_dict[k]
        if not isinstance(v, torch.Tensor): continue
        batch[k] = v.unsqueeze(0).to(device)
        
    # Remap keys for smolvla if necessary
    if 'observation.state' in batch and 'observation.state' not in sample_dict:
      pass # no need
    batch['observation.language.tokens'] = lang_ids.to(device)
    batch['observation.language.attention_mask'] = lang_mask.to(device)
    return batch


def evaluate_method(method_name, dataset, episode_indices, eval_ep_list, image_keys, state_key, action_key, ds_cfg, dyn_cfg, DEVICE):
    print(f"\n======================================================================")
    print(f"  EVALUATING ABLATION METHOD: {method_name}")
    print(f"======================================================================")

    output_dir = Path(f"d:/EyetechCode/results/ablation_{method_name}")
    checkpoint_path = output_dir / f"ablation_{method_name}_final.pt"

    if not checkpoint_path.exists():
        print(f"  [!] Checkpoint not found: {checkpoint_path}. Skipping.")
        return None

    # Load BaseModel
    print("  Loading pretrained SmolVLA base...")
    smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    tokenizer = smolvla.model.vlm_with_expert.processor.tokenizer
    _tok = tokenizer(ds_cfg["task_instruction"], return_tensors="pt", padding="max_length", max_length=64)
    LANG_IDS, LANG_MASK = _tok['input_ids'], _tok['attention_mask'].bool()
    
    # SmolVLA dynamically remaps key based on config
    KEY_REMAP_S = {dk: sk for i, (dk, sk) in enumerate(zip(image_keys, smolvla.config.image_features.keys()))}

    smolvla.to(DEVICE)
    
    # Initialize components
    flow_model = smolvla.model
    vlm_hidden = flow_model.vlm_with_expert.config.text_config.hidden_size
    star_router = STARRouter(hidden_dim=vlm_hidden, num_skippable_layers=dyn_cfg["num_skippable_layers"]).to(DEVICE)
    token_pruner = ActionAwareTokenPruner(v_threshold=dyn_cfg["adp_velocity_threshold"], min_keep_ratio=dyn_cfg["adp_min_keep_ratio"]).to(DEVICE)

    # Note: load_state_dict properly overrides original loaded weights
    try:
        print(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
        if 'smolvla' in ckpt:
            smolvla.load_state_dict(ckpt['smolvla'])
        if 'star_router' in ckpt and ckpt['star_router']:
            star_router.load_state_dict(ckpt['star_router'])
        if 'token_pruner' in ckpt and ckpt['token_pruner']:
            token_pruner.load_state_dict(ckpt['token_pruner'])
        print("  ✓ Checkpoint loaded successfully.")
    except Exception as e:
        print(f"  [X] Failed to load checkpoint: {e}")
        return None
        
    smolvla.eval()
    star_router.eval()
    token_pruner.eval()
    
    action_dim = dataset[0][action_key].shape[-1]
    
    results = {'mse_per_episode': [], 'latency_ms': [], 'predictions': {}, 'ground_truth': {}}

    for ep_idx in eval_ep_list:
        ep_preds, ep_gts = [], []
        indices = episode_indices[ep_idx]
        smolvla.reset()

        for step_idx in tqdm(indices, desc=f"  Ep{ep_idx}", ncols=85, leave=False):
            s = dataset[step_idx]
            gt = s[action_key].numpy()

            # Fix sample dict keys for smolvla standard
            smolvla_s = {KEY_REMAP_S.get(k, k): v for k, v in s.items()}
            batch = build_eval_batch(smolvla_s, DEVICE, image_keys, LANG_IDS, LANG_MASK)
            
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = smolvla.select_action(batch)
            t1 = time.perf_counter()

            pred_np = pred.squeeze().cpu().numpy()
            if pred_np.ndim > 1: pred_np = pred_np[0]
            pred_np = pred_np[:action_dim]

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

    print(f"  Overall MSE: {results['mse_total']:.4f}")
    
    # Save Report
    report = {
        'method': method_name,
        'checkpoint': str(checkpoint_path),
        'eval_episodes': list(eval_ep_list),
        'results': {
            'total_mse': results['mse_total'],
            'per_episode_mse': {f'ep{ep}': mse for ep, mse in zip(eval_ep_list, results['mse_per_episode'])},
            'per_joint_mse': results['mse_per_joint'],
            'avg_latency_ms': results['latency_mean_ms'],
        }
    }
    with open(output_dir / f'eval_report_{method_name}.json', 'w') as f:
        json.dump(report, f, indent=2)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="snapflow,token_prune,layer_skip")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    methods = args.methods.split(",")

    DATASET_KEY = "svla_so100_pickplace"
    ds_cfg = DATASETS[DATASET_KEY]
    dyn_cfg = TRAINING["dynamic_lora_pruning_snapflow"]

    print("Loading Dataset...")
    dataset = LeRobotDataset(ds_cfg["repo_id"])
    
    sample = dataset[0]
    all_keys = list(sample.keys())
    image_keys = [k for k in all_keys if 'image' in k.lower() and isinstance(sample[k], torch.Tensor)]
    state_key = next((k for k in all_keys if 'state' in k.lower() and isinstance(sample[k], torch.Tensor)), None)
    action_key = 'action'
    
    ep_col = dataset.hf_dataset['episode_index']
    episode_indices = {}
    for idx, ep in enumerate(ep_col):
        ep_int = ep.item() if isinstance(ep, torch.Tensor) else int(ep)
        if ep_int not in episode_indices: episode_indices[ep_int] = []
        episode_indices[ep_int].append(idx)
        
    all_eps = sorted(episode_indices.keys())
    n_train = ds_cfg["train_episodes"]
    eval_ep_list = all_eps[n_train:n_train + 3] # Evaluates first 3 (e.g. 40,41,42)
    
    all_results = {}
    for method in methods:
        method = method.strip()
        if not method: continue
        res = evaluate_method(method, dataset, episode_indices, eval_ep_list, image_keys, state_key, action_key, ds_cfg, dyn_cfg, DEVICE)
        if res:
            all_results[method] = res
            
            # Create Plot for this specific method tracking what we found
            output_dir = Path(f"d:/EyetechCode/results/ablation_{method}")
            fig, ax = plt.subplots(figsize=(8, 6))
            x_j = np.arange(dataset[0][action_key].shape[-1])
            ax.bar(x_j, res['mse_per_joint'], 0.6, color='#2196F3')
            ax.set_title(f'MSE per Joint - {method}')
            ax.set_xlabel('Joint'); ax.set_ylabel('MSE')
            ax.grid(True, alpha=0.3)
            plt.savefig(output_dir / f'eval_plot_{method}.png')
            plt.close()
            print(f"  ✓ Plot saved to: {output_dir / f'eval_plot_{method}.png'}")
            
    print("\n[Done] Evaluated methods:")
    for method, res in all_results.items():
        print(f"  - {method}: MSE={res['mse_total']:.4f}, Latency={res['latency_mean_ms']:.1f}ms")

if __name__ == "__main__":
    main()
