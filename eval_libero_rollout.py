"""
LIBERO Rollout Evaluation for Hierarchical Action-Aware SmolVLA
===============================================================
Runs closed-loop rollout evaluation in LIBERO simulation.
Reports per-task success rate, mean success rate, and latency.

Usage:
    python eval_libero_rollout.py [--n_rollouts N] [--suite libero_10] [--checkpoint PATH]

Standard protocol: 20 rollouts per task (10 tasks) = 200 total episodes.
"""

import os, sys, time, json, types, argparse, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import warnings
warnings.filterwarnings("ignore")

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
_spec = importlib.util.spec_from_file_location(
    'lerobot.robots.config', os.path.join(_pkg, 'robots', 'config.py'))
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
#  ARGS
# =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--n_rollouts", type=int, default=20, help="Rollouts per task")
parser.add_argument("--suite", type=str, default="libero_10", help="LIBERO suite name")
parser.add_argument("--checkpoint", type=str,
    default="d:/EyetechCode/outputs/hierarchical_action_aware/final_model.pt",
    help="Path to trained model checkpoint")
parser.add_argument("--baseline_checkpoint", type=str,
    default="d:/EyetechCode/outputs/smolvla_plain_baseline/final_model.pt",
    help="Path to baseline SmolVLA checkpoint (for comparison)")
parser.add_argument("--output_dir", type=str,
    default="d:/EyetechCode/outputs/libero_rollout_eval",
    help="Output directory for results")
parser.add_argument("--horizon", type=int, default=600,
    help="Max steps per rollout episode")
parser.add_argument("--skip_dynamic", action="store_true",
    help="Skip dynamic model, only run baseline")
parser.add_argument("--skip_baseline", action="store_true",
    help="Skip baseline model, only run dynamic model")
parser.add_argument("--device", type=str, default="cuda",
    help="Device: cuda or cpu")
parser.add_argument("--img_size", type=int, default=256,
    help="Image size fed to SmolVLA (default: 256 to match training)")
parser.add_argument("--no_flip", action="store_true",
    help="Disable vertical image flip (for A/B testing the flip fix)")
parser.add_argument("--settle_steps", type=int, default=5,
    help="No-op zero-action steps after set_init_state before the policy runs. "
         "LIBERO official protocol uses 5 (libero/lifelong/metric.py:120). "
         "Increase if physics settling is unstable.")
parser.add_argument("--no_skip", action="store_true",
    help="Run the FULL model (no layer skipping). The training task loss is "
         "computed on the full model — skipping is never applied during training "
         "— so eval-time skipping is a train/test mismatch. Use this to match "
         "training. Recommended for measuring task success.")
parser.add_argument("--base_model", default="lerobot/smolvla_libero",
    help="HF model id for the SmolVLA base. Use lerobot/smolvla_libero (already "
         "fine-tuned on LIBERO) — NOT smolvla_base (generic, never saw LIBERO).")
parser.add_argument("--pretrained_only", action="store_true",
    help="Evaluate the pretrained base model directly (skip loading the dynamic "
         "checkpoint and the router). Use to validate the rollout env with the "
         "known-good lerobot/smolvla_libero policy. Implies --no_skip.")
parser.add_argument("--no_normalize", action="store_true",
    help="Disable MEAN_STD state/action normalization. ONLY for models trained on "
         "RAW data; smolvla_libero (and finetunes of it) REQUIRE normalization.")
parser.add_argument("--stats_path", type=str, default=None,
    help="Path to dataset stats.json for MEAN_STD normalization. Defaults to "
         "HuggingFaceVLA/libero stats. Use data/datasets/libero_10_full/meta/stats.json "
         "for models trained on libero_10_full.")
parser.add_argument("--save_video", action="store_true",
    help="Save MP4 video for the first rollout of each task to --output_dir/videos/")
parser.add_argument("--video_fps", type=int, default=20,
    help="FPS for saved videos (default: 20)")
parser.add_argument("--no_adp", action="store_true",
    help="Disable ADP token pruning (layer-skip only). Use to ablate DTP vs DLS.")
args = parser.parse_args()

DEVICE = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_ROLLOUTS = args.n_rollouts
SUITE_NAME = args.suite
HORIZON = args.horizon
IMG_SIZE = args.img_size
FLIP_IMAGES = not args.no_flip
SETTLE_STEPS = args.settle_steps
PRETRAINED_ONLY = args.pretrained_only
NO_SKIP = args.no_skip or PRETRAINED_ONLY      # pretrained base has no trained router
NO_ADP  = args.no_adp or PRETRAINED_ONLY       # pretrained base wasn't trained with ADP
BASE_MODEL = args.base_model

# ── MEAN_STD normalization (REQUIRED for smolvla_libero) ──────────────────────
# smolvla_libero was trained on MEAN_STD-normalized state/action (config:
# normalization_mapping STATE/ACTION = MEAN_STD). Verified by open-loop probe:
# feeding RAW state → MSE 0.30 (broken); normalizing state in + unnormalizing
# action out → MSE 0.008 (near-perfect). So we MUST normalize here.
NORMALIZE = not args.no_normalize
# Default: use libero_10_full stats (matches training). --pretrained_only falls
# back to HuggingFaceVLA stats (that's what the base model was trained on).
_LIBERO10_STATS  = "data/datasets/libero_10_full/meta/stats.json"
_HF_STATS        = "data/datasets/HuggingFaceVLA/libero/meta/stats.json"
_STATS_PATH = (args.stats_path
               or (_HF_STATS if PRETRAINED_ONLY else _LIBERO10_STATS))
S_MEAN = S_STD = A_MEAN = A_STD = None
if NORMALIZE:
    import json as _json
    with open(_STATS_PATH) as _f:
        _stt = _json.load(_f)
    S_MEAN = np.array(_stt["observation.state"]["mean"], np.float32).ravel()
    S_STD  = np.array(_stt["observation.state"]["std"],  np.float32).ravel()
    A_MEAN = np.array(_stt["action"]["mean"], np.float32).ravel()
    A_STD  = np.array(_stt["action"]["std"],  np.float32).ravel()
    print(f"  [NORMALIZE] MEAN_STD on state(in)/action(out) from {_STATS_PATH}")

# ── LIBERO init states protocol ───────────────────────────────────────────────
# suite.get_task_init_states(task_idx) loads from pre-sampled .pruned_init files
# (libero/benchmark/__init__.py:158).  These 50 states are sampled INDEPENDENTLY
# from the HDF5 demo files — they are NOT in demo order and have NO overlap with
# training demo init states.  The LIBERO official eval uses ALL 50 states for
# evaluation, cycling as needed (metric.py:111):
#   indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
# Standard: 20 rollouts per task, cycling through the 50 pre-sampled states.
# There is no train/test split for init states.


def _normalize_state(state_np):
    return (state_np - S_MEAN) / (S_STD + 1e-8) if NORMALIZE else state_np


def _unnormalize_action(act_np):
    if not NORMALIZE:
        return act_np
    n = min(act_np.shape[-1], A_MEAN.shape[0])
    out = act_np.copy()
    out[..., :n] = act_np[..., :n] * A_STD[:n] + A_MEAN[:n]
    return out

# LIBERO single-arm Panda expects 7D action (6 arm + 1 gripper)
ACTION_DIM = 7
_action_shape_warned = False

print("=" * 70)
print(f"  LIBERO Rollout Evaluation — {SUITE_NAME}")
print("=" * 70)
print(f"  Device:      {DEVICE}")
print(f"  Rollouts/task: {N_ROLLOUTS}")
print(f"  Horizon:     {HORIZON} steps")
print(f"  Checkpoint:  {args.checkpoint}")
print(f"  Output:      {OUTPUT_DIR}")
print()

# =====================================================================
#  CONFIG
# =====================================================================
from finetune_config import TRAINING

dyn_cfg = TRAINING["hierarchical_action_aware"]
NUM_FIXED   = dyn_cfg["num_fixed_layers"]
NUM_SKIP    = dyn_cfg["num_skippable_layers"]
SCALE_M     = dyn_cfg.get("kinematic_scale_m", 0.015)
SCALE_J     = dyn_cfg.get("kinematic_scale_j", 80.0)
K_LAMBDA    = dyn_cfg.get("kinematic_lambda", 0.6)
KIN_DIM     = dyn_cfg.get("kinematic_state_dim", 16)
V_THRESH    = dyn_cfg.get("adp_velocity_threshold", 0.015)   # ADP token-prune threshold
R_MIN       = dyn_cfg.get("adp_min_keep_ratio", 0.20)        # ADP min keep ratio
N_ACTION_STEPS = None   # set after model load (= chunk replan cadence)

# =====================================================================
#  MODULE DEFINITIONS  (must match training script for state_dict load)
# =====================================================================

class ActionAwarePRouter(nn.Module):
    """Matches finetune_hierarchical_action_aware.py — exact copy."""

    def __init__(self, hidden_dim: int, num_skippable_layers: int = 8,
                 rho_min: float = 0.0, rho_max: float = 0.5,
                 init_gain: float = 1.0, gate_bias_init: float = -2.0):
        super().__init__()
        self.num_skippable_layers = num_skippable_layers
        self.rho_min = rho_min
        self.rho_max = rho_max

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_skippable_layers),
        )
        self.rho_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.theta_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_skippable_layers),
        )
        for head in (self.gate_net[-1], self.rho_net[-1], self.theta_net[-1]):
            with torch.no_grad():
                head.weight.mul_(init_gain)
        nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)
        with torch.no_grad():
            bias_spread = torch.linspace(-1.0, 1.0, num_skippable_layers)
            self.theta_net[-1].bias.copy_(bias_spread)

    def forward(self, hidden_pooled, e_view, delta_s_norm,
                kin_features, s_t, tau=1.0, hard=True):
        rho_logits = self.rho_net(kin_features)
        rho_target = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(rho_logits)
        theta_t = torch.sigmoid(self.theta_net(kin_features))
        x = torch.cat([hidden_pooled, e_view, delta_s_norm, s_t], dim=-1)
        logits = self.gate_net(x) - theta_t
        if hard or not self.training:
            gates = (torch.sigmoid(logits / max(tau, 1e-4)) > 0.5).float()
        else:
            logits_2c = torch.stack([torch.zeros_like(logits), logits], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_2c, tau=tau, hard=False, dim=-1)
            gates = gumbel_out[..., 1]
        gate_probs = torch.sigmoid(logits)
        gate_loss = gates.mean()
        return gates, gate_loss, rho_target, theta_t, gate_probs


# =====================================================================
#  LOAD SMOLVLA
# =====================================================================
print(f"  Loading SmolVLA base: {BASE_MODEL}")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

smolvla = SmolVLAPolicy.from_pretrained(BASE_MODEL, device=DEVICE)
smolvla.to(DEVICE)
smolvla.eval()

# CRITICAL (must match training): smolvla_base declares action_feature.shape=[6],
# so select_action would return only 6D and drop the gripper. LIBERO needs 7D.
# Override so the model emits the 7th dim (gripper) — trained models supervise it.
_af = smolvla.config.action_feature
if _af is not None and _af.shape[0] != ACTION_DIM:
    _old = tuple(_af.shape)
    _af.shape = (ACTION_DIM,)
    print(f"  [FIX] action_feature.shape {_old} -> ({ACTION_DIM},) (include gripper)")

flow_model    = smolvla.model
vlm           = flow_model.vlm_with_expert
vlm_hidden    = vlm.config.text_config.hidden_size
num_vlm_layers = vlm.num_vlm_layers
N_ACTION_STEPS = int(getattr(smolvla.config, "n_action_steps", 50))  # chunk replan cadence

# SmolVLA image keys (e.g. "observation.images.image", "observation.images.image2")
smolvla_img_keys = list(smolvla.config.image_features.keys())
print(f"  SmolVLA image keys: {smolvla_img_keys}")

# Tokenizer
tokenizer = vlm.processor.tokenizer

# =====================================================================
#  LOAD ACTION ROUTER
# =====================================================================
action_router = ActionAwarePRouter(
    hidden_dim=vlm_hidden,
    num_skippable_layers=NUM_SKIP,
    rho_min=dyn_cfg.get("rho_target_min", 0.0),
    rho_max=dyn_cfg.get("rho_target_max", 0.5),
    init_gain=dyn_cfg.get("router_init_gain", 1.0),
    gate_bias_init=dyn_cfg.get("router_gate_bias_init", -2.0),
).to(DEVICE)
action_router.eval()


def _load_checkpoint(ckpt_path, device):
    """Load checkpoint with integrity check."""
    p = Path(ckpt_path)
    candidates = [
        p,
        Path.cwd() / p,
        OUTPUT_DIR / p.name,
        Path("outputs") / "hierarchical_action_aware" / p.name,
        Path("outputs") / p.name,
    ]
    resolved = next((c for c in candidates if c.exists()), None)
    if resolved is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Tried: {', '.join(str(c) for c in candidates)}"
        )
    try:
        return torch.load(str(resolved), map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}")


# ── Load dynamic model checkpoint ──
if PRETRAINED_ONLY:
    print(f"  [PRETRAINED-ONLY] Using {BASE_MODEL} directly (no checkpoint, no router, no skip).")
elif not args.skip_dynamic:
    print(f"  Loading dynamic model from: {args.checkpoint}")
    ckpt = _load_checkpoint(args.checkpoint, DEVICE)
    smolvla.load_state_dict(ckpt['smolvla'])
    print("  ✓ SmolVLA weights loaded")
    if 'action_router' in ckpt:
        action_router.load_state_dict(ckpt['action_router'])
        print("  ✓ ActionAwarePRouter weights loaded")
    else:
        print("  ✗ No 'action_router' key in checkpoint — router uses random weights!")
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

# ── Load baseline checkpoint ──
baseline_smolvla = None
if not args.skip_baseline:
    print(f"\n  Loading baseline from: {args.baseline_checkpoint}")
    try:
        ckpt_bl = _load_checkpoint(args.baseline_checkpoint, DEVICE)
        baseline_smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        baseline_smolvla.load_state_dict(ckpt_bl['smolvla'])
        baseline_smolvla.to(DEVICE)
        baseline_smolvla.eval()
        print("  ✓ Baseline weights loaded")
        del ckpt_bl
        gc.collect()
    except Exception as e:
        print(f"  ✗ Baseline load failed: {e}  (will skip baseline eval)")
        baseline_smolvla = None


# =====================================================================
#  LAYER SKIP HOOKS
# =====================================================================

def apply_layer_skip_hooks(gate_binary_array, num_fixed):
    """Register forward hooks making skipped VLM layers identity pass-throughs."""
    try:
        vlm_layers = flow_model.vlm_with_expert.get_vlm_model().text_model.layers
    except Exception:
        return []
    handles = []
    for i, g in enumerate(gate_binary_array):
        idx = num_fixed + i
        if idx >= len(vlm_layers):
            break
        if float(g) < 0.5:
            def _make_hook():
                def _hook(module, inp, out):
                    h_in = inp[0]
                    if isinstance(out, tuple):
                        return (h_in,) + out[1:]
                    return h_in
                return _hook
            h = vlm_layers[idx].register_forward_hook(_make_hook())
            handles.append(h)
    return handles


# =====================================================================
#  OBSERVATION → BATCH CONVERTER
# =====================================================================

def obs_to_batch(obs, lang_ids, lang_mask, device, img_size=256):
    """Convert LIBERO robosuite obs dict → SmolVLA batch dict.

    LIBERO obs keys (robosuite):
      agentview_image          : (H, W, 3) uint8
      robot0_eye_in_hand_image : (H, W, 3) uint8
      robot0_eef_pos           : (3,) float64
      robot0_eef_quat          : (4,) float64
      robot0_gripper_qpos      : (2,) float64
    """
    batch = {}

    def _process_img(arr):
        """(H, W, C) uint8 → (C, H, W) float32 [0, 1], resized to img_size."""
        if arr is None:
            return torch.zeros(3, img_size, img_size, dtype=torch.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.asarray(arr).astype(np.uint8)
        # robosuite/MuJoCo off-screen render is rotated 180° vs the LeRobot LIBERO
        # dataset frames. Verified empirically by matching the SAME robot config
        # (_probe_sim_vs_data.py): a pure vertical flip [::-1] fixes top/bottom but
        # leaves the scene LEFT-RIGHT MIRRORED (basket on the wrong side) → the
        # policy reaches the wrong way and every rollout fails. The correct
        # transform is a 180° rotation = vertical + horizontal flip [::-1, ::-1].
        # (This matches the OpenVLA/LIBERO convention.)
        if FLIP_IMAGES:
            arr = arr[::-1, ::-1].copy()
        # robosuite images are (H, W, C) uint8
        img = cv2.resize(arr, (img_size, img_size),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

    # Camera images → SmolVLA image keys
    img_cam1 = _process_img(obs.get('agentview_image'))
    img_cam2 = _process_img(obs.get('robot0_eye_in_hand_image'))
    imgs = [img_cam1, img_cam2]

    for i, key in enumerate(smolvla_img_keys):
        if i < len(imgs):
            batch[key] = imgs[i].unsqueeze(0).to(device)

    # Robot state — MUST match the dataset layout the model was trained on.
    # Verified empirically (_probe_state.py): HuggingFaceVLA/libero state is
    #   [eef_pos(3), eef_AXIS_ANGLE(3), gripper_qpos(2)]   (NOT quat, NOT mean!).
    # The old code fed [pos, quat(4), gripper_mean(1)] → proprio dims 3-7 were
    # semantically wrong (e.g. dim3 ≈ quat_x≈0 where the model expects ≈π),
    # so the policy received garbage proprioception at test time.
    from robosuite.utils import transform_utils as _T
    eef_pos    = np.array(obs.get('robot0_eef_pos', [0.0, 0.0, 0.0]), dtype=np.float32)
    eef_quat   = np.array(obs.get('robot0_eef_quat', [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
    axis_angle = np.array(_T.quat2axisangle(eef_quat), dtype=np.float32)        # (3,)
    grip_qpos  = np.array(obs.get('robot0_gripper_qpos', [0.0, 0.0]), dtype=np.float32).reshape(-1)
    if grip_qpos.shape[0] < 2:
        grip_qpos = np.pad(grip_qpos, (0, 2 - grip_qpos.shape[0]))
    grip_qpos  = grip_qpos[:2]
    state_np   = np.concatenate([eef_pos, axis_angle, grip_qpos])   # (8,) = pos3+axisangle3+grip2
    state_np   = _normalize_state(state_np.astype(np.float32))       # MEAN_STD (smolvla_libero)
    batch['observation.state'] = torch.from_numpy(state_np).unsqueeze(0).to(device)

    # Language tokens
    batch['observation.language.tokens'] = lang_ids.to(device)
    batch['observation.language.attention_mask'] = lang_mask.to(device)

    return batch


# =====================================================================
#  GATE COMPUTATION (action_p router)
# =====================================================================

@torch.no_grad()
def compute_gates(batch, prev_state, prev_state2):
    """Compute layer skip gates from the action-aware router.

    Returns:
        gate_binary  : np.ndarray (NUM_SKIP,)
        new_prev     : current state tensor (for next step)
        new_prev2    : previous state tensor (for next step)
    """
    state = batch.get('observation.state')
    if state is None:
        state = torch.zeros(1, 8, device=DEVICE)
    else:
        state = state.float()

    if prev_state is None:
        prev_state  = torch.zeros_like(state)
        prev_state2 = torch.zeros_like(state)

    # Kinematic features
    s_dim = min(state.shape[-1], KIN_DIM)
    s_cur   = state[:, :s_dim]
    s_prv   = prev_state[:, :s_dim]
    s_prv2  = prev_state2[:, :s_dim]

    delta    = s_cur - s_prv
    m_raw    = 1.0 / (delta.norm(dim=-1, keepdim=True) + 1e-6)
    j_raw    = (s_cur - 2.0 * s_prv + s_prv2).norm(dim=-1, keepdim=True)
    m_norm   = torch.tanh(torch.log1p(m_raw * SCALE_M))
    j_norm   = torch.tanh(torch.log1p(j_raw * SCALE_J))
    s_t      = torch.relu(K_LAMBDA * m_norm + (1.0 - K_LAMBDA) * j_norm)
    kin_feat = torch.cat([m_norm, j_norm, s_t], dim=-1)

    # Image embeddings (for visual entropy)
    images, _ = smolvla.prepare_images(batch)
    img_embs  = [flow_model.vlm_with_expert.embed_image(img).float()
                 for img in images]
    # ADP keep-ratio (DTP): velocity-driven, same formula as HierarchicalADPPruner
    v_ee = delta.norm(dim=-1, keepdim=True)
    r_keep = torch.where(
        v_ee > V_THRESH,
        torch.clamp(R_MIN + (1.0 - R_MIN) * V_THRESH / (v_ee + 1e-8), R_MIN, 1.0),
        torch.ones_like(v_ee))
    r_keep_val = float(r_keep.mean().item())

    if not img_embs:
        return np.ones(NUM_SKIP, dtype=np.float32), r_keep_val, state, prev_state

    all_img_emb  = torch.cat(img_embs, dim=1)
    visual_entropy = all_img_emb.var(dim=1).mean(dim=-1, keepdim=True)
    hidden_pooled  = all_img_emb.mean(dim=1)

    expected_dim = action_router.gate_net[0].in_features - 3
    if hidden_pooled.shape[-1] != expected_dim:
        hidden_pooled = F.adaptive_avg_pool1d(
            hidden_pooled.unsqueeze(1), expected_dim).squeeze(1)

    delta_s_norm = delta.norm(dim=-1, keepdim=True)
    gates, _, _, _, _ = action_router(
        hidden_pooled, visual_entropy, delta_s_norm, kin_feat, s_t,
        tau=dyn_cfg.get("gumbel_tau_end", 0.35), hard=True)

    gate_binary = gates.detach().cpu().numpy()[0].astype(np.float32)
    return gate_binary, r_keep_val, state.detach(), prev_state.detach()


# =====================================================================
#  LIBERO ENVIRONMENT SETUP
# =====================================================================
print("\n  Initialising LIBERO benchmark …")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bddl_base = get_libero_path('bddl_files')
bdict = benchmark.get_benchmark_dict()
suite = bdict[SUITE_NAME]()
n_tasks = suite.get_num_tasks()
print(f"  Suite: {SUITE_NAME} — {n_tasks} tasks")

# =====================================================================
#  ROLLOUT FUNCTION
# =====================================================================

def run_rollout(env, lang_ids, lang_mask, init_state, policy_fn, record=False):
    """Run one rollout episode.

    Args:
        env        : OffScreenRenderEnv (already reset)
        lang_ids   : (1, L) tokenized language
        lang_mask  : (1, L) attention mask
        init_state : flat MuJoCo state vector for env.set_init_state()
        policy_fn  : callable(batch) → action np.ndarray (7,)
        record     : if True, collect agentview_image frames and return them

    Returns:
        success (bool), n_steps (int), latencies_ms (list of float)
    """
    # Fresh policy action-queue + chunk counter per episode (so gate/ADP recompute
    # aligns with the model's replan cadence).
    if hasattr(smolvla, "reset"):
        smolvla.reset()
    _reset_dyn()

    # LIBERO standard: reset() to clear the done flag / step counter BEFORE
    # set_init_state. Required when reusing one env across rollouts — otherwise a
    # prior episode leaves env.done=True and the next step() raises.
    env.reset()
    obs = env.set_init_state(init_state)

    # LIBERO standard: step the sim with no-op actions after set_init_state so
    # objects settle and the first real observation is physically valid.
    # Without this the policy acts on a transient/invalid initial frame.
    if SETTLE_STEPS > 0:
        dummy = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(dummy)

    latencies = []
    frames = []
    prev_state = None
    prev_state2 = None
    success = False

    for step in range(HORIZON):
        if record:
            frame = obs.get('agentview_image')
            if frame is not None:
                frames.append(frame.copy())

        batch = obs_to_batch(obs, lang_ids, lang_mask, DEVICE, IMG_SIZE)
        t0 = time.perf_counter()
        action = policy_fn(batch, prev_state, prev_state2)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        # Update prev_state for kinematics (uses observation.state from batch)
        state_cur = batch['observation.state']
        prev_state2 = prev_state if prev_state is not None else torch.zeros_like(state_cur)
        prev_state  = state_cur

        obs, _, done, info = env.step(action)
        if env.check_success():
            success = True
            if record:
                frame = obs.get('agentview_image')
                if frame is not None:
                    frames.append(frame.copy())
            break
        if done:
            break

    return success, step + 1, latencies, frames


def _normalize_action(action):
    """Ensure policy output is exactly ACTION_DIM for env.step()."""
    global _action_shape_warned
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] == ACTION_DIM:
        return arr
    if arr.shape[0] > ACTION_DIM:
        if not _action_shape_warned:
            print(f"    [warn] policy returned {arr.shape[0]}D action; truncating to {ACTION_DIM}D")
            _action_shape_warned = True
        return arr[:ACTION_DIM]
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[:arr.shape[0]] = arr
    if not _action_shape_warned:
        print(f"    [warn] policy returned {arr.shape[0]}D action; padding to {ACTION_DIM}D")
        _action_shape_warned = True
    return out


# =====================================================================
#  POLICY FUNCTIONS
# =====================================================================
_DYN_STEP = [0]   # per-episode call counter (reset in run_rollout)


def _reset_dyn():
    _DYN_STEP[0] = 0


def _patch_adp_eval(r_keep_val):
    """ADP (DTP) token pruning at eval — mirror of training: keep top-K image
    tokens (K=round(keep·N)), zero the rest, via an embed_image monkeypatch.
    Returns a restore() callable."""
    vlm_we = flow_model.vlm_with_expert
    orig = vlm_we.embed_image
    keep = max(0.05, min(1.0, float(r_keep_val)))

    def patched(img):
        emb = orig(img)
        N = emb.shape[1]
        K = max(int(round(keep * N)), 1)
        if K < N:
            m = torch.zeros(1, N, 1, device=emb.device, dtype=emb.dtype)
            m[:, :K, :] = 1.0
            emb = emb * m
        return emb

    vlm_we.embed_image = patched
    return lambda: setattr(vlm_we, "embed_image", orig)


def dynamic_policy(batch, prev_state, prev_state2):
    """Dynamic model. Gates + ADP keep-ratio are recomputed ONLY when the model
    re-plans (every N_ACTION_STEPS, when select_action runs the model). On the
    intervening steps select_action just pops the queue — computing gates there
    wasted ~15ms/step (the latency paradox). Layer-skip hooks + ADP token mask are
    active during the model forward; both realise the trained DTP→DLS cascade.
    """
    handles = []
    adp_restore = None
    replan = (_DYN_STEP[0] % N_ACTION_STEPS == 0)
    _DYN_STEP[0] += 1
    if replan and not (NO_SKIP and NO_ADP):
        gate_binary, r_keep_val, _, _ = compute_gates(batch, prev_state, prev_state2)
        if not NO_SKIP:
            handles = apply_layer_skip_hooks(gate_binary, NUM_FIXED)
        if not NO_ADP:
            adp_restore = _patch_adp_eval(r_keep_val)
    with torch.no_grad():
        pred = smolvla.select_action(batch)
    for h in handles:
        h.remove()
    if adp_restore is not None:
        adp_restore()
    arr = pred.detach().cpu().numpy().squeeze() if isinstance(pred, torch.Tensor) else np.array(pred).squeeze()
    return _normalize_action(_unnormalize_action(arr))   # model outputs normalized action


def baseline_policy(batch, prev_state, prev_state2):
    """Plain SmolVLA baseline policy (no dynamic skipping)."""
    with torch.no_grad():
        pred = baseline_smolvla.select_action(batch)
    arr = pred.detach().cpu().numpy().squeeze() if isinstance(pred, torch.Tensor) else np.array(pred).squeeze()
    return _normalize_action(_unnormalize_action(arr))


# =====================================================================
#  MAIN EVAL LOOP
# =====================================================================

def eval_suite(model_label, policy_fn, reset_fn=None):
    """Run all tasks in the suite with the given policy.

    Returns results dict with per-task success rates and latency stats.
    """
    results = {
        "model": model_label,
        "suite": SUITE_NAME,
        "n_rollouts_per_task": N_ROLLOUTS,
        "per_task": [],
        "overall_success_rate": 0.0,
        "latency_mean_ms": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "latency_max_ms": 0.0,
    }

    all_successes = []
    all_latencies = []

    for task_idx in tqdm(range(n_tasks), desc=f"Tasks [{model_label}]"):
        task  = suite.get_task(task_idx)
        inits = suite.get_task_init_states(task_idx)  # (50, state_dim)

        # Tokenize this task's language instruction
        # libero Task has .language attribute (or fallback to task name)
        instr = getattr(task, 'language',
                        getattr(task, 'language_instruction',
                                getattr(task, 'task_description',
                                        task.name.lower().replace('_', ' '))))
        tok   = tokenizer(instr, return_tensors="pt", padding="max_length",
                          max_length=64, truncation=True)
        lang_ids  = tok['input_ids']
        lang_mask = tok['attention_mask'].bool()

        # BDDL file full path
        bddl_suite = SUITE_NAME.replace("libero_", "libero_")
        bddl_path  = os.path.join(bddl_base, bddl_suite, task.bddl_file)
        if not os.path.exists(bddl_path):
            # Try without subdirectory
            bddl_path = os.path.join(bddl_base, task.bddl_file)
        if not os.path.exists(bddl_path):
            print(f"\n  WARNING: BDDL not found: {bddl_path} — skipping task {task_idx}")
            results["per_task"].append({
                "task_idx": task_idx, "task_name": task.name,
                "success_rate": 0.0, "n_success": 0,
                "instruction": instr, "bddl_missing": True,
            })
            continue

        # Create env for this task
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="agentview",
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=IMG_SIZE,
            camera_widths=IMG_SIZE,
            control_freq=20,
            # horizon padded for the settle steps; ignore_done=True so the env
            # never auto-terminates on horizon (which raised "executing action in
            # terminated episode" once settle steps pushed total past horizon).
            # We control episode length via the HORIZON loop + check_success().
            horizon=HORIZON + SETTLE_STEPS + 10,
            ignore_done=True,
        )

        # Reset each episode's policy state
        if reset_fn is not None:
            reset_fn()
        if hasattr(smolvla, 'reset'):
            smolvla.reset()

        task_successes = 0
        task_latencies = []

        # Video: record every rollout; at end save the first SUCCESS (or fallback
        # to rollout 0 if all fail).  LIBERO video_utils.py flips with [::-1] for
        # human-readable orientation — we do the same with np.flipud().
        best_frames  = None   # frames of first success (or rollout 0 if no success)
        best_success = False  # whether best_frames is a success rollout

        for rollout_i in range(N_ROLLOUTS):
            # Cycle through all 50 pre-sampled init states (LIBERO official protocol).
            # These states are independent from HDF5 demo init states — no train/test
            # overlap concern.  Standard: 20 rollouts, cycling inits[0..19].
            init_state = inits[rollout_i % len(inits)]

            do_record = args.save_video and (not best_success)

            try:
                success, n_steps, lats, frames = run_rollout(
                    env, lang_ids, lang_mask, init_state, policy_fn,
                    record=do_record)
                if success:
                    task_successes += 1
                task_latencies.extend(lats)

                if do_record and frames:
                    if success and not best_success:
                        # First success — this is the best video
                        best_frames  = frames
                        best_success = True
                    elif best_frames is None:
                        # No success yet — keep rollout 0 as fallback
                        best_frames = frames
            except Exception as e:
                print(f"\n    Rollout {rollout_i} task {task_idx} failed: {e}")

            # Reset policy context each episode
            if hasattr(smolvla, 'reset'):
                smolvla.reset()

        # Save best video after all rollouts for this task
        if args.save_video and best_frames:
            try:
                import imageio
                vid_dir = OUTPUT_DIR / "videos"
                vid_dir.mkdir(parents=True, exist_ok=True)
                tag = "success" if best_success else "fail"
                vid_path = vid_dir / f"task{task_idx:02d}_{task.name[:40]}_{tag}.mp4"
                imageio.mimwrite(
                    str(vid_path),
                    [np.flipud(f) for f in best_frames],
                    fps=args.video_fps,
                    macro_block_size=None,
                )
                print(f"    [video] saved → {vid_path.name}  ({len(best_frames)} frames, {tag})")
            except Exception as ve:
                print(f"    [video] save failed: {ve}")

        env.close()
        del env
        gc.collect()

        sr = task_successes / N_ROLLOUTS
        all_successes.append(sr)
        all_latencies.extend(task_latencies)

        print(f"  Task {task_idx:2d} [{task.name[:50]}]: "
              f"{task_successes}/{N_ROLLOUTS}  ({sr*100:.0f}%)")

        results["per_task"].append({
            "task_idx":       task_idx,
            "task_name":      task.name,
            "instruction":    instr,
            "n_success":      task_successes,
            "n_rollouts":     N_ROLLOUTS,
            "success_rate":   sr,
            "latency_mean_ms": float(np.mean(task_latencies)) if task_latencies else 0.0,
        })

    # Aggregate
    if all_successes:
        results["overall_success_rate"] = float(np.mean(all_successes))
    if all_latencies:
        results["latency_mean_ms"] = float(np.mean(all_latencies))
        results["latency_p50_ms"]  = float(np.percentile(all_latencies, 50))
        results["latency_p95_ms"]  = float(np.percentile(all_latencies, 95))
        results["latency_max_ms"]  = float(np.max(all_latencies))

    return results


# ─── Run evaluations ───────────────────────────────────────────────
all_results = {}

if not args.skip_dynamic:
    print("\n" + "=" * 70)
    print("  Running DYNAMIC model evaluation …")
    print("=" * 70)
    dyn_results = eval_suite("hierarchical_action_aware_dynamic", dynamic_policy)
    all_results["dynamic"] = dyn_results

    print(f"\n  Dynamic Model — Overall Success Rate: "
          f"{dyn_results['overall_success_rate']*100:.1f}%")
    print(f"  Latency (mean/p50/p95/max): "
          f"{dyn_results['latency_mean_ms']:.1f} / "
          f"{dyn_results['latency_p50_ms']:.1f} / "
          f"{dyn_results['latency_p95_ms']:.1f} / "
          f"{dyn_results['latency_max_ms']:.1f} ms")

if not args.skip_baseline and baseline_smolvla is not None:
    print("\n" + "=" * 70)
    print("  Running BASELINE model evaluation …")
    print("=" * 70)
    if hasattr(baseline_smolvla, 'reset'):
        baseline_smolvla.reset()
    bl_results = eval_suite("smolvla_baseline", baseline_policy)
    all_results["baseline"] = bl_results

    print(f"\n  Baseline — Overall Success Rate: "
          f"{bl_results['overall_success_rate']*100:.1f}%")
    print(f"  Latency (mean/p50/p95/max): "
          f"{bl_results['latency_mean_ms']:.1f} / "
          f"{bl_results['latency_p50_ms']:.1f} / "
          f"{bl_results['latency_p95_ms']:.1f} / "
          f"{bl_results['latency_max_ms']:.1f} ms")

# ── Comparison summary ──
if "dynamic" in all_results and "baseline" in all_results:
    d_sr = all_results["dynamic"]["overall_success_rate"]
    b_sr = all_results["baseline"]["overall_success_rate"]
    d_lat = all_results["dynamic"]["latency_p50_ms"]
    b_lat = all_results["baseline"]["latency_p50_ms"]
    speedup = b_lat / max(d_lat, 0.1)

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  Dynamic  success rate: {d_sr*100:.1f}%  latency p50: {d_lat:.1f}ms")
    print(f"  Baseline success rate: {b_sr*100:.1f}%  latency p50: {b_lat:.1f}ms")
    print(f"  Success Δ: {(d_sr-b_sr)*100:+.1f} pp")
    print(f"  Speedup (p50): {speedup:.2f}x  ({'FASTER' if speedup > 1 else 'SLOWER'})")

    all_results["comparison"] = {
        "dynamic_success_rate":  d_sr,
        "baseline_success_rate": b_sr,
        "success_delta_pp":      (d_sr - b_sr) * 100,
        "dynamic_latency_p50_ms":  d_lat,
        "baseline_latency_p50_ms": b_lat,
        "speedup_p50":             speedup,
    }

# ── Save results ──
out_file = OUTPUT_DIR / f"libero_rollout_{SUITE_NAME}.json"
with open(out_file, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results saved → {out_file}")
