"""
Test SmolVLA pretrained model loading.
Strategy: Patch lerobot.robots.__init__ to skip importing Robot class
(which triggers the transformers/GR00T crash chain).
Only RobotConfig is needed by the rest of the import chain.
"""
import sys
import types

# ─── Step 1: Pre-load lerobot.robots with only RobotConfig (skip Robot) ───
import lerobot
pkg = lerobot.__path__[0]

# Create a minimal robots module that only exports RobotConfig
robots_mod = types.ModuleType('lerobot.robots')
robots_mod.__path__ = [f'{pkg}\\robots']
robots_mod.__package__ = 'lerobot.robots'

# Load RobotConfig directly from config.py (it has no bad deps)
import importlib.util
spec = importlib.util.spec_from_file_location(
    'lerobot.robots.config', f'{pkg}\\robots\\config.py')
config_mod = importlib.util.module_from_spec(spec)
sys.modules['lerobot.robots.config'] = config_mod
spec.loader.exec_module(config_mod)

robots_mod.RobotConfig = config_mod.RobotConfig
sys.modules['lerobot.robots'] = robots_mod

# ─── Step 2: Stub lerobot.processor to avoid transformers chain ───
proc_mod = types.ModuleType('lerobot.processor')
proc_mod.__path__ = [f'{pkg}\\processor']
proc_mod.__package__ = 'lerobot.processor'
proc_mod.RobotAction = dict  # Dummy
proc_mod.RobotObservation = dict  # Dummy
proc_mod.PolicyAction = dict  # Dummy
sys.modules['lerobot.processor'] = proc_mod

# ─── Step 3: Stub lerobot.policies to avoid GR00T __init__ ───
policies_mod = types.ModuleType('lerobot.policies')
policies_mod.__path__ = [f'{pkg}\\policies']
policies_mod.__package__ = 'lerobot.policies'
sys.modules['lerobot.policies'] = policies_mod

print("[PATCH] Import chain patched OK")

# ─── Step 4: Now import SmolVLA ───
print("[LOAD] Importing SmolVLAPolicy...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print("[LOAD] SmolVLAPolicy imported OK!")

# ─── Step 5: Load pretrained model ───
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[LOAD] Loading lerobot/smolvla_base on {device}...")
print("       (This downloads ~900MB weights on first run)")

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
print("[LOAD] Model loaded successfully!")

total = sum(p.numel() for p in policy.parameters())
print(f"[INFO] Total params: {total / 1e6:.1f}M")
print(f"[INFO] Device: {next(policy.parameters()).device}")
print("")
print("[SUCCESS] SmolVLA pretrained model is ready!")
