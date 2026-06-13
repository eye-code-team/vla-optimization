"""
Scalable Fine-tune Config for SmolVLA vs Baseline Benchmarks.
Add new datasets as dict entries in DATASETS to scale experiments.
"""

DATASETS = {
    "svla_so100_pickplace": {
        "repo_id": "lerobot/svla_so100_pickplace",
        "task_instruction": "Pick up the cube and place it in the bin.",
        "train_episodes": 40,   # first N episodes for training
        "eval_episodes": 10,    # last N episodes for evaluation
    },
    # Future datasets — just add entries:
    # "so100_lego": {
    #     "repo_id": "lerobot/so100_lego_task",
    #     "task_instruction": "Place the lego on the red box.",
    #     "train_episodes": 35,
    #     "eval_episodes": 10,
    # },
    "so100_lego": {
        "repo_id": "lerobot/so100_lego_task",
        "task_instruction": "Place the lego on the red box.",
        "train_episodes": 60,
        "eval_episodes": 20,
    },
    "pusht": {
        "repo_id": "lerobot/pusht",
        "task_instruction": "Push the T-shaped block to the target zone.",
        "train_episodes": 40,
        "eval_episodes": 10,
    },
    "aloha_mobile_cabinet": {
        "repo_id": "lerobot/aloha_mobile_cabinet",
        "task_instruction": "Open the cabinet.",
        "train_episodes": 40,
        "eval_episodes": 10,
    },
    "xarm_lift_medium": {
        "repo_id": "lerobot/xarm_lift_medium",
        "task_instruction": "Lift the block.",
        "train_episodes": 40,
        "eval_episodes": 10,
    },
    "xarm_push_medium": {
        "repo_id": "lerobot/xarm_push_medium",
        "task_instruction": "Push the block.",
        "train_episodes": 40,
        "eval_episodes": 10,
    },
    "aloha_static_tape": {
        "repo_id": "lerobot/aloha_static_tape",
        "task_instruction": "Grasp the tape.",
        "train_episodes": 40,
        "eval_episodes": 10,
    },
    "aloha_sim_transfer_cube": {
        "repo_id": "lerobot/aloha_sim_transfer_cube_human",
        "task_instruction": "Transfer the cube between the arms.",
        "train_episodes": 40,
        "eval_episodes": 10,
    },
    "aloha_static_battery": {
        "repo_id": "lerobot/aloha_static_battery",
        "task_instruction": "Pick up the battery.",
        "train_episodes": 40,
        "eval_episodes": 10,
    },
    # ── LIBERO (LeRobot-format, HuggingFaceVLA/libero) ──────────────────────
    # Run download_libero.py first to cache dataset and populate episode list.
    # action_dim=7 (6D eef delta + gripper), state_dim=8 (pos+ori+gripper),
    # two cameras: observation.images.image + observation.images.image2
    "libero_spatial": {
        "repo_id": "HuggingFaceVLA/libero",
        "root": "data/datasets/HuggingFaceVLA/libero",
        # libero_spatial == dataset task_index 30-39 (resolved by instruction-string
        # match in build_libero_splits). 432 episodes. Auto-loaded from
        # libero_spatial_train_episodes.json. Evaluation uses LIBERO rollout sim.
        "episodes": None,
        "task_instruction": "Pick up the black bowl and place it on the plate.",
        "train_episodes": 432,
        "eval_episodes":  0,
    },
    "libero_object": {
        "repo_id": "HuggingFaceVLA/libero",
        "root": "data/datasets/HuggingFaceVLA/libero",
        "episodes": None,
        "task_instruction": "Perform the object manipulation task.",
        "train_episodes": 400,
        "eval_episodes":  100,
    },
    "libero_goal": {
        "repo_id": "HuggingFaceVLA/libero",
        "root": "data/datasets/HuggingFaceVLA/libero",
        "episodes": None,
        "task_instruction": "Reach the goal configuration.",
        "train_episodes": 400,
        "eval_episodes":  100,
    },
    "libero_10": {
        "repo_id": "HuggingFaceVLA/libero",
        "root": "data/datasets/HuggingFaceVLA/libero",
        "episodes": None,   # auto-loaded from libero_10_train_episodes.json
        "task_instruction": "Complete the long-horizon manipulation task.",
        # libero_10 == dataset task_index 0-9 (resolved by instruction-string match
        # in build_libero10_splits.py). 379 episodes total, 10 tasks × 29–49 demos.
        # NOTE: task_index 30-39 is libero_SPATIAL, NOT libero_10 — do not use.
        "train_episodes": 329,
        "eval_episodes":  50,
    },
    "libero_10_full": {
        # Original LIBERO-10 HDF5 converted to LeRobot format via
        # convert_libero_hdf5_to_lerobot.py (yifengzhu-hf/LIBERO-datasets).
        # FULL dataset: exactly 50 demos/task × 10 tasks = 500 episodes.
        # Split: first 45 → train (450), last 5 → test (50) per task.
        "repo_id": None,    # local only
        "root": "data/datasets/libero_10_full",
        "episodes": None,   # auto-loaded from libero_10_full_train_episodes.json
        "task_instruction": "Complete the long-horizon manipulation task.",
        "train_episodes": 450,
        "eval_episodes":  50,
    },
}

TRAINING = {
    "smolvla": {
        "steps": 2000,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "micro_batch": 2,
        "grad_accum": 8,        # effective batch = 2*8 = 16
        "freeze_vlm": True,     # Only train action expert (~50M)
        "fp16": True,           # Mixed precision for 12GB VRAM
        "max_grad_norm": 1.0,
        "save_every": 500,
    },
    "act": {
        "steps": 2000,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "micro_batch": 8,
        "grad_accum": 2,        # effective batch = 8*2 = 16
        "fp16": True,
        "max_grad_norm": 10.0,
        "save_every": 500,
    },
}

TRAINING["dynamic_lora_pruning_snapflow"] = {
    # Phase 1: Gating & Skip Layer (DySL)
    "phase1_steps": 2000,
    "phase1_lr": 1e-4,
    # Phase 2: LoRA-SP + ADP
    "phase2_steps": 1000,
    "phase2_lr": 1e-4,
    "lora_max_rank": 128,
    "lora_energy_threshold": 0.9,
    "adp_velocity_threshold": 0.15,
    "adp_min_keep_ratio": 0.3,
    # Phase 3: SnapFlow
    "phase3_steps": 1000,
    "phase3_lr": 5e-5,
    "snap_teacher_steps": 10,
    # Shared
    "micro_batch": 2,
    "grad_accum": 8,
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 1000,
    # Loss weights
    "lambda_spec": 0.01,
    "lambda_cost": 0.001,
    # DySL
    "num_fixed_layers": 8,
    "num_skippable_layers": 8,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.1,
    "freeze_vlm": True,
}

TRAINING["dynamic_P_loss_cogkd"] = {
    # Phase 1: Dynamic P gating
    "phase1_steps": 1000,
    "phase1_lr": 1e-4,

    # Phase 2: Joint optimization
    "phase2_steps": 500,
    "phase2_lr": 1e-4,

    # Phase 3: CogKD distillation
    "phase3_steps": 500,
    "phase3_lr": 5e-5,

    # Shared optimization
    "micro_batch": 2,
    "grad_accum": 8,
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 1000,
    "log_every": 100,

    # Dynamic layer skipping
    "num_fixed_layers": 8,
    "num_skippable_layers": 8,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.1,
    "freeze_vlm": True,
    "router_init_gain": 3.0,
    "router_gate_bias_init": 0.0,      # start at max-uncertainty (was 3.0)

    # Action-aware P budget
    "rho_target_min": 0.0,
    "rho_target_max": 0.5,
    "kinematic_state_dim": 16,
    "kinematic_lambda": 0.6,

    # Loss weights — core
    "lambda_gate": 0.001,
    "lambda_overbudget": 0.05,
    "lambda_spec": 0.01,
    "lambda_cogkd": 0.2,

    # Budget loss ramp: small early (task dominates) → large late (drives gate variation)
    "lambda_budget_start": 0.005,
    "lambda_budget_end": 0.10,
    "lambda_budget": 0.01,             # used in Phase 2/3 (no ramp)

    # New diversity loss weights
    "lambda_temporal_entropy": 0.1,    # L_temporal_entropy: force each layer to toggle ~50%
    "lambda_rho_spread": 0.05,         # L_rho_spread: rho_target must vary across batch
    "lambda_gate_flip": 0.1,           # L_gate_flip: reward flipping gates vs prev step

    # Minimum std dev of rho_target across batch before rho_spread penalty kicks in
    "min_rho_std": 0.05,

    # Sequential batch window for meaningful kinematic features
    "sequential_window": 8,

    # CogKD
    "cogkd_lambda": 0.3,
    "cogkd_temperature": 2.0,
    "toi_ratio": 0.3,
    "toi_min_tokens": 4,
}

TRAINING["dynamic_layerskip_snapflow_only"] = {
    # Phase 1: Dynamic layer skipping
    "phase1_steps": 5000,
    "phase1_lr": 1e-4,

    # Phase 2: SnapFlow with dynamic skipping enabled
    "phase2_steps": 5000,
    "phase2_lr": 5e-5,
    "snap_teacher_steps": 10,

    # Shared optimization
    "micro_batch": 2,
    "grad_accum": 8,
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 1000,
    "log_every": 100,

    # Loss weights
    "lambda_gate": 0.001,
    "lambda_snap": 0.5,

    # Dynamic layer skipping settings
    "num_fixed_layers": 8,
    "num_skippable_layers": 8,
    "skip_keep_ratio": 0.5,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.1,

    # Freeze full VLM backbone and train non-VLM + router
    "freeze_vlm": True,
}

TRAINING["dysl_core_snapflow"] = {
    # Phase 1: DySL core training (dynamic-static + prior/post guidance)
    "phase1_steps": 5000,
    "phase1_lr": 1e-4,

    # Phase 2: SnapFlow with DySL core enabled
    "phase2_steps": 5000,
    "phase2_lr": 5e-5,
    "snap_teacher_steps": 10,

    # Shared optimization
    "micro_batch": 2,
    "grad_accum": 8,
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 1000,
    "log_every": 100,

    # Loss weights
    "lambda_gate": 0.001,
    "lambda_snap": 0.5,

    # Dynamic-static layer setup
    "static_layer_ratio": 0.25,
    "num_anchor_front_layers": 2,
    "manual_static_layer_ids": [],
    "calibration_batches": 12,
    "calibration_batch": 2,

    # Prior guidance (pre-skip)
    "continuity_k": 5,
    "continuity_eta": 1e-3,
    "gate_threshold": 0.5,

    # Post guidance (verification)
    "post_verify_enabled": True,
    "post_verify_eta": 5e-4,
    "post_verify_cooldown": 2,

    # Router temperature
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.1,

    # Freeze full VLM backbone and train non-VLM + router + adapters
    "freeze_vlm": True,
}

TRAINING["deer_vlap_dora_a2a_snap"] = {
    # Phase 1: DeeR-VLA Multi-Exit training
    "phase1_steps": 5000,
    "phase1_lr": 1e-4,
    "num_exits": 4,                    # Exit points at layers 4, 8, 12, 16
    "exit_threshold_start": 0.5,       # Anneal from → exit_threshold_end
    "exit_threshold_end": 0.85,        # Action consistency threshold
    "exit_loss_weights": [0.1, 0.2, 0.3, 0.4],

    # Phase 2: VLA-Pruner + DoRA
    "phase2_steps": 8000,
    "phase2_lr": 1e-4,
    "pruner_semantic_weight": 0.4,
    "pruner_action_weight": 0.6,
    "pruner_temporal_momentum": 0.7,
    "pruner_min_keep_ratio": 0.25,
    "dora_rank": 64,
    "dora_alpha": 1.0,
    "dora_dropout": 0.05,

    # Phase 3: A2A Flow Matching
    "phase3_steps": 5000,
    "phase3_lr": 8e-5,
    "a2a_history_len": 5,
    "a2a_latent_dim": 128,

    # Phase 4: SnapFlow 1-NFE
    "phase4_steps": 5000,
    "phase4_lr": 5e-5,
    "snap_teacher_steps": 10,

    # Phase 5: RS-CL + Polish
    "phase5_steps": 3000,
    "phase5_lr": 3e-5,
    "rscl_temperature": 0.07,
    "rscl_lambda": 0.1,

    # Shared
    "micro_batch": 2,
    "grad_accum": 8,
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 1000,
    "freeze_vlm": True,

    # Loss weights
    "lambda_exit": 0.1,
    "lambda_prune": 0.001,
    "lambda_rscl": 0.1,

    # Architecture (SmolLM2)
    "num_fixed_layers": 4,       # First 4 layers always execute (no exit)
    "num_vlm_layers": 16,        # Total VLM layers
}

TRAINING["vlaiap_coral_snapflow"] = {
    # Phase 1: Hierarchical Gating + CORAL Expert routing
    "phase1_steps": 5000,
    "phase1_lr": 1e-4,
    # Phase 2: LoRA-SP + VLA-IAP
    "phase2_steps": 10000,
    "phase2_lr": 1e-4,
    "lora_max_rank": 128,
    "lora_energy_threshold": 0.9,
    # VLA-IAP params
    "iap_conservative_ratio": 0.8,
    "iap_aggressive_ratio": 0.3,
    "iap_iou_threshold": 0.5,
    "iap_temporal_momentum": 0.7,
    "iap_geometric_anchor_ratio": 0.1,
    # Phase 3: SnapFlow + CogKD
    "phase3_steps": 5000,
    "phase3_lr": 5e-5,
    "snap_teacher_steps": 10,
    "cogkd_lambda": 0.3,
    "cogkd_temperature": 2.0,
    # Shared
    "micro_batch": 2,
    "grad_accum": 8,
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 1000,
    # Loss weights
    "lambda_spec": 0.01,
    "lambda_cost": 0.001,
    "lambda_diversity": 0.005,
    "lambda_token_action_align": 0.01,
    # DySL — Hierarchical Groups
    "num_fixed_layers": 8,         # L0-L7 Foundation (never skip)
    "num_spatial_layers": 4,       # L8-L11 Spatial Awareness
    "num_action_layers": 4,        # L12-L15 Action Refinement
    "num_skippable_layers": 8,     # total skippable = spatial + action
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.1,
    "freeze_vlm": True,
    # CORAL Experts
    "coral_num_experts": 4,
    "coral_expert_rank": 32,
    # SnapFlow feedback
    "snap_mse_feedback_threshold": 1000.0,
    "snap_feedback_alpha": 0.1,
}

TRAINING["lyapunov_tokenbottleneck_snapflow02"] = {
    # Phase 1: Lyapunov-stable gating + diversity
    "phase1_steps": 5000,
    "phase1_lr": 1e-4,

    # Phase 2: VTB pruning + uncertainty-aware compute
    "phase2_steps": 10000,
    "phase2_lr": 1e-4,

    # Phase 3: SnapFlow2 + CogKD
    "phase3_steps": 5000,
    "phase3_lr": 5e-5,
    "snap_teacher_steps": 10,
    "snap_curvature_delta": 0.08,
    "cogkd_temperature": 2.0,

    # Shared optimization
    "micro_batch": 2,
    "grad_accum": 8,
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 1000,
    "freeze_vlm": True,

    # Architecture
    "num_fixed_layers": 8,
    "num_skippable_layers": 8,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.1,

    # LoRA-SP
    "lora_max_rank": 128,
    "lora_energy_threshold": 0.9,

    # VTB pruning
    "vtb_conservative_ratio": 0.8,
    "vtb_aggressive_ratio": 0.3,
    "interaction_lock_entropy_tau": 1.6,
    "vtb_temporal_gamma": 0.7,
    "vtb_semantic_weight": 0.65,
    "vtb_structural_weight": 0.35,
    "target_uncertainty": 0.25,

    # Loss weights
    "lambda_spec": 0.01,
    "lambda_cost": 0.001,
    "lambda_lyapunov": 0.01,
    "lambda_diversity": 0.005,
    "lambda_ib": 0.01,
    "lambda_uncertainty": 0.01,
    "lambda_curvature": 0.05,
    "lambda_isokinetic": 0.02,
    "lambda_cogkd": 0.2,
}

TRAINING["semifinetune_entropy_cim"] = {
    # Phase 1: Entropy-guided semi-finetuning (EGSF)
    "phase1_steps": 5000,
    "phase1_lr": 1e-4,

    # Phase 2: Contextual Interaction Masking (CIM)
    "phase2_steps": 10000,
    "phase2_lr": 1e-4,

    # Phase 3: SnapFlow alignment + stabilization
    "phase3_steps": 5000,
    "phase3_lr": 5e-5,
    "snap_teacher_steps": 10,

    # Shared optimization
    "micro_batch": 2,
    "grad_accum": 8,
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 1000,
    "freeze_vlm": True,

    # Architecture
    "num_fixed_layers": 8,
    "num_spatial_layers": 4,
    "num_action_layers": 4,
    "num_skippable_layers": 8,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.1,

    # LoRA-SP
    "lora_max_rank": 128,
    "lora_energy_threshold": 0.9,

    # Compatibility aliases for modules inherited from previous dynamic scaffold
    "iap_conservative_ratio": 0.82,
    "iap_aggressive_ratio": 0.30,
    "iap_iou_threshold": 0.50,
    "iap_temporal_momentum": 0.70,
    "iap_geometric_anchor_ratio": 0.10,
    "coral_num_experts": 4,
    "coral_expert_rank": 32,

    # EGSF complexity score
    "complexity_alpha_entropy": 0.40,
    "complexity_alpha_iou_proxy": 0.35,
    "complexity_alpha_grad_proxy": 0.25,

    # CIM thresholds / temporal smoothing
    "cim_conservative_ratio": 0.82,
    "cim_aggressive_ratio": 0.30,
    "cim_temporal_gamma": 0.70,
    "cim_interaction_entropy_tau": 1.60,
    "cim_iou_proxy_tau": 0.50,

    # Hybrid MSE thresholding (fixed then adaptive)
    "hybrid_tau_warmup_steps": 1000,
    "hybrid_tau_fixed": 1000.0,
    "hybrid_tau_percentile": 85.0,
    "hybrid_tau_window": 128,

    # Hard-sample curriculum
    "hard_sample_manifest_path": "d:/EyetechCode/results/dynamic_lora_pruning_snapflow/hard_samples_manifest.json",
    "hard_sample_weight_max": 2.5,
    "hard_sample_weight_default": 1.0,

    # Loss weights
    "lambda_spec": 0.01,
    "lambda_cost": 0.001,
    "lambda_diversity": 0.005,
    "lambda_ib": 0.01,
    "lambda_token_action_align": 0.01,
    "lambda_cim_align": 0.01,
    "lambda_cogkd": 0.2,

    # SnapFlow feedback knobs
    "snap_mse_feedback_threshold": 1000.0,
    "snap_feedback_alpha": 0.1,
    "cogkd_lambda": 0.2,
    "cogkd_temperature": 2.0,
}

TRAINING["hierarchical_action_aware"] = {
    # ------------------------------------------------------------------ #
    # Hierarchical Action-Aware Optimization                              #
    # DTP (token pruning) → DLS (layer skip) cascade  +  CogKD + Snap   #
    # ------------------------------------------------------------------ #

    # Phase 1: CascadeSTAR router + ADP initialisation
    # 10 epochs × 3,310 steps/epoch (432 eps, 52,970 frames, batch=16) = 33,100 total
    # Phase split 40% / 30% / 30%  →  13,240 / 9,930 / 9,930  (rounded below)
    "phase1_steps": 13000,
    "phase1_lr": 1e-4,

    # Phase 2: Joint optimisation — add LoRA-SP + budget coupling
    "phase2_steps": 10000,
    "phase2_lr": 1e-4,

    # Phase 3: CogKD (ToI-masked) + SnapFlow self-distillation
    "phase3_steps": 10000,
    "phase3_lr": 5e-5,
    "snap_teacher_steps": 10,

    # Shared optimisation
    "micro_batch": 8,   # RTX 4090 24GB bfloat16; drop to 4 if OOM on smaller GPU
    "grad_accum": 2,   # effective batch = micro_batch × grad_accum = 16
    "fp16": True,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "save_every": 2000,
    "log_every": 200,

    # Architecture (SmolLM2 — 16 VLM layers)
    "num_fixed_layers": 4
    ,         # L0-L7: always execute
    "num_skippable_layers": 12,     # L8-L15: CascadeSTAR can skip
    "freeze_vlm": True,

    # Gumbel temperature annealing
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.35,          # raised from 0.1 → prevents gradient vanishing at near-binary gates

    # ADP (Action-aware Dynamic Token Pruning)
    # LIBERO sequential frame velocity norm ~ 0.008–0.037 at 10 FPS (8D state).
    # Threshold lowered to 0.015 → triggers at ~60% of motion frames.
    # min_keep_ratio lowered to 0.20 → at peak velocity, 80% of tokens pruned.
    "adp_velocity_threshold": 0.015,
    "adp_min_keep_ratio": 0.20,

    # GPU wave-alignment for real latency gains (RTX 4070 SUPER → wave=32)
    "gpu_wave_size": 32,

    # DTP → DLS budget coupling (kept for Phase 2/3 reference)
    "budget_coupling_alpha": 0.70,

    # Upgraded CascadeSTARRouter: per-layer kinematic thresholds + dynamic rho
    # Target p-ratio = 0.7 (70% of skippable layers skipped); rho ramps 0 → 0.55
    # rho_target = fraction of skippable layers that are ACTIVE (not skipped)
    # Slow/critical motion: s_t≈1 → rho≈0.55 → 45% skip (need more layers)
    # Fast/transit motion:  s_t≈0 → rho≈0.15 → 85% skip (can afford to skip)
    "rho_target_min": 0.25,          # minimum fraction active — raised so budget_loss never targets near-zero
    "rho_target_max": 0.72,          # maximum fraction active — wider range for more dynamic variation
    "kinematic_state_dim": 16,       # how many state dims to use for kinematics
    "kinematic_lambda": 0.6,         # blend weight: λ*m_norm + (1-λ)*j_norm
    "router_gate_bias_init": 2.0,    # σ(2.0)=0.88 → start mostly ON; creates room to vary downward
    "router_init_gain": 1.0,         # neutral init gain

    # Sequential batch sampling: meaningful velocity / jerk features
    "sequential_window": 8,          # consecutive frames per training batch

    # Skip floor/ceiling: SOFT guards only — primary skip pressure comes from budget_loss.
    # lambda_skip_floor was 12.0 which dominated all other losses → static gate collapse.
    # Reduced to soft nudges; budget_loss (lambda_budget=5.0) is now the primary driver.
    "skip_target_ratio": 0.60,       # floor: >= 60% layers skipped per step (gentler target)
    "lambda_skip_floor": 0.0,        # DISABLED: was primary collapse driver; budget_loss handles skip pressure

    # Per-layer always-ON ceiling: each layer may be ON at most max_layer_on_rate
    "max_layer_on_rate": 0.80,       # allow up to 80% ON for complex motion phases
    "lambda_always_on": 0.2,         # minimal ceiling; anti-collapse constraints now handle the floor

    # Anti-collapse constraints: MUST dominate collapse forces
    # At all-skip: skip_ceiling gradient ≈ 6.0*(1.0-0.78)^2*2 ≈ 0.58 vs budget ≈ 2.0*(0-0.25)^2*2 ≈ 0.25
    # layer_min_on: 8.0*(0-0.15)^2*2 ≈ 0.36 per layer (×12 = 4.3 total). Dominates budget.
    "max_skip_ratio": 0.78,          # at most 78% skip → forces ≥22% active (2.6/12 layers)
    "lambda_skip_ceiling": 6.0,      # STRONG ceiling: prevents all-skip collapse
    "min_layer_on_rate": 0.15,       # each layer active >= 15% (raised from 8%)
    "lambda_min_layer_on": 8.0,      # DOMINANT anti-collapse: 8.0 >> budget=2.0

    # NEW: rho_supervision — explicitly trains rho_net to be kinematic-sensitive.
    # This is THE key fix: rho_net is now directly supervised by s_t signal,
    # so rho_target varies per-sample → budget_loss forces state-dependent gates.
    "lambda_rho_supervision": 3.0,   # reduced: kinematic scales are now better calibrated

    # Action-aware gate diversity constraints
    "min_gate_var": 0.08,            # per-layer gate variance across batch
    "lambda_gate_diversity": 7.0,   # STRONG: force different decisions per sample in each batch
    "lambda_visual_coupling": 2.0,  # visual entropy drives gate activation

    # Diversity loss weights
    "lambda_temporal_entropy": 4.0, # STRONG: force gates to flip between steps
    "lambda_rho_spread": 2.0,       # force rho_target to vary across batch
    "lambda_gate_flip": 0.5,
    "min_rho_std": 0.12,            # raised: require more rho variation

    # Kinematic normalization (log1p + tanh, replaces batch-norm)
    # Recalibrated for LIBERO velocity range (delta_s ≈ 0.008–0.037):
    #   scale_m=5.0 saturates m_norm=1.0 for ALL LIBERO frames → constant kin_features
    #   scale_m=0.015: m_norm(0.008)≈0.86, m_norm(0.037)≈0.41 → 0.45 discriminative range
    # Similarly scale_j=10.0 gives j_norm≈0 for small LIBERO jerks; scale_j=80.0 spreads [0.1,0.7]
    "kinematic_scale_m": 0.015,
    "kinematic_scale_j": 80.0,

    # Lambda budget ramp: gates learn to match kinematic-conditioned rho_target.
    # REDUCED from 5.0 to prevent budget_loss from dominating task_loss (5:1 ratio caused instability).
    # Anti-collapse constraints now hold the floor; budget drives the DYNAMIC variation.
    "lambda_budget_start": 1.0,      # gentle ramp start
    "lambda_budget_end": 3.0,        # moderate ramp end

    # LoRA-SP
    "lora_max_rank": 128,
    "lora_energy_threshold": 0.9,

    # CogKD (ToI-masked, uses frozen smolvla_base as permanent teacher)
    "cogkd_lambda": 0.3,
    "cogkd_temperature": 2.0,
    "toi_ratio": 0.30,
    "toi_min_tokens": 4,

    # Unified loss weights
    "lambda_task": 1.0,
    "lambda_gate": 0.3,        # light cost pressure (anti-collapse and budget now handle structure)
    "lambda_budget": 2.0,      # REDUCED from 5.0 → stable; anti-collapse constraints prevent gate_ratio=0
    "lambda_spec": 0.01,
    "lambda_distill": 0.2,
    "lambda_snap": 0.5,
    "lambda_cost": 0.001,
}

EVAL = {
    "num_eval_episodes": 5,     # Evaluate on last 5 of eval episodes
    # Proxy success criterion: episode is "success" if its MSE is below this threshold.
    # Scale reference for LIBERO (action in [-1,1]^7):
    #   random policy  ≈ 0.33  (uniform [-1,1] variance)
    #   undertrained   ≈ 0.10–0.20  (few hundred steps)
    #   well-trained   ≈ 0.02–0.05  (full training)
    # Use 0.13 for smoke-test / early checkpoints; tighten to 0.05 after full training.
    "success_mse_threshold": 0.13,
}
