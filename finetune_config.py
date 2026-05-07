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
    }
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
    "phase1_steps": 5000,
    "phase1_lr": 1e-4,
    # Phase 2: LoRA-SP + ADP
    "phase2_steps": 10000,
    "phase2_lr": 1e-4,
    "lora_max_rank": 128,
    "lora_energy_threshold": 0.9,
    "adp_velocity_threshold": 0.15,
    "adp_min_keep_ratio": 0.3,
    # Phase 3: SnapFlow
    "phase3_steps": 5000,
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

EVAL = {
    "num_eval_episodes": 5,     # Evaluate on last 5 of eval episodes
}
