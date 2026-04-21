import numpy as np
import torch

DEFAULT_MB_PPO_CONFIG = {
    # Seed and device
    'seed': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cuda_deterministic': False,
    'cuda_index': 0,

    # Network
    'n_layers': 2,
    'hidden_size': 256,
    'single_head': False,
    'antenna_count': 16,
    # Four-branch action architecture:
    # traj(2), common_precoding(2*M), private_precoding(2*K*M), resource(1 + (K+1) + K).
    'common_precoding_dim': 32,
    'private_precoding_dim': 192,
    'resource_dim': 14,
    # Legacy aggregated field kept for compatibility with old scripts.
    'precoding_dim': 224,
    'csi_complex_dim': 4,

    # PPO hyperparameters
    'lr': 8e-5,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'ppo_epochs': 10,
    'mini_batch_size': 64,
    'max_grad_norm': 0.5,
    'entropy_coef': 5e-3,
    'value_coef': 0.5,

    # Training loop
    'episodes': 3000,
    'episode_length': 100,
    'eval_interval': 50,
    'num_eval_episodes': 5,
    'save_freq': 200,

    # Environment and scenario (aligned to draft)
    'range_pos': 500,
    'n_community': 4,
    'n_ubs': 1,
    'n_powers': 10,
    'n_moves': 16,
    'n_gts': 6,
    'n_eve': 0,
    'velocity_bound': 20,
    'jamming_power_bound': 15,
    'r_sense': 1e9,
    'K_richan': 10,

    # Objective and reward terms (PF-EE + fixed physical penalties)
    'tx_power_max_dbm': 30.0,
    # reward_objective: pf_ee | sum_ee | max_min
    'reward_objective': 'pf_ee',
    'reward_pf_scale': 1.0,
    'pf_rate_ref_mbps': 0.1,
    'pf_log_scale': 100.0,
    # Objective-unit alignment refs: objective_main_aligned = objective_main_raw / objective_ref
    'reward_objective_ref_pf_ee': 8.0,
    'reward_objective_ref_sum_ee': 0.02,
    'reward_objective_ref_max_min': 0.06,
    # Final scalar passed to PPO: reward = reward_output_scale * (main_aligned - fixed_penalties)
    # Keep this around [1, 10] by tuning refs + this global multiplier.
    'reward_output_scale': 6.0,
    # Spatial penalties are divided by normalizer before applying fixed coefficients.
    # Keep 1.0 for backward compatibility with old runs.
    'wall_penalty_normalizer': 1.0,
    'core_penalty_normalizer': 1.0,
    'reward_power_penalty_scale': 1.0,
    'reward_wall_penalty_scale': 2.0,
    'reward_core_penalty_scale': 10.0,

    # Optional geometry controls
    'core_boundary_margin_ratio': 0.20,
    'terminate_on_core_violation': False,
    'core_violation_terminate_patience': 8,
    'core_violation_terminate_threshold': 2.0,
    'core_terminate_start_episode': 1,

    # Simplified propulsion model coefficients (MVP phase)
    'fly_power_base': 80.0,
    'fly_power_coeff': 0.12,

    # Second-round physical mapping controls
    'phy_mapping_blend': 1.0,
    'precoding_gain_scale': 1.0,
    'interference_scale': 1.0,
    'state_h_scale': 100.0,
    'alpha_common_logit_bias': 0.0,

    # Baseline switches: mbppo | ppo | circular | hover | sdma | noma
    'baseline': 'mbppo',
    'force_hard_mapping_for_ppo': False,
    'enforce_ppo_power_penalty_floor': True,
    'rsma_mode': 'rsma',
    # Ablation controls
    'freeze_uav_trajectory': False,
    # none | zf
    'fixed_precoding_scheme': 'none',
    # none | uniform
    'fixed_beta_mode': 'none',
    # full | rho_alpha
    'resource_learning_scope': 'full',

    # Runtime
    'fair_service': True,
    'map': None,
    'output_dir': None,

    # Saving and plotting support
    'save_episode_metrics': True,
}

if __name__ == '__main__':
    print('cuda' if torch.cuda.is_available() else 'cpu')
