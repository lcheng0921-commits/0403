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
    'n_heads': 3,
    'single_head': False,
    'use_mha': True,
    'antenna_count': 16,
    'precoding_dim': 224,
    'csi_complex_dim': 4,
    'use_qos_guided_attention': True,
    'qos_feature_index': -1,
    'qos_attn_bias_scale': 2.0,

    # PPO hyperparameters
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'ppo_epochs': 8,
    'mini_batch_size': 64,
    'max_grad_norm': 0.5,
    'entropy_coef': 1e-2,
    'value_coef': 0.5,

    # Constraint dual variable
    'lambda_init': 0.1,
    'lambda_lr': 1e-4,
    # Kept for backward compatibility in old scripts; not used by standard dual update.
    'lambda_max': 1.5,
    # Kept for backward compatibility in old scripts; not used by standard dual update.
    'lambda_growth_scale': 0.05,

    # Training loop
    'episodes': 5000,
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
    'n_eve': 4,
    'r_cov': 220,
    'velocity_bound': 20,
    'jamming_power_bound': 15,
    'r_sense': 1e9,
    'K_richan': 10,

    # QoS and power constraints from draft
    # Paper setting uses a fixed R_th during training.
    'paper_fixed_qos_threshold': 0.1,
    'enforce_fixed_qos_threshold': True,
    'qos_threshold': 0.1,
    'tx_power_max_dbm': 30.0,
    'reward_ee_scale': 10.0,
    'reward_qos_scale': 1.0,

    # Simplified propulsion model coefficients (MVP phase)
    'fly_power_base': 80.0,
    'fly_power_coeff': 0.12,

    # Second-round physical mapping controls
    'phy_mapping_blend': 1.0,
    'precoding_gain_scale': 1.0,
    'interference_scale': 1.0,

    # Baseline switches: mbppo | ppo | circular | hover | sdma | noma
    'baseline': 'mbppo',
    'rsma_mode': 'rsma',

    # Runtime
    'fair_service': True,
    'map': None,
    'output_dir': None,

    # Saving and plotting support
    'save_episode_metrics': True,
}

if __name__ == '__main__':
    print('cuda' if torch.cuda.is_available() else 'cpu')
