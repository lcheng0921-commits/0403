import copy
from pathlib import Path
from types import SimpleNamespace as SN

import numpy as np
import torch

try:
    from tensorboardX import SummaryWriter
except Exception:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        # Fallback writer for environments without tensorboard dependencies.
        class SummaryWriter:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

from algo.mb_ppo.mb_ppo_config import DEFAULT_MB_PPO_CONFIG
from algo.mb_ppo.mb_ppo_learner import MultiBranchPPOLearner
from algo.mb_ppo.ppo_buffer import RolloutBuffer
from algo.mb_ppo.utils import check_args_sanity, save_config, save_var, set_rand_seed
from mb_ppo_env.mb_ppo_environment import MbPpoEnv


def _create_output_dir(repo_root: Path, requested_output_dir: str = '') -> Path:
    data_root = repo_root / 'mb_ppo_data'
    data_root.mkdir(parents=True, exist_ok=True)

    if requested_output_dir:
        exp_dir = Path(requested_output_dir)
        if not exp_dir.is_absolute():
            exp_dir = repo_root / requested_output_dir
        if exp_dir.exists() and (exp_dir / 'config.json').exists():
            raise RuntimeError(f'Requested output directory already contains an experiment: {exp_dir}')
        (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (exp_dir / 'logs').mkdir(parents=True, exist_ok=True)
        (exp_dir / 'vars').mkdir(parents=True, exist_ok=True)
        return exp_dir

    for i in range(1, 1000):
        exp_dir = data_root / f'exp{i}'
        try:
            exp_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue

        (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=False)
        (exp_dir / 'logs').mkdir(parents=True, exist_ok=False)
        (exp_dir / 'vars').mkdir(parents=True, exist_ok=False)
        return exp_dir

    raise RuntimeError('No available experiment directory under mb_ppo_data.')


def _resolve_resume_checkpoint(repo_root: Path, resume_path: str) -> Path:
    path = Path(str(resume_path).strip())
    if not path.is_absolute():
        path = repo_root / path

    if not path.exists():
        raise FileNotFoundError(f'Resume path not found: {path}')

    if path.is_dir():
        all_ckpt = sorted(path.glob('checkpoint_episode*.pt'))
        if not all_ckpt:
            raise FileNotFoundError(f'No checkpoint files found under: {path}')

        def ep_num(ckpt_path: Path):
            stem = ckpt_path.stem.replace('checkpoint_episode', '')
            try:
                return int(stem)
            except Exception:
                return -1

        all_ckpt = sorted(all_ckpt, key=ep_num)
        return all_ckpt[-1]

    return path


def _build_env(args):
    return MbPpoEnv(
        map=args.map,
        fair_service=args.fair_service,
        range_pos=args.range_pos,
        episode_length=args.episode_length,
        n_ubs=args.n_ubs,
        n_powers=args.n_powers,
        n_moves=args.n_moves,
        n_gts=args.n_gts,
        n_eve=args.n_eve,
        r_sense=args.r_sense,
        n_community=args.n_community,
        K_richan=args.K_richan,
        jamming_power_bound=args.jamming_power_bound,
        velocity_bound=args.velocity_bound,
        tx_power_max_dbm=args.tx_power_max_dbm,
        reward_objective=args.reward_objective,
        reward_pf_scale=args.reward_pf_scale,
        pf_rate_ref_mbps=args.pf_rate_ref_mbps,
        pf_log_scale=args.pf_log_scale,
        reward_objective_ref_pf_ee=args.reward_objective_ref_pf_ee,
        reward_objective_ref_sum_ee=args.reward_objective_ref_sum_ee,
        reward_objective_ref_max_min=args.reward_objective_ref_max_min,
        reward_output_scale=args.reward_output_scale,
        wall_penalty_normalizer=args.wall_penalty_normalizer,
        core_penalty_normalizer=args.core_penalty_normalizer,
        reward_power_penalty_scale=args.reward_power_penalty_scale,
        reward_wall_penalty_scale=args.reward_wall_penalty_scale,
        reward_core_penalty_scale=args.reward_core_penalty_scale,
        rsma_mode=args.rsma_mode,
        baseline=args.baseline,
        force_hard_mapping_for_ppo=args.force_hard_mapping_for_ppo,
        common_precoding_dim=args.common_precoding_dim,
        private_precoding_dim=args.private_precoding_dim,
        resource_dim=args.resource_dim,
        precoding_dim=args.precoding_dim,
        antenna_count=args.antenna_count,
        csi_complex_dim=args.csi_complex_dim,
        fly_power_base=args.fly_power_base,
        fly_power_coeff=args.fly_power_coeff,
        phy_mapping_blend=args.phy_mapping_blend,
        precoding_gain_scale=args.precoding_gain_scale,
        interference_scale=args.interference_scale,
        state_h_scale=args.state_h_scale,
        alpha_common_logit_bias=args.alpha_common_logit_bias,
        core_boundary_margin_ratio=args.core_boundary_margin_ratio,
        terminate_on_core_violation=args.terminate_on_core_violation,
        core_violation_terminate_patience=args.core_violation_terminate_patience,
        core_violation_terminate_threshold=args.core_violation_terminate_threshold,
        core_terminate_start_episode=args.core_terminate_start_episode,
        freeze_uav_trajectory=args.freeze_uav_trajectory,
        fixed_precoding_scheme=args.fixed_precoding_scheme,
        fixed_beta_mode=args.fixed_beta_mode,
    )

def evaluate_policy(
    test_env,
    learner,
    num_eval_episodes,
    record_trace=False,
    current_episode=None,
    total_episodes=None,
):
    ep_returns = []
    ep_pf_ee = []
    ep_ee = []
    ep_pf_comm_ee = []
    ep_comm_ee = []
    ep_fairness = []
    ep_min_user_rate = []
    ep_total_energy = []
    ep_tx_energy = []
    ep_tx_power = []
    trace_payload = None

    if current_episode is not None and hasattr(test_env, 'set_training_progress'):
        eval_total = total_episodes if total_episodes is not None else current_episode
        test_env.set_training_progress(current_episode, eval_total)

    for episode_idx in range(num_eval_episodes):
        obs, state, init_info = test_env.reset()
        done = False
        ep_ret = 0.0
        trajectory = [test_env.pos_ubs.copy()]

        while not done:
            actions, _, _, _ = learner.take_actions(obs, state, deterministic=True)
            obs, state, reward, done, info = test_env.step(actions)
            ep_ret += float(np.mean(reward))
            trajectory.append(test_env.pos_ubs.copy())

        ep_returns.append(ep_ret)
        ep_pf_ee.append(float(info.get('episode_pf_energy_efficiency', info.get('pf_energy_efficiency', 0.0))))
        ep_ee.append(float(info.get('episode_energy_efficiency', info.get('energy_efficiency', 0.0))))
        ep_pf_comm_ee.append(float(info.get('episode_pf_comm_ee', info.get('slot_pf_comm_ee', 0.0))))
        ep_comm_ee.append(float(info.get('episode_comm_ee', info.get('slot_comm_ee', 0.0))))
        ep_fairness.append(float(info.get('jain_fairness', info.get('effective_fairness', 0.0))))
        ep_min_user_rate.append(float(info.get('episode_min_user_rate_avg', info.get('min_user_rate', 0.0))))
        ep_total_energy.append(float(info.get('episode_total_energy', 0.0)))
        ep_tx_energy.append(float(info.get('episode_tx_energy', 0.0)))
        ep_tx_power.append(float(info.get('tx_power', 0.0)))

        if record_trace and episode_idx == 0:
            trace_payload = {
                'trajectory': np.asarray(trajectory, dtype=np.float32),
                'uav_init_pos': np.asarray(init_info['uav_init_pos'], dtype=np.float32),
                'gts_init_pos': np.asarray(init_info['gts_init_pos'], dtype=np.float32),
                'range_pos': float(init_info['range_pos']),
            }

    metrics = {
        'eval_return_mean': float(np.mean(ep_returns)),
        'eval_return_std': float(np.std(ep_returns)),
        'eval_pf_energy_efficiency': float(np.mean(ep_pf_ee)),
        'eval_energy_efficiency': float(np.mean(ep_ee)),
        'eval_pf_comm_energy_efficiency': float(np.mean(ep_pf_comm_ee)),
        'eval_comm_energy_efficiency': float(np.mean(ep_comm_ee)),
        'eval_jain_fairness': float(np.mean(ep_fairness)),
        'eval_min_user_rate': float(np.mean(ep_min_user_rate)),
        'eval_total_energy': float(np.mean(ep_total_energy)),
        'eval_tx_energy': float(np.mean(ep_tx_energy)),
        'eval_tx_power': float(np.mean(ep_tx_power)),
    }
    if trace_payload is not None:
        metrics['trace_payload'] = trace_payload
    return metrics


def train(train_kwargs=None):
    train_kwargs = train_kwargs or {}

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = _create_output_dir(repo_root, requested_output_dir=train_kwargs.get('output_dir', ''))

    config = copy.deepcopy(DEFAULT_MB_PPO_CONFIG)
    config.update(train_kwargs)

    antenna_count = int(config.get('antenna_count', 16))
    n_gts = int(config['n_gts'])
    required_common_precoding_dim = 2 * antenna_count
    required_private_precoding_dim = 2 * n_gts * antenna_count
    required_resource_dim = 2 * n_gts + 2
    required_precoding_dim = required_common_precoding_dim + required_private_precoding_dim

    configured_common_precoding_dim = int(config.get('common_precoding_dim', required_common_precoding_dim))
    configured_private_precoding_dim = int(config.get('private_precoding_dim', required_private_precoding_dim))
    configured_resource_dim = int(config.get('resource_dim', required_resource_dim))
    configured_precoding_dim = int(config.get('precoding_dim', required_precoding_dim))

    if configured_common_precoding_dim != required_common_precoding_dim:
        raise ValueError(
            f"common_precoding_dim mismatch: got {configured_common_precoding_dim}, "
            f"required {required_common_precoding_dim}."
        )
    if configured_private_precoding_dim != required_private_precoding_dim:
        raise ValueError(
            f"private_precoding_dim mismatch: got {configured_private_precoding_dim}, "
            f"required {required_private_precoding_dim}."
        )
    if configured_resource_dim != required_resource_dim:
        raise ValueError(
            f"resource_dim mismatch: got {configured_resource_dim}, required {required_resource_dim}."
        )
    if configured_precoding_dim != required_precoding_dim:
        raise ValueError(
            f"precoding_dim mismatch: got {configured_precoding_dim}, required {required_precoding_dim} "
            f"(2 * (n_gts + 1) * antenna_count)."
        )

    config['common_precoding_dim'] = required_common_precoding_dim
    config['private_precoding_dim'] = required_private_precoding_dim
    config['resource_dim'] = required_resource_dim
    # Keep legacy aggregated field for backward compatibility.
    config['precoding_dim'] = required_precoding_dim

    if config['map'] is None:
        from experiment.mb_ppo.maps import ClusteredMap500

        config['map'] = ClusteredMap500

    # Baseline switches (backward-compatible aliases).
    if config['baseline'] in ['single_head', 'ppo']:
        config['single_head'] = True
        config['baseline'] = 'ppo'
    else:
        config['single_head'] = False
    if config['baseline'] == 'sdma':
        config['rsma_mode'] = 'sdma'
    elif config['baseline'] == 'noma':
        config['rsma_mode'] = 'noma'

    # Keep PPO baseline physically fair by default, while allowing strict apples-to-apples reward ablations.
    if (
        config['baseline'] == 'ppo'
        and bool(config.get('enforce_ppo_power_penalty_floor', True))
        and float(config.get('reward_power_penalty_scale', 0.0)) < 100.0
    ):
        print(
            f"[Info] PPO baseline enforces reward_power_penalty_scale>=100.0; "
            f"override {config.get('reward_power_penalty_scale')} -> 100.0."
        )
        config['reward_power_penalty_scale'] = 100.0

    args = SN(**config)
    args.output_dir = str(output_dir)
    args = check_args_sanity(args)
    args.episodes = int(args.episodes)

    if str(args.resource_learning_scope) not in ['full', 'rho_alpha']:
        raise ValueError(
            f"Unsupported resource_learning_scope: {args.resource_learning_scope}. "
            "Allowed: ['full', 'rho_alpha']."
        )

    save_config(output_dir=args.output_dir, config=args.__dict__)
    set_rand_seed(args.seed)

    if args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env = _build_env(args)
    test_env = _build_env(args)

    env_info = env.get_env_info()
    learner = MultiBranchPPOLearner(env_info, args)

    if int(args.episodes) <= 0:
        raise ValueError(f'episodes must be positive, got {args.episodes}')

    resume_episode = 0
    resume_ckpt_path = None
    if str(getattr(args, 'resume', '')).strip():
        resume_ckpt_path = _resolve_resume_checkpoint(repo_root, str(args.resume))
        checkpoint = torch.load(str(resume_ckpt_path), map_location='cpu')

        if 'model_state_dict' not in checkpoint or 'optimizer_state_dict' not in checkpoint:
            raise KeyError(f'Invalid checkpoint format: {resume_ckpt_path}')

        learner.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_episode = int(checkpoint.get('episode', 0))
        learner.actor_critic.train()

        print(
            f"[Info] Resume from {resume_ckpt_path} | "
            f"resume_episode={resume_episode}"
        )

    episode_start = int(resume_episode + 1)
    episode_end = int(resume_episode + int(args.episodes))
    print(
        f"[Info] Training episode range: [{episode_start}, {episode_end}] "
        f"(run_episodes={int(args.episodes)})"
    )

    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    buffer = RolloutBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)

    train_returns = []
    eval_history = []
    episode_metrics = []
    eval_traces = []

    # Paper-aligned single-stage training with optional controlled-branch ablations.
    policy_branch = 'all'
    if str(args.resource_learning_scope) == 'rho_alpha':
        policy_branch = 'rho_alpha'
    elif str(args.fixed_precoding_scheme) != 'none':
        policy_branch = 'resource'
    elif bool(args.freeze_uav_trajectory):
        policy_branch = 'comm'

    learner.set_traj_branch_trainable(policy_branch in ['all', 'traj'])
    print(
        f"[Info] Single-stage training: policy_branch={policy_branch}, "
        f"freeze_uav_trajectory={bool(args.freeze_uav_trajectory)}, "
        f"fixed_precoding_scheme={str(args.fixed_precoding_scheme)}, fixed_beta_mode={str(args.fixed_beta_mode)}."
    )

    for episode in range(episode_start, episode_end + 1):
        buffer.reset()

        if hasattr(env, 'set_training_progress'):
            env.set_training_progress(episode, episode_end)

        (obs, state, _), done = env.reset(), False
        episode_return = 0.0
        info = {}

        while not done:
            actions, action_tensor, log_prob, value = learner.take_actions(
                obs,
                state,
                deterministic=False,
                policy_branch=policy_branch,
            )
            next_obs, next_state, reward, done, info = env.step(actions)

            reward_scalar = float(np.mean(reward))
            episode_return += reward_scalar

            buffer.add(
                obs_other=obs[0],
                obs_gt=obs[1],
                state=state,
                action=action_tensor,
                log_prob=log_prob,
                reward=reward_scalar,
                done=done,
                value=value,
            )

            obs, state = next_obs, next_state

        buffer.compute_returns_and_advantages(last_value=0.0)
        metrics = learner.update(buffer, policy_branch=policy_branch)

        train_returns.append(episode_return)
        ep_len = int(info.get('EpLen', args.episode_length))
        step_reward_mean = float(episode_return / max(1, ep_len))
        per_episode_metrics = {
            'episode': episode,
            'train_return': float(episode_return),
            'step_reward_mean': float(step_reward_mean),
            'pf_energy_efficiency': float(info.get('pf_energy_efficiency', 0.0)),
            'energy_efficiency': float(info.get('energy_efficiency', 0.0)),
            'slot_pf_total_ee': float(info.get('slot_pf_total_ee', info.get('pf_energy_efficiency', 0.0))),
            'slot_total_ee': float(info.get('slot_total_ee', info.get('energy_efficiency', 0.0))),
            'slot_pf_comm_ee': float(info.get('slot_pf_comm_ee', 0.0)),
            'slot_comm_ee': float(info.get('slot_comm_ee', 0.0)),
            'reward_objective': str(info.get('reward_objective', args.reward_objective)),
            'reward_main_raw': float(info.get('reward_main_raw', 0.0)),
            'reward_main_aligned': float(info.get('reward_main_aligned', 0.0)),
            'reward_ref': float(info.get('reward_ref', 0.0)),
            'reward_penalty_total': float(info.get('reward_penalty_total', 0.0)),
            'reward_before_scale': float(info.get('reward_before_scale', 0.0)),
            'reward_output_scale': float(info.get('reward_output_scale', 0.0)),
            'reward_scalar': float(info.get('reward_scalar', 0.0)),
            'log_utility_raw': float(info.get('log_utility_raw', 0.0)),
            'log_utility': float(info.get('log_utility', 0.0)),
            'log_utility_scale': float(info.get('log_utility_scale', args.pf_log_scale)),
            'pf_rate_ref_mbps': float(info.get('pf_rate_ref_mbps', args.pf_rate_ref_mbps)),
            'jain_fairness': float(info.get('jain_fairness', info.get('effective_fairness', 0.0))),
            'min_user_rate': float(info.get('min_user_rate', 0.0)),
            'episode_pf_energy_efficiency': float(info.get('episode_pf_energy_efficiency', 0.0)),
            'episode_energy_efficiency': float(info.get('episode_energy_efficiency', 0.0)),
            'episode_pf_comm_ee': float(info.get('episode_pf_comm_ee', 0.0)),
            'episode_comm_ee': float(info.get('episode_comm_ee', 0.0)),
            'episode_min_user_rate_avg': float(info.get('episode_min_user_rate_avg', 0.0)),
            'episode_total_energy': float(info.get('episode_total_energy', 0.0)),
            'episode_tx_energy': float(info.get('episode_tx_energy', 0.0)),
            'total_throughput': float(info.get('total_throughput', 0.0)),
            'tx_power': float(info.get('tx_power', 0.0)),
            'tx_power_violation': float(info.get('tx_power_violation', 0.0)),
            'wall_violation': float(info.get('wall_violation', 0.0)),
            'wall_penalty': float(info.get('wall_penalty', 0.0)),
            'core_violation': float(info.get('core_violation', 0.0)),
            'core_penalty': float(info.get('core_penalty', 0.0)),
            'core_penalty_factor': float(info.get('core_penalty_factor', 1.0)),
            'effective_core_penalty_scale': float(info.get('effective_core_penalty_scale', 0.0)),
            'core_violation_steps': int(info.get('core_violation_steps', 0)),
            'core_terminate_enabled': bool(info.get('core_terminate_enabled', False)),
            'core_terminate_start_episode': int(info.get('core_terminate_start_episode', 1)),
            'velocity': float(info.get('velocity', 0.0)),
            'user_mean_serving_dist': float(info.get('user_mean_serving_dist', 0.0)),
            'beam_quality': float(info.get('beam_quality', 0.0)),
            'common_beam_norm': float(info.get('common_beam_norm', 0.0)),
            'private_beam_norm_mean': float(info.get('private_beam_norm_mean', 0.0)),
            'private_beam_norm_std': float(info.get('private_beam_norm_std', 0.0)),
            'private_beam_norm_max_dev': float(info.get('private_beam_norm_max_dev', 0.0)),
            'alpha_common': float(info.get('alpha_common', 0.0)),
            'gamma_common_min': float(info.get('gamma_common_min', 0.0)),
            'gamma_private_mean': float(info.get('gamma_private_mean', 0.0)),
            'common_interference': float(info.get('common_interference', 0.0)),
            'private_interference': float(info.get('private_interference', 0.0)),
            'rsma_common_rate': float(info.get('rsma_common_rate', 0.0)),
            'rsma_private_rate': float(info.get('rsma_private_rate', 0.0)),
            'loss_actor': float(metrics['LossActor']),
            'loss_critic': float(metrics['LossCritic']),
            'loss_entropy': float(metrics['Entropy']),
            'loss_kl': float(metrics['KL']),
            'policy_branch': str(policy_branch),
        }
        episode_metrics.append(per_episode_metrics)

        writer.add_scalar('train/episode_return', episode_return, episode)
        writer.add_scalar('train/step_reward_mean', step_reward_mean, episode)
        writer.add_scalar('train/pf_energy_efficiency', info.get('pf_energy_efficiency', 0.0), episode)
        writer.add_scalar('train/energy_efficiency', info.get('energy_efficiency', 0.0), episode)
        writer.add_scalar('train/slot_pf_total_ee', info.get('slot_pf_total_ee', info.get('pf_energy_efficiency', 0.0)), episode)
        writer.add_scalar('train/slot_total_ee', info.get('slot_total_ee', info.get('energy_efficiency', 0.0)), episode)
        writer.add_scalar('train/slot_pf_comm_ee', info.get('slot_pf_comm_ee', 0.0), episode)
        writer.add_scalar('train/slot_comm_ee', info.get('slot_comm_ee', 0.0), episode)
        writer.add_scalar('train/reward_main_raw', info.get('reward_main_raw', 0.0), episode)
        writer.add_scalar('train/reward_main_aligned', info.get('reward_main_aligned', 0.0), episode)
        writer.add_scalar('train/reward_ref', info.get('reward_ref', 0.0), episode)
        writer.add_scalar('train/reward_penalty_total', info.get('reward_penalty_total', 0.0), episode)
        writer.add_scalar('train/reward_before_scale', info.get('reward_before_scale', 0.0), episode)
        writer.add_scalar('train/reward_output_scale', info.get('reward_output_scale', 0.0), episode)
        writer.add_scalar('train/reward_scalar', info.get('reward_scalar', 0.0), episode)
        writer.add_scalar('train/log_utility_raw', info.get('log_utility_raw', 0.0), episode)
        writer.add_scalar('train/log_utility', info.get('log_utility', 0.0), episode)
        writer.add_scalar('train/log_utility_scale', info.get('log_utility_scale', args.pf_log_scale), episode)
        writer.add_scalar('train/pf_rate_ref_mbps', info.get('pf_rate_ref_mbps', args.pf_rate_ref_mbps), episode)
        writer.add_scalar('train/jain_fairness', info.get('jain_fairness', info.get('effective_fairness', 0.0)), episode)
        writer.add_scalar('train/min_user_rate', info.get('min_user_rate', 0.0), episode)
        writer.add_scalar('train/episode_pf_energy_efficiency', info.get('episode_pf_energy_efficiency', 0.0), episode)
        writer.add_scalar('train/episode_energy_efficiency', info.get('episode_energy_efficiency', 0.0), episode)
        writer.add_scalar('train/episode_pf_comm_ee', info.get('episode_pf_comm_ee', 0.0), episode)
        writer.add_scalar('train/episode_comm_ee', info.get('episode_comm_ee', 0.0), episode)
        writer.add_scalar('train/episode_min_user_rate_avg', info.get('episode_min_user_rate_avg', 0.0), episode)
        writer.add_scalar('train/episode_total_energy', info.get('episode_total_energy', 0.0), episode)
        writer.add_scalar('train/episode_tx_energy', info.get('episode_tx_energy', 0.0), episode)
        writer.add_scalar('train/total_throughput', info.get('total_throughput', 0.0), episode)
        writer.add_scalar('train/tx_power', info.get('tx_power', 0.0), episode)
        writer.add_scalar('train/tx_power_violation', info.get('tx_power_violation', 0.0), episode)
        writer.add_scalar('train/wall_violation', info.get('wall_violation', 0.0), episode)
        writer.add_scalar('train/wall_penalty', info.get('wall_penalty', 0.0), episode)
        writer.add_scalar('train/core_violation', info.get('core_violation', 0.0), episode)
        writer.add_scalar('train/core_penalty', info.get('core_penalty', 0.0), episode)
        writer.add_scalar('train/core_penalty_factor', info.get('core_penalty_factor', 1.0), episode)
        writer.add_scalar('train/effective_core_penalty_scale', info.get('effective_core_penalty_scale', 0.0), episode)
        writer.add_scalar('train/core_terminate_enabled', float(bool(info.get('core_terminate_enabled', False))), episode)
        writer.add_scalar('train/core_violation_steps', info.get('core_violation_steps', 0.0), episode)
        writer.add_scalar('train/beam_quality', info.get('beam_quality', 0.0), episode)
        writer.add_scalar('train/common_beam_norm', info.get('common_beam_norm', 0.0), episode)
        writer.add_scalar('train/private_beam_norm_mean', info.get('private_beam_norm_mean', 0.0), episode)
        writer.add_scalar('train/private_beam_norm_std', info.get('private_beam_norm_std', 0.0), episode)
        writer.add_scalar('train/private_beam_norm_max_dev', info.get('private_beam_norm_max_dev', 0.0), episode)
        writer.add_scalar('train/user_mean_serving_dist', info.get('user_mean_serving_dist', 0.0), episode)
        writer.add_scalar('train/alpha_common', info.get('alpha_common', 0.0), episode)
        writer.add_scalar('train/gamma_common_min', info.get('gamma_common_min', 0.0), episode)
        writer.add_scalar('train/gamma_private_mean', info.get('gamma_private_mean', 0.0), episode)
        writer.add_scalar('train/common_interference', info.get('common_interference', 0.0), episode)
        writer.add_scalar('train/private_interference', info.get('private_interference', 0.0), episode)
        writer.add_scalar('train/rsma_common_rate', info.get('rsma_common_rate', 0.0), episode)
        writer.add_scalar('train/rsma_private_rate', info.get('rsma_private_rate', 0.0), episode)
        writer.add_scalar('loss/actor', metrics['LossActor'], episode)
        writer.add_scalar('loss/critic', metrics['LossCritic'], episode)
        writer.add_scalar('loss/entropy', metrics['Entropy'], episode)
        writer.add_scalar('loss/kl', metrics['KL'], episode)
        if policy_branch == 'traj':
            policy_branch_code = 0.0
        elif policy_branch == 'comm':
            policy_branch_code = 1.0
        else:
            policy_branch_code = 2.0
        writer.add_scalar('train/policy_branch_code', policy_branch_code, episode)

        if episode % max(1, args.eval_interval) == 0:
            eval_stats = evaluate_policy(
                test_env,
                learner,
                args.num_eval_episodes,
                record_trace=True,
                current_episode=episode,
                total_episodes=episode_end,
            )
            trace_payload = eval_stats.pop('trace_payload', None)
            eval_stats['episode'] = episode
            eval_history.append(eval_stats)
            if trace_payload is not None:
                eval_traces.append({'episode': episode, **trace_payload})

            writer.add_scalar('eval/return_mean', eval_stats['eval_return_mean'], episode)
            writer.add_scalar('eval/return_std', eval_stats['eval_return_std'], episode)
            writer.add_scalar('eval/pf_energy_efficiency', eval_stats['eval_pf_energy_efficiency'], episode)
            writer.add_scalar('eval/energy_efficiency', eval_stats['eval_energy_efficiency'], episode)
            writer.add_scalar('eval/pf_comm_energy_efficiency', eval_stats['eval_pf_comm_energy_efficiency'], episode)
            writer.add_scalar('eval/comm_energy_efficiency', eval_stats['eval_comm_energy_efficiency'], episode)
            writer.add_scalar('eval/jain_fairness', eval_stats['eval_jain_fairness'], episode)
            writer.add_scalar('eval/min_user_rate', eval_stats['eval_min_user_rate'], episode)
            writer.add_scalar('eval/total_energy', eval_stats['eval_total_energy'], episode)
            writer.add_scalar('eval/tx_energy', eval_stats['eval_tx_energy'], episode)
            writer.add_scalar('eval/tx_power', eval_stats['eval_tx_power'], episode)

            print(
                'Episode {} | branch={} | train_return={:.4f} | eval_return={:.4f} | eval_pf_ee={:.6f} | eval_ee={:.6f} | eval_pf_comm_ee={:.6f} | eval_comm_ee={:.6f} | eval_jain={:.4f} | eval_rmin={:.6f}'.format(
                    episode,
                    policy_branch,
                    episode_return,
                    eval_stats['eval_return_mean'],
                    eval_stats['eval_pf_energy_efficiency'],
                    eval_stats['eval_energy_efficiency'],
                    eval_stats['eval_pf_comm_energy_efficiency'],
                    eval_stats['eval_comm_energy_efficiency'],
                    eval_stats['eval_jain_fairness'],
                    eval_stats['eval_min_user_rate'],
                )
            )

        if episode % max(1, args.save_freq) == 0 or episode == episode_end:
            save_path = output_dir / 'checkpoints' / f'checkpoint_episode{episode}.pt'
            learner.save_model(str(save_path), stamp=dict(episode=episode))

            save_var(str(output_dir / 'vars' / 'train_returns'), train_returns)
            save_var(str(output_dir / 'vars' / 'eval_history'), eval_history)
            if args.save_episode_metrics:
                save_var(str(output_dir / 'vars' / 'episode_metrics'), episode_metrics)
                save_var(str(output_dir / 'vars' / 'eval_traces'), eval_traces)

    writer.close()
    print('Complete MB-PPO training.')


def build_train_kwargs_from_cli(cli_args):
    baseline = cli_args.baseline

    # Normalize legacy naming.
    baseline_norm = 'ppo' if baseline == 'single_head' else baseline
    if baseline_norm not in ['mbppo', 'ppo', 'circular', 'hover', 'sdma', 'noma']:
        raise ValueError(f'Unsupported baseline: {baseline_norm}')

    if baseline_norm == 'sdma':
        rsma_mode = 'sdma'
    elif baseline_norm == 'noma':
        rsma_mode = 'noma'
    else:
        rsma_mode = 'rsma'

    return {
        'baseline': baseline_norm,
        'single_head': baseline_norm == 'ppo',
        'rsma_mode': rsma_mode,
        'resume': cli_args.resume,
        'output_dir': cli_args.output_dir,
        'episodes': cli_args.episodes,
        'eval_interval': cli_args.eval_interval,
        'save_freq': cli_args.save_freq,
        'seed': cli_args.seed,
        'n_gts': int(cli_args.n_gts),
        'tx_power_max_dbm': cli_args.tx_power_max_dbm,
        'reward_objective': str(cli_args.reward_objective),
        'reward_pf_scale': cli_args.reward_pf_scale,
        'pf_rate_ref_mbps': cli_args.pf_rate_ref_mbps,
        'pf_log_scale': cli_args.pf_log_scale,
        'reward_objective_ref_pf_ee': cli_args.reward_objective_ref_pf_ee,
        'reward_objective_ref_sum_ee': cli_args.reward_objective_ref_sum_ee,
        'reward_objective_ref_max_min': cli_args.reward_objective_ref_max_min,
        'reward_output_scale': cli_args.reward_output_scale,
        'wall_penalty_normalizer': cli_args.wall_penalty_normalizer,
        'core_penalty_normalizer': cli_args.core_penalty_normalizer,
        'reward_power_penalty_scale': cli_args.reward_power_penalty_scale,
        'reward_wall_penalty_scale': cli_args.reward_wall_penalty_scale,
        'reward_core_penalty_scale': cli_args.reward_core_penalty_scale,
        'core_boundary_margin_ratio': cli_args.core_boundary_margin_ratio,
        'terminate_on_core_violation': cli_args.terminate_on_core_violation,
        'core_violation_terminate_patience': cli_args.core_violation_terminate_patience,
        'core_violation_terminate_threshold': cli_args.core_violation_terminate_threshold,
        'core_terminate_start_episode': cli_args.core_terminate_start_episode,
        'phy_mapping_blend': cli_args.phy_mapping_blend,
        'precoding_gain_scale': cli_args.precoding_gain_scale,
        'interference_scale': cli_args.interference_scale,
        'state_h_scale': cli_args.state_h_scale,
        'alpha_common_logit_bias': cli_args.alpha_common_logit_bias,
        'force_hard_mapping_for_ppo': cli_args.force_hard_mapping_for_ppo,
        'enforce_ppo_power_penalty_floor': cli_args.enforce_ppo_power_penalty_floor,
        'freeze_uav_trajectory': cli_args.freeze_uav_trajectory,
        'fixed_precoding_scheme': cli_args.fixed_precoding_scheme,
        'fixed_beta_mode': cli_args.fixed_beta_mode,
        'resource_learning_scope': cli_args.resource_learning_scope,
        'antenna_count': cli_args.antenna_count,
        'common_precoding_dim': int(cli_args.common_precoding_dim),
        'private_precoding_dim': int(cli_args.private_precoding_dim),
        'resource_dim': int(cli_args.resource_dim),
        # Legacy aggregated field for compatibility.
        'precoding_dim': int(2 * (int(cli_args.n_gts) + 1) * int(cli_args.antenna_count)),
        'csi_complex_dim': cli_args.csi_complex_dim,
        'save_episode_metrics': not cli_args.no_save_episode_metrics,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train MB-PPO (multi-branch focus).')
    parser.add_argument(
        '--baseline',
        type=str,
        default='mbppo',
        choices=['mbppo', 'ppo', 'circular', 'hover', 'sdma', 'noma'],
        help='Training mode / baseline selector.',
    )
    parser.add_argument('--episodes', type=int, default=DEFAULT_MB_PPO_CONFIG['episodes'], help='Number of training episodes.')
    parser.add_argument('--eval-interval', type=int, default=20, help='Evaluate every N episodes.')
    parser.add_argument('--save-freq', type=int, default=100, help='Save checkpoint every N episodes.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--n-gts', type=int, default=DEFAULT_MB_PPO_CONFIG['n_gts'], help='Number of ground users.')
    parser.add_argument('--resume', type=str, default='', help='Optional checkpoint file or checkpoint directory to resume from.')
    parser.add_argument('--output-dir', type=str, default='', help='Optional explicit output directory (absolute or repo-relative).')
    parser.add_argument('--tx-power-max-dbm', type=float, default=30.0, help='Maximum transmit power in dBm.')
    parser.add_argument('--reward-objective', type=str, default=DEFAULT_MB_PPO_CONFIG['reward_objective'], choices=['pf_ee', 'sum_ee', 'max_min'], help='Primary objective used in reward (PF-EE / Sum-EE / Max-Min).')
    parser.add_argument('--reward-pf-scale', type=float, default=DEFAULT_MB_PPO_CONFIG['reward_pf_scale'], help='Scale factor for PF-EE term in reward.')
    parser.add_argument('--pf-rate-ref-mbps', type=float, default=DEFAULT_MB_PPO_CONFIG['pf_rate_ref_mbps'], help='Rate reference (Mbps) to make log utility dimensionless via log(1 + R/R_ref).')
    parser.add_argument('--pf-log-scale', type=float, default=DEFAULT_MB_PPO_CONFIG['pf_log_scale'], help='Scaling factor applied to summed log utility before PF-EE reward computation.')
    parser.add_argument('--reward-objective-ref-pf-ee', type=float, default=DEFAULT_MB_PPO_CONFIG['reward_objective_ref_pf_ee'], help='Reference scale for PF-EE objective alignment.')
    parser.add_argument('--reward-objective-ref-sum-ee', type=float, default=DEFAULT_MB_PPO_CONFIG['reward_objective_ref_sum_ee'], help='Reference scale for Sum-EE objective alignment.')
    parser.add_argument('--reward-objective-ref-max-min', type=float, default=DEFAULT_MB_PPO_CONFIG['reward_objective_ref_max_min'], help='Reference scale for Max-Min objective alignment.')
    parser.add_argument('--reward-output-scale', type=float, default=DEFAULT_MB_PPO_CONFIG['reward_output_scale'], help='Final scalar multiplier applied to aligned reward before PPO update.')
    parser.add_argument('--wall-penalty-normalizer', type=float, default=DEFAULT_MB_PPO_CONFIG['wall_penalty_normalizer'], help='Normalize wall violation by this value before applying wall penalty scale.')
    parser.add_argument('--core-penalty-normalizer', type=float, default=DEFAULT_MB_PPO_CONFIG['core_penalty_normalizer'], help='Normalize core violation by this value before applying core penalty scale.')
    parser.add_argument('--reward-power-penalty-scale', type=float, default=DEFAULT_MB_PPO_CONFIG['reward_power_penalty_scale'], help='Penalty coefficient for TX power violation ratio.')
    parser.add_argument('--reward-wall-penalty-scale', type=float, default=DEFAULT_MB_PPO_CONFIG['reward_wall_penalty_scale'], help='Penalty coefficient for wall overshoot distance before boundary clipping.')
    parser.add_argument('--reward-core-penalty-scale', type=float, default=DEFAULT_MB_PPO_CONFIG['reward_core_penalty_scale'], help='Penalty coefficient for violating the central core service region.')
    parser.add_argument('--core-boundary-margin-ratio', type=float, default=DEFAULT_MB_PPO_CONFIG['core_boundary_margin_ratio'], help='Core service-zone margin ratio for [margin, range-margin] constraint.')
    parser.add_argument('--terminate-on-core-violation', action=argparse.BooleanOptionalAction, default=DEFAULT_MB_PPO_CONFIG['terminate_on_core_violation'], help='Terminate episode when core-zone violation persists for several steps.')
    parser.add_argument('--core-violation-terminate-patience', type=int, default=DEFAULT_MB_PPO_CONFIG['core_violation_terminate_patience'], help='Consecutive violating steps required before early termination.')
    parser.add_argument('--core-violation-terminate-threshold', type=float, default=DEFAULT_MB_PPO_CONFIG['core_violation_terminate_threshold'], help='Per-step core violation threshold used by early termination.')
    parser.add_argument('--core-terminate-start-episode', type=int, default=DEFAULT_MB_PPO_CONFIG['core_terminate_start_episode'], help='Episode index from which core early termination becomes active (if enabled).')
    parser.add_argument('--phy-mapping-blend', type=float, default=1.0, help='Blend ratio between base and physical-mapped rates.')
    parser.add_argument('--precoding-gain-scale', type=float, default=1.0, help='Gain scaling factor used in physical mapping.')
    parser.add_argument('--interference-scale', type=float, default=1.0, help='Interference scaling factor used in physical mapping.')
    parser.add_argument('--state-h-scale', type=float, default=DEFAULT_MB_PPO_CONFIG['state_h_scale'], help='Scale factor applied to CSI features only in NN state/obs inputs.')
    parser.add_argument('--alpha-common-logit-bias', type=float, default=DEFAULT_MB_PPO_CONFIG['alpha_common_logit_bias'], help='Positive bias added to common-stream alpha logit before softmax mapping.')
    parser.add_argument('--force-hard-mapping-for-ppo', action='store_true', help='When baseline=ppo, force hard physical mapping (same feasibility mapping path as MB-PPO/SAC).')
    parser.add_argument('--enforce-ppo-power-penalty-floor', action=argparse.BooleanOptionalAction, default=DEFAULT_MB_PPO_CONFIG['enforce_ppo_power_penalty_floor'], help='If enabled, baseline=ppo forces reward_power_penalty_scale >= 100 for legacy safety.')
    parser.add_argument('--freeze-uav-trajectory', action=argparse.BooleanOptionalAction, default=DEFAULT_MB_PPO_CONFIG['freeze_uav_trajectory'], help='If enabled, UAV trajectory is fixed (hover), used for branch ablation.')
    parser.add_argument('--fixed-precoding-scheme', type=str, default=DEFAULT_MB_PPO_CONFIG['fixed_precoding_scheme'], choices=['none', 'zf'], help='Fix precoding direction by closed-form scheme; zf ignores learned precoding branch outputs.')
    parser.add_argument('--fixed-beta-mode', type=str, default=DEFAULT_MB_PPO_CONFIG['fixed_beta_mode'], choices=['none', 'uniform'], help='Fix beta allocation mode. uniform keeps common-rate split fixed.')
    parser.add_argument('--resource-learning-scope', type=str, default=DEFAULT_MB_PPO_CONFIG['resource_learning_scope'], choices=['full', 'rho_alpha'], help='Resource subspace used by PPO gradient. rho_alpha learns only rho and alpha.')
    parser.add_argument('--antenna-count', type=int, default=DEFAULT_MB_PPO_CONFIG['antenna_count'], help='Number of transmit antennas used for beam-direction mapping.')
    parser.add_argument('--common-precoding-dim', type=int, default=DEFAULT_MB_PPO_CONFIG['common_precoding_dim'], help='Dimension of common-stream direction branch; must equal 2*antenna_count.')
    parser.add_argument('--private-precoding-dim', type=int, default=DEFAULT_MB_PPO_CONFIG['private_precoding_dim'], help='Dimension of private-stream direction branch; must equal 2*n_gts*antenna_count.')
    parser.add_argument('--resource-dim', type=int, default=DEFAULT_MB_PPO_CONFIG['resource_dim'], help='Dimension of resource branch (rho+alpha+beta); must equal 2*n_gts+2.')
    parser.add_argument('--csi-complex-dim', type=int, default=4, help='Per-user CSI complex dimension used in observation features.')
    parser.add_argument('--no-save-episode-metrics', action='store_true', help='Disable saving per-episode metric pickle files.')
    cli_args = parser.parse_args()

    kwargs = build_train_kwargs_from_cli(cli_args)
    train(kwargs)
