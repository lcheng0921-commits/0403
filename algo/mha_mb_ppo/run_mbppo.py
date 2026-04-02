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

from algo.mha_mb_ppo.mb_ppo_config import DEFAULT_MB_PPO_CONFIG
from algo.mha_mb_ppo.mb_ppo_learner import MultiBranchPPOLearner
from algo.mha_mb_ppo.ppo_buffer import RolloutBuffer
from algo.mha_mb_ppo.utils import check_args_sanity, save_config, save_var, set_rand_seed
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
        r_cov=args.r_cov,
        r_sense=args.r_sense,
        n_community=args.n_community,
        K_richan=args.K_richan,
        jamming_power_bound=args.jamming_power_bound,
        velocity_bound=args.velocity_bound,
        qos_threshold=args.qos_threshold,
        tx_power_max_dbm=args.tx_power_max_dbm,
        reward_ee_scale=args.reward_ee_scale,
        reward_qos_scale=args.reward_qos_scale,
        lambda_penalty=args.lambda_init,
        rsma_mode=args.rsma_mode,
        baseline=args.baseline,
        precoding_dim=args.precoding_dim,
        antenna_count=args.antenna_count,
        csi_complex_dim=args.csi_complex_dim,
        fly_power_base=args.fly_power_base,
        fly_power_coeff=args.fly_power_coeff,
        phy_mapping_blend=args.phy_mapping_blend,
        precoding_gain_scale=args.precoding_gain_scale,
        interference_scale=args.interference_scale,
    )


def _apply_fixed_qos_threshold(config: dict):
    enforce = bool(config.get('enforce_fixed_qos_threshold', False))
    if not enforce:
        return

    fixed_rth = float(config.get('paper_fixed_qos_threshold', config.get('qos_threshold', 0.1)))
    requested_rth = float(config.get('qos_threshold', fixed_rth))
    config['paper_fixed_qos_threshold'] = fixed_rth
    config['qos_threshold'] = fixed_rth

    if abs(requested_rth - fixed_rth) > 1e-12:
        print(
            f'[Info] Enforce fixed R_th={fixed_rth:.4f}; '
            f'ignore requested qos_threshold={requested_rth:.4f}.'
        )


def evaluate_policy(test_env, learner, num_eval_episodes, record_trace=False):
    ep_returns = []
    ep_qos = []
    ep_qos_norm = []
    ep_qos_dual = []
    ep_qos_dual_norm = []
    ep_ee = []
    ep_fairness = []
    ep_tx_power = []
    trace_payload = None

    for episode_idx in range(num_eval_episodes):
        test_env.set_lambda(learner.lambda_penalty)
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
        ep_qos.append(float(info.get('qos_violation', 0.0)))
        ep_qos_norm.append(float(info.get('qos_violation_norm', 0.0)))
        ep_qos_dual.append(float(info.get('qos_dual_signal', 0.0)))
        ep_qos_dual_norm.append(float(info.get('qos_dual_signal_norm', 0.0)))
        ep_ee.append(float(info.get('energy_efficiency', 0.0)))
        ep_fairness.append(float(info.get('effective_fairness', 0.0)))
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
        'eval_qos_violation': float(np.mean(ep_qos)),
        'eval_qos_violation_norm': float(np.mean(ep_qos_norm)),
        'eval_qos_dual_signal': float(np.mean(ep_qos_dual)),
        'eval_qos_dual_signal_norm': float(np.mean(ep_qos_dual_norm)),
        'eval_energy_efficiency': float(np.mean(ep_ee)),
        'eval_fairness': float(np.mean(ep_fairness)),
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
    _apply_fixed_qos_threshold(config)

    required_precoding_dim = 2 * (int(config['n_gts']) + 1) * int(config.get('antenna_count', 16))
    if int(config.get('precoding_dim', 0)) < required_precoding_dim:
        print(
            f"[Info] Align precoding_dim to paper mapping: "
            f"{config.get('precoding_dim', 0)} -> {required_precoding_dim}."
        )
        config['precoding_dim'] = required_precoding_dim

    if config['map'] is None:
        from experiment.mb_ppo.maps import ClusteredMap500

        config['map'] = ClusteredMap500

    # Baseline switches (backward-compatible aliases).
    if config['baseline'] in ['single_head', 'ppo']:
        config['single_head'] = True
        # Plain PPO baseline should not use MHA features.
        config['use_mha'] = False
        config['baseline'] = 'ppo'
    if config['baseline'] == 'mbppo_nomha':
        config['single_head'] = False
        config['use_mha'] = False
    if config['baseline'] == 'sdma':
        config['rsma_mode'] = 'sdma'
    elif config['baseline'] == 'noma':
        config['rsma_mode'] = 'noma'

    args = SN(**config)
    args.output_dir = str(output_dir)
    args = check_args_sanity(args)

    save_config(output_dir=args.output_dir, config=args.__dict__)
    set_rand_seed(args.seed)

    if args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env = _build_env(args)
    test_env = _build_env(args)

    env_info = env.get_env_info()
    learner = MultiBranchPPOLearner(env_info, args)

    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    buffer = RolloutBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)

    train_returns = []
    eval_history = []
    episode_metrics = []
    eval_traces = []

    for episode in range(1, args.episodes + 1):
        buffer.reset()
        env.set_lambda(learner.lambda_penalty)

        (obs, state, _), done = env.reset(), False
        episode_return = 0.0
        info = {}

        while not done:
            actions, action_tensor, log_prob, value = learner.take_actions(obs, state, deterministic=False)
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
                # Paper dual update uses (1/N) * sum_n sum_k (R_th - R_k[n]).
                qos_violation=info.get('qos_dual_signal_sum', 0.0),
            )

            obs, state = next_obs, next_state

        buffer.compute_returns_and_advantages(last_value=0.0)
        metrics = learner.update(buffer)
        lambda_penalty = learner.update_lambda(buffer.mean_qos_signal())

        train_returns.append(episode_return)
        per_episode_metrics = {
            'episode': episode,
            'train_return': float(episode_return),
            'qos_violation': float(info.get('qos_violation', 0.0)),
            'qos_violation_sum': float(info.get('qos_violation_sum', 0.0)),
            'qos_violation_norm': float(info.get('qos_violation_norm', 0.0)),
            'qos_dual_signal': float(info.get('qos_dual_signal', 0.0)),
            'qos_dual_signal_sum': float(info.get('qos_dual_signal_sum', 0.0)),
            'qos_dual_signal_norm': float(info.get('qos_dual_signal_norm', 0.0)),
            'weighted_qos_gap': float(info.get('weighted_qos_gap', 0.0)),
            'weighted_qos_gap_norm': float(info.get('weighted_qos_gap_norm', 0.0)),
            'energy_efficiency': float(info.get('energy_efficiency', 0.0)),
            'effective_fairness': float(info.get('effective_fairness', 0.0)),
            'total_throughput': float(info.get('total_throughput', 0.0)),
            'tx_power': float(info.get('tx_power', 0.0)),
            'velocity': float(info.get('velocity', 0.0)),
            'beam_quality': float(info.get('beam_quality', 0.0)),
            'alpha_common': float(info.get('alpha_common', 0.0)),
            'gamma_common_min': float(info.get('gamma_common_min', 0.0)),
            'gamma_private_mean': float(info.get('gamma_private_mean', 0.0)),
            'common_interference': float(info.get('common_interference', 0.0)),
            'private_interference': float(info.get('private_interference', 0.0)),
            'rsma_common_rate': float(info.get('rsma_common_rate', 0.0)),
            'rsma_private_rate': float(info.get('rsma_private_rate', 0.0)),
            'lambda_penalty': float(lambda_penalty),
            'loss_actor': float(metrics['LossActor']),
            'loss_critic': float(metrics['LossCritic']),
            'loss_entropy': float(metrics['Entropy']),
            'loss_kl': float(metrics['KL']),
        }
        episode_metrics.append(per_episode_metrics)

        writer.add_scalar('train/episode_return', episode_return, episode)
        writer.add_scalar('train/lambda_penalty', lambda_penalty, episode)
        writer.add_scalar('train/qos_violation', info.get('qos_violation', 0.0), episode)
        writer.add_scalar('train/qos_violation_sum', info.get('qos_violation_sum', 0.0), episode)
        writer.add_scalar('train/qos_violation_norm', info.get('qos_violation_norm', 0.0), episode)
        writer.add_scalar('train/qos_dual_signal', info.get('qos_dual_signal', 0.0), episode)
        writer.add_scalar('train/qos_dual_signal_sum', info.get('qos_dual_signal_sum', 0.0), episode)
        writer.add_scalar('train/qos_dual_signal_norm', info.get('qos_dual_signal_norm', 0.0), episode)
        writer.add_scalar('train/weighted_qos_gap', info.get('weighted_qos_gap', 0.0), episode)
        writer.add_scalar('train/weighted_qos_gap_norm', info.get('weighted_qos_gap_norm', 0.0), episode)
        writer.add_scalar('train/energy_efficiency', info.get('energy_efficiency', 0.0), episode)
        writer.add_scalar('train/total_throughput', info.get('total_throughput', 0.0), episode)
        writer.add_scalar('train/fairness', info.get('effective_fairness', 0.0), episode)
        writer.add_scalar('train/tx_power', info.get('tx_power', 0.0), episode)
        writer.add_scalar('train/beam_quality', info.get('beam_quality', 0.0), episode)
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

        if episode % max(1, args.eval_interval) == 0:
            eval_stats = evaluate_policy(
                test_env,
                learner,
                args.num_eval_episodes,
                record_trace=True,
            )
            trace_payload = eval_stats.pop('trace_payload', None)
            eval_stats['episode'] = episode
            eval_history.append(eval_stats)
            if trace_payload is not None:
                eval_traces.append({'episode': episode, **trace_payload})

            writer.add_scalar('eval/return_mean', eval_stats['eval_return_mean'], episode)
            writer.add_scalar('eval/return_std', eval_stats['eval_return_std'], episode)
            writer.add_scalar('eval/qos_violation', eval_stats['eval_qos_violation'], episode)
            writer.add_scalar('eval/qos_violation_norm', eval_stats['eval_qos_violation_norm'], episode)
            writer.add_scalar('eval/qos_dual_signal', eval_stats['eval_qos_dual_signal'], episode)
            writer.add_scalar('eval/qos_dual_signal_norm', eval_stats['eval_qos_dual_signal_norm'], episode)
            writer.add_scalar('eval/energy_efficiency', eval_stats['eval_energy_efficiency'], episode)
            writer.add_scalar('eval/fairness', eval_stats['eval_fairness'], episode)
            writer.add_scalar('eval/tx_power', eval_stats['eval_tx_power'], episode)

            print(
                'Episode {} | train_return={:.4f} | eval_return={:.4f} | qos_gap={:.4f} | qos_dual={:.4f} | fair={:.4f} | lambda={:.4f}'.format(
                    episode,
                    episode_return,
                    eval_stats['eval_return_mean'],
                    eval_stats['eval_qos_violation'],
                    eval_stats['eval_qos_dual_signal'],
                    eval_stats['eval_fairness'],
                    lambda_penalty,
                )
            )

        if episode % max(1, args.save_freq) == 0 or episode == args.episodes:
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
    if baseline_norm not in ['mbppo', 'mbppo_nomha', 'ppo', 'circular', 'hover', 'sdma', 'noma']:
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
        # Plain PPO uses single-head actor without MHA.
        'use_mha': baseline_norm not in ['mbppo_nomha', 'ppo'],
        'rsma_mode': rsma_mode,
        'output_dir': cli_args.output_dir,
        'episodes': cli_args.episodes,
        'eval_interval': cli_args.eval_interval,
        'save_freq': cli_args.save_freq,
        'seed': cli_args.seed,
        'lambda_init': cli_args.lambda_init,
        'lambda_lr': cli_args.lambda_lr,
        'lambda_max': cli_args.lambda_max,
        'lambda_growth_scale': cli_args.lambda_growth_scale,
        'qos_threshold': cli_args.qos_threshold,
        'paper_fixed_qos_threshold': cli_args.paper_fixed_rth,
        'enforce_fixed_qos_threshold': not cli_args.disable_fixed_rth,
        'tx_power_max_dbm': cli_args.tx_power_max_dbm,
        'phy_mapping_blend': cli_args.phy_mapping_blend,
        'precoding_gain_scale': cli_args.precoding_gain_scale,
        'interference_scale': cli_args.interference_scale,
        'antenna_count': cli_args.antenna_count,
        'csi_complex_dim': cli_args.csi_complex_dim,
        'use_qos_guided_attention': not cli_args.disable_qos_guided_attention,
        'qos_feature_index': cli_args.qos_feature_index,
        'qos_attn_bias_scale': cli_args.qos_attn_bias_scale,
        'save_episode_metrics': not cli_args.no_save_episode_metrics,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train MB-PPO (second implementation round).')
    parser.add_argument(
        '--baseline',
        type=str,
        default='mbppo',
        choices=['mbppo', 'mbppo_nomha', 'ppo', 'single_head', 'circular', 'hover', 'sdma', 'noma'],
        help='Training mode / baseline selector.',
    )
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes.')
    parser.add_argument('--eval-interval', type=int, default=20, help='Evaluate every N episodes.')
    parser.add_argument('--save-freq', type=int, default=100, help='Save checkpoint every N episodes.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--lambda-init', type=float, default=0.1, help='Initial dual variable value.')
    parser.add_argument('--lambda-lr', type=float, default=DEFAULT_MB_PPO_CONFIG['lambda_lr'], help='Dual update step size for lambda.')
    parser.add_argument('--lambda-max', type=float, default=1.5, help='Hard upper bound of lambda penalty.')
    parser.add_argument('--lambda-growth-scale', type=float, default=0.05, help='Scaling factor for positive lambda growth step.')
    parser.add_argument('--output-dir', type=str, default='', help='Optional explicit output directory (absolute or repo-relative).')
    parser.add_argument('--qos-threshold', type=float, default=0.1, help='Requested QoS threshold; ignored when fixed R_th is enabled.')
    parser.add_argument('--paper-fixed-rth', type=float, default=DEFAULT_MB_PPO_CONFIG['paper_fixed_qos_threshold'], help='Fixed paper R_th used when fixed mode is enabled.')
    parser.add_argument('--disable-fixed-rth', action='store_true', help='Disable fixed R_th enforcement and use --qos-threshold directly.')
    parser.add_argument('--tx-power-max-dbm', type=float, default=30.0, help='Maximum transmit power in dBm.')
    parser.add_argument('--phy-mapping-blend', type=float, default=1.0, help='Blend ratio between base and physical-mapped rates.')
    parser.add_argument('--precoding-gain-scale', type=float, default=1.0, help='Gain scaling factor used in physical mapping.')
    parser.add_argument('--interference-scale', type=float, default=1.0, help='Interference scaling factor used in physical mapping.')
    parser.add_argument('--antenna-count', type=int, default=DEFAULT_MB_PPO_CONFIG['antenna_count'], help='Number of transmit antennas used for beam-direction mapping.')
    parser.add_argument('--csi-complex-dim', type=int, default=4, help='Per-user CSI complex dimension for observation (real+imag fed to MHA).')
    parser.add_argument('--disable-qos-guided-attention', action='store_true', help='Disable QoS-gap guided attention bias in actor MHA.')
    parser.add_argument('--qos-feature-index', type=int, default=-1, help='Feature index of QoS-gap in per-user GT features fed to MHA.')
    parser.add_argument('--qos-attn-bias-scale', type=float, default=2.0, help='Scale of QoS-guided additive bias for attention logits.')
    parser.add_argument('--no-save-episode-metrics', action='store_true', help='Disable saving per-episode metric pickle files.')
    cli_args = parser.parse_args()

    kwargs = build_train_kwargs_from_cli(cli_args)
    train(kwargs)
