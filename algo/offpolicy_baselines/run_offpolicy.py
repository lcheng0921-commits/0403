import copy
from pathlib import Path
from types import SimpleNamespace as SN
from typing import Dict, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for TD3/SAC baselines. "
        "Please install it in conda env uav_rsma."
    ) from exc

try:
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required for TD3/SAC baselines. "
        "Please install it in conda env uav_rsma."
    ) from exc

try:
    from tensorboardX import SummaryWriter
except Exception:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        class SummaryWriter:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

from algo.mha_mb_ppo.mb_ppo_config import DEFAULT_MB_PPO_CONFIG
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


def _build_env_kwargs(args: SN) -> Dict:
    return {
        'map': args.map,
        'fair_service': args.fair_service,
        'range_pos': args.range_pos,
        'episode_length': args.episode_length,
        'n_ubs': args.n_ubs,
        'n_powers': args.n_powers,
        'n_moves': args.n_moves,
        'n_gts': args.n_gts,
        'n_eve': args.n_eve,
        'r_cov': args.r_cov,
        'r_sense': args.r_sense,
        'n_community': args.n_community,
        'K_richan': args.K_richan,
        'jamming_power_bound': args.jamming_power_bound,
        'velocity_bound': args.velocity_bound,
        'qos_threshold': args.qos_threshold,
        'tx_power_max_dbm': args.tx_power_max_dbm,
        'reward_ee_scale': args.reward_ee_scale,
        'reward_qos_scale': args.reward_qos_scale,
        'lambda_penalty': args.lambda_init,
        'rsma_mode': args.rsma_mode,
        'baseline': args.baseline,
        'precoding_dim': args.precoding_dim,
        'antenna_count': args.antenna_count,
        'csi_complex_dim': args.csi_complex_dim,
        'fly_power_base': args.fly_power_base,
        'fly_power_coeff': args.fly_power_coeff,
        'phy_mapping_blend': args.phy_mapping_blend,
        'precoding_gain_scale': args.precoding_gain_scale,
        'interference_scale': args.interference_scale,
    }


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


class Sb3MbPpoEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, env_kwargs: Dict):
        super().__init__()
        self.raw_env = MbPpoEnv(**env_kwargs)
        self.lambda_penalty = float(env_kwargs.get('lambda_penalty', 1.0))
        self.raw_env.set_lambda(self.lambda_penalty)

        self.n_gts = self.raw_env.n_gts
        self.precoding_dim = int(env_kwargs.get('precoding_dim', 32))
        self.power_dim = self.n_gts + 2
        self.rate_dim = self.n_gts
        self.traj_dim = 2
        self.action_dim = self.traj_dim + self.precoding_dim + self.power_dim + self.rate_dim

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        obs, _, init_info = self.raw_env.reset()
        self._last_init_info = init_info
        obs_vec = self._flatten_obs(obs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_vec.shape[0],),
            dtype=np.float32,
        )

    @staticmethod
    def _flatten_obs(obs):
        other = obs[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        gt = obs[1].detach().cpu().numpy().astype(np.float32).reshape(-1)
        return np.concatenate([other, gt], axis=0).astype(np.float32)

    def _split_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.asarray(action, dtype=np.float32).reshape(-1)
        idx = 0

        traj = x[idx : idx + self.traj_dim]
        idx += self.traj_dim

        precoding = x[idx : idx + self.precoding_dim]
        idx += self.precoding_dim

        power = x[idx : idx + self.power_dim]
        idx += self.power_dim

        rate = x[idx : idx + self.rate_dim]

        return {
            'traj': traj.reshape(1, self.traj_dim),
            'precoding': precoding.reshape(1, self.precoding_dim),
            'power': power.reshape(1, self.power_dim),
            'rate': rate.reshape(1, self.rate_dim),
        }

    def set_lambda(self, value: float):
        self.lambda_penalty = float(value)
        self.raw_env.set_lambda(self.lambda_penalty)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        obs, _, init_info = self.raw_env.reset()
        self._last_init_info = init_info
        return self._flatten_obs(obs), init_info

    def step(self, action):
        # Keep dual variable in sync with wrapper attribute updated by callback.
        self.raw_env.set_lambda(float(self.lambda_penalty))
        action_dict = self._split_action(action)
        obs, _, reward, done, info = self.raw_env.step(action_dict)
        reward_scalar = float(np.mean(reward))
        terminated = bool(done)
        truncated = False
        return self._flatten_obs(obs), reward_scalar, terminated, truncated, info


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, writer, lambda_init: float, lambda_lr: float, qos_threshold: float):
        super().__init__()
        self.writer = writer
        self.lambda_penalty = float(lambda_init)
        self.lambda_lr = float(lambda_lr)
        self.qos_threshold = float(max(1e-6, qos_threshold))

        self.episodes_done = 0
        self.train_returns = []
        self.episode_metrics = []
        self._qos_dual_signal_sum = None
        self._qos_steps = None

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, 'num_envs', 1))
        self._qos_dual_signal_sum = np.zeros(n_envs, dtype=np.float32)
        self._qos_steps = np.zeros(n_envs, dtype=np.int32)
        self.training_env.set_attr('lambda_penalty', self.lambda_penalty)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])

        for idx, done in enumerate(dones):
            info = infos[idx] if idx < len(infos) else {}
            qos_gap = float(info.get('qos_violation', 0.0))
            qos_gap_sum = float(info.get('qos_violation_sum', qos_gap))
            qos_dual = float(info.get('qos_dual_signal', qos_gap))
            qos_dual_sum = float(info.get('qos_dual_signal_sum', qos_dual))
            qos_gap_norm = float(info.get('qos_violation_norm', qos_gap / self.qos_threshold))
            qos_dual_norm = float(info.get('qos_dual_signal_norm', qos_dual / self.qos_threshold))

            if self._qos_dual_signal_sum is not None and idx < len(self._qos_dual_signal_sum):
                self._qos_dual_signal_sum[idx] += qos_dual_sum
                self._qos_steps[idx] += 1

            if not done:
                continue

            ep_info = info.get('episode', {})
            ep_raw = ep_info.get('r', info.get('EpRet', 0.0))
            ep_return = float(np.mean(np.asarray(ep_raw, dtype=np.float32)))

            ep_qos_dual_signal = qos_dual_sum
            if self._qos_dual_signal_sum is not None and idx < len(self._qos_dual_signal_sum):
                steps = int(max(1, self._qos_steps[idx]))
                ep_qos_dual_signal = float(self._qos_dual_signal_sum[idx] / steps)
                self._qos_dual_signal_sum[idx] = 0.0
                self._qos_steps[idx] = 0

            self.lambda_penalty = max(0.0, self.lambda_penalty + self.lambda_lr * ep_qos_dual_signal)
            self.training_env.set_attr('lambda_penalty', self.lambda_penalty)

            self.episodes_done += 1
            self.train_returns.append(ep_return)

            metrics = {
                'episode': int(self.episodes_done),
                'train_return': ep_return,
                'qos_violation': qos_gap,
                'qos_violation_sum': qos_gap_sum,
                'qos_violation_norm': qos_gap_norm,
                'qos_dual_signal': qos_dual,
                'qos_dual_signal_sum': qos_dual_sum,
                'qos_dual_signal_norm': qos_dual_norm,
                'episode_qos_dual_signal': ep_qos_dual_signal,
                'weighted_qos_gap': float(info.get('weighted_qos_gap', 0.0)),
                'weighted_qos_gap_norm': float(info.get('weighted_qos_gap_norm', qos_gap_norm)),
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
                'lambda_penalty': float(self.lambda_penalty),
            }
            self.episode_metrics.append(metrics)

            self.writer.add_scalar('train/episode_return', ep_return, self.episodes_done)
            self.writer.add_scalar('train/lambda_penalty', self.lambda_penalty, self.episodes_done)
            self.writer.add_scalar('train/qos_violation', qos_gap, self.episodes_done)
            self.writer.add_scalar('train/qos_violation_sum', qos_gap_sum, self.episodes_done)
            self.writer.add_scalar('train/qos_violation_norm', qos_gap_norm, self.episodes_done)
            self.writer.add_scalar('train/qos_dual_signal', qos_dual, self.episodes_done)
            self.writer.add_scalar('train/qos_dual_signal_sum', qos_dual_sum, self.episodes_done)
            self.writer.add_scalar('train/qos_dual_signal_norm', qos_dual_norm, self.episodes_done)
            self.writer.add_scalar('train/episode_qos_dual_signal', ep_qos_dual_signal, self.episodes_done)
            self.writer.add_scalar('train/energy_efficiency', metrics['energy_efficiency'], self.episodes_done)
            self.writer.add_scalar('train/fairness', metrics['effective_fairness'], self.episodes_done)

        return True


def _build_model(algo: str, env, args: SN):
    policy_kwargs = {'net_arch': [256, 256]}

    if algo == 'td3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return TD3(
            'MlpPolicy',
            env,
            learning_rate=args.lr,
            buffer_size=200000,
            learning_starts=max(1000, args.episode_length),
            batch_size=256,
            tau=0.005,
            gamma=args.gamma,
            train_freq=(1, 'step'),
            gradient_steps=1,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=args.seed,
            device=args.device,
        )

    if algo == 'sac':
        return SAC(
            'MlpPolicy',
            env,
            learning_rate=args.lr,
            buffer_size=200000,
            learning_starts=max(1000, args.episode_length),
            batch_size=256,
            tau=0.005,
            gamma=args.gamma,
            train_freq=(1, 'step'),
            gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=args.seed,
            device=args.device,
        )

    raise ValueError(f'Unsupported algorithm: {algo}')


def evaluate_policy(eval_env: Sb3MbPpoEnv, model, num_eval_episodes: int, lambda_penalty: float, seed_base: int = 0, record_trace: bool = False):
    ep_returns = []
    ep_qos = []
    ep_qos_norm = []
    ep_qos_dual = []
    ep_qos_dual_norm = []
    ep_ee = []
    ep_fairness = []
    ep_tx_power = []
    trace_payload = None

    for ep_idx in range(num_eval_episodes):
        eval_env.set_lambda(lambda_penalty)
        obs, init_info = eval_env.reset(seed=seed_base + ep_idx)
        done = False
        ep_ret = 0.0
        trajectory = [eval_env.raw_env.pos_ubs.copy()]
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            trajectory.append(eval_env.raw_env.pos_ubs.copy())
            last_info = info

        ep_returns.append(ep_ret)
        ep_qos.append(float(last_info.get('qos_violation', 0.0)))
        ep_qos_norm.append(float(last_info.get('qos_violation_norm', 0.0)))
        ep_qos_dual.append(float(last_info.get('qos_dual_signal', 0.0)))
        ep_qos_dual_norm.append(float(last_info.get('qos_dual_signal_norm', 0.0)))
        ep_ee.append(float(last_info.get('energy_efficiency', 0.0)))
        ep_fairness.append(float(last_info.get('effective_fairness', 0.0)))
        ep_tx_power.append(float(last_info.get('tx_power', 0.0)))

        if record_trace and ep_idx == 0:
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


def _save_training_vars(output_dir: Path, callback: EpisodeMetricsCallback, eval_history, eval_traces, save_episode_metrics: bool):
    save_var(str(output_dir / 'vars' / 'train_returns'), callback.train_returns)
    save_var(str(output_dir / 'vars' / 'eval_history'), eval_history)
    if save_episode_metrics:
        save_var(str(output_dir / 'vars' / 'episode_metrics'), callback.episode_metrics)
        save_var(str(output_dir / 'vars' / 'eval_traces'), eval_traces)


def train_offpolicy(algo: str, train_kwargs=None):
    train_kwargs = train_kwargs or {}

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = _create_output_dir(repo_root, requested_output_dir=train_kwargs.get('output_dir', ''))

    config = copy.deepcopy(DEFAULT_MB_PPO_CONFIG)
    config.update(train_kwargs)
    config['baseline'] = algo
    _apply_fixed_qos_threshold(config)

    if config['map'] is None:
        from experiment.mb_ppo.maps import ClusteredMap500
        config['map'] = ClusteredMap500

    args = SN(**config)
    args.output_dir = str(output_dir)
    args = check_args_sanity(args)

    save_config(output_dir=args.output_dir, config=args.__dict__)
    set_rand_seed(args.seed)

    env_kwargs = _build_env_kwargs(args)
    train_env = Sb3MbPpoEnv(env_kwargs)
    eval_env = Sb3MbPpoEnv(env_kwargs)

    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    callback = EpisodeMetricsCallback(
        writer=writer,
        lambda_init=args.lambda_init,
        lambda_lr=args.lambda_lr,
        qos_threshold=args.qos_threshold,
    )

    model = _build_model(algo=algo, env=train_env, args=args)

    total_episodes = int(args.episodes)
    eval_interval = max(1, int(args.eval_interval))
    save_freq = max(1, int(args.save_freq))
    episode_len = int(args.episode_length)

    eval_history = []
    eval_traces = []

    next_eval = eval_interval
    next_save = save_freq

    while callback.episodes_done < total_episodes:
        target_episode = min(total_episodes, next_eval)
        remaining = target_episode - callback.episodes_done
        if remaining <= 0:
            break

        model.learn(
            total_timesteps=remaining * episode_len,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        while next_save <= callback.episodes_done:
            save_path = output_dir / 'checkpoints' / f'checkpoint_episode{next_save}.zip'
            model.save(str(save_path))
            _save_training_vars(output_dir, callback, eval_history, eval_traces, args.save_episode_metrics)
            next_save += save_freq

        if callback.episodes_done >= target_episode:
            eval_stats = evaluate_policy(
                eval_env=eval_env,
                model=model,
                num_eval_episodes=int(args.num_eval_episodes),
                lambda_penalty=float(callback.lambda_penalty),
                seed_base=int(args.seed * 1000 + callback.episodes_done),
                record_trace=True,
            )
            trace_payload = eval_stats.pop('trace_payload', None)
            eval_stats['episode'] = int(callback.episodes_done)
            eval_history.append(eval_stats)
            if trace_payload is not None:
                eval_traces.append({'episode': int(callback.episodes_done), **trace_payload})

            writer.add_scalar('eval/return_mean', eval_stats['eval_return_mean'], callback.episodes_done)
            writer.add_scalar('eval/return_std', eval_stats['eval_return_std'], callback.episodes_done)
            writer.add_scalar('eval/qos_violation', eval_stats['eval_qos_violation'], callback.episodes_done)
            writer.add_scalar('eval/qos_violation_norm', eval_stats['eval_qos_violation_norm'], callback.episodes_done)
            writer.add_scalar('eval/qos_dual_signal', eval_stats['eval_qos_dual_signal'], callback.episodes_done)
            writer.add_scalar('eval/qos_dual_signal_norm', eval_stats['eval_qos_dual_signal_norm'], callback.episodes_done)
            writer.add_scalar('eval/energy_efficiency', eval_stats['eval_energy_efficiency'], callback.episodes_done)
            writer.add_scalar('eval/fairness', eval_stats['eval_fairness'], callback.episodes_done)
            writer.add_scalar('eval/tx_power', eval_stats['eval_tx_power'], callback.episodes_done)

            print(
                'Episode {} | train_return={:.4f} | eval_return={:.4f} | qos_gap={:.4f} | qos_dual={:.4f} | fair={:.4f} | lambda={:.4f}'.format(
                    callback.episodes_done,
                    callback.train_returns[-1] if callback.train_returns else 0.0,
                    eval_stats['eval_return_mean'],
                    eval_stats['eval_qos_violation'],
                    eval_stats['eval_qos_dual_signal'],
                    eval_stats['eval_fairness'],
                    callback.lambda_penalty,
                )
            )

            next_eval += eval_interval

    final_ep = int(callback.episodes_done)
    final_save = output_dir / 'checkpoints' / f'checkpoint_episode{final_ep}.zip'
    model.save(str(final_save))
    _save_training_vars(output_dir, callback, eval_history, eval_traces, args.save_episode_metrics)

    writer.close()
    print(f'Complete {algo.upper()} training.')


def build_train_kwargs_from_cli(cli_args):
    algo = cli_args.algo.lower().strip()
    if algo not in ['td3', 'sac']:
        raise ValueError(f'Unsupported algo: {algo}')

    return {
        'baseline': algo,
        'rsma_mode': 'rsma',
        'output_dir': cli_args.output_dir,
        'episodes': cli_args.episodes,
        'eval_interval': cli_args.eval_interval,
        'save_freq': cli_args.save_freq,
        'seed': cli_args.seed,
        'lambda_init': cli_args.lambda_init,
        'lambda_lr': cli_args.lambda_lr,
        'qos_threshold': cli_args.qos_threshold,
        'paper_fixed_qos_threshold': cli_args.paper_fixed_rth,
        'enforce_fixed_qos_threshold': not cli_args.disable_fixed_rth,
        'tx_power_max_dbm': cli_args.tx_power_max_dbm,
        'phy_mapping_blend': cli_args.phy_mapping_blend,
        'precoding_gain_scale': cli_args.precoding_gain_scale,
        'interference_scale': cli_args.interference_scale,
        'antenna_count': cli_args.antenna_count,
        'csi_complex_dim': cli_args.csi_complex_dim,
        'save_episode_metrics': not cli_args.no_save_episode_metrics,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train TD3/SAC baselines on MB-PPO environment.')
    parser.add_argument('--algo', type=str, required=True, choices=['td3', 'sac'], help='Off-policy baseline algorithm.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes.')
    parser.add_argument('--eval-interval', type=int, default=20, help='Evaluate every N episodes.')
    parser.add_argument('--save-freq', type=int, default=100, help='Save checkpoint every N episodes.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--lambda-init', type=float, default=0.1, help='Initial dual variable value.')
    parser.add_argument('--lambda-lr', type=float, default=DEFAULT_MB_PPO_CONFIG['lambda_lr'], help='Dual update step size for lambda.')
    parser.add_argument('--output-dir', type=str, default='', help='Optional explicit output directory (absolute or repo-relative).')
    parser.add_argument('--qos-threshold', type=float, default=0.1, help='Requested QoS threshold; ignored when fixed R_th is enabled.')
    parser.add_argument('--paper-fixed-rth', type=float, default=DEFAULT_MB_PPO_CONFIG['paper_fixed_qos_threshold'], help='Fixed paper R_th used when fixed mode is enabled.')
    parser.add_argument('--disable-fixed-rth', action='store_true', help='Disable fixed R_th enforcement and use --qos-threshold directly.')
    parser.add_argument('--tx-power-max-dbm', type=float, default=30.0, help='Maximum transmit power in dBm.')
    parser.add_argument('--phy-mapping-blend', type=float, default=0.7, help='Blend ratio between base and mapped rates.')
    parser.add_argument('--precoding-gain-scale', type=float, default=1.0, help='Gain scaling in physical mapping.')
    parser.add_argument('--interference-scale', type=float, default=1.0, help='Interference scaling in physical mapping.')
    parser.add_argument('--antenna-count', type=int, default=DEFAULT_MB_PPO_CONFIG['antenna_count'], help='Number of transmit antennas used for beam-direction mapping.')
    parser.add_argument('--csi-complex-dim', type=int, default=4, help='Per-user CSI complex dimension for observation.')
    parser.add_argument('--no-save-episode-metrics', action='store_true', help='Disable saving per-episode metric files.')
    cli_args = parser.parse_args()

    kwargs = build_train_kwargs_from_cli(cli_args)
    train_offpolicy(algo=cli_args.algo, train_kwargs=kwargs)


if __name__ == '__main__':
    main()
