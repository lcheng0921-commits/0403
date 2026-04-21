import argparse
import copy
from pathlib import Path
from types import SimpleNamespace as SN

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError("gymnasium is required for SAC baseline training.") from exc

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
except Exception as exc:  # pragma: no cover
    raise ImportError("stable-baselines3 is required for SAC baseline training.") from exc

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

from algo.mb_ppo.mb_ppo_config import DEFAULT_MB_PPO_CONFIG
from algo.mb_ppo.utils import check_args_sanity, save_config, save_var, set_rand_seed
from mb_ppo_env.mb_ppo_environment import MbPpoEnv


def _create_output_dir(repo_root: Path, requested_output_dir: str = "") -> Path:
    data_root = repo_root / "mb_ppo_data"
    data_root.mkdir(parents=True, exist_ok=True)

    if requested_output_dir:
        exp_dir = Path(requested_output_dir)
        if not exp_dir.is_absolute():
            exp_dir = repo_root / requested_output_dir
        if exp_dir.exists() and (exp_dir / "config.json").exists():
            raise RuntimeError(f"Requested output directory already contains an experiment: {exp_dir}")
        (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        (exp_dir / "vars").mkdir(parents=True, exist_ok=True)
        return exp_dir

    for i in range(1, 1000):
        exp_dir = data_root / f"exp{i}"
        try:
            exp_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue

        (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
        (exp_dir / "logs").mkdir(parents=True, exist_ok=False)
        (exp_dir / "vars").mkdir(parents=True, exist_ok=False)
        return exp_dir

    raise RuntimeError("No available experiment directory under mb_ppo_data.")


def _build_env(args: SN) -> MbPpoEnv:
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
        reward_pf_scale=args.reward_pf_scale,
        pf_rate_ref_mbps=args.pf_rate_ref_mbps,
        pf_log_scale=args.pf_log_scale,
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


class Sb3MbPpoPfEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, raw_env: MbPpoEnv):
        super().__init__()
        self.raw_env = raw_env

        self.traj_dim = 2
        self.common_precoding_dim = int(self.raw_env.common_precoding_dim)
        self.private_precoding_dim = int(self.raw_env.private_precoding_dim)
        self.resource_dim = int(self.raw_env.resource_dim)
        self.action_dim = self.traj_dim + self.common_precoding_dim + self.private_precoding_dim + self.resource_dim

        obs, _, init_info = self.raw_env.reset()
        self._last_init_info = init_info
        flat_obs = self._flatten_obs(obs)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_obs.shape[0],),
            dtype=np.float32,
        )

    @staticmethod
    def _flatten_obs(obs):
        other = obs[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        gt = obs[1].detach().cpu().numpy().astype(np.float32).reshape(-1)
        return np.concatenate([other, gt], axis=0).astype(np.float32)

    def _split_action(self, action: np.ndarray):
        x = np.asarray(action, dtype=np.float32).reshape(-1)
        idx = 0

        traj = x[idx : idx + self.traj_dim]
        idx += self.traj_dim

        common_precoding = x[idx : idx + self.common_precoding_dim]
        idx += self.common_precoding_dim

        private_precoding = x[idx : idx + self.private_precoding_dim]
        idx += self.private_precoding_dim

        resource = x[idx : idx + self.resource_dim]

        return {
            "traj": traj.reshape(1, self.traj_dim),
            "common_precoding": common_precoding.reshape(1, self.common_precoding_dim),
            "private_precoding": private_precoding.reshape(1, self.private_precoding_dim),
            "resource": resource.reshape(1, self.resource_dim),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        obs, _, init_info = self.raw_env.reset()
        self._last_init_info = init_info
        return self._flatten_obs(obs), init_info

    def step(self, action):
        action_dict = self._split_action(action)
        obs, _, reward, done, info = self.raw_env.step(action_dict)
        reward_scalar = float(np.mean(reward))
        terminated = bool(done)
        truncated = False
        return self._flatten_obs(obs), reward_scalar, terminated, truncated, info


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.episodes_done = 0
        self._running_returns = None
        self.train_returns = []
        self.episode_metrics = []

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._running_returns = np.zeros(n_envs, dtype=np.float64)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if isinstance(infos, dict):
            infos = [infos]
        if isinstance(dones, (bool, np.bool_)):
            dones = [bool(dones)]
        if isinstance(rewards, (float, np.floating)):
            rewards = [float(rewards)]

        for idx, done in enumerate(dones):
            reward = float(rewards[idx]) if idx < len(rewards) else 0.0
            if self._running_returns is not None and idx < len(self._running_returns):
                self._running_returns[idx] += reward

            if not done:
                continue

            info = infos[idx] if idx < len(infos) else {}

            ep_return = 0.0
            if self._running_returns is not None and idx < len(self._running_returns):
                ep_return = float(self._running_returns[idx])
                self._running_returns[idx] = 0.0

            self.episodes_done += 1
            self.train_returns.append(ep_return)

            metrics = {
                "episode": int(self.episodes_done),
                "train_return": ep_return,
                "pf_energy_efficiency": float(info.get("pf_energy_efficiency", 0.0)),
                "energy_efficiency": float(info.get("energy_efficiency", 0.0)),
                "log_utility_raw": float(info.get("log_utility_raw", 0.0)),
                "log_utility": float(info.get("log_utility", 0.0)),
                "log_utility_scale": float(info.get("log_utility_scale", 0.0)),
                "pf_rate_ref_mbps": float(info.get("pf_rate_ref_mbps", 0.0)),
                "jain_fairness": float(info.get("jain_fairness", info.get("effective_fairness", 0.0))),
                "min_user_rate": float(info.get("min_user_rate", 0.0)),
                "episode_pf_energy_efficiency": float(info.get("episode_pf_energy_efficiency", 0.0)),
                "episode_energy_efficiency": float(info.get("episode_energy_efficiency", 0.0)),
                "episode_min_user_rate_avg": float(info.get("episode_min_user_rate_avg", 0.0)),
                "episode_total_energy": float(info.get("episode_total_energy", 0.0)),
                "total_throughput": float(info.get("total_throughput", 0.0)),
                "tx_power": float(info.get("tx_power", 0.0)),
                "tx_power_violation": float(info.get("tx_power_violation", 0.0)),
                "wall_violation": float(info.get("wall_violation", 0.0)),
                "wall_penalty": float(info.get("wall_penalty", 0.0)),
                "core_violation": float(info.get("core_violation", 0.0)),
                "core_penalty": float(info.get("core_penalty", 0.0)),
                "core_penalty_factor": float(info.get("core_penalty_factor", 1.0)),
                "effective_core_penalty_scale": float(info.get("effective_core_penalty_scale", 0.0)),
                "core_violation_steps": int(info.get("core_violation_steps", 0)),
                "core_terminate_enabled": bool(info.get("core_terminate_enabled", False)),
                "core_terminate_start_episode": int(info.get("core_terminate_start_episode", 1)),
                "velocity": float(info.get("velocity", 0.0)),
                "user_mean_serving_dist": float(info.get("user_mean_serving_dist", 0.0)),
                "beam_quality": float(info.get("beam_quality", 0.0)),
                "common_beam_norm": float(info.get("common_beam_norm", 0.0)),
                "private_beam_norm_mean": float(info.get("private_beam_norm_mean", 0.0)),
                "private_beam_norm_std": float(info.get("private_beam_norm_std", 0.0)),
                "private_beam_norm_max_dev": float(info.get("private_beam_norm_max_dev", 0.0)),
                "alpha_common": float(info.get("alpha_common", 0.0)),
                "gamma_common_min": float(info.get("gamma_common_min", 0.0)),
                "gamma_private_mean": float(info.get("gamma_private_mean", 0.0)),
                "common_interference": float(info.get("common_interference", 0.0)),
                "private_interference": float(info.get("private_interference", 0.0)),
                "rsma_common_rate": float(info.get("rsma_common_rate", 0.0)),
                "rsma_private_rate": float(info.get("rsma_private_rate", 0.0)),
            }
            self.episode_metrics.append(metrics)

            self.writer.add_scalar("train/episode_return", ep_return, self.episodes_done)
            self.writer.add_scalar("train/pf_energy_efficiency", metrics["pf_energy_efficiency"], self.episodes_done)
            self.writer.add_scalar("train/energy_efficiency", metrics["energy_efficiency"], self.episodes_done)
            self.writer.add_scalar("train/log_utility_raw", metrics["log_utility_raw"], self.episodes_done)
            self.writer.add_scalar("train/log_utility", metrics["log_utility"], self.episodes_done)
            self.writer.add_scalar("train/log_utility_scale", metrics["log_utility_scale"], self.episodes_done)
            self.writer.add_scalar("train/pf_rate_ref_mbps", metrics["pf_rate_ref_mbps"], self.episodes_done)
            self.writer.add_scalar("train/jain_fairness", metrics["jain_fairness"], self.episodes_done)
            self.writer.add_scalar("train/min_user_rate", metrics["min_user_rate"], self.episodes_done)
            self.writer.add_scalar("train/episode_pf_energy_efficiency", metrics["episode_pf_energy_efficiency"], self.episodes_done)
            self.writer.add_scalar("train/episode_energy_efficiency", metrics["episode_energy_efficiency"], self.episodes_done)
            self.writer.add_scalar("train/episode_min_user_rate_avg", metrics["episode_min_user_rate_avg"], self.episodes_done)
            self.writer.add_scalar("train/episode_total_energy", metrics["episode_total_energy"], self.episodes_done)
            self.writer.add_scalar("train/total_throughput", metrics["total_throughput"], self.episodes_done)
            self.writer.add_scalar("train/tx_power", metrics["tx_power"], self.episodes_done)
            self.writer.add_scalar("train/tx_power_violation", metrics["tx_power_violation"], self.episodes_done)
            self.writer.add_scalar("train/wall_violation", metrics["wall_violation"], self.episodes_done)
            self.writer.add_scalar("train/wall_penalty", metrics["wall_penalty"], self.episodes_done)
            self.writer.add_scalar("train/core_violation", metrics["core_violation"], self.episodes_done)
            self.writer.add_scalar("train/core_penalty", metrics["core_penalty"], self.episodes_done)
            self.writer.add_scalar("train/core_penalty_factor", metrics["core_penalty_factor"], self.episodes_done)
            self.writer.add_scalar("train/effective_core_penalty_scale", metrics["effective_core_penalty_scale"], self.episodes_done)
            self.writer.add_scalar("train/core_terminate_enabled", float(metrics["core_terminate_enabled"]), self.episodes_done)
            self.writer.add_scalar("train/core_violation_steps", metrics["core_violation_steps"], self.episodes_done)
            self.writer.add_scalar("train/beam_quality", metrics["beam_quality"], self.episodes_done)
            self.writer.add_scalar("train/common_beam_norm", metrics["common_beam_norm"], self.episodes_done)
            self.writer.add_scalar("train/private_beam_norm_mean", metrics["private_beam_norm_mean"], self.episodes_done)
            self.writer.add_scalar("train/private_beam_norm_std", metrics["private_beam_norm_std"], self.episodes_done)
            self.writer.add_scalar("train/private_beam_norm_max_dev", metrics["private_beam_norm_max_dev"], self.episodes_done)
            self.writer.add_scalar("train/user_mean_serving_dist", metrics["user_mean_serving_dist"], self.episodes_done)
            self.writer.add_scalar("train/alpha_common", metrics["alpha_common"], self.episodes_done)
            self.writer.add_scalar("train/gamma_common_min", metrics["gamma_common_min"], self.episodes_done)
            self.writer.add_scalar("train/gamma_private_mean", metrics["gamma_private_mean"], self.episodes_done)
            self.writer.add_scalar("train/common_interference", metrics["common_interference"], self.episodes_done)
            self.writer.add_scalar("train/private_interference", metrics["private_interference"], self.episodes_done)
            self.writer.add_scalar("train/rsma_common_rate", metrics["rsma_common_rate"], self.episodes_done)
            self.writer.add_scalar("train/rsma_private_rate", metrics["rsma_private_rate"], self.episodes_done)

        return True


def evaluate_policy(
    eval_env: Sb3MbPpoPfEnv,
    model,
    num_eval_episodes: int,
    seed_base: int = 0,
    record_trace: bool = False,
):
    ep_returns = []
    ep_pf_ee = []
    ep_ee = []
    ep_fairness = []
    ep_min_user_rate = []
    ep_total_energy = []
    ep_tx_power = []
    trace_payload = None

    for episode_idx in range(num_eval_episodes):
        obs, init_info = eval_env.reset(seed=seed_base + episode_idx)
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
        ep_pf_ee.append(float(last_info.get("episode_pf_energy_efficiency", last_info.get("pf_energy_efficiency", 0.0))))
        ep_ee.append(float(last_info.get("episode_energy_efficiency", last_info.get("energy_efficiency", 0.0))))
        ep_fairness.append(float(last_info.get("jain_fairness", last_info.get("effective_fairness", 0.0))))
        ep_min_user_rate.append(float(last_info.get("episode_min_user_rate_avg", last_info.get("min_user_rate", 0.0))))
        ep_total_energy.append(float(last_info.get("episode_total_energy", 0.0)))
        ep_tx_power.append(float(last_info.get("tx_power", 0.0)))

        if record_trace and episode_idx == 0:
            trace_payload = {
                "trajectory": np.asarray(trajectory, dtype=np.float32),
                "uav_init_pos": np.asarray(init_info["uav_init_pos"], dtype=np.float32),
                "gts_init_pos": np.asarray(init_info["gts_init_pos"], dtype=np.float32),
                "range_pos": float(init_info["range_pos"]),
            }

    metrics = {
        "eval_return_mean": float(np.mean(ep_returns)),
        "eval_return_std": float(np.std(ep_returns)),
        "eval_pf_energy_efficiency": float(np.mean(ep_pf_ee)),
        "eval_energy_efficiency": float(np.mean(ep_ee)),
        "eval_jain_fairness": float(np.mean(ep_fairness)),
        "eval_min_user_rate": float(np.mean(ep_min_user_rate)),
        "eval_total_energy": float(np.mean(ep_total_energy)),
        "eval_tx_power": float(np.mean(ep_tx_power)),
    }
    if trace_payload is not None:
        metrics["trace_payload"] = trace_payload
    return metrics


def _build_model(env, args: SN):
    policy_kwargs = {
        "net_arch": {
            "pi": [int(args.sac_hidden_size), int(args.sac_hidden_size)],
            "qf": [int(args.sac_hidden_size), int(args.sac_hidden_size)],
        }
    }
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=float(args.lr),
        buffer_size=int(args.sac_buffer_size),
        learning_starts=max(int(args.sac_learning_starts), int(args.episode_length)),
        batch_size=int(args.mini_batch_size),
        tau=float(args.sac_tau),
        gamma=float(args.gamma),
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=int(args.seed),
        device=str(args.device),
    )


def _save_training_vars(output_dir: Path, callback: EpisodeMetricsCallback, eval_history, eval_traces, save_episode_metrics: bool):
    save_var(str(output_dir / "vars" / "train_returns"), callback.train_returns)
    save_var(str(output_dir / "vars" / "eval_history"), eval_history)
    if save_episode_metrics:
        save_var(str(output_dir / "vars" / "episode_metrics"), callback.episode_metrics)
        save_var(str(output_dir / "vars" / "eval_traces"), eval_traces)


def train_sac_pf(train_kwargs=None):
    train_kwargs = train_kwargs or {}

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = _create_output_dir(repo_root, requested_output_dir=train_kwargs.get("output_dir", ""))

    config = copy.deepcopy(DEFAULT_MB_PPO_CONFIG)
    config.update(train_kwargs)

    if config["map"] is None:
        from experiment.mb_ppo.maps import ClusteredMap500

        config["map"] = ClusteredMap500

    config["baseline"] = "sac"
    config["single_head"] = False
    config.setdefault("sac_hidden_size", 176)
    config.setdefault("sac_buffer_size", 200000)
    config.setdefault("sac_learning_starts", 1000)
    config.setdefault("sac_tau", 0.005)

    args = SN(**config)
    args.output_dir = str(output_dir)
    args = check_args_sanity(args)

    save_config(output_dir=args.output_dir, config=args.__dict__)
    set_rand_seed(args.seed)

    train_env = Sb3MbPpoPfEnv(_build_env(args))
    eval_env = Sb3MbPpoPfEnv(_build_env(args))

    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    callback = EpisodeMetricsCallback(writer=writer)
    model = _build_model(env=train_env, args=args)

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
            save_path = output_dir / "checkpoints" / f"checkpoint_episode{next_save}.zip"
            model.save(str(save_path))
            _save_training_vars(output_dir, callback, eval_history, eval_traces, args.save_episode_metrics)
            next_save += save_freq

        if callback.episodes_done >= target_episode:
            eval_stats = evaluate_policy(
                eval_env=eval_env,
                model=model,
                num_eval_episodes=int(args.num_eval_episodes),
                seed_base=int(args.seed * 1000 + callback.episodes_done),
                record_trace=True,
            )
            trace_payload = eval_stats.pop("trace_payload", None)
            eval_stats["episode"] = int(callback.episodes_done)
            eval_history.append(eval_stats)
            if trace_payload is not None:
                eval_traces.append({"episode": int(callback.episodes_done), **trace_payload})

            writer.add_scalar("eval/return_mean", eval_stats["eval_return_mean"], callback.episodes_done)
            writer.add_scalar("eval/return_std", eval_stats["eval_return_std"], callback.episodes_done)
            writer.add_scalar("eval/pf_energy_efficiency", eval_stats["eval_pf_energy_efficiency"], callback.episodes_done)
            writer.add_scalar("eval/energy_efficiency", eval_stats["eval_energy_efficiency"], callback.episodes_done)
            writer.add_scalar("eval/jain_fairness", eval_stats["eval_jain_fairness"], callback.episodes_done)
            writer.add_scalar("eval/min_user_rate", eval_stats["eval_min_user_rate"], callback.episodes_done)
            writer.add_scalar("eval/total_energy", eval_stats["eval_total_energy"], callback.episodes_done)
            writer.add_scalar("eval/tx_power", eval_stats["eval_tx_power"], callback.episodes_done)

            print(
                "Episode {} | train_return={:.4f} | eval_return={:.4f} | eval_pf_ee={:.6f} | eval_ee={:.6f} | eval_jain={:.4f} | eval_rmin={:.6f}".format(
                    callback.episodes_done,
                    callback.train_returns[-1] if callback.train_returns else 0.0,
                    eval_stats["eval_return_mean"],
                    eval_stats["eval_pf_energy_efficiency"],
                    eval_stats["eval_energy_efficiency"],
                    eval_stats["eval_jain_fairness"],
                    eval_stats["eval_min_user_rate"],
                )
            )

            next_eval += eval_interval

    final_ep = int(callback.episodes_done)
    final_save = output_dir / "checkpoints" / f"checkpoint_episode{final_ep}.zip"
    model.save(str(final_save))
    _save_training_vars(output_dir, callback, eval_history, eval_traces, args.save_episode_metrics)

    writer.close()
    print("Complete SAC(PF) training.")


def build_train_kwargs_from_cli(cli_args):
    map_cls = None
    if cli_args.map_class:
        from experiment.mb_ppo import maps as maps_module

        if not hasattr(maps_module, cli_args.map_class):
            raise ValueError(f"Unknown map class: {cli_args.map_class}")
        map_cls = getattr(maps_module, cli_args.map_class)

    return {
        "output_dir": cli_args.output_dir,
        "episodes": int(cli_args.episodes),
        "eval_interval": int(cli_args.eval_interval),
        "save_freq": int(cli_args.save_freq),
        "seed": int(cli_args.seed),
        "map": map_cls,
        "n_gts": int(cli_args.n_gts),
        "tx_power_max_dbm": float(cli_args.tx_power_max_dbm),
        "reward_pf_scale": float(cli_args.reward_pf_scale),
        "pf_rate_ref_mbps": float(cli_args.pf_rate_ref_mbps),
        "pf_log_scale": float(cli_args.pf_log_scale),
        "reward_power_penalty_scale": float(cli_args.reward_power_penalty_scale),
        "reward_wall_penalty_scale": float(cli_args.reward_wall_penalty_scale),
        "reward_core_penalty_scale": float(cli_args.reward_core_penalty_scale),
        "core_boundary_margin_ratio": float(cli_args.core_boundary_margin_ratio),
        "terminate_on_core_violation": bool(cli_args.terminate_on_core_violation),
        "core_violation_terminate_patience": int(cli_args.core_violation_terminate_patience),
        "core_violation_terminate_threshold": float(cli_args.core_violation_terminate_threshold),
        "core_terminate_start_episode": int(cli_args.core_terminate_start_episode),
        "phy_mapping_blend": float(cli_args.phy_mapping_blend),
        "precoding_gain_scale": float(cli_args.precoding_gain_scale),
        "interference_scale": float(cli_args.interference_scale),
        "state_h_scale": float(cli_args.state_h_scale),
        "alpha_common_logit_bias": float(cli_args.alpha_common_logit_bias),
        "freeze_uav_trajectory": bool(cli_args.freeze_uav_trajectory),
        "fixed_precoding_scheme": str(cli_args.fixed_precoding_scheme),
        "fixed_beta_mode": str(cli_args.fixed_beta_mode),
        "lr": float(cli_args.lr),
        "gamma": float(cli_args.gamma),
        "mini_batch_size": int(cli_args.mini_batch_size),
        "sac_hidden_size": int(cli_args.sac_hidden_size),
        "sac_buffer_size": int(cli_args.sac_buffer_size),
        "sac_learning_starts": int(cli_args.sac_learning_starts),
        "sac_tau": float(cli_args.sac_tau),
        "save_episode_metrics": not bool(cli_args.no_save_episode_metrics),
        "force_hard_mapping_for_ppo": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SAC baseline on PF-EE objective with MB-PPO environment.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_MB_PPO_CONFIG["episodes"], help="Number of training episodes.")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluate every N episodes.")
    parser.add_argument("--save-freq", type=int, default=200, help="Save checkpoint every N episodes.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="", help="Optional explicit output directory (absolute or repo-relative).")
    parser.add_argument("--map-class", type=str, default="", help="Map class name under experiment.mb_ppo.maps (e.g. ClusteredMap500Split42DenseA).")
    parser.add_argument("--n-gts", type=int, default=DEFAULT_MB_PPO_CONFIG["n_gts"], help="Number of ground users.")

    parser.add_argument("--tx-power-max-dbm", type=float, default=DEFAULT_MB_PPO_CONFIG["tx_power_max_dbm"], help="Maximum transmit power in dBm.")
    parser.add_argument("--reward-pf-scale", type=float, default=DEFAULT_MB_PPO_CONFIG["reward_pf_scale"], help="Scale factor for PF-EE reward.")
    parser.add_argument("--pf-rate-ref-mbps", type=float, default=DEFAULT_MB_PPO_CONFIG["pf_rate_ref_mbps"], help="Rate reference used in log utility.")
    parser.add_argument("--pf-log-scale", type=float, default=DEFAULT_MB_PPO_CONFIG["pf_log_scale"], help="Scale factor for summed log utility.")
    parser.add_argument("--reward-power-penalty-scale", type=float, default=DEFAULT_MB_PPO_CONFIG["reward_power_penalty_scale"], help="Penalty coefficient for TX power violation ratio.")
    parser.add_argument("--reward-wall-penalty-scale", type=float, default=DEFAULT_MB_PPO_CONFIG["reward_wall_penalty_scale"], help="Penalty coefficient for wall overshoot.")
    parser.add_argument("--reward-core-penalty-scale", type=float, default=DEFAULT_MB_PPO_CONFIG["reward_core_penalty_scale"], help="Penalty coefficient for core-region violation.")
    parser.add_argument("--core-boundary-margin-ratio", type=float, default=DEFAULT_MB_PPO_CONFIG["core_boundary_margin_ratio"], help="Core service-zone margin ratio.")
    parser.add_argument("--terminate-on-core-violation", action=argparse.BooleanOptionalAction, default=DEFAULT_MB_PPO_CONFIG["terminate_on_core_violation"], help="Enable early stop when core violation persists.")
    parser.add_argument("--core-violation-terminate-patience", type=int, default=DEFAULT_MB_PPO_CONFIG["core_violation_terminate_patience"], help="Consecutive violating steps required before early termination.")
    parser.add_argument("--core-violation-terminate-threshold", type=float, default=DEFAULT_MB_PPO_CONFIG["core_violation_terminate_threshold"], help="Core violation threshold for early termination.")
    parser.add_argument("--core-terminate-start-episode", type=int, default=DEFAULT_MB_PPO_CONFIG["core_terminate_start_episode"], help="Episode index from which core early termination is active.")

    parser.add_argument("--phy-mapping-blend", type=float, default=DEFAULT_MB_PPO_CONFIG["phy_mapping_blend"], help="Blend ratio between base and physical-mapped rates.")
    parser.add_argument("--precoding-gain-scale", type=float, default=DEFAULT_MB_PPO_CONFIG["precoding_gain_scale"], help="Gain scaling used in physical mapping.")
    parser.add_argument("--interference-scale", type=float, default=DEFAULT_MB_PPO_CONFIG["interference_scale"], help="Interference scaling used in physical mapping.")
    parser.add_argument("--state-h-scale", type=float, default=DEFAULT_MB_PPO_CONFIG["state_h_scale"], help="Scale factor applied to CSI features in state/obs.")
    parser.add_argument("--alpha-common-logit-bias", type=float, default=DEFAULT_MB_PPO_CONFIG["alpha_common_logit_bias"], help="Bias added to common-stream alpha logit before softmax mapping.")

    parser.add_argument("--freeze-uav-trajectory", action=argparse.BooleanOptionalAction, default=DEFAULT_MB_PPO_CONFIG["freeze_uav_trajectory"], help="If enabled, UAV trajectory is fixed.")
    parser.add_argument("--fixed-precoding-scheme", type=str, default=DEFAULT_MB_PPO_CONFIG["fixed_precoding_scheme"], choices=["none", "zf"], help="Fix precoding direction by closed-form scheme.")
    parser.add_argument("--fixed-beta-mode", type=str, default=DEFAULT_MB_PPO_CONFIG["fixed_beta_mode"], choices=["none", "uniform"], help="Fix beta allocation mode.")

    parser.add_argument("--lr", type=float, default=DEFAULT_MB_PPO_CONFIG["lr"], help="SAC learning rate.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_MB_PPO_CONFIG["gamma"], help="Discount factor.")
    parser.add_argument("--mini-batch-size", type=int, default=DEFAULT_MB_PPO_CONFIG["mini_batch_size"], help="Batch size used for SAC updates.")
    parser.add_argument("--sac-hidden-size", type=int, default=176, help="Hidden width for SAC policy and Q networks.")
    parser.add_argument("--sac-buffer-size", type=int, default=200000, help="Replay buffer size.")
    parser.add_argument("--sac-learning-starts", type=int, default=1000, help="Number of steps before SAC gradient updates start.")
    parser.add_argument("--sac-tau", type=float, default=0.005, help="Target network soft-update coefficient.")

    parser.add_argument("--no-save-episode-metrics", action="store_true", help="Disable saving per-episode metric files.")
    cli_args = parser.parse_args()

    kwargs = build_train_kwargs_from_cli(cli_args)
    train_sac_pf(kwargs)


if __name__ == "__main__":
    main()