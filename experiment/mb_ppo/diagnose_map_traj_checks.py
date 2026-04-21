import argparse
import copy
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as SN

import matplotlib.pyplot as plt
import numpy as np

from algo.mb_ppo.mb_ppo_config import DEFAULT_MB_PPO_CONFIG
from algo.mb_ppo.mb_ppo_learner import MultiBranchPPOLearner
from algo.mb_ppo.run_mbppo import _build_env, train
from algo.mb_ppo.utils import check_args_sanity, load_var, set_rand_seed


def make_clustered_map_class(name: str, range_pos: float, n_gts: int):
    class DiagnosticClusteredMap:
        def __init__(self, range_pos=500, n_eve=4, n_gts=6, n_ubs=1, n_community=4):
            self.n_eve = int(n_eve)
            self.n_gts = int(n_gts)
            self.n_ubs = int(n_ubs)
            self.range_pos = float(range_pos)
            self.n_community = int(max(1, n_community))

            self.pos_eve = np.full((self.n_eve, 2), -200.0, dtype=np.float32)
            self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)
            self.pos_ubs = np.empty((self.n_ubs, 2), dtype=np.float32)

            self.fen = 2
            self.area = self.range_pos / self.fen
            self.gts_in_community = [[] for _ in range(self.n_community)]

        def set_eve(self):
            preset = np.array(
                [[0.2, 0.2], [0.8, 0.8], [-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32
            )
            for i in range(self.n_eve):
                if i < len(preset):
                    if preset[i, 0] < 0.0:
                        self.pos_eve[i] = np.array([-200.0, -200.0], dtype=np.float32)
                    else:
                        self.pos_eve[i] = self.range_pos * preset[i]
                else:
                    self.pos_eve[i] = np.array([-200.0, -200.0], dtype=np.float32)

        def set_gts(self):
            self.gts_in_community = [[] for _ in range(self.n_community)]
            c1 = np.array([0.2 * self.range_pos, 0.2 * self.range_pos], dtype=np.float32)
            c2 = np.array([0.8 * self.range_pos, 0.8 * self.range_pos], dtype=np.float32)
            sigma = np.array([0.06 * self.range_pos, 0.06 * self.range_pos], dtype=np.float32)

            n1 = int(np.ceil(self.n_gts * 0.5))
            n2 = int(self.n_gts - n1)
            clusters = [(c1, n1), (c2, n2)]

            idx = 0
            for center, cnt in clusters:
                for _ in range(cnt):
                    if idx >= self.n_gts:
                        break
                    point = np.random.normal(loc=center, scale=sigma)
                    point = np.clip(point, 0.05 * self.range_pos, 0.95 * self.range_pos)
                    self.pos_gts[idx] = point.astype(np.float32)

                    cid = int((point[1] // self.area) * self.fen + (point[0] // self.area))
                    cid = int(np.clip(cid, 0, self.n_community - 1))
                    self.gts_in_community[cid].append(idx)
                    idx += 1

            while idx < self.n_gts:
                point = np.random.uniform(low=0.05 * self.range_pos, high=0.95 * self.range_pos, size=(2,))
                self.pos_gts[idx] = point.astype(np.float32)
                cid = int((point[1] // self.area) * self.fen + (point[0] // self.area))
                cid = int(np.clip(cid, 0, self.n_community - 1))
                self.gts_in_community[cid].append(idx)
                idx += 1

        def set_ubs(self):
            if self.n_ubs >= 1:
                center = np.array([0.5 * self.range_pos, 0.5 * self.range_pos], dtype=np.float32)
                self.pos_ubs[0] = center
            for i in range(1, self.n_ubs):
                angle = 2.0 * np.pi * i / max(1, self.n_ubs)
                offset = 0.06 * self.range_pos * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                self.pos_ubs[i] = self.pos_ubs[0] + offset

        def get_map(self):
            self.set_eve()
            self.set_gts()
            self.set_ubs()
            return {
                "pos_gts": self.pos_gts,
                "pos_eve": self.pos_eve,
                "pos_ubs": self.pos_ubs,
                "area": self.area,
                "range_pos": self.range_pos,
                "gts_in_community": self.gts_in_community,
            }

    DiagnosticClusteredMap.__name__ = name
    DiagnosticClusteredMap.default_range_pos = float(range_pos)
    DiagnosticClusteredMap.default_n_gts = int(n_gts)
    return DiagnosticClusteredMap


@dataclass
class Scenario:
    name: str
    range_pos: int
    n_gts: int


def pairwise_distance_stats(points: np.ndarray):
    n = int(points.shape[0])
    if n <= 1:
        return 0.0, 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.eye(n, dtype=bool)
    dist_masked = dist.copy()
    dist_masked[mask] = np.inf
    nearest = float(np.mean(np.min(dist_masked, axis=1)))
    diameter = float(np.max(dist))
    return nearest, diameter


def estimate_map_harshness(map_cls, range_pos: int, n_gts: int, samples: int = 200):
    nearest_list = []
    diameter_list = []
    init_uav_dist_mean = []

    for _ in range(samples):
        m = map_cls(range_pos=range_pos, n_eve=0, n_gts=n_gts, n_ubs=1, n_community=4)
        data = m.get_map()
        gts = np.asarray(data["pos_gts"], dtype=np.float32)
        uav0 = np.asarray(data["pos_ubs"][0], dtype=np.float32)

        nearest, diameter = pairwise_distance_stats(gts)
        nearest_list.append(nearest)
        diameter_list.append(diameter)

        d0 = np.linalg.norm(gts - uav0[None, :], axis=1)
        init_uav_dist_mean.append(float(np.mean(d0)))

    return {
        "nearest_mean": float(np.mean(nearest_list)),
        "nearest_std": float(np.std(nearest_list)),
        "diameter_mean": float(np.mean(diameter_list)),
        "diameter_std": float(np.std(diameter_list)),
        "initial_uav_distance_mean": float(np.mean(init_uav_dist_mean)),
        "initial_uav_distance_std": float(np.std(init_uav_dist_mean)),
    }


def scenario_train(base_config: dict, scenario: Scenario, episodes: int, seed: int, output_tag: str):
    map_cls = make_clustered_map_class(
        name=f"DiagMap_{scenario.name}",
        range_pos=float(scenario.range_pos),
        n_gts=int(scenario.n_gts),
    )
    out_dir = f"mb_ppo_data/{output_tag}_{scenario.name}_ep{episodes}_seed{seed}"
    out_path = Path(out_dir)

    existing_metrics = out_path / "vars" / "episode_metrics.pickle"
    if existing_metrics.exists():
        try:
            old_metrics = load_var(str(out_path / "vars" / "episode_metrics"))
            if len(old_metrics) >= int(episodes):
                print(f"\n[Reuse] scenario={scenario.name}, use existing run: {out_dir}")
                return out_path
        except Exception:
            pass

    required_precoding_dim = 2 * (int(scenario.n_gts) + 1) * int(base_config.get("antenna_count", 16))

    kwargs = copy.deepcopy(base_config)
    kwargs.update(
        {
            "seed": int(seed),
            "episodes": int(episodes),
            "eval_interval": 50,
            "save_freq": 100,
            "range_pos": int(scenario.range_pos),
            "n_gts": int(scenario.n_gts),
            "precoding_dim": int(required_precoding_dim),
            "map": map_cls,
            "output_dir": out_dir,
            "save_episode_metrics": True,
        }
    )
    print(f"\n[Train] scenario={scenario.name}, range={scenario.range_pos}, n_gts={scenario.n_gts}, out={out_dir}")
    train(kwargs)
    return Path(out_dir)


def summarize_training(exp_dir: Path):
    metrics = load_var(str(exp_dir / "vars" / "episode_metrics"))
    qos = np.array([float(m.get("qos_violation_sum", 0.0)) for m in metrics], dtype=np.float32)
    lam = np.array([float(m.get("lambda_penalty", 0.0)) for m in metrics], dtype=np.float32)
    vel = np.array([float(m.get("velocity", 0.0)) for m in metrics], dtype=np.float32)

    tail = min(50, len(qos))
    qos_tail = float(np.mean(qos[-tail:]))
    qos_min = float(np.min(qos)) if len(qos) > 0 else 0.0

    if len(qos) >= 20:
        x = np.arange(20, dtype=np.float32)
        y = qos[-20:]
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    corr_lam_vel = 0.0
    if len(lam) > 5 and np.std(lam) > 1e-8 and np.std(vel) > 1e-8:
        corr_lam_vel = float(np.corrcoef(lam, vel)[0, 1])

    return {
        "episodes": int(len(metrics)),
        "qos_violation_sum_last": float(qos[-1]) if len(qos) > 0 else 0.0,
        "qos_violation_sum_tail50_mean": qos_tail,
        "qos_violation_sum_min": qos_min,
        "qos_tail20_slope": slope,
        "lambda_last": float(lam[-1]) if len(lam) > 0 else 0.0,
        "lambda_mean": float(np.mean(lam)) if len(lam) > 0 else 0.0,
        "velocity_mean": float(np.mean(vel)) if len(vel) > 0 else 0.0,
        "corr_lambda_velocity": corr_lam_vel,
        "qos_close_to_zero": bool(qos_tail < 1e-3),
    }


def load_exp_config(exp_dir: Path):
    with open(exp_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    merged = copy.deepcopy(DEFAULT_MB_PPO_CONFIG)
    merged.update(cfg)
    return merged


def prepare_cfg_for_replay(cfg: dict, exp_dir: Path):
    replay_cfg = copy.deepcopy(cfg)

    if not callable(replay_cfg.get("map", None)):
        replay_cfg["map"] = make_clustered_map_class(
            name=f"ReplayMap_{exp_dir.name}",
            range_pos=float(replay_cfg.get("range_pos", 500)),
            n_gts=int(replay_cfg.get("n_gts", 6)),
        )

    dev = str(replay_cfg.get("device", "cpu"))
    if dev.startswith("cuda"):
        replay_cfg["device"] = "cuda"

    return replay_cfg


def build_env_and_learner_from_checkpoint(exp_dir: Path, checkpoint_path: Path):
    cfg = prepare_cfg_for_replay(load_exp_config(exp_dir), exp_dir=exp_dir)

    args = SN(**cfg)
    args = check_args_sanity(args)

    env = _build_env(args)
    env_info = env.get_env_info()
    learner = MultiBranchPPOLearner(env_info=env_info, args=args)
    learner.load_model(str(checkpoint_path))
    return args, env, learner


def rollout_trace(exp_dir: Path, checkpoint_path: Path, steps: int, seed: int):
    set_rand_seed(seed)
    _, env, learner = build_env_and_learner_from_checkpoint(exp_dir=exp_dir, checkpoint_path=checkpoint_path)
    env.set_lambda(learner.lambda_penalty)

    obs, state, init_info = env.reset()
    rows = []
    positions = [env.pos_ubs.copy()]
    weak_user_positions = []

    for step in range(1, steps + 1):
        actions, _, _, _ = learner.take_actions(obs=obs, state=state, deterministic=True)
        obs, state, _, done, info = env.step(actions)

        weak_user = int(np.argmin(env.latest_effective_rate))
        weak_pos = env.pos_gts[weak_user].copy()
        weak_user_positions.append(weak_pos)
        weak_dist = float(np.linalg.norm(env.pos_ubs[0] - weak_pos))

        rows.append(
            {
                "step": step,
                "R_c": float(info.get("rsma_common_rate", 0.0)),
                "R_k": float(info.get("rsma_private_rate", 0.0)),
                "P_tx": float(info.get("tx_power", 0.0)),
                "lambda": float(info.get("lambda_penalty", learner.lambda_penalty)),
                "weak_user_idx": weak_user,
                "weak_user_dist": weak_dist,
                "uav_x": float(env.pos_ubs[0, 0]),
                "uav_y": float(env.pos_ubs[0, 1]),
            }
        )

        positions.append(env.pos_ubs.copy())
        if done:
            break

    return {
        "rows": rows,
        "uav_init": np.asarray(init_info["uav_init_pos"], dtype=np.float32),
        "gts_init": np.asarray(init_info["gts_init_pos"], dtype=np.float32),
        "uav_traj": np.asarray(positions, dtype=np.float32),
        "weak_pos_seq": np.asarray(weak_user_positions, dtype=np.float32),
        "range_pos": float(init_info["range_pos"]),
    }


def summarize_trace(trace_payload: dict):
    rows = trace_payload["rows"]
    if not rows:
        return {"steps": 0, "weak_dist_slope": 0.0, "path_length": 0.0}

    weak_dist = np.array([float(r["weak_user_dist"]) for r in rows], dtype=np.float32)
    if len(weak_dist) >= 2:
        x = np.arange(len(weak_dist), dtype=np.float32)
        slope = float(np.polyfit(x, weak_dist, 1)[0])
    else:
        slope = 0.0

    traj = np.asarray(trace_payload["uav_traj"], dtype=np.float32)[:, 0, :]
    if len(traj) >= 2:
        path_length = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    else:
        path_length = 0.0

    return {
        "steps": int(len(rows)),
        "weak_dist_start": float(weak_dist[0]),
        "weak_dist_end": float(weak_dist[-1]),
        "weak_dist_slope": slope,
        "path_length": path_length,
    }


def write_step_csv(rows, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "R_c", "R_k", "P_tx", "lambda", "weak_user_idx", "weak_user_dist", "uav_x", "uav_y"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_trajectory_compare(early_trace: dict, late_trace: dict, save_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, trace, subtitle in [
        (axes[0], early_trace, "Early checkpoint"),
        (axes[1], late_trace, "Late checkpoint"),
    ]:
        gts = np.asarray(trace["gts_init"], dtype=np.float32)
        traj = np.asarray(trace["uav_traj"], dtype=np.float32)[:, 0, :]
        rng = float(trace["range_pos"])

        ax.scatter(gts[:, 0], gts[:, 1], c="#ff7f0e", s=35, label="users")
        ax.plot(traj[:, 0], traj[:, 1], c="#1f77b4", lw=1.8, label="uav traj")
        ax.scatter(traj[0, 0], traj[0, 1], c="#2ca02c", s=55, marker="s", label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], c="#d62728", s=60, marker="*", label="end")

        ax.set_xlim(0.0, rng)
        ax.set_ylim(0.0, rng)
        ax.set_title(subtitle)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def check_channel_h_norm(exp_dir: Path, seed: int, samples: int = 2000):
    cfg = prepare_cfg_for_replay(load_exp_config(exp_dir), exp_dir=exp_dir)
    args = SN(**cfg)
    args = check_args_sanity(args)

    set_rand_seed(seed)
    env = _build_env(args)
    env.reset()

    norms = []
    powers = []
    for _ in range(samples):
        angle = np.random.uniform(-np.pi, np.pi)
        r = np.random.uniform(1.0, env.range_pos * np.sqrt(2.0))
        delta = np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
        d = float(np.linalg.norm(delta))
        h = env._channel_vector_complex(delta_xy=delta, distance=d, dim=env.antenna_count)
        norms.append(float(np.linalg.norm(h)))
        powers.append(float(np.mean(np.square(np.abs(h)))))

    norms = np.asarray(norms, dtype=np.float32)
    powers = np.asarray(powers, dtype=np.float32)
    is_unit_norm = bool(abs(float(np.mean(norms)) - 1.0) < 0.1 and float(np.std(norms)) < 0.1)

    return {
        "samples": int(samples),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "elem_power_mean": float(np.mean(powers)),
        "elem_power_std": float(np.std(powers)),
        "appears_unit_normed": is_unit_norm,
    }


def check_alpha_rho_init(exp_dir: Path, seed: int):
    cfg = prepare_cfg_for_replay(load_exp_config(exp_dir), exp_dir=exp_dir)
    args = SN(**cfg)
    args = check_args_sanity(args)

    set_rand_seed(seed)
    env = _build_env(args)
    env.reset()

    alpha = np.asarray(env.latest_alpha, dtype=np.float32)
    return {
        "rho_init": float(env.latest_rho),
        "alpha_init_min": float(np.min(alpha)),
        "alpha_init_max": float(np.max(alpha)),
        "alpha_init_sum": float(np.sum(alpha)),
        "tx_power_init_w": float(env.latest_tx_power),
        "tx_power_max_w": float(env.tx_power_max_w),
        "tx_power_init_ratio": float(env.latest_tx_power / max(env.tx_power_max_w, 1e-8)),
        "stream_count": int(env.stream_count),
    }


def pick_checkpoint_pair(exp_dir: Path):
    ckpt_dir = exp_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("checkpoint_episode*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")

    def ep_num(p: Path):
        stem = p.stem
        return int(stem.replace("checkpoint_episode", ""))

    early = ckpts[0]
    for p in ckpts:
        if ep_num(p) >= 100:
            early = p
            break
    late = ckpts[-1]
    return early, late


def main():
    parser = argparse.ArgumentParser(description="Diagnostic checks for map harshness, trajectory behavior, and PHY internals.")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes for each diagnostic scenario.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed used for diagnostic runs.")
    parser.add_argument("--steps", type=int, default=100, help="Number of rollout steps for detailed R_c/R_k/P_tx print.")
    parser.add_argument("--output-tag", type=str, default="diag_maptraj", help="Output prefix for experiment folders.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d")
    output_tag = f"{args.output_tag}_{timestamp}"

    scenarios = [
        Scenario(name="g6_r500", range_pos=500, n_gts=6),
        Scenario(name="g4_r500", range_pos=500, n_gts=4),
        Scenario(name="g3_r300", range_pos=300, n_gts=3),
    ]

    base_config = copy.deepcopy(DEFAULT_MB_PPO_CONFIG)
    results = {
        "meta": {
            "episodes_per_scenario": int(args.episodes),
            "seed": int(args.seed),
            "steps_reported": int(args.steps),
            "output_tag": output_tag,
            "qos_threshold": float(base_config.get("qos_threshold", 0.0)),
            "lambda_lr": float(base_config.get("lambda_lr", 0.0)),
            "lambda_decay_gate_ratio": float(base_config.get("lambda_decay_gate_ratio", 0.0)),
        },
        "map_harshness": {},
        "training_summary": {},
        "trajectory_summary": {},
        "channel_norm_check": {},
        "alpha_rho_init_check": {},
    }

    exp_dirs = {}
    for s in scenarios:
        map_cls = make_clustered_map_class(name=f"DiagMapStats_{s.name}", range_pos=s.range_pos, n_gts=s.n_gts)
        results["map_harshness"][s.name] = estimate_map_harshness(
            map_cls=map_cls,
            range_pos=s.range_pos,
            n_gts=s.n_gts,
            samples=200,
        )

        exp_dir = scenario_train(base_config=base_config, scenario=s, episodes=args.episodes, seed=args.seed, output_tag=output_tag)
        exp_dirs[s.name] = exp_dir
        results["training_summary"][s.name] = summarize_training(exp_dir=exp_dir)

    reference_exp = exp_dirs["g6_r500"]
    early_ckpt, late_ckpt = pick_checkpoint_pair(reference_exp)
    early_trace = rollout_trace(reference_exp, checkpoint_path=early_ckpt, steps=args.steps, seed=args.seed)
    late_trace = rollout_trace(reference_exp, checkpoint_path=late_ckpt, steps=args.steps, seed=args.seed)

    results["trajectory_summary"]["early_checkpoint"] = {
        "checkpoint": str(early_ckpt),
        **summarize_trace(early_trace),
    }
    results["trajectory_summary"]["late_checkpoint"] = {
        "checkpoint": str(late_ckpt),
        **summarize_trace(late_trace),
    }

    diag_dir = reference_exp / "diagnostics"
    write_step_csv(late_trace["rows"], save_path=diag_dir / "first100_steps_rc_rk_ptx.csv")
    plot_trajectory_compare(
        early_trace=early_trace,
        late_trace=late_trace,
        save_path=diag_dir / "trajectory_early_vs_late_lambda.png",
        title="Trajectory check under increasing lambda across training",
    )

    results["channel_norm_check"] = check_channel_h_norm(reference_exp, seed=args.seed, samples=3000)
    results["alpha_rho_init_check"] = check_alpha_rho_init(reference_exp, seed=args.seed)

    print("\n[First 100-step metrics from late checkpoint]")
    for row in late_trace["rows"]:
        print(
            "step={:03d}, R_c={:.6f}, R_k={:.6f}, P_tx={:.6f}, lambda={:.6f}, weak_user={}, weak_dist={:.3f}".format(
                int(row["step"]),
                float(row["R_c"]),
                float(row["R_k"]),
                float(row["P_tx"]),
                float(row["lambda"]),
                int(row["weak_user_idx"]),
                float(row["weak_user_dist"]),
            )
        )

    report_path = Path("mb_ppo_data") / f"{output_tag}_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[Done] Diagnostic summary saved to:", report_path)
    print("[Done] First-100-step CSV saved to:", diag_dir / "first100_steps_rc_rk_ptx.csv")
    print("[Done] Trajectory figure saved to:", diag_dir / "trajectory_early_vs_late_lambda.png")


if __name__ == "__main__":
    main()
