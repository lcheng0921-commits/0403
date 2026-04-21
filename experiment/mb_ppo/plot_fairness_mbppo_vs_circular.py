import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace as SN

import matplotlib.pyplot as plt
import numpy as np
from torch import load as torch_load

from algo.mb_ppo.mb_ppo_learner import MultiBranchPPOLearner
from algo.mb_ppo.run_mbppo import _build_env
from algo.mb_ppo.utils import check_args_sanity, set_rand_seed


def _load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "baseline" in data:
        return data
    if isinstance(data, dict) and len(data) == 1:
        nested = next(iter(data.values()))
        if isinstance(nested, dict):
            return nested
    return {}


def _resolve_map_class(map_name: str):
    from experiment.mb_ppo import maps as maps_mod

    if hasattr(maps_mod, map_name):
        return getattr(maps_mod, map_name)
    return maps_mod.ClusteredMap500


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    all_ckpt = sorted(ckpt_dir.glob("checkpoint_episode*.pt"))
    if not all_ckpt:
        raise FileNotFoundError(f"No checkpoint files found under {ckpt_dir}")

    def ep_num(path: Path):
        name = path.stem.replace("checkpoint_episode", "")
        try:
            return int(name)
        except Exception:
            return -1

    all_ckpt = sorted(all_ckpt, key=ep_num)
    return all_ckpt[-1]


def _build_args(config: dict, map_cls, baseline: str):
    cfg = copy.deepcopy(config)
    cfg["map"] = map_cls
    cfg["baseline"] = str(baseline)

    # Keep architecture flags compatible with trained checkpoint.
    if cfg["baseline"] == "ppo":
        cfg["single_head"] = True
    else:
        cfg["single_head"] = bool(config.get("single_head", False))

    dev = str(cfg.get("device", "cpu"))
    if dev.startswith("cuda"):
        cfg["device"] = "cuda"

    args = SN(**cfg)
    args = check_args_sanity(args)
    return args


def _rollout_fairness(env, learner, episodes: int):
    ep_final = []
    ep_step_mean = []

    for _ in range(int(episodes)):
        env.set_lambda(learner.lambda_penalty)
        obs, state, _ = env.reset()
        done = False
        fair_seq = []

        while not done:
            actions, _, _, _ = learner.take_actions(obs=obs, state=state, deterministic=True)
            obs, state, _, done, info = env.step(actions)
            fair_seq.append(float(info.get("effective_fairness", 0.0)))

        ep_final.append(float(info.get("effective_fairness", 0.0)))
        ep_step_mean.append(float(np.mean(fair_seq)) if fair_seq else 0.0)

    return {
        "final_mean": float(np.mean(ep_final)),
        "final_std": float(np.std(ep_final)),
        "step_mean_mean": float(np.mean(ep_step_mean)),
        "step_mean_std": float(np.std(ep_step_mean)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare fairness: optimized MB-PPO trajectory vs fixed circular trajectory.")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory under mb_ppo_data.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint filename. Empty means latest checkpoint.")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes for fairness averaging.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for replay.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiment/mb_ppo/pics/fig_fairness_mbppo_vs_circular.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = repo_root / exp_dir

    cfg_path = exp_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {exp_dir}")

    config = _load_config(cfg_path)
    map_name = str(config.get("map", "ClusteredMap500"))
    map_cls = _resolve_map_class(map_name)

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_path = (ckpt_dir / args.checkpoint) if args.checkpoint else _latest_checkpoint(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    set_rand_seed(int(args.seed))

    args_opt = _build_args(config=config, map_cls=map_cls, baseline="mbppo")
    env_opt = _build_env(args_opt)
    env_info = env_opt.get_env_info()

    learner = MultiBranchPPOLearner(env_info=env_info, args=args_opt)
    ckpt = torch_load(str(ckpt_path), map_location="cpu")
    learner.actor_critic.load_state_dict(ckpt["model_state_dict"])
    learner.lambda_penalty = float(ckpt.get("lambda_penalty", learner.lambda_penalty))
    learner.actor_critic.eval()

    args_circ = _build_args(config=config, map_cls=map_cls, baseline="circular")
    env_circ = _build_env(args_circ)

    res_opt = _rollout_fairness(env=env_opt, learner=learner, episodes=int(args.episodes))
    res_circ = _rollout_fairness(env=env_circ, learner=learner, episodes=int(args.episodes))

    labels = ["Optimized trajectory (MB-PPO)", "Fixed circular trajectory"]
    vals = [res_opt["step_mean_mean"], res_circ["step_mean_mean"]]
    errs = [res_opt["step_mean_std"], res_circ["step_mean_std"]]
    colors = ["#1d3557", "#f4a261"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.arange(len(labels), dtype=np.float32)
    bars = ax.bar(x, vals, yerr=errs, color=colors, width=0.62, alpha=0.95, capsize=4)

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() * 0.5,
            b.get_height() + 0.01,
            f"{vals[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Average fairness per episode")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Fairness Comparison: Optimized vs Fixed Circular")
    ax.grid(axis="y", alpha=0.25)

    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = repo_root / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {save_path}")
    print(f"source exp: {exp_dir}")
    print(f"checkpoint: {ckpt_path.name}")
    print(f"episodes: {int(args.episodes)}")
    print(f"optimized_step_mean_fairness: {res_opt['step_mean_mean']:.6f} +- {res_opt['step_mean_std']:.6f}")
    print(f"circular_step_mean_fairness: {res_circ['step_mean_mean']:.6f} +- {res_circ['step_mean_std']:.6f}")


if __name__ == "__main__":
    main()
