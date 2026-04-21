import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_load_pickle(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "baseline" in data:
        return data
    if isinstance(data, dict) and len(data) == 1:
        nested = next(iter(data.values()))
        if isinstance(nested, dict):
            return nested
    return {}


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    w = int(max(1, window))
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=np.float32) / float(w)
    padded = np.pad(x, (w - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _normalize_baseline(name: str) -> str:
    b = str(name).strip().lower()
    if b == "single_head":
        return "ppo"
    return b


def _load_experiment(repo_root: Path, exp_arg: str) -> Tuple[Path, Dict, List[Dict], List[float]]:
    exp_dir = Path(exp_arg)
    if not exp_dir.is_absolute():
        exp_dir = repo_root / exp_arg
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in: {exp_dir}")

    config = _load_config(config_path)
    episode_metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    train_returns = _safe_load_pickle(exp_dir / "vars" / "train_returns.pickle") or []
    return exp_dir, config, episode_metrics, train_returns


def _extract_episode_axis(episode_metrics: List[Dict], train_returns: List[float]) -> np.ndarray:
    if episode_metrics:
        episodes = [int(item.get("episode", i + 1)) for i, item in enumerate(episode_metrics)]
        return np.asarray(episodes, dtype=np.int32)
    if train_returns:
        return np.arange(1, len(train_returns) + 1, dtype=np.int32)
    return np.zeros((0,), dtype=np.int32)


def _extract_pf_ee(episode_metrics: List[Dict], train_returns: List[float]) -> Tuple[np.ndarray, str]:
    keys = ["episode_pf_energy_efficiency", "pf_energy_efficiency"]
    if episode_metrics:
        for k in keys:
            y = np.asarray([float(item.get(k, np.nan)) for item in episode_metrics], dtype=np.float32)
            if y.size > 0 and np.any(np.isfinite(y)):
                return y, k

    if train_returns:
        return np.asarray(train_returns, dtype=np.float32), "train_return_fallback"
    return np.zeros((0,), dtype=np.float32), "none"


def _print_alignment(records: Dict[str, Dict]):
    check_keys = [
        "seed",
        "episodes",
        "n_gts",
        "episode_length",
        "tx_power_max_dbm",
        "reward_pf_scale",
        "pf_rate_ref_mbps",
        "pf_log_scale",
        "reward_power_penalty_scale",
        "reward_wall_penalty_scale",
        "reward_core_penalty_scale",
        "phy_mapping_blend",
        "precoding_gain_scale",
        "interference_scale",
        "state_h_scale",
        "alpha_common_logit_bias",
    ]

    print("[Check] Alignment summary:")
    for name, payload in records.items():
        cfg = payload["config"]
        baseline = _normalize_baseline(cfg.get("baseline", "unknown"))
        force_hard = bool(cfg.get("force_hard_mapping_for_ppo", False))
        print(f"  - {name}: baseline={baseline}, force_hard_mapping_for_ppo={force_hard}, map={cfg.get('map', 'unknown')}")

    for k in check_keys:
        vals = {}
        for name, payload in records.items():
            v = payload["config"].get(k, None)
            if isinstance(v, float):
                v = round(float(v), 12)
            vals[name] = v
        uniq = {str(v) for v in vals.values()}
        state = "OK" if len(uniq) <= 1 else "MISMATCH"
        print(f"  - {k}: {state} -> {vals}")


def _clip_xy(x: np.ndarray, y: np.ndarray, max_episode: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_episode <= 0:
        return x, y
    keep = x <= max_episode
    return x[keep], y[keep]


def main():
    parser = argparse.ArgumentParser(description="Plot PF-EE convergence comparison for MB-PPO/PPO/SAC.")
    parser.add_argument("--exp-mbppo", type=str, required=True, help="MB-PPO experiment directory.")
    parser.add_argument("--exp-ppo", type=str, required=True, help="PPO(single-head) experiment directory.")
    parser.add_argument("--exp-sac", type=str, required=True, help="SAC experiment directory.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiment/mb_ppo/pics/fig_conv_pf_ee_mbppo_ppo_sac.png",
        help="Output figure path.",
    )
    parser.add_argument("--smooth-window", type=int, default=25, help="Rolling mean window.")
    parser.add_argument("--max-episode", type=int, default=0, help="If >0, clip x-axis to episodes <= this value.")
    parser.add_argument("--smoothed-only", action="store_true", help="Hide raw traces and show only smoothed curves.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    mbppo_dir, mbppo_cfg, mbppo_metrics, mbppo_train_returns = _load_experiment(repo_root, args.exp_mbppo)
    ppo_dir, ppo_cfg, ppo_metrics, ppo_train_returns = _load_experiment(repo_root, args.exp_ppo)
    sac_dir, sac_cfg, sac_metrics, sac_train_returns = _load_experiment(repo_root, args.exp_sac)

    records = {
        "MB-PPO": {"dir": mbppo_dir, "config": mbppo_cfg, "metrics": mbppo_metrics, "train_returns": mbppo_train_returns},
        "PPO(single-head)": {"dir": ppo_dir, "config": ppo_cfg, "metrics": ppo_metrics, "train_returns": ppo_train_returns},
        "SAC": {"dir": sac_dir, "config": sac_cfg, "metrics": sac_metrics, "train_returns": sac_train_returns},
    }
    _print_alignment(records)

    curves = {}
    for name, payload in records.items():
        x = _extract_episode_axis(payload["metrics"], payload["train_returns"])
        y, key_used = _extract_pf_ee(payload["metrics"], payload["train_returns"])
        x, y = _clip_xy(x, y, args.max_episode)
        curves[name] = {"x": x, "y": y, "key": key_used, "dir": payload["dir"]}

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 6.2))
    color_map = {
        "MB-PPO": "#1f77b4",
        "PPO(single-head)": "#2ca02c",
        "SAC": "#ff7f0e",
    }

    for name in ["MB-PPO", "PPO(single-head)", "SAC"]:
        item = curves[name]
        x = item["x"]
        y = item["y"]
        if x.size == 0 or y.size == 0:
            continue

        smooth = _rolling_mean(y, args.smooth_window)
        color = color_map.get(name, None)
        if not args.smoothed_only:
            ax.plot(x, y, linewidth=0.85, alpha=0.16, color=color)
        ax.plot(x[: smooth.size], smooth, linewidth=2.2, color=color, label=f"{name} ({item['key']})")

    ax.set_xlabel("Episodes")
    ax.set_ylabel("PF-EE")
    ax.set_title("Convergence Comparison on PF-EE: MB-PPO vs PPO(single-head) vs SAC")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = repo_root / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {save_path}")
    print(f"MB-PPO source: {mbppo_dir}")
    print(f"PPO source: {ppo_dir}")
    print(f"SAC source: {sac_dir}")


if __name__ == "__main__":
    main()