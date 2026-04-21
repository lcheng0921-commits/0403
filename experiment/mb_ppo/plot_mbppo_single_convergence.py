import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

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


def _extract_episode_axis(episode_metrics: List[Dict], train_returns: List[float]) -> np.ndarray:
    if episode_metrics:
        episodes = [int(item.get("episode", i + 1)) for i, item in enumerate(episode_metrics)]
        return np.asarray(episodes, dtype=np.int32)
    if train_returns:
        return np.arange(1, len(train_returns) + 1, dtype=np.int32)
    return np.zeros((0,), dtype=np.int32)


def _extract_metric(episode_metrics: List[Dict], train_returns: List[float], key: str) -> np.ndarray:
    if episode_metrics:
        vals = [float(item.get(key, np.nan)) for item in episode_metrics]
        return np.asarray(vals, dtype=np.float32)
    if key == "train_return" and train_returns:
        return np.asarray(train_returns, dtype=np.float32)
    return np.zeros((0,), dtype=np.float32)


def _extract_first_available(
    episode_metrics: List[Dict],
    train_returns: List[float],
    keys: List[str],
) -> Tuple[np.ndarray, str]:
    for key in keys:
        y = _extract_metric(episode_metrics, train_returns, key)
        if y.size > 0 and np.any(np.isfinite(y)):
            return y, key
    return np.zeros((0,), dtype=np.float32), "none"


def _plot_one(ax, x: np.ndarray, y: np.ndarray, smooth_window: int, title: str, y_label: str, max_episode: int, smoothed_only: bool):
    if x.size == 0 or y.size == 0:
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        return

    if max_episode > 0:
        keep = x <= max_episode
        x = x[keep]
        y = y[keep]

    smooth = _rolling_mean(y, smooth_window)

    if not smoothed_only:
        ax.plot(x, y, linewidth=0.9, alpha=0.18, color="#1f77b4", label="raw")
    ax.plot(x[: smooth.size], smooth, linewidth=2.0, color="#1f77b4", label=f"smooth(w={max(1, smooth_window)})")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Plot MB-PPO single-run convergence under PF-EE objective.")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory under mb_ppo_data.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiment/mb_ppo/pics/fig_conv_mbppo_pf_single_4metrics.png",
        help="Output figure path.",
    )
    parser.add_argument("--smooth-window", type=int, default=25, help="Rolling mean window.")
    parser.add_argument("--max-episode", type=int, default=0, help="If >0, clip x-axis to episodes <= this value.")
    parser.add_argument("--smoothed-only", action="store_true", help="Hide raw traces and only keep smoothed curves.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = repo_root / exp_dir

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in: {exp_dir}")

    config = _load_config(config_path)
    episode_metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    train_returns = _safe_load_pickle(exp_dir / "vars" / "train_returns.pickle") or []

    x = _extract_episode_axis(episode_metrics, train_returns)
    y_pf_ee, pf_key_used = _extract_first_available(
        episode_metrics,
        train_returns,
        ["episode_pf_energy_efficiency", "pf_energy_efficiency"],
    )
    y_ee, ee_key_used = _extract_first_available(
        episode_metrics,
        train_returns,
        ["episode_energy_efficiency", "energy_efficiency"],
    )
    y_jain, jain_key_used = _extract_first_available(
        episode_metrics,
        train_returns,
        ["jain_fairness", "effective_fairness"],
    )
    y_rmin, rmin_key_used = _extract_first_available(
        episode_metrics,
        train_returns,
        ["episode_min_user_rate_avg", "min_user_rate"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    axes = axes.reshape(-1)

    _plot_one(
        axes[0],
        x=x,
        y=y_pf_ee,
        smooth_window=args.smooth_window,
        title=f"PF-EE Convergence ({pf_key_used})",
        y_label="PF-EE",
        max_episode=args.max_episode,
        smoothed_only=args.smoothed_only,
    )
    _plot_one(
        axes[1],
        x=x,
        y=y_ee,
        smooth_window=args.smooth_window,
        title=f"EE Convergence ({ee_key_used})",
        y_label="EE",
        max_episode=args.max_episode,
        smoothed_only=args.smoothed_only,
    )
    _plot_one(
        axes[2],
        x=x,
        y=y_jain,
        smooth_window=args.smooth_window,
        title=f"Jain Fairness Convergence ({jain_key_used})",
        y_label="Jain index",
        max_episode=args.max_episode,
        smoothed_only=args.smoothed_only,
    )
    _plot_one(
        axes[3],
        x=x,
        y=y_rmin,
        smooth_window=args.smooth_window,
        title=f"Weakest-User Rate Convergence ({rmin_key_used})",
        y_label="R_min (Mbps)",
        max_episode=args.max_episode,
        smoothed_only=args.smoothed_only,
    )

    seed = config.get("seed", "unknown")
    baseline = config.get("baseline", "mbppo")
    fig.suptitle(
        f"MB-PPO PF Convergence | baseline={baseline} | seed={seed}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = repo_root / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {save_path}")
    print(f"source exp: {exp_dir}")
    print(f"seed: {seed}")
    print(f"pf metric key: {pf_key_used}")
    print(f"ee metric key: {ee_key_used}")
    print(f"jain metric key: {jain_key_used}")
    print(f"rmin metric key: {rmin_key_used}")


if __name__ == "__main__":
    main()
