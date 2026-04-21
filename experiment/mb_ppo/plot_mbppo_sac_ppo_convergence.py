import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExpRecord:
    exp_dir: Path
    baseline: str
    config: Dict
    episode_metrics: List[Dict]
    train_returns: List[float]


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


def _discover_experiment_dirs(repo_root: Path) -> List[Path]:
    data_root = repo_root / "mb_ppo_data"
    if not data_root.exists():
        return []

    dirs = []
    for p in data_root.iterdir():
        if p.is_dir() and (p / "config.json").exists():
            dirs.append(p)
    return sorted(dirs, key=lambda p: p.stat().st_mtime)


def _normalize_baseline(name: str) -> str:
    b = str(name).strip().lower()
    if b == "single_head":
        return "ppo"
    return b


def _load_record(exp_dir: Path) -> Optional[ExpRecord]:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return None

    config = _load_config(config_path)
    baseline = _normalize_baseline(config.get("baseline", "unknown"))

    episode_metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    train_returns = _safe_load_pickle(exp_dir / "vars" / "train_returns.pickle") or []

    return ExpRecord(
        exp_dir=exp_dir,
        baseline=baseline,
        config=config,
        episode_metrics=episode_metrics,
        train_returns=train_returns,
    )


def _read_seed(rec: ExpRecord) -> Optional[int]:
    seed = rec.config.get("seed", None)
    if seed is None:
        return None
    try:
        return int(seed)
    except Exception:
        return None


def _read_effective_qos_threshold(rec: ExpRecord) -> Optional[float]:
    enforce = bool(rec.config.get("enforce_fixed_qos_threshold", False))
    key = "paper_fixed_qos_threshold" if enforce else "qos_threshold"
    val = rec.config.get(key, rec.config.get("qos_threshold", None))
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _validate_alignment(records: Dict[str, ExpRecord]) -> None:
    seeds = {name: _read_seed(rec) for name, rec in records.items()}
    qos = {name: _read_effective_qos_threshold(rec) for name, rec in records.items()}

    seed_values = {v for v in seeds.values() if v is not None}
    qos_values = {round(v, 12) for v in qos.values() if v is not None}

    print("[Check] seed by baseline:", seeds)
    print("[Check] effective_qos_threshold by baseline:", qos)

    if len(seed_values) > 1:
        print("[Warn] Seed mismatch detected. Please retrain with the same --seed for fair comparison.")
    if len(qos_values) > 1:
        print("[Warn] QoS threshold mismatch detected. Please keep effective R_th identical.")


def _load_by_arg(repo_root: Path, target: str) -> Optional[ExpRecord]:
    candidate = Path(target)
    exp_dir = candidate if candidate.is_absolute() else (repo_root / candidate)
    if not exp_dir.exists():
        return None
    return _load_record(exp_dir)


def _pick_latest_by_baseline(repo_root: Path, baseline: str) -> Optional[ExpRecord]:
    want = _normalize_baseline(baseline)
    latest = None
    latest_mtime = -1.0

    for exp_dir in _discover_experiment_dirs(repo_root):
        rec = _load_record(exp_dir)
        if rec is None or rec.baseline != want:
            continue
        mtime = exp_dir.stat().st_mtime
        if latest is None or mtime > latest_mtime:
            latest = rec
            latest_mtime = mtime

    return latest


def _extract_episode_axis(rec: ExpRecord) -> np.ndarray:
    if rec.episode_metrics:
        episodes = [int(item.get("episode", i + 1)) for i, item in enumerate(rec.episode_metrics)]
        return np.asarray(episodes, dtype=np.int32)
    if rec.train_returns:
        return np.arange(1, len(rec.train_returns) + 1, dtype=np.int32)
    return np.zeros((0,), dtype=np.int32)


def _extract_metric(rec: ExpRecord, key: str) -> np.ndarray:
    if rec.episode_metrics:
        vals = [float(item.get(key, np.nan)) for item in rec.episode_metrics]
        return np.asarray(vals, dtype=np.float32)
    if key == "train_return" and rec.train_returns:
        return np.asarray(rec.train_returns, dtype=np.float32)
    return np.zeros((0,), dtype=np.float32)


def _extract_qos_violation_metric(rec: ExpRecord) -> np.ndarray:
    for key in [
        "qos_violation_sum_rollout_mean",
        "qos_violation_norm_rollout_mean",
        "qos_violation_sq_sum",
        "qos_violation_sum",
        "qos_violation",
    ]:
        y = _extract_metric(rec, key)
        if y.size > 0 and np.any(np.isfinite(y)):
            return y
    return np.zeros((0,), dtype=np.float32)


def _plot_panel(
    ax,
    records,
    metric_getter,
    title: str,
    y_label: str,
    smooth_window: int,
    max_episode: int,
    smoothed_only: bool,
):
    colors = {
        "MB-PPO": "#1f77b4",
        "SAC": "#ff7f0e",
        "PPO": "#2ca02c",
        "PPO+Physical": "#d62728",
    }

    for pretty_name, rec in records.items():
        x = _extract_episode_axis(rec)
        y = metric_getter(rec)

        if max_episode > 0:
            keep = x <= max_episode
            x = x[keep]
            y = y[keep]

        if x.size == 0 or y.size == 0:
            continue

        color = colors.get(pretty_name, None)
        smoothed = _rolling_mean(y, smooth_window)
        if not smoothed_only:
            ax.plot(x, y, alpha=0.15, linewidth=0.9, color=color)
        ax.plot(x[: smoothed.size], smoothed, linewidth=2.0, color=color, label=f"{pretty_name} ({rec.exp_dir.name})")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Plot MB-PPO/SAC/PPO four-metric convergence curves.")
    parser.add_argument("--exp-mbppo", type=str, default="", help="MB-PPO experiment folder path.")
    parser.add_argument("--exp-sac", type=str, default="", help="SAC experiment folder path.")
    parser.add_argument("--exp-ppo", type=str, default="", help="PPO experiment folder path.")
    parser.add_argument("--exp-ppo-physical", type=str, default="", help="PPO with hard physical mapping experiment folder path.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiment/mb_ppo/pics/fig_convergence_mbppo_sac_ppo_4metrics.png",
        help="Output figure path.",
    )
    parser.add_argument("--smooth-window", type=int, default=15, help="Rolling mean window for smoothing.")
    parser.add_argument(
        "--max-episode",
        type=int,
        default=0,
        help="If >0, clip all curves to episodes <= this value.",
    )
    parser.add_argument(
        "--smoothed-only",
        action="store_true",
        help="Plot only smoothed curves and hide raw noisy traces.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    mbppo_rec = _load_by_arg(repo_root, args.exp_mbppo) if args.exp_mbppo else _pick_latest_by_baseline(repo_root, "mbppo")
    sac_rec = _load_by_arg(repo_root, args.exp_sac) if args.exp_sac else _pick_latest_by_baseline(repo_root, "sac")
    ppo_rec = _load_by_arg(repo_root, args.exp_ppo) if args.exp_ppo else _pick_latest_by_baseline(repo_root, "ppo")
    ppo_physical_rec = _load_by_arg(repo_root, args.exp_ppo_physical) if args.exp_ppo_physical else None

    if mbppo_rec is None or sac_rec is None or ppo_rec is None:
        print("Cannot find MB-PPO, SAC and PPO records at the same time.")
        print("Tip: pass --exp-mbppo --exp-sac --exp-ppo explicitly.")
        return

    records = {
        "MB-PPO": mbppo_rec,
        "SAC": sac_rec,
        "PPO": ppo_rec,
    }
    if ppo_physical_rec is not None:
        records["PPO+Physical"] = ppo_physical_rec

    _validate_alignment(records)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    axes = axes.reshape(-1)

    _plot_panel(
        axes[0],
        records,
        metric_getter=lambda rec: _extract_metric(rec, "energy_efficiency"),
        title="Energy Efficiency vs Episode",
        y_label="EE",
        smooth_window=args.smooth_window,
        max_episode=args.max_episode,
        smoothed_only=args.smoothed_only,
    )
    _plot_panel(
        axes[1],
        records,
        metric_getter=_extract_qos_violation_metric,
        title="QoS Violation vs Episode",
        y_label="QoS violation amount",
        smooth_window=args.smooth_window,
        max_episode=args.max_episode,
        smoothed_only=args.smoothed_only,
    )
    _plot_panel(
        axes[2],
        records,
        metric_getter=lambda rec: _extract_metric(rec, "lambda_penalty"),
        title="Dual Variable lambda vs Episode",
        y_label="lambda",
        smooth_window=args.smooth_window,
        max_episode=args.max_episode,
        smoothed_only=args.smoothed_only,
    )
    _plot_panel(
        axes[3],
        records,
        metric_getter=lambda rec: _extract_metric(rec, "train_return"),
        title="Total Reward vs Episode",
        y_label="Total reward",
        smooth_window=args.smooth_window,
        max_episode=args.max_episode,
        smoothed_only=args.smoothed_only,
    )

    if "PPO+Physical" in records:
        fig.suptitle("Convergence Comparison: MB-PPO vs SAC vs PPO vs PPO+Physical", fontsize=13)
    else:
        fig.suptitle("Convergence Comparison: MB-PPO vs SAC vs PPO", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = repo_root / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {save_path}")
    if args.max_episode > 0:
        print(f"Clipped to max episode: {args.max_episode}")
    if args.smoothed_only:
        print("Plot mode: smoothed-only")
    print(f"MB-PPO source: {mbppo_rec.exp_dir}")
    print(f"SAC source: {sac_rec.exp_dir}")
    print(f"PPO source: {ppo_rec.exp_dir}")
    if ppo_physical_rec is not None:
        print(f"PPO+Physical source: {ppo_physical_rec.exp_dir}")


if __name__ == "__main__":
    main()
