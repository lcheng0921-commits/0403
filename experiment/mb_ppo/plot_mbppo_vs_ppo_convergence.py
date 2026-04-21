import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExpRecord:
    exp_dir: Path
    exp_id: int
    baseline: str
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


def _discover_experiments(repo_root: Path) -> List[Path]:
    data_root = repo_root / "mb_ppo_data"
    exp_dirs = [p for p in data_root.glob("exp*") if p.is_dir() and p.name[3:].isdigit()]
    return sorted(exp_dirs, key=lambda p: int(p.name[3:]))


def _load_record(exp_dir: Path) -> Optional[ExpRecord]:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return None

    config = _load_config(config_path)
    baseline = str(config.get("baseline", "unknown"))
    if baseline == "single_head":
        baseline = "ppo"

    episode_metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    train_returns = _safe_load_pickle(exp_dir / "vars" / "train_returns.pickle") or []

    exp_name = exp_dir.name
    exp_id = -1
    if exp_name.startswith("exp") and exp_name[3:].isdigit():
        exp_id = int(exp_name[3:])

    return ExpRecord(
        exp_dir=exp_dir,
        exp_id=exp_id,
        baseline=baseline,
        episode_metrics=episode_metrics,
        train_returns=train_returns,
    )


def _pick_latest_by_baseline(repo_root: Path, baseline: str) -> Optional[ExpRecord]:
    latest: Optional[ExpRecord] = None
    for exp_dir in _discover_experiments(repo_root):
        rec = _load_record(exp_dir)
        if rec is None:
            continue
        if rec.baseline != baseline:
            continue
        if latest is None or rec.exp_id > latest.exp_id:
            latest = rec
    return latest


def _load_by_arg(repo_root: Path, target: str) -> Optional[ExpRecord]:
    target = target.strip()
    if target.isdigit():
        exp_dir = repo_root / "mb_ppo_data" / f"exp{int(target)}"
    else:
        candidate = Path(target)
        exp_dir = candidate if candidate.is_absolute() else (repo_root / candidate)

    if not exp_dir.exists():
        return None
    return _load_record(exp_dir)


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
        arr = np.asarray(vals, dtype=np.float32)
        return arr

    if key == "train_return" and rec.train_returns:
        return np.asarray(rec.train_returns, dtype=np.float32)

    return np.zeros((0,), dtype=np.float32)


def _extract_qos_violation_metric(rec: ExpRecord) -> np.ndarray:
    # Prefer rollout-level QoS statistics first to align with lambda-update interpretation.
    for k in [
        "qos_violation_sum_rollout_mean",
        "qos_violation_norm_rollout_mean",
        "qos_violation_sq_sum",
        "qos_violation_sum",
        "qos_violation",
    ]:
        y = _extract_metric(rec, k)
        if y.size > 0 and np.any(np.isfinite(y)):
            return y
    return np.zeros((0,), dtype=np.float32)


def _make_panel(
    ax,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    label1: str,
    label2: str,
    title: str,
    y_label: str,
    smooth_window: int,
):
    if y1.size > 0:
        s1 = _rolling_mean(y1, smooth_window)
        ax.plot(x1, y1, alpha=0.18, linewidth=1.0, color="#1f77b4")
        ax.plot(x1[: s1.size], s1, linewidth=2.0, color="#1f77b4", label=label1)
    if y2.size > 0:
        s2 = _rolling_mean(y2, smooth_window)
        ax.plot(x2, y2, alpha=0.18, linewidth=1.0, color="#d62728")
        ax.plot(x2[: s2.size], s2, linewidth=2.0, color="#d62728", label=label2)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()


def _prepare_label(rec: ExpRecord, pretty: str) -> str:
    if rec.exp_id >= 0:
        return f"{pretty} (exp{rec.exp_id})"
    return f"{pretty} ({rec.exp_dir.name})"


def _resolve_records(repo_root: Path, args) -> Tuple[Optional[ExpRecord], Optional[ExpRecord]]:
    mbppo_rec = _load_by_arg(repo_root, args.exp_mbppo) if args.exp_mbppo else _pick_latest_by_baseline(repo_root, "mbppo")
    ppo_rec = _load_by_arg(repo_root, args.exp_ppo) if args.exp_ppo else _pick_latest_by_baseline(repo_root, "ppo")
    return mbppo_rec, ppo_rec


def main():
    parser = argparse.ArgumentParser(description="Plot MB-PPO vs PPO four-metric convergence curves.")
    parser.add_argument("--exp-mbppo", type=str, default="", help="MB-PPO experiment id or folder path.")
    parser.add_argument("--exp-ppo", type=str, default="", help="PPO experiment id or folder path.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiment/mb_ppo/pics/fig_convergence_mbppo_vs_ppo_4metrics.png",
        help="Output figure path.",
    )
    parser.add_argument("--smooth-window", type=int, default=15, help="Rolling mean window for smoothing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mbppo_rec, ppo_rec = _resolve_records(repo_root, args)

    if mbppo_rec is None or ppo_rec is None:
        print("Cannot find both MB-PPO and PPO experiment records.")
        print("Tip: pass --exp-mbppo and --exp-ppo explicitly (exp id or folder path).")
        return

    x_mbppo = _extract_episode_axis(mbppo_rec)
    x_ppo = _extract_episode_axis(ppo_rec)

    y_ee_mbppo = _extract_metric(mbppo_rec, "energy_efficiency")
    y_ee_ppo = _extract_metric(ppo_rec, "energy_efficiency")

    y_qos_mbppo = _extract_qos_violation_metric(mbppo_rec)
    y_qos_ppo = _extract_qos_violation_metric(ppo_rec)

    y_lambda_mbppo = _extract_metric(mbppo_rec, "lambda_penalty")
    y_lambda_ppo = _extract_metric(ppo_rec, "lambda_penalty")

    y_ret_mbppo = _extract_metric(mbppo_rec, "train_return")
    y_ret_ppo = _extract_metric(ppo_rec, "train_return")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axes = axes.reshape(-1)

    label_mbppo = _prepare_label(mbppo_rec, "MB-PPO")
    label_ppo = _prepare_label(ppo_rec, "PPO")

    _make_panel(
        axes[0],
        x_mbppo,
        y_ee_mbppo,
        x_ppo,
        y_ee_ppo,
        label_mbppo,
        label_ppo,
        "Energy Efficiency vs Episode",
        "EE",
        smooth_window=args.smooth_window,
    )
    _make_panel(
        axes[1],
        x_mbppo,
        y_qos_mbppo,
        x_ppo,
        y_qos_ppo,
        label_mbppo,
        label_ppo,
        "QoS Violation vs Episode",
        "QoS violation amount",
        smooth_window=args.smooth_window,
    )
    _make_panel(
        axes[2],
        x_mbppo,
        y_lambda_mbppo,
        x_ppo,
        y_lambda_ppo,
        label_mbppo,
        label_ppo,
        "Dual Variable lambda vs Episode",
        "lambda",
        smooth_window=args.smooth_window,
    )
    _make_panel(
        axes[3],
        x_mbppo,
        y_ret_mbppo,
        x_ppo,
        y_ret_ppo,
        label_mbppo,
        label_ppo,
        "Total Reward vs Episode",
        "Total reward",
        smooth_window=args.smooth_window,
    )

    fig.suptitle("Convergence Comparison: MB-PPO vs PPO", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = repo_root / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {save_path}")
    print(f"MB-PPO source: {mbppo_rec.exp_dir}")
    print(f"PPO source: {ppo_rec.exp_dir}")


if __name__ == "__main__":
    main()
