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


def _safe_load_json(path: Path):
    if not path.exists():
        return {}
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


def _extract_metric(metrics: List[Dict], keys: List[str]) -> np.ndarray:
    for k in keys:
        arr = np.asarray([float(m.get(k, np.nan)) for m in metrics], dtype=np.float32)
        if arr.size > 0 and np.any(np.isfinite(arr)):
            return arr
    return np.zeros((0,), dtype=np.float32)


def _load_series(exp_dir: Path, metric_source: str) -> Optional[Dict]:
    cfg = _safe_load_json(exp_dir / "config.json")

    if metric_source == "eval":
        metrics = _safe_load_pickle(exp_dir / "vars" / "eval_history.pickle") or []
        if not metrics:
            return None

        x = np.asarray([int(m.get("episode", i + 1)) for i, m in enumerate(metrics)], dtype=np.int32)
        return {
            "exp_dir": exp_dir,
            "config": cfg,
            "episode": x,
            "jain": _extract_metric(metrics, ["eval_jain_fairness"]),
            "ee": _extract_metric(metrics, ["eval_energy_efficiency"]),
            "comm_ee": _extract_metric(metrics, ["eval_comm_energy_efficiency"]),
            "pf_comm_ee": _extract_metric(metrics, ["eval_pf_comm_energy_efficiency"]),
            "min_rate": _extract_metric(metrics, ["eval_min_user_rate"]),
            "step_reward": np.zeros((0,), dtype=np.float32),
            "velocity": np.zeros((0,), dtype=np.float32),
            "power_violation": np.zeros((0,), dtype=np.float32),
        }

    metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    if not metrics:
        return None

    x = np.asarray([int(m.get("episode", i + 1)) for i, m in enumerate(metrics)], dtype=np.int32)
    return {
        "exp_dir": exp_dir,
        "config": cfg,
        "episode": x,
        "jain": _extract_metric(metrics, ["jain_fairness", "effective_fairness"]),
        "ee": _extract_metric(metrics, ["energy_efficiency", "episode_energy_efficiency"]),
        "comm_ee": _extract_metric(metrics, ["episode_comm_ee", "slot_comm_ee"]),
        "pf_comm_ee": _extract_metric(metrics, ["episode_pf_comm_ee", "slot_pf_comm_ee"]),
        "min_rate": _extract_metric(metrics, ["episode_min_user_rate_avg", "min_user_rate"]),
        "step_reward": _extract_metric(metrics, ["step_reward_mean"]),
        "velocity": _extract_metric(metrics, ["velocity"]),
        "power_violation": _extract_metric(metrics, ["tx_power_violation"]),
    }


def _tail_mean(arr: np.ndarray, tail: int) -> float:
    if arr.size == 0:
        return float("nan")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    tw = int(max(1, min(tail, valid.size)))
    return float(np.mean(valid[-tw:]))


def _resolve_dir(repo_root: Path, p: str) -> Path:
    path = Path(str(p).strip())
    if not path.is_absolute():
        path = repo_root / path
    return path


def _find_case_dir(root: Path, keyword: str) -> Optional[Path]:
    if not root.exists():
        return None
    cands = [p for p in root.iterdir() if p.is_dir() and keyword in p.name and (p / "config.json").exists()]
    if not cands:
        return None
    cands = sorted(cands, key=lambda p: p.stat().st_mtime)
    return cands[-1]


def _clip_xy(x: np.ndarray, y: np.ndarray, max_episode: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_episode <= 0:
        return x, y
    keep = x <= int(max_episode)
    return x[keep], y[keep]


def main():
    parser = argparse.ArgumentParser(description="Plot Jain + Total-EE + Comm-EE convergence under three reward objectives.")
    parser.add_argument("--root", type=str, default="", help="Batch root directory from run_objective_tradeoff_batch.py.")
    parser.add_argument("--exp-sum-ee", type=str, default="", help="Explicit exp dir for sum_ee objective.")
    parser.add_argument("--exp-pf-ee", type=str, default="", help="Explicit exp dir for pf_ee objective.")
    parser.add_argument("--exp-max-min", type=str, default="", help="Explicit exp dir for max_min objective.")
    parser.add_argument("--smooth-window", type=int, default=25, help="Rolling smooth window.")
    parser.add_argument("--max-episode", type=int, default=0, help="If >0, clip curves to this max episode.")
    parser.add_argument("--smoothed-only", action="store_true", help="Hide raw traces.")
    parser.add_argument(
        "--metric-source",
        type=str,
        default="eval",
        choices=["eval", "train"],
        help="Curve source: deterministic eval_history or stochastic episode_metrics.",
    )
    parser.add_argument("--tail-window", type=int, default=100, help="Tail window for printed summary.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiment/mb_ppo/pics/fig_conv_objective_tradeoff_jain_ee_split42denseA.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    root = None
    if args.root.strip():
        root = _resolve_dir(repo_root, args.root)

    exp_sum = _resolve_dir(repo_root, args.exp_sum_ee) if args.exp_sum_ee.strip() else None
    exp_pf = _resolve_dir(repo_root, args.exp_pf_ee) if args.exp_pf_ee.strip() else None
    exp_mm = _resolve_dir(repo_root, args.exp_max_min) if args.exp_max_min.strip() else None

    if root is not None:
        if exp_sum is None:
            exp_sum = _find_case_dir(root, "sum_ee")
        if exp_pf is None:
            exp_pf = _find_case_dir(root, "pf_ee")
        if exp_mm is None:
            exp_mm = _find_case_dir(root, "max_min")

    if exp_sum is None or exp_pf is None or exp_mm is None:
        raise RuntimeError(
            "Cannot resolve all three experiment dirs. "
            "Provide --root from batch runner, or pass --exp-sum-ee --exp-pf-ee --exp-max-min explicitly."
        )

    records = {
        "Max Sum-EE": _load_series(exp_sum, metric_source=args.metric_source),
        "Max PF-EE": _load_series(exp_pf, metric_source=args.metric_source),
        "Max-Min": _load_series(exp_mm, metric_source=args.metric_source),
    }

    for name, rec in records.items():
        if rec is None:
            source_file = "eval_history.pickle" if args.metric_source == "eval" else "episode_metrics.pickle"
            raise RuntimeError(f"Missing valid {source_file} for {name}")

    colors = {
        "Max Sum-EE": "#e76f51",
        "Max PF-EE": "#1d3557",
        "Max-Min": "#2a9d8f",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.0))

    for name in ["Max Sum-EE", "Max PF-EE", "Max-Min"]:
        rec = records[name]
        x = rec["episode"]

        y_jain = rec["jain"]
        y_ee = rec["ee"]
        y_comm_ee = rec["comm_ee"]
        xj, yj = _clip_xy(x, y_jain, args.max_episode)
        xe, ye = _clip_xy(x, y_ee, args.max_episode)
        xc, yc = _clip_xy(x, y_comm_ee, args.max_episode)

        sj = _rolling_mean(yj, args.smooth_window)
        se = _rolling_mean(ye, args.smooth_window)
        sc = _rolling_mean(yc, args.smooth_window)

        c = colors[name]
        if not args.smoothed_only:
            axes[0].plot(xj, yj, color=c, linewidth=0.85, alpha=0.15)
            axes[1].plot(xe, ye, color=c, linewidth=0.85, alpha=0.15)
            axes[2].plot(xc, yc, color=c, linewidth=0.85, alpha=0.15)

        axes[0].plot(xj[: sj.size], sj, color=c, linewidth=2.2, label=name)
        axes[1].plot(xe[: se.size], se, color=c, linewidth=2.2, label=name)
        axes[2].plot(xc[: sc.size], sc, color=c, linewidth=2.2, label=name)

    axes[0].set_title("Fairness Convergence (Jain)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Jain index")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Total Energy Efficiency Convergence")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total-EE")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].set_title("Communication Energy Efficiency Convergence")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Comm-EE")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    title_source = "Deterministic Eval" if args.metric_source == "eval" else "Training Rollout"
    fig.suptitle(
        f"Objective Tradeoff Convergence ({title_source})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = repo_root / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {save_path}")

    print("\nTail summary checks:")
    for name in ["Max Sum-EE", "Max PF-EE", "Max-Min"]:
        rec = records[name]
        if args.metric_source == "eval":
            print(
                f"- {name}: "
                f"Eval_Jain_tail={_tail_mean(rec['jain'], args.tail_window):.4f}, "
                f"Eval_TotalEE_tail={_tail_mean(rec['ee'], args.tail_window):.4f}, "
                f"Eval_CommEE_tail={_tail_mean(rec['comm_ee'], args.tail_window):.4f}, "
                f"Eval_PFCommEE_tail={_tail_mean(rec['pf_comm_ee'], args.tail_window):.4f}, "
                f"Eval_Rmin_tail={_tail_mean(rec['min_rate'], args.tail_window):.6f}, "
                f"source={rec['exp_dir']}"
            )
        else:
            print(
                f"- {name}: "
                f"Jain_tail={_tail_mean(rec['jain'], args.tail_window):.4f}, "
                f"TotalEE_tail={_tail_mean(rec['ee'], args.tail_window):.4f}, "
                f"CommEE_tail={_tail_mean(rec['comm_ee'], args.tail_window):.4f}, "
                f"PFCommEE_tail={_tail_mean(rec['pf_comm_ee'], args.tail_window):.4f}, "
                f"Rmin_tail={_tail_mean(rec['min_rate'], args.tail_window):.6f}, "
                f"step_reward_tail={_tail_mean(rec['step_reward'], args.tail_window):.3f}, "
                f"velocity_tail={_tail_mean(rec['velocity'], args.tail_window):.3f}, "
                f"power_violation_tail={_tail_mean(rec['power_violation'], args.tail_window):.5f}, "
                f"source={rec['exp_dir']}"
            )


if __name__ == "__main__":
    main()
