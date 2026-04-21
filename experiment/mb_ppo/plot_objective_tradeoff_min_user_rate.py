import argparse
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


def _tail_mean(arr: np.ndarray, tail: int) -> float:
    if arr.size == 0:
        return float("nan")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    tw = int(max(1, min(tail, valid.size)))
    return float(np.mean(valid[-tw:]))


def _load_series(exp_dir: Path) -> Optional[Dict]:
    metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    if not metrics:
        return None

    x = np.asarray([int(m.get("episode", i + 1)) for i, m in enumerate(metrics)], dtype=np.int32)
    rmin = _extract_metric(metrics, ["min_user_rate", "episode_min_user_rate_avg"])

    return {
        "exp_dir": exp_dir,
        "episode": x,
        "min_user_rate": rmin,
    }


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
    parser = argparse.ArgumentParser(description="Plot weakest-user-rate convergence under three reward objectives.")
    parser.add_argument("--root", type=str, default="", help="Batch root directory from run_objective_tradeoff_batch.py.")
    parser.add_argument("--exp-sum-ee", type=str, default="", help="Explicit exp dir for sum_ee objective.")
    parser.add_argument("--exp-pf-ee", type=str, default="", help="Explicit exp dir for pf_ee objective.")
    parser.add_argument("--exp-max-min", type=str, default="", help="Explicit exp dir for max_min objective.")
    parser.add_argument("--smooth-window", type=int, default=25, help="Rolling smooth window.")
    parser.add_argument("--max-episode", type=int, default=0, help="If >0, clip curves to this max episode.")
    parser.add_argument("--smoothed-only", action="store_true", help="Hide raw traces.")
    parser.add_argument("--tail-window", type=int, default=200, help="Tail window for printed summary.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiment/mb_ppo/pics/fig_conv_objective_tradeoff_min_user_rate_split42denseA.png",
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
        "Max Sum-EE": _load_series(exp_sum),
        "Max PF-EE": _load_series(exp_pf),
        "Max-Min": _load_series(exp_mm),
    }

    for name, rec in records.items():
        if rec is None:
            raise RuntimeError(f"Missing valid episode_metrics for {name}")

    colors = {
        "Max Sum-EE": "#e76f51",
        "Max PF-EE": "#1d3557",
        "Max-Min": "#2a9d8f",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))

    for name in ["Max Sum-EE", "Max PF-EE", "Max-Min"]:
        rec = records[name]
        x = rec["episode"]
        y = rec["min_user_rate"]
        xc, yc = _clip_xy(x, y, args.max_episode)

        sy = _rolling_mean(yc, args.smooth_window)
        c = colors[name]

        if not args.smoothed_only:
            ax.plot(xc, yc, color=c, linewidth=0.85, alpha=0.15)
        ax.plot(xc[: sy.size], sy, color=c, linewidth=2.2, label=name)

    ax.set_title("Weakest User Rate Convergence")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Min user rate")
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

    print("\nTail summary checks:")
    for name in ["Max Sum-EE", "Max PF-EE", "Max-Min"]:
        rec = records[name]
        print(
            f"- {name}: "
            f"rmin_tail={_tail_mean(rec['min_user_rate'], args.tail_window):.6f}, "
            f"source={rec['exp_dir']}"
        )


if __name__ == "__main__":
    main()
