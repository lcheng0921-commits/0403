import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(exp_dir: Path) -> List[Dict]:
    metrics_path = exp_dir / "vars" / "episode_metrics.pickle"
    if not metrics_path.exists():
        return []
    with metrics_path.open("rb") as f:
        return pickle.load(f)


def _to_array(metrics: List[Dict], key: str) -> np.ndarray:
    return np.asarray([float(m.get(key, np.nan)) for m in metrics], dtype=np.float64)


def _calc_stats(arr: np.ndarray, mask: np.ndarray):
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    valid = arr[mask & np.isfinite(arr)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(valid)), float(np.var(valid)), float(np.std(valid)), int(valid.size)


def _write_markdown(rows: List[Dict], md_path: Path):
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Objective | Jain Mean | Jain Variance | Jain Std | EE Mean | EE Variance | EE Std | N |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            n_show = min(int(r["jain_points"]), int(r["ee_points"]))
            f.write(
                "| {label} | {jm:.6f} | {jv:.6f} | {js:.6f} | {em:.6f} | {ev:.6f} | {es:.6f} | {n} |\n".format(
                    label=r["label"],
                    jm=r["jain_mean"],
                    jv=r["jain_variance"],
                    js=r["jain_std"],
                    em=r["ee_mean"],
                    ev=r["ee_variance"],
                    es=r["ee_std"],
                    n=n_show,
                )
            )


def _plot_bar(rows: List[Dict], title: str, save_path: Path):
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    colors = ["#3B8EA5", "#F18F01", "#6A994E"]

    jain_mean = np.asarray([r["jain_mean"] for r in rows], dtype=np.float64)
    jain_std = np.asarray([r["jain_std"] for r in rows], dtype=np.float64)
    ee_mean = np.asarray([r["ee_mean"] for r in rows], dtype=np.float64)
    ee_std = np.asarray([r["ee_std"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=180)

    axes[0].bar(x, jain_mean, yerr=jain_std, capsize=4, color=colors, alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12)
    axes[0].set_ylabel("Jain")
    axes[0].set_title("Jain Mean +- Std")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, ee_mean, yerr=ee_std, capsize=4, color=colors, alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=12)
    axes[1].set_ylabel("EE")
    axes[1].set_title("EE Mean +- Std")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute local tail stats (mean/variance/std) for objective tradeoff experiments."
    )
    parser.add_argument("--root", type=str, required=True, help="Experiment root directory.")
    parser.add_argument("--episode-start", type=int, default=2500)
    parser.add_argument("--episode-end", type=int, default=3000)
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="objective_tradeoff_tail2500_3000",
        help="Output file prefix under root.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()

    summary_path = root / "objective_tradeoff_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: List[Dict] = []

    for item in summary:
        exp_dir = Path(item["exp_dir"])
        metrics = _load_metrics(exp_dir)
        if not metrics:
            continue

        episodes = np.asarray([int(m.get("episode", i + 1)) for i, m in enumerate(metrics)], dtype=np.int32)
        mask = (episodes >= int(args.episode_start)) & (episodes <= int(args.episode_end))

        jain = _to_array(metrics, "jain_fairness")
        if not np.any(np.isfinite(jain)):
            jain = _to_array(metrics, "effective_fairness")
        ee = _to_array(metrics, "energy_efficiency")

        j_mean, j_var, j_std, j_n = _calc_stats(jain, mask)
        e_mean, e_var, e_std, e_n = _calc_stats(ee, mask)

        rows.append(
            {
                "objective": item.get("objective", ""),
                "label": item.get("label", item.get("objective", "")),
                "episode_start": int(args.episode_start),
                "episode_end": int(args.episode_end),
                "jain_mean": j_mean,
                "jain_variance": j_var,
                "jain_std": j_std,
                "ee_mean": e_mean,
                "ee_variance": e_var,
                "ee_std": e_std,
                "jain_points": j_n,
                "ee_points": e_n,
            }
        )

    if not rows:
        raise RuntimeError("No valid rows were computed. Check experiment files and range.")

    csv_path = root / f"{args.out_prefix}_stats.csv"
    md_path = root / f"{args.out_prefix}_stats.md"
    bar_path = root / f"{args.out_prefix}_bar.png"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_markdown(rows, md_path)
    _plot_bar(
        rows,
        title=f"Local Tail Statistics (Episode {int(args.episode_start)}-{int(args.episode_end)})",
        save_path=bar_path,
    )

    print("Saved:")
    print(f"- {csv_path}")
    print(f"- {md_path}")
    print(f"- {bar_path}")
    print("\nRows:")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
