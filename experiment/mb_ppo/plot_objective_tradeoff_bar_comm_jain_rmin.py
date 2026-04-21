import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


OBJECTIVE_SPECS = [
    ("sum_ee", "Max Sum-EE", "#e76f51"),
    ("pf_ee", "Max PF-EE", "#1d3557"),
    ("max_min", "Max-Min", "#2a9d8f"),
]


def _safe_load_pickle(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _resolve_dir(repo_root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _tail_slice(metrics: List[Dict], episode_start: int, episode_end: int) -> List[Dict]:
    return [
        m
        for m in metrics
        if episode_start <= int(m.get("episode", 0)) <= episode_end
    ]


def _mean_valid(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def _compute_comm_ee_from_eval_history(
    tail_metrics: List[Dict],
    episode_length: int,
) -> Tuple[np.ndarray, str]:
    comm_values: List[float] = []

    has_direct = any("eval_comm_energy_efficiency" in m for m in tail_metrics)
    has_tx_energy = any("eval_tx_energy" in m for m in tail_metrics)

    if has_direct:
        for m in tail_metrics:
            comm_values.append(float(m.get("eval_comm_energy_efficiency", np.nan)))
        return np.asarray(comm_values, dtype=np.float64), "direct_eval_comm_energy_efficiency"

    if has_tx_energy:
        for m in tail_metrics:
            ee = float(m.get("eval_energy_efficiency", np.nan))
            total_energy = float(m.get("eval_total_energy", np.nan))
            tx_energy = float(m.get("eval_tx_energy", np.nan))
            if np.isfinite(ee) and np.isfinite(total_energy) and tx_energy > 1e-12:
                comm_values.append(float(ee * total_energy / tx_energy))
            else:
                comm_values.append(float("nan"))
        return np.asarray(comm_values, dtype=np.float64), "reconstructed_from_eval_tx_energy"

    # Backward-compatible reconstruction for old logs without eval_tx_energy:
    # throughput = eval_energy_efficiency * eval_total_energy
    # comm_ee ~= throughput / (eval_tx_power * episode_length)
    for m in tail_metrics:
        ee = float(m.get("eval_energy_efficiency", np.nan))
        total_energy = float(m.get("eval_total_energy", np.nan))
        tx_power = float(m.get("eval_tx_power", np.nan))
        if np.isfinite(ee) and np.isfinite(total_energy) and tx_power > 1e-12:
            comm_values.append(float(ee * total_energy / (tx_power * max(1, episode_length))))
        else:
            comm_values.append(float("nan"))
    return np.asarray(comm_values, dtype=np.float64), "proxy_from_eval_tx_power_times_episode_length"


def _collect_tail_summary(
    root: Path,
    episode_start: int,
    episode_end: int,
    episode_length: int,
) -> List[Dict]:
    rows: List[Dict] = []

    for objective, label, _ in OBJECTIVE_SPECS:
        eval_path = root / f"{objective}_ep3000_seed10" / "vars" / "eval_history.pickle"
        metrics = _safe_load_pickle(eval_path) or []
        if not metrics:
            raise RuntimeError(f"Missing eval_history.pickle for objective={objective}: {eval_path}")

        tail = _tail_slice(metrics, episode_start=episode_start, episode_end=episode_end)
        if not tail:
            raise RuntimeError(
                f"No eval points in [{episode_start}, {episode_end}] for objective={objective}: {eval_path}"
            )

        comm_series, comm_source = _compute_comm_ee_from_eval_history(
            tail_metrics=tail,
            episode_length=episode_length,
        )
        jain_series = np.asarray([float(m.get("eval_jain_fairness", np.nan)) for m in tail], dtype=np.float64)
        rmin_series = np.asarray([float(m.get("eval_min_user_rate", np.nan)) for m in tail], dtype=np.float64)

        rows.append(
            {
                "objective": objective,
                "label": label,
                "tail_points": int(len(tail)),
                "episode_start": int(episode_start),
                "episode_end": int(episode_end),
                "comm_ee_source": comm_source,
                "comm_ee_mean": _mean_valid(comm_series),
                "jain_mean": _mean_valid(jain_series),
                "rmin_mean": _mean_valid(rmin_series),
            }
        )

    return rows


def _save_summary_files(rows: List[Dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _annotate_bars(ax, bars, values, fmt: str) -> None:
    for idx, bar in enumerate(bars):
        val = float(values[idx])
        if not np.isfinite(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() * 0.5,
            bar.get_height() + 0.01 * max(1e-6, float(np.nanmax(values))),
            format(val, fmt),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _plot_bar_triplet(rows: List[Dict], save_path: Path) -> None:
    labels = [r["label"] for r in rows]
    colors = [spec[2] for spec in OBJECTIVE_SPECS]
    x = np.arange(len(rows), dtype=np.float32)

    comm_vals = np.asarray([float(r["comm_ee_mean"]) for r in rows], dtype=np.float64)
    jain_vals = np.asarray([float(r["jain_mean"]) for r in rows], dtype=np.float64)
    rmin_vals = np.asarray([float(r["rmin_mean"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))

    bars0 = axes[0].bar(x, comm_vals, color=colors, width=0.62, alpha=0.95)
    axes[0].set_title("Comm-EE (SumRate / P_tx)")
    axes[0].set_ylabel("Comm-EE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=8)
    axes[0].grid(axis="y", alpha=0.25)
    _annotate_bars(axes[0], bars0, comm_vals, ".3f")

    bars1 = axes[1].bar(x, jain_vals, color=colors, width=0.62, alpha=0.95)
    axes[1].set_title("Jain Index")
    axes[1].set_ylabel("Jain")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=8)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(axis="y", alpha=0.25)
    _annotate_bars(axes[1], bars1, jain_vals, ".4f")

    bars2 = axes[2].bar(x, rmin_vals, color=colors, width=0.62, alpha=0.95)
    axes[2].set_title("Min-Rate")
    axes[2].set_ylabel("Rate (Mbps)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=8)
    axes[2].grid(axis="y", alpha=0.25)
    _annotate_bars(axes[2], bars2, rmin_vals, ".4f")

    fig.suptitle("Objective Tradeoff Tail Means (Deterministic Eval)", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Comm-EE/Jain/Min-Rate bars from deterministic eval tail window.")
    parser.add_argument("--root", type=str, required=True, help="Objective-tradeoff root directory.")
    parser.add_argument("--episode-start", type=int, default=2500, help="Tail start episode (inclusive).")
    parser.add_argument("--episode-end", type=int, default=3000, help="Tail end episode (inclusive).")
    parser.add_argument("--episode-length", type=int, default=100, help="Episode length used for old-log Comm-EE proxy reconstruction.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Output bar chart path. Default: <root>/objective_tradeoff_bar_comm_jain_rmin_eval_tail2500_3000.png",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="Output summary CSV path. Default: <root>/objective_tradeoff_tail2500_3000_comm_jain_rmin_summary.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Output summary JSON path. Default: <root>/objective_tradeoff_tail2500_3000_comm_jain_rmin_summary.json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    root = _resolve_dir(repo_root, args.root)

    rows = _collect_tail_summary(
        root=root,
        episode_start=int(args.episode_start),
        episode_end=int(args.episode_end),
        episode_length=int(args.episode_length),
    )

    save_path = _resolve_dir(repo_root, args.save_path) if args.save_path.strip() else (
        root / f"objective_tradeoff_bar_comm_jain_rmin_eval_tail{int(args.episode_start)}_{int(args.episode_end)}.png"
    )
    summary_csv = _resolve_dir(repo_root, args.summary_csv) if args.summary_csv.strip() else (
        root / f"objective_tradeoff_tail{int(args.episode_start)}_{int(args.episode_end)}_comm_jain_rmin_summary.csv"
    )
    summary_json = _resolve_dir(repo_root, args.summary_json) if args.summary_json.strip() else (
        root / f"objective_tradeoff_tail{int(args.episode_start)}_{int(args.episode_end)}_comm_jain_rmin_summary.json"
    )

    _plot_bar_triplet(rows, save_path=save_path)
    _save_summary_files(rows, csv_path=summary_csv, json_path=summary_json)

    print(f"Saved bar chart: {save_path}")
    print(f"Saved summary csv: {summary_csv}")
    print(f"Saved summary json: {summary_json}")
    print("\nTail mean summary:")
    for row in rows:
        print(
            "- {label}: Comm-EE={comm_ee_mean:.6f} ({comm_ee_source}), Jain={jain_mean:.6f}, Min-Rate={rmin_mean:.6f}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
