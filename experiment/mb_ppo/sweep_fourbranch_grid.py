import argparse
import csv
import itertools
import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from algo.mb_ppo.run_mbppo import train
from experiment.mb_ppo.maps import ClusteredMap500, ClusteredMap500Split24


@dataclass
class GridCase:
    case_id: str
    lr: float
    entropy_coef: float
    lambda_lr: float
    ppo_epochs: int


@dataclass
class SeriesRecord:
    case_id: str
    exp_dir: Path
    config: Dict
    episode: np.ndarray
    ee: np.ndarray
    qos: np.ndarray
    lamb: np.ndarray


def _safe_load_pickle(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _safe_load_json(path: Path) -> Dict:
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


def _parse_float_grid(text: str, name: str) -> List[float]:
    values = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception as exc:
            raise ValueError(f"Invalid float token in --{name}: {token}") from exc
    if not values:
        raise ValueError(f"--{name} cannot be empty")
    return values


def _parse_int_grid(text: str, name: str) -> List[int]:
    values = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except Exception as exc:
            raise ValueError(f"Invalid int token in --{name}: {token}") from exc
    if not values:
        raise ValueError(f"--{name} cannot be empty")
    return values


def _float_token(value: float) -> str:
    token = f"{value:.6g}"
    token = token.replace("+", "")
    token = token.replace("-", "m")
    token = token.replace(".", "p")
    return token


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    w = int(max(1, window))
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=np.float32) / float(w)
    padded = np.pad(x, (w - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _extract_episode_axis(episode_metrics: List[Dict]) -> np.ndarray:
    if not episode_metrics:
        return np.zeros((0,), dtype=np.int32)
    episodes = [int(item.get("episode", idx + 1)) for idx, item in enumerate(episode_metrics)]
    return np.asarray(episodes, dtype=np.int32)


def _extract_metric(episode_metrics: List[Dict], key: str) -> np.ndarray:
    if not episode_metrics:
        return np.zeros((0,), dtype=np.float32)
    values = [float(item.get(key, np.nan)) for item in episode_metrics]
    return np.asarray(values, dtype=np.float32)


def _extract_qos_metric(episode_metrics: List[Dict]) -> Tuple[np.ndarray, str]:
    prefer = [
        "qos_violation_sum_rollout_mean",
        "qos_violation_norm_rollout_mean",
        "qos_violation_sum",
        "qos_violation_sq_sum",
        "qos_violation",
        "qos_violation_norm",
    ]
    for key in prefer:
        arr = _extract_metric(episode_metrics, key)
        if arr.size > 0 and np.any(np.isfinite(arr)):
            return arr, key
    return np.zeros((0,), dtype=np.float32), "none"


def _tail_mean(arr: np.ndarray, tail_window: int) -> float:
    if arr.size == 0:
        return float("nan")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    tw = int(max(1, min(tail_window, valid.size)))
    return float(np.mean(valid[-tw:]))


def _tail_last(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    return float(valid[-1])


def _build_cases(args) -> List[GridCase]:
    lrs = _parse_float_grid(args.lr_grid, "lr-grid")
    entropy = _parse_float_grid(args.entropy_grid, "entropy-grid")
    lambda_lrs = _parse_float_grid(args.lambda_lr_grid, "lambda-lr-grid")
    ppo_epochs = _parse_int_grid(args.ppo_epochs_grid, "ppo-epochs-grid")

    products = list(itertools.product(lrs, entropy, lambda_lrs, ppo_epochs))
    cases = []
    for idx, (lr, ent, lam_lr, pe) in enumerate(products, start=1):
        case_id = f"g{idx:02d}"
        cases.append(
            GridCase(
                case_id=case_id,
                lr=float(lr),
                entropy_coef=float(ent),
                lambda_lr=float(lam_lr),
                ppo_epochs=int(pe),
            )
        )
    return cases


def _case_folder_name(case: GridCase) -> str:
    return (
        f"{case.case_id}"
        f"_lr{_float_token(case.lr)}"
        f"_ent{_float_token(case.entropy_coef)}"
        f"_llr{_float_token(case.lambda_lr)}"
        f"_pe{int(case.ppo_epochs)}"
    )


def _resolve_grid_root(repo_root: Path, args) -> Path:
    if args.grid_root.strip():
        custom = Path(args.grid_root.strip())
        if not custom.is_absolute():
            custom = repo_root / custom
        return custom

    date_tag = datetime.now().strftime("%Y%m%d")
    tag = args.tag.strip() if args.tag.strip() else f"fourbranch_grid_{date_tag}"
    return repo_root / "mb_ppo_data" / tag


def _build_plan(cases: Sequence[GridCase], grid_root: Path, args) -> List[Dict]:
    plan = []
    for idx, case in enumerate(cases, start=1):
        folder = _case_folder_name(case)
        exp_dir = grid_root / folder
        seed_val = int(args.seed + args.seed_stride * (idx - 1))
        plan.append(
            {
                "index": idx,
                "case_id": case.case_id,
                "folder": folder,
                "exp_dir": str(exp_dir),
                "seed": seed_val,
                **asdict(case),
            }
        )
    return plan


def _save_plan(plan_path: Path, plan: List[Dict], args):
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "cases": plan,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_plan(plan: List[Dict]):
    print("=" * 90)
    print("Four-branch hyperparameter grid plan")
    print("=" * 90)
    for item in plan:
        print(
            "[{index:02d}] {case_id} | lr={lr:.2e} | entropy={entropy_coef:.2e} | "
            "lambda_lr={lambda_lr:.2e} | ppo_epochs={ppo_epochs} | seed={seed}"
            .format(**item)
        )
    print("=" * 90)


def _select_map_class(map_name: str):
    key = str(map_name).strip().lower()
    if key == "split24":
        return ClusteredMap500Split24
    if key == "clustered":
        return ClusteredMap500
    raise ValueError(f"Unsupported map choice: {map_name}")


def _run_cases(repo_root: Path, grid_root: Path, plan: List[Dict], args):
    map_cls = _select_map_class(args.map)

    for item in plan:
        exp_dir = Path(item["exp_dir"])
        config_path = exp_dir / "config.json"

        if config_path.exists() and args.skip_existing:
            print(f"[Skip] Existing run found: {exp_dir}")
            continue

        if config_path.exists() and not args.skip_existing:
            raise RuntimeError(
                f"Experiment already exists: {exp_dir}. "
                "Use --skip-existing to continue without rerunning."
            )

        print(
            f"\n[Run] {item['case_id']} -> {exp_dir.name} | "
            f"lr={item['lr']:.2e}, ent={item['entropy_coef']:.2e}, "
            f"lambda_lr={item['lambda_lr']:.2e}, ppo_epochs={item['ppo_epochs']}"
        )

        kwargs = {
            "baseline": "mbppo",
            "map": map_cls,
            "grid_case_id": str(item["case_id"]),
            "grid_tag": str(grid_root.name),
            "output_dir": str(exp_dir),
            "episodes": int(args.episodes),
            "eval_interval": int(args.eval_interval),
            "save_freq": int(args.save_freq),
            "seed": int(item["seed"]),
            "n_gts": int(args.n_gts),
            "paper_fixed_qos_threshold": float(args.paper_fixed_rth),
            "qos_threshold": float(args.paper_fixed_rth),
            "enforce_fixed_qos_threshold": True,
            "tx_power_max_dbm": float(args.tx_power_max_dbm),
            "lr": float(item["lr"]),
            "entropy_coef": float(item["entropy_coef"]),
            "lambda_lr": float(item["lambda_lr"]),
            "ppo_epochs": int(item["ppo_epochs"]),
        }
        train(kwargs)


def _discover_case_dirs(grid_root: Path) -> List[Path]:
    if not grid_root.exists():
        return []

    dirs = []
    for p in sorted(grid_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "config.json").exists() and (p / "vars" / "episode_metrics.pickle").exists():
            dirs.append(p)
    return dirs


def _load_series(exp_dir: Path) -> Optional[SeriesRecord]:
    config = _safe_load_json(exp_dir / "config.json")
    metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    if not metrics:
        return None

    x = _extract_episode_axis(metrics)
    ee = _extract_metric(metrics, "energy_efficiency")
    qos, _ = _extract_qos_metric(metrics)
    lamb = _extract_metric(metrics, "lambda_penalty")

    case_id = str(config.get("grid_case_id", ""))
    if not case_id:
        # Fallback: parse from folder prefix gXX_...
        folder = exp_dir.name
        case_id = folder.split("_")[0] if "_" in folder else folder

    return SeriesRecord(
        case_id=case_id,
        exp_dir=exp_dir,
        config=config,
        episode=x,
        ee=ee,
        qos=qos,
        lamb=lamb,
    )


def _build_summary(series_records: Sequence[SeriesRecord], tail_window: int, qos_ok_threshold: float) -> List[Dict]:
    rows = []
    for rec in series_records:
        row = {
            "case_id": rec.case_id,
            "exp_dir": str(rec.exp_dir),
            "seed": int(rec.config.get("seed", -1)),
            "lr": float(rec.config.get("lr", np.nan)),
            "entropy_coef": float(rec.config.get("entropy_coef", np.nan)),
            "lambda_lr": float(rec.config.get("lambda_lr", np.nan)),
            "ppo_epochs": int(rec.config.get("ppo_epochs", -1)),
            "episodes": int(rec.episode[-1]) if rec.episode.size > 0 else -1,
            "ee_last": _tail_last(rec.ee),
            "ee_tail_mean": _tail_mean(rec.ee, tail_window=tail_window),
            "qos_last": _tail_last(rec.qos),
            "qos_tail_mean": _tail_mean(rec.qos, tail_window=tail_window),
            "lambda_last": _tail_last(rec.lamb),
            "lambda_tail_mean": _tail_mean(rec.lamb, tail_window=tail_window),
        }
        row["qos_ok"] = bool(np.isfinite(row["qos_tail_mean"]) and row["qos_tail_mean"] <= qos_ok_threshold)
        rows.append(row)

    rows.sort(
        key=lambda r: (
            0 if r["qos_ok"] else 1,
            -r["ee_tail_mean"] if np.isfinite(r["ee_tail_mean"]) else np.inf,
            r["qos_tail_mean"] if np.isfinite(r["qos_tail_mean"]) else np.inf,
            r["lambda_tail_mean"] if np.isfinite(r["lambda_tail_mean"]) else np.inf,
        )
    )

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return rows


def _save_summary(summary: List[Dict], grid_root: Path):
    json_path = grid_root / "grid_summary.json"
    csv_path = grid_root / "grid_summary.csv"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary:
        fields = list(summary[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in summary:
                writer.writerow(row)


def _plot_compare_curves(
    series_records: Sequence[SeriesRecord],
    summary: Sequence[Dict],
    save_path: Path,
    smooth_window: int,
    max_episode: int,
    smoothed_only: bool,
):
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.3))
    axes = axes.reshape(-1)
    cmap = plt.get_cmap("tab20")

    # Keep line order consistent with ranked summary.
    order = {row["case_id"]: i for i, row in enumerate(summary)}
    records = sorted(series_records, key=lambda r: order.get(r.case_id, 10_000))

    for idx, rec in enumerate(records):
        color = cmap(idx % 20)
        label = rec.case_id

        x = rec.episode
        y_ee = rec.ee
        y_qos = rec.qos
        y_lam = rec.lamb

        if max_episode > 0:
            keep = x <= max_episode
            x = x[keep]
            y_ee = y_ee[keep]
            y_qos = y_qos[keep]
            y_lam = y_lam[keep]

        if x.size == 0:
            continue

        s_ee = _rolling_mean(y_ee, smooth_window)
        s_qos = _rolling_mean(y_qos, smooth_window)
        s_lam = _rolling_mean(y_lam, smooth_window)

        if not smoothed_only:
            axes[0].plot(x, y_ee, linewidth=0.9, alpha=0.14, color=color)
            axes[1].plot(x, y_qos, linewidth=0.9, alpha=0.14, color=color)
            axes[2].plot(x, y_lam, linewidth=0.9, alpha=0.14, color=color)

        axes[0].plot(x[: s_ee.size], s_ee, linewidth=1.8, color=color, label=label)
        axes[1].plot(x[: s_qos.size], s_qos, linewidth=1.8, color=color, label=label)
        axes[2].plot(x[: s_lam.size], s_lam, linewidth=1.8, color=color, label=label)

    axes[0].set_title("Energy Efficiency")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("EE")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("QoS Violation")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("QoS violation")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Dual Variable lambda")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("lambda")
    axes[2].grid(alpha=0.25)

    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    title_mode = "smoothed-only" if smoothed_only else "raw+smoothed"
    fig.suptitle(f"Four-branch Grid Compare ({title_mode})", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.02, 0.88, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_pareto(summary: Sequence[Dict], save_path: Path):
    if not summary:
        return

    x = np.asarray([float(row.get("qos_tail_mean", np.nan)) for row in summary], dtype=np.float32)
    y = np.asarray([float(row.get("ee_tail_mean", np.nan)) for row in summary], dtype=np.float32)
    c = np.asarray([float(row.get("lambda_tail_mean", np.nan)) for row in summary], dtype=np.float32)
    labels = [str(row.get("case_id", "?")) for row in summary]

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if not np.any(keep):
        return

    x = x[keep]
    y = y[keep]
    c = c[keep]
    labels = [labels[i] for i, ok in enumerate(keep.tolist()) if ok]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.8))
    scatter = ax.scatter(x, y, c=c, cmap="viridis", s=70, alpha=0.92)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(4, 4), fontsize=9)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("lambda_tail_mean")

    ax.set_title("Four-branch Grid Pareto View")
    ax.set_xlabel("QoS violation tail mean (lower is better)")
    ax.set_ylabel("EE tail mean (higher is better)")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _inject_case_metadata(plan: Sequence[Dict]):
    # This metadata is only for summary readability; training itself does not require it.
    mapping = {}
    for item in plan:
        mapping[str(Path(item["exp_dir"]).resolve())] = item
    return mapping


def _save_case_mapping(summary: Sequence[Dict], grid_root: Path):
    path = grid_root / "grid_case_mapping.txt"
    lines = ["case_id -> (lr, entropy_coef, lambda_lr, ppo_epochs, seed)"]
    for row in summary:
        lines.append(
            "{case_id}: lr={lr:.2e}, entropy={entropy_coef:.2e}, lambda_lr={lambda_lr:.2e}, ppo_epochs={ppo_epochs}, seed={seed}".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_top(summary: Sequence[Dict], topk: int):
    print("\nTop cases (ranked):")
    for row in list(summary)[: max(1, topk)]:
        print(
            "#{rank:02d} {case_id} | qos_ok={qos_ok} | EE_tail={ee_tail_mean:.3f} | "
            "QoS_tail={qos_tail_mean:.6f} | lambda_tail={lambda_tail_mean:.6f}"
            .format(**row)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Four-branch MB-PPO hyperparameter grid + batch plotting (EE/QoS/lambda)."
    )

    parser.add_argument("--lr-grid", type=str, default="6e-5,8e-5,1e-4", help="Comma-separated learning rates.")
    parser.add_argument("--entropy-grid", type=str, default="3e-3,5e-3", help="Comma-separated entropy coefficients.")
    parser.add_argument("--lambda-lr-grid", type=str, default="6e-4,8e-4", help="Comma-separated lambda learning rates.")
    parser.add_argument("--ppo-epochs-grid", type=str, default="10", help="Comma-separated PPO epoch values.")

    parser.add_argument("--episodes", type=int, default=1200, help="Training episodes per case.")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval.")
    parser.add_argument("--save-freq", type=int, default=200, help="Checkpoint save frequency.")
    parser.add_argument("--seed", type=int, default=10, help="Base random seed.")
    parser.add_argument("--seed-stride", type=int, default=0, help="Added seed offset per case index.")
    parser.add_argument("--n-gts", type=int, default=6, help="Number of users.")
    parser.add_argument("--paper-fixed-rth", type=float, default=0.04, help="Fixed paper QoS threshold.")
    parser.add_argument("--tx-power-max-dbm", type=float, default=30.0, help="Max TX power in dBm.")
    parser.add_argument("--map", type=str, choices=["split24", "clustered"], default="split24", help="Map class used in sweep.")

    parser.add_argument("--tag", type=str, default="", help="Grid tag used when --grid-root is empty.")
    parser.add_argument("--grid-root", type=str, default="", help="Explicit grid root directory.")
    parser.add_argument("--max-runs", type=int, default=0, help="If >0, only run first N grid cases.")

    parser.add_argument("--dry-run", action="store_true", help="Only print grid plan and save plan file.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only analyze existing runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already completed runs under grid root.")

    parser.add_argument("--smooth-window", type=int, default=25, help="Smoothing window for curve plots.")
    parser.add_argument("--max-episode", type=int, default=0, help="If >0, clip compare curves by episode.")
    parser.add_argument("--smoothed-only", action="store_true", help="Hide raw traces in compare curves.")
    parser.add_argument("--tail-window", type=int, default=100, help="Tail window for summary statistics.")
    parser.add_argument("--qos-ok-threshold", type=float, default=0.05, help="Threshold for qos_ok flag.")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k ranked cases in terminal output.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    grid_root = _resolve_grid_root(repo_root, args)

    cases = _build_cases(args)
    plan = _build_plan(cases, grid_root, args)
    max_runs = int(args.max_runs)
    active_plan = plan[:max_runs] if max_runs > 0 else plan

    grid_root.mkdir(parents=True, exist_ok=True)
    plan_path = grid_root / "grid_plan.json"
    _save_plan(plan_path, plan, args)
    _print_plan(active_plan)
    print(f"Active runs: {len(active_plan)} / {len(plan)}")
    print(f"Plan file: {plan_path}")

    if args.dry_run:
        print("Dry-run only. No training executed.")
        return

    if not args.skip_train:
        _run_cases(repo_root=repo_root, grid_root=grid_root, plan=active_plan, args=args)

    case_dirs = _discover_case_dirs(grid_root)
    if not case_dirs:
        print(f"No completed case directories found under: {grid_root}")
        return

    series_records = []
    metadata = _inject_case_metadata(active_plan)
    for exp_dir in case_dirs:
        rec = _load_series(exp_dir)
        if rec is None:
            continue

        match = metadata.get(str(exp_dir.resolve()), None)
        if match is not None:
            rec.config["grid_case_id"] = str(match.get("case_id", rec.case_id))
            rec.case_id = str(match.get("case_id", rec.case_id))

        series_records.append(rec)

    if not series_records:
        print("No valid series records found for analysis.")
        return

    summary = _build_summary(
        series_records,
        tail_window=int(args.tail_window),
        qos_ok_threshold=float(args.qos_ok_threshold),
    )

    _save_summary(summary, grid_root)
    _save_case_mapping(summary, grid_root)

    compare_fig = grid_root / "fig_compare_ee_qos_lambda.png"
    pareto_fig = grid_root / "fig_pareto_ee_vs_qos.png"

    _plot_compare_curves(
        series_records=series_records,
        summary=summary,
        save_path=compare_fig,
        smooth_window=int(args.smooth_window),
        max_episode=int(args.max_episode),
        smoothed_only=bool(args.smoothed_only),
    )
    _plot_pareto(summary, pareto_fig)

    print(f"Saved summary JSON: {grid_root / 'grid_summary.json'}")
    print(f"Saved summary CSV: {grid_root / 'grid_summary.csv'}")
    print(f"Saved case mapping: {grid_root / 'grid_case_mapping.txt'}")
    print(f"Saved compare figure: {compare_fig}")
    print(f"Saved pareto figure: {pareto_fig}")

    _print_top(summary, topk=int(args.topk))


if __name__ == "__main__":
    main()
