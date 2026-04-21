import argparse
import csv
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from algo.mb_ppo.run_mbppo import train
from experiment.mb_ppo.maps import ClusteredMap500Split24


@dataclass
class ControlCase:
    case_id: str
    desc: str
    overrides: Dict


@dataclass
class SeriesRecord:
    case_id: str
    desc: str
    exp_dir: Path
    config: Dict
    episode: np.ndarray
    qos: np.ndarray
    lamb: np.ndarray
    qos_signal: np.ndarray


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


def _extract_qos_metric(episode_metrics: List[Dict]) -> np.ndarray:
    for key in [
        "qos_violation_sum_rollout_mean",
        "qos_violation_norm_rollout_mean",
        "qos_violation_sum",
        "qos_violation_sq_sum",
        "qos_violation",
        "qos_violation_norm",
    ]:
        arr = _extract_metric(episode_metrics, key)
        if arr.size > 0 and np.any(np.isfinite(arr)):
            return arr
    return np.zeros((0,), dtype=np.float32)


def _extract_qos_signal_metric(episode_metrics: List[Dict]) -> np.ndarray:
    for key in [
        "qos_signal_rollout_mean",
        "qos_signal_raw_rollout_mean",
        "qos_dual_signal_norm_for_lambda",
        "qos_dual_signal_norm",
    ]:
        arr = _extract_metric(episode_metrics, key)
        if arr.size > 0 and np.any(np.isfinite(arr)):
            return arr
    return np.zeros((0,), dtype=np.float32)


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


def _post_window_mean(episode: np.ndarray, arr: np.ndarray, start_episode: int) -> float:
    if episode.size == 0 or arr.size == 0:
        return float("nan")
    keep = episode >= int(start_episode)
    if not np.any(keep):
        return float("nan")
    segment = arr[keep]
    valid = segment[np.isfinite(segment)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def _post_zero_ratio(episode: np.ndarray, arr: np.ndarray, start_episode: int, eps: float = 1e-4) -> float:
    if episode.size == 0 or arr.size == 0:
        return float("nan")
    keep = episode >= int(start_episode)
    if not np.any(keep):
        return float("nan")
    segment = arr[keep]
    valid = segment[np.isfinite(segment)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid <= float(eps)))


def _post_lambda_slope(episode: np.ndarray, lamb: np.ndarray, start_episode: int) -> float:
    if episode.size == 0 or lamb.size == 0:
        return float("nan")
    keep = episode >= int(start_episode)
    if np.sum(keep) < 20:
        return float("nan")
    x = episode[keep].astype(np.float64)
    y = lamb[keep].astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 20:
        return float("nan")
    slope, _ = np.polyfit(x[valid], y[valid], 1)
    return float(slope)


def _resolve_root(repo_root: Path, args) -> Path:
    if args.root.strip():
        custom = Path(args.root.strip())
        if not custom.is_absolute():
            custom = repo_root / custom
        return custom

    date_tag = datetime.now().strftime("%Y%m%d")
    tag = args.tag.strip() if args.tag.strip() else f"qos_lambda_controls_{date_tag}"
    return repo_root / "mb_ppo_data" / tag


def _build_cases(args) -> List[ControlCase]:
    return [
        ControlCase(
            case_id="c00",
            desc="baseline_best",
            overrides={},
        ),
        ControlCase(
            case_id="c01",
            desc="scheme1_lower_rth",
            overrides={
                "paper_fixed_qos_threshold": float(args.s1_rth),
                "qos_threshold": float(args.s1_rth),
            },
        ),
        ControlCase(
            case_id="c02",
            desc="scheme2_higher_lambda_lr",
            overrides={
                "lambda_lr": float(args.s2_lambda_lr),
            },
        ),
        ControlCase(
            case_id="c03",
            desc="scheme3_reward_weight_pfg",
            overrides={
                "pfg_guidance_weight": float(args.s3_pfg_weight),
            },
        ),
        ControlCase(
            case_id="c04",
            desc="scheme4_longer_core_warmup",
            overrides={
                "core_penalty_warmup_episodes": int(args.s4_core_warmup),
            },
        ),
    ]


def _case_dir_name(case: ControlCase) -> str:
    return f"{case.case_id}_{case.desc}"


def _build_base_kwargs(args) -> Dict:
    return {
        "baseline": "mbppo",
        "map": ClusteredMap500Split24,
        "episodes": int(args.episodes),
        "eval_interval": int(args.eval_interval),
        "save_freq": int(args.save_freq),
        "seed": int(args.seed),
        "n_gts": int(args.n_gts),
        "paper_fixed_qos_threshold": float(args.base_rth),
        "qos_threshold": float(args.base_rth),
        "enforce_fixed_qos_threshold": True,
        "tx_power_max_dbm": float(args.tx_power_max_dbm),
        "lr": float(args.base_lr),
        "entropy_coef": float(args.base_entropy),
        "lambda_lr": float(args.base_lambda_lr),
        "ppo_epochs": int(args.base_ppo_epochs),
        "lambda_init": float(args.lambda_init),
        "lambda_max": float(args.lambda_max),
        "reward_ee_scale": float(args.reward_ee_scale),
        "reward_core_penalty_scale": float(args.reward_core_penalty_scale),
        "pfg_guidance_weight": float(args.base_pfg_weight),
        "core_penalty_warmup_episodes": int(args.base_core_warmup),
        "core_penalty_start_scale": float(args.core_penalty_start_scale),
        "terminate_on_core_violation": bool(args.terminate_on_core_violation),
        "core_terminate_start_episode": int(args.core_terminate_start_episode),
        "lambda_signal_aggregation": str(args.lambda_signal_aggregation),
        "save_episode_metrics": True,
    }


def _save_plan(root: Path, cases: Sequence[ControlCase], args, base_kwargs: Dict):
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "base_kwargs": {k: (str(v) if callable(v) else v) for k, v in base_kwargs.items()},
        "cases": [asdict(c) for c in cases],
    }
    (root / "control_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_cases(repo_root: Path, root: Path, cases: Sequence[ControlCase], args, base_kwargs: Dict):
    del repo_root
    for idx, case in enumerate(cases, start=1):
        exp_dir = root / _case_dir_name(case)
        config_path = exp_dir / "config.json"

        if config_path.exists() and args.skip_existing:
            print(f"[Skip] Existing run found: {exp_dir}")
            continue

        if config_path.exists() and not args.skip_existing:
            raise RuntimeError(
                f"Experiment already exists: {exp_dir}. Use --skip-existing to continue."
            )

        kwargs = dict(base_kwargs)
        kwargs.update(case.overrides)
        kwargs["output_dir"] = str(exp_dir)

        print("=" * 90)
        print(f"[Run {idx:02d}/{len(cases):02d}] {case.case_id} | {case.desc}")
        print(f"output_dir: {exp_dir}")
        print("overrides:", case.overrides)
        train(kwargs)


def _load_series(root: Path, case: ControlCase) -> Optional[SeriesRecord]:
    exp_dir = root / _case_dir_name(case)
    config = _safe_load_json(exp_dir / "config.json")
    episode_metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    if not episode_metrics:
        return None

    return SeriesRecord(
        case_id=case.case_id,
        desc=case.desc,
        exp_dir=exp_dir,
        config=config,
        episode=_extract_episode_axis(episode_metrics),
        qos=_extract_qos_metric(episode_metrics),
        lamb=_extract_metric(episode_metrics, "lambda_penalty"),
        qos_signal=_extract_qos_signal_metric(episode_metrics),
    )


def _build_summary(records: Sequence[SeriesRecord], tail_window: int, post_start_episode: int) -> List[Dict]:
    rows = []
    for rec in records:
        row = {
            "case_id": rec.case_id,
            "desc": rec.desc,
            "exp_dir": str(rec.exp_dir),
            "episodes": int(rec.episode[-1]) if rec.episode.size > 0 else -1,
            "seed": int(rec.config.get("seed", -1)),
            "rth": float(rec.config.get("paper_fixed_qos_threshold", np.nan)),
            "lambda_lr": float(rec.config.get("lambda_lr", np.nan)),
            "pfg_guidance_weight": float(rec.config.get("pfg_guidance_weight", np.nan)),
            "core_penalty_warmup_episodes": int(rec.config.get("core_penalty_warmup_episodes", -1)),
            "qos_last": _tail_last(rec.qos),
            "qos_tail_mean": _tail_mean(rec.qos, tail_window),
            "qos_post500_mean": _post_window_mean(rec.episode, rec.qos, post_start_episode),
            "qos_post500_zero_ratio": _post_zero_ratio(rec.episode, rec.qos, post_start_episode),
            "lambda_last": _tail_last(rec.lamb),
            "lambda_tail_mean": _tail_mean(rec.lamb, tail_window),
            "lambda_post500_slope": _post_lambda_slope(rec.episode, rec.lamb, post_start_episode),
            "qos_signal_tail_mean": _tail_mean(rec.qos_signal, tail_window),
            "qos_signal_post500_mean": _post_window_mean(rec.episode, rec.qos_signal, post_start_episode),
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["qos_post500_mean"] if np.isfinite(r["qos_post500_mean"]) else np.inf,
            r["qos_tail_mean"] if np.isfinite(r["qos_tail_mean"]) else np.inf,
            r["lambda_tail_mean"] if np.isfinite(r["lambda_tail_mean"]) else np.inf,
        )
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def _save_summary(root: Path, summary: Sequence[Dict]):
    json_path = root / "controls_summary.json"
    csv_path = root / "controls_summary.csv"

    json_path.write_text(json.dumps(list(summary), indent=2), encoding="utf-8")

    if summary:
        fields = list(summary[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in summary:
                writer.writerow(row)


def _plot_qos_lambda(records: Sequence[SeriesRecord], summary: Sequence[Dict], root: Path, smooth_window: int, max_episode: int):
    if not records:
        return

    rank_map = {row["case_id"]: int(row["rank"]) for row in summary}
    ordered = sorted(records, key=lambda r: rank_map.get(r.case_id, 10_000))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    cmap = plt.get_cmap("tab10")

    for idx, rec in enumerate(ordered):
        color = cmap(idx % 10)
        rank = rank_map.get(rec.case_id, -1)
        label = f"#{rank} {rec.case_id}:{rec.desc}"

        x = rec.episode
        y_qos = rec.qos
        y_lam = rec.lamb

        if max_episode > 0:
            keep = x <= max_episode
            x = x[keep]
            y_qos = y_qos[keep]
            y_lam = y_lam[keep]

        if x.size == 0:
            continue

        sq = _rolling_mean(y_qos, smooth_window)
        sl = _rolling_mean(y_lam, smooth_window)

        axes[0].plot(x, y_qos, linewidth=0.8, alpha=0.14, color=color)
        axes[1].plot(x, y_lam, linewidth=0.8, alpha=0.14, color=color)

        axes[0].plot(x[: sq.size], sq, linewidth=1.8, color=color, label=label)
        axes[1].plot(x[: sl.size], sl, linewidth=1.8, color=color, label=label)

    axes[0].set_title("QoS Violation Convergence")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("QoS violation")
    axes[0].grid(alpha=0.25)
    axes[0].axvline(500, linestyle="--", linewidth=1.0, color="black", alpha=0.35)

    axes[1].set_title("Dual Variable lambda Convergence")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("lambda")
    axes[1].grid(alpha=0.25)
    axes[1].axvline(500, linestyle="--", linewidth=1.0, color="black", alpha=0.35)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    fig.suptitle("QoS/Lambda Control-Variable Comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 0.85, 0.95])

    save_path = root / "fig_controls_qos_lambda.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _print_top(summary: Sequence[Dict], topk: int):
    print("\nTop cases by post-500 QoS:")
    for row in list(summary)[: max(1, topk)]:
        print(
            "#{rank:02d} {case_id} | {desc} | QoS_post500={qos_post500_mean:.6f} | "
            "QoS_tail={qos_tail_mean:.6f} | QoS_zero_ratio_post500={qos_post500_zero_ratio:.3f} | "
            "lambda_tail={lambda_tail_mean:.6f} | lambda_slope_post500={lambda_post500_slope:.6e}".format(**row)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Control-variable experiments for QoS/lambda convergence using the best four-branch MB-PPO set."
    )

    parser.add_argument("--tag", type=str, default="", help="Output tag under mb_ppo_data when --root is empty.")
    parser.add_argument("--root", type=str, default="", help="Explicit output root directory.")

    parser.add_argument("--episodes", type=int, default=600, help="Training episodes per case.")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval.")
    parser.add_argument("--save-freq", type=int, default=200, help="Checkpoint frequency.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed shared by all cases.")
    parser.add_argument("--n-gts", type=int, default=6, help="Number of users.")

    parser.add_argument("--base-lr", type=float, default=6e-5, help="Best baseline lr.")
    parser.add_argument("--base-entropy", type=float, default=3e-3, help="Best baseline entropy coef.")
    parser.add_argument("--base-lambda-lr", type=float, default=4e-4, help="Best baseline lambda lr.")
    parser.add_argument("--base-ppo-epochs", type=int, default=10, help="Best baseline PPO epochs.")
    parser.add_argument("--base-rth", type=float, default=0.04, help="Best baseline fixed QoS threshold.")
    parser.add_argument("--lambda-init", type=float, default=0.3, help="Initial lambda.")
    parser.add_argument("--lambda-max", type=float, default=35.0, help="Lambda upper bound.")
    parser.add_argument("--reward-ee-scale", type=float, default=1.0, help="Base EE reward weight.")
    parser.add_argument("--reward-core-penalty-scale", type=float, default=10.0, help="Base core penalty weight.")
    parser.add_argument("--base-pfg-weight", type=float, default=0.0, help="Base proactive fairness guidance weight.")
    parser.add_argument("--base-core-warmup", type=int, default=1000, help="Base core penalty warmup episodes.")
    parser.add_argument("--core-penalty-start-scale", type=float, default=0.15, help="Initial core penalty scale factor.")
    parser.add_argument("--terminate-on-core-violation", action="store_true", help="Enable core-violation early termination.")
    parser.add_argument("--core-terminate-start-episode", type=int, default=1000, help="Core early-termination start episode.")
    parser.add_argument("--lambda-signal-aggregation", type=str, default="positive_mean", choices=["positive_mean", "signed_mean"], help="Lambda signal aggregation mode.")
    parser.add_argument("--tx-power-max-dbm", type=float, default=30.0, help="TX power cap in dBm.")

    parser.add_argument("--s1-rth", type=float, default=0.035, help="Scheme1: lower fixed QoS threshold.")
    parser.add_argument("--s2-lambda-lr", type=float, default=8e-4, help="Scheme2: higher lambda lr.")
    parser.add_argument("--s3-pfg-weight", type=float, default=0.2, help="Scheme3: reward-weight change via proactive fairness guidance weight.")
    parser.add_argument("--s4-core-warmup", type=int, default=2000, help="Scheme4: longer core-penalty warmup episodes.")

    parser.add_argument("--tail-window", type=int, default=100, help="Tail window for summary metrics.")
    parser.add_argument("--post-start-episode", type=int, default=500, help="Post window start episode for trend checks.")
    parser.add_argument("--smooth-window", type=int, default=25, help="Smoothing window for plotting.")
    parser.add_argument("--max-episode", type=int, default=0, help="If >0, clip plots to episodes <= this value.")
    parser.add_argument("--topk", type=int, default=5, help="Print top-k cases.")

    parser.add_argument("--dry-run", action="store_true", help="Only create plan file.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only analyze existing runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing completed runs.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    root = _resolve_root(repo_root, args)
    root.mkdir(parents=True, exist_ok=True)

    cases = _build_cases(args)
    base_kwargs = _build_base_kwargs(args)
    _save_plan(root, cases, args, base_kwargs)

    print("=" * 90)
    print("QoS/Lambda control-variable plan")
    print("=" * 90)
    for c in cases:
        print(f"{c.case_id} | {c.desc} | overrides={c.overrides}")
    print("=" * 90)
    print(f"output root: {root}")

    if args.dry_run:
        print("Dry-run only. No training executed.")
        return

    if not args.skip_train:
        _run_cases(repo_root=repo_root, root=root, cases=cases, args=args, base_kwargs=base_kwargs)

    records = []
    for c in cases:
        rec = _load_series(root, c)
        if rec is not None:
            records.append(rec)

    if not records:
        print("No valid records found for summary/plotting.")
        return

    summary = _build_summary(records, tail_window=int(args.tail_window), post_start_episode=int(args.post_start_episode))
    _save_summary(root, summary)
    _plot_qos_lambda(
        records=records,
        summary=summary,
        root=root,
        smooth_window=int(args.smooth_window),
        max_episode=int(args.max_episode),
    )

    print(f"Saved summary JSON: {root / 'controls_summary.json'}")
    print(f"Saved summary CSV: {root / 'controls_summary.csv'}")
    print(f"Saved figure: {root / 'fig_controls_qos_lambda.png'}")
    _print_top(summary, topk=int(args.topk))


if __name__ == "__main__":
    main()
