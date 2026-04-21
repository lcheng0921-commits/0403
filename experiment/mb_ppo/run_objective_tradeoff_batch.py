import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from algo.mb_ppo.run_mbppo import train
from experiment.mb_ppo.maps import ClusteredMap500NearFarExtreme, ClusteredMap500Split42DenseA


@dataclass
class ObjectiveCase:
    objective: str
    label: str


MAP_REGISTRY = {
    "split42densea": ClusteredMap500Split42DenseA,
    "nearfar_extreme": ClusteredMap500NearFarExtreme,
}


def _safe_load_pickle(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _tail_mean(arr: np.ndarray, tail: int) -> float:
    if arr.size == 0:
        return float("nan")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    tw = int(max(1, min(tail, valid.size)))
    return float(np.mean(valid[-tw:]))


def _extract_metric(metrics: List[Dict], keys: List[str]) -> np.ndarray:
    for k in keys:
        vals = np.asarray([float(m.get(k, np.nan)) for m in metrics], dtype=np.float32)
        if vals.size > 0 and np.any(np.isfinite(vals)):
            return vals
    return np.zeros((0,), dtype=np.float32)


def _summarize_case(exp_dir: Path, tail_window: int) -> Dict:
    metrics = _safe_load_pickle(exp_dir / "vars" / "episode_metrics.pickle") or []
    if not metrics:
        return {
            "exp_dir": str(exp_dir),
            "episodes": 0,
        }

    x = np.asarray([int(m.get("episode", i + 1)) for i, m in enumerate(metrics)], dtype=np.int32)
    ee = _extract_metric(metrics, ["episode_energy_efficiency", "energy_efficiency"]) 
    comm_ee = _extract_metric(metrics, ["episode_comm_ee", "slot_comm_ee"])
    pf_ee = _extract_metric(metrics, ["episode_pf_energy_efficiency", "pf_energy_efficiency"])
    pf_comm_ee = _extract_metric(metrics, ["episode_pf_comm_ee", "slot_pf_comm_ee"])
    jain = _extract_metric(metrics, ["jain_fairness", "effective_fairness"])
    step_reward = _extract_metric(metrics, ["step_reward_mean"]) 
    velocity = _extract_metric(metrics, ["velocity"])
    tx_power = _extract_metric(metrics, ["tx_power"])
    power_violation = _extract_metric(metrics, ["tx_power_violation"])

    return {
        "exp_dir": str(exp_dir),
        "episodes": int(x[-1]) if x.size > 0 else 0,
        "pf_ee_tail_mean": _tail_mean(pf_ee, tail_window),
        "ee_tail_mean": _tail_mean(ee, tail_window),
        "pf_comm_ee_tail_mean": _tail_mean(pf_comm_ee, tail_window),
        "comm_ee_tail_mean": _tail_mean(comm_ee, tail_window),
        "jain_tail_mean": _tail_mean(jain, tail_window),
        "step_reward_tail_mean": _tail_mean(step_reward, tail_window),
        "velocity_tail_mean": _tail_mean(velocity, tail_window),
        "tx_power_tail_mean": _tail_mean(tx_power, tail_window),
        "power_violation_tail_mean": _tail_mean(power_violation, tail_window),
    }


def _save_summary(root: Path, rows: List[Dict]):
    json_path = root / "objective_tradeoff_summary.json"
    csv_path = root / "objective_tradeoff_summary.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _print_summary(rows: List[Dict]):
    print("\nSummary (tail means):")
    for r in rows:
        print(
            "{objective:>7} | reward_tail={step_reward_tail_mean:7.3f} | "
            "Jain_tail={jain_tail_mean:7.4f} | TotalEE_tail={ee_tail_mean:9.3f} | "
            "CommEE_tail={comm_ee_tail_mean:9.3f} | "
            "vel_tail={velocity_tail_mean:7.3f} | pwr_violation_tail={power_violation_tail_mean:8.5f}".format(**r)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train MB-PPO under three objectives (Max Sum-EE / Max PF-EE / Max-Min) on the same map."
    )
    parser.add_argument("--root", type=str, default="", help="Output root. Default: mb_ppo_data/objective_tradeoff_<map_name>_<date>")
    parser.add_argument(
        "--map-name",
        type=str,
        default="split42densea",
        choices=sorted(MAP_REGISTRY.keys()),
        help="Scenario topology used by all objective runs.",
    )
    parser.add_argument("--n-gts", type=int, default=6, help="Number of users.")
    parser.add_argument("--episodes", type=int, default=600, help="Episodes per objective run.")
    parser.add_argument("--eval-interval", type=int, default=20, help="Evaluation interval.")
    parser.add_argument("--save-freq", type=int, default=200, help="Checkpoint save frequency.")
    parser.add_argument("--seed", type=int, default=10, help="Shared seed for all objective runs.")
    parser.add_argument("--tail-window", type=int, default=100, help="Tail window used in summary checks.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a case if its config.json already exists.")
    parser.add_argument("--tx-power-max-dbm", type=float, default=30.0, help="Per-slot TX power cap (dBm).")
    parser.add_argument("--pf-rate-ref-mbps", type=float, default=0.1, help="PF log utility rate reference (Mbps).")
    parser.add_argument("--pf-log-scale", type=float, default=100.0, help="PF log utility scaling before EE division.")

    # Keep penalties fixed across objectives.
    parser.add_argument("--reward-power-penalty-scale", type=float, default=1.0)
    parser.add_argument("--reward-wall-penalty-scale", type=float, default=2.0)
    parser.add_argument("--reward-core-penalty-scale", type=float, default=10.0)

    # Objective alignment and final reward scaling.
    parser.add_argument("--reward-objective-ref-pf-ee", type=float, default=8.0)
    parser.add_argument("--reward-objective-ref-sum-ee", type=float, default=0.02)
    parser.add_argument("--reward-objective-ref-max-min", type=float, default=0.06)
    parser.add_argument("--reward-output-scale", type=float, default=6.0)

    args = parser.parse_args()
    map_name = str(args.map_name).strip().lower()
    map_cls = MAP_REGISTRY[map_name]

    repo_root = Path(__file__).resolve().parents[2]
    if args.root.strip():
        root = Path(args.root)
        if not root.is_absolute():
            root = repo_root / root
    else:
        date_tag = datetime.now().strftime("%Y%m%d")
        root = repo_root / "mb_ppo_data" / f"objective_tradeoff_{map_name}_{date_tag}"

    root.mkdir(parents=True, exist_ok=True)

    cases = [
        ObjectiveCase(objective="sum_ee", label="Max Sum-EE"),
        ObjectiveCase(objective="pf_ee", label="Max PF-EE"),
        ObjectiveCase(objective="max_min", label="Max-Min"),
    ]

    # Force same map family and same core training setup across all three objectives.
    common_kwargs = {
        "baseline": "mbppo",
        "map": map_cls,
        "episodes": int(args.episodes),
        "eval_interval": int(args.eval_interval),
        "save_freq": int(args.save_freq),
        "seed": int(args.seed),
        "n_gts": int(args.n_gts),
        "tx_power_max_dbm": float(args.tx_power_max_dbm),
        "pf_rate_ref_mbps": float(args.pf_rate_ref_mbps),
        "pf_log_scale": float(args.pf_log_scale),
        "reward_power_penalty_scale": float(args.reward_power_penalty_scale),
        "reward_wall_penalty_scale": float(args.reward_wall_penalty_scale),
        "reward_core_penalty_scale": float(args.reward_core_penalty_scale),
        "reward_objective_ref_pf_ee": float(args.reward_objective_ref_pf_ee),
        "reward_objective_ref_sum_ee": float(args.reward_objective_ref_sum_ee),
        "reward_objective_ref_max_min": float(args.reward_objective_ref_max_min),
        "reward_output_scale": float(args.reward_output_scale),
        "wall_penalty_normalizer": 500.0,
        "core_penalty_normalizer": 500.0,
        "terminate_on_core_violation": False,
        "core_terminate_start_episode": 10_000_000,
    }

    print("=" * 100)
    print("Objective tradeoff batch")
    print("=" * 100)
    print(f"Output root: {root}")
    print(f"Map: {map_cls.__name__} ({map_name})")
    print(f"Seed: {int(args.seed)} | Episodes/case: {int(args.episodes)}")
    print(f"n_gts={int(args.n_gts)} | tx_power_max_dbm={float(args.tx_power_max_dbm):.2f}")
    print(
        "PF shaping -> pf_rate_ref_mbps={:.4f}, pf_log_scale={:.1f}".format(
            float(args.pf_rate_ref_mbps),
            float(args.pf_log_scale),
        )
    )
    print(
        "Penalty scales fixed across objectives -> power={:.3f}, wall={:.3f}, core={:.3f}".format(
            float(args.reward_power_penalty_scale),
            float(args.reward_wall_penalty_scale),
            float(args.reward_core_penalty_scale),
        )
    )
    print("Spatial penalty normalization -> wall/500, core/500")
    print(
        "Reward refs -> pf_ee={:.3f}, sum_ee={:.3f}, max_min={:.3f}, output_scale={:.3f}".format(
            float(args.reward_objective_ref_pf_ee),
            float(args.reward_objective_ref_sum_ee),
            float(args.reward_objective_ref_max_min),
            float(args.reward_output_scale),
        )
    )

    summary_rows = []

    for idx, case in enumerate(cases, start=1):
        exp_dir = root / f"{case.objective}_ep{int(args.episodes)}_seed{int(args.seed)}"

        if (exp_dir / "config.json").exists() and args.skip_existing:
            print(f"[Skip {idx}/{len(cases)}] {case.label}: {exp_dir}")
        else:
            run_kwargs = dict(common_kwargs)
            run_kwargs.update(
                {
                    "reward_objective": str(case.objective),
                    "output_dir": str(exp_dir),
                }
            )
            print(f"\n[Run {idx}/{len(cases)}] {case.label} -> {exp_dir.name}")
            train(run_kwargs)

        row = {
            "objective": str(case.objective),
            "label": str(case.label),
        }
        row.update(_summarize_case(exp_dir=exp_dir, tail_window=int(args.tail_window)))
        summary_rows.append(row)

    _save_summary(root, summary_rows)
    _print_summary(summary_rows)

    print("\nSaved:")
    print(f"- {root / 'objective_tradeoff_summary.json'}")
    print(f"- {root / 'objective_tradeoff_summary.csv'}")


if __name__ == "__main__":
    main()
