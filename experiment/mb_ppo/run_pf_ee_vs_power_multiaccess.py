import argparse
import copy
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as SN
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algo.mb_ppo.mb_ppo_config import DEFAULT_MB_PPO_CONFIG
from algo.mb_ppo.mb_ppo_learner import MultiBranchPPOLearner
from algo.mb_ppo.run_mbppo import _build_env, train
from algo.mb_ppo.utils import check_args_sanity, set_rand_seed
from experiment.mb_ppo import maps as maps_mod


MAP_REGISTRY = {
    "clustered500": maps_mod.ClusteredMap500,
    "split24": maps_mod.ClusteredMap500Split24,
    "split42densea": maps_mod.ClusteredMap500Split42DenseA,
    "nearfar_extreme": maps_mod.ClusteredMap500NearFarExtreme,
}


def dbm_to_w(dbm: float) -> float:
    return float(1e-3 * np.power(10.0, float(dbm) / 10.0))


def _safe_load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Invalid JSON object in: {path}")


def _safe_dump_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_power_tag(dbm: float) -> str:
    s = f"{float(dbm):.2f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _baseline_label(baseline: str) -> str:
    key = str(baseline).strip().lower()
    if key == "mbppo":
        return "MB-PPO + RSMA"
    if key == "sdma":
        return "SDMA (alpha_c=0)"
    if key == "noma":
        return "NOMA"
    return key


def _select_map_class(map_name: str):
    key = str(map_name).strip().lower()
    if key not in MAP_REGISTRY:
        raise ValueError(f"Unsupported map_name: {map_name}. Choices: {sorted(MAP_REGISTRY.keys())}")
    return MAP_REGISTRY[key]


_CKPT_PATTERN = re.compile(r"checkpoint_episode(\d+)\.pt$")


def _latest_checkpoint(exp_dir: Path) -> Path:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    best_ep = -1
    best_path = None
    for p in ckpt_dir.glob("checkpoint_episode*.pt"):
        m = _CKPT_PATTERN.search(p.name)
        if m is None:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best_path = p

    if best_path is None:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")
    return best_path


def _extract_map_name(raw_map_value) -> str:
    text = str(raw_map_value)
    for cls in [
        "ClusteredMap500NearFarExtreme",
        "ClusteredMap500Split42DenseA",
        "ClusteredMap500Split24",
        "ClusteredMap500",
    ]:
        if cls in text:
            return cls
    if hasattr(maps_mod, text):
        return text
    return "ClusteredMap500"


def _map_class_from_saved_config(cfg: Dict):
    map_name = _extract_map_name(cfg.get("map", "ClusteredMap500"))
    if hasattr(maps_mod, map_name):
        return getattr(maps_mod, map_name)
    return maps_mod.ClusteredMap500


def _build_args_from_saved_config(cfg: Dict):
    merged = copy.deepcopy(DEFAULT_MB_PPO_CONFIG)
    merged.update(cfg)
    merged["map"] = _map_class_from_saved_config(cfg)

    args = SN(**merged)
    args = check_args_sanity(args)
    return args


def _evaluate_checkpoint_deterministic(
    config_path: Path,
    checkpoint_path: Path,
    eval_episodes: int,
    eval_seed: int,
) -> Dict:
    cfg = _safe_load_json(config_path)
    args = _build_args_from_saved_config(cfg)

    # Ensure all baselines use the exact same random map sequence at one power point.
    set_rand_seed(eval_seed)

    env = _build_env(args)
    env_info = env.get_env_info()
    learner = MultiBranchPPOLearner(env_info, args)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    learner.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    learner.actor_critic.eval()

    pf_ee_values = []
    for _ in range(int(eval_episodes)):
        obs, state, _ = env.reset()
        done = False
        info = {}

        while not done:
            actions, _, _, _ = learner.take_actions(obs, state, deterministic=True)
            obs, state, _, done, info = env.step(actions)

        pf_ee = float(info.get("episode_pf_energy_efficiency", info.get("pf_energy_efficiency", 0.0)))
        pf_ee_values.append(pf_ee)

    arr = np.asarray(pf_ee_values, dtype=np.float64)
    return {
        "pf_ee_mean": float(np.mean(arr)),
        "pf_ee_std": float(np.std(arr)),
        "pf_ee_var": float(np.var(arr)),
        "num_eval_episodes": int(arr.size),
    }


def _save_summary_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_pf_ee_vs_power(rows: List[Dict], baseline_order: List[str], save_path: Path):
    if not rows:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 6.2))
    colors = {
        "mbppo": "#1f77b4",
        "sdma": "#ff7f0e",
        "noma": "#2ca02c",
    }

    for baseline in baseline_order:
        b = str(baseline).strip().lower()
        data = [r for r in rows if str(r["baseline"]).strip().lower() == b]
        if not data:
            continue
        data = sorted(data, key=lambda x: float(x["pmax_dbm"]))

        x = np.asarray([float(r["pmax_dbm"]) for r in data], dtype=np.float64)
        y = np.asarray([float(r["pf_ee_mean"]) for r in data], dtype=np.float64)
        s = np.asarray([float(r["pf_ee_std"]) for r in data], dtype=np.float64)

        color = colors.get(b, None)
        ax.plot(x, y, marker="o", linewidth=2.0, label=_baseline_label(b), color=color)
        ax.fill_between(x, y - s, y + s, alpha=0.18, color=color)

    ax.set_xlabel("Max transmit power P_max (dBm)")
    ax.set_ylabel("PF-EE")
    ax.set_title("PF-EE vs P_max under Different Multiple Access Schemes")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _run_training_case(exp_dir: Path, run_kwargs: Dict, skip_existing: bool) -> Tuple[Path, str]:
    config_path = exp_dir / "config.json"
    if skip_existing and config_path.exists():
        ckpt = _latest_checkpoint(exp_dir)
        return ckpt, "skipped_existing"

    train(run_kwargs)
    ckpt = _latest_checkpoint(exp_dir)
    return ckpt, "trained"


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate PF-EE vs max TX power for MB-PPO+RSMA, SDMA, and NOMA with deterministic Monte Carlo evaluation."
    )
    parser.add_argument("--root", type=str, default="", help="Output root directory.")
    parser.add_argument("--map-name", type=str, default="nearfar_extreme", choices=sorted(MAP_REGISTRY.keys()))
    parser.add_argument("--baselines", type=str, nargs="*", default=["mbppo", "sdma", "noma"])
    parser.add_argument("--pmax-dbms", type=float, nargs="*", default=[20.0, 25.0, 30.0, 35.0, 40.0])
    parser.add_argument("--anchor-dbm", type=float, default=30.0, help="Anchor power used for pretraining before fine-tuning.")
    parser.add_argument("--disable-finetune", action="store_true", help="If set, each power point trains from scratch.")

    parser.add_argument("--episodes-anchor", type=int, default=3000, help="Training episodes at anchor power.")
    parser.add_argument("--episodes-finetune", type=int, default=600, help="Fine-tuning episodes for non-anchor power points.")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=200)

    parser.add_argument("--seed", type=int, default=10, help="Training seed used for all runs.")
    parser.add_argument("--seed-eval", type=int, default=20260420, help="Base eval seed (per power uses seed_eval + index).")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Deterministic Monte Carlo episodes per data point.")

    parser.add_argument("--n-gts", type=int, default=6)
    parser.add_argument("--reward-objective", type=str, default="pf_ee", choices=["pf_ee", "sum_ee", "max_min"])
    parser.add_argument("--pf-rate-ref-mbps", type=float, default=0.02)
    parser.add_argument("--pf-log-scale", type=float, default=100.0)
    parser.add_argument("--reward-objective-ref-pf-ee", type=float, default=12.0)
    parser.add_argument("--reward-objective-ref-sum-ee", type=float, default=0.02)
    parser.add_argument("--reward-objective-ref-max-min", type=float, default=0.06)
    parser.add_argument("--reward-output-scale", type=float, default=6.0)

    parser.add_argument("--reward-power-penalty-scale", type=float, default=1.0)
    parser.add_argument("--reward-wall-penalty-scale", type=float, default=2.0)
    parser.add_argument("--reward-core-penalty-scale", type=float, default=10.0)

    parser.add_argument("--skip-existing", action="store_true", help="Skip training if config/checkpoint already exists.")
    args = parser.parse_args()

    baselines = [str(b).strip().lower() for b in args.baselines]
    for b in baselines:
        if b not in {"mbppo", "sdma", "noma"}:
            raise ValueError(f"Unsupported baseline: {b}. Allowed: mbppo/sdma/noma.")

    pmax_dbms = sorted({float(v) for v in args.pmax_dbms})
    if len(pmax_dbms) == 0:
        raise ValueError("pmax_dbms is empty.")

    use_finetune = not bool(args.disable_finetune)
    anchor_dbm = float(args.anchor_dbm)
    if use_finetune and anchor_dbm not in pmax_dbms:
        raise ValueError(f"anchor_dbm={anchor_dbm} must be included in pmax_dbms when fine-tuning is enabled.")

    map_cls = _select_map_class(args.map_name)

    repo_root = Path(__file__).resolve().parents[2]
    if str(args.root).strip():
        root = Path(args.root)
        if not root.is_absolute():
            root = repo_root / root
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = repo_root / "mb_ppo_data" / f"pf_ee_vs_pmax_multiaccess_{args.map_name}_{stamp}"

    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "map": map_cls,
        "reward_objective": str(args.reward_objective),
        "eval_interval": int(args.eval_interval),
        "save_freq": int(args.save_freq),
        "seed": int(args.seed),
        "n_gts": int(args.n_gts),
        "pf_rate_ref_mbps": float(args.pf_rate_ref_mbps),
        "pf_log_scale": float(args.pf_log_scale),
        "reward_objective_ref_pf_ee": float(args.reward_objective_ref_pf_ee),
        "reward_objective_ref_sum_ee": float(args.reward_objective_ref_sum_ee),
        "reward_objective_ref_max_min": float(args.reward_objective_ref_max_min),
        "reward_output_scale": float(args.reward_output_scale),
        "reward_power_penalty_scale": float(args.reward_power_penalty_scale),
        "reward_wall_penalty_scale": float(args.reward_wall_penalty_scale),
        "reward_core_penalty_scale": float(args.reward_core_penalty_scale),
        "num_eval_episodes": 5,
        "wall_penalty_normalizer": 500.0,
        "core_penalty_normalizer": 500.0,
        "terminate_on_core_violation": False,
        "core_terminate_start_episode": 10_000_000,
    }

    run_index: Dict[Tuple[str, float], Dict] = {}

    print("=" * 100)
    print("PF-EE vs P_max | Multi-access sweep")
    print("=" * 100)
    print(f"Output root: {root}")
    print(f"Map: {map_cls.__name__} ({args.map_name})")
    print(f"Baselines: {baselines}")
    print(f"P_max points (dBm): {pmax_dbms}")
    print(f"P_max points (W): {[dbm_to_w(v) for v in pmax_dbms]}")
    print(f"Deterministic MC episodes per point: {int(args.eval_episodes)}")

    for baseline in baselines:
        baseline_root = runs_root / baseline
        baseline_root.mkdir(parents=True, exist_ok=True)

        anchor_tag = _format_power_tag(anchor_dbm)
        anchor_dir = baseline_root / f"pmax_{anchor_tag}dbm"

        anchor_kwargs = dict(common_kwargs)
        anchor_kwargs.update(
            {
                "baseline": baseline,
                "episodes": int(args.episodes_anchor),
                "tx_power_max_dbm": float(anchor_dbm),
                "output_dir": str(anchor_dir),
            }
        )

        anchor_ckpt, anchor_status = _run_training_case(
            exp_dir=anchor_dir,
            run_kwargs=anchor_kwargs,
            skip_existing=bool(args.skip_existing),
        )
        print(
            f"[Train] baseline={baseline:>5} | pmax={anchor_dbm:>5.1f} dBm ({dbm_to_w(anchor_dbm):.4f} W) | {anchor_status} | ckpt={anchor_ckpt.name}"
        )

        run_index[(baseline, float(anchor_dbm))] = {
            "exp_dir": anchor_dir,
            "checkpoint": anchor_ckpt,
            "trained_by": "anchor",
        }

        for pmax_dbm in pmax_dbms:
            p = float(pmax_dbm)
            if p == float(anchor_dbm):
                continue

            tag = _format_power_tag(p)
            exp_dir = baseline_root / f"pmax_{tag}dbm"

            run_kwargs = dict(common_kwargs)
            run_kwargs.update(
                {
                    "baseline": baseline,
                    "tx_power_max_dbm": p,
                    "output_dir": str(exp_dir),
                }
            )

            if use_finetune:
                run_kwargs["episodes"] = int(args.episodes_finetune)
                run_kwargs["resume"] = str(anchor_ckpt)
                trained_by = f"finetune_from_{anchor_dbm}dbm"
            else:
                run_kwargs["episodes"] = int(args.episodes_anchor)
                trained_by = "from_scratch"

            ckpt, status = _run_training_case(
                exp_dir=exp_dir,
                run_kwargs=run_kwargs,
                skip_existing=bool(args.skip_existing),
            )
            print(
                f"[Train] baseline={baseline:>5} | pmax={p:>5.1f} dBm ({dbm_to_w(p):.4f} W) | {status} | ckpt={ckpt.name}"
            )

            run_index[(baseline, p)] = {
                "exp_dir": exp_dir,
                "checkpoint": ckpt,
                "trained_by": trained_by,
            }

    rows: List[Dict] = []
    for p_idx, pmax_dbm in enumerate(pmax_dbms):
        eval_seed = int(args.seed_eval) + int(p_idx)

        for baseline in baselines:
            key = (baseline, float(pmax_dbm))
            if key not in run_index:
                raise KeyError(f"Missing run index entry for baseline={baseline}, pmax_dbm={pmax_dbm}")

            exp_dir = run_index[key]["exp_dir"]
            checkpoint = run_index[key]["checkpoint"]
            stats = _evaluate_checkpoint_deterministic(
                config_path=exp_dir / "config.json",
                checkpoint_path=checkpoint,
                eval_episodes=int(args.eval_episodes),
                eval_seed=eval_seed,
            )

            row = {
                "baseline": baseline,
                "scheme_label": _baseline_label(baseline),
                "pmax_dbm": float(pmax_dbm),
                "pmax_w": dbm_to_w(float(pmax_dbm)),
                "trained_by": str(run_index[key]["trained_by"]),
                "train_seed": int(args.seed),
                "eval_seed": int(eval_seed),
                "num_eval_episodes": int(stats["num_eval_episodes"]),
                "pf_ee_mean": float(stats["pf_ee_mean"]),
                "pf_ee_std": float(stats["pf_ee_std"]),
                "pf_ee_var": float(stats["pf_ee_var"]),
                "run_dir": str(exp_dir),
                "checkpoint": str(checkpoint),
            }
            rows.append(row)
            print(
                f"[Eval ] baseline={baseline:>5} | pmax={float(pmax_dbm):>5.1f} dBm | seed={eval_seed} | "
                f"PF-EE={row['pf_ee_mean']:.6f} ± {row['pf_ee_std']:.6f}"
            )

    summary_json = root / "pf_ee_vs_pmax_multiaccess_summary.json"
    summary_csv = root / "pf_ee_vs_pmax_multiaccess_summary.csv"
    plot_path = root / "fig_pf_ee_vs_pmax_multiaccess.png"

    _safe_dump_json(summary_json, rows)
    _save_summary_csv(summary_csv, rows)
    _plot_pf_ee_vs_power(rows, baseline_order=baselines, save_path=plot_path)

    print("\nSaved outputs:")
    print(f"- {summary_json}")
    print(f"- {summary_csv}")
    print(f"- {plot_path}")


if __name__ == "__main__":
    main()
