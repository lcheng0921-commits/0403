import argparse
from pathlib import Path

from experiment.mb_ppo.diagnose_map_traj_checks import rollout_trace, summarize_trace, write_step_csv


def main():
    parser = argparse.ArgumentParser(description="Export first-100 rollout metrics and weak-user distance slope.")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory under mb_ppo_data.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename, e.g. checkpoint_episode500.pt.")
    parser.add_argument("--steps", type=int, default=100, help="Rollout steps to export.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for replay.")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    ckpt_path = exp_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    trace = rollout_trace(exp_dir=exp_dir, checkpoint_path=ckpt_path, steps=int(args.steps), seed=int(args.seed))
    summary = summarize_trace(trace)

    diag_dir = exp_dir / "diagnostics"
    csv_path = diag_dir / "first100_steps_rc_rk_ptx.csv"
    write_step_csv(trace["rows"], save_path=csv_path)

    print(f"csv_path={csv_path}")
    print(f"weak_dist_slope={summary['weak_dist_slope']}")
    print(f"weak_dist_start={summary['weak_dist_start']}")
    print(f"weak_dist_end={summary['weak_dist_end']}")
    print(f"steps={summary['steps']}")


if __name__ == "__main__":
    main()
