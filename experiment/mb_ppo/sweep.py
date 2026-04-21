import argparse
import subprocess
import sys
from pathlib import Path


def _run_cmd(cmd):
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def _run_one(baseline: str, episodes: int, eval_interval: int, save_freq: int, seed: int, qos: float, pmax: float):
    cmd = [
        sys.executable,
        '-m',
        'algo.mb_ppo.run_mbppo',
        '--baseline',
        baseline,
        '--episodes',
        str(episodes),
        '--eval-interval',
        str(eval_interval),
        '--save-freq',
        str(save_freq),
        '--seed',
        str(seed),
        '--qos-threshold',
        str(qos),
        '--tx-power-max-dbm',
        str(pmax),
    ]
    _run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(description='Run MB-PPO sweep experiments for second-round figures.')
    parser.add_argument('--mode', type=str, default='both', choices=['qos', 'power', 'both'], help='Sweep dimension.')
    parser.add_argument('--baselines', type=str, nargs='*', default=['mbppo', 'ppo', 'sdma', 'noma'], help='Baselines to run.')
    parser.add_argument('--episodes', type=int, default=200, help='Training episodes per experiment.')
    parser.add_argument('--eval-interval', type=int, default=20, help='Evaluation interval.')
    parser.add_argument('--save-freq', type=int, default=50, help='Checkpoint save frequency.')
    parser.add_argument('--seed', type=int, default=10, help='Base random seed.')
    parser.add_argument('--qos-values', type=float, nargs='*', default=[0.05, 0.1, 0.3, 0.5], help='QoS sweep values.')
    parser.add_argument('--power-sweep-qos', type=float, default=0.1, help='QoS threshold used during power sweep.')
    parser.add_argument('--power-values', type=float, nargs='*', default=[24.0, 27.0, 30.0, 33.0], help='Power sweep values in dBm.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    print(f'Repo root: {repo_root}')

    exp_id = 0
    if args.mode in ['qos', 'both']:
        for baseline in args.baselines:
            for qos in args.qos_values:
                exp_id += 1
                _run_one(
                    baseline=baseline,
                    episodes=args.episodes,
                    eval_interval=args.eval_interval,
                    save_freq=args.save_freq,
                    seed=args.seed + exp_id,
                    qos=qos,
                    pmax=30.0,
                )

    if args.mode in ['power', 'both']:
        for baseline in args.baselines:
            for pmax in args.power_values:
                exp_id += 1
                _run_one(
                    baseline=baseline,
                    episodes=args.episodes,
                    eval_interval=args.eval_interval,
                    save_freq=args.save_freq,
                    seed=args.seed + exp_id,
                    qos=args.power_sweep_qos,
                    pmax=pmax,
                )


if __name__ == '__main__':
    main()
