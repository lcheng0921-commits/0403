import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExpRecord:
    exp_id: int
    exp_dir: Path
    baseline: str
    config: Dict
    train_returns: List[float]
    eval_history: List[Dict]
    episode_metrics: List[Dict]
    eval_traces: List[Dict]


def _safe_load_pickle(path: Path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        return pickle.load(f)


def _load_config(config_path: Path) -> Dict:
    with config_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        if 'baseline' in data:
            return data
        if len(data) == 1:
            nested = next(iter(data.values()))
            if isinstance(nested, dict):
                return nested

    return {}


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    w = int(max(1, window))
    if w == 1:
        return x
    kernel = np.ones(w, dtype=np.float32) / float(w)
    padded = np.pad(x, (w - 1, 0), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def _discover_exp_dirs(data_root: Path, exp_ids: Optional[List[int]]) -> List[Path]:
    if exp_ids:
        candidates = [data_root / f'exp{idx}' for idx in exp_ids]
        return [p for p in candidates if p.is_dir()]

    all_dirs = [p for p in data_root.glob('exp*') if p.is_dir() and p.name[3:].isdigit()]
    return sorted(all_dirs, key=lambda p: int(p.name[3:]))


def _load_records(repo_root: Path, exp_ids: Optional[List[int]]) -> List[ExpRecord]:
    data_root = repo_root / 'mb_ppo_data'
    exp_dirs = _discover_exp_dirs(data_root, exp_ids)
    records: List[ExpRecord] = []

    for exp_dir in exp_dirs:
        config_path = exp_dir / 'config.json'
        if not config_path.exists():
            continue

        config = _load_config(config_path)
        baseline = str(config.get('baseline', 'unknown'))
        exp_id = int(exp_dir.name[3:])

        train_returns = _safe_load_pickle(exp_dir / 'vars' / 'train_returns.pickle') or []
        eval_history = _safe_load_pickle(exp_dir / 'vars' / 'eval_history.pickle') or []
        episode_metrics = _safe_load_pickle(exp_dir / 'vars' / 'episode_metrics.pickle') or []
        eval_traces = _safe_load_pickle(exp_dir / 'vars' / 'eval_traces.pickle') or []

        records.append(
            ExpRecord(
                exp_id=exp_id,
                exp_dir=exp_dir,
                baseline=baseline,
                config=config,
                train_returns=train_returns,
                eval_history=eval_history,
                episode_metrics=episode_metrics,
                eval_traces=eval_traces,
            )
        )

    return records


def _pick_latest_by_baseline(records: List[ExpRecord]) -> Dict[str, ExpRecord]:
    chosen: Dict[str, ExpRecord] = {}
    for rec in records:
        old = chosen.get(rec.baseline)
        if old is None or rec.exp_id > old.exp_id:
            chosen[rec.baseline] = rec
    return chosen


def _final_metric_mean(rec: ExpRecord, key: str, ratio: float = 0.2) -> Optional[float]:
    if not rec.episode_metrics:
        return None

    values = [float(item.get(key, np.nan)) for item in rec.episode_metrics]
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    tail = max(1, int(arr.size * ratio))
    return float(np.mean(arr[-tail:]))


def plot_convergence(records: List[ExpRecord], output_dir: Path):
    target = [r for r in records if r.baseline == 'mbppo' and len(r.train_returns) > 0]
    if not target:
        return

    rec = sorted(target, key=lambda r: r.exp_id)[-1]
    train = np.asarray(rec.train_returns, dtype=np.float32)
    train_smooth = _rolling_mean(train, window=max(1, train.size // 30))

    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, train.size + 1), train, alpha=0.25, linewidth=1.0, label='train return raw')
    plt.plot(np.arange(1, train_smooth.size + 1), train_smooth, linewidth=2.0, label='train return smooth')

    if rec.eval_history:
        eval_ep = np.asarray([int(item.get('episode', 0)) for item in rec.eval_history], dtype=np.int32)
        eval_ret = np.asarray([float(item.get('eval_return_mean', 0.0)) for item in rec.eval_history], dtype=np.float32)
        plt.plot(eval_ep, eval_ret, marker='o', linewidth=1.6, label='eval return')

    plt.title(f'Convergence (MB-PPO, exp{rec.exp_id})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_convergence.png', dpi=300)
    plt.close()


def plot_trajectory(records: List[ExpRecord], output_dir: Path):
    target = [r for r in records if r.baseline == 'mbppo' and len(r.eval_traces) > 0]
    if not target:
        return

    rec = sorted(target, key=lambda r: r.exp_id)[-1]
    trace = rec.eval_traces[-1]

    traj = np.asarray(trace.get('trajectory', []), dtype=np.float32)
    if traj.size == 0:
        return

    gts_pos = np.asarray(trace.get('gts_init_pos', []), dtype=np.float32)
    uav_init = np.asarray(trace.get('uav_init_pos', []), dtype=np.float32)
    range_pos = float(trace.get('range_pos', rec.config.get('range_pos', 500.0)))

    plt.figure(figsize=(6.5, 6.0))

    if gts_pos.size > 0:
        plt.scatter(gts_pos[:, 0], gts_pos[:, 1], s=40, c='#2ca02c', label='GT')

    if uav_init.size > 0:
        plt.scatter(uav_init[:, 0], uav_init[:, 1], s=65, c='#1f77b4', marker='s', label='UAV init')

    traj2d = traj[:, 0, :]
    plt.plot(traj2d[:, 0], traj2d[:, 1], color='#1f77b4', linewidth=2.0, label='UAV trajectory')
    plt.scatter(traj2d[-1, 0], traj2d[-1, 1], s=70, c='#1f77b4', marker='*', label='UAV final')

    plt.xlim(0.0, range_pos)
    plt.ylim(0.0, range_pos)
    plt.title(f'2D Trajectory (MB-PPO, exp{rec.exp_id})')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_trajectory.png', dpi=300)
    plt.close()


def plot_fairness_bar(records: List[ExpRecord], output_dir: Path):
    latest = _pick_latest_by_baseline(records)
    if not latest:
        return

    baselines = []
    fairness = []
    for baseline, rec in sorted(latest.items(), key=lambda x: x[0]):
        value = _final_metric_mean(rec, key='effective_fairness')
        if value is None:
            continue
        baselines.append(baseline)
        fairness.append(value)

    if not fairness:
        return

    plt.figure(figsize=(7.2, 4.5))
    x = np.arange(len(baselines))
    bars = plt.bar(x, fairness, color='#4c72b0', width=0.62)
    for i, b in enumerate(bars):
        plt.text(b.get_x() + b.get_width() * 0.5, b.get_height() + 0.005, f'{fairness[i]:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(x, baselines, rotation=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel('Jain Fairness Index')
    plt.title('Fairness Comparison Across Baselines')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_fairness_bar.png', dpi=300)
    plt.close()


def _plot_xy_curve(records: List[ExpRecord], x_key: str, y_key: str, fig_name: str, title: str, x_label: str, y_label: str, output_dir: Path):
    grouped: Dict[str, List[tuple]] = {}

    for rec in records:
        x_val = rec.config.get(x_key)
        if x_val is None:
            continue
        y_val = _final_metric_mean(rec, key=y_key)
        if y_val is None:
            continue
        grouped.setdefault(rec.baseline, []).append((float(x_val), float(y_val)))

    if not grouped:
        return

    plt.figure(figsize=(7.2, 4.6))
    for baseline, points in sorted(grouped.items(), key=lambda x: x[0]):
        points = sorted(points, key=lambda t: t[0])
        x = np.asarray([p[0] for p in points], dtype=np.float32)
        y = np.asarray([p[1] for p in points], dtype=np.float32)
        plt.plot(x, y, marker='o', linewidth=1.8, label=baseline)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / fig_name, dpi=300)
    plt.close()


def plot_ee_qos(records: List[ExpRecord], output_dir: Path):
    _plot_xy_curve(
        records=records,
        x_key='qos_threshold',
        y_key='energy_efficiency',
        fig_name='fig_ee_vs_qos.png',
        title='Energy Efficiency vs QoS Threshold',
        x_label='QoS Threshold',
        y_label='Energy Efficiency',
        output_dir=output_dir,
    )


def plot_ee_power(records: List[ExpRecord], output_dir: Path):
    _plot_xy_curve(
        records=records,
        x_key='tx_power_max_dbm',
        y_key='energy_efficiency',
        fig_name='fig_ee_vs_power.png',
        title='Energy Efficiency vs Max Transmit Power',
        x_label='Max Transmit Power (dBm)',
        y_label='Energy Efficiency',
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description='Draw MB-PPO second-round figures from mb_ppo_data.')
    parser.add_argument('--exp-ids', type=int, nargs='*', default=None, help='Optional experiment ids to load, e.g. --exp-ids 3 7 9')
    parser.add_argument('--save-dir', type=str, default='experiment/mb_ppo/pics', help='Figure output directory.')
    parser.add_argument('--show', action='store_true', help='Display generated figures interactively.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / args.save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(repo_root=repo_root, exp_ids=args.exp_ids)
    if not records:
        print('No experiment records found under mb_ppo_data.')
        return

    plot_convergence(records, output_dir)
    plot_trajectory(records, output_dir)
    plot_ee_qos(records, output_dir)
    plot_fairness_bar(records, output_dir)
    plot_ee_power(records, output_dir)

    print(f'Generated figures in {output_dir}.')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
