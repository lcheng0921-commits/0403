import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_load_pickle(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "baseline" in data:
        return data
    if isinstance(data, dict) and len(data) == 1:
        nested = next(iter(data.values()))
        if isinstance(nested, dict):
            return nested
    return {}


def _kmeans_two_clusters(points: np.ndarray, max_iter: int = 30) -> np.ndarray:
    n = points.shape[0]
    if n <= 1:
        return np.zeros((n,), dtype=np.int32)

    idx_a = int(np.argmin(points[:, 0]))
    idx_b = int(np.argmax(points[:, 0]))
    c = np.stack([points[idx_a], points[idx_b]], axis=0).astype(np.float32)

    labels = np.zeros((n,), dtype=np.int32)
    for _ in range(max_iter):
        d0 = np.sum((points - c[0]) ** 2, axis=1)
        d1 = np.sum((points - c[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in (0, 1):
            mask = labels == k
            if np.any(mask):
                c[k] = np.mean(points[mask], axis=0)

    if np.sum(labels == 0) == 0 or np.sum(labels == 1) == 0:
        thr = float(np.median(points[:, 0]))
        labels = (points[:, 0] >= thr).astype(np.int32)

    return labels.astype(np.int32)


def _simulate_circular_reference(
    uav_init_xy: np.ndarray,
    steps: int,
    range_pos: float,
    episode_length: int,
    velocity_bound: float,
) -> np.ndarray:
    center = np.array([range_pos / 2.0, range_pos / 2.0], dtype=np.float32)
    radius = 0.30 * float(range_pos)

    traj = [uav_init_xy.astype(np.float32).copy()]
    cur = uav_init_xy.astype(np.float32).copy()

    den = max(1, int(episode_length))
    v_bound = float(max(0.0, velocity_bound))

    for t in range(1, steps + 1):
        phase = 2.0 * np.pi * (float(t) / float(den))
        target = center + radius * np.array([np.cos(phase), np.sin(phase)], dtype=np.float32)
        delta = target - cur
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            step = min(dist, v_bound)
            cur = cur + (delta / dist) * step
        traj.append(cur.copy())

    return np.asarray(traj, dtype=np.float32)


def _extract_style_traces(exp_dir: Path, style_last_k: int, style_min_episode: int) -> Tuple[Dict, List[Dict]]:
    cfg = _load_config(exp_dir / "config.json")
    eval_traces = _safe_load_pickle(exp_dir / "vars" / "eval_traces.pickle") or []
    if not eval_traces:
        raise FileNotFoundError(f"No eval_traces.pickle entries found under {exp_dir / 'vars'}")

    traces = [t for t in eval_traces if int(t.get("episode", 0)) >= int(max(0, style_min_episode))]
    if not traces:
        traces = eval_traces

    if style_last_k > 0:
        traces = traces[-int(style_last_k) :]

    return cfg, traces


def main():
    parser = argparse.ArgumentParser(description="Plot MB-PPO trajectory against fixed circular trajectory on clustered map.")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory under mb_ppo_data.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiment/mb_ppo/pics/fig_traj_mbppo_vs_circular.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--style-last-k",
        type=int,
        default=20,
        help="Number of latest eval traces to summarize as overall style. <=0 means use all traces.",
    )
    parser.add_argument(
        "--style-min-episode",
        type=int,
        default=0,
        help="Only use eval traces with episode >= this value.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = repo_root / exp_dir

    if not (exp_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in: {exp_dir}")

    cfg, traces = _extract_style_traces(
        exp_dir=exp_dir,
        style_last_k=int(args.style_last_k),
        style_min_episode=int(args.style_min_episode),
    )

    ref_trace = traces[-1]
    gts = np.asarray(ref_trace.get("gts_init_pos", []), dtype=np.float32)
    uav_init = np.asarray(ref_trace.get("uav_init_pos", []), dtype=np.float32)
    if uav_init.size == 0:
        raise ValueError("Trace payload missing uav_init_pos.")

    mbppo_trajs = []
    for tr in traces:
        traj_all = np.asarray(tr.get("trajectory", []), dtype=np.float32)
        if traj_all.size == 0:
            continue
        mbppo_trajs.append(traj_all[:, 0, :])
    if not mbppo_trajs:
        raise ValueError("No valid trajectory found in selected eval traces.")

    min_len = int(min(t.shape[0] for t in mbppo_trajs))
    mbppo_trajs = [t[:min_len] for t in mbppo_trajs]
    mbppo_stack = np.stack(mbppo_trajs, axis=0)
    mbppo_style_traj = np.mean(mbppo_stack, axis=0)

    range_pos = float(ref_trace.get("range_pos", cfg.get("range_pos", 500.0)))
    episode_length = int(cfg.get("episode_length", max(1, traj_all.shape[0] - 1)))
    velocity_bound = float(cfg.get("velocity_bound", 20.0))

    steps = int(max(0, mbppo_style_traj.shape[0] - 1))
    circular_traj = _simulate_circular_reference(
        uav_init_xy=uav_init[0],
        steps=steps,
        range_pos=range_pos,
        episode_length=episode_length,
        velocity_bound=velocity_bound,
    )

    labels = _kmeans_two_clusters(gts) if gts.size > 0 else np.zeros((0,), dtype=np.int32)
    if gts.size > 0:
        c0 = int(np.sum(labels == 0))
        c1 = int(np.sum(labels == 1))
        # Keep Cluster1 as the smaller group for the requested 1+2 display.
        if c0 > c1:
            labels = 1 - labels
    c1 = gts[labels == 0] if gts.size > 0 else np.zeros((0, 2), dtype=np.float32)
    c2 = gts[labels == 1] if gts.size > 0 else np.zeros((0, 2), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7.2, 6.8))

    if c1.size > 0:
        ax.scatter(c1[:, 0], c1[:, 1], s=44, color="#2a9d8f", alpha=0.95, label="Cluster1 users")
    if c2.size > 0:
        ax.scatter(c2[:, 0], c2[:, 1], s=44, color="#e76f51", alpha=0.95, label="Cluster2 users")

    for traj in mbppo_trajs:
        ax.plot(traj[:, 0], traj[:, 1], color="#1d3557", linewidth=0.9, alpha=0.18)
    ax.plot(
        mbppo_style_traj[:, 0],
        mbppo_style_traj[:, 1],
        color="#1d3557",
        linewidth=2.4,
        label=f"MB-PPO style trajectory (mean of {len(mbppo_trajs)} eval traces)",
    )
    ax.plot(
        circular_traj[:, 0],
        circular_traj[:, 1],
        color="#f4a261",
        linewidth=2.0,
        linestyle="--",
        label="Fixed circular trajectory",
    )

    ax.scatter(
        mbppo_style_traj[0, 0],
        mbppo_style_traj[0, 1],
        s=62,
        marker="s",
        color="#1d3557",
        label="UAV start (style)",
    )
    ax.scatter(
        mbppo_style_traj[-1, 0],
        mbppo_style_traj[-1, 1],
        s=78,
        marker="*",
        color="#1d3557",
        label="UAV end (style)",
    )

    ax.set_xlim(0.0, range_pos)
    ax.set_ylim(0.0, range_pos)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("MB-PPO Style Trajectory vs Fixed Circular (2D)")
    ax.grid(alpha=0.28)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = repo_root / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved trajectory comparison figure to: {save_path}")
    print(f"source exp: {exp_dir}")
    ep_first = traces[0].get("episode", "unknown")
    ep_last = traces[-1].get("episode", "unknown")
    print(f"style traces: {len(mbppo_trajs)} | eval episode range: {ep_first} -> {ep_last}")
    print(f"points in style trajectory: {mbppo_style_traj.shape[0]}")


if __name__ == "__main__":
    main()
