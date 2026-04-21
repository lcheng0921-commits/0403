import numpy as np

from mb_ppo_env.channel_model import AirToGroundChannel
from mb_ppo_env.utils import compute_jain_fairness_index, wrapper_obs, wrapper_state


def _softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    denom = float(np.sum(exp_x))
    if denom <= 1e-8:
        return np.ones_like(x, dtype=np.float32) / max(1, x.size)
    return (exp_x / denom).astype(np.float32)


def _safe_norm(x: np.ndarray, eps: float = 1e-6) -> float:
    return float(max(np.linalg.norm(x), eps))


class UbsRsmaEvn:
    """Base environment with mobility, user association and observation/state APIs.

    Legacy eavesdropper and secrecy-rate logic is intentionally removed.
    """

    unit = 100
    h_ubs = 100
    # Standard thermal noise PSD: -174 dBm/Hz.
    n0 = 1e-3 * np.power(10, -174 / 10)  # W/Hz
    bw = 1e6  # 1MHz
    fc = 2.4e9
    dt = 10
    scene = "urban"

    def __init__(
        self,
        map,
        fair_service=True,
        range_pos=800,
        episode_length=50,
        n_ubs=2,
        n_powers=10,
        n_moves=8,
        n_gts=100,
        n_eve=0,
        r_sense=np.inf,
        n_community=4,
        K_richan=10,
        jamming_power_bound=0,
        velocity_bound=50,
    ):
        self._fair_service = bool(fair_service)
        self.range_pos = float(range_pos)
        self.episode_length = int(episode_length)

        self.n_gts = int(n_gts)
        self.n_ubs = int(n_ubs)
        self.n_agents = int(n_ubs)
        self.n_powers = int(n_powers)
        self.n_moves = int(n_moves)
        self.n_eve = int(n_eve)  # kept for constructor compatibility
        self.r_sense = float(r_sense)
        self.K_richan = float(K_richan)
        self.n_community = int(n_community)
        self.velocity_bound = float(velocity_bound)
        self.jamming_power_bound = float(jamming_power_bound)

        self.map = map(
            range_pos=self.range_pos,
            n_eve=self.n_eve,
            n_gts=self.n_gts,
            n_ubs=self.n_ubs,
            n_community=self.n_community,
        )
        self.ATGChannel = AirToGroundChannel(self.scene, self.fc)

        self.t = 0
        self.ep_ret = np.zeros(self.n_agents, dtype=np.float32)
        self.mean_returns = 0.0

        self.pos_ubs = np.zeros((self.n_ubs, 2), dtype=np.float32)
        self.pos_gts = np.zeros((self.n_gts, 2), dtype=np.float32)

        self.d_u2g_level = np.zeros((self.n_ubs, self.n_gts), dtype=np.float32)
        self.d_u2u = np.zeros((self.n_ubs, self.n_ubs), dtype=np.float32)

        self.gt_served_by_uav = np.full(self.n_gts, -1, dtype=np.int32)
        self.ubs_serv_gts = [[] for _ in range(self.n_ubs)]

        self.rate_gt = np.zeros(self.n_gts, dtype=np.float32)
        self.rate_gt_t = np.zeros(self.n_gts, dtype=np.float32)
        self.rate_per_ubs_t = np.zeros(self.n_ubs, dtype=np.float32)
        self.aver_rate_per_gt = np.zeros(self.n_gts, dtype=np.float32)

        self.total_throughput_t = 0.0
        self.total_throughput = 0.0
        self.global_util = 0.0
        self.fair_idx = 1.0
        self.avg_fair_idx_per_episode = 0.0

        # Kept for compatibility with existing obs/state naming.
        self.ssr_ubsk_t = np.zeros(self.n_ubs, dtype=np.float32)

        self.latest_rho = 0.5
        self.latest_effective_rate = np.zeros(self.n_gts, dtype=np.float32)
        self.latest_qos_gap = np.zeros(self.n_gts, dtype=np.float32)

        g_max = float(self.ATGChannel.estimate_chan_gain(0.0, self.h_ubs, self.K_richan))
        snr_max = max(1e-6, (1e-3 * np.power(10, 30 / 10)) * g_max / (self.n0 * self.bw))
        self.achievable_rate_gts_max = float(self.bw * np.log2(1 + snr_max) * 1e-6)
        self.achievable_rate_ubs_max = float(self.achievable_rate_gts_max * max(1, self.n_gts))

        move_amounts = np.array([self.velocity_bound], dtype=np.float32).reshape(-1, 1)
        ang = 2 * np.pi * np.arange(max(1, self.n_moves)) / max(1, self.n_moves)
        move_dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
        self.avail_moves = np.concatenate((np.zeros((1, 2), dtype=np.float32), np.kron(move_amounts, move_dirs)))
        self.n_moves = int(self.avail_moves.shape[0])

        self.uav_traj = []
        self.throughput_list = []

    def reset(self):
        self.uav_traj = []
        self.throughput_list = []

        map_data = self.map.get_map()
        self.pos_ubs = np.asarray(map_data["pos_ubs"], dtype=np.float32).copy()
        self.pos_gts = np.asarray(map_data["pos_gts"], dtype=np.float32).copy()

        self.t = 1
        self.ep_ret = np.zeros(self.n_agents, dtype=np.float32)
        self.mean_returns = 0.0

        self.rate_gt.fill(0.0)
        self.rate_gt_t.fill(0.0)
        self.rate_per_ubs_t.fill(0.0)
        self.aver_rate_per_gt.fill(0.0)
        self.total_throughput_t = 0.0
        self.total_throughput = 0.0
        self.global_util = 0.0
        self.fair_idx = 1.0
        self.avg_fair_idx_per_episode = 0.0
        self.ssr_ubsk_t.fill(0.0)

        self.latest_rho = 0.5
        self.latest_effective_rate.fill(0.0)
        self.latest_qos_gap.fill(0.0)

        self.update_distance()
        self._assign_users()

        obs = wrapper_obs(self.get_obs())
        state = wrapper_state(self.get_state())

        init_info = {
            "range_pos": float(self.range_pos),
            "uav_init_pos": self.pos_ubs.copy(),
            "gts_init_pos": self.pos_gts.copy(),
        }
        return obs, state, init_info

    def update_distance(self):
        for u in range(self.n_ubs):
            for k in range(self.n_gts):
                self.d_u2g_level[u, k] = np.linalg.norm(self.pos_ubs[u] - self.pos_gts[k])

        for u in range(self.n_ubs):
            for v in range(self.n_ubs):
                self.d_u2u[u, v] = np.linalg.norm(self.pos_ubs[u] - self.pos_ubs[v])

    def _assign_users(self):
        self.gt_served_by_uav.fill(-1)
        self.ubs_serv_gts = [[] for _ in range(self.n_ubs)]
        for gt in range(self.n_gts):
            u = int(np.argmin(self.d_u2g_level[:, gt]))
            self.gt_served_by_uav[gt] = u
            self.ubs_serv_gts[u].append(gt)

    def get_env_info(self):
        obs = self.get_obs_size()
        gt_features_dim = int(obs["gt"][1])
        other_features_dim = int(obs["agent"] + np.prod(obs["ubs"]))
        return {
            "gt_features_dim": gt_features_dim,
            "other_features_dim": other_features_dim,
            "state_shape": int(self.get_state_size()),
            "n_moves": int(self.n_moves),
            "n_powers": int(self.n_powers),
            "n_agents": int(self.n_agents),
            "episode_limit": int(self.episode_length),
        }

    def step(self, actions):
        self.t += 1
        move_ids = np.asarray(actions.get("moves", np.zeros(self.n_agents)), dtype=np.int32)
        moves = self.avail_moves[np.clip(move_ids, 0, self.n_moves - 1)]
        self.pos_ubs = np.clip(self.pos_ubs + moves, 0.0, self.range_pos)

        self.update_distance()
        self._assign_users()
        self.latest_effective_rate.fill(0.0)
        self.rate_gt_t = self.latest_effective_rate.copy()
        self.total_throughput_t = 0.0
        self.total_throughput = float(self.total_throughput)

        reward = self.get_reward()
        self.ep_ret += reward
        self.mean_returns += float(np.mean(reward))
        self.fair_idx = float(compute_jain_fairness_index(self.aver_rate_per_gt + self.rate_gt_t))
        self.avg_fair_idx_per_episode += self.fair_idx
        done = self.get_terminate()

        info = {
            "EpRet": self.ep_ret,
            "EpLen": int(self.t),
            "mean_returns": float(self.mean_returns),
            "total_throughput": float(self.total_throughput),
            "global_util": float(self.global_util),
            "avg_fair_idx_per_episode": float(self.avg_fair_idx_per_episode),
            "BadMask": bool(self.t == self.episode_length),
        }

        obs = wrapper_obs(self.get_obs())
        state = wrapper_state(self.get_state())
        self.uav_traj.append(self.pos_ubs.copy())
        self.throughput_list.append(float(self.total_throughput_t))
        return obs, state, reward, done, info

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int):
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        ubs_feats = np.zeros(self.obs_ubs_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.obs_gt_feats_size, dtype=np.float32)

        own_feats[0:2] = self.pos_ubs[agent_id] / self.range_pos
        own_feats[2] = self.ssr_ubsk_t[agent_id] / max(self.achievable_rate_ubs_max, 1e-6)

        other_ubs = [u for u in range(self.n_agents) if u != agent_id]
        for j, u in enumerate(other_ubs):
            if self.d_u2u[agent_id, u] <= self.r_sense:
                ubs_feats[j, 0] = 1.0
                ubs_feats[j, 1:3] = (self.pos_ubs[u] - self.pos_ubs[agent_id]) / self.range_pos

        for gt in range(self.n_gts):
            gt_feats[gt, 0] = 1.0
            gt_feats[gt, 1:3] = (self.pos_gts[gt] - self.pos_ubs[agent_id]) / self.range_pos
            gt_feats[gt, 3] = self.latest_effective_rate[gt] / max(self.achievable_rate_gts_max, 1e-6)

        return {"agent": own_feats, "ubs": ubs_feats, "gt": gt_feats}

    def get_obs_size(self):
        return {"agent": self.obs_own_feats_size, "ubs": self.obs_ubs_feats_size, "gt": self.obs_gt_feats_size}

    @property
    def obs_own_feats_size(self):
        return 3

    @property
    def obs_ubs_feats_size(self):
        return self.n_agents - 1, 3

    @property
    def obs_gt_feats_size(self):
        return self.n_gts, 4

    def get_state(self):
        ubs_feats = np.zeros(self.state_ubs_feats_size(), dtype=np.float32)
        gt_feats = np.zeros(self.state_gt_feats_size(), dtype=np.float32)

        ubs_feats[:, 0:2] = self.pos_ubs / self.range_pos
        ubs_feats[:, 2] = self.ssr_ubsk_t / max(self.achievable_rate_ubs_max, 1e-6)

        gt_feats[:, 0:2] = self.pos_gts / self.range_pos
        gt_feats[:, 2] = self.latest_effective_rate / max(self.achievable_rate_gts_max, 1e-6)

        return np.concatenate((ubs_feats.flatten(), gt_feats.flatten()))

    def get_state_size(self):
        return int(np.prod(self.state_ubs_feats_size()) + np.prod(self.state_gt_feats_size()))

    def state_ubs_feats_size(self):
        return self.n_ubs, 3

    def state_gt_feats_size(self):
        return self.n_gts, 3

    def get_reward(self, reward_scale_rate=1.0):
        base = float(np.mean(self.latest_effective_rate))
        rewards = np.full(self.n_agents, base * reward_scale_rate, dtype=np.float32)
        return rewards

    def get_terminate(self):
        return bool(self.t == self.episode_length)


class MbPpoEnv(UbsRsmaEvn):
    """Continuous MB-PPO environment with paper-aligned RSMA transmission model.

    Key points:
    - Eavesdropper/secrecy pipeline removed.
    - RSMA rates computed via explicit SINR equations.
    - URA steering and complex Rician channel kept in-rate computation.
    - Baseline=ppo does not use hard L2/softmax power mapping.
    """

    def __init__(
        self,
        map,
        fair_service=True,
        range_pos=500,
        episode_length=100,
        n_ubs=1,
        n_powers=10,
        n_moves=16,
        n_gts=6,
        n_eve=0,
        r_sense=np.inf,
        n_community=4,
        K_richan=10,
        jamming_power_bound=0,
        velocity_bound=20,
        tx_power_max_dbm=30.0,
        reward_objective="pf_ee",
        reward_pf_scale=1.0,
        pf_rate_ref_mbps=0.1,
        pf_log_scale=100.0,
        reward_objective_ref_pf_ee=8.0,
        reward_objective_ref_sum_ee=0.02,
        reward_objective_ref_max_min=0.06,
        reward_output_scale=6.0,
        wall_penalty_normalizer=1.0,
        core_penalty_normalizer=1.0,
        reward_power_penalty_scale=1.0,
        reward_wall_penalty_scale=2.0,
        reward_core_penalty_scale=10.0,
        rsma_mode="rsma",
        baseline="mbppo",
        force_hard_mapping_for_ppo=False,
        common_precoding_dim=None,
        private_precoding_dim=None,
        resource_dim=None,
        precoding_dim=None,
        antenna_count=16,
        csi_complex_dim=4,
        fly_power_base=80.0,
        fly_power_coeff=0.12,
        phy_mapping_blend=1.0,
        precoding_gain_scale=1.0,
        interference_scale=1.0,
        state_h_scale=100.0,
        alpha_common_logit_bias=0.0,
        core_boundary_margin_ratio=0.20,
        terminate_on_core_violation=False,
        core_violation_terminate_patience=8,
        core_violation_terminate_threshold=2.0,
        core_terminate_start_episode=1,
        freeze_uav_trajectory=False,
        fixed_precoding_scheme="none",
        fixed_beta_mode="none",
        **legacy_kwargs,
    ):
        super().__init__(
            map=map,
            fair_service=fair_service,
            range_pos=range_pos,
            episode_length=episode_length,
            n_ubs=n_ubs,
            n_powers=n_powers,
            n_moves=n_moves,
            n_gts=n_gts,
            n_eve=n_eve,
            r_sense=r_sense,
            n_community=n_community,
            K_richan=K_richan,
            jamming_power_bound=jamming_power_bound,
            velocity_bound=velocity_bound,
        )

        self.reward_objective = str(reward_objective).strip().lower()
        if self.reward_objective not in {"pf_ee", "sum_ee", "max_min"}:
            raise ValueError(
                f"Unsupported reward_objective: {self.reward_objective}. "
                "Allowed: ['pf_ee', 'sum_ee', 'max_min']."
            )

        self.reward_pf_scale = float(reward_pf_scale)
        self.pf_rate_ref_mbps = float(max(1e-6, pf_rate_ref_mbps))
        self.pf_log_scale = float(max(1e-6, pf_log_scale))
        self.reward_objective_ref_pf_ee = float(max(1e-6, reward_objective_ref_pf_ee))
        self.reward_objective_ref_sum_ee = float(max(1e-6, reward_objective_ref_sum_ee))
        self.reward_objective_ref_max_min = float(max(1e-6, reward_objective_ref_max_min))
        self.reward_output_scale = float(max(1e-6, reward_output_scale))
        self.wall_penalty_normalizer = float(max(1e-6, wall_penalty_normalizer))
        self.core_penalty_normalizer = float(max(1e-6, core_penalty_normalizer))
        self.reward_power_penalty_scale = float(reward_power_penalty_scale)
        self.reward_wall_penalty_scale = float(max(0.0, reward_wall_penalty_scale))
        self.reward_core_penalty_scale = float(max(0.0, reward_core_penalty_scale))
        # Compatibility-only field for old runners; no longer used in reward update.
        self.lambda_penalty = float(max(0.0, legacy_kwargs.get("lambda_penalty", 0.0)))
        self.rsma_mode = str(rsma_mode)
        self.baseline = str(baseline)
        self.force_hard_mapping_for_ppo = bool(force_hard_mapping_for_ppo)

        self.antenna_count = int(max(1, antenna_count))
        self.csi_complex_dim = int(max(1, csi_complex_dim))

        self.tx_power_max_w = float(1e-3 * np.power(10.0, tx_power_max_dbm / 10.0))
        self.phy_mapping_blend = float(np.clip(phy_mapping_blend, 0.0, 1.0))
        self.precoding_gain_scale = float(max(1e-6, precoding_gain_scale))
        self.interference_scale = float(max(1e-6, interference_scale))
        # Scale CSI only in NN input features; keep physical channel equations unchanged.
        self.state_h_scale = float(max(1e-6, state_h_scale))
        # Positive logit bias for common stream before softmax(alpha).
        self.alpha_common_logit_bias = float(max(0.0, alpha_common_logit_bias))
        self.core_boundary_margin_ratio = float(np.clip(core_boundary_margin_ratio, 0.0, 0.49))
        self.core_boundary_margin = float(self.range_pos * self.core_boundary_margin_ratio)
        self.core_boundary_min = float(self.core_boundary_margin)
        self.core_boundary_max = float(self.range_pos - self.core_boundary_margin)
        self.terminate_on_core_violation = bool(terminate_on_core_violation)
        self.core_violation_terminate_patience = int(max(1, core_violation_terminate_patience))
        self.core_violation_terminate_threshold = float(max(0.0, core_violation_terminate_threshold))
        self.core_terminate_start_episode = int(max(1, core_terminate_start_episode))

        self.freeze_uav_trajectory = bool(freeze_uav_trajectory)
        fixed_precoding_scheme = str(fixed_precoding_scheme).strip().lower()
        if fixed_precoding_scheme not in {"none", "zf"}:
            raise ValueError(
                f"Unsupported fixed_precoding_scheme: {fixed_precoding_scheme}. "
                "Allowed: ['none', 'zf']."
            )
        self.fixed_precoding_scheme = fixed_precoding_scheme

        fixed_beta_mode = str(fixed_beta_mode).strip().lower()
        if fixed_beta_mode not in {"none", "uniform"}:
            raise ValueError(
                f"Unsupported fixed_beta_mode: {fixed_beta_mode}. "
                "Allowed: ['none', 'uniform']."
            )
        self.fixed_beta_mode = fixed_beta_mode

        self.curriculum_episode = 1
        self.curriculum_total_episodes = 1

        self.fly_power_base = float(fly_power_base)
        self.fly_power_coeff = float(fly_power_coeff)

        self.stream_count = self.n_gts + 1
        self.required_common_precoding_dim = 2 * self.antenna_count
        self.required_private_precoding_dim = 2 * self.n_gts * self.antenna_count
        self.required_resource_dim = 2 * self.n_gts + 2
        self.required_precoding_dim = self.required_common_precoding_dim + self.required_private_precoding_dim

        self.common_precoding_dim = int(
            self.required_common_precoding_dim if common_precoding_dim is None else common_precoding_dim
        )
        if self.common_precoding_dim != self.required_common_precoding_dim:
            raise ValueError(
                f"common_precoding_dim mismatch: got {self.common_precoding_dim}, "
                f"required {self.required_common_precoding_dim} (2 * antennas)."
            )

        self.private_precoding_dim = int(
            self.required_private_precoding_dim if private_precoding_dim is None else private_precoding_dim
        )
        if self.private_precoding_dim != self.required_private_precoding_dim:
            raise ValueError(
                f"private_precoding_dim mismatch: got {self.private_precoding_dim}, "
                f"required {self.required_private_precoding_dim} (2 * n_gts * antennas)."
            )

        self.resource_dim = int(self.required_resource_dim if resource_dim is None else resource_dim)
        if self.resource_dim != self.required_resource_dim:
            raise ValueError(
                f"resource_dim mismatch: got {self.resource_dim}, required {self.required_resource_dim} "
                "(1 rho + (n_gts+1) alpha + n_gts beta)."
            )

        # Legacy compatibility dimensions.
        self.power_dim = self.n_gts + 2
        self.rate_dim = self.n_gts
        self.precoding_dim = int(self.required_precoding_dim)
        if precoding_dim is not None and int(precoding_dim) != self.required_precoding_dim:
            raise ValueError(
                f"precoding_dim mismatch: got {int(precoding_dim)}, "
                f"required {self.required_precoding_dim} (2 * streams * antennas)."
            )

        self.latest_qos_margin = np.zeros(self.n_gts, dtype=np.float32)
        self.latest_qos_gap = np.zeros(self.n_gts, dtype=np.float32)
        self.latest_qos_gap_sum = 0.0
        self.latest_qos_gap_sq_sum = 0.0
        self.latest_qos_gap_mean = 0.0
        self.latest_qos_dual_signal = 0.0
        self.latest_qos_dual_signal_sum = 0.0
        self.latest_qos_gap_norm_mean = 0.0
        self.latest_qos_dual_signal_norm = 0.0
        self.latest_qos_dual_signal_norm_for_lambda = 0.0
        self.latest_qos_dual_signal_norm_for_lambda_sum = 0.0

        self.latest_beta = np.ones(self.n_gts, dtype=np.float32) / max(1, self.n_gts)
        self.latest_alpha = np.ones(self.stream_count, dtype=np.float32) / max(1, self.stream_count)
        self.latest_velocity = 0.0
        self.latest_rho = 0.5

        self.latest_ee = 0.0
        self.latest_pf_ee = 0.0
        self.latest_comm_ee = 0.0
        self.latest_pf_comm_ee = 0.0
        self.latest_log_utility_raw = 0.0
        self.latest_log_utility = 0.0
        self.latest_slot_total_power = 0.0
        self.latest_slot_total_energy = 0.0
        self.latest_slot_tx_energy = 0.0
        self.latest_min_user_rate = 0.0
        self.cum_log_utility = 0.0
        self.cum_sum_rate = 0.0
        self.cum_total_energy = 0.0
        self.cum_tx_energy = 0.0
        self.cum_min_user_rate = 0.0
        self.latest_guidance_reward = 0.0
        self.latest_guidance_mode = "off"
        self.latest_guidance_mode_code = 0.0
        self.latest_violation_centroid = np.zeros(2, dtype=np.float32)
        self.latest_violation_centroid_dist = 0.0
        self.latest_weak_user_idx = -1
        self.latest_weak_user_dist = 0.0
        self.latest_guidance_potential = 0.0
        self.latest_guidance_progress = 0.0
        self._prev_guidance_potential = None
        self._tracked_weak_user = -1
        self.latest_tx_power = 0.0
        self.latest_power_violation_ratio = 0.0
        self.latest_wall_violation = 0.0
        self.latest_wall_penalty = 0.0
        self.latest_core_violation = 0.0
        self.latest_core_penalty = 0.0
        self.latest_core_penalty_factor = 1.0
        self.latest_effective_core_penalty_scale = float(self.reward_core_penalty_scale)
        self.core_violation_steps = 0
        self.latest_user_mean_serving_dist = 0.0
        self.latest_distance_shaping_factor = 0.0
        self.latest_distance_shaping_reward = 0.0

        self.latest_weighted_qos_gap = 0.0
        self.latest_weighted_qos_gap_sq = 0.0
        self.latest_weighted_qos_gap_norm = 0.0
        self.latest_weighted_qos_penalty = 0.0
        self.latest_lagrangian_constraint = 0.0
        self.latest_effective_qos_scale = 0.0
        self.latest_qos_warmup_factor = 0.0
        self.latest_effective_rate = np.zeros(self.n_gts, dtype=np.float32)
        self.latest_effective_fairness = 1.0
        self.latest_common_rate_sum = 0.0
        self.latest_private_rate_sum = 0.0

        self.latest_reward_main_raw = 0.0
        self.latest_reward_main_aligned = 0.0
        self.latest_reward_ref = 1.0
        self.latest_reward_penalty_total = 0.0
        self.latest_reward_before_scale = 0.0
        self.latest_reward_output_scale = float(self.reward_output_scale)
        self.latest_reward_scalar = 0.0

        self.latest_beam_gain = 0.0
        self.latest_common_beam_norm = 0.0
        self.latest_private_beam_norm_mean = 0.0
        self.latest_private_beam_norm_std = 0.0
        self.latest_private_beam_norm_max_dev = 0.0
        self.latest_gamma_common_min = 0.0
        self.latest_gamma_private_mean = 0.0
        self.latest_common_interference = 0.0
        self.latest_private_interference = 0.0

        self.pathloss_alpha = 2.2
        self.pathloss_beta0 = 1.0
        self._channel_cache = np.zeros((self.n_ubs, self.n_gts, self.antenna_count), dtype=np.complex64)

        # Keep normalization bounds consistent with the same pathloss model used in physical rate computation.
        g_max = self.pathloss_beta0 * np.power(max(float(self.h_ubs), 1.0), -self.pathloss_alpha)
        snr_max = max(1e-6, self.tx_power_max_w * g_max / (self.n0 * self.bw))
        self.achievable_rate_gts_max = float(self.bw * np.log2(1.0 + snr_max) * 1e-6)
        self.achievable_rate_ubs_max = float(self.achievable_rate_gts_max * max(1, self.n_gts))

    def set_lambda(self, value: float):
        # Backward-compatibility no-op: PF reward no longer depends on lambda.
        self.lambda_penalty = float(max(0.0, value))

    def set_reward_mode(self, mode: str):
        reward_mode = str(mode).strip().lower()
        if reward_mode not in {"pf_ee", "sum_ee", "max_min"}:
            raise ValueError(
                f"Unsupported reward mode: {reward_mode}. "
                "Allowed: ['pf_ee', 'sum_ee', 'max_min']."
            )
        self.reward_objective = reward_mode

    def set_training_progress(self, episode: int, total_episodes: int):
        self.curriculum_episode = int(max(1, episode))
        self.curriculum_total_episodes = int(max(1, total_episodes))

    def _core_terminate_enabled(self) -> bool:
        if not self.terminate_on_core_violation:
            return False
        return bool(self.curriculum_episode >= self.core_terminate_start_episode)

    def reset(self):
        obs, state, init_info = super().reset()

        self.latest_qos_margin.fill(0.0)
        self.latest_qos_gap.fill(0.0)
        self.latest_qos_gap_sum = 0.0
        self.latest_qos_gap_sq_sum = 0.0
        self.latest_qos_gap_mean = 0.0
        self.latest_qos_dual_signal = 0.0
        self.latest_qos_dual_signal_sum = 0.0
        self.latest_qos_gap_norm_mean = 0.0
        self.latest_qos_dual_signal_norm = 0.0
        self.latest_qos_dual_signal_norm_for_lambda = 0.0
        self.latest_qos_dual_signal_norm_for_lambda_sum = 0.0

        self.latest_beta = np.ones(self.n_gts, dtype=np.float32) / max(1, self.n_gts)
        self.latest_alpha = np.ones(self.stream_count, dtype=np.float32) / max(1, self.stream_count)
        self.latest_velocity = 0.0
        self.latest_rho = 0.5

        self.latest_ee = 0.0
        self.latest_pf_ee = 0.0
        self.latest_comm_ee = 0.0
        self.latest_pf_comm_ee = 0.0
        self.latest_log_utility_raw = 0.0
        self.latest_log_utility = 0.0
        self.latest_slot_total_power = 0.0
        self.latest_slot_total_energy = 0.0
        self.latest_slot_tx_energy = 0.0
        self.latest_min_user_rate = 0.0
        self.cum_log_utility = 0.0
        self.cum_sum_rate = 0.0
        self.cum_total_energy = 0.0
        self.cum_tx_energy = 0.0
        self.cum_min_user_rate = 0.0
        self.latest_guidance_reward = 0.0
        self.latest_guidance_mode = "off"
        self.latest_guidance_mode_code = 0.0
        self.latest_violation_centroid = np.mean(self.pos_gts, axis=0).astype(np.float32)
        self.latest_violation_centroid_dist = float(np.min(np.linalg.norm(self.pos_ubs - self.latest_violation_centroid[None, :], axis=1)))
        self.latest_weak_user_idx = -1
        self.latest_weak_user_dist = 0.0
        self.latest_guidance_potential = 0.0
        self.latest_guidance_progress = 0.0
        self._prev_guidance_potential = None
        self._tracked_weak_user = -1
        self.latest_tx_power = self.latest_rho * self.tx_power_max_w
        self.latest_power_violation_ratio = 0.0
        self.latest_wall_violation = 0.0
        self.latest_wall_penalty = 0.0
        self.latest_core_violation = 0.0
        self.latest_core_penalty = 0.0
        self.latest_core_penalty_factor = 1.0
        self.latest_effective_core_penalty_scale = float(
            self.reward_core_penalty_scale * self.latest_core_penalty_factor
        )
        self.core_violation_steps = 0
        self.latest_weighted_qos_gap = self.latest_qos_gap_sq_sum
        self.latest_weighted_qos_gap_sq = self.latest_qos_gap_sq_sum
        self.latest_weighted_qos_gap_norm = 0.0
        self.latest_weighted_qos_penalty = 0.0
        self.latest_qos_warmup_factor = 0.0
        self.latest_effective_qos_scale = 0.0
        self.latest_user_mean_serving_dist = float(np.mean(np.linalg.norm(self.pos_ubs[:, None, :] - self.pos_gts[None, :, :], axis=2)))
        self.latest_distance_shaping_factor = 0.0
        self.latest_distance_shaping_reward = 0.0
        self.latest_effective_rate.fill(0.0)
        self.latest_effective_fairness = 1.0
        self.latest_common_rate_sum = 0.0
        self.latest_private_rate_sum = 0.0

        self.latest_reward_main_raw = 0.0
        self.latest_reward_main_aligned = 0.0
        self.latest_reward_ref = 1.0
        self.latest_reward_penalty_total = 0.0
        self.latest_reward_before_scale = 0.0
        self.latest_reward_output_scale = float(self.reward_output_scale)
        self.latest_reward_scalar = 0.0

        self.latest_beam_gain = 0.0
        self.latest_common_beam_norm = 0.0
        self.latest_private_beam_norm_mean = 0.0
        self.latest_private_beam_norm_std = 0.0
        self.latest_private_beam_norm_max_dev = 0.0
        self.latest_gamma_common_min = 0.0
        self.latest_gamma_private_mean = 0.0
        self.latest_common_interference = 0.0
        self.latest_private_interference = 0.0

        self._refresh_channel_cache()

        obs = wrapper_obs(self.get_obs())
        state = wrapper_state(self.get_state())
        return obs, state, init_info

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info.update(
            {
                "action_mode": "continuous",
                "traj_dim": 2,
                "common_precoding_dim": self.common_precoding_dim,
                "private_precoding_dim": self.private_precoding_dim,
                "resource_dim": self.resource_dim,
                "precoding_dim": self.precoding_dim,
                "antenna_count": self.antenna_count,
                "csi_complex_dim": self.csi_complex_dim,
                "power_dim": self.power_dim,
                "rate_dim": self.rate_dim,
            }
        )
        return env_info

    def _circular_moves(self):
        center = np.array([self.range_pos / 2.0, self.range_pos / 2.0], dtype=np.float32)
        radius = 0.30 * self.range_pos
        moves = np.zeros((self.n_agents, 2), dtype=np.float32)
        speed = np.zeros(self.n_agents, dtype=np.float32)

        for i in range(self.n_agents):
            phase = 2.0 * np.pi * (self.t / max(1, self.episode_length)) + i * 2.0 * np.pi / max(1, self.n_agents)
            target = center + radius * np.array([np.cos(phase), np.sin(phase)], dtype=np.float32)
            delta = target - self.pos_ubs[i]
            dist = np.linalg.norm(delta)
            if dist > 1e-6:
                step = min(dist, self.velocity_bound)
                moves[i] = delta / dist * step
                speed[i] = step
        return moves, speed

    def _ura_steering_vector(self, azimuth: float, elevation: float, dim: int) -> np.ndarray:
        nx = int(np.floor(np.sqrt(dim)))
        ny = int(np.ceil(dim / max(nx, 1)))
        idx_x = np.arange(nx, dtype=np.float32)
        idx_y = np.arange(ny, dtype=np.float32)

        kx = np.sin(elevation) * np.cos(azimuth)
        ky = np.sin(elevation) * np.sin(azimuth)

        phase_grid = np.pi * (idx_x[:, None] * kx + idx_y[None, :] * ky)
        a = np.exp(1j * phase_grid).reshape(-1)[:dim]
        a = a / np.sqrt(max(1, dim))
        return a.astype(np.complex64)

    def _channel_vector_complex(self, delta_xy: np.ndarray, distance: float, dim: int) -> np.ndarray:
        d3 = np.sqrt(np.square(distance) + np.square(self.h_ubs))
        azimuth = np.arctan2(float(delta_xy[1]), float(delta_xy[0] + 1e-8))
        elevation = np.arctan2(float(self.h_ubs), float(distance + 1e-8))

        beta = self.pathloss_beta0 * np.power(max(d3, 1.0), -self.pathloss_alpha)
        k_lin = max(0.0, float(self.K_richan))

        a_los = self._ura_steering_vector(azimuth=azimuth, elevation=elevation, dim=dim)
        g_nlos = (
            np.random.randn(dim).astype(np.float32) + 1j * np.random.randn(dim).astype(np.float32)
        ) / np.sqrt(2.0)

        if k_lin <= 1e-8:
            h = np.sqrt(beta) * g_nlos
        else:
            h = np.sqrt(beta) * (
                np.sqrt(k_lin / (k_lin + 1.0)) * a_los
                + np.sqrt(1.0 / (k_lin + 1.0)) * g_nlos
            )
        return h.astype(np.complex64)

    def _refresh_channel_cache(self):
        for uav_id in range(self.n_ubs):
            for gt_id in range(self.n_gts):
                delta = self.pos_gts[gt_id] - self.pos_ubs[uav_id]
                distance = float(np.linalg.norm(delta))
                self._channel_cache[uav_id, gt_id] = self._channel_vector_complex(
                    delta_xy=delta,
                    distance=distance,
                    dim=self.antenna_count,
                )

    def _build_obs_csi_vector(self, gt_id: int, uav_id: int) -> np.ndarray:
        h_full = self._channel_cache[uav_id, gt_id]
        keep_dim = min(self.csi_complex_dim, self.antenna_count)
        h_obs = np.zeros((self.csi_complex_dim,), dtype=np.complex64)
        h_obs[:keep_dim] = h_full[:keep_dim]
        csi_vec = np.concatenate([np.real(h_obs), np.imag(h_obs)], axis=0).astype(np.float32)
        return (csi_vec * self.state_h_scale).astype(np.float32)

    def _require_action_branch(self, actions: dict, key: str, expected_dim: int) -> np.ndarray:
        if key not in actions:
            raise KeyError(f"Missing action branch '{key}'. No implicit zero-filling is allowed.")

        branch = np.asarray(actions[key], dtype=np.float32)
        expected_size = self.n_agents * expected_dim
        if branch.size != expected_size:
            raise ValueError(
                f"action branch '{key}' dim mismatch: got total size {branch.size}, "
                f"required {expected_size} ({self.n_agents} x {expected_dim})."
            )

        return branch.reshape(self.n_agents, expected_dim)

    def _resolve_action_branches(self, actions: dict):
        traj_actions = self._require_action_branch(actions=actions, key="traj", expected_dim=2)

        new_keys = ["common_precoding", "private_precoding", "resource"]
        has_new = [k in actions for k in new_keys]
        if any(has_new) and not all(has_new):
            raise KeyError(
                "Action branches must provide all of ['common_precoding', 'private_precoding', 'resource'] "
                "or use legacy ['precoding', 'power', 'rate'] together."
            )

        if all(has_new):
            common_precoding_actions = self._require_action_branch(
                actions=actions,
                key="common_precoding",
                expected_dim=self.common_precoding_dim,
            )
            private_precoding_actions = self._require_action_branch(
                actions=actions,
                key="private_precoding",
                expected_dim=self.private_precoding_dim,
            )
            resource_actions = self._require_action_branch(
                actions=actions,
                key="resource",
                expected_dim=self.resource_dim,
            )
            return traj_actions, common_precoding_actions, private_precoding_actions, resource_actions

        # Legacy fallback: merge power/rate into resource branch and split precoding into common/private.
        precoding_actions = self._require_action_branch(
            actions=actions,
            key="precoding",
            expected_dim=self.precoding_dim,
        )
        power_actions = self._require_action_branch(
            actions=actions,
            key="power",
            expected_dim=self.power_dim,
        )
        rate_actions = self._require_action_branch(
            actions=actions,
            key="rate",
            expected_dim=self.rate_dim,
        )

        common_precoding_actions = precoding_actions[:, : self.common_precoding_dim]
        private_precoding_actions = precoding_actions[:, self.common_precoding_dim :]
        resource_actions = np.concatenate([power_actions, rate_actions], axis=1).astype(np.float32)
        return traj_actions, common_precoding_actions, private_precoding_actions, resource_actions

    def _normalize_beam_directions(self, beam_complex: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(beam_complex, axis=1, keepdims=True)
        beam_vectors = beam_complex / np.clip(norms, 1e-6, np.inf)

        # Guard against zero vectors to keep strict unit-norm feasibility.
        zero_mask = norms[:, 0] <= 1e-6
        if np.any(zero_mask):
            beam_vectors[zero_mask] = 0.0 + 0.0j
            beam_vectors[zero_mask, 0] = 1.0 + 0.0j

        norm_vals = np.linalg.norm(beam_vectors, axis=1)
        self.latest_common_beam_norm = float(norm_vals[0])
        if norm_vals.size > 1:
            private_norms = norm_vals[1:]
            self.latest_private_beam_norm_mean = float(np.mean(private_norms))
            self.latest_private_beam_norm_std = float(np.std(private_norms))
            self.latest_private_beam_norm_max_dev = float(np.max(np.abs(private_norms - 1.0)))
        else:
            self.latest_private_beam_norm_mean = 0.0
            self.latest_private_beam_norm_std = 0.0
            self.latest_private_beam_norm_max_dev = 0.0
        return beam_vectors.astype(np.complex64)

    def _build_noma_shared_beam(self, beam_vectors: np.ndarray, served_users: list[int], uav_id: int) -> np.ndarray:
        if len(served_users) == 0:
            shared = np.zeros((self.antenna_count,), dtype=np.complex64)
            shared[0] = 1.0 + 0.0j
            return shared

        candidate_beams = []
        for gt in served_users:
            idx = int(gt) + 1
            if 0 <= idx < beam_vectors.shape[0]:
                candidate_beams.append(beam_vectors[idx])

        if candidate_beams:
            shared = np.sum(np.asarray(candidate_beams, dtype=np.complex64), axis=0)
        else:
            shared = np.sum([self._channel_cache[uav_id, gt].conj() for gt in served_users], axis=0)

        shared_norm = np.linalg.norm(shared)
        if shared_norm <= 1e-8:
            shared = np.zeros((self.antenna_count,), dtype=np.complex64)
            shared[0] = 1.0 + 0.0j
        else:
            shared = (shared / shared_norm).astype(np.complex64)
        return shared

    def _compute_sdma_rates(self, beam_vectors: np.ndarray, stream_powers: np.ndarray, beta: np.ndarray):
        private_rate = np.zeros(self.n_gts, dtype=np.float32)
        common_decode_rate = np.zeros(self.n_gts, dtype=np.float32)

        beam_quality = []
        gamma_common_list = []
        gamma_private_list = []
        common_interf_list = []
        private_interf_list = []

        noise_power = float(self.bw * self.n0)
        stream_count = self.stream_count

        for gt in range(self.n_gts):
            uav_id = int(self.gt_served_by_uav[gt])
            if uav_id < 0:
                continue

            h_vec = self._channel_cache[uav_id, gt]
            projections = np.array([np.vdot(h_vec, beam_vectors[s]) for s in range(stream_count)], dtype=np.complex64)
            gains = self.precoding_gain_scale * np.maximum(np.square(np.abs(projections)), 1e-12)

            beam_quality.append(float(np.max(gains[1:])))

            common_signal = float(stream_powers[0] * gains[0])
            common_interference = float(np.sum(stream_powers[1:] * gains[1:] * self.interference_scale))
            gamma_c = common_signal / (noise_power + common_interference)
            common_decode_rate[gt] = float(self.bw * np.log2(1.0 + gamma_c) * 1e-6)

            private_signal = float(stream_powers[gt + 1] * gains[gt + 1])
            private_terms = stream_powers[1:] * gains[1:] * self.interference_scale
            private_interference = float(np.sum(private_terms) - private_terms[gt])
            gamma_p = private_signal / (noise_power + private_interference)
            private_rate[gt] = float(self.bw * np.log2(1.0 + gamma_p) * 1e-6)

            gamma_common_list.append(float(gamma_c))
            gamma_private_list.append(float(gamma_p))
            common_interf_list.append(float(common_interference))
            private_interf_list.append(float(private_interference))

        served_mask = self.gt_served_by_uav >= 0
        if np.any(served_mask):
            rc_budget = float(np.min(common_decode_rate[served_mask]))
        else:
            rc_budget = 0.0

        common_share = beta * rc_budget
        effective_rate = private_rate + common_share

        return {
            "private_rate": private_rate.astype(np.float32),
            "common_share": common_share.astype(np.float32),
            "effective_rate": effective_rate.astype(np.float32),
            "beam_quality": float(np.mean(beam_quality)) if beam_quality else 0.0,
            "gamma_common_min": float(np.min(gamma_common_list)) if gamma_common_list else 0.0,
            "gamma_private_mean": float(np.mean(gamma_private_list)) if gamma_private_list else 0.0,
            "common_interference": float(np.mean(common_interf_list)) if common_interf_list else 0.0,
            "private_interference": float(np.mean(private_interf_list)) if private_interf_list else 0.0,
        }

    def _compute_noma_rates(self, beam_vectors: np.ndarray, stream_powers: np.ndarray, beta: np.ndarray):
        private_rate = np.zeros(self.n_gts, dtype=np.float32)
        common_decode_rate = np.zeros(self.n_gts, dtype=np.float32)

        beam_quality = []
        gamma_common_list = []
        gamma_private_list = []
        common_interf_list = []
        private_interf_list = []

        noise_power = float(self.bw * self.n0)
        private_powers = np.asarray(stream_powers[1:], dtype=np.float32)

        for uav_id in range(self.n_ubs):
            served = [int(gt) for gt in self.ubs_serv_gts[uav_id] if 0 <= int(gt) < self.n_gts]
            if not served:
                continue

            shared_beam = self._build_noma_shared_beam(beam_vectors=beam_vectors, served_users=served, uav_id=uav_id)

            served_gains = []
            served_powers = []
            for gt in served:
                h_vec = self._channel_cache[uav_id, gt]
                gain = self.precoding_gain_scale * float(np.maximum(np.abs(np.vdot(h_vec, shared_beam)) ** 2, 1e-12))
                served_gains.append(gain)
                served_powers.append(float(private_powers[gt]))

            served_gains = np.asarray(served_gains, dtype=np.float32)
            served_powers = np.asarray(served_powers, dtype=np.float32)

            # Weak-to-strong SIC order; higher-power layers are decoded first.
            order = np.argsort(served_gains, kind="mergesort")
            served_sorted = [served[i] for i in order]
            gains_sorted = served_gains[order]
            powers_sorted = served_powers[order]

            for rank, gt in enumerate(served_sorted):
                h_gain = float(gains_sorted[rank])
                residual_interference = float(np.sum(powers_sorted[rank + 1 :] * h_gain))
                own_signal = float(powers_sorted[rank] * h_gain)
                own_gamma = own_signal / (noise_power + residual_interference)

                decodable_rates = []
                for decode_rank in range(rank, len(served_sorted)):
                    stronger_gain = float(gains_sorted[decode_rank])
                    decode_interference = float(np.sum(powers_sorted[rank + 1 :] * stronger_gain))
                    decode_signal = float(powers_sorted[rank] * stronger_gain)
                    decode_gamma = decode_signal / (noise_power + decode_interference)
                    decodable_rates.append(float(self.bw * np.log2(1.0 + decode_gamma) * 1e-6))

                layer_rate = float(np.min(decodable_rates)) if decodable_rates else 0.0
                private_rate[gt] = layer_rate

                beam_quality.append(float(h_gain))
                gamma_private_list.append(float(own_gamma))
                private_interf_list.append(float(residual_interference))

                # Record the strongest user's decodability as a proxy for SIC feasibility.
                if rank == len(served_sorted) - 1:
                    common_decode_rate[gt] = layer_rate

            gamma_common_list.append(float(np.min(served_gains)))
            common_interf_list.append(float(np.sum(powers_sorted[1:]) if powers_sorted.size > 1 else 0.0))

        common_share = np.zeros(self.n_gts, dtype=np.float32)
        effective_rate = private_rate

        return {
            "private_rate": private_rate.astype(np.float32),
            "common_share": common_share.astype(np.float32),
            "effective_rate": effective_rate.astype(np.float32),
            "beam_quality": float(np.mean(beam_quality)) if beam_quality else 0.0,
            "gamma_common_min": float(np.min(gamma_common_list)) if gamma_common_list else 0.0,
            "gamma_private_mean": float(np.mean(gamma_private_list)) if gamma_private_list else 0.0,
            "common_interference": float(np.mean(common_interf_list)) if common_interf_list else 0.0,
            "private_interference": float(np.mean(private_interf_list)) if private_interf_list else 0.0,
        }

    def _build_zf_beam_vectors(self) -> np.ndarray:
        beam_dim = self.antenna_count
        private_beams = np.zeros((self.n_gts, beam_dim), dtype=np.complex64)

        for uav_id in range(self.n_ubs):
            served = [int(gt) for gt in self.ubs_serv_gts[uav_id] if 0 <= int(gt) < self.n_gts]
            if not served:
                continue

            h_stack = np.stack([self._channel_cache[uav_id, gt] for gt in served], axis=0).astype(np.complex64)
            k_served = int(h_stack.shape[0])
            reg = (1e-4 * np.eye(k_served)).astype(np.complex64)
            gram = h_stack @ h_stack.conj().T
            zf_mat = h_stack.conj().T @ np.linalg.pinv(gram + reg)

            for col, gt in enumerate(served):
                vec = zf_mat[:, col] if col < zf_mat.shape[1] else h_stack[col].conj()
                vec_norm = np.linalg.norm(vec)
                if vec_norm <= 1e-8:
                    vec = h_stack[col].conj()
                    vec_norm = np.linalg.norm(vec)
                if vec_norm <= 1e-8:
                    vec = np.zeros((beam_dim,), dtype=np.complex64)
                    vec[0] = 1.0 + 0.0j
                else:
                    vec = vec / vec_norm
                private_beams[gt] = vec.astype(np.complex64)

        # Robust fallback for users not covered by the per-UAV ZF solve.
        for gt in range(self.n_gts):
            if np.linalg.norm(private_beams[gt]) > 1e-8:
                continue
            uav_id = int(np.clip(self.gt_served_by_uav[gt], 0, self.n_ubs - 1))
            h_vec = self._channel_cache[uav_id, gt]
            h_norm = np.linalg.norm(h_vec)
            if h_norm <= 1e-8:
                private_beams[gt, 0] = 1.0 + 0.0j
            else:
                private_beams[gt] = (h_vec.conj() / h_norm).astype(np.complex64)

        common_vec = np.zeros((beam_dim,), dtype=np.complex64)
        for gt in range(self.n_gts):
            uav_id = int(np.clip(self.gt_served_by_uav[gt], 0, self.n_ubs - 1))
            common_vec += self._channel_cache[uav_id, gt].conj()

        common_norm = np.linalg.norm(common_vec)
        if common_norm <= 1e-8:
            common_vec = np.zeros((beam_dim,), dtype=np.complex64)
            common_vec[0] = 1.0 + 0.0j
        else:
            common_vec = (common_vec / common_norm).astype(np.complex64)

        beam_complex = np.concatenate([common_vec[None, :], private_beams], axis=0)
        return self._normalize_beam_directions(beam_complex)

    def _build_beams_and_powers(
        self,
        common_precoding_actions: np.ndarray,
        private_precoding_actions: np.ndarray,
        resource_actions: np.ndarray,
        use_hard_mapping: bool,
    ):
        stream_count = self.stream_count
        beam_dim = self.antenna_count

        common_flat = np.mean(common_precoding_actions, axis=0)
        if common_flat.size != self.common_precoding_dim:
            raise ValueError(
                f"common_precoding action dim mismatch: got {common_flat.size}, "
                f"required {self.common_precoding_dim}."
            )

        private_flat = np.mean(private_precoding_actions, axis=0)
        if private_flat.size != self.private_precoding_dim:
            raise ValueError(
                f"private_precoding action dim mismatch: got {private_flat.size}, "
                f"required {self.private_precoding_dim}."
            )

        common_raw = common_flat.reshape(1, beam_dim, 2)
        private_raw = private_flat.reshape(self.n_gts, beam_dim, 2)
        common_complex = common_raw[:, :, 0] + 1j * common_raw[:, :, 1]
        private_complex = private_raw[:, :, 0] + 1j * private_raw[:, :, 1]
        beam_complex = np.concatenate([common_complex, private_complex], axis=0)

        resource_logits = np.mean(resource_actions, axis=0)
        if resource_logits.size != self.resource_dim:
            raise ValueError(
                f"resource action dim mismatch: got {resource_logits.size}, required {self.resource_dim}."
            )

        rho_logit = float(resource_logits[0])
        alpha_logits = resource_logits[1 : 1 + stream_count].copy()
        beta_logits = resource_logits[1 + stream_count :]
        if beta_logits.size != self.n_gts:
            raise ValueError(
                f"resource beta dim mismatch: got {beta_logits.size}, required {self.n_gts}."
            )
        if self.fixed_beta_mode == "uniform":
            beta = np.ones(self.n_gts, dtype=np.float32) / max(1, self.n_gts)
        else:
            beta = _softmax_np(beta_logits)

        if self.fixed_precoding_scheme == "zf":
            beam_vectors = self._build_zf_beam_vectors()
        else:
            beam_vectors = None

        if use_hard_mapping:
            if beam_vectors is None:
                beam_vectors = self._normalize_beam_directions(beam_complex)

            rho = 0.5 * (np.clip(rho_logit, -1.0, 1.0) + 1.0)
            alpha_logits[0] += self.alpha_common_logit_bias
            alpha = _softmax_np(alpha_logits)

            # For SDMA/NOMA baselines, enforce alpha_c=0 to remove common-stream TX power.
            if self.rsma_mode in {"sdma", "noma"} or self.baseline in {"sdma", "noma"}:
                alpha = alpha.copy()
                alpha[0] = 0.0
                private_sum = float(np.sum(alpha[1:]))
                if private_sum <= 1e-8:
                    alpha[1:] = 1.0 / max(1, stream_count - 1)
                else:
                    alpha[1:] = alpha[1:] / private_sum

            stream_powers = alpha * float(rho * self.tx_power_max_w)
            total_tx_power = float(np.sum(stream_powers))
        else:
            # PPO baseline without hard feasibility mapping.
            if beam_vectors is None:
                beam_vectors = beam_complex.astype(np.complex64)

            beam_norm_vals = np.linalg.norm(beam_vectors, axis=1)
            self.latest_common_beam_norm = float(beam_norm_vals[0]) if beam_norm_vals.size > 0 else 0.0
            if beam_norm_vals.size > 1:
                private_norms = beam_norm_vals[1:]
                self.latest_private_beam_norm_mean = float(np.mean(private_norms))
                self.latest_private_beam_norm_std = float(np.std(private_norms))
                self.latest_private_beam_norm_max_dev = float(np.max(np.abs(private_norms - 1.0)))
            else:
                self.latest_private_beam_norm_mean = 0.0
                self.latest_private_beam_norm_std = 0.0
                self.latest_private_beam_norm_max_dev = 0.0

            rho = 1.0 + float(np.clip(rho_logit, -1.0, 1.0))
            stream_raw = np.log1p(np.exp(alpha_logits))
            if float(np.sum(stream_raw)) <= 1e-8:
                alpha = np.ones(stream_count, dtype=np.float32) / max(1, stream_count)
            else:
                alpha = (stream_raw / float(np.sum(stream_raw))).astype(np.float32)

            # For SDMA/NOMA baselines, enforce alpha_c=0 to remove common-stream TX power.
            if self.rsma_mode in {"sdma", "noma"} or self.baseline in {"sdma", "noma"}:
                alpha = alpha.copy()
                alpha[0] = 0.0
                private_sum = float(np.sum(alpha[1:]))
                if private_sum <= 1e-8:
                    alpha[1:] = 1.0 / max(1, stream_count - 1)
                else:
                    alpha[1:] = alpha[1:] / private_sum

            stream_powers = alpha * float(rho * self.tx_power_max_w)
            total_tx_power = float(np.sum(stream_powers))

        return (
            beam_vectors.astype(np.complex64),
            stream_powers.astype(np.float32),
            float(rho),
            alpha.astype(np.float32),
            beta.astype(np.float32),
            total_tx_power,
        )

    def _compute_rsma_rates(self, beam_vectors: np.ndarray, stream_powers: np.ndarray, beta: np.ndarray):
        if self.rsma_mode == "sdma" or self.baseline == "sdma":
            return self._compute_sdma_rates(beam_vectors=beam_vectors, stream_powers=stream_powers, beta=beta)
        if self.rsma_mode == "noma" or self.baseline == "noma":
            return self._compute_noma_rates(beam_vectors=beam_vectors, stream_powers=stream_powers, beta=beta)

        private_rate = np.zeros(self.n_gts, dtype=np.float32)
        common_decode_rate = np.zeros(self.n_gts, dtype=np.float32)

        beam_quality = []
        gamma_common_list = []
        gamma_private_list = []
        common_interf_list = []
        private_interf_list = []

        noise_power = float(self.bw * self.n0)
        stream_count = self.stream_count

        for gt in range(self.n_gts):
            uav_id = int(self.gt_served_by_uav[gt])
            if uav_id < 0:
                continue

            h_vec = self._channel_cache[uav_id, gt]

            projections = np.array([np.vdot(h_vec, beam_vectors[s]) for s in range(stream_count)], dtype=np.complex64)
            gains = self.precoding_gain_scale * np.maximum(np.square(np.abs(projections)), 1e-12)

            beam_quality.append(float(np.max(gains[1:])))

            common_signal = float(stream_powers[0] * gains[0])
            common_interference = float(np.sum(stream_powers[1:] * gains[1:] * self.interference_scale))
            gamma_c = common_signal / (noise_power + common_interference)
            common_decode_rate[gt] = float(self.bw * np.log2(1.0 + gamma_c) * 1e-6)

            private_signal = float(stream_powers[gt + 1] * gains[gt + 1])
            private_terms = stream_powers[1:] * gains[1:] * self.interference_scale
            private_interference = float(np.sum(private_terms) - private_terms[gt])
            gamma_p = private_signal / (noise_power + private_interference)
            private_rate[gt] = float(self.bw * np.log2(1.0 + gamma_p) * 1e-6)

            gamma_common_list.append(float(gamma_c))
            gamma_private_list.append(float(gamma_p))
            common_interf_list.append(float(common_interference))
            private_interf_list.append(float(private_interference))

        served_mask = self.gt_served_by_uav >= 0
        if np.any(served_mask):
            rc_budget = float(np.min(common_decode_rate[served_mask]))
        else:
            rc_budget = 0.0

        common_share = beta * rc_budget
        effective_rate = private_rate + common_share

        return {
            "private_rate": private_rate.astype(np.float32),
            "common_share": common_share.astype(np.float32),
            "effective_rate": effective_rate.astype(np.float32),
            "beam_quality": float(np.mean(beam_quality)) if beam_quality else 0.0,
            "gamma_common_min": float(np.min(gamma_common_list)) if gamma_common_list else 0.0,
            "gamma_private_mean": float(np.mean(gamma_private_list)) if gamma_private_list else 0.0,
            "common_interference": float(np.mean(common_interf_list)) if common_interf_list else 0.0,
            "private_interference": float(np.mean(private_interf_list)) if private_interf_list else 0.0,
        }

    def _physical_rate_mapping(
        self,
        common_precoding_actions: np.ndarray,
        private_precoding_actions: np.ndarray,
        resource_actions: np.ndarray,
    ):
        use_hard_mapping = (self.baseline != "ppo") or self.force_hard_mapping_for_ppo

        beam_vectors, stream_powers, rho, alpha, beta, total_tx_power = self._build_beams_and_powers(
            common_precoding_actions=common_precoding_actions,
            private_precoding_actions=private_precoding_actions,
            resource_actions=resource_actions,
            use_hard_mapping=use_hard_mapping,
        )
        rsma_stats = self._compute_rsma_rates(beam_vectors=beam_vectors, stream_powers=stream_powers, beta=beta)

        power_violation_ratio = max(0.0, total_tx_power - self.tx_power_max_w) / max(self.tx_power_max_w, 1e-6)

        return {
            "rho": float(rho),
            "alpha": alpha,
            "beta": beta,
            "mapped_private": rsma_stats["private_rate"],
            "mapped_total": rsma_stats["effective_rate"],
            "common_share": rsma_stats["common_share"],
            "beam_quality": float(rsma_stats["beam_quality"]),
            "gamma_common_min": float(rsma_stats["gamma_common_min"]),
            "gamma_private_mean": float(rsma_stats["gamma_private_mean"]),
            "common_interference": float(rsma_stats["common_interference"]),
            "private_interference": float(rsma_stats["private_interference"]),
            "total_tx_power": float(total_tx_power),
            "power_violation_ratio": float(power_violation_ratio),
        }

    def step(self, actions):
        self.t += 1

        traj_actions, common_precoding_actions, private_precoding_actions, resource_actions = self._resolve_action_branches(actions)

        if self.freeze_uav_trajectory:
            moves = np.zeros((self.n_agents, 2), dtype=np.float32)
            speed = np.zeros(self.n_agents, dtype=np.float32)
        elif self.baseline == "hover":
            moves = np.zeros((self.n_agents, 2), dtype=np.float32)
            speed = np.zeros(self.n_agents, dtype=np.float32)
        elif self.baseline == "circular":
            moves, speed = self._circular_moves()
        else:
            # Paper mapping: action -> speed and heading, then convert to Cartesian velocity.
            a_v = np.clip(traj_actions[:, 0], -1.0, 1.0)
            a_theta = np.clip(traj_actions[:, 1], -1.0, 1.0)
            speed = 0.5 * self.velocity_bound * (a_v + 1.0)
            heading = np.pi * (a_theta + 1.0)
            moves = np.stack([speed * np.cos(heading), speed * np.sin(heading)], axis=1).astype(np.float32)

        self.latest_velocity = float(np.mean(speed))

        proposed_pos = self.pos_ubs + moves
        clipped_pos = np.clip(proposed_pos, 0.0, self.range_pos)
        wall_overshoot = proposed_pos - clipped_pos
        self.latest_wall_violation = float(np.mean(np.linalg.norm(wall_overshoot, axis=1)))
        self.pos_ubs = clipped_pos
        core_lower = np.maximum(self.core_boundary_min - self.pos_ubs, 0.0)
        core_upper = np.maximum(self.pos_ubs - self.core_boundary_max, 0.0)
        core_overshoot = core_lower + core_upper
        self.latest_core_violation = float(np.mean(np.linalg.norm(core_overshoot, axis=1)))
        if self.latest_core_violation > self.core_violation_terminate_threshold:
            self.core_violation_steps += 1
        else:
            self.core_violation_steps = 0

        self.update_distance()
        self._assign_users()
        self._refresh_channel_cache()

        mapping = self._physical_rate_mapping(
            common_precoding_actions=common_precoding_actions,
            private_precoding_actions=private_precoding_actions,
            resource_actions=resource_actions,
        )

        self.latest_rho = float(mapping["rho"])
        self.latest_alpha = mapping["alpha"].copy()
        self.latest_beta = mapping["beta"].copy()
        self.latest_tx_power = float(mapping["total_tx_power"])
        self.latest_power_violation_ratio = float(mapping["power_violation_ratio"])

        self.latest_common_rate_sum = float(np.sum(mapping["common_share"]))
        self.latest_private_rate_sum = float(np.sum(mapping["mapped_private"]))
        self.latest_gamma_common_min = float(mapping["gamma_common_min"])
        self.latest_gamma_private_mean = float(mapping["gamma_private_mean"])
        self.latest_common_interference = float(mapping["common_interference"])
        self.latest_private_interference = float(mapping["private_interference"])
        self.latest_beam_gain = float(mapping["beam_quality"])

        self.latest_effective_rate = mapping["mapped_total"].copy()
        self.rate_gt_t = self.latest_effective_rate.copy()
        self.total_throughput_t = float(np.sum(self.rate_gt_t))
        self.total_throughput += float(self.total_throughput_t)

        self.rate_gt += self.rate_gt_t
        self.aver_rate_per_gt = (self.aver_rate_per_gt * max(self.t - 1, 1) + self.rate_gt_t) / max(self.t, 1)
        self.fair_idx = float(compute_jain_fairness_index(self.aver_rate_per_gt))
        self.latest_effective_fairness = float(compute_jain_fairness_index(self.rate_gt_t + 1e-6))

        self.rate_per_ubs_t.fill(0.0)
        for u in range(self.n_ubs):
            if len(self.ubs_serv_gts[u]) > 0:
                self.rate_per_ubs_t[u] = float(np.sum(self.rate_gt_t[self.ubs_serv_gts[u]]))
        self.ssr_ubsk_t = self.rate_per_ubs_t.copy()
        self.global_util = float(self.fair_idx * np.mean(self.rate_gt_t))

        reward = self.get_reward(reward_scale_rate=1.0)
        self.ep_ret += reward
        self.mean_returns += float(np.mean(reward))
        self.avg_fair_idx_per_episode += self.latest_effective_fairness

        done = self.get_terminate()

        info = {
            "EpRet": self.ep_ret,
            "EpLen": int(self.t),
            "mean_returns": float(self.mean_returns),
            "total_throughput": float(self.total_throughput),
            "global_util": float(self.global_util),
            "avg_fair_idx_per_episode": float(self.avg_fair_idx_per_episode),
            "pf_energy_efficiency": float(self.latest_pf_ee),
            "energy_efficiency": float(self.latest_ee),
            "slot_pf_total_ee": float(self.latest_pf_ee),
            "slot_total_ee": float(self.latest_ee),
            "slot_pf_comm_ee": float(self.latest_pf_comm_ee),
            "slot_comm_ee": float(self.latest_comm_ee),
            "log_utility_raw": float(self.latest_log_utility_raw),
            "log_utility": float(self.latest_log_utility),
            "log_utility_scale": float(self.pf_log_scale),
            "pf_rate_ref_mbps": float(self.pf_rate_ref_mbps),
            "jain_fairness": float(self.latest_effective_fairness),
            "min_user_rate": float(self.latest_min_user_rate),
            "slot_total_power": float(self.latest_slot_total_power),
            "slot_total_energy": float(self.latest_slot_total_energy),
            "slot_tx_energy": float(self.latest_slot_tx_energy),
            "episode_pf_energy_efficiency": float(self.cum_log_utility / max(self.cum_total_energy, 1e-6)),
            "episode_energy_efficiency": float(self.cum_sum_rate / max(self.cum_total_energy, 1e-6)),
            "episode_pf_comm_ee": float(self.cum_log_utility / max(self.cum_tx_energy, 1e-6)),
            "episode_comm_ee": float(self.cum_sum_rate / max(self.cum_tx_energy, 1e-6)),
            "episode_min_user_rate_avg": float(self.cum_min_user_rate / max(1, self.t)),
            "episode_total_energy": float(self.cum_total_energy),
            "episode_tx_energy": float(self.cum_tx_energy),
            "guidance_reward": float(self.latest_guidance_reward),
            "guidance_mode": str(self.latest_guidance_mode),
            "guidance_mode_code": float(self.latest_guidance_mode_code),
            "guidance_potential": float(self.latest_guidance_potential),
            "guidance_progress": float(self.latest_guidance_progress),
            "weak_user_idx": int(self.latest_weak_user_idx),
            "weak_user_dist": float(self.latest_weak_user_dist),
            "tx_power": float(self.latest_tx_power),
            "tx_power_violation": float(self.latest_power_violation_ratio),
            "wall_violation": float(self.latest_wall_violation),
            "wall_violation_norm": float(self.latest_wall_violation / self.wall_penalty_normalizer),
            "wall_penalty": float(self.latest_wall_penalty),
            "core_violation": float(self.latest_core_violation),
            "core_violation_norm": float(self.latest_core_violation / self.core_penalty_normalizer),
            "core_penalty": float(self.latest_core_penalty),
            "core_penalty_factor": float(self.latest_core_penalty_factor),
            "effective_core_penalty_scale": float(self.latest_effective_core_penalty_scale),
            "lambda_penalty": float(self.lambda_penalty),
            "core_violation_steps": int(self.core_violation_steps),
            "core_terminate_enabled": bool(self._core_terminate_enabled()),
            "core_terminate_start_episode": int(self.core_terminate_start_episode),
            "freeze_uav_trajectory": bool(self.freeze_uav_trajectory),
            "fixed_precoding_scheme": str(self.fixed_precoding_scheme),
            "fixed_beta_mode": str(self.fixed_beta_mode),
            "velocity": float(self.latest_velocity),
            "user_mean_serving_dist": float(self.latest_user_mean_serving_dist),
            "distance_shaping_factor": float(self.latest_distance_shaping_factor),
            "distance_shaping_reward": float(self.latest_distance_shaping_reward),
            "rsma_common_rate": float(self.latest_common_rate_sum),
            "rsma_private_rate": float(self.latest_private_rate_sum),
            "effective_fairness": float(self.latest_effective_fairness),
            "reward_objective": str(self.reward_objective),
            "reward_main_raw": float(self.latest_reward_main_raw),
            "reward_main_aligned": float(self.latest_reward_main_aligned),
            "reward_ref": float(self.latest_reward_ref),
            "reward_penalty_total": float(self.latest_reward_penalty_total),
            "reward_before_scale": float(self.latest_reward_before_scale),
            "reward_output_scale": float(self.latest_reward_output_scale),
            "reward_scalar": float(self.latest_reward_scalar),
            "violation_centroid_x": float(self.latest_violation_centroid[0]),
            "violation_centroid_y": float(self.latest_violation_centroid[1]),
            "violation_centroid_dist": float(self.latest_violation_centroid_dist),
            "beam_quality": float(self.latest_beam_gain),
            "common_beam_norm": float(self.latest_common_beam_norm),
            "private_beam_norm_mean": float(self.latest_private_beam_norm_mean),
            "private_beam_norm_std": float(self.latest_private_beam_norm_std),
            "private_beam_norm_max_dev": float(self.latest_private_beam_norm_max_dev),
            "alpha_common": float(self.latest_alpha[0]),
            "gamma_common_min": float(self.latest_gamma_common_min),
            "gamma_private_mean": float(self.latest_gamma_private_mean),
            "common_interference": float(self.latest_common_interference),
            "private_interference": float(self.latest_private_interference),
            "BadMask": bool(self.t == self.episode_length),
        }

        obs = wrapper_obs(self.get_obs())
        state = wrapper_state(self.get_state())

        self.uav_traj.append(self.pos_ubs.copy())
        self.throughput_list.append(float(self.total_throughput_t))

        return obs, state, reward, done, info

    def get_terminate(self):
        if self.t == self.episode_length:
            return True
        if self._core_terminate_enabled() and self.core_violation_steps >= self.core_violation_terminate_patience:
            return True
        return False

    def get_obs_agent(self, agent_id: int):
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        ubs_feats = np.zeros(self.obs_ubs_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.obs_gt_feats_size, dtype=np.float32)

        own_feats[0:2] = self.pos_ubs[agent_id] / self.range_pos
        own_feats[2] = self.rate_per_ubs_t[agent_id] / max(self.achievable_rate_ubs_max, 1e-6)
        own_feats[3] = self.latest_rho
        if self.latest_weak_user_idx >= 0:
            target_pos = self.pos_gts[self.latest_weak_user_idx]
        else:
            target_pos = self.latest_violation_centroid
        own_feats[4:6] = (target_pos - self.pos_ubs[agent_id]) / self.range_pos

        other_ubs = [u for u in range(self.n_agents) if u != agent_id]
        for j, u in enumerate(other_ubs):
            if self.d_u2u[agent_id, u] <= self.r_sense:
                ubs_feats[j, 0] = 1.0
                ubs_feats[j, 1:3] = (self.pos_ubs[u] - self.pos_ubs[agent_id]) / self.range_pos

        for gt in range(self.n_gts):
            gt_feats[gt, 0] = 1.0
            gt_feats[gt, 1:3] = (self.pos_gts[gt] - self.pos_ubs[agent_id]) / self.range_pos

            csi_vec = self._build_obs_csi_vector(gt_id=gt, uav_id=agent_id)
            csi_start = 3
            csi_end = csi_start + 2 * self.csi_complex_dim
            gt_feats[gt, csi_start:csi_end] = csi_vec

            rate_idx = csi_end
            gt_feats[gt, rate_idx] = self.latest_effective_rate[gt] / max(self.achievable_rate_gts_max, 1e-6)

        return {"agent": own_feats, "ubs": ubs_feats, "gt": gt_feats}

    @property
    def obs_own_feats_size(self):
        # pos(2) + normalized served throughput(1) + rho(1) + weak-user relative vector (fallback to centroid)(2)
        return 6

    @property
    def obs_gt_feats_size(self):
        # vis(1) + rel_pos(2) + csi_real_imag(2*csi_dim) + norm_rate(1)
        return self.n_gts, 4 + 2 * self.csi_complex_dim

    def get_state(self):
        ubs_feats = np.zeros(self.state_ubs_feats_size(), dtype=np.float32)
        gt_feats = np.zeros(self.state_gt_feats_size(), dtype=np.float32)

        ubs_feats[:, 0:2] = self.pos_ubs / self.range_pos
        ubs_feats[:, 2] = self.rate_per_ubs_t / max(self.achievable_rate_ubs_max, 1e-6)
        ubs_feats[:, 3] = self.latest_rho
        if self.latest_weak_user_idx >= 0:
            target_pos = self.pos_gts[self.latest_weak_user_idx]
        else:
            target_pos = self.latest_violation_centroid
        ubs_feats[:, 4:6] = (target_pos[None, :] - self.pos_ubs) / self.range_pos

        gt_feats[:, 0:2] = self.pos_gts / self.range_pos
        gt_feats[:, 2] = self.latest_effective_rate / max(self.achievable_rate_gts_max, 1e-6)

        return np.concatenate((ubs_feats.flatten(), gt_feats.flatten()))

    def state_ubs_feats_size(self):
        # pos(2) + norm_uav_rate(1) + rho(1) + weak-user relative vector (fallback to centroid)(2)
        return self.n_ubs, 6

    def state_gt_feats_size(self):
        # pos(2) + norm_rate(1)
        return self.n_gts, 3

    def get_reward(self, reward_scale_rate=1.0):
        del reward_scale_rate

        effective_rate = np.maximum(self.latest_effective_rate, 0.0)
        throughput_mbps = float(np.sum(effective_rate))
        self.latest_min_user_rate = float(np.min(effective_rate)) if effective_rate.size > 0 else 0.0

        fly_power = self.fly_power_base + self.fly_power_coeff * float(np.square(self.latest_velocity))
        total_power = max(fly_power + self.latest_tx_power, 1e-6)
        tx_power = max(self.latest_tx_power, 1e-6)

        self.latest_slot_total_power = float(total_power)
        self.latest_slot_total_energy = float(total_power)
        self.latest_slot_tx_energy = float(self.latest_tx_power)
        self.latest_ee = float(throughput_mbps / total_power)
        # Use dimensionless log utility: log(1 + R / R_ref), then apply scale for critic signal strength.
        self.latest_log_utility_raw = float(np.sum(np.log1p(effective_rate / self.pf_rate_ref_mbps)))
        self.latest_log_utility = float(self.pf_log_scale * self.latest_log_utility_raw)
        self.latest_pf_ee = float(self.latest_log_utility / total_power)
        self.latest_comm_ee = float(throughput_mbps / tx_power)
        self.latest_pf_comm_ee = float(self.latest_log_utility / tx_power)

        self.cum_log_utility += self.latest_log_utility
        self.cum_sum_rate += throughput_mbps
        self.cum_total_energy += self.latest_slot_total_energy
        self.cum_tx_energy += float(self.latest_tx_power)
        self.cum_min_user_rate += self.latest_min_user_rate

        user_to_uav_dist = np.linalg.norm(self.pos_ubs[:, None, :] - self.pos_gts[None, :, :], axis=2)
        nearest_user_dist = np.min(user_to_uav_dist, axis=0)
        self.latest_user_mean_serving_dist = float(np.mean(nearest_user_dist))

        self.latest_qos_margin.fill(0.0)
        self.latest_qos_gap.fill(0.0)
        self.latest_qos_gap_sum = 0.0
        self.latest_qos_gap_sq_sum = 0.0
        self.latest_qos_gap_mean = 0.0
        self.latest_qos_dual_signal = 0.0
        self.latest_qos_dual_signal_sum = 0.0
        self.latest_qos_gap_norm_mean = 0.0
        self.latest_qos_dual_signal_norm = 0.0
        self.latest_qos_dual_signal_norm_for_lambda = 0.0
        self.latest_qos_dual_signal_norm_for_lambda_sum = 0.0
        self.latest_weighted_qos_gap = 0.0
        self.latest_weighted_qos_gap_sq = 0.0
        self.latest_weighted_qos_gap_norm = 0.0
        self.latest_weighted_qos_penalty = 0.0
        self.latest_qos_warmup_factor = 0.0
        self.latest_effective_qos_scale = 0.0
        self.latest_lagrangian_constraint = 0.0

        self.latest_guidance_reward = 0.0
        self.latest_guidance_mode = "off"
        self.latest_guidance_mode_code = 0.0
        self.latest_guidance_potential = 0.0
        self.latest_guidance_progress = 0.0
        self.latest_weak_user_idx = int(np.argmin(effective_rate)) if effective_rate.size > 0 else -1
        self.latest_weak_user_dist = 0.0
        self.latest_distance_shaping_factor = 0.0
        self.latest_distance_shaping_reward = 0.0

        wall_violation_norm = float(self.latest_wall_violation / self.wall_penalty_normalizer)
        core_violation_norm = float(self.latest_core_violation / self.core_penalty_normalizer)
        self.latest_wall_penalty = float(self.reward_wall_penalty_scale * wall_violation_norm)
        self.latest_core_penalty_factor = 1.0
        self.latest_effective_core_penalty_scale = float(self.reward_core_penalty_scale)
        self.latest_core_penalty = float(self.latest_effective_core_penalty_scale * core_violation_norm)

        if self.reward_objective == "pf_ee":
            objective_raw = float(self.latest_pf_ee)
            objective_ref = float(self.reward_objective_ref_pf_ee)
        elif self.reward_objective == "sum_ee":
            objective_raw = float(self.latest_ee)
            objective_ref = float(self.reward_objective_ref_sum_ee)
        else:
            objective_raw = float(self.latest_min_user_rate)
            objective_ref = float(self.reward_objective_ref_max_min)

        self.latest_reward_main_raw = float(self.reward_pf_scale * objective_raw)
        self.latest_reward_ref = float(max(1e-6, objective_ref))
        self.latest_reward_main_aligned = float(self.latest_reward_main_raw / self.latest_reward_ref)

        power_term = self.reward_power_penalty_scale * self.latest_power_violation_ratio
        self.latest_reward_penalty_total = float(
            power_term
            + self.latest_wall_penalty
            + self.latest_core_penalty
        )
        self.latest_reward_before_scale = float(self.latest_reward_main_aligned - self.latest_reward_penalty_total)
        self.latest_reward_output_scale = float(self.reward_output_scale)
        reward_scalar = float(self.reward_output_scale * self.latest_reward_before_scale)
        self.latest_reward_scalar = float(reward_scalar)

        rewards = np.full(self.n_agents, reward_scalar, dtype=np.float32)
        return rewards
