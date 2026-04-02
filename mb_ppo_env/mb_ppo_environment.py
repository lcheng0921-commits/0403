import numpy as np

from mb_ppo_env.environment import UbsRsmaEvn
from mb_ppo_env.utils import compute_jain_fairness_index, wrapper_obs, wrapper_state


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    denom = np.sum(ex)
    if denom <= 1e-8:
        return np.ones_like(x, dtype=np.float32) / max(len(x), 1)
    return ex / denom


class MbPpoEnv(UbsRsmaEvn):
    """Continuous-action wrapper used by MB-PPO second-round implementation."""

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
        n_eve=4,
        r_cov=220.0,
        r_sense=np.inf,
        n_community=4,
        K_richan=10,
        jamming_power_bound=15,
        velocity_bound=20,
        qos_threshold=0.5,
        tx_power_max_dbm=30.0,
        reward_ee_scale=1.0,
        reward_qos_scale=1.0,
        lambda_penalty=1.0,
        rsma_mode='rsma',
        baseline='mbppo',
        precoding_dim=32,
        antenna_count=16,
        csi_complex_dim=4,
        fly_power_base=80.0,
        fly_power_coeff=0.12,
        phy_mapping_blend=1.0,
        precoding_gain_scale=1.0,
        interference_scale=1.0,
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
            r_cov=r_cov,
            r_sense=r_sense,
            n_community=n_community,
            K_richan=K_richan,
            jamming_power_bound=jamming_power_bound,
            velocity_bound=velocity_bound,
        )

        self.qos_threshold = qos_threshold
        self.reward_ee_scale = reward_ee_scale
        self.reward_qos_scale = reward_qos_scale
        self.lambda_penalty = float(lambda_penalty)
        self.rsma_mode = rsma_mode
        self.baseline = baseline

        self.precoding_dim = int(precoding_dim)
        self.antenna_count = int(max(1, antenna_count))
        self.csi_complex_dim = int(max(1, csi_complex_dim))
        self.tx_power_max_w = 1e-3 * np.power(10.0, tx_power_max_dbm / 10.0)
        self.phy_mapping_blend = float(np.clip(phy_mapping_blend, 0.0, 1.0))
        self.precoding_gain_scale = float(max(1e-6, precoding_gain_scale))
        self.interference_scale = float(max(1e-6, interference_scale))

        # Simplified propulsion model for MVP.
        self.fly_power_base = float(fly_power_base)
        self.fly_power_coeff = float(fly_power_coeff)

        self.latest_qos_margin = np.zeros(self.n_gts, dtype=np.float32)
        self.latest_qos_gap = np.zeros(self.n_gts, dtype=np.float32)
        self.latest_qos_gap_sum = 0.0
        self.latest_qos_gap_mean = 0.0
        self.latest_qos_dual_signal = 0.0
        self.latest_qos_dual_signal_sum = 0.0
        self.latest_qos_gap_norm_mean = 0.0
        self.latest_qos_dual_signal_norm = 0.0
        self.latest_beta = np.ones(self.n_gts, dtype=np.float32) / max(self.n_gts, 1)
        self.latest_velocity = 0.0
        self.latest_rho = 0.5
        self.latest_ee = 0.0
        self.latest_tx_power = 0.0
        self.latest_weighted_qos_gap = 0.0
        self.latest_weighted_qos_gap_norm = 0.0
        self.latest_alpha = np.ones(self.n_gts + 1, dtype=np.float32) / max(self.n_gts + 1, 1)
        self.latest_effective_rate = np.zeros(self.n_gts, dtype=np.float32)
        self.latest_effective_fairness = 1.0
        self.latest_common_rate_sum = 0.0
        self.latest_private_rate_sum = 0.0
        self.latest_beam_gain = 0.0
        self.latest_base_throughput = 0.0
        self.latest_gamma_common_min = 0.0
        self.latest_gamma_private_mean = 0.0
        self.latest_common_interference = 0.0
        self.latest_private_interference = 0.0

    def set_lambda(self, value: float):
        self.lambda_penalty = float(max(0.0, value))

    def reset(self):
        obs, state, init_info = super().reset()
        qos_norm = max(self.qos_threshold, 1e-6)
        qos_sum_norm = max(self.qos_threshold * max(self.n_gts, 1), 1e-6)
        self.latest_qos_margin = (self.qos_threshold - self.rate_gt_t).astype(np.float32)
        self.latest_qos_gap = np.maximum(0.0, self.latest_qos_margin).astype(np.float32)
        self.latest_qos_gap_sum = float(np.sum(self.latest_qos_gap))
        self.latest_qos_gap_mean = float(np.mean(self.latest_qos_gap))
        self.latest_qos_dual_signal = float(np.mean(self.latest_qos_margin))
        self.latest_qos_dual_signal_sum = float(np.sum(self.latest_qos_margin))
        self.latest_qos_gap_norm_mean = self.latest_qos_gap_mean / qos_norm
        self.latest_qos_dual_signal_norm = self.latest_qos_dual_signal / qos_norm
        self.latest_beta = np.ones(self.n_gts, dtype=np.float32) / max(self.n_gts, 1)
        self.latest_velocity = 0.0
        self.latest_rho = 0.5
        self.latest_ee = 0.0
        self.latest_tx_power = self.latest_rho * self.tx_power_max_w
        # Paper-aligned reward penalty term uses the user-sum violation.
        self.latest_weighted_qos_gap = self.latest_qos_gap_sum
        self.latest_weighted_qos_gap_norm = self.latest_qos_gap_sum / qos_sum_norm
        self.latest_alpha = np.ones(self.n_gts + 1, dtype=np.float32) / max(self.n_gts + 1, 1)
        self.latest_effective_rate = self.rate_gt_t.astype(np.float32).copy()
        self.latest_effective_fairness = float(compute_jain_fairness_index(self.latest_effective_rate))
        self.latest_common_rate_sum = float(np.sum(self.cominfo_rate_gts_t))
        self.latest_private_rate_sum = float(np.sum(self.privateinfo_rate_gts_t))
        self.latest_beam_gain = 0.0
        self.latest_base_throughput = float(np.sum(self.rate_gt_t))
        self.latest_gamma_common_min = 0.0
        self.latest_gamma_private_mean = 0.0
        self.latest_common_interference = 0.0
        self.latest_private_interference = 0.0

        # Refresh wrapped tensors since QoS-gap feature is updated right after reset.
        obs = wrapper_obs(self.get_obs())
        state = wrapper_state(self.get_state())
        return obs, state, init_info

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info.update(
            dict(
                action_mode='continuous',
                traj_dim=2,
                precoding_dim=self.precoding_dim,
                antenna_count=self.antenna_count,
                csi_complex_dim=self.csi_complex_dim,
                power_dim=self.n_gts + 2,
                rate_dim=self.n_gts,
            )
        )
        return env_info

    def _build_obs_csi_vector(self, gt_id: int, uav_id: int) -> np.ndarray:
        delta = self.pos_gts[gt_id] - self.pos_ubs[uav_id]
        distance = np.linalg.norm(delta)
        gain_obs = self.ATGChannel.estimate_chan_gain(distance, self.h_ubs, self.K_richan)
        h_obs = self._channel_vector_complex(
            delta_xy=delta,
            distance=distance,
            dim=self.csi_complex_dim,
            power_gain=float(np.power(abs(gain_obs), 2) + 1e-9),
        )
        return np.concatenate([np.real(h_obs), np.imag(h_obs)], axis=0).astype(np.float32)

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

    def _channel_feature(self, delta_xy: np.ndarray, distance: float, dim: int) -> np.ndarray:
        angle = np.arctan2(float(delta_xy[1]), float(delta_xy[0] + 1e-8))
        inv_dist = 1.0 / (1.0 + distance / max(self.range_pos, 1e-6))
        base = np.array(
            [
                np.cos(angle),
                np.sin(angle),
                inv_dist,
                np.clip(distance / max(self.range_pos, 1e-6), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        if dim <= 4:
            return base[:dim]

        feats = [base]
        harmonic = 2
        while sum(arr.size for arr in feats) < dim:
            feats.append(
                np.array(
                    [
                        np.cos(harmonic * angle),
                        np.sin(harmonic * angle),
                        np.power(inv_dist, min(harmonic, 4)),
                        np.power(np.clip(distance / max(self.range_pos, 1e-6), 0.0, 1.0), min(harmonic, 4)),
                    ],
                    dtype=np.float32,
                )
            )
            harmonic += 1

        return np.concatenate(feats, axis=0)[:dim]

    def _channel_vector_complex(self, delta_xy: np.ndarray, distance: float, dim: int, power_gain: float) -> np.ndarray:
        dim = max(1, int(dim))
        idx = np.arange(dim, dtype=np.float32)

        azimuth = np.arctan2(float(delta_xy[1]), float(delta_xy[0] + 1e-8))
        elevation = np.arctan2(float(self.h_ubs), float(distance + 1e-8))

        # Deterministic steering-like phase profile for LoS component.
        phase_los = 2.0 * np.pi * idx * (0.35 * np.cos(azimuth) * np.cos(elevation) + 0.15 * np.sin(azimuth))
        los = np.exp(1j * phase_los)

        # Deterministic pseudo-scatter profile for NLoS component.
        dist_norm = np.clip(distance / max(self.range_pos, 1e-6), 0.0, 1.0)
        phase_scatter = 2.0 * np.pi * (
            idx * (0.27 * np.sin(azimuth) + 0.11 * np.cos(elevation))
            + 0.19 * dist_norm
        )
        scatter = np.exp(1j * phase_scatter)

        k_lin = float(max(self.K_richan, 0.0))
        if k_lin <= 1e-8:
            h = scatter
        else:
            h = np.sqrt(k_lin / (k_lin + 1.0)) * los + np.sqrt(1.0 / (k_lin + 1.0)) * scatter

        amp = np.sqrt(max(power_gain, 1e-12))
        h = (amp / np.sqrt(dim)) * h
        return h.astype(np.complex64)

    def _physical_rate_mapping(self, precoding_actions: np.ndarray, power_actions: np.ndarray, rate_actions: np.ndarray):
        stream_count = self.n_gts + 1
        beam_dim = self.antenna_count
        required_dim = stream_count * beam_dim * 2

        precoding_flat = np.mean(precoding_actions, axis=0)
        if precoding_flat.size < required_dim:
            precoding_flat = np.pad(precoding_flat, (0, required_dim - precoding_flat.size))
        beam_raw = precoding_flat[:required_dim].reshape(stream_count, beam_dim, 2)
        beam_complex = beam_raw[:, :, 0] + 1j * beam_raw[:, :, 1]
        beam_norm = np.linalg.norm(beam_complex, axis=1, keepdims=True)
        beam_dirs = beam_complex / np.clip(beam_norm, 1e-6, np.inf)

        power_logits = np.mean(power_actions, axis=0)
        rho = 0.5 * (np.clip(power_logits[0], -1.0, 1.0) + 1.0)
        alpha_logits = power_logits[1 : self.n_gts + 2]
        if alpha_logits.size < stream_count:
            alpha_logits = np.pad(alpha_logits, (0, stream_count - alpha_logits.size))
        alpha = _softmax_np(alpha_logits)

        beta_logits = np.mean(rate_actions, axis=0)
        beta_logits = beta_logits[: self.n_gts]
        if beta_logits.size < self.n_gts:
            beta_logits = np.pad(beta_logits, (0, self.n_gts - beta_logits.size))
        beta = _softmax_np(beta_logits)

        total_tx_power = float(rho * self.tx_power_max_w)
        p_common = float(total_tx_power * alpha[0])
        p_private = np.asarray(total_tx_power * alpha[1:], dtype=np.float32)
        p_streams = np.concatenate(([p_common], p_private), axis=0)

        private_rate = np.zeros(self.n_gts, dtype=np.float32)
        common_decode_rate = np.zeros(self.n_gts, dtype=np.float32)
        beam_quality = []
        gamma_common = []
        gamma_private = []
        common_interference_list = []
        private_interference_list = []

        noise_power = float(self.bw * self.n0)

        for gt_id in range(self.n_gts):
            uav_id = int(self.gt_served_by_uav[gt_id])
            if uav_id < 0:
                continue

            delta = self.pos_gts[gt_id] - self.pos_ubs[uav_id]
            distance = np.linalg.norm(delta)
            h_vec = self._channel_vector_complex(
                delta_xy=delta,
                distance=distance,
                dim=beam_dim,
                power_gain=float(self.g_recv_gts_t[gt_id] + 1e-9),
            )

            projected = np.array([np.vdot(h_vec, w) for w in beam_dirs], dtype=np.complex64)
            stream_gain = self.precoding_gain_scale * np.square(np.abs(projected))
            stream_gain = np.maximum(stream_gain, 1e-9)

            beam_quality.append(float(np.max(stream_gain[1:])) if stream_gain.size > 1 else float(stream_gain[0]))

            common_signal = float(p_streams[0] * stream_gain[0])
            inter_common = float(np.sum(p_streams[1:] * stream_gain[1:] * self.interference_scale))
            common_sinr = common_signal / (noise_power + inter_common)
            common_decode_rate[gt_id] = self.bw * np.log2(1.0 + common_sinr) * 1e-6
            common_interference_list.append(inter_common)
            gamma_common.append(common_sinr)

            private_signal = float(p_private[gt_id] * stream_gain[gt_id + 1])

            private_terms = p_private * stream_gain[1:] * self.interference_scale
            private_interf = float(np.sum(private_terms) - private_terms[gt_id])
            private_sinr = private_signal / (noise_power + private_interf)
            private_rate[gt_id] = self.bw * np.log2(1.0 + private_sinr) * 1e-6
            private_interference_list.append(private_interf)
            gamma_private.append(private_sinr)

        served_mask = self.gt_served_by_uav >= 0
        if np.any(served_mask):
            common_budget = float(np.min(common_decode_rate[served_mask]))
        else:
            common_budget = 0.0

        common_share = beta * common_budget

        mapped_private = (1.0 - self.phy_mapping_blend) * self.privateinfo_rate_gts_t + self.phy_mapping_blend * private_rate
        mapped_total = (1.0 - self.phy_mapping_blend) * self.rate_gt_t + self.phy_mapping_blend * (common_share + private_rate)

        return {
            'rho': float(rho),
            'alpha': alpha.astype(np.float32),
            'beta': beta.astype(np.float32),
            'mapped_private': mapped_private.astype(np.float32),
            'mapped_total': mapped_total.astype(np.float32),
            'common_share': common_share.astype(np.float32),
            'beam_quality': float(np.mean(beam_quality)) if beam_quality else 0.0,
            'gamma_common_min': float(np.min(gamma_common)) if gamma_common else 0.0,
            'gamma_private_mean': float(np.mean(gamma_private)) if gamma_private else 0.0,
            'common_interference': float(np.mean(common_interference_list)) if common_interference_list else 0.0,
            'private_interference': float(np.mean(private_interference_list)) if private_interference_list else 0.0,
        }

    def step(self, actions):
        self.t = self.t + 1

        traj_actions = np.asarray(actions.get('traj', np.zeros((self.n_agents, 2))), dtype=np.float32).reshape(
            self.n_agents, 2
        )
        precoding_actions = np.asarray(
            actions.get('precoding', np.zeros((self.n_agents, self.precoding_dim))), dtype=np.float32
        ).reshape(self.n_agents, self.precoding_dim)
        power_actions = np.asarray(
            actions.get('power', np.zeros((self.n_agents, self.n_gts + 2))), dtype=np.float32
        ).reshape(self.n_agents, self.n_gts + 2)
        rate_actions = np.asarray(actions.get('rate', np.zeros((self.n_agents, self.n_gts))), dtype=np.float32).reshape(
            self.n_agents, self.n_gts
        )

        if self.baseline == 'hover':
            moves = np.zeros((self.n_agents, 2), dtype=np.float32)
            speed = np.zeros(self.n_agents, dtype=np.float32)
        elif self.baseline == 'circular':
            moves, speed = self._circular_moves()
        else:
            a_v = np.clip(traj_actions[:, 0], -1.0, 1.0)
            a_theta = np.clip(traj_actions[:, 1], -1.0, 1.0)
            speed = 0.5 * self.velocity_bound * (a_v + 1.0)
            theta = np.pi * (a_theta + 1.0)
            moves = np.stack([speed * np.cos(theta), speed * np.sin(theta)], axis=1)

        self.latest_velocity = float(np.mean(speed))
        self.pos_ubs = np.clip(self.pos_ubs + moves, a_min=0.0, a_max=self.range_pos)

        # Paper system model does not couple a jamming action with transmit power.
        jamming_powers = np.zeros(self.n_agents, dtype=np.float32)

        self.update_distance()
        self.transmit_data(jamming_power=jamming_powers)
        self.sercurity_model()
        self.latest_base_throughput = float(np.sum(self.rate_gt_t))

        mapping = self._physical_rate_mapping(
            precoding_actions=precoding_actions,
            power_actions=power_actions,
            rate_actions=rate_actions,
        )
        self.latest_rho = float(mapping['rho'])
        self.latest_alpha = mapping['alpha']
        self.latest_beta = mapping['beta']
        self.latest_tx_power = float(self.latest_rho * self.tx_power_max_w)
        self.latest_beam_gain = float(mapping['beam_quality'])

        self.latest_common_rate_sum = float(np.sum(mapping['common_share']))
        self.latest_private_rate_sum = float(np.sum(mapping['mapped_private']))
        self.latest_gamma_common_min = float(mapping['gamma_common_min'])
        self.latest_gamma_private_mean = float(mapping['gamma_private_mean'])
        self.latest_common_interference = float(mapping['common_interference'])
        self.latest_private_interference = float(mapping['private_interference'])

        if self.rsma_mode == 'sdma' or self.baseline == 'sdma':
            self.latest_effective_rate = mapping['mapped_private']
        elif self.rsma_mode == 'noma' or self.baseline == 'noma':
            # Minimal NOMA baseline: no common-rate stream allocation, private-rate only.
            self.latest_effective_rate = mapping['mapped_private']
        else:
            self.latest_effective_rate = mapping['mapped_total']
        self.latest_effective_fairness = float(compute_jain_fairness_index(self.latest_effective_rate))

        reward = self.get_reward(self.reward_scale)
        self.ep_ret = self.ep_ret + reward
        self.mean_returns = self.mean_returns + reward.mean()
        self.avg_fair_idx_per_episode = self.avg_fair_idx_per_episode + self.latest_effective_fairness

        done = self.get_terminate()

        info = dict(
            EpRet=self.ep_ret,
            EpLen=self.t,
            mean_returns=self.mean_returns,
            total_throughput=self.total_throughput,
            Ssr_Sys=self.ssr_system_rate,
            global_util=self.global_util,
            avg_fair_idx_per_episode=self.avg_fair_idx_per_episode,
            qos_violation=float(self.latest_qos_gap.mean()),
            qos_violation_sum=float(self.latest_qos_gap_sum),
            qos_violation_norm=float(self.latest_qos_gap_norm_mean),
            qos_dual_signal=float(self.latest_qos_dual_signal),
            qos_dual_signal_sum=float(self.latest_qos_dual_signal_sum),
            qos_dual_signal_norm=float(self.latest_qos_dual_signal_norm),
            weighted_qos_gap=float(self.latest_weighted_qos_gap),
            weighted_qos_gap_norm=float(self.latest_weighted_qos_gap_norm),
            qos_gap_sum=float(self.latest_qos_gap_sum),
            qos_gap_mean=float(self.latest_qos_gap_mean),
            energy_efficiency=float(self.latest_ee),
            lambda_penalty=float(self.lambda_penalty),
            tx_power=float(self.latest_tx_power),
            velocity=float(self.latest_velocity),
            rsma_common_rate=float(self.latest_common_rate_sum),
            rsma_private_rate=float(self.latest_private_rate_sum),
            effective_fairness=float(self.latest_effective_fairness),
            beam_quality=float(self.latest_beam_gain),
            alpha_common=float(self.latest_alpha[0]),
            gamma_common_min=float(self.latest_gamma_common_min),
            gamma_private_mean=float(self.latest_gamma_private_mean),
            common_interference=float(self.latest_common_interference),
            private_interference=float(self.latest_private_interference),
        )

        obs = wrapper_obs(self.get_obs())
        state = wrapper_state(self.get_state())
        info['BadMask'] = True if (self.t == self.episode_length) else False

        self.uav_traj.append(self.pos_ubs.copy())
        self.jamming_power_list.append(jamming_powers.copy())
        self.fair_idx_list.append(self.fair_idx)
        self.ssr_list.append(self.ssr_community_sum_t)
        self.throughput_list.append(self.total_throughput_t)

        return obs, state, reward, done, info

    def get_obs_agent(self, agent_id: int) -> dict:
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        ubs_feats = np.zeros(self.obs_ubs_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.obs_gt_feats_size, dtype=np.float32)

        own_feats[0:2] = self.pos_ubs[agent_id] / self.range_pos
        own_feats[2] = self.ssr_ubsk_t[agent_id] / max(self.achievable_rate_ubs_max, 1e-6)
        own_feats[3] = self.latest_rho

        other_ubs = [ubs_id for ubs_id in range(self.n_agents) if ubs_id != agent_id]
        for j, ubs_id in enumerate(other_ubs):
            if self.d_u2u[agent_id][ubs_id] <= self.r_sense:
                ubs_feats[j, 0] = 1.0
                ubs_feats[j, 1:3] = (self.pos_ubs[ubs_id] - self.pos_ubs[agent_id]) / self.range_pos

        qos_norm = max(self.qos_threshold, 1e-6)
        for m in range(self.n_gts):
            if self.d_u2g_level[agent_id][m] <= self.r_cov:
                gt_feats[m, 0] = 1.0
                gt_feats[m, 1:3] = (self.pos_gts[m] - self.pos_ubs[agent_id]) / self.range_pos

                csi_vec = self._build_obs_csi_vector(gt_id=m, uav_id=agent_id)
                csi_start = 3
                csi_end = csi_start + 2 * self.csi_complex_dim
                gt_feats[m, csi_start:csi_end] = csi_vec

                ssr_idx = csi_end
                qos_idx = csi_end + 1
                gt_feats[m, ssr_idx] = self.ssr_gt_rate[m] / max(self.achievable_rate_gts_max, 1e-6)
                gt_feats[m, qos_idx] = self.latest_qos_gap[m] / qos_norm

        return dict(agent=own_feats, ubs=ubs_feats, gt=gt_feats)

    @property
    def obs_own_feats_size(self) -> int:
        # pos(2) + ssr(1) + rho(1)
        return 4

    @property
    def obs_gt_feats_size(self) -> tuple:
        # visibility(1) + relative pos(2) + CSI real/imag(2*csi_complex_dim) + normalized ssr(1) + normalized qos-gap(1)
        return self.n_gts, 5 + 2 * self.csi_complex_dim

    def get_state(self) -> np.ndarray:
        ubs_feats = np.zeros(self.state_ubs_feats_size(), dtype=np.float32)
        gt_feats = np.zeros(self.state_gt_feats_size(), dtype=np.float32)

        ubs_feats[:, 0:2] = self.pos_ubs / self.range_pos
        ubs_feats[:, 2] = self.ssr_ubsk_t / max(self.achievable_rate_ubs_max, 1e-6)
        ubs_feats[:, 3] = self.latest_rho

        qos_norm = max(self.qos_threshold, 1e-6)
        gt_feats[:, 0:2] = self.pos_gts / self.range_pos
        gt_feats[:, 2] = self.latest_effective_rate / max(self.achievable_rate_gts_max, 1e-6)
        gt_feats[:, 3] = self.latest_qos_gap / qos_norm

        return np.concatenate((ubs_feats.flatten(), gt_feats.flatten()))

    def state_ubs_feats_size(self) -> tuple:
        # pos(2) + ssr(1) + rho(1)
        return self.n_ubs, 4

    def state_gt_feats_size(self) -> tuple:
        # pos(2) + rate(1) + qos-gap(1)
        return self.n_gts, 4

    def get_reward(self, reward_scale_rate=1.0) -> float:
        effective_rate = self.latest_effective_rate
        qos_norm = max(self.qos_threshold, 1e-6)
        qos_sum_norm = max(self.qos_threshold * max(self.n_gts, 1), 1e-6)

        # Paper-aligned dual residual: (R_th - R_k) can be positive or negative.
        self.latest_qos_margin = (self.qos_threshold - effective_rate).astype(np.float32)
        self.latest_qos_gap = np.maximum(0.0, self.latest_qos_margin).astype(np.float32)
        self.latest_qos_gap_sum = float(np.sum(self.latest_qos_gap))
        self.latest_qos_gap_mean = float(np.mean(self.latest_qos_gap))
        self.latest_qos_dual_signal = float(np.mean(self.latest_qos_margin))
        self.latest_qos_dual_signal_sum = float(np.sum(self.latest_qos_margin))
        self.latest_qos_gap_norm_mean = self.latest_qos_gap_mean / qos_norm
        self.latest_qos_dual_signal_norm = self.latest_qos_dual_signal / qos_norm
        throughput_mbps = float(np.sum(effective_rate))
        self.total_throughput_t = throughput_mbps
        self.total_throughput = float(self.total_throughput) - float(self.latest_base_throughput) + throughput_mbps

        fly_power = self.fly_power_base + self.fly_power_coeff * (self.latest_velocity ** 2)
        total_power = max(fly_power + self.latest_tx_power, 1e-6)
        self.latest_ee = (throughput_mbps * 1e6) / total_power

        # Paper reward: sum_k max(0, R_th - R_k).
        self.latest_weighted_qos_gap = self.latest_qos_gap_sum
        self.latest_weighted_qos_gap_norm = self.latest_qos_gap_sum / qos_sum_norm

        reward_scalar = (
            self.reward_ee_scale * self.latest_ee * 1e-6
            - self.lambda_penalty * self.reward_qos_scale * self.latest_weighted_qos_gap
        )

        # Fairness is enforced through QoS constraints and dual updates,
        # so we avoid an extra multiplicative fairness term here.

        rewards = np.full(self.n_agents, reward_scalar, dtype=np.float32)
        idle_ubs_mask = self.rate_per_ubs_t == 0
        rewards = rewards * (1 - idle_ubs_mask.astype(np.float32))
        return rewards
