import torch
import torch.nn as nn
from torch.distributions import Normal


class MultiBranchActorCritic(nn.Module):
    def __init__(
        self,
        gt_features_dim,
        other_features_dim,
        state_dim,
        n_gts,
        n_heads=2,
        n_layers=2,
        hidden_size=256,
        traj_dim=2,
        precoding_dim=32,
        csi_complex_dim=4,
        power_dim=8,
        rate_dim=6,
        single_head=False,
        use_mha=True,
        use_qos_guided_attention=True,
        qos_feature_index=-1,
        qos_attn_bias_scale=2.0,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()

        self.n_gts = n_gts
        self.gt_features_dim = gt_features_dim
        self.other_features_dim = other_features_dim

        self.traj_dim = traj_dim
        self.precoding_dim = precoding_dim
        self.power_dim = power_dim
        self.rate_dim = rate_dim
        self.total_action_dim = traj_dim + precoding_dim + power_dim + rate_dim

        self.n_heads = n_heads
        if self.n_heads != 3:
            raise ValueError('Semantic MHA requires n_heads=3 (position/CSI/QoS).')

        self.csi_complex_dim = int(max(1, csi_complex_dim))
        self.single_head = single_head
        self.use_mha = bool(use_mha)
        self.use_qos_guided_attention = bool(use_qos_guided_attention)
        self.qos_feature_index = int(qos_feature_index)
        self.qos_attn_bias_scale = float(max(0.0, qos_attn_bias_scale))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Diagnostics for analysis and potential logging.
        self.latest_attn_weights = None
        self.latest_token_weights = None

        # Semantic-head MHA:
        #   head-1 uses position-only tokens,
        #   head-2 uses explicit CSI(real/imag)-only tokens,
        #   head-3 uses QoS-gap-only tokens.
        self.pos_input_dim = 3  # rel_x, rel_y, rel_distance
        self.csi_input_dim = 2 * self.csi_complex_dim  # real + imag
        self.qos_input_dim = 1
        self.semantic_head_dim = max(32, hidden_size // 2)

        self.pos_proj = nn.Sequential(
            nn.Linear(self.pos_input_dim, self.semantic_head_dim),
            nn.ReLU(),
        )
        self.csi_proj = nn.Sequential(
            nn.Linear(self.csi_input_dim, self.semantic_head_dim),
            nn.ReLU(),
        )
        self.qos_proj = nn.Sequential(
            nn.Linear(self.qos_input_dim, self.semantic_head_dim),
            nn.ReLU(),
        )

        self.pos_attn = nn.MultiheadAttention(embed_dim=self.semantic_head_dim, num_heads=1, batch_first=True)
        self.csi_attn = nn.MultiheadAttention(embed_dim=self.semantic_head_dim, num_heads=1, batch_first=True)
        self.qos_attn = nn.MultiheadAttention(embed_dim=self.semantic_head_dim, num_heads=1, batch_first=True)

        self.semantic_fusion = nn.Sequential(
            nn.Linear(self.semantic_head_dim * 3, self.semantic_head_dim),
            nn.ReLU(),
        )

        self.no_mha_encoder = nn.Sequential(
            nn.Linear(self.n_gts * self.gt_features_dim, self.semantic_head_dim),
            nn.ReLU(),
            nn.Linear(self.semantic_head_dim, self.semantic_head_dim),
            nn.ReLU(),
        )
        attn_dim = self.semantic_head_dim

        actor_layers = [nn.Linear(attn_dim + other_features_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            actor_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.actor_encoder = nn.Sequential(*actor_layers)

        if self.single_head:
            self.actor_mean = nn.Linear(hidden_size, self.total_action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(self.total_action_dim))
        else:
            self.traj_mean = nn.Linear(hidden_size, self.traj_dim)
            self.precoding_mean = nn.Linear(hidden_size, self.precoding_dim)
            self.power_mean = nn.Linear(hidden_size, self.power_dim)
            self.rate_mean = nn.Linear(hidden_size, self.rate_dim)

            self.traj_log_std = nn.Parameter(torch.zeros(self.traj_dim))
            self.precoding_log_std = nn.Parameter(torch.zeros(self.precoding_dim))
            self.power_log_std = nn.Parameter(torch.zeros(self.power_dim))
            self.rate_log_std = nn.Parameter(torch.zeros(self.rate_dim))

        critic_layers = [nn.Linear(state_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            critic_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        critic_layers += [nn.Linear(hidden_size, 1)]
        self.critic = nn.Sequential(*critic_layers)

    def _atanh(self, x):
        x = torch.clamp(x, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _build_qos_guided_attn_bias(self, qos_gap):
        if not self.use_qos_guided_attention:
            return None, None

        qos_gap = torch.clamp(qos_gap.squeeze(-1), min=0.0)
        # Normalize per sample to produce stable attention guidance.
        qos_scale = torch.mean(qos_gap, dim=1, keepdim=True) + 1e-6
        qos_norm = qos_gap / qos_scale

        # Centered key bias: positive means the token is more disadvantaged.
        key_bias = qos_norm - torch.mean(qos_norm, dim=1, keepdim=True)
        key_bias = self.qos_attn_bias_scale * key_bias

        batch_size, n_tokens = key_bias.shape
        attn_bias = key_bias.unsqueeze(1).expand(batch_size, n_tokens, n_tokens)
        attn_bias = attn_bias.repeat_interleave(1, dim=0)
        return attn_bias, qos_norm

    def _extract_semantic_tokens(self, gt_features):
        # Layout from env:
        # [visible(1), rel_pos(2), csi_real(c), csi_imag(c), ssr(1), qos_gap(1)]
        n_feat = gt_features.size(-1)

        rel_pos = gt_features[..., 1:3]
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        pos_token = torch.cat([rel_pos, rel_dist], dim=-1)

        csi_start = 3
        csi_end = min(csi_start + self.csi_input_dim, n_feat)
        csi_token = gt_features[..., csi_start:csi_end]
        if csi_token.size(-1) < self.csi_input_dim:
            pad = torch.zeros(*csi_token.shape[:-1], self.csi_input_dim - csi_token.size(-1), device=csi_token.device, dtype=csi_token.dtype)
            csi_token = torch.cat([csi_token, pad], dim=-1)

        qos_idx = self.qos_feature_index if self.qos_feature_index >= 0 else n_feat + self.qos_feature_index
        qos_idx = min(max(qos_idx, 0), n_feat - 1)
        qos_token = gt_features[..., qos_idx:qos_idx + 1]

        return {
            'pos': pos_token,
            'csi': csi_token,
            'qos': qos_token,
        }

    def _pool_tokens(self, attn_out, attn_weights, extra_bias=None):
        token_importance = attn_weights.mean(dim=1).mean(dim=1)
        if extra_bias is not None:
            token_importance = token_importance + self.qos_attn_bias_scale * extra_bias
        token_weights = torch.softmax(token_importance, dim=-1)
        pooled = torch.sum(attn_out * token_weights.unsqueeze(-1), dim=1)
        return pooled, token_weights

    def _actor_forward(self, gt_features, other_features):
        if not self.use_mha:
            flat_gt = gt_features.reshape(gt_features.size(0), -1)
            fused = self.no_mha_encoder(flat_gt)
            self.latest_attn_weights = None
            self.latest_token_weights = None
            x = torch.cat([fused, other_features], dim=-1)
            return self.actor_encoder(x)

        semantic_tokens = self._extract_semantic_tokens(gt_features)

        pos_embed = self.pos_proj(semantic_tokens['pos'])
        csi_embed = self.csi_proj(semantic_tokens['csi'])
        qos_embed = self.qos_proj(semantic_tokens['qos'])

        pos_out, pos_attn_w = self.pos_attn(
            pos_embed,
            pos_embed,
            pos_embed,
            need_weights=True,
            average_attn_weights=False,
        )
        csi_out, csi_attn_w = self.csi_attn(
            csi_embed,
            csi_embed,
            csi_embed,
            need_weights=True,
            average_attn_weights=False,
        )

        qos_attn_mask, qos_norm = self._build_qos_guided_attn_bias(semantic_tokens['qos'])
        qos_out, qos_attn_w = self.qos_attn(
            qos_embed,
            qos_embed,
            qos_embed,
            attn_mask=qos_attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        pos_pooled, pos_token_w = self._pool_tokens(pos_out, pos_attn_w)
        csi_pooled, csi_token_w = self._pool_tokens(csi_out, csi_attn_w)
        qos_pooled, qos_token_w = self._pool_tokens(qos_out, qos_attn_w, extra_bias=qos_norm)

        fused = torch.cat([pos_pooled, csi_pooled, qos_pooled], dim=-1)
        fused = self.semantic_fusion(fused)

        self.latest_attn_weights = {
            'pos': pos_attn_w.detach(),
            'csi': csi_attn_w.detach(),
            'qos': qos_attn_w.detach(),
        }
        self.latest_token_weights = {
            'pos': pos_token_w.detach(),
            'csi': csi_token_w.detach(),
            'qos': qos_token_w.detach(),
        }

        x = torch.cat([fused, other_features], dim=-1)
        return self.actor_encoder(x)

    def policy(self, gt_features, other_features):
        x = self._actor_forward(gt_features, other_features)

        if self.single_head:
            mean = self.actor_mean(x)
            log_std = self.actor_log_std.unsqueeze(0).expand_as(mean)
        else:
            mean = torch.cat(
                [
                    self.traj_mean(x),
                    self.precoding_mean(x),
                    self.power_mean(x),
                    self.rate_mean(x),
                ],
                dim=-1,
            )
            log_std = torch.cat(
                [
                    self.traj_log_std,
                    self.precoding_log_std,
                    self.power_log_std,
                    self.rate_log_std,
                ],
                dim=0,
            )
            log_std = log_std.unsqueeze(0).expand_as(mean)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def value(self, state):
        return self.critic(state).squeeze(-1)

    def _log_prob_from_pre_tanh(self, dist, pre_tanh, action):
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def sample_actions(self, gt_features, other_features, deterministic=False):
        mean, log_std = self.policy(gt_features, other_features)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = dist.rsample()

        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh, action)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate_actions(self, gt_features, other_features, action):
        mean, log_std = self.policy(gt_features, other_features)
        std = log_std.exp()
        dist = Normal(mean, std)

        pre_tanh = self._atanh(action)
        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh, action)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

    def split_actions(self, action):
        idx = 0
        traj = action[..., idx : idx + self.traj_dim]
        idx += self.traj_dim

        precoding = action[..., idx : idx + self.precoding_dim]
        idx += self.precoding_dim

        power = action[..., idx : idx + self.power_dim]
        idx += self.power_dim

        rate = action[..., idx : idx + self.rate_dim]

        return {
            'traj': traj,
            'precoding': precoding,
            'power': power,
            'rate': rate,
        }
