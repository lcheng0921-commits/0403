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
        n_layers=2,
        hidden_size=256,
        traj_dim=2,
        common_precoding_dim=32,
        private_precoding_dim=192,
        resource_dim=14,
        single_head=False,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()

        self.n_gts = n_gts
        self.gt_features_dim = gt_features_dim
        self.other_features_dim = other_features_dim

        self.traj_dim = traj_dim
        self.common_precoding_dim = common_precoding_dim
        self.private_precoding_dim = private_precoding_dim
        self.resource_dim = resource_dim
        self.total_action_dim = (
            traj_dim
            + common_precoding_dim
            + private_precoding_dim
            + resource_dim
        )

        self.single_head = single_head
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Deeper shared encoder improves cross-task global context for trajectory/beam/power branches.
        self.gt_embed_dim = max(128, hidden_size)
        self.gt_encoder = nn.Sequential(
            nn.Linear(self.n_gts * self.gt_features_dim, self.gt_embed_dim),
            nn.ReLU(),
            nn.Linear(self.gt_embed_dim, self.gt_embed_dim),
            nn.ReLU(),
            nn.Linear(self.gt_embed_dim, self.gt_embed_dim),
            nn.ReLU(),
            nn.Linear(self.gt_embed_dim, self.gt_embed_dim),
            nn.ReLU(),
        )

        actor_layers = [nn.Linear(self.gt_embed_dim + other_features_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            actor_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.actor_encoder = nn.Sequential(*actor_layers)

        if self.single_head:
            self.actor_mean = nn.Linear(hidden_size, self.total_action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(self.total_action_dim))
        else:
            self.traj_mean = nn.Linear(hidden_size, self.traj_dim)
            self.common_precoding_mean = nn.Linear(hidden_size, self.common_precoding_dim)
            self.private_precoding_mean = nn.Linear(hidden_size, self.private_precoding_dim)
            self.resource_mean = nn.Linear(hidden_size, self.resource_dim)

            self.traj_log_std = nn.Parameter(torch.zeros(self.traj_dim))
            self.common_precoding_log_std = nn.Parameter(torch.zeros(self.common_precoding_dim))
            self.private_precoding_log_std = nn.Parameter(torch.zeros(self.private_precoding_dim))
            self.resource_log_std = nn.Parameter(torch.zeros(self.resource_dim))

        critic_layers = [nn.Linear(state_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            critic_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        critic_layers += [nn.Linear(hidden_size, 1)]
        self.critic = nn.Sequential(*critic_layers)

    def _atanh(self, x):
        x = torch.clamp(x, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _branch_slice(self, branch):
        if branch in (None, 'all'):
            return slice(0, self.total_action_dim)

        traj_end = self.traj_dim
        common_end = traj_end + self.common_precoding_dim
        precoding_end = common_end + self.private_precoding_dim
        resource_start = precoding_end
        # Resource branch layout: [rho(1), alpha(n_gts+1), beta(n_gts)].
        rho_alpha_end = resource_start + 1 + (self.n_gts + 1)

        if branch == 'traj':
            return slice(0, traj_end)
        if branch == 'precoding':
            return slice(traj_end, precoding_end)
        if branch == 'resource':
            return slice(resource_start, self.total_action_dim)
        if branch == 'rho_alpha':
            return slice(resource_start, min(rho_alpha_end, self.total_action_dim))
        if branch == 'comm':
            return slice(traj_end, self.total_action_dim)
        raise ValueError(f"Unsupported policy branch: {branch}")

    def _actor_forward(self, gt_features, other_features):
        flat_gt = gt_features.reshape(gt_features.size(0), -1)
        fused = self.gt_encoder(flat_gt)
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
                    self.common_precoding_mean(x),
                    self.private_precoding_mean(x),
                    self.resource_mean(x),
                ],
                dim=-1,
            )
            log_std = torch.cat(
                [
                    self.traj_log_std,
                    self.common_precoding_log_std,
                    self.private_precoding_log_std,
                    self.resource_log_std,
                ],
                dim=0,
            )
            log_std = log_std.unsqueeze(0).expand_as(mean)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def value(self, state):
        return self.critic(state).squeeze(-1)

    def _log_prob_from_pre_tanh(self, dist, pre_tanh, action, branch='all'):
        branch_slice = self._branch_slice(branch)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        return log_prob[..., branch_slice].sum(dim=-1)

    def sample_actions(self, gt_features, other_features, deterministic=False, policy_branch='all'):
        mean, log_std = self.policy(gt_features, other_features)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = dist.rsample()

        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh, action, branch=policy_branch)
        branch_slice = self._branch_slice(policy_branch)
        entropy = dist.entropy()[..., branch_slice].sum(dim=-1)

        return action, log_prob, entropy

    def evaluate_actions(self, gt_features, other_features, action, policy_branch='all'):
        mean, log_std = self.policy(gt_features, other_features)
        std = log_std.exp()
        dist = Normal(mean, std)

        pre_tanh = self._atanh(action)
        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh, action, branch=policy_branch)
        branch_slice = self._branch_slice(policy_branch)
        entropy = dist.entropy()[..., branch_slice].sum(dim=-1)

        return log_prob, entropy

    def split_actions(self, action):
        idx = 0
        traj = action[..., idx : idx + self.traj_dim]
        idx += self.traj_dim

        common_precoding = action[..., idx : idx + self.common_precoding_dim]
        idx += self.common_precoding_dim

        private_precoding = action[..., idx : idx + self.private_precoding_dim]
        idx += self.private_precoding_dim

        resource = action[..., idx : idx + self.resource_dim]

        # Legacy aliases for compatibility with old code paths.
        precoding = torch.cat([common_precoding, private_precoding], dim=-1)
        power_dim = self.n_gts + 2
        power = resource[..., :power_dim]
        rate = resource[..., power_dim:]

        return {
            'traj': traj,
            'common_precoding': common_precoding,
            'private_precoding': private_precoding,
            'resource': resource,
            'precoding': precoding,
            'power': power,
            'rate': rate,
        }
