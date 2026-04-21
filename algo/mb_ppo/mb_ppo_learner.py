import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import AdamW

from algo.mb_ppo.agents.mb_ppo_agent import MultiBranchActorCritic


class MultiBranchPPOLearner:
    def __init__(self, env_info, args):
        self.args = args
        self.device = args.device

        self.n_agents = env_info['n_agents']
        self.n_gts = args.n_gts
        self.gt_features_dim = env_info['gt_features_dim']
        self.other_features_dim = env_info['other_features_dim']
        self.state_dim = env_info['state_shape']

        self.traj_dim = env_info.get('traj_dim', 2)
        antenna_count = int(env_info.get('antenna_count', getattr(args, 'antenna_count', 16)))
        legacy_precoding_dim = int(env_info.get('precoding_dim', getattr(args, 'precoding_dim', 0)))

        self.common_precoding_dim = int(env_info.get('common_precoding_dim', 2 * antenna_count))
        self.private_precoding_dim = int(
            env_info.get(
                'private_precoding_dim',
                max(0, legacy_precoding_dim - self.common_precoding_dim),
            )
        )
        if self.private_precoding_dim <= 0:
            self.private_precoding_dim = int(2 * self.n_gts * antenna_count)

        self.resource_dim = int(
            env_info.get(
                'resource_dim',
                int(env_info.get('power_dim', self.n_gts + 2)) + int(env_info.get('rate_dim', self.n_gts)),
            )
        )

        self.actor_critic = MultiBranchActorCritic(
            gt_features_dim=self.gt_features_dim,
            other_features_dim=self.other_features_dim,
            state_dim=self.state_dim,
            n_gts=self.n_gts,
            n_layers=args.n_layers,
            hidden_size=args.hidden_size,
            traj_dim=self.traj_dim,
            common_precoding_dim=self.common_precoding_dim,
            private_precoding_dim=self.private_precoding_dim,
            resource_dim=self.resource_dim,
            single_head=args.single_head,
        ).to(self.device)

        self.total_action_dim = self.actor_critic.total_action_dim

        self.optimizer = AdamW(self.actor_critic.parameters(), lr=args.lr)

        self.clip_ratio = args.clip_ratio
        self.ppo_epochs = args.ppo_epochs
        self.mini_batch_size = args.mini_batch_size
        self.max_grad_norm = args.max_grad_norm
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef

    def set_traj_branch_trainable(self, trainable: bool):
        if self.actor_critic.single_head:
            return

        for p in self.actor_critic.traj_mean.parameters():
            p.requires_grad = bool(trainable)
        self.actor_critic.traj_log_std.requires_grad = bool(trainable)

    def _to_device_obs(self, obs):
        obs_other = obs[0].to(self.device).float()
        obs_gt = obs[1].to(self.device).float()
        return obs_other, obs_gt

    def take_actions(self, obs, state, deterministic=False, policy_branch='all'):
        obs_other, obs_gt = self._to_device_obs(obs)
        state = state.to(self.device).float()

        with torch.no_grad():
            action_tensor, log_prob, _ = self.actor_critic.sample_actions(
                gt_features=obs_gt,
                other_features=obs_other,
                deterministic=deterministic,
                policy_branch=policy_branch,
            )
            value = self.actor_critic.value(state).squeeze(0)

        env_action = self.actor_critic.split_actions(action_tensor)
        env_action = {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in env_action.items()
        }

        return env_action, action_tensor.detach().cpu(), log_prob.detach().cpu(), float(value.item())

    def evaluate_value(self, state):
        state = state.to(self.device).float()
        with torch.no_grad():
            value = self.actor_critic.value(state).squeeze(0)
        return float(value.item())

    def update(self, rollout_buffer, policy_branch='all'):
        data = rollout_buffer.as_tensors(self.device)
        obs_other = data['obs_other']
        obs_gt = data['obs_gt']
        states = data['state']
        actions = data['action']
        old_log_prob = data['old_log_prob']
        returns = data['returns']
        advantages = data['advantages']

        if len(returns) == 0:
            return {
                'LossActor': 0.0,
                'LossCritic': 0.0,
                'Entropy': 0.0,
                'KL': 0.0,
            }

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss_list = []
        critic_loss_list = []
        entropy_list = []
        kl_list = []

        n_steps = len(returns)

        for _ in range(self.ppo_epochs):
            for batch_idx in rollout_buffer.iter_indices(self.mini_batch_size, shuffle=True):
                b_obs_other = obs_other[batch_idx]
                b_obs_gt = obs_gt[batch_idx]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_prob = old_log_prob[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                # Flatten timestep-agent dimensions for actor update.
                flat_obs_other = b_obs_other.reshape(-1, self.other_features_dim)
                flat_obs_gt = b_obs_gt.reshape(-1, self.n_gts, self.gt_features_dim)
                flat_actions = b_actions.reshape(-1, self.total_action_dim)

                flat_old_log_prob = b_old_log_prob.reshape(-1)
                flat_advantages = b_advantages.repeat_interleave(self.n_agents)

                log_prob, entropy = self.actor_critic.evaluate_actions(
                    gt_features=flat_obs_gt,
                    other_features=flat_obs_other,
                    action=flat_actions,
                    policy_branch=policy_branch,
                )

                ratio = torch.exp(log_prob - flat_old_log_prob)
                surr1 = ratio * flat_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * flat_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.actor_critic.value(b_states)
                critic_loss = F.mse_loss(values, b_returns)

                entropy_mean = entropy.mean()
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (flat_old_log_prob - log_prob).mean().item()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                entropy_list.append(entropy_mean.item())
                kl_list.append(approx_kl)

        return {
            'LossActor': float(np.mean(actor_loss_list)),
            'LossCritic': float(np.mean(critic_loss_list)),
            'Entropy': float(np.mean(entropy_list)),
            'KL': float(np.mean(kl_list)),
            'PolicyBranch': str(policy_branch),
        }

    def save_model(self, path, stamp):
        checkpoint = dict()
        checkpoint.update(stamp)
        checkpoint['model_state_dict'] = self.actor_critic.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f'Save checkpoint to {path}.')

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.actor_critic.eval()
        print(f'Load checkpoint from {path}.')
