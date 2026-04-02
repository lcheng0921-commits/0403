import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.obs_other = []
        self.obs_gt = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.qos_signals = []

        self.returns = None
        self.advantages = None

    def add(
        self,
        obs_other,
        obs_gt,
        state,
        action,
        log_prob,
        reward,
        done,
        value,
        qos_violation,
    ):
        self.obs_other.append(obs_other.detach().cpu().float())
        self.obs_gt.append(obs_gt.detach().cpu().float())
        self.states.append(state.detach().cpu().float())
        self.actions.append(action.detach().cpu().float())
        self.log_probs.append(log_prob.detach().cpu().float())

        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.qos_signals.append(float(qos_violation))

    def compute_returns_and_advantages(self, last_value=0.0):
        n_steps = len(self.rewards)
        if n_steps == 0:
            self.returns = torch.zeros(0, dtype=torch.float32)
            self.advantages = torch.zeros(0, dtype=torch.float32)
            return

        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)

        gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(n_steps)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * next_value * mask - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
            next_value = self.values[t]

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)

    def as_tensors(self, device):
        if self.returns is None or self.advantages is None:
            raise RuntimeError('Call compute_returns_and_advantages before as_tensors.')

        state_tensor = torch.stack(self.states)
        if state_tensor.dim() == 3 and state_tensor.size(1) == 1:
            state_tensor = state_tensor.squeeze(1)

        return {
            'obs_other': torch.stack(self.obs_other).to(device),
            'obs_gt': torch.stack(self.obs_gt).to(device),
            'state': state_tensor.to(device),
            'action': torch.stack(self.actions).to(device),
            'old_log_prob': torch.stack(self.log_probs).to(device),
            'returns': self.returns.to(device),
            'advantages': self.advantages.to(device),
        }

    def iter_indices(self, mini_batch_size, shuffle=True):
        indices = np.arange(len(self.rewards))
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), mini_batch_size):
            yield indices[start : start + mini_batch_size]

    def mean_qos_violation(self):
        if not self.qos_signals:
            return 0.0
        return float(np.mean(self.qos_signals))

    def mean_qos_signal(self):
        # Stored value is the dual-update signal collected during rollout.
        return self.mean_qos_violation()

    def __len__(self):
        return len(self.rewards)
