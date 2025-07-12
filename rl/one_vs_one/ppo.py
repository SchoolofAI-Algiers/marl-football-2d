from pydantic import BaseModel
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution, TanhTransform

# Action space constants
ACTION_LOW = torch.tensor([0.0, -1.0, 0.0, -1.0])
ACTION_HIGH = torch.tensor([1.0, 1.0, 1.0, 1.0])

class PPOConfig(BaseModel):
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_length: int = 2560
    mini_batch_size: int = 128
    epochs: int = 8

class PPOMetrics(BaseModel):
    policy_loss: float
    value_loss: float
    entropy: float
    loss: float

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)  # initial std = exp(-0.5) ≈ 0.6

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.actor_mean(h)
        std = self.log_std.exp().expand_as(mu)
        value = self.critic(h)
        return mu, std, value

    def get_dist(self, obs: torch.Tensor):
        mu, std, _ = self.forward(obs)
        base_dist = Normal(mu, std)
        return TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.net(obs)
            return self.critic(h).squeeze(-1)

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig):
        self.config = config
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.policy = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        
        self.low = ACTION_LOW.to(self.device)
        self.high = ACTION_HIGH.to(self.device)

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Maps action from [-1, 1] to actual action range."""
        return self.low + 0.5 * (action + 1.0) * (self.high - self.low)

    def select_action(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs_t = torch.FloatTensor(obs).to(self.device)  # shape: [B, obs_dim]
        obs_t = obs
        with torch.no_grad():
            dist = self.policy.get_dist(obs_t)
            raw_actions = dist.sample()
            log_probs = dist.log_prob(raw_actions).sum(dim=1)
            values = self.policy.get_value(obs_t)
            scaled_actions = self.rescale_action(raw_actions)
        return scaled_actions, log_probs, values

    def compute_gae(self, rewards, values, dones):
        config = self.config
        gae = 0
        returns = []
        values = values + [0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + config.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + config.gamma * config.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, buffer: Dict[str, List]) -> PPOMetrics:
        obs = torch.FloatTensor(np.array(buffer['obs'])).to(self.device)
        actions = torch.FloatTensor(np.array(buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(buffer['log_probs']).to(self.device)
        returns = torch.FloatTensor(buffer['returns']).to(self.device)
        values = torch.FloatTensor(buffer['values']).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Inverse rescale actions to raw Tanh range [-1, 1]
        raw_actions = 2 * (actions - self.low) / (self.high - self.low) - 1.0
        
        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []

        for _ in range(self.config.epochs):
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)
            for start in range(0, len(obs), self.config.mini_batch_size):
                mb_idx = idxs[start:start+self.config.mini_batch_size]
                mb_obs = obs[mb_idx]
                mb_raw_actions = raw_actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                dist = self.policy.get_dist(mb_obs)
                new_log_probs = dist.log_prob(mb_raw_actions).sum(1)
                entropy = dist.base_dist.entropy().sum(1).mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                _, _, value = self.policy(mb_obs)
                value_loss = (mb_returns - value.squeeze()).pow(2).mean()

                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                total_losses.append(loss.item())
                
        return PPOMetrics(
            policy_loss=np.mean(policy_losses),
            value_loss=np.mean(value_losses),
            entropy=np.mean(entropies),
            loss=np.mean(total_losses)
        )
