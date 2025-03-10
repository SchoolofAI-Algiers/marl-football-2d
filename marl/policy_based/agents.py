import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


class WoLF_PHCAgent:
    def __init__(self, agent_name, actions, alpha=0.1, gamma=0.9, epsilon=0.1, l_w=0.01, l_l=0.04):
        self.agent_name = agent_name
        self.actions = actions
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon
        self.l_w = l_w  # WoLF slow learning rate (when agent is winning)    
        self.l_l = l_l  # WoLF fast learning rate (when agent is loosing)
        self.Q = {}  
        self.policy = {}  
        self.avg_policy = {}  # Average policy
        self.C = {}  # State visit counts

    # State form: (self.positions["A"], self.positions["B"], self.ball_owner, self.score, self.timestep)

    def _hash_state(self, state):
        """Convert state dictionary into a hashable tuple with immutable values."""
        return (state[0], state[1], state[2], (state[3]['A'], state[3]['B']))

    def _initialize_state(self, state):
        """Ensure state exists in Q-table and policies"""
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}
            self.policy[state] = {a: 1.0 / len(self.actions) for a in self.actions}
            self.avg_policy[state] = {a: 1.0 / len(self.actions) for a in self.actions}
            self.C[state] = 0

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        state = self._hash_state(state)  # Convert to tuple of immutable values
        self._initialize_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.policy[state], key=self.policy[state].get)
    
    def update(self, state, action, reward, next_state):
        """Update Q-values, average policy, and policy using WoLF-PHC."""
        state = self._hash_state(state)
        next_state = self._hash_state(next_state)

        self._initialize_state(state)
        self._initialize_state(next_state)

        # Q-learning update rule
        best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
        self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][best_next_action] - self.Q[state][action])

        # Update average policy
        self.C[state] += 1
        for a in self.actions:
            self.avg_policy[state][a] += (self.policy[state][a] - self.avg_policy[state][a]) / self.C[state]

        # Compute expected values for current and average policy
        excepted_value_policy = sum(self.policy[state][a] * self.Q[state][a] for a in self.actions)
        expected_value_avg_policy = sum(self.avg_policy[state][a] * self.Q[state][a] for a in self.actions)

        # Determine if agent is "winning" or "losing"
        delta = self.l_w if excepted_value_policy > expected_value_avg_policy else self.l_l

        # Update policy
        best_action = max(self.Q[state], key=self.Q[state].get)
        for a in self.actions:
            if a == best_action:
                self.policy[state][a] = min(self.policy[state][a] + delta, 1.0)
            else:
                self.policy[state][a] = max(self.policy[state][a] - delta / len(self.actions) - 1, 0.0)
            
        # Normalize policy to sum to 1
        total_prob = sum(self.policy[state].values())
        for a in self.actions:
            self.policy[state][a] /= total_prob

###########################################################

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=34):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class IReinforceAgent:
    def __init__(self, name: str, input_dim: int, action_dim: int, lr=0.01, gamma=0.99):
        self.name = name
        self.gamma = gamma
        self.policy_network = PolicyNetwork(input_dim=input_dim, output_dim=action_dim, hidden_dim=34)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.memory = []  # Stores (log_prob, reward) pairs for training

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy_network(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()  
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def store_outcome(self, log_prob, reward):
        self.memory.append((log_prob, reward))

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []

        for _, reward in reversed(self.memory):
            R = reward + self.gamma * R
            returns.insert(0, R)      

        
        # Normalize
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Policy gradient loss
        for (log_prob, R) in zip(self.memory, returns):
            policy_loss.append(-log_prob[0] * R)
        
        self.optimizer.zero_grad()

        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        self.memory = []

###########################################################

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=34):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class IA2CAgent:
    def __init__(self):
        pass