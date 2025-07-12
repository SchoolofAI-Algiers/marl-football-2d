import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.clear()
        
    def add_experience(self, obs, action, log_prob, value, reward, done):
        """Add a single experience to the buffer"""
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = []
