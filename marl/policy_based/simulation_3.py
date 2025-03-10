import pygame
import numpy as np
from marl.gridenv.config import FPS, STEP_DELAY, GOAL_DELAY
from marl.gridenv.environment import FootballEnv
from marl.policy_based.agents import IReinforceAgent

pygame.init()
clock = pygame.time.Clock()

env = FootballEnv()
action_space = env.action_space
state_dim = 7     # State form: (self.positions["A"], self.positions["B"], self.ball_owner, self.score, self.timestep), can include time here
action_dim = len(action_space)
agent_A = IReinforceAgent('Player A', input_dim=state_dim, action_dim=action_dim)
agent_B = IReinforceAgent('Player B', input_dim=state_dim, action_dim=action_dim)

num_episodes = 10
max_score = 5
max_timesteps = 100

for episode in range(num_episodes):
    state = env.reset()
    done = False
    score = 0
    timesteps = 0
    total_rewards = {"A": 0, "B": 0}

    while score < max_score and timesteps < max_timesteps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        state_vector = np.array([
            *state[0],
            *state[1],
            0 if state[2] == 'A' else 1, 
            state[3]["A"] - state[3]["B"],
            state[4]   # For the timestamp, might be useful
        ], dtype=np.float32)

        # print(state_vector)

        action_A, log_prob_A = agent_A.select_action(state_vector)
        action_B, log_prob_B = agent_B.select_action(state_vector)
        actions = {"A": env.action_space[action_A], "B": env.action_space[action_B]}

        next_state, rewards, done, info = env.step(actions)

        agent_A.store_outcome(log_prob_A, action_A)
        agent_B.store_outcome(log_prob_B, action_B)

        total_rewards['A'] += rewards['A']
        total_rewards['B'] += rewards['B']

        state = next_state
        timesteps = state[4]

        score = max(env.score['A'], env.score['B'])

        # if episode % 1000 == 0:
        print(f"Episode {episode+1}, Score: A {env.score['A']} - {env.score['B']} B")
        if info["goal"]:
            env.render()
            pygame.time.delay(GOAL_DELAY)
            env.reset(on_goal=True, conceeding_team=info["conceeding_team"])
        env.render()

        pygame.time.delay(STEP_DELAY)
        clock.tick(FPS)

    agent_A.update_policy()
    agent_B.update_policy()

pygame.quit()
