import pygame
import random
from marl.gridenv.config import FPS, STEP_DELAY, GOAL_DELAY
from marl.gridenv.environment import FootballEnv
from marl.policy_based.agents import WoLF_PHCAgent

pygame.init()
clock = pygame.time.Clock()

env = FootballEnv()
action_space = env.action_space
agent_A = WoLF_PHCAgent('Player A', actions=action_space, epsilon=0.3)
agent_B = WoLF_PHCAgent('Player B', actions=action_space, epsilon=0.3)

num_episodes = 10
max_score = 5

for episode in range(num_episodes):
    state = env.reset()
    done = False
    score = 0

    while score < max_score:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action_A = agent_A.select_action(state[:-1])
        action_B = agent_B.select_action(state[:-1])
        actions = {'A': action_A, 'B': action_B}

        next_state, rewards, done, info = env.step(actions)

        agent_A.update(state, action_A, rewards["A"] - rewards["B"], next_state[:-1])
        agent_B.update(state, action_B, rewards["B"] - rewards["A"], next_state[:-1])

        state = next_state

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

pygame.quit()
