import pygame
import random
from config import FPS, STEP_DELAY, GOAL_DELAY
from environment import FootballEnv

pygame.init()
clock = pygame.time.Clock()

env = FootballEnv()
action_space = env.action_space
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    actions = {
        "A": random.choice(action_space),
        "B": random.choice(action_space)
    }

    state, rewards, done, info = env.step(actions)
    if info["goal"]:
        env.render()
        pygame.time.delay(GOAL_DELAY)
        env.reset(on_goal=True, conceeding_team=info["conceeding_team"])
    env.render()

    pygame.time.delay(STEP_DELAY)
    clock.tick(FPS)

pygame.quit()
