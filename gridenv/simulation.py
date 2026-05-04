import pygame
import random
from config import FPS, STEP_DELAY, GOAL_DELAY
from environment import FootballEnv

pygame.init()
clock = pygame.time.Clock()

env = FootballEnv(team_size=3, size="S")
action_space = env.action_space
done = False

env.render()
pygame.time.delay(STEP_DELAY)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    actions = {
        "A": [random.choice(action_space) for _ in range(env.team_size)],
        "B": [random.choice(action_space) for _ in range(env.team_size)]
    }

    state, rewards, done, info = env.step(actions)
    if info["goal"]:
        env.render()
        pygame.time.delay(GOAL_DELAY)
        env.reset(on_goal=True, conceding_team=info["conceding_team"])
    env.render()

    pygame.time.delay(STEP_DELAY)
    clock.tick(FPS)

pygame.quit()
