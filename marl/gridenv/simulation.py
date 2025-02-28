import pygame
import random
from config import FPS
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

    state, rewards = env.step(actions)
    env.render()

    pygame.time.delay(500)
    clock.tick(FPS)

pygame.quit()
