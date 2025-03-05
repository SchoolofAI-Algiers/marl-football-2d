import pygame
import numpy as np
from config import FPS
from environment import FootballEnv
import time


pygame.init()
clock = pygame.time.Clock()

env = FootballEnv(team_size=11)

state = env.reset()
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    act = 1
    actions = [np.random.uniform(-act, act, 5) for _ in range(env.num_players)]
    state, rewards, done, _ = env.step(actions)
    env.render()
    
    clock.tick(FPS)

pygame.quit()