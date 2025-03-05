import pygame
import numpy as np
from env.config import FPS, PLAYER_MAX_SPEED, PLAYER_MAX_ROTATION, PLAYER_MAX_KICKING_FORCE
from env.environment import FootballEnv

pygame.init()
clock = pygame.time.Clock()

env = FootballEnv(team_size=11)

state = env.reset()
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    actions = [list(np.random.uniform(-PLAYER_MAX_SPEED, PLAYER_MAX_SPEED, 2)) + [np.random.random() * PLAYER_MAX_ROTATION] + list(np.random.uniform(-PLAYER_MAX_KICKING_FORCE, PLAYER_MAX_KICKING_FORCE, 2)) for _ in range(env.num_players)]
    state, rewards, done, _ = env.step(actions)
    env.render()
    
    clock.tick(FPS)

pygame.quit()