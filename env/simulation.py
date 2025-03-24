import pygame
import numpy as np
from env.config import FPS, PLAYER_MAX_ACCELERATION, PLAYER_MAX_ANGULAR_ACCELERATION, PLAYER_MAX_KICKING_FORCE
from env.environment import FootballEnv

pygame.init()
clock = pygame.time.Clock()

env = FootballEnv(team_size=5)

state = env.reset()
done = False

def random_action():
    acceleration = np.random.uniform(0, PLAYER_MAX_ACCELERATION)
    angular_acceleration = np.random.uniform(-PLAYER_MAX_ANGULAR_ACCELERATION, PLAYER_MAX_ANGULAR_ACCELERATION)
    kicking_force = np.random.uniform(0, PLAYER_MAX_KICKING_FORCE)
    kicking_angle = np.random.uniform(-np.pi, np.pi)
    return [acceleration, angular_acceleration, kicking_force, kicking_angle]

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    actions = [random_action() for _ in range(env.num_players)]
    state, rewards, done, _ = env.step(actions)
    env.render()
    
    clock.tick(FPS)

pygame.quit()