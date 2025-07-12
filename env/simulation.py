import os
os.environ["SDL_AUDIODRIVER"] = "dummy" # Disable audio (wsl)

import pygame
import numpy as np
from env.config import FPS, PLAYER_MAX_ACCELERATION, PLAYER_MAX_ANGULAR_ACCELERATION, PLAYER_MAX_KICKING_FORCE
from env.environment import FootballEnv
from env.schema import PlayerAction, TeamActions

pygame.init()
clock = pygame.time.Clock()

env = FootballEnv(team_size=1)

state = env.reset()
done = False

def random_action():
    acceleration = np.random.uniform(0, 1)
    angular_acceleration = np.random.uniform(-1, 1)
    kicking_force = np.random.uniform(0, 1)
    kicking_angle = np.random.uniform(-1, 1)
    return PlayerAction(
        acceleration=acceleration,
        angular_acceleration=angular_acceleration,
        kicking_force=kicking_force,
        kicking_angle=kicking_angle
    )

i = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    actions = TeamActions(
        team1=[random_action() for _ in range(env.team_size)],
        team2=[random_action() for _ in range(env.team_size)]
    )
    step_result = env.step(actions)
    state, rewards, done, _ = step_result.state, step_result.rewards, step_result.done, step_result.info
    env.render()
    
    clock.tick(FPS)
    
    # print(f"State: {state}")
    i += 1

print(i)
pygame.quit()