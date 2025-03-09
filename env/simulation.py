import pygame
import random 
import numpy as np
from env.config import FPS, PLAYER_MAX_SPEED, PLAYER_MAX_ROTATION, PLAYER_MAX_KICKING_FORCE
from env.environment import FootballEnv
import env.config as config

pygame.init()
clock = pygame.time.Clock()

env = FootballEnv(team_size=11)

state = env.reset()
done = False

# Stadium dimensions
stadium_width = config.STADIUM_LENGTH * config.LENGTH_TO_WIDTH_RATIO - 10
stadium_height = config.STADIUM_LENGTH - 10

# Function to generate random positions within the stadium
def random_position():
    return np.array([
        random.uniform(5, stadium_width),
        random.uniform(5, stadium_height )
    ])


# Randomly place all players
player_positions = [random_position() for _ in range(22)]  # 22 players in total

team_roles = [
        ["pass"] + ["catch"] + ["follow"] + ["tackle"] * 5 + ["wander"] * 3,  # Team 0
        ["pass"] + ["catch"] + ["follow"] + ["tackle"] * 5 + ["wander"] * 3   # Team 1
    ]


while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            
    # Shuffle actions within each team
    player_positions = [random_position() for _ in range(22)]  # 22 players in total
    
    for team in team_roles:
        random.shuffle(team)

    actions = team_roles[0] + team_roles[1]

    # Assign the ball to the passer from team 0
    passer_index = actions.index("pass")  # Find the passer in team 0
    ball_position = player_positions[passer_index]

    # Place a tackler from team 1 near the passer
    tackler_index = 11 + actions[11:].index("tackle")  # Find a tackler in team 1
    player_positions[tackler_index] = ball_position + np.random.uniform(-5, 5, size=2)  # Place near the passer


    #actions = [list(np.random.uniform(-PLAYER_MAX_SPEED, PLAYER_MAX_SPEED, 2)) + [np.random.random() * PLAYER_MAX_ROTATION] + list(np.random.uniform(-PLAYER_MAX_KICKING_FORCE, PLAYER_MAX_KICKING_FORCE, 2)) for _ in range(env.num_players)]
    
    state, rewards, done, _ = env.step(actions)
    env.render()
    
    clock.tick(FPS)

pygame.quit()
