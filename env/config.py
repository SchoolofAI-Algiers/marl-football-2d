import numpy as np

# stadium attributes
STADIUM_LENGTH = 100
PLAYERS_TO_LENGTH_RATIO = 1/22
LENGTH_TO_WIDTH_RATIO = 6/10

# player attributes
PLAYER_RADIUS = 1
PLAYER_MAX_SPEED = 30
PLAYER_MAX_ROTATION = np.pi

# ball attributes
BALL_RADIUS = 0.5

# simulation attributes
DT = 0.1
MAX_GAME_TIME = 300
FPS = 60

# physics attributes
FRICTION_FACTOR = 0.9
ANGULAR_FRICTION_FACTOR = 0.9
AIR_RESISTANCE = 0.98