import numpy as np

# stadium attributes
STADIUM_LENGTH = 100
PLAYERS_TO_LENGTH_RATIO = 1/22
LENGTH_TO_WIDTH_RATIO = 6/10

# player attributes
PLAYER_RADIUS = 1
PLAYER_MAX_SPEED = 30
PLAYER_MAX_ROTATION = np.pi
PLAYER_MAX_ACCELERATION = 2.0

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


# Steering movements Consts

TARGET_RADIUS = 1 #needed in the align behavior
SLOW_RADIUS = 1.5
MAX_PREDICTION_TIME=3 #max pursuit time
STOP_THRESHOLD=5      # Seperation threshold
DECAY_COEFFICIENT = 3000  # Controls separation force intensity
#wandering
WANDER_RADIUS=  3.0
WANDER_DIST= 5.0
JITTER=1.0
LOOKAHEAD_DISTANCE=5