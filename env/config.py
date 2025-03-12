import numpy as np

# stadium attributes
# stadium attributes
# stadium
STADIUM_LENGTH = 105
STADIUM_WIDTH = 68
# center
CENTER_CIRCLE_RADIUS = 9.15
# penalty area
PENALTY_AREA_LENGTH = 16
PENALTY_AREA_WIDTH = 40
# goal area
GOAL_AREA_LENGTH = 6
GOAL_AREA_WIDTH = 20
# goal
GOAL_WIDTH = 12
# penalty spot
PENALTY_SPOT_DISTANCE = 11
PENALTY_SPOT_RADIUS = 0.25
# player attributes
PLAYER_RADIUS = 1
PLAYER_MAX_SPEED = 20
PLAYER_MAX_ROTATION = np.pi
PLAYER_MAX_KICKING_FORCE = 10
PLAYER_MAX_ACCELERATION = 5
PLAYER_DEFAULT_SPEED=10

# ball attributes
BALL_RADIUS = 0.5
BALL_MAX_SPEED = 25

# simulation attributes
DT = 0.1
MAX_GAME_TIME = 1000
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
WANDER_DIST= 5
JITTER=2
LOOKAHEAD_DISTANCE=5

#catch
CATCH_RADIUS=4
TACKLE_RADIUS=6