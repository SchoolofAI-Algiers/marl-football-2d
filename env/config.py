import numpy as np

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
PLAYER_RADIUS = 0.75

PLAYER_MAX_SPEED = 5  # Max sprint speed (m/s)
PLAYER_MAX_ACCELERATION = 5  # Max acceleration (m/s²)
PLAYER_MAX_ANGULAR_SPEED = np.pi  # Max turning speed (rad/s)
PLAYER_MAX_ANGULAR_ACCELERATION = np.pi  # Max turning acceleration (rad/s²)
PLAYER_MAX_KICKING_FORCE = 2  # Max kicking acceleration (m/s²)

# run (1-10), pass (20-100), shoot (200-500), action (continuous), 1 - 500, mean, variance, discrete action (kick, pass, shoot) -> [-1, 1] * action_type_max_force

# ball attributes
BALL_RADIUS = 0.4

# simulation attributes
DT = 0.1
MAX_GAME_TIME = 1000
FPS = 60

# visual attributes
RENDER_SCALE = 12
PADDING = 10
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
ORANGE = (255, 165, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# physics attributes
FRICTION_FACTOR = 0.9
ANGULAR_FRICTION_FACTOR = 0.9
AIR_RESISTANCE = 0.98