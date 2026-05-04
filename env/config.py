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
GOAL_DEPTH = 3
# penalty spot
PENALTY_SPOT_DISTANCE = 11
PENALTY_SPOT_RADIUS = 0.25

# player attributes
PLAYER_RADIUS = 1

PLAYER_MAX_SPEED = 7  # Max sprint speed (m/s) — reference for full-size 22-player stadium
PLAYER_MAX_ACCELERATION = 14  # Max acceleration (m/s²) — scales with stadium size
PLAYER_MAX_ANGULAR_SPEED = np.pi  # Max turning speed (rad/s) — does NOT scale
PLAYER_MAX_ANGULAR_ACCELERATION = np.pi / 2  # Max turning acceleration (rad/s²) — does NOT scale
PLAYER_MAX_KICK_IMPULSE = 40  # Max kick impulse (m/s) — scales with stadium size

BALL_MAX_SPEED = 60  # Max ball speed (m/s) — scales with stadium size
KICK_COOLDOWN = 0.5

# run (1-10), pass (20-100), shoot (200-500), action (continuous), 1 - 500, mean, variance, discrete action (kick, pass, shoot) -> [-1, 1] * action_type_max_force

# ball attributes
BALL_RADIUS = 0.5

# simulation attributes
DT = 0.1
MAX_GAME_TIME = 128
FPS = 60

# visual attributes
RENDER_SCALE = 12
PADDING = 10
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
ORANGE = (255, 165, 0)
YELLOW = (255, 215, 0) 
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# physics attributes
PLAYER_FRICTION_FACTOR = 0.93
BALL_FRICTION_FACTOR = 0.96
ANGULAR_FRICTION_FACTOR = 0.95

# Reward
STEP_REWARD = 0 # -0.0001  # Small negative reward for each step to encourage efficiency
KICK_REWARD = 0.1  # Positive reward for kicking — must dominate proximity penalty
PROXIMITY_REWARD_SCALE = 0.01  # Delta-based reward for approaching the ball (kept small to avoid punishing kicks)
BALL_POSITION_REWARD_SCALE = 0.1  # Delta-based reward for ball advancing toward opponent goal
CORNER_AVOIDANCE_REWARD_SCALE = 0.05  # Delta-based reward for moving away from corners