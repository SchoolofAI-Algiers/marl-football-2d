# Path
ASSETS_PATH = './marl/gridenv/assets'

# Colors 
A_COLOR = (255,0,0)  # red
B_COLOR = (0,0,255)  # blue
GREEN = (34, 139, 34)
GRAY = (211, 211, 211)
WHITE = (255, 255, 255)  

# Dimensions settings
GRID_SIZES = {
    "S": (4, 7),
    "M": (6, 9),
    "L": (9, 11)
}
DEFAULT_SIZE = "S"
GRID_SIZE = GRID_SIZES[DEFAULT_SIZE]
CELL_SIZE = 100

# Simulation settings
FPS = 30
MAX_TIMESTEP = 100
STEP_DELAY = 1000
GOAL_DELAY = 1000