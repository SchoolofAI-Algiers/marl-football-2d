# Path
ASSETS_PATH = './marl/gridenv/assets'

# Colors 
A_COLOR = (238,161,205)  # baby pink 
B_COLOR = (194,195,255)  # lavender purple
GREEN = (0, 159, 63)  
BLUE = (173,216,230)   
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
# WIDTH = GRID_SIZE[1] * CELL_SIZE
# HEIGHT = GRID_SIZE[0] * CELL_SIZE

# Simulation settings
FPS = 30
STEP_DELAY = 500
GOAL_DELAY = 1000