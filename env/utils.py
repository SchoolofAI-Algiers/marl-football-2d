import math

from env.config import (
    STADIUM_LENGTH, STADIUM_WIDTH, CENTER_CIRCLE_RADIUS, PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH, GOAL_AREA_LENGTH, GOAL_AREA_WIDTH, GOAL_WIDTH, PENALTY_SPOT_DISTANCE, PENALTY_SPOT_RADIUS,
    PLAYER_RADIUS, BALL_RADIUS, PLAYER_MAX_SPEED, PLAYER_MAX_ROTATION,
    DT, MAX_GAME_TIME,
    FRICTION_FACTOR, ANGULAR_FRICTION_FACTOR, AIR_RESISTANCE
)
from env.models import Dimensions, Physics, Simulation

def scale_dimension(value: float, num_players: int, exponent: float = 1.0) -> float:
    """Scale a dimension with a logarithmic function and a tunable exponent."""
    return int(value * (math.log1p(num_players) / math.log1p(22)) ** exponent)

def get_dimensions(num_players: int) -> Dimensions:
    return Dimensions(
        stadium_length=scale_dimension(STADIUM_LENGTH, num_players, exponent=1.0),
        stadium_width=scale_dimension(STADIUM_WIDTH, num_players, exponent=1.0),
        center_circle_radius=scale_dimension(CENTER_CIRCLE_RADIUS, num_players, exponent=0.9),
        penalty_area_length=scale_dimension(PENALTY_AREA_LENGTH, num_players, exponent=0.7),
        penalty_area_width=scale_dimension(PENALTY_AREA_WIDTH, num_players, exponent=0.7),
        goal_area_length=scale_dimension(GOAL_AREA_LENGTH, num_players, exponent=0.5),
        goal_area_width=scale_dimension(GOAL_AREA_WIDTH, num_players, exponent=0.5),
        goal_width=scale_dimension(GOAL_WIDTH, num_players, exponent=0.3),
        penalty_spot_distance=scale_dimension(PENALTY_SPOT_DISTANCE, num_players, exponent=0.8),
        penalty_spot_radius=PENALTY_SPOT_RADIUS,
        player_radius=PLAYER_RADIUS,
        ball_radius=BALL_RADIUS
    )
    
def get_physics() -> Physics:
    return Physics(
        player_max_speed=PLAYER_MAX_SPEED,
        player_max_rotation=PLAYER_MAX_ROTATION,
        friction_factor=FRICTION_FACTOR,
        angular_friction_factor=ANGULAR_FRICTION_FACTOR,
        air_resistance=AIR_RESISTANCE
    )
    
def get_simulation() -> Simulation:
    return Simulation(
        dt=DT,
        max_game_time=MAX_GAME_TIME
    )
