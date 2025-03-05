from config import (
    STADIUM_LENGTH, PLAYERS_TO_LENGTH_RATIO, LENGTH_TO_WIDTH_RATIO,
    PLAYER_RADIUS, BALL_RADIUS, PLAYER_MAX_SPEED, PLAYER_MAX_ROTATION,
    DT, MAX_GAME_TIME,
    FRICTION_FACTOR, ANGULAR_FRICTION_FACTOR, AIR_RESISTANCE
    
)
from models import Dimensions, Physics, Simulation

def get_dimensions(num_players: int) -> Dimensions:
    return Dimensions(
        stadium_length=int(num_players * PLAYERS_TO_LENGTH_RATIO * STADIUM_LENGTH),
        stadium_width=int(num_players * PLAYERS_TO_LENGTH_RATIO * STADIUM_LENGTH * LENGTH_TO_WIDTH_RATIO),
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
