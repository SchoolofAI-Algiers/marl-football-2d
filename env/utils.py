import os
import math
import random
from typing import Tuple

import numpy as np

from env.config import (
    STADIUM_LENGTH, STADIUM_WIDTH, CENTER_CIRCLE_RADIUS, PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH, GOAL_AREA_LENGTH, GOAL_AREA_WIDTH, GOAL_WIDTH, GOAL_DEPTH, PENALTY_SPOT_DISTANCE, PENALTY_SPOT_RADIUS,
    PLAYER_RADIUS, BALL_RADIUS, PLAYER_MAX_SPEED, PLAYER_MAX_ACCELERATION, PLAYER_MAX_ANGULAR_SPEED, PLAYER_MAX_ANGULAR_ACCELERATION, PLAYER_MAX_KICK_IMPULSE, BALL_MAX_SPEED,
    DT, MAX_GAME_TIME,
    PLAYER_FRICTION_FACTOR, BALL_FRICTION_FACTOR, ANGULAR_FRICTION_FACTOR, KICK_COOLDOWN
)
from env.schema import Dimensions, Physics, Simulation

# System

def fix_sdl():
    import os
    os.environ["SDL_AUDIODRIVER"] = "dummy"

# Physics

def scale_dimension(value: float, num_players: int, exponent: float = 1.0) -> float:
    """Scale a dimension with a logarithmic function and a tunable exponent."""
    if num_players >= 22:
        return value
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
        goal_depth=GOAL_DEPTH,
        penalty_spot_distance=scale_dimension(PENALTY_SPOT_DISTANCE, num_players, exponent=0.8),
        penalty_spot_radius=PENALTY_SPOT_RADIUS,
        player_radius=PLAYER_RADIUS,
        ball_radius=BALL_RADIUS
    )
    
def get_physics(num_players: int) -> Physics:
    scale = get_dimensions(num_players).stadium_length / STADIUM_LENGTH

    return Physics(
        player_max_speed=PLAYER_MAX_SPEED * scale,
        player_max_acceleration=PLAYER_MAX_ACCELERATION * scale,
        player_max_angular_speed=PLAYER_MAX_ANGULAR_SPEED,
        player_max_angular_acceleration=PLAYER_MAX_ANGULAR_ACCELERATION,
        player_max_kick_impulse=PLAYER_MAX_KICK_IMPULSE * scale,
        player_friction_factor=PLAYER_FRICTION_FACTOR,
        ball_friction_factor=BALL_FRICTION_FACTOR,
        angular_friction_factor=ANGULAR_FRICTION_FACTOR,
        ball_max_speed=BALL_MAX_SPEED * scale,
        kick_cooldown=KICK_COOLDOWN,
    )
    
def get_simulation() -> Simulation:
    return Simulation(
        dt=DT,
        max_game_time=MAX_GAME_TIME
    )

def random_float():
    """Generate a random float between 0 and 1."""
    return random.random()

# Mirroring (for self-play)

def mirror_position(pos: Tuple[float, float]) -> Tuple[float, float]:
    """Mirror position both horizontally and vertically (180° rotation).
    Assumes positions are normalized to [-1, 1]."""
    return (-pos[0], -pos[1])

def mirror_velocity(vel: Tuple[float, float]) -> Tuple[float, float]:
    """Mirror velocity both horizontally and vertically (180° rotation)."""
    return (-vel[0], -vel[1])

def mirror_orientation(orientation: float) -> float:
    """Mirror orientation (180° rotation). Input/output in normalized [-1, 1] range."""
    angle_rad = orientation * np.pi
    mirrored_angle = angle_rad + np.pi
    while mirrored_angle > np.pi:
        mirrored_angle -= 2 * np.pi
    while mirrored_angle < -np.pi:
        mirrored_angle += 2 * np.pi
    return mirrored_angle / np.pi

def mirror_angular_velocity(angular_vel: float) -> float:
    """Mirror angular velocity (flip sign)."""
    return -angular_vel

def mirror_action(action: np.ndarray) -> np.ndarray:
    """Mirror a 4-element action array [acc, ang_acc, kick_force, kick_angle].

    - acceleration: unchanged (forward/backward relative to player)
    - angular_acceleration: flips sign (left becomes right)
    - kicking_force: unchanged
    - kicking_angle: flips sign (left becomes right relative to player)
    """
    return np.array([
        action[0],
        -action[1],
        action[2],
        -action[3],
    ], dtype=action.dtype)