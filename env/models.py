from typing import Tuple, Union, List
from pydantic import BaseModel

class Dimensions(BaseModel):
    stadium_length: int
    stadium_width: int
    center_circle_radius: float
    penalty_area_length: float
    penalty_area_width: float
    goal_area_length: float
    goal_area_width: float
    goal_width: float
    penalty_spot_distance: float
    penalty_spot_radius: float
    player_radius: float
    ball_radius: float

class Physics(BaseModel):
    player_max_speed: float
    player_max_angular_speed: float
    friction_factor: float
    angular_friction_factor: float
    air_resistance: float
    
class Simulation(BaseModel):
    dt: float
    max_game_time: int

class GameState(BaseModel):
    game_time: int
    score: List[int]
    possession: Union[int, None]
    possession_time: List[int]
