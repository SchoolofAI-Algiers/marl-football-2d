from typing import Tuple, Union, List
from pydantic import BaseModel
import numpy as np

class Dimensions(BaseModel):
    stadium_length: int
    stadium_width: int
    player_radius: float
    ball_radius: float

class Physics(BaseModel):
    player_max_speed: float
    player_max_rotation: float
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


#the input data structure used by the steering bahviors
class MovementInput(BaseModel):
    #radius :float
    position : List[float]
    velocity : List[float]
    rotation : float
    angular_velocity : float

#data structure to store the output of the movement
class MovementOutput(BaseModel):
    linear: List[float] # type: ignore
    angular: float  # Angular acceleration