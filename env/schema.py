from typing import Tuple, Union, List, Dict
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
    goal_depth: float
    penalty_spot_distance: float
    penalty_spot_radius: float
    player_radius: float
    ball_radius: float

class Physics(BaseModel):
    player_max_speed: float
    player_max_acceleration: float
    player_max_angular_speed: float
    player_max_angular_acceleration: float
    player_max_kicking_force: float
    player_friction_factor: float
    ball_friction_factor: float
    angular_friction_factor: float
    ball_max_speed: float
    kick_cooldown: float
    
class Simulation(BaseModel):
    dt: float
    max_game_time: int

class GameState(BaseModel):
    game_time: float
    score: List[int]
    possession: Union[int, None]
    possession_time: List[float]
    last_kicker: Union[int, None]
    last_kick_time: Union[float, None]

class PlayerState(BaseModel):
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    orientation: float
    angular_velocity: float

class TeamState(BaseModel):
    players: List[PlayerState]

class BallState(BaseModel):
    position: Tuple[float, float]
    velocity: Tuple[float, float]

class EnvironmentState(BaseModel):
    team1: TeamState
    team2: TeamState
    ball: BallState
    
class PlayerReward(BaseModel):
    player_id: int
    reward: float

class TeamRewards(BaseModel):
    team1: List[PlayerReward]
    team2: List[PlayerReward]

class StepResult(BaseModel):
    state: EnvironmentState
    rewards: TeamRewards
    done: bool
    info: Dict
    
class PlayerAction(BaseModel):
    acceleration: float
    angular_acceleration: float
    kicking_force: float
    kicking_angle: float

class TeamActions(BaseModel):
    team1: List[PlayerAction]
    team2: List[PlayerAction]