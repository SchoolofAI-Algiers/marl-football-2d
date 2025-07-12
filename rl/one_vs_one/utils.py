import numpy as np
from typing import Tuple
from env.schema import PlayerState, TeamState, BallState, PlayerAction, EnvironmentState

def mirror_position(pos: Tuple[float, float], field_length: float = 1.0, field_width: float = 1.0) -> Tuple[float, float]:
    """Mirror position both horizontally and vertically (180° rotation)"""
    return (field_length - pos[0], field_width - pos[1])

def mirror_velocity(vel: Tuple[float, float]) -> Tuple[float, float]:
    """Mirror velocity both horizontally and vertically (180° rotation)"""
    return (-vel[0], -vel[1])

def mirror_orientation(orientation: float) -> float:
    """Mirror orientation (180° rotation)"""
    # Convert from [-1, 1] normalized range back to [-π, π]
    angle_rad = orientation * np.pi
    
    # Add π to rotate 180 degrees
    mirrored_angle = angle_rad + np.pi
    
    # Normalize back to [-π, π]
    while mirrored_angle > np.pi:
        mirrored_angle -= 2 * np.pi
    while mirrored_angle < -np.pi:
        mirrored_angle += 2 * np.pi
    
    # Convert back to [-1, 1] normalized range
    return mirrored_angle / np.pi

def mirror_angular_velocity(angular_vel: float) -> float:
    """Mirror angular velocity (flip sign)"""
    return -angular_vel

def mirror_player_state(player_state: PlayerState) -> PlayerState:
    """Mirror a single player's state"""
    return PlayerState(
        position=mirror_position(player_state.position),
        velocity=mirror_velocity(player_state.velocity),
        orientation=mirror_orientation(player_state.orientation),
        angular_velocity=mirror_angular_velocity(player_state.angular_velocity)
    )

def mirror_team_state(team_state: TeamState) -> TeamState:
    """Mirror all players in a team"""
    return TeamState(
        players=[mirror_player_state(player) for player in team_state.players]
    )

def mirror_ball_state(ball_state: BallState) -> BallState:
    """Mirror ball state"""
    return BallState(
        position=mirror_position(ball_state.position),
        velocity=mirror_velocity(ball_state.velocity)
    )

def mirror_environment_state(state: EnvironmentState) -> EnvironmentState:
    """
    Mirror the entire environment state from team2's perspective.
    This swaps the teams and mirrors all positions/orientations.
    """
    return EnvironmentState(
        team1=mirror_team_state(state.team2),  # team2 becomes team1
        team2=mirror_team_state(state.team1),  # team1 becomes team2
        ball=mirror_ball_state(state.ball)
    )

def mirror_action(action: PlayerAction) -> PlayerAction:
    """
    Mirror a player action for the opposite team.
    - acceleration stays the same (forward/backward relative to player)
    - angular_acceleration flips sign (left becomes right)
    - kicking_force stays the same
    - kicking_angle flips sign (left becomes right relative to player)
    """
    return PlayerAction(
        acceleration=action.acceleration,
        angular_acceleration=-action.angular_acceleration,
        kicking_force=action.kicking_force,
        kicking_angle=-action.kicking_angle
    )

def env_to_obs(state: EnvironmentState, team: int = 0) -> np.ndarray:
    """
    Convert environment state to observation from a specific team's perspective.
    If team=1, mirror the state so team2 sees itself as team1.
    """
    if team == 1:
        state = mirror_environment_state(state)
    
    return np.array(
        [
            *state.team1.players[0].position, *state.team1.players[0].velocity, state.team1.players[0].orientation, state.team1.players[0].angular_velocity, 1.0,
            *state.team2.players[0].position, *state.team2.players[0].velocity, state.team2.players[0].orientation, state.team2.players[0].angular_velocity, 0.0,
            *state.ball.position, *state.ball.velocity
        ]
    )
    