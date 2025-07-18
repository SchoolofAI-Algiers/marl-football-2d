import numpy as np
from env.schema import EnvironmentState

def env_to_obs(state: EnvironmentState) -> np.ndarray:
    return np.array(
        [
            *state.team1.players[0].position, *state.team1.players[0].velocity, state.team1.players[0].orientation, state.team1.players[0].angular_velocity,
            *state.ball.position, *state.ball.velocity
        ]
    )
