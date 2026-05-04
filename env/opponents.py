from typing import Callable, Dict, Optional

import numpy as np


def make_random_opponent(seed: Optional[int] = None) -> Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Returns an opponent policy that produces random actions.

    The callable has the same signature as a PettingZoo policy:
      (observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]
    """
    rng = np.random.default_rng(seed)

    def policy(observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            agent: rng.uniform(-1, 1, size=(4,)).astype(np.float32)
            for agent in observations
        }

    return policy


def make_noop_opponent() -> Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Returns an opponent policy that produces zero actions (stationary players)."""

    def policy(observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            agent: np.zeros(4, dtype=np.float32)
            for agent in observations
        }

    return policy
