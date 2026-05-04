# marl-football-2d

A 2D football simulator for multi-agent reinforcement learning research. The environment follows the [PettingZoo](https://pettingzoo.farama.org/) Parallel API and ships with [RLlib](https://docs.ray.io/en/latest/rllib/) training scripts for PPO and SAC.

## Components

- [env/](env/) — the continuous-physics football environment.
  - [environment.py](env/environment.py): `FootballEnv`, the PettingZoo `ParallelEnv`. Supports single-team rollouts, self-play (`opponent="agents"`), and play against a fixed policy (`opponent=callable`).
  - [engine.py](env/engine.py): rigid-body player and ball dynamics with collisions, friction, and goal handling.
  - [config.py](env/config.py): stadium dimensions, physics constants, and reward weights.
  - [opponents.py](env/opponents.py): ready-made opponent policies (`make_random_opponent`, `make_noop_opponent`).
  - [schema.py](env/schema.py), [simulation.py](env/simulation.py), [utils.py](env/utils.py): pydantic models, scaling helpers, and SDL/rendering utilities.

- [rl/](rl/) — training entry points built on Ray RLlib.
  - [rl/ppo/](rl/ppo/): PPO training. Run with `./rl/ppo/train.sh <experiment-name>` or `python -m rl.ppo.train`.
  - [rl/sac/](rl/sac/): SAC training. Run with `./rl/sac/train.sh <experiment-name>` or `python -m rl.sac.train`.
  - Both scripts share a single policy across teammates, log to TensorBoard under `logs/<algo>/<run>`, and accept the same env flags (`--opponent`, `--ball-placement`, `--max-game-time`, `--render`).

- [playground/](playground/) — Jupyter notebooks for manual scripting, physics tuning, and stadium-size scaling experiments.

- [gridenv/](gridenv/) — earlier grid-based prototype, kept for reference.

- [marl/](marl/), [gameai/](gameai/) — placeholders for upcoming multi-agent algorithms and scripted bots.

- [tests/](tests/) — sanity scripts (e.g. CUDA availability check).

## Environment at a glance

- **Agents.** `team{0,1}_player{i}` (PettingZoo IDs). Set `team_size` for `NvN` matches.
- **Observation.** `Box(-1, 1)` vector per agent: own state, teammates, opponents, ball, ego-frame ball offset, kick-availability flag, and time remaining. Team-1 observations are mirrored so both teams attack rightward.
- **Action.** `Box` with `[acceleration, angular_acceleration, kick_force, kick_angle]`, all in `[-1, 1]` (acceleration and kick force in `[0, 1]`), denormalized internally using the limits in [env/config.py](env/config.py).
- **Rewards.** Step penalty, kick bonus, ball-proximity shaping, ball-advance shaping, corner-avoidance shaping, and goal reward. Per-component breakdowns are returned in `info["reward_components"]`.

## Quick start

Install everything as described in [SETUP.md](SETUP.md), then:

```python
from env import FootballEnv, make_random_opponent

env = FootballEnv(team_size=2, opponent=make_random_opponent(), render_mode="human")
obs, info = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, info = env.step(actions)
env.close()
```

To launch a training run:

```sh
./rl/ppo/train.sh baseline --opponent agents --num-workers 4
tensorboard --logdir logs/ppo
```

## Setup and contributing

- Setup instructions: [SETUP.md](SETUP.md).
- Contribution workflow and branching rules: [CONTRIBUTION.md](CONTRIBUTION.md).
