import argparse

from env.utils import fix_sdl
fix_sdl()

from env.environment import FootballEnv
from env.opponents import make_noop_opponent, make_random_opponent

OPPONENT_FACTORIES = {
    "none": lambda: None,
    "agents": lambda: "agents",
    "noop": make_noop_opponent,
    "random": make_random_opponent,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run a football simulation with random actions")
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--opponent", choices=OPPONENT_FACTORIES.keys(), default="agents")
    parser.add_argument("--ball", choices=["center", "random"], default="center")
    parser.add_argument("--render", choices=["human", "rgb_array", "none"], default="human")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()

    render_mode = None if args.render == "none" else args.render
    opponent = OPPONENT_FACTORIES[args.opponent]()

    env = FootballEnv(
        team_size=args.team_size,
        opponent=opponent,
        render_mode=render_mode,
        ball_placement=args.ball,
    )

    observations, infos = env.reset()
    done = False
    step_count = 0

    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        done = len(env.agents) == 0

        if not done and args.log_interval and step_count % args.log_interval == 0:
            first_agent = list(rewards.keys())[0]
            print(f"Step {step_count:05d} | Reward={rewards[first_agent]:.3f} | score={infos[first_agent]['score']}")

        step_count += 1

    print(f"Episode finished after {step_count} steps")
    env.close()

if __name__ == "__main__":
    main()
