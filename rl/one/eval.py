import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # Disable audio (wsl)

import torch
import numpy as np

from env.environment import FootballNoOpponentEnv
from env.schema import PlayerAction, TeamActions
from rl.one.utils import env_to_obs
from rl.one.ppo import PPOAgent, PPOConfig

def evaluate_agent(model_path: str, num_episodes: int = 20, render: bool = False):
    config = PPOConfig()
    env = FootballNoOpponentEnv(team_size=1)

    # Get obs and action dimensions
    sample_obs = env.reset()
    obs_dim = env_to_obs(sample_obs).shape[0]
    act_dim = 4

    agent = PPOAgent(obs_dim, act_dim, config)
    agent.policy.load_state_dict(torch.load(model_path, weights_only=True))
    agent.policy.eval()

    device = agent.device
    print(f"Loaded model from {model_path} onto {device}")

    rewards = []
    win, draw, loss = 0, 0, 0

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            obs = env_to_obs(state)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action_t, _, _ = agent.select_action(obs_t)
            action = action_t.squeeze(0).cpu().numpy()

            team_actions = TeamActions(
                team1=[PlayerAction(
                    acceleration=action[0],
                    angular_acceleration=action[1],
                    kicking_force=action[2],
                    kicking_angle=action[3]
                )]
            )

            result = env.step(team_actions)
            state = result.state
            reward = result.rewards.team1[0].reward
            done = result.done

            episode_reward += reward
            if render:
                env.render()

        rewards.append(episode_reward)
        g1, g2 = env.game_state.score
        if g1 > g2:
            win += 1
        elif g1 < g2:
            loss += 1
        else:
            draw += 1

        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Score = {g1}:{g2}")

    print("\n--- Evaluation Summary ---")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Win/Draw/Loss: {win}/{draw}/{loss}")
    print(f"Win rate:  {win / num_episodes:.2%}")
    print(f"Draw rate: {draw / num_episodes:.2%}")
    print(f"Loss rate: {loss / num_episodes:.2%}")

if __name__ == "__main__":
    evaluate_agent("./rl/one/models/ppo_one_player.pt", num_episodes=20, render=True)
