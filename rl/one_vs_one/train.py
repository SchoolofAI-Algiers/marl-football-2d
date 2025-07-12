import os
import torch

from env.environment import FootballEnv
from env.schema import PlayerAction, TeamActions, StepResult

from rl.one_vs_one.rollout import RolloutBuffer
from rl.one_vs_one.ppo import PPOAgent, PPOConfig
from rl.one_vs_one.utils import mirror_action, env_to_obs

def train_selfplay_mirrored(total_timesteps: int = 1_280_000):
    env = FootballEnv(team_size=1)
    sample_obs = env.reset()
    obs_dim = env_to_obs(sample_obs).shape[0]
    act_dim = 4

    config = PPOConfig()
    agent = PPOAgent(obs_dim, act_dim, config)
    opponent = PPOAgent(obs_dim, act_dim, config)
    opponent.policy.load_state_dict(agent.policy.state_dict())

    buffer = RolloutBuffer()
    timestep = 0
    episode_count = 0

    while timestep < total_timesteps:
        state = env.reset()
        episode_count += 1

        for step in range(config.rollout_length):
            obs1 = env_to_obs(state, team=0)
            obs2 = env_to_obs(state, team=1)

            # Convert obs to tensors and send to correct device
            obs1_t = torch.FloatTensor(obs1).unsqueeze(0).to(agent.device)
            obs2_t = torch.FloatTensor(obs2).unsqueeze(0).to(opponent.device)

            # Agent (team1)
            with torch.no_grad():
                action1_t, logp1_t, val1_t = agent.select_action(obs1_t)
            action1 = action1_t.squeeze(0).cpu().numpy()
            logp1 = logp1_t.item()
            val1 = val1_t.item()

            # Opponent (team2)
            with torch.no_grad():
                action2_t, _, _ = opponent.select_action(obs2_t)
            action2 = action2_t.squeeze(0).cpu().numpy()

            # Create actions
            team_actions = TeamActions(
                team1=[PlayerAction(
                    acceleration=action1[0],
                    angular_acceleration=action1[1],
                    kicking_force=action1[2],
                    kicking_angle=action1[3]
                )],
                team2=[mirror_action(PlayerAction(
                    acceleration=action2[0],
                    angular_acceleration=action2[1],
                    kicking_force=action2[2],
                    kicking_angle=action2[3]
                ))]
            )

            result: StepResult = env.step(team_actions)
            next_state = result.state
            reward = result.rewards.team1[0].reward
            done = result.done

            buffer.obs.append(obs1)
            buffer.actions.append(action1)
            buffer.log_probs.append(logp1)
            buffer.values.append(val1)
            buffer.rewards.append(reward)
            buffer.dones.append(done)

            state = next_state
            timestep += 1

            if done:
                break

        # Compute final value
        if done:
            last_val = 0
        else:
            obs_final = env_to_obs(state, team=0)
            obs_final_t = torch.FloatTensor(obs_final).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                _, _, last_val_t = agent.select_action(obs_final_t)
            last_val = last_val_t.item()

        buffer.values.append(last_val)
        buffer.returns = agent.compute_gae(buffer.rewards, buffer.values, buffer.dones)
        buffer.values = buffer.values[:-1]

        agent.update(buffer.__dict__)
        buffer.clear()

        if episode_count % 5 == 0:
            opponent.policy.load_state_dict(agent.policy.state_dict())
            print(f"Episode {episode_count}, Timestep {timestep}: Updated opponent policy")

        if episode_count % 100 == 0:
            print(f"Episode {episode_count}, Timestep {timestep}")

    os.makedirs("./models", exist_ok=True)
    torch.save(agent.policy.state_dict(), os.path.join("./models", "ppo_selfplay_mirrored.pt"))
    print("Training completed and model saved!")

if __name__ == "__main__":
    train_selfplay_mirrored()
