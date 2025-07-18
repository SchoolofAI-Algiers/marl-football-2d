import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # Disable audio (wsl)

import torch
import mlflow
import random
import time

from env.environment import FootballNoOpponentEnv
from env.schema import PlayerAction, TeamActions, StepResult

from rl.one.rollout import RolloutBuffer
from rl.one.ppo import PPOAgent, PPOConfig, PPOMetrics
from rl.one.utils import env_to_obs

def train_one_player(total_timesteps: int = 5_120_000):
    config = PPOConfig()
    total_episodes = total_timesteps // config.rollout_length

    os.makedirs("./rl/one/models", exist_ok=True)
    os.makedirs("./rl/one/models/checkpoints", exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ppo_one_player_experiments")
    run_name = f"PPO-One-Player-{random.randint(10, 99)}"

    start_time = time.time()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "rollout_length": config.rollout_length,
            "epochs": config.epochs,
            "mini_batch_size": config.mini_batch_size,
            "eps_clip": config.eps_clip,
            "vf_coef": config.vf_coef,
            "ent_coef": config.ent_coef,
            "max_grad_norm": config.max_grad_norm,
            "learning_rate": config.lr,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda
        })

        env = FootballNoOpponentEnv(team_size=1)
        sample_obs = env.reset()
        obs_dim = env_to_obs(sample_obs).shape[0]
        act_dim = 4

        agent = PPOAgent(obs_dim, act_dim, config)
        buffer = RolloutBuffer()

        timestep = 0
        episode_count = 0
        wins_count = 0
        draws_count = 0
        losses_count = 0
        
        goals_for = 0
        goals_against = 0

        while timestep < total_timesteps:
            buffer.clear()
            steps_collected = 0
            episode_returns = []

            while steps_collected < config.rollout_length:
                state = env.reset()
                episode_return = 0
                done = False

                while not done and steps_collected < config.rollout_length:
                    obs = env_to_obs(state)
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)

                    with torch.no_grad():
                        action_t, logp_t, val_t = agent.select_action(obs_t)

                    action = action_t.squeeze(0).cpu().numpy()
                    logp = logp_t.item()
                    val = val_t.item()

                    team_actions = TeamActions(
                        team1=[PlayerAction(
                            acceleration=action[0],
                            angular_acceleration=action[1],
                            kicking_force=action[2],
                            kicking_angle=action[3]
                        )]
                    )

                    result: StepResult = env.step(team_actions)
                    next_state = result.state
                    reward = result.rewards.team1[0].reward
                    done = result.done

                    env.render()

                    buffer.obs.append(obs)
                    buffer.actions.append(action)
                    buffer.log_probs.append(logp)
                    buffer.values.append(val)
                    buffer.rewards.append(reward)
                    buffer.dones.append(done)

                    episode_return += reward
                    state = next_state
                    timestep += 1
                    steps_collected += 1

                    if done:
                        episode_returns.append(episode_return)
                        episode_count += 1
                        goals_team1 = env.game_state.score[0]
                        goals_team2 = env.game_state.score[1]
                        goal_diff = goals_team1 - goals_team2
                        
                        goals_for += goals_team1
                        goals_against += goals_team2

                        mlflow.log_metric("2_1_goals_scored", goals_team1, step=episode_count)
                        mlflow.log_metric("2_2_goals_conceded", goals_team2, step=episode_count)
                        mlflow.log_metric("2_3_goals_difference", goal_diff, step=episode_count)
                        
                        mlflow.log_metric("2_4_goals_for", goals_for, step=episode_count)
                        mlflow.log_metric("2_5_goals_against", goals_against, step=episode_count)

                        if goal_diff > 0:
                            wins_count += 1
                            mlflow.log_metric("3_1_wins", wins_count, step=episode_count)
                        elif goal_diff < 0:
                            losses_count += 1
                            mlflow.log_metric("3_2_losses", losses_count, step=episode_count)
                        else:
                            draws_count += 1
                            mlflow.log_metric("3_3_draws", draws_count, step=episode_count)

                        mlflow.log_metric("1_1_win_ratio", wins_count / episode_count, step=episode_count)
                        mlflow.log_metric("1_2_draw_ratio", draws_count / episode_count, step=episode_count)
                        mlflow.log_metric("1_3_loss_ratio", losses_count / episode_count, step=episode_count)

            # Bootstrapping for GAE
            if done:
                last_val = 0
            else:
                obs_final = env_to_obs(state)
                obs_final_t = torch.FloatTensor(obs_final).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    _, _, last_val_t = agent.select_action(obs_final_t)
                last_val = last_val_t.item()

            buffer.values.append(last_val)
            buffer.returns = agent.compute_gae(buffer.rewards, buffer.values, buffer.dones)
            buffer.values = buffer.values[:-1]

            mean_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0
            mlflow.log_metric("4_1_mean_return", mean_return, step=episode_count)
            
            metrics: PPOMetrics = agent.update(buffer.__dict__)

            mlflow.log_metric("4_2_loss", metrics.loss, step=episode_count)
            mlflow.log_metric("4_3_entropy", metrics.entropy, step=episode_count)
            mlflow.log_metric("4_4_policy_loss", metrics.policy_loss, step=episode_count)
            mlflow.log_metric("4_5_value_loss", metrics.value_loss, step=episode_count)

            if episode_count % 50 == 0:
                torch.save({
                    "agent_policy": agent.policy.state_dict(),
                }, f"./rl/one/models/checkpoints/episode_{episode_count}.pth")
                print(f"Checkpoint saved at Episode {episode_count}")

            if episode_count % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"Episode {episode_count} / {total_episodes} | Timestep {timestep} | Time: {elapsed:.2f} minutes")

        torch.save(agent.policy.state_dict(), "./rl/one/models/ppo_one_player.pt")
        print("Training completed and model saved!")

if __name__ == "__main__":
    train_one_player()
