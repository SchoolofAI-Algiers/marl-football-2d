import os
os.environ["SDL_AUDIODRIVER"] = "dummy" # Disable audio (wsl)
import torch
import mlflow
import random
import time

from env.environment import FootballNoOpponentEnv
from env.schema import PlayerAction, TeamActions, StepResult

from rl.one.rollout import RolloutBuffer
from rl.one.ppo import PPOAgent, PPOConfig, PPOMetrics
from rl.one.utils import env_to_obs

def train_selfplay_mirrored(total_timesteps: int = 5_120_000):
    config = PPOConfig()
    total_episodes = total_timesteps // config.rollout_length 

    os.makedirs("./rl/one/models", exist_ok=True)
    os.makedirs("./rl/one/models/checkpoints", exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    
    mlflow.set_experiment("ppo_one_player_experiments")
    run_name = f"PPO-One-Player-{random.randint(10, 99)}"

    start_time = time.time()

    with mlflow.start_run(run_name=run_name):

        # Log hyperparameters once
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

        config = PPOConfig()
        agent = PPOAgent(obs_dim, act_dim, config)

        buffer = RolloutBuffer()
        timestep = 0
        episode_count = 0
        wins_count = 0
        draws_count = 0
        losses_count = 0

        while timestep < total_timesteps:
            state = env.reset()
            episode_count += 1

            episode_return = 0

            for step in range(config.rollout_length):
                obs1 = env_to_obs(state)

                obs1_t = torch.FloatTensor(obs1).unsqueeze(0).to(agent.device)

                with torch.no_grad():
                    action1_t, logp1_t, val1_t = agent.select_action(obs1_t)
                action1 = action1_t.squeeze(0).cpu().numpy()
                logp1 = logp1_t.item()
                val1 = val1_t.item()


                team_actions = TeamActions(
                    team1=[PlayerAction(
                        acceleration=action1[0],
                        angular_acceleration=action1[1],
                        kicking_force=action1[2],
                        kicking_angle=action1[3]
                    )],
                )

                result: StepResult = env.step(team_actions)
                # print(result)
                next_state = result.state
                reward = result.rewards.team1[0].reward
                done = result.done
                
                env.render()

                buffer.obs.append(obs1)
                buffer.actions.append(action1)
                buffer.log_probs.append(logp1)
                buffer.values.append(val1)
                buffer.rewards.append(reward)
                buffer.dones.append(done)

                episode_return += reward
                state = next_state
                timestep += 1

                if done:
                    goals_team1 = env.game_state.score[0]
                    goals_team2 = env.game_state.score[1]
                    goal_difference = goals_team1 - goals_team2
                    
                    mlflow.log_metric("goals_scored", goals_team1, step=episode_count)
                    mlflow.log_metric("goals_conceded", goals_team2, step=episode_count)
                    mlflow.log_metric("goals_difference", goal_difference, step=episode_count)
                    
                    if goal_difference > 0:
                        wins_count += 1
                        mlflow.log_metric("wins", wins_count, step=episode_count)
                    elif goal_difference < 0:
                        losses_count += 1
                        mlflow.log_metric("losses", losses_count, step=episode_count)
                    else:
                        draws_count += 1
                        mlflow.log_metric("draws", draws_count, step=episode_count)
                        
                    mlflow.log_metric("win_ratio", wins_count / episode_count, step=episode_count)
                    mlflow.log_metric("draw_ratio", draws_count / episode_count, step=episode_count)
                    mlflow.log_metric("loss_ratio", losses_count / episode_count, step=episode_count)

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

            metrics: PPOMetrics = agent.update(buffer.__dict__)
            buffer.clear()

            # Log to MLflow
            mlflow.log_metric("policy_loss", metrics.policy_loss, step=episode_count)
            mlflow.log_metric("value_loss", metrics.value_loss, step=episode_count)
            mlflow.log_metric("entropy", metrics.entropy, step=episode_count)
            mlflow.log_metric("loss", metrics.loss, step=episode_count)
            mlflow.log_metric("mean_return", metrics.mean_return, step=episode_count)

            # if episode_count % 50 == 0:
            #     opponent.policy.load_state_dict(agent.policy.state_dict())
            #     print(f"Episode {episode_count} / {total_episodes}: Updated opponent policy")
                
            if episode_count % 50 == 0:
                torch.save({
                    "agent_policy": agent.policy.state_dict(),
                }, f"./rl/one/models/checkpoints/episode_{episode_count}.pth")
                print(f"Checkpoint saved at Episode {episode_count}")

            if episode_count % 5 == 0:
                print(f"Episode {episode_count} / {total_episodes}, Time: {(time.time() - start_time)/60:.2f} minutes")

        torch.save(agent.policy.state_dict(), os.path.join("./rl/one/models", "ppo_selfplay_mirrored.pt"))
        print("Training completed and model saved!")

if __name__ == "__main__":
    train_selfplay_mirrored()
