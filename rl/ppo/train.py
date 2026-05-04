import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["RAY_LOG_LEVEL"] = "fatal"

from env.utils import fix_sdl
fix_sdl()

import uuid
import logging
logging.getLogger("ray").setLevel(logging.CRITICAL)

import argparse
import time

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter

from env.environment import FootballEnv

_env_count = 0

def env_creator(config):
    global _env_count
    render = config.get("render_mode", None) if _env_count == 0 else None
    _env_count += 1
    opponent_cfg = config.get("opponent", "agents")
    opponent = None if opponent_cfg == "none" else opponent_cfg
    env = FootballEnv(
        team_size=config.get("team_size", 1),
        opponent=opponent,
        render_mode=render,
        ball_placement=config.get("ball_placement", "center"),
        max_game_time=config.get("max_game_time", None),
    )
    return ParallelPettingZooEnv(env)

class FootballMetricsCallback(DefaultCallbacks):

    def on_episode_end(self, *, episode, worker=None, env=None, **kwargs):
        info = episode.last_info_for("team0_player0")
        if info and "score" in info:
            score = info["score"]
            t0, t1 = score[0], score[1]
            diff = t0 - t1
            episode.custom_metrics["goals_scored_t0"] = t0
            episode.custom_metrics["goals_conceded_t0"] = t1
            episode.custom_metrics["goal_difference"] = diff
            episode.custom_metrics["win"] = float(diff > 0)
            episode.custom_metrics["draw"] = float(diff == 0)
            episode.custom_metrics["loss"] = float(diff < 0)
            episode.custom_metrics["reward_t0"] = episode.agent_rewards[("team0_player0", "shared")]

        for agent_id in episode.get_agents():
            agent_info = episode.last_info_for(agent_id)
            if agent_info and "reward_components" in agent_info:
                for comp, val in agent_info["reward_components"].items():
                    episode.custom_metrics[f"rc_{comp}_{agent_id}"] = val

def build_config(args):
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="football_selfplay",
            env_config={
                "team_size": 1,
                "opponent": args.opponent,
                "render_mode": "human" if args.render else None,
                "ball_placement": args.ball_placement,
                "max_game_time": args.max_game_time,
            },
        )
        .framework("torch")
        .multi_agent(
            policies={"shared": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
            batch_mode="truncate_episodes",
            sample_timeout_s=600,
        )
        .training(
            gamma=args.gamma,
            lr=args.lr,
            lambda_=args.gae_lambda,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            train_batch_size=args.train_batch_size,
            grad_clip=args.grad_clip,
            vf_clip_param=args.vf_clip_param,
        )
        .callbacks(FootballMetricsCallback)
        .reporting(
            metrics_num_episodes_for_smoothing=100,
            min_sample_timesteps_per_iteration=args.min_sample_timesteps,
        )
        .resources(num_gpus=args.num_gpus)
    )

def parse_args():
    p = argparse.ArgumentParser(description="PPO training for 2D football")

    g = p.add_argument_group("PPO hyperparameters")
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--gae-lambda", type=float, default=0.95)
    g.add_argument("--clip-param", type=float, default=0.2)
    g.add_argument("--entropy-coeff", type=float, default=0.01)
    g.add_argument("--vf-loss-coeff", type=float, default=0.5)
    g.add_argument("--num_epochs", type=int, default=10)
    g.add_argument("--minibatch-size", type=int, default=128)
    g.add_argument("--train-batch-size", type=int, default=4096)
    g.add_argument("--grad-clip", type=float, default=0.5)
    g.add_argument("--vf-clip-param", type=float, default=10.0)
    g.add_argument("--rollout-fragment-length", type=int, default=256)

    g = p.add_argument_group("infrastructure")
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--num-envs-per-worker", type=int, default=1)
    g.add_argument("--num-gpus", type=float, default=1)

    g = p.add_argument_group("run control")
    g.add_argument("--stop-timesteps", type=int, default=12_800_000)
    g.add_argument("--min-sample-timesteps", type=int, default=1280)
    g.add_argument("--checkpoint-freq", type=int, default=100)
    g.add_argument("--log-interval", type=int, default=50)
    g.add_argument("--results-dir", type=str, default="logs/ppo")
    g.add_argument("--experiment-name", type=str, default="test_run_" + uuid.uuid4().hex[:8])

    g = p.add_argument_group("environment")
    g.add_argument("--opponent", choices=["agents", "none"], default="none")
    g.add_argument("--ball-placement", choices=["center", "random"], default="random")
    g.add_argument("--max-game-time", type=float, default=128,
                   help="Episode length in sim seconds (default: T => T x 10 steps)")
    g.add_argument("--render", action="store_true",
                   help="Render env during training (requires --num-workers 0)")

    return p.parse_args()

def main():
    args = parse_args()

    ray.init()
    register_env("football_selfplay", env_creator)

    config = build_config(args)
    algo = config.build_algo()

    log_dir = os.path.join(args.results_dir, args.experiment_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Logging to: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")

    iteration = 0
    start_time = time.time()

    try:
        while True:
            result = algo.train()
            iteration += 1

            timesteps_total = result.get("timesteps_total", 0)
            if timesteps_total >= args.stop_timesteps:
                break

            env_runners = result.get("env_runners", {})
            mean_reward = env_runners.get("episode_reward_mean", float("nan"))
            ep_len = env_runners.get("episode_len_mean", float("nan"))
            sampled = result.get("num_env_steps_sampled", 0)
            trained = result.get("num_env_steps_trained", 0)

            writer.add_scalar("train/episode_reward_mean", mean_reward, iteration)
            writer.add_scalar("train/episode_len_mean", ep_len, iteration)
            writer.add_scalar("train/total_sampled", sampled, iteration)
            writer.add_scalar("train/total_trained", trained, iteration)

            learner = result.get("info", {}).get("learner", {})
            for policy_id, metrics in learner.items():
                if not isinstance(metrics, dict):
                    continue
                ls = metrics.get("learner_stats", {})
                for name in ["policy_loss", "vf_loss", "entropy", "kl",
                             "total_loss", "cur_lr"]:
                    if name in ls:
                        writer.add_scalar(f"learner/{policy_id}/{name}", ls[name], iteration)

            custom = env_runners.get("custom_metrics", {})
            for key in ["goals_scored_t0", "goals_conceded_t0", "goal_difference",
                        "win", "draw", "loss", "reward_t0"]:
                mean_key = f"{key}_mean"
                if mean_key in custom:
                    writer.add_scalar(f"football/{key}", custom[mean_key], iteration)

            for key, val in custom.items():
                if key.startswith("rc_") and key.endswith("_mean"):
                    tag = key[len("rc_"):-len("_mean")]
                    writer.add_scalar(f"reward/{tag}", val, iteration)

            writer.flush()

            if iteration == 1 or iteration % args.log_interval == 0:
                goals_for = custom.get("goals_scored_t0_mean", 0)
                goals_against = custom.get("goals_conceded_t0_mean", 0)
                win_rate = custom.get("win_mean", 0)
                draw_rate = custom.get("draw_mean", 0)
                elapsed = time.time() - start_time
                print(
                    f"[it {iteration:4d}] "
                    f"ts = {timesteps_total:>8d} | "
                    f"reward = {mean_reward:>6.4f} | "
                    f"goals = {goals_for:.2f} - {goals_against:.2f} | "
                    f"win = {win_rate:.1%} | "
                    f"draw = {draw_rate:.1%} | "
                    f"{(elapsed/60):.0f} min"
                )

            if iteration % args.checkpoint_freq == 0:
                path = algo.save(checkpoint_dir=checkpoint_dir)
                print(f"  -> checkpoint: {path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    path = algo.save(checkpoint_dir=checkpoint_dir)
    print(f"Final checkpoint: {path}")

    writer.close()
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
