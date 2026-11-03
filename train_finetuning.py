#! /usr/bin/env python
import os
import pickle

import gymnasium as gym
import tqdm
from absl import app, flags

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer, combine_batches
from rlpd.data.robomimic_datasets import RobomimicDataset
from rlpd.envs.robomimic import (
    DEFAULT_HORIZON,
    RobomimicEnv,
    download_dataset,
    read_env_args,
)
from rlpd.evaluation import evaluate
from rlpd.wrappers import seed_env

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string(
    "dataset",
    "lift/ph",
    "Robomimic task/quality, e.g. lift/ph or can/mh. Downloaded on first use; "
    "the env is built from the dataset's recorded env config.",
)
flags.DEFINE_float(
    "offline_ratio",
    0.5,
    "Fraction of each batch drawn from the demos; 0 trains purely online.",
)
flags.DEFINE_float("success_reward", 10.0, "Reward on the (terminal) success step.")
flags.DEFINE_integer("horizon", DEFAULT_HORIZON, "Max steps per episode.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    assert 0.0 <= FLAGS.offline_ratio < 1.0
    use_demos = FLAGS.offline_ratio > 0 or FLAGS.pretrain_steps > 0
    # Sizes are per UTD minibatch first, so both split evenly across the updates.
    offline_per_minibatch = int(FLAGS.batch_size * FLAGS.offline_ratio)
    num_offline = offline_per_minibatch * FLAGS.utd_ratio
    num_online = (FLAGS.batch_size - offline_per_minibatch) * FLAGS.utd_ratio

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    dataset_path = download_dataset(FLAGS.dataset)
    env_args = read_env_args(dataset_path)
    ds = RobomimicDataset(dataset_path, FLAGS.success_reward) if use_demos else None

    env = RobomimicEnv(env_args, FLAGS.horizon, FLAGS.success_reward)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)
    observation = seed_env(env, FLAGS.seed)

    render_mode = "rgb_array" if FLAGS.save_video else None
    eval_env = RobomimicEnv(
        env_args, FLAGS.horizon, FLAGS.success_reward, render_mode=render_mode
    )
    seed_env(eval_env, FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)

    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Only true terminations stop bootstrapping; time-limit truncations do not.
        mask = 0.0 if terminated else 1.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, _ = env.reset()
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)
            wandb.log(
                {"training/success": float(info["is_success"])},
                step=i + FLAGS.pretrain_steps,
            )

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(num_online)
            if num_offline > 0:
                batch = combine_batches(ds.sample(num_offline), batch, FLAGS.utd_ratio)

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")


if __name__ == "__main__":
    app.run(main)
