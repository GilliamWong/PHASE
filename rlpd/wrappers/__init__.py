import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from rlpd.wrappers.pixels import make_pixel_env
from rlpd.wrappers.single_precision import SinglePrecision
from rlpd.wrappers.wandb_video import WANDBVideo


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env


def seed_env(env: gym.Env, seed: int):
    """Seeds the env's dynamics and its spaces' samplers; returns the first obs.

    Gymnasium seeds through reset(), so this replaces the old env.seed().
    """
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    return observation
