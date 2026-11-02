import gymnasium as gym
import numpy as np
import shimmy

from rlpd.wrappers.frame_stack import FrameStack
from rlpd.wrappers.repeat_action import RepeatAction

# Registers the "dm_control/<domain>-<task>-v0" environment ids.
gym.register_envs(shimmy)

PIXEL_KEY = "pixels"


class PixelObservation(gym.ObservationWrapper):
    """Replaces observations with {"pixels": env.render()}.

    Gymnasium 1.x removed PixelObservationWrapper; this keeps its pixels_only
    behavior. The env must have been made with render_mode="rgb_array" at the
    given image size.
    """

    def __init__(self, env: gym.Env, image_size: int):
        super().__init__(env)
        pixel_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict({PIXEL_KEY: pixel_space})

    def observation(self, observation):
        return {PIXEL_KEY: self.env.render()}


def make_pixel_env(
    env_name: str,
    action_repeat: int,
    image_size: int = 84,
    num_stack: int = 3,
    camera_id: int = 0,
):
    """Makes a pixel-observation dm_control env, e.g. "dm_control/cheetah-run-v0".

    Returns the env and the pixel observation keys.
    """
    env = gym.make(
        env_name,
        render_mode="rgb_array",
        render_kwargs={"height": image_size, "width": image_size, "camera_id": camera_id},
    )
    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    env = gym.wrappers.RescaleAction(env, -1, 1)
    env = PixelObservation(env, image_size)

    if num_stack is not None:
        env = FrameStack(env, num_stack=num_stack)

    env = gym.wrappers.ClipAction(env)

    return env, (PIXEL_KEY,)
