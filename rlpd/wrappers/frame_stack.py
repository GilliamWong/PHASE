import collections
import copy

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stacking_key: str = "pixels"):
        super().__init__(env)
        self._num_stack = num_stack
        self._stacking_key = stacking_key

        assert stacking_key in self.env.observation_space.spaces
        pixel_obs_spaces = self.env.observation_space.spaces[stacking_key]

        self._env_dim = pixel_obs_spaces.shape[-1]

        low = np.repeat(pixel_obs_spaces.low[..., np.newaxis], num_stack, axis=-1)
        high = np.repeat(pixel_obs_spaces.high[..., np.newaxis], num_stack, axis=-1)
        new_pixel_obs_spaces = Box(low=low, high=high, dtype=pixel_obs_spaces.dtype)
        # Copy so the wrapped env keeps its own (unstacked) space.
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.observation_space.spaces[stacking_key] = new_pixel_obs_spaces

        self._frames = collections.deque(maxlen=num_stack)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for i in range(self._num_stack):
            self._frames.append(obs[self._stacking_key])
        obs[self._stacking_key] = self.frames
        return obs, info

    @property
    def frames(self):
        return np.stack(self._frames, axis=-1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs[self._stacking_key])
        obs[self._stacking_key] = self.frames
        return obs, reward, terminated, truncated, info
