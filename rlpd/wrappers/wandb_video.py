from typing import Optional

import gymnasium as gym
import numpy as np

import wandb


class WANDBVideo(gym.Wrapper):
    """Logs one video per episode to wandb.

    Uses the "pixels" observation when present; otherwise the env must have been
    made with render_mode="rgb_array".
    """

    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
        max_videos: Optional[int] = None,
    ):
        super().__init__(env)

        self._name = name
        self._max_videos = max_videos
        self._video = []

    def _add_frame(self, obs):
        if self._max_videos is not None and self._max_videos <= 0:
            return
        if isinstance(obs, dict) and "pixels" in obs:
            if obs["pixels"].ndim == 4:
                self._video.append(obs["pixels"][..., -1])
            else:
                self._video.append(obs["pixels"])
        else:
            self._video.append(self.render())

    def reset(self, **kwargs):
        self._video.clear()
        obs, info = super().reset(**kwargs)
        self._add_frame(obs)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)
        self._add_frame(obs)

        if (terminated or truncated) and len(self._video) > 0:
            if self._max_videos is not None:
                self._max_videos -= 1
            video = np.moveaxis(np.stack(self._video), -1, 1)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, 1)
            video = wandb.Video(video, fps=20, format="mp4")
            wandb.log({self._name: video}, commit=False)

        return obs, reward, terminated, truncated, info
