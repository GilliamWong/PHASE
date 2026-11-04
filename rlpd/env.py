"""Robomimic tasks (robosuite 1.5) as Gymnasium environments.

The env is built from the env config stored in a Robomimic dataset, so online
interaction matches the setup the demonstrations were recorded in.
"""

import contextlib
import json
from typing import Mapping, Optional

import gymnasium as gym
import h5py
import numpy as np
import robosuite
from huggingface_hub import hf_hub_download

# Robomimic's standard low-dimensional observation. Dataset and env
# observations are both flattened in this order by flatten_observation.
OBS_KEYS = ("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

# Rollout horizon Robomimic uses for Lift and Can.
DEFAULT_HORIZON = 400

_HF_REPO = "robomimic/robomimic_datasets"

# Evaluation videos: square frames from Robomimic's default viewpoint.
_RENDER_SIZE = 256
_RENDER_CAMERA = "agentview"


def download_dataset(dataset: str) -> str:
    """Downloads a low-dim Robomimic dataset, e.g. "lift/ph", and returns its path.

    Files are cached by huggingface_hub (under ~/.cache/huggingface).
    """
    task, quality = dataset.split("/")
    return hf_hub_download(
        _HF_REPO, f"v1.5/{task}/{quality}/low_dim_v15.hdf5", repo_type="dataset"
    )


def read_env_args(dataset_path: str) -> dict:
    with h5py.File(dataset_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def flatten_observation(obs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Concatenates OBS_KEYS along the last axis (works for single steps or batches)."""
    return np.concatenate([obs[k] for k in OBS_KEYS], axis=-1).astype(np.float32)


@contextlib.contextmanager
def _use_as_global_numpy_rng(rng: np.random.RandomState):
    # robosuite's object placement samplers draw from NumPy's global RNG.
    saved = np.random.get_state()
    np.random.set_state(rng.get_state())
    try:
        yield
    finally:
        rng.set_state(np.random.get_state())
        np.random.set_state(saved)


class RobomimicEnv(gym.Env):
    """A Robomimic task with a sparse success reward.

    The episode terminates on success with reward success_reward; every other
    step gives 0. Episodes are truncated after `horizon` steps.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env_args: dict,
        horizon: int = DEFAULT_HORIZON,
        success_reward: float = 10.0,
        render_mode: Optional[str] = None,
    ):
        self.render_mode = render_mode
        self._horizon = horizon
        self._success_reward = success_reward

        env_kwargs = dict(env_args["env_kwargs"])
        env_kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=render_mode == "rgb_array",
            use_camera_obs=False,
            # Horizon and termination are handled here, not by robosuite.
            ignore_done=True,
        )
        self._env = robosuite.make(env_args["env_name"], **env_kwargs)

        obs_spec = self._get_obs_dict(self._env.observation_spec())
        obs_dim = flatten_observation(obs_spec).shape[-1]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        low, high = self._env.action_spec
        self.action_space = gym.spaces.Box(low.astype(np.float32), high.astype(np.float32))

        self._placement_rng = np.random.RandomState()
        self._num_steps = 0

    @staticmethod
    def _get_obs_dict(raw_obs: Mapping[str, np.ndarray]) -> dict:
        obs = dict(raw_obs)
        # robosuite names the object observables "object-state"; Robomimic, "object".
        obs["object"] = obs["object-state"]
        return obs

    @property
    def sim(self):
        """The MuJoCo simulation, e.g. for saving and restoring states."""
        return self._env.sim

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._placement_rng = np.random.RandomState(seed)
        with _use_as_global_numpy_rng(self._placement_rng):
            raw_obs = self._env.reset()
        self._num_steps = 0
        return flatten_observation(self._get_obs_dict(raw_obs)), {}

    def step(self, action):
        raw_obs, _, _, info = self._env.step(action)
        self._num_steps += 1
        success = bool(self._env._check_success())
        info = dict(info, is_success=success)
        reward = self._success_reward if success else 0.0
        truncated = self._num_steps >= self._horizon and not success
        obs = flatten_observation(self._get_obs_dict(raw_obs))
        return obs, reward, success, truncated, info

    def render(self):
        frame = self._env.sim.render(
            width=_RENDER_SIZE,
            height=_RENDER_SIZE,
            camera_name=_RENDER_CAMERA,
        )
        # MuJoCo's offscreen buffer is upside down.
        return frame[::-1]

    def close(self):
        self._env.close()


def seed_env(env: gym.Env, seed: int) -> np.ndarray:
    """Seeds the env's dynamics and its spaces' samplers; returns the first obs."""
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    return observation
