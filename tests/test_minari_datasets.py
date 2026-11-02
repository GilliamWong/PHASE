"""Checks that Minari episodes become transitions aligned with the online env."""

from collections import OrderedDict

import gymnasium as gym
import numpy as np
from minari import EpisodeData

from rlpd.data.minari_datasets import episodes_to_transitions
from rlpd.wrappers import wrap_gym

# Keys deliberately out of sorted order: wrap_gym re-sorts them before flattening.
DICT_SPACE = gym.spaces.Dict(
    OrderedDict(
        observation=gym.spaces.Box(-10, 10, (3,), np.float64),
        desired_goal=gym.spaces.Box(-10, 10, (2,), np.float64),
    )
)
ACTION_SPACE = gym.spaces.Box(-1, 1, (2,), np.float32)


def _episode(episode_id, num_steps, terminated, rng):
    observations = {k: rng.normal(size=(num_steps + 1, *s.shape)) for k, s in DICT_SPACE.items()}
    terminations = np.zeros(num_steps, dtype=bool)
    truncations = np.zeros(num_steps, dtype=bool)
    (terminations if terminated else truncations)[-1] = True
    return EpisodeData(
        id=episode_id,
        observations=observations,
        actions=rng.uniform(-1, 1, size=(num_steps, 2)).astype(np.float32),
        rewards=rng.normal(size=num_steps),
        terminations=terminations,
        truncations=truncations,
        infos={},
    )


class _ReplayEnv(gym.Env):
    """Emits a fixed observation so wrap_gym's output can be compared."""

    observation_space = DICT_SPACE
    action_space = ACTION_SPACE

    def __init__(self, observation):
        self._observation = observation

    def reset(self, *, seed=None, options=None):
        return self._observation, {}


def test_dict_observations_match_wrapped_env_flattening():
    # Catches offline/online observation misalignment, e.g. AntMaze goal ordering.
    rng = np.random.default_rng(0)
    episode = _episode(0, 4, terminated=False, rng=rng)
    transitions = episodes_to_transitions([episode], DICT_SPACE)

    for t in range(4):
        raw = {k: v[t] for k, v in episode.observations.items()}
        online, _ = wrap_gym(_ReplayEnv(raw)).reset()
        np.testing.assert_allclose(transitions["observations"][t], online, rtol=1e-6)


def test_transitions_follow_episode_boundaries():
    # Catches next_observations crossing episodes and truncations being treated
    # as terminations (which would wrongly zero the bootstrap mask).
    rng = np.random.default_rng(1)
    episodes = [_episode(0, 3, terminated=True, rng=rng), _episode(1, 5, terminated=False, rng=rng)]
    transitions = episodes_to_transitions(episodes, DICT_SPACE)

    assert len(transitions["observations"]) == 8
    np.testing.assert_array_equal(transitions["dones"], [0, 0, 1, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(transitions["terminals"], [0, 0, 1, 0, 0, 0, 0, 0])
    # Within an episode, s'_t == s_{t+1}; the last s' is the episode's final obs.
    np.testing.assert_array_equal(
        transitions["next_observations"][:2], transitions["observations"][1:3]
    )
    final_obs = np.concatenate([episodes[0].observations[k][-1] for k in sorted(DICT_SPACE.keys())])
    np.testing.assert_allclose(transitions["next_observations"][2], final_obs)
