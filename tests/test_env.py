"""Checks that the online env reproduces the Robomimic demos' observations."""

import h5py
import numpy as np
import pytest

from rlpd.env import OBS_KEYS, RobomimicEnv, flatten_observation, read_env_args


def cached_dataset(name):
    """Path to an already-downloaded Robomimic dataset; skips the test otherwise."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    task, quality = name.split("/")
    try:
        return hf_hub_download(
            "robomimic/robomimic_datasets",
            f"v1.5/{task}/{quality}/low_dim_v15.hdf5",
            repo_type="dataset",
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.skip(f"{name} not downloaded; run training once to fetch it")


@pytest.mark.parametrize("name", ["lift/ph", "can/ph"])
def test_env_observations_match_demo_observations(name):
    # Catches online observations that differ from the demos' (key order,
    # robosuite's "object-state" naming), restoring each recorded state into
    # the demo's own model as Robomimic does.
    path = cached_dataset(name)
    env = RobomimicEnv(read_env_args(path))
    with h5py.File(path) as f:
        demo = f["data/demo_0"]
        states = demo["states"][:]
        demo_obs = flatten_observation({k: demo["obs"][k][:] for k in OBS_KEYS})
        model_xml = demo.attrs["model_file"]

    robosuite_env = env.unwrapped._env
    robosuite_env.reset_from_xml_string(model_xml)
    # Visit every state in order: some robosuite observables reuse values cached
    # by the previous update, as they did when the demos were recorded.
    for t in range(len(states)):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        raw = robosuite_env._get_observations(force_update=True)
        online = flatten_observation(env._get_obs_dict(raw))
        np.testing.assert_allclose(online, demo_obs[t], atol=1e-5)


def test_seeded_resets_are_reproducible():
    # Catches object placement drawing from NumPy's unseeded global RNG.
    env_args = read_env_args(cached_dataset("lift/ph"))
    env_a, env_b = RobomimicEnv(env_args), RobomimicEnv(env_args)

    obs_a, _ = env_a.reset(seed=3)
    np.random.seed(123)  # unrelated global RNG use must not change placements
    obs_b, _ = env_b.reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert not np.allclose(env_a.reset()[0], obs_a)  # later resets still vary
