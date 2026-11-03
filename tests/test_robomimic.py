"""Checks that Robomimic demos and the online env describe the same MDP."""

import h5py
import numpy as np
import pytest

from rlpd.data import combine_batches
from rlpd.data.robomimic_datasets import load_demo_transitions
from rlpd.envs.robomimic import OBS_KEYS, flatten_observation

OBS_DIMS = {"object": 3, "robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2}


def _write_demo(group, rewards, rng):
    num_steps = len(rewards)
    for obs_group in ("obs", "next_obs"):
        for key, dim in OBS_DIMS.items():
            group.create_dataset(f"{obs_group}/{key}", data=rng.normal(size=(num_steps, dim)))
        # A key outside OBS_KEYS, which must be ignored.
        group.create_dataset(f"{obs_group}/robot0_joint_vel", data=rng.normal(size=(num_steps, 7)))
    group.create_dataset("actions", data=rng.uniform(-1, 1, (num_steps, 7)))
    group.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float64))


def test_demos_end_at_first_success(tmp_path):
    # Catches demos running past the success step, which RobomimicEnv never
    # produces online, and success not being terminal.
    path = tmp_path / "demos.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        # demo_10 sorts before demo_2 as a string; transitions follow numeric order.
        _write_demo(f.create_group("data/demo_2"), [0, 0, 0, 1, 1], rng)
        _write_demo(f.create_group("data/demo_10"), [0, 0, 0], rng)  # never succeeds

    transitions = load_demo_transitions(str(path), success_reward=10.0)

    np.testing.assert_array_equal(transitions["rewards"], [0, 0, 0, 10, 0, 0, 0])
    np.testing.assert_array_equal(transitions["terminals"], [0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(transitions["dones"], [0, 0, 0, 1, 0, 0, 1])
    with h5py.File(path) as f:
        demo = f["data/demo_2"]
        expected_obs = flatten_observation({k: demo["obs"][k][:4] for k in OBS_KEYS})
        expected_next = flatten_observation({k: demo["next_obs"][k][:4] for k in OBS_KEYS})
    np.testing.assert_allclose(transitions["observations"][:4], expected_obs)
    np.testing.assert_allclose(transitions["next_observations"][:4], expected_next)


def test_combine_batches_gives_each_minibatch_the_same_mix():
    # Catches minibatches that are all offline or all online, which would happen
    # if the batches were simply concatenated before SACLearner.update slices them.
    num_minibatches, offline_per, online_per = 4, 3, 5
    offline = {"x": np.zeros(num_minibatches * offline_per)}
    online = {"x": np.ones(num_minibatches * online_per)}

    combined = combine_batches(offline, online, num_minibatches)["x"]

    for minibatch in combined.reshape(num_minibatches, -1):
        assert (minibatch == 0).sum() == offline_per
        assert (minibatch == 1).sum() == online_per


def _cached_dataset(name):
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
    from rlpd.envs.robomimic import RobomimicEnv, read_env_args

    path = _cached_dataset(name)
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
    from rlpd.envs.robomimic import RobomimicEnv, read_env_args

    env_args = read_env_args(_cached_dataset("lift/ph"))
    env_a, env_b = RobomimicEnv(env_args), RobomimicEnv(env_args)

    obs_a, _ = env_a.reset(seed=3)
    np.random.seed(123)  # unrelated global RNG use must not change placements
    obs_b, _ = env_b.reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert not np.allclose(env_a.reset()[0], obs_a)  # later resets still vary
