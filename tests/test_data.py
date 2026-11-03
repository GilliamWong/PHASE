"""Checks demo loading, the replay buffer, and RLPD's batch mixing."""

import h5py
import numpy as np
import torch

from rlpd.data import TransitionBuffer, load_demo_transitions, sample_mixed
from rlpd.env import OBS_KEYS, flatten_observation

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
    np.testing.assert_array_equal(transitions["masks"], [1, 1, 1, 0, 1, 1, 1])
    with h5py.File(path) as f:
        demo = f["data/demo_2"]
        expected_obs = flatten_observation({k: demo["obs"][k][:4] for k in OBS_KEYS})
        expected_next = flatten_observation({k: demo["next_obs"][k][:4] for k in OBS_KEYS})
    np.testing.assert_allclose(transitions["observations"][:4], expected_obs)
    np.testing.assert_allclose(transitions["next_observations"][:4], expected_next)


def _buffer_with(values, capacity, obs_dim=2, action_dim=1):
    """A buffer whose i-th added transition has every field equal to values[i]."""
    buffer = TransitionBuffer(obs_dim, action_dim, capacity, torch.device("cpu"))
    for v in values:
        buffer.add(np.full(obs_dim, v), np.full(action_dim, v), v, v, np.full(obs_dim, v))
    return buffer


def test_fields_round_trip_through_rows():
    # Catches column slices that disagree between add() and split().
    buffer = TransitionBuffer(2, 3, 4, torch.device("cpu"))
    buffer.add(np.array([1, 2]), np.array([3, 4, 5]), 6.0, 0.0, np.array([7, 8]))

    batch = buffer.split(buffer.rows[:1])

    np.testing.assert_array_equal(batch.observations[0], [1, 2])
    np.testing.assert_array_equal(batch.actions[0], [3, 4, 5])
    assert batch.rewards[0] == 6 and batch.masks[0] == 0
    np.testing.assert_array_equal(batch.next_observations[0], [7, 8])


def test_sampling_only_returns_filled_rows():
    # Catches indices reaching unfilled rows (all zeros here) while the buffer
    # is partly full, and the ring buffer not overwriting its oldest rows.
    partly_full = _buffer_with([1.0] * 3, capacity=100)
    assert (partly_full.sample_rows((10_000,)) == 1.0).all()

    wrapped = _buffer_with([1.0, 2.0, 3.0, 4.0, 5.0], capacity=3)
    assert len(wrapped) == 3
    assert set(wrapped.sample_rows((1_000,))[:, 0].tolist()) == {3.0, 4.0, 5.0}


def test_each_minibatch_gets_the_same_demo_online_mix():
    # Catches minibatches that are all demos or all online, which per-minibatch
    # critic updates would then see as alternating data sources.
    demos, online = _buffer_with([0.0], 1), _buffer_with([1.0], 1)

    batch = sample_mixed(demos, online, 3, 5, num_minibatches=4)

    assert batch.rewards.shape == (4, 8)
    assert (batch.rewards.sum(dim=1) == 5).all()
