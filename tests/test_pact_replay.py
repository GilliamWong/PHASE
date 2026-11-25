import numpy as np
import pytest
import torch

from pact.replay import OnlineReplay, annotate_episode
from pact.train import update_from_replay
from rlif.data import Batch, TransitionBuffer


def test_takeovers_link_only_their_own_learner_segments():
    observations = np.arange(8).reshape(-1, 1)
    actions = observations + 10
    intervention = [False, False, True, True, False, False, True, False]
    sampled = {2: [-2], 6: [-6]}
    progress = [0.6, 0.4, 0.41, 0.8, 0.7, 0.5, 0.51, 0.9]
    inputs = (observations, actions, intervention, sampled, progress)
    result = annotate_episode(*inputs, success=True)
    np.testing.assert_array_equal(result.preference_indices, [0, 0, -1, -1, 1, 1, -1, -1])
    np.testing.assert_allclose(result.correction_weights, [np.exp(-3), 1, 0, 0, np.exp(-3), 1, 0, 0])
    assert len(result.preferences) == 2
    np.testing.assert_array_equal(result.preferences[0].observation, [2])
    np.testing.assert_array_equal(result.preferences[0].human_action, [12])
    np.testing.assert_array_equal(result.preferences[0].policy_action, [-2])

    failed = annotate_episode(*inputs, success=False)
    assert len(failed.preferences) == 2
    assert (failed.preference_indices == -1).all()
    assert not failed.correction_weights.any()


def test_takeover_at_reset_has_a_preference_without_a_segment():
    result = annotate_episode([[0]], [[1]], [True], {0: [-1]}, [0.0], success=True)
    assert len(result.preferences) == 1
    assert result.preference_indices.tolist() == [-1]
    with pytest.raises(ValueError, match="resampled policy action"):
        annotate_episode([[0]], [[1]], [True], {}, [0.0], success=True)


def test_shared_preference_survives_actor_eviction_until_linked_row_is_overwritten():
    replay = OnlineReplay(3, 1, torch.device("cpu"))
    rng = np.random.default_rng(42)

    def add(values, flags, sampled, progress):
        observations = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
        rewards = torch.zeros(len(values))
        batch = Batch(observations, observations, rewards, torch.ones_like(rewards), observations)
        replay.add_episode(batch, annotate_episode(
            observations.numpy(), observations.numpy(), flags, sampled, progress, success=True
        ))

    add([0, 1], [False, True], {1: [-1]}, [0.8, 0.2])
    add([2], [True], {0: [-2]}, [0.2])
    assert replay.sample_actor_preferences(1, rng).observations.item() == 2
    batch, correction = replay.sample(32, rng, row_offset=16)
    preferences, rows, weights = correction
    assert len(rows) > 0
    assert (batch.observations[rows - 16] == 0).all()
    assert (preferences.observations == 1).all()
    assert (weights == 1).all()
    add([3], [False], {}, [0.0])
    assert len(replay) == 3
    assert replay.sample(32, rng)[1] is None


def test_driver_mixes_equal_halves_and_passes_preferences_to_learner():
    device = torch.device("cpu")
    demos = TransitionBuffer(1, 1, 1, device)
    demos.add([9], [0], 0, 1, [9])
    online = OnlineReplay(2, 2, device)
    obs = torch.tensor([[0.8], [0.2]])
    reward = torch.zeros(2)
    online.add_episode(Batch(obs, obs, reward, torch.ones(2), obs),
                       annotate_episode(obs, obs, [False, True], {1: [-1]},
                                        [0.8, 0.2], success=True))

    class Learner:
        def update(self, batch, *, actor_preferences, critic_preferences):
            assert batch.observations[:16].eq(9).all()
            assert batch.observations[16:].lt(1).all()
            assert len(actor_preferences.observations) == 16
            assert critic_preferences[1].min() >= 16
            return "updated"

    assert update_from_replay(Learner(), demos, online, 32, np.random.default_rng(42)) == "updated"
