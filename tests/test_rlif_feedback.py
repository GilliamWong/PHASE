"""Checks RLIF's temporally shifted intervention-feedback rewards."""

import dataclasses
import json
import numpy as np
import pytest
import torch

from rlif.data import TransitionBuffer
from rlif.human import InterventionFeedback
from rlif.agent import AgentConfig, RLPDAgent
from rlif.train import RoundConfig, collect_episode, load_agent, train_rounds


def step(feedback, index, intervening):
    labeled = feedback.begin_step(intervening)
    feedback.record_step(
        np.array([index], np.float32),
        np.array([index + 0.25], np.float32),
        1.0,
        np.array([index + 1], np.float32),
    )
    return labeled


def test_only_transition_leading_to_takeover_gets_negative_reward():
    feedback = InterventionFeedback()

    assert step(feedback, 0, False) is None
    before_takeover = step(feedback, 1, True)
    during_takeover = step(feedback, 2, True)
    after_handback = step(feedback, 3, False)
    final = feedback.end_episode()

    assert before_takeover.reward == -1
    assert before_takeover.action == pytest.approx([0.25])
    assert not before_takeover.controlled_by_expert
    assert during_takeover.reward == 0
    assert during_takeover.controlled_by_expert
    assert after_handback.reward == 0
    assert after_handback.controlled_by_expert
    assert final.reward == 0
    assert not final.controlled_by_expert


def test_repeated_takeovers_each_penalize_the_preceding_policy_step():
    feedback = InterventionFeedback()
    rewards = []
    for index, intervening in enumerate([False, True, False, False, True, True]):
        labeled = step(feedback, index, intervening)
        if labeled is not None:
            rewards.append(labeled.reward)
    rewards.append(feedback.end_episode().reward)

    assert rewards == [-1, 0, 0, -1, 0, 0]


def test_takeover_at_initial_state_has_no_transition_to_penalize():
    feedback = InterventionFeedback()
    assert step(feedback, 0, True) is None
    assert feedback.end_episode().reward == 0


def test_episode_boundary_clears_takeover_state():
    feedback = InterventionFeedback()
    step(feedback, 0, True)
    feedback.end_episode()
    feedback.reset()

    assert step(feedback, 0, False) is None
    assert step(feedback, 1, True).reward == -1


def test_call_order_errors_are_explicit():
    feedback = InterventionFeedback()
    with pytest.raises(RuntimeError, match="begin_step"):
        feedback.record_step(np.zeros(1), np.zeros(1), 1.0, np.zeros(1))
    feedback.begin_step(False)
    with pytest.raises(RuntimeError, match="record"):
        feedback.begin_step(False)


def test_collection_ignores_task_reward_and_counts_one_takeover_event():
    class Agent:
        def act(self, observation, deterministic=False):
            assert not deterministic
            return np.array([observation[0] + 0.5], np.float32)

    class Controller:
        def reset(self):
            pass

        def act(self, observation, policy_action):
            return np.array([9.0], np.float32) if observation[0] in (1, 2) else None

    class Env:
        def reset(self):
            self.t = 0
            return np.array([0.0], np.float32), {}

        def step(self, action):
            self.t += 1
            # Deliberately nonzero: RLIF must discard the task reward.
            return np.array([self.t], np.float32), 10.0, self.t == 3, False, {"t": self.t}

    replay = TransitionBuffer(1, 1, 3, torch.device("cpu"))
    metrics = collect_episode(Env(), Agent(), Controller(), replay)
    batch = replay.split(replay.rows[: len(replay)])

    assert batch.rewards.tolist() == [-1.0, 0.0, 0.0]
    assert batch.actions[:, 0].tolist() == pytest.approx([0.5, 9.0, 9.0])
    assert batch.masks.tolist() == [1.0, 1.0, 0.0]
    assert metrics["intervention_events"] == 1
    assert metrics["intervention_steps"] == 2


def test_round_driver_respects_pretrain_and_per_round_update_counts():
    class Agent:
        def __init__(self):
            self.updates = 0

        def act(self, observation, deterministic=False):
            return np.zeros(1, np.float32)

        def update(self, batch):
            self.updates += 1
            return {"updates": torch.tensor(self.updates)}

    class Controller:
        def reset(self):
            pass

        def act(self, observation, policy_action):
            return None

    class OneStepEnv:
        def reset(self):
            return np.zeros(1, np.float32), {}

        def step(self, action):
            return np.ones(1, np.float32), 123.0, True, False, {}

    demos = TransitionBuffer(1, 1, 1, torch.device("cpu"))
    demos.add(np.zeros(1), np.zeros(1), 0.0, 0.0, np.ones(1))
    online = TransitionBuffer(1, 1, 2, torch.device("cpu"))
    agent = Agent()
    config = RoundConfig(
        num_rounds=2,
        episodes_per_round=1,
        pretrain_updates=3,
        updates_per_round=4,
        batch_size=2,
        offline_ratio=0.5,
        utd_ratio=1,
    )

    history = train_rounds(OneStepEnv(), agent, Controller(), demos, online, config)

    assert agent.updates == 3 + 2 * 4
    assert len(history) == 2
    assert len(online) == 2


def test_rlpd_checkpoint_loads_into_independent_rlif_learner(tmp_path):
    config = AgentConfig(hidden_dims=(8,), num_qs=2, num_min_qs=2)
    original = RLPDAgent(2, 1, config, torch.device("cpu"))
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(config), f)
    path = checkpoint_dir / "step.pt"
    torch.save({"model": original.state_dict()}, path)

    restored = load_agent(str(path), 2, 1, torch.device("cpu"))

    for expected, actual in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)
