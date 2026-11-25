"""Checks the paper-native and PHASE-adapted RLIF intervention gates."""

import numpy as np

from rlif.human import (
    OfficialRandomController,
    OfficialValueController,
    PhaseProgressController,
    PhaseRandomController,
)


OBS = np.array([0.0], np.float32)
POLICY_ACTION = np.array([-1.0], np.float32)


def corrector(observation):
    return np.array([1.0], np.float32)


def test_official_random_takeover_has_fixed_duration():
    controller = OfficialRandomController(corrector, np.random.default_rng(0), 1.0, 3)
    controller.reset()
    actions = [controller.act(OBS, POLICY_ACTION) for _ in range(3)]
    assert all(action is not None for action in actions)

    # With probability zero, completing a takeover returns control permanently.
    controller.intervention_probability = 0.0
    assert controller.act(OBS, POLICY_ACTION) is None


def test_stateful_corrector_observes_every_state_and_resets():
    class StatefulCorrector:
        def __init__(self):
            self.observations = []
            self.resets = 0

        def reset(self):
            self.resets += 1
            self.observations.clear()

        def observe(self, observation):
            self.observations.append(float(observation[0]))
            return np.array([len(self.observations)], np.float32)

    corrector = StatefulCorrector()
    controller = OfficialRandomController(corrector, np.random.default_rng(0), 0.0, 3)
    controller.reset()
    assert controller.act(np.array([1.0], np.float32), POLICY_ACTION) is None
    assert controller.act(np.array([2.0], np.float32), POLICY_ACTION) is None
    assert corrector.observations == [1.0, 2.0]
    controller.intervention_probability = 1.0
    np.testing.assert_array_equal(
        controller.act(np.array([3.0], np.float32), POLICY_ACTION), np.array([3.0], np.float32)
    )
    assert corrector.resets == 1


def test_official_value_gate_uses_reference_critic_comparison():
    def q_value(observation, action):
        return float(action[0])

    controller = OfficialValueController(
        corrector,
        q_value,
        np.random.default_rng(0),
        threshold=0.5,
        takeover_steps=1,
        bad_probability=1.0,
        good_probability=0.0,
    )
    controller.reset()
    assert controller.act(OBS, POLICY_ACTION) is not None
    assert controller.act(OBS, np.array([2.0], np.float32)) is None


def test_phase_random_supports_bounded_takeovers():
    controller = PhaseRandomController(
        corrector,
        np.random.default_rng(0),
        intervention_probability=1.0,
        full_takeover_probability=0.0,
        min_takeover_steps=2,
        max_takeover_steps=2,
    )
    controller.reset()
    assert controller.act(OBS, POLICY_ACTION) is not None
    controller.intervention_probability = 0.0
    assert controller.act(OBS, POLICY_ACTION) is not None
    assert controller.act(OBS, POLICY_ACTION) is None


def test_phase_progress_gate_triggers_on_stall_only_after_full_window():
    values = iter([1.0, 1.0, 0.0])
    controller = PhaseProgressController(
        corrector,
        lambda observation: next(values),
        np.random.default_rng(0),
        intervention_probability=0.0,
        progress_window=2,
        progress_threshold=-0.5,
        stall_intervention_probability=1.0,
        full_takeover_probability=0.0,
        min_takeover_steps=1,
        max_takeover_steps=1,
    )
    controller.reset()
    assert controller.act(OBS, POLICY_ACTION) is None
    assert controller.act(OBS, POLICY_ACTION) is None
    assert controller.act(OBS, POLICY_ACTION) is not None
