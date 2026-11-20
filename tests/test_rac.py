"""Checks the RaC-style synthetic human: recovery first, then the corrector,
and the hand-back flag used for Rule 2."""

import numpy as np
import pytest

from rlpd.human import CORRECTION, LEARNER, RECOVERY, HumanConfig, SyntheticHuman, eef_to_object

CORRECTOR_ACTION = np.full(7, 0.5, dtype=np.float32)


class FixedCorrector:
    def act(self, observation, deterministic):
        return CORRECTOR_ACTION


def _obs(object_pos, eef_pos, object_dim=10):
    """A Lift-shaped observation: object state (10), eef pos (3), quat (4), gripper (2)."""
    obs = np.zeros(object_dim + 9, dtype=np.float32)
    obs[:3] = object_pos
    obs[object_dim : object_dim + 3] = eef_pos
    return obs


# Two demo states: gripper 10 cm above the object (open), and at it (closed).
DEMO_OBS = np.stack([_obs([0, 0, 0], [0, 0, 0.10]), _obs([0, 0, 0], [0, 0, 0.0])])
DEMO_ACTIONS = np.zeros((2, 7), dtype=np.float32)
DEMO_ACTIONS[:, -1] = [-1.0, 1.0]


def _human(**overrides):
    config = dict(
        corrector="unused", style="rac", intervention_rate=1.0, full_takeover_prob=0.0,
        min_takeover_steps=2, max_takeover_steps=2, max_recovery_steps=5,
    )
    config.update(overrides)
    return SyntheticHuman(
        HumanConfig(**config), FixedCorrector(), np.random.default_rng(0),
        demo_observations=DEMO_OBS, demo_actions=DEMO_ACTIONS,
    )


def test_eef_to_object_uses_the_eef_block():
    assert np.allclose(eef_to_object(_obs([1, 2, 3], [1.5, 2, 3.25])), [0.5, 0, 0.25])


def test_recovery_moves_toward_nearest_demo_state_then_corrector_takes_over():
    human = _human()
    # 8 cm above the object: nearest demo state is the one 10 cm above (open gripper).
    obs = _obs([0, 0, 0], [0, 0, 0.08])
    action = human.act(obs)
    assert human.phase == RECOVERY
    assert action[2] > 0 and np.allclose(action[:2], 0) and action[-1] == -1.0
    # At the target: recovery ends and the corrector acts for the 2 sampled steps.
    at_target = _obs([0, 0, 0], [0, 0, 0.10])
    assert np.array_equal(human.act(at_target), CORRECTOR_ACTION) and human.phase == CORRECTION
    assert not human.hands_back
    human.act(at_target)
    assert human.phase == CORRECTION and human.hands_back


def test_recovery_gives_up_after_max_steps():
    human = _human(max_recovery_steps=3)
    far = _obs([0, 0, 0], [0.5, 0.5, 0.5])
    phases = []
    for _ in range(5):
        human.act(far)
        phases.append(human.phase)
    assert phases == [RECOVERY] * 3 + [CORRECTION] * 2


def test_qoil_style_has_no_recovery_phase():
    human = _human(style="qoil")
    human.act(_obs([0, 0, 0], [0.5, 0.5, 0.5]))
    assert human.phase == CORRECTION


def test_no_takeover_is_learner_phase():
    human = _human(intervention_rate=0.0)
    assert human.act(_obs([0, 0, 0], [0, 0, 0])) is None and human.phase == LEARNER


def test_rule_two_needs_takeovers_that_end():
    with pytest.raises(ValueError):
        _human(end_after_takeover=True, full_takeover_prob=0.5)


def test_rac_needs_demos():
    with pytest.raises(ValueError):
        SyntheticHuman(HumanConfig(corrector="unused", style="rac"), FixedCorrector(), np.random.default_rng(0))
