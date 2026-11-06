"""Checks the synthetic human's gate and takeover lengths."""

import numpy as np

from rlpd.human import HumanConfig, SyntheticHuman

HUMAN_ACTION = np.array([0.5])


class FixedCorrector:
    def act(self, observation, deterministic):
        self.deterministic = deterministic
        return HUMAN_ACTION


def _human(**overrides):
    config = HumanConfig(corrector="unused", **overrides)
    # The value of an observation is its first entry.
    return SyntheticHuman(config, FixedCorrector(), np.random.default_rng(0), lambda o: o[0])


def test_progress_gate_fires_when_value_falls_over_the_window():
    # Catches an off-by-one window (comparing with V(s_{t-k+1}) or
    # V(s_{t-k-1})) or a flipped sign on progress.
    # Never take over unless stalled, then always, for one step.
    human = _human(
        gate="progress",
        intervention_rate=0.0,
        stall_intervention_prob=1.0,
        progress_window=3,
        progress_threshold=-0.5,
        full_takeover_prob=0.0,
        min_takeover_steps=1,
        max_takeover_steps=1,
    )
    # Values rise, then fall by 0.2 per step: V_t - V_{t-3} is first below
    # -0.5 (at -0.6) at the step whose value is 1.4.
    values = [0.0, 1.0, 2.0, 1.8, 1.6, 1.4, 1.2]
    controlled = [human.act(np.array([v])) is not None for v in values]

    assert controlled == [False, False, False, False, False, True, True]


def test_takeovers_last_the_sampled_length_or_until_reset():
    # Catches takeovers that end early, overrun, or survive the episode end.
    short = _human(intervention_rate=1.0, full_takeover_prob=0.0, min_takeover_steps=3, max_takeover_steps=3)
    short.act(np.zeros(1))  # starts a 3-step takeover
    short._config.intervention_rate = 0.0  # no new takeovers from here on
    assert [short.act(np.zeros(1)) is not None for _ in range(4)] == [True, True, False, False]

    full = _human(intervention_rate=1.0, full_takeover_prob=1.0)
    full.act(np.zeros(1))
    full._config.intervention_rate = 0.0
    assert all(full.act(np.zeros(1)) is not None for _ in range(500))
    full.reset()
    assert full.act(np.zeros(1)) is None


def test_corrector_acts_with_its_mean_unless_told_to_sample():
    # Catches --no-corrector_deterministic being ignored, or the default changing.
    for deterministic in (True, False):
        human = _human(intervention_rate=1.0, corrector_deterministic=deterministic)
        human.act(np.zeros(1))
        assert human._corrector.deterministic is deterministic
