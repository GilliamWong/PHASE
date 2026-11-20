"""A synthetic human that takes over from the learner during training.

It stands in for a person supervising training, after Q-OIL's simulated human
(Teegavarapu et al., CoRL 2026, Sec. 4.2 and App. F.4): a gate decides when to
take over, and a trained agent with partial success (the corrector) chooses
the actions until it hands control back.

With style="rac", each takeover follows RaC's Rule 1 (Hu et al., 2025,
arXiv:2509.07953): first a scripted recovery that moves the gripper back to
the nearest demo state (relative to the object), then the corrector. Rule 2
(end the episode when the takeover ends) is end_after_takeover; train.py
applies it as a time-limit truncation, so the critic still bootstraps.
"""

import collections
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from rlpd.agent import RLPDAgent


@dataclass
class HumanConfig:
    # Checkpoint (a run's checkpoints/step_<step>.pt) of the agent that takes
    # over. Empty means no interventions: plain RLPD training.
    corrector: str = ""
    # When to take over, checked at every step the learner controls:
    # "random" with probability intervention_rate; "progress" (Q-OIL's PAM)
    # also with stall_intervention_prob when progress has stalled.
    gate: str = "random"
    intervention_rate: float = 0.005  # PAM's beta when gate=progress
    # Checkpoint whose critic scores progress as V(s) = mean_i Q_i(s, mean
    # action), for gate=progress. Its rewards set the scale of
    # progress_threshold; the PAM values below assume rewards of 0 or 10.
    progress_reference: str = ""
    progress_window: int = 12  # k: progress is V(s_t) - V(s_{t-k})
    progress_threshold: float = -0.158  # delta: stalled when progress is below it
    stall_intervention_prob: float = 0.599  # alpha
    # Each takeover lasts until the episode ends with this probability, and
    # otherwise for a uniform number of steps in [min, max]_takeover_steps.
    full_takeover_prob: float = 0.5
    min_takeover_steps: int = 10
    max_takeover_steps: int = 40
    # The corrector gives its mean action, as in evaluation, so its measured
    # success rate describes the actions it gives; --no-corrector_deterministic
    # samples from its policy instead (the paper does not say which it used).
    corrector_deterministic: bool = True
    # "qoil": the corrector acts for the whole takeover. "rac": a scripted
    # recovery to the nearest demo state first, then the corrector.
    style: str = "qoil"
    recovery_tolerance: float = 0.02  # meters: recovery ends this close to the target
    max_recovery_steps: int = 30
    recovery_gain: float = 10.0  # delta-position action per meter of error, clipped to [-1, 1]
    # RaC's Rule 2: end the episode when the takeover hands control back.
    end_after_takeover: bool = False


# Phase codes for each step, saved by train.py as phases.npy (RaC style only
# has recovery steps; with style="qoil" every takeover step is CORRECTION).
LEARNER, RECOVERY, CORRECTION = 0, 1, 2

# Layout of the flattened Robomimic observation (rlpd.env.OBS_KEYS): object
# state first (its position is the first 3 entries for Lift, Can and Square),
# then eef position (3), eef quaternion (4) and gripper joint positions (2).
_EEF_FROM_END = 9  # eef position starts 9 entries before the end


def eef_to_object(observations: np.ndarray) -> np.ndarray:
    """The end effector's position relative to the object, in meters."""
    eef = observations[..., -_EEF_FROM_END : -_EEF_FROM_END + 3]
    return eef - observations[..., :3]


def critic_value(agent: RLPDAgent) -> Callable[[np.ndarray], float]:
    """V(s) as the ensemble-mean Q of the agent's mean action."""

    @torch.no_grad()
    def value(observation: np.ndarray) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=agent.device)[None]
        return agent.critic(obs, agent.actor.mode(obs)).mean().item()

    return value


class SyntheticHuman:
    """Call reset() at the start of every episode and act() at every step."""

    def __init__(
        self,
        config: HumanConfig,
        corrector: RLPDAgent,
        rng: np.random.Generator,
        value: Optional[Callable[[np.ndarray], float]] = None,
        demo_observations: Optional[np.ndarray] = None,
        demo_actions: Optional[np.ndarray] = None,
    ):
        if config.gate not in ("random", "progress"):
            raise ValueError(f"unknown gate {config.gate!r}; use 'random' or 'progress'")
        if config.gate == "progress" and value is None:
            raise ValueError("gate=progress needs a value function (--progress_reference)")
        if not 1 <= config.min_takeover_steps <= config.max_takeover_steps:
            raise ValueError("need 1 <= min_takeover_steps <= max_takeover_steps")
        if config.style not in ("qoil", "rac"):
            raise ValueError(f"unknown style {config.style!r}; use 'qoil' or 'rac'")
        if config.style == "rac" and (demo_observations is None or demo_actions is None):
            raise ValueError("style=rac needs the demos (--offline_ratio > 0)")
        if config.end_after_takeover and config.full_takeover_prob > 0:
            raise ValueError("end_after_takeover needs --full_takeover_prob=0 (takeovers must end)")
        self._config = config
        if config.style == "rac":
            # Recovery targets: each demo state's eef-to-object offset, and
            # the gripper command the demo gave there.
            self._demo_offsets = eef_to_object(np.asarray(demo_observations))
            self._demo_gripper = np.asarray(demo_actions)[:, -1]
        self._corrector = corrector
        self._rng = rng
        self._value = value
        # V of the last k + 1 observations this episode, for gate=progress.
        self._values = collections.deque(maxlen=config.progress_window + 1)
        self.reset()

    def reset(self):
        self._takeover_steps_left = 0.0
        self._values.clear()
        self._recovering = False
        self._recovery_steps = 0
        self._target = None
        self.phase = LEARNER  # the phase of the last act() call
        self.hands_back = False  # True if the last act() was a takeover's last step

    def act(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """The human's action if the human controls this step, else None."""
        config = self._config
        self.hands_back = False
        if config.gate == "progress":
            self._values.append(self._value(observation))
        if self._takeover_steps_left == 0 and self._rng.random() < self._takeover_prob():
            if self._rng.random() < config.full_takeover_prob:
                self._takeover_steps_left = math.inf  # until reset()
            else:
                self._takeover_steps_left = self._rng.integers(
                    config.min_takeover_steps, config.max_takeover_steps + 1
                )
            if config.style == "rac":
                self._start_recovery(observation)
        if self._takeover_steps_left == 0:
            self.phase = LEARNER
            return None
        if self._recovering:
            action = self._recovery_action(observation)
            if action is not None:
                self.phase = RECOVERY
                return action
        self._takeover_steps_left -= 1
        self.hands_back = self._takeover_steps_left == 0
        self.phase = CORRECTION
        return self._corrector.act(observation, deterministic=config.corrector_deterministic)

    def _start_recovery(self, observation: np.ndarray):
        offsets = self._demo_offsets
        nearest = np.argmin(np.linalg.norm(offsets - eef_to_object(observation), axis=-1))
        self._target = (offsets[nearest], self._demo_gripper[nearest])
        self._recovering = True
        self._recovery_steps = 0

    def _recovery_action(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """A proportional step toward the recovery target, or None once
        recovery is over (target reached or max_recovery_steps used)."""
        config = self._config
        offset, gripper = self._target
        error = offset - eef_to_object(observation)
        if np.linalg.norm(error) < config.recovery_tolerance or self._recovery_steps >= config.max_recovery_steps:
            self._recovering = False
            return None
        self._recovery_steps += 1
        # OSC_POSE deltas: position (3), rotation (3, held), gripper (1).
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(config.recovery_gain * error, -1.0, 1.0)
        action[-1] = gripper
        return action

    def _takeover_prob(self) -> float:
        config = self._config
        values = self._values
        stalled = (
            config.gate == "progress"
            and len(values) == values.maxlen
            and values[-1] - values[0] < config.progress_threshold
        )
        return config.stall_intervention_prob if stalled else config.intervention_rate
