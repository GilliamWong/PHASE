"""A synthetic human that takes over from the learner during training.

It stands in for a person supervising training, after Q-OIL's simulated human
(Teegavarapu et al., CoRL 2026, Sec. 4.2 and App. F.4): a gate decides when to
take over, and a trained agent with partial success (the corrector) chooses
the actions until it hands control back.
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
    ):
        if config.gate not in ("random", "progress"):
            raise ValueError(f"unknown gate {config.gate!r}; use 'random' or 'progress'")
        if config.gate == "progress" and value is None:
            raise ValueError("gate=progress needs a value function (--progress_reference)")
        if not 1 <= config.min_takeover_steps <= config.max_takeover_steps:
            raise ValueError("need 1 <= min_takeover_steps <= max_takeover_steps")
        self._config = config
        self._corrector = corrector
        self._rng = rng
        self._value = value
        # V of the last k + 1 observations this episode, for gate=progress.
        self._values = collections.deque(maxlen=config.progress_window + 1)
        self._takeover_steps_left = 0.0

    def reset(self):
        self._takeover_steps_left = 0.0
        self._values.clear()

    def act(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """The corrector's action if the human controls this step, else None."""
        config = self._config
        if config.gate == "progress":
            self._values.append(self._value(observation))
        if self._takeover_steps_left == 0 and self._rng.random() < self._takeover_prob():
            if self._rng.random() < config.full_takeover_prob:
                self._takeover_steps_left = math.inf  # until reset()
            else:
                self._takeover_steps_left = self._rng.integers(
                    config.min_takeover_steps, config.max_takeover_steps + 1
                )
        if self._takeover_steps_left == 0:
            return None
        self._takeover_steps_left -= 1
        return self._corrector.act(observation, deterministic=config.corrector_deterministic)

    def _takeover_prob(self) -> float:
        config = self._config
        values = self._values
        stalled = (
            config.gate == "progress"
            and len(values) == values.maxlen
            and values[-1] - values[0] < config.progress_threshold
        )
        return config.stall_intervention_prob if stalled else config.intervention_rate
