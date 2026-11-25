"""Intervention controllers and the RLIF reward-labeling state machine."""

import collections
import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np


class InterventionController(Protocol):
    """A human or synthetic corrector used while collecting an episode."""

    def reset(self) -> None:
        """Starts a new episode."""

    def act(
        self, observation: np.ndarray, policy_action: np.ndarray
    ) -> Optional[np.ndarray]:
        """Returns a corrective action, or ``None`` to leave policy control."""


ActionPolicy = Callable[[np.ndarray], np.ndarray]
ValueFunction = Callable[[np.ndarray, np.ndarray], float]
StateValueFunction = Callable[[np.ndarray], float]


class _TakeoverController:
    """Shared fixed/full-duration takeover mechanics for synthetic experts."""

    def __init__(
        self,
        corrector: ActionPolicy,
        rng: np.random.Generator,
        min_steps: int,
        max_steps: int,
        full_takeover_probability: float = 0.0,
    ):
        if not 1 <= min_steps <= max_steps:
            raise ValueError("need 1 <= min_steps <= max_steps")
        if not 0 <= full_takeover_probability <= 1:
            raise ValueError("full_takeover_probability must be in [0, 1]")
        self.corrector = corrector
        self.rng = rng
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.full_takeover_probability = full_takeover_probability
        self._steps_left = 0.0
        self._candidate_action = None

    def reset(self) -> None:
        self._steps_left = 0.0
        self._candidate_action = None
        reset = getattr(self.corrector, "reset", None)
        if reset is not None:
            reset()

    def _observe(self, observation: np.ndarray) -> None:
        """Advances a stateful corrector before choosing whether to intervene."""
        observe = getattr(self.corrector, "observe", None)
        self._candidate_action = (
            None if observe is None else np.asarray(observe(observation), dtype=np.float32)
        )

    def _start_takeover(self) -> None:
        if self.rng.random() < self.full_takeover_probability:
            self._steps_left = math.inf
        else:
            self._steps_left = int(self.rng.integers(self.min_steps, self.max_steps + 1))

    def _correct(self, observation: np.ndarray) -> np.ndarray:
        self._steps_left -= 1
        if self._candidate_action is None:
            self._candidate_action = np.asarray(self.corrector(observation), dtype=np.float32)
        return self._candidate_action.copy()


class OfficialRandomController(_TakeoverController):
    """RLIF's random gate with the paper's intended fixed-length takeover."""

    def __init__(
        self,
        corrector: ActionPolicy,
        rng: np.random.Generator,
        intervention_probability: float,
        takeover_steps: int = 4,
    ):
        if not 0 <= intervention_probability <= 1:
            raise ValueError("intervention_probability must be in [0, 1]")
        super().__init__(corrector, rng, takeover_steps, takeover_steps)
        self.intervention_probability = intervention_probability

    def act(self, observation: np.ndarray, policy_action: np.ndarray) -> Optional[np.ndarray]:
        self._observe(observation)
        if self._steps_left == 0 and self.rng.random() < self.intervention_probability:
            self._start_takeover()
        return None if self._steps_left == 0 else self._correct(observation)


class OfficialValueController(_TakeoverController):
    """The official reference-Q gate used in RLIF's value-based experiments.

    The reference critic compares the learner action with either the corrector
    action or another supplied reference-policy action. A failed comparison
    triggers takeover with probability 0.95 in the official experiments;
    otherwise a 0.05 probability is retained as noisy intervention feedback.
    """

    def __init__(
        self,
        corrector: ActionPolicy,
        reference_q: ValueFunction,
        rng: np.random.Generator,
        threshold: float,
        takeover_steps: int = 4,
        reference_policy: Optional[ActionPolicy] = None,
        bad_probability: float = 0.95,
        good_probability: float = 0.05,
    ):
        for name, value in (("bad_probability", bad_probability), ("good_probability", good_probability)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        super().__init__(corrector, rng, takeover_steps, takeover_steps)
        self.reference_q = reference_q
        self.reference_policy = reference_policy or corrector
        self.threshold = threshold
        self.bad_probability = bad_probability
        self.good_probability = good_probability

    def act(self, observation: np.ndarray, policy_action: np.ndarray) -> Optional[np.ndarray]:
        self._observe(observation)
        if self._steps_left == 0:
            if self.reference_policy is self.corrector and self._candidate_action is not None:
                reference_action = self._candidate_action
            else:
                reference_action = np.asarray(self.reference_policy(observation))
            policy_q = self.reference_q(observation, policy_action)
            reference_q = self.reference_q(observation, reference_action)
            probability = (
                self.bad_probability
                if policy_q < reference_q * self.threshold
                else self.good_probability
            )
            if self.rng.random() < probability:
                self._start_takeover()
        return None if self._steps_left == 0 else self._correct(observation)


class PhaseRandomController(_TakeoverController):
    """Q-OIL branch's random gate with its takeover-duration distribution."""

    def __init__(
        self,
        corrector: ActionPolicy,
        rng: np.random.Generator,
        intervention_probability: float = 0.005,
        full_takeover_probability: float = 0.5,
        min_takeover_steps: int = 10,
        max_takeover_steps: int = 40,
    ):
        if not 0 <= intervention_probability <= 1:
            raise ValueError("intervention_probability must be in [0, 1]")
        super().__init__(
            corrector,
            rng,
            min_takeover_steps,
            max_takeover_steps,
            full_takeover_probability,
        )
        self.intervention_probability = intervention_probability

    def act(self, observation: np.ndarray, policy_action: np.ndarray) -> Optional[np.ndarray]:
        self._observe(observation)
        if self._steps_left == 0 and self.rng.random() < self.intervention_probability:
            self._start_takeover()
        return None if self._steps_left == 0 else self._correct(observation)


class PhaseProgressController(PhaseRandomController):
    """Q-OIL branch's progress-stall gate, adapted as an RLIF corrector."""

    def __init__(
        self,
        corrector: ActionPolicy,
        value: StateValueFunction,
        rng: np.random.Generator,
        intervention_probability: float = 0.005,
        progress_window: int = 12,
        progress_threshold: float = -0.158,
        stall_intervention_probability: float = 0.599,
        **takeover_kwargs,
    ):
        super().__init__(corrector, rng, intervention_probability, **takeover_kwargs)
        if progress_window < 1:
            raise ValueError("progress_window must be positive")
        if not 0 <= stall_intervention_probability <= 1:
            raise ValueError("stall_intervention_probability must be in [0, 1]")
        self.value = value
        self.progress_threshold = progress_threshold
        self.stall_intervention_probability = stall_intervention_probability
        self._values = collections.deque(maxlen=progress_window + 1)

    def reset(self) -> None:
        super().reset()
        self._values.clear()

    def act(self, observation: np.ndarray, policy_action: np.ndarray) -> Optional[np.ndarray]:
        self._observe(observation)
        self._values.append(float(self.value(observation)))
        if self._steps_left == 0:
            stalled = (
                len(self._values) == self._values.maxlen
                and self._values[-1] - self._values[0] < self.progress_threshold
            )
            probability = (
                self.stall_intervention_probability if stalled else self.intervention_probability
            )
            if self.rng.random() < probability:
                self._start_takeover()
        return None if self._steps_left == 0 else self._correct(observation)


@dataclass(frozen=True)
class PendingTransition:
    """A transition whose RLIF reward is not known until the next state."""

    observation: np.ndarray
    action: np.ndarray
    mask: float
    next_observation: np.ndarray
    controlled_by_expert: bool


@dataclass(frozen=True)
class LabeledTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    mask: float
    next_observation: np.ndarray
    controlled_by_expert: bool


class InterventionFeedback:
    """Labels the transition before each takeover onset with reward ``-1``.

    RLIF observes whether the expert intervenes at state ``s_t`` and assigns
    the resulting feedback to the policy transition that reached that state.
    Consequently, collection holds one transition pending until control at
    the following state is known. Continued expert control is not a new onset.
    """

    def __init__(self):
        self._pending: Optional[PendingTransition] = None
        self._intervening = False
        self._began_step = False

    @property
    def intervening(self) -> bool:
        """Whether the expert controlled the preceding step."""
        return self._intervening

    def reset(self) -> None:
        if self._pending is not None:
            raise RuntimeError("finish the previous episode before reset")
        self._intervening = False
        self._began_step = False

    def begin_step(self, intervening: bool) -> Optional[LabeledTransition]:
        """Declares who controls this state and finalizes the prior step."""
        if self._began_step:
            raise RuntimeError("record the current step before beginning another")
        onset = bool(intervening) and not self._intervening
        labeled = self._label_pending(-1.0 if onset else 0.0)
        self._intervening = bool(intervening)
        self._began_step = True
        return labeled

    def record_step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        mask: float,
        next_observation: np.ndarray,
    ) -> None:
        """Stores the executed step until control at its successor is known."""
        if not self._began_step:
            raise RuntimeError("call begin_step before record_step")
        if self._pending is not None:
            raise RuntimeError("the previous transition has not been finalized")
        self._pending = PendingTransition(
            np.asarray(observation).copy(),
            np.asarray(action).copy(),
            float(mask),
            np.asarray(next_observation).copy(),
            self._intervening,
        )
        self._began_step = False

    def end_episode(self) -> Optional[LabeledTransition]:
        """Finalizes the last step with zero reward and clears control state."""
        if self._began_step:
            raise RuntimeError("record the current step before ending the episode")
        labeled = self._label_pending(0.0)
        self._intervening = False
        return labeled

    def _label_pending(self, reward: float) -> Optional[LabeledTransition]:
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        # A takeover onset should follow a policy-controlled transition.  Keep
        # the guard explicit so a malformed controller cannot punish a human
        # action while silently appearing to implement RLIF.
        if reward < 0 and pending.controlled_by_expert:
            raise RuntimeError("intervention onset cannot follow expert control")
        return LabeledTransition(
            pending.observation,
            pending.action,
            reward,
            pending.mask,
            pending.next_observation,
            pending.controlled_by_expert,
        )
