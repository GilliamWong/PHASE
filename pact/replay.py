"""PACT preference tuples and transition annotations for episode replay."""

from dataclasses import dataclass
from collections import deque

import numpy as np
import torch

from pact.agent import PreferenceBatch
from pact.segments import correction_weights, select_segment
from rlif.data import Batch


@dataclass(frozen=True)
class Preference:
    """Actions compared at the same takeover observation."""

    observation: np.ndarray
    human_action: np.ndarray
    policy_action: np.ndarray


@dataclass(frozen=True)
class EpisodeAnnotations:
    """Per-transition links into this episode's preference tuples."""

    preferences: tuple[Preference, ...]
    preference_indices: np.ndarray
    correction_weights: np.ndarray


def annotate_episode(
    observations,
    executed_actions,
    intervening,
    policy_actions_at_takeover,
    progress,
    *,
    success: bool,
) -> EpisodeAnnotations:
    """Build one preference per onset and annotate eligible learner segments.

    Inputs are aligned by action timestep. The caller supplies a freshly
    sampled policy action keyed by each takeover onset. Unlinked rows have
    index -1 and weight zero. Indices are local to the returned episode.
    """
    length = len(intervening)
    if any(len(values) != length for values in (observations, executed_actions, progress)):
        raise ValueError("episode inputs must have one entry per transition")

    preferences = []
    indices = np.full(length, -1, dtype=np.int64)
    weights = np.zeros(length, dtype=np.float32)
    previous_intervention = False
    learner_start = 0
    for step, intervention in enumerate(intervening):
        if intervention and not previous_intervention:
            if step not in policy_actions_at_takeover:
                raise ValueError(f"missing resampled policy action at takeover step {step}")
            preference_index = len(preferences)
            preferences.append(Preference(
                np.array(observations[step], copy=True),
                np.array(executed_actions[step], copy=True),
                np.array(policy_actions_at_takeover[step], copy=True),
            ))
            if success:
                bounds = select_segment(progress, step, start=learner_start)
                if bounds is not None:
                    start, end = bounds
                    indices[start:end + 1] = preference_index
                    weights[start:end + 1] = correction_weights(start, end)
        if intervention:
            learner_start = step + 1
        previous_intervention = intervention
    return EpisodeAnnotations(tuple(preferences), indices, weights)


class OnlineReplay:
    """Bounded transition ring with shared preference references and weights."""

    def __init__(self, capacity: int, preference_capacity: int, device: torch.device):
        if capacity < 1 or preference_capacity < 1:
            raise ValueError("replay capacities must be positive")
        self.transitions = []
        self.capacity = capacity
        self.next_slot = 0
        self.preferences = deque(maxlen=preference_capacity)
        self.device = device

    def __len__(self) -> int:
        return len(self.transitions)

    def add_episode(self, batch: Batch, annotations: EpisodeAnnotations) -> None:
        """Store transitions and resolve episode-local links before sampling."""
        count = len(batch.rewards)
        indices = annotations.preference_indices
        weights = annotations.correction_weights
        if any(len(field) != count for field in batch) or (
            indices.shape != (count,) or weights.shape != (count,)
        ):
            raise ValueError("episode annotations must align with transitions")
        if np.any(indices < -1) or np.any(indices >= len(annotations.preferences)):
            raise ValueError("invalid episode preference index")

        # Normalize ownership on CPU. Each row shares a tuple reference and
        # overwriting a row releases it independently of actor-buffer eviction.
        cpu_batch = [torch.as_tensor(field).detach().cpu().clone() for field in batch]
        self.preferences.extend(annotations.preferences)
        for transition, index, weight in zip(zip(*cpu_batch), indices, weights):
            preference = annotations.preferences[index] if index >= 0 else None
            row = (transition, preference, float(weight))
            if len(self.transitions) < self.capacity:
                self.transitions.append(row)
            else:
                self.transitions[self.next_slot] = row
            self.next_slot = (self.next_slot + 1) % self.capacity

    def _preference_batch(self, preferences) -> PreferenceBatch:
        return PreferenceBatch(*(
            torch.as_tensor(np.stack(values), dtype=torch.float32, device=self.device)
            for values in zip(*(
                (p.observation, p.human_action, p.policy_action) for p in preferences
            ))
        ))

    def sample(self, size: int, rng: np.random.Generator, *, row_offset: int = 0):
        """Return an online batch and its optional critic-correction arguments.

        Use row_offset when demo rows precede online rows in the learner batch.
        """
        if size < 1 or not self.transitions or row_offset < 0:
            raise ValueError("sampling requires a nonempty replay and valid batch size and offset")
        selected = [self.transitions[index] for index in rng.integers(len(self), size=size)]
        batch = Batch(*(torch.stack(field).to(self.device) for field in zip(*(r[0] for r in selected))))
        linked = [(row + row_offset, pref, weight)
                  for row, (_, pref, weight) in enumerate(selected) if pref is not None]
        if not linked:
            return batch, None
        rows, preferences, weights = zip(*linked)
        return batch, (
            self._preference_batch(preferences),
            torch.tensor(rows, dtype=torch.long, device=self.device),
            torch.tensor(weights, dtype=batch.rewards.dtype, device=self.device),
        )

    def sample_actor_preferences(self, size: int, rng: np.random.Generator):
        if size < 1:
            raise ValueError("preference batch size must be positive")
        if not self.preferences:
            return None
        indices = rng.integers(len(self.preferences), size=size)
        return self._preference_batch([self.preferences[index] for index in indices])
