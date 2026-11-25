"""Localize a candidate critic-correction segment before one takeover."""

from math import exp
from typing import Sequence


def select_segment(
    progress: Sequence[float],
    takeover: int,
    *,
    start: int = 0,
    window: int = 4,
    margin: float = 0.045,
    recovery_steps: int = 3,
) -> tuple[int, int] | None:
    """Return inclusive bounds within the learner-controlled prefix."""
    if not 0 <= start <= takeover < len(progress):
        raise ValueError("invalid learner prefix or takeover index")
    if window < 1 or recovery_steps < 1 or margin < 0:
        raise ValueError("invalid segment detector parameters")

    # Treat consecutive regression flags as one anomaly and use the nearest
    # anomaly start, following Appendix C.3.
    anchor = None
    previous_flag = False
    for t in range(start, takeover):
        flagged = any(
            progress[j] < progress[t] - margin
            for j in range(t + 1, min(t + window, takeover) + 1)
        )
        if flagged and not previous_flag:
            anchor = t
        previous_flag = flagged
    if anchor is None:
        return None

    for t in range(anchor + 1, takeover - recovery_steps):
        if all(progress[j] > progress[j - 1] for j in range(t + 1, t + recovery_steps + 1)):
            return anchor, t
    return anchor, takeover - 1


def correction_weights(start: int, end: int, beta: float = 3.0) -> tuple[float, ...]:
    """Increase the correction weight toward the final segment step."""
    if end < start:
        raise ValueError("segment end must be at or after its start")
    return tuple(exp(-beta * (end - t) / max(1, end - start)) for t in range(start, end + 1))
