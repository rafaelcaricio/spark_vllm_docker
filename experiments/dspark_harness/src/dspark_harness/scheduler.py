from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product


StepCurve = Callable[[int], float]


@dataclass(frozen=True)
class ScheduleResult:
    lengths: tuple[int, ...]
    expected_accepts: float
    batch_tokens: int
    expected_throughput: float


def cumulative_survival(confidences: Sequence[float]) -> tuple[float, ...]:
    out: list[float] = []
    acc = 1.0
    for confidence in confidences:
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        acc *= float(confidence)
        out.append(acc)
    return tuple(out)


def hardware_aware_prefix_schedule(
    confidence_rows: Sequence[Sequence[float]],
    *,
    steps_per_second: StepCurve,
    early_stop: bool = True,
) -> ScheduleResult:
    """Greedy DSpark prefix scheduler over a batch of active requests.

    `confidence_rows[r][j]` is the conditional acceptance probability for
    request r at draft position j. The scheduler works with cumulative prefix
    survival probabilities and chooses per-request prefix lengths that maximize
    expected accepted tokens times the profiled engine step rate.
    """

    request_count = len(confidence_rows)
    lengths = [0] * request_count
    best_lengths = tuple(lengths)
    batch_tokens = request_count
    expected_accepts = float(request_count)
    best_batch_tokens = batch_tokens
    best_expected_accepts = expected_accepts
    best_throughput = expected_accepts * float(steps_per_second(batch_tokens))

    candidates: list[tuple[float, int, int]] = []
    for request_idx, confidences in enumerate(confidence_rows):
        for position_idx, survival in enumerate(cumulative_survival(confidences), start=1):
            if survival > 0.0:
                candidates.append((survival, request_idx, position_idx))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    for survival, request_idx, position_idx in candidates:
        if position_idx != lengths[request_idx] + 1:
            continue

        lengths[request_idx] = position_idx
        batch_tokens += 1
        expected_accepts += survival
        throughput = expected_accepts * float(steps_per_second(batch_tokens))

        if throughput > best_throughput:
            best_lengths = tuple(lengths)
            best_batch_tokens = batch_tokens
            best_expected_accepts = expected_accepts
            best_throughput = throughput
            continue

        if early_stop:
            break

    return ScheduleResult(
        lengths=best_lengths,
        expected_accepts=best_expected_accepts,
        batch_tokens=best_batch_tokens,
        expected_throughput=best_throughput,
    )


def score_lengths(
    confidence_rows: Sequence[Sequence[float]],
    lengths: Sequence[int],
    *,
    steps_per_second: StepCurve,
) -> ScheduleResult:
    if len(confidence_rows) != len(lengths):
        raise ValueError("confidence_rows and lengths must have the same length")

    expected_accepts = float(len(confidence_rows))
    batch_tokens = len(confidence_rows)
    for confidences, length in zip(confidence_rows, lengths, strict=True):
        if length < 0 or length > len(confidences):
            raise ValueError(f"invalid prefix length {length}")
        survivals = cumulative_survival(confidences)
        expected_accepts += sum(survivals[:length])
        batch_tokens += length

    return ScheduleResult(
        lengths=tuple(int(length) for length in lengths),
        expected_accepts=expected_accepts,
        batch_tokens=batch_tokens,
        expected_throughput=expected_accepts * float(steps_per_second(batch_tokens)),
    )


def brute_force_prefix_schedule(
    confidence_rows: Sequence[Sequence[float]],
    *,
    steps_per_second: StepCurve,
) -> ScheduleResult:
    best: ScheduleResult | None = None
    ranges = [range(len(row) + 1) for row in confidence_rows]
    for lengths in product(*ranges):
        result = score_lengths(
            confidence_rows,
            lengths,
            steps_per_second=steps_per_second,
        )
        if best is None or result.expected_throughput > best.expected_throughput:
            best = result
    assert best is not None
    return best
