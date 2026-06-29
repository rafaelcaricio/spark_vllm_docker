from __future__ import annotations

import pytest

from dspark_harness.scheduler import (
    cumulative_survival,
    hardware_aware_prefix_schedule,
)


def test_cumulative_survival_multiplies_conditional_confidences() -> None:
    assert cumulative_survival([0.9, 0.8, 0.5]) == pytest.approx((0.9, 0.72, 0.36))


def test_scheduler_prioritizes_high_survival_prefixes() -> None:
    result = hardware_aware_prefix_schedule(
        [
            [0.95, 0.90, 0.80],
            [0.60, 0.60, 0.60],
        ],
        steps_per_second=lambda batch_tokens: 100.0,
    )

    assert result.lengths[0] >= result.lengths[1]
    assert result.lengths == (3, 3)
    assert result.batch_tokens == 8


def test_scheduler_prunes_when_batch_capacity_drops_sharply() -> None:
    result = hardware_aware_prefix_schedule(
        [
            [0.95, 0.90, 0.80],
            [0.60, 0.60, 0.60],
        ],
        steps_per_second=lambda batch_tokens: {2: 100.0, 3: 100.0, 4: 55.0}.get(
            batch_tokens,
            30.0,
        ),
    )

    assert result.lengths == (1, 0)
    assert result.batch_tokens == 3


def test_unconstrained_search_can_cross_jagged_capacity_cliffs() -> None:
    confidence_rows = [
        [0.99, 0.99, 0.99],
        [0.99, 0.99, 0.99],
    ]
    jagged = lambda batch_tokens: {2: 100.0, 3: 50.0, 4: 150.0, 5: 140.0}.get(
        batch_tokens,
        130.0,
    )

    early = hardware_aware_prefix_schedule(
        confidence_rows,
        steps_per_second=jagged,
        early_stop=True,
    )
    unconstrained = hardware_aware_prefix_schedule(
        confidence_rows,
        steps_per_second=jagged,
        early_stop=False,
    )

    assert early.lengths == (0, 0)
    assert unconstrained.expected_throughput > early.expected_throughput
    assert sum(unconstrained.lengths) >= 2
