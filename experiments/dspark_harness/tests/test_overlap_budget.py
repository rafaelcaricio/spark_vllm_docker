from __future__ import annotations

import pytest

from dspark_harness.overlap import dspark_overlap_budget


def test_dspark_v4_flash_target_layers_leave_only_tail_overlap() -> None:
    budget = dspark_overlap_budget(
        target_layer_ids=(40, 41, 42),
        num_target_layers=43,
        serialized_cycle_ms=71.8,
        draft_graph_ms=10.6,
        post_feature_tail_ms=3.0,
    )

    assert budget.feature_is_tail_dependency
    assert budget.legal_hidden_ms == pytest.approx(3.0)
    assert budget.legal_speedup == pytest.approx(0.0436, abs=0.0001)
    assert budget.theoretical_full_draft_speedup == pytest.approx(0.1732, abs=0.0001)


def test_overlap_budget_rejects_impossible_target_layer() -> None:
    with pytest.raises(ValueError, match="exceed the final target layer"):
        dspark_overlap_budget(
            target_layer_ids=(43,),
            num_target_layers=43,
            serialized_cycle_ms=71.8,
            draft_graph_ms=10.6,
            post_feature_tail_ms=3.0,
        )


def test_overlap_budget_clamps_tail_to_draft_work() -> None:
    budget = dspark_overlap_budget(
        target_layer_ids=(10,),
        num_target_layers=43,
        serialized_cycle_ms=71.8,
        draft_graph_ms=10.6,
        post_feature_tail_ms=20.0,
    )

    assert not budget.feature_is_tail_dependency
    assert budget.legal_hidden_ms == pytest.approx(10.6)
    assert budget.legal_speedup == pytest.approx(budget.theoretical_full_draft_speedup)
