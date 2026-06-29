from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OverlapBudget:
    """Budget for legally hiding DSpark draft work behind verifier tail work."""

    max_hidden_layer: int
    final_target_layer: int
    serialized_cycle_ms: float
    draft_graph_ms: float
    post_feature_tail_ms: float

    @property
    def feature_is_tail_dependency(self) -> bool:
        return self.max_hidden_layer >= self.final_target_layer

    @property
    def legal_hidden_ms(self) -> float:
        return min(self.draft_graph_ms, max(0.0, self.post_feature_tail_ms))

    @property
    def legal_speedup(self) -> float:
        remaining = self.serialized_cycle_ms - self.legal_hidden_ms
        if remaining <= 0.0:
            raise ValueError("overlap budget cannot consume the whole cycle")
        return self.serialized_cycle_ms / remaining - 1.0

    @property
    def theoretical_full_draft_speedup(self) -> float:
        hidden = min(self.draft_graph_ms, self.serialized_cycle_ms)
        remaining = self.serialized_cycle_ms - hidden
        if remaining <= 0.0:
            raise ValueError("draft graph cannot be larger than the whole cycle")
        return self.serialized_cycle_ms / remaining - 1.0


def dspark_overlap_budget(
    *,
    target_layer_ids: Sequence[int],
    num_target_layers: int,
    serialized_cycle_ms: float,
    draft_graph_ms: float,
    post_feature_tail_ms: float,
) -> OverlapBudget:
    if num_target_layers <= 0:
        raise ValueError("num_target_layers must be positive")
    if not target_layer_ids:
        raise ValueError("target_layer_ids must not be empty")
    if serialized_cycle_ms <= 0.0:
        raise ValueError("serialized_cycle_ms must be positive")
    if draft_graph_ms < 0.0 or post_feature_tail_ms < 0.0:
        raise ValueError("timings must be non-negative")

    max_hidden_layer = max(int(layer) for layer in target_layer_ids)
    final_target_layer = num_target_layers - 1
    if max_hidden_layer > final_target_layer:
        raise ValueError(
            "target_layer_ids cannot exceed the final target layer: "
            f"{max_hidden_layer} > {final_target_layer}"
        )
    return OverlapBudget(
        max_hidden_layer=max_hidden_layer,
        final_target_layer=final_target_layer,
        serialized_cycle_ms=float(serialized_cycle_ms),
        draft_graph_ms=float(draft_graph_ms),
        post_feature_tail_ms=float(post_feature_tail_ms),
    )
