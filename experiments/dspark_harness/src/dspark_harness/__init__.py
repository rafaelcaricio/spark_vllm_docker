"""Validation helpers for DSpark-style speculative decoding."""

from .simulator import (
    DraftStep,
    ToyLanguageModel,
    acceptance_probability,
    confidence_threshold_scheduler,
    dspark_draft,
    empirical_distribution,
    exact_speculative_distribution,
    fixed_length_scheduler,
    lookahead_biased_scheduler,
    sample_speculative_cycle,
    sample_speculative_once,
)
from .scheduler import (
    ScheduleResult,
    brute_force_prefix_schedule,
    cumulative_survival,
    hardware_aware_prefix_schedule,
    score_lengths,
)

__all__ = [
    "DraftStep",
    "ToyLanguageModel",
    "acceptance_probability",
    "confidence_threshold_scheduler",
    "dspark_draft",
    "empirical_distribution",
    "exact_speculative_distribution",
    "fixed_length_scheduler",
    "lookahead_biased_scheduler",
    "sample_speculative_cycle",
    "sample_speculative_once",
    "ScheduleResult",
    "brute_force_prefix_schedule",
    "cumulative_survival",
    "hardware_aware_prefix_schedule",
    "score_lengths",
]
