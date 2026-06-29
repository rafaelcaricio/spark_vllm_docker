from __future__ import annotations

from math import prod

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dspark_harness.scheduler import (
    brute_force_prefix_schedule,
    cumulative_survival,
    hardware_aware_prefix_schedule,
)
from dspark_harness.simulator import (
    ToyLanguageModel,
    acceptance_probability,
    assert_close_distribution,
    confidence_threshold_scheduler,
    dspark_draft,
    exact_speculative_distribution,
    fixed_length_scheduler,
)


VOCAB = ("A", "B", "C", "D")
PREFIX = ("<bos>",)


def probability_vector(size: int = 4):
    return st.lists(
        st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size,
    )


def as_dist(values: list[float]) -> dict[str, float]:
    total = sum(values)
    return {token: value / total for token, value in zip(VOCAB, values, strict=True)}


@given(target_values=probability_vector(), draft_values=probability_vector())
@settings(max_examples=1000)
def test_rejection_sampling_preserves_target_distribution_for_one_draft_token(
    target_values: list[float],
    draft_values: list[float],
) -> None:
    target_dist = as_dist(target_values)
    draft_dist = as_dist(draft_values)
    target = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: target_dist})
    draft = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: draft_dist})

    out = exact_speculative_distribution(
        target_model=target,
        draft_model=draft,
        prefix=PREFIX,
        scheduler=fixed_length_scheduler(1),
    )

    assert_close_distribution(out, target_dist, abs_tol=1e-12)


@given(
    target_values=probability_vector(),
    draft_values=probability_vector(),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=1000)
def test_confidence_threshold_pruning_is_lossless_when_non_anticipating(
    target_values: list[float],
    draft_values: list[float],
    threshold: float,
) -> None:
    target_dist = as_dist(target_values)
    draft_dist = as_dist(draft_values)
    target = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: target_dist})
    draft = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: draft_dist})

    out = exact_speculative_distribution(
        target_model=target,
        draft_model=draft,
        prefix=PREFIX,
        scheduler=confidence_threshold_scheduler(threshold),
    )

    assert_close_distribution(out, target_dist, abs_tol=1e-12)


@given(confidences=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=8))
@settings(max_examples=1000)
def test_prefix_survival_is_chain_rule_product(confidences: list[float]) -> None:
    survivals = cumulative_survival(confidences)

    for idx, survival in enumerate(survivals, start=1):
        assert survival == prod(confidences[:idx])


@given(
    base_values=probability_vector(size=3),
    target_values=probability_vector(size=3),
    previous_bias=st.lists(
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
)
@settings(max_examples=1000)
def test_dspark_confidence_matches_total_variation_acceptance_formula(
    base_values: list[float],
    target_values: list[float],
    previous_bias: list[float],
) -> None:
    vocab = ("x", "y", "z")
    base_dist = {
        token: value / sum(base_values)
        for token, value in zip(vocab, base_values, strict=True)
    }
    target_dist = {
        token: value / sum(target_values)
        for token, value in zip(vocab, target_values, strict=True)
    }
    target = ToyLanguageModel(vocab=vocab, transitions={("x",): target_dist})
    parallel_base = ToyLanguageModel(
        vocab=vocab,
        transitions={("x", "<draft:0>"): base_dist},
    )
    markov_bias = {
        ("x", token): bias
        for token, bias in zip(vocab, previous_bias, strict=True)
    }

    draft_step = dspark_draft(
        target_model=target,
        parallel_base_model=parallel_base,
        prefix=("x",),
        block_size=1,
        markov_bias=markov_bias,
        rng=__import__("random").Random(7),
    )[0]

    tv_acceptance = 1.0 - 0.5 * sum(
        abs(draft_step.draft_dist[token] - draft_step.target_dist[token])
        for token in vocab
    )
    assert draft_step.confidence == acceptance_probability(
        draft_step.draft_dist,
        draft_step.target_dist,
    )
    assert draft_step.confidence == pytest.approx(tv_acceptance)


@given(
    rows=st.lists(
        st.lists(
            st.floats(min_value=0.05, max_value=0.99, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=4,
        ),
        min_size=1,
        max_size=4,
    ),
    decay=st.floats(min_value=0.90, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=1000)
def test_unconstrained_greedy_scheduler_matches_bruteforce_for_profiled_capacity(
    rows: list[list[float]],
    decay: float,
) -> None:
    # For a fixed verification batch size, sorting by cumulative survival gives
    # the optimal prefix allocation. Searching the whole greedy path should
    # therefore match exhaustive search over all prefix lengths.
    steps_per_second = lambda batch_tokens: 100.0 * (decay ** max(batch_tokens - len(rows), 0))

    greedy = hardware_aware_prefix_schedule(
        rows,
        steps_per_second=steps_per_second,
        early_stop=False,
    )
    brute = brute_force_prefix_schedule(rows, steps_per_second=steps_per_second)

    assert greedy.expected_throughput == pytest.approx(brute.expected_throughput)
