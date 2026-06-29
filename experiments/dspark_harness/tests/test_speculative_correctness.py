from __future__ import annotations

from random import Random

import pytest

from dspark_harness.simulator import (
    ToyLanguageModel,
    assert_close_distribution,
    confidence_threshold_scheduler,
    dspark_draft,
    empirical_distribution,
    exact_speculative_distribution,
    fixed_length_scheduler,
    lookahead_biased_scheduler,
    sample_speculative_once,
)


VOCAB = ("A", "B", "C")
PREFIX = ("<bos>",)
TARGET_DIST = {"A": 0.7, "B": 0.2, "C": 0.1}
DRAFT_DIST = {"A": 0.2, "B": 0.5, "C": 0.3}


def test_fixed_length_speculative_decoding_is_lossless_exactly() -> None:
    target = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: TARGET_DIST})
    draft = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: DRAFT_DIST})

    out = exact_speculative_distribution(
        target_model=target,
        draft_model=draft,
        prefix=PREFIX,
        scheduler=fixed_length_scheduler(1),
    )

    assert_close_distribution(out, TARGET_DIST, abs_tol=1e-12)


def test_confidence_threshold_scheduler_is_lossless_exactly() -> None:
    target = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: TARGET_DIST})
    draft = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: DRAFT_DIST})

    verify_all = exact_speculative_distribution(
        target_model=target,
        draft_model=draft,
        prefix=PREFIX,
        scheduler=confidence_threshold_scheduler(0.0),
    )
    verify_none = exact_speculative_distribution(
        target_model=target,
        draft_model=draft,
        prefix=PREFIX,
        scheduler=confidence_threshold_scheduler(0.99),
    )

    assert_close_distribution(verify_all, TARGET_DIST, abs_tol=1e-12)
    assert_close_distribution(verify_none, TARGET_DIST, abs_tol=1e-12)


def test_lookahead_scheduler_biases_distribution() -> None:
    target = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: TARGET_DIST})
    draft = ToyLanguageModel(vocab=VOCAB, transitions={PREFIX: DRAFT_DIST})

    out = exact_speculative_distribution(
        target_model=target,
        draft_model=draft,
        prefix=PREFIX,
        scheduler=lookahead_biased_scheduler(good_token="A"),
    )

    assert out["A"] == pytest.approx(0.76)
    with pytest.raises(AssertionError):
        assert_close_distribution(out, TARGET_DIST, abs_tol=1e-12)


def test_dspark_markov_drafting_remains_lossless_empirically() -> None:
    target = ToyLanguageModel(
        vocab=("of", "course", "problem"),
        transitions={
            ("of",): {"of": 0.05, "course": 0.85, "problem": 0.10},
            ("of", "course"): {"of": 0.10, "course": 0.20, "problem": 0.70},
            ("of", "problem"): {"of": 0.20, "course": 0.20, "problem": 0.60},
        },
        default={"of": 0.20, "course": 0.40, "problem": 0.40},
    )
    parallel_base = ToyLanguageModel(
        vocab=("of", "course", "problem"),
        transitions={
            ("of", "<draft:0>"): {"of": 0.10, "course": 0.50, "problem": 0.40},
            ("of", "<draft:1>"): {"of": 0.10, "course": 0.45, "problem": 0.45},
            ("of", "<draft:2>"): {"of": 0.10, "course": 0.45, "problem": 0.45},
        },
        default={"of": 0.10, "course": 0.45, "problem": 0.45},
    )
    markov_bias = {
        ("of", "course"): 4.0,
        ("of", "problem"): 0.25,
        ("course", "problem"): 3.0,
    }
    rng = Random(20260627)
    samples = []
    for _ in range(30_000):
        draft_steps = dspark_draft(
            target_model=target,
            parallel_base_model=parallel_base,
            prefix=("of",),
            block_size=3,
            markov_bias=markov_bias,
            rng=rng,
        )
        assert draft_steps[0].draft_dist["course"] > 0.5
        samples.append(
            sample_speculative_once(
                target_model=target,
                draft_steps=draft_steps,
                prefix=("of",),
                scheduler=fixed_length_scheduler(3),
                rng=rng,
            )
        )

    assert_close_distribution(
        empirical_distribution(samples),
        target.next_distribution(("of",)),
        abs_tol=0.018,
    )
