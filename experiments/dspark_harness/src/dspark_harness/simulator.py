from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import isclose
from random import Random
from typing import Callable, Iterable, Mapping, Sequence


Token = str
Prefix = tuple[Token, ...]
Distribution = dict[Token, float]
Scheduler = Callable[[Sequence["DraftStep"]], int]


@dataclass(frozen=True)
class DraftStep:
    token: Token
    draft_dist: Distribution
    target_dist: Distribution
    confidence: float


class ToyLanguageModel:
    """Small conditional language model for exact speculative-decoding tests."""

    def __init__(
        self,
        *,
        vocab: Sequence[Token],
        transitions: Mapping[Prefix, Distribution] | None = None,
        default: Distribution | None = None,
    ) -> None:
        if not vocab:
            raise ValueError("vocab must not be empty")
        self.vocab = tuple(vocab)
        self.transitions = {
            tuple(prefix): normalize_distribution(dist, self.vocab)
            for prefix, dist in (transitions or {}).items()
        }
        self.default = normalize_distribution(
            default or {token: 1.0 for token in self.vocab},
            self.vocab,
        )

    def next_distribution(self, prefix: Iterable[Token]) -> Distribution:
        return dict(self.transitions.get(tuple(prefix), self.default))

    def sample_next(self, prefix: Iterable[Token], rng: Random) -> Token:
        return sample_distribution(self.next_distribution(prefix), rng)


def normalize_distribution(dist: Mapping[Token, float], vocab: Sequence[Token]) -> Distribution:
    out = {token: float(dist.get(token, 0.0)) for token in vocab}
    total = sum(out.values())
    if total <= 0.0:
        raise ValueError(f"distribution has no positive mass: {dist!r}")
    return {token: probability / total for token, probability in out.items()}


def sample_distribution(dist: Mapping[Token, float], rng: Random) -> Token:
    threshold = rng.random()
    cumulative = 0.0
    last_token = None
    for token, probability in dist.items():
        last_token = token
        cumulative += probability
        if threshold <= cumulative:
            return token
    assert last_token is not None
    return last_token


def acceptance_probability(draft_dist: Mapping[Token, float], target_dist: Mapping[Token, float]) -> float:
    """Analytical per-step speculative acceptance probability."""

    return sum(min(float(draft_dist.get(t, 0.0)), float(target_dist.get(t, 0.0))) for t in target_dist)


def add_markov_bias(base_dist: Distribution, previous_token: Token, markov_bias: Mapping[tuple[Token, Token], float]) -> Distribution:
    biased = {
        token: probability * float(markov_bias.get((previous_token, token), 1.0))
        for token, probability in base_dist.items()
    }
    return normalize_distribution(biased, tuple(base_dist))


def dspark_draft(
    *,
    target_model: ToyLanguageModel,
    parallel_base_model: ToyLanguageModel,
    prefix: Prefix,
    block_size: int,
    markov_bias: Mapping[tuple[Token, Token], float] | None,
    rng: Random,
) -> list[DraftStep]:
    """Draft a DSpark-style semi-autoregressive block.

    The parallel base distribution is read from the original prefix plus a draft
    position marker. The sampled previous draft token then applies a Markov bias,
    which gives the later draft positions intra-block dependency.
    """

    if block_size < 0:
        raise ValueError("block_size must be non-negative")
    bias = markov_bias or {}
    steps: list[DraftStep] = []
    sampled_prefix: list[Token] = []
    previous_token = prefix[-1]
    for position in range(block_size):
        base_prefix = prefix + (f"<draft:{position}>",)
        base_dist = parallel_base_model.next_distribution(base_prefix)
        draft_dist = add_markov_bias(base_dist, previous_token, bias)
        token = sample_distribution(draft_dist, rng)
        target_dist = target_model.next_distribution(prefix + tuple(sampled_prefix))
        confidence = acceptance_probability(draft_dist, target_dist)
        steps.append(
            DraftStep(
                token=token,
                draft_dist=draft_dist,
                target_dist=target_dist,
                confidence=confidence,
            )
        )
        sampled_prefix.append(token)
        previous_token = token
    return steps


def fixed_length_scheduler(length: int) -> Scheduler:
    def schedule(steps: Sequence[DraftStep]) -> int:
        return min(max(int(length), 0), len(steps))

    return schedule


def confidence_threshold_scheduler(threshold: float) -> Scheduler:
    """Safe prefix scheduler that only inspects current/past prefix scores."""

    def schedule(steps: Sequence[DraftStep]) -> int:
        cumulative = 1.0
        admitted = 0
        for step in steps:
            cumulative *= step.confidence
            if cumulative < threshold:
                break
            admitted += 1
        return admitted

    return schedule


def lookahead_biased_scheduler(*, good_token: Token) -> Scheduler:
    """Unsafe scheduler used as a negative control.

    It peeks at the first sampled draft token and only verifies when the token
    value is favorable. That violates the non-anticipating property and biases
    the output distribution.
    """

    def schedule(steps: Sequence[DraftStep]) -> int:
        if not steps:
            return 0
        return 1 if steps[0].token == good_token else 0

    return schedule


def sample_speculative_cycle(
    *,
    target_model: ToyLanguageModel,
    draft_steps: Sequence[DraftStep],
    prefix: Prefix,
    scheduler: Scheduler,
    rng: Random,
) -> tuple[Token, ...]:
    verify_len = scheduler(draft_steps)
    if verify_len < 0 or verify_len > len(draft_steps):
        raise ValueError(f"scheduler returned invalid length {verify_len}")

    committed: list[Token] = []
    for accepted, step in enumerate(draft_steps[:verify_len]):
        proposed = step.token
        draft_prob = step.draft_dist[proposed]
        target_prob = step.target_dist[proposed]
        if rng.random() <= min(1.0, target_prob / max(draft_prob, 1e-12)):
            committed.append(proposed)
            continue
        residual = residual_distribution(step.target_dist, step.draft_dist)
        committed.append(sample_distribution(residual, rng))
        return tuple(committed)

    target_prefix = prefix + tuple(step.token for step in draft_steps[:verify_len])
    committed.append(target_model.sample_next(target_prefix, rng))
    return tuple(committed)


def sample_speculative_once(
    *,
    target_model: ToyLanguageModel,
    draft_steps: Sequence[DraftStep],
    prefix: Prefix,
    scheduler: Scheduler,
    rng: Random,
) -> Token:
    return sample_speculative_cycle(
        target_model=target_model,
        draft_steps=draft_steps,
        prefix=prefix,
        scheduler=scheduler,
        rng=rng,
    )[0]


def residual_distribution(target_dist: Mapping[Token, float], draft_dist: Mapping[Token, float]) -> Distribution:
    residual = {
        token: max(float(target_dist.get(token, 0.0)) - float(draft_dist.get(token, 0.0)), 0.0)
        for token in target_dist
    }
    total = sum(residual.values())
    if total <= 1e-12:
        return normalize_distribution(target_dist, tuple(target_dist))
    return {token: probability / total for token, probability in residual.items()}


def exact_speculative_distribution(
    *,
    target_model: ToyLanguageModel,
    draft_model: ToyLanguageModel,
    prefix: Prefix,
    scheduler: Scheduler,
) -> Distribution:
    """Enumerate a one-token, one-draft speculative step exactly."""

    draft_dist = draft_model.next_distribution(prefix)
    target_dist = target_model.next_distribution(prefix)
    out = {token: 0.0 for token in target_model.vocab}

    for draft_token, draft_prob in draft_dist.items():
        step = DraftStep(
            token=draft_token,
            draft_dist=draft_dist,
            target_dist=target_dist,
            confidence=acceptance_probability(draft_dist, target_dist),
        )
        if scheduler([step]) == 0:
            for token, probability in target_dist.items():
                out[token] += draft_prob * probability
            continue

        accept_prob = min(1.0, target_dist[draft_token] / max(draft_prob, 1e-12))
        out[draft_token] += draft_prob * accept_prob
        residual = residual_distribution(target_dist, draft_dist)
        for token, probability in residual.items():
            out[token] += draft_prob * (1.0 - accept_prob) * probability
    return out


def empirical_distribution(samples: Iterable[Token]) -> Distribution:
    counts = Counter(samples)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot build empirical distribution from no samples")
    return {token: count / total for token, count in counts.items()}


def assert_close_distribution(
    actual: Mapping[Token, float],
    expected: Mapping[Token, float],
    *,
    abs_tol: float,
) -> None:
    tokens = set(actual) | set(expected)
    diffs = {
        token: float(actual.get(token, 0.0)) - float(expected.get(token, 0.0))
        for token in tokens
    }
    bad = {token: diff for token, diff in diffs.items() if not isclose(diff, 0.0, abs_tol=abs_tol)}
    if bad:
        raise AssertionError(f"distribution mismatch: actual={dict(actual)} expected={dict(expected)} diff={bad}")


def all_sequences(vocab: Sequence[Token], length: int) -> Iterable[tuple[Token, ...]]:
    return product(vocab, repeat=length)
