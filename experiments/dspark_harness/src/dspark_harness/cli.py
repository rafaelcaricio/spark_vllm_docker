from __future__ import annotations

import argparse
from random import Random

from .scheduler import brute_force_prefix_schedule, hardware_aware_prefix_schedule
from .simulator import (
    ToyLanguageModel,
    assert_close_distribution,
    dspark_draft,
    empirical_distribution,
    exact_speculative_distribution,
    fixed_length_scheduler,
    sample_speculative_once,
)


def _random_distribution(vocab: tuple[str, ...], rng: Random) -> dict[str, float]:
    values = [rng.random() + 1e-6 for _ in vocab]
    total = sum(values)
    return {token: value / total for token, value in zip(vocab, values, strict=True)}


def run_stress(*, iterations: int, samples: int, seed: int) -> None:
    rng = Random(seed)
    vocab = ("A", "B", "C", "D")
    prefix = ("<bos>",)

    for idx in range(iterations):
        target_dist = _random_distribution(vocab, rng)
        draft_dist = _random_distribution(vocab, rng)
        target = ToyLanguageModel(vocab=vocab, transitions={prefix: target_dist})
        draft = ToyLanguageModel(vocab=vocab, transitions={prefix: draft_dist})
        exact = exact_speculative_distribution(
            target_model=target,
            draft_model=draft,
            prefix=prefix,
            scheduler=fixed_length_scheduler(1),
        )
        assert_close_distribution(exact, target_dist, abs_tol=1e-12)

        row_count = rng.randint(1, 4)
        rows = [
            [rng.uniform(0.05, 0.99) for _ in range(rng.randint(1, 4))]
            for _ in range(row_count)
        ]
        decay = rng.uniform(0.9, 1.0)
        steps_per_second = lambda batch_tokens, row_count=row_count, decay=decay: (
            100.0 * (decay ** max(batch_tokens - row_count, 0))
        )
        greedy = hardware_aware_prefix_schedule(
            rows,
            steps_per_second=steps_per_second,
            early_stop=False,
        )
        brute = brute_force_prefix_schedule(rows, steps_per_second=steps_per_second)
        if abs(greedy.expected_throughput - brute.expected_throughput) > 1e-9:
            raise AssertionError((idx, rows, decay, greedy, brute))

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
    sample_rng = Random(seed + 1)
    outputs = []
    for _ in range(samples):
        draft_steps = dspark_draft(
            target_model=target,
            parallel_base_model=parallel_base,
            prefix=("of",),
            block_size=3,
            markov_bias=markov_bias,
            rng=sample_rng,
        )
        outputs.append(
            sample_speculative_once(
                target_model=target,
                draft_steps=draft_steps,
                prefix=("of",),
                scheduler=fixed_length_scheduler(3),
                rng=sample_rng,
            )
        )
    assert_close_distribution(
        empirical_distribution(outputs),
        target.next_distribution(("of",)),
        abs_tol=0.012,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--samples", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()
    run_stress(iterations=args.iterations, samples=args.samples, seed=args.seed)
    print(
        "DSpark harness stress passed "
        f"(iterations={args.iterations}, samples={args.samples}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
