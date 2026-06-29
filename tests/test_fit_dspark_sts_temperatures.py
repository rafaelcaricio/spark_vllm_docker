#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "fit_dspark_sts_temperatures.py"
)
SPEC = importlib.util.spec_from_file_location("fit_dspark_sts_temperatures", SCRIPT)
assert SPEC is not None
fit_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = fit_module
SPEC.loader.exec_module(fit_module)


def make_payload(
    counts: list[list[int]],
    accepted: list[list[int]],
    confidence_sums: list[list[float]],
) -> dict:
    return {
        "bins": len(counts[0]),
        "counts": counts,
        "accepted": accepted,
        "confidence_sums": confidence_sums,
    }


def test_parse_payloads_from_vllm_log_lines() -> None:
    payload = make_payload(
        counts=[[0, 10], [2, 3]],
        accepted=[[0, 8], [1, 1]],
        confidence_sums=[[0.0, 8.0], [0.4, 2.7]],
    )
    text = "\n".join(
        [
            "ordinary log line",
            "INFO DSpark STS calibration bins: " + json.dumps(payload),
        ]
    )

    assert fit_module.parse_payloads(text) == [payload]


def test_aggregate_payloads_adds_counts_and_confidence_sums() -> None:
    first = make_payload(
        counts=[[0, 10]],
        accepted=[[0, 8]],
        confidence_sums=[[0.0, 8.0]],
    )
    second = make_payload(
        counts=[[2, 3]],
        accepted=[[1, 1]],
        confidence_sums=[[0.4, 2.7]],
    )

    aggregate = fit_module.aggregate_payloads([first, second])

    assert aggregate["counts"] == [[2, 13]]
    assert aggregate["accepted"] == [[1, 9]]
    assert aggregate["confidence_sums"] == [[0.4, 10.7]]


def test_fit_temperature_keeps_well_calibrated_bins_near_identity() -> None:
    payload = make_payload(
        counts=[[100, 100]],
        accepted=[[20, 80]],
        confidence_sums=[[20.0, 80.0]],
    )
    aggregate = fit_module.aggregate_payloads([payload])
    fit = fit_module.fit_positions(
        aggregate,
        min_temperature=0.05,
        max_temperature=20.0,
    )[0]

    assert 0.95 < fit.temperature < 1.05
    assert fit.calibrated_ece <= fit.raw_ece + 1e-8


def test_fit_temperature_reduces_overconfident_bins() -> None:
    payload = make_payload(
        counts=[[100]],
        accepted=[[50]],
        confidence_sums=[[90.0]],
    )
    aggregate = fit_module.aggregate_payloads([payload])
    fit = fit_module.fit_positions(
        aggregate,
        min_temperature=0.05,
        max_temperature=20.0,
    )[0]

    assert fit.temperature > 5.0
    assert fit.calibrated_ece < fit.raw_ece
    assert fit.calibrated_nll < fit.raw_nll
