#!/usr/bin/env python3
"""Fit DSpark STS temperatures from calibration-bin log payloads."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOG_MARKER = "DSpark STS calibration bins: "
EPS = 1.0e-6


@dataclass(frozen=True)
class BinObservation:
    count: int
    accepted: int
    confidence_sum: float

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.count

    @property
    def empirical_acceptance(self) -> float:
        return self.accepted / self.count


@dataclass(frozen=True)
class PositionFit:
    position: int
    temperature: float
    samples: int
    raw_nll: float
    calibrated_nll: float
    raw_ece: float
    calibrated_ece: float
    raw_brier: float
    calibrated_brier: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "position": self.position,
            "temperature": self.temperature,
            "samples": self.samples,
            "raw_nll": self.raw_nll,
            "calibrated_nll": self.calibrated_nll,
            "raw_ece": self.raw_ece,
            "calibrated_ece": self.calibrated_ece,
            "raw_brier": self.raw_brier,
            "calibrated_brier": self.calibrated_brier,
        }


def _new_matrix(rows: int, cols: int, value: int = 0) -> list[list[int]]:
    return [[value] * cols for _ in range(rows)]


def _new_float_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def parse_payloads(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        marker_index = line.find(LOG_MARKER)
        if marker_index < 0:
            continue
        raw = line[marker_index + len(LOG_MARKER) :].strip()
        payload, _end = decoder.raw_decode(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"expected object payload, got {type(payload).__name__}")
        payloads.append(payload)
    return payloads


def _check_matrix(
    matrix: Any,
    *,
    name: str,
    rows: int,
    cols: int,
) -> None:
    if not isinstance(matrix, list) or len(matrix) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"{name}[{row_index}] must have {cols} columns")


def aggregate_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("no DSpark STS calibration payloads found")

    first = payloads[0]
    bins = int(first["bins"])
    rows = len(first["counts"])
    counts = _new_matrix(rows, bins)
    accepted = _new_matrix(rows, bins)
    confidence_sums = _new_float_matrix(rows, bins)

    for payload in payloads:
        if int(payload["bins"]) != bins:
            raise ValueError("all payloads must use the same bin count")
        _check_matrix(payload["counts"], name="counts", rows=rows, cols=bins)
        _check_matrix(payload["accepted"], name="accepted", rows=rows, cols=bins)
        _check_matrix(
            payload["confidence_sums"],
            name="confidence_sums",
            rows=rows,
            cols=bins,
        )
        for row_index in range(rows):
            for bin_index in range(bins):
                count = int(payload["counts"][row_index][bin_index])
                accept = int(payload["accepted"][row_index][bin_index])
                conf_sum = float(payload["confidence_sums"][row_index][bin_index])
                if count < 0 or accept < 0 or accept > count:
                    raise ValueError(
                        "invalid count/accepted values at "
                        f"position={row_index} bin={bin_index}"
                    )
                if conf_sum < -EPS or conf_sum > count + EPS:
                    raise ValueError(
                        "invalid confidence sum at "
                        f"position={row_index} bin={bin_index}"
                    )
                counts[row_index][bin_index] += count
                accepted[row_index][bin_index] += accept
                confidence_sums[row_index][bin_index] += conf_sum

    return {
        "bins": bins,
        "counts": counts,
        "accepted": accepted,
        "confidence_sums": confidence_sums,
    }


def observations_for_position(
    aggregate: dict[str, Any],
    position: int,
) -> list[BinObservation]:
    counts = aggregate["counts"][position]
    accepted = aggregate["accepted"][position]
    confidence_sums = aggregate["confidence_sums"][position]
    observations: list[BinObservation] = []
    for count, accept, conf_sum in zip(
        counts,
        accepted,
        confidence_sums,
        strict=True,
    ):
        count = int(count)
        if count <= 0:
            continue
        observations.append(
            BinObservation(
                count=count,
                accepted=int(accept),
                confidence_sum=float(conf_sum),
            )
        )
    return observations


def calibrate_probability(probability: float, temperature: float) -> float:
    probability = min(max(float(probability), EPS), 1.0 - EPS)
    temperature = max(float(temperature), EPS)
    logit = math.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def _nll_for_temperature(
    observations: list[BinObservation],
    temperature: float,
) -> float:
    loss = 0.0
    samples = 0
    for observation in observations:
        q = min(
            max(
                calibrate_probability(
                    observation.mean_confidence,
                    temperature,
                ),
                EPS,
            ),
            1.0 - EPS,
        )
        loss -= observation.accepted * math.log(q)
        loss -= (observation.count - observation.accepted) * math.log(1.0 - q)
        samples += observation.count
    return loss / max(samples, 1)


def _score_observations(
    observations: list[BinObservation],
    temperature: float,
) -> tuple[float, float, float]:
    nll = _nll_for_temperature(observations, temperature)
    ece = 0.0
    brier = 0.0
    samples = sum(observation.count for observation in observations)
    if samples <= 0:
        return float("nan"), float("nan"), float("nan")
    for observation in observations:
        q = calibrate_probability(observation.mean_confidence, temperature)
        rate = observation.empirical_acceptance
        ece += observation.count * abs(q - rate)
        brier += observation.accepted * ((1.0 - q) ** 2)
        brier += (observation.count - observation.accepted) * (q**2)
    return nll, ece / samples, brier / samples


def fit_temperature(
    observations: list[BinObservation],
    *,
    min_temperature: float,
    max_temperature: float,
    iterations: int = 96,
) -> float:
    if not observations:
        return 1.0
    if min_temperature <= 0.0 or max_temperature <= min_temperature:
        raise ValueError("temperature bounds must satisfy 0 < min < max")

    lo = math.log(min_temperature)
    hi = math.log(max_temperature)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0
    left = lo + inv_phi_sq * (hi - lo)
    right = lo + inv_phi * (hi - lo)
    left_score = _nll_for_temperature(observations, math.exp(left))
    right_score = _nll_for_temperature(observations, math.exp(right))

    for _ in range(iterations):
        if left_score < right_score:
            hi = right
            right = left
            right_score = left_score
            left = lo + inv_phi_sq * (hi - lo)
            left_score = _nll_for_temperature(observations, math.exp(left))
        else:
            lo = left
            left = right
            left_score = right_score
            right = lo + inv_phi * (hi - lo)
            right_score = _nll_for_temperature(observations, math.exp(right))

    candidates = [
        min_temperature,
        max_temperature,
        math.exp((lo + hi) / 2.0),
    ]
    return min(
        candidates,
        key=lambda temperature: _nll_for_temperature(observations, temperature),
    )


def fit_positions(
    aggregate: dict[str, Any],
    *,
    min_temperature: float,
    max_temperature: float,
) -> list[PositionFit]:
    fits: list[PositionFit] = []
    rows = len(aggregate["counts"])
    for position in range(rows):
        observations = observations_for_position(aggregate, position)
        samples = sum(observation.count for observation in observations)
        temperature = fit_temperature(
            observations,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
        )
        raw_nll, raw_ece, raw_brier = _score_observations(observations, 1.0)
        calibrated_nll, calibrated_ece, calibrated_brier = _score_observations(
            observations,
            temperature,
        )
        fits.append(
            PositionFit(
                position=position,
                temperature=temperature,
                samples=samples,
                raw_nll=raw_nll,
                calibrated_nll=calibrated_nll,
                raw_ece=raw_ece,
                calibrated_ece=calibrated_ece,
                raw_brier=raw_brier,
                calibrated_brier=calibrated_brier,
            )
        )
    return fits


def _read_inputs(paths: list[str]) -> str:
    if not paths:
        return sys.stdin.read()
    parts: list[str] = []
    for raw_path in paths:
        parts.append(Path(raw_path).read_text())
    return "\n".join(parts)


def _format_temperature(value: float) -> str:
    return f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs",
        nargs="*",
        help="Files containing vLLM logs with DSpark STS calibration bins.",
    )
    parser.add_argument("--min-temperature", type=float, default=0.05)
    parser.add_argument("--max-temperature", type=float, default=20.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only.",
    )
    args = parser.parse_args()

    payloads = parse_payloads(_read_inputs(args.logs))
    aggregate = aggregate_payloads(payloads)
    fits = fit_positions(
        aggregate,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    )
    env_value = ",".join(_format_temperature(fit.temperature) for fit in fits)
    result = {
        "payloads": len(payloads),
        "bins": aggregate["bins"],
        "temperatures": [fit.temperature for fit in fits],
        "env": f"VLLM_DSPARK_STS_TEMPERATURES={env_value}",
        "positions": [fit.as_dict() for fit in fits],
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(result["env"])
    print(f"payloads={result['payloads']} bins={result['bins']}")
    for fit in fits:
        print(
            "pos={position} samples={samples} temp={temp:.6g} "
            "nll={raw_nll:.6f}->{cal_nll:.6f} "
            "ece={raw_ece:.6f}->{cal_ece:.6f} "
            "brier={raw_brier:.6f}->{cal_brier:.6f}".format(
                position=fit.position,
                samples=fit.samples,
                temp=fit.temperature,
                raw_nll=fit.raw_nll,
                cal_nll=fit.calibrated_nll,
                raw_ece=fit.raw_ece,
                cal_ece=fit.calibrated_ece,
                raw_brier=fit.raw_brier,
                cal_brier=fit.calibrated_brier,
            )
        )


if __name__ == "__main__":
    main()
