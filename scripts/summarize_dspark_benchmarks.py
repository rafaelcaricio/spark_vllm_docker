#!/usr/bin/env python3
"""Summarize DSpark single-stream benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("server_tok_s", ("decode_tokens_per_second_server_metrics",)),
    ("approx_tok_s", ("decode_tokens_per_second_approx",)),
    ("draft_acceptance", ("dspark_quality", "acceptance_rate")),
    ("accepted_per_draft", ("dspark_quality", "accepted_tokens_per_draft")),
    ("first_acceptance", ("dspark_quality", "first_token_acceptance")),
    (
        "suffix_conditional_acceptance",
        ("dspark_quality", "mean_suffix_conditional_acceptance"),
    ),
)


T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
}


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    return float(value)


def summarize(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        row: dict[str, Any] = {"file": path.name}
        row["benchmark_mode"] = data.get("benchmark_mode", "single_stream")
        row["concurrency"] = data.get("concurrency", 1)
        for name, metric_path in METRICS:
            row[name] = nested_get(data, metric_path)
        server_tok_s = row["server_tok_s"]
        accepted_per_draft = row["accepted_per_draft"]
        if server_tok_s is not None and accepted_per_draft is not None:
            cycle_tokens = 1.0 + accepted_per_draft
            row["cycle_tokens_est"] = cycle_tokens
            row["cycles_per_second_est"] = server_tok_s / cycle_tokens
            row["cycle_ms_est"] = 1000.0 / row["cycles_per_second_est"]
        else:
            row["cycle_tokens_est"] = None
            row["cycles_per_second_est"] = None
            row["cycle_ms_est"] = None
        rows.append(row)

    aggregates: dict[str, Any] = {}
    for name, _metric_path in METRICS:
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            continue
        count = len(values)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if count > 1 else 0.0
        t_critical = T_CRITICAL_95.get(count - 1, 1.96)
        ci95_half_width = (
            t_critical * stdev / math.sqrt(count) if count > 1 else 0.0
        )
        aggregates[name] = {
            "count": count,
            "mean": mean,
            "stdev": stdev,
            "cv": stdev / mean if mean else None,
            "ci95_half_width": ci95_half_width,
            "min": min(values),
            "max": max(values),
        }
    for name in ("cycle_tokens_est", "cycles_per_second_est", "cycle_ms_est"):
        values = [row[name] for row in rows if row[name] is not None]
        if values:
            count = len(values)
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if count > 1 else 0.0
            t_critical = T_CRITICAL_95.get(count - 1, 1.96)
            ci95_half_width = (
                t_critical * stdev / math.sqrt(count) if count > 1 else 0.0
            )
            aggregates[name] = {
                "count": count,
                "mean": mean,
                "stdev": stdev,
                "cv": stdev / mean if mean else None,
                "ci95_half_width": ci95_half_width,
                "min": min(values),
                "max": max(values),
            }

    return {"runs": rows, "aggregate": aggregates}


def fmt_optional(value: Any, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="+",
        help="Glob patterns or concrete JSON paths to summarize.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/dspark-benchmarks",
        help="Base directory used for relative glob patterns.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary as JSON.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    paths: list[Path] = []
    for pattern in args.patterns:
        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate)
            continue
        if not candidate.is_absolute():
            matches = sorted(out_dir.glob(pattern))
        else:
            matches = sorted(candidate.parent.glob(candidate.name))
        paths.extend(matches)

    paths = sorted(dict.fromkeys(path for path in paths if path.suffix == ".json"))
    if not paths:
        raise SystemExit("no benchmark JSON files matched")

    summary = summarize(paths)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    for row in summary["runs"]:
        print(
            "{file}: mode={benchmark_mode} c={concurrency} "
            "server_tok_s={server_tok_s} "
            "draft_acceptance={draft_acceptance} "
            "accepted_per_draft={accepted_per_draft} "
            "cycle_ms={cycle_ms} "
            "first={first_acceptance} suffix={suffix_conditional_acceptance}"
            .format(
                file=row["file"],
                benchmark_mode=row["benchmark_mode"],
                concurrency=row["concurrency"],
                server_tok_s=fmt_optional(row["server_tok_s"]),
                draft_acceptance=fmt_optional(row["draft_acceptance"]),
                accepted_per_draft=fmt_optional(row["accepted_per_draft"]),
                cycle_ms=fmt_optional(row["cycle_ms_est"], precision=3),
                first_acceptance=fmt_optional(row["first_acceptance"]),
                suffix_conditional_acceptance=fmt_optional(
                    row["suffix_conditional_acceptance"]
                ),
            )
        )

    print("aggregate:")
    for name, stats in summary["aggregate"].items():
        cv = stats["cv"]
        cv_text = "n/a" if cv is None else f"{cv:.6f}"
        print(
            f"  {name}: mean={stats['mean']:.6f} "
            f"stdev={stats['stdev']:.6f} "
            f"cv={cv_text} "
            f"ci95=±{stats['ci95_half_width']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f}"
        )


if __name__ == "__main__":
    main()
