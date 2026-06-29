#!/usr/bin/env python3
"""Build a DSpark hardware-scheduler SPS curve from benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


DEFAULT_LENGTH_PATTERN = r"(?:^|[_-])(?:len|length|force|forced|fl)(\d+)(?:[_-]|$)"


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def infer_forced_length(path: Path, data: dict[str, Any], pattern: re.Pattern[str]) -> int:
    for candidate in (
        data.get("forced_draft_length"),
        data.get("force_draft_length"),
        data.get("dspark_forced_draft_length"),
    ):
        if candidate is not None:
            return int(candidate)

    match = pattern.search(path.stem)
    if match:
        return int(match.group(1))

    draft_tokens_per_draft = nested_get(
        data,
        ("dspark_quality", "draft_tokens_per_draft"),
    )
    if draft_tokens_per_draft is not None:
        rounded = round(float(draft_tokens_per_draft))
        if math.isclose(float(draft_tokens_per_draft), rounded, abs_tol=1e-6):
            return int(rounded)

    raise ValueError(
        f"could not infer forced draft length for {path}. Include len<N> in "
        "the filename or pass benchmark JSONs with forced_draft_length metadata."
    )


def load_points(
    paths: list[Path],
    *,
    length_pattern: re.Pattern[str],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        concurrency = int(data.get("concurrency", 1))
        if concurrency <= 0:
            raise ValueError(f"{path}: concurrency must be positive")

        forced_length = infer_forced_length(path, data, length_pattern)
        if forced_length < 0:
            raise ValueError(f"{path}: forced length must be non-negative")

        drafts = nested_get(data, ("dspark_quality", "drafts"))
        elapsed = nested_get(
            data,
            ("timing_summary", "wall_decode_elapsed_after_first_content_s"),
        )
        if elapsed is None:
            elapsed = data.get("decode_elapsed_after_first_content_s")
        if drafts is None or elapsed is None:
            raise ValueError(f"{path}: missing dspark_quality.drafts or decode elapsed")
        drafts = float(drafts)
        elapsed = float(elapsed)
        if drafts <= 0.0 or elapsed <= 0.0:
            raise ValueError(f"{path}: drafts and decode elapsed must be positive")

        request_drafts_per_second = drafts / elapsed
        steps_per_second = request_drafts_per_second / concurrency
        batch_tokens = concurrency * (1 + forced_length)

        points.append(
            {
                "file": str(path),
                "concurrency": concurrency,
                "forced_length": forced_length,
                "batch_tokens": batch_tokens,
                "drafts": drafts,
                "decode_elapsed_s": elapsed,
                "request_drafts_per_second": request_drafts_per_second,
                "steps_per_second": steps_per_second,
                "server_tok_s": nested_get(
                    data,
                    (
                        "timing_summary",
                        "aggregate_decode_tokens_per_second_server_metrics",
                    ),
                ),
                "acceptance_rate": nested_get(
                    data,
                    ("dspark_quality", "acceptance_rate"),
                ),
                "accepted_tokens_per_draft": nested_get(
                    data,
                    ("dspark_quality", "accepted_tokens_per_draft"),
                ),
            }
        )
    return points


def summarize_points(points: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(int(point["batch_tokens"]), []).append(point)

    curve_rows: list[dict[str, Any]] = []
    for batch_tokens in sorted(grouped):
        rows = grouped[batch_tokens]
        rates = [float(row["steps_per_second"]) for row in rows]
        mean = statistics.mean(rates)
        stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
        curve_rows.append(
            {
                "batch_tokens": batch_tokens,
                "steps_per_second_mean": mean,
                "steps_per_second_stdev": stdev,
                "cv": stdev / mean if mean else None,
                "runs": len(rows),
                "sources": [
                    {
                        "file": Path(row["file"]).name,
                        "concurrency": row["concurrency"],
                        "forced_length": row["forced_length"],
                        "steps_per_second": row["steps_per_second"],
                    }
                    for row in rows
                ],
            }
        )

    curve = ",".join(
        f"{row['batch_tokens']}:{row['steps_per_second_mean']:.6f}"
        for row in curve_rows
    )
    return {
        "curve": curve,
        "curve_rows": curve_rows,
        "points": points,
    }


def resolve_paths(patterns: list[str], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate)
            continue
        if candidate.is_absolute():
            matches = sorted(candidate.parent.glob(candidate.name))
        else:
            matches = sorted(out_dir.glob(pattern))
        paths.extend(matches)
    return sorted(dict.fromkeys(path for path in paths if path.suffix == ".json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="+",
        help="Benchmark JSON paths or glob patterns. Filenames should include len<N>.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/dspark-benchmarks",
        help="Base directory for relative glob patterns.",
    )
    parser.add_argument(
        "--length-regex",
        default=DEFAULT_LENGTH_PATTERN,
        help="Regex with one capture group for forced draft length.",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    paths = resolve_paths(args.patterns, out_dir)
    if not paths:
        raise SystemExit("no benchmark JSON files matched")

    summary = summarize_points(
        load_points(
            paths,
            length_pattern=re.compile(args.length_regex),
        )
    )

    print(f"VLLM_DSPARK_SPS_CURVE={summary['curve']}")
    for row in summary["curve_rows"]:
        cv = row["cv"]
        cv_text = "n/a" if cv is None else f"{cv:.4f}"
        print(
            "  B={batch_tokens}: sps={mean:.6f} stdev={stdev:.6f} "
            "cv={cv} runs={runs}".format(
                batch_tokens=row["batch_tokens"],
                mean=row["steps_per_second_mean"],
                stdev=row["steps_per_second_stdev"],
                cv=cv_text,
                runs=row["runs"],
            )
        )

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
