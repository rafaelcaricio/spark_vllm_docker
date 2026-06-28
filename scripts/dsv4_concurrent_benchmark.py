#!/usr/bin/env python3
"""Concurrent DeepSeek V4 benchmark using the DSpark parser and prompt shapes."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from dspark_single_stream_benchmark import (
    benchmark_payload,
    build_prompt,
    dspark_quality_summary,
    http_get_text,
    interesting_metrics,
    metric_delta,
    post_stream_openai,
    server_max_model_len,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--endpoint",
        choices=("chat", "completion"),
        default="chat",
    )
    parser.add_argument(
        "--scenario",
        choices=("context_confirm", "code_completion"),
        default="code_completion",
    )
    parser.add_argument(
        "--thinking",
        choices=("default", "true", "false"),
        default="false",
    )
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--stable-prompt", action="store_true")
    parser.add_argument("--cache-salt", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Send and discard this many concurrent batches before measuring.",
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def request_summary(
    tokenizer: Any,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    endpoint: str,
    request_index: int,
) -> dict[str, Any]:
    stream = post_stream_openai(
        base_url,
        path,
        payload,
        timeout=timeout,
        endpoint=endpoint,
    )
    output_tokens = len(tokenizer.encode(stream["text"], add_special_tokens=False))
    first_chunk_at = stream["first_chunk_at"]
    start = stream["start"]
    end = stream["end"]
    decode_elapsed = None
    decode_tps = None
    if first_chunk_at is not None:
        decode_elapsed = max(end - first_chunk_at, 0.0)
        if decode_elapsed > 0.0:
            decode_tps = max(output_tokens - 1, 0) / decode_elapsed
    return {
        "request_index": request_index,
        "start": start,
        "first_byte_at": stream["first_byte_at"],
        "first_chunk_at": first_chunk_at,
        "end": end,
        "stream_chunks": stream["chunk_count"],
        "output_tokens_local": output_tokens,
        "time_to_first_content_s": (
            None if first_chunk_at is None else first_chunk_at - start
        ),
        "time_to_first_byte_s": (
            None if stream["first_byte_at"] is None else stream["first_byte_at"] - start
        ),
        "decode_elapsed_after_first_content_s": decode_elapsed,
        "decode_tokens_per_second_approx": decode_tps,
        "end_to_end_elapsed_s": end - start,
        "response_preview": stream["text"][:300],
    }


def run_batch(
    tokenizer: Any,
    base_url: str,
    path: str,
    base_payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    phase: str,
) -> dict[str, Any]:
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for request_index in range(args.concurrency):
            payload = dict(base_payload)
            if args.cache_salt:
                payload["cache_salt"] = (
                    f"{args.cache_salt}-{phase}-req{request_index + 1}"
                )
            futures.append(
                executor.submit(
                    request_summary,
                    tokenizer,
                    base_url,
                    path,
                    payload,
                    args.timeout,
                    endpoint=args.endpoint,
                    request_index=request_index + 1,
                )
            )
        requests = [future.result() for future in futures]
    batch_end = time.perf_counter()
    return {
        "phase": phase,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "wall_elapsed_s": batch_end - batch_start,
        "requests": sorted(requests, key=lambda row: row["request_index"]),
    }


def batch_timing_summary(
    batch: dict[str, Any],
    metrics_delta: dict[str, Any],
    *,
    concurrency: int,
) -> dict[str, Any]:
    requests = batch["requests"]
    earliest_start = min(row["start"] for row in requests)
    latest_end = max(row["end"] for row in requests)
    first_content_times = [
        row["first_chunk_at"] for row in requests if row["first_chunk_at"] is not None
    ]
    earliest_first_content = min(first_content_times) if first_content_times else None
    wall_e2e = latest_end - earliest_start
    wall_decode = (
        None
        if earliest_first_content is None
        else max(latest_end - earliest_first_content, 0.0)
    )
    total_output_tokens_local = sum(row["output_tokens_local"] for row in requests)
    generation_tokens_delta = float(metrics_delta.get("generation_tokens") or 0.0)

    aggregate_e2e_local = (
        total_output_tokens_local / wall_e2e if wall_e2e > 0.0 else None
    )
    aggregate_decode_local = None
    aggregate_decode_server = None
    if wall_decode and wall_decode > 0.0:
        aggregate_decode_local = (
            max(total_output_tokens_local - concurrency, 0) / wall_decode
        )
        aggregate_decode_server = (
            max(generation_tokens_delta - concurrency, 0.0) / wall_decode
            if generation_tokens_delta
            else None
        )
    aggregate_e2e_server = (
        generation_tokens_delta / wall_e2e
        if wall_e2e > 0.0 and generation_tokens_delta
        else None
    )

    return {
        "earliest_start": earliest_start,
        "earliest_first_content": earliest_first_content,
        "latest_end": latest_end,
        "wall_end_to_end_elapsed_s": wall_e2e,
        "wall_decode_elapsed_after_first_content_s": wall_decode,
        "total_output_tokens_local": total_output_tokens_local,
        "generation_tokens_server_metrics": generation_tokens_delta,
        "aggregate_end_to_end_tokens_per_second_local": aggregate_e2e_local,
        "aggregate_decode_tokens_per_second_local": aggregate_decode_local,
        "aggregate_end_to_end_tokens_per_second_server_metrics": aggregate_e2e_server,
        "aggregate_decode_tokens_per_second_server_metrics": aggregate_decode_server,
        "per_user_end_to_end_tokens_per_second_local": (
            aggregate_e2e_local / concurrency if aggregate_e2e_local is not None else None
        ),
        "per_user_decode_tokens_per_second_local": (
            aggregate_decode_local / concurrency
            if aggregate_decode_local is not None
            else None
        ),
        "per_user_end_to_end_tokens_per_second_server_metrics": (
            aggregate_e2e_server / concurrency
            if aggregate_e2e_server is not None
            else None
        ),
        "per_user_decode_tokens_per_second_server_metrics": (
            aggregate_decode_server / concurrency
            if aggregate_decode_server is not None
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.warmup_batches < 0:
        raise SystemExit("--warmup-batches must be >= 0")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    visible_prompt_suffix = "" if args.stable_prompt else args.prompt_suffix
    prompt, prompt_tokens = build_prompt(
        tokenizer,
        args.prompt_tokens,
        visible_prompt_suffix,
        endpoint=args.endpoint,
        scenario=args.scenario,
    )

    base_url = args.base_url.rstrip("/")
    health = http_get_text(base_url + "/health", timeout=10.0)
    path, payload = benchmark_payload(args, prompt)

    warmup_batches = [
        run_batch(
            tokenizer,
            base_url,
            path,
            payload,
            args,
            phase=f"warmup{idx + 1}",
        )
        for idx in range(args.warmup_batches)
    ]

    metrics_before = http_get_text(base_url + "/metrics", timeout=10.0)
    measured = run_batch(
        tokenizer,
        base_url,
        path,
        payload,
        args,
        phase="measured",
    )
    metrics_after = http_get_text(base_url + "/metrics", timeout=10.0)
    delta = metric_delta(metrics_before, metrics_after)
    quality_summary = dspark_quality_summary(delta)
    timing_summary = batch_timing_summary(
        measured,
        delta,
        concurrency=args.concurrency,
    )

    result = {
        "benchmark_mode": "concurrent",
        "model": args.model,
        "base_url": base_url,
        "concurrency": args.concurrency,
        "prompt_tokens_local": prompt_tokens,
        "target_prompt_tokens": args.prompt_tokens,
        "served_max_model_len": server_max_model_len(base_url, args.model),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "endpoint": args.endpoint,
        "scenario": args.scenario,
        "thinking": args.thinking,
        "prompt_suffix": args.prompt_suffix,
        "stable_prompt": args.stable_prompt,
        "visible_prompt_suffix": visible_prompt_suffix,
        "cache_salt": args.cache_salt,
        "warmup_batches": args.warmup_batches,
        "warmup_batch_summaries": [
            {
                "phase": batch["phase"],
                "wall_elapsed_s": batch["wall_elapsed_s"],
                "total_output_tokens_local": sum(
                    row["output_tokens_local"] for row in batch["requests"]
                ),
            }
            for batch in warmup_batches
        ],
        "measured_batch": measured,
        "timing_summary": timing_summary,
        "metrics_delta": delta,
        "dspark_quality": quality_summary,
        "decode_tokens_per_second_approx": (
            timing_summary["per_user_decode_tokens_per_second_local"]
        ),
        "decode_tokens_per_second_server_metrics": (
            timing_summary["per_user_decode_tokens_per_second_server_metrics"]
        ),
        "end_to_end_output_tokens_per_second": (
            timing_summary["per_user_end_to_end_tokens_per_second_local"]
        ),
        "health": health.strip(),
        "metrics_before_interesting": interesting_metrics(metrics_before),
        "metrics_after_interesting": interesting_metrics(metrics_after),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        out.with_suffix(".metrics.before.txt").write_text(metrics_before)
        out.with_suffix(".metrics.after.txt").write_text(metrics_after)


if __name__ == "__main__":
    main()
