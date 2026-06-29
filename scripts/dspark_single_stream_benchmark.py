#!/usr/bin/env python3
"""Single-stream long-context benchmark for the DSpark experiment server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=200_000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--endpoint",
        choices=("chat", "completion"),
        default="chat",
        help="Use chat completions or raw text completions.",
    )
    parser.add_argument(
        "--scenario",
        choices=("context_confirm", "code_completion"),
        default="context_confirm",
        help="Prompt shape to benchmark.",
    )
    parser.add_argument(
        "--thinking",
        choices=("default", "true", "false"),
        default="default",
        help=(
            "Override DeepSeek chat-template thinking mode. The DSpark paper "
            "evaluates non-thinking mode."
        ),
    )
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument(
        "--stable-prompt",
        action="store_true",
        help=(
            "Keep the visible prompt fixed even when --prompt-suffix is set. "
            "Use --cache-salt for per-run cache isolation."
        ),
    )
    parser.add_argument("--cache-salt", default="")
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Send and discard this many requests before measuring metrics/time.",
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def http_get_text(url: str, timeout: float) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return f"ERROR: {exc}"


def interesting_metrics(raw: str) -> list[str]:
    lines: list[str] = []
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if "spec" in lowered or "draft" in lowered or "dspark" in lowered:
            lines.append(line)
    return lines[:200]


def metric_values(raw: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            key, value = line.rsplit(" ", 1)
            values[key] = float(value)
        except ValueError:
            continue
    return values


def metric_sum(values: dict[str, float], name: str) -> float:
    return sum(value for key, value in values.items() if key.startswith(name))


def metric_delta(before: str, after: str) -> dict[str, Any]:
    before_values = metric_values(before)
    after_values = metric_values(after)
    delta: dict[str, Any] = {}
    for result_key, metric_name in (
        ("spec_decode_num_drafts", "vllm:spec_decode_num_drafts_total"),
        ("spec_decode_draft_tokens", "vllm:spec_decode_num_draft_tokens_total"),
        ("spec_decode_accepted_tokens",
         "vllm:spec_decode_num_accepted_tokens_total"),
        ("prompt_tokens", "vllm:prompt_tokens_total"),
        ("generation_tokens", "vllm:generation_tokens_total"),
    ):
        delta[result_key] = metric_sum(after_values, metric_name) - metric_sum(
            before_values, metric_name)

    accepted_per_pos: dict[str, float] = {}
    prefix = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    for key, after_value in after_values.items():
        if not key.startswith(prefix):
            continue
        before_value = before_values.get(key, 0.0)
        marker = 'position="'
        if marker in key:
            position = key.split(marker, 1)[1].split('"', 1)[0]
            accepted_per_pos[position] = after_value - before_value
    delta["spec_decode_accepted_per_pos"] = accepted_per_pos

    drafts_by_length: dict[str, float] = {}
    prefix = "vllm:spec_decode_num_drafts_by_draft_length_total"
    for key, after_value in after_values.items():
        if not key.startswith(prefix):
            continue
        before_value = before_values.get(key, 0.0)
        marker = 'draft_length="'
        if marker in key:
            draft_length = key.split(marker, 1)[1].split('"', 1)[0]
            drafts_by_length[draft_length] = after_value - before_value
    if drafts_by_length:
        delta["spec_decode_drafts_by_draft_length"] = drafts_by_length
    return delta


def dspark_quality_summary(metrics_delta: dict[str, Any]) -> dict[str, Any]:
    drafts = float(metrics_delta.get("spec_decode_num_drafts") or 0.0)
    draft_tokens = float(metrics_delta.get("spec_decode_draft_tokens") or 0.0)
    accepted_tokens = float(
        metrics_delta.get("spec_decode_accepted_tokens") or 0.0
    )
    accepted_per_pos = metrics_delta.get("spec_decode_accepted_per_pos") or {}
    accepted_counts_by_pos = {
        str(position): float(count)
        for position, count in sorted(
            accepted_per_pos.items(),
            key=lambda item: int(item[0]),
        )
    }
    drafts_by_length = metrics_delta.get("spec_decode_drafts_by_draft_length") or {}
    draft_length_histogram = {
        str(draft_length): float(count)
        for draft_length, count in sorted(
            drafts_by_length.items(),
            key=lambda item: int(item[0]),
        )
    }
    draft_length_histogram_total = sum(draft_length_histogram.values())
    mean_scheduled_draft_length = (
        sum(int(length) * count for length, count in draft_length_histogram.items())
        / draft_length_histogram_total
        if draft_length_histogram_total > 0.0
        else None
    )
    max_observed_draft_length = max(
        [int(length) for length in draft_length_histogram]
        + [len(accepted_counts_by_pos)],
        default=0,
    )
    scheduled_draft_prune_rate = (
        1.0 - (mean_scheduled_draft_length / max_observed_draft_length)
        if mean_scheduled_draft_length is not None and max_observed_draft_length > 0
        else None
    )

    per_position_acceptance = {
        position: (count / drafts if drafts > 0.0 else None)
        for position, count in accepted_counts_by_pos.items()
    }
    conditional_acceptance: dict[str, float | None] = {}
    previous_denominator = drafts
    for position, count in accepted_counts_by_pos.items():
        conditional_acceptance[position] = (
            count / previous_denominator if previous_denominator > 0.0 else None
        )
        previous_denominator = count
    suffix_values = [
        value
        for position, value in conditional_acceptance.items()
        if int(position) > 0 and value is not None
    ]
    return {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "draft_tokens_per_draft": (
            draft_tokens / drafts if drafts > 0.0 else None
        ),
        "accepted_tokens_per_draft": (
            accepted_tokens / drafts if drafts > 0.0 else None
        ),
        "acceptance_rate": (
            accepted_tokens / draft_tokens if draft_tokens > 0.0 else None
        ),
        "draft_length_histogram": draft_length_histogram,
        "mean_scheduled_draft_length": mean_scheduled_draft_length,
        "scheduled_draft_prune_rate": scheduled_draft_prune_rate,
        "per_position_acceptance": per_position_acceptance,
        "conditional_acceptance_per_position": conditional_acceptance,
        "first_token_acceptance": conditional_acceptance.get("0"),
        "mean_suffix_conditional_acceptance": (
            sum(suffix_values) / len(suffix_values) if suffix_values else None
        ),
    }


def server_max_model_len(base_url: str, model: str) -> int | None:
    raw = http_get_text(base_url.rstrip("/") + "/v1/models", timeout=10.0)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    for entry in payload.get("data", []):
        if entry.get("id") == model:
            max_len = entry.get("max_model_len")
            return int(max_len) if max_len is not None else None
    return None


def prompt_token_count(tokenizer: Any, content: str, *, endpoint: str) -> int:
    if endpoint == "completion":
        return len(tokenizer.encode(content, add_special_tokens=False))

    messages = [{"role": "user", "content": content}]
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)
    except Exception:
        return len(tokenizer.encode(content, add_special_tokens=False))


def scenario_parts(scenario: str, prompt_suffix: str) -> tuple[str, str]:
    if scenario == "code_completion":
        marker = f"# Profile run marker: {prompt_suffix}\n" if prompt_suffix else ""
        unit = (
            marker
            + "# DSpark decode benchmark helper.\n"
            + "def acceptance_case(seed: int) -> tuple[str, int, int]:\n"
            + "    label = f\"case_{seed:04d}\"\n"
            + "    draft_tokens = 5\n"
            + "    accepted_tokens = 4 if seed % 7 else 5\n"
            + "    return label, accepted_tokens, draft_tokens\n\n"
            + "def expected_speedup(seed: int) -> float:\n"
            + "    _label, accepted, drafted = acceptance_case(seed)\n"
            + "    return (accepted + 1) / max(drafted, 1)\n\n"
        )
        tail = (
            "\n# Continue this pytest module with deterministic checks. "
            "Output Python code only.\n\n"
            "def test_dspark_prefix_scheduler_keeps_fast_path_hot() -> None:\n"
            "    cases = [\n"
            "        (\"prefill\", 1, 5),\n"
            "        (\"decode\", 2, 5),\n"
            "        (\"verify\", 3, 5),\n"
        )
        return unit, tail

    unit = (
        "Long-context benchmark fact: DSpark speculative decoding validates "
        "draft tokens against the target model while using target-layer hidden "
        "features, Markov correction, and confidence estimates. "
    )
    if prompt_suffix:
        unit = f"Profile run marker {prompt_suffix}. " + unit
    tail = (
        "\n\nUse the preceding repeated benchmark facts as inert context. "
        "Answer in one concise sentence: confirm the context was received."
    )
    if prompt_suffix:
        tail += f"\n\nProfile run marker: {prompt_suffix}"
    return unit, tail


def build_prompt(
    tokenizer: Any,
    target_tokens: int,
    prompt_suffix: str = "",
    *,
    endpoint: str = "chat",
    scenario: str = "context_confirm",
) -> tuple[str, int]:
    unit, tail = scenario_parts(scenario, prompt_suffix)

    low = 0
    high = max(1, target_tokens // 4)
    while (
        prompt_token_count(tokenizer, unit * high + tail, endpoint=endpoint)
        < target_tokens
    ):
        high *= 2

    best_text = tail
    best_count = prompt_token_count(tokenizer, best_text, endpoint=endpoint)
    while low <= high:
        mid = (low + high) // 2
        text = unit * mid + tail
        count = prompt_token_count(tokenizer, text, endpoint=endpoint)
        if count <= target_tokens:
            best_text = text
            best_count = count
            low = mid + 1
        else:
            high = mid - 1

    return best_text, best_count


def post_stream_openai(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    endpoint: str,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks: list[str] = []
    first_chunk_at: float | None = None
    first_byte_at: float | None = None
    chunk_count = 0
    start = time.perf_counter()

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            now = time.perf_counter()
            if first_byte_at is None:
                first_byte_at = now
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_text = line.removeprefix("data:").strip()
            if data_text == "[DONE]":
                break
            if not data_text:
                continue
            event = json.loads(data_text)
            choice = event.get("choices", [{}])[0]
            if endpoint == "completion":
                text = choice.get("text") or ""
            else:
                delta = choice.get("delta") or {}
                text = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
            if text:
                if first_chunk_at is None:
                    first_chunk_at = now
                chunks.append(text)
                chunk_count += 1

    end = time.perf_counter()
    return {
        "start": start,
        "first_byte_at": first_byte_at,
        "first_chunk_at": first_chunk_at,
        "end": end,
        "chunk_count": chunk_count,
        "text": "".join(chunks),
    }


def benchmark_payload(args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    if args.endpoint == "completion":
        path = "/v1/completions"
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
        }
    else:
        path = "/v1/chat/completions"
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
        }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.endpoint == "chat" and args.thinking != "default":
        payload["chat_template_kwargs"] = {"thinking": args.thinking == "true"}
    if args.cache_salt:
        payload["cache_salt"] = args.cache_salt
    return path, payload


def main() -> None:
    args = parse_args()
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

    warmup_summaries: list[dict[str, Any]] = []
    for warmup_idx in range(args.warmup_requests):
        warmup_payload = dict(payload)
        if args.cache_salt:
            warmup_payload["cache_salt"] = f"{args.cache_salt}-warmup{warmup_idx + 1}"
        warmup_stream = post_stream_openai(
            base_url,
            path,
            warmup_payload,
            timeout=args.timeout,
            endpoint=args.endpoint,
        )
        warmup_output_tokens = len(
            tokenizer.encode(warmup_stream["text"], add_special_tokens=False)
        )
        warmup_summaries.append({
            "index": warmup_idx + 1,
            "output_tokens_local": warmup_output_tokens,
            "end_to_end_elapsed_s": warmup_stream["end"] - warmup_stream["start"],
            "stream_chunks": warmup_stream["chunk_count"],
        })

    metrics_before = http_get_text(base_url + "/metrics", timeout=10.0)
    stream = post_stream_openai(
        base_url,
        path,
        payload,
        timeout=args.timeout,
        endpoint=args.endpoint,
    )
    metrics_after = http_get_text(base_url + "/metrics", timeout=10.0)
    metrics_delta = metric_delta(metrics_before, metrics_after)
    quality_summary = dspark_quality_summary(metrics_delta)

    output_ids = tokenizer.encode(stream["text"], add_special_tokens=False)
    output_tokens = len(output_ids)
    first_chunk_at = stream["first_chunk_at"]
    first_byte_at = stream["first_byte_at"]
    end = stream["end"]
    start = stream["start"]

    ttft = None if first_chunk_at is None else first_chunk_at - start
    first_byte_latency = None if first_byte_at is None else first_byte_at - start
    decode_elapsed = None if first_chunk_at is None else max(end - first_chunk_at, 0.0)
    decode_tps = None
    if decode_elapsed and decode_elapsed > 0:
        decode_tps = max(output_tokens - 1, 0) / decode_elapsed
    server_decode_tps = None
    generation_tokens_delta = metrics_delta.get("generation_tokens")
    if decode_elapsed and decode_elapsed > 0 and generation_tokens_delta:
        server_decode_tps = max(generation_tokens_delta - 1, 0) / decode_elapsed

    result = {
        "model": args.model,
        "base_url": base_url,
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
        "warmup_requests": args.warmup_requests,
        "warmup_summaries": warmup_summaries,
        "output_tokens_local": output_tokens,
        "stream_chunks": stream["chunk_count"],
        "time_to_first_content_s": ttft,
        "time_to_first_byte_s": first_byte_latency,
        "decode_elapsed_after_first_content_s": decode_elapsed,
        "decode_tokens_per_second_approx": decode_tps,
        "decode_tokens_per_second_server_metrics": server_decode_tps,
        "end_to_end_elapsed_s": end - start,
        "end_to_end_output_tokens_per_second": (
            output_tokens / (end - start) if end > start else None
        ),
        "metrics_delta": metrics_delta,
        "dspark_quality": quality_summary,
        "health": health.strip(),
        "metrics_before_interesting": interesting_metrics(metrics_before),
        "metrics_after_interesting": interesting_metrics(metrics_after),
        "response_preview": stream["text"][:500],
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
