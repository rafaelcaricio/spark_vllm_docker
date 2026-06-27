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
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=200_000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--cache-salt", default="")
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
    return delta


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


def chat_token_count(tokenizer: Any, content: str) -> int:
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


def build_prompt(tokenizer: Any, target_tokens: int,
                 prompt_suffix: str = "") -> tuple[str, int]:
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

    low = 0
    high = max(1, target_tokens // 4)
    while chat_token_count(tokenizer, unit * high + tail) < target_tokens:
        high *= 2

    best_text = tail
    best_count = chat_token_count(tokenizer, best_text)
    while low <= high:
        mid = (low + high) // 2
        text = unit * mid + tail
        count = chat_token_count(tokenizer, text)
        if count <= target_tokens:
            best_text = text
            best_count = count
            low = mid + 1
        else:
            high = mid - 1

    return best_text, best_count


def post_stream_chat(
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
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
            delta = choice.get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content") or ""
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


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    prompt, prompt_tokens = build_prompt(tokenizer, args.prompt_tokens,
                                         args.prompt_suffix)

    base_url = args.base_url.rstrip("/")
    health = http_get_text(base_url + "/health", timeout=10.0)
    metrics_before = http_get_text(base_url + "/metrics", timeout=10.0)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.cache_salt:
        payload["cache_salt"] = args.cache_salt
    stream = post_stream_chat(base_url, payload, timeout=args.timeout)
    metrics_after = http_get_text(base_url + "/metrics", timeout=10.0)
    metrics_delta = metric_delta(metrics_before, metrics_after)

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
        "prompt_suffix": args.prompt_suffix,
        "cache_salt": args.cache_salt,
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
