#!/usr/bin/env python3
"""DSpark request-stability smoke tests against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


METRIC_NAMES = {
    "generation_tokens": "vllm:generation_tokens_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


@dataclass
class ChatResult:
    ok: bool
    elapsed_s: float
    content: str
    completion_tokens: int
    error: str | None = None


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> ChatResult:
    started = time.perf_counter()
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1234,
        }
        data = _post_json(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            payload,
            timeout,
        )
        elapsed = time.perf_counter() - started
        content = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return ChatResult(True, elapsed, content, completion_tokens)
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return ChatResult(
            False,
            time.perf_counter() - started,
            "",
            0,
            error=repr(exc),
        )


def scrape_metrics(base_url: str, timeout: float) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return {}

    out: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in METRIC_NAMES.values():
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, metric_name in METRIC_NAMES.items():
        delta[key] = after.get(metric_name, 0.0) - before.get(metric_name, 0.0)
    drafts = delta.get("drafts", 0.0)
    draft_tokens = delta.get("draft_tokens", 0.0)
    accepted = delta.get("accepted_tokens", 0.0)
    delta["accepted_per_draft"] = accepted / drafts if drafts > 0 else 0.0
    delta["acceptance"] = accepted / draft_tokens if draft_tokens > 0 else 0.0
    return delta


def summarize_results(results: list[ChatResult], wall_s: float) -> dict[str, Any]:
    completion_tokens = sum(result.completion_tokens for result in results)
    elapsed = [result.elapsed_s for result in results]
    return {
        "ok": all(result.ok for result in results),
        "requests": len(results),
        "completion_tokens": completion_tokens,
        "wall_s": wall_s,
        "aggregate_tok_s": completion_tokens / wall_s if wall_s > 0 else 0.0,
        "mean_request_s": statistics.fmean(elapsed) if elapsed else 0.0,
        "max_request_s": max(elapsed) if elapsed else 0.0,
        "errors": [result.error for result in results if result.error],
    }


def run_concurrent(
    *,
    base_url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    timeout: float,
    stagger_s: float = 0.0,
) -> dict[str, Any]:
    prompts = [
        (
            "Write a deterministic numbered list of practical DSpark validation "
            f"checks for request {idx}. Keep going until the token budget stops you."
        )
        for idx in range(concurrency)
    ]

    before = scrape_metrics(base_url, timeout=5)
    started = time.perf_counter()

    def worker(idx: int) -> ChatResult:
        if stagger_s > 0:
            time.sleep(idx * stagger_s)
        return chat_once(
            base_url=base_url,
            model=model,
            prompt=prompts[idx],
            max_tokens=max_tokens,
            timeout=timeout,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, idx) for idx in range(concurrency)]
        results = [future.result() for future in as_completed(futures)]

    wall_s = time.perf_counter() - started
    after = scrape_metrics(base_url, timeout=5)
    summary = summarize_results(results, wall_s)
    summary["metrics_delta"] = metric_delta(before, after)
    return summary


def run_condense(
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    churn_threads: int,
    churn_max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    expected = condense_victim_payload()
    victim_prompt = (
        "You are a byte-for-byte copy engine. Return exactly the payload between "
        "<payload> and </payload>. Do not include the tags. Do not add markdown, "
        "quotes, commentary, bullets, or trailing text.\n"
        "<payload>\n"
        f"{expected}\n"
        "</payload>"
    )
    reference = chat_once(
        base_url=base_url,
        model=model,
        prompt=victim_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if not reference.ok:
        return {"ok": False, "phase": "reference", "error": reference.error}
    reference_matches_expected = reference.content.strip() == expected

    stop = threading.Event()
    churn_results: list[ChatResult] = []

    def churn_worker(worker_id: int) -> None:
        idx = 0
        while not stop.is_set():
            result = chat_once(
                base_url=base_url,
                model=model,
                prompt=(
                    f"Return a compact deterministic greeting for churn worker "
                    f"{worker_id}, iteration {idx}."
                ),
                max_tokens=churn_max_tokens,
                timeout=timeout,
            )
            churn_results.append(result)
            idx += 1

    before = scrape_metrics(base_url, timeout=5)
    started = time.perf_counter()
    threads = [
        threading.Thread(target=churn_worker, args=(idx,), daemon=True)
        for idx in range(churn_threads)
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    victim = chat_once(
        base_url=base_url,
        model=model,
        prompt=victim_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    stop.set()
    for thread in threads:
        thread.join(timeout=timeout)
    wall_s = time.perf_counter() - started
    after = scrape_metrics(base_url, timeout=5)
    victim_matches_expected = victim.content.strip() == expected

    return {
        "ok": victim.ok and reference_matches_expected and victim_matches_expected,
        "victim_ok": victim.ok,
        "reference_matches_expected": reference_matches_expected,
        "victim_matches_expected": victim_matches_expected,
        "byte_for_byte_match": victim.content == reference.content,
        "reference_completion_tokens": reference.completion_tokens,
        "victim_completion_tokens": victim.completion_tokens,
        "wall_s": wall_s,
        "victim_tok_s": (
            victim.completion_tokens / victim.elapsed_s if victim.elapsed_s > 0 else 0.0
        ),
        "churn_requests": len(churn_results),
        "churn_errors": sum(1 for result in churn_results if not result.ok),
        "reference_expected_diff_offset": first_diff(
            reference.content.strip(), expected),
        "victim_expected_diff_offset": first_diff(victim.content.strip(), expected),
        "first_diff_offset": first_diff(reference.content, victim.content),
        "metrics_delta": metric_delta(before, after),
    }


def condense_victim_payload() -> str:
    lines = ["DSpark request-stable slot validation ledger"]
    lines.extend(
        f"{idx:03d}|victim|persistent-main-kv|condense-safe|draft-read-slot"
        for idx in range(40)
    )
    lines.append("END-OF-LEDGER")
    return "\n".join(lines)


def first_diff(left: str, right: str) -> int | None:
    for idx, (lch, rch) in enumerate(zip(left, right)):
        if lch != rch:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument(
        "--mode",
        choices=("static", "staggered", "condense", "all"),
        default="all",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stagger-s", type=float, default=0.25)
    parser.add_argument("--churn-threads", type=int, default=6)
    parser.add_argument("--churn-max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    payload: dict[str, Any] = {}
    if args.mode in ("static", "all"):
        payload["static"] = run_concurrent(
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    if args.mode in ("staggered", "all"):
        payload["staggered"] = run_concurrent(
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            stagger_s=args.stagger_s,
        )
    if args.mode in ("condense", "all"):
        payload["condense"] = run_condense(
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            churn_threads=args.churn_threads,
            churn_max_tokens=args.churn_max_tokens,
            timeout=args.timeout,
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(section.get("ok") for section in payload.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
