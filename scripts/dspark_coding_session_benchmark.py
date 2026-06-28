#!/usr/bin/env python3
"""Realistic coding-session benchmark for DSpark single-stream decode.

Drives a READ-ONLY agentic multi-turn conversation about this project against
the served `deepseek-v4-flash-dspark` model, capturing per-turn and per-session
decode speed + spec-decode acceptance / tau by diffing /metrics. The goal is to
reproduce the context-growth acceptance decay seen in real coding sessions
(tau ~3.0-3.4) rather than the optimistic synthetic short-prompt number
(tau ~4.27), so tau / Tverify changes can be A/B'd under realistic conditions.

The model is given ONLY read-only tools (read_file/grep/glob/list_dir) executed
locally against configured workspace roots. There is no edit/write/exec tool and
every path is containment-checked. Reuses the proven /metrics-diff helpers from
dspark_single_stream_benchmark.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from transformers import AutoTokenizer

from dspark_single_stream_benchmark import (
    dspark_quality_summary,
    http_get_text,
    interesting_metrics,
    metric_delta,
    server_max_model_len,
)
from dspark_coding_session_corpus import CORPUS, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ROOTS = [
    "/home/pieter/Code/bjk110_spark-vllm-docker",
    "/home/pieter/Code/vllm-dspark-unholy",
]
# directories never descended into by grep/glob (size + noise hygiene)
_SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".hypothesis",
    "node_modules", "build", "dist", ".ruff_cache", ".cache", "cache",
    ".mypy_cache", "site-packages", ".eggs", ".tox",
}
_MAX_FILE_BYTES = 1_500_000  # skip files larger than this in grep


# ---------------------------------------------------------------------------
# Read-only tool runtime (SAFETY-CRITICAL: no writes, ever)
# ---------------------------------------------------------------------------

def _roots_real(roots: list[str]) -> list[str]:
    out = []
    for r in roots:
        rp = os.path.realpath(os.path.expanduser(r))
        if os.path.isdir(rp):
            out.append(rp)
    # de-duplicate, preserve order
    seen = set()
    deduped = []
    for r in out:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def _contained(real_path: str, roots_real: list[str]) -> bool:
    rp = os.path.realpath(real_path)
    return any(rp == r or rp.startswith(r + os.sep) for r in roots_real)


def _resolve(path: str, roots_real: list[str]) -> str | None:
    """Resolve a (relative-to-any-root or absolute) path to a contained realpath."""
    if not path:
        return None
    p = os.path.expanduser(path.strip().strip("`"))
    candidates = [p] if os.path.isabs(p) else [os.path.join(r, p) for r in roots_real]
    for c in candidates:
        try:
            rp = os.path.realpath(c)
        except OSError:
            continue
        if _contained(rp, roots_real) and os.path.exists(rp):
            return rp
    return None


def _resolve_dir(path: str | None, roots_real: list[str]) -> list[str]:
    """Resolve a scope dir for grep/glob; defaults to all roots."""
    if not path:
        return list(roots_real)
    rp = _resolve(path, roots_real)
    if rp is None:
        return []
    return [rp] if os.path.isdir(rp) else [os.path.dirname(rp)]


def tool_read_file(args: dict[str, Any], roots_real: list[str], line_cap: int) -> str:
    path = args.get("path", "")
    rp = _resolve(path, roots_real)
    if rp is None:
        return f"ERROR: path not found or outside workspace: {path!r}"
    if os.path.isdir(rp):
        return f"ERROR: {rp} is a directory; use list_dir instead."
    try:
        with open(rp, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"ERROR: cannot read {rp}: {exc}"
    n = len(lines)
    start = int(args.get("start_line") or 1)
    end = int(args.get("end_line") or n)
    s = max(0, min(start - 1, n))
    e = max(s, min(end, n))
    capped = (e - s) > line_cap
    if capped:
        e = s + line_cap
    body = "".join(f"{i + 1:6d}\t{lines[i]}" for i in range(s, e))
    note = f"\n... [truncated at {line_cap} lines of {n}; pass start_line to continue]" if capped else ""
    return f"[{rp}] ({n} lines)\n{body}{note}"


def tool_grep(args: dict[str, Any], roots_real: list[str]) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "ERROR: pattern is required"
    try:
        cre = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: invalid regex {pattern!r}: {exc}"
    scope = _resolve_dir(args.get("path"), roots_real)
    if not scope:
        return f"ERROR: scope path not found/contained: {args.get('path')!r}"
    max_results = max(1, min(int(args.get("max_results") or 50), 200))
    results: list[str] = []
    for root in scope:
        for dirpath, dirs, files in os.walk(root):
            if not _contained(os.path.realpath(dirpath), roots_real):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in files:
                fp = os.path.join(dirpath, fn)
                try:
                    if not _contained(os.path.realpath(fp), roots_real):
                        continue
                    if os.path.getsize(fp) > _MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if cre.search(line):
                                results.append(f"{fp}:{i}:{line.rstrip()[:240]}")
                                if len(results) >= max_results:
                                    return "\n".join(results) + f"\n... [truncated at {max_results} matches]"
                except OSError:
                    continue
    return "\n".join(results) if results else "(no matches)"


def tool_glob(args: dict[str, Any], roots_real: list[str]) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "ERROR: pattern is required"
    scope = _resolve_dir(args.get("path"), roots_real)
    if not scope:
        return f"ERROR: scope path not found/contained: {args.get('path')!r}"
    matches: list[str] = []
    for root in scope:
        for m in _glob.glob(os.path.join(root, "**", pattern), recursive=True):
            rp = os.path.realpath(m)
            if _contained(rp, roots_real):
                matches.append(rp)
            if len(matches) >= 200:
                break
    return "\n".join(matches) if matches else "(no matches)"


def tool_list_dir(args: dict[str, Any], roots_real: list[str]) -> str:
    path = args.get("path", "")
    rp = _resolve(path, roots_real)
    if rp is None:
        # allow listing a root by name fragment
        for r in roots_real:
            if os.path.basename(r) == path or r.endswith(path):
                rp = r
                break
    if rp is None or not os.path.isdir(rp):
        return f"ERROR: directory not found/contained: {path!r}"
    try:
        entries = sorted(os.scandir(rp), key=lambda e: (not e.is_dir(), e.name))
    except OSError as exc:
        return f"ERROR: {exc}"
    out = [f"[{rp}]"]
    for e in entries[:200]:
        kind = "d" if e.is_dir() else ("l" if e.is_symlink() else "f")
        out.append(f"{kind}  {e.name}")
    return "\n".join(out)


def execute_tool(name: str, arguments: dict[str, Any], roots_real: list[str], line_cap: int) -> str:
    """Dispatch ONE read-only tool call. No write/edit/exec path exists here."""
    if not isinstance(arguments, dict):
        arguments = {}
    if name == "read_file":
        return tool_read_file(arguments, roots_real, line_cap)
    if name == "grep":
        return tool_grep(arguments, roots_real)
    if name == "glob":
        return tool_glob(arguments, roots_real)
    if name == "list_dir":
        return tool_list_dir(arguments, roots_real)
    return f"ERROR: unknown tool {name!r} (only read-only tools are available)"


# ---------------------------------------------------------------------------
# OpenAI tool schemas (read-only only)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the project (read-only). Returns lines with 1-based line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, repo-root-relative or absolute."},
                    "start_line": {"type": "integer", "description": "1-based first line (default 1)."},
                    "end_line": {"type": "integer", "description": "1-based last line (default EOF)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex search file contents under a scope (read-only). Returns file:line:match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex."},
                    "path": {"type": "string", "description": "Optional scope dir/file (repo-relative or absolute). Default: whole workspace."},
                    "max_results": {"type": "integer", "description": "Max matches (default 50)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by name pattern (read-only). Supports ** for recursive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Filename glob, e.g. '*.py' or 'dspark*.py'."},
                    "path": {"type": "string", "description": "Optional scope dir."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory (read-only).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path (repo-relative or absolute)."}},
                "required": ["path"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool-call streaming over /v1/chat/completions
# ---------------------------------------------------------------------------

def post_chat_turn(
    base_url: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    thinking: str,
    cache_salt: str,
    timeout: float,
) -> dict[str, Any]:
    """Stream one chat completion, accumulating text + tool_call deltas.

    Returns finish_reason, text, parsed tool_calls (with both parsed args and raw
    argument string), the OpenAI-format assistant tool_calls message fragment,
    and the same timing anchors as post_stream_openai.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if thinking != "default":
        payload["chat_template_kwargs"] = {"thinking": thinking == "true"}
    if cache_salt:
        payload["cache_salt"] = cache_salt

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    text_parts: list[str] = []
    tc_acc: dict[int, dict[str, str]] = {}
    finish_reason = "stop"
    first_byte_at: float | None = None
    first_chunk_at: float | None = None
    start = time.perf_counter()

    try:
        resp_ctx = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
        except OSError:
            pass
        raise RuntimeError(f"chat HTTP {exc.code} {exc.reason}: {body}") from exc

    with resp_ctx as resp:
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
            try:
                event = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            choice = (event.get("choices") or [{}])[0]
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr
            delta = choice.get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if text:
                if first_chunk_at is None:
                    first_chunk_at = now
                text_parts.append(text)
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                slot = tc_acc.setdefault(idx, {"id": tc.get("id") or f"call_{idx}", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
    end = time.perf_counter()

    text = "".join(text_parts)
    tool_calls: list[dict[str, Any]] = []
    for idx in sorted(tc_acc):
        slot = tc_acc[idx]
        raw_args = slot["args"]
        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed = {}
        tool_calls.append(
            {"id": slot["id"], "name": slot["name"], "arguments": parsed, "arguments_raw": raw_args}
        )
    tool_calls_msg = [
        {
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments_raw"] or "{}"},
        }
        for tc in tool_calls
    ]
    decode_elapsed = max(end - (first_chunk_at or end), 0.0) if first_chunk_at else 0.0
    return {
        "finish_reason": finish_reason,
        "text": text,
        "tool_calls": tool_calls,
        "tool_calls_msg": tool_calls_msg,
        "first_byte_at": first_byte_at,
        "first_chunk_at": first_chunk_at,
        "decode_elapsed": decode_elapsed,
        "start": start,
        "end": end,
    }


# ---------------------------------------------------------------------------
# Token counting for the growing multi-turn context
# ---------------------------------------------------------------------------

def count_messages_tokens(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return int(len(ids))
    except Exception:
        return sum(
            len(tokenizer.encode(str(m.get("content") or ""), add_special_tokens=False))
            for m in messages
        )


def count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Turn + session loops
# ---------------------------------------------------------------------------

def run_turn(
    args: argparse.Namespace,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    base_url: str,
    roots_real: list[str],
    context_tokens: int,
) -> dict[str, Any]:
    """Run one user turn: a tool sub-loop until the model answers (finish=stop).

    Diffs /metrics around the whole turn for decode speed + acceptance.
    """
    metrics_before = http_get_text(base_url.rstrip("/") + "/metrics", timeout=10.0)
    turn_start = time.perf_counter()

    tool_rounds = 0
    tool_call_log: list[dict[str, Any]] = []
    total_decode_elapsed = 0.0
    first_chunk_at: float | None = None
    finish = None
    force_finish = False  # once set, send tool-less requests so the model must answer

    while True:
        resp = post_chat_turn(
            base_url,
            messages,
            [] if force_finish else TOOL_SCHEMAS,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking=args.thinking,
            cache_salt=args.cache_salt,
            timeout=args.timeout,
        )
        if first_chunk_at is None:
            first_chunk_at = resp["first_chunk_at"]
        total_decode_elapsed += resp["decode_elapsed"]
        finish = resp["finish_reason"]

        if resp["tool_calls"] and not force_finish:
            messages.append(
                {
                    "role": "assistant",
                    "content": resp["text"] or "",
                    "tool_calls": resp["tool_calls_msg"],
                }
            )
            for tc in resp["tool_calls"]:
                result = execute_tool(tc["name"], tc["arguments"], roots_real, args.read_file_line_cap)
                tool_call_log.append(
                    {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "result_chars": len(result),
                        "result_preview": result[:200],
                    }
                )
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            tool_rounds += 1
            if tool_rounds >= args.max_tool_rounds_per_turn:
                # HARD cap: subsequent requests omit tools so the model cannot keep
                # calling them and must emit a text answer.
                force_finish = True
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool-round budget reached for this turn. "
                        "Answer the original question now using what you have read.",
                    }
                )
            continue
        # no tool calls (natural stop, forced finish, or length): final assistant answer
        messages.append({"role": "assistant", "content": resp["text"]})
        break

    metrics_after = http_get_text(base_url.rstrip("/") + "/metrics", timeout=10.0)
    delta = metric_delta(metrics_before, metrics_after)
    quality = dspark_quality_summary(delta)
    gen_delta = float(delta.get("generation_tokens") or 0.0)
    decode_tps = (
        max(gen_delta - 1.0, 0.0) / total_decode_elapsed
        if total_decode_elapsed > 0 and gen_delta > 0
        else None
    )
    ttft = None
    if first_chunk_at is not None:
        ttft = first_chunk_at - turn_start
    return {
        "context_tokens_at_turn_start": context_tokens,
        "tool_rounds": tool_rounds,
        "tool_calls": tool_call_log,
        "generation_tokens": gen_delta,
        "decode_elapsed_s": total_decode_elapsed,
        "time_to_first_content_s": ttft,
        "decode_tokens_per_second_server": decode_tps,
        "finish_reason": finish,
        "dspark_quality": quality,
        "metrics_before_interesting": interesting_metrics(metrics_before),
        "metrics_after_interesting": interesting_metrics(metrics_after),
    }


def _acceptance_curve(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bucket turns by context length and report mean acceptance/tau per bucket."""
    buckets = {
        "0-2k": [], "2k-4k": [], "4k-8k": [], "8k-16k": [], "16k+": [],
    }

    def bucket(ct: int) -> str:
        if ct < 2000:
            return "0-2k"
        if ct < 4000:
            return "2k-4k"
        if ct < 8000:
            return "4k-8k"
        if ct < 16000:
            return "8k-16k"
        return "16k+"

    for t in turns:
        q = t.get("dspark_quality") or {}
        acc = q.get("acceptance_rate")
        apd = q.get("accepted_tokens_per_draft")
        if acc is None:
            continue
        buckets[bucket(int(t.get("context_tokens_at_turn_start") or 0))].append((acc, apd))

    curve = []
    for name, vals in buckets.items():
        if not vals:
            continue
        accs = [v[0] for v in vals if v[0] is not None]
        apds = [v[1] for v in vals if v[1] is not None]
        curve.append(
            {
                "context_bucket": name,
                "n": len(vals),
                "mean_acceptance": statistics.fmean(accs) if accs else None,
                "mean_accepted_per_draft": statistics.fmean(apds) if apds else None,
            }
        )
    return curve


def _fmt_num(v: Any, spec: str = ".1f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def _emit_turn_progress(idx: int, total: int, ctx: int, trec: dict[str, Any]) -> None:
    """Stream one line per turn to stderr so the run is followable in real time."""
    q = trec.get("dspark_quality") or {}
    acc = q.get("acceptance_rate")
    apd = q.get("accepted_tokens_per_draft")
    dtps = trec.get("decode_tokens_per_second_server")
    tau = (apd + 1.0) if isinstance(apd, (int, float)) else None
    print(
        f"[session] turn {idx + 1}/{total} ctx={ctx}tok "
        f"rounds={trec.get('tool_rounds')} calls={len(trec.get('tool_calls', []))} "
        f"gen={trec.get('generation_tokens')} decode_tps={_fmt_num(dtps)} "
        f"accept={_fmt_num(acc, '.3f')} acc/draft={_fmt_num(apd, '.2f')} "
        f"tau~={_fmt_num(tau, '.2f')}",
        file=sys.stderr,
        flush=True,
    )


def _check_early_stop(
    args: argparse.Namespace,
    turn_index_zero_based: int,
    ctx_tokens: int,
    last_acceptance: Any,
) -> str | None:
    """Return a stop reason if the session should stop after this turn.

    Two modes: observer-driven (touch args.stop_file) and signal-driven
    (acceptance floor / context ceiling, only after args.early_stop_min_turns).
    """
    if getattr(args, "stop_file", "") and os.path.exists(args.stop_file):
        return "stop_file"
    if (turn_index_zero_based + 1) < int(getattr(args, "early_stop_min_turns", 0) or 0):
        return None
    floor = getattr(args, "early_stop_acceptance_floor", None)
    if floor is not None and last_acceptance is not None and last_acceptance < floor:
        return "acceptance_floor"
    max_ctx = getattr(args, "early_stop_max_context", None)
    if max_ctx is not None and ctx_tokens is not None and ctx_tokens >= max_ctx:
        return "max_context"
    return None


def run_session(
    args: argparse.Namespace,
    tokenizer: Any,
    base_url: str,
    roots_real: list[str],
    served_mml: int | None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns: list[dict[str, Any]] = []
    session_start = time.perf_counter()
    corpus = CORPUS[: args.max_turns]
    session_stop_reason = "completed"

    for idx, prompt in enumerate(corpus):
        messages.append({"role": "user", "content": prompt})
        ctx = count_messages_tokens(tokenizer, messages)
        headroom = 256
        if served_mml and ctx > served_mml - headroom:
            turns.append({"turn": idx, "skipped": "context_near_max_model_len", "context_tokens": ctx})
            session_stop_reason = "context_near_max_model_len"
            break
        trec = run_turn(args, tokenizer, messages, base_url, roots_real, ctx)
        trec["turn"] = idx
        trec["user_prompt"] = prompt
        turns.append(trec)

        # stream this turn's metrics to stderr for live progress visibility
        _emit_turn_progress(idx, len(corpus), ctx, trec)

        # early-stop when enough value is observed (observer file or signal threshold)
        q = trec.get("dspark_quality") or {}
        stop_reason = _check_early_stop(args, idx, ctx, q.get("acceptance_rate"))
        if stop_reason:
            print(f"[session] early-stop: {stop_reason}", file=sys.stderr, flush=True)
            session_stop_reason = stop_reason
            break

    session_wall = time.perf_counter() - session_start

    # aggregate over turns that produced acceptance signal
    accs = [t["dspark_quality"]["acceptance_rate"] for t in turns
            if (t.get("dspark_quality") or {}).get("acceptance_rate") is not None]
    apds = [t["dspark_quality"]["accepted_tokens_per_draft"] for t in turns
            if (t.get("dspark_quality") or {}).get("accepted_tokens_per_draft") is not None]
    tps_list = [t["decode_tokens_per_second_server"] for t in turns
                if t.get("decode_tokens_per_second_server") is not None]
    total_gen = sum(float(t.get("generation_tokens") or 0.0) for t in turns)
    total_ctx = sum(int(t.get("context_tokens_at_turn_start") or 0) for t in turns)

    aggregate = {
        "turns_completed": sum(1 for t in turns if "skipped" not in t),
        "mean_acceptance": statistics.fmean(accs) if accs else None,
        "mean_accepted_per_draft": statistics.fmean(apds) if apds else None,
        "mean_tau_est": (statistics.fmean(apds) + 1.0) if apds else None,
        "mean_decode_tok_s": statistics.fmean(tps_list) if tps_list else None,
        "total_generation_tokens": total_gen,
        "total_context_tokens_seen": total_ctx,
        "session_wall_s": session_wall,
        "effective_session_tok_s": (total_gen / session_wall) if session_wall > 0 else None,
    }
    print(
        f"[session] DONE stop={session_stop_reason} turns={aggregate['turns_completed']} "
        f"mean_accept={_fmt_num(aggregate.get('mean_acceptance'), '.3f')} "
        f"mean_acc/draft={_fmt_num(aggregate.get('mean_accepted_per_draft'), '.2f')} "
        f"mean_tau~={_fmt_num(aggregate.get('mean_tau_est'), '.2f')} "
        f"mean_decode_tps={_fmt_num(aggregate.get('mean_decode_tok_s'))} "
        f"eff_tps={_fmt_num(aggregate.get('effective_session_tok_s'))} "
        f"wall={_fmt_num(aggregate.get('session_wall_s'))}s",
        file=sys.stderr,
        flush=True,
    )
    return {
        "turns": turns,
        "stop_reason": session_stop_reason,
        "aggregate": aggregate,
        "acceptance_by_context_bucket": _acceptance_curve(turns),
    }


# ---------------------------------------------------------------------------
# Warmup + main
# ---------------------------------------------------------------------------

def warmup(args: argparse.Namespace, base_url: str, tokenizer: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for i in range(args.warmup_requests):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Reply with: ready."},
        ]
        try:
            resp = post_chat_turn(
                base_url,
                msgs,
                [],
                model=args.model,
                max_tokens=16,
                temperature=args.temperature,
                thinking=args.thinking,
                cache_salt=f"{args.cache_salt}-warmup{i + 1}",
                timeout=args.timeout,
            )
            summaries.append(
                {
                    "index": i + 1,
                    "output_chars": len(resp["text"]),
                    "finish_reason": resp["finish_reason"],
                }
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            summaries.append({"index": i + 1, "error": str(exc)})
    return summaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="deepseek-v4-flash-dspark")
    p.add_argument("--model-dir", required=True)
    p.add_argument(
        "--workspace-root",
        action="append",
        default=None,
        help="Readable repo root (repeatable). Defaults to docker repo + vLLM fork.",
    )
    p.add_argument("--max-turns", type=int, default=len(CORPUS))
    p.add_argument(
        "--stop-file",
        default="",
        help="Observer-driven early stop: if this file exists at a turn boundary, "
        "the session stops gracefully (touch the file to stop once enough value is seen).",
    )
    p.add_argument(
        "--early-stop-min-turns",
        type=int,
        default=0,
        help="Minimum turns before signal-driven early-stop can fire (default 0 = off unless a floor/ceiling is set).",
    )
    p.add_argument(
        "--early-stop-acceptance-floor",
        type=float,
        default=None,
        help="Stop once a turn's acceptance drops below this (after --early-stop-min-turns).",
    )
    p.add_argument(
        "--early-stop-max-context",
        type=int,
        default=None,
        help="Stop once context tokens reach this (after --early-stop-min-turns).",
    )
    p.add_argument("--max-tool-rounds-per-turn", type=int, default=6)
    p.add_argument("--read-file-line-cap", type=int, default=400)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--thinking",
        choices=("default", "true", "false"),
        default="false",
        help="Override chat-template thinking mode. Coding gate uses false.",
    )
    p.add_argument("--cache-salt", default="")
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--timeout", type=float, default=7200.0)
    p.add_argument("--output-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    roots_real = _roots_real(args.workspace_root or DEFAULT_ROOTS)
    if not roots_real:
        raise SystemExit("No valid workspace roots resolved.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    base_url = args.base_url.rstrip("/")
    health = http_get_text(base_url + "/health", timeout=10.0)
    served_mml = server_max_model_len(base_url, args.model)

    warmup_summaries = warmup(args, base_url, tokenizer)

    session = run_session(args, tokenizer, base_url, roots_real, served_mml)

    result = {
        "model": args.model,
        "base_url": base_url,
        "served_max_model_len": served_mml,
        "workspace_roots": roots_real,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "max_tool_rounds_per_turn": args.max_tool_rounds_per_turn,
        "read_file_line_cap": args.read_file_line_cap,
        "cache_salt": args.cache_salt,
        "warmup_requests": args.warmup_requests,
        "warmup_summaries": warmup_summaries,
        "health": health.strip(),
        "session": session,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        from pathlib import Path

        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
