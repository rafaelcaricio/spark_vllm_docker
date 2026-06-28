#!/usr/bin/env python3
"""Scripted prompt corpus + system prompt for the DSpark coding-session benchmark.

These prompts drive a read-only agentic multi-turn conversation about *this*
project (the bjk110_spark-vllm-docker wrapper repo and the vllm-dspark-unholy
fork). They are ordered locate -> read -> trace -> debug so that context and
file reads compound across turns, reproducing the context-growth acceptance
decay seen in real coding sessions.

Each prompt is grounded in real files/anchors so the model must use its
read-only tools (read_file/grep/glob/list_dir) against the actual repos.
"""

from __future__ import annotations

# The model is a read-only coding assistant. It MUST NOT be given any edit/write
# tool; the harness only exposes read_file/grep/glob/list_dir.
SYSTEM_PROMPT = (
    "You are a careful coding assistant helping maintain this project. You have "
    "read-only tools: read_file, grep, glob, list_dir. You CANNOT edit, write, "
    "or run code. To answer, READ the relevant files first and ground every claim "
    "in concrete file:line references. Be precise and concise. The project spans "
    "two repos: the docker/compose/benchmark wrapper and the vLLM fork with the "
    "DSpark speculative-decoding + B12X kernel code. Use relative paths from a "
    "repo root, or absolute paths."
)

# Each entry: a single user turn. Chained so later turns build on earlier reads.
CORPUS: list[str] = [
    # --- locate / orient ---
    "Locate the DSpark speculative-decoding proposer and verifier in the vLLM "
    "fork. List the relevant files and give a one-line role for each.",
    # --- read a large module and explain ---
    "Read the DSpark proposer source and explain how a draft block is proposed: "
    "how the Markov head is applied, where draft tokens are produced, and how "
    "rejected target-context suffixes are trimmed before the next verify. Cite "
    "function names and line numbers.",
    # --- trace a mechanism ---
    "How does the runtime decide whether the target verify forward runs as a "
    "captured CUDA graph (FULL/PIECEWISE) vs eager? Trace dispatch_cudagraph and "
    "note which features force eager or disable FULL.",
    # --- config / env hunt ---
    "Find every place that reads VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM, "
    "VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT, and VLLM_DSPARK_CONFIDENCE_SCHEDULER. "
    "For each, state the file, the default, and what enabling it changes.",
    # --- warmup / JIT ---
    "What does the DeepSeek V4 B12X route-pack warmup cover and why prompt-sized "
    "token shapes? Summarize _deepseek_v4_b12x_route_pack_warmup and which "
    "kernels it pre-JITs.",
    # --- compare two paths ---
    "Compare the DSpark proposer against the MTP-1 proposer: draft generation, "
    "verify width, and where they diverge. Reference both source files.",
    # --- model-architecture read (large file) ---
    "In the DeepSeek V4 DSpark model, how are target-layer hidden features at "
    "layers 40/41/42 captured for the draft, and what does the deferred-capture "
    "path change? Read the relevant model file and cite line numbers.",
    # --- wrapper repo / ops ---
    "In the docker wrapper repo, summarize the TP=2 head/worker setup: which "
    "compose service is which, how VLLM_USE_B12X_WO_PROJECTION and MTP_NUM_TOKENS "
    "are passed, and how the worker joins the head.",
    # --- debug-style (the real-session failure mode) ---
    "Suppose DSpark draft acceptance collapses under long context during a real "
    "coding session while the synthetic short-prompt benchmark looks fine. Which "
    "diagnostic flags and log lines would you check, and what would each tell "
    "you? Reference the env vars and the metrics-logging code.",
    # --- synthesis / capstone ---
    "Based on what you've read across this session, name the three highest-"
    "leverage, parity-preserving ways to speed up single-stream DSpark decode, "
    "and for each cite the specific code path it would touch.",
]
