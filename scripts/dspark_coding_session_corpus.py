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
    "in concrete file:line references. Be precise; when asked for detail, explain "
    "thoroughly and step by step rather than tersely. The project spans two repos: "
    "the docker/compose/benchmark wrapper and the vLLM fork with the DSpark "
    "speculative-decoding + B12X kernel code. Use relative paths from a repo "
    "root, or absolute paths."
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


# "Detail" corpus: every prompt asks for an EXHAUSTIVE, step-by-step explanation.
# The point is to force long, reasoning-heavy output (not code generation), which
# depresses draft acceptance toward the real coding-session regime (~tau 3.0,
# ~40 tok/s) that the terse 'main' corpus under-represents (~tau 4.0). No prompt
# asks the model to write or modify code.
CORPUS_DETAIL: list[str] = [
    # --- orient, then explain a cycle in full detail ---
    "Locate the DSpark proposer and verifier in the vLLM fork, READ them, and "
    "then explain in EXHAUSTIVE detail how a single draft-verify cycle works end "
    "to end. Walk through every stage, the tensors passed between stages, exactly "
    "where and how the Markov head is applied, and why each step is necessary. Be "
    "thorough and step by step; do not write any code.",
    # --- explain a mechanism in depth ---
    "READ the B12X W4A16 MoE selector code and explain in FULL detail how it "
    "chooses a tile configuration for a given shape. Explain what each parameter "
    "(tile_k, tile_n, cta_threads, blocks_per_sm) controls, the trade-offs it "
    "balances, the edge cases, and the reasoning a maintainer would use to safely "
    "override it. Cover everything; do not write code.",
    # --- trace a full request path in detail ---
    "Trace, in COMPLETE detail, the life of a single decode request from the HTTP "
    "endpoint to the first streamed token. Walk through every queue, scheduler "
    "decision, CUDA-graph dispatch choice, and the kernels that run, explaining "
    "the reasoning behind each stage and identifying every place latency could "
    "hide. Be exhaustive.",
    # --- debug-style deep reasoning (the real failure mode) ---
    "Explain in DEPTH why DSpark draft acceptance decays during a real, long "
    "coding session while the short-prompt benchmark stays high. Enumerate EVERY "
    "candidate cause, the evidence for and against each drawn from the code and "
    "the metrics, and exactly how you would distinguish them experimentally. "
    "Ground each claim in specific files, env vars, or log lines. Be exhaustive.",
    # --- topology / reasoning ---
    "READ the TP=2 serving topology (compose, entrypoint, env) and explain in "
    "THOROUGH detail how the two ranks coordinate during decode: what data "
    "crosses the interconnect, why all-reduces appear at the exact points they "
    "do, and the full reasoning for why single-stream latency is structured the "
    "way it is. Walk through the whole picture step by step.",
    # --- invariant + failure reasoning ---
    "Explain in COMPLETE detail how rejected-context trimming works in the "
    "proposer and exactly what would break if it were removed. Describe the "
    "invariant it enforces, construct a concrete step-by-step failure sequence "
    "if it were absent, and explain the reasoning. Do not write code.",
    # --- exhaustive comparison ---
    "Compare the DSpark path and the MTP-1 path in EXHAUSTIVE detail. Enumerate "
    "every difference in draft mechanics, verify width, sampling, and latency "
    "structure, with the reasoning behind each difference and its implication for "
    "acceptance and speed. Be comprehensive; reference both source files.",
    # --- warmup / JIT reasoning ---
    "READ the kernel warmup code and explain in DEPTH the relationship between "
    "warmup coverage and first-request Triton JIT spikes: why specific shapes "
    "compile late, the reasoning behind each warmup choice, and precisely what a "
    "maintainer should check and change to close a coverage gap. Be exhaustive.",
    # --- capstone: detailed synthesis ---
    "Synthesizing everything you have read this session, explain in THOROUGH "
    "detail the three most promising parity-preserving ways to speed up "
    "single-stream DSpark decode. For each, explain the exact code path it "
    "touches, the mechanism of the speedup, the risks to acceptance, and how you "
    "would validate that acceptance is preserved. Reason through each fully.",
]


# Registry so the harness can select a corpus by name.
CORPORA: dict[str, list[str]] = {
    "main": CORPUS,
    "detail": CORPUS_DETAIL,
}
