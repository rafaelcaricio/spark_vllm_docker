# DSpark Direction Review — Feedback for the Implementer

Date: 2026-06-28 (revised twice same day: scope clarified, then §5.2 corrected
and warmed anchors recorded)
Reviewer: outside review pass (no code changes made; this is a direction note).
Reads: the DSpark paper summary, `dspark_quality_worklog.md`, the
`experiments/dspark_harness/` scheduler/simulator/tests, the single-stream
benchmark script, `.env.dspark-experiment`, sampled run outputs, and the
implementer's MTP-1/no-spec measurement.

> **Aligned goal:** maximize **single-stream coding decode speed** for DSpark on
> DeepSeek-V4-Flash, gated on **warmed MTP-1**. Single-stream is the correct
> regime (the owner's real usage); do not redirect to a concurrent benchmark.
>
> **Warmed anchors (2026-06-28, apples-to-apples):** no-spec `26.33 tok/s` ·
> MTP-1 `39.88 tok/s` · DSpark `63.41 tok/s` = **1.59× MTP-1** (2.41× no-spec).
> Paper-grade already; the remaining work is closing the gap to the physics
> ceiling.
>
> **Main path:** reduce **`Tverify` (priority) + `Tdraft`** while **preserving
> τ**, via parity-preserving target-side work — not numeric shortcuts. Scheduler
> work (incl. paper §5.2) is **secondary** for single-stream.

---

## TL;DR — three claims

1. **Single-stream is the right target; spec decoding is the right tool.** What
   was misguided was the *gate* (an absolute `>100 tok/s` with no reference) and
   the *missing baseline*. Both are now fixed: the gate is the **warmed MTP-1
   ratio**, and the baseline is measured.
2. **Measured: DSpark is 1.59× MTP-1 and 2.41× no-spec — already paper-grade
   single-stream.** The paper's higher 60–85% includes the **concurrency-only**
   load scheduler (incl. §5.2's async scheduler), which is structurally inert
   here. **Do not read the gap to 60–85% as a single-stream deficit to close** —
   that delta lives in the scheduler, which needs batching to activate.
3. **The recoverable headroom is `Tverify` and τ, with parity as the hard
   constraint.** Target verification is ~83% of the cycle. Every `Tverify`
   shortcut that changes numerics breaks τ (compressed-MLA 0.97% acceptance,
   deferred-capture 62%→39%). So the rule is *parity-preserving* target-side
   work — graph capture, autotune, fused feature capture — not numeric kernels.

The implementation is careful and correctness instincts are right (see "What's
strong"). This is a **targeting** redirect, not a criticism of the engineering.

---

## What's strong — keep doing this

- **Algorithmic harness is the right instinct and well executed.** Hypothesis
  properties prove rejection sampling preserves the target distribution,
  non-anticipating threshold pruning is safe, lookahead pruning is unsafe, the
  greedy scheduler matches brute force, and **early-stop gets trapped by jagged
  capacity cliffs** — exactly the paper's appendix caveat.
- **Two real correctness bugs, both well diagnosed and fixed:** the
  rejected-context suffix cached as accepted target context (trim fix), and the
  confidence-tensor aliasing bug (detached GPU view reused → "logit-like" 0.31
  confidences → clone fix → 0.977).
- **The single most valuable engineering lesson, learned twice:** *"target-side
  kernel work cannot substitute an implementation unless it preserves
  verification numerics."* This is now the governing constraint for the main path.
- **Disciplined negative-result hygiene.** Every regressing path is kept
  flag-gated and off by default, not baked into the image.
- **The no-spec anchor's near-zero variance is a diagnostic gift:** it proves the
  DSpark run-to-run variance is spec/JIT-specific, not hardware noise — which
  points straight at graph-capture / warmup as the fix (Lever 1 below).

---

## Step 1 (DONE 2026-06-28): MTP-1 / no-spec anchor measured

| path | tok/s | vs no-spec | vs MTP-1 |
| --- | ---: | ---: | ---: |
| no-spec | `26.33` (≈0 variance) | 1.00× | 0.66× |
| MTP-1 (warmed) | `39.88` | 1.52× | 1.00× |
| DSpark-5 (warmed) | `63.41` | **2.41×** | **1.59× (+59%)** |

This confirms the thesis: **DSpark is already delivering paper-grade single-stream
speedup** (+59% over MTP-1). The earlier "can't reach `>100 tok/s`" narrative was
a phantom — we were already winning the metric that matters.

The primary gate is now the **warmed DSpark / MTP-1 ratio** (currently 1.59×),
recorded back-to-back on warmed servers with identical prompts so no-spec /
MTP-1 / DSpark are true apples-to-apples.

---

## The main path: reduce `Tverify` + `Tdraft` while preserving τ

`iter_timing` (best data in the worklog):

| stage | time |
| --- | ---: |
| target forward | ~61.0 ms |
| draft propose | ~8.6 ms |
| target postprocess logits | ~2.5 ms |
| total iteration | ~73.9 ms |

Target verification is ~83% of the cycle, so **`Tverify` is the priority**;
`Tdraft` (~12%) is secondary. The crux constraint, learned twice: **every
`Tverify` shortcut that changes numerics breaks τ**. So the main path is
*parity-preserving* target-side work — anything that perturbs the verifier's
outputs is rejected by definition. Gate every change on a verifier-output /
acceptance check before trusting it.

> **Correction to an earlier draft of this note (important):** paper §5.2 is the
> **async hardware-aware prefix scheduler** (a verification-length scheduler made
> async), *not* draft/verify pipelining. The worklog — and this note, following
> it — misread §5.2 as "async two-step" draft/verify overlap. **Scheduler work is
> secondary for single-stream** (the load-aware scheduler is inert here whether
> sync or async). Cross-cycle draft/verify overlap is *not* paper-grounded and is
> **set aside** — it may merit a separate look much later, but it is not the main
> path and must not be attributed to the paper.

### Lever 1 (primary) — `Tverify` reduction with verifier parity
Where ~83% of the cycle lives and the largest recoverable gain sits. Parity is
non-negotiable. Candidates, all bit-equivalent or graph/autotune-level:
- **Whole-cycle CUDA-graph capture of the verify path**, including the
  JIT-escaping kernels (`eagle_prepare_next_token_padded_kernel`,
  `_pack_topk_routes_prefix/post_kernel`). Also removes launch/Python overhead
  and kills the bimodal first-request variance (the no-spec anchor's flat
  variance is the proof this variance is spec/JIT-specific).
- **FlashInfer sparse-MLA decode autotune buckets** for the actual DSpark decode
  shapes (a listed custom-kernel opportunity), so verify attention uses the best
  tactic instead of an untuned fallback.
- **Cut duplicate DSpark feature materialization** at target layers [40,41,42].
  The `hc_post_mean` direction already shaved ~0.9 ms *with* parity — fuse the
  capture into the existing residual chain without recomputing `hc_post`.
- **Bit-equivalent MoE decode kernels** — explicitly NOT numeric shortcuts like
  compressed-MLA. A faster MoE path is fair game only if shown output-equal to the
  current one.

### Lever 2 (secondary, compounding) — raise/preserve τ via reference parity
More accepted tokens per cycle at the same cycle time is a direct tok/s win. Path:
the reference-parity fixes previously deprioritized — KV quant-dequant on the
no-RoPE slice, non-greedy Markov sampling at the paper's `temperature=1.0`. The
run-2 first-token collapses (~48 tok/s) are a τ-stability signal worth fixing
here. τ is hard-capped at 6 (γ=5 + bonus) by the released checkpoint, so this
lever compounds with Lever 1 but can't reach `>100` alone.

### Lever 3 (secondary) — `Tdraft` reduction
~8.6 ms, the smaller term. The fused-Markov kernel already lost to the vendor
rank-256 projection, so this needs a different approach; lower priority than
Lever 1. Worth revisiting only after Lever 1 plateaus.

### Ceiling and target
At γ=5, τ_max=6; the cycle floor is `Tverify` (~61 ms) once overhead is removed.
At the current cycle (~68 ms), realistic τ≈5 → ~74 tok/s and τ≈5.5 → ~81; with a
parity-preserving `Tverify` cut to ~58 ms and τ≈5.5, ~95 tok/s. So the
**realistic ceiling is ~2.0–2.3× MTP-1 (~80–90 tok/s)**, and **~2× MTP-1
(~80 tok/s) is the honest near-term target**. `>100 tok/s` sits at/beyond the
ceiling (needs τ≈5.5 *and* `Tverify` ≤ ~55 ms) — a stretch, not the plan.

---

## Recommended next steps (in order)

1. ~~Measure MTP-1 / no-spec baseline~~ — **DONE** (Step 1). Re-measure whenever
   the server image changes, back-to-back, warmed.
2. **Add in-harness warmup + confidence intervals** so future deltas are
   trustworthy (no more go/no-go on 3-run means with bimodal tails).
3. **Parity-preserving `Tverify` reduction (Lever 1)** — whole-cycle graph capture
   of the verify path, FlashInfer sparse-MLA autotune for DSpark shapes, fused
   DSpark feature capture at [40,41,42]. Gate every change on verifier-output /
   acceptance parity *before* trusting a speed number.
4. **Raise/preserve τ via reference parity (Lever 2)** — KV quant-dequant +
   non-greedy Markov sampling; fix the run-2 first-token collapse.
5. **`Tdraft` reduction (Lever 3)** only if 3–4 plateau.
6. Keep scheduler / §5.2-async work **secondary**; do not budget it for
   single-stream.

## What to stop doing

- Stop proposer-side A/Bs in single-stream (fused Markov, W1 replication,
  fast-output, local-argmax). Profiling says second-order.
- Stop target-kernel toggles that change verifier numerics (B12X MHC,
  sparse-indexer, W4A16 TC, compressed MLA). Lesson learned twice.
- Stop wiring the **load-aware confidence scheduler / STS / `SPS(B)` / §5.2-async
  path** for this workload — it is a concurrency feature and is inert
  single-stream. Leave it built and unit-tested.
- Stop attributing draft/verify overlap to paper §5.2 (it is the async scheduler);
  set cross-cycle overlap aside as not-paper-grounded.
- Stop treating absolute `>100 tok/s` as the definition of done; the gate is the
  warmed DSpark/MTP-1 ratio, with `>100` as a stretch beyond the ceiling.

## Smaller notes (lower priority)

- **`τ` vs "accepted/draft":** the `dspark_quality_summary` conditional math
  (`count_pos / count_{pos-1}`) correctly implements the paper's `c_k`; always
  state whether a number includes the +1 bonus.
- **Non-greedy drafting still uses greedy Markov selection** (real parity gap) —
  now promoted to Lever 2 since it is a τ lever for this workload.
- **TTFT matters for coding feel.** Track time-to-first-token separately; verify
  that Lever 1 graph-capture work does not regress TTFT.

---

## Bottom line

**The warmed MTP-1 anchor is in and it vindicates the work: DSpark is 1.59× MTP-1
and 2.41× no-spec single-stream — paper-grade.** The next budget goes to closing
the gap toward the ~2.0–2.3× ceiling (~80–90 tok/s) via **parity-preserving
`Tverify` reduction (Lever 1) + τ via reference parity (Lever 2)**, with `Tdraft`
(Lever 3) as a follow-on. Set scheduler/§5.2-async work aside (concurrency-only),
stop the proposer/kernel A/B loop (dry), and treat absolute `>100 tok/s` as a
stretch beyond the ceiling — not the definition of done.

## Sources

- [DeepSeek Releases DSpark — MarkTechPost](https://www.marktechpost.com/2026/06/27/deepseek-releases-dspark-a-speculative-decoding-framework-that-accelerates-deepseek-v4-per-user-generation-60-85-over-mtp-1/)
- [DeepSpec codebase (GitHub)](https://github.com/deepseek-ai/DeepSpec)
