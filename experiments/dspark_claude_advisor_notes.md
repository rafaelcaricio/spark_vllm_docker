# DSpark Advisor Notes — Full Reasoning Record

Date: 2026-06-29
Role: Claude (read-only advisor/research partner to the implementer, Codex)
Status: Living document — update as evidence accumulates.

---

## 1. The central tension: draft length vs verification length

The DSpark paper defines per-token decode latency as:

```
L = (Tdraft + Tverify) / τ
```

where `τ` is the number of tokens accepted per verification cycle. Three levers:
- **Tdraft** — make drafting faster (cheaper draft forward).
- **Tverify** — make verification cheaper (fewer tokens to verify per step).
- **τ** — make drafting better (higher acceptance → more accepted tokens per cycle).

These levers are **not independent**. The paper's key architectural insight is that
DSpark separates **draft length** (γ, how many tokens the draft model proposes) from
**verification length** (how many of those γ tokens the target actually verifies).
This separation is what enables confidence-scheduled verification: the scheduler can
choose to verify **fewer** than γ tokens (pruning the suffix), trading slightly lower
τ for lower Tverify — and the tradeoff depends on the hardware capacity curve SPS(B).

### Why `dspark_block_size=5` (γ=5) matters

The released DeepSeek-V4-Flash-DSpark checkpoint trains and ships with γ=5
(DSpark-5). This is the draft model's **block size** — it always proposes 5 tokens
internally, even if fewer are verified. Reducing γ below 5 at runtime is **not**
equivalent to the paper scheduler because:

1. The draft model computes all 5 positions regardless (`dspark_block_size=5` in the
   config; the `draft()` method always loops `range(block_size=5)`). Setting
   `MTP_NUM_TOKENS=3` or `VLLM_DSPARK_FORCE_DRAFT_LENGTH=3` only changes how many
   tokens are *verified*, not how many are *drafted*. So Tdraft is unchanged — only
   Tverify shrinks.

2. The paper's scheduler exploits this: it drafts 5 (cheap parallel backbone + weak
   Markov head) but *verifies* a variable-length prefix (1..5) per request, based on
   confidence + hardware capacity. The draft cost is fixed; the scheduler optimizes
   the verify cost.

3. Our measurements confirmed this: the forced-length curve at single-stream showed
   γ=5 (full verify) is optimal for tok/s. The γ=3 head-to-head settle (clean
   FULL@4 graph, MNS=1) measured **56.63 tok/s (−6% vs γ=5's 60.24)**. The cycle
   dropped 71→61ms (−14%) but τ dropped 4.27→3.47 (−19%), so net tok/s fell. This
   is the weight-bound forward at batch 6: verifying fewer tokens barely speeds up
   Tverify (MoE weight-load dominates), but τ drops proportionally.

**Implication for the hardware scheduler**: at single-stream (batch 6, MAX_NUM_SEQS=1),
pruning verification length is net-negative because the forward is weight-bandwidth-
bound (Tverify ≈ constant regardless of token count). The scheduler's value emerges
only under **concurrency** (batch ≥ 16-24), where the forward becomes compute-bound
(more tokens → proportionally more compute → Tverify grows), making shorter
verification genuinely cheaper.

---

## 2. Interpretation of our hardware scheduler measurements so far

### Single-stream levers exhausted (proven with evidence)

| Lever | Status | Evidence |
|---|---|---|
| Tverify (forward) | **flat** with context | cycle ≈ 70ms at 0.1k..80k context (sparse-MLA windowed) |
| τ (draft quality) | **at ceiling** | draft path audited paper-faithful; pos0 stable ~0.88; suffix decay inherent to rank-256 Markov head on novel content |
| MoE W4A16 (43% of cycle) | **tapped** | tile changes collapse acceptance; blocks_per_sm shmem-hangs; b12x selector frozen |
| Confidence pruning (γ=3) | **inert** (−6%) | clean FULL@4: 56.63 vs 60.24 tok/s; τ drops faster than cycle |
| NCCL/DBO overlap | **gated off** by design | thresholds 32/256/1024; overlap doesn't pay at batch 6 |
| Draft/verify overlap | **infeasible** | draft is feature-conditioned (`project_main(main_hidden)`) → hard data dependency on target forward → strictly sequential pipeline |
| KVQ reference parity | **neutral** | clean A/B: 60.75 tok/s / 0.642 acceptance vs 58.48 / 0.653 baseline |

Single-stream ceiling: **~60 tok/s benchmark / ~40 tok/s real coding session (1.5×
MTP-1)**. No remaining single-stream lever of meaningful size.

### Concurrency path opened (validated)

| Config | c=1 | c=2 | c=4 |
|---|---|---|---|
| MNS=1 (serialized) | 61 agg | 55 agg (flat) | 54 agg (flat) |
| MNS=2 (batched, pre-ragged) | 58 agg | **88 agg (+60% per-user vs serialized)** | 💥 crash (mixed batch) |
| MNS=2 + compact ragged | 61 agg | 82 agg | **crash resolved → stable** |

The compact ragged grouping (request_indices + index_select/index_copy in
`store_main_kv`) is the correct fix for mixed prefill+verify batches. It bypasses
the `_view_by_request` uniform-length assertion by slicing via `query_starts`
directly, groups by chunk length, and maps compact groups back to cache slots.

### SPS(B) curve profiled (first pass)

From c=4 forced-length sweep (MNS=4) + c=8 full-length:

| B (batch tokens) | SPS (steps/sec) | Source |
|---:|---:|---|
| 4 | **invalid** (excluded) | forced_length=0: draft model still computed γ=5; only 4 drafts in 26s |
| 8 | 7.235 | c=4, γ=1 |
| 12 | 7.749 | c=4, γ=2 |
| 16 | 6.334 | c=4, γ=3 |
| 20 | 6.767 | c=4, γ=4 |
| 24 | 7.280 | c=4, γ=5 (full) |
| 48 | 5.397 | c=8, γ=5 (full) |

**Key observations:**
1. The curve is **jagged** (non-monotonic): B=12→16 drops (7.7→6.3), B=16→24
   recovers (6.3→7.3). This is atypical — SPS(B) should be roughly monotonic
   decreasing. Possible causes: measurement noise (check CV), cudagraph-capture
   boundary effects (B=16 may hit a different capture bucket than B=12/B=20), or
   ragged-grouping overhead (more groups at certain B values).

2. The B=24→48 drop (7.3→5.4) shows the GPU saturating between batch 24 and 48 —
   the expected capacity cliff.

3. The B=4 point was invalid (the forced-length override didn't suppress drafting
   when forced_length=0 — the draft model still computed `dspark_block_size=5`
   tokens internally). Excluding B=4 is correct.

---

## 3. SPS curve / stepwise lookup / early_stop=0: soundness assessment

### Stepwise floor lookup (no interpolation)

The `_steps_per_second` method (dspark_proposer.py:252-263) implements:
```python
selected_rate = curve[0][1]  # floor = first (smallest) entry
for profiled_tokens, rate in curve:
    if batch_tokens < profiled_tokens:
        break
    selected_rate = rate  # keep the largest profiled B ≤ query
return selected_rate
```

This returns the rate at the **largest profiled B ≤ the query**. For queries below
the first entry (B<8), it returns the floor (7.235, the B=8 rate).

**Sound for first benchmark?** Yes. The plateaus between profiled points mean the
scheduler sees 6 distinct SPS levels. This is coarse but tests the scheduler's
*direction* (does variable-length verification help under concurrency?). Interpolation
is a refinement — not needed until the scheduler is validated and we're optimizing
marginal decisions.

**Risk**: at B values between profiled points (e.g., B=28..47), the scheduler sees a
flat plateau at 7.280 (the B=24 rate) until B=48 (cliff to 5.397). This means at
c=8 the scheduler might think capacity is still high (7.3) at B=32 and over-verify.
But this is conservative in the **safe direction** — over-verifying means slightly
more Tverify compute, not a throughput collapse. The worst case is the scheduler
behaves like fixed γ=5 at c=8 (which is the baseline anyway).

### early_stop=0

The paper recommends removing early-stop when the SPS curve is jagged. With
early_stop=True, the greedy scheduler stops at the first throughput drop (B=16, SPS
6.3) and misses the recovery at B=20/24 (SPS 6.8/7.3). With early_stop=False, it
exhaustively traverses the full greedy prefix path and picks the globally best point.

**Sound for this curve?** Yes — the curve is explicitly jagged (non-monotonic at
3 points). early_stop=0 is the paper-aligned setting. The exhaustive traversal adds
negligible compute (the greedy path has at most γ×N candidates; at c=4, γ=5, that's
20 iterations of a Python loop with tensor ops on a small confidence matrix).

### Excluding B=4

B=4 was invalid (forced_length=0 produced an anomalous measurement). The floor at
B=8's rate (7.235) for B<8 is slightly conservative — the real SPS at B=4 is likely
higher (fewer tokens → faster steps), so the scheduler under-estimates capacity at
very small B → slightly under-verifies. This is safe.

**Verdict**: the SPS curve, stepwise lookup, and early_stop=0 are all sound for a
first grounded benchmark. The choices are conservative where uncertain, which is the
right posture for a validation pass.

---

## 4. Whether to profile c=8 forced lengths or B=16 repeats

### B=16 repeat check (quick, pre-benchmark)

The B=16 SPS=6.334 is the curve's biggest anomaly (lower than both B=12 and B=20).
If this is a one-off measurement artifact (e.g., a transient during the forced-length
run), the curve would be smoother and the scheduler's decisions more reliable.

**Recommendation**: re-run c=4 forced_length=3 (2 repeats) before the benchmark.
If SPS reproduces at ~6.3 → it's a real capacity feature (likely a capture-size
boundary or ragged-grouping artifact at B=16) → the scheduler handles it correctly
with early_stop=0. If SPS jumps to ~7.0+ → update the curve → smoother decisions.
Cost: 2 forced-length runs (~5 min). Not blocking, but high-value for curve quality.

### c=8 forced lengths (post-validation)

The gap B=24→48 has no profiled points. The scheduler sees a flat plateau (7.280)
from B=25..47, then a cliff (5.397) at B=48. At c=8, the scheduler might choose
B=32 (γ=3) or B=40 (γ=4) → stepwise lookup returns 7.280 → over-estimates capacity
→ potentially over-verifies.

**Recommendation**: profile c=8 forced lengths (γ=1..5 → B=16..48 at MNS=8, or
B=8..24 at MNS=4) **AFTER the first benchmark validates the approach**. If the
scheduler helps at c=4/c=8 with the current curve, fill the gap for better c=8
decisions. If it doesn't help, the gap isn't the issue (it's the confidence quality
or the algorithm).

**Why not profile c=8 first?** Each forced-length point requires a server restart
(server-side env). Profiling c=8 × 6 lengths = 6+ restarts (~30 min). The first
benchmark with the current 6-point curve answers "does the scheduler help at all?"
— that's the gating question. If no, the gap is irrelevant. If yes, the gap becomes
worth filling.

---

## 5. Why the scheduler may stay near full length (γ=5)

The DSpark paper says: "verifying extra tokens has minimal opportunity cost under
light system load." At our batch sizes (c=4 → B=24, c=8 → B=48), the GPU may not
be saturated enough for shorter verification to pay off. The SPS curve shows:

- B=8→24: SPS is ~7.0-7.7 (nearly flat). The capacity cost of verifying 5 vs 1 token
  per request is ~0 (SPS barely changes). So the scheduler has **no incentive to
  prune** — full γ=5 gives the most accepted tokens at nearly the same step rate.
- B=48: SPS drops to 5.4. Here, verifying fewer tokens (B=24 → SPS 7.3) could be
  worth it — the capacity savings (7.3 vs 5.4, +35% step rate) might offset fewer
  accepted tokens.

**Prediction**: at c=4 (B=24, SPS 7.3), the scheduler will stay at full γ=5
(minimal capacity gain from pruning). At c=8 (B=48, SPS 5.4), it might prune to
γ=3-4 (B=32-40, stepwise SPS 7.3 → step rate +35%) IF the confidence head says the
suffix tokens are likely rejected anyway. The scheduler's value emerges only at the
capacity cliff (B≥48), exactly as the paper predicts.

**If the scheduler stays at full γ=5 everywhere**: that's a VALID result — it means
the hardware isn't saturated enough at our concurrency levels for pruning to pay. The
scheduler correctly identifies that verifying extra tokens is nearly free (the paper's
"light load" regime). The next step would be to push concurrency higher (c=16+) where
the GPU saturates and pruning becomes valuable.

---

## 6. Likely next high-value steps

### Immediate (validate the scheduler)

1. **Re-run B=16 point** (c=4, γ=3, 2 repeats) — confirm or correct the 6.334 dip.
2. **Benchmark hardware scheduler** at c=4/c/8 with the profiled SPS curve +
   early_stop=0. Compare per-user tok/s, acceptance, and aggregate vs the compact-
   ragged checkpoint (scheduler off, fixed γ=5).
3. **Gate metrics**: per-user tok/s ≥ compact-ragged baseline (no regression);
   acceptance preserved (the scheduler is lossless — it should never lower τ below
   what confidence warrants); aggregate scaling holds or improves.

### Short-term (if the scheduler helps)

4. **Profile c=8 forced lengths** — fill the B=24→48 gap. Refine the SPS curve for
   better c=8 decisions.
5. **Add STS calibration** — the confidence head produces ~0.977 post-clone-fix,
   but STS (Sequential Temperature Scaling) would cut ECE from ~3-8% to ~1%, making
   the scheduler's survival estimates sharper. The paper says raw confidence is
   "usually overconfident" — STS is the calibration step.
6. **Push concurrency higher** (c=16, MNS=8 if memory allows) — this is where the
   scheduler's value compounds (GPU saturation → pruning becomes valuable → per-user
   speedup at matched aggregate load, the paper's headline).

### Medium-term (structural gains)

7. **Raise MAX_NUM_SEQS** — the GB10 memory limit caps at ~4 sequences for the
   unholy path. The DSpark path (smaller draft model, windowed KV) might tolerate
   more. If MNS=8 is stable with the ragged fix, c=8 batches 8 → B=48 → the capacity
   cliff is hit → the scheduler's pruning becomes valuable.
8. **Async two-step (§5.2)** — make the scheduling decision non-blocking. Relevant
   only after the synchronous scheduler is validated AND scheduling overhead is
   measurable at high concurrency (c=16+). At c=4/c=8, the Python scheduling loop
   is microseconds — negligible.

### Long-term (paper-aligned, large effort)

9. **DSpark vs MTP-1 at matched aggregate load** — the paper's actual comparison.
   At the same server throughput (aggregate tok/s), DSpark should give each user
   faster generation than MTP-1. This is the 60-85% per-user claim. Measurable with
   the concurrent benchmark once multi-seq DSpark is stable + the scheduler is
   validated.

---

## 7. What evidence would change my recommendation

| If we observe... | Then... |
|---|---|
| Scheduler stays at full γ=5 at c=4/c=8 (no pruning) | The GPU isn't saturated at these batch sizes. Push to c=16/MNS=8 to reach the capacity cliff where pruning pays. |
| Scheduler prunes but acceptance drops (τ falls) | The confidence head is uncalibrated → STS calibration becomes the priority. The scheduler is making bad pruning decisions with overconfident survival estimates. |
| Scheduler prunes and per-user tok/s improves | The approach works. Profile finer SPS points + add STS + push concurrency higher. |
| Scheduler prunes but per-user tok/s regresses | The SPS curve is wrong (measurement noise or missing B points) or the pruning math is incorrect. Re-profile + verify the `hardware_aware_prefix_schedule()` algorithm against the harness's property tests. |
| c=8 crashes with the ragged fix | Another single-stream assumption in the proposer or model. Diagnose via the crash stack; likely a shape assertion or a buffer-size limit at higher batch. |
| B=16 SPS reproduces at ~6.3 on re-run | Real capacity feature (capture boundary or ragged-grouping overhead). early_stop=0 handles it. Document it as a known capacity cliff in the SPS curve. |
| B=16 SPS jumps to ~7.0 on re-run | One-off noise. Update the curve. The scheduler's decisions become smoother (no spurious dip to avoid). |

---

## 8. Specific actionable next steps for Codex

1. **Pre-benchmark** (5 min): re-run c=4 forced_length=3, 2 repeats. If SPS ≠ 6.3,
   update `VLLM_DSPARK_SPS_CURVE`.

2. **Benchmark** (30 min): restart with:
   ```
   VLLM_DSPARK_CONFIDENCE_SCHEDULER=hardware
   VLLM_DSPARK_SPS_CURVE=8:7.235357,12:7.749286,16:6.334372,20:6.767292,24:7.280485,48:5.396963
   VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=0
   MAX_NUM_SEQS=4
   ```
   Run concurrent sweep c=1/2/4/8 (2 repeats each). Compare per-user tok/s,
   acceptance, aggregate vs the compact-ragged checkpoint.

3. **Post-benchmark analysis**: check whether the scheduler actually varied
   verification lengths (log `_last_draft_lengths` or add a diagnostic). If it stayed
   at γ=5 everywhere → the GPU isn't saturated → push MNS/concurrency. If it pruned
   → check acceptance held.

4. **If scheduler helps**: profile c=8 forced lengths (fill B=24→48 gap) + add STS
   calibration. If not: check confidence quality via `VLLM_DSPARK_POSITION0_DIAGNOSTICS=1`
   at concurrency (the confidence head's calibration may degrade under batched decode).

5. **Commit**: the compact ragged fix + the SPS profiling harness + the hardware
   scheduler benchmark results (all measured, all grounded).

---

## Appendix: Key numbers reference

### Single-stream anchors (warmed, MNS=1)
- No-spec: 26.33 tok/s
- MTP-1: 39.88 tok/s
- DSpark-5 (γ=5): 60.24 tok/s (1.51× MTP-1, 2.29× no-spec)
- DSpark-3 (γ=3, clean FULL@4): 56.63 tok/s (−6% vs γ=5)

### Real coding session (detail harness, MNS=1)
- Mean: ~40 tok/s, τ ≈ 3.0 (acceptance decays with context; pos0 stable ~0.88,
  suffix decays on novel content — inherent to rank-256 Markov head)

### Concurrency (MNS=2, compact ragged)
- c=1: 61.01 tok/s agg, accept 0.668
- c=2: 81.94 tok/s agg (+49% vs MNS=1 serialized), per-user 40.97, accept 0.608

### Cycle breakdown (single-stream, clean profile)
- Target forward (MoE 43% + dense 30% + NCCL 5% + attn 1%): ~61ms
- Draft propose: ~8.6ms
- Postprocess logits: ~2.5ms
- Total cycle: ~70ms (flat to 80k context)

### SPS(B) curve (c=4 forced-length + c=8 full)
- B=8: 7.235 | B=12: 7.749 | B=16: 6.334 | B=20: 6.767 | B=24: 7.280 | B=48: 5.397
