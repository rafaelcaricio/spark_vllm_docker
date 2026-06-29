# DSpark Harness

Standalone, uv-managed validation harness for DSpark speculative-decoding logic.

The harness is intentionally independent from vLLM. It validates the algorithmic
pieces before we attach them to the experimental vLLM source tree:

- standard speculative rejection sampling preserves the target distribution,
- confidence-threshold prefix pruning is safe when it is non-anticipating,
- token-value lookahead pruning is unsafe and measurably biases output,
- DSpark-style Markov-biased drafting still remains lossless after target
  verification,
- cumulative prefix survival follows the confidence chain rule,
- the hardware-aware greedy prefix scheduler matches exhaustive search when it
  searches the full greedy path over profiled batch capacities,
- early-stop scheduling can get trapped by jagged hardware capacity cliffs, which
  matches the production caveat in the DSpark paper,
- measured draft/verify overlap budgets stay honest about target-layer
  dependencies, separating legal tail overlap from the theoretical full-draft
  hiding ceiling.

Run the grounded pytest suite:

```bash
uv run pytest --hypothesis-show-statistics
```

Run the heavier stress gate:

```bash
uv run dspark-harness --iterations 5000 --samples 80000 --seed 20260627
```

This does not validate vLLM tensor layouts, CUDA graph behavior, KV cache
updates, or DeepSeek-V4 kernels. Those belong in the next integration layer.
