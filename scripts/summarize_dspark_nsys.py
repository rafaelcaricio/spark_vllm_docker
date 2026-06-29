#!/usr/bin/env python3
"""Summarize DSpark Nsight Systems SQLite exports.

The runner writes one SQLite export per profiled rank. This script extracts the
small set of metrics we care about for DSpark decode work: CUDA graph replay
shape, launch gaps, synchronization waits, NVTX iteration ranges, and dominant
kernel families.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * p)]


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "sum_ms": sum(values),
        "mean_ms": mean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
    }


def kernel_bucket(name: str) -> str:
    low = name.lower()
    if "b12xmoefusedw4a16" in low and "fusedmoekernel" in low:
        return "B12X W4A16 MoE fused"
    if "cutlass_80_wmma_tensorop_bf16_s161616gemm" in low:
        return "CUTLASS bf16/s1616 GEMM"
    if "deep_gemm::sm120_fp8_fp4" in low:
        return "DeepGEMM fp8/fp4"
    if "b12xgemmdense" in low:
        return "B12X dense GEMM"
    if "nccldevkernel_allreduce" in low:
        return "NCCL all-reduce"
    if "nccldevkernel_allgather" in low:
        return "NCCL all-gather"
    if "sparse_mla" in low:
        return "FlashInfer sparse MLA"
    if "mhc" in low:
        return "MHC"
    if "argmax" in low or "reduce_kernel" in low:
        return "Argmax/reduce"
    if "quantize_attention" in low or "qnormropekvropequant" in low:
        return "Quantize/RoPE/KV insert"
    return "Other"


def summarize_db(path: Path) -> dict[str, Any]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row

    runtime_start, runtime_end, runtime_count = con.execute(
        "select min(start), max(end), count(*) from CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()
    capture_span_ms = (
        (runtime_end - runtime_start) / 1e6
        if runtime_start is not None and runtime_end is not None
        else None
    )

    graph_rows = [
        ((row["end"] - row["start"]) / 1e6, row["start"], row["end"])
        for row in con.execute(
            "select start,end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start"
        )
    ]
    graph_durations = [row[0] for row in graph_rows]
    graph_gaps = [
        (graph_rows[i][1] - graph_rows[i - 1][2]) / 1e6
        for i in range(1, len(graph_rows))
    ]
    graph_pairs = [
        graph_durations[i] + graph_durations[i + 1]
        for i in range(0, len(graph_durations) - 1, 2)
    ]
    short_graphs = [x for x in graph_durations if 5.0 <= x < 20.0]
    long_graphs = [x for x in graph_durations if 40.0 <= x < 80.0]

    generation_ranges = [
        (row["end"] - row["start"]) / 1e6
        for row in con.execute(
            """
            select n.start,n.end
            from NVTX_EVENTS n
            left join StringIds s on n.textId=s.id
            where coalesce(n.text, s.value) like 'execute_context_%generation_1(6)%'
              and n.end is not null
            order by n.start
            """
        )
    ]

    sync_rows = [
        {
            "name": row["name"],
            "count": row["count"],
            "sum_ms": row["sum_ms"],
            "mean_us": row["mean_us"],
        }
        for row in con.execute(
            """
            select s.value as name,
                   count(*) as count,
                   sum(r.end-r.start)/1e6 as sum_ms,
                   avg(r.end-r.start)/1e3 as mean_us
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            join StringIds s on r.nameId=s.id
            where s.value like 'cuda%Synchronize%'
            group by s.value
            order by sum_ms desc
            """
        )
    ]

    buckets: dict[str, dict[str, float | int]] = {}
    for row in con.execute(
        """
        select s.value as name,
               count(*) as count,
               sum(k.end-k.start)/1e6 as sum_ms
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on k.demangledName=s.id
        group by s.value
        """
    ):
        bucket = kernel_bucket(row["name"])
        entry = buckets.setdefault(bucket, {"count": 0, "sum_ms": 0.0})
        entry["count"] += int(row["count"])
        entry["sum_ms"] += float(row["sum_ms"] or 0.0)

    kernel_buckets = [
        {"bucket": key, **value}
        for key, value in sorted(
            buckets.items(), key=lambda item: item[1]["sum_ms"], reverse=True
        )
    ]

    con.close()
    return {
        "path": str(path),
        "runtime_count": runtime_count,
        "capture_span_ms": capture_span_ms,
        "generation_nvtx": stats(generation_ranges),
        "cuda_graphs": stats(graph_durations),
        "cuda_graph_gaps": stats(graph_gaps),
        "cuda_graph_pairs": stats(graph_pairs),
        "short_graphs_5_20ms": stats(short_graphs),
        "long_graphs_40_80ms": stats(long_graphs),
        "sync_runtime": sync_rows,
        "kernel_buckets": kernel_buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sqlite_paths = sorted(args.profile_dir.glob("*/*.sqlite"))
    if not sqlite_paths:
        raise SystemExit(f"no .sqlite files found under {args.profile_dir}")

    summaries = [summarize_db(path) for path in sqlite_paths]
    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    for summary in summaries:
        role = Path(summary["path"]).parent.name
        print(f"{role}: {summary['path']}")
        print(f"  capture_span_ms: {summary['capture_span_ms']:.3f}")
        print(
            "  generation_nvtx: "
            f"count={summary['generation_nvtx']['count']} "
            f"sum={summary['generation_nvtx']['sum_ms']:.3f}ms "
            f"p50={summary['generation_nvtx']['p50_ms']:.3f}ms "
            f"p90={summary['generation_nvtx']['p90_ms']:.3f}ms"
        )
        print(
            "  cuda_graphs: "
            f"count={summary['cuda_graphs']['count']} "
            f"sum={summary['cuda_graphs']['sum_ms']:.3f}ms "
            f"p50={summary['cuda_graphs']['p50_ms']:.3f}ms "
            f"p90={summary['cuda_graphs']['p90_ms']:.3f}ms"
        )
        print(
            "  graph_pairs: "
            f"count={summary['cuda_graph_pairs']['count']} "
            f"mean={summary['cuda_graph_pairs']['mean_ms']:.3f}ms "
            f"p50={summary['cuda_graph_pairs']['p50_ms']:.3f}ms "
            f"p90={summary['cuda_graph_pairs']['p90_ms']:.3f}ms"
        )
        print(
            "  short_graphs_5_20ms: "
            f"count={summary['short_graphs_5_20ms']['count']} "
            f"sum={summary['short_graphs_5_20ms']['sum_ms']:.3f}ms "
            f"mean={summary['short_graphs_5_20ms']['mean_ms']:.3f}ms"
        )
        print(
            "  long_graphs_40_80ms: "
            f"count={summary['long_graphs_40_80ms']['count']} "
            f"sum={summary['long_graphs_40_80ms']['sum_ms']:.3f}ms "
            f"mean={summary['long_graphs_40_80ms']['mean_ms']:.3f}ms"
        )
        print("  sync_runtime:")
        for row in summary["sync_runtime"][:4]:
            print(
                f"    {row['name']}: count={row['count']} "
                f"sum={row['sum_ms']:.3f}ms mean={row['mean_us']:.3f}us"
            )
        print("  kernel_buckets:")
        for row in summary["kernel_buckets"][:10]:
            print(f"    {row['bucket']}: count={row['count']} sum={row['sum_ms']:.3f}ms")


if __name__ == "__main__":
    main()
