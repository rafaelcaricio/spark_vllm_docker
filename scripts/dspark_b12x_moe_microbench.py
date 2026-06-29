#!/usr/bin/env python3
"""Standalone B12X W4A16 MoE microbenchmark for DSpark decode shapes.

This intentionally avoids loading DeepSeek weights.  The W4A16 kernel's memory
traffic and launch shape are determined by the packed tensor shapes, routing
shape, and selector result, so synthetic packed tensors are sufficient for
Nsight Compute bandwidth/occupancy checks.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any

import torch

from b12x.moe.fused.w4a16.host import make_w4a16_packed_buffers
from b12x.moe.fused.w4a16.kernel import (
    _select_tile_config,
    compile_w4a16_fused_moe,
    compile_w4a16_topk_sum,
    run_w4a16_moe,
    select_route_block_size_m,
)
from b12x.moe.fused.w4a16.prepare import W4A16PackedWeights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic packed-W4A16 MoE benchmark for DSpark decode.",
    )
    parser.add_argument("--m", type=int, default=6)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--activation", default="silu", choices=("silu", "relu2"))
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--scale-format", default="e8m0_k32", choices=("e8m0_k32", "e4m3_k16"))
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sms", type=int, default=48)
    parser.add_argument("--max-shared-mem", type=int, default=101376)
    parser.add_argument(
        "--force-tile-config",
        default="",
        metavar="TILE_K,TILE_N,CTA_THREADS",
        help="Restrict B12X W4A16 selector to one candidate for microbench A/B.",
    )
    parser.add_argument(
        "--force-blocks-per-sm",
        type=int,
        default=0,
        help="Override selector blocks_per_sm for the selected W4A16 tile.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a small compile/run shape for call-surface smoke tests.",
    )
    parser.add_argument(
        "--zero-scales",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero synthetic scales/global scales to avoid overflow in random output.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype {name!r}")


def gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3)


def packed_weight_shape(num_experts: int, size_k: int, size_n: int) -> tuple[int, int, int]:
    if size_k % 16 or size_n % 64:
        raise ValueError(f"W4A16 packed weights require K%16==0 and N%64==0, got {size_k=}, {size_n=}")
    return (num_experts, size_k // 16, (size_n // 64) * 128)


def e8m0_scale_shape(num_experts: int, size_k: int, size_n: int) -> tuple[int, int, int]:
    if size_k % 32:
        raise ValueError(f"E8M0 K/32 scales require K%32==0, got {size_k=}")
    return (num_experts, size_k // 32, size_n)


def e4m3_scale_shape(num_experts: int, size_k: int, size_n: int) -> tuple[int, int]:
    # This matches the packed e4m3 view consumed by the kernel after preparation.
    if size_k % 16:
        raise ValueError(f"E4M3 K/16 scales require K%16==0, got {size_k=}")
    return (num_experts, (size_k // 16) * size_n)


def tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size()) * int(torch.tensor(shape).prod().item())


def estimate_static_bytes(args: argparse.Namespace) -> dict[str, Any]:
    hidden = int(args.hidden_size)
    intermediate = int(args.intermediate_size)
    experts = int(args.num_experts)
    w13_rows = intermediate * 2
    w13_shape = packed_weight_shape(experts, hidden, w13_rows)
    w2_shape = packed_weight_shape(experts, intermediate, hidden)
    if args.scale_format == "e8m0_k32":
        w13_scale_shape = e8m0_scale_shape(experts, hidden, w13_rows)
        w2_scale_shape = e8m0_scale_shape(experts, intermediate, hidden)
        scale_dtype = torch.uint8
    else:
        w13_scale_shape = e4m3_scale_shape(experts, hidden, w13_rows)
        w2_scale_shape = e4m3_scale_shape(experts, intermediate, hidden)
        scale_dtype = torch.uint8

    dtype = dtype_from_name(args.dtype)
    pieces = {
        "w13_int32": tensor_nbytes(w13_shape, torch.int32),
        "w2_int32": tensor_nbytes(w2_shape, torch.int32),
        "w13_scale": tensor_nbytes(w13_scale_shape, scale_dtype),
        "w2_scale": tensor_nbytes(w2_scale_shape, scale_dtype),
        "a_input": int(args.m) * hidden * torch.empty((), dtype=dtype).element_size(),
    }
    total = sum(pieces.values())
    return {
        "pieces_bytes": pieces,
        "total_static_bytes": total,
        "total_static_gib": gib(total),
        "w13_shape": w13_shape,
        "w2_shape": w2_shape,
        "w13_scale_shape": w13_scale_shape,
        "w2_scale_shape": w2_scale_shape,
    }


def maybe_tiny(args: argparse.Namespace) -> None:
    if not args.tiny:
        return
    args.m = 2
    args.topk = 2
    args.hidden_size = 512
    args.intermediate_size = 512
    args.num_experts = 8
    args.warmup = min(args.warmup, 2)
    args.iterations = min(args.iterations, 4)


def parse_force_tile_config(value: str) -> tuple[int, int, int] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--force-tile-config must be TILE_K,TILE_N,CTA_THREADS")
    tile_k, tile_n, cta_threads = (int(p) for p in parts)
    return tile_k, tile_n, cta_threads


def apply_selector_overrides(
    config: tuple[int, int, int] | None,
    blocks_per_sm: int,
) -> None:
    if config is None and int(blocks_per_sm) <= 0:
        return
    import b12x.moe.fused.w4a16.kernel as w4a16_kernel

    if config is not None:
        w4a16_kernel._SMALL_BATCH_TILE_CONFIGS = (config,)
        w4a16_kernel._LARGE_BATCH_TILE_CONFIGS = (config,)
    if int(blocks_per_sm) > 0:
        forced_blocks = int(blocks_per_sm)

        def _forced_blocks_per_sm(**_: Any) -> int:
            return forced_blocks

        w4a16_kernel._determine_blocks_per_sm = _forced_blocks_per_sm


def build_prepared(args: argparse.Namespace, device: torch.device) -> W4A16PackedWeights:
    static = estimate_static_bytes(args)
    dtype = dtype_from_name(args.dtype)
    w13 = torch.empty(static["w13_shape"], dtype=torch.int32, device=device)
    w2 = torch.empty(static["w2_shape"], dtype=torch.int32, device=device)
    w13_scale = torch.empty(static["w13_scale_shape"], dtype=torch.uint8, device=device)
    w2_scale = torch.empty(static["w2_scale_shape"], dtype=torch.uint8, device=device)
    w13_global = torch.empty((args.num_experts,), dtype=torch.float32, device=device)
    w2_global = torch.empty((args.num_experts,), dtype=torch.float32, device=device)
    if args.zero_scales:
        w13_scale.zero_()
        w2_scale.zero_()
        w13_global.zero_()
        w2_global.zero_()
    else:
        w13_scale.fill_(127)
        w2_scale.fill_(127)
        w13_global.fill_(1.0)
        w2_global.fill_(1.0)

    props = torch.cuda.get_device_properties(device)
    workspace = torch.zeros(
        (int(props.multi_processor_count) * 4 + 2,),
        dtype=torch.int32,
        device=device,
    )
    return W4A16PackedWeights(
        w13=w13,
        w13_scale=w13_scale,
        w13_global_scale=w13_global,
        w2=w2,
        w2_scale=w2_scale,
        w2_global_scale=w2_global,
        workspace=workspace,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        is_gated=args.activation == "silu",
        params_dtype=dtype,
        source_format="fp4_e8m0_k32" if args.scale_format == "e8m0_k32" else "modelopt_nvfp4",
        w13_layout="packed",
        weight_layout="packed",
        scale_format=args.scale_format,
    )


def build_routes(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(args.seed)
    route_ids = torch.arange(args.m * args.topk, dtype=torch.int32, device=device)
    route_ids = (route_ids % args.num_experts).reshape(args.m, args.topk).contiguous()
    weights = torch.full(
        (args.m, args.topk),
        1.0 / float(args.topk),
        dtype=torch.float32,
        device=device,
    )
    return weights, route_ids


def run_once(
    args: argparse.Namespace,
    prepared: W4A16PackedWeights,
    a_input: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    buffers: Any,
    fused_launch: Any,
    topk_sum_launch: Any,
) -> torch.Tensor:
    return run_w4a16_moe(
        a_input,
        prepared,
        topk_weights,
        topk_ids,
        activation=args.activation,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        apply_router_weight_on_input=False,
        fast_math=True,
        fused_launch=fused_launch,
        topk_sum_launch=topk_sum_launch,
    )


def main() -> int:
    args = parse_args()
    maybe_tiny(args)
    forced_tile_config = parse_force_tile_config(args.force_tile_config)

    block_size = select_route_block_size_m(args.m, args.topk, args.num_experts)
    if forced_tile_config is None and args.force_blocks_per_sm > 0:
        forced_tile_config = _select_tile_config(
            problem_m=args.m,
            problem_n=args.intermediate_size * 2,
            problem_k=args.hidden_size,
            top_k=args.topk,
            moe_block_size=block_size,
            sms=args.sms,
            max_shared_mem=args.max_shared_mem,
            scale_format=args.scale_format,
        )[:3]
    apply_selector_overrides(forced_tile_config, args.force_blocks_per_sm)

    tile_k, tile_n, cta_threads, blocks_per_sm = _select_tile_config(
        problem_m=args.m,
        problem_n=args.intermediate_size * 2,
        problem_k=args.hidden_size,
        top_k=args.topk,
        moe_block_size=block_size,
        sms=args.sms,
        max_shared_mem=args.max_shared_mem,
        scale_format=args.scale_format,
    )
    summary: dict[str, Any] = {
        "shape": {
            "m": args.m,
            "topk": args.topk,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_experts": args.num_experts,
            "activation": args.activation,
            "dtype": args.dtype,
            "scale_format": args.scale_format,
        },
        "selector": {
            "moe_block_size": block_size,
            "fc1_tile_k": tile_k,
            "fc1_tile_n": tile_n,
            "cta_threads": cta_threads,
            "blocks_per_sm": blocks_per_sm,
            "forced_tile_config": forced_tile_config,
            "forced_blocks_per_sm": args.force_blocks_per_sm or None,
        },
        "memory_estimate": estimate_static_bytes(args),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --dry-run is used")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    args.sms = int(props.multi_processor_count)
    args.max_shared_mem = int(getattr(props, "shared_memory_per_block_optin", args.max_shared_mem))

    prepared = build_prepared(args, device)
    dtype = dtype_from_name(args.dtype)
    a_input = torch.empty((args.m, args.hidden_size), dtype=dtype, device=device)
    a_input.zero_()
    topk_weights, topk_ids = build_routes(args, device)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=args.m,
        topk=args.topk,
        dtype=dtype,
        device=device,
    )
    fused_launch = compile_w4a16_fused_moe(
        size_m=args.m,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.topk,
        activation=args.activation,
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size,
        max_m_blocks=args.m * args.topk,
        element_dtype="bf16" if dtype is torch.bfloat16 else "fp16",
        fast_math=True,
        sms=args.sms,
        max_shared_mem=args.max_shared_mem,
        weight_layout="packed",
        scale_format=args.scale_format,
        w13_layout="packed",
        direct_topk_routes=True,
        tc_decode_fused_sum=False,
    )
    topk_sum_launch = compile_w4a16_topk_sum(
        m=args.m,
        topk=args.topk,
        hidden_size=args.hidden_size,
        element_dtype="bf16" if dtype is torch.bfloat16 else "fp16",
    )

    torch.cuda.synchronize(device)
    for _ in range(args.warmup):
        run_once(args, prepared, a_input, topk_weights, topk_ids, buffers, fused_launch, topk_sum_launch)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(args.iterations):
        run_once(args, prepared, a_input, topk_weights, topk_ids, buffers, fused_launch, topk_sum_launch)
    end.record()
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - wall_start
    elapsed_ms = float(start.elapsed_time(end))
    summary.update(
        {
            "device": {
                "name": props.name,
                "sms": args.sms,
                "max_shared_mem": args.max_shared_mem,
            },
            "compiled_launch": {
                "fc1_tile_k": int(fused_launch.fc1_tile_k),
                "fc1_tile_n": int(fused_launch.fc1_tile_n),
                "fc2_tile_k": int(fused_launch.fc2_tile_k),
                "fc2_tile_n": int(fused_launch.fc2_tile_n),
                "blocks_per_sm": int(fused_launch.blocks_per_sm),
                "direct_topk_routes": bool(fused_launch.direct_topk_routes),
                "tc_decode_fused_sum": bool(fused_launch.tc_decode_fused_sum),
            },
            "timing": {
                "warmup": args.warmup,
                "iterations": args.iterations,
                "cuda_elapsed_ms": elapsed_ms,
                "cuda_ms_per_iter": elapsed_ms / max(args.iterations, 1),
                "wall_s": wall_s,
            },
            "output_checksum": float(buffers.output.float().sum().item()),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
