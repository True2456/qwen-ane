#!/usr/bin/env python3
"""Probe oMLX's native affine MoE kernel on DeepSeek V4 decode shapes.

This deliberately uses synthetic packed weights.  It measures kernel dispatch
and memory traffic without loading the 100+ GB checkpoint, while preserving
the checkpoint's exact packed dimensions and affine metadata dtypes.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
from omlx.custom_kernels.bonsai import fast as bonsai_fast
from omlx.patches.deepseek_v4.switch_layers import (
    _build_mxfp4_blocks,
    _gather_sort,
    _scatter_unsort,
)


def _median_ms(fn, iterations: int) -> tuple[float, mx.array]:
    samples = []
    output = None
    for _ in range(iterations):
        start = time.perf_counter_ns()
        output = fn()
        mx.eval(output)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    assert output is not None
    return statistics.median(samples), output


def _stock(x, weight, scales, biases, indices, group_size, bits):
    return mx.gather_qmm(
        x,
        weight,
        scales,
        biases,
        rhs_indices=indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
        sorted_indices=False,
    )


def _stock_sorted(x, weight, scales, biases, indices, group_size, bits):
    sorted_x, sorted_indices, inv_order = _gather_sort(x, indices)
    output = mx.gather_qmm(
        sorted_x,
        weight,
        scales,
        biases,
        rhs_indices=sorted_indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
        sorted_indices=True,
    )
    return _scatter_unsort(output, inv_order, indices.shape)


def _native_blocks(x, weight, scales, biases, indices, group_size, bits):
    sorted_x, sorted_indices, inv_order = _gather_sort(x, indices)
    block_meta, block_count = _build_mxfp4_blocks(
        sorted_indices, weight.shape[0], 16
    )
    output = glm_fast.deepseek_affine_gather_qmm_blocks(
        sorted_x,
        weight,
        scales,
        biases,
        block_meta,
        block_count,
        group_size,
        bits,
        1,
    )
    return _scatter_unsort(output, inv_order, indices.shape)


def _make_case(experts: int, routes: int, input_dims: int, output_dims: int,
               group_size: int, bits: int):
    if input_dims % 32 or input_dims % group_size:
        raise ValueError("input_dims must be divisible by 32 and group size")
    packed_dims = input_dims // 32 * bits

    # Match the model's BF16 activations and affine metadata exactly.
    x = mx.random.normal((1, 1, 1, input_dims)).astype(mx.bfloat16)
    weight = mx.random.randint(
        0,
        2**31,
        (experts, output_dims, packed_dims),
        dtype=mx.uint32,
    )
    scales = (
        mx.random.uniform(
            low=1e-3,
            high=2e-2,
            shape=(experts, output_dims, input_dims // group_size),
        )
        .astype(mx.bfloat16)
    )
    biases = (
        mx.random.uniform(
            low=-3e-2,
            high=0.0,
            shape=scales.shape,
        )
        .astype(mx.bfloat16)
    )
    indices = mx.array(
        [[(i * 37 + 5) % experts for i in range(routes)]], dtype=mx.int32
    )
    mx.eval(x, weight, scales, biases, indices)
    return x, weight, scales, biases, indices


def _run_case(name: str, args, input_dims: int, output_dims: int,
              group_size: int, bits: int):
    tensors = _make_case(
        args.experts,
        args.routes,
        input_dims,
        output_dims,
        group_size,
        bits,
    )
    x, weight, scales, biases, indices = tensors
    calls = {
        "stock": lambda: _stock(
            x, weight, scales, biases, indices, group_size, bits
        ),
        "stock_sorted": lambda: _stock_sorted(
            x, weight, scales, biases, indices, group_size, bits
        ),
        "native_blocks": lambda: _native_blocks(
            x, weight, scales, biases, indices, group_size, bits
        ),
    }

    print(
        f"\n{name}: experts={args.experts} routes={args.routes} "
        f"K={input_dims} N={output_dims} q{bits}/gs{group_size}"
    )
    reference = calls["stock"]()
    mx.eval(reference)
    for label, call in calls.items():
        try:
            for _ in range(args.warmup):
                mx.eval(call())
            latency, output = _median_ms(call, args.iterations)
            diff = mx.abs(output.astype(mx.float32) - reference.astype(mx.float32))
            max_abs = float(mx.max(diff).item())
            mean_abs = float(mx.mean(diff).item())
            speedup = None
            if label == "stock":
                args.stock_ms = latency
            elif getattr(args, "stock_ms", None):
                speedup = args.stock_ms / latency
            suffix = f" speedup={speedup:.3f}x" if speedup is not None else ""
            print(
                f"  {label:14s} {latency:8.3f} ms{suffix} "
                f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
            )
        except Exception as exc:
            print(f"  {label:14s} ERROR: {type(exc).__name__}: {exc}")


def _run_bonsai_q2(args):
    x, weight, scales, biases, indices = _make_case(
        args.experts, args.routes, 4096, 2048, 128, 2
    )
    x_single = x.reshape(1, 4096)
    single_weight = weight[0]
    single_scales = scales[0]
    single_biases = biases[0]
    mx.eval(x_single, single_weight, single_scales, single_biases)

    single_calls = {
        "stock_qmv": lambda: mx.quantized_matmul(
            x_single,
            single_weight,
            scales=single_scales,
            biases=single_biases,
            transpose=True,
            group_size=128,
            bits=2,
        ),
        "bonsai_qmv": lambda: bonsai_fast.bonsai_q2_affine_qmv(
            x_single, single_weight, single_scales, single_biases
        ),
    }
    print("\nq2 single selected expert (no dynamic gather):")
    reference = single_calls["stock_qmv"]()
    mx.eval(reference)
    stock_ms = None
    for label, call in single_calls.items():
        try:
            for _ in range(args.warmup):
                mx.eval(call())
            latency, output = _median_ms(call, args.iterations)
            diff = mx.abs(output.astype(mx.float32) - reference.astype(mx.float32))
            max_abs = float(mx.max(diff).item())
            mean_abs = float(mx.mean(diff).item())
            if stock_ms is None:
                stock_ms = latency
                suffix = ""
            else:
                suffix = f" speedup={stock_ms / latency:.3f}x"
            print(
                f"  {label:14s} {latency:8.3f} ms{suffix} "
                f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
            )
        except Exception as exc:
            print(f"  {label:14s} ERROR: {type(exc).__name__}: {exc}")

    # This represents the only direct way to feed dynamic router output to the
    # current Bonsai ABI: gather six full expert matrices, then run batched QMV.
    route_indices = indices.flatten()
    route_x = mx.broadcast_to(x_single, (args.routes, 1, 4096))

    def gathered_bonsai():
        return bonsai_fast.bonsai_q2_affine_qmv(
            route_x,
            weight[route_indices],
            scales[route_indices],
            biases[route_indices],
        )

    print("\nq2 dynamic six-expert gather + Bonsai:")
    try:
        for _ in range(args.warmup):
            mx.eval(gathered_bonsai())
        latency, output = _median_ms(gathered_bonsai, args.iterations)
        stock = _stock(x, weight, scales, biases, indices, 128, 2)
        mx.eval(stock)
        stock_latency, _ = _median_ms(
            lambda: _stock(x, weight, scales, biases, indices, 128, 2),
            args.iterations,
        )
        diff = mx.abs(
            output.reshape(stock.shape).astype(mx.float32)
            - stock.astype(mx.float32)
        )
        print(
            f"  gathered_bonsai {latency:8.3f} ms "
            f"vs stock={stock_latency:.3f} ms speedup={stock_latency / latency:.3f}x "
            f"max_abs={float(mx.max(diff).item()):.6g} "
            f"mean_abs={float(mx.mean(diff).item()):.6g}"
        )
    except Exception as exc:
        print(f"  gathered_bonsai ERROR: {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--routes", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()

    print(mx.device_info())
    print(
        "native affine blocks:",
        glm_fast.has_symbol("deepseek_affine_gather_qmm_blocks"),
    )
    print("native Bonsai Q2:", bonsai_fast.has_symbol("bonsai_q2_affine_qmv"))
    _run_case("gate/up", args, 4096, 2048, 128, 2)
    _run_case("down", args, 2048, 4096, 64, 3)
    _run_bonsai_q2(args)


if __name__ == "__main__":
    main()
