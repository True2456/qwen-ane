#!/usr/bin/env python3
"""Benchmark a fused Q2/gs128 DeepSeek MoE gate+up+SwiGLU kernel."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx


_KERNEL_CACHE = {}


def _kernel(dtype):
    cached = _KERNEL_CACHE.get(dtype)
    if cached is not None:
        return cached
    source = r"""
        using namespace metal;

        constexpr int GS = 128;
        constexpr int VALUES_PER_WORD = 16;
        constexpr int WORDS_PER_GROUP = GS / VALUES_PER_WORD;

        uint lane = thread_index_in_simdgroup;
        uint tile = threadgroup_position_in_grid.x;
        uint route = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int routes_per_token = int(routes_size);
        int K_words = K / VALUES_PER_WORD;
        int K_groups = K / GS;
        int n0 = int(tile) * 4;
        if (n0 + 3 >= N) return;

        int expert = int(indices[route]);
        int token = int(route) / routes_per_token;
        const device T* xr = x + token * K;

        long expert_w = long(expert) * long(N) * long(K_words);
        long expert_m = long(expert) * long(N) * long(K_groups);
        float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int word = int(lane); word < K_words; word += 32) {
            int k0 = word * VALUES_PER_WORD;
            int group = word / WORDS_PER_GROUP;
            T xv[VALUES_PER_WORD];
            _Pragma("unroll")
            for (int ki = 0; ki < VALUES_PER_WORD; ++ki) {
                xv[ki] = xr[k0 + ki];
            }

            _Pragma("unroll")
            for (int j = 0; j < 4; ++j) {
                long woff = expert_w + long(n0 + j) * long(K_words) + word;
                long moff = expert_m + long(n0 + j) * long(K_groups) + group;
                uint gw = gate_w[woff];
                uint uw = up_w[woff];
                float gs = float(gate_scales[moff]);
                float gb = float(gate_biases[moff]);
                float us = float(up_scales[moff]);
                float ub = float(up_biases[moff]);
                _Pragma("unroll")
                for (int ki = 0; ki < VALUES_PER_WORD; ++ki) {
                    float a = float(xv[ki]);
                    gate_acc[j] += a * (float((gw >> (2 * ki)) & 3u) * gs + gb);
                    up_acc[j] += a * (float((uw >> (2 * ki)) & 3u) * us + ub);
                }
            }
        }

        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) {
            gate_acc[j] = simd_sum(gate_acc[j]);
            up_acc[j] = simd_sum(up_acc[j]);
        }
        if (lane < 4) {
            float g = gate_acc[int(lane)];
            float u = up_acc[int(lane)];
            float silu = g / (1.0f + exp(-g));
            y[long(route) * long(N) + n0 + int(lane)] = T(silu * u);
        }
    """
    dtype_tag = "bf16" if dtype == mx.bfloat16 else "fp16"
    built = mx.fast.metal_kernel(
        name=f"rindi_deepseek_q2_gs128_pair_swiglu_{dtype_tag}",
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
            "indices",
            "K_size",
            "N_size",
            "routes_size",
        ],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[dtype] = built
    return built


def fused_q2_swiglu(x, gate_w, gate_scales, gate_biases,
                    up_w, up_scales, up_biases, indices):
    batch = int(indices.shape[0])
    routes = int(indices.shape[1])
    K = int(gate_scales.shape[-1]) * 128
    N = int(gate_w.shape[-2])
    flat_indices = indices.reshape(-1)
    x2 = x.reshape(batch, K)
    kernel = _kernel(x.dtype)
    (output,) = kernel(
        inputs=[
            x2,
            gate_w,
            gate_scales,
            gate_biases,
            up_w,
            up_scales,
            up_biases,
            flat_indices,
            K,
            N,
            routes,
        ],
        template=[("T", x.dtype)],
        grid=(32 * (N // 4), batch * routes, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, routes, 1, N)],
        output_dtypes=[x.dtype],
    )
    return output


def stock_swiglu(x, gate_w, gate_scales, gate_biases,
                 up_w, up_scales, up_biases, indices):
    kwargs = dict(
        rhs_indices=indices,
        transpose=True,
        group_size=128,
        bits=2,
        mode="affine",
        sorted_indices=False,
    )
    gate = mx.gather_qmm(x, gate_w, gate_scales, gate_biases, **kwargs)
    up = mx.gather_qmm(x, up_w, up_scales, up_biases, **kwargs)
    return (gate * mx.sigmoid(gate)) * up


def _median_ms(fn, iterations):
    values = []
    output = None
    for _ in range(iterations):
        start = time.perf_counter_ns()
        output = fn()
        mx.eval(output)
        values.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(values), output


def _weights(experts, N, K, dtype):
    w = mx.random.randint(
        0, 2**31, (experts, N, K // 16), dtype=mx.uint32
    )
    s = mx.random.uniform(
        low=1e-3, high=1e-2, shape=(experts, N, K // 128)
    ).astype(dtype)
    b = mx.random.uniform(
        low=-2e-2, high=0.0, shape=s.shape
    ).astype(dtype)
    return w, s, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--routes", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    K, N = 4096, 2048
    dtype = mx.bfloat16
    x = mx.random.normal((args.batch, 1, 1, K)).astype(dtype)
    gate = _weights(args.experts, N, K, dtype)
    up = _weights(args.experts, N, K, dtype)
    indices = mx.array(
        [
            [(row * 19 + route * 37 + 5) % args.experts
             for route in range(args.routes)]
            for row in range(args.batch)
        ],
        dtype=mx.int32,
    )
    mx.eval(x, *gate, *up, indices)

    calls = {
        "stock_pair_swiglu": lambda: stock_swiglu(x, *gate, *up, indices),
        "fused_q2_swiglu": lambda: fused_q2_swiglu(x, *gate, *up, indices),
    }
    reference = calls["stock_pair_swiglu"]()
    mx.eval(reference)
    print(mx.device_info())
    print(
        f"experts={args.experts} batch={args.batch} routes={args.routes} "
        f"K={K} N={N} q2/gs128"
    )
    stock_ms = None
    for name, call in calls.items():
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
            f"{name:20s} {latency:8.3f} ms{suffix} "
            f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
        )


if __name__ == "__main__":
    main()
