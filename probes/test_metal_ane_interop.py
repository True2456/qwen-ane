#!/usr/bin/env python3
"""test_metal_ane_interop.py - Cross-check probe for custom Metal C engine against GPU/NumPy/MLX."""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runtime.metal_engine import MetalEngine, MetalSharedEvent


def test_rmsnorm_cross_check(engine: MetalEngine, S=32, C=5120, eps=1e-6):
    print(f"\n[1] Cross-checking RMSNorm FP16 (S={S}, C={C})...")
    
    # Input data
    np.random.seed(42)
    x_np = np.random.randn(S, C).astype(np.float16)
    w_np = np.random.randn(C).astype(np.float16)

    # Reference NumPy (FP32 precision accumulation matching GPU shader)
    x_f32 = x_np.astype(np.float32)
    w_f32 = w_np.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    ref_np = (x_f32 / rms * w_f32).astype(np.float16)

    # Optional MLX cross check
    mlx_available = False
    try:
        import mlx.core as mx
        x_mx = mx.array(x_np)
        w_mx = mx.array(w_np)
        ref_mlx = (x_mx * mx.rsqrt(mx.mean(x_mx * x_mx, axis=-1, keepdims=True) + eps) * w_mx)
        mx.eval(ref_mlx)
        ref_mlx_np = np.array(ref_mlx)
        mlx_available = True
    except ImportError:
        pass

    # Metal C execution in zero-copy IOSurface
    in_buf = engine.create_iosurface_buffer(x_np.nbytes)
    w_buf = engine.create_buffer(w_np.nbytes)
    out_buf = engine.create_iosurface_buffer(ref_np.nbytes)

    # Copy inputs into mapped views
    in_buf.numpy_view((S, C), dtype=np.float16)[:] = x_np
    w_buf.numpy_view((C,), dtype=np.float16)[:] = w_np

    # Warmup
    for _ in range(5):
        engine.dispatch_rmsnorm(in_buf, w_buf, out_buf, S, C, eps)

    # Benchmark latency
    runs = 500
    t0 = time.perf_counter()
    for _ in range(runs):
        engine.dispatch_rmsnorm(in_buf, w_buf, out_buf, S, C, eps)
    t_end = time.perf_counter()
    avg_us = (t_end - t0) / runs * 1e6

    metal_out = out_buf.numpy_view((S, C), dtype=np.float16).copy()

    # Numerical difference
    diff_np = np.abs(metal_out.astype(np.float32) - ref_np.astype(np.float32))
    max_err_np = np.max(diff_np)
    rel_err_np = max_err_np / (np.max(np.abs(ref_np.astype(np.float32))) + 1e-6)

    print(f"  NumPy reference max abs err: {max_err_np:.6f}, relative err: {rel_err_np:.6e}")
    if mlx_available:
        diff_mlx = np.abs(metal_out.astype(np.float32) - ref_mlx_np.astype(np.float32))
        print(f"  MLX reference max abs err:   {np.max(diff_mlx):.6f}")
    print(f"  Metal latency per dispatch:  {avg_us:.2f} µs")
    assert rel_err_np < 2e-3, f"RMSNorm precision error too high: {rel_err_np}"
    print("  ✓ RMSNorm FP16 PASSED")


def test_layout_transform(engine: MetalEngine, S=32, C=5120):
    print(f"\n[2] Cross-checking ANE Layout Transformations (S={S}, C={C})...")
    np.random.seed(43)
    x_linear = np.random.randn(S, C).astype(np.float16)

    # Reference ANE planar [1, C, 1, S]
    ref_ane = np.ascontiguousarray(x_linear.T).reshape(1, C, 1, S)

    in_buf = engine.create_iosurface_buffer(x_linear.nbytes)
    out_ane_buf = engine.create_iosurface_buffer(ref_ane.nbytes)
    out_linear_buf = engine.create_iosurface_buffer(x_linear.nbytes)

    in_buf.numpy_view((S, C), dtype=np.float16)[:] = x_linear

    # Linear -> ANE
    engine.dispatch_layout_transform(in_buf, out_ane_buf, C, S, mode=0)
    ane_view = out_ane_buf.numpy_view((1, C, 1, S), dtype=np.float16)
    assert np.array_equal(ane_view, ref_ane), "Linear -> ANE transform failed"

    # ANE -> Linear
    engine.dispatch_layout_transform(out_ane_buf, out_linear_buf, C, S, mode=1)
    linear_view = out_linear_buf.numpy_view((S, C), dtype=np.float16)
    assert np.array_equal(linear_view, x_linear), "ANE -> Linear transform failed"

    # Benchmark
    runs = 1000
    t0 = time.perf_counter()
    for _ in range(runs):
        engine.dispatch_layout_transform(in_buf, out_ane_buf, C, S, mode=0)
    t_end = time.perf_counter()
    avg_us = (t_end - t0) / runs * 1e6
    gb_s = (x_linear.nbytes * 2) / (avg_us * 1e-6) / 1e9

    print(f"  Layout transform latency:    {avg_us:.2f} µs ({gb_s:.1f} GB/s effective)")
    print("  ✓ Layout Transforms PASSED")


def test_moe_gather_scatter(engine: MetalEngine, num_tokens=16, top_k=8, hidden_dim=1536):
    print(f"\n[3] Cross-checking Dynamic MoE Gather & Scatter (tokens={num_tokens}, k={top_k}, hidden={hidden_dim})...")
    np.random.seed(44)
    activations = np.random.randn(num_tokens, hidden_dim).astype(np.float16)
    indices = np.random.randint(0, 128, size=(num_tokens, top_k), dtype=np.int32)
    weights = np.random.rand(num_tokens, top_k).astype(np.float32)
    weights /= np.sum(weights, axis=-1, keepdims=True)

    # Reference Gather
    ref_gathered = np.zeros((num_tokens * top_k, hidden_dim), dtype=np.float16)
    for t in range(num_tokens):
        for k in range(top_k):
            ref_gathered[t * top_k + k] = activations[t]

    act_buf = engine.create_buffer(activations.nbytes)
    idx_buf = engine.create_buffer(indices.nbytes)
    gathered_buf = engine.create_iosurface_buffer(ref_gathered.nbytes)

    act_buf.numpy_view((num_tokens, hidden_dim), dtype=np.float16)[:] = activations
    idx_buf.numpy_view((num_tokens, top_k), dtype=np.int32)[:] = indices

    engine.dispatch_moe_gather(act_buf, idx_buf, gathered_buf, num_tokens, top_k, hidden_dim)
    metal_gathered = gathered_buf.numpy_view((num_tokens * top_k, hidden_dim), dtype=np.float16)
    assert np.array_equal(metal_gathered, ref_gathered), "MoE Gather output mismatch"

    # Reference Scatter
    fake_expert_outputs = np.random.randn(num_tokens * top_k, hidden_dim).astype(np.float16)
    ref_combined = np.zeros((num_tokens, hidden_dim), dtype=np.float16)
    for t in range(num_tokens):
        acc = np.zeros(hidden_dim, dtype=np.float32)
        for k in range(top_k):
            acc += weights[t, k] * fake_expert_outputs[t * top_k + k].astype(np.float32)
        ref_combined[t] = acc.astype(np.float16)

    exp_out_buf = engine.create_buffer(fake_expert_outputs.nbytes)
    weight_buf = engine.create_buffer(weights.nbytes)
    dst_buf = engine.create_iosurface_buffer(ref_combined.nbytes)

    exp_out_buf.numpy_view((num_tokens * top_k, hidden_dim), dtype=np.float16)[:] = fake_expert_outputs
    weight_buf.numpy_view((num_tokens, top_k), dtype=np.float32)[:] = weights

    engine.dispatch_moe_scatter(exp_out_buf, weight_buf, idx_buf, dst_buf, num_tokens, top_k, hidden_dim)
    metal_combined = dst_buf.numpy_view((num_tokens, hidden_dim), dtype=np.float16)

    diff = np.abs(metal_combined.astype(np.float32) - ref_combined.astype(np.float32))
    assert np.max(diff) < 2e-3, f"MoE Scatter output mismatch: max err {np.max(diff)}"

    # Benchmark gather + scatter combined
    runs = 500
    t0 = time.perf_counter()
    for _ in range(runs):
        engine.dispatch_moe_gather(act_buf, idx_buf, gathered_buf, num_tokens, top_k, hidden_dim)
        engine.dispatch_moe_scatter(exp_out_buf, weight_buf, idx_buf, dst_buf, num_tokens, top_k, hidden_dim)
    t_end = time.perf_counter()
    avg_us = (t_end - t0) / runs * 1e6
    print(f"  MoE Gather + Scatter latency: {avg_us:.2f} µs")
    print("  ✓ Dynamic MoE Gather & Scatter PASSED")


def test_shared_event_sync(engine: MetalEngine):
    print("\n[4] Testing Hardware MTLSharedEvent Signaling...")
    event = engine.create_shared_event()
    assert event.value == 0, "Initial event value should be 0"

    event.value = 42
    assert event.value == 42, f"Expected 42, got {event.value}"

    buf1 = engine.create_buffer(1024)
    buf2 = engine.create_buffer(1024)
    w_buf = engine.create_buffer(1024)

    # Dispatch with GPU signaling event value = 100
    engine.dispatch_rmsnorm(buf1, w_buf, buf2, S=1, C=512, signal_event=(event, 100))
    assert event.value == 100, f"Expected GPU signaled value 100, got {event.value}"
    print("  ✓ MTLSharedEvent Hardware Sync PASSED")


def test_gemm_argmax(engine: MetalEngine, M=4, N=5120, K=10240):
    print(f"\n[5] Cross-checking GPU GEMM & Argmax FP16 (M={M}, N={N}, K={K})...")
    np.random.seed(45)
    A_np = (np.random.randn(M, K) / np.sqrt(K)).astype(np.float16)
    B_np = (np.random.randn(N, K) / np.sqrt(K)).astype(np.float16)

    # Reference GEMM C = A @ B.T
    ref_C = (A_np.astype(np.float32) @ B_np.astype(np.float32).T).astype(np.float16)

    A_buf = engine.create_buffer(A_np.nbytes)
    B_buf = engine.create_buffer(B_np.nbytes)
    C_buf = engine.create_iosurface_buffer(ref_C.nbytes)

    A_buf.numpy_view((M, K), dtype=np.float16)[:] = A_np
    B_buf.numpy_view((N, K), dtype=np.float16)[:] = B_np

    engine.dispatch_gemm(A_buf, B_buf, C_buf, M, N, K)
    metal_C = C_buf.numpy_view((M, N), dtype=np.float16)

    diff = np.abs(metal_C.astype(np.float32) - ref_C.astype(np.float32))
    max_err = np.max(diff)
    print(f"  GEMM FP16 max abs error vs NumPy: {max_err:.6f}")
    assert max_err < 5e-2, f"GEMM error too high: {max_err}"
    print("  ✓ GEMM FP16 PASSED")

    # Test Argmax
    ref_tokens = np.argmax(ref_C, axis=-1).astype(np.int32)
    tok_buf = engine.create_buffer(M * 4)

    engine.dispatch_argmax(C_buf, tok_buf, M, N)
    metal_tokens = tok_buf.numpy_view((M,), dtype=np.int32)

    assert np.array_equal(metal_tokens, ref_tokens), f"Argmax mismatch: {metal_tokens} vs {ref_tokens}"
    print("  ✓ GPU Argmax FP16 PASSED")


def main():
    print("=" * 60)
    print("  METAL C ENGINE & ZERO-COPY ANE INTEROP PROBE")
    print("=" * 60)

    engine = MetalEngine()
    print(f"Metal Device: {engine.device_name}")

    test_rmsnorm_cross_check(engine)
    test_layout_transform(engine)
    test_moe_gather_scatter(engine)
    test_shared_event_sync(engine)
    test_gemm_argmax(engine)

    print("\n" + "=" * 60)
    print("  ALL METAL CROSS-CHECKS & HARDWARE PROBES PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

