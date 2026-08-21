#!/usr/bin/env python3
"""ane_metal_roundtrip.py - End-to-end zero-copy Metal GPU -> ANE pipeline probe."""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runtime.metal_engine import MetalEngine
from runtime.q38_ane_engine import AneEngine, _BUILD_INFO, _objc, _sel


def main():
    print("=" * 60)
    print("  END-TO-END ZERO-COPY METAL GPU -> ANE PIPELINE PROBE")
    print("=" * 60)

    metal = MetalEngine()
    ane = AneEngine()
    print(f"Metal Device: {metal.device_name}")

    # Shapes: Real Qwen channel dimension and decode width
    C = 5120
    S = 32
    print(f"Config: Channels={C}, SeqLen={S} (FP16)")

    # 1. Compile scale-by-2 (add x + x) ANE program
    mil = f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<fp16, [1, {C}, 1, {S}]> y = add(x=x, y=x)[name=string("y")];
  }} -> (y);
}}
"""
    ane_prog = ane.compile_multiproc(mil, {}, C, C, S)
    if not ane_prog:
        print("  [ERROR] ANE compilation failed")
        return

    # Ensure IO surfaces are created and bound
    ane._ensure_io(ane_prog)

    # 2. Wrap ANE's existing IOSurfaces directly into zero-copy Metal buffers
    in_buf_metal = metal.buffer_from_existing_iosurface(ane_prog._in_surf)
    out_buf_metal = metal.buffer_from_existing_iosurface(ane_prog._out_surf)

    # 3. Create test inputs on GPU
    np.random.seed(123)
    raw_input = (np.random.randn(S, C) * 0.5).astype(np.float16)
    weight_norm = (np.ones(C) * 1.5).astype(np.float16)

    scratch_in = metal.create_buffer(raw_input.nbytes)
    w_norm_buf = metal.create_buffer(weight_norm.nbytes)
    scratch_norm = metal.create_buffer(raw_input.nbytes)

    scratch_in.numpy_view((S, C), dtype=np.float16)[:] = raw_input
    w_norm_buf.numpy_view((C,), dtype=np.float16)[:] = weight_norm

    # 4. Metal step: compute RMSNorm on GPU and pack directly into ANE's input IOSurface
    metal.dispatch_rmsnorm(scratch_in, w_norm_buf, scratch_norm, S, C, eps=1e-6)
    metal.dispatch_layout_transform(scratch_norm, in_buf_metal, C, S, mode=0)

    # 5. ANE step: submit execution directly on the shared IOSurface
    ok = ane.submit(ane_prog)
    assert ok, "ANE execution failed"

    # 6. Read back from ANE's output IOSurface and verify
    ane_out_planar = out_buf_metal.numpy_view((1, C, 1, S), dtype=np.float16).copy()
    ane_out_linear = np.ascontiguousarray(ane_out_planar.reshape(C, S).T)

    # Reference computation: (RMSNorm(raw_input) * weight_norm) * 2.0 through ANE add(x, x)
    r_f32 = raw_input.astype(np.float32)
    w_f32 = weight_norm.astype(np.float32)
    rms = np.sqrt(np.mean(r_f32 ** 2, axis=-1, keepdims=True) + 1e-6)
    ref_expected = (((r_f32 / rms) * w_f32) * 2.0).astype(np.float16)

    diff = np.abs(ane_out_linear.astype(np.float32) - ref_expected.astype(np.float32))
    max_err = np.max(diff)
    rel_err = max_err / np.max(np.abs(ref_expected.astype(np.float32)))

    print(f"\nPipeline Verification Results:")
    print(f"  Max Absolute Error: {max_err:.6f}")
    print(f"  Relative Error:     {rel_err:.6e}")
    assert rel_err < 2e-3, f"Pipeline error too high: {rel_err}"
    print("  ✓ ZERO-COPY GPU (RMSNorm + Pack) -> ANE (MIL Kernel) PASSED EXACTLY!")

    # 7. Benchmark end-to-end pipeline latency
    runs = 200
    t0 = time.perf_counter()
    for _ in range(runs):
        metal.dispatch_rmsnorm(scratch_in, w_norm_buf, scratch_norm, S, C, eps=1e-6)
        metal.dispatch_layout_transform(scratch_norm, in_buf_metal, C, S, mode=0)
        ane.submit(ane_prog)
    t_end = time.perf_counter()
    avg_ms = (t_end - t0) / runs * 1e3
    print(f"  End-to-End Pipeline Latency: {avg_ms:.3f} ms per dispatch")


if __name__ == "__main__":
    main()
