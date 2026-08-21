#!/usr/bin/env python3
"""ane_int8_tops_probe.py - Empirical probe exploring true INT8 (W8A8) execution on M5 Max ANE."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _BUILD_INFO, _iosurface_view


def test_mil_variant(name: str, mil_template: str, blobs: dict, C: int, M: int, S: int):
    print(f"\n--- Testing Variant: {name} (In={C}, Out={M}, S={S}) ---")
    eng = AneEngine()
    buf = io.StringIO()
    prog = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            prog = eng.compile_multiproc(mil_template, blobs, C, M, S)
        except Exception as e:
            print(f"  Compile Exception: {e}")
    
    if not prog:
        output_tail = "\n".join(buf.getvalue().strip().splitlines()[-4:])
        print(f"  Result: FAILED TO COMPILE")
        if output_tail:
            print(f"  Compiler message:\n{output_tail}")
        return None

    print(f"  Result: COMPILED SUCCESSFULLY!")
    eng._ensure_io(prog)

    # Benchmark execution
    # Warmup
    for _ in range(10):
        eng.submit(prog)

    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        eng.submit(prog)
    t1 = time.perf_counter()
    
    avg_ms = (t1 - t0) / runs * 1000.0
    ops = 2.0 * M * C * S
    tops = (ops / (avg_ms / 1000.0)) / 1e12
    print(f"  Latency: {avg_ms:.3f} ms")
    print(f"  Measured Throughput: {tops:.2f} TOPS (INT8-equivalent)")
    return tops


def main():
    print("============================================================")
    print("  EXPLORING TRUE INT8 ARITHMETIC ON APPLE NEURAL ENGINE")
    print("============================================================")
    
    C = 5120
    M = 8192
    S = 256

    rng = np.random.default_rng(42)
    w_i8 = rng.integers(-128, 127, size=(M, C, 1, 1), dtype=np.int8)
    w_fp16 = (rng.standard_normal((M, C, 1, 1)) * 0.02).astype(np.float16)

    blobs_i8 = {"w.bin": w_i8.tobytes()}
    blobs_fp16 = {"w.bin": w_fp16.tobytes()}

    # -------------------------------------------------------------
    # Baseline 1: Standard FP16 Conv
    # -------------------------------------------------------------
    mil_fp16 = f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<fp16, [{M}, {C}, 1, 1]> w = const()[name=string("w"), val=tensor<fp16, [{M}, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {M}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
  }} -> (y);
}}
"""
    test_mil_variant("FP16 Input + FP16 Weight", mil_fp16, blobs_fp16, C, M, S)

    # -------------------------------------------------------------
    # Baseline 2: FP16 Input + INT8 Weight (constexpr dequant)
    # -------------------------------------------------------------
    s_fp16 = (np.ones((M, 1, 1, 1)) * 0.01).astype(np.float16)
    blobs_w8a16 = {"w.bin": w_i8.tobytes(), "s.bin": s_fp16.tobytes()}
    mil_w8a16 = f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<int8, [{M}, {C}, 1, 1]> wq = const()[name=string("wq"), val=tensor<int8, [{M}, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<fp16, [{M}, 1, 1, 1]> sc = const()[name=string("sc"), val=tensor<fp16, [{M}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/s.bin"), offset=uint64(64)))];
    tensor<fp16, [{M}, {C}, 1, 1]> w = constexpr_blockwise_shift_scale(data=wq, scale=sc)[name=string("dq")];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {M}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
  }} -> (y);
}}
"""
    test_mil_variant("FP16 Input + INT8 Weight (W8A16)", mil_w8a16, blobs_w8a16, C, M, S)

    # -------------------------------------------------------------
    # Experiment 1: INT8 Input + INT8 Weight -> INT32 output
    # -------------------------------------------------------------
    mil_int8_direct = f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<int8, [1, {C}, 1, {S}]> x) {{
    tensor<int8, [{M}, {C}, 1, 1]> w = const()[name=string("w"), val=tensor<int8, [{M}, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<int32, [1, {M}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
  }} -> (y);
}}
"""
    test_mil_variant("INT8 Input + INT8 Weight -> INT32 Output", mil_int8_direct, blobs_i8, C, M, S)

    # -------------------------------------------------------------
    # Experiment 2: INT8 Input + INT8 Weight -> INT8 output (requantized)
    # -------------------------------------------------------------
    mil_int8_i8out = f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<int8, [1, {C}, 1, {S}]> x) {{
    tensor<int8, [{M}, {C}, 1, 1]> w = const()[name=string("w"), val=tensor<int8, [{M}, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<int8, [1, {M}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
  }} -> (y);
}}
"""
    test_mil_variant("INT8 Input + INT8 Weight -> INT8 Output", mil_int8_i8out, blobs_i8, C, M, S)


if __name__ == "__main__":
    main()
