#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ane_as.py - Standalone Apple Neural Engine Bytecode Synthesizer & Direct Evaluation CLI."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (
    AneEngine,
    _desc,
    _msg,
    _iosurface_view,
    _nsdict,
    _objc,
    _sel,
    _cls,
)
from probes.ane_gdn_conv1d import run as compile_conv1d


class AneAssembler:
    """Lightweight standalone ANE compiler and direct hardware runner."""

    def __init__(self):
        self.engine = AneEngine()
        assert self.engine.available, "AppleNeuralEngine.framework is not available on this system"
        
        self.client_cls = _cls("_ANEClient")
        self.client = _msg(self.client_cls, "sharedConnection")
        if not self.client:
            self.client = _msg(_msg(self.client_cls, "alloc"), "init")
        assert self.client, "Failed to connect to _ANEClient"

    def assemble_and_evaluate_conv1d(
        self,
        channels: int = 64,
        seq_len: int = 32,
        kernel_size: int = 4,
        iterations: int = 50,
    ) -> dict:
        """Assemble a depthwise causal 1d conv model and execute on ANE hardware."""
        ret = compile_conv1d(channels, "none")
        if not (isinstance(ret, tuple) and ret[0] and ret[1] is not None):
            raise RuntimeError(f"Failed to compile ANE model for channels={channels}")
        ok, prog = ret

        ane_model = _msg(prog.model, "model")
        prog_handle = _msg(ane_model, "programHandle", restype=ctypes.c_ulonglong)

        # Setup IO
        self.engine._ensure_io(prog)
        rng = np.random.default_rng(42)
        x_data = rng.normal(0, 0.2, (channels, seq_len)).astype(np.float32)
        with _iosurface_view(prog._in_surf, (channels, seq_len), np.float16) as dst:
            dst[:] = x_data.astype(np.float16)

        DoEvalDirect = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
        )
        err_ptr = ctypes.c_void_p(0)
        opts = _nsdict({})

        # Warmup
        for _ in range(5):
            DoEvalDirect(("objc_msgSend", _objc))(
                self.client,
                _sel("doEvaluateDirectWithModel:options:request:qos:error:"),
                ane_model,
                opts,
                prog._request,
                21,
                ctypes.byref(err_ptr),
            )

        # Measure direct hardware execution
        t0 = time.perf_counter()
        for _ in range(iterations):
            ok = DoEvalDirect(("objc_msgSend", _objc))(
                self.client,
                _sel("doEvaluateDirectWithModel:options:request:qos:error:"),
                ane_model,
                opts,
                prog._request,
                21,
                ctypes.byref(err_ptr),
            )
            if not ok:
                raise RuntimeError(f"Direct ANE evaluation failed: {_desc(err_ptr.value)}")
        total_time = time.perf_counter() - t0
        avg_eval_ms = (total_time / iterations) * 1e3

        # Numerical verification against output surface
        with _iosurface_view(prog._out_surf, (channels, seq_len), np.float16) as src:
            got = np.array(src, np.float32)

        # Scalar reference computation
        rng_w = np.random.default_rng(2000 + channels)
        w = rng_w.normal(0, 0.15, (channels, 1, 1, kernel_size)).astype(np.float16)
        xp = np.pad(x_data, ((0, 0), (kernel_size - 1, 0)))
        ref = np.empty_like(x_data)
        wf = w.astype(np.float32).reshape(channels, kernel_size)
        for t in range(seq_len):
            ref[:, t] = np.sum(xp[:, t:t+kernel_size] * wf, axis=1)

        abs_err = float(np.max(np.abs(got - ref)))
        rel_err = float(abs_err / (np.max(np.abs(ref)) + 1e-9))

        # Compute FLOPs and effective compute rate
        flops = 2.0 * channels * kernel_size * seq_len
        gflops_s = (flops / (avg_eval_ms * 1e-3)) / 1e9

        return {
            "channels": channels,
            "seq_len": seq_len,
            "kernel_size": kernel_size,
            "program_handle": hex(prog_handle),
            "avg_eval_ms": avg_eval_ms,
            "gflops_s": gflops_s,
            "rel_err": rel_err,
            "abs_err": abs_err,
        }


def main():
    parser = argparse.ArgumentParser(description="Apple Neural Engine Direct Synthesizer & Evaluator (ane_as)")
    parser.add_argument("--channels", type=int, default=64, help="Channel dimension C (default: 64)")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence/spatial length S (default: 32)")
    parser.add_argument("--kernel", type=int, default=4, help="Kernel width K (default: 4)")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations (default: 50)")
    parser.add_argument("--test-all", action="store_true", help="Run full suite across multiple shapes")

    args = parser.parse_args()

    print("=" * 70)
    print("  Apple Neural Engine Direct Synthesizer & Evaluator (ane_as)")
    print("=" * 70)

    assembler = AneAssembler()

    if args.test_all:
        test_shapes = [64, 10240]
        print(f"\nRunning benchmark suite across {len(test_shapes)} dimensions...")
        for c in test_shapes:
            res = assembler.assemble_and_evaluate_conv1d(c, args.seq_len, args.kernel, args.iters)
            status = "PASS" if res["rel_err"] < 1e-3 else "FAIL"
            print(f"  [{status}] C={c:<5} S={res['seq_len']} | Handle: {res['program_handle']} | "
                  f"Eval: {res['avg_eval_ms']:.3f} ms | Throughput: {res['gflops_s']:.2f} GFLOP/s | "
                  f"RelErr: {res['rel_err']:.2e}")
    else:
        res = assembler.assemble_and_evaluate_conv1d(args.channels, args.seq_len, args.kernel, args.iters)
        status = "PASS" if res["rel_err"] < 1e-3 else "FAIL"
        print(f"\nResults for C={res['channels']}, S={res['seq_len']}, K={res['kernel_size']}:")
        print(f"  Status:             {status}")
        print(f"  Program Handle:     {res['program_handle']}")
        print(f"  Hardware Eval:      {res['avg_eval_ms']:.3f} ms / dispatch")
        print(f"  Compute Throughput: {res['gflops_s']:.2f} GFLOP/s")
        print(f"  Max Absolute Error: {res['abs_err']:.6e}")
        print(f"  Relative Error:     {res['rel_err']:.6e}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
