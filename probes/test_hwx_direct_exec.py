#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""test_hwx_direct_exec.py - Phase 2: Direct _ANEModel & _ANEClient Hardware Execution Gate."""

from __future__ import annotations

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


def run_direct_hwx_gate():
    print("=" * 70)
    print("  PHASE 2: DIRECT _ANEModel & _ANEClient HARDWARE EXECUTION GATE (M5 Max)")
    print("=" * 70)

    # 1. Initialize direct _ANEClient
    client_cls = _cls("_ANEClient")
    client = _msg(client_cls, "sharedConnection")
    if not client:
        client = _msg(_msg(client_cls, "alloc"), "init")
    assert client, "Failed to get _ANEClient sharedConnection"
    print("✓ Direct _ANEClient connection established.")

    # 2. Test shapes: C=64 (micro) and C=10240 (full Qwen GDN layer)
    shapes = [64, 10240]
    eng = AneEngine()

    DoEvalDirect = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
    )

    EvalWithModel = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
    )

    for C in shapes:
        print(f"\n--- Testing C={C:<5} (Spatial width S=32, K=4) ---")
        ret = compile_conv1d(C, "none")
        assert isinstance(ret, tuple) and ret[0] and ret[1] is not None, f"Compilation failed for C={C}"
        ok, prog = ret

        ane_model = _msg(prog.model, "model")
        assert ane_model, f"Failed to get underlying _ANEModel for C={C}"

        prog_handle = _msg(ane_model, "programHandle", restype=ctypes.c_ulonglong)
        print(f"  ✓ _ANEModel extracted: 0x{ane_model:x} (programHandle: 0x{prog_handle:x})")

        # Setup input data
        eng._ensure_io(prog)
        rng = np.random.default_rng(1000 + C)
        x_data = rng.normal(0, 0.2, (C, 32)).astype(np.float32)
        with _iosurface_view(prog._in_surf, (C, 32), np.float16) as dst:
            dst[:] = x_data.astype(np.float16)

        # 2a. Execute via doEvaluateDirectWithModel:
        err_direct = ctypes.c_void_p(0)
        opts = _nsdict({})
        t0 = time.perf_counter()
        direct_ok = DoEvalDirect(("objc_msgSend", _objc))(
            client,
            _sel("doEvaluateDirectWithModel:options:request:qos:error:"),
            ane_model,
            opts,
            prog._request,
            21,
            ctypes.byref(err_direct),
        )
        t_direct = (time.perf_counter() - t0) * 1e3
        assert direct_ok, f"doEvaluateDirect failed: {_desc(err_direct.value)}"
        print(f"  ✓ doEvaluateDirectWithModel: PASS ({t_direct:.3f} ms)")

        # 2b. Execute via evaluateWithModel:
        err_eval = ctypes.c_void_p(0)
        t0 = time.perf_counter()
        eval_ok = EvalWithModel(("objc_msgSend", _objc))(
            client,
            _sel("evaluateWithModel:options:request:qos:error:"),
            ane_model,
            opts,
            prog._request,
            21,
            ctypes.byref(err_eval),
        )
        t_eval = (time.perf_counter() - t0) * 1e3
        assert eval_ok, f"evaluateWithModel:         PASS ({t_eval:.3f} ms)"

        # 2c. Benchmark raw hardware dispatch floor
        n_bench = 100 if C < 10240 else 30
        t0 = time.perf_counter()
        for _ in range(n_bench):
            DoEvalDirect(("objc_msgSend", _objc))(
                client,
                _sel("doEvaluateDirectWithModel:options:request:qos:error:"),
                ane_model,
                opts,
                prog._request,
                21,
                ctypes.byref(err_direct),
            )
        bench_avg_ms = (time.perf_counter() - t0) * 1e3 / n_bench
        print(f"  ✓ Hardware dispatch average: {bench_avg_ms:.3f} ms/call ({n_bench} iterations)")

    print("\n" + "=" * 70)
    print("  ALL DIRECT _ANEClient HARDWARE EXECUTION TESTS PASSED (100% SUCCESS)")
    print("=" * 70)


if __name__ == "__main__":
    run_direct_hwx_gate()
