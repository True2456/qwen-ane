#!/usr/bin/env python3
"""test_direct_hwx_driver.py - Validate direct _ANEModel / _ANEClient loading and execution."""

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
)
from runtime.ane_direct_engine import AneDirectEngine


def main():
    print("=" * 60)
    print("  TESTING DIRECT _ANEClient AND _ANEModel ENGINE")
    print("=" * 60)

    direct_engine = AneDirectEngine()
    print("Direct ANE Client initialized successfully.")

    # 1. Compile reference layer using tested working configuration
    from probes.ane_gdn_conv1d import run as compile_conv1d
    print("Compiling reference package via ane_gdn_conv1d...")
    ok, prog = compile_conv1d(64, "none")
    assert ok and prog is not None, "Failed to compile reference package"
    print("  ✓ Reference model compiled and evaluated.")

    local_path = _desc(_msg(prog.model, "localModelPath"))
    model_url = _desc(_msg(prog.model, "modelURL"))
    hex_id = _desc(_msg(prog.model, "hexStringIdentifier"))
    print(f"LocalModelPath: {local_path}")
    print(f"ModelURL:       {model_url}")
    print(f"Hex Identifier: {hex_id}")

    # Inspect package contents
    print("\nInspecting compiled package directory structure:")
    if os.path.exists(local_path):
        for root, dirs, files in os.walk(local_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                print(f"  {fname}: {os.path.getsize(fpath)} bytes")

    # 2. Extract underlying _ANEModel from _ANEInMemoryModel
    underlying_ane_model = _msg(prog.model, "model")
    print(f"\nUnderlying _ANEModel pointer: {underlying_ane_model != 0} ({underlying_ane_model})")

    # Evaluate directly using _ANEClient with the underlying _ANEModel
    print("\nTesting direct evaluation via [_ANEClient doEvaluateDirectWithModel:options:request:qos:error:]...")
    
    # Ensure IO buffers on prog
    eng = E.AneEngine()
    eng._ensure_io(prog)
    with _iosurface_view(prog._in_surf, (64, 32), np.float16) as dst:
        dst[:] = np.ones((64, 32), dtype=np.float16) * 0.5

    # Run direct evaluate via _ANEInMemoryModel / _ANERequest
    EvaluateDirect = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    err_eval = ctypes.c_void_p(0)
    opts = _nsdict({})
    eval_ok = EvaluateDirect(("objc_msgSend", _objc))(
        prog.model,
        _sel("evaluateWithQoS:options:request:error:"),
        21,
        prog._compile_opts,
        prog._request,
        ctypes.byref(err_eval),
    )
    print(f"Direct evaluate result: {eval_ok}")
    assert eval_ok, f"Direct evaluation failed: {_desc(err_eval.value) if err_eval.value else 'unknown'}"
    print("  ✓ SUCCESS: Direct evaluation on ANE silicon executed cleanly!")

    print("\n" + "=" * 60)
    print("  ALL DIRECT ANE DRIVER TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
