# SPDX-License-Identifier: Apache-2.0
"""Constexpr stacked-expert SwiGLU for Flash-Next pin-and-bake decode."""
from __future__ import annotations

import contextlib
import io
import time

import numpy as np

from runtime.ane_fused_w8a8 import MultiOut, Prog


def _swiglu(p: Prog, x: str, gate, up, down, prefix: str, out_name: str) -> str:
    M = int(gate.shape[0])
    H = int(gate.shape[1])
    p.weight_int8(f"{prefix}gw", gate)
    p.weight_int8(f"{prefix}uw", up)
    p.weight_int8(f"{prefix}dw", down)
    p.conv(f"{prefix}gate", x, f"{prefix}gw", M)
    p.sigmoid(f"{prefix}sig", f"{prefix}gate", M)
    p.mul(f"{prefix}silu", f"{prefix}gate", f"{prefix}sig", M)
    p.conv(f"{prefix}upv", x, f"{prefix}uw", M)
    p.mul(f"{prefix}act", f"{prefix}silu", f"{prefix}upv", M)
    p.conv(out_name, f"{prefix}act", f"{prefix}dw", H)
    return out_name


def compile_stacked_swiglu(
    engine, gate, up, down, *, seq: int = 32, tag: str = "swiglu",
    shared_gate=None, shared_up=None, shared_down=None, sgate=None,
):
    """x[H] -> routed SwiGLU (+ optional shared expert). Constexpr int8.

    Shared path returns extra outputs `sh` [H] and `sg` [1]; the host applies
    `y + sigmoid(sg) * sh` so we do not depend on MIL broadcast.
    """
    gate = np.ascontiguousarray(gate, np.float32)
    up = np.ascontiguousarray(up, np.float32)
    down = np.ascontiguousarray(down, np.float32)
    M, H = int(gate.shape[0]), int(gate.shape[1])
    if up.shape != (M, H) or down.shape != (H, M):
        raise ValueError(f"swiglu shapes gate={gate.shape} up={up.shape} down={down.shape}")
    p = Prog(H, seq)
    y = _swiglu(p, "x", gate, up, down, "", "y")
    p.out("y", H)
    has_shared = shared_gate is not None
    if has_shared:
        sg = np.ascontiguousarray(shared_gate, np.float32)
        su = np.ascontiguousarray(shared_up, np.float32)
        sd = np.ascontiguousarray(shared_down, np.float32)
        sw = np.ascontiguousarray(np.asarray(sgate, np.float32).reshape(1, -1))
        _swiglu(p, "x", sg, su, sd, "s", "sh")
        p.out("sh", H)
        p.weight_int8("wsg", sw)
        p.conv("sg", "x", "wsg", 1)
        p.out("sg", 1)
    mil, packed = p.mil()
    cap = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        try:
            prog = engine.compile_multiproc(
                mil, {"weight.bin": packed}, H, H, seq,
                raw_weight_files=frozenset({"weight.bin"}))
        except Exception as exc:  # noqa: BLE001
            cap.write(f"exception: {exc!r}\n")
            prog = None
    if prog is None:
        tail = " | ".join(cap.getvalue().strip().splitlines()[-6:])
        raise RuntimeError(f"stacked swiglu compile failed for {tag}: {tail}")
    runner = MultiOut(prog, H, seq, p.outputs)
    extra = "+shared" if has_shared else ""
    print(f"  bake {tag:<36} swiglu M={M}{extra}  "
          f"{time.perf_counter()-t0:.2f}s", flush=True)
    return runner
