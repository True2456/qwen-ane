#!/usr/bin/env python3
"""Compile one Flash-Next stacked-expert SwiGLU as constexpr int8 and time it.

    python3 -u probes/ane_baked_moe_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("Q38_ANE_ENGINE", str(ROOT))
sys.path.insert(0, str(ROOT))

from runtime.q38_ane_engine import AneEngine  # noqa: E402

H, I, K, S = 2560, 640, 16, 32


def main() -> int:
    rng = np.random.default_rng(0)
    gu = rng.standard_normal((K * 2 * I, H), dtype=np.float32).astype(np.float32) * 0.02
    dn = rng.standard_normal((H, K * I), dtype=np.float32).astype(np.float32) * 0.02
    x = rng.standard_normal((S, H), dtype=np.float32).astype(np.float32) * 0.02
    h = rng.standard_normal((S, K * I), dtype=np.float32).astype(np.float32) * 0.02

    eng = AneEngine()
    t0 = time.perf_counter()
    gu_p = eng.compile_linear(gu, S, quantized=True, keep_weight_dequant=False)
    print(f"compile gu {gu.shape}  {time.perf_counter()-t0:.2f}s  ok={gu_p is not None}", flush=True)
    t0 = time.perf_counter()
    dn_p = eng.compile_linear(dn, S, quantized=True, keep_weight_dequant=False)
    print(f"compile dn {dn.shape}  {time.perf_counter()-t0:.2f}s  ok={dn_p is not None}", flush=True)
    if gu_p is None or dn_p is None:
        return 1

    def once():
        y1 = eng.evaluate(gu_p, x)
        y2 = eng.evaluate(dn_p, h)
        return y1, y2

    once()
    ts = []
    for _ in range(11):
        t = time.perf_counter()
        once()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    med = ts[len(ts) // 2]
    flops = 2 * K * 2 * I * H * S + 2 * H * K * I * S
    print(f"eval pair median {med:.2f} ms  ({flops/med/1e9:.1f} TFLOP/s at S={S} K={K})")
    print(f"48 layers * median = {48*med:.0f} ms/token  ({1000/(48*med):.2f} tok/s if MoE-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
