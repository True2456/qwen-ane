#!/usr/bin/env python3
"""How wide can one GDN MIL submit go, and what does it cost per token?

Decode wants a narrow graph: a submit costs ~1.13 ms that does not scale with
tokens, plus ~0.146 ms a token, so K=4 is the right shape when only two or
three drafts are accepted. Prefill has no such limit — every slot carries a
real token — so the fixed cost should amortise until the graph stops
compiling. The emitter's sequence width is 32, which caps k there.

    ~/.rindi/venvs/coreai/bin/python probes/mil_wide_prefill.py [k ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_flashnext_coreai import _load_layer, HC_W, SEQ_DEFAULT
from flashnext_multitoken_step import MultiTokenStep
from runtime.mil_gdn_backend import MilGdnLayer

S = SEQ_DEFAULT
REPEAT = 31


def main() -> None:
    ks = [int(v) for v in sys.argv[1:]] or [1, 4, 8, 16, 32]
    loader, w = _load_layer(0)
    ref = MultiTokenStep(w, 1).eval().half()
    rng = np.random.default_rng(7)
    x = np.zeros((1, HC_W, 1, S), np.float16)
    x[0, :, 0, :] = (rng.standard_normal((HC_W, S)) * 0.05).astype(np.float16)
    print(f"{'k':>4} {'compile s':>10} {'ms/submit':>10} {'ms/token':>9} "
          f"{'36 layers, tok/s':>17}")
    for k in ks:
        t0 = time.perf_counter()
        try:
            lay = MilGdnLayer(0, w, ref, k=k)
        except Exception as exc:  # noqa: BLE001
            print(f"{k:4d}  FAILED: {exc}")
            continue
        tc = time.perf_counter() - t0
        lay(x, n=k)
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            lay(x, n=k)
        ms = (time.perf_counter() - t0) / REPEAT * 1e3
        print(f"{k:4d} {tc:10.1f} {ms:10.3f} {ms / k:9.3f} "
              f"{1e3 / (ms / k * 36):17.1f}")
        del lay
    loader.close()


if __name__ == "__main__":
    main()
