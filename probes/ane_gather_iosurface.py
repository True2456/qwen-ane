#!/usr/bin/env python3
"""IOSurface gather vs CoreML predict-copy.

If gather is implemented like AneDynamicLinear, write_table once then N
id-only evaluates must be fast. CoreML predict that re-feeds the table
every call is the broken path.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _rel(a, b) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def bench_coreml(vocab: int, dim: int, k: int, table, ids, expect) -> None:
    from runtime.ane_lookup import _compile_espresso_gather

    net, shape, weights, ml = _compile_espresso_gather(vocab, dim, k)
    print(f"  espresso.shape {shape}")
    feed = {"table": table.astype(np.float32), "ids": ids}
    for _ in range(3):
        y = ml.predict(feed)["y"]
    t0 = time.perf_counter()
    for _ in range(20):
        y = ml.predict(feed)["y"]
    ms = (time.perf_counter() - t0) / 20 * 1e3
    got = np.reshape(np.asarray(y, np.float32), expect.shape)
    print(f"  coreml predict-copy  {ms:.3f} ms  rel={_rel(got, expect):.3e}")


def bench_iosurface(vocab: int, dim: int, k: int, table, ids, expect) -> None:
    import ctypes
    from runtime.ane_lookup import AneGather
    from runtime.q38_ane_engine import _desc, _objc_call

    g = AneGather(vocab, dim, k)
    ident = _desc(_objc_call(ctypes.c_void_p, (), g.model, "hexStringIdentifier"))
    print(f"  symbols={g.symbols}")
    print(f"  ident={ident[:96]}")
    t_w0 = time.perf_counter()
    g.write_table(table)
    t_w = (time.perf_counter() - t_w0) * 1e3
    y = g.gather(ids)
    y2 = g.gather((ids + 1) % vocab)
    print(f"  write_table {t_w:.3f} ms  first gather rel={_rel(y, expect):.3e}")
    print(f"  ids vs ids+1 L2={np.linalg.norm(y - y2):.4g}")
    for _ in range(3):
        g.gather(ids)
    t0 = time.perf_counter()
    for _ in range(20):
        y = g.gather(ids)
    ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"  ids-only evaluate    {ms:.3f} ms  rel={_rel(y, expect):.3e}")


def main() -> int:
    rng = np.random.default_rng(0)
    k = 10
    for vocab, dim in ((64, 2560), (512, 2560)):
        print(f"\n== V={vocab} D={dim} K={k} ==")
        table = rng.standard_normal((vocab, dim), dtype=np.float32)
        ids = rng.integers(0, vocab, size=k, dtype=np.int32)
        expect = table[ids]
        try:
            bench_coreml(vocab, dim, k, table, ids, expect)
        except Exception as exc:
            print(f"  coreml failed: {exc!r}")
        try:
            bench_iosurface(vocab, dim, k, table, ids, expect)
        except Exception as exc:
            import traceback
            print(f"  iosurface failed: {exc!r}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
