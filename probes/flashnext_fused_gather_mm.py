"""Live ANE expert selection + projection; validate routing and table mutation.

Run each shape in its own process because compiler failures can abort Python.
Tables are synthetic FP16, with Flash-Next matrix dimensions in the large cases.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.ane_lookup import AneGatherMM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=["tiny", "gate", "down"], default="tiny")
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--method", choices=["conv", "matmul", "reduce"], default="conv")
    args = parser.parse_args()
    e, o, i, k = ((8, 32, 64, 2) if args.shape == "tiny" else
                  (args.experts, 1280, 2560, 10) if args.shape == "gate" else
                  (args.experts, 2560, 640, 10))
    rng = np.random.default_rng(918)
    print(f"shape E={e} out={o} in={i} k={k}", flush=True)
    table = np.empty((e, o, i), np.float16)
    for row in table:
        row[:] = rng.normal(0, 1 / np.sqrt(i), row.shape)
    g = AneGatherMM(e, o, i, k, method=args.method)
    print(f"compiled; symbols={g.symbols}", flush=True)
    g.write_table(table)
    times, errors = [], []
    for step in range(12):
        if step == 6:
            table *= np.float16(-0.5)
            g.write_table(table)
        x = rng.normal(0, 0.2, i).astype(np.float16)
        ids = rng.choice(e, k, replace=False)
        t = time.perf_counter()
        got = g.project(x, ids)
        ms = (time.perf_counter() - t) * 1e3
        want = table[ids].astype(np.float32) @ x.astype(np.float32)
        rel = float(np.linalg.norm(got-want) / max(np.linalg.norm(want), 1e-12))
        print(f"step={step} ms={ms:.3f} rel={rel:.6g}", flush=True)
        if not np.isfinite(got).all() or rel > 0.02:
            raise AssertionError((step, rel))
        errors.append(rel)
        if step >= 2 and step != 6:
            times.append(ms)
    print(f"PASS changing ids, activations, table; median={np.median(times):.3f}ms "
          f"p95={np.percentile(times,95):.3f}ms max_rel={max(errors):.6g}", flush=True)


if __name__ == "__main__":
    main()
