"""What does compiling every program at width 32 cost during prefill?

tools/pure_ane.py builds every ANE program with width=32 and fills at most
three real lanes, so a 261-token prompt is ~87 sequential passes of the whole
64-layer stack.  The projections are the weight-heavy blocks and their cost is
nearly flat in S until the array fills, so per-token cost should fall steeply
with width.  Measures the real projection shapes at int4 across widths, and
re-tests the down_proj input-channel split at each width to see whether the
fix matters for decode (S=32) or only for prefill.
"""
import contextlib, io, os, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ane_peak_real import build, build_split, measure   # same idioms, verified

eng = AneEngine()

REAL = (
    ("mlp gate+up", 34816, 5120),
    ("mlp down", 5120, 17408),
    ("gdn in_proj", 16480, 5120),
    ("attn qkv", 14336, 5120),
)
WIDTHS = (32, 64, 128, 256, 512)

print("1. int4 projections: microseconds PER TOKEN at each compiled width")
print(f"  {'projection':>14} " + "".join(f"{('S=%d'%s):>11}" for s in WIDTHS))
base = {}
for label, M, H in REAL:
    row = ""
    for S in WIDTHS:
        p, _ = build(M, H, S, 4)
        if p is None:
            row += f"{'rej':>11}"; continue
        ms, _ = measure(p, M, H, S, n=9)
        if ms is None:
            row += f"{'ZERO':>11}"; del p; continue
        us = ms * 1000 / S
        if S == 32:
            base[label] = us
        row += f"{us:>11.1f}"
        del p
    print(f"  {label:>14} " + row, flush=True)

print(f"\n  speedup per token, S=512 vs S=32:")
for label, M, H in REAL:
    p, _ = build(M, H, 512, 4)
    ms, _ = measure(p, M, H, 512, n=9)
    if ms and label in base:
        print(f"    {label:>14} {base[label]/(ms*1000/512):>5.1f}x")
    del p

print("\n2. Does the down_proj split help at DECODE width too?")
print(f"  {'width':>6} {'parts=1 ms':>11} {'parts=4 ms':>11} {'gain':>7}")
for S in WIDTHS:
    M, H = 5120, 17408
    p1, _ = build(M, H, S, 4)
    m1, _ = measure(p1, M, H, S, n=9) if p1 else (None, None)
    del p1
    p4, deq = build_split(M, H, S, 4, 4)
    m4, rel = measure(p4, M, H, S, deq=deq, n=9) if p4 else (None, None)
    del p4
    if m1 and m4:
        print(f"  {S:>6} {m1:>11.2f} {m4:>11.2f} {m1/m4:>6.2f}x   rel={rel:.2e}",
              flush=True)
    else:
        print(f"  {S:>6} {'--':>11} {'--':>11}", flush=True)
