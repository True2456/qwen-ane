"""How many decode lanes are free? Cost per CALL, not per token.

The hardware pads decode to width 32, so a 1-token step computes 32 lanes and
discards 31.  If a call at width 64 costs the same as a call at width 32, the
free capacity is 64 positions per dispatch, not 32 -- and speculative decoding
can verify twice as many candidates for nothing.  Measures ms per dispatch, so
a flat row means the extra lanes are literally free.

down_proj is measured both unsplit and split 4 ways across input channels,
because it is the one projection that does not stay flat.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ane_peak_real import build, build_split, measure

WIDTHS = (32, 48, 64, 96, 128)
SHAPES = (
    ("mlp gate+up", 34816, 5120, 1),
    ("gdn in_proj", 16480, 5120, 1),
    ("attn qkv",    14336, 5120, 1),
    ("mlp down x1",  5120, 17408, 1),
    ("mlp down x4",  5120, 17408, 4),
)

print("ms per DISPATCH at int4 (flat row = those lanes cost nothing)\n")
print(f"  {'block':>13} " + "".join(f"{('S=%d'%s):>9}" for s in WIDTHS) + "   vs S=32")
for label, M, H, parts in SHAPES:
    row, first = "", None
    for S in WIDTHS:
        p, deq = (build(M, H, S, 4) if parts == 1
                  else build_split(M, H, S, 4, parts))
        if p is None:
            row += f"{'rej':>9}"; continue
        ms, _ = measure(p, M, H, S, n=15)
        del p
        if ms is None:
            row += f"{'ZERO':>9}"; continue
        first = first if first is not None else ms
        row += f"{ms:>9.3f}"
    tail = f"{ms/first:>6.2f}x" if first else ""
    print(f"  {label:>13} " + row + f"   {tail}", flush=True)

print("\nA flat row means N lanes cost the same as 32. Lanes beyond the first")
print("are free verification slots for speculative decoding.")
