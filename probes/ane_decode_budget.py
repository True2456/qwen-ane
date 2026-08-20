"""Where does a decode token's time actually go?

Whole-model pure int4 measures 2.770 tok/s (361 ms/token).  Ranking any
optimization needs to know how much of that is learned-weight convolution and
how much is everything else.  Measures every distinct real conv shape at the
production decode width of 32, weights the result by how many times each fires
per token, and reports the remainder against the known end-to-end rate.

The remainder is not attributed here -- it is the sequence cores, norms, host
IOSurface copies and per-dispatch driver floor together.  Naming it as one
number is the point: it says whether to attack convolution or overhead.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ane_peak_real import build, measure

MEASURED_TOK_S = 2.770          # README, pure int4, no MTP, 32 generated tokens
S = 32                          # every pure_ane.py program compiles at width 32

# shape, how many fire per decode token, label
BLOCKS = (
    ("mlp gate+up", 34816, 5120, 64),
    ("mlp down",     5120, 17408, 64),
    ("gdn in_proj", 16480, 5120, 48),
    ("attn qkv",    14336, 5120, 16),
    ("lm_head x4",  62080, 5120, 4),
)

print(f"int4 convolution cost per decode token, all programs at width S={S}\n")
print(f"  {'block':>14} {'calls':>6} {'ms/call':>9} {'ms/token':>10} {'% of token':>11}")
total = 0.0
rows = []
for label, M, H, calls in BLOCKS:
    p, _ = build(M, H, S, 4)
    if p is None:
        print(f"  {label:>14} {'compile rejected':>28}"); continue
    ms, _ = measure(p, M, H, S, n=15)
    del p
    if ms is None:
        print(f"  {label:>14} {'ZERO output':>28}"); continue
    per_token = ms * calls
    total += per_token
    rows.append((label, calls, ms, per_token))

budget = 1000.0 / MEASURED_TOK_S
for label, calls, ms, per_token in rows:
    print(f"  {label:>14} {calls:>6} {ms:>9.3f} {per_token:>10.1f} "
          f"{100*per_token/budget:>10.1f}%")

print(f"\n  {'convolution total':>14} {'':>6} {'':>9} {total:>10.1f} "
      f"{100*total/budget:>10.1f}%")
print(f"  {'token budget':>14} {'':>6} {'':>9} {budget:>10.1f} "
      f"{'100.0%':>11}   (at {MEASURED_TOK_S} tok/s)")
print(f"  {'everything else':>14} {'':>6} {'':>9} {budget-total:>10.1f} "
      f"{100*(budget-total)/budget:>10.1f}%")

# The driver's fixed cost per dispatch, measured directly: a trivially small
# program whose arithmetic is negligible against the submit path.
p, _ = build(64, 64, S, 4)
if p is not None:
    ms, _ = measure(p, 64, 64, S, n=200)
    calls = sum(c for _, _, _, c in BLOCKS) + 48 + 48 + 16 + 16   # + cores/norms
    print(f"\n  per-dispatch floor {ms:.3f} ms; ~{calls} dispatches/token "
          f"= {ms*calls:.1f} ms ({100*ms*calls/budget:.0f}% of the token budget)")
