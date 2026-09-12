# Round 3: what is left on the Flash-Next ANE port

Three problems, hardest first. Each one is self-contained, has a measured
starting point in this repository, and has a correctness gate that does not
depend on anyone's judgement. Rounds 1 and 2 are in `ANE-MIL-COST-MODEL.md`
and `ANE-GPU-OVERLAP.md`; read both before starting, along with `PREFILL.md`.

Where things stand. Decode is 20.6 tok/s on a code prompt and 14.5 on prose.
Prefill is 25.6 tok/s on the decode graphs, 58.4 with a second set baked at
k=32. Quality costs about 2% perplexity against the 4-bit MLX reference.

## Problem 1: the chunked delta rule on the ANE

**Worth: prefill from 58 to somewhere near 100 tok/s, and it is the only term
that does not shrink when the graph gets wider.**

The gated delta net recurrence runs one token at a time. Per head, with a
scalar gate:

```
S <- S * g_t                     # S is [Hv=48, Dv=128, Dk=128], 1.57 MB fp16
u <- S k_t                       # [Dv]
d <- (v_t - u) * beta_t          # [Dv]
S <- S + d k_t^T
y_t <- S q_t                     # note: the updated S
```

`gdn_core` in `probes/flashnext_mil_layer.py` emits this once per slot and the
graph is unrolled over k slots. `probes/mil_wide_prefill.py` prices it: a
submit costs 1.17 ms that does not depend on k, plus 0.145 ms a token. That
0.145 ms is the state being read and written through memory once per token,
and across 36 layers it is 5.2 ms a token — 28% of prefill at k=32, and the
one term that a wider graph does not amortise.

The reference gets past it with a Metal kernel that holds the state in
registers for a whole sequence (`mlx_lm/models/gated_delta.py`,
`_make_gated_delta_kernel`). The ANE has no equivalent. The way out is the
chunked form: process C tokens with matmuls and touch the state once per
chunk. The intra-chunk dependency is the delta rule's `u <- S k_t`, which is
resolved by the WY / UT transform — build the strictly lower triangular
`A = tril(diag(beta) K K_decayed^T, -1)`, invert `(I + A)` once per chunk, and
the whole chunk becomes matmuls. For C=32 that inverse is a 32x32 matrix per
head. Forward substitution is 32 sequential steps and probably not worth it;
blocked doubling on a strictly lower triangular matrix converges in log2(C)
matmuls, which is five.

Start from the published chunkwise algorithm rather than deriving it. Note
the gate here is scalar per head, not per key dimension — `compute_g` returns
`[B, T, Hv]` — which is the easy case.

**Gate.** `probes/mil_k_check.py K` for K in 1, 4, 8, 16, 32 must show per-slot
mixed error in the 0.026 to 0.035 band, final state under 0.01 and conv under
0.02, matching what it prints today. Then 1024 tokens of
`eval/prose.txt` scored after a 512-token prefill must land within 0.01 in log
likelihood of 6.58.

**Measure with** `probes/mil_wide_prefill.py` for the per-token slope, and
`generate --prompt-ids <511 ids> --max-new 4` for the end-to-end prefill rate
and its breakdown.

**Traps**, all of which have cost a day each here. Every ANE I/O last dimension
must be a multiple of 32, or the host and the ANE silently disagree about the
stride. Output surfaces bind in alphabetical symbol order. The conv window is
written 64 elements wide whatever the graph declares once more than 16 are
kept. The emitter's sequence width is 32, so C > 32 means widening `S` first,
and `S` must stay a multiple of 32. And `ANE-MIL-COST-MODEL.md` says the cost
of a reduction follows its output grid, so reducing a short axis is far worse
than reducing a long one.

## Problem 2: the long-context quality residual

**Worth: it decides whether the 256k claim is real.**

Perplexity against the 4-bit MLX reference costs about 2% at 2048 tokens and
grows with position — roughly a factor of 2.7 between an empty context and
eight thousand tokens, after controlling for the corpus changing.
`QUALITY-EVAL.md` has the method and the numbers.

Three causes are already eliminated with measurements: the indexer's int8
projection, the key set shared across a block's slots, and a silent fallback
to recency. Forcing recency-only prices the block selection at 0.019 in log
likelihood out of the 0.043 that remains at 4096 tokens.

The decisive experiment nobody has run is a direct comparison of the selected
index sets: drive the port's indexer and `mlx_lm`'s over the same hidden
states and measure how much of the top-512 block set they agree on, position
by position. If the sets match, the loss is in the int8 attention over a
sparse window and not in the selection at all.

**Gate.** Either a fix that closes most of the 0.043 at 4096 tokens without
costing decode speed, or a measurement that says where it comes from and why
it cannot be closed.

## Problem 3: the routed experts on the ANE

**Worth: the 3.55 ms a token the GPU takes during prefill and the 29 ms a
block during decode, plus the whole energy argument.**

Still open from round 2, still the highest ceiling and the lowest probability.
512 experts, ten per token, 4-bit, selected per token. The ANE cannot gather
weights dynamically, so the shapes of an answer are a resident subset with a
GPU fallback for misses, or restructuring so the selection becomes a
procedure index. `compile_multiproc` already supports one procedure per expert.

Doing this is also what would let adjacent layers fuse into one submit, which
removes the 1.17 ms fixed cost that currently gets paid 36 times a block.

**Gate.** End to end, same tokens as the GPU path, faster, and with the ANE
doing the expert arithmetic rather than staging weights through IOSurfaces at
the same bandwidth the GPU already achieves.
