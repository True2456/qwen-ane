# Three open problems on the Flash-Next Neural Engine port

Written 2026-09-12, after the port reached 15.4 tok/s. Apple M5 Max, 137 GB,
macOS build 26A428. Everything below is measured on this machine.

The first round of this document asked for the routed MoE on the ANE. That is
still open and is still the biggest single win, but two new problems have
become better posed than it, because this session disproved the models people
(including me) were optimizing against.

## Where the time goes

Decode runs all 48 layers as hand-written MIL through the private ANE path,
with MTP speculative decoding. A verification pass emits 2.46 tokens on
predictable text and 1.96 on free prose:

| | ms of a 148 ms pass |
|---|---|
| 36 GDN layers, ANE | 70 |
| 12 QSA layers, ANE + host | 31 |
| routed MoE, GPU | 28 |
| router, state commit, lm_head, embed, host | 16 |

Run it with:

```
FLASHNEXT_SPEC=4 FLASHNEXT_MOE=mlxresident FLASHNEXT_HEAD=mlx \
FLASHNEXT_MIL_GDN=1 FLASHNEXT_MIL_QSA=1 \
~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py generate \
  --prompt-ids 760,6511,3177,314,9338,369,11751,11,321,279,6511,3177,314,9564,369,19241,13,561,6511,3177,314,14898,369,21047,11,321,279,6511,3177,314,17163,369 \
  --max-new 64
```

Getting to 25 tok/s at the current 2.46 tokens a pass means a 98 ms pass.
Getting there by drafting better instead would need 3.7 tokens a pass, which
at a 92% first-draft and 40% second-draft hit rate is not reachable. **The pass
has to get cheaper.** These three problems are the three ways it can.

---

## Problem 1 — What actually costs 1.85 ms in a MIL GDN layer?

**Worth:** up to 70 ms a pass. Nothing else can be optimized until this is
understood, because the two obvious cost models are both wrong.

One GDN layer, four live token slots, measured in isolation by
`probes/mil_k_check.py`:

| K | ms |
|---|---|
| 1 | 1.367 |
| 2 | 1.616 |
| 4 | 1.841 |
| 8 | 2.466 |

So there is a fixed 1.2 ms and a marginal 0.157 ms per extra token step. The
fixed part is what matters: 36 layers × 1.2 ms = 43 ms of the 148 ms pass.

**Hypothesis A, weight bandwidth — disproved.** The layer holds about 94 MB:
in_proj int8 42 MB, out_proj int8 16 MB, two fp16 mixers 26 MB, shared expert
fp16 10 MB. 94 MB in 1.2 ms would be 78 GB/s, which looked exactly right. But
taking in_proj to per-channel int4, halving the single largest weight from
42 MB to 21 MB, moved the time from 1.897 ms to 1.880 — **1%** — while the
error went from 0.028 to 0.22. Taking the mixers and shared expert from fp16 to
int8, another 18 MB, gave 5%.

**Hypothesis B, op dispatch — disproved.** The layer is roughly 290 MIL ops at
K=4, which at 1.85 ms is about 6 us an op, and that also looked right. The
hyper-connection mixer slices its four branches apart and puts them back, 55
ops, when the same arithmetic is 26 ops on a `(1, HC, H, S)` reshape — and
`(1, HC_W, 1, S)` *is* that tensor in row-major order, so the reshape is free.
The rewrite was bit-identical and **20% slower**: 1.849 ms to 2.247.

That second result is the interesting one. Four ops over 2560 channels beat one
op over 4 channels covering the same elements. The ANE evidently cares about
which axis the work sits on, and the cost is neither per-byte nor per-op.

**The task.** Build a cost model for MIL ops on this hardware that predicts
these three measurements, then use it to make the layer faster. Concretely:

1. A microbenchmark harness that compiles single-op and small-chain MIL
   programs through `runtime/q38_ane_engine.py` and times them, sweeping shape,
   rank, axis placement, dtype and op kind.
2. A written model: what does an ANE op cost as a function of its tensors?
   Explain why 4×(1,2560,1,32) beats 1×(1,4,2560,32), and why halving a
   42 MB weight is free.
3. Apply it. A 20% cut in the fixed 1.2 ms is 8 ms a pass. A 50% cut is 22 ms.

The MIL emitter is `probes/flashnext_mil_layer.py` (`build_mil`, `gdn_core`,
`mixer`). `docs/W8A8-PROJECTIONS.md` has every spelling the compiler rejected.

**Ground rules.** `probes/mil_k_check.py` prints per-slot error against the
torch reference; per-slot error must stay at or below 0.034 and the final state
at or below 0.007. A faster layer that moves the error is not a result.

---

## Problem 2 — Overlap the ANE and the GPU

**Worth:** 28 ms a pass, which is 15.4 to about 20 tok/s, and it costs no
accuracy at all.

Today the pass is a strictly serial alternation. For each of 48 layers: the ANE
computes the layer, the GPU runs the routed MoE over that layer's output, the
host recombines, the next layer's ANE call begins. The ANE is idle for the
28 ms the GPU works and the GPU is idle for the 101 ms the ANE works.

The dependency looks total, but it is not, because a speculative pass carries K
independent token slots. Split the block into two micro-batches and the layer
stack becomes a two-stage pipeline:

```
ANE(layer i, half 1)
GPU MoE(i, h1)      ||  ANE(i, h2)
GPU MoE(i, h2)      ||  ANE(i+1, h1)
...
```

`ANE(i, h2)` needs the recurrent state that `ANE(i, h1)` produced, which it has.
`ANE(i+1, h1)` needs `MoE(i, h1)`, which finished in the previous slot. The
wavefront is valid. It should hide almost all of the GPU time.

**What makes this hard.** `AneEngine.submit` in `runtime/q38_ane_engine.py`
calls `evaluateWithQoS:options:request:error:` and blocks. Pipelining needs
asynchronous submission with a completion signal, from the private framework,
driven from Python. That is the real work. Watch for:

* The GDN recurrent state currently lives in its input IOSurface and is
  updated in place by `commit`. Two in-flight halves need two state slots per
  layer, or an explicit fence before the overwrite.
* MLX evaluation and the ANE submit must run on different threads without
  either serializing the other. MLX's stream semantics are the lever.
* Correctness is checkable: output must be byte-identical to the serial path,
  which prints `vs BF16 greedy ...: MATCH` at the end of a run.

A second, smaller piece of the same idea: the MTP drafter is 7 ms of GPU work
that must finish before the pass starts. Eager drafting — continuing the chain
past the verification boundary while the backbone verifies, and discarding on
mismatch — hides it for free.

---

## Problem 3 — The routed MoE on the ANE

**Worth:** 28 ms a pass directly, and it is the only change that would let
consecutive layers fuse into one MIL program, which is worth more again. It
also removes the GPU rail from the decode loop entirely, which is the actual
point of this port: the ANE runs at 6-8 W against the GPU's 64-84.

This was the first round's challenge and it is still open. The state of it:

* Apple's compiler assigns `GatherMM` to the GPU and rejects every fused form
  tried so far. `docs/CHALLENGE-ANE-MOE.md` lists them.
* Core AI cannot stream compressed weights: int8 gives `Failed to
  HandleANELayer`, and palettized weights are decompressed on load.
* Grouped scales are `InvalidMILProgram` at every width; only per-channel
  works, and per-channel int4 costs too much accuracy (Problem 1 measured it:
  error 0.028 to 0.22 on a dense projection).
* 512 experts × 10 selected × 48 layers, 4-bit, group 64. The bank is 69 GB
  resident on the GPU today.

Read `docs/CHALLENGE-ANE-MOE.md` and `docs/CHALLENGE-ANE-MOE-RESULT.md` before
starting. The previous attempt did not solve it; it found an unrelated 3.3x in
the GPU path, which is already landed.

An angle that has not been tried: the ANE supports multi-procedure programs
(`compile_multiproc`, one procedure per expert, already used elsewhere in this
repo). Ten selected experts is ten procedure selections per layer. The question
is whether ten small submits beat one GPU gather once the weights are baked and
resident, and whether the ~80 resident-model ceiling survives it.

---

## Which to hand out

Problem 1 is the one I would give a smarter agent. It is self-contained, it has
three measurements that any correct answer must explain, it needs no new
framework surface, and everything downstream of it is currently guesswork.

Problem 2 is the surest win in tokens per second but it is systems work, not
insight work — the difficulty is entirely in getting async ANE submission out
of a private framework through ctypes.

Problem 3 has the highest ceiling and the lowest probability.

## Ground rules for all three

* `docs/*.md` are lab notes, not truth. Several entries in them were retracted
  by this session's measurements. Re-measure anything you depend on.
* Verify against the production Swift backend in `~/.mlx128/Rindi-NativeMLX`,
  not against `mlx_lm`.
* Do not take the experts below 4 bits.
* Report negative results with numbers. Half this document is negative results
  and they are the reason the remaining problems are well posed.
