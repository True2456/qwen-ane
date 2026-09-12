# A measured cost model for text MIL on the M5 Max ANE

Measured 2026-09-12 on Apple M5 Max, macOS 27.0 build 26A428. This note is
specific to the private text-MIL compiler and `AneEngine.submit` path used by
this repository. It replaces the two useful-looking but false models “time is
weight bytes / bandwidth” and “time is MIL node count times a dispatch cost.”

## Reproduction

`probes/ane_mil_cost.py` compiles programs through
`runtime/q38_ane_engine.py::compile_multiproc`, allocates the same IOSurface
kind as the production layer, warms the program, and times only synchronous
`AneEngine.submit`. Compilation and host copies are outside the timed region.
JSON output includes every sample and compile failures.

```sh
PY=~/.rindi/venvs/coreai/bin/python

$PY probes/ane_mil_cost.py --suite layout --ops square,reduce_mean \
  --depth 8 --warmup 12 --repeats 101 --json /tmp/ane-layout.json
$PY probes/ane_mil_cost.py --suite axis --depth 8 \
  --warmup 12 --repeats 101 --json /tmp/ane-axis.json
$PY probes/ane_mil_cost.py --suite dtype --ops square,tanh --depth 8 \
  --warmup 12 --repeats 101 --json /tmp/ane-dtype.json
$PY probes/ane_mil_cost.py --suite conv --depth 1 --filter conv_i \
  --warmup 12 --repeats 101 --json /tmp/ane-projections.json
$PY probes/ane_mil_cost.py --suite core --filter group_rms --depth 8 \
  --warmup 12 --repeats 101 --json /tmp/ane-group-rms.json
```

The reduction cases contain one reduction followed by a broadcast add per
logical repeat. This keeps the next repeat and external output at the original,
IOSurface-safe shape. `us/repeat` is therefore not a per-MIL-node estimate.
The grouped-RMS cases use four logical normalization blocks (`depth / 2`), and
compare exactly the two layouts used by the mixer experiment.

Absolute medians drift by a few percent with thermal and system state. The
conclusions below use paired cases from the same process or alternating runs.

## The model

A useful first-order model is:

```text
T(program) = Tsubmit
           + schedule({ops})
           + sum Tpointwise(physical tiles, live intermediates)
           + sum Treduce(reduced axis, reduction length, output grid)
           + sum Tconv(I tiles, O tiles, spatial tiles, weight form)
           + sum Tretile(producer layout -> consumer layout)
           + Tinput/output surfaces
```

The terms are deliberately not summed per source-level node. The compiler
fuses and eliminates pointwise work, and schedules a connected region as a
tile pipeline. The tensor axes select different physical kernels. A reshape is
free as a row-major indexing statement but can still force a retile when the
next kernel expects the factor on a different physical axis.

The calibrated rules are:

1. **There is a program/schedule floor.** Small pointwise chains land near
   0.11–0.13 ms regardless of their source node count. A 64-deep square chain
   was 0.127 ms at C=4, 0.119 ms at C=128, 0.295 ms at C=2560, and 0.863 ms at
   C=10240. Node count matters only after it creates additional live tile
   traffic that the compiler cannot fuse away.
2. **Pointwise cost follows physical tiles, not just element count.** For
   eight squares over the same 327,680 elements, `(1,2560,4,32)` took
   0.192 ms and `(1,4,2560,32)` took 0.147 ms. Folding is not intrinsically
   slow; in this isolated pointwise case it is faster.
3. **Reduction cost is dominated by the output grid and axis kernel.** A
   reduction is approximately a stream term for input tiles plus a setup/tree
   term for every independent output reduction. Reducing the four-element C
   axis of `(1,4,2560,32)` creates 81,920 tiny reductions and took 1.690 ms for
   eight reduce+broadcast repeats. Reducing its 2560-element H axis creates
   only 128 long reductions and took 0.355 ms. Reducing C=2560 in
   `(1,2560,4,32)` also took 0.350 ms. Same elements and arithmetic, 4.8x
   different time.
4. **Rank alone is mostly metadata.** With the same extents, rank-2 through
   rank-5 pointwise chains were all about 0.11 ms; changing which axis owns an
   extent is what changes the kernel.
5. **Convolution has compute/tile, activation, and constant-fetch terms.** The
   stored weight byte count affects only the last term. A
   `constexpr_blockwise_shift_scale` int4 or int8 constant feeds the same
   logical fp16 convolution tensor, with the same output tiles and activation
   traffic. The compressed backing store is not a proxy for total conv cost.
6. **External dtype and internal dtype are different questions.** fp32 and
   int8 function signatures failed to compile on this build. An fp16 signature
   with an internal int8 quantize/dequantize pair compiled, but at decode width
   it did not beat fp16: square was 0.121 vs 0.118 ms and tanh 0.131 vs
   0.106 ms. This does not contradict the W8A8 wins at S>=256 in
   `W8A8-PROJECTIONS.md`; those wins come from reducing traffic between large
   conv tiles in a fused chain, not from changing the program I/O type.

This is a predictive topology model, not an attempt to infer Apple's private
tile dimensions. For a proposed rewrite, count physical output reductions,
producer/consumer axis changes, live tiled intermediates, and external surface
bytes before counting MIL statements.

## The three previously unexplained measurements

### Why halving the 42 MB in-projection buys about 1%

The exact projection shapes, re-run as single conv programs, were:

| projection | fp16 | int8 | int4 | int4 vs int8 |
|---|---:|---:|---:|---:|
| 2560 -> 16480 (`in_proj`) | 0.759 ms | 0.385 ms | 0.378 ms | 1.8% |
| 6144 -> 2560 (`out_proj`) | 0.308 ms | 0.215 ms | 0.208 ms | 3.4% |

Int4 halves the compressed int8 blob, but does not halve the convolution's
I/O tiles, MAC schedule, activation traffic, or logical fp16 dequantized
weight tiles. It can therefore remove only a residual constant-fetch term.
The projection probe predicts a low-single-digit ceiling, and the full-layer
measurement—42 MB to 21 MB, 1.897 to 1.880 ms, about 1%—is inside it. The
full-layer int4 result also failed numerically: mixed error rose from the
0.028–0.034 band to 0.22–0.29.

### Why 55 mixer ops beat 26

The source node reduction combined three distinct effects:

- Pointwise work moved from four C=2560 branches to one C=4, H=2560 tensor.
  The isolated square result says this part should help, not hurt.
- Group normalization crossed the C/H layout. The exact four-block grouped-RMS
  microbenchmark took 0.225 ms unrolled and 0.247 ms folded: folding was 9.8%
  slower even before the branch sum.
- The folded weighted branch sum became a reduction over C=4 at every one of
  2560 x 32 spatial positions. The axis sweep measures this topology at
  1.690 ms, versus 0.355 ms when 2560 values are reduced at 4 x 32 positions.
  The original graph uses three wide pointwise adds instead of that grid of
  tiny reductions.

Thus the 26-node graph asks for worse reduction and layout schedules. A 20%
full-mixer loss (1.849 to 2.247 ms) is the expected direction despite fewer
MIL statements. “Four 2560-channel ops beat one four-channel op” is true for
that connected mixer, not a universal pointwise rule.

### Why the axis matters so much

For a reduction of shape `d0 x d1 x d2 x d3` over axis `a`, the number of
independent results is `product(dims except a)` and the reduction length is
`dims[a]`. `(1,4,2560,32)` over C asks the ANE to start/combine 81,920
four-value reductions. The same tensor over H asks for 128 2560-value
reductions. The latter maps to the ANE's wide reduction pipeline; the former
is dominated by per-output work and poor lane occupancy. Equal element count
does not imply equal scheduled work.

## Applied change

The model points to external surface traffic as a safe target because it can
be removed without changing arithmetic or internal axes.

`fseq` has `S+3=35` columns, but a K-slot pass can commit only columns
`0..K+2`. The old graph exported all 35 columns in a QKV x 64 IOSurface. For
K=4, the new graph slices the seven observable columns and exports a QKV x 32
surface, reducing this output from 1,310,720 to 655,360 bytes per layer. The
runtime now also leaves immutable `c_param` and `d_hcn` in their input
IOSurfaces and updates the already-resident conv-cache column at commit rather
than re-locking and restaging all three inputs on every call.

The legacy graph can be selected with `MIL_GDN_FULL_CACHE_OUTPUT=1`.
`probes/mil_k_check.py` can reproduce the former host staging with
`MIL_K_RESTAGE_INPUTS=1`.

Three alternating 301-submit K=4 runs gave these medians of run medians:

| path | ms/layer | change |
|---|---:|---:|
| old 64-column output + repeated staging | 1.835 | baseline |
| new 32-column output + resident inputs | 1.783 | -0.052 ms, **2.8%** |

The cache-output-only alternating A/B was 1.801 to 1.771 ms wall time and
1.715 to 1.694 ms submit-only, a 1.7% and 1.2% saving respectively. The total
measured saving is about 1.9 ms over 36 GDN layers per speculative pass—not
the hoped-for 8 ms. It is kept because it is exact, small, and repeatable.

Correctness after the change:

```text
K=4 mixed: t0=0.0284 t1=0.0338 t2=0.0319 t3=0.0289
final state: 0.00613
conv cache: 0.01217
```

All four slots remain <=0.034 and final state remains <=0.007. K=1, K=2, and
K=8 were also checked; their worst mixed errors were 0.0269, 0.0285, and
0.0321, with final-state errors 0.00558, 0.00628, and 0.00505.

## Negative results from applying the model

All timings below are K=4 layer medians unless stated otherwise. A change that
crossed either numerical threshold is a failure even if faster.

| rewrite | result |
|---|---|
| Replace multiply+reduce recurrence arms with batched `matmul` | 1.872 vs 1.844 ms; mixed errors 0.107–0.132, fail |
| Replace only the outer product with `matmul` | 1.831 ms, but t1=0.0344, fail |
| Replace only the state/query reduction with `matmul` | 1.858 ms; mixed errors 0.107–0.133, fail |
| Flatten state reductions from `(48,128,128)` to `(6144,1,128)` | 1.942 vs 1.844 ms (+5.3%), numerically acceptable |
| Keep recurrent state transposed `(128,48,128)` | 2.004 vs 1.844 ms (+8.7%), numerically acceptable |
| Fold only mixer group normalization | 1.827 vs 1.836 ms, within run noise; exact microbenchmark was 9.8% slower |
| Direct `x*sigmoid(x)` SiLU in the mixer | 1.817 vs 1.844 ms, but t1=0.0358, fail |
| Direct SiLU in the GDN | mixed errors 0.164–0.212 and state 0.0288, fail |
| Fuse mixer down and injection projections | long-run submit 1.701–1.714 vs baseline 1.695–1.703 ms; no win |
| Run only K live columns internally and pad outputs | 1.774 vs 1.765 ms in the controlled follow-up; no win |
| Pack four state outputs into one surface | 1.775 vs 1.765 ms; no win |
| Pack non-state outputs at live width | 1.880 vs 1.844 ms; slower |
| Split exact int8 conv outputs into 2/4/8 projections | 2 and 8 parts failed compile; 4 parts was 0.389 vs 0.385 ms for `in_proj`, and 0.212 vs 0.215 ms for `out_proj` |

Several short 25-repeat trials falsely suggested 0.04–0.07 ms wins for
projection fusion, live-width execution, or packing. The 301-repeat alternating
runs erased or reversed them. Do not promote a layer rewrite from a single
short run on this path.

## Independent verification

Re-run 2026-09-12 in a separate session, same machine.

**Correctness is exact.** The per-slot errors are unchanged from the graph the
change replaced, digit for digit, at every K:

| K | worst slot | final state |
|---|---|---|
| 1 | 0.0269 | 0.00558 |
| 2 | 0.0285 | 0.00628 |
| 4 | 0.0338 | 0.00613 |
| 8 | 0.0321 | 0.00505 |

**The layer gain reproduces.** Alternating 201-repeat runs, new / legacy / new:
1.771, 1.828, 1.768 ms. That is 3.2%, slightly better than the 2.8% claimed.
Across the four widths the layer went 1.367 to 1.275 ms at K=1, 1.616 to 1.446
at K=2, 1.857 to 1.775 at K=4 and 2.466 to 2.299 at K=8.

**The axis result reproduces and is the load-bearing one.** Re-running the axis
suite: reducing the 4-element axis of `(1,4,2560,32)` takes 1.711 ms where
reducing its 2560-element axis takes 0.352 ms and reducing `(1,2560,4,32)` over
C takes 0.360 ms. Same elements, same arithmetic, 4.8x.

**End to end**, 64 tokens after a 32-token prompt, speculative K=4: 15.4 to
**15.7 tok/s**, generated ids unchanged, and plain decode still prints
`vs BF16 greedy [220, 17, 15, 16]: MATCH`.

### One more negative result, from applying the model

The model says a reduction over a long axis with few outputs is the cheap
regime, and the mixer's grouped RMS is already in it: 4 branches x 32 positions
is 128 outputs of length 2560. That suggested its five ops per branch — square,
reduce_mean, add eps, pow(-0.5), multiply — might be collapsible to three by
asking the compiler for the norm directly with `reduce_l2_norm`, `pow(-1)` and
a multiply, folding the sqrt(H) that turns an L2 normalize into an RMS
normalize into `hc_norm` on the host.

It compiles, and it is numerically equivalent (0.0282 / 0.0342 / 0.0319 /
0.0288 against 0.0284 / 0.0338 / 0.0319 / 0.0289). It is **not faster**: 1.782
against 1.770 ms. The compiler already fuses the square into the reduction, so
there was no intermediate to remove. This is consistent with rule 1 — source
node count only matters when it creates tile traffic the compiler cannot fuse
away — and it is a second, independent confirmation of that rule.

### What the model implies for the rest of the layer

Reading the projection table together with the K sweep accounts for the whole
1.77 ms at K=4: in_proj 0.385, out_proj 0.215, four recurrence steps at the
measured marginal 0.146 ms each, and about 0.47 ms of mixers, shared expert,
depthwise conv and recombine.

The conv numbers say fp16 to int8 scales cleanly at about 110 GB/s of weight
fetch (84 MB at 0.759 ms, 42 MB at 0.385) and then stops, because for that
shape the int8 fetch time and the conv's own floor coincide at 0.38 ms. So the
two big projections are already at their floor, and the remaining fp16 weights
— 26 MB of mixers and 10 MB of shared expert per layer — are the only fetch
left to cut. That is worth about 4 ms a pass at int8, against a measured error
cost that already fails the gate for the mixers (0.028 to 0.031-0.038).

The ANE side of the pass is therefore close to done. The next 19% is not on the
ANE at all: it is the 28 ms the GPU spends on the routed MoE while the ANE
sits idle, which is Problem 2 of `CHALLENGE-ANE-ROUND-2.md`.
