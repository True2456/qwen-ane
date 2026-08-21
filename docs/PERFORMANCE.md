# Measured performance

All figures M5 Max, Qwen3.8-27B, int4 weights on the ANE, bf16 on the GPU.
Reproduce with the probes in `probes/` and the scripts in `tools/`.

## The headline

**The measured ANE kernel is 1.7× more efficient per joule and ~4.5× slower
than the measured GPU kernel** on whole-model decode. That slowness is a
scheduling result, not an arithmetic ceiling: on the model's real projection
shapes at int4 the ANE sustains **18.7–19.3 TFLOP/s**, which is 89–92% of this
part's fp16-equivalent peak. See "The arithmetic ceiling" below — an earlier
version of this document reported ~10 TFLOP/s as the hardware limit and that
was wrong.

`sudo tools/tflops_per_watt.sh` — same MLP pinned on each engine, idle-corrected,
CPU cost of driving each included:

| engine | S | TFLOP/s | ANE W | GPU W | CPU W | net W | TFLOP/W |
|---|---|---|---|---|---|---|---|
| ANE int4 | 128 | 9.00 | 5.92 | 0.00 | 1.93 | 7.26 | **1.24** |
| ANE int4 | 512 | 10.19 | 6.42 | 0.00 | 2.49 | 8.32 | 1.22 |
| GPU bf16 | 512 | 47.93 | 0.00 | 64.75 | 0.82 | 64.98 | 0.74 |
| GPU int4 | 512 | 47.84 | 0.00 | 83.74 | 0.94 | 84.09 | 0.57 |

The GPU buys its 4.7× throughput with **10× the power** (64–84 W vs 6–8 W).
Note GPU int4 costs *more* power than bf16 for the same throughput — dequant
overhead.

## The arithmetic ceiling

M5 Max's ANE is a 16-core, **42 TOPS INT8** engine (the widely quoted 38 TOPS
is M4's). TOPS counts two operations per MAC, so 42 TOPS is 21e12 INT8 MACs/s,
and a dual-lane INT8/FP16 array does half that in fp16 — call it **~21 TFLOP/s
fp16-equivalent**. Note that M5's headline "4x AI compute" and its ~70 TFLOPS
FP16 figure describe the *GPU's* per-core Neural Accelerators, which are a
different unit and not what this repository drives.

`probes/ane_peak_tflops.py` and `probes/ane_peak_real.py`, one baked conv per
measurement, S=512:

| projection | fp16 | int8 | int4 | int4 % of peak |
|---|---:|---:|---:|---:|
| mlp gate+up `[34816,5120]` | 7.0 | 13.9 | **19.1** | 91% |
| gdn in_proj `[16480,5120]` | 7.0 | 13.9 | **18.7** | 89% |
| attn qkv `[14336,5120]` | 7.0 | 13.8 | **18.9** | 90% |
| lm_head chunk `[62080,5120]` | 7.1 | 14.0 | **19.3** | 92% |
| mlp down `[5120,17408]` | 1.1 | 2.1 | 5.0 | 24% |

The best shape measured anywhere is 20.3 TFLOP/s (int4, `[32768,2048]`,
S=512), and nothing exceeded it — independent evidence that the real ceiling
sits near the 21 TFLOP/s the spec implies, and that we are already close to it
on four of the five shapes.

Two claims elsewhere in this repo were corrected by these measurements:

* **"~10 TFLOP/s sustained, flat from S=64 up" was a property of one graph**,
  the chained 27B MLP, whose cost is dominated by `down_proj`. It is not a
  hardware ceiling.
* **"Decode cost is invariant to weight precision" is true only at S=32**,
  where dispatch dominates. At prefill widths, int4 is **2.7× fp16** on the
  same shape. fp16 large-M convs are weight-bandwidth-bound: the array streams
  weights once per S-tile, so cutting weight bytes 4× converts a
  bandwidth-bound conv into a compute-bound one.

### `down_proj` is pathological, and splitting it fixes it

`[5120,17408]` — few output rows, very deep input — runs at 24% of peak and,
uniquely, gets *worse* with width (21.3 µs/token at S=32 against 35.7 at
S=512). Splitting it into N convs over disjoint input channels and summing the
partials recovers nearly all of it:

| parts | int4 ms @S=512 | TFLOP/s | % peak | max rel vs dequant ref |
|---:|---:|---:|---:|---:|
| 1 | 18.2 | 5.0 | 24% | 7.09e-4 |
| 2 | 9.2 | 9.9 | 47% | 9.31e-4 |
| **4** | **4.84** | **18.8** | **90%** | 9.27e-4 |
| 8 | 4.59 | 19.9 | 95% | 1.37e-3 |

**3.81× on that projection at S=512**, verified against a dequantized-weight
reference, with error no worse than the unsplit conv at 4 parts. At 8 parts
fp16 accumulation across partial sums starts to show.

The gain is width-dependent and **exactly 1.00× at S=32**, so this is a
prefill optimization, not a decode one:

| width | parts=1 | parts=4 | gain |
|---:|---:|---:|---:|
| 32 | 0.68 ms | 0.68 ms | 1.00× |
| 64 | 1.27 | 0.67 | 1.89× |
| 128 | 4.42 | 1.29 | 3.43× |
| 512 | 18.47 | 4.85 | 3.81× |

The packed four-way form is now deployed. It slices the existing row-quantized
payload (so int4 values and scales are unchanged), writes the four data tensors
and their scale tensors as milinternal records in one `down.bin`, and references
them at absolute offsets. This stays below the compiler's 16-weight-file rule:
the chained GDN tail uses 12 files and the attention tail 9. The old form is
available as `--down-proj-parts 1`; packed 4 is the pure runtime default.

On complete real tails, packed four-way changes width-64 GDN from 3.289 to
**2.683 ms** (-18.4%) and attention from 3.261 to **2.714 ms** (-16.8%). At
width 32 both tails are about 1% slower in isolation, but the full server is
flat: identical 16-token output and 3.499 tok/s for both forms; profiled decode
was 284.82 versus 284.39 ms/token. Thus it removes the width-64 pathology
without a measurable production regression.
Depth-2 MTP was also requalified with the packed tail: 126 programs compiled,
the same 16-token output was produced, and acceptance remained 1.875
tokens/cycle. This compatibility result does not change MTP's opt-in status.

### Prefill runs 32 wide, and pays about 2.2× for it

Every program in `tools/pure_ane.py` is compiled at `width=32` with at most
three real lanes, so a 261-token prompt is ~87 sequential passes of the whole
64-layer stack. Per-token cost at int4, µs/token:

| projection | S=32 | S=64 | S=128 | S=512 | S=512 vs S=32 |
|---|---:|---:|---:|---:|---:|
| mlp gate+up | 40.3 | 20.2 | 18.1 | 18.6 | **2.2×** |
| gdn in_proj | 20.5 | 10.2 | 9.0 | 9.0 | **2.3×** |
| attn qkv | 17.8 | 8.8 | 7.7 | 7.8 | **2.3×** |
| mlp down (unsplit) | 21.3 | 19.7 | 34.3 | 35.7 | 0.6× |
| mlp down (4 parts) | 21.3 | 10.5 | 10.1 | 9.5 | **2.2×** |

Most of the win is already at S=64–128; the curve is flat after that.

**This bounds the projections only.** The original 261-token measurement was
35.9 s ≈ 137 ms/token, while the projection work above totals ~5.2 ms/token at
S=32 — so projections were a small fraction of prefill, and most time was in
per-position GDN recurrence, attention, and per-pass dispatch overhead.

The production runtime now handles each complete 16-token prompt block with a
single shared GDN recurrence program. The graph performs the exact ordered
Qwen recurrence, Q/K RMS normalization, polynomial softplus, decay and sigmoid
gating on the ANE, and chains arbitrary incoming compact state. A continued
real layer-0 block measures 3.499 ms versus 12.756 ms stepwise (**3.65×**),
with identical fused and production-stepwise output/state. Sixty-four tokens
measure 54.310 versus 48.693 ms and are intentionally not selected.

End to end, a 148-token prompt (nine fused blocks plus a four-token remainder)
measured **41.31 ms/token** over two warm runs. Decode remained **274–283
ms/token (3.53–3.65 tok/s)**. Exact prefix-cache reuse also passed with 23
tokens reused, zero evaluated, and 6.8 ms to cached logits. These figures are
from the int4, context-4096, MTP-off server with 126 resident programs; partial
blocks and decode continue to use the one-step recurrence. MTP draft-2 also
loads at exactly 127 resident programs and passed semantic and speculative
generation checks.

### The INT8 lane is not reached

Everything above is fp16 arithmetic; int4/int8 weights are dequantized by
`constexpr_blockwise_shift_scale` before the conv, which is why they buy
bandwidth rather than MACs. Reaching 42 TOPS needs int8 *activations* into an
int8 MAC path. int8 and uint8 feature-map inputs do compile, but only through
a `cast` to fp16 ahead of the conv, so the arithmetic stays fp16; int4
activations are rejected outright. No MIL spelling was found that engages an
int8 multiply lane. The remaining ~2× to the INT8 figure is therefore
unproven and unreached, not merely unoptimized.

## Whole model

| config | tok/s |
|---|---|
| GPU baseline | **8.8** |
| 16 of 64 MLPs on ANE | 7.2 |
| all 64 MLPs on ANE | 4.3 |
| 64 MLPs + lm_head | 4.1 |
| + MTP speculation, draft 2 | **6.7** |
| `--ane-chain --ane-lm-head` (current best coverage) | 3.5–3.8 |

### Framework-free pure backend and MTP

These measurements use the standalone `tools/pure_ane.py` int4 runtime. The
timer includes a 13-token templated prompt and 32 generated tokens, so they are
end-to-end rates rather than decode-only rates.

| pure configuration | programs | blobs | tok/s |
|---|---:|---:|---:|
| original one-token path | 122 | 12.86 GB | 2.467 |
| three-lane prompt ingestion, no MTP | 122 | 12.86 GB | 2.770 |
| **pure MTP, draft 2** | **124** | **13.16 GB** | **3.185** |

The MTP run averaged 2.385 accepted tokens per speculative cycle. Its complete
32-token output was byte-for-byte identical to the non-MTP target run. The gain
is 15.0% against the current batched baseline and 29.1% against the original
path. This is lower than the older hybrid 6.7 tok/s result because the pure
runtime snapshots roughly 75 MB of GDN state per cycle and still advances the
small recurrent and causal-attention cores sequentially; the learned
weight-heavy projections and MLPs are the operations batched across real lanes.

## Decode is flat to 32 tokens

The hardware refuses widths below 32, so a 1-token decode step computes a full
32-token batch and discards 31/32. Consequences:

* Decode cost is **invariant to weight precision** — int8 1.989 ms/layer vs int4
  1.941. Quantising further cannot speed up decode.
* It is **invariant to dispatch count** — fusing gate+up from 3 convs to 2 moved
  nothing (1.929 → 1.939 ms).
* Speculative decoding is unusually valuable here: verifying k+1 draft tokens
  in the weight-heavy blocks costs about the same as verifying one.
  `tools/mtp_specdec.py` does longest-prefix
  acceptance, which took acceptance from 1.91 to **3.20 tokens/step**. Draft
  depth 2 wins overall, because drafting is sequential — each draft token costs
  a full MTP layer plus a full `lm_head`.

The framework-free equivalent is `tools/ane pure-infer --bits 4 --mtp-draft 2`.
For statistically useful measurements, run `tools/ane pure-serve` once and
then `tools/ane pure-bench`. The persistent benchmark endpoint performs an
optional warmup followed by multiple clean-state runs without including model
compilation in each sample. It reports end-to-end tokens/s, decode-only
tokens/s, time to first token, queue time, prompt/completion counts, and every
individual run rather than only an average.
It uses accept-all-or-longest-prefix rollback and contains no MLX/GPU path.

### Production-loop profile and projection chaining

`pure-serve --profile-decode` now measures the real loop. Before chaining,
mean decode across three warm runs was 297.41 ms/token: GDN tails 43.2%,
projection heads 16.5%, attention tails 14.2%, recurrence 12.1%, GDN conv 7.1%,
and Python scheduler remainder only 0.2–0.3%. The previously derived 44%
“unattributed host” bucket was not real; isolated conv costs cannot be
subtracted from fused tail programs that contain additional arithmetic.

The target now folds each next-layer projection into the preceding tail's
already-computed RMSNorm output. This removes 63 dispatches and two projection
banks, reducing the long-context int4 server from 127 to 125 programs and
12.86 to 12.82 GB. Identical 16-token outputs measured 3.352 → **3.503 tok/s**
(+4.48%) and 297.41 → **284.39 ms/token** decode. Five unprofiled runs averaged
3.499 tok/s, confirming profiling overhead is negligible.

Linear depth-2 MTP was requalified after chaining: all three outputs matched
the target exactly, but this short raw-prompt workload accepted only 1.875
tokens/cycle and ran at 2.907 tok/s versus the target's 3.499. The faster target
has moved the break-even point; `--mtp-draft 2` remains correct and opt-in, but
tree-shaped drafting is required to turn the free width-64 lanes into a gain.

On the live int4/context-4096 server, an identical 17-token chat changed from a
3.066 s cold request and 2.746 s TTFT to a 0.341 s cached request and 8.4 ms
TTFT. The cache reported all 17 prompt tokens reused and zero evaluated. A
38-token continuation reused the original 17-token boundary, evaluated the 21
new tokens, and produced the expected answer. `/v1/benchmarks` disables prefix
reuse by default so warmups do not silently turn measured runs into cache-hit
tests; opt in explicitly when measuring interactive continuation latency.

## Long-context scaling

`--context` now accepts the checkpoint maximum of 262,144. This is a capacity
result, not a claim that 256K decode has short-chat latency. Attention remains
O(context). At 256K, each full-attention layer scans 1,024 256-token KV blocks;
the int4 target groups those into 32 scan submissions plus 31 ANE online-softmax
merges. Across 16 full-attention layers, dispatch and memory traffic are
substantial even though all attention arithmetic remains on ANE.

Correctness measurements on M5 Max:

| test | result |
|---|---|
| 512 capacity, first streamed position 257 | relative error 2.11e-3 |
| 262,144 capacity, position 257 | relative error 1.69e-3 |
| 262,144 capacity, position 8,193 (32-block scan + cross-group merge) | relative error 2.19e-3 |
| full 64-layer int4, 261-token raw prompt | pass; generated token 248046 |

The full boundary run baked 127 programs / 12.86 GB, ingested 261 prompt
tokens, and generated one token in 35.934 seconds end-to-end. That run is a
state-transition qualification, not a decode-throughput benchmark. KV capacity
is 64 KiB per configured token for the 16 target attention layers: 16 GiB at
256K, plus 1 GiB if MTP is enabled.

The GPU is flat to T=64 for the same reason (weight-bandwidth-bound), so this
lever is not ANE-specific.

## Component costs (decode, per call)

| block | ms |
|---|---|
| fused layer tail | 2.14 (submit), 2.93 wall |
| GDN input projection | 0.82–0.87 |
| attention q/k/v | 0.74–0.89 |
| `lm_head`, 4 chunks | 3.33 (vs 4.86 on GPU) |
| GDN recurrence, old state round trip | 4.39 |
| GDN gates + recurrence, resident IOSurface state | **0.344** (vs ~0.66 GPU recurrence) |

The resident figure includes the 1.57 MB state copy into the compiler-accepted
width-160 input, parameter writes, ANE dispatch, and y read. The state copy by
itself is 0.036 ms. The ANE row includes polynomial softplus/decay and beta; it
excludes upstream q/k/v/a/b projection time. The cited GPU number is the
recurrence reference and does not make the ANE comparison look artificially
better by adding projection time.

### End-to-end inference caveat

`--ane-gdn-step` is correct but slower while surrounded by GPU operators:

| test | GPU | ANE GDN hybrid |
|---|---:|---:|
| 16-token bench | 7.4 tok/s | 5.0 tok/s |
| 64-token bench | 8.5 tok/s | 5.6 tok/s |

Split timing over 816 layer-calls measured 3.057 ms evaluating/marshalling the
GPU-produced q/k/v/a/b tensors and 0.499 ms in the resident ANE call. The
standalone 0.34–0.35 ms result is therefore real, but the GPU→host→ANE boundary
erases it. The next performance step is direct IOSurface chaining from the ANE
GDN projection and temporal convolution—not further recurrence tuning.

Host overhead is negligible: `cast 0.001 / write 0.017 / submit 1.882 /
read 0.018 ms` — **98% of a dispatch is the ANE itself**. There is no plumbing
win available; gains must come from more work per dispatch or more of the model
on the ANE.

## Startup

The pure server now reopens the private framework's content-addressed compiled
artifacts across processes. A measured int4/context-4096 restart hit **127/127**
artifacts: compiler time **0.00 s**, compiler input materialization **0.00 s**,
ANE load 0.46 s, descriptor construction/hash 5.27 s, BF16→int4 quantization
53.96 s, total 60.97 s. `probes/ane_compile_cache.py` independently verifies
that a second process executes a cached program exactly without calling the
compiler.

The remaining cold-start cost is quantization, not compilation. The persistent
server's `--bake-cache` stores versioned Zstandard-compressed prequantized blobs
and is enabled by default; `--no-bake-cache` disables it. The complete cache is
9.18 GB on disk (8.5 GiB via `du`) for 12.86 GB of unpacked blobs.

A measured warm restart hit **504/504** quantized tensors and **127/127**
compiled artifacts: quantization 0.00 s, read/decompress 7.54 s, descriptor
construction/hash 5.20 s, ANE load 0.45 s, compiler/materialization 0.00 s,
and **14.50 s total**. New compressed writes stop when they would leave under
8 GB free (20 GB on the uncompressed Python fallback). Inspect exact phase
totals at `GET /v1/metrics` → `startup`.

The older hybrid `tools/ane_serve.py` measurements were cold 52.6 s and warm
28.5 s. They established that prebaking helps, but its claim that the compiled
artifact could not be reopened no longer applies to the pure loader: direct
`compiledModelExists` → `loadWithQoS` reuse is now tested and implemented.

## A caution about the energy numbers

An earlier measurement here reported the ANE at **0.43× the GPU's tokens per
joule** — the opposite conclusion. It was wrong: the sampling window spanned
model load and a 33 s bake with the ANE idle ~85% of it, so it measured the
plumbing, not the silicon. Sample only the generation window
(`tools/ane_power_ab.sh` now emits `MEASURE_START`/`MEASURE_END`), and pin the
engine before comparing (`tools/tflops_per_watt.sh`).


## Why ANE prefill lags the GPU far more than ANE decode does

Measured on the same machine, same 127-token prompt, Qwen3.8-27B:

| | ANE (int4) | MLX (bf16, GPU) | ANE is |
|---|---:|---:|---:|
| decode | ~3.5 tok/s | 10.2 | **2.9x slower** |
| prefill | 23.7 | 304 | **12.8x slower** |

Decode is nearly competitive and prefill is 4.4x worse in relative terms. The
two modes are limited by different things.

**Decode is weight-bandwidth-bound.** Every token reads the whole model. The
ANE reads 12.8 GB of int4; MLX reads 54 GB of bf16 (its 54.19 GB peak memory
confirms it). The ANE therefore moves 4x fewer bytes, which nearly cancels its
lower bandwidth: 45 GB/s effective against MLX's 551 GB/s, yet only 2.9x behind
in tokens. **int4 is what rescues decode.**

**Prefill is compute-bound**, and the ANE only reaches its arithmetic rate at
width. An int4 conv measures 19.1 TFLOP/s at S=512 but 8.9 at S=32, and prefill
runs at 16 lanes, so roughly half the achievable rate while the GPU runs near
its own ~45 TFLOP/s peak.

### Why widening the programs does not fix it

Compiling at width 64 and running 32 lanes measured 23.69 tok/s against 23.68
at width 32 with 16 lanes -- exactly neutral. The per-block profile explains it:

| block | ms | % of prefill | scales with width? |
|---|---:|---:|---|
| gdn tail | 1356 | 24.4% | yes -- flat in width, so per-token cost halves |
| gdn recurrence, unrolled | 1211 | 21.8% | **no** -- doubling U doubles the graph's work |
| attention tail | 444 | 8.0% | yes |
| attention prepare, per position | 355 | 6.4% | would batch, but it is only 6% |
| gdn conv | 242 | 4.4% | yes |

The projection half halves per token and the recurrence half does not, so the
two cancel. That is a real result, not a null one.

An earlier hypothesis in this repo -- that Qwen's per-position attention loop
held a 2-3x -- is **wrong**: it is 6.4% of prefill, and batching it would move
22.83 to 24.2 tok/s.

The recurrence is the part that scales with tokens no matter how wide the
programs get. Unrolling removes its dispatch overhead but not its arithmetic.
Beating it needs the chunked/parallel formulation
(`flash-linear-attention`'s `chunk_kda` shape), which is an algorithm change.

Caveat: about 35% of prefill is unattributed here. `AneAttentionCore` recorded
zero calls because at context 512 the runtime uses
`AneLongContextAttentionCore`, a different class, so the long-context core plus
host glue are outside this table.


## Speculative decoding (MTP): what it is actually worth

`mtp_lanes` was hardcoded to 3, capping draft depth at 2. That cap was not
justified by cost: the verify pass batches up to `active_lanes` positions and
measures 273.4 ms for one, 370.5 for three, 620.3 for sixteen — so the first
candidate costs 273 ms and each **marginal** one about 23. It is now
`Q38_ANE_MTP_LANES`, default 5, allowing depth 3. `AneLinearProjectionBank`
also defaults to three active lanes and the MTP bank inherited that, which is
why deeper drafts failed with `expected (5120, 1..3) lanes`.

**Correctness first: MTP reproduces greedy decode exactly** — five prompts, 32
tokens each, token-for-token identical to `mtp_draft=0`.

Measured over five varied prompts:

| draft depth | tok/s | accepted/cycle |
|---:|---:|---:|
| 0 | 3.45 | — |
| 2 | 3.78 | 2.19 |
| **3** | **3.98** | 2.69 |
| 4 | 3.68 | 2.84 |

**MTP is worth 1.15x**, and lifting the cap from depth 2 to 3 is 1.05x of that.
Acceptance is 67% per token, matching the 68.8% recorded elsewhere in this
document.

### Two traps in measuring it

**Do not benchmark speculation on a predictable prompt.** Depth 4 measured
6.42 tok/s against 3.52 on `"The capital of France is"`, which continues
`"Paris. The capital of France is Paris. The capital of France is..."`. A
drafter predicts that almost perfectly, so acceptance ran at 4.4 of 5 and the
apparent speedup was 1.82x — against 1.15x on varied prompts. Any speculative
benchmark needs prompts whose continuations are not near-deterministic.

**MTP does not improve with generation length.** 2.96 tok/s at both 32 and 128
generated tokens, acceptance 2.214 and 2.268. Prefill is already amortized by
32 tokens, so there is nothing further to collect from longer runs.

`generate()` returns `(tokens, elapsed)`, so `len(result)` is 2 regardless —
easy to mistake for a two-token generation.
