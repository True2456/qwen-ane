# Measured performance

All figures M5 Max, Qwen3.8-27B, int4 weights on the ANE, bf16 on the GPU.
Reproduce with the probes in `probes/` and the scripts in `tools/`.

## The headline

**The measured ANE kernel is 1.7× more efficient per joule and ~4.5× slower
than the measured GPU kernel.** This is sustained performance for the compiled
graphs below, not the ANE's quoted 38+ TOPS theoretical INT8/FP16 dual-lane
peak. The gap to that peak is a utilization target, especially for multi-lane
and more deeply pipelined graphs.

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

## Throughput ceiling

`probes/ane_lane_occupancy.py` — one real 27B MLP, sweeping program width:

| S | ANE ms | GPU ms | ANE TFLOP/s | ANE vs GPU |
|---|---|---|---|---|
| 32 | 2.548 | 2.912 | 6.72 | **1.14×** |
| 64 | 2.948 | 2.706 | 11.61 | 0.92× |
| 512 | 26.933 | 6.328 | 10.17 | 0.23× |
| 2048 | 125.046 | 24.654 | 8.76 | 0.20× |

The ANE flattens at ~10 TFLOP/s from S=64 up and never improves; the GPU reaches
~45. The single crossover at S=32 exists only because the GPU is latency-bound
there, not compute-bound.

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
It uses accept-all-or-longest-prefix rollback and contains no MLX/GPU path.

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

| | bake | total |
|---|---|---|
| cold | 44 s | 52.6 s |
| `--bake-cache` warm | 21 s | 28.5 s |

Quantisation is 83% of the bake (0.45 s of 0.54 s per layer) and is cacheable.
The MIL compile is 0.09 s and is **not** — `ANECCompile` re-runs from MIL every
time, and preserving its content-addressed output saves only 1.1×.

## A caution about the energy numbers

An earlier measurement here reported the ANE at **0.43× the GPU's tokens per
joule** — the opposite conclusion. It was wrong: the sampling window spanned
model load and a 33 s bake with the ANE idle ~85% of it, so it measured the
plumbing, not the silicon. Sample only the generation window
(`tools/ane_power_ab.sh` now emits `MEASURE_START`/`MEASURE_END`), and pin the
engine before comparing (`tools/tflops_per_watt.sh`).
