# Routed MoE: full-bank dtype conversion in the decode loop

Experiment date: 2026-09-12, Apple M5 Max, macOS build 26A428.

## Finding

`ResidentMoe` stored BF16 scales/biases and supplied FP16 activations.
In the installed MLX source, `mlx/ops.cpp`, `gather_qmm` first promotes their
types, then casts **the entire scale/bias arrays**, and only subsequently
executes the indexed multiplication. FP16 + BF16 promotes to FP32. This is
not a different low-throughput multiplication kernel selected for FP16 input.

Each layer has these scale shapes (and identically shaped biases):

| Projection | Scale shape |
|---|---|
| gate | 512 × 640 × 40 |
| up | 512 × 640 × 40 |
| down | 512 × 2560 × 10 |

Together they contain 78,643,200 scale/bias elements. Their FP32 conversion
creates 314,572,800 bytes per layer, or **15,099,494,400 bytes per token** over
48 layers, plus reading the original BF16 arrays. Selection of ten experts
does not restrict these conversion operations. These are logical tensor bytes,
not a hardware counter measurement of DRAM traffic.

The fix materializes matching FP32 scales/biases once at model load and uses
FP32 activations and scores. Packed four-bit expert weights are unchanged.
`FLASHNEXT_MOE_DTYPE=float16` also loads matching FP16 scales by default.
`FLASHNEXT_MOE_SCALE_DTYPE=bfloat16` explicitly restores the old mismatch for
the controlled baseline. Host readback conversion precedes the single eval,
so a low-precision mode does not require a second GPU submit just to cast its
output to FP32.

Resident MoE storage increases from **68.42 GB to 76.44 GB**, including shared
experts. This is a resident-memory tradeoff, not checkpoint requantization.

## Making the measurement possible

The first, unmodified command aborted after 64 generated tokens:

```
CoreAIRuntime/NDArray+Pool.swift:77: Fatal error: Failed to allocate storage
for NDArray with byteCount: 655360, sk: ioSurface, st: float16
```

There was 35 GiB free. Copying recurrent outputs into fresh host-backed arrays
also aborted (after 74 tokens, failing on a 32768-byte allocation). Explicit
autorelease pools and `gc.collect()` every 48-layer sweep did not prevent the
failure in `probes/flashnext_pool_repro.py`. These observations do not prove
that dynamic MoE is impossible on ANE.

`FLASHNEXT_ANE_WORKER=1` runs Core AI inference in a spawned process and
restarts that process after 1536 layer calls. The decoder and GPU bank stay
resident in the parent for the entire generated stream. Every recurrent state
is explicit input/output, copied without changing dtype or values. QSA KV and
indexer state also remain in the decoder. A worker restart does not reset the
context or replay a prompt. No selected expert weights are sent to the worker.

All worker IPC, copies, startup, model loading, and restarts are included in
the measured ANE call and end-to-end latency. This workaround is costly;
it is not a sub-150-ms/token decode result or an energy-efficiency result.
The first 74 generated IDs match direct in-process inference exactly, across
two worker restarts.

The command in the challenge automatically selected experimental folded QSA
assets when present and produced `The 2011–12…` before any MoE change. Bare QSA
restores `The 2016–17 season marked a pivotal`. It is now the default;
`FLASHNEXT_QSA_FOLDED=1` explicitly opts into folded mixers. The previously
unexecuted bare wide-QSA path also needed its missing `layer_kv` dictionary
initialized.

## Measurement protocol and reproduction

Run arms sequentially: they each hold most of the machine's RAM. Check
`df -h /System/Volumes/Data` and retain at least 30 GB before model loading.
The worker's default recycle interval is 32 full tokens, safely below the
observed allocation limit.

```sh
FLASHNEXT_MOE=mlxresident FLASHNEXT_HEAD=mlx FLASHNEXT_PREFILL_K=16 \
FLASHNEXT_QSA_FOLDED=0 FLASHNEXT_ANE_WORKER=1 FLASHNEXT_STRICT_LOAD=1 \
FLASHNEXT_MOE_DTYPE=float32 \
PYTHONPATH=/Users/true/.mlx128/mlx/python \
/Users/true/.rindi/venvs/coreai/bin/python -u probes/flashnext_moe_inloop.py \
  --tokens 700 --warmup 100 --validate --output /tmp/ane-moe-fp32.json
```

For the old mixed-dtype arm, add `FLASHNEXT_MOE_DTYPE=float16` and
`FLASHNEXT_MOE_SCALE_DTYPE=bfloat16`; use a different output filename.
`--swift-fixtures /tmp/ane-moe-fixtures` saves selected expert weights and real
decoder inputs from layers 0, 3, 24 and 47 at the start and end of the measured
window. Then run:

```sh
/Users/true/.rindi/venvs/coreai/bin/python probes/run_flashnext_moe_swift.py \
  /tmp/ane-moe-fixtures
```

The verifier links the existing production Swift build under
`~/.mlx128/Rindi-NativeMLX`, including its matching precompiled Metal library.
It calls the actual `SwitchGLU` and `weightedExpertSum` implementations used
by `Sources/RindiInference/Qwen4Exp.swift`. No `mlx_lm` model implementation is
used for verification.

`flashnext_moe_inloop.py` instruments `stage_generate` itself. It excludes the
first 100 generated tokens and measures the following 600 continuously across
all 48 layers. The GPU MoE timing includes NumPy input construction, graph
construction, eval/synchronization, output readback, and the shared expert.
Routing has a separate timer. CPU correctness checks run **after** generation,
so they do not warm the GPU or perturb a following layer's timings.

The CPU oracle unpacks the selected four-bit weights in NumPy, applies their
scales and biases, and computes SwiGLU, weighted reduction and shared expert
in FP32. It checks all 48 layers at both ends of the measured window.

## Results

Long-run results and reference errors are recorded here after the runs finish.

---

## Independent verification (same day, separate session)

The finding reproduces and the fix is real.

Isolated, `gate_proj`, ten changing experts, warm:

| activations / scales | per call |
|---|---|
| fp16 / bf16 — **the shipped combination** | 0.573 ms |
| fp16 / fp16 | **0.175 ms** |
| fp32 / fp32 | **0.175 ms** |

3.3x. Note what this corrects: an earlier session measured fp32/bf16 at
0.581 ms, concluded *activations* were at fault, switched them to fp16 and left
scales at bf16 — preserving the mismatch exactly. The pair that shipped was
never measured. Whenever a dtype mismatch is suspected, measure the pair that
ships, not a neighbouring pair.

In the decode loop, `--prompt-ids 760 --max-new 12`, bare QSA:

| | before | after |
|---|---|---|
| MoE, 48 layers | 90 ms | **54 ms** (1.06 ms/layer) |
| ANE + I/O | 114 ms | 119 ms |
| per token | ~220 ms | **~185 ms** |
| output | `The 2016-17 season marked a pivotal` | unchanged, **BF16 greedy MATCH** |

**Win condition 2 is met**: routed MoE at 1.06 ms/layer measured in the real
loop, below the 1.5 ms threshold, with output unchanged. Win condition 1
(<150 ms/token) is not: decode is ~185 ms/token, ~5.4 tok/s, up from 4.3.

### Refinement: use fp16 scales, not fp32

The fix defaults `FLASHNEXT_MOE_SCALE_DTYPE` to match the activation dtype, and
the writeup materialises fp32. Measured side by side, fp32 buys nothing and
costs 8 GB:

| scale dtype | ANE | MoE | resident bank |
|---|---|---|---|
| fp32 | 119-120 ms | 51-54 ms | 76.44 GB |
| **fp16** | 117-119 ms | 51-53 ms | **68.42 GB** |

Both match BF16 greedy. Prefer `FLASHNEXT_MOE_DTYPE=float16`; the resident
bank stays where it was and the speed is identical.

### Caveat on the loop configuration

These runs use `FLASHNEXT_QSA_FOLDED=0`. With folded QSA plus the width-rung
ladder, the load exceeds the **~80 resident model** ANE ceiling and fails with
`Program load failed — no ANE resources`. Whoever reconciles the two paths
needs to budget model count, not just correctness.
