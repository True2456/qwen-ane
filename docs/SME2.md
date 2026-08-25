# SME2 Q4 projection backend

`runtime/rindi_sme_engine.*` is a shared, model-independent CPU projection
backend for dense and MoE execution plans. It currently implements:

- pointer-driven INT8 matrix multiplication;
- signed row-wise INT4 weight × per-token INT8 activation GEMV;
- FP32 and FP16 output APIs;
- scalar quantized and full-FP16-activation references;
- configurable output-row parallelism; and
- a genuinely asynchronous compatibility projection API.

The Q4 kernel consumes the dense chain's existing byte layout: two signed
two's-complement nibbles per byte, low nibble first, with one FP16 scale per
output row. Weights are unpacked into streaming SVE registers and immediately
consumed by `SDOT`; no expanded weight matrix is allocated.

## Build and validate

```sh
make test-sme2
make bench-sme2

# Fast Ling-expert shapes only
/tmp/rindi-bench-sme2 --quick --workers 4 --iterations 20

# One exact dense shape
/tmp/rindi-bench-sme2 --shape dense_down --workers 4 --iterations 20
```

The test requires SME2 hardware for the accelerated path but retains scalar
fallback behavior. `otool -tvV runtime/rindi_sme_engine.o` should show
`smstart`, `sdot`, and `smstop`.

## Efficiency gate

Raw latency is not enough because SME2, Metal, and ANE share memory bandwidth.
Run the long-window A/B before selecting SME2 in the dense execution plan:

```sh
sudo tools/sme2_power_ab.sh

# Optional longer/two-worker measurement
WORKERS=2 ITERS=4000 sudo -E tools/sme2_power_ab.sh
```

The script reports isolated projections per joule for the dense fused gate/up
and down shapes. It intentionally does not call that number model tokens per
joule. Whole-model promotion still requires a feature-gated dense run with
logit and generation regression checks.

On the initial M5 Max latency run, SME2 was strongly favorable for both small
Ling expert projections and modestly favorable for the dense down projection;
Metal remained substantially faster for the dense fused gate/up projection.
The intended dense candidate is therefore a per-shape plan (Metal gate/up,
SME2 down), not an all-SME replacement.

## Current integration boundary

The object is part of the shared native library, but no dense or MoE model is
routed through it by default. This preserves the existing dense hot path. A
feature-gated lane-1 dense experiment keeps gate/up on Metal and moves only the
down projection to SME2:

```sh
RINDI_SME2_DOWN=1 RINDI_SME2_WORKERS=4 bin/rindi serve
```

In CoreAI mode the flag also prepares the narrow Metal decode tails needed by
the hybrid path. Wide prefill lanes remain on CoreAI/ANE. `RINDI_DISABLE_METAL_TAIL`
still overrides the experiment and leaves SME2 unused.

The split introduces a Metal completion wait before SME2 and a second Metal
command buffer afterward. It must therefore pass full-model output, tok/s, and
joules/token A/B tests before becoming a default execution plan.

The initial 27B greedy decode A/B produced an identical output hash, but the
hybrid measured 4.180 tok/s versus 4.622 tok/s for the all-Metal narrow tail
(about 9.6% slower). The isolated down-projection energy result is favorable,
but the synchronization cost fails the model-level throughput promotion gate.
Keep the flag off by default. Re-run both modes with:

```sh
make bench-sme2-dense-decode
```

## Heterogeneous row split and SIMD QMV

`make bench-sme2-hetero` sweeps disjoint GPU/SME2 output-row splits and also
compares the original lane-1 Metal kernel with the SIMD-reduced QMV. After
switching its inner loop to packed 32-bit loads, the stabilized M5 Max run
reached about 196 GiB/s for dense gate/up and 99 GiB/s for dense down. A
concurrent split reached about 197 and 103 GiB/s respectively, demonstrating
additive CPU/GPU bandwidth, although the margin is workload and thermal-state
sensitive. Against installed MLX 0.32.0, the native kernels measured about
211 versus 214 GiB/s on gate/up and 152 versus 153 GiB/s on down. Lane-1
rowwise Metal projections use the SIMD QMV by default; long-K/small-N shapes
automatically use four-way split-K. `RINDI_METAL_QMV_NO_SPLITK=1` disables
that specialization, while
`RINDI_METAL_QMV_LEGACY=1` restores the scalar kernel for regression A/B.

`RINDI_SME2_DOWN_SPLIT=80` is the experimental concurrent path: the GPU
computes the first 80% of aligned output rows while SME2 computes the remainder.
It is intentionally not selected automatically because the SIMD GPU QMV won
the initial throughput sweep. The flag remains useful for power tests and for
future MoE expert scheduling where the CPU and GPU may touch separate experts.

In the corresponding full 27B decode run, the legacy Metal tail reached 4.70
tok/s, the all-GPU SIMD/split-K QMV reached 10.79 tok/s, and 80/20 and 90/10
down splits reached 7.39 and 7.71 tok/s. All three produced the same short-run
greedy hash.
The dense split loses because SwiGLU must finish before either backend can read
the down input, forcing an extra command-buffer boundary. MoE experts can be
scheduled independently and remain the stronger heterogeneous target.

## Apple streaming-mode ABI note

The SME object is compiled separately with `-march=armv9.2-a+sme2` and automatic
vectorization disabled. The kernel explicitly brackets streaming-vector code
with `SMSTART SM`/`SMSTOP SM`. It also declares v8-v15 clobbers: changing the
streaming vector register file without preserving ordinary AAPCS callee-saved
vector state corrupts caller floating-point values. Do not apply the SME target
flags to the rest of the server.
