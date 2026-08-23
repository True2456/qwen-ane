# Weight-format study: EXL3-class formats vs the engine's ANE format

Probed 23 Aug 2026, M5 Max. Two experiments: (1) a compile-acceptance gate run
through `ane_c_bridge.m` — the engine's own C++ path, on the current OS; (2) a
quantization-quality study on real Qwen3.6-35B-A3B bf16 expert weights and
Qwen3.8-27B dense weights. Probe files: `probes/test_ane_format_gate.cpp`
(`make test-ane-format-gate`), `probes/ane_exl3_format_study.py`.

## 1. Compile gate: every EXL3 mechanism is rejected

EXL3 (exllamav3) is a QTIP variant: weights are trellis-coded into packed
indices, and dequantization at inference is lookup-table based. Mapping that
onto MIL needs exactly three capabilities. All three fail:

| gate case | op class | result |
|---|---|---|
| `perchannel_int4_positive_control` | engine's exact int4 spelling | **PASS** |
| `blockwise_gs64_scales` | scale tensor `[O, 2, 1, 1]` (gs=64 granularity) | **FAIL** (`InvalidMILProgram`) |
| `lut_palettized_a/b` | `constexpr_lut_to_dense`, two arities | **FAIL** ×2 |
| `gather_codebook_lookup` | runtime-indexed codebook read | **FAIL** |

Consistent with the earlier Python-path scans (`ANE_MOE_LING.md`,
`ANE-MOE-HANDOFF.md` §10.1): int4/int8 with **per-output-channel scales only**;
no blockwise, no zero-points, no LUTs, no gather. There are also no exposed
integer/bitwise ALU ops for trellis decode. A trellis format cannot be spelled
in MIL for this compiler. Decompressing EXL3 → ANE-native offline would work
but is strictly dominated by quantizing from bf16 directly (requantizing a
lossy format compounds error and forfeits EXL3's only benefit, footprint).

On Metal, exllamav3 remains CUDA-only; a community macOS port effort exists
(exllamav3 issue #226 / PonyExl3, Jun 2026) but nothing shippable. So EXL3
checkpoints are not a practical serving path on Apple Silicon in general today.

## 2. Quality study: what per-channel-only costs, and what recovers it

18 real expert matrices `[512, 2048]` (layers 5/20/35 × gate/up/down ×
experts 0/63, bf16 source) + 2 dense 27B matrices `[17408, 5120]`-class.
Relative Frobenius error against bf16 (activation-weighted iid-N(0,1) numbers
were near-identical; iid activations understate outlier effects):

| scheme | bpw | MoE gate | MoE up | MoE down | dense 27B |
|---|---|---|---|---|---|
| **ANE int4 RTN (engine today)** | 4.03 | 0.1581 | 0.1549 | 0.1372 | 0.1752 |
| ANE int4 + clip-optimal scales | 4.03 | 0.1121 | 0.1107 | 0.1085 | 0.1115 |
| + block-Hadamard rotation (QuaRot-style, blk=128) | 4.03 | 0.1085 | 0.1085 | 0.1068 | 0.1088 |
| mixed 4/8 rows (hot rows → int8) | ~5 | 0.1105 | 0.1104 | 0.1077 | 0.1108 |
| ANE int8 RTN | 8.06 | 0.0087 | 0.0085 | 0.0076 | 0.0097 |
| *Metal gw64 affine (GPU-shipped bar)* | 4.25 | *0.0907* | *0.0905* | *0.0902* | *0.0903* |
| *gw32 affine (granularity reference)* | 4.5 | *0.0788* | *0.0788* | *0.0785* | *0.0786* |

Findings:

1. **The current ANE format is measurably inefficient — but fixably so.**
   Engine RTN int4 (max-abs/7 scale, no search) is the worst scheme measured:
   ~1.7× the weight error of the group-wise gs64 format MLX ships at the same
   bpw. The gap is not per-channel-vs-group-wise alone; it is mostly the naive
   max-abs scale.
2. **Clip-optimal scale selection closes most of it for free.** Searching the
   per-row scale multiplier (25-point grid, bake-time only, zero runtime cost,
   zero format change) takes 0.158 → 0.112 (−29%), nearly matching shipped
   gs64 quality. One function changes: `RindiAneProjection::compile_int4`
   (and the matching exporter so both paths agree).
3. **Block-Hadamard rotation adds little under iid activations** (0.112 →
   0.109). Its real benefit is suppressing *activation* outliers, which this
   test cannot see. Re-test with real calibration activations before
   investing; note it requires fusing a rotation into the preceding norm.
4. **Mixed 4/8 is not worth it as tested**: +~1 bpw buys −1.5% rel err.
5. **int8 remains in a different league** (~0.9% err), consistent with all
   prior measurements. Where size allows, int8 dominates any int4 refinement.

## 3. Clip-search scales: implemented and validated on hardware

`RindiAneProjection::compile_int4` now searches a 25-point per-row scale grid
(0.40–1.00 × max_abs/7, minimum reconstruction SSE) instead of taking max_abs/7.
Payload format unchanged; bake-time cost only, blob-cache amortised. Default on;
pass `optimize_scales=false` for legacy RTN.

A/B through the actual ANE (`probes/test_ane_int4_clip_ab.cpp`, real
gpu_backbone tensors, random input, fp32 reference):

| tensor | RTN rel err | clip rel err | improvement |
|---|---|---|---|
| layers.0.linear_attn.in_proj_a | 0.157 | **0.079** | **2.0×** |
| layers.10.linear_attn.in_proj_a | 0.264 | **0.122** | **2.2×** |
| layers.20.linear_attn.in_proj_a | 0.324 | **0.211** | **1.5×** |

Larger than the §2 prediction because these backbone tensors are *already*
group-wise quantised — the production path double-quantises (gw64 → dequant →
per-channel int4), which amplifies whatever scale-selection error remains.
That observation suggests a further win: feed `compile_int4` from the bf16
checkpoint rather than the packed backbone.

**Landmine found while probing:** `RindiAneProjection::evaluate()` copies the
input contiguously into the `[1, I, 1, width]` surface. That is only correct at
`lanes == width_` (32). At `lanes == 1` the vector lands in the wrong stride and
the ANE returns uncorrelated garbage. Production callers always pad to 32 lanes,
so nothing today hits it — but any future single-lane caller will. Fix is a
strided copy or an explicit guard.

## 4. The AWQ model (truemod/Qwen3.8-27B-AWQ-gs64-mm): GPU-only asset

Measured on layer-20 MLP weights (`probes/ane_awq_reuse_study.py`):

| path | rel err vs bf16 |
|---|---|
| AWQ dequantised → ANE int4 clip (**chain**) | 0.188 / 0.194 |
| bf16 → ANE int4 clip (**direct**) | **0.110 / 0.113** |
| AWQ as-is vs bf16 source tensor | 0.162 / 0.169 |
| naive gw64 affine on same tensor | 0.090 / 0.091 |

Two conclusions:

1. **Never requantise into ANE format from any 4-bit build.** The chain adds
   ~70% more error than quantising directly from bf16. This also applies to
   the current backbone-fed path (see §3).
2. The AWQ checkpoint stores a *transformed* graph: its activation-aware
   scaling was fused into neighbouring weight matrices (no separate smoothing
   tensors exist in the index). Its weights are therefore not drop-in
   comparable to the original graph per-tensor — the 0.162-vs-0.090 gap above
   reflects that transform, not necessarily worse serving quality. But that is
   exactly why it cannot be reused piecemeal by another engine: mixing its
   matrices with original-graph weights elsewhere breaks the equivalence.

**Verdict: the AWQ model is a GPU serving artifact (16 GB, MLX/LM Studio) and
has no role in the ANE path. Quantize ANE blobs from bf16, always.**

## 5. Answers

* **Can the engine run EXL3?** No — natively impossible (gate above); via
  offline decompression pointless. Not runnable on Metal either today.
* **Does the ANE need its own model format?** Yes: MIL text + 128-byte-header
  blob files compiled per shape into an Espresso package; weights must be
  pre-laid-out per-channel int4/int8 (packed low-nibble-first) or fp16. Every
  checkpoint needs an offline conversion pass (`export_rindi_package.py`).
  The format is best understood as a strict subset of mainstream quantization:
  same int4/int8 idea, minus groups, zero-points, and codebooks.
* **Is the format we use inefficient?** The *format* is fine within its
  constraints; the *scale selection* is not. Clip-search scales recover ~29%
  of the int4 error at zero cost. Ship that first; evaluate rotation with real
  activations second; use int8 where the size budget allows.

## 6. Reproduce

```bash
cd ane-port && make test-ane-format-gate && /tmp/rindi-test-ane-format-gate
../.venv/bin/python probes/ane_exl3_format_study.py   # from repo root
```
