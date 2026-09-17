# Flash-Next on ANE: fused projections + routing (Sep 2026)

Machine: M5 Max, ANE h17, macOS 27.0 (26A428). Model:
`~/models/Qwen3.8-Flash-Next`.

## What is proven

### Fused W8A8 projection block (all six convs of linear-attn)

`probes/ane_flashnext_block.py`, S=256, real layer-0 weights, outlier
activations, scales calibrated on an independent draw. fp16 control at
stage 1 = 2.257e-3 (harness sound).

| stage | arm | ms | worst rel L2 vs fp32 |
|---|---|---:|---:|
| S1 z→out_proj (2 conv) | fp16 | 2.33 | 2.26e-3 |
| | w8a16 | 1.28 | 4.66e-4 |
| | w8a8 | 1.32 | 2.79e-3 |
| S4 + qkv + a/b + causal dw conv1d (6 conv) | fp16 | 3.40 | 2.33e-3 |
| | w8a16 | 2.15 | 4.73e-4 |
| | w8a8 | 2.24 | 3.13e-3 |

Error does not compound across the fused graph. Depthwise causal conv1d
(`groups=10240`, k=4, pad `[0,0,3,0]`) compiles and verifies.

Bind surfaces in compiled symbol order (`kANEFModelOutputSymbolsArrayKey`,
alphabetical), not MIL return order, or eval fails Code=42.

### Container / bank packer

`runtime/q38_ane_engine.py` `_BlobPacker`: header byte 80 is a file-absolute
payload pointer. Concatenating raw `_make_blob` chunks aliased every
procedure onto tensor 0 (int8 banks → `inf`). `tests/test_blob_pack.py`
PASSes fp16/int8/int4, aligned and unaligned.

`probes/ane_w8a8_accuracy.py` still concatenates raw `_make_blob` — its
error columns remain invalid.

### CPU reference

`tools/flashnext_reference.py`, layer 0:
- prefill vs step self-check: max rel ~1e-6 (tol 2e-4)
- mlx-lm CPU cross-check: full layer max_rel 2.2e-6
- HF vs MLX l2-eps variant drifts 7.8e-3; use the MLX placement

## Routing: two rungs, both real

Text MIL rejects `top_k` / `sort` / `gather` / `reduce_argmax`
(`InvalidMILProgram`). That is the **frontend**, not the engine. The
private gather is espresso `gather_nd` (live table + indices) or
`inner_product { is_lookup: 1 }` (const embedding). Compiler symbol
`_ANECGatherLayer`. Helper: `runtime/ane_lookup.py` — **one** compiled
program. Table is a persistent IOSurface; `gather` writes 10 fp16 indices
and evaluates. CoreML `predict({"table", "ids"})` DMA-copies the table
every call and cannot beat host `copyto`.

ANE IOSurfaces are fp16. Writing int32 ids reads as ~0 and always returns
row 0. Empty espresso weights hashed every gather_nd to one ident
(SHA-256 of ""); salt the weights dict per `(vocab, dim, k)` or the second
shape silently runs the first program.

| lookup | ANE | programs | notes |
|---|---|---|---|
| `gather_nd` live 64×2560, k=10, ids-only | **0.14 ms**, rel 2e-4 | **1** | IOSurface table stays bound |
| `gather_nd` live 512×2560, k=10, ids-only | **0.14 ms**, rel 2e-4 | **1** | same; CoreML predict-copy is 0.58 ms |
| fp16 const embedding nB≥127 | zeros | 1 | ANE tile cliff; CPU-only exact |
| flattened expert `D=32768` | compile fails | — | ANEC_IR serialize I/O; not a 2560-d gather |

### Sep 12 follow-up: rank-3 expert gather executes, but is slow

Re-ran `probes/ane_gather_expert_shape.py '3D mini' '3D gu 64' eval`.
Evidence: `artifacts/coreai/gather_review_verified.log`. Synthetic data,
real gate/up matrix dimensions; this is a 64-expert subset, not all 512.

| live table shape, top-10 | host FP32 take | ANE ids + eval + FP16 snapshot | relative L2 |
|---|---:|---:|---:|
| `[32,16,2560]` | 0.03 ms | 0.57 ms | 2.082e-4 |
| `[64,1280,2560]` | 1.96 ms | 32.42 ms | 2.078e-4 |

The larger table is 419.4 MB in the FP16 surface; selected output is
65.5 MB. Initial table upload was 135.1 ms and is excluded from the
per-call measurement. These are API wall times, not isolated ANE compute
or equal-dtype bandwidth benchmarks. The new evidence supersedes any
inference that the flattened width failure also rules out rank-3 execution.
It does **not** make standalone gather a decode optimization.

Fixed `AneGather.gather(as_float32=False)` returning an alias of the
unlocked output surface: `ascontiguousarray` did not copy contiguous FP16.
Both return modes now provide an owned snapshot. The probe checks changed
indices against the reference and verifies that a subsequent evaluate
does not mutate a retained FP16 result. Both shapes pass.

Next useful ANE experiment is fused selection + multiplication, returning
activations rather than a materialized expert slab. First verify changing
ids and live table updates on a small graph, then the full expert matrix
dimensions. Compilation alone is insufficient. Retain the resident Q4 GPU
MoE as a comparison under the user's energy-per-answer objective; actual
energy measurements remain outstanding.

### Sep 12: fused projection follow-up and export correction

`probes/flashnext_fused_gather_mm.py` now tests live-table selection followed
by matmul, multiply/reduce, or dynamic convolution through Espresso. All
three tiny fused forms fail ANECIR serialization on this installation;
standalone gather with the same `[8,32,64]`, k=2 input loads. This isolates
the failure from the standalone gather shape, without proving a hardware
limitation.

The older GatherMM probes exported an already-decomposed Torch program.
Merely instantiating `GatherMM` did not preserve the composite declaration.
Corrected them to use `add_pytorch_module` with
`ExternalizeSpec(GatherMM, "gather_mm", ["num_batch_axes"])` and uint16
indices, per [Apple's GatherMM API](https://apple.github.io/coreai-torch/main/api/composite-ops/gather-mm.html).
ANE preference permits CPU/GPU fallback: successful execution alone does
not establish ANE placement. Old labels that implied otherwise were changed.

New independent numerical probe: `probes/flashnext_coreai_gather_mm.py`.
It changes activations and expert IDs on every call, changes the live
weight table halfway through, and compares to FP32 NumPy projection.
`--composite` preserves the op; `--pretranspose` moves layout conversion
outside inference; `--storage` controls NDArray backing; `--debug-out`
saves specialization metadata. Timings include input wrapping, execution,
and output snapshot; initial table upload and compilation are excluded.

Measured 64 experts, top-10, gate/up `[1280,2560]` per expert:

| configuration | median ms | worst relative L2 |
|---|---:|---:|
| decomposed, live table, ANE preferred, bytes | 19.824 | 0.002564 |
| composite, live, ANE preferred, debug specialization | 40.229 | 0.000211 |
| composite, pretransposed live, ANE preferred, bytes | 32.705 | 0.002564 |
| composite, pretransposed live, ANE preferred, IOSurface | 32.723 | 0.002564 |
| composite, pretransposed live, GPU preferred, Metal | 2.367 | 0.000211 |

These are different specialization/storage configurations, not a pure
ANE-vs-GPU hardware benchmark. The debug metadata for the composite
gate probe reports `Unsupported mps.matmul op for this ANE architecture`
and GPU residency for that operation. Other operations have ANE residency.
Thus the slow ANE-preferred case cannot be described as fused all-ANE
GatherMM. See `artifacts/coreai/gather_composite_gate_placement.json` and
`gather_*gate.log` for evidence. This does not justify integrating the
FP16 bank into the full decoder: resident Q4 MLX remains the smaller bank.

Do **not** paste `gather` into `modelWithMILText`. Do **not** compile one
program per 64-row bank. Do **not** feed int32 indices into the ids
surface.

One level below, ANECompiler's C API **accepts the descriptors** on this
part. Recipe recovered in `probes/ane_compiler_validate.py`:

```
_ANECUnitValidatorCreate(NULL, CFSTR("h17"), &out)  // status 0
_ANECTopKLayerDescInitialize(desc)                  // "Max", k=1, "Channel"
_ANECValidateTopKLayer(validator, desc, ...)        // returns 1
```

`"H17C"` is rejected. All-NULL create returns status 1; a garbage arg1
segfaults (treated as a CFType).

| op | validator (default desc) |
|---|---|
| TopK, Sort, Gather, SDPA, ArgMinMax, MatrixMult | accept (1) |
| RingBufferWriter | reject (6) — defaults incomplete |
| default Conv (no kernel sizes) | reject (6) — control, not a rubber stamp |

Execution still needs `_ANECCreateModelDictionary` → populate →
`_ANECCompile`. That schema is not recovered. `CreateModelDictionary(0,0)`
returns NULL cleanly.

## The espresso / ANECIR front end (in progress)

`_ANEInMemoryModelDescriptor` has a second factory,
`modelWithNetworkDescription:weights:optionsPlist:` (`isMILModel=NO`).
It is the Espresso path (`_ANEEspressoIRTranslator`), **not** raw MLIR.

- Looks for `model.espresso.net` under `localModelPath`.
- Shipping nets are JSON `format_version: 200` (`layers[].type`) or
  compressed `pbze`. No shipping net on this machine contains a top_k
  layer, but Espresso accepted handmade `type: "topk"` and `type: "softmax"`
  and got as far as *serializing ANECIR*.
- `compilerOptionsWithOptions:` on this path emits
  `kANEFModelType = kANEFModelANECIR` and filename `net.plist`. Passing
  those options skips the translator and ANECCompile then fails
  `InvalidNetworkSourceFileName`. Leave options nil so the translator runs.
- Current failure: `Cannot serialize ANEC_IR_repr` (I/O). The net is
  accepted; the write-out of the intermediate IR is not.

ANECompiler also embeds an MLIR `mps`/`mpsx` dialect inventory including
`mps.top_k`, `mps.sort`, `mps.gather`, `mpsx.quantized_gather`. That is
the compiler's *internal* IR, reached via espresso→ANECIR today, not via
pasting `func.func` into `modelWithMILText`.

## Full ANE layer run (the 27B contract)

`tools/flashnext_ane.py` executes one Flash-Next `linear_attention` + 512-expert
MoE decoder layer GPU-free. Same definition as 27B `pure_ane`: every GEMM,
depthwise conv, GDN state update, and expert SwiGLU is an ANE evaluate.
CPU is control — embedding-sized copies, residual recombine, top-10 of 512
router logits, paging the ten chosen expert matrices onto a live weight
surface.

Dark horse that makes 512 experts fit: **one pair of `AneDynamicLinear`
programs**, not 512 compiled procedures. The 127-model loader budget cannot
hold a procedure per expert; paging `[1280,2560]` + `[2560,640]` onto the
already-validated dynamic-weight surfaces does. Compile `S=32` (S=1 is
silently padded and a 64 KiB surface then fails Code=42).

Measured on this machine, layer 0, 4 sequential tokens, fp16 weights vs
`flashnext_reference.py` (mlx-lm l2-eps):

| token | hidden max_rel | notes |
|---|---:|---|
| 0 | 1.69e-2 | |
| 1 | 1.95e-2 | |
| 2 | 4.52e-3 | error does not grow |
| 3 | 3.64e-3 | |
| final SSM | 1.78e-3 | ANE-resident `[48,128,128]` |

37.9 ms/tok for one layer (compile ~0.8 s cold). GDN `y` tensor from the
compiled graph is ~6% off; the state surface matches at 5e-4, so readout is
`einsum(S, q)` from that surface — same class of IOSurface copy 27B already
does.

### 48-layer generate (`tools/flashnext_generate.py`)

GPU-free decode through all 48 layers, embed, mixer, and chunked `lm_head`.
Shared programs: one `AneDynamicLinear` per unique GEMM shape, one GDN step,
one GQA core (24q / 2kv / d=256 / L=256), four lm_head chunks. 512-expert MoE
pages the top-10 slabs onto one `[1280,2560]` + `[2560,640]` pair. Linear-attn
depthwise conv is CPU (same arithmetic as `FlashNextConv`) so 36 extra models
do not blow the 127-model cap. PLE/n-gram is a zero table (`ngram_index.json`
absent; layer() skips the 128 shard tensors so it does not mmap ~102 GB).

Greedy next-token after `"The"` (id 760) matches the numpy fp32 full-model
reference: **220** (a space). Chat-template prompt `Reply with exactly: OK`
generated `[248068, 198, 760, 1156]` → `<think>\nThe user`, the same first
four tokens as 27B `pure_ane` vs MLX.

Decode was 2.5–4.2 s/tok when every expert GEMM was its own evaluate and
every slab went BF16→fp32→fp16 on one thread. Packed top-10 staging (two
GEMMs per layer) plus 16-way convert is ~1.3 s/tok on a cache miss and
faster when the same experts repeat. The ANE itself does those two packed
GEMMs in ~6 ms; the rest is converting 4.7 GB of BF16 expert slabs into
the weight surfaces. 27B is faster because its weights are baked int4
inside the compiled blob and never re-paged.

```
Q38_ANE_REUSE_COMPILED=0 python3 -u tools/flashnext_generate.py --prompt 'Reply with exactly: OK' --tokens 4
python3 -u tools/flashnext_generate.py --cpu --ids 760 --tokens 1
```

`--cpu` is the same 48-layer graph in numpy (no ANE) for token checks.

## Still open

1. ANE depthwise conv (`FlashNextConv`) and fused shared-expert gate+up.
2. Espresso `type: "topk"` serialize — would move the 512-way argpartition
   on-engine. Not required for a GPU-free run.
3. Rail power (`sudo powermetrics --samplers ane_power,gpu_power`).
4. Expert-slab bf16→fp16 without an fp32 round-trip; that is the decode
   bandwidth term (~10 MB × 10 experts × 48 layers per token).
