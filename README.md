# Running Qwen3.8-27B on the Apple Neural Engine

A working 27B LLM whose weights live on the Apple Neural Engine of an M5 Max,
driven through the private `AppleNeuralEngine.framework` — no CoreML.

```bash
tools/ane serve --ane-chain --ane-lm-head --dense-bits 4
```

Answers on `http://127.0.0.1:1239/v1` (OpenAI-compatible).

## Framework-free pure ANE backend

`tools/pure_ane.py` is a second backend, independent of the hybrid server. It
does not import MLX, oMLX, PyTorch, Core ML, Transformers, or a GPU runtime. It
reads BF16 safetensors directly, uses `tokenizers` for token IDs, and submits
MIL programs and IOSurfaces straight to `AppleNeuralEngine.framework` through
the private driver vendored at `runtime/q38_ane_engine.py`, which itself needs
only the standard library and numpy.

```bash
tools/ane pure-loader-smoke
tools/ane pure-gdn-layer-smoke --bits 16
tools/ane pure-attention-layer-smoke --bits 16
tools/ane pure-infer --bits 16 --tokens 4 --verify-reference
tools/ane pure-infer --bits 4 --tokens 32 --mtp-draft 2
```

The complete 64-layer scheduler now bakes and dispatches without a GPU or MLX:

| precision | ANE programs | learned-weight blobs | status |
|---|---:|---:|---|
| per-output int4 | 122 | 12.86 GB | exact for 4 MLX tokens; coherent after divergence; 2.770 tok/s with three-lane prompt ingestion |
| int4 + pure MTP depth 2 | 124 | 13.16 GB | exact same 32 target tokens; **3.185 tok/s**, 2.385 accepted tokens/cycle |
| per-output int8 | 122 | 25.72 GB | exact for 16 MLX tokens; then diverges but correctly answers `OK`; 2.421 tok/s |
| fp16 | 125 | 51.42 GB | **full semantic inference passes**; known prompt generated the same first four tokens as MLX |

The earlier int4 incoherence was caused by shared fp16 arithmetic bugs, not
necessarily by quantization: after fixing those bugs, int4 generated the exact
same four-token reference prefix in 4.791 seconds with 12.86 GB of blobs. In a
32-token comparison it diverged at token 5 but continued with coherent,
semantically equivalent reasoning. Int8 matched the MLX reference for 16 tokens
and then took a different but correct path, producing the requested `OK`. This
is strong evidence that both formats work, but not yet a broad benchmark.
`pure-infer`
continues to default to fp16. CPU work is limited to tokenization,
embedding-row selection, IOSurface byte movement, cache bookkeeping, and greedy
sampling; all learned tensor arithmetic, attention, GDN convolution and
recurrence, normalization, MLPs, and the final head run on the ANE.

The standalone backend now consumes the checkpoint's one MTP layer without
MLX. `--mtp-draft 2` batches `[confirmed, draft1, draft2]` through the target's
large projections and MLPs, while causal attention and GDN recurrence advance
in order. Rejections restore all 48 compact GDN states, convolution histories,
and attention offsets, then replay only the proven prefix. The 32-token MTP and
non-MTP runs emitted identical token IDs; MTP improved the current batched
baseline by 15.0% and the original one-lane-prompt result by 29.1%.

Measured full-fidelity check on M5 Max:

```text
PURE_ANE_BAKE=PASS programs=125 blobs=51.42GB seconds=82.3
PURE_ANE_REFERENCE=PASS tokens=[248068,198,760,1156]
PURE_ANE_EXECUTION=PASS prompt_tokens=13 generated=4 seconds=11.594
token_ids=[248068, 198, 760, 1156]
<think>
The user
```

The output exactly matches the four-token MLX reference for the same templated
prompt. Allow roughly 55 GB of genuinely available disk space during an fp16
cold bake: the private compiler materializes weights and macOS may use swap.
Deleted files still held open by another application do not count as free
space (`lsof +L1` is useful when `df` and Finder disagree).

The measured resident-program failure near 134 programs was avoided with
projection procedure banks (two quantized or five fp16), one shared
attention-preparation program, one
shared attention core, and one shared GDN recurrence. This is why the pure
layout differs from the 69-program hybrid chain described below.
Pure MTP adds two programs: an fp16 embedding/hidden fusion projection and the
MTP decoder tail. Its QKV projection is another procedure in the existing
attention bank, and it shares the target's four dynamically-normalized
vocabulary-head programs.

## What actually runs on the ANE

| block | coverage | programs |
|---|---|---|
| layer tails: `out_proj → +residual → RMSNorm → gate/up → silu → mul → down → +residual` | 64/64 | 64 |
| next-layer `input_layernorm` + input projection, folded into the layer before it | 63/64 | 0 (rides along) |
| layer 0's input projection (nothing precedes it) | 1 | 1 |
| `lm_head` [248320, 5120], split along the vocabulary | 1 | 4 |
| **total** | | **69 / 127** |

12.19 GB of int4 blobs. Every linear projection in the model is on the ANE.

Still on the GPU in the default build: embeddings, the attention core
(softmax·QK<sup>T</sup>·V), and GDN `conv1d`. With `--ane-gdn-step`, the
gated-delta recurrence keeps state resident in ANE-layout IOSurfaces, computes
polynomial softplus/decay and beta inside the same ANE graph, and runs at
0.344 ms/layer-token in the standalone benchmark (state rel 1.06e-3), versus
4.39 ms for the old host round trip and about 0.66 ms on GPU.

In the current hybrid server, this option is still slower end-to-end (5.6 tok/s
versus an 8.5 tok/s GPU baseline) because GPU-produced q/k/v/a/b force a
3.06 ms synchronization/marshalling boundary per GDN call. Real inference is
coherent; performance requires chaining the preceding ANE projection and
temporal convolution directly into the resident recurrence surface.

## Start here

* **[docs/SETUP.md](docs/SETUP.md)** — moving this to another machine, paths, the
  `libomp` crash, verifying the install.
* **[docs/ANE-REFERENCE.md](docs/ANE-REFERENCE.md)** — what the ANE accepts and
  rejects. Read before writing any MIL: the ops that silently fail, the width
  and alignment rules, the program limit. This is the part that took the longest
  to learn.
* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how the model is mapped onto
  ANE programs, and why it is shaped this way.
* **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** — measured throughput and
  energy, including where the ANE wins and where it does not.
* **[docs/FULL-ANE-FEASIBILITY.md](docs/FULL-ANE-FEASIBILITY.md)** — measured
  feasibility of removing the remaining GPU blocks: full attention and GDN
  arithmetic work, fp16-safe softplus is solved, and recurrent state now stays
  in ANE-layout IOSurfaces across decode steps.
* **[docs/FULL-HANDOFF.md](docs/FULL-HANDOFF.md)** — the complete lab notebook,
  44 sections, including the dead ends and the claims that turned out wrong.

## Honest summary

The ANE is **1.7× more efficient per joule** than the M5 Max GPU (1.24 vs
0.74 TFLOP/W) and draws ~6 W against 64–84 W. The quoted 38+ TOPS ANE figure is
a theoretical INT8/FP16 dual-lane peak; it is not contradicted by this repo's
~10–16.4 TFLOP/s-equivalent sustained kernels, which measure specific compiled
graphs. The comparable GPU kernel reaches ~45–48. Whole-model pure int4 now
measures 2.770 tok/s without MTP and 3.185 with pure MTP; the older hybrid path
measures 3.5–3.8 tok/s against 8.8 tok/s on the GPU.

So this is currently a power-efficiency and GPU-availability result, with real
unused ANE headroom still visible between whole-model throughput, sustained
kernel throughput, and theoretical peak.
