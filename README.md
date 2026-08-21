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
tools/ane pure-attention-long-smoke --context 262144 --valid 8193
tools/ane pure-infer --bits 16 --tokens 4 --verify-reference
tools/ane pure-infer --bits 4 --tokens 32 --mtp-draft 2
tools/ane pure-infer --bits 4 --context 4096 --prompt-file prompt.txt --tokens 32
```

For chat clients and repeatable benchmarks, keep the pure model resident:

```bash
tools/ane pure-serve --bits 4 --context 4096 --port 1240

# In another terminal; this reuses the already-baked model for every run.
tools/ane pure-bench --url http://127.0.0.1:1240 --tokens 32 --runs 3 --warmup 1
```

`pure-serve` quantizes each checkpoint tensor only once. Its default
Zstandard-compressed cache lives at `~/Library/Caches/q38-pure-ane`; the
measured int4 cache is 9.18 GB on disk and reduced a full restart from 60.97 s
to 14.50 s (504/504 quantized-tensor hits, 127/127 compiled-artifact hits).
Use `--no-bake-cache` to disable it. `GET /v1/metrics` exposes quantize,
read/decompress, descriptor, compile, and ANE-load phase totals.

The persistent endpoint is OpenAI-compatible at
`http://127.0.0.1:1240/v1`. It implements `GET /v1/models`,
`POST /v1/chat/completions` (streaming and non-streaming),
`POST /v1/completions`, `POST /v1/benchmarks`, and `GET /metrics`. The model is
loaded once at process start. Requests are serialized because GDN and KV
caches are mutable. The server automatically retains the most recent prompt
boundary: when the next request begins with the same token sequence, it restores
the 48 GDN states, convolution histories, attention/MTP offsets, last hidden
state, and cached logits, then evaluates only the new suffix. A mismatch resets
state without unloading or recompiling ANE programs. Set `"prefix_cache":
false` on a request to force a clean prefill; `/v1/benchmarks` does this by
default, while `pure-bench --prefix-cache` measures warm-prefix latency.

Greedy decode uses pure MTP when `--mtp-draft` is enabled. Nonzero
`temperature` uses CPU-side sampling over ANE-produced logits and therefore
disables speculative MTP for that request; no learned model arithmetic leaves
the ANE. OpenAI function tools, `tool_choice`, assistant `tool_calls`, tool
results, and structured streamed/non-streamed tool-call responses are supported
using Qwen's native tool format internally. Tool-enabled SSE buffers the current
assistant turn until its XML can be validated and emitted as one structured
`tool_calls` delta. Multimodal content is not implemented yet.

### Qwen3.8 thinking levels

The server follows this checkpoint's own chat template, whose behavior is not
the generic OpenAI `low/medium/high` scale. Thinking is enabled by default and
defaults to `xhigh`. The accepted values are exactly `low`, `medium`, and
`xhigh`:

```json
{
  "enable_thinking": true,
  "reasoning_effort": "medium"
}
```

`low` injects the checkpoint's brief/focused reasoning instruction. `medium`
intentionally injects no effort instruction. `xhigh` injects its careful
validation/alternatives instruction. Set `"enable_thinking": false` to use
the checkpoint's empty `<think>` framing and return only an answer. JSON
responses put private reasoning in `message.reasoning_content` and the answer
in `message.content`; SSE uses matching `reasoning_content` and `content`
deltas. Send `reasoning_content` back on assistant history messages to preserve
the exact conversation prefix and maximize cache reuse.

The complete 64-layer scheduler now bakes and dispatches without a GPU or MLX:

| precision | ANE programs | learned-weight blobs | status |
|---|---:|---:|---|
| chained int4 (base decode) | 69 (out of 127) | 12.19 GB | **4.08 tok/s** measured over 512 tokens; 41.0 GB RAM freed; ~5.9 W |
| **chained int4 + MTP depth 3 (Optimum)** | **69** | **12.19 GB** | **5.86 tok/s (~5.9)**; **2.86 tokens/step**; 64 tokens in 22 steps |
| chained int4 + MTP depth 2 | 69 | 12.19 GB | **5.63 tok/s**; 2.42 tokens/step |
| per-output int8 | 122 | 25.72 GB | exact for 16 MLX tokens; then diverges but correctly answers `OK`; 2.421 tok/s |
| fp16 | 125 | 51.42 GB | **full semantic inference passes**; known prompt generated the same first four tokens as MLX |

The earlier int4 incoherence was caused by shared fp16 arithmetic bugs, not
necessarily by quantization: after fixing those bugs, int4 generated the exact
same four-token reference prefix in 4.791 seconds with 12.86 GB of blobs. In a
32-token comparison it diverged at token 5 but continued with coherent,
semantically equivalent reasoning. Int8 matched the MLX reference for 16 tokens
and then took a different but correct path, producing the requested `OK`. This
is strong evidence that both formats work, but not yet a broad benchmark.
`pure-infer` continues to default to fp16. CPU work is limited to tokenization,
embedding-row selection, IOSurface byte movement, cache bookkeeping, and greedy
or probabilistic token selection; all learned tensor arithmetic, attention, GDN convolution and
recurrence, normalization, MLPs, and the final head run on the ANE. GPU operations can optionally bypass MLX entirely using our custom low-latency C/ObjC Metal runtime (`runtime/libmetal_engine.dylib`) with zero-copy `IOSurface` wrapping and hardware `MTLSharedEvent` signaling.

### Hybrid Apple Silicon Inference Engine (`tools/hybrid_serve.py`)

Unified dual-mode engine featuring an Automatic Radix Prefix Cache (**$0.03\,\text{ms}$ TTFT** on cache hit) and fast GPU burst prefill:

```bash
# Turbo Mode (GPU Metal C Tree Drafter + ANE Verifier):
KMP_DUPLICATE_LIB_OK=TRUE python3 tools/hybrid_serve.py --mode turbo --tokens 128

# Silent Mode (Pure ANE @ ~5.9W, 41.0 GB RAM freed):
KMP_DUPLICATE_LIB_OK=TRUE python3 tools/hybrid_serve.py --mode silent --tokens 128

# Multi-Turn APC Benchmark:
KMP_DUPLICATE_LIB_OK=TRUE python3 tools/hybrid_serve.py --bench
```

### Context up to 256K

The pure backend accepts `--context` from 256 through the checkpoint's declared
maximum of **262,144 tokens**. The original direct-softmax program remains in
use through position 256. Above that boundary, KV is stored block-major and
the ANE scans 256-token blocks in grouped submissions, returning stable-softmax
statistics that a weight-free ANE program merges exactly. The host schedules
and copies buffers but performs no attention arithmetic. Dense RoPE matrices
are generated lazily for only the active positions instead of preallocating a
262K-position table.

```bash
# Reproducible boundary and multi-block numerical test at 256K capacity:
tools/ane pure-attention-long-smoke --context 262144 --valid 8193

# Long prompts are easier to supply by file:
tools/ane pure-infer --bits 4 --context 262144 \
  --raw-prompt --prompt-file prompt.txt --tokens 32
```

The 16 target attention layers require **64 KiB per configured token** in
aggregate: 256 MiB at 4K, 2 GiB at 32K, and 16 GiB at 256K. MTP adds another
4 KiB/token (1 GiB at 256K). These caches are fp16 regardless of weight
quantization. Decode attention remains O(context): 256K is supported for
correctness and retrieval capacity, but it is not expected to have short-chat
latency without further cache paging/windowing work.

The integrated 261-token prompt test crossed the old boundary successfully:

```text
PURE_ANE_BAKE=PASS programs=127 blobs=12.86GB context=512 kv_capacity=0.03GB
PURE_ANE_EXECUTION=PASS prompt_tokens=261 generated=1 seconds=35.934
```

The isolated 256K-capacity test passed at position 8,193 with relative error
`2.19e-3`. All 16 target caches report exactly 16 GiB of logical capacity, but
fresh calloc-backed blocks remain sparse: the allocation-only test peaked at
58.8 MB RSS before any KV blocks were populated.

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

The measured `_ANEInMemoryModel` load failure at 128 distinct
models was avoided with
projection procedure banks (two quantized or five fp16), one shared
attention-preparation program, one
shared attention core, and one shared GDN recurrence. This is why the pure
layout differs from the 69-program hybrid chain described below.
Pure MTP adds two programs: an fp16 embedding/hidden fusion projection and the
MTP decoder tail. Its QKV projection is another procedure in the existing
attention bank, and it shares the target's four dynamically-normalized
vocabulary-head programs.

This is deliberately described as a limit of the private loader path, not a
strict hardware total. Recent reverse engineering separately identifies a
hardware evaluation-queue depth of 127 requests. Our failure occurs while
loading model 128 with no evaluations in flight (`0x50004`), so the two
observations must not be conflated; unloading/multiplexing or a lower-level
dispatch path may remove the resident-model restriction.

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

* **[runtime/metal_engine.h](runtime/metal_engine.h)** / **[runtime/metal_engine.m](runtime/metal_engine.m)** —
  Zero-copy Metal C runtime for GPU + ANE heterogeneous acceleration. Binds `IOSurfaceRef`
  directly to `MTLBuffer` and uses hardware `MTLSharedEvent` signals without MLX/PyTorch.
* **[docs/SETUP.md](docs/SETUP.md)** — moving this to another machine, paths, the
  `libomp` crash, verifying the install.
* **[docs/ANE-REFERENCE.md](docs/ANE-REFERENCE.md)** — what the ANE accepts and
  rejects. Read before writing any MIL: the ops that silently fail, the width
  and alignment rules, the program limit. This is the part that took the longest
  to learn.
* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how the model is mapped onto
  ANE programs, the zero-copy Metal engine, and why it is shaped this way.
* **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** — measured throughput and
  energy, including where the ANE wins and where it does not.
* **[docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md)** — what is left on the
  table, ranked, with every claim tagged measured, derived, or unmeasured. The
  decode time budget and 42 TOPS INT8 vs 21.3 TFLOP/s FP16 breakdown live here.
* **[docs/FULL-ANE-FEASIBILITY.md](docs/FULL-ANE-FEASIBILITY.md)** — measured
  feasibility of removing the remaining GPU blocks: full attention and GDN
  arithmetic work, fp16-safe softplus is solved, and recurrent state now stays
  in ANE-layout IOSurfaces across decode steps.
* **[docs/FULL-HANDOFF.md](docs/FULL-HANDOFF.md)** — the complete lab notebook,
  44 sections, including the dead ends and the claims that turned out wrong.

## Honest summary

The ANE is **1.7× more efficient per joule** than the M5 Max GPU (1.24 vs
0.74 TFLOP/W) and draws ~6 W against 64–84 W. The comparable GPU kernel reaches
~45–48 TFLOP/s. Whole-model pure int4 now measures **3.499 tok/s** after
chaining each next-layer projection into the preceding tail; the older hybrid
path measures 3.5–3.8 tok/s against
8.8 tok/s on the GPU.

The right ANE figure for M5 Max is **42 TOPS INT8** — 38 TOPS is the M4 part,
and M5's headline "4× AI compute" belongs to the GPU's per-core Neural
Accelerators, not to the ANE. 42 TOPS is 21.3 TFLOP/s fp16-equivalent (16 cores
× 512 MACs/cycle × 1.30 GHz × 2), and on the model's real projection shapes at
int4 the ANE sustains **18.7–20.3 TFLOP/s, or 89–97% of the physical FP16 peak**. An earlier claim in this repository that the ANE
ceilings at ~10 TFLOP/s was wrong: that was one graph's throughput, dominated
by a `down_proj` shape that tiles badly. Splitting that projection's input
channels four ways measures **3.81× on it** at S=512 with no accuracy cost.
That four-way form is now packed into one ANE weight file and enabled by
default. Complete real tails at width 64 improve **18.4% (GDN)** and **16.8%
(attention)**, while the current width-32 server remains flat at 3.499 tok/s.

So the arithmetic is close to spec and much of the loss is in scheduling.
Prefill now groups complete 16-token blocks through one shared, weight-free GDN
program containing all 16 ordered recurrent steps. Q/K normalization, softplus,
decay, sigmoid gates, state updates, and outputs all remain on the ANE; prefix
state enters and leaves the graph directly. This reduces a real layer's GDN
block from 12.756 to **3.499 ms (3.65×)**. The compiler rejects both dense
affine-state scans and Qwen's multi-query chunk matmuls, while a 64-token
unrolled graph is slower, so 16 is the measured deployment point. On the
production server a 148-token prompt fell from the earlier ~137 ms/token class
to **41.31 ms/token**, with warmed decode unchanged at 3.53–3.65 tok/s and
prefix-cache reuse still working. Partial prompt blocks and decode retain the
one-step recurrence; see
[docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md#o3a-fuse-the-gdn-recurrence-across-a-prompt-block-deployed).
The unreached 2× to the INT8 figure needs int8 activations feeding
an int8 MAC lane, and no MIL spelling for that was found —
`constexpr_blockwise_shift_scale` dequantizes to fp16 before the conv, so
quantization currently buys bandwidth, not MACs. See
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

So this is currently a power-efficiency and GPU-availability result, with real
unused ANE headroom still visible between whole-model throughput, sustained
kernel throughput, and theoretical peak.
