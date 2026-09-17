# Qwen3.8-27B private-MIL chunked GDN prefill

This change targets `PureAneRuntime`, int4 Qwen3.8-27B, M5 Max, macOS 27.
It does not use the Flash-Next Core AI packages. No MLX/torch/coreml is loaded
into the qualification or benchmark process.

## Implementation

`tools/ane_gdn_chunk.py` implements the chunkwise gated delta rule from
[Yang et al. §3.3](https://arxiv.org/abs/2412.06464), using the product-scan
and blocked-triangular-inverse spelling in `probes/flashnext_mil_chunk.py`.

Full 16-token prefill batches use the shared program. Single-token decode and
ragged remainders stay on `AneGdnRecurrence`. Compact state and
materialize/restore are unchanged. Per-layer causal convolution is unchanged.
Program width stays 32 and active lanes stay 16. The 16-step unroll remains
loaded for A/B (`Q38_ANE_GDN_PREFILL=unroll`).

Default is `Q38_ANE_GDN_PREFILL=chunk`. `PureAneRuntime.set_prefill_recurrence()`
can switch a resident process under the serving lock after clearing the request
prefix cache. Loading the chunk graph adds one program and does not rebake
weight banks.

The shared preparation emitter is the original Q/K normalization, head
repetition, softplus polynomial, decay and beta preparation. The retained
unroll emits byte-identical MIL at widths 8, 16 and 32.

Native 16-wide matrices reached the compiler after packing constants, then
failed instruction validation (the documented sub-32 matmul wall). The chunk
uses `[1,48,32,128]` tiles containing 16 live rows. Padded rows have zero
Q/K/V/beta and unit decay. Final state uses live row 15; ragged runtime
batches are not padded into state. C=32 live was not deployed: active lanes
stay 16.

Pairwise decay is a product scan, including exact zero gates. There is no
reciprocal cumulative decay and no logarithm. `(I+A)` is inverted by five
blocked doubling stages over the 32-wide tile. Query scaling by 64 is already
present in the preparation/output convention; `AneGdnTail` undoes it. Delta is
multiplied by 64 for the state-update matmul and that product is divided by 64
before adding the decayed initial state.

All constants are packed into one registered `weight.bin`. Separate mask files
caused `verifyBundleAtPath: hash mismatch`.

Host packing on the full-batch path copies compact fp16 GDN state
(`snapshot` / `(H·D, D)`) rather than materializing a float32 `(H, Dv, Dk)`
tensor and transposing it twice. `run_loaded` keeps fp16. That does not change
the MIL.

## Numerical qualification

```sh
env -u PYTHONPATH Q38_ANE_REUSE_COMPILED=1 \
  /opt/homebrew/bin/python3 -u -P probes/pure27_chunk_check.py
```

Inputs are real layer-0 int4 projection and causal-convolution activations
from `eval/code.txt`. The sequence is 32 stepwise prefix tokens, a 16-token
chunk, then eight stepwise decode tokens.

| Comparison | Output relative error | State relative error |
| --- | ---: | ---: |
| NumPy chunk vs sequential equations | 4.74e-7 | 1.68e-7 |
| Chunk ANE vs NumPy using captured ANE preparation | 0.001061 | 0.000529 |
| Chunk ANE vs stepwise ANE | 0.001301 | 0.001983 |
| Eight following stepwise decode tokens | 0.001169 | 0.001735 |

Errors use `max(abs(got-reference)) / max(abs(reference))`, as in
`ane_gdn_scan64.py`. Gate: output <0.012 and state <0.01 against stepwise ANE,
with the captured-preparation oracle additionally <0.01 on both.

The ideal float32 preparation gives state error 0.02742 for the chunk and
0.02791 for the baseline recurrence. That pre-existing normalization/gate
rounding difference is not attributed to chunking.

Layer microbenchmark (median of 11 after warmup):

| Path | `run_loaded` | `load` + `run_loaded` |
| --- | ---: | ---: |
| 16× stepwise | 12.47 ms | — |
| Unroll(16) | 3.31 ms | 3.50 ms |
| Chunk(16 live / 32-wide tile) | 1.12 ms | 1.26 ms |

`load`+`run` is **2.79×** versus the unroll. These are layer microbenchmarks,
not end-to-end prefill claims.

CPU tests cover zero gates, correlated keys, nonzero prefix states, chunk
composition, and compact-state packing:

```sh
env -u PYTHONPATH /opt/homebrew/bin/python3 -u -P -m pytest \
  tests/test_pure27_gdn_chunk.py tests/test_gdn_chunk.py \
  tests/test_tool_parse.py tests/test_sampling.py -q
```

## Resident 4k and agentic measurement

```sh
env -u PYTHONPATH Q38_ANE_REUSE_COMPILED=1 \
  Q38_MODEL=/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B \
  /opt/homebrew/bin/python3 -u -P probes/pure27_chunk_bench.py
```

One process owns `PureAneService` on port 1240. It bakes at context 4096,
measures the unroll, then loads the chunk graph. Cold requests use
`POST /v1/benchmarks` with `prefix_cache: false`. Agentic turns use
`POST /v1/chat/completions` with `prefix_cache: true`.

The cold prompt is exactly **4,080** tokens from `eval/prose.txt` (255 complete
16-token GDN batches), leaving room for eight generated tokens inside context
4096. One excluded warmup and two timed runs per mode. Peak RSS **22.79 GB**.
Program count **70** after the chunk graph is loaded (69 at an unroll-only bake).

Official numbers below are `results/pure27_chunk/benchmark.json` after compact
fp16 state packing. `benchmark_before_host.json` is the same workloads before
that packing (`gdn_recurrence` 1.81×).

### Cold 4k (timed mean of two runs)

| Mode | prompt / evaluated | TTFT | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Unroll | 4080 / 4080 | 286.68 s | 14.23 | 2.958 |
| Chunk | 4080 / 4080 | 259.24 s | 15.74 | 2.957 |

Overall prefill **1.11×**. Decode after the 4k prefill is unregressed versus
this unroll. Both sit below the historical ~3.4 tok/s short-context decode
figure because `AneLongContextAttentionCore` is heavier at 4k; the unroll
already was.

Prefill profile, milliseconds per token of 4,080, timed mean:

| Operation | Unroll ms/tok | Unroll % | Chunk ms/tok | Chunk % |
| --- | ---: | ---: | ---: | ---: |
| `attention_core` | 36.08 | 51.4 | 35.99 | 56.7 |
| `gdn_recurrence` | 11.65 | 16.6 | 5.11 | 8.0 |
| `gdn_tail` | 9.48 | 13.5 | 9.43 | 14.8 |
| `projection_head` | 3.88 | 5.5 | 3.86 | 6.1 |
| `attention_tail` | 3.07 | 4.4 | 3.07 | 4.8 |
| `attention_prepare` | 3.00 | 4.3 | 2.99 | 4.7 |
| `gdn_conv` | 1.68 | 2.4 | 1.67 | 2.6 |
| `final_head` | 0.68 | 1.0 | 0.68 | 1.1 |
| `embedding` | 0.39 | 0.6 | 0.41 | 0.6 |

`gdn_recurrence` **2.28×** (11.65 → 5.11 ms/tok). Attention core is unchanged
and is half of 4k prefill, which is why the wall-clock win is 1.11×.
