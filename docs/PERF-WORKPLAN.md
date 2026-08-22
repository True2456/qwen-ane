# Perf & Truthfulness Work Plan (C/C++ backend only)

Status: P0-P2 deployed and verified on the live server (2026-08-22).
Measured on M5 Max, Qwen3.8-27B int4, native server, greedy:

| metric | before | after | note |
|---|---:|---:|---|
| prefill (197-tok request) | ~40 tok/s | **~66 tok/s** | memset/lock/alloc removal + Metal attention |
| decode | ~4.0 tok/s | **~4.3 tok/s** | tails still dominate; MTP is the next lever |
| server truthfulness | fake metrics | real timers only | see P0 |

Validation: `make test-metal-attention` PASS (Metal attention vs scalar CPU
oracle, max_abs 2e-3 = fp16 rounding), `make test-native-generate` PASS,
live OpenAI smoke tests pass with real usage counters.

Operational warning discovered en route: the ANE compiler materializes models
under /private/var/folders; killed/crashed runs orphaned 44 GB of them and a
100%-full disk made every cache-missing compile fail with a nil error
(misleadingly like the program-budget failure). Keep >5 GB free; purge the
hex-named materialization dirs when cleaning up.

Baseline (M5 Max, Qwen3.8-27B int4, native server):
- decode ~3.5-4 tok/s (284 ms/token documented budget: 64 ANE tails ~160 ms,
  ~324 dispatch floor ~29 ms, scalar CPU attention core, GDN surface copies)
- prefill ~40 tok/s (29-lane ANE tails, scalar CPU attention core dominates)
- MLX GPU reference: pp1024 ~900 tok/s, tg128 ~30.5 tok/s

Order of attack. Each item is verified by an existing `make test-*` target or a
new probe before the next one starts. No fake numbers anywhere: every metric
the server or TUI reports must come from a measured timer.

## P0 — Purge fabricated values (truthfulness) — DONE
- [x] `rindi_server.cpp`: remove simulated prefill (`sleep_for(prompt*1000/950us)`)
- [x] `rindi_server.cpp`: remove keyword-triggered canned tool_calls
- [x] `rindi_server.cpp`: remove canned `reasoning_content` strings; route real
      `<think>...</think>` from the model instead
- [x] `rindi_server.cpp`: usage counts real prompt/generated tokens
      (was `body.size()/4`, `gen_output.size()/4`, `tokens_saved = prompt/2`)
- [x] `rindi_tui.cpp`: remove hardcoded `2850.0` prefill tok/s + fake APC stats
      in the interactive loop
- [x] Engine exposes a real `GenerationStats` (prompt tokens, generated tokens,
      prefill ms, first-token ms, decode ms); server + TUI consume only that

## P1 — Decode hot-path overhead (safe, structural) — DONE
- [x] `evaluate_tail`/`evaluate_tail_batch`: stop memsetting the full input
      surface per call; zero once and only when the lane count changes
      (~88 MB/token of memset removed at decode, ~88 MB/chunk at prefill)
- [x] Drop per-call `IOSurfaceLock/Unlock` on the tail/recurrence/conv surfaces
      (same-process CPU writes complete before the synchronous evaluate; the
      Python driver never locked these either)
- [x] `RindiGdnRecurrence::step`: remove dead memset (every byte is overwritten)
- [x] `sample_next_token`: lm_head GEMM + argmax in ONE command buffer
      (was two create/commit/wait round trips)
- [x] RoPE angle: replace per-element `std::pow` with a precomputed inv_freq
      table (~14k pow calls/token removed)
- [x] `forward_token`: reuse member scratch buffers instead of per-layer
      vector allocations

## P2 — Metal attention core — DONE, validated
- [x] KV caches resident in Metal buffers (host mirrors one 2 KB row/token)
- [x] Kernels: `attn_scores` (q·K^T), `attn_softmax` (per head), `attn_pv`
      (Σ p·V), all encoded in one command buffer per layer
- [x] Decode path (lanes=1) and prefill batch path (lanes≤29, causal within
      chunk) share the kernels; base-position mask handles the chunk offset
- [x] Env kill-switch `RINDI_DISABLE_METAL_ATTENTION` falls back to the scalar
      CPU core; `RINDI_COMPARE_METAL_ATTENTION` logs max/mean abs diff vs CPU
- [x] Probe: `probes/test_metal_attention.cpp` — Metal vs CPU reference on a
      real layer, decode + batch shapes

## P3 — MTP / speculative decode into the free ANE lanes — IMPLEMENTED, EXACT, ~break-even
- [x] GDN state store UNIFIED (the Metal batched path and the ANE single-step
      path now share one IOSurface-backed buffer; previously prefill advanced a
      Metal copy while decode read a stale ANE surface — a latent correctness
      bug affecting all multi-lane prefill + decode sequences)
- [x] Snapshot/restore: attention position rewind (stale KV rows unreachable
      through causal masking), GDN conv windows + recurrent states copied
      (~75 MB, 1.4 ms measured), draft-cache realignment over confirmed tokens
- [x] MtpBlock: fc + full attention layer + MLP loaded from mtp.* BF16 tensors,
      quantized to groupwise int4 at load; fused q/k/v and gate/up projections
- [x] Draft-time lm_head: dedicated int4 copy (610 MB vs 2.4 GB BF16 traffic)
- [x] gemv_int4_groupwise kernel: threadgroup-per-row, K-parallel, vectorized
      nibble loads + tree reduction (serves ALL lanes=1 groupwise projections)
- [x] EXACTNESS PROVEN: greedy output token streams identical to non-spec
      decode (probes/test_mtp.cpp, MTP_EXACT=PASS); max_tokens cap honored
- [x] Measured: accepted/step 1.8-2.0 at depth 2; net speedup ~0.98x (break-even)

### Decode economics (measured on the current verified build)
- RINDI_DEBUG_TIMING decode forward: per layer core_ms=0.8 + tail_ms=2.7
  (~3.5 ms) x 64 layers = ~224 ms/forward = ~4.5 tok/s decode. Layer 63
  (no next-proj) tail_ms=2.1: the ~0.6 ms delta is next-proj GEM compute;
  ~2.1 ms is the fused MLP tail on the NPU itself.
- probes/ane_decode_budget.py: convolution ~174.6 ms/token (48%),
  "everything else" ~186.4 ms (52%), per-dispatch submit floor only
  0.091 ms x ~324 = ~29.4 ms (8% of the token budget).
- CONCLUSION (revises prior plan): decode is ANE-COMPUTE-BOUND, not
  submit-bound. Pipelined ANE submit recovers at most ~8% (the submit
  floor), NOT the 229->120 ms hoped. The "pipelined submission" lever is
  retired on evidence.
- RINDI_ENABLE_METAL_TAIL=1 is much SLOWER than ANE (0.6 vs 4.5 tok/s).
  The ANE fused tail is the right backend; do not chase the Metal GEM tail
  as a decode speed-up.
- REVERTED a failed optimization: per-lane checkpoint-replay (GDN
  recurrence ckpts + arithmetic conv-window rebuild + adaptive depth) broke
  MTP exactness (MTP_EXACT=FAIL, diverged at token 14). Restored verified
  HEAD (MTP_EXACT=PASS, 0.93x). Reintroduce only with a per-token golden
  replay test.

## P4 — Metal GEMM kernel — DONE, 3x, bit-exact (prefill still ANE-bound)
- [x] Root cause: the batched prefill path (lanes>1) used the naive
  gemm_int4_groupwise/rowwise: one thread per (row, lane) dots the full K
  serially, so the 34836x5120 gate took ~16 ms vs MLX's ~0.6 ms on the same
  GPU (gpu_saturate.py: full MLP int4 @ S=32 = 1.62 ms = ~0.5 ms/projection).
  The old tiled kernel was dead code (no dispatch) AND was roWWise-layout,
  not the groupwise the path actually hit.
- [x] New gemm_int4_groupwise_batch + gemm_int4_rowwise_batched: one
  threadgroup per output row; threads tile [lane, kpar]; W word loaded once
  into threadgroup (kills the lanes-redundant re-read) and K split across a
  power-of-two slice with an exact per-lane tree reduction.
- [x] Wired in RindiAneProjection::metal_dispatch for lanes>1/offset==0
  (env kill RINDI_DISABLE_BATCH_GEMM); folded next_proj (offset!=0) still
  uses the offset kernels.
- [x] CORRECTNESS: probes/test_metal_gemm_micro.cpp asserts bit-exact vs the
  old kernel: groupwise mismatches=0/1114112 maxdiff=0;
  rowwise-batched vs naive mismatches=0/557056. MTP_EXACT=PASS.
- [x] SPEED: 15.9 -> 5.4 ms/call for the 34816x5120 gateways (3.0x),
  tail_ms/layer 30 -> 18 ms, Metal prefill 12.6 -> 19.6 tok/s.
- HONEST REMAINDER: Metal-tail prefill is now 19.6 tok/s vs ANE's ~71. The
  K-parallel kernel still leaves 5.4 ms/call (MLX ~0.5 ms). Next lever is
  2D row/col output-tiling + shared block loaders (below). Until that lands,
  ANE fused tails stay the prefill backend.

### Why MLX is still ~10x faster (source-verified, not guessed)
MLX's real int4 GEMM lives locally at
  /Users/true/AppleLLM/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/
  metal/steel/gemm/{gemm.h, mma.h, loader.h} + kernels/steel_gemm_splitk.h.
- gemm.h GEMMKernel: blocks the output into BMxBN tiles (NOT one row like
  ours); each threadgroup owns an output tile and streams A/B K-blocks
  through threadgroup shared memory (BlockLoader, loaders).
- loader.h: block loaders read A-tile + B-tile chunks once per K-step into
  shared threadgroup memory; DRAM A/B traffic amortized across the tile,
  so A+B are*not* re-read per output row (our kernel re-reads the whole
  A matrix for every threadgroup row).
- mma.h: register-blocked MMAFrag - 8x8 metal::simdgroup_matrix fragments,
  each thread holds WM*WN accumulators in fp32 registers and does the full
  inner product with fragment loads + unrolled outer product. This is the
  SIMD/HMA unit that gets the NU-factor; our per-thread scalar inner loop
  cannot reach it.
- kernels: gemm_splitk split-K across tid.z partitions with atomic
  accumulation. Also gemm_fused (loads A operand tiles) etc.
- CONCLUSION: the 12 TFLOP/s ceiling is reachable in our engine by porting
  this structure into WGSL (output tiles + threadgroup block loaders +
  register-blocked accumulation). It is a large careful port, NOT a oneshot
  glyph dump - the failed drafts underscore this.

## P4b - Attempted row-tile - REJECTED by harness (measured facts)
- ATTEMPTED gemm_int4_groupwise_tile: one threadgroup per 16-row x 32-lane
  block, A-block + W-block streamed into threadgroup once per K-step (the
  A-amortization idea). 
- WAS REJECTED by probes/test_metal_gemm_micro.cpp:
  TILE-vs-BATCH mism=1114112/1114112 maxdiff=22.7 (WRONG) AND 21.9ms vs
  BATCH 5.4ms (SLOWER). Reverted to the committed BATCH kernel. The harness
  gate did its job - a silent-bug kernel was caught before wiring.
- WHY: row-tile with serial-K per thread LOSES the K-parallelism that makes
  BATCH fast. A-sharing helps DRAM but serial-K per thread dominates.
- VERIFIED: `simdgroup_matrix<float,N,M>` COMPILES in this engine's runtime
  WGSL dialect (via newLibraryWithSource test). So MLX's register-blocked
  MMA IS reachable here - the true MLX-speed path is NOT just row-tiling but
  output-tiles + register-blocked simdgroup MMA.
- REAL NEXT: port a minimal register-blocked MMAFrag accumulation into a
  GEMM, bit-exact gate it in the harness BEFORE wiring. Prefer MLX's
  gemm_splitk/steel_gemm structure as the reference (in mlx-swift).

## Targets
- decode: ~4.5 tok/s measured now (was 4.0->4.3). Single forward near the
  per-layer ANE floor; >6 needs concurrency, not submit-rate.
- prefill >= 150 tok/s @1-2K ctx after P2 (from ~40)
- every metric traceable to a timer in this repo (tools/bench_latency.py)
