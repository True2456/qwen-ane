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

### Why break-even (measured, not guessed)
- verify (3 lanes through 64 layers): 229-233 ms - barely more than one
  1-lane step (tails are lane-parallel). This is the unit of progress.
- draft (2 MTP forwards): ~25-50 ms - cheap enough.
- partial-rejection rounds replay the proven prefix through the full stack:
  another ~230 ms. At ~1/3 rejections per depth this consumes the gain.
- The lever that flips this positive: pipelined ANE submission (layer N+1
  submitted while N runs; the driver queue is 127 deep) which cuts the 229 ms
  verify floor AND plain-decode latency alike. Next work item.

## Targets
- decode ≥ 6 tok/s after P1+P2 (from ~4), ≥ 8-10 with P3 (lane occupancy)
- prefill ≥ 150 tok/s @1-2K ctx after P2 (from ~40)
- every reported number traceable to a timer in this repo
