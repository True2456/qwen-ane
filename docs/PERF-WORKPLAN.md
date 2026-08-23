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

## P5b - Lever 1 RESULT: wide-width tails work but prefill does NOT accelerate
- MEASURED: width 32 and 62 both ~90 tok/s? NO - ~70 tok/s prefill on the same
  ~1500-token prompt (~14.5s vs ~14.8s wall incl decode). Raising the lane
  ceiling from 29 (width 32) to 59 (width 62, ~2x fewer chunk passes) does NOT
  move prefill throughput.
- The ~70 tok/s is a THROUGHPUT PLATEAU independent of lane count. The tail
  cost per pass does not amortize with lanes (double lanes -> ~same time per
  pass, ~2x tokens but not 2x throughput suggests a fixed per-FLOP/weight
  ceiling, not chunk overhead).
- Widening DOES evaluate: width 62 runs correctly, 64 rejected; the shipped
  width-32 is over-conservative. Parametric plumbing (P5a) is correct, kept.
- CONCLUSION: wides prefill is not the throughput lever; the unique path is
  compute (ALU int8) or fusing. Defer prefill. PIVOT to Lever 2 (decode):
  per-lane recurrence state checkpoints for O(1) MTP rollback - the only
  clear win and width-independent.
## P5a - Lever 1 (wide-width tails): compile OK, EVAL blocked - groundwork landed
- CONFIRMED: the ANE fused tails are runtime-compiled from build_tail_mil()
  (NOT frozen package blobs - earlier conclusion was wrong). At
  RINDI_ANE_WIDTH=64 the engine compiles all 64 tails + cores at seq=64
  ("Compiled 64 fused transformer tails").
- Parametric fixes landed (all width-32-preserving, verified 69.9 tok/s
  prefill / 4.0 tok/s decode unchanged):
  * forward_prompt_batch guard: hardcoded 29 -> chain seq_len-3
  * attention: fused-QKV gate + Metal workspaces sized by width_ not 32
  * GDN recurrence: buffers + step_batch guard sized by width_ (160)
  * GDN layer: recurrence kept at proven 160 (64 rejected by Espresso)
  * ane_c_bridge: in-memory eval now falls through to direct-client
    failover on failure (was hard-return false)
- BLOCKER: seq>32 programs COMPILE but do NOT EVALUATE through the in-memory
  request path (even 2 lanes fails at width 64). Direct-client failover added
  but ctx->aneClient is unset for runtime-compiled models, so it never runs.
  This matches the shipped package being 32-wide by design.
- NEXT for Lever 1: populate ctx->aneClient for runtime models and test
  doEvaluateDirectWithModel on a seq-64 tail; if that also fails, the
  in-memory API caps spatial width at 32 and the raw-.hwx assembler route
  (ANE-CUSTOM-COMPILER-RESEARCH.md) is the only path to wider prefill.
- Lever 2 (per-lane state checkpoints) does NOT need width>32 and remains
  the higher-confidence next step.

## P5 - Custom ANE compiler access: the two hardware-unlocked levers
Context: reverse-engineered private-framework access exists (docs/
ANE-CUSTOM-COMPILER-RESEARCH.md): direct _ANEModel/_ANEClient loading,
captured Espresso net.plist dirs, parametric MIL builders, ALU mode
registers. Physics-grounded plan, ranked:

WALLS (measured, do not fight):
- Decode floor: 12.7 GB int4 weights streamed/token at ~55 GB/s ANE DRAM BW
  => 4.7 tok/s hard max; current 4.27 = 91% of wall. Only amortization
  (speculation) or fewer bytes move it.
- INT8 dual-lane ALU mode: irrelevant while weight-BW-bound; revisit only
  for prefill after width scaling saturates.

LEVER 1 - PREFILL x2+: recompile fused tails at width 64/128.
- build_tail_mil() (rindi_native_chain.cpp) is already fully parametric in
  seq; package weight payloads (___.bin/__s.bin per layer) are
  width-independent and reusable directly.
- Steps: (a) port build_tail_mil to the Python harness or dump MIL from C++;
  (b) repack per-layer weight blobs at our own offsets (o/gu/dn/ip + norms);
  (c) compile at seq=64 via q38_ane_engine; (d) A/B vs 2x width-32 evals:
  bit-tolerance + ms/chunk; (e) wire engine to load regenerated tails
  (compile_prepared pattern already exists for cores).
- Expected: pp1024 71 -> ~120-140 tok/s (fewer chunk passes, compute still
  ~20% of fp16 peak). Also frees us from the frozen shipped package.

LEVER 2 - DECODE/MTP: per-lane state-checkpoint recurrence.
- GDN recurrence MIL is already state-in/state-out (-> (y, s2)); MTP partial
  rounds cost a 230ms replay ONLY because intermediate states inside a
  multi-lane verify are not materialized.
- Fix: emit an unrolled variant whose MIL exposes s2 after EACH lane as
  extra outputs (MIL supports multi-output functions; harness compiles
  arbitrary MIL). Partial acceptance then restores snapshot[n_ok] by copy -
  O(1) instead of O(replay). Attention KV already truncates O(1).
- Expected: kills the 230ms penalty; depth-1 spec projects ~4.9-5.2 tok/s
  (>base), depth 2-4 with good acceptance up to ~6-8 tok/s.
- Memory: state snapshot = HK*V*2B = 1.5 MB/layer - keep only the CURRENT
  round's per-lane states (k <= 8 => <=12 MB transient).

ORDER: Lever 1 first (isolated, big prefill win, no decode risk); Lever 2
second (needs careful exactness validation vs sequential recurrence).
## P4g - MTP root-caused: replay-on-partial economics + two real fixes landed
- The probe was overriding RINDI_MTP_DEPTH (setenv "2" after env read) - all
  earlier depth sweeps were void. With the fix: depth 1/2/4/8 -> accepted/step
  1.60/1.78/2.00/2.00. Deeper drafts genuinely accept more.
- Phase attribution (instrumented): draft iteration ~0.7-4 ms (MTP core +
  int4 draft head are CHEAP); verify ~230 ms (lane-invariant, same as a plain
  step); replay-on-partial ~230 ms more. Full-accept rounds emit k+1 tokens
  for one verify => win; partial/miss rounds pay a second forward => lose.
  Net: 0.89-0.91x at depths 1-8. THIS ARCHITECTURE (stateful GDN/ANE layers,
  O(n) state rollback) cannot truncate state like KV-only stacks, so partials
  inherently cost a rebuild pass. The Python prototype's speedup came from
  KV-only rollback (O(1) trim) - not reproducible here without ANE-side
  state snapshots.
- EXACTNESS: base greedy decode now uses the GPU BF16 head
  (greedy_argmax -> argmax_over_hidden lanes=1) - the SAME head speculation
  verifies with. Plain rounds are now bit-identical to base (guard-ON run
  PASSES). Remaining divergence enters via partial/replay rounds: batched
  lane eval vs sequential eval differ in low-order fp16 bits, so near-tie
  tokens can flip after a replay. Bit-exact MTP requires lane-invariant
  numerics from the fused tails - open item.
- LANDED: base decode 4.27 tok/s (up from ~3.9, GPU-head sampling);
  probe depth override fixed; acceptance-EMA guard verified EXACT at parity
  (4.15 tok/s, 0.98x) - recommended config for greedy traffic until
  lane-invariant numerics land.

## P4f - APC prefix cache: IMPLEMENTED + MTP findings
- APC (exact-prefix prompt cache) now wired end-to-end. The TUI/metrics
  plumbing existed but no cache was ever implemented (server passed
  hardcoded `false, 0`).
  Design: after prefill, snapshot FULL state (attention KV rows + positions
  incl. Metal mirrors via snapshot_kv/restore_kv, GDN conv+recurrence,
  MTP draft pos, and the end-of-prefill hidden). Later request whose tokens
  extend the cached sequence restores state and prefills only the suffix;
  equal-length regenerations reuse everything (no suffix prefill; cached
  hidden feeds decode directly).
  OFF-BY-ONE that mattered: equal-length resume must NOT replay the final
  token - the snapshot is AFTER it (double-advances every recurrent state,
  outputs diverge). Fixed by caching last_hidden.
- VALIDATED (temperature 0): cold 9.8s vs cached 4.7s on a ~500-token prompt
  (2.9x TTFT at 16-token generations; pure-prefill saving is larger),
  outputs BIT-IDENTICAL across cold/cached runs, single-chunk and
  multi-chunk both exact. RINDI_DISABLE_APC=1 disables.
- KNOWN LIMIT: multi-turn growth does NOT hit when the previous prompt ended
  with generation-prime tokens - a grown conversation is never a strict
  token-prefix of the old prompt. Fixing needs segment-boundary snapshots
  (cache state before the prime), tracked as follow-up.
- MTP STATUS (separate pre-existing issue): accepted/step ~1.1-1.8,
  speedup 0.84-0.99x (net loss) ALREADY BEFORE today's kernel work (clean
  tree measured 0.90x this morning vs d17de73-era claims of 4.57 acc).
  RINDI_MTP_DEPTH has NO effect (identical steps across depths 1-4);
  deep drafts keep missing. Added an acceptance-EMA guard
  (RINDI_MTP_EMA_MIN, default OFF) that cools speculation for 7 rounds
  when EMA-accepted drafts fall below threshold; verified replay keeps
  exactness regardless. Root-causing the draft-quality regression is an
  open item - today's changes are ruled out (fails identically with
  RINDI_DISABLE_APC=1 and guard disabled).

## P4e - ANE transfer audit: what the Metal work does/doesn't give the ANE
- DIRECT: nothing - kernels do not cross processors. What transfers is
  measurement + strategy:
  1. BUDGETS (measured, pp1024 lanes=32): pure-ANE fused layer 6.9 ms
     (442 ms/pass); ANE attention/GDN cores alone 4.5 ms; ANE MLP share
     ~2.2 ms => ANE effective ~3.8 TFLOPS whole-layer, ~8.5 TFLOPS on MLP.
  2. STRATEGIC NEGATIVE RESULT: documented GPU MMA ceiling is 15.4 TFLOPS
     (bf16==fp16) and real-world GEMM kernels reach ~4-5. ANE already runs
     matrices at ~8.5 effective. Moving ANE matrix work to Metal can never
     win big; RINDI_ENABLE_METAL_TAIL stays OFF for prefill. Metal's role:
     glue, draft head, fallback.
  3. CPU-GLUE AUDIT FALSIFIED: the per-attention-layer scalar sigmoid gate
     loop costs only 0.79 ms/layer (12.7 ms/pass, 2.8%); vForce rewrite
     would save ~5 ms/pass (~1%). Not worth precision churn. (Benchmark:
     /tmp/gatebench.cpp pattern - scalar exp is ~4 ns warm.)
- CHUNK-WIDTH EXPERIMENT (RINDI_ANE_WIDTH): runtime plumbing added (chain
  surface, core compile width, kPrefillLanes = width-3). RESULT: width>32
  fails at first prompt batch ("prompt batch failed ... lanes=61") because
  the FUSED ANE tails are precompiled blobs in the .rindi package, frozen at
  32 columns. Widening prefill chunks requires an offline package rebuild,
  not a runtime change. Default behavior unchanged; MTP_EXACT=PASS after.
- REMAINING ANE LEVERS: (a) offline rebuild of .rindi tails at width 64+
  (the only route to materially >71 tok/s prefill); (b) APC prefix caching
  (currently 0%) for repeated prefixes; (c) decode is MTP-bound - recent
  accepted/step dropped vs earlier sessions; retune draft depth/threshold.

## P4d - hardware ceilings measured + ROWWISE simd kernel wired
- MMA rate probes (this machine, Apple M5 Max): simdgroup_multiply_accumulate
  saturates at ~15.4 TFLOPS (8 independent fragment chains/simdgroup;
  16 chains regresses to 14.1; 2 chains 7.6). bf16 == fp16 exactly (15.4 vs
  15.4) - there is NO bf16 advantage on this documented MSL path. Scalar fp32
  FMA peak ~11.5 TFLOPS. The rumored 70 TFLOPS is NOT reachable through
  simdgroup_matrix; it would require the neural-accelerator tensor path,
  which this SDK's metal_tensor/metal_cooperative_tensor headers do not yet
  expose as an MMA op. Practical planning ceiling: ~15 TFLOPS.
- ROOT CAUSE of "prefill lower than ANE": ALL transformer-tail projections
  are compile_chain_int4 => ROWWISE layout, so P4c's groupwise kernel never
  ran in-server. Built gemm_int4_rw_simd (rowwise: signed int4 nibbles,
  2/byte, per-row fp16 scale; same validated tile structure).
  GOTCHA: scale_sh[] preload needs its own barrier before cross-simdgroup
  readers; rowwise nibbles are SIGNED (q>=8 -> -=16), unlike groupwise.
- GATE (probes/test_metal_rw_simd.mm): bad=0 vs scalar ref on 5 shapes incl
  lanes=7; production shapes 1.7-3.3x faster than gemm_int4_rowwise_batched
  (gate/up 9.15->2.75 ms, down 4.73->1.59 ms @ lanes=32).
- IN-SERVER A/B pp1024/tg128 with RINDI_ENABLE_METAL_TAIL=1:
  batched 84.3s vs rw_simd 54.2s TTFT = 1.55x. MTP_EXACT=PASS; output sane.
- HONEST STATUS: even with fast kernels, Metal tails (19.8 tok/s) still lose
  to the pure-ANE fallback (71.5 tok/s) because each layer serializes ANE
  attention core eval then one Metal MLP submit+wait with zero overlap.
  RECOMMENDATION: run WITHOUT RINDI_ENABLE_METAL_TAIL until the tail path is
  restructured (overlap ANE/Metal via double-buffered passes, or move MLP to
  ANE too). The simd kernels are the necessary fast primitive for either.

## P4c - simdgroup MMA port: WIRED (validated, 2.5x over batch)
- gemm_int4_simd (runtime/metal_engine.m): MLX steel-gemm structure adapted to
  int4-groupwise. BM=32 x BN=32 x BK=64 tiles (BK == quant group width);
  128-thread threadgroups = 4 simdgroups; each simdgroup accumulates four
  8x8 fp32 fragments over rows [sg*8,+8) via simdgroup_multiply_accumulate.
  Per K-step the W block is dequantized to fp16 in threadgroup memory and
  activations staged alongside; output staged fp32 then bounds-check
  converted to fp16 C (fully general rows/lanes, zero-padded overhang).
- GOTCHA that cost a debug cycle: kernel threadgroup args MUST use the
  pointer + [[threadgroup(N)]] form WITH setThreadgroupMemoryLength at
  dispatch. Fixed-size array params silently give zeros otherwise.
- GOTCHA 2: the probe's self-contained preamble (includes + bf16_to_float)
  must NOT be pasted into kDefaultShadersSource - duplicate definition kills
  the WHOLE library => NULL defaultLibrary => every Metal dispatch silently
  no-ops (garbage output). Library compile must be asserted, not assumed.
- GATE RESULTS (probes/test_metal_simd_gemm.mm):
  correctness bad=0 vs scalar ref on 5 shapes incl. lanes 8/16/24/32,
  maxabs ~1-2e-3 = fp16-dequant rounding class (batch itself is ~1e-3);
  perf 34816x5120 lanes=32: 2.17 ms vs batch 5.43 ms = 2.50x (5.25 TFLOPS).
  NOT bit-exact vs scalar by construction (same tradeoff MLX makes).
- END-TO-END: MTP_EXACT=PASS with simd wired; generation coherent.
  Fallback toggle RINDI_METAL_GEMM_BATCH forces the old batch kernel.
- HONEST CAVEAT: in-server prefill measured ~21 tok/s (simd) vs ~19 tok/s
  (batch) at pp1024 - statistically equal; the tail path itself is currently
  far below its historical ~900 tok/s for BOTH kernels and both show
  "Batched tail unavailable"/fallback unless RINDI_ENABLE_METAL_TAIL=1.
  In-engine prefill parity + the tail regression are separate open items;
  the 2.5x kernel win is real and isolated by the micro harness.
- Cumulative GEMM: naive 15.98 ms -> batch 5.43 ms -> simd MMA 2.17 ms
  = 7.4x total; ~42% of MLX fp16 steel-gemm throughput on this shape.

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

## P6 - Lever 2 implementation spec (O(1) MTP partial rollback)
BLOCKER analysis (why replay is 230ms but shouldn't be):
- In forward_prompt_batch, ONLY the CORES are stateful (attention KV rows,
  GDN conv history_ + recurrence metal_state_). The tails (MLP o/gu/dn + ip
  projections) are STATELESS - they only compute.
- Verify batch already computed: hidden lanes (batch_hidden), final
  next_projection, and KV rows for every lane.
- Partial with n_ok accepted: hidden for fix = batch_hidden lane n_ok (free).
  Attention KV rows pre-fix are already correct; restore_snapshot already
  rewinds position. So the ONLY missing state is GDN conv history +
  recurrence state AFTER the accepted prefix (48 GDN layers).
PLAN:
1. forward_prompt_batch: keep a per-layer vector of the batched hidden
   (layer_hiddens[l] = hidden at layer l entry, lanes x hidden_dim) so the
   partial path can feed any layer's core directly.
2. Add forward_state_only(input, keep_len): per layer, do ONLY the core stage
   (attention core_step_batch from stored q/k/v slices OR GDN core_step from
   stored slices), skipping evaluate_tail_batch entirely. GDN core_step is
   self-contained (project() internally + conv + recurrence) - feeding it the
   accepted-prefix hidden lanes rebuilds conv_history_ + metal_state_ via the
   normal (cheap) kernels. Attention needs nothing (KV rows already correct).
3. Partial branch in generate(): restore_snapshot(snap) [O(1) copies],
   forward_state_only(accepted_prefix, n_ok+1) [no weight stream: ~48 x
   small kernels, tens of ms total], hidden_state = batch_hidden lane n_ok,
   cur = fix, emit. Replay of MLP weights eliminated: 230ms -> ~30-60ms.
4. Exactness: state-only path must produce identical conv/recurrence state
   as the full verify for the accepted prefix. Verify by rerunning
   MTP_EXACT=PASS + tracing.
EXPECTED: partial rounds drop from ~468ms to ~270ms; with acc~1.7 (36%
    partials) round avg ~300ms -> ~5.3 tok/s vs base 4.27 = +25%; with
   deeper depth and better draft quality (hot-streak), toward +50%.
RISK: low-medium. Contained to engine.cpp generate() partial branch + one
   new forward_state_only; verified by MTP_EXACT.

## P6 result (attempted, reverted - cost premise VALIDATED, correctness does not)
- Implemented forward_prompt_batch capture + rebuild_gdn_state_only + partial
  branch rewrite (RINDI_STATE_REBUILD=1). Result:
  * rebuild ENGAGES and costs ~22ms restore vs ~230ms legacy replay - cost
    premise is correct (the replay really is weight streaming).
  * BUT MTP_EXACT=FAIL and acceptance DROPS (1.78 -> 1.07): the state-rebuild
    does NOT reproduce the legacy replay's conv/recurrence state exactly. The
    48 GDN layers' conv history + recurrence metal_state end up different
    after my per-layer rebuild than after the single wide-batch verify that
    the legacy replay was undoing. Next round's verifies reject => fewer
    accepts => slower.
  * Pre-existing note: depth-2 default MTP ALREADY FAILS exactness on partial
    rounds (batch-vs-sequential drift) independent of P6.
- WHY the rebuild diverges (hypothesis, untested): conv_.evaluate() threads
  history_ left-to-right per LANE; calling it with keep=n_ok+1 lanes vs the
  verify's k+1 lanes may advance history_ differently OR the causal conv
  within-batch neighbor columns means a shorter batch sees different padding
  than a longer batch at the same lane index. The test_tail_sync lane-count
  invariance probe was written precisely for this and never concluded.
- REVERTED. NEXT STEP (not a blind guess): before any P6 correctness work,
  directly answer "is conv_.evaluate(lanes=k) state-identical to lanes=m for
  the first k lanes?" via the standalone test_tail_sync.cpp route. Only if
  yes is per-layer rebuild viable; if no, the O(1) rollback needs the
  per-lane-conv-eval approach (call conv.evaluate K times, snapshot history_
  after each) which is exact by construction at ~lanes cost.

## P7 - GDN state compositionality: PROVEN (unblocks P6 properly)
- probes/test_conv_compose.cpp compiles one real GDN layer (layer 1) and
  compares state after (a) one wide call (lanes=4) vs (b) chunked 3+1 vs
  (c) chunked 2+2, hashing conv history_ + recurrence state.
- RESULT: all three produce BYTE-IDENTICAL conv+recurrence state
  (conv_hash/rec_hash equal across wide and both chunkings). COMPOSITIONAL=YES.
- IMPLICATIONS:
  * MTP's batched-verify does NOT inherently diverge GDN state from
    sequential decode - the earlier "batch vs sequential drift" explanation
    for MTP_EXACT=FAIL is WRONG. The drift must come from elsewhere
    (attention numeric path, lm_head, or the tail fp16 staging).
  * P6's state-only rebuild is FUNDAMENTALLY SOUND: re-running the GDN core
    over the accepted prefix reproduces the exact state. My earlier P6
    failure (MTP_EXACT=FAIL + lower acceptance) was an IMPLEMENTATION BUG,
    not a wall.
- NEXT: re-instate P6 but make the rebuild FAITHFUL and complete:
  * do NOT hand-slice projections - capture per-layer np AND, for the
    forward core, reuse the verify's stored np with an injected tail-skip
    path OR re-run each layer's core (attention core_step_batch + GDN
    core_step/core_from_projected_view) exactly as forward_prompt_batch does,
    feeding stored np for layers >= 1.
  * include attention core_step_batch (needed for KV) and gate_core_batch.
  * verify bit-exact state + MTP_EXACT=PASS before declaring win.
