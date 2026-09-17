# Flash-Next → Core AI / Neural Engine

Target: serve **Qwen3.8-Flash-Next** (`qwen4_exp`) on M5 Max (h17, 128 GB)
GPU-free, toward AFM-like decode (60–120 tok/s). Work lives in
`ane-port/scripts/export_flashnext_coreai.py`. This is a **new architecture
vs Apple’s Qwen3 recipes** and vs the old DynBank / 400-submit MIL GEMM
server — do not copy either.

Machine measurements below are Sep 2026, macOS 27, `coreai-torch` 0.4.1,
venv `~/.rindi/venvs/coreai`.

## Guardrails

- Base BF16 `~/models/Qwen3.8-Flash-Next` is **read-only mmap**.
  Never write there. Artifacts go to `ane-port/artifacts/coreai/`.
- PLE n-gram (~51B, `ple_layer_ids: [2]`) stays **off** the `.aimodel`.
- 4-bit MLX / AWQ 2/3 are GPU/MLX artifacts, not the Core AI source. Convert
  from **base BF16**, then palettize with `coreai-opt` after fp16 verifies.
- `apple-neural-engine` lab notes are **not a ceiling**. Core AI Conv2d on
  this chip already beats the DynBank 2.7 tok/s path.

## Model (text)

48 layers, `hidden=2560`, `hc_count=4`. Pattern **12 × (3 GDN + 1 QSA)**.
MoE 512 experts, top-10 + 1 shared, `I=640`. GDN: 48 V / 16 K heads, d=128,
conv k=4. QSA: 24q / 2kv, `head_dim=256`, RoPE dim 64, budget 2048. Vocab
248320. MTP 1 layer (unused until GDN+QSA decode works).

## What already works (layer 0 GDN)

| Graph | ANE | vs PyTorch | Notes |
|---|---|---|---|
| Conv2d smoke S=32 | 0.27 ms | rel 0.002 | BC1S, last dim 32 |
| Body: in_proj + out_proj + pin-10 MoE | **1.82 ms** | **rel 0.002** | 214 MB fp16 |
| Isolated S=1 GDN (prepared q/k/v) | 0.77 ms | rel 0.0008 | 4D, last dim 128 |
| Split fused (front + host + GDN+MoE) | 4.63 ms | y rel **0.003** | two submits, wrong residual |
| **P1 split: mixers + scored MoE** | **4.94 ms compute** | layer vs numpy **0.009** | +8.2 ms host route+pack |
| **P2 QSA L3 max_S=32** | **1.11 ms** | vs PyTorch **0.0017** (GPU 4.16 ms / 0.0004) | dense GQA; 99.7 MB |
| Compact GDN tail | **0.64 ms** | 16-step PASS | 36 layers exported |
| **Connected GDN prod** | ~2 ms | one-shot **0.005**; 16-step numpy attn **0.0049** | 116 MB; tanh-SiLU; 36 layers |

P1 quality (layer 0, S=1 token in slot 0, Sep 11):

| Compare | rel |
|---|---|
| host mixers vs numpy `gated_residual` | **0.0000** (`hc_norm` is stored as w−1) |
| fp16 PyTorch GDN vs numpy `linear_attention` | 0.0083 |
| ANE GDN attn vs PyTorch / numpy | 0.0168 / 0.0180 (SSM **0.0013**) |
| fp16 scored MoE vs numpy `moe_layer` | **0.0003** |
| routed proxy vs numpy `decoder_layer` | **0.0024** |
| ANE 10240 residual vs numpy `decoder_layer` | **0.0092** |

Routing: host softmax 512 → top-10 + renormalized scores `(1, 10, 1, 32)`. ANE graph is scored SwiGLU with those 10 experts **baked** as Conv2d + shared. Weight-as-input paging works (GPU rel 0.0005) but is **~380 ms/token** from ~100 MB I/O — do not use for decode. `--weight-inputs` keeps that probe.

Breakdown (baked): mix 0.36 + front 1.12 + prep 1.05 + gdn 0.83 + mlp-mix 0.46 + moe 1.03 + recombine 0.09 = **4.94 ms**. Host route+pack **8.16 ms** (BF16 mmap of 10 experts) currently dominates; x48 ≈ 1.6 tok/s until pack is sped up. Compute-only x48 ≈ **4.2 tok/s**.

**Authoring rules learned here (not just Apple’s docs):**

- Last I/O dim ≥ 32. Conv-state last dim 3 → ANE abort (MPS −19).
- No `exp` / `softplus` / unrolled S=32 Python loop in the `.aimodel`.
- `F.silu` in a big fused graph poisons GDN (rel 0.09–0.55). Use the tanh
  identity `x/2 + (x/2)*tanh(x/2)` (connected one-shot attn rel **0.005**).
  Empty/tiny-h rel **0.48** was a **false alarm** (fp16 noise vs attn rms ~7e-4).
- Mixers 10240→320→10240 drift on ANE; host numpy `host_gated_residual`.
- Scalar `0.25` / Python `1.0` / `1e-6` as fp32 literals segment off ANE.
- 4-bit palettize of all Conv2d: 1.18 ms but rel 0.31 — skip until fp16
  stack is correct.
- `coreai-build` CLI is missing; Python `AIModel.load` +
  `ComputeUnitKind.neural_engine()` is the compile path.

Run:

```bash
~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py host_test
~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py split --reuse
~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py qsa --reuse
~/.rindi/venvs/coreai/bin/python probes/flashnext_connected_gdn.py --reuse --export-all
~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py generate --max-new 4 --prompt-ids 760
```

## Target decode loop (one token)

Host residual stream is BC1S `(1, 10240, 1, 32)` (4 hyper branches × 2560).

For each of 48 layers:

1. **Host** attn hyper mix → `(1, 2560, 1, 32)` mixed hidden.
2. If `linear_attention` (36 layers):
   - **Default when all 36 connected graphs exist:** one ANE submit
     (front + tanh-SiLU + L2 + decay + compact tail). Probe:
     `probes/flashnext_connected_gdn.py --reuse --export-all`.
   - **Fallback:** ANE **front** (no SiLU) → host SiLU/L2/decay → ANE
     **gdn-only**. `--host-front` / `FLASHNEXT_CONNECTED=0` keep this split.
   - Host: recombine onto 10240.
   - Host: MLP hyper mix → `(1, 2560, 1, 32)`.
   - ANE **MoE**: scored top-10 SwiGLU (experts baked or paged; scores are inputs).
   - Host: recombine onto 10240.
3. If `full_attention` (12 layers, ids 3,7,…,47):
   - Host: indexer / budget gather (or dense GQA while context ≤ 2048).
   - ANE **QSA**: per-head GQA, readonly KV, RoPE cos/sin as `(1, 32, 1, 32)`
     (rotary/2 freqs, last dim ≥ 32 — do not ship `(1, 64, 1, 1)`).
   - Host recombine, then the same MLP mix + ANE MoE graph as GDN layers.
4. Layer 2 only: **host PLE** add from mmap’d n-gram table (zeros if the
   table is absent — mlx-lm fallback). Never bake 51B into the `.aimodel`.

Then: final hyper mixer, `lm_head` (host or a dedicated ANE Conv2d), sample.

Weight paging: one compiled front + gdn-only + MoE + QSA shape, **swap
per-layer weights** rather than 48× resident graphs.

## Phases

### P0 — quality split (done)

- [x] Conv2d BC1S smoke on Neural Engine
- [x] Layer-0 body fp16, rel 0.002
- [x] Isolated S=1 GDN, rel 0.0008
- [x] Two-graph split, y rel 0.003
- [x] Numpy `HostPrep` with reused fp16 buffers
- [x] Numpy `host_gated_residual` / `host_recombine` (wired in P1)
- [x] `FlashNextQSADecode` authored **and exported** (layer 3, max_S=32)

### P1 — one full GDN layer, correct residual

- [x] Wire host mixers + recombine into `split` (input 10240, output 10240)
- [x] Attn mix → GDN-only → recombine → MLP mix → MoE graph → recombine
- [x] Compare vs `tools/flashnext_reference.py` `decoder_layer` (pin-10
      proxy vs real top-10 is rel 0.10; ANE vs that proxy is rel 0.01)
- [x] Real top-10 routing on host: softmax-512 → packed experts + scores
      `(1, 10, 1, 32)`. ANE scored MoE vs numpy decoder_layer **rel 0.009**.
      Do not page expert weights as graph inputs (~380 ms). Bake (or later
      swap compiled Conv2d blobs). Host pack is 8 ms — next speed target.

### P2 — QSA layer (done for max_S=32)

- [x] Export `FlashNextQSADecode` for layer 3, `max_S=32` (`flashnext_qsa_L3_s32.aimodel`)
- [x] Dense GQA (context ≤ indexer budget); indexer still host-side
- [x] Readonly KV I/O; host writes `new_k`/`new_v` slot 0 into the cache
- [x] RoPE as `(1, 32, 1, 32)` freqs (last dim 1 aborts); mask `-40000`
- [x] Wire mix → QSA → recombine → MLP mix → numpy MoE → recombine
      (fp16 QSA vs numpy `full_attention_layer` **rel 0.0003**; residual vs
      `decoder_layer` **rel 0.0003**)
- [ ] max_S=2048 when short-context greedy is solid

### P3 — 48-layer decode (first generate done on host numpy)

- [x] Host embeddings + final `hyper_connection_mixer` + host `lm_head`
- [x] All 48 layers via `flashnext_reference.decoder_layer` (GDN+QSA+MoE),
      PLE zeros. Greedy `"The"` (id **760**) → **220, 17, 15, 16** = `'The 201'`
      in 13.3 s (~0.30 tok/s) on host numpy. MLX 4-bit prefix is
      `220, 17, 15, 15` (`'The 2000…'`). First **3 tokens match**; 4th is
      digit `1` vs `0`. `lm_logits` is `(T, vocab)` — argmax the flat row.
- [x] **Per-layer baked graphs** (Core AI has no weight-swap API): `layers`
      exported 36 GDN front+gdn-only + 12 QSA. Generate loads all 48 on
      Neural Engine (24.9 s). Same greedy prefix `'The 201'`. **0.61 tok/s**.
      ANE GDN ~31 ms/layer wall includes numpy MoE.
- [x] **Packed host MoE generate** (no frozen ANE MoE): same greedy
      prefix `'The 201'`. First **0.84 tok/s** (numpy `moe_layer` was 0.61).
- [x] **GDN-sticky MoE bake is false** (8-token BF16 greedy `"The"` →
      `'The 2011–12'`). Adjacent-token Jaccard is **~0.27–0.44** on both
      GDN and QSA (no GDN advantage). Token-0’s ten experts hold only
      **~20% softmax mass** even on token 0 (diffuse 512-way router);
      mass on that set is **~0.13 by token 1**, **0/48 layers ≥ 0.85**.
      Do not bake a frozen top-10 for decode. ANE keeps GDN/QSA; MoE
      stays packed `I=640` on host/GPU.

### P4 — PLE n-gram (layer index 1 / `ple_layer_ids: [2]`)

- [x] **No** `ngram_index.json` in `~/models/Qwen3.8-Flash-Next`.
      128 `ngram_embedding.shard_*` tensors **do** live in the main safetensors
      (`shard_0` is `(2500012, 160)` ≈ 51B bf16 values / ~102 GB) but mlx-lm
      only wires `NGramTable` when the index JSON exists. Without it the
      lookup is **zeros** — match that; do not invent a table.
- [x] PLE is **not** a 48-layer GEMM. Hashed trigram ids → huge row lookup →
      `key_proj`/`value_proj` + gate into the residual **before** attn mix on
      that one layer, so the transformer does not memorize surface statistics.
- [x] Quantize the table separately (int8/mmap bf16 off-ANE). It does **not**
      license 2-bit `in_proj`/`out_proj`/GDN. Never bake 51B into `.aimodel`.
      Palettize the Core AI graphs only after greedy prefix is stable.

### P5 — speed (prefix `'The 201'` holds; tok/s still short of chat)

Measured `generate --max-new 4 --prompt-ids 760` (Sep 11):

| Cut | tok/s | token walls (s) | token-4 MoE | prefix |
|---|---|---|---|---|
| numpy `moe_layer` | 0.61 | — | — | `'The 201'` |
| packed CPU MoE, BF16 mmap | 0.84 | 1.48 → 0.74 | 470 | `'The 201'` |
| compact fp16 disk, on-demand | 0.866 | 1.73 → 0.61 | — | `'The 201'` |
| fp16 RAM hot + 2-submit GDN | **1.11 / 1.257** | 1.64 → 0.42 / 1.19 → 0.45 → 0.46 → 0.42 | **226 / 219** | `'The 201'` |
| connected GDN 1-submit + fp16 hot (copy all 10) | **1.305** | 1.02 → 0.58 → 0.37 → 0.34 | **206** (gather 120 / GEMV 81) | `'The 201'` |
| **connected + sticky packed slots** | **1.560** | **0.98 → 0.47 → 0.27 → 0.26** | **138** (gather **58** / GEMV 76) | `'The 201'` |
| mlx-lm 4-bit bank, fused C GEMV (`FLASHNEXT_MOE=q4gemv`) | 0.805 | 1.97 → 0.94 → 0.68 → 0.65 | **427** (GEMV 413) | `'The 201'` |
| q4gemv + `--host-front` (12 tok, opt-in) | 2.025 | — | — | `'The 201'` |
| hybrid 4-bit store → dequant new top-10 → packed GEMV (`FLASHNEXT_MOE=hybrid`) | 0.447 | 5.58 → 1.35 → 0.73 → 0.64 | **431** (gather 348 / GEMV 79 = 1.65 ms/layer) | `'The 201'` |

Connected GDN (L0, Sep 11 night): one-shot attn rel **0.005**; generate-start
**0.014** vs Torch / **0.012** vs numpy; 16-step worst vs Torch **0.0069**, vs
numpy attn **0.0049** / ssm **0.0062** / conv **0.0027**. The 0.48 “fail” on
empty/tiny-h is fp16 noise (attn rms ~7e-4) — **false alarm**. Generate uses
connected graphs when all 36 exist (1 submit/layer); incomplete sets and
`FLASHNEXT_CONNECTED=0` keep the 2-submit split. PLE stays zeros unless
`FLASHNEXT_PLE=1`. Do not default `q4gemv`. The 1.2 tok/s wall was **host
MoE gather** (fp16→fp32 `copyto` of all 10 packed experts/layer), not ANE
compute (~1–2 ms/graph; ANE+io is ~84 ms of 48 submits). Sticky packed
slots skip copyto for resident `(layer, eid)` — hot gather 120→58 ms,
**1.305 → 1.560 tok/s**. Remaining hot token-4: ANE I/O 84 + GEMV 76 +
new-expert copyto 58.

**mlx-lm 4-bit ≠ Core AI palettize.** Rel 0.31 was `coreai-opt` on GDN/QSA
Conv2d — leave those graphs fp16. **Default generate is connected GDN + the
fp16 hot expert store** when the 36 prod graphs are on disk
(`artifacts/experts_f16/` + in-process LRU; BF16 mmap on miss only). The 4-bit
`switch_mlp` bank (`runtime/expert_bank.py`, 60.4 GB packed, affine gs=64)
**fits** but first-touch dequant of 480 experts lost the 4-token race
(token-1 wall 5.58 s; token-4 gather 348 ms vs isolated ~76 ms because the
bank is cache-cold). Keep it **opt-in**: `FLASHNEXT_MOE=hybrid` (dequant →
packed GEMV) and `FLASHNEXT_MOE=q4gemv` (fused 4-bit C GEMV, 0.805 4-tok /
2.025 with `--host-front` over 12 tok). Shared expert is 8-bit. Router stays
BF16. No ngram touch, no write to the BF16 tree.

Pure MLX greedy is `220, 17, 15, 15`. ANE-fp16 attention + fp16 MoE kept
`220, 17, 15, 16` (first 3 match both; token 4 stayed BF16 `16`).
Isolated (1 layer, RAM packed): C dequant 10-expert **2.22 ms**, 7-new
**1.58 ms**, packed GEMV **1.56 ms/layer** (was 6.0 ms 4-bit SwiGLU).
Hybrid generate token-4 GEMV matches that (**1.65 ms/layer**); gather
does not. Do not default hybrid until dequant-first-touch is fixed.

- [x] Packed host SwiGLU + RAM expert LRU + cached `lm_head` / mixers
- [x] Compact hot-expert store (`artifacts/experts_f16/`, FN16 fp16)
- [x] **MLX 4-bit routed-expert RAM bank** (60.4 GB packed; opt-in)
- [x] **Default fp16 hot decode** (`FLASHNEXT_MOE=fp16`). Prefix `'The 201'`
      held; restore run **1.257 tok/s** (was 1.11; hybrid 0.447).
      Hybrid/q4gemv remain opt-in.
- [x] **Connected GDN 1-submit** (tanh-SiLU, 36×116 MB). Recurrent PASS vs
      numpy `linear_attention` (16-step attn 0.0049). Fresh generate **1.305
      tok/s**, prefix `'The 201'`. Default when all 36 assets exist; 2-submit
      split remains the fallback (`FLASHNEXT_CONNECTED=0`).
- [x] **Sticky packed MoE slots** (skip fp16→fp32 `copyto` when `(layer, eid)`
      already occupies a packed slot; remap scores). **1.560 tok/s**. Token-4
      gather 120→58 ms (slot_hit=268 / copy=212). Prefix held. The 400 ms
      wall was host gather, not ANE compute (~1–2 ms/graph).
- [ ] Hot token-4 ~260 ms is now ANE+io **84 ms** (48 submit/I/O) + MoE GEMV
      **76 ms** + remaining new-expert copyto **58 ms**. Mixers ~25, HostPrep
      0 (connected), lm_head ~11. Tens of tok/s still needs new-expert gather
      and ANE I/O — not “ANE is slow”.
- [ ] Palettize GDN/QSA Conv2d **only after** measuring (rel 0.31 last try)
- [ ] MTP draft layer if it pays
- [ ] Do **not** put `F.silu`/`exp`/`softplus` back in the GDN graph; tanh-SiLU
      is the fused form that matched (one-shot 0.005). Tiny-h 0.48 was a false alarm.
- [ ] Do **not** page routed expert weights as ANE activations
- [ ] Do **not** default MoE to MPS (20–45 ms/layer launch+sync; flipped `16`/`17`)
- [ ] Do **not** bake 48 frozen MoE graphs / DynBank / weight-as-input experts
- [ ] Do **not** default PLE (`FLASHNEXT_PLE=1` only) or `FLASHNEXT_MOE=q4gemv`

## Sep 12: disk, dtype, and the real per-layer floors

Two blockers this session were environment, not architecture. Both invalidate
earlier conclusions that were recorded as ANE limitations.

### 1. A full disk looked like "the ANE refuses to load"

Every `.aimodel` load failed with `Foundation._GenericObjCError error 0` while
freshly converted models still loaded and ran. Cause: the Data volume was at
**100% (11 GB free of 1.8 TB)**, `~/Library/Caches/com.apple.e5rt.e5bundlecache`
had been purged to 0 B, and `ANECompilerService` could not complete a compile
(`ANECCreatePrepareInfoFromMLIR: no constants source available`, then
`Could not handle common config for oplayer`).

Freed 30 GB (`artifacts/experts_f16`, the superseded fp16 hot store) and 39 GB
(`~/Library/Caches/coreai-cache`). After that every model loads in <1 s.

**This retires "QSA L7/L11/L15 only load with `.with_debug(enabled=True)`".**
Debug specialization was never the fix; disk headroom was. It also explains the
7.77 s cold token and the multi-second-per-layer stalls seen while the expert
bank was paging against a full disk and 5 GB of 6 GB swap in use.

Check `df -h /System/Volumes/Data` before believing any ANE load failure.

### 2. fp32 activations against bf16 scales cost 2.6x in `gather_qmm`

`runtime/flashnext_mlx_moe.py` fed fp32 activations to a bank whose scales and
biases are bf16. That combination leaves the fast kernel:

| activations / scales | one gate_proj |
|---|---|
| fp32 / bf16 (what shipped) | 0.581 ms |
| bf16 / bf16 | 0.225 ms |
| fp16 / fp16 | 0.248 ms |
| fp32 / fp32 | 0.226 ms |

bf16 activations are fast but cost **rel 0.0117** against the fp32 CPU
reference — too coarse. fp16 gives **rel 0.00033** at the same speed, so the
resident MoE now defaults to fp16 (`FLASHNEXT_MOE_DTYPE`).

`mx.set_cache_limit(256 MB)` also cost 35%: 2.00 ms/layer at 256 MB vs
1.30 ms at 1 GB, because every routed call re-mapped its buffers. Now
`FLASHNEXT_MLX_CACHE_MB`, default 2048.

### 3. Measured floors (probes/flashnext_boundary_floor.py, ane_weight_bandwidth.py)

ANE, one `pure_step` GDN layer, persistent surfaces, no host work between:

| | |
|---|---|
| submit + feeds + take | 2.07 ms |
| submit only | 1.81 ms |
| host feeds + take | 0.17 ms |
| submit at concurrency 8 | 1.43 ms |

So the ANE side is **execution-bound, not dispatch-bound**: issuing eight
independent layers at once recovers only 21%. Host buffer handling is noise.

ANE weight-streaming ceiling, single Conv2d, seq=32:

| weights | eval | GB/s |
|---|---|---|
| 16 MB | 0.310 ms | 50 |
| 64 MB | 0.703 ms | 89 |
| 136 MB | 0.777 ms | 171 |
| 272 MB | 2.330 ms | 114 |

A `pure_step` layer carries ~71 M fp16 parameters (in_proj 42 M, out_proj
16 M, two mixers 13 M) = **136 MB per layer per token**, 4.9 GB across 36 GDN
layers. At 1.81 ms that is 75 GB/s, against a ~170 GB/s ceiling at the same
footprint. **~1 ms per layer is graph overhead, ~0.8 ms is weight streaming.**

GPU MoE, 48 routed layers, fp16, one resident layer bank:

| | per layer | per token |
|---|---|---|
| eval per layer (what decode needs) | 0.404 ms | 19.4 ms |
| 48 independent, one eval | 0.078 ms | 3.7 ms |
| 48 chained serially, one eval | 0.086 ms | 4.1 ms |

The GPU can do the whole routed MoE in ~4 ms/token. Decode cannot reach that
while the ANE consumes each layer's output: a minimal numpy → GPU → eval →
numpy round trip is **0.45 ms**, so 48 alternations cost ~22 ms/token in pure
latency before any MoE work.

### 4. The PLE divergence was not a bug

`FLASHNEXT_PLE=1` produced ` passage states:` where the recorded MLX greedy is
`[220, 17, 15, 15]` (`The 201…`). That reference was generated with the n-gram
table **off**: `~/models/Qwen3.8-Flash-Next/ngram_index.json` does not
exist, so `set_ngram_lookup` is never called and `NGramEmbedding` returns zeros.
Zero-PLE runs matched it because both sides had PLE disabled. A PLE-on run must
be compared against MLX with a lookup installed, not against this prefix. The
40–45 tok/s MLX baseline is likewise a zeros-PLE number.

### Where the budget actually is

Alternating ANE attention with GPU MoE has a floor near 6–8 tok/s:
36 GDN x 1.8 ms + 12 QSA + 48 x 0.45 ms of GPU round trip. Beating that needs
the per-layer ANE cost down (graph overhead first, then compressed weights),
not another MoE placement.

## Sep 12 (later): one token per submit was the ceiling, not the ANE

The 6–8 tok/s estimate above applies only to a decoder that submits one token
at a time. Every cost on this path is weight-streaming, so the fix is to put
more real tokens in each submit. Two measurements set this up.

**seq=32 is the fast shape, not padding waste.** The same 136 MB projection:

| tokens | eval |
|---|---|
| 1 | 2.198 ms |
| 2 | 2.046 ms |
| 8 | 2.344 ms |
| 16 | 2.686 ms |
| 32 | **1.060 ms** |

One token costs *twice* what 32 cost. The 32 slots are the ANE's efficient
tiling and 31 of them currently carry padding. Nothing was being wasted by
padding; the waste is leaving those slots empty.

**The recurrence amortizes.** `probes/flashnext_multistep_tail.py`, CompactTail
unrolled k times, exact (0.00000) against k chained single steps:

| k | tail | per token |
|---|---|---|
| 1 | 0.620 ms | 0.620 ms |
| 2 | 0.802 ms | 0.401 ms |
| 4 | 0.990 ms | 0.247 ms |
| 8 | 1.425 ms | 0.178 ms |

### Multi-token layer: `probes/flashnext_multitoken_step.py`

A whole `pure_step` layer consuming k real tokens per submit. Only two things
needed changing, because the mixers, recombine and out_proj were already
per-slot:

1. the 4-tap depthwise conv becomes a causal shift across slots — the 3-deep
   cache supplies only the tokens *before* the batch, and slot i takes taps
   from i-3..i. The shipped front applies the same cached columns to every
   slot, which is correct only for slot 0;
2. the GDN recurrence unrolls k times, threading one state.

Verified against k sequential single-token steps of the shipped path — mixed
output, recurrent state and conv cache all **rel 0.00000**:

| k | layer | per token | vs k=1 |
|---|---|---|---|
| 1 | 1.938 ms | 1.938 ms | 1.0x |
| 2 | 2.013 ms | 1.006 ms | 1.9x |
| 4 | 2.386 ms | 0.597 ms | 3.2x |
| 8 | 2.968 ms | 0.371 ms | 5.2x |
| 16 | 4.160 ms | 0.260 ms | **7.5x** |

The routed MoE amortizes the same way — same expert weights serve all k tokens
(`mx.expand_dims(x, (-2, -3))`, indices `(1, k, 10)`), 48 layers, one layer's
bank resident:

| k | MoE per token |
|---|---|
| 1 | 28.0 ms |
| 4 | 6.4 ms |
| 8 | 4.5 ms |
| 16 | 3.4 ms |

### What this projects to

At k=8, per token: GDN 36 x 2.968 / 8 = 13.4 ms, MoE ~4.5 ms (optimistic — one
layer's bank; the full 68 GB working set is worse), QSA and head a few ms more.
Roughly **30 ms/token, ~30 tok/s at full acceptance**.

Decode only reaches that with tokens to verify, so this needs the MTP drafter
(the checkpoint has one, and `qwen4_exp` already carries `rollback_state` and
`draft_snapshots` for exactly this). At a realistic 3–4 accepted of 8 it is
~2.5–3x today's 4.5 tok/s. **Prefill needs no drafter at all** — every token is
real, so prefill should go straight to the k=16/32 column.

### Decode after the fixes above

`FLASHNEXT_MOE=mlxresident FLASHNEXT_HEAD=mlx`, 0.22 s/token warm (4.5 tok/s),
from 0.29 s this morning. Output `The 2016–17 season marked a pivotal` —
**matches BF16 greedy exactly** (the MLX 4-bit prefix differs at token 4, which
is the known 4-bit digit).

| | this morning | now |
|---|---|---|
| ANE + I/O | 120 ms | 114 ms |
| MoE | 135 ms | 92 ms |
| lm_head | 13 ms | 2 ms |
| per token | 290 ms | 220 ms |

`FLASHNEXT_MOE=q4gemv` (CPU) lands at the same 82–112 ms as the GPU bank with
identical output, which is the tell that neither is compute-bound: both are
streaming 1.18 GB of expert weights for a single token.

### Do not

- Do not use `.with_debug(enabled=True)` to work around a load failure. It makes
  compilation enormously slower (4+ min on a layer that compiles in 0.9 s
  without it). The loads it appeared to fix were a full disk.
- Do not feed fp32 activations to a bf16-scaled quantized bank.
- QSA L7 compiles pathologically slowly from cold (>20 min) while the
  identically shaped L3/L11/L15 take about a second. Weights are unremarkable
  (no fp16 overflow, no zeros, same range). It caches once it finishes.

## Sep 12: calibration against the Swift MLX backend (Rindi-NativeMLX)

`~/.mlx128/Rindi-NativeMLX` already runs this checkpoint with its MTP head, so
it settles two things the ANE plan was guessing at.

**Measure long generations.** A 160-token run reports 34.9 tok/s; the same
prompt at 600 tokens reports **42.6 tok/s**, and a second warm turn 42.3. The
short run is still amortizing first-touch paging of expert weights — with 512
experts per layer and top-10 routing, a few hundred tokens do not make the
working set resident. Use >=600 generated tokens for any decode number, and
warm the model first the way `Qwen4ExpBenchmark` does.

`--session --repeat`, code prompt, 600 tokens, `Qwen3.8-Flash-Next-MLX-4bit`:

| | turn 1 | turn 2 (warm) |
|---|---|---|
| decode, MTP | **42.6 tok/s** | **42.3 tok/s** |
| prefill | 11.0 tok/s (cold) | 354.8 tok/s (8 reused) |
| acceptance | 52% of 602 | 50% of 626 |

So **the GPU reference is ~42 tok/s** and the gap from today's ANE 4.5 tok/s is
about 9x. Implied tokens per backbone pass is ~2.1 (600 emitted, ~313 drafts
confirmed, so ~287 passes), i.e. ~50 ms per backbone pass.

Plain greedy at the same 600 tokens is **32.6 tok/s**, so MTP is worth
**1.30x** on the GPU while delivering ~2.1 tokens per pass. The gap is the
drafter: each draft is `embed + one MTP layer + the full lm_head`
(`Qwen4Exp.swift` `mtpForward`), serialized because the head mutates its KV
cache in place. Against a ~31 ms GPU backbone pass that overhead is most of
the win.

**That is the asymmetry the ANE path exploits.** The drafter's cost is roughly
fixed and small; an ANE backbone pass costs ~220 ms. The same 2.1 tokens per
pass therefore converts close to its full value, where on the GPU it does not.

Projected ANE decode at k=3, from measured pieces: GDN 36 x 2.2 = 79 ms, QSA
12 x ~2 = 24 ms (multi-token QSA graph not built yet), MoE ~83 ms (the measured
k/1 ratio applied to the real 92 ms full-bank cost), mixers and head ~12 ms —
about 198 ms for 2.1 tokens, i.e. **~94 ms/token, ~10-11 tok/s**, from 4.5
today.

That is 2.4x, not 9x. Reaching the GPU's 42 tok/s needs either much higher
acceptance than 50% (a better or tree-structured drafter, since acceptance caps
useful k near 3) or an MoE that is not weight-streaming-bound per pass — it is
42% of the projected budget. **But throughput parity is not the goal.** At
10 tok/s the ANE wins on energy if it draws under about a quarter of the GPU's
power, which is plausible and remains unmeasured: `powermetrics` needs a
password on this machine, so no tokens/joule number in this repo is measured.

**This asymmetry is why speculation is worth more here than it is on the GPU.**
The drafter's cost is nearly fixed and small — one layer plus one lm_head, a
few ms — while an ANE backbone pass currently costs ~220 ms. The same 2.46
tokens per pass therefore converts to close to its full value:

At k=4, per backbone pass: GDN 36 x 2.386 = 86 ms, QSA ~24 ms, MoE ~84 ms
(scaling the measured k=4/k=1 ratio onto the real 92 ms full-bank cost), head
and mixers ~12 ms — about 200 ms for 2.46 tokens, i.e. **~81 ms/token, ~12
tok/s**, from 4.5 today. After that the MoE is again the largest single term.

### What to reuse rather than reinvent

`Qwen4ExpMTP.swift` / `Qwen4ExpSpeculative.swift` already solve the parts that
are independent of where the backbone runs:

- `Qwen4ExpRecurrentCache.beginVerification(draftCount:)` /
  `rewind(accepted:)` — snapshots recurrent state per draft position, since
  GDN and PLE state cannot be trimmed the way a KV cache can. Our ANE layer
  returns `new_ssm` and `new_conv` per submit, so the same snapshot-per-draft
  scheme applies directly.
- **Adaptive depth**: `target = min(drafts, max(1, round(recentAccepted) + 1))`
  over an EMA of accepted drafts. Chained drafts each cost a serialized round
  trip, so a fixed-depth chain wastes work on unpredictable text. Our k-token
  graph is exported per k, so adaptive depth means selecting among a small set
  of exported k, or exporting one k and feeding fewer real slots.
- Greedy acceptance: a draft is confirmed when the backbone predicts the same
  token; on rejection the sampled target token is emitted, so the pass is never
  wasted.

The drafter itself should stay on the GPU. It is one layer plus the lm_head,
it is already correct there, and it is small next to an ANE backbone pass.


## Sep 12: where the MoE time actually goes, and the shared expert

Power is settled — the ANE tops out at 7 W — so this is a throughput problem
only. No `powermetrics` needed.

**The 68 GB bank is not the problem.** Routed MoE cost is flat against resident
footprint: 0.26–0.38 ms/layer at 1.4 GB and at 34.0 GB alike (1, 2, 4, 8, 16,
24 layers resident). That is 12–18 ms/token for 48 layers, against the 92 ms
decode actually pays.

**The gap is GPU wake latency.** Inserting a gap before each call, standing in
for the ANE submit that separates them in decode:

| gap before call | per layer | per token (48) |
|---|---|---|
| none | 0.378 ms | 18.1 ms |
| 1 ms | 0.262 ms | 12.6 ms |
| 2 ms | 0.695 ms | 33.3 ms |
| 4 ms | 0.780 ms | 37.4 ms |

A GPU that just sat idle through a ~2 ms ANE submit costs ~0.4 ms extra to
restart, 48 times per token — about 22 ms. Host numpy conversions add only
~0.05–0.13 ms, far less than assumed. This cost is structural to alternating
two devices per layer; batching k tokens amortizes it, nothing else does.

**The dense shared expert should not be on the GPU.** It is 9.8 MB of fp16 and
pays that wake penalty like everything else — ~0.5 ms/layer, ~24 ms/token.
Folded into the layer graph instead (`SharedExpert` in
`probes/flashnext_multitoken_step.py`) it costs **+0.08 ms/layer**:

| | without | with shared |
|---|---|---|
| k=1 | 1.938 ms | 2.020 ms |
| k=4 | 2.386 ms | 2.452 ms |

rel 0.00064 against the fp32 reference, and the recurrence stays exact
(0.00000). Only the *routed* experts need the GPU; the always-on part does not.
The 12 QSA layers need the same treatment.

### Budget after this, k=3, ~2.1 tokens/pass

| | per pass |
|---|---|
| GDN, 36 x 2.28 | 82 ms |
| QSA, 12 x ~2 | 24 ms |
| routed MoE (18 compute + 22 wake) | 40 ms |
| mixers, head | 12 ms |
| **total** | **158 ms → ~75 ms/token, ~13 tok/s** |

GDN is then 52% of the budget and sits at 2.28 ms against a 1.06 ms
weight-streaming floor for 136 MB. Compressed ANE weights are the remaining
untested lever worth ~18 ms/pass.

**Compressed weights do not work through Core AI — but they do through MIL,
and they are worth ~1.8x at decode width.** `coreai_opt` int8 quantization at
the real projection shape makes the ANE compiler emit `Compiler internal error:
Codegen Error: Failed to HandleANELayer` and times at 1.009 ms against fp16's
1.060 ms. The 4-bit palettize note (1.18 vs 1.11 ms) says the same. **That is a
property of the Core AI quantizer path, not of the hardware.**

`probes/ane_w8a8_projection.py` (private MIL, `constexpr_blockwise_shift_scale`
per-channel weights) re-run at decode widths rather than the 128/256/512 in
W8A8-PROJECTIONS.md:

| family | S | fp16 | w8a16 | w8a8 |
|---|---:|---:|---:|---:|
| gdn pair 2560<->16384 | 32 | 4.341 ms | **1.86x** | 1.84x |
| gdn pair | 64 | 6.539 ms | 1.89x | 2.81x |
| gdn pair | 128 | 10.877 ms | 1.02x | 1.66x |
| square 2560 | 32 | 0.788 ms | **1.76x** | 1.70x |
| square 2560 | 64 | 0.778 ms | 1.69x | 1.71x |
| square 2560 | 128 | 0.796 ms | 1.00x | 1.02x |
| expert 2560<->1280 | 32 | 0.430 ms | **1.49x** | 1.57x |
| expert | 128 | 0.464 ms | 1.04x | 1.01x |

Two corrections to W8A8-PROJECTIONS.md's reading:

1. **"Decode will see none of this" is wrong.** That doc's S=128 rows show no
   w8a16 gain because at S=128 the chain is compute-bound (16.9 TFLOPS, near
   the fp16 peak). At S=32 a `pure_step` layer runs ~4.3 TFLOPS and is
   weight-bandwidth-bound, which is precisely where halving weight bytes pays.
   The gain is largest at exactly the widths decode uses and disappears at 128.
2. **At decode width the win is the weights, not the activations.** w8a16 and
   w8a8 are within noise at S=32 (1.86x vs 1.84x, 1.76x vs 1.70x). Activation
   bytes are negligible against 136 MB of weights at 32 slots, so the calibrated
   activation scales — the doc's main accuracy risk — are not needed for decode.
   Per-channel int8 weights alone measured **4.7e-4** against fp32, better than
   fp16's 2.3e-3.

**Precondition: the layer graphs must move from Core AI `.aimodel` to the
private MIL engine** (`runtime/q38_ane_engine.py`), since Core AI will not hand
the ANE compressed weights. The blob container bug is fixed and
`tests/test_blob_pack.py` guards it, so the engine is ready.

### The hard floor, and the only lever that reaches 40 tok/s

With compression off the table, one backbone pass cannot go below roughly:

| | per pass |
|---|---|
| GDN projections, 36 x 1.06 (streaming floor) | 38 ms |
| GDN recurrence at k=3, 36 x ~0.9 (latency, not bytes) | 32 ms |
| QSA, 12 x ~1.1 | 13 ms |
| routed MoE, 18 compute + 22 wake | 40 ms |
| head, mixers | 12 ms |
| **floor** | **~135 ms** |

With int8 weights that floor becomes:

| | per pass |
|---|---|
| GDN projections, 1.8x on 38 ms | 21 ms |
| GDN recurrence at k=3 (latency, unaffected) | 32 ms |
| QSA | 8 ms |
| routed MoE, 18 compute + 22 wake | 40 ms |
| head, mixers | 12 ms |
| **floor** | **~113 ms** |

The pass cost is nearly flat in k (1.938 ms at k=1, 4.160 ms at k=16 for the
whole layer), so **tokens per pass is the remaining multiplier**:

| tokens/pass | ms/token | tok/s |
|---|---|---|
| 2.1 (today's MTP acceptance) | 54 | 18 |
| 4 | 28 | 35 |
| 4.5 | 25 | **40** |
| 6 | 19 | 53 |

This is why **tree-structured speculation belongs on the ANE and not on the
GPU**. Verifying 16 candidate tokens costs the GPU real time, which is why MTP
only buys 1.30x there; on the ANE a k=16 submit costs 4.16 ms against 1.94 ms
for k=1. Widening the draft into a tree of candidates — several branches
verified in one submit, accepting the longest matching path — is close to free
here and is the route from 15 to 40 tok/s. Linear chained drafting at 50%
acceptance is not.

At 7 W, 15 tok/s is already ~1.9 tokens/J against ~1.2 for the GPU at 42 tok/s
and ~35 W, so the energy goal is met well before the throughput goal.


## Sep 12: Core AI vs the private MIL engine, and what order to do things in

**Core AI cannot hand the ANE compressed weights, and the API offers no way
around it.**

| Core AI path | result |
|---|---|
| `coreai_opt` int8, per-channel | `Codegen Error: Failed to HandleANELayer` |
| `coreai_opt` int8, per-tensor | same failure; falls back to fp16 speed (0.896 ms at 136 MB) |
| 4-bit palettize | compiles, ANE-resident, **no bandwidth gain** (1.18 vs 1.11 ms at 96 MB) |

A palettized conv shows ANE residency with no validation errors but carries an
extra ANE op next to the fp16 control, consistent with a dequantize node
materializing fp16 before the convolution. Both int8 spellings fail inside the
ANE compiler's own codegen, which is not reachable from the Python API. Only
the private MIL path (`constexpr_blockwise_shift_scale`) delivers the 1.8x.

**Splitting is not an option.** Projections in MIL plus recurrence in Core AI
means 2 submits per layer, 72 per pass; at ~0.2-0.4 ms of extra submit overhead
each that is 7-14 ms against the 17 ms the compression saves. Whichever engine
wins, one program per layer.

### Do the drafting work first, the port second

The two levers are not equal, and the cheaper one is worth more:

| | tokens/pass | ms/pass | ms/token | tok/s |
|---|---|---|---|---|
| today | 2.1 | 220 (measured) | 105 | 4.5 |
| + multi-token layer + shared expert on ANE | 2.1 | 135 | 64 | 15 |
| + wider/tree drafting (no port) | 4.5 | 135 | 30 | **33** |
| + MIL int8 projections | 4.5 | 113 | 25 | **40** |

Tree drafting is worth 15 -> 33 tok/s and needs no engine port; the MIL port is
worth a further ~20%. Sequencing matters because the port carries real risk:
the MIL notes already record crashes for `exp`/`softplus` in-graph and for a 2D
`(rows, 128)` recurrence, and the GDN decay term is
`pow(sigmoid(-(a + dt)), gamma)` — `pow` with a const exponent is unverified in
MIL, and the obvious rewrite goes through `exp`, which is one of the known
crashes.

So: build one full layer in MIL with int8 weights and diff it against the
multi-token Core AI layer, which is numerically exact today and makes a sound
reference. That surfaces the `pow`/`tanh`/recurrence questions on one layer
instead of 36, and it is the only part of the port with genuine unknowns —
the projection chain itself is already proven by
`probes/ane_w8a8_projection.py`.


## Sep 12: multi-token QSA is free

`probes/flashnext_multitoken_qsa.py`. **No graph change was needed.**
`FlashNextQSADecode` already computes q/k/v for all 32 slots and takes a
per-query-slot mask, so only the host feed had to stop being single-token:

* RoPE gets one position per slot (`off + i`), not one position broadcast to
  every slot;
* the mask lets query slot i see the cache `0..off-1` **plus batch slots
  `0..i`** — causal within the batch. The shipped `_qsa_feeds_into` opens only
  query slot 0 (`[..., :1]`), which is why k>1 was impossible before.

Verified against k sequential single-token calls with the KV cache advanced
between them:

| k | out vs k single calls | new_k | ANE | per token |
|---|---|---|---|---|
| 1 | 0.00000 | 0.00000 | 1.073 ms | 1.073 ms |
| 2 | 0.00000 | 0.00000 | 1.063 ms | 0.531 ms |
| 4 | 0.00000 | 0.00000 | 1.046 ms | 0.261 ms |
| 8 | 0.00005 | 0.00000 | 1.065 ms | 0.133 ms |

Flat in k, because the padded slots were already being computed. Both halves of
the backbone now take k tokens and both are verified exact:
`flashnext_multitoken_step.py` (GDN, needed a causal conv shift and an unrolled
recurrence) and this one.


## Sep 12: MIL op support for a GDN layer — green light, two adaptations

`probes/ane_mil_gdn_ops.py`, compile-only, smallest useful shapes. Run before
porting 36 layers, because the MIL notes record crashes for `exp`/`softplus`
and for a 2D `(rows, 128)` recurrence.

| construct | result |
|---|---|
| `tanh`, `sigmoid`, tanh-SiLU chain | OK |
| `pow`, const **scalar** exponent | OK |
| `exp` | **OK** — compiles, despite the recorded-crash note |
| `reduce_sum` / `reduce_mean`, last axis, 4D | OK |
| `transpose` 4D, `slice_by_index`, `concat` | OK |
| `mul(x, x)` on `(1,48,128,128)` | OK |
| `mul(state, slice)` broadcasting over dim 2 | **OK** — the recurrence's shape |
| `rsqrt` | **FAIL** InvalidMILProgram |
| `pow(x, -0.5)` as rsqrt | **OK** |
| rank-4 const tensor (inline **or** BLOBFILE) | **FAIL** InvalidMILProgram |

Two adaptations, no blocker:

1. **`rsqrt` is unsupported; `pow(x, -0.5)` is the substitute.** Both L2 norms
   and the RMS norms in the GDN tail go through it.
2. **Rank-4 const tensors are rejected**, at `(1,48,1,128)`, `(1,48,1,1)` and
   `(1,48,128,128)`, inline and blob-backed alike. `mul` and the broadcast are
   fine — it is specifically a const of that rank. Per-head parameters (`gamma`
   from `A_log`, `dt_bias`) must arrive as runtime inputs instead, which the
   compact-tail design already does: it ships six per-head param rows per
   token. Conv weights are unaffected (`[out, in, 1, 1]` blob consts are what
   `ane_w8a8_projection.py` already compiles).

The risky part of the port was the decay term `pow(sigmoid(-(a + dt)), gamma)`.
`pow` with a const scalar exponent compiles, and `gamma` moves to an input row,
so that term is reachable. Nothing found so far requires abandoning the port.


## Sep 12: k=16 exported for all 36 GDN layers, ANE half measured

`probes/flashnext_multitoken_step.py --steps 16 --shared --export-all` writes
`flashnext_multitoken_step_k16_L{i}.aimodel` for all 36 GDN layers (145 MB
each, 5.2 GB total) with the shared expert folded in. First load compiles in
~3 s per layer; afterwards all 36 load in **0.4 s** total.

`probes/flashnext_prefill_ane_pass.py` then runs one 16-token pass through the
whole stack — 36 graphs resident, per-layer recurrent surfaces ping-ponged, QSA
on the multi-token mask and per-slot RoPE. MoE excluded, so this is the ANE
contribution only:

| | measured | per layer | projected from L0 |
|---|---|---|---|
| 36 GDN, k=16 | 154.1 ms | 4.281 ms | 4.160 ms |
| 12 QSA | 14.0 ms | 1.165 ms | 1.073 ms |
| **ANE per pass** | **168.1 ms / 16 tokens** | | |

**10.5 ms/token against 114 ms today — 10.8x on the ANE half**, and the
single-layer projection holds across the real stack (within 3%).

Adding batched MoE (1.138 ms/layer at k=16, 55 ms for 48) and ~12 ms of head
and mixers puts a prefill pass near 235 ms for 16 tokens, ~15 ms/token, against
the 220 ms/token the current **serial prefix prefill** pays. That is the next
thing to wire: `stage_generate` feeds prompt tokens through the decode loop one
at a time (`prefill_steps = len(prompt_ids) - 1`), which is where the 15x sits.

### Blocking limit: QSA is exported at max_S=32

The QSA graphs, their KV cache and their mask are all sized `max_S=32`, so the
whole context is capped at 32 tokens no matter how many tokens a pass carries.
The model's own budget is 2048. Real prompts need that raised, and it is an
export change that should land **before** the prefill wiring, not after.


## Sep 12: the max_S=32 cap is gone, and all three k-token pieces are ready

**QSA context was an export shape, not a code limit.** `max_s` is never read in
`FlashNextQSADecode.forward` — the widths come from the example inputs — so
raising it is an export parameter. Cost of 64x the context, k=16:

| max_S | ANE/layer | vs sequential calls |
|---|---|---|
| 32 | 1.088 ms | 0.00009 |
| 128 | 1.097 ms | 0.00009 |
| 512 | 1.158 ms | 0.00006 |
| 1024 | 1.262 ms | 0.00006 |
| **2048** (model budget) | **1.444 ms** | 0.00009 |

All 12 QSA layers are exported at `max_S=2048`
(`flashnext_multitoken_qsa_L{i}_m2048.aimodel`). Full stack, 16 tokens:

| | max_S=32 | max_S=2048 |
|---|---|---|
| 36 GDN | 154.1 ms | 157.7 ms |
| 12 QSA | 14.0 ms | 19.9 ms |
| per token | 10.51 ms | **11.10 ms** |

Full context costs 5.6% of the pass. The cap that would have made a fast
prefill useless is removed.

### Three pieces, all verified, all exported

| piece | verified against | error |
|---|---|---|
| multi-token GDN layer, k=16, shared expert folded in, 36 layers | k sequential single-token steps | 0.00000 (0.00064 shared) |
| multi-token QSA, max_S=2048, 12 layers | k sequential single-token calls | 0.00009 |
| `ResidentMoe.routed_multi`, k tokens in one submit | k separate `routed()` calls | **0** (bit-exact) |

### What is left: the wiring

`stage_generate` is single-token from the embedding down
(`real_embedding_hidden(loader, [tid])`, `[:, :1, :]` slices throughout), and
its prompt handling is a **serial prefix prefill** — `prefill_steps =
len(prompt_ids) - 1` iterations of the decode loop at ~220 ms each. Replacing
that with chunked k-token passes is the remaining work:

1. load `multitoken_step_k16_L*` and `multitoken_qsa_L*_m2048` when a prefill
   k is requested, alongside the existing single-token graphs;
2. chunk prompt tokens through them, `routed_multi` for the MoE, k slots
   written into the QSA KV cache per pass;
3. hand the recurrent state, conv cache and KV cache to the existing decode
   loop unchanged — the multi-token and single-token graphs share those
   formats, which is what makes the handoff safe.

`real_embedding_hidden` already takes a list, and `_route` already takes
`(-1, H)`, so neither needs changing. Expect ~15 ms/token prefill against
220 ms/token today.


## Sep 12: 256k context needs the indexer, not bigger graphs

`max_position_embeddings` is **262144** and `indexer_budget` is **2048**. The
model never attends to more than 2048 keys at any context length: past the
budget the QSA indexer pools keys into blocks of `compress_ratio` 4, scores the
blocks, and gathers the top `2048/4 = 512` blocks plus the ragged tail.

**So the exported `max_S=2048` QSA graphs are already correct for 256k.** What
256k needs is a host KV cache that grows (today `AttnCache.empty(max_len=256)`)
and the selector. Memory at full context: ~6.4 GB of fp16 KV across 12 layers
plus ~200 MB of indexer block keys.

`runtime/flashnext_indexer.py` is that selector, ported from the **production
Swift backend** (`Rindi-NativeMLX` `Qwen4Exp.swift` `updateAndSelect`), not
from mlx_lm. Verify against the Swift app, not the Python port — it is what
ships, and it is ahead in four ways that all bite once speculation and chunked
prefill are in play:

| Swift production | mlx_lm / a naive port |
|---|---|
| `coversPrefix` disables selection when the index has a hole | none — a gap silently points at the **wrong tokens** |
| retains 192 raw keys so a rollback into mid-block can rebuild it | keeps only the ragged tail; a rewind destroys blocks that can never be rebuilt |
| block count `min(total/ratio, heldBlocks)`, from the array being partitioned | derived from the offset; the two can disagree and abort inside argpartition |
| `L > 16` skips selection, uses fused causal attention | always selects |

The retained-key window is the one that would have bitten hardest: speculative
rejection is exactly a rewind into the middle of a block.

**`L > 16` makes chunked prefill work at any prompt length.** A k=16 chunk is
not `> 16`, so it takes the selection path. Under the budget the selector
returns `None` and dense attention over <=2048 keys covers it; over the budget
it returns exactly 2048 indices. Either way the exported `max_S=2048` graphs
fit, so **a 256k prompt needs no new graph** — only the growable host KV cache.

The port still agrees exactly with mlx_lm on the linear decode path where the
two overlap. `probes/flashnext_indexer_check.py` drives both implementations
from the same layer-3 weights and token stream:

| | ours | mlx_lm | agreement |
|---|---|---|---|
| after 2100 single-token steps | 2048 | 2048 | **EXACT** |
| decode step at offset 2100 | 2049 | 2049 | **EXACT** |
| offset 2101 / 2102 / 2103 | 2050 / 2051 / 2048 | same | **EXACT** |

Same token indices, across the budget boundary and after it.

**mlx_lm's own L>1 selection path has never run.** `update_and_select` calls
`mx.unique` for the multi-token branch and this MLX build has no such symbol,
so any prompt longer than the budget raises `AttributeError` there. Our port
handles L>1 with `np.unique`; the check above therefore drives both one token
at a time, which is the path that works on their side.

### 8-bit KV cache

Viable, not on the critical path. 6.4 GB fp16 vs 3.2 GB int8 at 256k, and
memory is not the constraint on a 137 GB machine. The ANE graph cannot take
int8 at the function signature (W8A8-PROJECTIONS.md established that spelling
is rejected), so gathered keys need a dequantize before the call regardless.
Worth it for concurrent long contexts or to halve indexer scoring bandwidth.
Its quality cost is unmeasured here — do not enable it blind.


## Sep 12: chunked prefill is wired and produces identical output

`FLASHNEXT_PREFILL_K=16` in `stage_generate`. Same 17-token prompt, 6 new
tokens, chunked prefill vs the serial prefix path:

| | serial | chunked k=16 |
|---|---|---|
| 16 prefill tokens | 3.553 s | **0.775 s** |
| per token | 222 ms | **48.4 ms** |
| generated ids | [5006, 279, 3098, 303, 220, 16] | **identical** |

4.6x, with the continuation unchanged — which is the real check, since the
chunked path has to leave the recurrent state, conv cache and KV cache in
exactly the state serial decode would have produced.

**48.4 ms/token is not steady state.** That run had a single chunk, so it
carries every first-touch cost; the ANE half alone measures 11.1 ms/token.
A longer prompt is needed to separate the two, and that is currently blocked
(see below).

### Two things this surfaced

**A short final chunk corrupts the recurrence.** The k-step graph is unrolled
to exactly k steps, so a chunk of n < k still runs k of them and advances the
state over zero-padding. The first wired run did this — a 4-token prefill
against k=16 — and produced fluent but wrong output (`980年， 加拿` instead of
the expected continuation), which is the failure mode to watch for: plausible
text, silently wrong state. `prefill_chunked` now consumes only whole chunks
and returns the count; the remainder stays serial.

**The QSA indexer weights were being freed.** The loader drops `self_attn.*`
once the ANE asset owns those weights, which also dropped
`self_attn.indexer.*`. Those are host-side — chunked prefill selects keys with
them — so the drop now keeps `.indexer.` keys, about 6.5 MB per QSA layer.

### The 32-token context cap is lifted

Decode now drives the **same 2048-wide multi-token QSA graphs with one slot**,
rather than the single-token graphs baked at `max_S=32`. No new export: the
graphs already existed and were verified. `qsa_multi_attn(i, mixed, n)` is
shared — prefill calls it with n=k, decode with n=1 — so the indexer state and
KV buffers carry across the prefill/decode boundary instead of being rebuilt.
The guard now reads `FLASHNEXT_PREFILL_MAX_S` instead of `seq`.

Same 17-token prompt, decode on the wide graphs: **identical ids again**
([5006, 279, 3098, 303, 220, 16]).

A 65-token prompt — previously rejected outright — now runs:

| prefill | per token | vs serial |
|---|---|---|
| serial (old path) | 222 ms | 1.0x |
| chunked k=16, 1 chunk (17-token prompt) | 48.4 ms | 4.6x |
| chunked k=16, 4 chunks (65-token prompt) | **35.2 ms** | **6.3x** |
| ANE half alone | 11.1 ms | |

Per-token cost keeps falling as first-touch costs amortize over more chunks.

### Where the remaining prefill time goes

A 16-token chunk measures ~563 ms against an ANE pass of 177.6 ms, so ~385 ms
is host work — about 8 ms per layer. The routed MoE is 1.1 ms of that and the
router ~1 ms. The rest is the host mixers, and they are concentrated in the 12
QSA layers: each runs **two** `host_gated_residual_cached` calls plus two
`host_recombine` over 16 slots of a 10240-wide stream, in numpy. The 36 GDN
layers do none of this — `pure_step` already folds their mixers into the ANE
graph.

**That estimate was wrong, and folding the QSA mixers did not help.** Built and
exported anyway (`probes/flashnext_qsa_step.py`, 12 layers): attn mix + QSA +
recombine + mlp mix + shared expert in one graph, 1.995 ms vs 1.444 ms bare.
The host still has to run the attention mix, because the indexer selects keys
from it and selection must precede the ANE call — so the fold only removes one
recombine and one mixer pass. Measured end to end it is within noise:
35.2 ms/token bare, 33.1-37.7 folded across runs.

Instrumenting instead of guessing (`prefill_timers`), 64 tokens in 4 chunks:

| component | ms | share |
|---|---|---|
| ANE, 36 GDN layers | 907 | 43% |
| MoE (GPU) | 589 | 28% |
| ANE, 12 QSA layers | 271 | 13% |
| host mixers | 147 | 7% |
| output `.numpy()` copies | 60 | 3% |
| recombine | 43 | 2% |
| router | 41 | 2% |
| KV feed + indexer | 29 | 1% |

Host mixers were 7%, not the 20%+ the arithmetic suggested. The two real
targets are the ANE GDN pass and the MoE:

* **ANE GDN is 227 ms/chunk against 157.7 ms measured standalone.** The extra
  ~70 ms is the per-layer feed — `pool.x_hc.fill(0)` plus `_bsh_to_bc1s` over a
  655 KB buffer, 36 times per chunk. Worth attacking: it is pure host copying.
* **MoE is 3.07 ms/layer against 1.138 ms measured standalone.** Same GPU wake
  penalty as decode: every call follows a ~4.3 ms ANE submit, and a GPU that
  just idled costs ~0.4-0.8 ms to restart.

The fp16 mixers in the folded graph also flip greedy near-ties: on one prompt
the top-2 were (303, 440) at a 0.28 margin bare and (440, 303) at 0.047 folded.
Same pair, reordered. Benign, but greedy output is not bit-identical once the
QSA mixers move onto the ANE.

For reference, MLX warm prefill on this machine is 354.8 tok/s (2.8 ms/token),
so prefill is still ~12x off the GPU.


## Sep 12: host micro-optimisation is not where prefill time is

Two attempts, both neutral:

1. **Folding the QSA mixers into the graph** — built, verified, 12 layers
   exported. 35.2 ms/token bare vs 33.1-37.7 folded.
2. **Removing the bsh<->bc1s round trip** between layers (a 655 KB layout
   conversion 36x per chunk, plus a redundant `fill(0)`) — 35.5 ms/token.

Identical configs measured 33.1, 35.2, 35.5 and 37.7 ms/token across runs, so
**run-to-run noise here is about +/-10%** and neither change is distinguishable
from it. Quote prefill as ~35 ms/token, not to three digits.

**The gap is the two-device alternation, on both sides.** The 36 GDN layers
cost 227 ms/chunk in the real loop against 157.7 ms measured standalone, and
the MoE costs 3.07 ms/layer against 1.138 standalone. The standalone runs issue
their calls back to back; the real loop alternates ANE and GPU every layer, and
each device pays a restart cost when the other has just been running. That is
the same ~0.4-0.8 ms GPU wake penalty measured for decode, mirrored on the ANE
side.

So the remaining prefill headroom is not in numpy. It is in reducing
alternations — more tokens per submit, or getting the routed MoE off the GPU —
and neither is a host-side fix.

## Where decode actually stands

Decode is **unchanged by all of today's prefill work**, because prefill and
decode share graphs but not the bottleneck:

| | before | now |
|---|---|---|
| ANE + I/O | 114 ms | 163 ms |
| MoE | 95 ms | 70 ms |
| per token | 220 ms | 234 ms |
| context | 32 tokens | **2048** |

The ANE/MoE shift is mostly bookkeeping — the 12 QSA layers' MoE now runs
inside `qsa_step_layer` and is charged to the ANE timer. Decode is ~4.3 tok/s
either way, now with real context instead of 32 tokens.

**Nothing short of speculative decode moves that number.** The k-token graphs,
the batched MoE, the indexer and the rollback-capable cache design are all in
place; what is missing is the drafter and the accept/rollback loop.

## Sep 12 (evening): 8k / 16k context

The 2048-token generate guard treated `FLASHNEXT_PREFILL_MAX_S` as max context.
That is the QSA **graph width** (indexer budget), not host context. Lifted:
host KV grows, the indexer gathers a budget-sized subset, `--prompt-len N`
pads a seeded id list. `generate` will not print 8k tokens of decoded junk.

### Host path is flat in context

`probes/flashnext_longctx.py`, layer-3 indexer + gather, 16k stream in 0.7 s:

| ctx | k=16 chunk | decode L=1 | KV gather fp16 | ANE QSA (filled 2048 cache) |
|---:|---:|---:|---:|---:|
| 2048 | 0.17 ms (no select) | 0.02 ms | 1.08 ms | 1.92 ms |
| 4096 | 0.49 ms | 0.06 ms | 1.10 ms | 1.86 ms |
| 8192 | 0.76 ms | 0.07 ms | 1.12 ms | 1.85 ms |
| 16384 | 1.15 ms | 0.09 ms | 1.16 ms | 1.84 ms |

ANE QSA is **flat in host context** — it always sees 2048 keys. Indexer
scoring of 4096 blocks at 16k is 1.2 ms per k=16 chunk (~0.07 ms/token).
Gather of 2048 keys from a 16k cache is 1.16 ms vs 1.08 ms at 2k. Twelve
QSA layers add ~14 ms/token of host gather at decode, already paid at 2k.
**8–16k does not move decode off 4.3 tok/s.** Speculation is still the lever.

`keep[:2048]` on indexer output dropped the ragged tail (mlx_lm returns
2049 keys at offset 2101). `clip_to_budget` always keeps the tail, then the
most recent selected body keys. L>1 unique-block overflow is ranked to fit.
`sel is None` past the budget now raises rather than feeding prefix[:2048].
KV cache in generate is fp16.

### Core AI IOSurface pool is a ~2500-submit process budget

`FLASHNEXT_MOE=mlxresident FLASHNEXT_PREFILL_K=16 --prompt-len 8193` gets
through 512 prefill tokens and then aborts:

```
CoreAIRuntime/NDArray+Pool.swift:77: Fatal error: Failed to allocate
storage for NDArray with byteCount: 32768, sk: ioSurface, st: float16
```

32768 bytes is QSA `new_k` at seq=32 — that is the *next* allocation when
the pool is already empty, not a special QSA bug. Every I/O descriptor on
these graphs is `storage_kind=ioSurface`. `NDArray.numpy()` is a snapshot
(mutating it does not write back). Graphs have `States: []`, so the
`state=` output-backing slot is unused. Host wraps default to BYTES;
Core AI still materializes IOSurfaces for ANE outputs.

The budget is **total ANE evaluates in the process**, not concurrent
models and not Python refs:

| resident layers | chunks until abort | layer-submits | tokens (k=16) |
|---:|---:|---:|---:|
| 4 (3 GDN + 1 QSA) | **512 PASS** | 2048 | **8192** |
| 8 | ~320 | ~2560 | ~5120 |
| 12 | ~224 | ~2700 | ~3584 |
| 24 | ~110 | ~2640 | ~1760 |
| 36 GDN only | ~70 | ~2500 | ~1120 |
| 48 | ~40 | ~1920 | ~640 |
| 12 QSA only | 80 PASS | 960 | 1280 |

~2500 submits per process, then `new_ssm` (1 572 864 B) or `h`/`new_k`
fails. 4-layer ANE-only **8k tokens is 1.45 ms/token / 692 tok/s** — the
graphs are fine at 8k; the 48-layer process cannot issue 24 576 submits.
Reloading `.aimodel`s every 12 layers does **not** drain the pool (and
can `ANE evaluateWithModel failed` once ping-pong surfaces outlive the
model that produced them). Isolated wrap of 8k IOSurfaces (drop or hold)
does not abort — the leak is inside inference, not `NDArray()` wrap.

Last healthy **full** generate sample, 512 tokens, mlxresident + folded
QSA, `max_S=2048`:

| | tok/s | ms/token |
|---|---:|---:|
| chunk 1 (cold) | 12.6–28 | 35–80 |
| chunk 32 (512 tok) | **41** | **24.4** |

24.4 ms/token at 512 is already better than the 35 ms/token 64-token
figure. ANE-only 48-layer (no MoE) on a clean runtime is **12.5 ms/token
/ ~80 tok/s** through 512 tokens, then the same abort.

8k/16k **full** prefill+decode needs either Apple to free that pool or a
subprocess every ~40 chunks with GDN/QSA/indexer state handoff (~13 hops
for 8k). Isolated 8–16k indexer, gather and QSA numbers above are the
realistic context-scaling result.

Do not quote 8k generate tok/s until that abort is gone.

Probes: `probes/flashnext_longctx.py`, `probes/flashnext_iosurface_gc.py`,
`probes/flashnext_ndarray_pool.py`.


## Sep 12: the IOSurface abort, root-caused

Long prefill dies with a Swift `fatalError`, not an exception, so it kills the
process:

```
CoreAIRuntime/NDArray+Pool.swift:77: Fatal error: Failed to allocate storage
for NDArray with byteCount: 32768, sk: ioSurface, st: float16
```

32768 bytes is QSA `new_k` (1, 512, 1, 32). Reproduces between 32 and 64 chunks
at k=16: a 512-token prefill completes, a 1024-token one aborts.

What it is **not**:

| ruled out | evidence |
|---|---|
| input surfaces | `flashnext_iosurface_gc.py` wrap mode: 4000 allocations, held=0, PASS |
| allocation volume per se | infer mode: 4000 submits of the same graph, PASS |
| Python holding references | `gc.collect()` every 2 chunks is already in the loop; still aborts |
| our input backing choice | `FLASHNEXT_ANE_IOSURFACE=0` still aborts — the failing allocation is an **output**, which Core AI allocates itself |

What it is: **Core AI pools output NDArrays and there is no way to bind or
release them.** `flashnext_ndarray_pool.py` shows the function descriptor
exposes `input_names` / `output_names` / `state_names`, and our graphs report
`States: []`, so every submit allocates fresh output surfaces. `NDArray` has
only `dtype / from_descriptor / numpy / shape / strides` — no pool control.
`.numpy()` returns a copy, not a surface view, so outputs cannot be written in
place either. The trigger is 48 resident models rather than raw submit count,
which is why one model survives 4000 submits.

### The fix: export the recurrent buffers as state

`TorchConverter.add_pytorch_module` takes `state_names`, and `_utils.py`
derives state from **mutated buffers** that `torch.export` detects. So
registering `conv`, `ssm` and the QSA KV cache as buffers the module mutates in
place — instead of passing them in and returning them as outputs — makes Core
AI own those buffers persistently.

That pays three ways:

1. no per-submit output allocation for state, which is the abort;
2. the host round trip for `new_ssm` / `new_conv` / `new_k` / `new_v`
   disappears — currently `take` (464 ms) and `kvfeed` (187 ms) on a 512-token
   prefill;
3. fewer output surfaces to bind per submit, which is part of the
   alternation inflation (loss 3).

It needs the layer modules reworked to mutate buffers and all 48 graphs
re-exported.

### Steady-state prefill, and the long-context probe

512 tokens, 32 chunks: **13.925 s, 27.2 ms/token (36.8 tok/s)** — better than
the 33-37 ms/token measured over 4 chunks, as first-touch costs keep
amortizing. Breakdown: ane_gdn 5908 ms (42%), moe 4501 (32%), ane_qsa 1135,
mixers 600, take 464, recombine 318, route 308, kvfeed 187.

`flashnext_longctx.py` at 2k/4k/8k/16k host context:

* indexer scales gently — decode (L=1) 0.024 ms at 2k to 0.100 ms at 16k per
  layer; chunk (L=16) 0.174 to 1.197 ms. Block scoring is O(context/4), so at
  256k this becomes the dominant QSA cost and will need attention.
* **KV gather is flat and expensive**: ~1.1-1.6 ms per layer for the 2048
  selected keys regardless of context, which is ~13-19 ms per decode step
  across 12 layers. fp16 beats fp32 by ~30% (1.14 vs 1.63 ms at 16k), which is
  the concrete argument for a lower-precision KV cache — bandwidth, not
  capacity.

`tail_in_clip=False` at 8k/16k in that probe is a **false alarm for this
architecture**: the only index dropped is the current token, whose key comes
from the graph's `new_k` concat and is already covered by the mask, not from
the host cache.

### clip_to_budget was a real bug in the port

The indexer returns 512 blocks plus a ragged tail, i.e. 2049-2051 keys, and the
port clipped with `keep[:budget]` — which drops the **tail**, the most recent
tokens. `clip_to_budget` keeps the tail first, then the most recent body keys.
Both call sites in `stage_generate` use it.


## Sep 12: QSA width ladder, and what speculation is actually worth here

**The QSA graph's K/V inputs are `max_S` wide and `wrap_ndarray` copies them
per layer per submit.** A 2048 rung therefore moves 4 MB/layer whatever the
real context is — which is why decode went 114 -> 163 ms when the wide graphs
went in. Exporting a ladder and picking the smallest rung that fits
(`FLASHNEXT_QSA_RUNGS`, default 256 plus `FLASHNEXT_PREFILL_MAX_S`) puts
`ane+io` back to **150-153 ms**, with 2048 capability retained for when the
context needs it. Decode ~220 ms/token again, output unchanged.

Watch the model budget: ANE resources run out around **~80 resident models**
(`Program load failed — no ANE resources`). The ladder loads k=1 at every rung
plus the chunk width at the widest rung only: 36 GDN + 36 QSA = 72.

**A GPU keepalive thread is not the fix for the wake penalty.** It recovers
0.667 -> 0.483 ms/call at a 2 ms gap and nothing at 0 or 4 ms, and burning GPU
cycles to stay warm is directly against the tokens/joule goal.

### What speculation is worth, with measured inflation

A k=4 backbone pass costs ~243 ms (GDN 36 x 2.386 x 1.44, QSA 12 x ~2 x 1.44,
MoE 48 x 0.532 x 2.7, host ~15) against 220 ms for a single token. So:

| accepted per pass | ms/token | tok/s |
|---|---|---|
| 1 | 243 | 4.1 (worse than today) |
| 2 | 122 | 8.2 |
| 3 | 81 | 12.3 |
| 4 | 61 | 16.4 |

At the Swift backend's measured 52% acceptance the expected j is ~2-2.5, so
**~10 tok/s** — consistent with the earlier revision, and 2.2x today.

**But the rollback cost has to be counted.** The k-step GDN graph emits only
the final recurrent state, so a partial accept leaves the state wrong. The
options are all expensive:

* re-run a j-token pass to fix the state: 243 + ~230 ms for j tokens, which at
  j=2.5 is ~190 ms/token — barely better than doing nothing;
* emit per-step states: 1.5 MB x k per layer, 216 MB per pass at k=4, straight
  into the IOSurface allocation ceiling that already aborts long prefill;
* accept-all-or-nothing: at 50% per-token acceptance, p(all 4) is 6%.

This is why the Swift backend runs GDN verification **one chunk per draft
token** — it is not an optimisation choice, it is forced by the recurrence.
Doing the same here keeps the state correct but means the GDN layers get no
batching: k x 70 ms for the recurrent half, with only QSA and MoE amortised.
At k=4 that is ~96 ms/token at *full* acceptance, worse than the table above.

So the honest position: **speculation on this architecture is worth roughly
8-12 tok/s, not 30**, and getting there needs the state-rollback problem solved
in a way that does not multiply output surfaces. That is the next real design
decision, and it is upstream of any further host-side tuning.

## Explicit non-goals

- DynBank / 400 `_ANEInMemoryModel` GEMM RPC
- Writing int8/AWQ into the BF16 tree
- iOS / `coreai-build` until the Python Neural Engine path is correct
- Full 512-expert `SwitchGLU` on ANE (Apple documents that for GPU)

## Pointers

| What | Where |
|---|---|
| Export / stages | `scripts/export_flashnext_coreai.py`; connected GDN: `probes/flashnext_connected_gdn.py` |
| Numpy ground truth | `tools/flashnext_reference.py` |
| MLX source of truth | `~/.mlx128/mlx-lm/mlx_lm/models/qwen4_exp.py` |
| 8–16k / IOSurface probes | `probes/flashnext_longctx.py`, `probes/flashnext_iosurface_gc.py`, `probes/flashnext_ndarray_pool.py` |
| Skills | `~/.claude/skills/apple-core-ai`, `apple-neural-engine`, `mlx` |
