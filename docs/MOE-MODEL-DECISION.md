# MoE model decision: which model the C/C++ engine should target

Researched 22 Aug 2026. Ground truth = local checkpoints + our own measured
probes (docs/LING-PORT.md, ../../docs/ANE-MOE-HANDOFF.md), plus vendor pages.

## Verdict

**Target Qwen3.6-35B-A3B first. Ling-3.0-flash second. Park DeepSeek-V4-Flash
and stop investing in Ling-3.0-tiny.**

Qwen3.6-35B-A3B is the only candidate whose attention stack, MTP machinery,
tokenizer, and vocabulary the rindi engine already runs today. The single new
component is the sparse MoE FFN, and its exact expert geometry has already been
measured on our ANE toolchain on real weights. Everything else adds a second
hard problem on top of the first.

---

## 1. The candidates (local ground truth)

| | Ling-3.0-tiny | **Qwen3.6-35B-A3B** | Ling-3.0-flash | DeepSeek-V4-Flash-0731 |
|---|---|---|---|---|
| total / active params | 7.89B / 1.38B | 35.1B / ~3B | 124B / 5.1B | **284B / 13B** |
| local copy | 15 GB bf16 (complete) | 67 GB bf16 (complete) + oQ8-mtp | 255 GB bf16 (complete) | 155 GB fp8 (complete) |
| arch | `bailing_hybrid` (KDA + MLA) | **`qwen3_5_moe`** | `bailing_hybrid` | `deepseek_v4` |
| layers | 24 (18 KDA / 6 MLA) | 40 (30 linear-GDN / 10 full) | 42 (2 dense) | 43 (+1 MTP) |
| experts | 128 top-8 + 1 shared | 256 top-8 + 1 shared | 512 top-8 + 1 shared | 256 top-6 + 1 shared |
| expert shape | [512, 1536] | **[512, 2048]** | [768, 2560] | [2048, 4096] |
| router | sigmoid, noaux_tc, n_group=8/topk_group=4 | softmax top-8, no groups | same as tiny | sqrtsoftplus, noaux_tc |
| hidden / vocab | 1536 / 157184 | **2048 / 248320** | 2560 / 157184 | 4096 / 129280 |
| MTP head | **none** (0 tensors) | **yes** (`mtp_num_hidden_layers: 1`) | yes | yes |
| source precision | bf16 | bf16 | bf16 | **fp8 blockwise 128×128** |
| attention novelty vs engine | KDA gate/recurrence + absorbed MLA (all probed OK) | **none — engine already runs qwen3_5 GDN hybrid** | same as tiny | DSA indexer (topk 512), per-layer compress_ratios, 3 hash layers, sliding_window 128 |

Qwen3.8-27B (running today) is `qwen3_5`: 64 layers (48 GDN / 16 full), hidden
5120, vocab 248320, `mtp_num_hidden_layers: 1`. Qwen3.6-35B-A3B is the same
family with the dense MLP replaced by a 256-way sparse block. Same GDN layout
(linear_num_key_heads 16, key-head-dim 128), same conv kernel 4, same
mrope parameters, same tokenizer.

## 2. What we already measured (the evidence base)

From `ANE-MOE-HANDOFF.md` and `LING-PORT.md`, all on this machine:

* **Formats:** ANE accepts int8/int4 *per-output-channel* only; any blockwise
  scale, zero-point, LUT, fp4 → rejected or garbage. int4 packed low-nibble
  first, else silent garbage. Program switching is free; 127 resident programs;
  16 blobs/program; S < 32 silently returns zeros; dispatch floor 0.09 ms.
* **Expert accuracy on real Qwen3.6 weights:** int8 per-channel **2.15e-02**
  rel err — ~6x better than the shipped MLX int4-gs64 build (1.39e-01). The
  narrow `[512,2048]` expert is exactly where the ANE beats the GPU per expert
  (12 TFLOP/s vs 4.7).
* **Fused + stacked layer:** gate/up stacked along output channels, down along
  input channels (`sum_e down_e·a_e == [down_0|…|down_7]@[a_0;…;a_7]`) in ONE
  program = 0.244 ms/layer, flat from S=32 to S=128 — **2.4–3.8x faster than
  MLX SwitchGLU when the expert set is fixed**, and ~2.2–2.7x less energy.
* **The catch (§26):** real routing spreads a token's top-8 across ~7.3 baked
  groups → 1.35 ms/layer → 2.3x slower than GPU. Dynamic staging of routed
  weights → 3.6x slower. Crossover is 2 groups.
* **Ling-tiny end state:** correct output, decode 5.5–8.7 tok/s vs MLX 143,
  prefill ~100 tok/s after dispatch-count fixes. Dispatch-bound: 24 layers of
  small blocks cannot hide the floor. Its KDA gate/recurrence and absorbed MLA
  blocks are fully validated on hardware — reusable for flash later.
* **End-to-end MoE-on-ANE attempt (§27)** hit 11x-slower-than-GPU because
  expert pools cost 1.6 GB/layer and staging copies dominate. All pre-fusion
  Python-path measurements; the C++ chain + MTP lane-filling changes the
  constants, not the ranking.

## 3. Why Qwen3.6-35B-A3B wins

1. **~90% code reuse.** GDN recurrence, conv, full attention, MTP drafter,
   lm_head chunking (same 248320 vocab), radix prefix cache, scheduler, server:
   all exist and work. New code = router (CPU fp32, trivial) + expert matmul +
   combine. Ling needs KDA + MLA + new vocab + new router semantics; DeepSeek
   needs a new attention stack end to end.
2. **The new part is the best-measured part.** Our only per-expert accuracy AND
   speed measurements on real MoE weights are on this exact model.
3. **It ships an MTP head.** Lane-filling is the biggest lever we have
   (O1: 4.7% occupancy today; 27B MTP measured 2.86 tokens/step). Speculated
   verify passes are multi-position by construction — exactly what the flat
   S≥32 ANE cost profile wants, and what amortizes both the dispatch floor and
   the S=32 padding waste. Ling-tiny has no drafter at all.
4. **It fits.** Experts int8 ≈ 32 GB, everything else int4/fp16 ≈ 4–6 GB →
   ~38 GB resident, ~90 GB left for KV + OS. No SSD paging required for v1.
   Bake-all-int4 experts ≈ 16 GB if we go compute-all/mask-routed.
5. **Model quality.** Newest-gen Qwen, agentic/tool-trained, big vocab, and we
   already run LoRA experiments against it. Tiny is the weakest model here;
   flash and V4-Flash beat it on quality but not by enough to pay for what
   they cost below.

## 4. Why not the others

**Ling-3.0-flash (124B/5.1B)** — right architecture lineage (tiny's probes all
transfer, incl. per-(head,key-channel) decay fitting our GDN state layout) and
it has MTP. But per-channel int4 measures 2.68e-01 (worse than shipped gs64),
so experts must be int8 → ~124 GB of expert weights alone. Day-one residency
requires either SSD-paged experts or REAP-pruned subsets *and* the MoE runtime.
That is two unsolved problems instead of one. It is the ideal **second** target
— and the perfect showcase for the REAP + AFM-style pinning combination this
repo exists for — once the MoE machinery exists and works.

**DeepSeek-V4-Flash (284B/13B)** — 13B active is more per-token FLOPs than our
dense 27B; fp8 128×128-blockwise source needs full requant for the ANE's
per-channel-only formats; DSA indexer (topk 512 over ≤1M context),
per-layer compress ratios, hash layers, sqrtsoftplus router are all new engine
code with zero probe coverage. Nothing about it is incremental for us. Park.

**Ling-3.0-tiny** — already effectively answered: it loses to the 27B dense and
to MLX, has no MTP, and is the weakest model. Its value is the validated KDA/
MLA probe suite, which we keep for the flash port.

## 5. AFM-3-style SSD/pinning story (how this maps)

Apple's AFM 3 "Core Advanced" (20B MoE, 1–4B active) solves the same constraint
we measured in §26: NAND bandwidth forbids per-token expert swapping, so they
**pin a fixed expert set per prompt** (dense selector at prefill, periodic
re-select), keep a large always-active shared-expert fraction, and patch routed
experts in to form a dense-in-DRAM model ("inference-time elasticity").

That is precisely the regime where our measurements say the ANE *wins*
(fixed set = one stacked fused program, 0.244 ms/layer, 2.4x GPU) and precisely
what `q38_afm.py` / `ling_afm_run.py` prototyped (record prefill routes → pin →
refresh every N tokens). For Qwen3.6-35B on 128 GB this is an optimization, not
a necessity; on 16–32 GB laptops it becomes the feature. Design the expert
store as (resident pinned set) + (mmapped cold set) from day one — on the Metal
side mmap gives SSD paging almost for free; on the ANE side the pinned set is
what gets baked.

## 6. Integration plan (rindi engine)

Phase 0 — spec & reference
- Extend the checkpoint spec/loader for `model.safetensors.index.json` naming:
  `language_model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.weight` stacked
  `[256, 512, 2048]` (VLM wrapper; ignore vision tower), `shared_expert`,
  `gate.weight` router, MTP layer `layers.40.*`.
- CPU fp32 reference forward for one MoE layer + full-model logit check
  against mlx_lm (the Ling-tiny playbook).

Phase 1 — GPU-hybrid correctness (ship something that talks)
- Router on CPU fp32 (softmax top-8, `norm_topk_prob` semantics verified).
- Routed experts via our Metal engine as a fused gathered matmul over the
  stacked weight tensor (read-in-place, no staging copy — the one place the GPU
  should keep the work; mirrors SwitchGLU).
- Shared expert + norms + GDN/full-attn tails through the existing ANE chain /
  Metal tails unchanged.

Phase 2 — ANE expert paths
- Quantize experts int8 per-channel from the **bf16** checkpoint (never from a
  quantized build); 6 blobs/expert → 2 experts/procedure, program switching free.
- Baked pinned-set path (AFM policy): one stacked fused program per live group;
  refresh cadence measured, default 16–32 tokens.
- Optional compute-all/mask-routed int4 variant (~16 GB) for batched verify
  passes — decide from measured S≥32 lane occupancy, don't argue.

Phase 3 — MTP + measure
- Wire the shipped MTP head into rindi_mtp (depth 1–3); measure accepted/step.
- Report decode tok/s, prefill tok/s, W (powermetrics), vs oQ8-mtp under
  LM Studio and vs mlx_lm bf16 — machine idle, per LING-PORT's benchmarking
  caveat.

## 7. Honest expectations

Pure-ANE single-stream decode will not beat MLX/GPU on raw tok/s for any MoE —
our own §27 says so, and the Ling-tiny clean benchmark confirms it. The win
condition is the one the 27B already demonstrates: competitive tok/s at ~6 W,
GPU-free silent mode, strong prefill/batch throughput, and MTP multiplying the
fixed per-step cost. Qwen3.6-35B-A3B is the model that gets us there with one
new component instead of five.
