# Porting Ling-3.0-tiny to the pure ANE runtime

Status: **Stage 0 complete, strategy decided.** Runtime not yet built.

Ling-3.0-tiny (`BailingMoeV3` / `bailing_hybrid`) is 7.89B total / 1.38B
activated, 15.79 GB bf16. 24 layers, hidden 1536, vocab 157184.

## Architecture, verified against the checkpoint index

Not derived from config alone — every claim below was checked against
`model.safetensors.index.json`:

| property | value |
|---|---|
| MLA (full attention) layers | **3, 7, 11, 15, 19, 23** — the only ones with `q_a_proj` |
| KDA (linear attention) layers | the other **18** — the only ones with `A_log` |
| layer rule | `(i+1) % 4 == 0` → MLA |
| dense MLP | layer 0 only (`first_k_dense_replace=1`) |
| MoE layers | 1–23, **128 experts** each, top-8, +1 shared |
| expert shapes | `gate/up [512,1536]`, `down [1536,512]` → 4.5 MiB bf16, 1.125 MiB int4 |
| router | `mlp.gate.weight [128,1536]` + `expert_bias [128]` F32 |

`max_window_layers`, `num_kv_heads_for_linear_attn` and `use_qk_norm` are
present in `config.json` and **unused by the modeling code** — there are no
q/k norm tensors in the checkpoint.

## Stage 0: which MoE strategy?

`probes/ane_ling_moe_strategy.py`. The prior verdict
(`docs/ANE-MOE-HANDOFF.md` §27) was that MoE loses on the ANE — Qwen3.6-35B-A3B
measured ~11× slower than the GPU, because the ANE has no `gather` so routed
experts must be physically staged, in fp16, at 50 MB/layer/token. Ling-tiny's
experts are small enough that a second option exists: bake **all 128** as int4
constants and compute every one, masking to the routed 8.

Each case runs in its own process — ANE programs are never unloaded and the
~127-program budget is system-wide.

| strategy | ms/layer | ms/token, 1 position | ms/token, lanes filled |
|---|---:|---:|---:|
| (A) dynamic staging, S=32 | 1.602 | 36.8 | 36.8 — cannot batch |
| (B) baked dense, S=32 | 2.447 | 56.3 | **1.76** (32 lanes) |
| (B) baked dense, S=64 | 3.163 | 72.7 | **1.14** (64 lanes) |

A stages 37.7 MB per layer per token (0.577 ms) and dispatches in 0.956 ms.
B holds 151 MB of int4 per layer — **3.47 GB for all 23 MoE layers** — and
copies nothing.

**Decision: (B) baked dense is the default; (A) stays available behind a flag.**

B loses by 1.53× at strictly one position per step and wins everywhere else:

* **A cannot batch at all.** Its program is compiled for exactly one token's 8
  experts, so lanes carrying different routing need separate dispatches. B is
  routing-agnostic: any set of lanes costs the same dispatch.
* **Routing is diffuse** — `moe_routing_locality.py` measured consecutive tokens
  sharing only 1.74 of 8 experts. So k tokens touch close to `min(128, 8k)`
  distinct experts, and A's expert-major prefill has to stage nearly the whole
  layer as fp16 (576 MiB) once k reaches ~16. B already has those weights
  resident as int4. **B overtakes A from roughly 4 tokens onward.**
* **B is the only one compatible with filling the 64 free decode lanes**
  (`docs/OPTIMIZATIONS.md` O1), which is the largest identified optimization.
* B allows int4; dynamic weights are fp16-only because
  `constexpr_blockwise_shift_scale` needs a const operand.
* B costs **5 programs total** for all 23 MoE layers (4 `gate|up` chunks + 1
  `down`), since every MoE layer is shape-identical and banks by procedure.

### Shape notes measured along the way

* `gate|up` stacked is `[131072,1536]`, above the 62080–124160 single-conv
  output-channel limit, so it is chunked 4×. Each chunk is **flat from S=32 to
  S=64** (0.432 → 0.437 ms) — the free-lane property holds.
* `down` is `[1536,65536]`. Unsplit it hits the deep-input tiling pathology; split
  across 4 input-channel groups it runs at 0.771 ms. Unlike `gate|up` it does
  **not** stay flat with width (0.771 → 1.433 → 4.874 ms at S=32/64/128) because
  its 65536-channel activation surface grows with S. It is activation-bound, not
  weight-bound. **S=64 is the best operating point; S=128 is not worth it.**

### Router precision

`router_dtype: "fp32"` is explicit in the config, and scoring is sigmoid (not
softmax) with group-limited top-k, so near-ties are a real risk.

Measured on the real layer-1 router weights: fp16 vs fp32 agree on **99.96% of
the top-8**, and **99.7% of tokens select an identical set**. That de-risks a
future ANE router, but the router is 23 × `[128,1536]` matvecs per token (~4.5M
MACs) — negligible on CPU — so the port starts with **routing on CPU in fp32**
for exactness. This is a documented deviation from "all learned arithmetic on
the ANE" and must stay documented rather than quietly asserted away.

Caveat: activations for this test were a distributional stand-in (unit-RMS
scaled by the real `post_attention_layernorm` weight), not real hidden states.
The definitive check is milestone 4, against a real forward pass.

## Changes to `pure_ane.py` so far

Three additive hunks, all default-preserving; the Qwen path is unchanged:

1. `Checkpoint.EMBEDDING_NAMES` gains `model.word_embeddings.weight`. Qwen's
   names are matched first, so nothing changes for Qwen.
2. `Checkpoint(..., shifted_norms=True)`. Qwen3.5/3.8 store RMSNorm weights as
   deltas from one; BailingMoeV3 stores them plainly and passes `False`.
3. `AneFinalHead(..., head_name=..., norm_name=...)` instead of hardcoded
   `"lm_head.weight"` / `"model.language_model.norm.weight"`, defaults unchanged.

## Next

Milestones 1–6 are in the plan. Next up: the loader and `pure-ling-inspect`,
then the KDA layer smoke — where the two things most likely to be silently
wrong are the **safe gate** (`exp(-5·sigmoid(...))`, *not* softplus; mlx-lm's
in-tree `bailing_hybrid.py` has the softplus bug and must not be used as an
oracle) and the **per-(head, key-channel) decay**, which is a 128-vector rather
than Qwen's per-head scalar.
