# Porting Ling-3.0-tiny to the pure ANE runtime

Status: **Milestones 0-2a done.** Strategy decided, loader verified, KDA
gate validated on hardware. Recurrence, MLA and MoE blocks not yet built.

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

## Milestone 1: loader and spec, verified against the index

`tools/pure_ling.py` holds `LingSpec`, which reads the architecture from
`config.json` and then **checks every derived name and shape against the
checkpoint's tensor index** rather than trusting either. That matters because
`config.json` carries keys the modeling code never reads
(`max_window_layers`, `num_kv_heads_for_linear_attn`, `use_qk_norm`), and
building against them would silently produce the wrong model.

```
tensors=9283 shards=32
embedding=model.word_embeddings.weight (157184, 1536)
full_attention (MLA) layers=[3, 7, 11, 15, 19, 23]
linear_attention (KDA) layers=[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
moe layers=1..23 (23), dense mlp layers=[0]
PURE_LING_INSPECT=PASS spec matches every verified tensor
```

`verify()` also cross-checks the layer split against which tensors exist: the
MLA list must equal the layers carrying `q_a_proj`, and the KDA list the layers
carrying `A_log`. Both matched.

## Milestone 2a: the KDA safe gate on hardware

`probes/ane_ling_kda_gate.py`, against the real layer-1 `A_log` and `dt_bias`.

The gate is **not** Qwen's GatedDeltaNet form:

```text
Qwen GDN   g = exp(-exp(A_log) * softplus(a + dt_bias))       per HEAD
Ling KDA   g = exp(-5 * sigmoid(exp(A_log) * (f + dt_bias)))  per (HEAD, KEY-CHANNEL)
```

Two consequences. The fp16-safe polynomial softplus in `docs/ANE-REFERENCE.md`
is **not needed** — this gate is simpler. But the gate is a 2048-vector rather
than a 16-vector, and `exp(A_log)` (per head, measured 1.4607–2.3331) must be
folded into both the `f_proj` rows and `dt_bias` at load.

Measured, comparing the two sigmoid spellings against a float64 reference:

| spelling | small \|f'\|≤1 | moderate ≤8 | large ≤40 |
|---|---:|---:|---:|
| MIL `sigmoid` | 1.94e-2 | 2.27e-2 | 2.27e-2 |
| `x/(1+exp(-x))` | **4.43e-3** | 1.01e-2 | 1.38e-2 |

`exp_divide` is 2.2x better, confirming the `sigmoid` warning in
`docs/ANE-REFERENCE.md`. **Use it.**

Max relative error over all channels is the wrong summary, though. The gate
multiplies the recurrent state, so a channel at `g ≈ exp(-5)` is being
deliberately erased and its relative error costs nothing, while a channel at
`g ≈ 1` persists and compounds. Bucketed:

| g range | count | max rel | max abs |
|---|---:|---:|---:|
| [0.0067, 0.02) | 8309 | 6.84e-3 | 1.07e-4 |
| [0.02, 0.1) | 3453 | 1.01e-2 | 8.44e-4 |
| [0.1, 0.5) | 4968 | 9.46e-3 | 2.08e-3 |
| [0.5, 0.9) | 6977 | 4.76e-3 | 2.44e-3 |
| **[0.9, 1.001)** | 41829 | **8.38e-4** | 7.56e-4 |

Precision is best exactly where it matters: **8.38e-4 in the persisting
regime, against Qwen's 9.95e-4 decay-error reference.** Mean `|d log g|` is
4.93e-4. fp16 spacing at the exponent 5 is 3.91e-3, which alone floors the
small-`g` tail, so the tail figures are at the hardware limit and not worth
chasing.

The probe also sizes the mistake it exists to catch: using Qwen's softplus gate
here would be off by up to **9.92x**. That is not a subtle divergence, so the
layer smoke has a clear number to fail against.

## Next

Milestone 2b is the KDA recurrence with per-(head, key-channel) decay — Qwen's
`AneGdnRecurrence` broadcasts a width-1 column per head, and this needs a full
128-vector, which is the layout question to settle before the layer smoke.
Then MLA (milestone 3), the MoE block (4), full inference (5), measurement (6).
