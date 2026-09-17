# Porting Ling-3.0-tiny to the pure ANE runtime

Status: **The model runs and generates correct text, and is far slower
than MLX.** See "Clean benchmark" below before quoting any figure in this
document -- several earlier numbers were measured against a contended baseline
and are wrong by two orders of magnitude.

Status (original): **The model runs and generates correct text.** A numpy reference
forward over the real checkpoint produces ' Paris.' for 'The capital of France
is', which validates KDA, MLA and the MoE router together. Every ANE building
block is separately validated on hardware. What remains is wiring those blocks
into a full ANE runtime.

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

## Milestone 2b: the KDA recurrence fits Qwen's layout unchanged

`probes/ane_ling_kda_step.py`. This was the risk that could have forced a
redesign: Qwen's `AneGdnRecurrence` decays with a per-head scalar, while KDA's
decay is per **(head, key-channel)** — a 128-vector per head.

It turned out to be *simpler*, not harder. Qwen stores `state[h,dv,dk]` at
channel `h*Dk+dk`, width `dv`, so `(h,dk)` **is** the channel index: a
per-key-channel decay is a per-channel scalar, i.e. a width-1 column broadcast
across the width — the same idiom `k` and `q` already use. Qwen's per-head
scalar is the one that needs a grouped-conv broadcast. Everything else (the
`Dk→1` reduction, the `1→Dk` delta broadcast) is unchanged.

Measured over six *dependent* steps, feeding the ANE's own state forward, against
a float64 reference:

| step | y rel | state rel |
|---:|---:|---:|
| 0 | 1.13e-3 | 8.40e-4 |
| 2 | 1.58e-3 | 1.20e-3 |
| 5 | 1.41e-3 | 1.08e-3 |

No drift across steps. Qwen's GDN resident-state reference is 1.06e-3.

Both results leave on one surface: `concat` does not exist and `pad`+`add` caps
near 9216 channels, comfortably above the 2064 needed (`HK`=2048 state + `H`=16
output). That avoids the secondary-output binding helper, which is hardcoded for
width 32 while these outputs are width 128.

### `q` must not be pre-scaled by `Dk**-0.5`

The reference kernel scales `q` by `128**-0.5 = 0.0884` before the readout.
Doing that ahead of the ANE step is a **9x accuracy loss**:

| q scale | typical \|s2·q\| | y rel |
|---|---:|---:|
| `Dk**-0.5` | 5.08e-05 — **denormal** | 1.91e-2 |
| l2 only | 5.20e-04 | 2.06e-3 |
| × 8 | 4.66e-03 | 1.32e-3 |

fp16's smallest normal is 6.10e-05, so the pre-scaled products fall into the
denormal range and lose precision before the reduction ever runs.

The fix is exact rather than a fudge: `o_norm` is an RMSNorm applied directly to
`y`, and RMSNorm is scale-invariant, so the factor is absorbed completely —
only its epsilon needs multiplying by `Dk`. This is the same reasoning behind
the 64x carry in Qwen's GDN recurrence (`docs/ARCHITECTURE.md`).

## Milestone 3a: MLA algebra, and the absorbed projections on hardware

Two probes. `ane_ling_mla_ref.py` is a float64 oracle over the real layer-3
weights and settles the algebra before any MIL exists; `ane_ling_mla_absorb.py`
runs the piece Qwen's port has no equivalent of.

### The absorbed form is exact, not merely close

| check | rel |
|---|---:|
| absorbed vs materialized KV | **1.37e-15** |
| through the head gate and `dense` | 6.03e-16 |

That is float64 epsilon, so folding `kv_b_proj` into the query and the output is
an identity rather than an approximation. It pays for itself in cache:

| form | bytes/token/layer | 6 MLA layers at 32K |
|---|---:|---:|
| absorbed (512 latent + 64 k_pe) | **1152** | 0.23 GB |
| materialized K and V | 10240 | 2.01 GB |

### Interleaved RoPE is not optional and not eyeballable

The checkpoint stores rope dims interleaved, and the modeling code's
non-interleaved branch is literally `x = 1/0` — there is no fallback. Applying
the wrong one measures rel **2.99e-2 at cosine similarity 0.9989**: output that
looks almost right. This is why it is a numeric check and not a code review
item.

### The absorbed maps are grouped convolutions, and the blocks stay diagonal

`q_abs[h] = q_nope[h] @ W_K[h]` (`[128]→[512]`) and `out[h] = W_V[h] @ ctx[h]`
(`[512]→[128]`) are block-diagonal over 16 heads, i.e. a `groups=H` conv.

| map | fp16 | int8 | int4 |
|---|---:|---:|---:|
| `q_abs` `W_K` | **2.95e-4** | 7.36e-3 | 1.75e-1 |
| `out` `W_V` | **2.38e-3** | 1.13e-2 | 1.91e-1 |

Each row is paired with a head-0 isolation check — head 0's output recomputed
against head 0's weight alone. It tracks the overall error at every precision,
which is what proves the conv is genuinely block-diagonal; a grouped conv that
quietly mixed heads would still produce smallish-looking numbers.

**fp16 is the shipping choice, decided on arithmetic.** `kv_b_proj` is 12.6M
params across all six MLA layers, so int4 would save 18.9 MB of a 4.48 GB model
— 0.42% — while measuring 1.8e-1. Consistent with the Ling-3.0-flash result in
`docs/ANE-MOE-HANDOFF.md` that per-channel int4 on this family measures 2.68e-1.

### Precision plan for the whole model

| block | params | precision | size |
|---|---:|---|---:|
| routed experts | 6945.8M | int4 | 3.47 GB |
| all attention (KDA + MLA) | 383.7M | fp16 | 0.77 GB |
| embeddings + `lm_head` | 482.9M | int4 | 0.24 GB |
| **total** | | | **4.48 GB** |

## Milestone 3b: the absorbed MLA attention core on hardware

`probes/ane_ling_mla_core.py`. After absorption the query and key are both
576-wide (512 latent + 64 rope) and there is **one** shared key stream rather
than Qwen's four KV heads, so this is MQA. Qwen's `AneAttentionCore` layout
carries over unchanged: width is the head dim, channels carry the query heads,
the cache rows and the mask.

Because both score terms share one contiguous 576-wide key, `q_abs·lat` and
`q_rope·k_rope` collapse into a **single matmul** rather than two.

Measured against a float64 reference, cache L=256:

| valid positions | ctx rel | max attention prob | masked-row leak |
|---:|---:|---:|---:|
| 1 | 8.76e-6 | 1.0000 | 0.00e+00 |
| 7 | 1.29e-3 | 0.1879 | 0.00e+00 |
| 64 | 1.96e-3 | 0.0211 | 0.00e+00 |
| 255 | 6.34e-3 | 0.0063 | 0.00e+00 |
| 256 | 5.70e-3 | 0.0061 | 0.00e+00 |

Error grows with cache occupancy, which is fp16 accumulation over more terms;
Qwen's own core measures 3.83e-3 at L=256, so this is the same regime.

**Masked-row leak is exactly zero at every length.** That column refills the
masked cache rows with 8-sigma garbage and remeasures the context: any softmax
mass escaping the mask would move it. An earlier version of this probe tried to
recover the ANE's probabilities by multiplying the context through a `pinv` of
the cache and reported 88% argmax agreement — that was measuring the pinv's
conditioning, not the model, and was replaced.

### One deliberate inefficiency

The value contraction only needs the 512 latent columns, but slicing the width
down would make the output narrower than the input, which
`docs/ANE-REFERENCE.md` records as failing with `status=0x1d` unless the output
surface is allocated explicitly. The full 576 is contracted instead and the 64
rope columns are discarded on read: 11% more work in one matmul, widths stay
equal, and the discarded columns are never read so they are arithmetically
inert.

## It generates text

`tools/pure_ling.py reference-generate` is a float32 numpy forward over the real
checkpoint — no ANE, no MLX. It exists to prove the architecture end to end and
to be the oracle the ANE runtime is checked against.

```
$ tools/ane ling-reference-generate --raw-prompt --prompt "The capital of France is"
  token   13997  ' Paris'
  token      13  '.'
text=' Paris.\n\nOkay'
```

and through the checkpoint's own Bailing V3 chat template, thinking on:

```
$ tools/ane ling-reference-generate --prompt "What is 2+2? Answer in one word."
text='\n1.  **Analyze the Request'
```

That is one pass through all 24 layers: 18 KDA, 6 MLA, 23 MoE blocks with
group-limited top-8 routing, a shared expert, and the dense layer 0. Getting
' Paris.' means the routing details are right — sigmoid rather than softmax
scoring, `expert_bias` steering selection only, the group score being the sum of
the top two, and the weights coming from the pre-bias scores. Any of those
inverted produces fluent nonsense rather than a wrong-looking crash.

Weights are pulled lazily and routed experts are read per token, so a few tokens
cost far less than the 15.8 GB checkpoint.

### What it actually costs, and why

Two corrections, in order of how wrong they were.

**The reference is 1.27-1.34 tok/s warm, not 0.5.** The 0.5 figure timed the
first tokens of a cold process, so most of it was reading and converting bf16
weights off a 15.8 GB checkpoint rather than arithmetic.

**~5 tok/s is not a ceiling for anything except this numpy code.** That number
came from dividing 5.53 GB of fp32 weight traffic by 27.4 GB/s, the
single-threaded numpy GEMV rate on this machine -- both of which are properties
of the reference implementation, not of the model or the hardware. The real
baseline, measured:

```
$ mlx_lm.generate --model .../Ling-3.0-tiny --max-tokens 48
Generation: 41 tokens, 27.269 tokens-per-sec     # bf16, GPU, 16.4 GB peak
```

**MLX runs this model at 27.3 tok/s**, 20x the numpy reference, and answers
correctly ("The capital of France is **Paris**."). That is the number the ANE
port has to be measured against.

| path | bytes/token | measured | effective bandwidth |
|---|---:|---:|---:|
| numpy reference, fp32, CPU | 5.53 GB | 1.3 tok/s | 7.0 GB/s |
| **MLX, bf16, GPU** | **2.76 GB** | **27.3 tok/s** | **~75 GB/s** |
| ANE target, int4 | 0.69 GB | — | 150 GB/s streaming (measured, `ANE-REFERENCE.md`) |

Decode here is weight-bandwidth-bound, so bytes per token is the lever: int4 is
a quarter of bf16, and the ANE's measured weight streaming is about double the
75 GB/s MLX achieves. That is the headroom the port is chasing. It is headroom
on paper -- the ANE also pads decode to 32 lanes and sustains ~10-20 TFLOP/s,
and `docs/ANE-MOE-HANDOFF.md` records a 35B MoE losing to the GPU by 11x -- but
the arithmetic is why Ling-tiny is worth trying where the 35B was not.

The numpy reference stays what it is: an oracle for checking ANE blocks layer by
layer, not a performance path. Three real inefficiencies in it were still worth
fixing, each verified to leave the generated tokens identical:

* every projection was `x @ W.T` on a non-contiguous transposed view, so numpy
  materialized a fresh copy per call -- 966 MB per token for `lm_head` alone.
  The transpose is now cached contiguously.
* `_silu` and `_sigmoid` promoted to float64.
* stacking the routed experts into one matmul measured *worse* than the loop
  (746 ms against 786), because the concatenation copies 75 MB per layer per
  token, so it was reverted rather than kept for looking tidier.

Per-block, warm: MoE 384 ms (51%), KDA 264 ms (35%), MLA 20 ms, `lm_head` 6.8 ms.

## Validated ANE building blocks

| block | probe | result |
|---|---|---|
| KDA safe gate | `ane_ling_kda_gate.py` | 8.38e-4 where `g >= 0.9` |
| KDA recurrence | `ane_ling_kda_step.py` | y 1.4e-3, state 1.1e-3, 6 dependent steps |
| MLA absorbed algebra | `ane_ling_mla_ref.py` | 1.37e-15 vs materialized KV |
| MLA absorbed projections | `ane_ling_mla_absorb.py` | fp16 2.95e-4 / 2.38e-3, blocks diagonal |
| MLA attention core | `ane_ling_mla_core.py` | ctx 5.7e-3 at L=256, zero mask leak |
| MoE strategy | `ane_ling_moe_strategy.py` | baked-dense int4, 1.14 ms/token at 64 lanes |

## What the ANE port would actually cost, and why

`probes/ane_ling_budget.py` measures every block at Ling's real shapes and
weights it by how often it fires per token. Against MLX's measured 36.7 ms/token:

| block | calls | ms/call | ms/token | % of MLX budget |
|---|---:|---:|---:|---:|
| **`moe_gu`** (4 chunks x 23) | 92 | 0.428 | **39.38** | **107%** |
| **`moe_down`** (23) | 23 | 0.793 | **18.23** | 50% |
| `kda_in` (q\|k\|v\|f\|g\|b fused) | 18 | 0.288 | 5.18 | 14% |
| shared expert (gate\|up + down) | 46 | ~0.11 | 4.92 | 13% |
| `kda_out` | 18 | 0.136 | 2.45 | 7% |
| `lm_head`, 3 chunks | 3 | 0.647 | 1.94 | 5% |
| all six MLA blocks | 36 | ~0.12 | 4.20 | 11% |

**MoE is 67% of the token.** Everything else together is 28 ms.

### Cost 1: no `gather`, so 16x the expert bytes are read

A token routes to 8 of 128 experts — 217 MB at int4 across 23 layers. Baking
all 128 reads **3.47 GB**, sixteen times what is needed. The ANE has no
`gather`, so the alternatives are to compute every expert or to physically copy
the routed ones in (`docs/ANE-MOE-HANDOFF.md` §27, and Stage 0 above).

Ling has something Qwen's MoE did not: `n_group=8, topk_group=4`, so the router
**confines every token to 4 of the 8 groups**. At most 64 of 128 experts can be
live, and half the work is skippable for free. Measured per-group dispatch:

| experts per dispatch | gate\|up | down |
|---:|---:|---:|
| 64 | 0.773 ms | 0.763 ms |
| 32 | 0.422 | 0.424 |
| 16 | 0.255 | 0.175 |

### Cost 2: the per-dispatch floor, which this model is too small to amortize

The driver floor is 0.090 ms per dispatch regardless of work
(`probes/ane_decode_budget.py`). It is visible directly in the shared expert:
0.098 ms/call to move 1.6 MB is essentially **all floor**.

| configuration | MoE | total | tok/s | dispatches | floor |
|---|---:|---:|---:|---:|---:|
| all 128 experts | 57.6 ms | 85.6 ms | **11.7** | 272 | 24.5 ms (29%) |
| 4 live groups of 16 | 39.6 ms | 67.6 ms | **14.8** | 341 | 30.7 ms (45%) |
| MLX bf16 GPU | — | 36.7 ms | **27.3** | — | — |

**The two costs oppose each other.** Finer groups read fewer bytes but need more
dispatches: halving expert traffic (3.47 -> 1.74 GB) raises dispatches 272 -> 341
and the floor 24.5 -> 30.7 ms. It is still a net win, 11.7 -> 14.8 tok/s, but it
does not reach MLX.

For scale: on Qwen3.8-27B the same floor was 29 ms of a 361 ms token, **8%**.
Here it is 29-45%. Ling-tiny activates 1.4B parameters across 24 layers of many
small blocks, so there is far less arithmetic per dispatch to hide the fixed
cost behind.

If the routed experts could be read directly, MoE would be 3.6 ms and the whole
token ~32 tok/s — past MLX. That single missing op is the difference.

### The lever that actually changes the answer

Single-token decode is the wrong target. Measured, every Ling block except one
costs the same at width 64 as at width 32:

| block | S=32 | S=64 | ratio |
|---|---:|---:|---:|
| `kda_in`, `kda_out`, all MLA, `moe_gu`, `lm_head`, shared `gate\|up` | — | — | **0.87-1.02x** |
| `moe_down` | 0.758 | 1.462 | 1.93x |
| `shared_down` | 0.092 | 0.128 | 1.39x |

Only `moe_down` fails to stay flat, because its 65536-channel activation
surface grows with width. Everything weight-heavy is free at 64 lanes.

**But the recurrence is not batchable.** KDA state and MLA attention advance one
position at a time, so 18 KDA layers plus 6 MLA cores are a strictly sequential
per-position cost no amount of lane filling removes. That is the real wall, and
it was hiding a bug.

### The recurrence was 4x slower than its own arithmetic

Ling's KDA recurrence measured 0.509 ms/layer-position against Qwen's 0.358,
despite having 16 heads to Qwen's 48 — a third of the work taking 1.4x the time.
Decomposing it:

| variant | ms |
|---|---:|
| full arithmetic, emit `y` only | 0.118 |
| full arithmetic, emit state only | 0.123 |
| emit both via `pad`+`pad`+`add` | 0.508 |
| **emit both as two bound outputs** | **0.138** |

The arithmetic is 0.12 ms. **Merging the two results into one surface with
`pad`+`pad`+`add` cost +0.385 ms, four times everything else.** The probe used
that spelling only to sidestep `_bind_secondary_output`, which is hardcoded for
width 32 while these outputs are width 128 — a two-line fix, not a design
constraint. Qwen's GDN recurrence already binds two outputs, which is why it was
faster with three times the heads.

Fixing it takes the sequential cost from **9.16 to 2.48 ms per position**, and
the finding is now recorded in `docs/ANE-REFERENCE.md` because it applies to any
program tempted to merge results.

### Where that leaves the two paths

| path | per token | tok/s | MLX |
|---|---:|---:|---:|
| decode, one position per step | ~78.9 ms | **12.7** | 27.3 |
| batched, 64 lanes filled | ~4.8 ms | **~208** | 25.8 (prompt) |

Batched: 3.32 ms of sequential recurrence and attention, plus ~1.5 ms of
everything else amortized across 64 lanes.

**So the ANE loses decode by ~2x and wins batched work by ~8x.** Prefill and
any batch-shaped workload are where this port pays.

**Decode speculation is harder here than for Qwen.** Ling-3.0-tiny sets
`num_nextn_predict_layers: 0` and the checkpoint contains **zero MTP tensors**,
so there is no built-in drafter to fill the lanes with. Qwen's 2.385
accepted-tokens-per-cycle result does not transfer. Filling lanes at decode
would need an external draft model, prompt-lookup/n-gram speculation, or
trained Medusa-style heads — none of which exist yet.

### Cheap wins, in order

1. **Dispatch only the 4 live groups** — 1.42x on MoE, measured, and it needs
   nothing new: the router already produces the group mask.
2. **Fold the shared expert into a routed group dispatch** — it is 4.92 ms/token
   of almost pure floor for 4.7 MB, and removing 46 dispatches saves ~4 ms.
3. **Fill the lanes.** Everything above is worth ~2x; this is worth ~20x.

## Next

Build `LingRuntime` against `LingReference`, group-limited dispatch from the
start, and measure the batched path rather than the single-token one.


## Clean benchmark, and a correction that matters

Every MLX comparison earlier in this document was measured while a
background server held 25 GB at 30% CPU. Re-measured with that process idle:

| | ANE port | MLX (bf16, GPU) | ratio |
|---|---:|---:|---:|
| prefill, 193-token prompt | 24.5-28.3 tok/s | **2112-3260 tok/s** | **0.013x** |
| decode | 8.67 tok/s | **143-144 tok/s** | **0.061x** |

**The ANE port is ~16x slower at decode and ~80-100x slower at prefill.**

### Why the earlier numbers were wrong, and the ANE's were not

The ANE figures barely moved: prefill 28.1 -> 28.3, decode 9.0 -> 8.67. The MLX
figures moved by 5x for generation and 100x for prompt.

That asymmetry is the whole lesson. **MLX and the competing server share the
GPU and its memory bandwidth; the ANE is a separate engine.** So background load
crushed one side of the comparison and left the other untouched. A benchmark run
under load does not degrade uniformly, and "both sides were contended, so the
ratio is roughly fair" -- which this document previously assumed -- is exactly
the wrong inference.

The claim that ANE prefill beat MLX "1.1x" was 1.1x against a baseline
depressed 100-fold. It was never true.

### What the real gap is made of

Prefill: MLX processes the whole prompt as one large GEMM per projection. This
port is capped at 32 lanes by `AneLinearProjectionBank`'s hardcoded width and
issues ~1500 dispatches per 32-token chunk, each carrying a 0.09 ms floor and
~0.19 ms of host marshalling. Widening the banks to 64 is worth about 2x; the
gap is a hundred.

Decode: 115 ms/token, of which `bank.run` is ~83 ms across 371 calls. The ANE
arithmetic is a small fraction; dispatch floor and host copies dominate.

This is consistent with, and worse than, `docs/ANE-MOE-HANDOFF.md` 27, which
measured a 35B MoE at ~11x slower than the GPU and closed the line. Ling-tiny
does not escape that verdict -- it is a smaller model with more, smaller blocks,
so the fixed per-dispatch cost is amortized even less well.

### What would have to change

Not tuning. The port would need to stop being dispatch-bound: far fewer, far
larger dispatches, which for a 1.4B-activated MoE with 24 layers of small blocks
means fusing whole layers into single programs rather than one program per
projection. That is the shape of the Qwen chain fusion in
`docs/ARCHITECTURE.md`, and it is a rewrite of the runtime, not an optimization
of it.


## Would the GPU help with the leftover work?

Measured, identical work on the KDA prepare block `[6144, 64]`, median of 40:

| | per block | per 192-token prompt |
|---|---:|---:|
| numpy (today) | 0.591 ms | 32 ms |
| MLX GPU, tensors resident | 0.314 ms | 17 ms |
| MLX GPU including both transfers | **0.313 ms** | 17 ms |

**Host/GPU transfer is free** on unified memory: 0.019 ms each way, and the
with-transfer figure is indistinguishable from the resident one. That corrects
the note carried from the Qwen work that the "GPU->host->ANE boundary erases"
an ANE win -- that measurement was catching MLX lazy-evaluation forcing, not
the transfer.

But it does not change the outcome. The GPU is 1.89x faster on that block, and
the block is 32 ms of a 4200 ms prefill: 45.7 -> 45.9 tok/s. Applied to *all*
remaining CPU work (1109 ms of the 3003 ms measured), 1.89x would give roughly
**64 -> 77 tok/s, about 1.2x**.

The reason it cannot do more is the split itself. The ANE portion is 1894 ms
for 192 tokens. MLX runs the **entire model** in 91 ms. So the ANE doing its
63% share is already about **20x slower than the GPU doing all of it**, and
optimising the other 37% cannot reach that.

This is the same conclusion `docs/ANE-MOE-HANDOFF.md` 27 reached for a 35B MoE,
arrived at from the opposite direction: there the ANE lost because expert
staging cost more than the GPU's free gather; here it loses because 35
dispatches per token at a 0.09 ms floor cost more than the GPU's whole forward
pass.


## Prefill: 23.4 -> ~100 tok/s

Measured on a 193-token prompt, width-64 banks, int8, 104 programs, 9.97 GB.

| change | prefill tok/s |
|---|---:|
| starting point | 23.4 |
| MLA absorbed maps onto the ANE | 32.3 |
| width-64 banks | 37.4 |
| KDA recurrence onto the ANE | 41.2 |
| stop materializing `[T, experts, 2, M]` | 49.3 |
| **stacked `down` for prefill** | **58.1** |
| **KDA recurrence unrolled 16/dispatch** | **~100** |

The two large wins were dispatch count, not arithmetic. Dispatches for a
137-token prompt went 6807 -> ~450:

* `down` was dispatched once per (layer, routed expert). Correct for decode --
  8 experts, 8 small dispatches -- and wrong for prefill, where 64 positions
  times 8 experts touch ~110 of 128 experts, so a layer cost ~110 dispatches at
  0.105 ms = 11.5 ms against one stacked 1.509 ms dispatch.
* the recurrence was dispatched once per position, 2466 of them, 73% of all
  dispatches after the `down` fix. Unrolling 16 positions into one graph keeps
  the state on the ANE between them. U=8 and U=16 measure the same, so the win
  is removing the per-position dispatch, not the unroll depth.

This is the same lesson as Apple shipping AFM as essentially one compiled ANE
program. The gap was never silicon; it was 2269 dispatches to process 64 tokens.

### What does NOT pay

Fusing the KDA depthwise conv, SiLU and per-head L2 norm onto the ANE
(`AneKdaPrepare`, correct at rel 2.2e-03) measured **slower**: 45.6 -> 40.2
tok/s. The numpy it replaces is already batched over 64 positions, so moving it
adds 18 dispatches and two IOSurface round trips per layer per chunk to save
arithmetic that was never the bottleneck. Kept behind `ane_prepare=False` as
the evidence.

The rule the measurements support: move **weight-heavy** and **per-position**
work to the ANE; leave **batched elementwise** work in numpy.

### Benchmarking caveat, learned the hard way

Decode measured 8.0 tok/s at one point and 5.9 later for identical code, and
two features were wrongly suspected before isolation cleared them. The cause was
a background inference server holding 23.6 GB resident. With it idle, decode is stable at
5.55-5.94 (spread 1.07x) and prefill at 93.5-98.3.

Decode is the more sensitive of the two because it is dispatch- and
latency-bound rather than throughput-bound. **Any decode figure quoted from this
port needs the machine otherwise idle**, and the same is true of the MLX
baseline, which moved 100x for prompt processing between a loaded and an idle
machine.
