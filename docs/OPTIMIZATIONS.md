# Optimizations: what is left, and what each one is worth

Every claim here is tagged with how it is known:

* **Measured** — a probe in `probes/` produced the number on this machine.
  The probe and the command are named so it can be re-run.
* **Derived** — arithmetic on measured numbers, with the arithmetic shown.
* **Unmeasured** — a design argument. No number is claimed. These are ranked
  last on purpose.

All figures are M5 Max, Qwen3.8-27B, int4 weights, `tools/pure_ane.py`.

## Startup: quantize once, compile once

**Measured** — `probes/ane_compile_cache.py`, run in two fresh Python
processes. `_ANEInMemoryModel.compiledModelExists` exposes the private
framework's content-addressed artifact cache. The loader now checks it and
loads the artifact directly, falling back to materialize + compile only when
the direct load fails:

| same identity program | compile | materialize | load | output |
|---|---:|---:|---:|---|
| first process | 13.12 ms | 0.94 ms | 15.98 ms | exact |
| second process | **0** | **0** | 4.71 ms | exact |

**Measured on the complete model before populating the quantized cache** —
persistent server restart, int4,
context 4096, 127 programs, 12.86 GB of learned-weight blobs:

| phase | result |
|---|---:|
| compiled-artifact hits | **127 / 127** |
| compiler time | **0.00 s** |
| compiler input materialization | **0.00 s** |
| ANE load | 0.46 s |
| descriptor construction/hash | 5.27 s |
| BF16 → int4 quantization | **53.96 s** |
| total startup | 60.97 s |

Compilation was already the small part. The private descriptor
identity includes the weight payloads, so the host must still recreate or read
12.86 GB of int4 blobs before it can find the compiled artifact.

The requested fix is the persistent prequantized-weight cache (`--bake-cache`,
enabled by default under `~/Library/Caches/q38-pure-ane`). It stores each
row-quantized tensor as a versioned Zstandard frame and validates the exact
decompressed payload size before use. A sampled 44.57 MB real Qwen tensor
compressed to 31.32 MB (70.3%). The complete cache occupies 9.18 GB of files
(8.5 GiB reported by `du`) instead of another raw 12.86-GB copy.

**Measured warm restart after populating that cache:**

| phase | result |
|---|---:|
| prequantized tensor hits | **504 / 504** |
| BF16 → int4 quantization | **0.00 s** |
| compressed bytes read | 9.18 GB |
| read + decompress | 7.54 s |
| compiled-artifact hits | **127 / 127** |
| compiler time / materialization | **0.00 / 0.00 s** |
| descriptor construction/hash | 5.20 s |
| ANE load | 0.45 s |
| **total server startup** | **14.50 s** |

The cache refuses new writes unless the volume will retain 8 GB of free
headroom (20 GB on Python versions without Zstandard support).
`--no-bake-cache` disables it. Exact phase and cache counters are returned by
`GET /v1/metrics` under `startup`.

### What the “42” roof means

Do not compare the decode table directly to 42 TFLOP/s. Apple officially lists
a faster **16-core** Neural Engine for M5 Max, but does not publish a 42-TOPS
precision table. The 42 figure used by the probes is the reported **INT8 TOPS**
roof, not a 42-TFLOP/s FP16 roof. Under the dual-lane assumption used in this
repository it corresponds to roughly **21 TFLOP/s FP16-equivalent**. Our best
measured real-shape int4 kernels at 18.7–20.3 TFLOP/s are therefore about
89–97% of that arithmetic ceiling. Decode is slow because its width-32 graphs
use few lanes and pay hundreds of dispatches, not because the saturated array
is delivering only 10–15 out of an available 42 FP16 TFLOP/s.

Apple's public statement is deliberately less specific: [M5 Max has a faster
16-core Neural Engine with a higher-bandwidth memory connection](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/).

## Where a decode token actually goes

**Measured** — `probes/ane_decode_budget.py`. Each of the model's real
projection shapes, compiled at the production decode width of 32, weighted by
how many times it fires per token, against the 361 ms budget implied by the
2.770 tok/s end-to-end result:

| block | calls/token | ms/call | ms/token | % of token |
|---|---:|---:|---:|---:|
| mlp gate+up `[34816,5120]` | 64 | 1.308 | 83.7 | 23.2% |
| mlp down `[5120,17408]` | 64 | 0.681 | 43.6 | 12.1% |
| gdn in_proj `[16480,5120]` | 48 | 0.652 | 31.3 | 8.7% |
| attn qkv `[14336,5120]` | 16 | 0.585 | 9.4 | 2.6% |
| lm_head, 4 chunks | 4 | 2.164 | 8.7 | 2.4% |
| **convolution total** | | | **176.6** | **48.9%** |
| **everything else** | | | **184.4** | **51.1%** |

Sequence cores, measured separately at real shapes:

| core | ms/call | calls | ms/token | probe |
|---|---:|---:|---:|---|
| GDN depthwise conv, C=10240 | 0.146 | 48 | 7.0 | `ane_gdn_conv1d.py` |
| GDN resident-state recurrence | 0.358 | 48 | 17.2 | `ane_gdn_resident_state.py` |
| attention core, L=256 | 0.139 | 16 | 2.2 | `ane_attention_core.py` |
| per-dispatch driver floor | 0.090 | ~324 | 29.3 | `ane_decode_budget.py` |

**Important caveat.** The convolution table measures *isolated* convs. The
production engine fuses `out_proj → norm → gate/up → silu → mul → down →
residual` into one program, so those rows are the arithmetic content of a
token, not a measurement of the programs that actually run. Adding the
measured cores to the isolated convs leaves roughly 160 ms unattributed —
fused-program overhead beyond the convs, normalization, host IOSurface copies,
and Python driving the loop. **That unattributed remainder is the largest
single bucket in the budget and nobody has measured what is in it.** See O5.

## The optimizations, ranked

### O1. Fill the 64 free decode lanes. *(largest, unbuilt)*

**Measured** — `probes/ane_free_lanes.py`, ms per dispatch at int4:

| block | S=32 | S=64 | S=96 | S=128 |
|---|---:|---:|---:|---:|
| mlp gate+up | 1.287 | **1.285** | 2.266 | 2.264 |
| gdn in_proj | 0.644 | **0.634** | 1.105 | 1.110 |
| attn qkv | 0.564 | **0.582** | 0.966 | 0.985 |
| mlp down, unsplit | 0.680 | 1.263 | 3.220 | 4.332 |
| mlp down, split 4 | 0.670 | **0.668** | 1.248 | 1.290 |

A dispatch at width 64 costs the same as one at width 32 — within 1% on three
of four projections, and on the fourth once it is split (O2). The step is at
96, and 96 and 128 then cost the same as each other.

So decode has **64 free positions per dispatch, not 32**. The engine currently
uses 1 without MTP and 3 with `--mtp-draft 2`: **4.7% lane occupancy**.

**Derived** — the 176.6 ms/token of convolution is already paying for 64
positions. Anything that converts free lanes into accepted tokens multiplies
throughput against that fixed cost. The arithmetic ceiling is the lane count;
the realized gain is whatever acceptance a drafter sustains.

**Why the current MTP cannot get there.** Drafting is sequential — each draft
token costs a full MTP layer plus a full `lm_head` — which is exactly why
`docs/PERFORMANCE.md` records depth 2 as the best linear depth. A linear chain
cannot fill 64 lanes at any acceptable drafting cost. Filling them needs a
**tree**: branch the drafter at each step, lay the candidate tree across the
free lanes, and verify the whole tree in one batched target pass, keeping the
longest accepted path. The verify side is already batched (the pure MTP
scheduler batches `[confirmed, draft1, draft2]` through projections and MLPs
today); what is missing is tree-shaped drafting and tree-aware causal masking.

**Unmeasured** — the achieved speedup. Measured inputs that bound it: 68.8%
per-token MTP acceptance and 2.385 accepted tokens/cycle at depth 2
(`docs/PERFORMANCE.md`) on the earlier benchmark. After projection chaining,
the raw `Hello` benchmark sustained only 1.875 accepted tokens/cycle: MTP was
bit-identical but ran at 2.907 tok/s versus the target's 3.499. No tree-speedup
number is claimed here; build it and measure.

**Cost** — tree attention masking in the shared attention core, and GDN
recurrence must advance along the accepted path only. The rollback machinery
that makes this safe already exists: the scheduler snapshots 48 compact GDN
states plus convolution histories and replays the proven prefix.

### O2. Split `down_proj` across input channels. *(deployed and measured)*

**Measured** — `probes/ane_peak_real.py`. `[5120,17408]` — few output rows,
very deep input — runs at 24% of peak and is the only projection that gets
*worse* with width. Splitting the input channels across N convs and summing
the partials, at S=512:

| parts | ms | TFLOP/s | % of peak | max rel vs dequant ref |
|---:|---:|---:|---:|---:|
| 1 | 18.2 | 5.0 | 24% | 7.09e-4 |
| 2 | 9.2 | 9.9 | 47% | 9.31e-4 |
| **4** | **4.84** | **18.8** | **90%** | 9.27e-4 |
| 8 | 4.59 | 19.9 | 95% | 1.37e-3 |

**3.81× on that projection**, verified against a dequantized-weight reference,
with error no worse than the unsplit conv at 4 parts. At 8 parts fp16
accumulation across partial sums starts to degrade it.

**It buys nothing at decode width 32** — measured 1.00×. Its value is entirely
that it (a) unlocks O1's 64 free lanes, since unsplit `down_proj` alone doubles
at S=64, and (b) is worth 3.81× on prefill convolution.

**Implemented workaround for the 16-weight-file rule.** The already-quantized
int4 matrix is sliced by input channel, preserving its exact nibbles and row
scales. Four data tensors and four copies of the scale tensor are then stored
as milinternal records inside one `down.bin`, with file-absolute payload
pointers and eight MIL offsets. This changes `AneGdnTail` from 13 weight files
to 12 and `AneAttentionTail` from 10 to 9, so the compiler accepts the complete
chained programs. No requantization is involved; only fp16 partial-sum order
can differ. `--down-proj-parts 1` retains the old graph for A/B tests; 4 is the
pure runtime and server default.

**Measured on real complete tails**, nine post-warmup calls per case:

| complete int4 tail | width | parts=1 | packed parts=4 | change |
|---|---:|---:|---:|---:|
| GDN | 32 | 2.421 ms | 2.447 ms | +1.1% |
| attention | 32 | 2.479 ms | 2.504 ms | +1.0% |
| GDN | 64 | 3.289 ms | **2.683 ms** | **-18.4%** |
| attention | 64 | 3.261 ms | **2.714 ms** | **-16.8%** |

Both width-64 graphs passed their full-layer BF16-checkpoint oracle. GDN output
was unchanged in the sampled layer; attention changed only within the existing
int4 error envelope (20.37% → 20.33% max-relative error against the unquantized
layer oracle).

At the current one-token width-32 production setting the packed graph is
neutral end to end. It emitted identical 16-token text and measured 3.499
tok/s versus the saved one-way 3.499 tok/s. Profiled decode was 284.82 versus
284.39 ms/token. Its production purpose is now proven: it removes the
`down_proj` width-64 cliff needed by O1 without regressing today's server.
The optional depth-2 MTP configuration also compiled with the packed tail (126
programs), emitted the same 16-token target text, and retained its measured
1.875 accepted tokens/cycle; it remains slower than target-only and opt-in.

### O3. Stop compiling prefill 32 wide. *(measured on projections only)*

**Measured** — `probes/ane_prefill_width.py`, µs per token at int4:

| projection | S=32 | S=64 | S=128 | S=512 | gain |
|---|---:|---:|---:|---:|---:|
| mlp gate+up | 40.3 | 20.2 | 18.1 | 18.6 | **2.2×** |
| gdn in_proj | 20.5 | 10.2 | 9.0 | 9.0 | **2.3×** |
| attn qkv | 17.8 | 8.8 | 7.7 | 7.8 | **2.3×** |
| mlp down, unsplit | 21.3 | 19.7 | 34.3 | 35.7 | 0.6× |
| mlp down, split 4 | 21.3 | 10.5 | 10.1 | 9.5 | **2.2×** |

Every program in `tools/pure_ane.py` is built with `width=32` and at most three
real lanes, so a 261-token prompt is ~87 sequential passes of the 64-layer
stack. Most of the available gain is already back by S=64–128; the curve is
flat after that, so this does not require a large-width redesign.

**This bounds the projections only, and they are a small part of prefill.**
**Derived**: the measured 261-token prompt took 35.9 s ≈ 137 ms/token, while
the projection work totals ~5.5 ms/token at S=32 (176.6 ms ÷ 32 positions). So
projections are roughly 4% of prefill and a 2.2× on them is worth ~2% overall.
**The real prefill cost is elsewhere and is unmeasured.** Do O5 before
investing here.

**Cost** — width is a compile-time property of every program, so this interacts
with the resident-program budget: a separate wide prefill set would need its
own programs. Widths must be a multiple of 32 — **measured** today, S=48 builds
but fails `evaluate` with `status=0x1d`, which is the documented 64-byte
row-stride rule.

#### O3a. Fuse the GDN recurrence across a prompt block. *(deployed)*

The projection-only O3 result does not solve the dominant GDN prefill problem:
the current runtime still submits one resident-state recurrence evaluation per
token and per GDN layer. `probes/ane_gdn_scan64.py` now qualifies three exact
formulations using real layer-0 int4 projection and causal-convolution inputs,
with every token output and the final `48×128×128` state checked against both
NumPy and the production ANE recurrence.

**Measured compiler boundary.** The obvious associative affine scan is exact,
but its transition is a per-head `128×128` matrix, not a scalar decay. General
dynamic `128×128` transition composition is rejected by the private compiler
even at two tokens. Qwen's official chunked delta rule avoids those dense
transitions, but its first multi-query token-space product
`N×128 · 128×N` is also rejected. The accepted dynamic attention graph in this
backend is decode-shaped; adding softmax and the value product did not make a
multi-token query axis compile at N=2 or N=32. This rules out describing the
current compiler as supporting a parallel 64-position GDN scan.

**Deployed fallback.** One shared, weight-free ANE program contains 16 exact
recurrent steps, including raw Q/K RMS normalization, polynomial softplus,
decay, sigmoid beta, state updates, and output contractions. It accepts and
returns the production compact state, so prompt blocks and restored prefix
caches chain without CPU model arithmetic. This is sequential in dependency
order inside the graph, but removes 15 submissions and state-surface round
trips. It costs one program-budget slot, not one per layer.

**Measured real layer-0 continuation after a 32-token prefix:**

| block | fused ANE | stepwise ANE | speedup | fused vs step output/state |
|---:|---:|---:|---:|---:|
| 2 | 0.633 ms | 2.181 ms | 3.45× | identical |
| **16** | **3.499 ms** | **12.756 ms** | **3.65×** | **identical** |
| 64 | 54.310 ms | 48.693 ms | 0.90× | numerically passes, rejected for deployment |

The earlier 4.27× result was optimistic because Q/K normalization and gates
were precomputed by NumPy for the scan while included in the stepwise baseline.
The deployed 3.65× result times all of that model arithmetic inside the ANE
graph. Sixteen is the selected block; the compiler scheduler crosses a cliff
before 64.

To let the rest of the model consume 16 positions without another 64-program
prefill set, RMSNorm was vectorized: transpose `[channels, lanes]` to put
channels on the compiler-supported width reduction axis, normalize every lane
together, then transpose the denominators back. The standalone 16-lane test
measured 1.35e-3 relative error and 0.133 ms. A complete chained GDN tail is
bit-identical through the 3- and 16-lane APIs; one lane measured 3.08 versus
3.12 ms, while 16 lanes measured 3.16 versus 18.88 ms as six old calls.

**Measured production server, int4/context 4096, MTP off:**

| test | result |
|---|---:|
| resident programs | 126 |
| semantic check | exact `OK` |
| 148-token prompt, two warm runs | **41.31 ms/token mean** |
| earlier 261-token baseline | ~137 ms/token |
| warmed decode | **274–283 ms/token (3.53–3.65 tok/s)** |
| exact 23-token prefix hit | 23 reused, 0 evaluated, 6.8 ms to cached logits |

The 148-token profile contained nine full 16-token blocks and one four-token
remainder. Its 624 GDN recurrence calls are exactly 432 fused layer calls plus
192 stepwise remainder calls, confirming that production—not only the probe—
uses the new path. Partial blocks and decode retain the one-step recurrence.
The MTP configuration also loads successfully at the 127-program boundary;
`--mtp-draft 2` passed exact `OK` generation and a six-token speculative run
with `mtp_used=true`.

### O4. Keep weights at int4; do not "upgrade" for accuracy without measuring

**Measured** — `probes/ane_peak_real.py`, same shape, S=512: fp16 7.0, int8
13.9, int4 **18.7** TFLOP/s. Large fp16 convs are weight-bandwidth-bound
because the array restreams weights per S-tile; cutting weight bytes 4×
converts them into compute-bound convs.

This is already the production default, so it is not a new gain — it is a
warning. The claim elsewhere in this repo that decode is "invariant to weight
precision" holds **only at S=32**, where dispatch dominates. Any move to int8
or fp16 for accuracy costs up to 2.7× on every widened path.

### O5. Measure the real production loop. *(completed)*

**Measured** — `pure-serve --profile-decode`, three 16-token clean-state runs
after one warmup. The opt-in profiler times the actual `PureAneRuntime` calls
and is returned under each request's `profile`; cumulative totals are exposed
by `GET /v1/metrics`. Mean decode was 297.41 ms/token:

| production phase | ms/token | share |
|---|---:|---:|
| 48 GDN tails | 128.56 | 43.2% |
| 64 projection heads | 49.00 | 16.5% |
| 16 attention tails | 42.27 | 14.2% |
| 48 GDN recurrence calls | 35.93 | 12.1% |
| 48 GDN convolutions | 21.02 | 7.1% |
| vocabulary head | 10.10 | 3.4% |
| attention core + prepare | 8.30 | 2.8% |
| embedding + Python scheduler remainder | 1.90 | 0.6% |

The earlier 44% “unattributed host” bucket was an invalid subtraction of
isolated conv arithmetic from fused production programs. Python scheduling is
only 0.2–0.3%; fused tails are the real majority.

**Implemented from this trace — chained next-layer projections.** Each tail
already computes the next input RMSNorm. It now applies the next layer's
projection before returning its secondary output, eliminating 63 projection
dispatches and both target projection-bank programs. The complete model fell
from 127 to 125 programs and 12.86 to 12.82 GB. Three profiled A/B runs emitted
identical 16-token text and changed:

| | before | chained | change |
|---|---:|---:|---:|
| mean end-to-end | 3.352 tok/s | **3.503 tok/s** | **+4.48%** |
| mean decode | 297.41 ms/token | **284.39 ms/token** | **-13.02 ms** |
| projection-head calls/token | 64 | **1** | -63 |

Five unprofiled production runs measured 3.499 tok/s mean (3.470–3.532), so
the profiler itself does not explain the gain. A semantic request still
returned exactly `OK`.

### O6. Do not run two ANE processes at once. *(measured, operational)*

**Measured** — today, while another process held ~122 programs, a fresh process
failed to load a **16×16** program — a few KB — with `0x50004`. The
127-program budget is therefore **system-wide, not per-process**, which
contradicts the "per-process" framing in `README.md` and `docs/SETUP.md`.

Consequences: a server and a probe cannot coexist; benchmarks must run with
nothing else on the ANE or they measure a failure; and "run a second sequence
concurrently with the GPU", listed as a standing use in
`docs/FULL-HANDOFF.md` §31, cannot mean a second ANE process.

## Measured dead ends — do not retry

| idea | result | source |
|---|---|---|
| INT8 activations to reach 42 TOPS | int8/uint8 inputs compile only through a `cast` to fp16, so arithmetic stays fp16; int4 activations rejected outright | this session |
| `kANEFAneInstanceHint` for parallelism | 3.810 ms concurrent vs 4.057 serialized = 1.06×, where real parallelism would be 2.03 ms | `ane_instances.py` |
| Quantizing further to speed decode | int8 1.989 vs int4 1.941 ms/layer at S=32 | `docs/PERFORMANCE.md` |
| Fusing dispatches to cut count | gate+up from 3 convs to 2 moved 1.929 → 1.939 ms | `docs/PERFORMANCE.md` |
| Preserving only the MIL source directory | still calls `ANECCompile` and saves 1.1×; direct `compiledModelExists` reuse is the implemented solution | `docs/PERFORMANCE.md` |
| Expert co-activation clustering | fails; a token's experts span ~7 groups | `docs/FULL-HANDOFF.md` §23.2, §26.2 |
| In-graph slicing of weights | catastrophic | `docs/FULL-HANDOFF.md` §25.2 |

## Order of work

1. **O1** — tree speculation into 64 lanes. O2's packed parts=4 prerequisite
   is now deployed and measured on the complete width-64 tails. This is the
   largest available multiplier, and
   the only one that attacks the fixed convolution cost rather than shaving it.
2. **O3** — only if O5 shows prefill is projection-bound, which the current
   arithmetic suggests it is not.

## Reproducing

```bash
env -u PYTHONPATH /opt/homebrew/bin/python3 -u -P probes/ane_peak_tflops.py
env -u PYTHONPATH /opt/homebrew/bin/python3 -u -P probes/ane_peak_real.py
env -u PYTHONPATH /opt/homebrew/bin/python3 -u -P probes/ane_prefill_width.py
env -u PYTHONPATH /opt/homebrew/bin/python3 -u -P probes/ane_free_lanes.py
env -u PYTHONPATH /opt/homebrew/bin/python3 -u -P probes/ane_decode_budget.py
```

Nothing else may be using the ANE while these run (O6), or every compile
returns `0x50004` and the probe reports "rejected" for shapes that are fine.


## Solving the recurrence: what is tractable and what is not

Both ports end at the same wall from opposite directions. The gated-delta scan
is flat per token no matter how wide the programs get, so it is what stops
prefill from reaching the ANE's arithmetic rate (19.1 TFLOP/s at S=512 against
8.9 at S=32). The fix is the chunked/WY formulation, which replaces the
sequential scan with matmuls:

```text
G_t   = prod_{s<=t} g_s                     cumulative decay
A[t,s] = (k_s . k_t) * (G_t/G_s)   s < t    strictly lower triangular
delta  = (I + diag(beta) A)^-1 u            one triangular solve
Y      = Q~ S_0^T + tril(Q~ K~^T) delta     matmuls
```

Everything is a matmul over `[C, Dk]` and `[C, C]`, which the ANE does well,
and the state is never materialized per position.

### Qwen's GDN: tractable

Its decay is a **per-head scalar**, so `G_t/G_s` is one number per (head, t, s)
and `A = (K K^T) * D` is a plain masked matmul. Nothing in it is
fp16-hostile. This is the version worth building.

### Ling's KDA: blocked in fp16, measured

Its decay is per **(head, key-channel)**, so the relative decay is a
`[C, C, Dk]` tensor and the contraction is no longer a plain matmul. The usual
way around that is to form `k~ = k / G` and keep plain matmuls -- but `G`
underflows. Measured against the real layer-1 `A_log`/`dt_bias`:

| chunk C | median G | 1st pct | min | % of channels below fp16 min normal |
|---:|---:|---:|---:|---:|
| 8 | 4.19e-03 | 4.43e-12 | 6.03e-16 | **33.6%** |
| 16 | 5.91e-06 | 1.75e-21 | 8.39e-29 | **56.2%** |
| 32 | 1.25e-11 | 2.90e-38 | 0 | **72.3%** |
| 64 | 3.28e-23 | 0 | 0 | **82.3%** |

At C=8 a third of channels already underflow, so `k/G` overflows. The relative
form `G_t/G_s` is bounded by 1 and underflows harmlessly -- zero means fully
decayed, which is correct -- but it is the form that is not a plain matmul.
On fp16-only hardware this needs the log-space `chunk_kda` treatment, and it
should be treated as research rather than porting.

### What solving it for Qwen would be worth

The recurrence is 21.8% of prefill, so removing it alone is only ~1.2x. The
larger effect is second-order: the recurrence is the term that does **not**
scale with program width, which is why widening to 64 measured exactly neutral
(the tail's per-token cost halved and the recurrence's did not). Take it out and
width finally pays, letting the projections run at 19 TFLOP/s instead of 8.9.

Estimated end state, from measured components: **Qwen prefill ~40-55 tok/s
against the GPU's 304.** Still 6x behind, because the ANE's sustained 19
TFLOP/s is 2.4x under the GPU's ~45 and that part is irreducible.

At roughly 6 W against 60-84 W, that 6x gap is about 1.7x **better** per joule,
which is the same ratio `docs/PERFORMANCE.md` already measures for the isolated
kernels. Whether that holds end-to-end has not been measured here -- it needs
`sudo powermetrics` over the generation window, and no power number in this
document was measured this session.
