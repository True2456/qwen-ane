# Optimizations: what is left, and what each one is worth

Every claim here is tagged with how it is known:

* **Measured** — a probe in `probes/` produced the number on this machine.
  The probe and the command are named so it can be re-run.
* **Derived** — arithmetic on measured numbers, with the arithmetic shown.
* **Unmeasured** — a design argument. No number is claimed. These are ranked
  last on purpose.

All figures are M5 Max, Qwen3.8-27B, int4 weights, `tools/pure_ane.py`.

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
(`docs/PERFORMANCE.md`). No number is claimed here; build it and measure.

**Cost** — tree attention masking in the shared attention core, and GDN
recurrence must advance along the accepted path only. The rollback machinery
that makes this safe already exists: the scheduler snapshots 48 compact GDN
states plus convolution histories and replays the proven prefix.

### O2. Split `down_proj` across input channels. *(measured, not deployed)*

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

**Cost** — the **16-blobs-per-program** rule. `AneGdnTail` already uses 11
(`o`,`os`,`d`,`ds`,`gu`,`gus`,`gn`,`gmean`,`grep`,`pn`,`next`), and int4 costs
two blobs per part, so parts=2 fits at 13 and parts=4 needs 17.
**Unmeasured** workaround: pack the four payloads into a single blob at four
absolute offsets, which is the mechanism the projection banks already use
(`docs/ANE-REFERENCE.md`, procedure-bank weight offsets). Until that is tested,
parts=2 is the deployable version at 1.93× on the same shape.

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

### O4. Keep weights at int4; do not "upgrade" for accuracy without measuring

**Measured** — `probes/ane_peak_real.py`, same shape, S=512: fp16 7.0, int8
13.9, int4 **18.7** TFLOP/s. Large fp16 convs are weight-bandwidth-bound
because the array restreams weights per S-tile; cutting weight bytes 4×
converts them into compute-bound convs.

This is already the production default, so it is not a new gain — it is a
warning. The claim elsewhere in this repo that decode is "invariant to weight
precision" holds **only at S=32**, where dispatch dominates. Any move to int8
or fp16 for accuracy costs up to 2.7× on every widened path.

### O5. Measure the unattributed half before optimizing it. *(not an optimization yet)*

**Derived** — convolution is 48.9% of the token budget and the measured
sequence cores add ~26 ms, leaving roughly 160 ms/token (44%) unattributed
between fused-program overhead, normalization, host IOSurface copies, and
Python. **This is the largest single bucket in the budget and its contents are
unknown.**

`docs/PERFORMANCE.md` records host overhead as negligible on a *single*
dispatch — `cast 0.001 / write 0.017 / submit 1.882 / read 0.018 ms`, 98% ANE —
but that was one dispatch of one fused layer, not ~324 dispatches with
per-layer state copies, and it should not be generalized to the whole token.

The measurement to write: instrument the real `PureAneRuntime` decode loop with
per-phase timers around each program class and each host copy, rather than
timing isolated shapes. Until that exists, any effort spent on this half is
guesswork.

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
| Caching the MIL compile | `ANECCompile` re-runs from MIL every time; preserving its output saves 1.1× | `docs/PERFORMANCE.md` |
| Expert co-activation clustering | fails; a token's experts span ~7 groups | `docs/FULL-HANDOFF.md` §23.2, §26.2 |
| In-graph slicing of weights | catastrophic | `docs/FULL-HANDOFF.md` §25.2 |

## Order of work

1. **O5** — instrument the real decode loop. 51% of the token is unexplained
   and everything below is being ranked without it.
2. **O2 at parts=2** — deployable now at 13 blobs, and a precondition for O1.
   Test the packed-blob route to parts=4.
3. **O1** — tree speculation into 64 lanes. Largest available multiplier, and
   the only one that attacks the fixed convolution cost rather than shaving it.
4. **O3** — only if O5 shows prefill is projection-bound, which the current
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
