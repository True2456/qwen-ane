# ANE MoE port (Ling-3.0-flash) — state of play

Handoff doc. Covers Option B: **baked multi-procedure ANE programs** holding MoE
expert weights, dispatched per-expert by `procedureIndex:`. Everything below was
measured on an **M5 Max, 128 GB**, macOS 26.x, against the real Ling checkpoint —
not synthetic tensors, except where stated.

Read `docs/ANE-M5-MAX.md` first for the ANE API surface itself (`_ANEInMemoryModel`,
MIL dialect, IOSurface plumbing). This doc is only the MoE port.

---

## 1. Verdict so far

**The ANE beats the GPU on Ling's expert geometry, and accuracy is not the problem.**

Ling experts are narrow: `gate/up = [768, 2560]`, `down = [2560, 768]`. That shape is
bad for the GPU (low arithmetic intensity) and fine for the ANE.

| | ms / expert @ S=512 | TFLOP/s |
|---|---|---|
| GPU (MLX fp16) | 1.124 | 5.38 |
| **ANE (baked fp16)** | **0.534** | **11.30** |

→ **2.10× the GPU**, on real layer-2 weights.

## 2. Accuracy — settled, and better than expected

Per-expert SwiGLU, ANE fp16 vs an fp32 reference, real weights:

| layer/expert | ANE fp16 | MLX fp16 (fp32 accum) |
|---|---|---|
| 2/0 | 4.06e-03 | 5.05e-04 |
| 22/0 | 3.91e-03 | 5.07e-04 |
| 42/0 | 3.91e-03 | 5.12e-04 |

The ANE is consistently **~7.7× worse than fp16-with-fp32-accumulate**, and that ratio is
*flat across all 41 MoE layers and independent of weight magnitude* (absmax ranged
0.155–0.695). That signature means a **fixed internal accumulator width**, not a
dynamic-range problem — so it will *not* be improved by rescaling weights. A CPU
simulation of fp16-accumulated reduction lands at 1.6e-03, same order, confirming
accumulation as the dominant cause.

**This does not matter.** Against the quantisation the shipped model already spends:

| scheme | rel err | vs ANE fp16 |
|---|---|---|
| **ANE fp16** | **3.94e-03** | — |
| MLX 8-bit gs=64 | 9.49e-03 | ANE 2.4× cleaner |
| MLX 4-bit gs=64 | 1.589e-01 | **ANE 40× cleaner** |

The ANE's accumulation error is ~40× below the error `True2456/Ling-3.0-Flash-4.6bpw-MLX`
already accepts. It is noise inside a budget already being spent. **Close this question.**

## 3. The real blocker is size, not accuracy

fp16 experts *are* the bf16 checkpoint: **237 GB**. Does not fit 128 GB.

| precision | full-model size | fits 128 GB? |
|---|---|---|
| fp16 | ~237 GB | no |
| int8 | ~123 GB | not with headroom |
| **int4** | **~62 GB** | **yes** |
| shipped 4.6bpw | 67 GB | yes |

So the port only works at **int4-class** weights. That made in-graph int4 dequant the gate —
and it works (§4), and is *faster* than fp16 (§6).

## 4. What the ANE actually accepts for quantised weights

Mapped by sweeping dtype × scale-shape (`artifacts/ane_probes/ane_quant_matrix.py`).
`constexpr_blockwise_shift_scale` is **misleadingly named** — it is not blockwise here:

| spelling | result |
|---|---|
| int8, per-channel scale `[OUT,1,1,1]` | **OK**, err 3.40e-04 vs own dequant |
| **int4, per-channel, packed 2/byte** | **OK**, err 3.77e-04 — *this is the one* |
| int4, per-channel, **unpacked** | **compiles and silently returns garbage** (err 1.25) |
| int8/int4 **blockwise** scale `[OUT,nb,1,1]` | REJECTED (`InvalidMILProgram`) |
| any **offset / zero-point** argument | REJECTED |

Two traps worth remembering:
- **int4 must be packed two values per byte, low nibble first.** Unpacked int4 *compiles*
  and produces wrong numbers with no error at all. Always validate against a CPU dequant.
- Scale must be **one per output channel**. No group-wise, no zero-point → weights must be
  **symmetric per-output-channel int4**, which is coarser than the gs=64 affine that ships.

## 5b. RESOLVED — and the target model changed

Everything in §5 below was written before the int4/int8 results landed. Superseded by §10.

## 5. The open question (as it stood then)

Per-channel symmetric int4 is coarser than gs=64 affine. **But a per-INPUT-channel scale is
free**: it folds into the preceding RMSNorm weight and never appears in the ANE graph at all.
That is exactly AWQ, and this repo already has the AWQ machinery.

`artifacts/ane_probes/ane_int4_equalize.py` is **written but never ran** — it measures
naive per-channel int4 vs per-channel int4 + folded AWQ scale vs MLX 4-bit gs=64/32, on
real weights. It was blocked by the TCC failure in §7, not by anything technical.

**Run that first — it is now the only thing standing between here and a working port.**
Speed, dispatch and the size budget are all settled; accuracy of per-channel int4 is the
last unknown. If folding closes the gap to gs=64, the port is unblocked at ~62 GB.
If it does not, fall back to int8 over a *subset* of pinned experts (REAP + residency),
which was the original AFM-style plan anyway.

Also worth doing, per the 67 GB figure: bake from **the shipped 4.6bpw MLX checkpoint**
rather than the 237 GB bf16 — dequantise its groups to fp16, then re-quantise
per-channel int4. Avoids re-deriving a quant that already has calibration behind it.

## 6. Structural constraints — measured, and less binding than feared

**int4 is faster than fp16 on the ANE, and the blob ceiling turns out not to matter.**
(`artifacts/ane_probes/ane_int4_speed.py`, Ling geometry, S=512, synthetic weights —
this measures dispatch and throughput, not accuracy. Compare only within this run;
the GPU baseline moves a little between runs.)

| | ms/expert | TFLOP/s | vs GPU |
|---|---|---|---|
| GPU (MLX fp16) | 0.755 | 7.99 | — |
| ANE fp16 | 0.540 | 11.19 | 1.40× |
| **ANE int4** | **0.400** | **15.11** | **1.89×** |

int4 wins because the ANE is bandwidth-bound here, and dequant is in-graph and free.

Blob ceiling confirmed at **16 per program** → fp16 fits **5 experts/program**,
int4 fits **2** (6 blobs each: 3 data + 3 scale). Ling activates 8 routed experts, so
int4 needs **4 programs per layer** — and that costs nothing:

| dispatch | ms |
|---|---|
| procedure switch, same program | 0.396 |
| switching across 4 programs | 0.398 |

**+0.002 ms.** Program switching is free, so the 2-experts-per-program limit is a
bookkeeping detail, not a design constraint. A full 8-expert layer projects to
**~3.19 ms on ANE vs ~6.04 ms on GPU**.
- **No BLOBFILE offset packing** — each const needs its own blob file, which is what makes
  the 16-blob ceiling bite.
- Layer map: layers **0–1 are dense**, layers **2–42 are MoE**, 512 experts each
  (20,992 = 512 × 41 tensors per projection), plus one shared expert per layer.
- **Attention is still unsolved on ANE.** `bailing_hybrid` (KDA/GDN) does not map to 1×1
  conv. First cut of the port should be **ANE MLPs + GPU attention**.

## 7. Environment traps (cost real time this session)

- **The checkpoint is bf16, which numpy has no dtype for.** `safe_open(framework="np")`
  throws; `framework="mlx"` silently falls back to numpy in this build and throws the same.
  Working approach — parse the safetensors header and shift, reading only what you need
  out of 237 GB via mmap:
  ```python
  raw = np.frombuffer(mm, np.uint16, count, offset).astype(np.uint32)
  w = (raw << 16).view(np.float32).astype(np.float16)   # bf16 == top 16 bits of fp32
  ```
- Run probes with the **oMLX bundled interpreter**, not system python:
  ```
  O=/Applications/oMLX.app/Contents/Resources
  PYTHONPATH="$O/Python/framework-mlx-base/lib/python3.11/site-packages:$O:$HOME/AppleLLM/q38_native_engine" \
    "$O/Python/cpython-3.11/bin/python3.11" -P artifacts/ane_probes/<probe>.py
  ```
  `-P` matters: having the script's own directory on `sys.path` triggered
  `PermissionError` from the import path hooks under macOS file-access restrictions.
- **Never let helper calls mutate `objc_msgSend.argtypes` mid-expression** — hoist every
  `_nsdata`/`_cls`/`_sel` into locals first, or you get a segfault. See
  `q38_ane_engine.compile_multiproc`.

### TCC revocation (live issue at time of writing)
Partway through this session macOS revoked file access for the Claude Code app:
`readdir`/`open` on `~/Desktop` began returning `EPERM` while `stat` still succeeded, and
`git` failed with `Unable to read current working directory`. Nothing in the repo changed.
Fix: re-grant the app **Full Disk Access** (or Files & Folders → Desktop) in
System Settings → Privacy & Security, then restart it. Until then the probes cannot read
the checkpoint and nothing can be committed.

## 8. Probe inventory (`artifacts/ane_probes/`)

| file | what it establishes |
|---|---|
| `ane_ling_real.py` | bakes N real Ling experts fp16, checks accuracy + ANE-vs-GPU speed |
| `ane_ling_err_control.py` | ANE fp16 vs MLX fp16 across layers 2→42 — isolates accumulator error |
| `ane_ling_err_yardstick.py` | prices that error against 8/6/5/4-bit quant |
| `ane_quant_matrix.py` | **dtype × scale-shape acceptance map** (§4) — the important one |
| `ane_int4_speed.py` | int4 vs fp16 vs GPU throughput; blob ceiling; **program-switch cost** (§6) |
| `ane_int4_equalize.py` | AWQ-fold rescue of per-channel int4 — **written, not yet run** |
| `ane_multiproc.py` | multi-procedure dispatch + blob ceiling |
| `ane_baked_vs_dynamic.py` | baked weights vs dynamic-weight upload |
| `ane_pure_energy.py`, `ane_energy_sweep.py` | power measurements |
| `swizzle_capture.py`, `dump_blobs.py` | capture a real MIL program + its blobs from CoreML |

## 9. Uncommitted / loose ends

- `/tmp/replay_stream.py` and `/tmp/multiturn.py` — the oMLX↔q38 parity harness that found
  the CoT bug. **Still orphaned in /tmp**, will be lost on reboot. Move into the repo.
- `_system_boundary` at `q38_patches.py:1267` is dead code.
- The TCC revocation in §7 resolved on its own; everything is committed now.


---

# 10. Conclusion: the target is Qwen3.6-35B-A3B at int8, not Ling

> **SUPERSEDED BY SECTION 11.** The speed claims in this section were measured
> against an unfused per-expert GPU baseline and do not survive comparison with
> MLX's real MoE kernel. The *accuracy* and *dtype* findings here still stand.

## 10.1 The ANE offers exactly two weight formats

Scanned every plausible spelling (`ane_dtype_scan.py`). With a per-output-channel
scale and `groups=1`:

| dtype | result |
|---|---|
| **int8** | OK, correct |
| **int4** (packed 2/byte) | OK, correct |
| uint8 | compiles, **silently wrong** (5.16 err) |
| int6 / uint6 / int5 / uint5 / int3 / uint3 / int2 | REJECTED |

**There is no 6-bit.** Any intermediate bpw must be a *mix* of int4 and int8 per
tensor — which is the bit-allocation problem `mlx-compress` already solves.
Unsigned is unusable: `uint*` needs a zero-point, and offsets are rejected too.

Traps: int4 **must** be packed two-per-byte, low nibble first — unpacked int4
compiles and returns garbage silently. Same for `uint8`. Always validate values
against a CPU dequant; the compiler will not tell you.

## 10.2 Approaches that were tried and are dead

- **Grouped conv to fake group-wise scales.** Numerically perfect (tracks exact
  group dequant to ~9e-04) but **10x too slow**: gs=64 costs 1.959 ms vs 0.188 ms
  ungrouped. The GPU's own int4 gs=64 beats it on both speed *and* error. Dead.
- **Residual int4 (W ≈ Q1 + Q2, two ungrouped convs).** Strictly dominated by
  plain int8 — same 8 bpw, worse error (1.12e-02 vs 8.65e-03), slower
  (0.289 vs 0.189 ms). Dead.
- **AWQ per-input-channel fold.** Moved the error from 2.545e-01 to 2.540e-01,
  i.e. nothing. Caveat: measured with iid Gaussian activations, and AWQ's premise
  is activation *outliers*, so this under-measures it. But AWQ redistributes
  difficulty; it cannot manufacture group granularity, which is the actual gap.

## 10.3 Why Ling loses and Qwen wins

Ling-3.0-flash is **127.5B params, 97% in routed experts**. That forces a bad choice:

| Ling config | size | err | verdict |
|---|---|---|---|
| int8 per-channel | 129 GB | 8.65e-03 | **does not fit** |
| int4 per-channel | 65 GB | 2.68e-01 | fits, but **worse than shipped** |
| shipped MLX affine int4 gs=64 | 73 GB | 1.57e-01 | the bar to beat |

Mixing helps but not enough — best fitting mix (gate8/up8/down4, 6.8 bpw, 108 GB)
still only reaches 1.45e-01, barely matching the shipped build while using 35 GB more.

**Qwen3.6-35B-A3B has no such tradeoff.** 35.1B params, 92% routed experts,
0.45B vision tower (it is a VLM), experts `[512,2048]` — *narrower* than Ling's
`[768,2560]`, and narrow is exactly where the ANE beats the GPU.

| Qwen expert, S=512 | ms | TFLOP/s | err |
|---|---|---|---|
| **ANE int8** | **0.268** | **12.03** | **2.15e-02** |
| ANE int4 | 0.245 | 13.14 | 2.26e-01 |
| GPU fp16 | 0.682 | 4.72 | — |
| MLX affine int4 gs=64 | — | — | 1.39e-01 |

**int8 on ANE is 2.54x the GPU and 6.4x more accurate than int4 gs=64**, at
**35.5 GB** — leaving ~90 GB free. int4 buys 9% more speed for 10x the error;
there is no reason to use it here.

## 10.4 Recommended configuration

**Qwen3.6-35B-A3B, routed experts int8 per-output-channel, baked onto ANE.**

- Quantise from the **bf16** checkpoint at
  `~/.lmstudio/models/mlx-community/Qwen3.6-35B-A3B-bf16`, *not* from an existing
  4-bit build — requantising from a quantised source is strictly lossy, and the
  ANE needs per-channel scales that a gs=64 affine build cannot provide anyway.
- Experts are **MLX fused `switch_mlp`**: `layers.N.mlp.switch_mlp.{gate,up,down}_proj.weight`
  stacked `[256, 512, 2048]`. Slice expert e from axis 0. Note the
  `language_model.` prefix — the same VLM shape that bit the DeepSeek-V4 port.
- Blob budget: int8 needs 6 blobs/expert (3 data + 3 scale) → **2 experts/program**,
  8 active → 4 programs/layer. **Program switching is free** (+0.002 ms measured),
  so this costs nothing.
- **Attention stays on GPU.** 30 of 40 layers are `linear_attn` (Mamba-style:
  `A_log`, `conv1d`, `dt_bias`) and 10 are full attention; neither maps to 1x1 conv.
  First cut is ANE experts + GPU attention, same conclusion as for Ling.

## 10.5 What is genuinely unmeasured

- **No end-to-end model has been run on the ANE.** Every number here is a single
  expert or matmul in isolation. Per-expert error does not linearly predict output
  quality across 40 layers — it needs a real eval, and the repo's own history says
  PPL over-credits (see the REAM result). Run the agentic/repetition evals, not PPL.
- **Routing, gather/scatter, and the residual path are not implemented.** The
  measured 0.268 ms is expert compute only, excluding dispatch of 8 experts per
  token and combining their outputs.
- **The new 72 GB `truemod/Ling-3.0-Flash-mlx-int8attn-int4moe`** (int4 gs=64
  affine, `routed_expert_int4_bf16_passthrough`) is a GPU build. It is not directly
  bakeable — the ANE cannot use its group-wise affine scales.


---

# 11. CORRECTION: the ANE loses to the GPU on MoE layers

Section 10 claimed ANE int8 was 2.5x the GPU. **That was a baseline artifact.**
Two independent errors, both found by checking rather than assuming:

## 11.1 The GPU baseline was the slow path

Every earlier comparison timed the GPU doing **one expert as three separate
matmuls**. Real MoE inference uses `mlx_lm.models.switch_layers.SwitchGLU`,
which fuses all 8 experts into a single `gather_mm` / `gather_qmm` launch.

At S=512, both engines doing the identical 25.8 GFLOP of work:

| | time | throughput |
|---|---|---|
| ANE fp16, 8 dispatches | 2.762 ms | 9.3 TFLOP/s |
| **MLX SwitchGLU int4 (fused)** | **0.816 ms** | **31.6 TFLOP/s** |

The unfused baseline ran at 4.7–8 TFLOP/s. That ~4–6x gap is the entire
"ANE advantage" reported in section 10.

## 11.2 The ANE silently returns zeros below S=32

`ane_seqlen_correctness.py`:

| S | output | verdict |
|---|---|---|
| 1, 2, 4, 8, 16 | all zero | **no compute, no error** |
| 32, 64, 128, 512 | correct (1.8e-02) | OK |

Earlier S=1 timings (0.032 ms, "7.9x the GPU", and the "3.61x cold rotation")
were **timing a no-op**. There is no error, no warning, and the call returns
promptly — it just does nothing. Anything measured below S=32 is void.

Consequence: **decode cannot run at S=1 on the ANE.** It must be padded to 32,
paying 32x the compute for one token of output.

## 11.3 Honest full-layer numbers

Full MoE layer, top-8, ANE fp16 vs MLX SwitchGLU, all outputs verified correct.
"ANE disp" excludes the host-side combine, matching what SwitchGLU is timed doing.

| tokens | S used | ANE disp | ANE+combine | GPU fp16 | GPU int4 | ANE vs int4 |
|---|---|---|---|---|---|---|
| 1 | 32 | 1.140 ms | 1.247 ms | 0.273 ms | 0.176 ms | **0.15x** |
| 8 | 32 | 1.189 ms | 1.294 ms | 0.435 ms | 0.266 ms | 0.22x |
| 32 | 32 | 1.084 ms | 1.197 ms | 0.528 ms | 0.553 ms | 0.51x |
| 128 | 128 | 1.096 ms | 1.429 ms | 0.447 ms | 0.345 ms | 0.31x |
| 512 | 512 | 2.762 ms | 3.944 ms | 0.962 ms | 0.816 ms | 0.30x |

**The ANE loses at every sequence length, by 2-7x.** Best case is S=32 (0.51x),
where the GPU's fused kernel is least efficient.

## 11.4 Why, structurally

- The ANE has **no gather**. Each expert is a separate program dispatch, so a
  top-8 layer costs 8 round trips; the GPU does one fused launch over stacked
  weights. This is architectural, not an implementation detail.
- Peak ANE throughput on this shape is ~9-12 TFLOP/s against the GPU's ~32.
- The S>=32 floor makes single-token decode structurally wasteful.

## 11.5 What still stands from earlier sections

- The dtype/scale acceptance map (§4, §10.1) — int4 and int8 only, per-output-channel
  scale only, packed int4 mandatory, silent-garbage traps. All independently verified.
- fp16 and int8 accuracy on ANE are fine (1.8e-02 fp16 on Qwen; far better than
  int4 gs=64 at 1.39e-01).
- Program switching is free; the ~120-program load ceiling and 16-blob ceiling are real.
- Inline MIL scale literals are **rejected** — scales must be blob files.

## 11.6 The one open question: energy

The original goal was **battery life, not speed**. A 3x slower engine can still
win on joules per token, and nothing here has measured that for a real MoE layer.
`ane_moe_energy.py` does it (needs sudo for powermetrics):

```
O=/Applications/oMLX.app/Contents/Resources
sudo env PYTHONPATH="$O/Python/framework-mlx-base/lib/python3.11/site-packages:$O:$HOME/AppleLLM/q38_native_engine" \
    "$O/Python/cpython-3.11/bin/python3.11" -P artifacts/ane_probes/ane_moe_energy.py
```

**Measured — see §12. The answer is no.**


---

# 12. Energy: first measurement (SUPERSEDED BY §13)

> The ANE numbers below were taken with an unoptimised dispatch loop that left
> the ANE idle most of the time. §13 re-measures after fixing that and the
> conclusion **reverses**. Kept because the method and the crossover reasoning
> are still correct.

# 12(old). Energy: measured, and it does not rescue the ANE

Ran `ane_moe_energy.py` (sudo, powermetrics, one engine at a time, S=512, top-8):

| engine | ANE mW | GPU mW | CPU mW | layers/s | mJ/layer |
|---|---|---|---|---|---|
| ANE | 4042 | 639 | 5101 | 363.4 | **26.92** |
| GPU | 0 | 38199 | 4998 | 1175.1 | **36.76** |

Taken alone this reads as a win: **the ANE uses 1.37x less energy per layer**,
drawing 9.8 W total against the GPU's 43.2 W. That is a large instantaneous
power difference — real, and worth knowing.

## 12.1 But the machine is on 4.4x longer

Energy per layer is not battery life. The whole platform — display, SoC
uncore, memory, everything — keeps drawing power for as long as the task runs,
and the ANE takes 3.913 ms per layer against the GPU's 0.887 ms.

Total energy per layer at a given platform baseline draw:

| baseline | GPU total | ANE total | winner |
|---|---|---|---|
| 0 W | 36.8 mJ | 26.9 mJ | ANE |
| 2 W | 38.5 mJ | 34.7 mJ | ANE |
| **3.25 W** | **39.6 mJ** | **39.6 mJ** | *crossover* |
| 5 W | 41.2 mJ | 46.5 mJ | GPU |
| 8 W | 43.9 mJ | 58.2 mJ | GPU |
| 12 W | 47.4 mJ | 73.9 mJ | GPU |

**The crossover is 3.25 W.** A MacBook Pro with the display on idles well above
that — typically 8–15 W. Above the crossover the ANE's 27% compute-energy saving
is swamped by running 4.4x longer, and it loses on total battery too.

The ANE only wins if the rest of the machine draws under ~3.25 W, which does not
describe a laptop anyone is using.

## 12.2 Host CPU is half the ANE's power, and mostly unavoidable

The ANE arm burns 5101 mW of CPU against 4042 mW of actual ANE — over half its
budget is host-side work, because the ANE has no gather and every expert needs a
separate dispatch with its own surface traffic.

Sharing one input IOSurface across all programs (`ane_shared_input.py`) works and
is numerically identical, but saves only **8%** (4.237 → 3.913 ms). So the cost is
the *output* reads and the fp32 accumulate, not the input copy. Moving the combine
into the graph or onto the GPU could recover more, but the gap to close is 4.4x.

## 12.3 Verdict (WRONG — see §13)

This section concluded "stop the MoE-on-ANE line". That was based on the ANE
running 4.4x slower than the GPU, which turned out to be dispatch serialisation,
not the hardware.


---

# 13. Corrected: the ANE wins on battery by 2.07x  (INVALID — see §15)

> **The benchmark behind this section was wrong.** It issued 8 expert dispatches
> at every sequence length, which is only correct for S=1. Real routing at S=512
> touches 254 of 256 experts. The dispatch-optimisation findings (request
> caching, threading) are real and still apply; the energy conclusion does not.

The tell was that the ANE drew only **4 W while nominally busy**. The M5 Max ANE
can pull far more than that, so it was idling on serialized host round-trips
rather than being saturated.

## 13.1 Three serialisation bugs

1. **`submit()` rebuilt the `_ANERequest` on every dispatch.** It rebuilds
   whenever `procedure_index` differs from last call — which in a MoE loop is
   every single expert. Caching one request per (program, procedure): **11%**.
2. **The input was copied per call** instead of once per program. All 8 experts
   read the same `x`.
3. **`evaluateWithQoS:options:request:error:` is synchronous**, so exactly one
   request was ever in flight. This is why `setQueueDepth:` alone did nothing —
   depth is meaningless with a blocking submit. Issuing the 8 experts across a
   **4-thread pool** actually overlaps them: 2.33 -> 1.79 ms. Saturates at 4.

Result: **3.96 ms -> 1.79 ms per layer**, GPU 0.87 ms. Gap 4.5x -> **2.06x**.
Output unchanged at 1.84e-02 throughout.

## 13.2 Energy, re-measured

| engine | ANE mW | GPU mW | CPU mW | layers/s | mJ/layer |
|---|---|---|---|---|---|
| ANE | **5978** | 41 | **3428** | 558.7 | **16.91** |
| GPU | 0 | 37968 | 3158 | 1173.7 | 35.04 |

Both directions confirm the diagnosis: ANE draw **rose** 4042 -> 5978 mW (it is
being fed), and CPU **fell** 5101 -> 3428 mW (fewer copies, no rebuilds).

**The ANE uses 2.07x less energy per layer**, at 9.4 W total against the GPU's
41.1 W.

## 13.3 Battery, including platform power

| baseline | GPU | ANE | winner |
|---|---|---|---|
| 0 W | 35.0 mJ | 16.9 mJ | ANE |
| 5 W | 39.3 mJ | 25.9 mJ | ANE |
| 8 W | 41.9 mJ | 31.2 mJ | ANE |
| 12 W | 45.3 mJ | 38.4 mJ | ANE |
| 15 W | 47.8 mJ | 43.8 mJ | ANE |

**Crossover is 19.3 W**, up from 3.25 W. A laptop idles at 8–15 W, so the ANE now
wins on total battery in every realistic configuration — the case §12 said was
impossible.

## 13.4 Standing verdict

- **Speed: GPU wins, 2.06x.** Use the GPU when plugged in or latency-bound.
- **Battery: ANE wins, 2.07x energy, crossover 19.3 W.** Worth building for
  battery mode — which was the original goal.
- The S>=32 floor still stands, so decode must be padded to 32 tokens. That suits
  speculative decode (Qwen3.6-35B-A3B ships an MTP head) and batching far better
  than plain single-token decode.

## 13.5 Remaining headroom

At 5978 mW the ANE is still likely not saturated. `_ANERequest` exposes
**`setCompletionHandler:`**, which allows fully async submission with no thread
pool — removing the 3428 mW of CPU the threads cost. Not attempted; it needs ObjC
blocks constructed from ctypes. If it reaches speed parity, the energy advantage
would approach 4x.


---

# 14. Scope limit: these numbers are PREFILL numbers  (SUPERSEDED BY §15)

Everything favourable in §13 was measured at **S=512, which is prefill-shaped**.
Inference has two phases and they behave completely differently.

- **Prefill** processes the whole prompt at once — hundreds to thousands of
  tokens. Always well above the S>=32 floor.
- **Decode** emits **one token per step, by definition** — each token depends on
  the one before it, so it cannot be batched. Always S=1.

The ANE cannot run S=1, so decode must pad 1 -> 32 and discard 31/32 of the work.

Measured with the optimised dispatch (cached requests + 4 threads):

| real tokens | ANE S | ANE ms | GPU ms | ANE/GPU | wasted |
|---|---|---|---|---|---|
| **1 (decode)** | 32 | 0.940 | 0.202 | **4.65x** | 97% |
| 4 | 32 | 0.934 | 0.570 | 1.64x | 88% |
| 8 | 32 | 0.957 | 0.809 | 1.18x | 75% |
| **32** | 32 | 0.915 | 0.939 | **0.97x** | 0% |
| 128 | 128 | 0.920 | 0.484 | 1.90x | 0% |
| 512 | 512 | 1.780 | 0.830 | 2.14x | 0% |

Note the shape: ANE cost is nearly **flat** from S=1 to S=128 (~0.92 ms) because
it is dispatch-bound, not compute-bound, in that whole range. The GPU scales with
actual work. So the ANE is worst exactly where it wastes most (S=1) and reaches
**parity at S=32**, the point where the padding waste hits zero.

## 14.1 What this means per phase

| phase | shape | verdict |
|---|---|---|
| prefill | S >> 32 | ANE ~2x slower, **~2x less energy** — good on battery |
| decode, single stream | S=1 | ANE **4.65x slower**, 97% wasted — bad on both |
| 8–32 tokens/step | S=32 | **ANE at parity on speed, still ahead on energy** |

The S=32 window is reachable by speculative decode with a deep draft, or by
batching 8–32 concurrent requests. Qwen's MTP head gives only ~2–4 tokens per
step, which lands at 1.2–1.6x slower — not the parity point.

## 14.2 Where this lands

The measurements point at **ANE for prefill, GPU for decode** — which is exactly
the split oMLX already ships as its ANE prefill option. The MoE work does not
displace the GPU for interactive generation; it makes the prefill half cheaper on
battery, and becomes genuinely attractive only under batching.

## 14.3 Async submission: attempted, failed

`_ANERequest setCompletionHandler:` would allow firing all 8 experts from one
thread and dropping the pool's ~3.4 W of CPU. `ane_async_block.py` constructs a
global ObjC block by hand (isa / flags / invoke / descriptor) and **segfaults
inside the setter** — the sequential path runs fine immediately before it, so the
crash is in the block layout or the expected handler signature, neither of which
is documented. Setting `BLOCK_HAS_SIGNATURE` without a signature field crashes
too. Left unsolved; the 4-thread pool already captures most of the win.


---

# 15. The ANE cannot do MoE  (reason corrected in §16 -- "no gather" was wrong)

Sections 13 and 14 both rest on a benchmark that issued **8 expert dispatches
regardless of sequence length**. That is correct only for a single token. In a
real MoE layer **every token routes to its own top-8**, so S tokens touch far
more than 8 distinct experts — and the ANE, having no gather, needs one dispatch
per active expert.

Counted with the real router (`ane_routing_reality.py`, top-8 of 256):

| tokens | distinct experts hit | dispatches | ANE ms | GPU ms | ANE/GPU |
|---|---|---|---|---|---|
| 1 | 8 | 8 | 0.94 | 0.202 | **4.7x** |
| 8 | 57 | 57 | 6.70 | 0.809 | 8.3x |
| 32 | 138 | 138 | 16.21 | 0.939 | **17.3x** |
| 128 | 230 | 230 | 27.02 | 0.484 | 55.8x |
| 512 | 254 | 254 | 29.84 | 0.830 | **36.0x** |

Expert count saturates at ~256 by a few hundred tokens: with top-8 of 256, the
expected distinct count is `256 * (1 - (1-8/256)^S)`, which is already 163 at
S=32. **The ANE gets worse as the batch grows** — the opposite of §14's claim.

## 15.1 So the padded columns cannot be used

The question "can the other 31 columns do useful work?" has a clean answer: **no.**
Filling them with real tokens does not amortise the dispatch — those tokens want
*different* experts, so each one added pulls in more experts and more dispatches.
This applies equally to MTP, tree speculation, and batching: the candidates are
different tokens, so they route differently. There is no arrangement of tokens
that keeps the dispatch count at 8 while doing more than one token of work.

## 15.2 Every configuration loses

The only benchmark that was ever structurally valid is **S=1 decode**, where 8
dispatches is genuinely correct — and there the ANE is **4.65x slower**. Energy
at that shape is roughly a wash (ANE ~8.8 mJ vs GPU ~8.3 mJ per layer), and once
platform draw over 4.65x the wall time is included the GPU wins outright.

The §13 energy result (2.07x, crossover 19.3 W) was measured at S=512 with 8
dispatches — a workload that does not exist. It is void.

## 15.3 What is actually true

- **MoE on ANE is dead.** Not for want of tuning: the ANE has no gather, and MoE
  is a gather. One dispatch per active expert, and nearly all experts activate.
- **Dense models on ANE are fine** — this whole problem is MoE-specific. A dense
  layer is one dispatch regardless of token count, which is why oMLX's existing
  ANE prefill (a dense drafter) works well and should be kept.
- The **dispatch optimisations are real and reusable**: caching one `_ANERequest`
  per procedure (11%), writing inputs once per program, and issuing work across
  4 threads (23%). Any future ANE work should start with these.
- The **hardware reference stands** (§4, §10.1, §11.2): int4/int8 only,
  per-output-channel scales only, packed int4 mandatory, silent-garbage traps,
  S>=32 floor, 16-blob and ~120-program ceilings, inline scale literals rejected.

## 15.4 Methodological note

This document reversed its conclusion three times. Each reversal came from the
benchmark not matching the real workload, never from the hardware:

1. GPU baseline was unfused per-expert matmuls -> made the ANE look 2.5x faster.
2. ANE dispatch was serialised -> made the ANE look 4.4x slower and lose on energy.
3. Routing was fixed to 8 experts at all S -> made prefill look viable.

The measurement that would have caught all three earlier is the same one: run the
actual layer, with actual routing, against the framework's real kernel — not an
isolated matmul against a hand-rolled baseline.


---

# 16. Correction to §15, and the real reason

§15 claimed the ANE "has no gather" and called MoE architecturally impossible.
**The premise was untested.** Probing the op set directly (`ane_gather_probe.py`):

| construct | result |
|---|---|
| `matmul`, const operand | **compiles** |
| `matmul`, both operands dynamic | **compiles** |
| batched `matmul` over stacked `[NE,M,H]` experts | **compiles** |
| `gather` (even with a const index) | rejected |
| `conv` with a gathered weight | rejected |

So the ANE *does* have matmul, *does* support dynamic (non-const) weight
operands, and *does* batch over an expert axis. An earlier note in this repo that
"the ANE has no matmul, everything is conv" is wrong. A whole MoE layer —
gate, up, SwiGLU, down, combine, 8 experts — **compiles and runs correctly in one
dispatch** (`ane_batched_swiglu.py`, rel err 9.4e-03).

## 16.1 It still loses, for a different reason

| approach | time (S=32, 8 experts) | throughput |
|---|---|---|
| batched matmul, 1 dispatch, whole layer | 3.085 ms | **0.52 TFLOP/s** |
| baked conv, 8 dispatches | 0.940 ms | ~12 TFLOP/s |
| MLX SwitchGLU int4 | 0.939 ms | — |

**The ANE's matmul path runs ~20x slower than its conv path.** Batching does
remove the per-expert dispatch (0.041 ms/expert at NE=64 vs 0.117 ms separate),
but the op it forces you onto costs far more than the dispatches it saves. On top
of that, dynamic weights cost a 50 MB upload per layer at 69 GB/s.

The two paths are a bind:

- **conv**: fast (12 TFLOP/s) but weights must be compile-time const, so one
  program per expert and one dispatch per active expert.
- **matmul**: takes live weights and batches, but at 0.5-6 TFLOP/s.

Neither saturates the ANE, which is why it never drew more than ~6 W: the conv
path is dispatch-bound and idle between round trips, and the matmul path keeps it
busy on a much slower unit.

## 16.2 Where that leaves it

Best measured ANE config for a top-8 layer is baked conv at **0.940 ms**, which
*ties* MLX at S=32 (0.939 ms) — but only when all tokens share the same 8 experts.
With real routing S=32 touches 138 distinct experts, and both paths degrade badly
from there (§15).

This is a statement about what could be expressed here, not a proof of
impossibility. The op probe was shallow — it tested a handful of constructs, and
a formulation that keeps the conv path's throughput while avoiding per-expert
dispatch may exist. What is established is that the obvious one (batched matmul)
is 20x off the pace.

## 16.3 On "Apple has this working"

Worth separating two Apple models. The **on-device** Apple foundation model is a
**dense ~3B**, which has no gather problem at all — one dispatch per layer,
exactly the shape the ANE conv path is good at. Apple's **MoE** model is the
server one, running on Apple silicon servers rather than a laptop ANE. So the
existence of AFM does not by itself demonstrate MoE running on this ANE, and the
dense/MoE split matches what is measured here: dense suits this hardware, MoE
does not.


---

# 17. The ANE was being run at 4% of its capability

Every MoE number in this document up to §16 was measured on a **starved** ANE.
The tell was the one that kept getting ignored: it never drew more than ~6 W.

`ane_peak_shape.py` sweeps a plain baked conv over output width and sequence
length (input H=2048):

| | M=512 | M=1024 | M=2048 | M=4096 | M=8192 |
|---|---|---|---|---|---|
| S=32 | **0.7** | 1.3 | 2.0 | 2.8 | 3.6 |
| S=64 | 1.3 | 2.5 | 3.8 | 5.5 | 7.2 |
| S=128 | 2.4 | 5.0 | 7.1 | 11.0 | 13.6 |
| S=256 | 3.8 | 7.9 | 11.0 | 13.8 | **15.8** |

(TFLOP/s. int8 at the widest shapes reaches the same ~15.7.)

**Qwen's experts are M=512.** At S=32 that is **0.7 TFLOP/s — about 4% of the
15.8 the hardware reaches when given wide work.** Every "the ANE is slow at MoE"
conclusion in §11–§16 was measured in that corner. The ANE needs both width and
depth to fill its cores; a narrow expert leaves almost all of it idle.

## 17.1 The right formulation: stack experts along output channels

Instead of one dispatch per expert (M=512 each), concatenate the selected
experts into a single conv weight (M = NE x 512 = 4096). Same fast conv path,
one dispatch, and the width the hardware wants.

`ane_stacked_experts.py`, 8 experts, one projection:

| S | 8 separate convs | stacked dispatch | weight upload | stacked total |
|---|---|---|---|---|
| 32 | 1.012 ms | 0.414 | 0.249 | **0.663** |
| 64 | 1.001 ms | 0.415 | 0.244 | **0.659** |
| 128 | 0.955 ms | 0.425 | 0.250 | **0.675** |
| 256 | 0.984 ms | 0.538 | 0.235 | **0.773** |

The separate-dispatch column is **flat at ~1.0 ms regardless of S** — it is
essentially all dispatch latency and almost no compute, which is exactly what
"4% of peak" looks like from the other side.

## 17.2 The new bottleneck is weight upload, not compute

Selecting *which* experts requires live weights, and `constexpr_blockwise_shift_scale`
needs a const operand — so dynamic weights must be fp16, at **16.8 MB per
projection, ~50 MB per layer, ~67 GB/s**. Upload is now a third of the cost and
does not shrink with quantisation.

A **baked** stack pays no upload but cannot choose its experts. That is precisely
the case for **expert pinning** — the AFM-style idea this work started from: keep
hot experts baked as stacked programs and upload only on a miss.

## 17.3 Status

Not a verdict. Stacking is a real ~1.5x over per-expert dispatch and the gap to
MLX is far smaller than §15/§16 claimed, but a full layer (gate+up+down) is still
roughly 2x the GPU, with upload the limiting term. Unmeasured and worth doing:

1. **All three projections in one stacked program** — one dispatch, one upload.
2. **Energy at the wide shape.** The earlier energy runs were also at M=512, i.e.
   at 4% utilisation. Higher utilisation may change joules/token substantially in
   either direction.
3. **Expert pinning** to eliminate upload on hits.
4. Whether `S>=32` still holds for wide dispatches, and how far throughput climbs
   past M=8192.

## 17.4 Note on Apple's models

§16.3 argued Apple's on-device model is dense and its MoE is server-side. That
reflects the AFM generation and may be out of date: current high-end Macs are
reported to run a ~20B MoE with 1-4B active on-device. If so, MoE on this ANE is
demonstrably achievable and the limit is the formulation, not the hardware —
consistent with §17, where simply widening the dispatch recovered 4x.


---

# 18. Dense 27B on the ANE: great at prefill, hopeless at decode

Qwen3.8-27B is `hidden 5120, intermediate 17408, 64 layers`, so its MLP
projections are **M=17408** — 34x wider than a Qwen MoE expert, and exactly the
shape §17 showed the ANE needs.

`ane_dense27b.py`, real projection `[17408, 5120]`:

| bits | S | ms | TFLOP/s | GB/s |
|---|---|---|---|---|
| fp16 | 32 | 1.223 | 4.7 | 146 |
| fp16 | 64 | 1.220 | 9.4 | 146 |
| fp16 | 256 | 7.123 | 6.4 | 25 |
| int8 | 32 | 0.685 | 8.3 | 130 |
| **int8** | **64** | **0.695** | **16.4** | 128 |
| int8 | 256 | 3.621 | 12.6 | 25 |

**Compute is a non-issue.** At int8/S=64 the ANE hits 16.4 TFLOP/s — at or above
the §17 ceiling. Wide dense projections saturate it completely; there is no
starvation here.

Note the collapse at S=256 (25 GB/s, and fp16 gets *slower* in absolute terms).
The output surface is 17408 x 256 fp16 = 8.9 MB and appears to blow a tiling
limit. **There is a sweet spot around S=64**, which is worth respecting when
chunking prefill.

## 18.1 Decode is bandwidth-bound, and the ANE loses that fight

| | weight-read bandwidth |
|---|---|
| **ANE** | **130–146 GB/s** |
| GPU (fp16) | 218 GB/s |

Dense decode reads **every** weight for **every** token:

- 27 GB (int8) / 130 GB/s = **208 ms/token = 4.8 tok/s**

Unusable for interactive generation, and the gap to the GPU is structural — the
ANE simply has less bandwidth to unified memory.

## 18.2 The synthesis

Two independent requirements, and the two model families each satisfy only one:

| | wide enough to fill the ANE? | reads few weights per token? |
|---|---|---|
| dense 27B | **yes** (M=17408) | no — reads all 27 GB |
| MoE experts | no (M=512, 4% of peak) | **yes** — only ~8/256 |

Which is why neither has worked as-is. It also says what *would*: something both
wide and sparse. **Stacking MoE experts along the output-channel axis (§17.1)
does exactly that** — M=4096 instead of 512, while still touching only the 8
routed experts. That remains the most promising direction.

## 18.3 Practical answer for the 27B

- **Prefill on ANE: yes.** 16.4 TFLOP/s at int8/S=64, compute-bound, saturating.
  This is precisely what oMLX's existing ANE prefill option does, and the
  measurements support keeping it — ideally chunked at S~64, not 256.
- **Decode on ANE: no.** 4.8 tok/s, bandwidth-bound.
- So for this model the split stays **ANE prefill + GPU decode**, and the 27B is
  a better ANE prefill engine than the MoE ever was.


---

# 19. Python is not the bottleneck. The dispatch floor is, and fusing fixes it.

## 19.1 Python/ctypes costs 1-2%, not more

`ane_python_overhead.py` calls `evaluateWithQoS:` three ways: through
`eng.submit()`, through a raw `objc_msgSend` with the CFUNCTYPE, selector and
args all hoisted out of the loop, and on a trivial M=16 program.

| M | eng.submit | raw msgSend (hoisted) | python cost |
|---|---|---|---|
| 16 | 0.0923 ms | 0.0916 ms | 1% |
| 512 | 0.0992 | 0.1023 | ~0% |
| 4096 | 0.1785 | 0.1813 | ~0% |

A bare `objc_msgSend` round trip is **0.17 us** — four orders of magnitude below
a dispatch. **Rewriting the harness in C or ObjC would gain about 2%.**

## 19.2 The floor is the driver, ~0.092 ms per dispatch

An M=16 conv — essentially no work — still costs **0.0923 ms**. That is the
per-dispatch cost of a synchronous `evaluateWithQoS:` round trip, and it is paid
every time regardless of how much work the program does. It is the single
explanation for most of this document: 8 experts x 0.092 ms is 0.74 ms of pure
floor, which is why per-expert dispatch was ~90% overhead and why the ANE idled
at 4-6 W.

## 19.3 Fusing layers amortises it (this is how Apple's stack works)

CoreML compiles a whole model into one program and dispatches once. Chaining K
conv+SiLU layers into a single program (`ane_fused_layers.py`, M=H=2048, S=32):

| layers in one program | total | per layer | vs K separate dispatches |
|---|---|---|---|
| 1 | 0.133 ms | 0.1326 | 0.96x |
| 2 | 0.184 | 0.0922 | 1.38x |
| 4 | 0.292 | 0.0730 | 1.74x |
| 8 | 0.524 | 0.0656 | 1.94x |
| **16** | **0.954** | **0.0596** | **2.13x** |

Per-layer cost falls from 0.133 to 0.060 ms and flattens there — at which point
it is reading 8.4 MB of weights in 0.0596 ms = **141 GB/s**, matching the
bandwidth measured in §18. So fusion takes the ANE from dispatch-bound to
**bandwidth-bound**, which is the correct place to be.

## 19.4 Weights cannot be consolidated into one blob

The obvious companion — put all weights in a single BLOBFILE and `slice_by_index`
them in-graph, sidestepping the 16-blob ceiling — **does not work**:

| K | one-blob result |
|---|---|
| 1 | compiles, but 0.248 ms vs 0.133 (the slice is not folded — 2x slower) |
| 2, 4, 8, 16 | **rejected** |

So the **16-blob-per-program ceiling stands**, capping roughly 16 weight tensors
per program. For Qwen3.8-27B (64 layers x 3 projections = 192 tensors) that is
~12 programs per forward pass instead of 192 dispatches.

## 19.5 What this does and does not fix

- **Fixes**: dispatch-bound execution. 2.13x, and the ANE now runs at its
  bandwidth limit rather than idling between round trips.
- **Does not fix**: bandwidth. Dense 27B decode still reads 27 GB/token at
  ~141 GB/s = ~191 ms/token (~5 tok/s). Fusion cannot help a bandwidth-bound
  workload.
- **Most promising remaining combination**: fused programs (amortised floor) +
  stacked experts (§17.1, wide dispatches) + MoE sparsity (reads only ~8/256 of
  weights, so bandwidth is not the limit it is for dense).

## 19.6 Where ObjC would actually help

Not for call latency. The one place a native harness matters is
`_ANERequest setCompletionHandler:` — async submission, which needs ObjC blocks
and segfaulted from ctypes (§14.3). Overlapping dispatches is worth up to the
0.092 ms floor per overlap; the 4-thread pool already captured ~23% of that, and
async might capture more.


---

# 20. RESULT: wide + sparse + fused beats the GPU by 2.4-3.8x

The combination §19.5 pointed at, built and measured (`ane_wide_sparse_moe.py`),
on real Qwen3.6-35B-A3B weights, verified against an fp32 reference.

## 20.1 The layout

Three things at once, none of which worked alone:

- **fused**: gate, SiLU, up, multiply and down in ONE program -> the 0.092 ms
  driver floor is paid once per layer, not once per expert.
- **stacked**: the 8 routed experts concatenated so every conv is wide.
  `gate`/`up` stack along **output** channels -> `[8*512, 2048] = [4096, 2048]`.
  `down` stacks along its **input** axis -> `[2048, 4096]`, because

      sum_e  down_e @ a_e  ==  [down_0 | ... | down_7] @ [a_0 ; ... ; a_7]

  so the expert sum falls out of the matmul. No grouped conv (10x too slow, §11),
  no `reduce_sum`, and every conv stays `groups=1`.
- **sparse**: only the 8 routed experts are touched, so unlike dense 27B (§18)
  this is nowhere near the bandwidth wall.

## 20.2 Measured

| bits | S | ANE | TFLOP/s | GPU int4 (SwitchGLU) | ANE/GPU | rel err |
|---|---|---|---|---|---|---|
| fp16 | 32 | 0.411 ms | 3.9 | 0.587 | 0.70x | 1.95e-02 |
| fp16 | 64 | 0.407 | 7.9 | 1.026 | **0.40x** | 1.93e-02 |
| fp16 | 128 | 0.427 | 15.1 | 0.404 | 1.06x | 1.95e-02 |
| int8 | 32 | 0.244 | 6.6 | 0.600 | **0.41x** | 2.30e-02 |
| **int8** | **64** | **0.251** | 12.8 | 0.965 | **0.26x** | 2.32e-02 |
| int8 | 128 | 0.420 | 15.4 | 0.444 | 0.94x | 2.32e-02 |

**2.4x faster at S=32, 3.8x at S=64**, at 15.4 TFLOP/s — the ceiling from §17,
against the 0.7 TFLOP/s that per-expert dispatch was getting. Accuracy 2.3e-02 is
~6x cleaner than the int4 gs=64 the shipped model uses (1.39e-01).

Note the ANE cost is nearly flat (0.24-0.42 ms) from S=32 to S=128 while the GPU
scales with work, so the ANE's advantage is largest at small batches — exactly
the decode-shaped regime — and gone by S=128.

## 20.3 The remaining gap: these weights are baked

The program above bakes one **fixed** set of 8 experts. Real routing picks a
different set per token. Two ways to close it:

1. **Expert pinning** (the AFM-style idea this work started from). Keep hot
   experts baked as stacked programs; the measured 0.244 ms stands on a hit.
2. **Dynamic weights.** Upload the selected stack per layer: 3 x 8.4 MB = 25 MB
   int8 at the measured ~67 GB/s = ~0.38 ms, giving ~0.62 ms/layer against the
   GPU's 0.60 — roughly parity, no better.

So the win is real but **conditional on not re-uploading weights every token**.
Pinning is what converts it from parity into 2.4x. Routing concentration (how
often the top-8 repeats across consecutive tokens) is the number that decides how
much of the 2.4x survives, and it has not been measured.

## 20.4 Why this took so long to find

The earlier conclusions were not wrong about what they measured — they were
measuring the wrong thing:

| § | claim | actual cause |
|---|---|---|
| 11 | ANE 2-7x slower | unfused GPU baseline |
| 13 | ANE wins battery 2.07x | 8 dispatches at every S |
| 15 | ANE has no gather, MoE dead | never probed the op set |
| 17 | narrow experts, 4% of peak | M=512 starves 16 cores |
| 19 | 0.092 ms floor per dispatch | paid per expert, 8x per layer |

Every fix came from one habit: measuring the actual workload against the real
kernel instead of extrapolating from an isolated benchmark.


---

# 21. The ANE's memory bandwidth really is ~150 GB/s, and it does not scale

The ANE is on the same unified memory as the GPU, so in principle nothing stops
it decoding as fast. Measured, its *interface* to that memory is narrower.

`ane_bandwidth_ceiling.py`, int8 weights, S=32, dispatch floor subtracted:

| weight size | raw GB/s | net GB/s |
|---|---|---|
| 16.8 MB | 82 | **149** |
| 33.6 MB | 107 | **152** |
| 67.1 MB | 123 | **147** |
| 134.2 MB | 139 | **153** |

Flat at ~150 GB/s across an 8x range in size. And concurrency does **not** open
more channels:

| threads | aggregate GB/s |
|---|---|
| 1 | 120 |
| 2 | 133 |
| 4 | 141 |

GPU on the same test scales past it — 86 GB/s at 67 MB, 153 at 134 MB, **263 at
268 MB**, still climbing.

This is the one number in this document that did **not** improve once the ANE
stopped being starved. Everything else in §11-§19 was a measurement artifact;
this is a hardware limit: roughly **1.75x less read bandwidth than the GPU**.

## 21.1 Which is exactly why MoE is the right ANE workload, and dense is not

§18 leaned the wrong way. Ranking the two by what actually binds them:

| workload | bytes read per token | bound by | ANE verdict |
|---|---|---|---|
| dense 27B decode | all 27 GB | **bandwidth** | ~1.75x slower, unfixable |
| MoE decode (top-8 of 256) | ~3% of weights | compute / dispatch | **2.4-3.8x faster** (§20) |

Dense decode is precisely the workload that leans on the ANE's single weakest
property. MoE decode barely touches it — 8 of 256 experts is ~25 MB per layer,
which at 150 GB/s is well under the compute time. That is why the wide+sparse+
fused layer in §20 beats the GPU while dense 27B cannot.

**So the conclusion inverts: sparsity is not a problem for the ANE, it is the
reason to use the ANE.** The ANE has compute (15.4 TFLOP/s measured) but not
bandwidth; MoE needs compute and little bandwidth; dense needs bandwidth above
all. The 27B belongs on the GPU for decode, and on the ANE only for prefill,
where it is compute-bound.


---

# 22. RESULT: MoE on ANE is faster AND uses 2.66x less energy

Measured on the deployable configuration — wide + sparse + fused int8 layer
(§20) with the per-token expert staging real routing requires (§21) — against
MLX SwitchGLU int4. `ane_moe_energy_v2.py`, powermetrics, one engine at a time.

| engine | ANE mW | GPU mW | CPU mW | layers/s | mJ/layer |
|---|---|---|---|---|---|
| **ANE** | 1562 | 849 | 18063 | **1744.8** | **11.73** |
| GPU | 0 | 36784 | 14476 | 1644.0 | 31.18 |

- **Speed: ANE 0.573 ms vs GPU 0.608 ms** — slightly faster, with staging included.
- **Energy: 2.66x less per layer** (11.73 vs 31.18 mJ).
- Faster *and* lower energy means it wins at **every** platform baseline; unlike
  §13 there is no crossover to argue about.

## 22.1 The ANE is nearly free; the copy around it is not

`ANE 1562 mW` versus `CPU 18063 mW`. **The accelerator costs a twelfth of what
the staging memmove costs.** The GPU arm's 36784 mW is doing the actual matmuls;
the ANE arm's power is dominated by moving 25 MB per layer on CPU threads.

This is where the remaining headroom is, and it is large:

- If staging were free, the ANE layer would draw ~2.4 W against the GPU's ~51 W
  total — an order-of-magnitude energy difference rather than 2.66x.
- Both arms run a Python busy-loop, which inflates CPU in *both*, so the 2.66x is
  if anything conservative for the ANE (the GPU arm's 14476 mW is mostly loop
  overhead, while the ANE arm's 18063 is loop **plus** real copying).

Ways to attack it, none yet tried:
1. **Do the gather on the GPU** (a Metal blit into the IOSurface) instead of CPU
   threads — the GPU is far more efficient at bulk copies, and it is otherwise
   idle in this configuration.
2. **Avoid the copy**: contiguous expert layouts, so the weight surface can be
   bound at an offset into the resident pool rather than assembled. Requires
   co-activated experts to be adjacent — i.e. reordering experts by co-activation
   clustering, which the routing capture in §21 has the data to evaluate.
3. In-graph concat of weight inputs would have been the clean answer but is
   **rejected** by the compiler (§21).

## 22.2 Status

**MoE inference on the ANE works, is numerically correct, and beats the GPU on
both speed and energy.** What exists is the layer: one fused dispatch, 8 stacked
experts, int8, real weights, verified against fp32.

What a full engine still needs:
- **Attention stays on the GPU.** `linear_attn` (Mamba-style) does not map to
  1x1 conv; this was never solved.
- Wiring the layer into oMLX's forward pass, with the router driving staging.
- The S>=32 floor means decode must be padded or batched (§14).
- Multi-prompt validation: §21's locality numbers come from a single 64-token
  generation.


---

# 23. Attacking the 18 W staging copy: three attempts

§22 showed the ANE itself draws 1562 mW while the staging memmove costs
18063 mW. Three ways at it were tried.

## 23.1 GPU Metal blit — works, but does NOT help (measured, see §24)

Both the resident expert pool (805 MB) and the destination are wrapped with
`newBufferWithBytesNoCopy:` — unified memory, so no data moves to create them —
and the 8 expert blocks are copied by one blit encoder.

| staging | ms | GB/s | layer total | vs GPU |
|---|---|---|---|---|
| cpu memmove | 0.381 | 66 | 0.625 | 0.96x |
| cpu memmove x4 threads | 0.315 | 80 | 0.567 | 1.06x |
| **GPU metal blit** | 0.353 | 71 | **0.510** | **1.18x** |

Verified byte-exact against the source. Bandwidth is a wash — all three sit near
the ~70-85 GB/s copy ceiling — but the blit moves 25 MB/layer of traffic off the
CPU and onto the GPU's DMA engine, which is where the 18 W lives.
`ane_moe_energy_v3.py` measures it (needs sudo).

## 23.2 Expert co-activation clustering — FAILS

If co-activated experts were adjacent, the weight surface could be *bound* at an
offset instead of assembled: zero copy. Captured real routing over 4 prompts /
11040 routing vectors / 40 layers, built the co-activation matrix, and ordered
experts by the Fiedler vector of its normalised Laplacian.

| block size | blocks touched, random order | clustered |
|---|---|---|
| 8 | 7.28 | **5.65** |
| 16 | 6.48 | 4.47 |
| 32 | 5.25 | 3.29 |

Clustering helps (7.28 -> 5.65) but nowhere near enough: 5.65 baked dispatches is
1.378 ms/layer against 0.544 ms for assemble-then-dispatch, and 0.600 for the GPU.

**MoE routers are trained to spread load**, so the routing is genuinely diffuse —
there is no tight co-activation structure to exploit. This approach is dead.

## 23.3 In-graph concat of weight inputs — INCONCLUSIVE

The clean answer would be N weight inputs concatenated in-graph, each bound to an
offset in the resident pool. §21 recorded this as "rejected", but the follow-up
differential rejected **all four** arms including `concat` on the activation
path — which is known to work from earlier probes. So the multi-input test
harness is broken, not the compiler.

**This is unresolved, not disproven.** Worth redoing with a correct harness,
since it is the only route to genuinely zero-copy expert assembly.

## 23.4 Where staging stands

- Best measured: **GPU blit, 0.510 ms/layer, 1.18x the GPU on speed**, with the
  copy off the CPU.
- The copy itself cannot go much faster — 66-85 GB/s across CPU, threads and GPU
  DMA alike looks like a unified-memory copy ceiling, not an implementation flaw.
- Eliminating the copy entirely needs §23.3 to work; clustering (§23.2) cannot
  deliver it.


---

# 24. The blit measured: power moves, it does not shrink

`ane_moe_energy_v3.py` (Metal blit staging) against v2 (CPU memmove x4):

| run | ANE mW | GPU mW | CPU mW | total mW | layers/s | mJ/layer |
|---|---|---|---|---|---|---|
| v2, cpu memmove x4 | 1562 | 849 | 18063 | 20474 | 1744.8 | **11.73** |
| v3, GPU metal blit | 1419 | 3381 | 15679 | 20479 | 1530.3 | 13.38 |

**Total power is identical** — 20474 vs 20479 mW. The blit took ~2.4 W off the
CPU and put ~2.5 W onto the GPU, essentially one for one, then lost ~12%
throughput to blit submission and `waitUntilCompleted`. Net: worse, 2.23x vs
2.66x energy advantage.

**Use v2 (CPU memmove across 4 threads).** The blit is correct and available, but
there is no reason to prefer it.

## 24.1 This corrects §22's diagnosis

§22 read `CPU 18063 mW` next to `ANE 1562 mW` and concluded the staging copy was
twelve times the cost of the accelerator, with an order of magnitude of headroom
if it could be eliminated. **That was wrong.** Removing the memmove entirely —
which the blit does — dropped CPU by only 2.4 W.

So the ANE arm's CPU is dominated by the Python busy-loop and the driver submit
path, not by the copy. The copy is worth perhaps 2-6 W of the ~20 W total. The
projected "~2.4 W if staging were free" does not exist.

Note also the run-to-run variance: the GPU arm's CPU reads 14476 mW in v2 and
9506 mW in v3 for identical work. Differences of a few watts between these runs
are not meaningful; only the ~2.2-2.7x gap is.

## 24.2 Standing result

**MoE on ANE: ~2.2-2.7x less energy than the GPU at comparable speed** (ANE
0.573 ms vs GPU 0.608 in the best configuration). It wins at every platform
baseline. That is the deliverable, and it is not going to improve much further by
attacking staging — the remaining CPU is harness overhead that a real engine
would not have in the same form, and the copy itself sits at a unified-memory
ceiling of 66-85 GB/s no matter which engine performs it.


---

# 25. Working ANE MoE inference — and the honest cost

## 25.1 The harness bug that invalidated §21/§23.3

`concat` of weight inputs was never rejected by the compiler. The engine requires
the weight input to be named **`wimg`**; the probes used `w0`, and
`AneDynamicLinear.compile` returned None with
`ANE dynamic unexpected input symbols ['w0', 'x']`. Every "rejected" result in
§21 and §23.3 was that, not a MIL limitation.

## 25.2 In-graph slicing of weights is catastrophic

The full single-dispatch layer (gate+up+down+router weights packed into one
weight input, sliced in-graph) **compiles and is numerically correct**
(3.67e-02) but takes **353 ms** — a thousand times slower than baked. This
matches §19.4, where one-blob slicing was 2x slower at K=1. The compiler cannot
fold slices on the weight path; weights must arrive already shaped.

## 25.3 The deployable dynamic path

Weights pre-shaped, two dispatches per layer (gate|up fused into one wide conv,
then down), router weights applied to the activation in between:

| | ms/layer | vs GPU |
|---|---|---|
| baked, **fixed** experts (§20) | 0.244 | **0.42x — 2.4x faster** |
| dynamic, **real per-token routing** | 2.076 | **3.60x slower** |
| MLX SwitchGLU int4 | 0.576 | — |

Dynamic weights arrive as feature maps rather than compiler-optimised blobs:
1.209 ms of dispatch versus 0.244 baked, plus 0.867 ms staging 50 MB.

**The 2.4x win requires pinned experts.** §21 showed consecutive tokens share
only 22% of their top-8, so real routing does not provide that. This also means
the §22/§24 energy result overstated the case: it used baked compute (0.244 ms)
with realistic staging, a combination that does not exist.

## 25.4 End-to-end: it runs, and it is correct

`ane_moe_inference.py` replaces layer 20's `Qwen3NextSparseMoeBlock` with an ANE
implementation — routing on GPU, expert math on ANE, decode padded from S=1 to
the S>=32 floor — during real generation.

**Token agreement with the GPU baseline: 24/24.** Identical output.

Cost: **6.11 ms per token-position** against the GPU's ~0.5 ms. Worse than the
2.076 ms of §25.3 because the per-token weight assembly converts fp32 to fp16 out
of a 1 GB array; a real engine would keep an fp16 pool and memcpy. The tok/s
figures printed by that script are **not** a valid comparison — the baseline run
absorbs model warmup.

## 25.5 Where this actually stands

- **Correctness: proven.** A real MoE layer runs on the ANE inside real
  generation with identical tokens.
- **Speed: the ANE loses at ~3.6x** for real routing, and wins 2.4x only for
  fixed/pinned expert sets, which routing does not supply.
- **The blocker is dynamic weight bandwidth**, not dispatch, not width, not
  gather. Baked weights get an optimised layout; feature-map weights do not.

Worth trying next, in order:
1. **int8 dynamic weights** — halves the 50 MB staging and may narrow the
   feature-map penalty. Untested for the dynamic path.
2. A **resident fp16 expert pool** with memcpy staging (the current code converts
   dtypes per token, which is most of the 6.11 ms).
3. Whether the feature-map weight penalty is layout-dependent — if the surface
   can be pre-tiled the way the compiler tiles blobs, the 2.2x dynamic penalty
   might disappear, which would make the whole thing win outright.


---

# 26. FINAL: baked wins per-dispatch, loses to routing spread

Baked is far better than dynamic (0.244 vs 2.076 ms/layer) and needs no staging
copy at all. Two things stop it being the answer.

## 26.1 Hard ceiling of 127 resident programs

`ane_program_ceiling.py`: **127** stacked-8 programs load, then compilation
fails. Not blobs (762 were fine), not memory (3.2 GB of 128). It looks like a
hard model-count limit.

127 groups = **4.0 layers** of a 256-expert model. A 40-layer model needs 1280
groups. Packing 5 fp16 groups per program via procedures raises it to ~635
groups (~20 layers), and baking only the hot 128 experts per layer (99.5%
coverage, §21) needs 640 — just about reachable. So residency is tight but not
the real blocker.

## 26.2 The real blocker: a token's experts span ~7 groups

Baked groups are *fixed* sets of 8. Real routing picks 8 arbitrary experts, which
land in ~7.3 different groups (5.65 after co-activation clustering, §23.2), so
the layer needs that many dispatches. Measured:

| groups dispatched | serial | 4-thread | vs GPU 0.576 |
|---|---|---|---|
| 1 | 0.245 | 0.253 | **2.35x faster** |
| 2 | 0.495 | 0.468 | **1.23x faster** |
| 4 | 0.983 | 0.815 | 0.71x |
| 6 | 1.473 | 1.163 | 0.50x |
| **7 (real routing)** | 1.739 | **1.348** | **0.43x — 2.3x slower** |

**The crossover is 2 groups.** The ANE wins if and only if a token's top-8 comes
from at most 16 specific experts. Real routing gives 58 (7.3 x 8).

## 26.3 Complete picture

| configuration | ms/layer | vs GPU | exists? |
|---|---|---|---|
| baked, all 8 experts in one group | 0.244 | **2.4x faster** | only with fixed routing |
| baked, 2 groups | 0.468 | 1.2x faster | needs group-limited routing |
| baked, 7.3 groups (real) | 1.348 | 2.3x slower | **yes** |
| dynamic staged weights (real) | 2.076 | 3.6x slower | **yes** |
| MLX SwitchGLU int4 on GPU | 0.576 | — | yes |

Best real option is baked groups at 1.348 ms — still 2.3x slower than the GPU,
but notably it needs **no staging copy**, so its energy profile is much better
than the dynamic path and is worth measuring if power matters more than latency.

## 26.4 The one thing that would make this win

The ANE is fast enough — 2.35x the GPU when the experts are in one group. The
mismatch is purely that routing is unconstrained. **A model trained with
group-limited routing** — top-8 drawn from a small number of fixed blocks, which
is a real technique (DeepSeek's device-limited routing is the same idea for a
different reason) — would land in the 1-2 group regime and beat the GPU by
1.2-2.4x with zero staging.

That is a model-side change, not something achievable by tuning the ANE path.
Everything on the ANE side is now understood and measured; the remaining gap is
in how the router is trained.


---

# 27. FINAL, measured end to end: dense beats MoE on the ANE, by a lot

Everything below is the real model in one process, A/B'd with a single flag
(`tools/ane_serve.py`), not composed from per-layer constants.

## 27.1 MoE, Qwen3.6-35B-A3B (bf16)

| MoE layers on ANE | tok/s | ms/token | marginal/layer |
|---|---|---|---|
| 0 (GPU baseline) | **53.0** | 18.9 | — |
| 8 | 17.7 | 56.5 | +4.7 ms |
| 16 | 10.8 | 92.6 | +4.5 ms |

~4.6 ms per MoE layer per token, consistently. All 40 layers projects to
~203 ms/token, **4.9 tok/s against the GPU's 53 — about 11x slower**.

Only 16 layers could be measured directly: expert pools cost 1.6 GB/layer, so 40
would need 64 GB on top of the 65 GB model, over the Metal cap.

## 27.2 Dense, Qwen3.8-27B

| config | tok/s | rel err vs bf16 |
|---|---|---|
| GPU 4-bit AWQ | 24.3 | 1.07e-01 |
| GPU bf16 | 8.7 | 0 |
| ANE int8 baked from bf16 | 4.4–4.8 | **9.7e-03** |

Accuracy-matched this is ~2x, not 5x: int8-from-bf16 is effectively bf16 quality
and the GPU needs bf16 to match it, where it runs 8.7.

## 27.3 The correction: sparsity favours the GPU, not the ANE

§18 and §26 argued MoE was the better ANE target because it reads only ~3% of
weights per token. **That was wrong**, and the end-to-end numbers show why.

MLX reads the 8 routed experts *in place* through `gather_qmm` at no extra cost.
The ANE cannot: it has no gather, so those experts must be physically staged into
a weight surface first — 50 MB per layer per token, paid twice (copied in, then
read out). The GPU's MoE decode at 18.9 ms/token is *faster than its own dense
27B decode*, precisely because skipping 97% of the weights is free for it.

Sparsity is an advantage the GPU can take and the ANE cannot.

## 27.4 Where the ANE actually stands on this machine

- **Dense, weight-heavy, one dispatch per layer** is its best shape: ~2x behind
  the GPU accuracy-matched, and baking is clean (no staging, 64 layers = 64
  programs, 17 GB of int8 blobs, ~70 s).
- **MoE is its worst shape**, because expert selection forces a copy the GPU
  never makes.
- Neither wins on speed. The remaining reason to care is power, which is
  measured per-layer (§22, §24) but never end-to-end on these configurations.

---

## 28. Decode on the ANE is dispatch-bound, not bandwidth-bound (M5 Max, 27B dense)

Measured with `tools/ane_serve.py --dense-layers 16 --bench`, decode and prefill
counted separately (`ANE decode` / `ANE prefill` lines):

| weights | ANE decode (ms per layer-token) | ANE prefill (ms per layer-token) |
|---|---|---|
| int8 (4.28 GB / 16 layers) | 1.989 | 0.095 |
| int4 (2.14 GB / 16 layers) | 1.941 | 0.098 |

**Halving the weight bytes moved decode by 2.4%.** If decode were bandwidth-bound
the int4 row would be ~2x faster (134 MB vs 268 MB per layer at the measured
150 GB/s = 0.89 vs 1.79 ms). It is not. The cost is ~0.65 ms per dispatch across
the MLP's 3 convs -- 7x the 0.092 ms dispatch floor measured on small ops.

Consequences, and they overturn two earlier sections:

* **Quantising further cannot help decode.** int4 was the obvious lever and it is
  worth 2%. Any future "go to int2/int3" plan for decode latency is dead on
  arrival for the same reason.
* **The concurrency win in section 27 does not transfer to decode.** ANE/GPU
  overlap was measured on a compute-bound probe. At T=1 the MLP is latency-bound
  and splitting a layer across both engines *loses*: end-to-end 6.7 tok/s split
  vs 7.1 full-ANE vs **8.8 GPU baseline**. The split adds a second engine's
  latency to the critical path without removing work from it.
* **The host round-trip is not the problem.** `--sync-only` (pay the
  `np.array()` MLX pipeline flush, then compute on the GPU anyway) measures
  8.6 tok/s against the 8.8 baseline: the flush costs ~2%. The large "ANE ms"
  figure that mode reports is the timer absorbing already-queued GPU work, not
  overhead -- do not read it as a cost.

**The ANE is ~2x slower than the GPU per MLP layer at decode** (1.94 ms vs the
GPU's ~1.07 ms) and this holds regardless of weight precision. Prefill is the
opposite: 0.095 ms/layer-token, ~20x cheaper per token than decode, because the
work batches into one dispatch.

Standing conclusion: **on M5 Max the ANE is a prefill engine, not a decode
engine.** Effort on decode should go to reducing dispatch *count*, not bytes.

## 29. Why decode cost is invariant: 31 of 32 lanes are discarded

Section 28 found decode cost flat in weight bits. It is also flat in dispatch
count -- fusing gate+up into one 2I-channel conv (the trick already used on the
MoE path, now `--no-fuse-gu` to disable) moved nothing: 1.929 ms separate vs
1.939 ms fused. The reason is neither bytes nor dispatches:

**The hardware refuses widths below S=32, so `AneDenseMLP` pads T=1 up to 32.
Every decode step already computes a full 32-token batch and discards 31/32.**

`artifacts/ane_probes/ane_lane_occupancy.py`, real 27B layer-0 MLP
(gate [17408, 5120], down [5120, 17408]), int4 on ANE vs bf16 on GPU:

| T | ANE ms | GPU ms | ANE us/token | GPU us/token |
|---|---|---|---|---|
| 1 | 2.012 | 2.641 | 2012 | 2641 |
| 4 | 1.871 | 2.718 | 468 | 680 |
| 8 | 1.893 | 2.702 | 237 | 338 |
| 16 | 1.944 | 2.805 | 122 | 175 |
| 32 | 1.998 | 2.677 | 62 | 84 |

**Both engines are flat.** A 32-token step costs what a 1-token step costs, on
either unit, because decode is weight-bandwidth-bound: the layer's weights are
streamed once regardless of how many tokens ride along. (The GPU column is
pessimistic -- it pays a full `mx.eval` sync per iteration, where in-model MLX
pipelines. Do not read 2.68 ms as the in-model GPU cost, which is nearer 1.1 ms.)

Two consequences:

1. **The free capacity is real but it is not ANE-specific.** Filling the 32 lanes
   -- speculative decoding, MTP tree verification, or batching several sequences
   -- is worth up to ~32x on *either* engine. This is the single largest lever
   found in this work, and it does not require the ANE at all.
2. **Moving whole layers to the ANE cannot overlap.** Layers form one dependency
   chain, so "16 MLPs on ANE, the rest on GPU" runs strictly sequentially: it
   moves work to the slower unit and buys nothing. That is exactly the measured
   8.8 -> 7.1 tok/s. Genuine ANE/GPU overlap needs two *independent* chains --
   two concurrent sequences, not two halves of one layer.

## 30. Multi-procedure banks compile but never dispatch (M5 Max)

`AneEngine.compile_procedure_bank` packs N linears into one resident
`_ANEInMemoryModel` as `func procedure000..NNN`, selected per call by
`_ANERequest.procedureIndex`. It is meant to dodge the ~127-resident-program
ceiling. **It silently returns procedure 0's weights for every index.**

Reproducer: `scratchpad/bankcheck.py` -- four random [512,256] weights in one
bank, each evaluated and matched against all four references:

```
proc 0: matches weight 0  OK
proc 1: matches weight 0  *** WRONG ***
proc 2: matches weight 0  *** WRONG ***
proc 3: matches weight 0  *** WRONG ***
```

Everything checks out at the API level, which is what makes it dangerous:

* The MIL is correct -- four `func procedureNNN`, distinct const names
  (`w0..w3`), distinct BLOBFILE offsets (0, 262272, 524544, 786816).
* The loaded `_ANEModel` reports all four: `procedureInfoForProcedureIndex:`
  returns `ANEFModelProcedureID = 3` with its own symbol arrays for index 3.
* `inputSymbolIndicesForProcedureIndex:` returns distinct symbols per procedure
  (proc *i* -> symbol *i*).
* `_ANERequest.procedureIndex` reads back exactly what was passed (0,1,2,3).

Things ruled out by measurement, so do not retry them:

* `kANEFProcedureVariantHint` = 1, 4 or 0 -- no effect.
* `procedureIndex` as a raw integer instead of `NSNumber` -- crashes (NSNumber
  is the correct ABI).
* Evaluation order -- running procedure 3 first on a fresh model still returns
  procedure 0, so it is not a stale cached `_ANERequest`.
* Binding all N symbols at once (`inputs=[surf]*N, inputIndices=[0..N-1]`) --
  rejected with `ANEProgramProcessRequestDirect() ... statusType=0x9 Program
  Inference error`. One surface with `inputIndices=[i]` is the correct shape.
* Bypassing `evaluateWithQoS:` and calling
  `_ANEProgramForEvaluation processRequest:model:qos:qIndex:...` directly with
  `qIndex = i` -- still procedure 0. `qIndex` is a queue index, not a procedure
  selector.

**Workaround: do not slice MLPs into per-projection programs.** The bank existed
because gate/up/down as separate programs is 3 x 64 = 192 resident models,
over the ceiling. One program per layer holding the *whole* MLP (fused gate+up,
silu, mul, down) is 64 -- comfortably under it, and that path
(`AneDenseMLP` in `tools/ane_serve.py`) is measured correct and in use.

## 31. The ceiling: ANE sustains ~10 TFLOP/s, the GPU ~45

`artifacts/ane_probes/ane_lane_occupancy.py`, real 27B layer-0 MLP, int4 on ANE
vs bf16 on GPU, sweeping the compiled program width S:

| S | ANE ms | GPU ms | ANE us/tok | GPU us/tok | ANE TFLOP/s | ANE vs GPU |
|---|---|---|---|---|---|---|
| 32 | 2.548 | 2.912 | 79.6 | 91.0 | 6.72 | **1.14x** |
| 64 | 2.948 | 2.706 | 46.1 | 42.3 | 11.61 | 0.92x |
| 128 | 7.585 | 3.106 | 59.3 | 24.3 | 9.02 | 0.41x |
| 256 | 13.917 | 3.887 | 54.4 | 15.2 | 9.84 | 0.28x |
| 512 | 26.933 | 6.328 | 52.6 | 12.4 | 10.17 | 0.23x |
| 1024 | 58.773 | 12.190 | 57.4 | 11.9 | 9.32 | 0.21x |
| 2048 | 125.046 | 24.654 | 61.1 | 12.0 | 8.76 | 0.20x |

The ANE flattens at **~10 TFLOP/s** from S=64 up and never improves. The GPU
keeps scaling to ~12 us/token, which on this shape (534.8 MFLOP/token) is
**~45 TFLOP/s**. The GPU is ~4.5x the ANE's sustained throughput.

The single crossover is S=32, where the ANE is 1.14x -- and only because the GPU
is latency-bound there, not compute-bound. In-model that GPU overhead disappears
(MLX pipelines across layers), which is why whole-model runs go the other way:

| config | tok/s |
|---|---|
| GPU baseline | **8.8** |
| 16 of 64 MLPs on ANE (int4) | 7.2 |
| **all 64 MLPs on ANE (int4, 8.56 GB)** | **4.3** |
| 16 MLPs split ANE+GPU concurrently | 6.7 |

All 64 layers on the ANE is correct and stable -- it is simply half the speed.
Sections 27's concurrency result does not rescue it: overlap helps only when two
*independent* chains exist, and a single decode is one dependency chain.

**Conclusion, after measuring bits, dispatch count, fusion, width, splitting and
full-model wiring: the ANE cannot beat the M5 Max GPU on throughput for this
model.** Remaining honest uses are (a) perf-per-watt, the original motivation,
(b) running a second independent sequence concurrently with the GPU, and
(c) offloading while the GPU does something else. Throughput-per-token is not
one of them.

## 32. Perf-per-watt: the ANE loses there too (the CPU is the reason)

`sudo tools/ane_power_ab.sh` -- samples `cpu_power,gpu_power,ane_power` while
running the GPU baseline and the all-64-MLPs-on-ANE config back to back, then
attributes samples to each benchmark's wall-clock window.

| window | tok/s | ANE W | GPU W | CPU W | total W | tok/s/W | samples |
|---|---|---|---|---|---|---|---|
| gpu | 8.9 | 0.00 | 3.32 | 5.35 | 8.67 | **1.026** | 41 |
| ane | 4.3 | 0.72 | 1.77 | 7.35 | 9.84 | **0.437** | 125 |

speed 0.48x, power 1.13x, **efficiency 0.43x (tokens per joule)**.

Read the columns, not the total -- they say something more useful than "it lost":

* **The ANE rail is genuinely frugal: 0.72 W** while doing every MLP in a 27B
  model. The silicon is not the problem.
* Moving that work off the GPU did exactly what it should: **3.32 -> 1.77 W**,
  saving 1.55 W.
* **The CPU went up 5.35 -> 7.35 W, +2.00 W** -- more than the GPU saving. That
  is the cost of *driving* the ANE from Python: 8320 dispatches (557/s), each a
  ctypes/ObjC round trip plus a strided fp16 transpose into an IOSurface and
  another back out, ~364 MB/s of host-side copying.

The overhead is attackable (planar activations, batched dispatch, driving the
loop from native code), but it does not change the verdict. **Even at zero host
overhead** the ANE config would be 4.3 tok/s at 1.77 + 0.72 + 5.35 = 7.84 W =
0.55 tok/s/W, still well under the GPU's 1.026. The arithmetic is structural:
during inference this machine's power is **CPU-dominated** (5.35 of 8.67 W), the
GPU rail is only 3.3 W, and halving throughput to save at most ~1.5 W of GPU can
never pay for itself.

**This closes the ANE question on its original motivation.** Throughput was lost
in section 31; perf-per-watt is lost here, and by a wider margin. The ANE is not
a win for single-stream LLM inference on M5 Max at any of the axes measured
(speed, weight precision, dispatch count, program width, layer splitting,
concurrency, or energy).

## 33. CORRECTION to sections 31-32: the ANE is the efficient engine, by 1.7x

Sections 31-32 concluded the ANE was not worth pursuing. **That conclusion was
wrong**, and the error is instructive: both measurements were taken on an
unsaturated decode, where the ANE idles ~85% of the wall clock behind a Python
dispatch loop. 0.43x tok/s/W described our *plumbing*, not the silicon.

Pinning the same 27B MLP on each engine (`sudo tools/tflops_per_watt.sh`,
idle-corrected, CPU cost of driving each engine included in "net W"):

| engine | S | TFLOP/s | ANE W | GPU W | CPU W | net W | TFLOP/W |
|---|---|---|---|---|---|---|---|
| ANE int4 | 32 | 8.49 | 5.69 | 0.02 | 2.29 | 7.40 | 1.15 |
| ANE int4 | 128 | 9.00 | 5.92 | 0.00 | 1.93 | 7.26 | **1.24** |
| ANE int4 | 512 | 10.19 | 6.42 | 0.00 | 2.49 | 8.32 | 1.22 |
| GPU bf16 | 512 | 47.93 | 0.00 | 64.75 | 0.82 | 64.98 | 0.74 |
| GPU int4 | 512 | 47.84 | 0.00 | 83.74 | 0.94 | 84.09 | 0.57 |

**ANE 1.24 vs GPU 0.74 TFLOP/W = 1.68x**, and 2.2x against GPU int4. The GPU
buys its 4.7x throughput with 10x the power (64-84 W against 6-8 W).

Two further corrections to the record:

* **The ANE draws 5.7-6.4 W when saturated, not 0.72 W.** The 0.72 W in section
  32 was an average over a window that included the 33 s bake and ~85% idle.
* **Quantisation helps the ANE and hurts the GPU here.** GPU int4 costs *more*
  power than bf16 (83.7 vs 64.8 W) for the same 48 TFLOP/s -- dequant overhead.

The real problem is duty cycle, not the hardware. In-model each MLP dispatch
takes 1.913 ms wall, while the saturated loop does the identical dispatch in
2.021 ms at ~100% duty. Reconciling the measured 0.72 W against 5.7 W saturated
puts actual ANE-busy time at **~0.95 ms of the 1.913 ms** -- the other half is
host-side: the strided fp16 transpose into the IOSurface, the read back, and the
Python/ctypes round trip.

**So the target is host overhead.** Removing it takes decode from 1.9 ms to
~0.95 ms per layer, which is level with the GPU's ~1.07 ms -- at a tenth of the
power. That is the work: raise duty cycle, do not chase TFLOP/s.

## 34. Working ANE inference: what runs on the ANE today, and how fast

Everything below runs the 27B on the ANE and is measured end-to-end with
coherent output. No GPU comparisons -- these are absolute ANE-path numbers.

| config | tok/s |
|---|---|
| 64 MLPs on ANE (int4) | 4.3 |
| 64 MLPs + lm_head on ANE | 4.1 |
| + MTP speculation, draft 4 | 6.5 |
| **+ MTP speculation, draft 2** | **6.7** |

**lm_head on the ANE** (`--ane-lm-head`): the head is [248320, 5120], untied,
1.27 GB params -> 0.64 GB int4, bakes in 3 s. Too many output rows for one conv,
so it is split along the vocabulary; splitting on output rows is exact (each row
keeps its own scale, no partial sums cross a boundary). Chunk sweep:

| chunks | rows each | ms/position |
|---|---|---|
| 2 | 124160 | **fails to compile** |
| 4 | 62080 | **3.328** |
| 8 | 31040 | 3.474 |
| 16 | 15520 | 3.835 |

4 is the default. The output-channel ceiling for one conv is between 62080 and
124160. On the ANE the head costs 3.33 ms/position against 4.86 ms on the GPU.

**Speculation is unusually valuable on the ANE.** ANE cost is flat to S=32
(section 29), so verifying k+1 draft tokens costs the same 1.9 ms/layer as
verifying one -- the verify pass is free up to 32 tokens. `tools/mtp_specdec.py`
now does **longest-prefix acceptance** (was accept-all-or-one), which took
acceptance from 1.91 to 3.20 tokens/step. Draft depth 2 wins overall because
drafting is sequential: each draft token costs a full MTP layer plus a full
lm_head, so depth beyond ~4 costs more than the extra accepted tokens return.

Run it:

```
tools/ane_serve.py --model <27B> --dense-layers 64 --ane-lm-head --dense-bits 4
tools/mtp_specdec.py --ane-layers 64 --ane-lm-head --dense-bits 4 --draft 2
```

**Still on the GPU:** attention (16 full_attention layers), GDN/linear_attn
(48 layers), norms, embeddings -- together ~116 ms of a 244 ms token. That is
the next target. Program budget matters: 64 MLPs + 4 head chunks = 68 resident
models against a ~127 ceiling, so per-layer projections must be fused (one
conv with concatenated q/k/v/z outputs) rather than baked one per projection.

Two gotchas found while wiring this:

* **Patch the lm_head instance, not its class.** `type(head).__call__ = ...`
  hijacks every `nn.Linear` in the model, including the GDN `in_proj_*` layers,
  which then receive vocab-sized outputs. Use
  `head.__class__ = type("...", (type(head),), {"__call__": patched})`.
* The per-layer host round trip is cheap: `cast 0.001 / write 0.017 /
  submit 1.882 / read 0.018 ms`. **98% of a dispatch is the ANE itself**, so
  there is no host-side win available; gains must come from ANE work per
  dispatch (lanes) or from moving more of the model onto it.

## 35. Maximum ANE coverage: 126 programs, 11.5 GB, everything that fits

```
tools/ane bench --dense-layers 64 --ane-gdn --ane-attn --ane-lm-head --dense-bits 4
```

| block | count | blobs | ms per call | programs |
|---|---|---|---|---|
| dense MLP (fused gate+up, silu, mul, down) | 64/64 | 8.56 GB | 1.920 | 64 |
| GDN input projections (qkv+z+b+a fused) | 42/48 | 1.77 GB | 0.823 | 42 |
| attention q/k/v (fused) | 16/16 | 0.59 GB | 0.743 | 16 |
| lm_head (4 vocab chunks) | 1 | 0.64 GB | 3.238 | 4 |
| **total** | | **11.56 GB** | | **126** |

Coherent output, 3.7 tok/s. ANE time per token: 64x1.920 + 42x0.823 + 16x0.743
+ 3.24 = **172.6 ms** of a 270 ms token.

**The binding constraint is the resident-program ceiling: exactly 126.** Building
64 MLP + 48 GDN + 16 attention + 4 lm_head = 132 fails deterministically -- with
126 already resident, the next `compile_multiproc` returns None ("GDN layer 46:
fused projection [16480,5120] failed to compile", then lm_head failed too).
`ANE_MAX_PROGRAMS = 126` is now a constant, and baking is ordered lm_head ->
attention -> GDN so the cheapest-value block absorbs the shortfall (6 GDN layers
stay on the GPU) instead of losing the head.

**Fusing projections that share an input is the technique that made this fit.**
GDN calls in_proj_qkv/_z/_b/_a on the same tensor, so their rows concatenate into
one [16480, 5120] conv: the qkv module runs the dispatch and stashes the result,
the other three read their slice back. Four GPU matmuls become one ANE dispatch,
and one program instead of four. Same for attention q/k/v. Without this, GDN
alone would need 192 programs.

**Next, and it solves the ceiling rather than working around it:** fuse at the
*layer* level instead of the *linear* level. Within a layer the sequence is
`out_proj -> residual add -> RMSNorm -> gate/up -> silu -> mul -> down`, and the
add and the norm are both ops the ANE already runs. Folding out_proj and the MLP
into a single program per layer would put out_proj (currently GPU, 48+16 of them)
onto the ANE *and* cost zero extra programs -- freeing the 6 GDN layers and
leaving headroom. That is the way to full coverage, since procedure banks (the
other route past the ceiling) do not dispatch (section 30).

## 36. RMSNorm on the ANE, and layer-tail fusion

**RMSNorm runs on the ANE**, but not written the obvious way. Probed in
`artifacts/ane_probes/ane_rmsnorm.py`:

| formulation | result |
|---|---|
| `reduce_mean(axes=[1])` + `rsqrt` | **compile fails** |
| `l2_norm(axes=[1])` | **compile fails** |
| `layer_norm(axes=[1])` | compiles, rel 0.0301 -- but subtracts the mean, so it is not RMSNorm |
| **conv with 1/C weights, `sqrt`, `real_div`** | **OK, rel 0.0015** |

The ANE rejects channel-axis reductions and `rsqrt`, but **a 1x1 conv with
constant 1/C weights is a cross-channel mean**, and `sqrt` + `real_div` stand in
for `rsqrt`. `[1,1,1,S]` broadcasts against `[1,C,1,S]` in `real_div`. The block
costs 0.212 ms at H=5120, S=32 -- and a 64-channel variant that slices is
cheaper still (0.106 ms) if it ever matters.

That unlocked `AneFusedLayer` (`--ane-fused-layers -1`): one program per layer
doing `out_proj -> +residual -> RMSNorm -> gate/up -> silu -> mul -> down ->
+residual`. The two inputs (attention/GDN core output, residual) are
concatenated into one surface and sliced apart in-graph, avoiding multi-input
request plumbing. out_proj is replaced by an identity so the attention module
hands back its pre-projection core.

Cost, built up piece by piece (`artifacts/ane_probes/ane_fused_cost.py`):

| program | ms |
|---|---|
| MLP alone (baseline) | 1.829 |
| MLP with wide input + slice | 1.858 |
| out_proj + add + MLP | 2.026 |
| full fused layer, in situ | 2.140 (submit) |

**This adds out_proj and post_attention_layernorm to the ANE at zero program
cost** -- still one program per layer.

Full build: `tools/ane bench --ane-fused-layers -1 --ane-gdn --ane-attn
--ane-lm-head --dense-bits 4` -> 64 fused tails + 42 GDN + 16 attention +
4 lm_head = 126 programs, coherent output, 3.5 tok/s.

**Debugging note.** The fused layer first measured 30 ms/call, which looked like
a pathological op. It was not: `mx` (the MLX->numpy conversion) was 25 ms of it,
because `np.array(core)` blocks until the GDN forward for that layer evaluates.
That work was always there, merely attributed elsewhere before. In the full
build the same counter reads 0.548 ms. **Beware phase timers that straddle an
MLX sync -- they bill upstream GPU work to whatever forced the evaluation.**

**Next, and it breaks the 126 ceiling instead of fitting under it:** have each
fused program also emit the *next* layer's `input_layernorm` and input
projection, concatenated onto its output. The GDN in_proj programs (42) would
disappear entirely, freeing ~42 slots, and all 48 GDN layers plus all 64
input_layernorms would land on the ANE. Each program then computes the tail of
layer N and the head of layer N+1; it must branch on whether layer N+1 is
linear_attention or full_attention when choosing the projection weights.

## 37. Chained fusion: built, but blocked by a 9216-channel output limit

The plan from section 36 was to have each fused program also emit the *next*
layer's `input_layernorm` and input projection, which would delete the 42 GDN
and 16 attention projection programs and put the whole model comfortably under
the 126 ceiling. It is implemented (`--ane-chain`, `attach_ane_chain`) and it
works -- but not for this model's projection widths.

**`concat` does not compile on the ANE. Neither does `stack`.** Emitting two
tensors from one program needs `pad` + `pad` + `add`, which is exact and does
compile. But that path has its own width ceiling:

| output width | pad+add | plain conv |
|---|---|---|
| 6144, 7168, 8192, 9216 | OK | OK |
| 11264, 15360, 21600 | **fails** | OK (lm_head reaches 62080) |

So a chained program can carry at most `H + ~4096` output channels
(`ANE_MAX_CHAIN_PROJ = 4096`). Bisecting confirms the limit is the output path,
not weight volume: with P=16480 the program fails even when the MLP is shrunk to
89 MB of weights, while P=2048 compiles at 155 MB.

GDN's fused projection is 16480 rows and attention's q/k/v ~14450, so **neither
fits**. `attach_ane_chain` therefore falls back per layer: the tail is still
fused, and the next layer keeps its own projection program. The full build is
unchanged at 126 programs, 42/48 GDN layers, 3.4-3.5 tok/s, coherent output.

The chain machinery is correct and will pay off on any model whose input
projection is under ~4096 rows -- it is the width of *this* model that defeats
it. Routes still open for full GDN coverage:

* Split the next-layer projection so only a <=4096-row slice rides along
  (e.g. `in_proj_b`/`in_proj_a` plus part of `z`), leaving a narrower program
  behind. Saves no programs by itself, so only useful combined with something else.
* Find whether the 126-program ceiling is a count or a memory limit -- if
  memory, smaller per-program blobs would raise it. Untested.
* Emit the second tensor through a *conv* rather than pad+add, since conv
  outputs reach 62080. Needs the two halves to share one input, which they do
  not here (`o0` is not a linear function of the normed value... though it is
  recoverable as `nn / ilw * sd`, a per-channel scale times a per-token scalar).
  That last observation may be the way in and is untested.

## 38. [PARTLY SUPERSEDED by section 41] One ANE; the count claim here is WRONG

Two questions settled by measurement.

**There is one ANE, and instance hints do not parallelise it.**
`ioreg` shows a single `ANE0` node. `kANEFAneInstanceHint` accepts 1-4, and the
hybrid code was spreading gate=1/up=2/down=1 across it, but
`artifacts/ane_probes/ane_instances.py` shows that buys nothing -- the same 27B
MLP compiled under hints 1 and 2, then driven from two threads:

```
hint=1 alone         2.034 ms
hint=2 alone         2.023 ms
both concurrently    3.810 ms      <- serialized
serialized would be  4.057 ms
truly parallel would be 2.034 ms
```

1.06x over serial. **Do not plan on two ANE engines on M5 Max.**

**WRONG -- see section 41.** The table below is an artifact of a broken probe:
every program had identical MIL text, and ANECCompile is content-addressed, so
the ANE deduplicated them into ONE resident model and handed it back 500 times.
The limit IS a program count, and it is exactly 127. The single-ANE result above
stands; everything from here to the end of this section does not.

`artifacts/ane_probes/ane_program_limit.py` (since corrected) compiled uniform
programs until failure:

| program | reached | resident | outcome |
|---|---|---|---|
| tiny [128,128] | 400 | 0.00 GB | hit my cap |
| small [1024,1024] | 400 | 0.21 GB | hit my cap |
| [8192,5120] | 200 | 4.20 GB | hit my cap |
| [16384,5120] | **500** | **20.99 GB** | hit my cap |

500 resident programs *appeared* to load fine -- they were the same program 500
times. With `uniq` varying each program's output width so every MIL is distinct,
the corrected probe stops at **127 programs holding just 5.31 GB**, which also
rules out memory independently of the real build. (IOSurface allocation and the
fd limit of 1048576 are genuinely not factors.)

So `ANE_MAX_PROGRAMS` is now an env-tunable guard rail (default 512), not a
hardware constant. But the real build still stops at 127 programs / ~12.6 GB,
and surfacing the swallowed compiler output gives the actual reason -- a **load**
failure, not a compile failure:

```
createProgramInstanceForModel:...memoryPoolID:...optOutOfModelMemoryUnwiring:error::
Program load failure (0x50004)
```

The selector names a memory pool and model-memory *wiring*. So the limit depends
on the **mix** of program sizes and blob counts, not on count or total bytes
alone: 500 uniform 42 MB programs with 2 blobs each are fine, while 64 fused
layers of 150 MB with 11 blobs each plus 63 smaller programs are not. Untested
follow-ups, in the order worth trying:

1. Free the MLX-side weights after baking each block. The full bf16 model stays
   resident next to the blobs, and nothing needs it once a block is on the ANE.
2. Shrink the per-program footprint (the fused layer's 150 MB is dominated by
   the 89 MB gate+up blob) and see whether the reachable count rises.
3. Probe 0x50004 directly by compiling mixed-size programs until it appears,
   to learn whether it tracks total wired bytes, largest program, or blob count.

Current best build is unchanged: 64 fused tails + 43/48 GDN + 16 attention +
4 lm_head, coherent output, 3.3-3.5 tok/s.

## 39. Freeing the MLX weights: 50.4 GB back, and what 0x50004 is not

`--keep-mlx-weights` disables it; freeing is the default whenever a block is
baked. After a module's work moves to the ANE nothing reads its MLX array, so
`_free_mlx` swaps in a 1-element placeholder and `_reclaim` (`mx.clear_cache` +
`gc.collect`) hands the buffer back. **The full build frees 50.4 GB**, output
unchanged, 3.3 tok/s. This deliberately kills the GPU fallback for baked
modules -- that is the trade.

It did **not** move the 0x50004 wall: the build still stops at 43/48 GDN layers,
so host memory pressure was never its cause. The failure is inside the ANE.
Ruled out, all by measurement:

| hypothesis | verdict |
|---|---|
| resident program count | no -- 500 programs load fine |
| total resident bytes | no -- 20.99 GB loads fine |
| IOSurface allocation | no -- same limit with and without `_ensure_io` |
| file descriptors | no -- limit is 1048576 |
| blob files per program | no -- 4 and 12 blobs both reach 260 programs / 10.91 GB |

Blob count does have a separate hard limit: **24 blobs per program fails
immediately** (consistent with the known 16-blob ceiling), but that is a
per-program constraint, not the thing capping the build.

What remains is an ANE memory-pool/wiring limit that depends on the size *mix*:
uniform 42 MB programs reach 21 GB, while this build's 64x150 MB fused layers
plus 63 smaller programs stop near 12.6 GB. The selector that fails names it --
`memoryPoolID` and `optOutOfModelMemoryUnwiring`. Untried: shrinking the fused
layer's dominant 89 MB gate+up blob (e.g. splitting gate and up back into
separate programs trades count for size), or passing
`optOutOfModelMemoryUnwiring` explicitly.

## 40. Lazy load, why quantisation is mandatory, and a prebaked blob cache

**The 0x50004 wall was our own memory footprint.** Instrumenting the build with
free-memory readings showed the machine at **0.2 GB free** mid-bake with RSS
52.4 GB -- the whole bf16 model resident while the ANE tried to wire 12.6 GB of
blobs. Loading with `mlx_lm.load(..., lazy=True)` leaves weights memory-mapped
until touched; baking touches each once and frees it:

| | RSS after fused layers | free during bake |
|---|---|---|
| eager (`--eager-load`) | 52.4 GB | **0.2 GB** |
| **lazy (default)** | **31.4 GB** | **13.2 GB** |

This did not by itself recover the last 5 GDN layers (free still falls to 1.1 GB
by the GDN stage), which means **the ANE wires substantially more than the blob
size** -- 43 GDN programs of 42 MB each consume ~9 GB of headroom, not 1.8 GB.
That ratio is the thing to chase next.

**Why quantise at all, given no fixed memory ceiling?** The ANE does support
fp16 conv weights (the `bits=16` path works), so quantisation is not required
for correctness. It is required for *fit*: int4 -> fp16 is 4x, turning 9.57 GB
of blobs into ~38 GB, and the ANE **wires** its model pages. Quantisation here
buys wired-memory headroom, not disk. Accuracy cost is small -- per-output-channel
int4 scales measure rel 1.5e-3 on a real MLP.

**Prebaking the quantised blobs is worth it, and it is now the default path for
fused layers** (`--bake-cache`). Splitting the bake shows why:

```
quantize only            0.45 s
quantize + MIL compile   0.54 s
=> compile portion       0.09 s   (17%)
```

Quantisation is 83% of the work and is cacheable; the MIL compile is not
(ANECCompile re-runs from MIL every time -- preserving its content-addressed
output saves only 1.1x). Measured end to end:

| run | bake | total startup | cache on disk |
|---|---|---|---|
| cold | 44 s | 52.6 s | 8.9 GB |
| warm | **21 s** | **28.5 s** | reused |

`_quant_blob` takes the weight as a callable so a cache hit never materialises
the fused layer's 713 MB gate+up concat, and refuses to write below
`_CACHE_MIN_FREE` (20 GB) so it cannot fill the boot disk.

## 41. SOLVED: the limit is exactly 127 distinct resident programs

`Program load failure (0x50004)` is a **program-count** limit, not memory, not
blob size, not blob count. Established by prediction rather than correlation:

| build | programs | where it stopped |
|---|---|---|
| 64 fused + 4 head + 16 attn + GDN | 64+4+16+**43** | 127 |
| 64 fused + **8** head + 16 attn + GDN | 64+8+16+**39** | 127 |
| head + attn + GDN first, fused last | 4+16+48+**59** | 127 |

Changing the lm_head chunk count from 4 to 8 moved GDN's stopping point from 43
to exactly 39 -- the prediction the count hypothesis makes and the others do not.
Reordering so the small blocks go first moved the shortfall onto the fused
layers but left the total identical. It fails with 9.8 GB free, so it is not
memory; the earlier "0.2 GB free" reading was a separate (real) problem fixed by
lazy loading and the streaming quantiser.

**Section 38's claim that 500 programs load fine was wrong.** Every program in
that probe had identical MIL text, and ANECCompile is content-addressed, so the
ANE deduplicated them into a single resident model: it measured one program
loaded 500 times. Any future capacity probe must vary the MIL, not just the
weights.

**The fix is a program budget spent in value order.** `ANE_MAX_PROGRAMS = 127`
(env-overridable), and `_bake_small` now allocates what the fused layers leave
by how much weight each program carries -- lm_head ~160 MB/program, GDN ~42 MB,
attention ~37 MB -- so attention absorbs any shortfall instead of GDN:

```
ANE fused layer tails: 64
baked lm_head 248320x5120 in 4 chunks, 0.64 GB
fused 48 GDN layers, 2.03 GB          <- was 43
fused 11 attention layers, 0.40 GB    <- absorbs the shortfall
ANE programs resident: 127/127
```

Everything the model can put on the ANE within the budget is now there:
**64/64 layer tails, 48/48 GDN projections, 11/16 attention, lm_head** --
12.6 GB of int4 blobs, coherent output, 3.5 tok/s.

To go further the *count* has to come down, which means fewer, larger programs:
layer-level chaining (section 37) is the right shape but is capped by the
9216-channel pad+add output width, and procedure banks (section 30) do not
dispatch. Those two remain the only routes to full coverage.

### 41a. Independent confirmation of the 127 limit

`artifacts/ane_probes/ane_program_limit.py` now sets `O = O - uniq` so every
program has a distinct output width and therefore distinct MIL text. Result:

```
... 100 programs, 4.2 GB
... 125 programs, 5.2 GB
stopped at 127 programs, 5.31 GB resident  (COMPILE FAILED)
```

**127 distinct programs, 5.31 GB.** The real build also stops at 127 while
holding 12.6 GB. Same count, less than half the memory -- the limit is the count,
conclusively.

Lesson for any future ANE capacity probe: **vary the MIL text, not the weights.**
Identical MIL is content-addressed to a single resident model, and a probe that
does not vary it measures nothing.

## 42. SOLVED: two output tensors per program -> 68 programs, every projection on the ANE

The 127-program limit is no longer binding. A MIL func can return **two**
tensors, and `_ANERequest` takes an array of outputs -- we had only ever passed
one. So each layer's program now emits `y` (the layer output) and `y2` (the NEXT
layer's input projection) through separate surfaces, with no width limit at all.
That deletes the 48 GDN and 16 attention projection programs outright.

```
tools/ane serve --ane-chain --ane-lm-head --dense-bits 4
  chained 64 layers (63 with the next layer's head folded in), 12.14 GB int4
  ANE programs resident: 68/127
  Tokyo. 161.        <- correct, 3.8 tok/s
```

| | before | after |
|---|---|---|
| programs | 127/127 (full) | **68/127** |
| GDN projections | 43/48 | **64/64 folded** |
| attention q/k/v | 11/16 | **folded** |
| free slots | 0 | **59** |

Four things had to be right, and each failed loudly first:

* **The compiler reorders outputs.** The MIL declares `-> (y, y2)` but the model
  reports `y2@output` as symbol 0 and `y@output` as symbol 1. Binding by
  position gives each surface the wrong size and inference fails with
  `status=0x1d`. `_build_two_output_request` parses the model description and
  binds by *channel count*. The regex needs a `(?!Channels =)` lookahead or it
  runs from the input's Channels to the first output Name and reports the input
  width.
* **`identity` as a program's SOLE output returns NaN**, so the single-output
  build names the residual add `y` directly. With two outputs the identity copy
  is required instead -- emitting the add as an output while the norm branch
  also consumes it likewise gives NaN. Two opposite rules, both measured.
* **`mx.array` on a strided numpy view takes a CPU path.** Slicing the projection
  out of the second surface needs `np.ascontiguousarray`.
* **MLX streams are per-thread.** `ThreadingHTTPServer` handed each request to a
  fresh worker and generation died with "There is no Stream(cpu, 0) in current
  thread". Binding a stream in the handler was not enough (the ANE path builds
  arrays deeper in the stack), so the server is now single-threaded -- decoding
  is sequential anyway with one ANE and one KV cache.

**A caution about synthetic tests.** While debugging this I chased a NaN for
several rounds that was an artifact of my own probe: random Gaussian weights
through a 17408-wide MLP overflow fp16. The single-output path was reported NaN
by the probe while the real model generated coherent text throughout. Verify ANE
numerics against real weights, not random ones.

Still on the GPU: layer 0's input projection (no predecessor to fold it into),
the GDN recurrence (conv1d + gated delta rule), the attention core, and
embeddings. With 59 free program slots there is now room for all of it.

## 43. Every projection on the ANE; the GDN recurrence is feasible

Layer 0 has no predecessor to fold its head into, so `attach_ane_chain` now
bakes it as one extra program. **69/127 programs, and every linear projection in
the model runs on the ANE**: 64 layer tails (out_proj + residual + RMSNorm +
MLP + residual), 63 folded next-layer heads, layer 0's own head, and lm_head.
12.19 GB of int4 blobs, coherent output.

Still on the GPU: the GDN recurrence (conv1d + gated delta rule), the attention
core, and embeddings. **58 free program slots.**

The gated delta step, per layer per token, with state `[H=48, Dv=128, Dk=128]`:

```
state  *= decay                    elementwise
kv_mem  = (state * k).sum(-1)      reduction over Dk
delta   = (v - kv_mem) * beta      elementwise
state  += k * delta                rank-1 outer product
y       = (state * q).sum(-1)      reduction over Dk
```

`artifacts/ane_probes/ane_gdn_ops.py` checks every op it needs, with state laid
out as `[1, Dk, 1, H*Dv]` so the Dk reduction falls on the channel axis:

| op | needed for | result |
|---|---|---|
| ones-conv reduction over Dk (S=6144) | kv_mem, y | OK |
| `mul` of two dynamic inputs | state * k | OK |
| `matmul` dynamic x dynamic | reductions as matmuls | OK |
| `transpose` 4d | layout | OK |
| `reduce_sum axes=[1]` | the natural spelling | OK |
| `exp` | decay gate | OK |

**All of it compiles.** Note `reduce_sum` over the channel axis works at
C=128/S=6144 while section 36's `reduce_mean` failed at C=512/S=32 -- channel
reductions are shape-dependent, not categorically rejected. Re-test rather than
assuming.

Shape of the build, when it happens: one program per GDN layer (48, well inside
the 58 free slots), taking state plus q/k/v/decay/beta as inputs and returning
`(y, new_state)` -- the two-output mechanism from section 42. State traffic is
48x128x128x2 bytes in and out per layer per token = 147 MB/token across 48
layers, ~2.1 ms/token at the measured 70 GB/s IOSurface rate.

## 44. The GDN recurrence runs on the ANE (verified)

`artifacts/ane_probes/ane_gdn_step.py` implements a full gated-delta step as one
ANE program and checks it against the numpy reference:

```
compiled: in [6240,160] -> y [48,128] + state [6144,128]
output symbol order (channels): [6144, 48]
y     rel 0.0009  OK
state rel 0.0009  OK
```

**The layout is what makes it work.** Store `state[h, dv, dk]` at channel
`h*Dk+dk`, width `dv`. Then:

* the sum over Dk is a **grouped conv** with ones weights (groups=H, Dk->1),
  verified rel 0.0004
* broadcasting delta back over Dk is a **grouped conv** the other way
  (groups=H, 1->Dk), rel 0.0003
* `k`, `q`, `decay`, `beta` become width-1 columns broadcast across Dv, rel 0.0008

All six tensors ride in on one surface (state in columns 0..Dv-1, decay/k/q in
columns Dv..Dv+2, v and beta in extra channels); `y` and the new state leave on
two output surfaces via the section-42 mechanism.

**Two traps, both of which cost a debugging round:**

* **Row stride must be a multiple of 64 bytes.** Width 131 (262 bytes) compiles
  fine, builds a valid request, and then fails `evaluate` with status 0x1d.
  Padding the input to width 160 (320 bytes) fixed it. Every previously working
  program happened to satisfy this by accident (32 -> 64 bytes, 1024 -> 2048).
* **A program whose output is narrower than its input** trips the same 0x1d
  through `_ensure_io`, which sizes the output surface from `seq_len`. Allocate
  output surfaces explicitly when the widths differ.

Also worth recording: `reduce_sum(axes=[1])` compiles at C=128/S=6144 while
section 36's `reduce_mean(axes=[1])` failed at C=512/S=32. Channel reductions are
shape-dependent -- re-test rather than assuming they are banned.

Remaining to wire this into the model: the causal `conv1d` before the recurrence
(depthwise, kernel 4 -- a grouped conv the ANE already supports), the `g` gate
(`-exp(A_log) * softplus(a + dt_bias)`; subsequently solved with the fp16-safe
`t*P5(t)` formulation in `docs/FULL-ANE-FEASIBILITY.md`),
`RMSNormGated` on the output, and a prefill path (the program is a single step,
so T>1 loops it). Budget is not a concern: 48 GDN programs against 58 free slots.

## 45. Persistent GDN state no longer round-trips through MLX

`tools/ane_serve.py::AneGdnStep` now owns a compact `[6144,128]` IOSurface for
each live cache marker. The full state is imported once after prefill. On every
later decode step the compact state is copied into the first 128 columns of the
accepted `[6240,160]` program input, and `new_state` is bound directly back to
the cache-owned surface. The server returns the original MLX cache array only
as an opaque identity marker; it never reads or transposes the evolved state.

Measured M5 Max results:

```
1.57 MB IOSurface row copy                         0.036 ms
resident gates+recurrence (copy+write+dispatch+y) 0.344 ms
old state round trip                              4.393 ms
GPU recurrence reference                         ~0.66 ms
two-step state relative error                     0.00106
```

The resident program now takes raw a/b and computes the fp16-safe polynomial
softplus, decay, and explicit exp/divide beta in the same dispatch; the
supported decode path no longer calls MLX for gate arithmetic.

Literal zero-copy ping-pong remains rejected by the private compiler. Separate
ordinary activation inputs, width padding/projection, compact reshape packing,
and a grouped-convolution parameter demultiplexer all return
`InvalidMILProgram`. That is now an optional compiler investigation rather than
a performance blocker: recurrent-state arithmetic and storage are GPU-free.
