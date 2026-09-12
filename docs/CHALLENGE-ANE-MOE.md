# Challenge: get the routed MoE off the GPU (or prove it cannot be done)

**Status:** open. Attempted repeatedly, failed every time, each time at a
different layer of Apple's stack. This is the highest-value unsolved problem on
the Flash-Next Neural Engine port, and the one most likely to be genuinely
blocked rather than merely hard.

---

## 1. The goal

Serve **Qwen3.8-Flash-Next** (`qwen4_exp`, 125B total / ~5B active) on an
M5 Max Neural Engine, optimising **tokens per joule**, not tokens per second.
The ANE tops out at 7 W; the GPU under load is roughly 30-40 W. So the ANE path
wins on energy at a fraction of the GPU's throughput — but not at any
throughput.

| | |
|---|---|
| MLX GPU, warm, 600 tokens, MTP on | **42.3 tok/s** |
| MLX GPU, same, no speculation | 32.6 tok/s |
| This ANE path, decode | **4.3 tok/s** (~220 ms/token) |
| This ANE path, prefill (k=16 chunks) | ~27 ms/token |

Correctness is established: the ANE path reproduces BF16 greedy output exactly
on the bare configuration.

## 2. The problem in one paragraph

Decode alternates devices **every layer**: the ANE runs attention (36 gated
delta net layers + 12 sparse attention layers), then the GPU runs that layer's
top-10-of-512 routed experts, 48 times per token. Each handoff costs both
devices a restart. The result is that every component costs 1.4-2.8x more
in the real loop than the same component measured standalone.

**SUPERSEDED — see `CHALLENGE-ANE-MOE-RESULT.md`.** Most of the inflation
below was a dtype bug, not device alternation: `gather_qmm` cast the entire
512-expert scale/bias bank on every call because activations were fp16 and
scales bf16. With matched dtypes:

| component | standalone | in-loop | inflation |
|---|---|---|---|
| routed MoE, per layer | 0.755 ms | 1.13 ms | **1.50x** |
| ANE, 48 layers | 83 ms | 119 ms | **1.43x** |

Alternation is real but worth ~54 ms of a 185 ms token (29%), not the dominant
factor. **The largest single item is now the ANE's own fp16 weight streaming:
83 ms, 45% of decode.** Original (wrong) figures kept below for the record:

| component | standalone | in the decode/prefill loop | inflation |
|---|---|---|---|
| 36 GDN layers, per 16-token chunk | 157.7 ms | 227 ms | 1.44x |
| QSA, per layer | 1.995 ms | 5.65 ms | 2.83x |
| routed MoE, per layer | 1.138 ms | 3.07 ms | 2.70x |
A bare numpy -> GPU -> numpy round trip costs **0.45 ms**, and a GPU that has
just idled through a 2-4 ms ANE submit costs **0.4-0.8 ms extra** to restart
(measured: 0.378 ms/call at zero gap, 0.695 at 2 ms, 0.780 at 4 ms).

**Solve the alternation and all three inflations should collapse at once.**

## 3. What the MoE actually is

Per layer, per token:

* router: `hidden(2560) @ gate.weight.T` -> 512 logits -> softmax -> top-10,
  renormalised. Selection **changes every token**, so expert weights cannot be
  baked into a graph.
* `gate_up_proj`: `(512, 1280, 2560)` — `[expert, 2*I, H]`, 4-bit, group 64
* `down_proj`: `(512, 2560, 640)` — `[expert, H, I]`, 4-bit, group 64
* SwiGLU over the 10 selected experts, weighted by the renormalised scores
* plus one dense always-on shared expert (already moved onto the ANE, +0.08
  ms/layer, `SharedExpert` in `probes/flashnext_multitoken_step.py`)

Bytes touched per layer per token: **24.6 MB** of 4-bit weights (10 experts x 3
matrices). Across 48 layers: **1.18 GB/token**. The full resident bank is
68.4 GB.

## 4. Everything already tried, and the exact failure

| approach | result |
|---|---|
| Core AI `GatherMM` composite, ANE-preferred | compiles, but the compiler assigns the indexed matmul to **GPU** and emits an ANE validation error; 33 ms/layer |
| same, GPU-preferred | 2.37 ms — works, but that is the status quo |
| Core AI `GatherMM`, int32 indices | ANE validation rejects; uint16 indices required |
| private MIL `gather_nd` + `batch_matmul` fused | Espresso rejects the graph |
| MIL gather + multiply + reduce_sum | rejected |
| MIL dynamic-convolution weights | rejected |
| standalone ANE gather at `[64, 1280, 2560]` | **works**, rel 2.1e-4 — but 32.42 ms vs 1.96 ms for a host gather |
| `coreai_opt` int8 weights, per-channel and per-tensor | `Compiler internal error: Codegen Error: Failed to HandleANELayer` |
| 4-bit palettize (Core AI) | compiles, ANE-resident, **no bandwidth gain** (1.18 vs 1.11 ms at 96 MB) — weights are expanded before the conv |
| CPU vectorised int4 GEMV (`FLASHNEXT_MOE=q4gemv`) | 82-112 ms/token, same as GPU; not compute-bound either |
| GPU keepalive thread to defeat the wake penalty | 0.667 -> 0.483 ms at a 2 ms gap, nothing at 0 or 4 ms; and it burns the energy budget |

**Ruled out by project decision, do not revisit:** baked per-expert graphs,
DynBank / 400-submit MIL GEMM RPC, paging routed expert weights in as ANE
activations, writing int8/AWQ into the BF16 checkpoint tree.

The recurring signature is `Codegen Error` **inside** `ANECCompile`. When the
failure is in Apple's compiler backend there is no Python-side workaround.
Part of this challenge is determining whether that is a hard wall.

## 5. Untested leads

None of these have been tried. They are listed in rough order of promise.

1. **MIL dynamic weights.** `docs/HWX-ISA-SPEC.md` section 9 test [10]
   validates a surface-packed dynamic-weight kernel at rel 5.4e-4. If selected
   expert weights can be bound as a runtime *weight* input rather than an
   activation, the "no baking" constraint disappears. The open question is
   whether the binding can be zero-copy against unified memory — `wrap_ndarray`
   copies, which would move 24.6 MB/layer and defeat the purpose.
2. **One-hot reformulation.** Express the gather as a matmul against a
   `(10, 512)` one-hot selector. Mathematically exact, and matmul is the one op
   the ANE definitely does well. The cost is that it touches all 512 experts,
   so it only wins if the ANE can stream the whole 1.4 GB layer bank faster
   than the GPU streams 24.6 MB — it almost certainly cannot, but the
   arithmetic should be checked rather than assumed.
3. **Expert residency / hot-set paging.** Routing is not uniform. If a small
   hot set covers most tokens, a baked hot-set graph plus a GPU fallback for
   misses could cut alternations by a large factor without solving the general
   problem. Requires measuring expert reuse across real token streams first.
4. **Overlap rather than eliminate.** Under speculative decoding the drafter
   (one layer + lm_head, GPU) is independent of the backbone pass. Pipelining
   those could hide part of the GPU cost. Does not fix inflation, but converts
   some of it into useful work.
5. **Attack the other end.** `pure_step` achieves 70 GB/s where a plain
   convolution chain of the same byte count achieves 171 GB/s. That 2.4x is
   unexplained, model-independent (the same ~11-13 tok/s ceiling shows up on a
   dense 27B), entirely inside our own code, and needs no new architecture.
   **If the MoE turns out to be blocked, this is the fallback target.**

## 6. Success criteria

Any one of these is a win, in descending order of value:

1. Decode below **150 ms/token** with output still matching BF16 greedy.
2. Routed MoE below **1.5 ms/layer measured in the real decode loop** (not
   standalone), 48 layers, correct against the fp32 CPU reference.
3. A demonstration that the per-layer ANE<->GPU alternation cost can be
   removed or hidden, with the inflation table above re-measured.
4. A *rigorous* negative result: the specific Apple-side limitation, the
   minimal reproducer, and what would have to change.

A negative result is genuinely valuable here. Several sessions have been spent
rediscovering the same rejections.

## 7. Ground rules

* **Measure in the loop, not standalone.** Every projection built from
  isolated probe timings has been wrong, by 1.4-2.8x. Instrument the real path.
* **Verify against the production Swift backend** (`~/.mlx128/Rindi-NativeMLX`,
  `Sources/RindiInference/Qwen4Exp.swift`), not `mlx_lm`. The Python port is
  behind it and its multi-token indexer path has never executed
  (`mx.unique` does not exist in this MLX build).
* **Treat the existing docs as lab notes, not truth.** Several recorded
  conclusions were artefacts of how they were tested. Two examples: "QSA layers
  only load with debug specialization" was a full disk, and "decode will see no
  W8A8 benefit" was measured at a width where the chain was compute-bound — at
  decode width int8 weights are worth 1.8x.
* Run numbers over **>=600 generated tokens** and warm the model first; short
  runs are dominated by first-touch paging of a 68 GB expert bank.
* Do not regress correctness. `--prompt-ids 760` must still produce
  `The 2016-17 season marked a pivotal`.

## 8. Environment

```bash
cd ~/Desktop/LLM\ -\ Reap/ane-port
# Core AI / ANE work
/Users/true/.rindi/venvs/coreai/bin/python
# anything touching MLX
PYTHONPATH=/Users/true/.mlx128/mlx/python /Users/true/.rindi/venvs/coreai/bin/python

# the decode loop
FLASHNEXT_MOE=mlxresident FLASHNEXT_HEAD=mlx FLASHNEXT_PREFILL_K=16 \
PYTHONPATH=/Users/true/.mlx128/mlx/python \
/Users/true/.rindi/venvs/coreai/bin/python -u scripts/export_flashnext_coreai.py \
  generate --max-new 12 --prompt-ids 760
```

**Check `df -h /System/Volumes/Data` before starting.** A full disk makes every
`.aimodel` load fail with a generic Objective-C error that looks exactly like an
ANE limitation. It cost most of a day. Keep >30 GB free.

**ANE resources run out around 80 resident models** (`Program load failed — no
ANE resources`). The current decode loads 72.

**Long prefill aborts** at ~64 chunks with
`NDArray+Pool.swift: Failed to allocate storage for NDArray ... sk: ioSurface`.
Core AI pools output surfaces with no binding or release API, and our graphs
report `States: []`. Stateful export via `state_names` is the documented escape
hatch; it converts and runs but produces wrong results (rel 0.255, suspected
read/write aliasing on the mutated buffer) and is 2x slower per submit.

## 9. Map

| what | where |
|---|---|
| decode + prefill loop | `scripts/export_flashnext_coreai.py`, `stage_generate` |
| resident quantized MoE | `runtime/flashnext_mlx_moe.py` |
| GPU wake penalty measurement | described in `FLASHNEXT-COREAI-PLAN.md`, Sep 12 |
| Core AI GatherMM attempts | `probes/flashnext_coreai_gather_mm.py`, `probes/gathermm_flashnext.py` |
| MIL gather attempts | `runtime/ane_lookup.py`, `probes/flashnext_fused_gather_mm.py` |
| MIL op support survey | `probes/ane_mil_gdn_ops.py` |
| W8A8 / int8 on real shapes | `probes/ane_w8a8_projection.py`, `docs/W8A8-PROJECTIONS.md` |
| private MIL engine | `runtime/q38_ane_engine.py` |
| full history and measurements | `docs/FLASHNEXT-COREAI-PLAN.md` |
