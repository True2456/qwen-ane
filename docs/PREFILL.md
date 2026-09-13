# Prefill

Decode confirms two or three tokens a block, so its graphs are baked narrow:
a submit costs about 1.1 ms that does not scale with the token count, and at
K=4 that fixed cost is worth paying. Prefill has no such limit. Every slot
carries a real token, so the right width is the widest graph that compiles.

Three changes, each measured on 511 tokens of prose:

| prefill path | 511 tokens | tok/s |
| --- | --- | --- |
| one token a submit | 66.0s | 7.7 |
| the decode graphs, full width (K=4) | 20.0s | 25.6 |
| the decode graphs at K=8 | 12.8s | 40.0 |
| a second set of graphs at k=32 | 8.7s | 58.4 |

None of it costs quality. Scoring 1024 tokens after a 512-token prefill:
6.6188 one token at a time, 6.5828 at K=4, 6.5574 at k=16, 6.6037 at k=32.

## How wide a graph can go

`probes/mil_wide_prefill.py` prices the unroll on one GDN layer.

| k | ms a submit | ms a token | 36 layers, tok/s |
| --- | --- | --- | --- |
| 1 | 1.32 | 1.32 | 21 |
| 4 | 1.78 | 0.45 | 62 |
| 8 | 2.33 | 0.29 | 95 |
| 16 | 3.48 | 0.22 | 128 |
| 32 | 5.81 | 0.18 | 153 |

k is capped at 32 because the emitter's sequence width is 32. Past k=8 two
things had to be fixed first. The recurrent state outputs were named
`q_state0` upward and surfaces bind in alphabetical order, so from ten slots
on every prefix state came back under the wrong name — silently, with the
per-slot outputs still looking right. And the conv-window output is not always
written at the width the graph declares:

| kept window, elements | 4 | 5 | 7 | 11 | 19 | 35 |
| --- | --- | --- | --- | --- | --- | --- |
| declared | 32 | 32 | 32 | 32 | 32 | 64 |
| actually written | 32 | 32 | 32 | 32 | **64** | 64 |

A declared 32 is honoured only while the window is 16 elements or fewer, so
k=16 wrote 64 into a surface sized for 32. Declaring 64 always is honoured at
every width, costs 1.3 MB a layer, and makes the runtime's stride right by
construction.

## Two sets of graphs

`FLASHNEXT_PREFILL_MIL_K=32` builds a second set of GDN and QSA graphs at that
width, walks the prompt through them, and hands the result to the decode
graphs. A GDN layer carries exactly the recurrent state and the conv window
across a pass; the QSA cache and the indexer's blocks were host side already,
so the handover is a copy of two arrays a layer. Chunks of less than the full
width go through the decode graphs instead.

The ANE holds both sets. 132 resident programs were loaded and run in
`probes/mil_two_widths.py`, against a note in the exporter that put the
ceiling near 80, so program count is not the constraint.

Memory is. The prefill graphs export only the last slot's state rather than
one per prefix — speculation needs every prefix so a partly accepted block can
unwind, and a prompt chunk never does — which at k=32 is 1.6 MB of output
surface a layer instead of 50 MB. Even so this is off by default: the second
set is another copy of the baked weights, and building it needs compiler
scratch space.

## Where the time goes

Per token at k=32, on a 511-token prompt:

| | ms a token |
| --- | --- |
| GDN on the ANE | 8.93 |
| routed experts on the GPU | 3.55 |
| QSA | 3.31 |
| router | 1.09 |
| recombine, head, commit, embed | 1.69 |

The floor under the GDN term is the recurrence itself, 0.155 ms a token a
layer, or 5.6 ms a token across 36 layers. That is the state being read and
written through memory once per token. The reference gets past it with a Metal
kernel that holds the state in registers for a whole sequence; the ANE cannot,
so matching it needs the chunked form of the delta rule, where a chunk of C
tokens is matmuls and the state is updated once. That is the next real step on
prefill, and it is worth about 5 of the 18.6 ms a token.

## One program, both widths

Two graph sets meant two copies of the baked weights, 94.1 MB a layer twice,
because a compiled program carries its own. `compile_multiproc` already puts
many procedures in one program for the routed experts, and the two unrolls fit
the same way: procedure 0 at k=4 exporting a state per prefix, procedure 1 at
k=32 exporting only its end-of-chunk state, sharing one weights dictionary.

Two things had to be arranged. The request pairs output surfaces with a
procedure's symbol indices by position, so a procedure declaring fewer outputs
bound to the front of the list and the evaluate failed with no error text;
programs now carry a per-procedure surface map. And the wide procedure's single
state is mapped onto the *last* state surface, which is exactly where the
narrow procedure's `commit` already looks — so there is no handover between
prefill and decode at all, only a `select`.

It is also faster, which was not the point:

| | prefill, 511 tokens | GDN, ms a token |
| --- | --- | --- |
| two programs | 59.7 and 65.2 tok/s | 6.10 and 6.17 |
| one program, two procedures | 78.0 and 82.0 tok/s | 4.76 |

Halving the resident weights is the likely reason. Perplexity is unchanged to
every digit, 1.886208 and 6.5943 scoring 1024 tokens after a 512-token
prefill, and a decode run emits the same ids at 20.7 tok/s.

It costs nothing at startup either, which the first measurement got wrong. The
136s that building 36 two-procedure programs appeared to take was a cold
compile: the ANE keeps compiled model packages in the process temp directory
and reuses them through `compiledModelExists` / `loadWithQoS:`, which
`Q38_ANE_REUSE_COMPILED` leaves on by default, and that cache had just been
deleted to free disk. Warm, the same 36 programs load in 16s and the 12 QSA
layers in 6s, the same as the two separate sets took before.

Prefill across this whole sequence: 7.4, then 25.6 filling every slot, 40 at
K=8, 58 with a second graph set, 65 with the chunked delta rule, and 82 with
both unrolls in one program.

## Where prefill stands at 83 tok/s

Per token, k=32, on a 511-token prompt. Total 11.98 ms.

| | ms a token |
| --- | --- |
| GDN submit | 4.33 |
| QSA (1.45 ANE, 1.11 experts, 0.32 mixer) | 3.09 |
| routed experts after a GDN layer | 2.81 |
| router | 0.49 |
| recombine, head, commit, embed, staging | 1.26 |

Splitting the GDN call settles where its time goes: writing the input surface
is 0.06 ms a token and reading the four outputs back is 0.11, so the remaining
4.33 is the evaluate. There is no host slack left in that path.

Roughly half the pass is ANE submits and a third is the experts on the GPU.

Three things worth trying next, in order.

**QSA still builds two programs.** The GDN layers share one program between the
two unrolls; QSA does not, so its weights are resident twice. Doing the same
for QSA should pay what it paid for GDN, which was 23% off the submit, plus
the memory.

**The experts cost 3.92 ms a token across both layer types.** The resident bank
is close to flat in the token count, so most of that is per-call: 48 host to
device copies, evaluates and syncs a chunk. Keeping the hidden state on the GPU
across route, experts, shared and recombine instead of returning to NumPy each
layer is the obvious thing to try.

**The submit is slower in the run than on the bench.** One layer's chunked k=32
submit is 2.37 ms in isolation and 3.85 ms a layer inside the pass. That gap
across 36 different layers is weight residency, not host code, and halving the
resident weights already bought 23% once.

### Ruled out: overlapping the experts with the ANE

The wavefront that lost at decode widths also loses here, and now for a
measured reason. Chunked submits cost 1.979 ms at k=16 and 2.367 at k=32,
because the chunk arithmetic is fixed at 32x32 whatever the token count. So
splitting a chunk in half to overlap costs 1.59 ms a layer and hides about
1.25 ms of experts. It would need a chunk width that actually scales first.

## QSA Program Sharing and GPU/Host Boundary Optimizations

Two further passes on the k=32 prefill path: sharing compiled programs between
decode and prefill for QSA (matching what GDN already did), and optimizing the
per-layer GPU/host boundary during MoE execution.

### Task 1: QSA Multi-Procedure Program Sharing

The 12 QSA layers previously built separate programs for decode ($k=4$) and
prefill ($k=32$), each with its own baked weight files (`weight_data.bin` and
`weight_scale.bin`). On the 2048 rung and the shared `_front` program, this
duplicated ~500 MB of resident weights on the ANE and required a second compile
pass.

`build_program_multi` in `probes/flashnext_mil_qsa_layer.py` now captures both
widths (`procedure000` at $k=4$, `procedure001` at $k=32$) and splices them into
one MIL program with a single unified weights dictionary.
- **Surface mapping**: Input mask surfaces differ in shape between procedures
  ($G \times 4 \times KV$ vs $G \times 32 \times KV$). `prog.proc_in_map` binds
  surface index 6 to procedure 0 and surface index 7 to procedure 1, while
  sharing input projections, residual surfaces, and rotary buffers (indices 0..5,
  8..9).
- **Output symmetry**: Unlike GDN (which exports intermediate prefix states at
  $k=4$ but only the terminal state at $k=32$), QSA has no recurrent state. Both
  procedures emit the same 6 output tensors (`t_newk`, `u_shared`, `v_mixed`,
  `w_hyper`, `x_inj`, `y_newv`) at identical shape ($S=32$). Output surface
  remapping (`proc_out_map`) was verified unnecessary.
- **Equivalence**: Output parity was verified in `probes/test_qsa_two_proc.py`:
  relative error between the multi-procedure and single-procedure programs is
  **0.000000** across all tensors.
- **Setup & Residency**: Pre-building the multi-procedure rung in `MilQsaLayer`
  drops prefill setup time to 0.0s (`mil_qsa_pf[i] = mil_qsa[i]`).
- **ANE submit**: The QSA ANE submit dropped from **1.56 ms / token** to
  **1.27 ms / token** (a 19% reduction), comfortably clearing the 1.45 ms gate.

### Task 2: GPU/Host Boundary and Hidden State Transposition

Between the ANE returning a layer's mixed hidden state and the next ANE layer's
input surface, the baseline executed host routing in NumPy, MLX array conversion,
`gather_qmm` on the GPU, `mx.eval`, conversion back to NumPy, shared-expert
addition, and hyper-connection recombine.

We measured each candidate for GPU fusion against host execution:

#### Negative Result: GPU Router at k=32
Tested moving softmax, argpartition, and top-10 expert indexing to MLX GPU:
- **NumPy CPU route at k=32**: **0.1915 ms / layer** (0.49 ms / token across 48 layers).
- **MLX GPU route at k=32**: **0.2072 ms / layer** (0.60 ms / token across 48 layers).
- **Finding**: Host routing via Apple Accelerate BLAS and C-level quickselect
  is faster than GPU routing at $k=32$. GPU `argpartition` over 512 experts at
  small batch is latency-bound and adds dispatch latency (+0.11 ms / token net loss).
  Host routing was kept.

#### Negative Result: GPU Hyper-Connection Recombine (`FLASHNEXT_FUSED_MOE=all`)
Tested computing the 4-branch residual recombine ($h + \text{out} \times \text{inj}$)
in the same MLX graph on the GPU before copying back to host:
- **Host CPU recombine (`host_fastpath.recombine`)**: **0.0820 ms / layer** (0.20 ms / token).
- **MLX GPU recombine**: **0.2277 ms / layer** (0.61 ms / token).
- **End-to-end impact**: Prefill rate dropped from **84.4 tok/s** to **74.4 tok/s**
  (6.053s vs 6.869s), with GDN MoE jumping from 2.99 ms to 3.37 ms / token.
- **Finding**: The residual stream $m_{\text{hyp}}$ is 1.31 MB per chunk
  ($10,240 \times 32$ floats). Copying 1.31 MB to Metal memory and reading the
  result back across the unified memory boundary costs 0.145 ms per layer more
  than in-place vectorized NEON/AVX addition on CPU cache. Recombine remains on CPU.

#### Negative Result: GPU Shared Expert Addition (`FLASHNEXT_FUSED_MOE=shared`)
Tested adding $m_{\text{sh}}$ to routed experts in MLX before `mx.eval`:
- **Host CPU add**: GDN MoE = **2.96–2.99 ms / token**.
- **MLX GPU add**: GDN MoE = **3.01–3.02 ms / token**.
- **Finding**: Transferring the 327 KB shared expert output ($32 \times 2560$) to
  the GPU and appending an add kernel to the MLX stream costs ~0.03 ms per layer
  more than vectorized addition in host cache alongside recombine.

#### Win: Direct MLX Dtype Creation
In `routed_multi`, the baseline created float32 MLX arrays and called `.astype()`:
```python
x = mx.array(np.ascontiguousarray(x_k, np.float32)).astype(self.dtype)
sc = mx.array(np.ascontiguousarray(scores_k, np.float32)).astype(self.dtype)
```
Calling `.astype(self.dtype)` scheduled two separate Metal conversion kernels per
layer. Replacing this with direct dtype ingestion:
```python
x = mx.array(x_k, dtype=self.dtype).reshape(1, k, -1)
sc = mx.array(scores_k, dtype=self.dtype).reshape(1, k, -1, 1)
```
- Activation array creation: **0.1450 ms** -> **0.0083 ms / layer** (17x faster).
- Scores array creation: **0.1518 ms** -> **0.0245 ms / layer** (6x faster).
- Saved **~0.40 ms / token** across all 48 layers.

#### Win: BC1S Layout Persistence Across Layer Transitions
The ANE outputs activations in BC1S layout `(1, C, 1, S)`, and the next ANE layer's
input surface `_spec_xb` is sized `(1, 10240, 1, seq)`. The previous pipeline
transposed activations to BSH `(1, S, 10240)` upon leaving each layer, then
transposed back to BC1S at the next layer's staging:
- Transposing $10,240 \times 32$ floats twice per layer cost **0.1060 ms / layer**.
- Keeping hidden activations in BC1S format across layers and slicing directly
  `_spec_xb[..., :n] = np.asarray(hid[..., :n], np.float16)` dropped layer staging
  to **0.0241 ms / layer**.
- Saved **~0.20 ms / token** across 48 layers.

### Per-Token Timing Breakdown

Measured on a 511-token prompt at $k=32$ (Apple M5 Max):

| Term | Baseline (83.5 tok/s) | After Tasks 1 & 2 (84.4 tok/s) | Delta |
| --- | --- | --- | --- |
| GDN submit | 4.33 ms | 4.34 ms | +0.01 ms |
| GDN total ANE | 4.48 ms | 4.52 ms | +0.04 ms |
| QSA ANE submit | 1.45 ms | **1.27 ms** | **-0.18 ms** (-12.4%) |
| QSA total | 3.09 ms | **2.86 ms** | **-0.23 ms** (-7.4%) |
| Routed experts (GDN) | 2.81 ms | 3.01 ms | +0.20 ms |
| Router | 0.49 ms | 0.49 ms | 0.00 ms |
| Recombine | 0.22 ms | **0.20 ms** | -0.02 ms |
| Staging / Head / Commit / Embed | 1.04 ms | **0.72 ms** | **-0.32 ms** |
| **Total per token** | **11.98 ms** | **11.84 ms** | **-0.14 ms** |
| **Prefill Throughput** | **83.5 tok/s** | **84.4 tok/s** | **+0.9 tok/s** |

### Gate Verification

All three gates pass:

1. **Prefill Rate Gate**:
   - Gate: $> 83\text{ tok/s}$, QSA ANE term $< 1.45\text{ ms / token}$.
   - Result: **84.4 tok/s** (511 tokens in 6.053s), QSA ANE submit = **1.27 ms / token** (total QSA ANE = 1.34 ms). **PASSED**.
2. **Quality Gate (`eval/prose.txt`, 1024 tokens scored after 512 prefill)**:
   - Gate: Within 0.01 in log likelihood of 6.5943 ($6.5843 \le \text{PPL} \le 6.6043$).
   - Result: NLL = **1.886208**, PPL = **6.5943** (exact match to every reported decimal place). **PASSED**.
3. **Decode Gate (64 tokens greedy decode, `FLASHNEXT_PREFILL_MIL_K` omitted)**:
   - Gate: Exact token ID sequence match with baseline.
   - Result: Emitted greedy IDs `[279, 5492, 1752, 13, 271, 550, 2088, 38012, 2961, 271, 13962, 13425, 25, 279, 1788, 19214, 944, 42103, 539, 220, 16, 13, 22, 87, 271, 91, 21826, 735, 1510, 25434, 16436, 63, 7772, 735, 586, 10442, 10658, 735, 198, 91, 4277, 91, 4277, 25, 91, 4277, 25, 91, 198, 91, 74988, 16, 11, 16, 21, 11, 16, 17, 23, 11, 16, 17, 23, 60]`. **PASSED**.


### Independent verification of the QSA sharing and the MoE boundary work

**The QSA change is correct and its component is faster.** The ANE submit for
a QSA layer measured 1.45 and 1.63 ms a token before and 1.27 and 1.29 after,
across separate runs. Perplexity is unchanged to every digit at 1.886208 and
6.5943, and a decode run emits the same 64 token ids at 21.0 tok/s.

**It does not show up end to end.** Two runs here give 82.2 and 82.2 tok/s
against 78.0, 82.0 and 83.0 measured on the code before this change. The
claimed 83.5 to 84.4 is inside that spread. The arithmetic says why: the QSA
ANE term is about an eighth of the pass, so cutting it 15% is worth 2% of the
total, which this measurement cannot resolve.

That is worth carrying forward as a rule. End-to-end prefill on a 511-token
prompt has a spread of roughly 5 tok/s run to run, so anything worth less than
about 5% has to be argued from the component timers rather than the headline.
The change is still worth keeping: the component is genuinely faster and it
frees a second copy of the QSA weights.

The three negative results in task 2 were measured properly and reverted, which
is the right outcome. The pass total is unchanged at about 82 tok/s.

## The wide prefill graphs damage the tokens right after a prompt

`FLASHNEXT_PREFILL_MIL_K` must stay 0. Everything above about 82 tok/s stands
as a throughput measurement and is not usable, because walking a prompt through
the wide graphs leaves the state wrong enough to change what the model does.

The clearest case is a tool-calling prompt. Same 324-token prompt, greedy, the
only difference being the width the prompt was walked at:

```
prefill_k=0    The user wants me to list files in the current directory. Simple task.
               </think>
               <tool_call><function=run_shell><parameter=cmd>ls -la</parameter>...

prefill_k=32   The user wants me to run a command. They haven't given me a specific
               command, but the system prompt suggests I should use the bash tool...
```

The first is character-for-character what the 4-bit MLX reference produces. The
second ignores the request. It is not the chunked delta rule: the plain
recurrent graph at k=32 is just as wrong, and so is k=16. It is the wide
prefill path itself.

### Why the gate missed it

Perplexity was measured over 1024 tokens scored after a 512-token prefill,
where the difference is 0.8% and looks like noise. The damage is concentrated
in the tokens immediately after the prompt and washes out as real tokens flow
through the recurrence:

| tokens scored after the prefill | prefill_k=0 | prefill_k=32 | cost |
| --- | --- | --- | --- |
| 32 | 1.944132 | 2.015541 | +7.4% |
| 1024 | 1.845533 | 1.853263 | +0.8% |

Greedy generation only ever sees the first case. One wrong argmax at the first
token sends the whole completion somewhere else, which is how 7% of perplexity
becomes the difference between calling a tool and asking the user what they
want.

**So the gate for any prefill change is a short scoring window immediately
after the prompt, not a long one.** Use `--ppl-tokens 32 --ppl-prefill 512`.
A long window measures how fast the recurrence recovers, which is not the
question.

## Why prefill is barely faster than decode, and what to fix

Prefill runs at decode's batch size. A block is 4 tokens either way, and decode
confirms about 2.9 of them, so prefill gets 4 tokens a pass against decode's
2.9. That 1.4x is the whole difference. Nothing about the arithmetic makes
prefill slow; it simply is not batched.

Widening the block is the fix and it works: 32 tokens a pass is 82 tok/s
against 25.6. It is unusable today because it corrupts the state, and the
suspect is now narrower than "the wide path".

The GDN side is verified at every width. `probes/mil_k_check.py` compares it
against `MultiTokenStep` and prints the same per-slot errors at K=1, 4, 8, 16
and 32.

The QSA side has never been verified at any width, and the probe that looks
like it does is not an oracle. `probes/flashnext_mil_qsa_layer.py` run as
intended reports:

```
QSA FULL LAYER (int8=True, m=256, k=1): mixed 0.99220  hyper 0.89212
                                        shared 0.97159  new_k 1.00000
```

Relative errors of about 1.0 mean the two sides share nothing, yet the shipped
decode path built from this same graph reproduces the 4-bit MLX reference
character for character on a tool-calling prompt. So the layer is right and its
reference is wrong. It reports this at k=1 as well as k=32, so it has been
uninformative the whole time.

**The next step on prefill is to build a QSA check that works**, along the
lines of `mil_k_check.py`: drive the layer and a numpy reference from the same
weights, the same key cache and the same rotary table, and compare at K=1 first
to establish the oracle before trusting it at 32. Then find out whether the
wide QSA layer is what corrupts the prompt. If it is, prefill is a fix away
from 82 tok/s.

## Fixed: the corruption was a different recurrence, not a broken layer

The section above said the wide prefill graphs must stay off. That is no longer
true, and the cause turned out to be neither of the suspects it named.

**The QSA layer was never wrong.** `probes/mil_qsa_k_check.py` replaces the old
probe, which never fed the front program's mixed and inject inputs and so
reported errors near 1.0 on a graph that was working. The new check shares
weights, key cache, rotary table and mask with `MilQsaLayer`, and agrees in the
same band the GDN check does:

| | mixed | hyper | shared | new_k | new_v |
| --- | --- | --- | --- | --- | --- |
| K=1 | 0.0283 | 0.0262 | 0.0414 | 0.0138 | 0.0198 |
| K=32 | 0.0347 | 0.0294 | 0.0467 | 0.0150 | 0.0225 |

**Serve had a real bug.** It walked 32-token chunks through the k=4 graph, which
tiled four keys into the cache. Generate already selected the wide procedure;
serve now does too, and a chunk wider than the selected graph is rejected.

**The corruption was the GDN recurrence at a different width.** Each layer
passes `mil_k_check` at 32, but a single 32-wide chunk is a different
approximation from eight 4-wide ones, and the difference is enough to flip a
few routed experts per layer. Across 48 layers that compounds. So the k=32
procedure now runs the same 4-wide chunk eight times inside one submit, with the
state carried between them: mixed, state and conv match eight k=4 passes at
0.000, and it stays one submit.

That gives up the chunked delta rule on the prefill path, which is why the rate
is 58 tok/s rather than 82. It is still 2.3x the 25.6 of walking at decode
width, and it is correct.

### Verified

| gate | result |
| --- | --- |
| 32 tokens scored after a 512-token prefill | nll 1.946449 against 1.944132 at k=4, inside 0.01 |
| the tool-calling prompt | correct `run_shell` call with `ls -la`, as the MLX reference |
| `mil_k_check` K=4 / K=32 | unchanged |
| decode | same 64 ids, 20.3 tok/s |
| footprint | 78 GB, peak 81, unchanged: the wide unroll shares the decode program's weights |

So it is on by default: `FLASHNEXT_PREFILL_MIL_K=32`, and the server and eval
client pass 32.

The 24 tok/s between this and the 82 of the single wide chunk is still on the
table, and now has a precise statement: find a 32-wide recurrence whose output
does not flip the experts the 4-wide one picks.

### Intermediate tile widths do not pass either

`MIL_GDN_CHUNK_TILE` sets the width of each chunk inside the 32-slot prefill
procedure. Same checks as above, one run each, all deterministic:

| tile | prefill tok/s | nll, 32 tokens after 512 | vs 1.944132 | tool call |
| --- | --- | --- | --- | --- |
| 4 (default) | 57.9 | 1.946449 | +0.002 | correct |
| 8 | 63.8 | 1.983824 | +0.040 | correct |
| 16 | 66.7 | 1.967086 | +0.023 | correct |
| 32 | 82 | 2.015541 | +0.071 | wrong |

Only tile 4 is inside the 0.01 gate. Tiles 8 and 16 still produce the right
tool call, so the damage is smaller than at 32, but it is well outside what
perplexity allows.

Two things worth noting for whoever picks this up. The error is not monotonic
in width: tile 16 is closer than tile 8. And a 4-wide chunk matches the
token-by-token recurrence at 0.000 while wider ones do not, so the loss is in
the chunked arithmetic itself, not in running prefill as one submit. The
likely places are fp16 precision in the triangular inversion, the x64 query
scaling, and the neutral padding that brings narrower chunks up to 32.


## Full-width state update: the missing scale (2026-09-13 remeasurement)

The four-slot tiling above was a workaround. The untiled 32-slot graph can
pass the short-window gate: the final GDN state-update matmul needs scaling,
just as the query matmuls already did. The default is now a single 32-slot
chunk with `MIL_GDN_CHUNK_UPDATE_SCALE=64`. Decode's prefix-state recurrence
is unchanged. `MIL_GDN_CHUNK_TILE=4 MIL_GDN_CHUNK_UPDATE_SCALE=1` reproduces
the former workaround; `MIL_GDN_CHUNK_TILE=32 MIL_GDN_CHUNK_UPDATE_SCALE=1`
reproduces the former fast path.

### Evidence that isolates the state update

On identical fp16 inputs in `probes/mil_chunk_core.py`, changing only the
state-update scale from 1 to 64 gives:

| relative error against NumPy float64 | scale 1 | scale 64 |
| --- | --- | --- |
| current chunk output | 0.00141437 | 0.00141437 |
| final recurrent state | 0.01562905 | 0.00164694 |

The graph multiplies delta by 64 before `delta.T @ kend`, then multiplies the
result by 1/64 before adding the decayed incoming state. The unchanged current
output explains why an output-only check can miss the problem. The improvement
from power-of-two scaling is evidence of lost small products in the ANE
matmul; it does not establish the engine's exact internal rounding mode.
The new default also passes correlated-key and zero-decay core checks:
output/state errors 0.002879/0.000338 and 0.001296/0.001646 respectively.
The core probe now fails above 0.005, including for the final state.

`probes/mil_k_check.py` is unchanged. Fresh results with the new default:

| K | per-slot mixed band | final state | conv |
| --- | --- | --- | --- |
| 4 | 0.0282–0.0341 | 0.00618 | 0.01217 |
| 32 | 0.0263–0.0340 | 0.00683 | 0.01129 |

K=32 is 2.349 ms/call versus 4.598 ms for the old tiled default, measured
before and after on this checkout. `probes/mil_gdn_walk_check.py` additionally
compares against the production four-slot **prefix-state** graph, not a
four-slot single-state chunk: after 128 identical input tokens, mixed error
is 0.007952, recurrent state error 0.006751, and conv is identical.

### QSA oracle rechecked before investigating GDN

The K=1 check ran first. NumPy versus Torch attention error was 0.00070.
The full width sweep then measured:

| K | mixed | hyper | shared | new_k | new_v |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.02830 | 0.02623 | 0.04138 | 0.01382 | 0.01978 |
| 4 | 0.03433 | 0.02797 | 0.04038 | 0.01472 | 0.02145 |
| 8 | 0.03202 | 0.02840 | 0.04392 | 0.01478 | 0.02143 |
| 16 | 0.03668 | 0.02910 | 0.05076 | 0.01493 | 0.02207 |
| 32 | 0.03469 | 0.02941 | 0.04670 | 0.01495 | 0.02253 |

The shared-program K=32 procedure gives the same numbers. Walking 64 tokens
at K=1 versus K=32 gives identical key/value caches and last-token mixed error
0.00055. No QSA graph, causal mask, or rotary change was needed. Both gate
prompts are below the 2048-key indexer budget, so skipping selection for L>16
cannot explain this failure; sparse long-context selection is a separate case.

The old `flashnext_mil_qsa_layer.py` executable now delegates to this oracle
instead of submitting without its required mixed/inject feeds. Every invocation
checks K=1 first and fails on nonfinite or excessive errors; the walk also has
failure thresholds. Run:

```bash
~/.rindi/venvs/coreai/bin/python probes/flashnext_mil_qsa_layer.py --all --two-proc --walk 64
```

`AneClient(prefill_k=0)` now explicitly sets the child's environment to 0.
Previously it omitted the setting, silently inheriting either the parent's
width or the exporter's new default of 32. The environment regression test
covers both inherited and absent settings.

### Full-model gate configuration

These runs keep real PLE enabled (the checkout's default), the resident MLX
expert bank, MLX head, speculative width 4, and both MIL layer backends:

```bash
FLASHNEXT_PREFILL_MIL_K=32 FLASHNEXT_SPEC=4 FLASHNEXT_MOE=mlxresident \
FLASHNEXT_HEAD=mlx FLASHNEXT_MIL_GDN=1 FLASHNEXT_MIL_QSA=1 \
~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py generate \
  --ppl-file eval/prose.txt --ppl-tokens 32 --ppl-prefill 512
```

The same command with `FLASHNEXT_PREFILL_MIL_K=0` freshly reproduces
**NLL 1.944132**. The first corrected full-width run gives **1.947003**,
a difference of **+0.002871**, inside the 0.01 gate. These are 32-token
scores immediately after prefill, not the misleading 1024-token window.

The scorer prints the timing for walking 511 prompt tokens, keeping the
512th token as input to the first scored prediction. Initial measurements:

| term (ms/token) | K=0 baseline | corrected K=32 |
| --- | --- | --- |
| embed | 0.01 | 0.01 |
| GDN staging | 1.00 | 0.10 |
| GDN ANE total | 17.97 | 4.59 |
| GDN router | 1.64 | 1.34 |
| GDN MoE | 7.55 | 2.97 |
| GDN recombine | 0.33 | 0.21 |
| QSA total | 9.52 | 3.11 |
| head | 0.87 | 0.29 |
| commit | 1.27 | 0.25 |
| PLE | 1.70 | 1.43 |
| PLE lookup (included in PLE) | 0.69 | 0.35 |
| **prefill** | **511 / 21.459s = 23.8 tok/s** | **511 / 7.336s = 69.7 tok/s** |

GDN call breakdown (write / submit / take): baseline 0.25 / 17.06 / 0.63,
corrected 0.08 / 4.38 / 0.13 ms/token. QSA breakdown (mix / index / feed /
ANE / MoE / recombine): baseline 1.46 / 0.18 / 0.08 / 4.59 / 3.08 / 0.11,
corrected 0.34 / 0.04 / 0.02 / 1.26 / 1.38 / 0.07 ms/token.
These component breakdowns are nested inside the totals above.


A fresh negative-control run with `MIL_GDN_CHUNK_UPDATE_SCALE=1` reproduces
**NLL 2.015541 exactly**, at 70.3 tok/s. Thus this checkout reproduces the
quality failure and the scale alone recovers it. The historical 82 tok/s is
not reproduced here even on the uncorrected fast path; the corrected path
preserves the measured fast-path rate within run-to-run variation.

The exact tool fixture is now checked in as `eval/prefill_tool.json`.
`FLASHNEXT_PREFILL_MIL_K=32 python probes/prefill_tool_check.py` verifies a
cold (0 reused tokens), 324-token prompt, explicit K=32 ready state, and a
literal `run_shell` call with `cmd="ls -la"`. It generated 47 tokens:

```text
The user wants me to list files in the current directory. Simple task.
</think>

<tool_call>
<function=run_shell>
<parameter=cmd>
ls -la
</parameter>
</function>
</tool_call>
```

No tool is executed by the check. All model and ANE runs were sequential;
compiled caches were retained. Disk headroom went from 45 GiB to 38 GiB
through the new graph builds, with no resource or compilation failures.

### Final run using only the documented environment variables

A second corrected run, with no `MIL_GDN_CHUNK_*` overrides, gives the same
**NLL 1.947003** and **511 tokens in 7.175s = 71.2 tok/s** (3.0x the freshly
measured 23.8 baseline). The complete printed breakdown is:

```text
prefill ms/token  embed=0.01  gdn_stage=0.10  gdn_ane=4.55  gdn_route=1.30  gdn_moe=2.89  gdn_rec=0.21  qsa=3.07  head=0.26  commit=0.24  ple=1.36  ple_lookup=0.36
prefill gdn call ms/token  write=0.07  submit=4.34  take=0.13
prefill qsa ms/token  mix=0.33  index=0.04  feed=0.02  ane=1.24  moe=1.37  recombine=0.07
```

The final QSA sweep, shared-program check, 64-token QSA cache walk,
128-token GDN walk against prefix-state decode, and isolated core all pass
their executable thresholds. The isolated core negative control with
`MIL_GDN_CHUNK_UPDATE_SCALE=1` is expected to fail on `q_state31` at 0.01562905.

## Remeasured: the wide chunk loses precision in the state-update matmul

The section above already switched the default to one 32-slot chunk with
`MIL_GDN_CHUNK_UPDATE_SCALE=64`. This pass re-measured the claim against the
token-by-token recurrence *before* any full-model run, because `mil_k_check`
against `MultiTokenStep` passes at every width and still lets experts flip.

`probes/mil_chunk_vs_recurrent.py` compiles the isolated GDN core (no
projections) once as 32 unrolled `gdn_core` steps and once as a chunk, same
fp16 inputs. NumPy fp64 WY versus that recurrence is `y=2.4e-7` /
`state=8e-8`, so the equations are not the loss. Neutral padding (gate 1,
beta 0, zero q/k/v) in NumPy fp16 is also a non-event:

| pad live→32 vs native-live | y | state |
| --- | --- | --- |
| 4 | 5e-8 | 0 |
| 8 | 6e-8 | 2e-8 |
| 16 | 1e-8 | 3e-8 |

The non-monotonic tile-8-versus-16 quality table is therefore not padding in
the arithmetic. The ANE is where the widths come apart.

### Which step diverges

Against the ANE recurrent graph, current-chunk *output* `cy` is the same
whether the state update is scaled or not. Final *state* is not:

| ANE config vs recurrent | per-slot y (t00 … t31) | state |
| --- | --- | --- |
| tile 32, update 64, q 64 (default) | 0.00098 … 0.00325 | **0.00170** |
| tile 32, update **1**, q 64 | identical y | **0.01564** |
| tile 32, update 64, q **1** | **0.047 … 0.214** | 0.00170 |
| tile 4, update 64 | 0.00106 … 0.00809 (spikes at chunk boundaries) | 0.00156 |
| tile 4, update 1 | same pattern, larger | 0.01014 |
| tile 8, update 64 | 0.00106 … 0.00382 | 0.00163 |
| tile 16, update 64 | 0.00106 … 0.00480 | 0.00172 |

`mil_chunk_core.py` on the same seed reproduces the documented pair exactly:
`cy=0.00141437` either way; `q_state31=0.00164694` at scale 64 and
`0.01562905` at scale 1 (the probe now fails closed above 0.005). Correlated
keys at scale 1 are *better* (`state=0.00132`), so this is lost small
products in the `delta.T @ kend` matmul, not the blocked inverse blowing up.

`MIL_GDN_CHUNK_Q_SCALE=64` is the older underflow fix for `Y`. Turning it
off wrecks the current outputs and leaves the state alone. Turning the
update scale off wrecks the state that the next chunk inherits and leaves
`Y` alone. An output-only check, including `mil_k_check`'s mixed column,
cannot see the failure.

Tile 4 versus token-by-token is **not** 0.000. That 0.000 was eight k=4
chunks against eight k=4 chunks. Versus `gdn_core`, tile 4 at the old
unscaled update is state 0.010, and the first token of each new 4-wide
chunk spikes because it reads the already-rounded state. Scale 64 brings
every width into the same 0.0016–0.0017 state band, so the width to ship
is 32: no padding, one matmul, same accuracy.

### Layer checks, unchanged

`chunk_tile=32 update_scale=64 q_scale=64 inverse=blocked` is what generate
now prints when the MIL GDN set loads.

| probe | result |
| --- | --- |
| `mil_k_check.py` 4 | mixed 0.0282–0.0341, state 0.00618, conv 0.01217 |
| `mil_k_check.py` 32 | mixed 0.0263–0.0340, state 0.00683, conv 0.01129; vs 8×k=4 mixed 0.00268 state 0.00138 conv 0 |
| `mil_qsa_k_check.py` 4 / 32 / two-proc | mixed 0.03433 / 0.03469 / 0.03469, K=1 first 0.02830 |
| `mil_gdn_walk_check.py` 128 tokens | mixed 0.007952, state 0.006751, conv 0 |

### Full-model gates (default tile 32, this checkout)

```bash
FLASHNEXT_PREFILL_MIL_K=32 FLASHNEXT_SPEC=4 FLASHNEXT_MOE=mlxresident \
FLASHNEXT_HEAD=mlx FLASHNEXT_MIL_GDN=1 FLASHNEXT_MIL_QSA=1 \
FLASHNEXT_NO_DRAFTER=1 \
~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py generate \
  --ppl-file eval/prose.txt --ppl-tokens 32 --ppl-prefill 512
```

| gate | result |
| --- | --- |
| 32 tokens scored after a 512-token prefill | **nll 1.947003** vs 1.944132 (**+0.002871**), inside 0.01 |
| 511-token prefill | **68.0 tok/s** (7.510 s), GDN submit 4.40 ms/token; above 57.9 |
| tool prompt, `probes/prefill_tool_check.py` | 324 tokens, 0 reused, 47 generated, **`run_shell(cmd='ls -la')`** |
| decode `--prompt-ids 760 --max-new 64` (spec 4, PLE on, drafter on) | ids **identical** to the PLE-on baseline in `results/ple/decode_on.log`, **8.288 tok/s** |

The 64-id list earlier in this file (`279, 5492, 1752, …` at 20.3 tok/s) is
the PLE-off decode. With the table on, greedy from `"The"` is the bedbugs
continuation starting `20438, 5134, 25, 1001, …`. Chunking does not touch
the prefix-state decode graphs.

### Negatives from this pass

- **`MIL_GDN_CHUNK_UPDATE_SCALE=1`** at tile 32: core `q_state31=0.01562905`.
  Historical e2e nll **2.015541**. Do not ship.
- **`MIL_GDN_CHUNK_Q_SCALE=1`**: current y vs recurrent 0.047–0.214. State
  unchanged. The query scale is necessary and is not the state bug.
- **Neutral padding** in NumPy: not the tile-8-versus-16 quality wobble.
- **Tile 4 as an oracle for tile 32**: matches other 4-wide chunks at 0.000,
  not the token-by-token recurrence.
- **`FLASHNEXT_NO_DRAFTER=1` on decode**: 1.00 tokens/pass, 7.134 tok/s, ids
  diverge after token 3 (`271, 29` inserted). Prefill scoring still used it
  and matched nll 1.947003; do not use it as the decode-id gate.

No emitter change was required beyond what `1b1779b` already defaulted:
tile 32, update scale 64. The path print on `MIL int8 GDN:` is so a flag
that does not reach `gdn_chunk` cannot look like a no-op.
