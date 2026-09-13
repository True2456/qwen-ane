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

The cost is compile time. Building 36 two-procedure programs took 136s against
about 22s for the two separate sets, and ANE compile times here are volatile
enough that the number should be re-checked on a warm cache before anyone
plans around it.

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

