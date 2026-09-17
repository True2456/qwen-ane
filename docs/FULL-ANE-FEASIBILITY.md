# Full-transformer ANE feasibility

This document answers a narrower question than the existing hybrid benchmarks:
**can Qwen3.8-27B run without using the GPU for model arithmetic?** The answer
is now **yes at the execution-stack level**, not just as isolated arithmetic
probes. `tools/pure_ane.py` loads the checkpoint without MLX/PyTorch/Core
ML, bakes all 64 layers, executes full attention and GDN, and runs the final
head through direct private-ANE requests. Persistent GDN state has no
per-token MLX/GPU round trip; the private compiler still prevents a literally
zero-copy binding, so the path uses one 1.57 MB IOSurface row copy per GDN
layer-token.

All results below were run on the M5 Max against MIL `program(1.3)`, `ios18`,
through the private `AppleNeuralEngine.framework`.  The probes use the real
Qwen shapes from `config.json`, not reduced toy dimensions unless a control row
is explicitly shown.

## What “full ANE” means

The practical target is **GPU-free model execution**:

1. CPU selects an embedding row and writes it to the first ANE surface.
2. All 64 transformer layers execute on the ANE.
3. Final norm and `lm_head` execute on the ANE.
4. CPU reads/samples logits.

Tokenization, an integer embedding lookup, request scheduling, and sampling are
control work.  The ANE MIL dialect has no usable `gather`, so “literally every
operation including token lookup runs on ANE” is not a useful or currently
achievable definition.  The important target is no MLX/GPU tensor arithmetic
and no GPU synchronization in the decode loop.

## Current standalone implementation

At direct context 256 the pure runtime uses the following loaded-model layout:

| component | programs |
|---|---:|
| 64 layer tails; 63 also emit the next layer's normalized hidden state | 64 |
| GDN K=4 depthwise conv | 48 |
| projection procedure banks | 2 at int4/int8; 5 at fp16 |
| shared dynamic Q/K norm + RoPE | 1 |
| shared grouped-query attention core | 1 |
| shared GDN norm/gates/recurrence | 1 |
| layer-0 norm + projection | 1 |
| final norm + vocabulary head | 4 |
| **total** | **122 quantized; 125 fp16** |

Long context through 262,144 replaces the one attention core with additional
stream/merge models and reaches 127 loaded models. The current
`_ANEInMemoryModel` path rejects distinct model 128 with `0x50004`. This is an
empirical private-loader/process budget, not a claim that the hardware's total
program capacity is 127; the separately reverse-engineered 127 value describes
concurrent evaluation queue depth.

End-to-end allocation and dispatch are proven at 12.86 GB int4 and 25.72 GB
int8. The original quantized semantic failures were measured before the stable
RMSNorm and scaled-GDN-carrier fixes. After those fixes, int4 generated the
exact four-token reference `[248068, 198, 760, 1156]` in 4.791 seconds, using
122 programs and 12.86 GB of blobs. That overturns the earlier conclusion that
int4 itself caused the incoherent output, although a broader benchmark is still
required before calling it release-quality. In a 32-token greedy comparison,
int4 diverged from MLX at token 5 but continued with coherent, semantically
equivalent reasoning at 2.467 generated tok/s.

Int8 generated the exact four-token reference in 5.092 seconds with 122
programs and 25.72 GB. Its 32-token run matched MLX exactly for the first 16
tokens, then diverged but correctly completed the instruction with `OK`; the
measured rate was 2.421 generated tok/s. Neither quantized run exhibited the
old corruption. Full-fidelity mode is fp16: the complete 64-layer runtime now bakes 125
resident programs with 51.42 GB of learned-weight blobs and passes semantic
inference. The known 13-token prompt generated token IDs
`[248068, 198, 760, 1156]`, decoding to `<think>\nThe user`, exactly matching
the first four MLX reference tokens. The measured cold bake took 82.3 seconds;
13-token prompt processing plus four-token generation took 11.594 seconds
(0.345 generated tok/s including prompt work).

Allow about 55 GB of genuinely available disk space for an fp16 cold bake.
The compiler materializes temporary weight/data files and macOS may grow swap;
a deleted file held open by another process still consumes space even though
it is absent from directory-size totals.

Re-run the semantic oracle with:

```bash
tools/ane pure-infer --bits 16 --tokens 4 --verify-reference
```

The standalone dependency guard checks `sys.modules` at startup, after compile,
and after every token, and aborts if `mlx`, `torch`, or `coremltools` enters the
process. The launcher also removes `PYTHONPATH` and uses system Python for every
`pure-*` command.

### Numerical formulations required for semantic parity

Two fp16 details were necessary beyond making each graph merely execute:

1. **Overflow-safe RMSNorm.** Fixed pre-scaling works on small probes but
   overflows once Qwen's residual-stream outliers exceed the fp16 square range.
   The pure runtime slices one of the 32 identical decode lanes, reshapes the
   5,120 channels onto width, reduces `max(abs(x))`, divides by that maximum,
   multiplies by 64, and performs `reduce_mean` on the bounded squares. The 64
   factor keeps the mean out of fp16's low-accuracy range while every square
   remains at most 4096. Epsilon is transformed by the same scale. Tests across
   input maxima 27, 52, 428, and 1,776 measured relative error from 6.3e-5 to
   7.3e-4.
2. **Scaled GDN carrier.** The gated-delta output contains meaningful values
   near `1e-6`, which BF16 preserves but the ANE fp16 contraction quantized.
   The shared recurrence emits this tensor at 64x scale. Its following gated
   RMSNorm uses `epsilon * 64^2`, exactly absorbing the carrier scale; recurrent
   state and model semantics are unchanged. With this change, the third prompt
   token's first three layer RMS values became 0.41209, 0.76120, and 0.83686,
   versus MLX 0.41404, 0.75974, and 0.84090.

## Exact model inventory

Qwen3.8-27B has 64 transformer layers:

| layer kind | count | important shape |
|---|---:|---|
| Gated DeltaNet | 48 | H=48, Dk=128, Dv=128, depthwise conv C=10240, K=4 |
| full grouped-query attention | 16 | Hq=24, Hkv=4, head dim=256 |

Every learned linear projection is already baked onto the ANE by
`--ane-chain`.  The missing work is therefore the sequence core between the
input projections and each layer tail.

## Full attention: feasible

`probes/ane_attention_core.py` executes the real grouped-query decode core:

```text
Q @ K^T * (1/sqrt(256)) → additive mask → stable softmax → P @ V
```

Both matmul operands are dynamic.  The program preserves grouped-query
attention as `[4 KV groups, 6 query heads/group]`; it does not physically repeat
the four KV heads six times.  A host-written additive mask permits a fixed
context bucket to represent any logical cache length.

Measured results:

| cache length | formulation | result | relative error | ANE time |
|---:|---|---|---:|---:|
| 32 | direct `softmax` | pass | 1.33e-3 | 0.098 ms |
| 32 | `reduce_max` + `exp` + `reduce_sum` + divide | pass | 1.33e-3 | 0.210 ms |
| 128 | direct | pass | 1.87e-3 | 0.108 ms |
| 256 | direct | pass | 3.83e-3 | 0.111 ms |
| 512 | one monolithic program | compile rejected | — | — |
| 512 | streamed 256-token chunks | pass | 1.30e-3 | 0.390 ms/chunk |
| 1024 | streamed 256-token chunks | pass | 1.44e-3 | 0.309 ms/chunk |
| 8193 | integrated 32-block scan + ANE merge, 256K capacity | pass | 2.19e-3 | — |

The direct program hits a compiler-complexity/shape ceiling above 256 tokens.
This is not an attention-math ceiling.  The streamed path returns each chunk's
normalized value, maximum, and exponential sum, then combines chunks with the
exact online-softmax identity:

```text
m = max(m_i)
w_i = sumexp_i * exp(m_i - m)
y = sum(y_i * w_i) / sum(w_i)
```

The original probe performs that tiny combination on CPU. The production pure
runtime now uses a shared, weight-free ANE combine program, so the host performs
no attention arithmetic. KV is block-major in 256-token blocks, and scan
programs handle 1/4/16/32 blocks per submission. A common `1/8192` scaling of
every exponential sum prevents fp16 denominator overflow at 256K without
changing relative weights or the normalized result.

RoPE is not a blocker: positions are known to the host, so precomputed sin/cos
rows can be supplied as data and rotation is elementwise multiply/add.  Q/K
RMSNorm is already expressible using the verified RMSNorm formulation.

## GDN temporal convolution: feasible

`probes/ane_gdn_conv1d.py` runs the real C=10240, K=4 depthwise causal
convolution with a width-32 decode surface.

| operation | result | relative error | ANE time |
|---|---|---:|---:|
| depthwise causal conv only | pass | 4.42e-4 | 0.278 ms |
| conv + direct `sigmoid` SiLU | numerically poor | 3.98e-2 | 0.266 ms |
| conv + `x / (1 + exp(-x))` | pass | 7.08e-4 | 0.133 ms |

The surprising row is direct sigmoid.  It compiles and runs but is not accurate
enough at the small real activation magnitudes.  Do not use MIL `sigmoid` for
this block.  The `exp`/divide spelling is both accurate and, in this run, no
slower.

The three cached convolution samples can occupy the three lanes before the
current projection in the width-32 input.  The K=4 custom-left-padded conv
produces the current result without GPU work.

## GDN recurrence: arithmetic and resident state are implemented

The recurrence itself is already implemented in `probes/ane_gdn_step.py` and
verified at about 9e-4 relative error.  Grouped convolutions provide the Dk
reductions and outer-product broadcast.

The original server path was slow because every call:

1. transposes `[H,Dv,Dk]` into `[H*Dk,Dv]` with NumPy;
2. writes roughly 3.5 MB into a combined IOSurface;
3. reads the new state back; and
4. transposes it to MLX layout again.

`probes/ane_gdn_persistent_state.py` explores binding the state output directly
as the next request's input.  The obvious formulations are currently rejected:

| attempted state formulation | result |
|---|---|
| separate state and per-token activation inputs | `InvalidMILProgram` |
| pad state output from width 128 to input stride 160 | `InvalidMILProgram` |
| reshape densely packed parameters beside a width-128 state | `InvalidMILProgram` |
| project state 128→160 for a stride-compatible ping-pong surface | `InvalidMILProgram` |

`probes/ane_gdn_compact_state.py` also isolates the width-128 packing attempt.
Both reshape-based packing and a grouped-convolution demultiplexer are rejected
before the recurrence. This is a private-compiler layout restriction, not an
unsupported GDN operation.

The implemented path in `tools/ane_serve.py::AneGdnStep`, verified by
`probes/ane_gdn_resident_state.py`, keeps one compact
`[H*Dk,Dv]` IOSurface per live MLX cache marker:

1. import the full MLX state once, after prefill;
2. copy compact state into columns `0:128` of the accepted width-160 input;
3. write q/k/v plus raw a/b and the per-head A/dt constants;
4. compute polynomial softplus, decay, beta, and the complete recurrence on ANE;
5. bind `new_state` directly to the cache-owned compact IOSurface; and
6. return the same MLX cache object as an opaque marker, without reading or
   transposing the evolved state.

Measured on M5 Max:

| path | time per layer-token |
|---|---:|
| old MLX/NumPy round trip | 4.39 ms |
| IOSurface state copy alone (1.57 MB) | 0.036 ms |
| resident gates + recurrence including copy, writes, dispatch, and y read | 0.344 ms |
| existing GPU recurrence reference | about 0.66 ms |

Two dependent resident steps pass with state relative error `1.06e-3` and y
relative error about `4.66e-3`. No model arithmetic or recurrent-state tensor is
sent to the GPU. This is not literally zero-copy, but it solves the performance
problem that zero-copy was intended to solve.

### Real 27B inference test

The isolated result does **not** make the current hybrid server faster by
itself. On a real greedy generation, q/k/v/a/b are still produced lazily by the
GPU. Converting them to host arrays forces the whole upstream GDN projection and
convolution to finish at every layer boundary:

| 76-token prompt | GPU baseline | `--ane-gdn-step` |
|---|---:|---:|
| 16 generated tokens | 7.4 tok/s | 5.0 tok/s |
| 64 generated tokens | 8.5 tok/s | 5.6 tok/s |
| TTFT, 64-token run | 449 ms | 594 ms |

Instrumenting 816 real GDN calls split the 3.556 ms/call wall time into
`3.057 ms` GPU-to-host evaluation/marshalling and `0.499 ms` for the resident
ANE gate+recurrence call. Thus the remaining slowdown is a hybrid boundary,
not recurrent-state traffic or ANE gate math.

Correctness remained healthy. In a same-process 64-token greedy A/B, the first
45 tokens were identical and 59/64 token positions matched. The decoded text
was semantically the same; the first difference was only “count maybe around
100” versus “count around 100 words.” A normal API request also completed with
the requested final answer `OK`.

This measurement supports finishing the ANE-only GDN chain before attempting
GPU overlap: make the ANE projection/conv feed this recurrence through
IOSurfaces directly, eliminating the measured 3.057 ms synchronization edge.

## GDN scalar gates: solved on ANE

The per-head decay uses:

```text
decay = exp(-exp(A_log) * softplus(a + dt_bias))
beta  = sigmoid(b)
```

The first obvious spellings are unsafe:

| spelling | result |
|---|---|
| MIL `softplus` / `softplus_parametric` | compile, silently wrong above about 10.4 |
| `log(1 + exp(x))` | overflows for large positive x |
| `relu(x) + log(1 + exp(-abs(x)))` | stable in fp32, but fp16 rounds small `1+t` to one |
| `(1 + exp(x)) ** -A` | same overflow/cancellation problem |

The working formulation avoids the fp16 `1+t` cancellation.  Let
`t=exp(-abs(x))`, approximate the smooth `log1p(t)/t` on `[0,1]` with a degree-5
polynomial, then restore the positive asymptote:

```text
P5(t) = 1
      - 0.499267578125 t
      + 0.324462890625 t^2
      - 0.2086181640625 t^3
      + 0.10028076171875 t^4
      - 0.023681640625 t^5

softplus(x) = relu(x) + t * P5(t)
beta(b)     = 1 / (1 + exp(-b))
decay       = exp(-A * softplus(a + dt_bias))
```

`probes/ane_gdn_gates.py` loads the real checkpoint lazily, selects adversarial
heads spanning `A=0.003839..139.4` and `dt_bias=-8.938..19.25`, and sweeps the
gate input over `[-20,20]`:

| output | maximum relative-to-peak error |
|---|---:|
| decay | 9.95e-4 |
| beta | 6.46e-4 |

The standalone gate program takes about 0.10–0.17 ms in repeated runs. The same
formulation is now fused into `AneGdnStep`, so `--ane-gdn-step` no longer calls
MLX `softplus` or `sigmoid` during its supported decode path. Direct MIL sigmoid
remains too inaccurate, so use the explicit exp/divide beta above. The scalar
gate is no longer a blocker to strict ANE arithmetic.

## Implemented GPU-free decode architecture

```text
CPU embedding row
  ↓
48 GDN layers:
  ANE input projections
  → ANE K=4 depthwise conv + exp/divide SiLU
  → ANE fp16-safe polynomial softplus + decay/beta
  → ANE gated-delta recurrence in permanent ANE layout
  → ANE norm/gate/out projection + residual + MLP

16 full-attention layers:
  ANE Q/K/V/gate projections + Q/K norm + RoPE
  → ANE 256-token attention chunks
  → exact online-softmax combine
  → ANE gate/out projection + residual + MLP
  ↓
ANE final norm + lm_head
  ↓
CPU sampling
```

This graph is implemented in `tools/pure_ane.py` and uses no GPU. The direct
attention core handles the first 256 positions and exact streamed online
softmax extends it through the checkpoint maximum of 262,144. The pure path is not guaranteed to
beat the hybrid runtime: the ANE still computes at least 32 decode lanes and
sustains about 10 TFLOP/s. Its purpose is to establish GPU-free throughput,
tokens per joule, and the true cost of state movement without MLX
synchronization.

### Pure MTP is implemented

`tools/ane pure-infer --bits 4 --tokens 32 --mtp-draft 2` loads the checkpoint's
single MTP layer into the standalone backend. Overflow-safe RMSNorm handles
three independent physical lanes, target verification batches the learned
weight-heavy blocks, and GDN/attention state is reversible. The real 32-token
test used 124 programs and 13.16 GB of blobs, averaged 2.385 accepted tokens per
cycle, emitted exactly the same token IDs as non-MTP greedy decode, and improved
end-to-end throughput from 2.770 to 3.185 tok/s.

## Remaining order

1. Persist compiled artifacts so cold start does not rebake 124/127 programs.
2. Build persistent streaming/chat serving and session reset around the now-long-context runtime.
3. Test the `_ANEInMemoryModel` unload lifecycle or lower-level dispatch route;
   the observed 127 distinct-model load budget is separate from the reported
   127-request hardware queue depth.
4. Run broad behavioral/perplexity evaluation of native int4 and int8. Int4
   now passes the short semantic oracle, so do not replace the format before
   measuring it properly.
5. Measure steady-state wall power, decode-only throughput, and tokens/joule
   with the standalone process.
6. Only then test selective GPU overlap against the now-working pure-ANE
   baseline.
