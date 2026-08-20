# What the ANE accepts and rejects

Measured on M5 Max through `AppleNeuralEngine.framework`'s private
`_ANEInMemoryModel` API, MIL `program(1.3)`, `ios18` opset. **Read this before
writing MIL.** Most of these fail silently or with an opaque status code, and
several contradict what the previous section of this document would lead you to
expect — so measure, don't assume.

## Ops

| op | status |
|---|---|
| `conv` (1×1 = linear, and grouped) | works, the workhorse |
| `mul`, `add`, `sub`, `sigmoid`, `sqrt`, `exp`, `real_div` | work |
| `softmax(axis=-1)` | works through the measured 256-token attention shape |
| `reduce_max`, `reduce_mean`, `reduce_sum` on width (`axes=[3]`) | work, including width 5120 after channel-to-width reshape |
| depthwise spatial `conv`, `groups=C`, kernel 1×4 | works at Qwen's real C=10240 shape |
| `slice_by_index`, `transpose`, `pad`, `identity`, `matmul` | work |
| `constexpr_blockwise_shift_scale` (int4/int8 dequant) | works, per-output-channel scales only |
| `concat` | **compile fails** |
| `stack` | **compile fails** |
| `rsqrt` | **compile fails** — use `sqrt` + `real_div` |
| `l2_norm` | **compile fails** |
| `gather`, `tile` | **compile fails** |
| `log(epsilon=fp16(0), ...)`, `pow` | compile and run correctly; spell `log`'s epsilon explicitly |
| `softplus`, `softplus_parametric` | compile but become silently wrong at large positive inputs |
| `reduce_mean(axes=[1])` | **fails at C=512, S=32** |
| `reduce_sum(axes=[1])` | **works at C=128, S=6144** |

Channel reductions are **shape-dependent**, not categorically banned. The two
rows above are the same class of op with opposite results. Re-test at your
shapes.

`sigmoid` also needs a numerical warning: it compiles, but Qwen's real
depthwise-conv SiLU measured about 4e-2 relative error.  Spelling SiLU as
`x / (1 + exp(-x))` reduced this to 7.1e-4.  Acceptance is not accuracy.

The GDN softplus has an accurate ANE-safe formulation.  Native `softplus`
silently fails above roughly 10.4, while the nominally stable
`relu(x) + log(1 + exp(-abs(x)))` loses small negative tails when fp16 rounds
`1+t` to one.  Use `t=exp(-abs(x))` and approximate `log1p(t)` as `t*P5(t)`:

```text
P5(t) = 1
      - 0.499267578125 t
      + 0.324462890625 t^2
      - 0.2086181640625 t^3
      + 0.10028076171875 t^4
      - 0.023681640625 t^5
softplus(x) = relu(x) + t*P5(t)
```

`probes/ane_gdn_gates.py` validates the complete decay and beta gates against
real checkpoint constants spanning A=0.003839..139.4 and
dt_bias=-8.938..19.25: decay error 9.95e-4, beta error 6.46e-4, about
0.10–0.17 ms in repeated runs.

Full grouped-query attention is verified in `probes/ane_attention_core.py`.
One direct program compiles through a 256-token cache but not at 512.  Exact
online-softmax streaming in 256-token chunks passes at 512 and 1024 tokens; see
`docs/FULL-ANE-FEASIBILITY.md`.

## Idioms that replace the missing ops

**Cross-channel mean** — a 1×1 conv with constant `1/C` weights is the compact
RMSNorm formulation for controlled ranges:

```
sq = mul(x, x)  →  ms = conv(sq, ones/C)  →  sqrt(ms + eps)  →  real_div(x, sd)  →  mul(w)
```
Verified rel 1.5e-3. Costs 0.212 ms at H=5120, S=32.

For the full fp16 model, fixed scaling is not robust: residual outliers grow
past the safe square range, while scaling down far enough makes the early-layer
mean inaccurate. The production pure backend uses a stable formulation:

```text
x0 = one of the 32 identical decode lanes
flat = reshape channels onto width
m = max(abs(flat))
s = 64 * flat / max(m, sqrt(eps))
rms = (m / 64) * sqrt(mean(s*s) + (64*sqrt(eps)/m)^2)
y = x / rms * weight
```

`s*s <= 4096`, and because `max/RMS <= sqrt(5120)`, its mean stays in an
fp16-accurate range. Real-shape tests with maxima from 27 to 1776 measured
relative error between 6.3e-5 and 7.3e-4.

**Emitting two tensors** — `concat` does not exist, and the `pad`+`pad`+`add`
substitute caps out near 9216 output channels. Instead return two outputs from
the MIL func and bind two surfaces (see below). No width limit.

**Reduction over a sub-range of channels** — grouped conv, `groups=H`, `Dk→1`
per group. Verified rel 4e-4.

**Broadcasting a value back over a sub-range** — grouped conv the other way,
`groups=H`, `1→Dk`. Verified rel 3e-4.

**Broadcasting one value across the width** — put it in a width-1 column and
`mul`; `[1,C,1,1]` broadcasts against `[1,C,1,S]`. Verified rel 8e-4.

**Persistent GDN state** — keep state in a compact `[H*Dk,Dv]` IOSurface and
bind `new_state` directly back to it. The accepted recurrence input has width
160 while state has width 128, so copy the compact surface into columns
`0:128` before dispatch. This 1.57 MB row-strided IOSurface copy measures
0.036 ms; the full resident call, including polynomial softplus/decay and beta,
is 0.344 ms and avoids all per-token state transposes and MLX/GPU gate
arithmetic. Literal zero-copy packing through a second
activation input, pad/projection, reshape, or grouped-conv demultiplexer is
rejected with `InvalidMILProgram` at the real shape.

## Two output tensors per program

A MIL func may return `-> (y, y2)`, and `_ANERequest` takes an *array* of
outputs. This is what removes the output-width limit and, by letting each layer
also emit the next layer's projection, what got the whole model under the
program ceiling.

**The compiler does not preserve declared output order.** A func declaring
`-> (y, y2)` reported `y2@output` as symbol 0. Binding by position hands each
surface the wrong size and inference dies with `status=0x1d`. Bind by the
channel count the model reports:

```python
chans = [int(c) for c, _, _ in re.findall(
    r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";', desc, re.S)]
```
The `(?!Channels =)` lookahead is required — without it the match runs from the
*input's* Channels to the first output Name and reports the input width.

## Shape and alignment rules

* **Width ≥ 32.** Below that the ANE silently returns zeros. `[1,C,32,1]` and
  `[1,C,4,8]` do not count — it must be the width axis.
* **Row stride must be a multiple of 64 bytes**, i.e. width a multiple of 32
  fp16 elements. Width 131 compiles, builds a valid request, then fails
  `evaluate` with `status=0x1d`. Pad to 160.
* **An output narrower than the input** trips the same 0x1d if you let
  `_ensure_io` size the output surface from `seq_len`. Allocate explicitly when
  widths differ.
* **Max output channels for one conv** is between 62080 (works) and 124160
  (fails). `lm_head` is chunked along the vocabulary for this reason.
* **`pad`+`add` output width** caps between 9216 (works) and 11264 (fails),
  far below a plain conv's 62080.
* **16 blobs per program.** 24 fails immediately.

## Weights

* Quantized paths use signed int4/int8 with per-output-channel scales. `uint8`
  and unpacked int4 compile and silently return garbage; int4 must be packed.
* fp16 weights work and are the current semantic-correctness path. The complete
  model uses 51.42 GB of fp16 blobs. `kANEFKeepModelMemoryWiredKey=0` is required
  so this footprint remains pageable.
* Blockwise scales, inline scale literals and offsets are all rejected.
* Concatenating weights along output rows is exact — each row keeps its own
  scale, and no partial sums cross the boundary. This is what makes fused
  projections and vocabulary chunking safe.

## Program limits

**Exactly 127 distinct resident programs.** Program 128 fails with
`Program load failure (0x50004)`. It is not memory (fails with 9.8 GB free, and
a synthetic build fails at 127 holding only 5.31 GB), not blob size, and not
blob count.

> **Capacity probes must vary the MIL *text*, not just the weights.**
> `ANECCompile` is content-addressed: identical MIL deduplicates to a single
> resident model. A probe that varies only weight values will appear to load
> hundreds of programs and is measuring one program loaded repeatedly. This
> error cost real time here.

## Procedure-bank weight offsets

Multi-procedure dispatch works. The earlier result where every procedure
appeared to use procedure 0's weights was a packed-blob bug: each milinternal
tensor header's payload pointer (header offset 80) must contain the payload's
**absolute file offset**, not `0x80` for every concatenated tensor. Once fixed,
procedures 0, 1, and 2 produced bit-identical results to separate one-procedure
models. The pointer is uint32, so fp16 banks must be split before any payload
offset reaches 4 GiB; the pure runtime uses four 12-layer GDN banks plus one
attention bank.

## Hardware facts

* **One ANE.** `ioreg` shows a single `ANE0`. `kANEFAneInstanceHint` accepts 1–4
  but gives no parallelism: two programs under hints 1 and 2 driven concurrently
  take 3.810 ms against 4.057 ms serialized (1.06×), where true parallelism
  would be 2.03 ms.
* **~10 TFLOP/s sustained**, flat from S=64 upward. 5.7–6.4 W saturated.
* **Weight streaming 150 GB/s**, IOSurface activations 70 GB/s, CoreML state
  7–20 GB/s.
* **Decode cost is flat from 1 to 32 tokens** — the hardware pads to width 32,
  so a 1-token step computes 32 lanes and discards 31.

## Debugging notes

* The engine swallows compiler output into a discarded buffer. Surface it on
  failure or you will be guessing.
* `identity` as a program's **sole** output returns NaN. With two outputs it is
  **required** — emitting a tensor as an output while other ops also consume it
  likewise gives NaN. Two opposite rules, both measured.
* `mx.array` on a strided numpy view takes a CPU path; use
  `np.ascontiguousarray`.
* MLX streams are per-thread. `ThreadingHTTPServer` kills generation with
  "There is no Stream(cpu, 0) in current thread"; serve single-threaded.
* **Verify ANE numerics against real weights, not random ones.** Random Gaussian
  weights through a 17408-wide MLP overflow fp16 and produce NaN that looks like
  a hardware bug. This sent the investigation down a false path more than once.
* Phase timers that straddle an MLX sync bill upstream GPU work to whatever
  forced the evaluation. A fused layer "costing 30 ms" was 25 ms of waiting for
  the GDN forward.
