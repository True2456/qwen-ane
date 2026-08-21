# How the model is mapped onto ANE programs

There are two backends in this repository. The original hybrid server uses a
69-program chained layout and leaves sequence cores on MLX/GPU. The standalone
`tools/pure_ane.py` backend uses 120 programs at direct context 256 and 125 for
the default int4 long-context scan set, and executes every learned tensor
operation on the ANE. Both
designs are constrained by an empirical system-wide limit in the private
`_ANEInMemoryModel` loader; a naive mapping wants several hundred.

The pure backend also has a persistent HTTP wrapper in
`tools/pure_ane_server.py`. It retains one baked `PureAneRuntime` for the life
of the process and serializes requests around its mutable sequence state. A
request reset zeros the 48 compact GDN IOSurfaces and convolution histories,
sets attention offsets back to zero, and leaves long-context KV pages alone;
masked stale entries are overwritten before they can become valid. Repeated
chat and benchmark requests therefore do not recompile and cannot accidentally
inherit another conversation's state.

One prompt-boundary snapshot is retained for automatic prefix reuse. It copies
the compact GDN recurrence states and convolution histories plus the last hidden
vector and logits; attention K/V stays in the existing block-major arrays, so
only logical offsets need snapshotting. A hit is allowed only when the cached
token tuple is an exact prefix and its execution mode matches (greedy/MTP or
sampled target-only). New suffix tokens overwrite only positions after that
boundary. An unrelated request invalidates the entry before overwriting early
KV slots. This is state caching, not response memoization: decoding still runs.

Reasoning controls are rendered before tokenization, using the three values and
exact instruction strings in the checkpoint's `tokenizer_config.json`. They
therefore participate naturally in prefix identity: changing from `medium` to
`low` or `xhigh` changes the token prefix and forces a safe miss. Generated text
is split at Qwen's closing `</think>` marker; the implicit opening marker lives
in the generation prompt. Reasoning, final content, and native tool XML are then
routed independently to their OpenAI response fields.

## Pure backend layout

| block | programs |
|---|---:|
| 48 GDN depthwise convolutions | 48 |
| 64 complete layer tails | 64 |
| shared GDN recurrence, attention prepare, and attention core | 3 |
| layer-0 norm/projection | 1 |
| final norm + vocabulary head chunks | 4 |
| **direct-context target total** | **120** |

Layers 1–63 have no standalone input-projection dispatch: every preceding tail
returns both its current hidden state and the next layer's normalized learned
projection. This keeps the blob count below 16 (13 for GDN, 10 for attention),
removes the two quantized target projection banks, and saves 63 dispatches.

Pure MTP adds three resident programs: its projection bank plus fusion and
tail. The fusion projection remains fp16 for
acceptance quality; the MTP attention/MLP tail uses the selected target
precision. MTP QKV is procedure 0 in its draft-only projection bank,
and both target and drafter share four vocabulary projection programs by
supplying either `model.language_model.norm.weight` or `mtp.norm.weight` as
runtime data.

Long context replaces the one direct attention core with a direct core, a
one-block statistics core, an ANE online-softmax combiner, and selected
multi-block scan programs. The configured set is chosen to stay within 127
distinct loaded models:

| long-context configuration | scan groups | total programs |
|---|---|---:|
| int4/int8 target | 1, 4, 16, 32 blocks | 125 |
| int4/int8 + MTP | 1, 32 blocks | 126 |
| fp16 target | 1 block | 122 |
| fp16 + MTP | — | unsupported on the current loader path (would need 129) |

Missing group sizes fall back to repeated one-block scans, changing dispatch
count but not attention math or maximum context.

## Long-context attention state

The checkpoint declares 262,144 positions. For positions 1–256 the original
direct-softmax core is retained, preserving the qualified short-context token
path. Later positions use exact online softmax:

```text
block-major fp16 K/V
  → ANE scans 256-token blocks (up to 32 blocks per submission)
  → each scan emits normalized value, block max, scaled exp-sum
  → ANE pairwise merge emits the exact global normalized value
```

Every scan uses the same `1/8192` denominator scale. Since it is common to all
blocks it cancels from the normalized result, while preventing fp16 denominator
overflow at 256K. K/V storage is `[block, kv_head, 256, head_dim]`, so each
submitted block is contiguous. Reset only restores the logical offset; stale
entries remain masked until overwritten, avoiding a 16 GiB clear at 256K.

The 16 target caches consume 64 KiB per configured token in aggregate. MTP's
17th attention cache adds 4 KiB/token. Dense 256x256 RoPE matrices are held in
an eight-position lazy cache shared by all attention layers rather than one per
possible position.

The remaining MTP projection bank is a multi-procedure-capable model. Packed weight payload offsets
inside `weight.bin` are absolute file offsets; treating every payload as if it
started at `0x80` makes procedure 0 appear correct while later procedures read
the wrong weights. Splitting fp16 GDN projections into 12-layer banks also
keeps every internal uint32 payload offset below 4 GiB.

Two numerical carriers are essential. Residual RMSNorm now slices up to three
independent decode lanes, reshapes each from channels onto width, max-scales
dynamically, multiplies by 64, reduces bounded squares, pads each result back to
its physical lane, and adds the disjoint tensors. This avoids both fp16 overflow
and inaccurate tiny means without assuming duplicate lanes. GDN recurrence
output is also carried at 64x so values near `1e-6` are
not quantized away, with `epsilon*64^2` in the immediately following RMSNorm.

The remainder of this document describes the original hybrid chain.

## The naive mapping, and why it fails

One program per linear: 64 MLPs × 3, plus 48 GDN layers × 4 input projections,
plus 16 attention layers × 4, plus `lm_head` — several hundred programs. Dead on
arrival.

## Fusion 1: linears that share an input become one conv

Weights concatenate along output rows, which is exact (each row keeps its own
int4 scale, no partial sums cross the boundary). The first module runs the
dispatch and stashes the result; the others read their slice back out.

* GDN calls `in_proj_qkv`, `_z`, `_b`, `_a` on the same tensor → one
  `[16480, 5120]` conv. Four matmuls, one dispatch, one program instead of four.
* Attention's `q_proj`/`k_proj`/`v_proj` → one `[14336, 5120]` conv.
* MLP's `gate_proj`/`up_proj` → one `[34816, 5120]` conv, sliced in-graph.

Without this, GDN alone needs 192 programs.

## Fusion 2: the layer tail

Within a layer the sequence after the attention/GDN core is

```
out_proj → +residual → RMSNorm → gate/up → silu → mul → down → +residual
```

and the `add` and the norm both run on the ANE, so all of it is **one program**.
The two inputs (the core output and the residual) are concatenated into one
surface and sliced apart in-graph, which avoids multi-input request plumbing.
`out_proj` is replaced by an identity so the attention module hands back its
pre-projection core.

This puts `out_proj` and `post_attention_layernorm` on the ANE at **zero extra
program cost** — still one program per layer.

## Fusion 3: the chain — each layer emits the next layer's head

This is what broke the ceiling. A MIL func can return **two** tensors, so each
layer's program also computes the *next* layer's `input_layernorm` and input
projection, and emits it on a second output surface:

```
program for layer N:
  in:  concat(core_N, residual_N)
  out: y      = layer N's output
       y2     = in_proj(input_layernorm(y))  for layer N+1
```

The next layer's projection modules then read their slice out of `y2` instead of
computing anything. **The 48 GDN and 16 attention projection programs disappear
entirely.**

An earlier attempt emitted both halves in one wide tensor via `pad`+`pad`+`add`;
that caps near 9216 channels, too narrow for GDN's 16480 rows. Two output
surfaces have no width limit.

## The resulting budget

| block | programs |
|---|---|
| 64 layer tails, 63 with the next layer's head folded in | 64 |
| layer 0's input projection (no predecessor to fold into) | 1 |
| `lm_head`, 4 vocabulary chunks | 4 |
| **total** | **69 / 127** |

58 slots spare. `lm_head` is chunked because one conv tops out between 62080 and
124160 output rows; 4 chunks measured fastest (3.328 ms/position vs 3.474 at 8
and 3.835 at 16), and 2 chunks fails to compile at 124160 rows.

## The gated-delta recurrence

Implemented and verified (`probes/ane_gdn_step.py`, rel 9e-4), enabled with
`--ane-gdn-step`. **One program serves all 48 layers** — the recurrence carries
no per-layer weights, state arrives as data.

The layout is the whole trick. Store `state[h, dv, dk]` at channel `h*Dk+dk`,
width `dv`; then

| operation | becomes |
|---|---|
| `kv_mem = (state·k).sum(-1)` | grouped conv, ones weights, `groups=H`, Dk→1 |
| `state += k ⊗ delta` | grouped conv `1→Dk` to broadcast delta, then `mul`+`add` |
| `y = (state·q).sum(-1)` | grouped conv, ones weights, Dk→1 |
| `k`, `q`, `decay`, `beta` | width-1 columns broadcast across Dv |

All six tensors ride in on one surface (state in columns 0..Dv-1, decay/k/q in
columns Dv..Dv+2, v and beta in extra channels); `y` and the new state leave on
two output surfaces. Input width is padded to 160 for the 64-byte row-stride
rule.

The initial implementation was 4.393 ms/call against ~0.66 ms on GPU because
it round-tripped 3.5 MB of state through NumPy and transposed
`[H,Dv,Dk]` twice. The server now keeps a compact `[H*Dk,Dv]` IOSurface per
cache marker, imports MLX state once after prefill, and binds each new state
directly back to that surface. The width-160 compiler input still requires one
row-strided 1.57 MB IOSurface copy (`0.036 ms` measured), but no per-token state
read, transpose, MLX conversion, or GPU work. The graph also forms decay and
beta from raw a/b with polynomial softplus and exp/divide sigmoid. It measures
`0.344 ms` including gates, state copy, parameter writes, ANE dispatch, and y
read.

## Still on the GPU in the hybrid backend

Embeddings (needs `gather`, which the ANE rejects), the attention core
(softmax·QKᵀ·V), the GDN `conv1d`, and by default the recurrence.

These are current runtime boundaries, not all hard ANE limitations.  Real-shape
probes now establish that the full grouped-query attention core and the
10,240-channel GDN depthwise convolution execute correctly on ANE.  Attention
needs 256-token online-softmax chunks above that direct-program ceiling; GDN
needs an exp/divide SiLU rather than MIL's inaccurate direct sigmoid.  The open
state-layout compiler restriction is handled by the resident IOSurface path.
The scalar softplus/decay gate passes using an fp16-safe `t*P5(t)` log1p
approximation. See [FULL-ANE-FEASIBILITY.md](FULL-ANE-FEASIBILITY.md).

The pure backend implements those sequence cores directly: grouped-query
attention through the checkpoint maximum of 262,144 positions, per-layer causal GDN convolution, polynomial
softplus/decay/beta, and resident gated-delta state. Its fp16 semantic inference
test matches the MLX reference prefix without importing a GPU tensor runtime.

## Pure MTP state machine

The target accepts 1–3 positions per call. Projection banks, layer tails, and
the shared vocabulary head process the positions together. Causality is
preserved by advancing each GDN recurrence and attention cache in position
order before the batched tail executes.

Before verification the scheduler copies the 48 compact fp16 GDN surfaces and
their depthwise-convolution histories, and records attention offsets. If all
drafts match, the verified state is already committed. On a mismatch it
restores that snapshot, replays `[confirmed] + accepted_drafts`, and folds the
target correction back through the MTP cache. Appended KV entries need no copy:
restoring the offset masks them and replay overwrites them.

## Memory

Two separate problems, both fixed:

* `mlx_lm.load(lazy=True)` — eager loading left **0.2 GB free** mid-bake with
  RSS 52.4 GB, because the whole bf16 model sat resident while the ANE tried to
  wire 12 GB of blobs. Lazy loading: RSS 31.4 GB, 13.2 GB free.
* A **row-blocked quantiser** — quantising a `[34816, 5120]` weight in one shot
  allocated 27.5× the resulting blob size in numpy temporaries.

After baking, MLX-side weights are replaced with 1-element placeholders
(`_free_mlx`), returning ~50 GB. This deliberately kills the GPU fallback for
baked modules; `--keep-mlx-weights` opts out.
