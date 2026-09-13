# What the reverse-engineered ANE guide says, and what it changes

Source: arXiv 2606.22283v1, a reverse-engineered guide to the Neural Engine
built from silicon measurement plus decompilation of the runtime, compiler,
driver and firmware. It is evidence, not documentation, and the claims below
should be re-measured before anything depends on them. Three of them bear
directly on decisions in this repository.

## The routed experts cannot go on the ANE

> control flow must be static ... the work the engine performs cannot depend on
> values computed during the run ... weights cannot be dynamically selected or
> indexed at runtime

That is problem 3 of `CHALLENGE-ANE-ROUND-3.md`, and it says the shape we were
hoping for does not exist. A mixture of experts picks ten weight sets per token
from five hundred and twelve, from a routing decision computed during the run.
The engine cannot do that.

The one remaining spelling is a procedure per expert with the host supplying
the index, which `compile_multiproc` already supports. Ten experts across
forty-eight layers is four hundred and eighty submits a token, and a submit
costs about 1.17 ms before it computes anything. That is not a close call.

So the item worth about a third of both prefill and decode, and the only one
that would let adjacent layers fuse, is closed. What remains of it is the
weaker version: keep a resident subset of experts baked into the graph and fall
back to the GPU for misses, which is a cache hit-rate question rather than an
architecture one.

## The 2 MB working set explains the recurrence

> a 2 MB on-chip working-set threshold ... weights stream into the datapath
> on-demand ... fetched per layer from DRAM

The GDN recurrent state is 48 x 128 x 128 in fp16, which is 1.5 MiB. Against a
2 MB working set that leaves almost nothing for the activations and the weights
streaming alongside it, so the state cannot stay on chip and makes a DRAM round
trip every token.

That is exactly the cost we measured and could not explain: 0.145 ms a token a
layer, 5.2 ms a token across 36 layers, and the one term a wider graph does not
amortise. It also explains why the chunked delta rule helped at all, since its
whole effect is to touch the state once a chunk instead of once a token.

It suggests something not yet tried: **shrink the state under the threshold**.
At int8 it is 786 KB, which leaves room for the rest of the layer. If that
keeps the recurrence on chip the per-token cost should fall by much more than
the halved bytes. The risk is precision, since this state accumulates over
thousands of tokens, and `probes/mil_k_check.py` is the gate that would say so.

## Weight formats

> int4 palette lookup ... streams natively across all generations, about 2.37x
> over dense fp16. int8 affine streams natively from A14/M2, half the bytes of
> fp16. sparsity streams at 50% zeros or better, 1.55-1.64x at 0.43x the bytes.

The int8 projections are on the streaming path, which is what the cost model
already implied. The remaining fp16 weights are the mixers and the shared
expert, about 36 MB and roughly 4 ms a pass, and int8 was rejected for them on
error rather than on speed.

Int4 palette is a different offer: a palette maps arbitrary fp16 values, so it
is not the uniform int4 that failed the error gate. Against it,
`runtime/expert_bank.py` already records Core AI palettisation of the GDN and
QSA convolutions at rel 0.31, which is far outside any gate here. Worth
re-checking per channel rather than globally before believing either number.
