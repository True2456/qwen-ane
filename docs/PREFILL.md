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
