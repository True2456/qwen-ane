# ANE/GPU overlap: what happened when we tried it

Problem 2 of `CHALLENGE-ANE-ROUND-2.md` asked whether the 28 ms the GPU spends
on the routed MoE can be hidden behind ANE work. The answer is no, for a
structural reason, and the attempt produced one smaller win that we kept.

Everything below was re-measured in this repository on the 32-token capitals
prompt with `--max-new 64` and `FLASHNEXT_SPEC=4`. Every configuration emits
the same 64 token ids, so the comparisons are like for like.

## Result 1: token-split pipelining is a large regression

`FLASHNEXT_PIPE=1` splits the K=4 verification block into two micro-batches and
wavefronts them, so the GPU MoE of half *h* runs while the ANE computes half
*h+1*. It works and it is correct. It is also much slower.

| configuration | tok/s |
| --- | --- |
| serial (`FLASHNEXT_PIPE=0`) | 16.04 |
| wavefront (`FLASHNEXT_PIPE=1`) | 12.20 |

The overlap counter says 50.8 ms of GPU work really did run concurrently with
the ANE, so the pipeline is not broken. It loses anyway, on both sides at once:

| per block, ms | serial | pipelined |
| --- | --- | --- |
| GDN on the ANE | 68.1 | 88.1 |
| GDN MoE on the GPU | 29.3 | 50.9 |
| router | 6.3 | 14.8 |

**Why the ANE side got worse.** A GDN submit costs about 1.13 ms that does not
depend on the token count, plus about 0.146 ms per token. The fixed part is the
two big projections, whose cost follows the weight tiles rather than the
activations — see `ANE-MIL-COST-MODEL.md`. Splitting K=4 into two submits of
two therefore pays the fixed 1.13 ms twice and buys nothing.

**Why the GPU side got worse too.** The routed experts are gather-bound, and
two `mx.eval` calls over two tokens each move nearly the same expert rows as one
call over four while paying the launch cost twice.

So the thing being hidden, about 0.82 ms of MoE per GDN layer, is smaller than
the 1.13 ms of extra ANE submit cost that hiding it creates. That inequality is
not close, and it does not depend on how the split is scheduled.

## Result 2: the ANE cannot be enqueued ahead of the GPU

The only way to overlap without splitting the block is to start layer *i+1* on
the ANE before the GPU finishes layer *i*. That needs a non-blocking submit.
None of the private-path routes to one work:

* `evaluateWithQoS:options:request:error:` blocks. Running it on a worker
  thread does overlap with MLX, because the call releases the GIL, but the
  ANE itself is still busy for the whole call.
* Attaching a Metal `IOSurfaceSharedEvent` through `setSharedEvents:` is
  ignored. `evaluate` returned in 0.20 ms with the event still at 0.
* `_ANEVirtualClient` is nil on `sharedConnection`, so `completionEvent:` is
  not reachable.
* `enqueueSetsWithModel:` returns false.

`AneEngine.submit_async` and `runtime/ane_async.py` are the worker-thread
version and are kept, because the GDN backend's `run`/`finish` split is what
made the wavefront experiment measurable. They are not on the default path.

## Result 3: eager MTP drafting is a real win, and is now the default

The drafter and the commit touch different engines: drafting is MLX on the GPU,
while committing a block copies recurrent state between ANE IOSurfaces and
trims the QSA cache. Running the next block's draft on the main thread while
the commit runs on a worker hides the whole draft.

| configuration | tok/s |
| --- | --- |
| `FLASHNEXT_EAGER=1` | 16.06, 15.89, 16.04 |
| `FLASHNEXT_EAGER=0` | 15.01, 15.69 |

Drafting goes from 374 ms to 0 ms of measured serial time, with 336 ms of it
reported as overlap. Roughly 3% end to end, same tokens, and it is on by
default.

MLX streams are thread-local, so the draft must stay on the thread that owns
the stream; moving it to the worker instead fails with `There is no
Stream(gpu, 1)`. The commit is pure NumPy and IOSurface work and is safe there.

## What is left

Getting past 20 tok/s needs either a per-layer ANE mailbox with a mid-graph GPU
kickoff, which the private path does not expose, or the routed MoE itself
moved off the GPU. The block is otherwise a strictly serial alternation and
each half is at its own floor.
