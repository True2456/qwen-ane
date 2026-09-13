# Sampling, and speculation that stays honest under it

The model card asks for `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0` in
thinking mode. Until now this port could only run at temperature 0, and not
because sampling was missing: the verification step was wrong for anything else.

A block accepted a draft when it equalled the backbone's argmax. That is right
at temperature 0. Sample the final token while accepting drafts on an argmax
match and the output is biased toward whatever the drafter proposed — it looks
sampled, and comes from the wrong distribution.

## The rule

Speculative sampling accepts a draft `x` drawn from the drafter's `q` with
probability `min(1, p(x)/q(x))`, and on rejection draws from the normalised
positive part of `p - q`. Both drafters here propose a single token with no
distribution of their own: the MTP head is used greedily and the context lookup
has none at all. With `q` a point mass the rule collapses to something simple.

* Accept the draft with probability `p(x)`, the target's own mass on it.
* On rejection, draw from `p` with `x` removed and renormalised.
* Having accepted every draft, draw the bonus token from `p` as usual.

`p` is the distribution after temperature, top_k, nucleus and min_p, applied in
that order, because that is the order serving frameworks use and it is what the
numbers on the model card mean.

`runtime/flashnext_sampling.py` holds it and `tests/test_sampling.py` checks
it: forty thousand trials with a deliberately mediocre drafter, and the emitted
token distribution matches the target to within one point for every token in
the filtered set. Temperature 0 still takes the old path, and a greedy decode
emits the same 64 ids it always has at 21.4 tok/s.

## What it costs

Sixty-four tokens on the code prompt, table on, seed fixed:

| | tok/s | tokens a pass | drafts accepted |
| --- | --- | --- | --- |
| greedy | 18.5 | 2.91 | 44/63 (70%) |
| thinking mode | 17.5 | 2.78 | 42/66 (64%) |

5%, which is less than expected. The worry was that acceptance would collapse,
since a draft now survives only with probability equal to the target's mass on
it rather than on an exact match. It holds up because `top_k=20` with
`top_p=0.95` still leaves a peaked distribution on this kind of text. Flatter
text should cost more, and that has not been measured.

## Using it

Per request in serve mode, which is where it belongs:

```json
{"op": "gen", "template": true, "messages": [...], "max_new": 400,
 "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
```

Or as `FLASHNEXT_TEMP`, `FLASHNEXT_TOP_P`, `FLASHNEXT_TOP_K`, `FLASHNEXT_MIN_P`
for a whole process, with `FLASHNEXT_SEED` to make a run repeatable. The
default stays at temperature 0 so the evaluation harness remains reproducible.

`presence_penalty` is not implemented. Thinking mode wants 0.0 so nothing is
missing for that configuration, but instruct mode asks for 1.5 and would need
per-request token counting. It would also lower acceptance, because the drafter
does not know about the penalty and its proposals would be penalised at
verification.
