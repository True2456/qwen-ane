# Context lookup drafts: 16.4 to 20.7 tok/s on code

Speculative decoding here is limited by how good the drafts are, not by how
fast the block verifies. The MTP head gets the first draft position right about
92% of the time and then collapses: 40% at the second position, less at the
third. It is a one-layer module being handed its own output where it was
trained on a forty-eight-layer backbone state, so chaining it compounds the
mismatch. Widening the block does not rescue it — at K=8 only 24% of drafts
were accepted, and throughput fell from 16.0 to 12.5 tok/s.

Text that repeats itself does not need the head. When the last three tokens
have occurred before, the token that followed them then is a very good guess
and costs a dictionary lookup. `runtime/flashnext_ngram.py` keeps a
last-occurrence index of 3-grams over the prompt plus everything generated,
and `MtpDrafter.draft` consults it at every step.

A hit replaces the head's argmax and is fed back into the chain, so the head
continues along the matched path rather than its own. The head still runs every
step even on a hit, because the drafter's attention cache needs a row per
drafted position for a partly accepted block to unwind. The lookup therefore
costs nothing but a dictionary probe and can only change which token is
proposed, never how many.

## Measured

Sixty-four tokens, greedy, `FLASHNEXT_SPEC=4`. Every configuration below emits
byte-identical output, which is guaranteed: the block still only ever emits the
backbone's own argmax, so a bad draft costs a rejected slot and nothing else.

| prompt | head only | with lookup | lookup fired |
| --- | --- | --- | --- |
| code, an edit-style repetition | 16.43 | 20.73 | 38 of 57 steps |
| a repeating list of capitals | 16.04 | 16.63 | 20 of 66 steps |
| prose | 14.33 | 14.52 | 16 of 81 steps |

Accepted drafts go from 58% to 77% on the code prompt, and tokens per pass from
2.67 to 3.20 against a K=4 ceiling of 4.

## Order

Three is the default and won on every prompt.

| order | code tok/s | fired | prose tok/s | fired |
| --- | --- | --- | --- | --- |
| 2 | 17.27 | 55 of 69 | 14.80 | 33 of 81 |
| 3 | 20.73 | 38 of 57 | 14.52 | 16 of 81 |
| 4 | 18.44 | 43 of 66 | 14.58 | 11 of 84 |
| 5 | 19.18 | 35 of 63 | 14.50 | 7 of 84 |

Two fires on any common bigram and proposes noise. Four and five fire rarely,
and mostly where the head was already going to be right. Set with
`FLASHNEXT_NGRAM_G`; turn the whole thing off with `FLASHNEXT_NGRAM=0`.

## Wider blocks, once the drafts are good

With the lookup carrying the chain, K=8 becomes the better width on repetitive
text, because the drafts survive far enough to fill it.

| prompt | K=4 | K=8 |
| --- | --- | --- |
| code | 20.73 | **24.41** |
| capitals | 16.63 | 16.80 |
| prose | 14.52 | 12.02 |

K=8 confirms 4.92 tokens a pass on the code prompt against 3.20 at K=4. It
loses on prose for the original reason: with the lookup quiet, seven chained
head drafts are mostly wrong and the wider block costs more to verify. Choosing
the width per block would need both sets of graphs compiled, which is about 19
extra seconds of startup and another copy of the baked weights on the ANE.
The default stays at K=4.
