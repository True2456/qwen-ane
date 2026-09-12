# What the ANE port costs in quality

Everything in this port so far has been measured in milliseconds. This is the
first measurement of what it costs in accuracy: the whole ANE path against the
unmodified 4-bit MLX model, on the same tokens, with context accumulating the
same way in both.

The port differs from the reference in three ways at once — int8 weights and
int8 activations on the ANE, forty-eight hand-written MIL graphs instead of the
MLX layers, and fp16 mixers and shared expert — so this is one number for all
of it, not an ablation.

## Result

Teacher-forced negative log likelihood over 2048 tokens, continuous context.

| corpus | reference ppl | ANE ppl | cost |
| --- | --- | --- | --- |
| technical prose | 6.5553 | 6.6482 | +1.42% |
| Python source | 3.0806 | 3.1476 | +2.18% |

Over the first 1024 tokens of the same files the split runs the other way
(prose +3.35%, code +1.66%), so the honest reading is that the port costs
somewhere around 2% perplexity and the corpus-to-corpus difference is noise at
this sample size.

This is in line with the earlier standalone measurement of int8 activations
alone, which cost about 1.5%. The MIL graphs and the fp16 mixers add little on
top of it.

## Method

Both arms score the same token stream: token *i* is predicted from tokens
0 through *i-1*, and the prediction for every token is scored, not just the
last of a window. The ANE arm drives the same K=4 graphs the speculator
verifies with, feeding the true continuation instead of drafts, so every slot
is accepted and every commit is a real commit. It therefore measures the
decode path as shipped, at the block rate rather than the token rate — about
22 tok/s for scoring.

The reference arm feeds the file through `mlx_lm` in 256-token chunks sharing
one prompt cache, so its context grows the same way.

The two corpora are checked in under `eval/`. Prose is this repository's own
documentation and code is three of its runtime modules, which is a fair proxy
for what this model is asked to do and avoids any question about what a
downloaded benchmark set contains.

```bash
PYTHONPATH=/Users/true/.mlx128/mlx-lm:/Users/true/.mlx128/mlx/python \
  ~/.rindi/venvs/coreai/bin/python probes/flashnext_ppl_mlx.py eval/prose.txt 2048
```

```bash
FLASHNEXT_SPEC=4 FLASHNEXT_MOE=mlxresident FLASHNEXT_HEAD=mlx \
FLASHNEXT_MIL_GDN=1 FLASHNEXT_MIL_QSA=1 FLASHNEXT_NO_DRAFTER=1 \
~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py generate \
  --ppl-file eval/prose.txt --ppl-tokens 2048
```

## Long context: the gap widens with position

The port claims 256k. Scoring further into the same file shows the cost is not
flat in position.

| tokens scored | reference ppl | ANE ppl | cost |
| --- | --- | --- | --- |
| 1024 | 7.4534 | 7.7028 | +3.35% |
| 2048 | 6.5553 | 6.6482 | +1.42% |
| 4096 | 5.4735 | 5.7135 | +4.38% |
| 8192 | 4.8725 | 5.4899 | +12.67% |

Read on its own that table overstates the effect, because the corpus is not
homogeneous: the later documents are dense with numbers and tables, and the
port is simply worse on that material wherever it appears. Taking tokens 6144
to 8192 out and scoring them as their own document, starting from an empty
context, gives 7.8363 against 8.5403 — a gap of 0.086 in log likelihood where
the first 2048 tokens of the same corpus gave 0.014. Content alone moves the
cost by six times.

Controlling for it by comparing those same tokens in both places: 0.086 when
they start the context, about 0.23 when they sit at position 6-8k. So there is
a real position effect on top of the content effect, worth roughly a factor of
2.7 between an empty context and eight thousand tokens.

## Where the long-context loss is not

Three candidates were tested and eliminated.

**Not the indexer's precision.** The front program projects the indexer's
queries and keys in int8 on the ANE. Recomputing that projection on the host in
fp32 changes nothing: 5.7226 against 5.7135 at 4096 tokens, which is noise in
the wrong direction.

**Not the shared key set.** A K=4 block selects one set of keys for all four
slots, where the reference selects per token. If that were the cost it would
land on the later slots, and it does not — per-slot log likelihood across 4096
tokens is 1.7615, 1.7065, 1.7042, 1.7992. Narrowing to K=2 is slightly worse,
not better: 5.7563.

**Not a silent fallback.** Past the budget the selection ran on every one of
6144 calls, never fell back, never lost its prefix, and 71.3% of the keys it
chose were in the most recent 2048 — so it is selecting rather than quietly
doing recency.

What the selection is worth can be priced directly. Forcing recency only
(`FLASHNEXT_IDX_RECENCY=1`) gives 5.8207 at 4096 tokens against 5.7135 with
selection, so the block selection buys 0.019 in log likelihood while the
remaining gap to the reference is 0.043. The selection works; it is just worth
less here than it is in the reference, and the rest of the gap is elsewhere.

## What this does not cover

Perplexity is not a task score. It says the distributions stayed close; it does
not say the model still gets a benchmark right. Nothing here goes past 8192
tokens, which is a thirty-second of what the port claims to support, and the
remaining long-context gap is measured but not explained.
