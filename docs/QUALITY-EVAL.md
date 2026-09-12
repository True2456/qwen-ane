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

## What this does not cover

Perplexity is not a task score. It says the distributions stayed close; it does
not say the model still gets a benchmark right. It also says nothing about long
context: 2048 tokens exercises the first QSA rung and one recurrent state per
layer, and the port claims to support 256k.
