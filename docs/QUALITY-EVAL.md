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

## The per-layer embedding table is on

`FLASHNEXT_PLE` now defaults to 1. Every quality and throughput number
**above** this section was measured with the table as zeros — `mlx_lm` has the
same fallback — so those comparisons stay like for like, and below what the
checkpoint can do.

Turning the table on, scoring 1024 tokens of `eval/prose.txt` after a 512-token
prefill:

| | nll | ppl | scoring rate |
| --- | --- | --- | --- |
| zero table | 1.886208 | 6.5943 | 23.3 tok/s |
| checkpoint rows, serial `pread` | 1.853263 | **6.3806** | 21.5 tok/s |
| checkpoint rows, threaded `pread` + row cache (now the default) | **1.853263** | **6.3806** | 21.8 tok/s |

3.2% better perplexity. For scale, the entire ANE port costs around 2%. The
nll with the faster gather is identical to six decimal places; the table's
contents did not change, only how fast the selected rows arrive.

### What the lookup was doing

One PLE layer (layer 1; `ple_layer_ids: [2]`). Each token hashes to **16 rows**
(trigram, 8 heads per order-2 and order-3), each row 160 BF16 = **320 bytes**,
so 5.12 KB of table per token out of a **102.4 GB** shard set (320,001,536
rows). The table is never materialized. `NgramRows.lookup` issued one `pread`
per row, serially, queue depth one.

On the 1536 tokens of this eval (512 prefill + 1024 scored):

- 24,576 lookups, **18,752 unique** (hit rate 0.237 if cached)
- no duplicates among a token's 16 ids
- **zero overlap** from one token to the next (every hash includes the current
  token)
- 16,704 rows seen once, 1,328 seen 2–4 times, 720 seen 5+ (max 26)
- 7.86 MB if every `pread` happens, 6.00 MB if unique rows are kept

A small cache helps the repeats. It cannot hide the first touch of the 76% of
rows that never come back.

### Isolated timings, measured here (not the 16x note)

Cold disjoint random rows, 256 tokens of 16 ids, two orderings so the second
strategy is not reading the first's pages:

| | ms/token |
| --- | --- |
| serial `pread` | 1.22–1.27 |
| 16 threads, persistent pool | 0.26 |

That is **4.8x**, not 16x. A 4096-row one-shot gather was 307.6 ms serial
against 43.1 ms with 64 threads (7.1x). The 16x figure (630 ms → 39 ms at 8192
rows) is mlx-lm's mmap page-fault gather, a different path, and that same note
turns threading **off** end-to-end because mmap of the table evicts the model.

Other negatives, with numbers:

- Threading **loses** once rows are already in the page cache: 0.025 ms serial
  vs 0.16 ms threaded on the same prose ids after a warm pass. The pool's
  wakeup is larger than a 320-byte hit.
- mmap of the shards looks free after that same warmup (0.025 ms) and is still
  the wrong design under a 79 GB process: each unique row faults a 16 KB page,
  which at long context is tens of gigabytes of cache fighting the model.
- `F_NOCACHE` on the shard fds after a warm process still reads at 0.025 ms —
  the SSD/OS has its own cache. The honest cold number is the disjoint-id
  run above, not a second pass over `eval/prose.txt`.
- Opening a new `ThreadPoolExecutor` per token (cache + thread the misses)
  was **0.35 ms/token**, slower than a persistent pool's 0.15–0.26. The pool
  has to live on `NgramRows`.
- `key_proj` (10240×2560) plus `value_proj` (2560×2560) is **0.66 ms** in
  isolation, ~0.9–1.2 ms inside `CpuPLE.step`. That work runs whether lookup
  is free or not. The zero table skips the whole module, not just the `pread`.
- Batching those GEMVs across K=32 is 10.7x (23.96 ms serial vs 2.24 ms
  batched). Across K=4 it is **0.90x** — slower. Scoring and decode are K=4,
  so this was not shipped; it would also be a different reduction order.

### What changed

`NgramRows.lookup` keeps a userspace cache of the 320-byte rows (6.00 MB on
this eval, cap 262144 rows) and issues cache misses through a persistent
16-thread pool. Shard fds get `F_NOCACHE` so the 16 KB kernel pages do not
accumulate. The BF16→fp32 conversion is the same shift-and-view as before.
`FLASHNEXT_PLE=0` still runs zeros; `FLASHNEXT_PLE_CACHE=0` /
`FLASHNEXT_PLE_THREADS=1` restore the old gather for A/B.

### The throughput gate, from the component timers

End-to-end scoring has a ~5 tok/s run-to-run spread, so 21.8 against 23.3 is
not a measurement of the lookup. The K=4 block that scoring actually runs:

| | ms / block | ms / token | share of a 21.8 tok/s token (45.9 ms) |
| --- | --- | --- | --- |
| PLE total | 7.21 | 1.80 | 3.9% |
| of which lookup | 2.52 | 0.63 | 1.4% |
| of which projections + conv | 4.69 | 1.17 | 2.6% |

Lookup went from ~1.2 ms/token cold serial to **0.63 ms** in the 79 GB
process. The remaining 1.17 ms is `CpuPLE` arithmetic the zero table never
does. Together that is about 4% of scoring, which is why the headline did not
land inside 2% of 23.3 (that bar is 22.83 tok/s). The floor is those GEMVs,
not the gather.

Same block on decode (`--prompt-ids 760 --max-new 64`, spec=4): **8.50 tok/s**
over 64 tokens (1.52 tokens/pass, 19% drafts accepted). Per block
`ple=6.9 ms` / `lookup=2.2 ms` — the same 4% of the block. The 8.50 headline
is the accept rate on `"The"`, not the table.

Prefill, k=32 graphs, 511 tokens:

| run | tok/s | PLE ms/token | lookup ms/token |
| --- | --- | --- | --- |
| ppl prompt | 71.4 | 1.54 | 0.31 |
| `--prompt-len 512 --max-new 4` | **75.0** | 1.50 | 0.40 |

Zero-table prefill on this machine sits around 82 tok/s with a 5 tok/s spread.
PLE is ~1.5 ms of a ~13.3 ms token (**11% of prefill**); lookup is ~3%. The
extra vs scoring is the K=32 loop of serial GEMVs, which isolation says would
batch, and which was left serial so the nll bits stay the ones above.

Cache in the 79 GB process matched the isolated reuse: scoring
`hits=5824 misses=18736 hit_rate=0.237 resident=6.00 MB`; decode of 64 tokens
almost no reuse (`hit_rate=0.030`).

Nothing here has been run through MMLU or GSM8K yet. Perplexity says the table
helps; a task score would say how much.

### Verifying the gather work, and what switching the table on really costs

The bits are unchanged, which was the gate that mattered: scoring 1024 tokens
after a 512-token prefill gives nll 1.853263 with the table on, to every digit,
and `probes/mil_k_check.py` prints the same per-slot errors at K=4 and K=32.

The cost was measured on the scorer, and the scorer is the wrong axis. On the
scorer the table costs 5%, 22.5 tok/s against 23.7, down from 8% before the
gather was threaded. On decode it costs three times that:

| | tok/s | tokens a pass | drafts accepted |
| --- | --- | --- | --- |
| table off | 21.1 | 3.20 | 44/57 |
| table on | 18.7 | 2.91 | 44/63 |

11%, and only part of it is the lookup. The rest is speculation: the table
changes the hidden state the MTP head drafts from, the accept rate falls from
77% to 70%, and a block confirms 2.91 tokens instead of 3.20. The scorer never
sees that because it accepts every slot by construction.

So the trade is 3.2% perplexity for 11% of decode, not for 5%. That is still
probably worth taking, and it is now the default, but it should be decided
against a task score rather than against perplexity, and re-checked after any
change to the drafter.
