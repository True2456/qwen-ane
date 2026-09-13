# Serving this for agent work

`generate --serve` holds the model and answers one JSON request a line on
stdin. Ops: `gen` (text or chat messages in, text out), `score` (next-token
logprobs for given continuations), `reset`, `state_hash` and `restore_check`
for debugging, `quit`.

```json
{"op": "gen", "template": true, "messages": [...], "max_new": 400,
 "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
```

Generation stops at the model's own end-of-turn tokens from
`generation_config.json`, or at any `stop` string, or at `max_new`.

## Prefix reuse

An agent loop resends its whole history every turn, and at 84 tok/s a 16k
conversation would spend three minutes prefilling before each reply. It does
not, because the state is checkpointed.

Four turns of a 1100-token agent conversation:

| turn | prompt | reused | prefill | whole turn |
| --- | --- | --- | --- | --- |
| 0 | 1097 | 0 | 12.9s | 16.6s |
| 1 | 1165 | 1096 | 0.84s | 6.2s |
| 2 | 1233 | 1164 | 0.83s | 6.2s |
| 3 | 1301 | 1232 | 0.84s | 6.3s |

15x on prefill and 2.7x on the turn.

A GDN layer's state is recurrent, so there is no rewinding to an arbitrary
point — only to somewhere a copy was kept. Each prompt is checkpointed after it
is prefilled, and a new request restores the longest checkpoint that is a
prefix of it. The first checkpoint is never evicted, because it is usually the
system prompt and tool list that every turn shares;
`FLASHNEXT_CHECKPOINTS` sets how many of the rest to keep (4).

A checkpoint is the recurrent state and conv window of each GDN layer, each
QSA layer's keys, values, offset and indexer blocks, the drafter's cache, and
the per-layer embedding's history and conv window. About 90 MB at 1200 tokens,
growing with the QSA cache.

Appending alone is not enough, which is why this needs checkpoints rather than
a served-token comparison: the chat template rewrites history, and Qwen drops
thinking blocks from previous assistant turns, so the next prompt is not an
extension of what was last served.

## Reuse changes the text, and does not lose information

The copy is exact. `restore_check` snapshots the live state, puts it straight
back, and compares 109 hashes across every layer: none differ.

The generated text still changes between a warm and a cold run of the same
prompt. That is not the restore. The state depends on how tokens were grouped
into blocks as well as on the tokens themselves, because the chunked prefill
graph and the K-slot decode graph are different approximations of the same
recurrence. Reuse changes where the block boundaries fall, so it changes the
state by about the port's own error.

Scoring the same prompt warm and cold, the logprobs of eight candidate
continuations differ by 0.1 to 1.9 nats and agree on the top one. Both sit 2 to
7 nats from the MLX arm, which is further than they are from each other. So
warm and cold are the same model rounding differently, not one of them being
wrong. Greedy output is reproducible for a given cache state, not across cache
states — the same property a batching server has.

## Not there yet

No HTTP endpoint; this speaks JSON lines on a pipe. No streaming, so a caller
waits for the whole turn. No tool definitions passed to the template. No
`presence_penalty`, which thinking mode does not want anyway. Long context is
validated to 8192 tokens.
