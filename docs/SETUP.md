# Setup on another machine

## Requirements

* Apple Silicon with an ANE. Everything here was measured on an **M5 Max**;
  program limits and throughput will differ on other parts.
* **oMLX.app** installed at `/Applications/oMLX.app` for the hybrid server. It
  is not imported or required by the framework-free `pure-*` backend.
* Nothing else. The ANE driver — MIL compilation, IOSurface allocation and the
  `_ANEInMemoryModel` plumbing — is vendored at `runtime/q38_ane_engine.py` and
  needs only the standard library and numpy. Point `Q38_ANE_ENGINE` at another
  checkout of `q38_native_engine` to use that copy instead.
* The model: `Qwen3.8-27B` in MLX/safetensors form. Default path is
  `/Users/<you>/.lmstudio/models/Qwen/Qwen3.8-27B`; override with `Q38_MODEL`.
* Hybrid int4: about 13 GB for ANE blobs, plus model/runtime headroom.
* Pure fp16: 51.42 GB of pageable learned-weight blobs plus runtime headroom,
  and roughly 55 GB of genuinely available disk during a cold compiler bake.

## Run it

```bash
tools/ane serve --ane-chain --ane-lm-head --dense-bits 4
tools/ane bench --ane-chain --ane-lm-head --dense-bits 4 --max-tokens 32
tools/ane spec  --ane-layers 64 --ane-lm-head --draft 2

# Framework-free, GPU-free model execution under system Python:
tools/ane pure-loader-smoke
tools/ane pure-gdn-layer-smoke --bits 16
tools/ane pure-attention-layer-smoke --bits 16
tools/ane pure-attention-long-smoke --context 262144 --valid 8193
tools/ane pure-infer --bits 16 --tokens 4 --verify-reference
tools/ane pure-infer --bits 4 --tokens 32 --mtp-draft 2
tools/ane pure-infer --bits 4 --context 4096 --prompt-file prompt.txt --tokens 32

# Persistent pure-ANE OpenAI endpoint and compile-free repeated benchmark:
tools/ane pure-serve --bits 4 --context 4096 --port 1240
tools/ane pure-bench --url http://127.0.0.1:1240 --tokens 32 --runs 3
```

`tools/ane` is a launcher, and you should use it rather than running the Python
directly. Hybrid commands select oMLX's isolated interpreter. `pure-*` commands
select `/opt/homebrew/bin/python3`, remove `PYTHONPATH`, and enforce a runtime
guard against MLX, PyTorch, and Core ML imports.

> **Do not run `tools/ane_serve.py` with the system Python.** It aborts with
> `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
> initialized.` The launcher runs it under oMLX's interpreter with an isolated
> import path (`-P`), which is what keeps a second `libomp` out of the process.

## Verify the install

Run the probes in dependency order; each is standalone and prints pass/fail.

```bash
O=/Applications/oMLX.app/Contents/Resources
export PYTHONPATH="$O/Python/framework-mlx-base/lib/python3.11/site-packages:$O:$PWD"
PY="$O/Python/cpython-3.11/bin/python3.11"

$PY -u -P probes/ane_rmsnorm.py        # RMSNorm formulations
env -u PYTHONPATH /opt/homebrew/bin/python3 -u -P probes/ane_rmsnorm_lanes.py # three independent stable lanes
$PY -u -P probes/ane_two_outputs.py    # two output tensors per program
$PY -u -P probes/ane_gdn_mech.py       # grouped-conv reduction / broadcast
$PY -u -P probes/ane_gdn_step.py       # a full gated-delta step vs numpy
$PY -u -P probes/ane_gdn_resident_state.py # two dependent resident-state steps
$PY -u -P probes/ane_program_limit.py  # where the program ceiling sits here
```

`ane_program_limit.py` is the one to re-run on new hardware. This repository's
private `_ANEInMemoryModel` path fails loading distinct model 128 on the M5
Max, even with no evaluations in flight. That is not the same claim as the
reverse-engineered hardware queue depth of 127 concurrent evaluation requests;
treat 127 as an API-path budget here, not a documented total ANE limit.

> **The budget is system-wide, not per-process.** Measured: while another
> process held ~122 programs, a fresh process failed to load a 16x16 program
> with `0x50004`. Run nothing else on the ANE while probing or serving, or
> compiles fail for shapes that are otherwise fine.

## Useful flags

| flag | effect |
|---|---|
| `--ane-chain` | layer tails + next layer's head folded in (**the main mode**) |
| `--ane-lm-head` | lm_head on the ANE, `--lm-head-chunks N` (4 is optimal) |
| `--ane-gdn-step` | ANE softplus/decay/beta + recurrence with resident state (0.344 ms standalone) |
| `--dense-bits` | 4 or 8; 4 is the default and the practical choice |
| `--bake-cache` | cache quantised blobs — 52 s cold start → 28 s warm |
| `--eager-load` | materialise the whole model at load (old behaviour, ~55 GB RSS) |
| `--keep-mlx-weights` | keep the GPU fallback alive, at full model memory |
| `--ane-fused-layers -1 --ane-gdn --ane-attn` | the older per-block mode, 127 programs |
| pure `--mtp-draft 0/1/2` | standalone ANE-only MTP; depth 2 measured best |
| pure `--context N` | attention/KV capacity, 256..262144; above 256 uses streamed exact ANE attention |
| pure `--down-proj-parts 1/4` | packed input-channel split for MLP `down_proj`; 4 is the default and unlocks efficient width 64 |
| pure `--prompt-file PATH` | UTF-8 prompt file, useful for long-context tests; overrides `--prompt` |
| `pure-serve --port N` | persistent OpenAI-compatible pure-ANE endpoint; default 1240 |
| `pure-serve --bake-cache DIR` | compressed prequantized cache; measured 60.97 s → 14.50 s warm startup |
| `pure-serve --no-bake-cache` | disable the default `~/Library/Caches/q38-pure-ane` cache |
| `pure-serve --max-tokens N` | default request output limit; clients can override it |
| `pure-bench --runs/--warmup` | benchmark an already-running server without compiling between samples |
| `pure-bench --prefix-cache` | measure cached-prefix latency instead of clean-state benchmark runs |

At 256K, the target's 16 fp16 KV caches are 16 GiB in aggregate; pure MTP adds
1 GiB. Int4 learned-weight blobs add 12.86 GB (13.16 GB with MTP). The 256K
arrays are allocated at runtime and become resident as their blocks are filled.
Long-context fp16 + MTP is currently rejected because that combination needs
129 distinct loaded models; int4/int8 + MTP and fp16 target-only fit at 127.

### Persistent API

Wait for `PURE_ANE_SERVER_READY`, then point an OpenAI-compatible client at
`http://127.0.0.1:1240/v1` with any placeholder API key. The server supports
chat/completions, text completions, SSE streaming, health, metrics, and a
dedicated benchmark endpoint. It accepts one active inference at a time and
queues overlapping requests so mutable recurrent/KV state cannot be mixed.
The most recent prompt snapshot is reused automatically when it is an exact
token prefix of the next request. API responses expose `prefix_cache_hit`,
`prefix_tokens_reused`, and `prompt_tokens_evaluated`; aggregate values are in
`GET /metrics`. Send `"prefix_cache": false` to force clean state. A 256K
configuration reserves sparse KV address space; reset does not zero or fault
all 16 GiB of target cache pages.

OpenAI `tools` and `tool_choice` are accepted for function tools. The server
renders Qwen's native tool schema, converts generated XML to OpenAI
`message.tool_calls`, accepts those calls plus `role: "tool"` results on the
next request, and supports multiple calls. Tool-enabled SSE is structurally
correct but buffers the assistant turn until it can distinguish ordinary text
from a complete tool call.

### Thinking levels

Qwen3.8-27B's bundled template enables thinking when `enable_thinking` is
omitted and resolves an omitted `reasoning_effort` to `xhigh`. It accepts only
`low`, `medium`, and `xhigh`—`high` is not a valid alias for this checkpoint.
The pure server matches those defaults and the exact checkpoint-authored system
instructions. `medium` has no extra system instruction by design.

```bash
curl http://127.0.0.1:1240/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Solve 17*23"}],
    "enable_thinking": true,
    "reasoning_effort": "low"
  }'
```

Non-streaming responses separate `message.reasoning_content` from final
`message.content`. Streaming sends the same two fields as deltas and recognizes
`</think>` even when it crosses token/chunk boundaries. Historical assistant
messages may include `reasoning_content`; the chat renderer preserves it exactly
as the model template requires. `pure-bench` keeps thinking off unless
`--thinking` is supplied, with `--reasoning-effort low|medium|xhigh` selecting
the level.

## Environment

| var | meaning |
|---|---|
| `Q38_MODEL` | model directory |
| `Q38_ANE_ENGINE` | ANE driver checkout; defaults to this repository |
| `ANE_MAX_PROGRAMS` | program budget (default 127) |
| `ANE_ORDER=gdn_first` | bake small blocks before the large fused layers |
| `Q38_ANE_KEEP_WIRED` | `kANEFKeepModelMemoryWiredKey` (measured: no effect) |
