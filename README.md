# qwen-ane

Local Qwen3.8 inference on Apple Silicon. One CLI for an Apple `fm`-style
terminal chat and an OpenAI-compatible server that coding agents can point at.

Two models, two silicon paths:

| model | flag | where it runs | context | rough footprint |
|---|---|---|---|---|
| **Qwen3.8-Flash-Next** | `-model flash-next` | ANE for GDN / QSA, MLX GPU for MoE | up to 128k | ~23 GB |
| **Qwen3.8-27B** | `-model 27b` | Apple Neural Engine only (GPU idle) | 4k tile | ~13 GB ANE-resident |

Flash-Next is the default. 27B is the low-power path: measured around **4 tok/s
decode at ~6 W** on an M5 Max, with the GPU free for a display or another job.
That is not the same engine as native C++ `rindi`, whose **12 tok/s** decode
is Metal, not ANE.

Apple Silicon only. Measured on **M5 Max / macOS 27**. Other M-series parts
will compile, but program limits and speed will differ.

## Install

```bash
git clone https://github.com/True2456/qwen-ane.git
cd qwen-ane
pip install -e .
```

Flash-Next also needs MLX:

```bash
pip install -e ".[mlx]"
```

Or run the launcher with no install:

```bash
./bin/qwen-ane --help
```

Python 3.10+, numpy, `huggingface_hub`, `safetensors`, and `tokenizers`.
27B talks to `AppleNeuralEngine.framework` through the driver vendored at
`runtime/q38_ane_engine.py`.

## Quick start

```bash
# Interactive chat (starts a local server if one is not already up)
qwen-ane chat

# Flash-Next, 128k context, thinking off
qwen-ane chat -model flash-next -ctx 128k

# Pure-ANE 27B
qwen-ane chat -model 27b -ctx 4096

# OpenAI-compatible API for agents
qwen-ane serve
qwen-ane serve -model 27b -port 2457 -ctx 4096
```

First launch uses a checkpoint already on the machine (`~/models/…`,
`~/.lmstudio/models/Qwen/…`, or `--model-path`). Hugging Face is only a
fallback if nothing local is found.

```text
  Qwen ANE CLI
  Apple Silicon Neural Engine Inference

  Model:    Qwen3.8-Flash-Next
  Context:  128k
  Endpoint: http://127.0.0.1:2457/v1
  Cache:    Enabled (LRU prefix reuse)
  Thinking: off

  Commands: /exit (quit), /clear (new chat), /think [level], /help

you> Name 3 fruits.
qwen> 1. Apple
2. Banana
3. Orange
```

Slash commands inside chat: `/exit`, `/quit`, `/clear`, `/think off|low|medium|xhigh`, `/help`.
Resume a session with `qwen-ane chat --resume <session-id>`.

## Models and weights

If the weights are already on disk, you do not need to download anything.
`qwen-ane models` shows what it found. Common locations are picked up
automatically:

- Flash-Next: `~/models/Qwen3.8-Flash-Next`, `~/models/Qwen3.8-Flash-Next-MLX-4bit`
- 27B: `~/.lmstudio/models/Qwen/Qwen3.8-27B`, `~/models/Qwen3.8-27B`

Or pass the directory:

```bash
qwen-ane chat -model flash-next --model-path ~/models/Qwen3.8-Flash-Next
qwen-ane chat -model 27b --model-path ~/.lmstudio/models/Qwen/Qwen3.8-27B
```

`qwen-ane pull` is only for a machine that does not already have the
checkpoint. Default Hub repos are
[True2456/Qwen3.8-Flash-Next-ANE](https://huggingface.co/True2456/Qwen3.8-Flash-Next-ANE)
and [True2456/Qwen3.8-27B-ANE](https://huggingface.co/True2456/Qwen3.8-27B-ANE).

`qwen-ane build <model> --source /path/to/bf16` links a local checkpoint into
`~/.qwenANE/models/`. 27B quantizes INT4 into `~/Library/Caches/q38-pure-ane`
on the first serve; later launches reuse the bake.

Discovery order: `~/.qwenANE/models/<name>`, then `~/models/…` and
`~/.lmstudio/models/Qwen/…`. Hub download is last, and only if those miss.

## Talking to the server

Default listener is `http://127.0.0.1:2457/v1`. The API key is not checked.
Requests are serialized: GDN / KV state is mutable, so one in-flight
completion at a time. A second `qwen-ane serve` on the same port refuses to
start a duplicate and tells you to `qwen-ane chat --port 2457` instead.

```bash
curl http://127.0.0.1:2457/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Pi:

```bash
# Flash-Next on :2457
pi -e extensions/flashnext-pi.ts --provider flashnext --model Qwen3.8-Flash-Next

# 27B pure ANE (server on :1240 if you launched tools/ane pure-serve;
# qwen-ane serve -model 27b uses the port you passed, default 2457)
pi -e extensions/pure27-pi.ts --provider pure27 --model Qwen3.8-27B --thinking off
```

Prefix reuse (`-lru`, on by default) is what makes agent loops tolerable:
the client resends the whole history, the engine restores matching GDN / KV
state and only evaluates the new suffix.

## Storage and config

Everything user-local lives in **`~/.qwenANE/`** (override with `QWEN_ANE_HOME`):

```text
~/.qwenANE/
├── config.json     persistent defaults
├── models/         pulled or linked checkpoints
│   ├── flash-next/
│   └── 27b/
└── sessions/       resumable chats
```

```bash
qwen-ane config show
qwen-ane config set default_model 27b
qwen-ane config set default_port 2457
```

| flag | default | meaning |
|---|---|---|
| `-m`, `--model` | `flash-next` | `flash-next` or `27b` |
| `-c`, `--ctx` | `128k` | `128k`, `64k`, `32k`, `8192`, `4096` (27B clamps to 4096) |
| `-p`, `--port` | `2457` | server or chat client port |
| `-lru` / `--no-lru` | on | prefix cache |
| `--thinking` | `off` | `off`, `low`, `medium`, `xhigh` |
| `--host` | `127.0.0.1` | bind / connect address |
| `--model-path` | auto | explicit checkpoint directory |
| `--hf-repo` | auto | override Hub repo on pull |
| `-r`, `--resume` | — | chat session id |
| `--spec` | `4` | Flash-Next speculative lookahead (`serve` only) |

Single-dash long flags work (`-model`, `-ctx`, `-port`, `-lru`) so the CLI
feels like `/usr/bin/fm`.

## What actually runs where

**Flash-Next** is a hybrid decode: linear recurrence and QSA on the Neural
Engine, routed MoE experts on MLX. Context is 128k. This is the daily-driver
chat / agent model.

**27B pure ANE** submits MIL programs straight to
`AppleNeuralEngine.framework`. No MLX, no PyTorch, no Metal GEMMs on the
decode path. INT4 weights, greedy or sampled decode. MTP (`--mtp-draft`)
exists in `tools/pure_ane_server.py` but `qwen-ane` leaves it off: warming
the extra layer over the prompt lost wall-clock time on the agent loop we
care about.

A third engine, native C++ `rindi`, lives in this repo for research. Its
headline **95 tok/s prefill / 12 tok/s decode** is width-128 ANE prefill
plus **lane-1 Metal decode**, not ANE decode. See
[docs/QWEN-PREFILL-FAST.md](docs/QWEN-PREFILL-FAST.md). Use `qwen-ane` unless
you are working on that path.

## Requirements and caveats

- Apple Silicon. One heavy ANE process at a time; a second model on the Neural
  Engine will fail compiles that otherwise succeed.
- 27B first load bakes INT4 (~9 GB cache) and compiles ~70 ANE programs.
  Later starts reuse `~/Library/Caches/q38-pure-ane` and the private compiler
  artifact cache.
- Do not load Flash-Next (~78 GB) and 27B in the same process. Separate ports,
  separate processes.
- Private ANE APIs. This is research software; expect breakage across macOS
  builds.

Full CLI notes: [docs/QWEN-ANE.md](docs/QWEN-ANE.md).
Setup / moving machines: [docs/SETUP.md](docs/SETUP.md).
Measured ANE vs GPU energy: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
ANE op rules: [docs/ANE-REFERENCE.md](docs/ANE-REFERENCE.md).

Apache-2.0.
