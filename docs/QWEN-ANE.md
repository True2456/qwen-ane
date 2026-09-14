# qwen-ane CLI reference

`qwen-ane` is the user-facing CLI for this repository. It runs Qwen3.8-Flash-Next
(hybrid ANE + MLX) and Qwen3.8-27B (pure ANE) from one command, with an Apple
`fm`-style chat and an OpenAI-compatible server.

See the [README](../README.md) for install, what each model is, and what
silicon each path actually uses.

## Commands

```bash
qwen-ane chat      # interactive session (default if you pass only flags)
qwen-ane serve     # OpenAI-compatible HTTP server
qwen-ane models    # local / Hub status
qwen-ane pull      # download into ~/.qwenANE/models/
qwen-ane build     # link a local BF16/MLX checkpoint into ~/.qwenANE/models/
qwen-ane config    # show / get / set ~/.qwenANE/config.json
```

`qwen-ane` with no subcommand is `chat`.

## Chat

```bash
qwen-ane chat
qwen-ane chat -model flash-next -ctx 128k -port 2457 -lru
qwen-ane chat -model 27b -ctx 4096
qwen-ane chat -thinking medium
qwen-ane chat --resume 20260915-080117-f829bf
qwen-ane chat -system "Be terse."
```

If nothing is listening on the port, chat starts the matching server in the
background and waits until `/v1/models` answers. If a server is already up,
chat attaches to it.

In-session:

| command | effect |
|---|---|
| `/exit`, `/quit` | leave; session is saved |
| `/clear` | drop history |
| `/think off\|low\|medium\|xhigh` | reasoning effort |
| `/help` | this list |

Thinking (`low` / `medium` / `xhigh`) follows the checkpoint chat template,
not a generic OpenAI scale. `high` is not valid; the CLI maps it to `xhigh`
on the server.

## Serve

```bash
qwen-ane serve --port 2457 --ctx 128k
qwen-ane serve --model 27b --port 2457
```

Endpoints: `GET /v1/models`, `POST /v1/chat/completions` (streaming SSE,
tools, `reasoning_content`). A second serve on the same port exits 0 after
printing the already-running model list.

27B context is clamped to 4096 (hardware tile). Flash-Next `--spec` is
speculative lookahead (default 4).

## Models, pull, build

```bash
qwen-ane models
qwen-ane pull flash-next
qwen-ane pull 27b --hf-repo True2456/Qwen3.8-27B-ANE
qwen-ane build flash-next --source ~/models/Qwen3.8-Flash-Next
qwen-ane build 27b --source ~/.lmstudio/models/Qwen/Qwen3.8-27B
```

Hub defaults live in `~/.qwenANE/config.json` under `hf_repos`.

`build` does not re-quantize. It links (or copies) the source directory into
`~/.qwenANE/models/<name>/`. 27B INT4 bake happens on first `serve` /
`chat` into `~/Library/Caches/q38-pure-ane`.

## Storage

Canonical directory: **`~/.qwenANE/`**. Override with `QWEN_ANE_HOME`.
A leftover non-empty `.qwenANE` in the git checkout is still honoured so an
old install keeps working.

```text
~/.qwenANE/
├── config.json
├── models/
│   ├── flash-next/
│   └── 27b/
└── sessions/
```

```bash
qwen-ane config show
qwen-ane config set default_model 27b
qwen-ane config set default_port 2457
```

## Flags

| Flag | Aliases | Default | Description |
|---|---|---|---|
| `-m`, `--model` | `-model` | `flash-next` | `flash-next` or `27b` |
| `-c`, `--ctx` | `-ctx` | `128k` | `128k`, `64k`, `32k`, `8192`, `4096` |
| `-p`, `--port` | `-port` | `2457` | server or client port |
| `-lru`, `--lru` | | on | prefix cache |
| `--no-lru` | `--no-cache` | | disable prefix cache |
| `--host` | `-host` | `127.0.0.1` | bind / connect host |
| `--thinking` | `-thinking` | `off` | `off`, `low`, `medium`, `xhigh` |
| `-i`, `--instructions` | `-system` | none | system prompt |
| `-r`, `--resume` | `-resume` | none | session id |
| `--model-path` | `-model-path` | auto | checkpoint directory |
| `--hf-repo` | `-hf-repo` | auto | Hub repo for pull |
| `--spec` | `-spec` | `4` | Flash-Next draft depth (`serve`) |
| `--max-tokens` | `-max-new` | `2048` | generation cap |
