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
tools/ane pure-infer --bits 16 --tokens 4 --verify-reference
tools/ane pure-infer --bits 4 --tokens 32 --mtp-draft 2
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

`ane_program_limit.py` is the one to re-run on new hardware: the 127-program
ceiling is an M5 Max measurement, not a documented constant.

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

## Environment

| var | meaning |
|---|---|
| `Q38_MODEL` | model directory |
| `Q38_ANE_ENGINE` | ANE driver checkout; defaults to this repository |
| `ANE_MAX_PROGRAMS` | program budget (default 127) |
| `ANE_ORDER=gdn_first` | bake small blocks before the large fused layers |
| `Q38_ANE_KEEP_WIRED` | `kANEFKeepModelMemoryWiredKey` (measured: no effect) |
