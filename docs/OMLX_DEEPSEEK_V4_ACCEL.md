# oMLX DeepSeek V4 acceleration trial

## Checkpoint

The local checkpoint used for these probes is:

`DeepSeek-V4-Flash-0731-AWQ-mtp2bit`

Its routed target experts use affine Q2/group-128 gate and up projections,
and affine Q3/group-64 down projections.  Its embedded DSpark drafter has
three decoder stages and a declared block size of five.

## Enabled trial

The oMLX model setting and active profile now set:

```json
"mtp_num_draft_tokens": 5
```

This is a maximum, not a fixed draft depth.  oMLX's `_DepthController`
measures depths 1 through 5 (and its depth-0 baseline), then selects the best
depth from observed acceptance and wall time.  The checkpoint had previously
used oMLX's default maximum of three despite advertising a DSpark block size
of five.

The setting takes effect on the next load of this model.  A server restart is
not required while the model is unloaded.  To revert, remove the field from
both `~/.omlx/model_settings.json` and the active `profile-16-copy` entry in
`~/.omlx/model_profiles.json`, or set it back to `3` in the UI.

Promotion criterion: compare the same oMLX benchmark corpus, prompt lengths,
generation length, sampling settings, and thermal state at maximum depths 3
and 5.  Keep depth 5 only if median generation tok/s improves without a
regression in generated-token identity for greedy tests.  The MTP completion
log should also show whether depths 4 and 5 are actually selected often enough
to matter.

## Native kernel results

The probes preserve the checkpoint's projection dimensions, quantization
metadata dtype, route count, and packed layouts while using synthetic weight
values so they can run without loading the 105 GB model.

On this M5 Max with the bundled MLX 0.32.0:

| Candidate | Shape | Result versus stock |
|---|---:|---:|
| Sort six decode routes | Q2/gs128 gate/up | `0.76-0.99x` |
| Existing DeepSeek affine blocks | Q2/gs128 gate/up | unsupported |
| Existing DeepSeek affine blocks | Q3/gs64 down | `0.40-0.58x` |
| Bonsai single-expert Q2 | 4096 -> 2048 | `0.89x` |
| Gather six experts then Bonsai | Q2/gs128 gate/up | `0.62x`, unsupported batched numerics |
| Fused routed Q2 gate+up+SwiGLU | 1-5 verify rows | `0.96-1.00x` |

The fused prototype matches stock closely (BF16 accumulation-order
differences only), but it does not clear the performance promotion threshold.
It remains a probe and is not patched into the installed oMLX application.

Run the probes with oMLX's bundled Python environment:

```sh
PYTHONHOME=/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11 \
PYTHONPATH=/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages \
/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 \
probes/bench_omlx_deepseek_affine.py --experts 64 --routes 6

PYTHONHOME=/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11 \
PYTHONPATH=/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages \
/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 \
probes/bench_omlx_deepseek_q2_pair.py --experts 64 --batch 4 --routes 6
```

## ANE and SME2 decision

ANE is not the first target for this checkpoint.  The DSpark head is already
small relative to the target backbone, and moving it to ANE would add a
synchronization boundary plus a second representation of weights while the
model is already close to the Metal wired-memory limit.

SME2 remains architecturally interesting for whole routed experts, not for a
split inside one projection.  A useful implementation must let the GPU and
SME2 consume different selected experts concurrently and merge once per MoE
layer.  Calling CPU work between MLX projections would recreate the command-
buffer boundary that already made the dense GPU/SME2 split slower in this
repository.  The current oMLX Python/MLX switch layer exposes no zero-copy,
asynchronous CPU primitive for that schedule, so no SME2 path is enabled until
that integration can be benchmarked end to end.
