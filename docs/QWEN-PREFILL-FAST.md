# Qwen native prefill fast path

The prefill-only fast path is the default for the native Qwen3.8-27B engine.
It does not change lane-1 decode. Set `RINDI_DISABLE_QWEN_PREFILL_FAST=1` for
an immediate scalar rollback. The path combines five accelerated stages:

- a Metal four-tap causal GDN convolution, replacing the macOS 27 scalar CPU
  fallback;
- GPU Q/K normalization and decay/beta preparation;
- a fixed-order, column-parallel Metal GDN recurrence;
- fused per-head GDN RMSNorm and SiLU gating;
- 32-lane Metal Q/K/V projection GEMMs, including automatic split/stitch for
  width-128 prompts (lane-1 decode retains the exact GEMV path).

The Metal convolution stores its three causal history values separately, so
all 128 lanes of a width-128 CoreAI tail are live instead of only 125.

Recommended throughput benchmark:

```sh
RINDI_DISABLE_MTP=1 \
RINDI_TAIL_COREAI=1 \
RINDI_ENABLE_METAL_TAIL=1 \
RINDI_PREFILL_BATCH_ATTENTION=1 \
RINDI_QWEN_PREFILL_FAST=1 \
RINDI_ANE_WIDTH=128 \
/tmp/rindi-bench-native-mtp \
  "$HOME/.lmstudio/models/Qwen/Qwen3.8-27B.rindi" 1024 128
```

For the rail-power A/B script, preserve the environment through `sudo`:

```sh
WIDTHS="128 32" COOLDOWN_SECONDS=20 \
RINDI_QWEN_PREFILL_FAST=1 sudo -E tools/native_power_ab.sh
```

On the development M5 Max running macOS 27.0 beta build 26A5421a, two cooled
1,024-prompt/128-generation runs measured 94.24 and 94.62 prompt tok/s with the
same `7c8bafad161ce549` generated-output hash. The earlier simdgroup recurrence
reached 98.47 tok/s but produced run-dependent output on this driver and is no
longer the default. It now requires both `RINDI_GDN_PARALLEL_K=1` and
`RINDI_GDN_SIMD_REDUCTION=1`.

Four independent 255-token WikiText-2 validation slices measured aggregate
PPL 15.1084 on the scalar path and 14.6641 on the deterministic fast path, a
2.94% improvement; every slice improved individually. Repeating a fast slice
produced an identical NLL and PPL. This qualifies the deterministic path as
the Qwen default.

The final reverse-order rail-power run measured width 32 at 82.86 prompt tok/s,
11.68 active W, and 7.096 prompt tok/J; width 128 measured 94.95 prompt tok/s,
10.89 active W, and 8.718 prompt tok/J. Thus width 128 was 1.146x faster while
using 0.933x the active prefill power, for 1.229x the energy efficiency. Both
configurations produced the same `7c8bafad161ce549` output hash. Decode uses
the same lane-1 path in both configurations; its A/B difference is thermal and
run-order noise rather than a width-dependent code path.

Power measurements still require root. Run the reverse-order validation to
control for thermal and idle-baseline order:

```sh
WIDTHS="32 128" COOLDOWN_SECONDS=20 \
sudo -E tools/native_power_ab.sh
```

The repository also contains `scripts/export_gdn_prefill_recurrence.py`, an
official CoreAI export of the shared weight-free recurrence. On the current OS,
the 16/32-token graphs exceed the accepted ANE region and the accepted
8-token graph measures about 9.4 ms per layer evaluation. That is slower than
the optimized Metal recurrence, so it remains an experiment rather than the
native scheduler default.

## Optional INT4 target LM head

Set `RINDI_INT4_LM_HEAD=1` to quantize the untied 248,320-by-5,120 BF16 output
projection to affine groupwise INT4 (group size 64, BF16 scale/bias, FP32
accumulation, FP16 logits). This applies to greedy decode, temperature
sampling, speculative target verification, and native perplexity scoring. It
does not alter the default BF16-head path when the flag is absent.

With the flag enabled, the dense engine does not also allocate the BF16 Metal
head: the head allocation falls from about 2.37 GiB to about 698 MiB. The
source BF16 tensor remains memory-mapped while the process is alive, but is not
copied into a second Metal buffer. No quantized cache file is written to NAND.

The first 1,024-prompt/128-generation M5 Max run measured 11.877 decode tok/s
and 94.831 prompt tok/s. The established BF16 control range was 9.67--10.51
decode tok/s; use 10.42 as the conservative comparison (+14.0%). A same-session
BF16 run fell to 7.28 tok/s because of thermal/run-order variance and is not
used as the headline comparison. Across four independent 255-token WikiText-2
slices, aggregate PPL changed from 14.6641 (BF16) to 14.7480 (INT4), a 0.572%
increase.

```sh
RINDI_INT4_LM_HEAD=1 \
RINDI_DISABLE_MTP=1 \
RINDI_QWEN_PREFILL_FAST=1 \
RINDI_ANE_WIDTH=128 \
/tmp/rindi-bench-native-mtp \
  "$HOME/.lmstudio/models/Qwen/Qwen3.8-27B.rindi" 1024 128
```

For an idle-subtracted rail measurement of the INT4 decode path:

```sh
WIDTHS="128" COOLDOWN_SECONDS=20 \
RINDI_INT4_LM_HEAD=1 \
sudo -E tools/native_power_ab.sh
```
