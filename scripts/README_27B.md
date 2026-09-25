---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: qwen-ane
pipeline_tag: text-generation
tags:
- apple-silicon
- apple-neural-engine
- ane
- qwen
- qwen3.8
- qwen-ane
---

# Qwen3.8-27B-ANE

INT4 runtime package for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on Apple Silicon, for use with [qwen-ane](https://github.com/True2456/qwen-ane).

Learned matmuls, Gated DeltaNet recurrence, attention, and MLP tails run on the Apple Neural Engine through `AppleNeuralEngine.framework`. The Metal GPU is not used. This is not the Flash-Next hybrid checkpoint (ANE attention + GPU MoE), and it is not a Transformers-loadable full-weight dump.

The Hub safetensors widget reports only the host tensors in `model.safetensors` (~2.5B F16 values). The 27B INT4 layer weights are in `quant_cache/`.

## Package layout

| Path | Size | Role |
| --- | --- | --- |
| `model.safetensors` | 5.09 GB | Host embeddings, RMSNorms, and LM head (F16) |
| `quant_cache/` | 9.98 GB | Pre-quantized INT4 blobs, per-row F16 scales, and `manifest.json` |
| `mtp.safetensors` | 0.85 GB | Optional MTP draft-layer tensors (BF16) |
| tokenizer / config | — | `config.json`, `tokenizer.json`, `chat_template.jinja`, and related files |

Total package is about **16 GB**. First load uses the packaged INT4 cache; it does not re-quantize from BF16.

## Requirements

- Apple Silicon (developed and measured on an M5 Max, macOS 27)
- [qwen-ane](https://github.com/True2456/qwen-ane)
- `AppleNeuralEngine.framework` (present on macOS; the runtime talks to it through the vendored driver)

## Install and run

```bash
git clone https://github.com/True2456/qwen-ane.git
cd qwen-ane
pip install -e .
```

Download into `~/.qwenANE/models/27b`:

```bash
qwen-ane pull 27b
```

Or:

```bash
hf download True2456/Qwen3.8-27B-ANE --local-dir ~/.qwenANE/models/27b
```

Chat and serve:

```bash
qwen-ane chat -model 27b
qwen-ane serve -model 27b -port 2457
```

Optional speculative decode (`--mtp-draft 2` uses `mtp.safetensors`):

```bash
qwen-ane serve -model 27b -port 2457 --mtp-draft 2
```

Coding agents can point an OpenAI-compatible client at that server. A Pi provider lives in the qwen-ane tree as `extensions/pure27-pi.ts`.

## Measured performance (M5 Max)

Figures below are from `qwen-ane` on an Apple M5 Max running 100% on-chip Apple Neural Engine inference (0% Metal GPU).

### Context scale & power metrics (2026-09-25)

Cold prefix (`reset` between lengths, salted prompts, **reused=0**). Pure ANE server context 8576, INT4 weights. Watts are **mean** `powermetrics --samplers cpu_power,gpu_power,ane_power -i 500` over the generate. PeakMem is `footprint -p` `phys_footprint_peak`. Prompt tokens were 1000 / 1999 / 3997 / 7993; completion 128.

| Test | TTFT(ms) | TPOT(ms) | ppTPS | tgTPS | E2E(s) | Throughput | PeakMem | ANE / GPU / CPU W |
|---|---:|---:|---:|---:|---:|---:|---|---|
| pp 1024 / tg 128 | 31856.7 | 251.4 | 31.4 | 4.0 | 63.8 | 17.7 | 21.0 GB | 2.86 / 0.24 / 7.04 |
| pp 2048 / tg 128 | 69915.0 | 260.6 | 28.6 | 3.8 | 103.0 | 20.6 | 21.0 GB | 2.67 / 0.11 / 5.91 |
| pp 4096 / tg 128 | 165730.8 | 278.2 | 24.1 | 3.6 | 201.1 | 20.5 | 21.0 GB | 2.44 / 0.17 / 6.27 |
| pp 8192 / tg 128 | 439329.1 | 319.8 | 18.2 | 3.1 | 480.0 | 16.9 | 21.0 GB | 2.23 / 0.25 / 7.34 |

- **Zero GPU utilization**: Metal GPU stays idle at 0.1–0.2 W while the 27B model runs entirely on the ANE.
- **Ultra-low power draw**: Sustained ANE power draw is only **~2.2–2.9 W**; total package power stays under **10 W**.
- **Rock-solid memory footprint**: Physical memory footprint stays flat at **21.0 GB** from 1k through 8k+ tokens.

### Microbenchmarks & Agent turns

| Workload | Result |
| --- | --- |
| Greedy decode, short / agent turns | about 3.4–4.3 tok/s |
| MTP-2, repetitive capitals probe | 5.58 tok/s vs 4.34 greedy; unique prose often falls back toward greedy |
| Prefill, 300-token agent prompt | 5.27 s TTFT (~57 tok/s), width-64 / 32-lane programs |
| Prefill, 4080-token cold prompt | 26.71 tok/s at the earlier 16-lane chunked-GDN config |
| Decode power (documented ANE-only path) | about 6 W; GPU idle |

Prefix cache is on for the HTTP server: a follow-up agent turn that reuses 300 prompt tokens prefills the 56-token suffix in about 1.2 s on the same machine.

Decode is slower than GPU INT4 (MLX) because each token walks 64 layers as many small ANE programs rather than one large GPU GEMM. INT4 shrinks the package and the DRAM traffic; the ANE datapath still executes in F16.

## Architecture

64 layers in a 3:1 pattern: 48 Gated DeltaNet (linear attention) layers and 16 full-attention layers (every fourth layer). Projections, convolutions, and SwiGLU MLPs are compiled MIL graphs on the Neural Engine. Host NumPy packs IO, samples tokens, and holds GDN / KV state.

## License

Apache-2.0. Base weights: Qwen Team. INT4 packaging and ANE runtime: [True2456/qwen-ane](https://github.com/True2456/qwen-ane).
