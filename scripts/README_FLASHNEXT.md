---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - apple-neural-engine
  - apple-silicon
  - mlx
  - qwen
  - qwen3.8
  - qwen4_exp
  - mixture-of-experts
  - qwen-ane
model-index:
  - name: Qwen3.8-Flash-Next-ANE
    results:
      - task:
          type: question-answering
          name: MMLU
        dataset:
          type: cais/mmlu
          name: MMLU (600 questions, 5 subjects)
          split: test
        metrics:
          - type: accuracy
            value: 87.33
            name: 5-shot likelihood
      - task:
          type: text-generation
          name: GSM8K
        dataset:
          type: openai/gsm8k
          name: GSM8K (200 test questions)
          split: test
        metrics:
          - type: accuracy
            value: 96.5
            name: 8-shot greedy
---

# Qwen3.8-Flash-Next-ANE

Hybrid inference package for [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) on Apple Silicon, designed for use with [qwen-ane](https://github.com/True2456/qwen-ane).

## What it is

Flash-Next is a hybrid silicon deployment:
- **Attention on Apple Neural Engine (ANE)**: 36 Gated DeltaNet (linear recurrence) layers and 12 Qwen Sparse Attention layers run on the Neural Engine via precompiled MIL graphs, drawing approximately 1.1 W.
- **MoE Experts on Metal GPU**: The 512-expert routed MoE bank (68 GB) and output projections remain resident in Unified Memory and execute via MLX, drawing approximately 9 W.
- **Speculative Drafting**: 4-slot MTP speculative drafting and PLE n-gram tables deliver high decode throughput.

For a 100% pure on-chip ANE model (0% GPU), see [True2456/Qwen3.8-27B-ANE](https://huggingface.co/True2456/Qwen3.8-27B-ANE).

## Package layout

| Path | Size | Role |
| --- | --- | --- |
| `ane-h17/` | ~5.6 GB | Precompiled ANE MIL graphs and weight blobs for 36 GDN + 12 QSA layers |
| `model.safetensors` | 76.7 GB | 4-bit language weights (experts 4-bit gs64; embeddings/mixers 8-bit) |
| `ngram/` | ~95 MB | BF16 PLE n-gram table shards |
| `mtp/` | ~1.4 GB | 4-bit MTP drafter weights |
| `model-indexer.safetensors` | 42.6 MB | QSA indexer weights |
| tokenizer / config | — | `config.json`, `tokenizer.json`, `chat_template.jinja`, and related metadata |

## Requirements

- Apple Silicon Mac with 96 GB or 128 GB Unified Memory (tested on M5 Max, macOS 27)
- Python 3.10+
- [qwen-ane](https://github.com/True2456/qwen-ane)

## Install and run

### 1. Install qwen-ane

```bash
git clone https://github.com/True2456/qwen-ane.git
cd qwen-ane
pip install -e ".[mlx]"
```

### 2. Download weights

Download automatically into `~/.qwenANE/models/flash-next`:

```bash
qwen-ane pull flash-next
```

Or using the Hugging Face CLI:

```bash
hf download True2456/Qwen3.8-Flash-Next-ANE --local-dir ~/.qwenANE/models/flash-next
```

### 3. Interactive chat

```bash
qwen-ane chat -model flash-next -ctx 32768
```

### 4. OpenAI-compatible API server

```bash
qwen-ane serve -model flash-next -port 2457 -ctx 32768
```

Point any OpenAI-compatible client, web UI, or coding agent to `http://127.0.0.1:2457/v1`.

For Pi the coding agent:

```bash
pi -e extensions/flashnext-pi.ts --provider flashnext --model Qwen3.8-Flash-Next
```

## Measured performance (Apple M5 Max)

Cold prefix (`reused=0`), serve context 33,792, `FLASHNEXT_SPEC=4`, `FLASHNEXT_PREFILL_MIL_K=32`, `FLASHNEXT_MOE=mlxresident`, `FLASHNEXT_HEAD=mlx`. Power measured via `powermetrics`:

| Prompt Tokens | Completion | TTFT | Prefill Speed | Decode Speed | Active Package Power |
| --- | --- | ---: | ---: | ---: | --- |
| 4,096 | 128 | 49.3 s | 80.6 tok/s | 15.1 tok/s | ~17.9 W (1.1 W ANE + 9.0 W GPU + 7.8 W CPU) |
| 8,192 | 128 | 100.5 s | 79.2 tok/s | 14.8 tok/s | ~17.9 W (1.1 W ANE + 9.1 W GPU + 7.8 W CPU) |
| 16,384 | 128 | 201.7 s | 79.1 tok/s | 15.7 tok/s | ~17.7 W (1.1 W ANE + 9.5 W GPU + 7.1 W CPU) |
| 32,768 | 128 | 399.2 s | 79.9 tok/s | 15.8 tok/s | ~19.4 W (1.1 W ANE + 10.5 W GPU + 7.8 W CPU) |

- **Prefill**: Sustained ~80 tok/s linearly up to 32k context.
- **Decode**: 15–16 tok/s with 4-slot MTP speculative decoding (or 4.5 tok/s single-step).
- **Prefix cache**: Multi-turn agent follow-ups with prefix hits evaluate in ~0.8 s.

## Quality verification

Measured on M5 Max against unmodified MLX 4-bit reference on identical prompts:

| Benchmark | Setting | ANE + MLX Hybrid | Reference MLX 4-bit GPU |
| --- | --- | ---: | ---: |
| MMLU | 5-shot, likelihood of A/B/C/D (600 questions) | 87.33% (524/600) | 86.33% (518/600) |
| GSM8K | 8-shot greedy, last number (200 test questions) | 96.50% (193/200) | — |

## License

Qwen Community License 1.0 (see `LICENSE`).
Inference engine: [True2456/qwen-ane](https://github.com/True2456/qwen-ane).
