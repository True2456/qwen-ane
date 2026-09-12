#!/usr/bin/env python3
"""Reference perplexity for Flash-Next through mlx_lm, for the ANE arm to beat.

The ANE path scores the same file with `generate --ppl-file`. This one runs the
unmodified 4-bit MLX model so the two numbers differ only by what the port
changed: int8 activations and weights on the ANE, a hand-written MIL graph per
layer, and fp16 mixers.

    PYTHONPATH=/Users/true/.mlx128/mlx-lm:/Users/true/.mlx128/mlx/python \
    ~/.rindi/venvs/coreai/bin/python probes/flashnext_ppl_mlx.py FILE [N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

if not hasattr(mx, "unique"):
    # The indexer's block selection past the 2048-token budget calls
    # mx.unique, which this MLX build does not have. Sorted unique values is
    # the whole contract, and the sync it costs only affects this arm.
    import numpy as _np

    def _unique(a, *_a, **_k):
        return mx.array(_np.unique(_np.array(a)))

    mx.unique = _unique

MODEL = "/Users/true/models/Qwen3.8-Flash-Next-MLX-4bit"
# Feed the file in chunks through one prompt cache, so context accumulates the
# way it does on the ANE arm. Scoring independent windows would measure a
# different thing and would flatter neither arm honestly.
WINDOW = 256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("tokens", type=int, nargs="?", default=2048)
    parser.add_argument("--prefill", type=int, default=1,
                        help="leading prompt length; score the following tokens")
    args = parser.parse_args()
    path, want = args.file, args.tokens
    prefill = max(1, args.prefill)
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL)
    print(f"loaded in {time.perf_counter() - t0:.1f}s  "
          f"active {mx.get_active_memory() / 1e9:.1f} GB", flush=True)

    ids = tokenizer.encode(path.read_text(), add_special_tokens=False)[:want + prefill]
    print(f"scoring {len(ids) - prefill} tokens of {path} after {prefill} prompt tokens", flush=True)

    total = 0.0
    n = 0
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    # Cache all but the last prompt token; its logits score the first target.
    for lo in range(0, prefill - 1, WINDOW):
        warm = model(mx.array([ids[lo:min(lo + WINDOW, prefill - 1)]]), cache=cache)
        mx.eval(warm)
        del warm
    for lo in range(prefill - 1, len(ids) - 1, WINDOW):
        chunk = ids[lo:lo + WINDOW + 1]
        if len(chunk) < 2:
            break
        x = mx.array([chunk[:-1]])
        tgt = mx.array([chunk[1:]])
        lg = model(x, cache=cache).astype(mx.float32)
        lse = mx.logsumexp(lg, axis=-1)
        pick = mx.take_along_axis(lg, tgt[..., None], axis=-1).squeeze(-1)
        nll = (lse - pick).sum()
        mx.eval(nll)
        total += float(nll)
        n += len(chunk) - 1
        print(f"  {n} tokens  ppl {mx.exp(mx.array(total / n)).item():.4f}",
              flush=True)
    el = time.perf_counter() - t0
    print(f"reference ppl over {n} tokens: nll {total / n:.6f}  "
          f"ppl {mx.exp(mx.array(total / n)).item():.4f}  in {el:.1f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
