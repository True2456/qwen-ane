"""Does the host indexer pick the same keys as mlx_lm's QSAIndexer?

256k context needs this selector, not bigger ANE graphs: the model attends to
at most `indexer_budget` keys however long the context is. Correctness here is
selection agreement — the same token indices, in any order — against the MLX
implementation driven with the same layer weights and the same token stream.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from runtime.flashnext_indexer import QSAIndexer, IndexerState  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--prefill", type=int, default=2100)
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm.models.qwen4_exp import QSAIndexer as MlxIndexer, IndexerCache, TextConfig
    from export_flashnext_coreai import _load_layer, BASE

    cfg_raw = json.loads((BASE / "config.json").read_text())
    cfg_raw = cfg_raw.get("text_config", cfg_raw)
    loader, w = _load_layer(a.layer)

    ours = QSAIndexer(w, cfg_raw)
    cfg = TextConfig(**{k: v for k, v in cfg_raw.items()
                        if k in TextConfig.__dataclass_fields__})
    theirs = MlxIndexer(cfg)
    p = "self_attn.indexer."
    theirs.index_qk_proj.weight = mx.array(np.asarray(w[p + "index_qk_proj.weight"], np.float32))
    theirs.q_layernorm.weight = mx.array(np.asarray(w[p + "q_layernorm.weight"], np.float32).reshape(-1))
    theirs.k_layernorm.weight = mx.array(np.asarray(w[p + "k_layernorm.weight"], np.float32).reshape(-1))

    H = int(cfg_raw["hidden_size"])
    rng = np.random.default_rng(7)
    state = IndexerState()
    cache = IndexerCache()

    # mlx_lm's L>1 selection path calls mx.unique, which this MLX build does
    # not have, so it has never run. Drive both one token at a time instead.
    xs = (rng.standard_normal((a.prefill, H)) * 0.05).astype(np.float32)
    ours_sel = theirs_sel = None
    for i in range(a.prefill):
        ours_sel = ours.update_and_select(xs[i:i + 1], i, state)
        theirs_sel = theirs.update_and_select(mx.array(xs[i:i + 1]).reshape(1, 1, H),
                                              offset=i, idx_cache=cache)
    def cmp(o, t, label):
        if o is None and t is None:
            print(f"  {label}: both None (context within budget)")
            return
        if (o is None) != (t is None):
            print(f"  {label}: MISMATCH ours={'None' if o is None else len(o)} "
                  f"theirs={'None' if t is None else t.size}")
            return
        a_ = set(np.asarray(o).ravel().tolist())
        b_ = set(np.array(t).ravel().tolist())
        inter = len(a_ & b_)
        print(f"  {label}: ours {len(a_)} theirs {len(b_)}  agree {inter}  "
              f"{'EXACT' if a_ == b_ else f'jaccard {inter/len(a_|b_):.4f}'}")

    cmp(ours_sel, theirs_sel, f"after {a.prefill} single-token steps")

    off = a.prefill
    for s in range(a.steps):
        x1 = (rng.standard_normal((1, H)) * 0.05).astype(np.float32)
        o = ours.update_and_select(x1, off, state)
        t = theirs.update_and_select(mx.array(x1).reshape(1, 1, H), offset=off, idx_cache=cache)
        cmp(o, t, f"decode step {s} (offset {off})")
        off += 1
    loader.close()


if __name__ == "__main__":
    main()
