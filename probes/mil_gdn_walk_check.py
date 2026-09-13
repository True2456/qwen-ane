"""Compare wide prefill with production K=4 recurrence over identical inputs.

Unlike mil_k_check's one-block Torch comparison, this starts with zero state,
walks multiple blocks, and uses prefix-state decode graphs for the narrow arm.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "probes"):
    sys.path.insert(0, str(path))
from export_flashnext_coreai import _load_layer, H, HC_W, SEQ_DEFAULT
from flashnext_multitoken_step import MultiTokenStep
from runtime.mil_gdn_backend import MilGdnLayer


def rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()
    if args.tokens % 32:
        ap.error("--tokens must be a multiple of 32")
    loader, w = _load_layer(args.layer)
    try:
        lay = MilGdnLayer(args.layer, w, MultiTokenStep(w, 1).eval().half(),
                          k=4, prefill_k=32, single_state=False)
    finally:
        loader.close()
    x = np.random.default_rng(12).normal(0, .05, (HC_W, args.tokens)).astype(np.float16)

    def walk(proc, width):
        lay.reset()
        lay.select(proc)
        out, states, convs = [], [], []
        for off in range(0, args.tokens, width):
            xb = np.zeros((HC_W, SEQ_DEFAULT), np.float16)
            xb[:, :width] = x[:, off:off+width]
            out.append(lay(xb, n=width)[0].reshape(H, width))
            lay.commit(width-1)
            if (off+width) % 32 == 0:
                states.append(lay.current_state().copy())
                convs.append(lay._conv.copy())
        return np.concatenate(out, axis=1), states, convs

    narrow, ns, nc = walk(0, 4)
    wide, ws, wc = walk(1, 32)
    for i, (s, ref, cv, rc) in enumerate(zip(ws, ns, wc, nc)):
        part = slice(i*32, (i+1)*32)
        errors = {"mixed": rel(wide[:,part], narrow[:,part]),
                  "state": rel(s,ref), "conv": rel(cv,rc)}
        print(f"L{args.layer} through {(i+1)*32} tokens: " +
              " ".join(f"{name}={value:.6f}" for name,value in errors.items()),
              flush=True)
        assert all(np.isfinite(value) and value < .015
                   for value in errors.values()), errors


if __name__ == "__main__":
    main()
