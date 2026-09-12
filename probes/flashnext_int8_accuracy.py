"""int8 per-channel accuracy on the real Flash-Next projections.

Go/no-go for the MIL int8 port. The 1.74x speedup is measured; what has not
been checked is whether per-channel int8 holds up on the actual weight
distribution of in_proj / out_proj, against an fp32 reference, with realistic
activations. fp16 is the control — it is what ships today, so int8 only has to
match it, not beat fp32.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import runtime.q38_ane_engine as E  # noqa: E402
from export_flashnext_coreai import _load_layer  # noqa: E402


def _rel(got, ref):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12))


def main() -> None:
    rng = np.random.default_rng(11)
    for layer in (0, 12, 24):
        loader, w = _load_layer(layer)
        tensors = {
            "in_proj_qkv": np.concatenate([
                np.asarray(w["linear_attn.in_proj_qkv.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_z.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_b.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_a.weight"], np.float32),
            ], axis=0),
            "out_proj": np.asarray(w["linear_attn.out_proj.weight"], np.float32),
        }
        for name, mat in tensors.items():
            o, i = mat.shape
            # realistic activations: unit-ish with a heavy tail, as the
            # residual stream has after the hyper mixer
            x = (rng.standard_normal((i, 32)) * 0.6).astype(np.float32)
            x[rng.integers(0, i, i // 64), :] *= 6.0
            ref = mat @ x

            f16 = (mat.astype(np.float16).astype(np.float32)
                   @ x.astype(np.float16).astype(np.float32))

            q8, s8 = E.quantize_linear_int8(mat)
            deq8 = q8.astype(np.float32) * np.asarray(s8, np.float32).reshape(-1, 1)
            r8 = deq8 @ x

            q4, s4 = E.quantize_linear_int4(mat)
            lo = (q4 & 0x0F).astype(np.int8)
            hi = ((q4 >> 4) & 0x0F).astype(np.int8)
            lo = np.where(lo > 7, lo - 16, lo)
            hi = np.where(hi > 7, hi - 16, hi)
            unpacked = np.empty((o, i), np.float32)
            unpacked[:, 0::2] = lo
            unpacked[:, 1::2] = hi
            deq4 = unpacked * np.asarray(s4, np.float32).reshape(-1, 1)
            r4 = deq4 @ x

            print(f"  L{layer:<2} {name:12s} {str(mat.shape):14s} "
                  f"fp16 {_rel(f16, ref):.5f}   int8 {_rel(r8, ref):.5f}   "
                  f"int4 {_rel(r4, ref):.5f}", flush=True)
        loader.close()


if __name__ == "__main__":
    main()
