"""Real Flash-Next weights through the ANE at int8, verified end to end.

Speed and numpy-side accuracy are already measured. What is not measured is the
actual ANE output for the real in_proj / out_proj at int8 — i.e. that the
engine's per-channel path reproduces the dequantized reference on hardware, not
just in a simulation of it.

Reference is the fp32 matmul. fp16-on-ANE is the control, since that is what
ships. The bar is the shipping MLX build's own error (group-64 4-bit, ~0.10 on
these tensors), not bit-exactness with fp32.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "probes"))

from ane_w8a8_projection import eng  # noqa: E402
from runtime.q38_ane_engine import _iosurface_view  # noqa: E402
from export_flashnext_coreai import _load_layer  # noqa: E402

S = 32


def _run(prog, x, out_dim):
    eng._ensure_io(prog)
    with _iosurface_view(prog._in_surf, x.shape, np.float16) as dst:
        np.copyto(dst, x)
    eng.submit(prog, procedure_index=0)
    with _iosurface_view(prog._out_surf, (out_dim, S), np.float16) as o:
        return np.array(o, np.float32)


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main() -> None:
    rng = np.random.default_rng(5)
    for layer in (0, 24):
        loader, w = _load_layer(layer)
        mats = {
            "in_proj_qkv": np.ascontiguousarray(
                np.asarray(w["linear_attn.in_proj_qkv.weight"], np.float32)),
            "out_proj": np.ascontiguousarray(
                np.asarray(w["linear_attn.out_proj.weight"], np.float32)),
        }
        for name, mat in mats.items():
            o, i = mat.shape
            x = np.ascontiguousarray((rng.standard_normal((i, S)) * 0.4).astype(np.float16))
            ref = mat @ np.asarray(x, np.float32)
            line = f"  L{layer:<2} {name:12s} {str(mat.shape):14s}"
            for fmt in ("fp16", "int8"):
                progs = eng.compile_procedure_bank([mat], S, weight_format=fmt)
                if not progs:
                    line += f"   {fmt}: COMPILE FAILED"
                    continue
                got = _run(progs[0], x, o)
                line += f"   {fmt} {_rel(got, ref):.5f}"
                del progs
            print(line, flush=True)
        loader.close()
    print("\n  reference points on these tensors: MLX group-64 8-bit 0.0061, "
          "4-bit 0.102 (the shipping build), 3-bit 0.215", flush=True)


if __name__ == "__main__":
    main()
