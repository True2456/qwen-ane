#!/usr/bin/env python3
"""Verify two-procedure QSA MIL compilation and numerical equivalence against single-procedure."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_flashnext_coreai import (
    _load_layer, FlashNextQSADecode, H, HC, HC_W, QSA_HQ, QSA_HKV, QSA_HD,
    QSA_ROTARY, QSA_MASK, SEQ_DEFAULT,
)
import flashnext_mil_qsa_layer as QL
from runtime.q38_ane_engine import _iosurface_view

S = SEQ_DEFAULT
G = QSA_HQ // QSA_HKV
HALF = QSA_ROTARY // 2
KVC = QSA_HKV * QSA_HD


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def test_qsa_two_proc(layer_idx: int = 3, m: int = 2048, k_narrow: int = 4, k_wide: int = 32):
    print(f"Testing QSA Layer {layer_idx} multi-procedure (M={m}, k={k_narrow}, {k_wide})...")
    loader, w = _load_layer(layer_idx)
    ref = QL._Ref(w)
    qsa = FlashNextQSADecode(max_s=m).eval().half()
    qsa.load_from_layer(w)
    loader.close()

    # 1. Compile single-procedure k=k_narrow
    print(f"  Compiling single-procedure k={k_narrow}...")
    QL.LAYER[0] = layer_idx
    QL.KVM[0] = m
    QL.KTOK[0] = k_narrow
    prog_narrow, hcn = QL.build_layer(w, ref, qsa)

    # 2. Compile single-procedure k=k_wide
    print(f"  Compiling single-procedure k={k_wide}...")
    QL.KTOK[0] = k_wide
    prog_wide, _ = QL.build_layer(w, ref, qsa)

    # 3. Compile two-procedure program
    print(f"  Compiling two-procedure program [{k_narrow}, {k_wide}]...")
    prog_multi, _ = QL.build_program_multi(w, ref, qsa, [(k_narrow, m), (k_wide, m)])
    assert prog_multi is not None, "build_program_multi returned None"

    print("  Compiled successfully. Comparing outputs...")

    rng = np.random.default_rng(42)
    # Test narrow and wide widths
    for proc_idx, (kt, prog_single) in enumerate([(k_narrow, prog_narrow), (k_wide, prog_wide)]):
        x = np.zeros((HC_W, S), np.float16)
        x[:, :kt] = (rng.standard_normal((HC_W, kt)) * 0.05).astype(np.float16)
        kc = np.ascontiguousarray((rng.standard_normal((KVC, m)) * 0.05).astype(np.float16))
        vc = np.ascontiguousarray((rng.standard_normal((KVC, m)) * 0.05).astype(np.float16))
        cos = np.zeros((HALF, S), np.float16)
        sin = np.zeros((HALF, S), np.float16)
        pos = (np.arange(kt, dtype=np.float32) + 17.0)[:, None] * np.arange(HALF)[None, :] * 0.01
        cos[:, :kt] = np.cos(pos).T.astype(np.float16)
        sin[:, :kt] = np.sin(pos).T.astype(np.float16)
        mixed = np.zeros((H, S), np.float16)
        inj = np.zeros((HC, S), np.float16)
        mixed[:, :kt] = (rng.standard_normal((H, kt)) * 0.05).astype(np.float16)
        inj[:, :kt] = (rng.standard_normal((HC, kt)) * 0.05).astype(np.float16)

        row = np.full((kt, m + S), QSA_MASK, np.float16)
        row[:, :17] = 0
        for t in range(kt):
            row[t, m:m + t + 1] = 0
        mask = np.ascontiguousarray(np.tile(row, (G, 1)))

        # Run single prog
        for surf, val in zip(prog_single._in_surfs, (x, cos, sin, hcn, kc, vc, mask, mixed, inj)):
            with _iosurface_view(surf, val.shape, np.float16) as d:
                np.copyto(d, val)
        assert QL.eng.submit(prog_single, procedure_index=0), "single submit failed"

        out_single = {}
        for idx, nm, shape in ((0, "new_k", (KVC, S)), (1, "shared", (H, S)),
                               (2, "mixed", (H, S)), (3, "hyper", (HC_W, S)),
                               (4, "inj", (HC, S)), (5, "new_v", (KVC, S))):
            with _iosurface_view(prog_single._out_surfs[idx], shape, np.float16) as o:
                out_single[nm] = np.array(o[:, :kt], np.float32)

        # Run multi prog
        in_map = prog_multi.proc_in_map[proc_idx]
        surfs_multi = [prog_multi._in_surfs[i] for i in in_map]
        for surf, val in zip(surfs_multi, (x, cos, sin, hcn, kc, vc, mask, mixed, inj)):
            with _iosurface_view(surf, val.shape, np.float16) as d:
                np.copyto(d, val)
        assert QL.eng.submit(prog_multi, procedure_index=proc_idx), f"multi submit {proc_idx} failed"

        out_multi = {}
        for idx, nm, shape in ((0, "new_k", (KVC, S)), (1, "shared", (H, S)),
                               (2, "mixed", (H, S)), (3, "hyper", (HC_W, S)),
                               (4, "inj", (HC, S)), (5, "new_v", (KVC, S))):
            with _iosurface_view(prog_multi._out_surfs[idx], shape, np.float16) as o:
                out_multi[nm] = np.array(o[:, :kt], np.float32)

        for nm in ("mixed", "hyper", "shared", "new_k", "new_v"):
            err = rel(out_multi[nm], out_single[nm])
            print(f"    proc {proc_idx} (k={kt}) {nm} rel err vs single: {err:.6f}")
            assert err < 1e-4, f"Mismatch in {nm} for proc {proc_idx}: {err}"

    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    test_qsa_two_proc()
