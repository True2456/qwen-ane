#!/usr/bin/env python3
"""Correctness: packed ANE MoE vs host SwiGLU on Flash-Next layer 0.

    FLASHNEXT_MOE=anepacked \\
    ~/.rindi/venvs/coreai/bin/python -u probes/flashnext_ane_packed_check.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("Q38_ANE_ENGINE", str(ROOT))
os.environ["FLASHNEXT_MOE"] = "anepacked"

from runtime.host_fastpath import silu  # noqa: E402
from export_flashnext_coreai import (  # noqa: E402
    H, I, K_PIN, HostMoE, expert_f16_store, BASE,
)


def _ref(moe: HostMoE, x: np.ndarray, ids, scores, *, fp16: bool = True) -> np.ndarray:
    """Host SwiGLU. fp16=True matches ANE weight surfaces (fp16 matmul)."""
    x1 = np.asarray(x, np.float32).reshape(-1, H)
    if fp16:
        x1 = x1.astype(np.float16).astype(np.float32)
    slabs = moe.store.ensure_many(moe.layer, ids, moe.gu, moe.dn)
    routed = np.zeros(H, np.float32)
    for (g, d), s in zip(slabs, scores):
        g32 = np.asarray(g, np.float16).astype(np.float32) if fp16 else np.asarray(g, np.float32)
        d32 = np.asarray(d, np.float16).astype(np.float32) if fp16 else np.asarray(d, np.float32)
        gu = g32 @ x1[0]
        if fp16:
            gu = gu.astype(np.float16).astype(np.float32)
        act = silu(gu[:I]) * gu[I:] * float(s)
        if fp16:
            act = act.astype(np.float16).astype(np.float32)
        routed += d32 @ act
    if fp16:
        routed = routed.astype(np.float16).astype(np.float32)
    sg = silu(x1 @ moe.shared_gate.T)
    su = x1 @ moe.shared_up.T
    shared = (sg * su) @ moe.shared_down.T
    sgate = 1.0 / (1.0 + np.exp(-np.clip(x1 @ moe.shared_sgate.T, -80, 80)))
    return routed + (sgate * shared)[0]


def main() -> int:
    from tools.flashnext_reference import FlashNextLoader

    loader = FlashNextLoader(str(BASE))
    w = loader.layer(0)
    store = expert_f16_store()
    moe = HostMoE(w, seq=32, store=store)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, H), dtype=np.float32).astype(np.float32)
    t0 = time.perf_counter()
    y = moe.apply(x)
    print(f"decode apply {time.perf_counter()-t0:.3f}s  y{y.shape}", flush=True)
    ids, sc = moe.inds[0], moe.scores[0, :, 0, 0]
    ref = _ref(moe, x, ids, sc)
    y1 = np.asarray(y, np.float32).reshape(-1)
    rel = float(np.linalg.norm(y1 - ref) / max(np.linalg.norm(ref), 1e-12))
    print(
        f"  vs host SwiGLU rel={rel:.3e}  ||y||={float(np.linalg.norm(y1)):.4f}  "
        f"||ref||={float(np.linalg.norm(ref)):.4f}  max|y-ref|={float(np.max(np.abs(y1-ref))):.4f}  "
        f"ids={np.asarray(ids).tolist()}",
        flush=True,
    )

    xt = rng.standard_normal((8, H), dtype=np.float32).astype(np.float32)
    t0 = time.perf_counter()
    inds, scores = moe._route(xt)
    y8 = moe._resident.routed_multi(xt, inds, scores)
    print(f"prefill T=8  {time.perf_counter()-t0:.3f}s  y{y8.shape}", flush=True)
    refs = np.stack([_ref(moe, xt[t], inds[t], scores[t]) for t in range(8)])
    y8r = np.asarray(y8, np.float32).reshape(8, H)
    rel8 = float(np.linalg.norm(y8r - refs) / max(np.linalg.norm(refs), 1e-12))
    print(f"  vs host SwiGLU rel={rel8:.3e}", flush=True)
    # ANE fp16 vs host fp16: ~1e-2 is normal; vs fp32 host is a bit higher.
    ok = rel < 3e-2 and rel8 < 3e-2
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
