"""Multi-token QSA: the missing half of a k-token backbone pass.

The GDN layers needed a real graph change to consume k tokens (causal conv
shift + unrolled recurrence). QSA does not: `FlashNextQSADecode` already
computes q/k/v for all 32 slots and takes a per-query-slot mask, so the graph
is unchanged and only the host feed has to stop being single-token.

Two things move:
  * RoPE — one position per slot (off + i), not one position for every slot;
  * mask — query slot i sees the cache 0..off-1 plus batch slots 0..i, i.e.
    causal *within* the batch. The shipped feed opens only query slot 0.

Verified against k sequential single-token calls with the cache advanced
between them, which is what decode does today.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_flashnext_coreai import (  # noqa: E402
    FlashNextQSADecode, _load_layer, _export, _pad32_into, _bsh_to_bc1s,
    H, QSA_HKV, QSA_HD, QSA_ROTARY, QSA_MASK, SEQ_DEFAULT,
)

S = SEQ_DEFAULT
KV_C = QSA_HKV * QSA_HD
MAX_S = SEQ_DEFAULT   # KV history width; independent of the S token slots


def rope(positions) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin with one RoPE position per token slot."""
    half = QSA_ROTARY // 2
    idx = np.arange(0, QSA_ROTARY, 2, dtype=np.float32)
    inv = 1.0 / (10_000_000.0 ** (idx / np.float32(QSA_ROTARY)))
    cos = np.zeros((1, half, 1, S), np.float16)
    sin = np.zeros((1, half, 1, S), np.float16)
    for slot, pos in enumerate(positions):
        f = np.float32(pos) * inv
        cos[0, :, 0, slot] = np.cos(f).astype(np.float16)
        sin[0, :, 0, slot] = np.sin(f).astype(np.float16)
    return cos, sin


def mask_multi(offset: int, k: int) -> np.ndarray:
    """(1, MAX_S+S, 1, S). Query slot i sees cache 0..offset-1 and batch 0..i."""
    m = np.full((1, MAX_S + S, 1, S), QSA_MASK, np.float16)
    if offset > 0:
        m[:, :offset, :, :k] = 0
    for i in range(k):
        m[:, MAX_S:MAX_S + i + 1, :, i] = 0
    return m


def feeds(h_bsh, k_hist, v_hist, offset, k):
    """h_bsh (1, k, H) -> the six ANE inputs, with k real slots."""
    h = np.zeros((1, H, 1, S), np.float16)
    bc = _bsh_to_bc1s(np.asarray(h_bsh, np.float32))
    h[..., :k] = np.asarray(bc[..., :k], np.float16)
    kc = np.zeros((1, KV_C, 1, MAX_S), np.float16)
    vc = np.zeros((1, KV_C, 1, MAX_S), np.float16)
    if offset:
        kc[0, :, 0, :offset] = k_hist[:offset].T
        vc[0, :, 0, :offset] = v_hist[:offset].T
    cos, sin = rope(range(offset, offset + k))
    return h, kc, vc, cos, sin, mask_multi(offset, k)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--offset", type=int, default=5)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--export-all", action="store_true",
                    help="export every QSA layer at this max_s")
    ap.add_argument("--max-s", type=int, default=SEQ_DEFAULT,
                    help="KV history width baked into the exported graph")
    a = ap.parse_args()
    global MAX_S
    MAX_S = a.max_s

    if getattr(a, "export_all", False):
        import asyncio as _aio
        rng0 = np.random.default_rng(1)
        k = a.steps[0]
        for i in [x for x in range(48) if x % 4 == 3]:
            out = Path(f"artifacts/coreai/flashnext_multitoken_qsa_L{i}_m{MAX_S}.aimodel")
            if out.is_dir():
                print(f"  L{i} exists, skipping", flush=True)
                continue
            ld, lw = _load_layer(i)
            mm = FlashNextQSADecode().eval().half()
            mm.load_from_layer(lw)
            hs0 = (rng0.standard_normal((1, k, H)) * 0.05).astype(np.float32)
            kh0 = (rng0.standard_normal((MAX_S, KV_C)) * 0.05).astype(np.float16)
            vh0 = (rng0.standard_normal((MAX_S, KV_C)) * 0.05).astype(np.float16)
            f0 = feeds(hs0, kh0, vh0, 0, k)
            _export(mm, tuple(_t(v) for v in f0),
                    (["h", "k_cache", "v_cache", "cos", "sin", "mask"],
                     ["out", "new_k", "new_v"]),
                    f"multitoken_qsa_L{i}_m{MAX_S}")
            ld.close()
            print(f"  exported QSA L{i} max_S={MAX_S}", flush=True)
        return

    loader, w = _load_layer(a.layer)
    m = FlashNextQSADecode().eval().half()
    m.load_from_layer(w)

    rng = np.random.default_rng(11)
    for k in a.steps:
        hs = (rng.standard_normal((1, k, H)) * 0.05).astype(np.float32)
        k_hist = (rng.standard_normal((MAX_S, KV_C)) * 0.05).astype(np.float16)
        v_hist = (rng.standard_normal((MAX_S, KV_C)) * 0.05).astype(np.float16)

        # reference: k single-token calls, cache advanced between them
        kh = k_hist.copy()
        vh = v_hist.copy()
        ref = []
        with torch.no_grad():
            for i in range(k):
                f = feeds(hs[:, i:i + 1], kh, vh, a.offset + i, 1)
                out, nk, nv = m(*[_t(v) for v in f])
                ref.append(out[..., :1].clone())
                kh[a.offset + i] = nk.numpy()[0, :, 0, 0]
                vh[a.offset + i] = nv.numpy()[0, :, 0, 0]

        f = feeds(hs, k_hist, v_hist, a.offset, k)
        with torch.no_grad():
            out, nk, nv = m(*[_t(v) for v in f])

        def rel(p, q):
            p = np.asarray(p.float().numpy(), np.float64)
            q = np.asarray(q.float().numpy(), np.float64)
            return float(np.linalg.norm(p - q) / max(np.linalg.norm(q), 1e-12))

        err = max(rel(out[..., i:i + 1], ref[i]) for i in range(k))
        kv_err = max(
            float(np.abs(nk.numpy()[0, :, 0, i] - kh[a.offset + i]).max()) for i in range(k)
        )
        line = f"  k={k:<2} vs {k} single calls: out {err:.5f}  new_k {kv_err:.5f}"
        if a.bench:
            line += "  " + asyncio.run(bench(m, f, k, a.layer, a.reps))
        print(line, flush=True)
    loader.close()


async def bench(m, f, k, layer, reps) -> str:
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from runtime.coreai_surfaces import wrap_ndarray
    names = ["h", "k_cache", "v_cache", "cos", "sin", "mask"]
    path = _export(m, tuple(_t(v) for v in f), (names, ["out", "new_k", "new_v"]),
                   f"multitoken_qsa_L{layer}_m{MAX_S}")
    ane = [x for x in ComputeUnitKind.available_kinds() if str(x) == "Neural Engine"][0]
    model = await AIModel.load(str(path),
                               specialization_options=SpecializationOptions
                               .from_preferred_compute_unit_kind(ane))
    fn = model.load_function("main")
    feed = {n: wrap_ndarray(v) for n, v in zip(names, f)}
    for _ in range(4):
        await fn(feed)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        await fn(feed)
        ts.append(time.perf_counter() - t)
    ms = float(np.median(ts)) * 1e3
    return f"ANE {ms:6.3f} ms  {ms / k:6.3f} ms/token"


if __name__ == "__main__":
    main()
