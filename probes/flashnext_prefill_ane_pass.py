"""Time the ANE half of a k-token pass across all 48 real layers.

The k=16 layer numbers so far come from layer 0 in isolation. This loads every
exported k=16 GDN graph plus the QSA layers and runs one full pass with 16 real
token slots, so the projection is checked against the whole stack: 36 graphs
resident at once, per-layer recurrent surfaces ping-ponged, QSA fed the
multi-token mask and per-slot RoPE.

MoE is deliberately excluded — this measures only what the ANE contributes, so
it can be compared against the 36 x 4.2 + 12 x 1.07 projection.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "probes"))

from export_flashnext_coreai import (  # noqa: E402
    H, HC_W, HV, DK, DV, QKV, QSA_HKV, QSA_HD, SEQ_DEFAULT,
)
from flashnext_multitoken_qsa import feeds as qsa_feeds  # noqa: E402

S = SEQ_DEFAULT
KV_C = QSA_HKV * QSA_HD


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--qsa-layer", type=int, default=3)
    ap.add_argument("--max-s", type=int, default=0,
                    help="use the QSA graphs exported at this KV width")
    a = ap.parse_args()

    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from runtime.coreai_surfaces import SurfacePool, wrap_ndarray

    ane = [x for x in ComputeUnitKind.available_kinds() if str(x) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)

    gdn = [i for i in range(48) if i % 4 != 3]
    qsa = [i for i in range(48) if i % 4 == 3]
    keep, fn_gdn = [], {}
    t0 = time.perf_counter()
    for i in gdn:
        p = ROOT / "artifacts" / "coreai" / f"flashnext_multitoken_step_k{a.k}_L{i}.aimodel"
        if not p.is_dir():
            print(f"  missing {p.name}; run flashnext_multitoken_step.py --export-all")
            return
        m = await AIModel.load(str(p), specialization_options=spec)
        keep.append(m)
        fn_gdn[i] = m.load_function("main")
    # one QSA graph, reused for all 12 slots: identical shape, and this probe
    # measures time, not values.
    suffix = f"_m{a.max_s}" if a.max_s else ""
    pq = (ROOT / "artifacts" / "coreai"
          / f"flashnext_multitoken_qsa_L{a.qsa_layer}{suffix}.aimodel")
    m_q = await AIModel.load(str(pq), specialization_options=spec)
    keep.append(m_q)
    fn_q = m_q.load_function("main")
    print(f"  loaded {len(fn_gdn)} GDN k={a.k} + QSA in {time.perf_counter()-t0:.1f}s",
          flush=True)

    pool = SurfacePool(seq=S)
    for i in gdn:
        pool.add_gdn(i)

    rng = np.random.default_rng(3)
    x = np.zeros((1, HC_W, 1, S), np.float16)
    x[..., :a.k] = (rng.standard_normal((1, HC_W, 1, a.k)) * 0.05).astype(np.float16)
    hs = (rng.standard_normal((1, a.k, H)) * 0.05).astype(np.float32)
    k_hist = (rng.standard_normal((S, KV_C)) * 0.05).astype(np.float16)
    v_hist = (rng.standard_normal((S, KV_C)) * 0.05).astype(np.float16)
    if a.max_s:
        import flashnext_multitoken_qsa as Q
        Q.MAX_S = a.max_s
        k_hist = (rng.standard_normal((a.max_s, KV_C)) * 0.05).astype(np.float16)
        v_hist = (rng.standard_normal((a.max_s, KV_C)) * 0.05).astype(np.float16)
    qf = qsa_feeds(hs, k_hist, v_hist, 0, a.k)
    qnames = ["h", "k_cache", "v_cache", "cos", "sin", "mask"]

    async def one_pass():
        t_g = 0.0
        for i in gdn:
            np.copyto(pool.x_hc, x)
            t = time.perf_counter()
            out = await fn_gdn[i](pool.pure_feeds(i))
            pool.accept_connected(i, out)
            out["mixed"].numpy()
            t_g += time.perf_counter() - t
        t_q = 0.0
        for _ in qsa:
            feed = {n: wrap_ndarray(v) for n, v in zip(qnames, qf)}
            t = time.perf_counter()
            o = await fn_q(feed)
            o["out"].numpy()
            t_q += time.perf_counter() - t
        return t_g, t_q

    for _ in range(2):
        await one_pass()
    gs, qs = [], []
    for _ in range(a.reps):
        g, q = await one_pass()
        gs.append(g)
        qs.append(q)
    g = float(np.median(gs)) * 1e3
    q = float(np.median(qs)) * 1e3
    print(f"  36 GDN layers   {g:7.1f} ms   ({g/36:5.3f} ms/layer)")
    print(f"  12 QSA layers   {q:7.1f} ms   ({q/12:5.3f} ms/layer)")
    print(f"  ANE per pass    {g+q:7.1f} ms for {a.k} tokens "
          f"-> {(g+q)/a.k:5.2f} ms/token")


if __name__ == "__main__":
    asyncio.run(main())
