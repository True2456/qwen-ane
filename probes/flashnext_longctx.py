"""Indexer, KV gather, and QSA cost at 2k / 8k / 16k host context.

The exported QSA graphs are max_S=2048 (indexer budget). Host context can be
much longer: the indexer gathers a budget-sized subset. This probe times that
path at realistic lengths without running a 48-layer prefill.

  python probes/flashnext_longctx.py
  python probes/flashnext_longctx.py --ane --decode-steps 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from runtime.flashnext_indexer import (  # noqa: E402
    QSAIndexer, IndexerState, clip_to_budget,
)


def _clip_selfcheck() -> None:
    # 512 blocks * 4 + 3 tail = 2051; [:2048] would drop the tail.
    body = np.arange(2048)
    tail = np.array([2048, 2049, 2050])
    keep = np.concatenate([body, tail])
    clipped = clip_to_budget(keep, offset=2051, budget=2048, compress_ratio=4)
    assert tail.tolist() == [int(x) for x in clipped if x >= 2048], clipped[-4:]
    assert clipped.size == 2048
    # Dense prefix longer than budget must keep the most recent window.
    dense = clip_to_budget(np.arange(8192), offset=8192, budget=2048)
    assert dense.size == 2048
    assert int(dense.max()) == 8191
    print("  clip_to_budget: PASS (tail kept, dense window is most-recent)", flush=True)


def _gather_ms(keys, keep, reps: int = 20) -> float:
    kv_c = keys.shape[0] * keys.shape[2]
    buf = np.zeros((kv_c, keep.size), keys.dtype)
    # warmup
    buf[:] = keys[:, keep].transpose(0, 2, 1).reshape(kv_c, keep.size)
    t0 = time.perf_counter()
    for _ in range(reps):
        buf[:] = keys[:, keep].transpose(0, 2, 1).reshape(kv_c, keep.size)
    return (time.perf_counter() - t0) / reps * 1e3


async def _ane_qsa(offsets, decode_steps: int, max_s: int) -> None:
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from export_flashnext_coreai import H, QSA_HKV, QSA_HD, QSA_MASK, QSA_ROTARY, SEQ_DEFAULT
    from runtime.coreai_surfaces import wrap_ndarray

    pq = ROOT / "artifacts" / "coreai" / f"flashnext_multitoken_qsa_L3_m{max_s}.aimodel"
    if not pq.is_dir():
        print(f"  missing {pq.name}; skip --ane")
        return
    ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
    print(f"  loading {pq.name}…", flush=True)
    t0 = time.perf_counter()
    model = await AIModel.load(str(pq), specialization_options=spec)
    fn = model.load_function("main")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    kv_c = QSA_HKV * QSA_HD
    seq = SEQ_DEFAULT
    rng = np.random.default_rng(3)
    h = np.zeros((1, H, 1, seq), np.float16)
    h[..., :1] = (rng.standard_normal((1, H, 1, 1)) * 0.05).astype(np.float16)
    k = np.zeros((1, kv_c, 1, max_s), np.float16)
    v = np.zeros((1, kv_c, 1, max_s), np.float16)
    k[0, :, 0, :] = (rng.standard_normal((kv_c, max_s)) * 0.05).astype(np.float16)
    v[0, :, 0, :] = (rng.standard_normal((kv_c, max_s)) * 0.05).astype(np.float16)
    inv = 1.0 / (10_000_000.0 ** (
        np.arange(0, QSA_ROTARY, 2, dtype=np.float32) / np.float32(QSA_ROTARY)))
    cos = np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16)
    sin = np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16)
    mask = np.full((1, max_s + seq, 1, seq), QSA_MASK, np.float16)
    mask[:, :max_s, :, :1] = 0
    mask[:, max_s:max_s + 1, :, :1] = 0

    def feeds(off: int):
        f = np.float32(off) * inv
        cos[0, :, 0, 0] = np.cos(f).astype(np.float16)
        sin[0, :, 0, 0] = np.sin(f).astype(np.float16)
        return {
            "h": wrap_ndarray(h), "k_cache": wrap_ndarray(k),
            "v_cache": wrap_ndarray(v), "cos": wrap_ndarray(cos),
            "sin": wrap_ndarray(sin), "mask": wrap_ndarray(mask),
        }

    # warmup
    out = await fn(feeds(2048))
    _ = out["out"].numpy()
    for off in offsets:
        times = []
        for _ in range(decode_steps):
            t1 = time.perf_counter()
            out = await fn(feeds(off))
            _ = out["out"].numpy()
            times.append((time.perf_counter() - t1) * 1e3)
        print(f"  ANE QSA decode  off={off:<6d}  "
              f"median {float(np.median(times)):.2f} ms  "
              f"mean {float(np.mean(times)):.2f} ms  n={len(times)}", flush=True)


async def _ane_stack(n_chunks: int, k: int, max_s: int, n_layers: int = 48,
                    no_pong: bool = False, skip_qsa: bool = False,
                    skip_gdn: bool = False, page: int = 0) -> None:
    """ANE prefill without MoE. Isolates the IOSurface / context cost."""
    import gc
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from export_flashnext_coreai import HC_W, QSA_HKV, QSA_HD, QSA_MASK, QSA_ROTARY, SEQ_DEFAULT
    from runtime.coreai_surfaces import SurfacePool, wrap_ndarray

    ane = [x for x in ComputeUnitKind.available_kinds() if str(x) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
    n_layers = int(n_layers)
    active = []
    for i in range(n_layers):
        is_qsa = i % 4 == 3
        if is_qsa and skip_qsa:
            continue
        if (not is_qsa) and skip_gdn:
            continue
        active.append(i)

    async def load_ids(ids):
        models, fng, fnq = [], {}, {}
        for i in ids:
            if i % 4 != 3:
                p = ROOT / "artifacts" / "coreai" / f"flashnext_multitoken_step_k{k}_L{i}.aimodel"
                m = await AIModel.load(str(p), specialization_options=spec)
                models.append(m)
                fng[i] = m.load_function("main")
            else:
                p = (ROOT / "artifacts" / "coreai"
                     / f"flashnext_qsa_step_k{k}_L{i}_m{max_s}.aimodel")
                m = await AIModel.load(str(p), specialization_options=spec)
                models.append(m)
                fnq[i] = m.load_function("main")
        return models, fng, fnq

    t0 = time.perf_counter()
    page = int(page or 0)
    if page <= 0:
        keep, fng, fnq = await load_ids(active)
        print(f"  loaded {len(fng)} GDN + {len(fnq)} QSA  layers={n_layers}  "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)
        pages = [active]
    else:
        keep, fng, fnq = [], {}, {}
        pages = [active[i:i + page] for i in range(0, len(active), page)]
        print(f"  paging {len(active)} layers in {len(pages)} pages of {page}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    pool = SurfacePool(seq=SEQ_DEFAULT)
    for i in active:
        if i % 4 != 3:
            pool.add_gdn(i)
    seq = SEQ_DEFAULT
    kv_c = QSA_HKV * QSA_HD
    pool.x_hc[..., :k] = np.float16(0.01)
    kbuf = np.zeros((1, kv_c, 1, max_s), np.float16)
    vbuf = np.zeros_like(kbuf)
    cos = np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16)
    sin = np.zeros_like(cos)
    mask = np.full((1, max_s + seq, 1, seq), QSA_MASK)
    mask[:, : min(64, max_s), :, :k] = 0
    xb = np.zeros((1, HC_W, 1, seq), np.float16)
    xb[..., :k] = np.float16(0.01)
    t_run = time.perf_counter()
    marks = {n_chunks // 8, n_chunks // 4, n_chunks // 2, n_chunks * 3 // 4, n_chunks}
    marks.update({1, 32, 64})
    if no_pong:
        print("  no-pong: rewrap conv/ssm every submit", flush=True)
    for step in range(1, n_chunks + 1):
        for ids in pages:
            if page > 0:
                keep, fng, fnq = await load_ids(ids)
            for i in ids:
                if i in fng:
                    if no_pong:
                        pool._conv_nd[i] = None
                        pool._ssm_nd[i] = None
                    out = await fng[i](pool.pure_feeds(i))
                    if not no_pong:
                        pool.accept_connected(i, out)
                    for key in list(out):
                        _ = np.array(out.pop(key).numpy(), copy=True)
                    del out
                elif i in fnq:
                    feed = {
                        "x": wrap_ndarray(xb), "k_cache": wrap_ndarray(kbuf),
                        "v_cache": wrap_ndarray(vbuf), "cos": wrap_ndarray(cos),
                        "sin": wrap_ndarray(sin), "mask": wrap_ndarray(mask),
                    }
                    out = await fnq[i](feed)
                    for key in list(out):
                        _ = np.array(out.pop(key).numpy(), copy=True)
                    del out, feed
            if page > 0:
                del keep, fng, fnq
                gc.collect()
        if step % 2 == 0:
            gc.collect()
        if step in marks or step % 32 == 0:
            dt = time.perf_counter() - t_run
            toks = step * k
            print(f"  chunk {step}/{n_chunks}  {toks} tok  {dt:.1f}s  "
                  f"{toks / max(dt, 1e-9):.1f} tok/s  "
                  f"{dt / toks * 1e3:.2f} ms/token", flush=True)
    dt = time.perf_counter() - t_run
    toks = n_chunks * k
    print(f"  ANE-only {toks} tokens  {dt:.2f}s  {toks / dt:.1f} tok/s  "
          f"{dt / toks * 1e3:.2f} ms/token", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ane", action="store_true")
    ap.add_argument("--ane-stack", type=int, default=0,
                    help="run N ANE-only chunks of --k (no MoE)")
    ap.add_argument("--layers", type=int, default=48,
                    help="how many layers the ANE-only stack loads")
    ap.add_argument("--no-pong", action="store_true",
                    help="rewrap GDN conv/ssm every submit instead of ping-pong")
    ap.add_argument("--skip-qsa", action="store_true")
    ap.add_argument("--skip-gdn", action="store_true")
    ap.add_argument("--page", type=int, default=0,
                    help="keep at most N ANE models loaded; reload per page")
    ap.add_argument("--skip-host", action="store_true")
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--max-s", type=int, default=2048)
    a = ap.parse_args()

    if a.skip_host:
        if a.ane:
            import asyncio
            asyncio.run(_ane_qsa((2048, 8192, 16384), a.decode_steps, a.max_s))
        if a.ane_stack:
            print(f"\n== ANE {a.layers}-layer stack, {a.ane_stack} chunks of k={a.k} (no MoE) ==",
                  flush=True)
            import asyncio
            asyncio.run(_ane_stack(
                a.ane_stack, a.k, a.max_s, a.layers, a.no_pong,
                a.skip_qsa, a.skip_gdn, a.page))
        return

    print("== clip_to_budget ==", flush=True)
    _clip_selfcheck()

    from export_flashnext_coreai import _load_layer, BASE, H, QSA_HKV, QSA_HD

    cfg = json.loads((BASE / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    loader, w = _load_layer(a.layer)
    indexer = QSAIndexer(w, cfg)
    budget = indexer.budget
    print(f"  layer {a.layer}  H={H}  budget={budget}  ratio={indexer.compress_ratio}",
          flush=True)

    rng = np.random.default_rng(7)
    state = IndexerState()
    marks = (2048, 4096, 8192, 16384)
    marks = tuple(m for m in marks if m <= a.max_ctx)
    xs = (rng.standard_normal((a.max_ctx, H)) * 0.05).astype(np.float32)

    print(f"\n== indexer stream to {a.max_ctx} (chunk k={a.k}) ==", flush=True)
    t_all = time.perf_counter()
    pos = 0
    last = {m: None for m in marks}
    while pos < a.max_ctx:
        n = min(a.k, a.max_ctx - pos)
        t0 = time.perf_counter()
        sel = indexer.update_and_select(xs[pos:pos + n], pos, state)
        dt = (time.perf_counter() - t0) * 1e3
        raw_n = 0 if sel is None else int(np.asarray(sel).size)
        clipped = (np.arange(pos, dtype=np.int64) if sel is None
                   else np.asarray(sel, np.int64))
        clipped = clip_to_budget(clipped, pos, budget)
        rec = {
            "ms": dt, "raw": raw_n, "clipped": int(clipped.size),
            "blocks": 0 if state.block_keys is None else int(state.block_keys.shape[0]),
            "sel_none": sel is None,
        }
        for m in marks:
            if pos < m <= pos + n:
                last[m] = rec
        pos += n
    print(f"  streamed {a.max_ctx} tokens in {time.perf_counter() - t_all:.2f}s",
          flush=True)
    print(f"  {'ctx':>6}  {'chunk_ms':>8}  {'raw':>6}  {'clip':>6}  {'blocks':>7}  none",
          flush=True)
    for m in marks:
        r = last[m]
        if r is None:
            continue
        print(f"  {m:6d}  {r['ms']:8.3f}  {r['raw']:6d}  {r['clipped']:6d}  "
              f"{r['blocks']:7d}  {r['sel_none']}", flush=True)

    print(f"\n== decode (L=1) indexer at mark ==", flush=True)

    def _clone(st: IndexerState) -> IndexerState:
        out = IndexerState()
        out.block_keys = None if st.block_keys is None else st.block_keys.copy()
        out.recent = None if st.recent is None else st.recent.copy()
        out.recent_start = st.recent_start
        out.covers_prefix = st.covers_prefix
        return out

    print(f"  {'ctx':>6}  {'L1_ms':>8}  {'raw':>6}  {'clip':>6}  tail_in_clip", flush=True)
    # Rebuild once, snapshot at each mark.
    state_d = IndexerState()
    pos = 0
    while pos < a.max_ctx:
        n = min(a.k, a.max_ctx - pos)
        indexer.update_and_select(xs[pos:pos + n], pos, state_d)
        pos += n
        if pos not in marks:
            continue
        x1 = (rng.standard_normal((1, H)) * 0.05).astype(np.float32)
        times, raws, clips, tails = [], [], [], []
        for _ in range(8):
            snap = _clone(state_d)
            t0 = time.perf_counter()
            sel = indexer.update_and_select(x1, pos, snap)
            times.append((time.perf_counter() - t0) * 1e3)
            raw = 0 if sel is None else int(np.asarray(sel).size)
            clipped = clip_to_budget(
                np.arange(pos) if sel is None else np.asarray(sel, np.int64),
                pos, budget)
            raws.append(raw)
            clips.append(int(clipped.size))
            tails.append(int(clipped.max()) == pos - 1 if clipped.size else False)
        print(f"  {pos:6d}  {float(np.median(times)):8.3f}  {raws[-1]:6d}  "
              f"{clips[-1]:6d}  {all(tails)}", flush=True)

    print("\n== KV gather (2 heads × 256 d, 2048 selected) ==", flush=True)
    nsel = budget
    keep = np.linspace(0, a.max_ctx - 1, nsel, dtype=np.int64)
    print(f"  {'ctx':>6}  {'fp32_ms':>8}  {'fp16_ms':>8}  {'MB':>6}", flush=True)
    for ctx in marks:
        keep_c = keep.copy()
        keep_c[keep_c >= ctx] = ctx - 1
        k32 = np.ascontiguousarray(
            rng.standard_normal((QSA_HKV, ctx, QSA_HD)).astype(np.float32))
        k16 = np.ascontiguousarray(k32.astype(np.float16))
        ms32 = _gather_ms(k32, keep_c)
        ms16 = _gather_ms(k16, keep_c)
        mb = k16.nbytes * 2 / 1e6  # k+v
        print(f"  {ctx:6d}  {ms32:8.3f}  {ms16:8.3f}  {mb:6.1f}", flush=True)

    loader.close()

    if a.ane:
        print("\n== ANE QSA decode at filled 2048-wide cache ==", flush=True)
        import asyncio
        asyncio.run(_ane_qsa(marks, a.decode_steps, a.max_s))

    if a.ane_stack:
        print(f"\n== ANE {a.layers}-layer stack, {a.ane_stack} chunks of k={a.k} (no MoE) ==",
              flush=True)
        import asyncio
        asyncio.run(_ane_stack(
            a.ane_stack, a.k, a.max_s, a.layers, a.no_pong,
            a.skip_qsa, a.skip_gdn, a.page))


if __name__ == "__main__":
    main()
