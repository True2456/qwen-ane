"""Separate real device execution from host<->device boundary cost.

Decode is ~285 ms/token: ane+io ~120 ms over 48 submits and MoE ~135 ms over
48 mx.eval syncs. This probe measures each side's floor in isolation:

  ANE : repeat one already-exported pure_step / QSA program with persistent
        surfaces, no host work between calls; with and without debug spec;
        serial vs asyncio.gather to expose queue latency.
  GPU : one resident 4-bit expert layer, 48 routed applications, eval per
        layer vs a single eval for the whole chain.

Nothing here is a decoder; it only tells us which cost to attack.
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

OUT = ROOT / "artifacts" / "coreai"


def _stats(ts):
    a = np.asarray(sorted(ts)) * 1e3
    return f"median={a[len(a)//2]:.3f} ms  p10={a[len(a)//10]:.3f}  p90={a[(9*len(a))//10]:.3f}"


async def ane_floor(debug: bool, n: int, layers: list[int]) -> None:
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from runtime.coreai_surfaces import SurfacePool

    ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
    if debug:
        spec = spec.with_debug(enabled=True)

    keep, fns = [], {}
    for i in layers:
        p = OUT / f"flashnext_pure_step_L{i}.aimodel"
        if not p.is_dir():
            print(f"  [skip] no pure_step L{i}")
            continue
        m = fn = None
        for attempt in range(4):
            try:
                m = await AIModel.load(str(p), specialization_options=spec)
                fn = m.load_function("main")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  [retry {attempt}] L{i}: {exc}")
                m = fn = None
                await asyncio.sleep(0.5 * (attempt + 1))
        if fn is None:
            print(f"  [give up] L{i}")
            continue
        keep.append(m)
        fns[i] = fn
    if not fns:
        print("  no pure_step assets; export first")
        return
    print(f"  loaded {len(fns)} pure_step  debug={debug}")

    pool = SurfacePool(seq=32)
    for i in fns:
        pool.add_gdn(i)

    first = next(iter(fns))
    for _ in range(5):
        out = await fns[first](pool.pure_feeds(first))
        pool.accept_connected(first, out)

    # 1. full per-layer cost as generate pays it
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        feeds = pool.pure_feeds(first)
        out = await fns[first](feeds)
        pool.accept_connected(first, out)
        pool.take_pure(out)
        ts.append(time.perf_counter() - t)
    print(f"  submit+feeds+take   {_stats(ts)}")

    # 2. submit only (feeds built once, outputs untouched)
    feeds = pool.pure_feeds(first)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = await fns[first](feeds)
        ts.append(time.perf_counter() - t)
    print(f"  submit only         {_stats(ts)}")

    # 3. host-side feed/take cost with no device call
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        pool.pure_feeds(first)
        pool.take_pure(out)
        ts.append(time.perf_counter() - t)
    print(f"  feeds+take only     {_stats(ts)}")

    # 4. queue latency: independent layers issued together
    ids = list(fns)[:8]
    if len(ids) >= 4:
        for k in (2, 4, min(8, len(ids))):
            sel = ids[:k]
            batch = {i: pool.pure_feeds(i) for i in sel}
            for _ in range(3):
                await asyncio.gather(*(fns[i](batch[i]) for i in sel))
            ts = []
            for _ in range(max(4, n // 4)):
                t = time.perf_counter()
                await asyncio.gather(*(fns[i](batch[i]) for i in sel))
                ts.append((time.perf_counter() - t) / k)
            print(f"  concurrent x{k:<2} per-call {_stats(ts)}")


def gpu_floor(n_layers: int, reps: int) -> None:
    import mlx.core as mx
    from runtime.flashnext_mlx_moe import ResidentMoe
    from tools.flashnext_reference import FlashNextLoader  # noqa: F401  (weights path)

    from runtime.expert_bank import Mlx4ExpertBank, MLX4_DEFAULT
    import runtime.expert_bank as eb
    bank = Mlx4ExpertBank(MLX4_DEFAULT)
    sg, su, sd = bank.shared_fp32(0)
    src = eb.MlxSafe(MLX4_DEFAULT)
    try:
        sgate = np.asarray(src.f32("model.layers.0.mlp.shared_expert_gate.weight"), np.float32).reshape(1, -1)
    finally:
        src.close()
    shared = (sg, su, sd, sgate)

    moe = ResidentMoe(0, shared)
    print(f"  layer bank {moe.nbytes / 1e9:.2f} GB")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2560).astype(np.float32) * 0.05
    sel = [rng.choice(512, 10, replace=False).astype(np.uint32) for _ in range(n_layers)]
    sc = np.full(10, 0.1, np.float32)

    for i in range(4):
        moe.apply(x, sel[i], sc)

    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for i in range(n_layers):
            moe.apply(x, sel[i], sc)
        ts.append((time.perf_counter() - t) / n_layers)
    print(f"  eval per layer      {_stats(ts)}  -> {np.median(ts)*n_layers*1e3:.1f} ms / token")

    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        ys = [moe.routed(x, sel[i], sc) for i in range(n_layers)]
        mx.eval(ys)
        ts.append((time.perf_counter() - t) / n_layers)
    print(f"  one eval, routed    {_stats(ts)}  -> {np.median(ts)*n_layers*1e3:.1f} ms / token")

    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        y = moe.routed(x, sel[0], sc)
        mx.eval(y)
        ts.append(time.perf_counter() - t)
    print(f"  single routed+eval  {_stats(ts)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ane", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--debug-spec", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--layers", type=int, default=8)
    a = ap.parse_args()
    if a.ane:
        print("[ANE floor]")
        asyncio.run(ane_floor(a.debug_spec, a.n, list(range(a.layers))))
    if a.gpu:
        print("[GPU MoE floor]")
        gpu_floor(48, 6)


if __name__ == "__main__":
    main()
