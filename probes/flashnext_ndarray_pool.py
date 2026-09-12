"""Can we reuse Core AI output NDArrays instead of allocating a new IOSurface
per submit? That abort at ~30-60 chunks is ``Failed to allocate storage for
NDArray ... sk: ioSurface`` on QSA ``new_k`` (32768 bytes).

  python probes/flashnext_ndarray_pool.py
  python probes/flashnext_ndarray_pool.py --submits 400
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _dump_desc(fn) -> None:
    d = fn.desc
    print(f"  desc type={type(d).__name__}", flush=True)
    print(f"  desc={d}", flush=True)
    names = [n for n in dir(d) if not n.startswith("_")]
    print(f"  attrs={names}", flush=True)
    for n in names:
        try:
            v = getattr(d, n)
        except Exception as e:
            print(f"    {n}: <err {e}>", flush=True)
            continue
        if callable(v):
            continue
        print(f"    {n}={v!r}", flush=True)
        # Descriptor maps are the interesting bit: shape / dtype / storage.
        if n.endswith("_descriptor"):
            try:
                items = list(v.items()) if hasattr(v, "items") else list(v)
            except Exception:
                items = []
            for item in items:
                print(f"      item={item!r}  type={type(item).__name__}", flush=True)
                obj = item[1] if isinstance(item, tuple) and len(item) == 2 else item
                attrs = [a for a in dir(obj) if not a.startswith("_")]
                print(f"      attrs={attrs}", flush=True)
                for a in attrs:
                    try:
                        av = getattr(obj, a)
                    except Exception:
                        continue
                    if callable(av):
                        continue
                    print(f"        {a}={av!r}", flush=True)


async def _load_qsa(max_s: int):
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from export_flashnext_coreai import H, QSA_HKV, QSA_HD, QSA_MASK, QSA_ROTARY, SEQ_DEFAULT
    from runtime.coreai_surfaces import wrap_ndarray

    p = ROOT / "artifacts" / "coreai" / f"flashnext_multitoken_qsa_L3_m{max_s}.aimodel"
    if not p.is_dir():
        p = ROOT / "artifacts" / "coreai" / f"flashnext_qsa_step_k16_L3_m{max_s}.aimodel"
    ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
    print(f"  loading {p.name}…", flush=True)
    t0 = time.perf_counter()
    model = await AIModel.load(str(p), specialization_options=spec)
    fn = model.load_function("main")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    return fn, wrap_ndarray, H, QSA_HKV, QSA_HD, QSA_MASK, QSA_ROTARY, SEQ_DEFAULT, max_s, p.name


def _feeds(wrap, H, kv_c, max_s, seq, mask_val, rng):
    h = np.zeros((1, H, 1, seq), np.float16)
    h[..., :1] = (rng.standard_normal((1, H, 1, 1)) * 0.05).astype(np.float16)
    k = (rng.standard_normal((1, kv_c, 1, max_s)) * 0.05).astype(np.float16)
    v = (rng.standard_normal((1, kv_c, 1, max_s)) * 0.05).astype(np.float16)
    cos = np.zeros((1, 32, 1, seq), np.float16)
    sin = np.zeros_like(cos)
    mask = np.full((1, max_s + seq, 1, seq), mask_val, np.float16)
    mask[:, :max_s, :, :1] = 0
    mask[:, max_s:max_s + 1, :, :1] = 0
    return {
        "h": wrap(h), "k_cache": wrap(k), "v_cache": wrap(v),
        "cos": wrap(cos), "sin": wrap(sin), "mask": wrap(mask),
    }


async def _try_state(fn, inputs) -> dict | None:
    out = await fn(inputs)
    keys = list(out)
    print(f"  outputs: {[(k, tuple(out[k].shape), out[k].dtype) for k in keys]}",
          flush=True)
    # Reuse the same NDArrays as `state` on the next call.
    try:
        out2 = await fn(inputs, state=out)
        print(f"  state=prev_outputs: OK  keys={list(out2)}", flush=True)
        same = {k: out[k] is out2[k] for k in keys if k in out2}
        print(f"  identity vs prev: {same}", flush=True)
        return out2
    except Exception as e:
        print(f"  state=prev_outputs FAILED: {type(e).__name__}: {e}", flush=True)
        return None


async def _stress(fn, make_inputs, n: int, *, reuse_state: bool, copy_out: bool) -> None:
    t0 = time.perf_counter()
    state = None
    for i in range(1, n + 1):
        inputs = make_inputs()
        out = await fn(inputs, state=state) if state is not None else await fn(inputs)
        if copy_out:
            for k in list(out):
                _ = np.array(out[k].numpy(), copy=True)
        if reuse_state:
            state = out
        else:
            out.clear()
            del out
        if i % 50 == 0 or i in (1, 8, 16, 32):
            dt = time.perf_counter() - t0
            print(f"  submit {i}/{n}  {dt:.1f}s  {dt / i * 1e3:.2f} ms/call"
                  f"  reuse={reuse_state}", flush=True)
            gc.collect()
    dt = time.perf_counter() - t0
    print(f"  DONE {n} submits in {dt:.1f}s  {dt / n * 1e3:.2f} ms/call"
          f"  reuse={reuse_state}", flush=True)


async def main_async(args) -> None:
    rng = np.random.default_rng(3)
    fn, wrap, H, QSA_HKV, QSA_HD, QSA_MASK, QSA_ROTARY, seq, max_s, name = (
        await _load_qsa(args.max_s)
    )
    print(f"  graph={name}  rotary={QSA_ROTARY}", flush=True)
    print("\n== descriptor ==", flush=True)
    _dump_desc(fn)

    kv_c = QSA_HKV * QSA_HD
    inputs = _feeds(wrap, H, kv_c, max_s, seq, QSA_MASK, rng)
    print("\n== try state=outputs ==", flush=True)
    reused = await _try_state(fn, inputs)
    if reused is not None and not reused:
        print("  (empty outputs — graphs declare States: []; cannot bind backings)",
              flush=True)
        reused = None

    def make():
        return _feeds(wrap, H, kv_c, max_s, seq, QSA_MASK, rng)

    n = args.submits
    print(f"\n== stress drop-outputs  n={n} ==", flush=True)
    await _stress(fn, make, n, reuse_state=False, copy_out=True)
    if reused is not None:
        print(f"\n== stress reuse-state  n={n} ==", flush=True)
        await _stress(fn, make, n, reuse_state=True, copy_out=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submits", type=int, default=200)
    ap.add_argument("--max-s", type=int, default=2048)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
