"""Does Core AI free IOSurface NDArrays when Python drops them?

  python probes/flashnext_iosurface_gc.py
  python probes/flashnext_iosurface_gc.py --mode infer --n 4000
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


async def _infer(n: int, hold: bool) -> None:
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from runtime.coreai_surfaces import wrap_ndarray

    p = ROOT / "artifacts" / "coreai" / "flashnext_multitoken_qsa_L3_m2048.aimodel"
    ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
    print(f"  loading {p.name}", flush=True)
    m = await AIModel.load(str(p), specialization_options=spec)
    fn = m.load_function("main")
    rng = np.random.default_rng(1)
    H, seq, max_s, kv_c = 2560, 32, 2048, 512
    h = np.zeros((1, H, 1, seq), np.float16); h[..., :1] = np.float16(0.05)
    k = (rng.standard_normal((1, kv_c, 1, max_s)) * 0.05).astype(np.float16)
    v = (rng.standard_normal((1, kv_c, 1, max_s)) * 0.05).astype(np.float16)
    cos = np.zeros((1, 32, 1, seq), np.float16)
    sin = np.zeros_like(cos)
    mask = np.full((1, max_s + seq, 1, seq), np.float16(-40000))
    mask[:, :max_s, :, :1] = 0
    mask[:, max_s:max_s + 1, :, :1] = 0

    held = []
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        feed = {
            "h": wrap_ndarray(h), "k_cache": wrap_ndarray(k),
            "v_cache": wrap_ndarray(v), "cos": wrap_ndarray(cos),
            "sin": wrap_ndarray(sin), "mask": wrap_ndarray(mask),
        }
        out = await fn(feed)
        if hold:
            held.append(out)
        else:
            for key in list(out):
                _ = np.array(out.pop(key).numpy(), copy=True)
            del out, feed
        if i % 100 == 0 or i in (1, 32, 64):
            gc.collect()
            print(f"  infer {i}/{n}  held={len(held)}  "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
    print("  infer PASS", flush=True)


def _wrap_only(n: int, hold: bool, ios: bool, shape: tuple[int, ...]) -> None:
    from runtime.coreai_surfaces import wrap_ndarray
    buf = np.zeros(shape, np.float16)
    print(f"  wrap shape={shape} bytes={buf.nbytes}", flush=True)
    held = []
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        nd = wrap_ndarray(buf, iosurface=ios)
        if hold:
            held.append(nd)
        else:
            del nd
        if i % 500 == 0 or i in (1, 100, 250):
            gc.collect()
            print(f"  wrap {i}/{n}  ios={ios} held={len(held)}  "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
    print("  wrap PASS", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("wrap", "infer"), default="wrap")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--bytes", action="store_true",
                    help="wrap as BYTES instead of IOSurface")
    ap.add_argument("--kcache", action="store_true",
                    help="wrap the 2 MiB QSA k_cache shape instead of 32 KiB new_k")
    args = ap.parse_args()
    print(f"mode={args.mode} n={args.n} hold={args.hold}", flush=True)
    if args.mode == "wrap":
        shape = (1, 512, 1, 2048) if args.kcache else (1, 512, 1, 32)
        _wrap_only(args.n, args.hold, ios=not args.bytes, shape=shape)
    else:
        asyncio.run(_infer(args.n, args.hold))


if __name__ == "__main__":
    main()
