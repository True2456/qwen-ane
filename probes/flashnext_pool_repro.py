"""Functional lifetime test, not a throughput benchmark or decode projection.

Repeat the actual 36 GDN / 12 QSA assets, with deterministic feeds and live
recurrent state. --autorelease drains an Objective-C pool each complete sweep.
"""
import argparse
import asyncio
import ctypes
import gc
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

async def run(args):
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from runtime.coreai_surfaces import SurfacePool
    objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
    objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(
        next(k for k in ComputeUnitKind.available_kinds() if str(k) == 'Neural Engine'))
    models, fns = [], []
    pool = SurfacePool(32)
    for i in range(48):
        name = f'flashnext_qsa_L{i}_s32' if i % 4 == 3 else f'flashnext_pure_step_L{i}'
        m = await AIModel.load(ROOT / 'artifacts/coreai' / (name + '.aimodel'), specialization_options=spec)
        models.append(m)
        fns.append(m.load_function('main'))
        if i % 4 != 3:
            pool.add_gdn(i)
    print('loaded 48 assets', flush=True)
    for step in range(args.steps):
        token_pool = objc.objc_autoreleasePoolPush() if args.autorelease else None
        for i, fn in enumerate(fns):
            if i % 4 == 3:
                out = await fn(pool.qsa_feeds())
                for v in out.values():
                    np.array(v.numpy(), copy=True)
                del v
            else:
                out = await fn(pool.pure_feeds(i))
                pool.accept_connected(i, out)
                pool.take_pure(out)
            out.clear()
            del out
        gc.collect()
        if token_pool:
            objc.objc_autoreleasePoolPop(token_pool)
        if step % 8 == 7:
            print(f'PASS sweeps={step+1} submits={(step+1)*48}', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--autorelease', action='store_true')
    asyncio.run(run(ap.parse_args()))
