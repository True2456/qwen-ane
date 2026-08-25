#!/usr/bin/env python3
"""P21: batch-export INT4 tails for ALL 64 layers of Qwen3.8-27B.

Loops scripts/export_real_tail.py logic per layer. Each bundle ~150MB;
total ~9.6GB unified-memory footprint when all are resident.

Env: LAYERS="0,1,2" subset, GDN_BITS/GDN_GROUP/GDN_S as usual.
Per-layer runtime benchmarking is disabled by default here because CoreAI's
Python specialization cache can consume several GB for every exported layer.
Set GDN_SKIP_BENCH=0 explicitly when a runtime benchmark is required.
"""
import asyncio, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = Path.home() / ".rindi/aimodels"

def export_layer(lyr: str) -> tuple:
    t0 = time.perf_counter()
    env = dict(os.environ)
    env.setdefault("GDN_SKIP_BENCH", "1")
    r = subprocess.run(
        [sys.executable, str(HERE / "export_real_tail.py"), lyr],
        env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    return int(lyr), r.returncode, dt, r.stdout, r.stderr

def main():
    layers = (os.environ.get("LAYERS", "").split(",")
              if os.environ.get("LAYERS") else [str(i) for i in range(64)])
    workers = int(os.environ.get("WORKERS", "8"))
    print(f"Exporting {len(layers)} layers in parallel with {workers} workers...")
    t0 = time.perf_counter()
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(export_layer, lyr): lyr for lyr in layers}
        completed = 0
        for f in as_completed(futures):
            lyr, rc, dt, out, err = f.result()
            completed += 1
            if rc == 0:
                print(f"[{completed}/{len(layers)}] Layer {lyr} OK ({dt:.1f}s)")
            else:
                print(f"[{completed}/{len(layers)}] Layer {lyr} FAILED (rc={rc}) in {dt:.1f}s")
                print(err[-300:] if err else out[-300:])
                failed.append(lyr)
    dt = time.perf_counter() - t0
    total = sum(f.stat().st_size
                for b in OUT_DIR.glob("qwen38_27b_tail_*.aimodel") if b.is_dir()
                for f in b.rglob("*") if f.is_file()) / 1e9
    print(f"\nAll done in {dt/60:.1f} min ({dt:.1f}s); total bundle bytes ~{total:.2f} GB")
    if failed:
        print(f"Failed layers: {failed}")
        return 1
    return 0

if __name__ == "__main__":
    main()
