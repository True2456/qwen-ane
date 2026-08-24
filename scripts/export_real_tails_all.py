#!/usr/bin/env python3
"""P21: batch-export INT4 tails for ALL 64 layers of Qwen3.8-27B.

Loops scripts/export_real_tail.py logic per layer. Each bundle ~150MB;
total ~9.6GB unified-memory footprint when all are resident.

Env: LAYERS="0,1,2" subset, GDN_BITS/GDN_GROUP/GDN_S as usual.
"""
import asyncio, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = Path.home() / ".rindi/aimodels"

def main():
    layers = (os.environ.get("LAYERS", "").split(",")
              if os.environ.get("LAYERS") else [str(i) for i in range(64)])
    t0 = time.perf_counter()
    for i, lyr in enumerate(layers):
        print(f"\n=== layer {lyr} ({i+1}/{len(layers)}) ===")
        env = dict(os.environ)
        r = subprocess.run(
            [sys.executable, str(HERE / "export_real_tail.py"), lyr],
            env=env)
        if r.returncode != 0:
            print(f"LAYER {lyr} FAILED (rc={r.returncode})")
            return r.returncode
    dt = time.perf_counter() - t0
    total = sum(f.stat().st_size
                for b in OUT_DIR.glob("qwen38_27b_tail_*.aimodel") if b.is_dir()
                for f in b.rglob("*") if f.is_file()) / 1e9
    print(f"\nall done in {dt/60:.1f} min; total bundle bytes ~{total:.2f} GB")

if __name__ == "__main__":
    main()
