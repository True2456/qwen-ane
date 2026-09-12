"""Compare resident GPU Q4 routing with the CPU dequant/GEMV reference.

Run in an MLX environment; uses real layer-0 weights and changing top-10
sets, includes NumPy input/output synchronization, excludes model loading.
"""
import sys
import time
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.flashnext_mlx_moe import ResidentMoe
from runtime.expert_bank import Mlx4ExpertBank, H, I


def main():
    bank = Mlx4ExpertBank()
    shared = (*bank.shared_fp32(0), np.zeros((1, H), np.float32))
    t = time.perf_counter()
    model = ResidentMoe(0, shared)
    print(f"loaded {model.nbytes / 1e9:.3f} GB in {time.perf_counter()-t:.2f}s", flush=True)
    rng = np.random.default_rng(914)
    x = rng.normal(0, .1, H).astype(np.float32)
    gu = np.empty((2*I, H), np.float32)
    dn = np.empty((H, I), np.float32)
    errors, times = [], []
    for step in range(24):
        ids = rng.choice(512, 10, replace=False)
        scores = rng.uniform(.01, 1, 10).astype(np.float32)
        scores /= scores.sum()
        if step < 3:
            want = np.zeros(H, np.float32)
            t = time.perf_counter()
            for e, s in zip(ids, scores):
                bank.dequant_expert(0, int(e), gu, dn)
                gate, up = np.split(gu @ x, 2)
                want += (dn @ (gate / (1+np.exp(-gate)) * up)) * s
            cpu_ms = (time.perf_counter()-t)*1e3
            got = model.routed(x, ids, scores)
            mx.eval(got)
            got = np.array(got.astype(mx.float32)).ravel()
            rel = np.linalg.norm(got-want)/np.linalg.norm(want)
            errors.append(float(rel))
            print(f"step {step}: routed rel={rel:.6g}, CPU dequant+GEMV={cpu_ms:.2f}ms", flush=True)
            assert rel < .002, (step, rel)
        full, ms = model.apply(x, ids, scores)
        if step < 3:
            g, u, d, sgate = shared
            gate = x @ g.T
            shared_ref = ((gate / (1+np.exp(-gate))) * (x @ u.T)) @ d.T
            full_ref = want + shared_ref / (1+np.exp(-(x @ sgate.T)))
            rel_full = np.linalg.norm(full.ravel()-full_ref)/np.linalg.norm(full_ref)
            assert rel_full < .002, (step, "shared", rel_full)
        if step >= 4:
            times.append(ms)
    print(f"resident GPU MoE + shared + host boundary: median={np.median(times):.3f}ms "
          f"p95={np.percentile(times,95):.3f}ms; max routed rel={max(errors):.6g}")


if __name__ == "__main__":
    main()
