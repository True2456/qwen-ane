"""Real layer-0 top-10 routed SwiGLU, scalar vs vector Q4 CPU kernels."""
import sys
import os
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime.expert_bank import Mlx4ExpertBank, _q4lib, H

bank = Mlx4ExpertBank()
bank.load_ram([0])
lib = _q4lib()
vector, scalar = lib.affine_q4_gemv, lib.affine_q4_gemv_scalar
scalar.argtypes, scalar.restype = vector.argtypes, None
rng = np.random.default_rng(917)
x = rng.normal(0,.1,H).astype(np.float32)
sets = [rng.choice(512,10,replace=False) for _ in range(12)]
scores = np.full(10,.1,np.float32)
results = []
for label, fn in (("scalar",scalar),("vector",vector),("parallel",vector)):
    os.environ["FLASHNEXT_Q4_PARALLEL"] = "1" if label == "parallel" else "0"
    lib.affine_q4_gemv = fn
    outputs, times = [], []
    for ids in sets:
        bank.swiglu_routed(0,ids,x,scores)
        t = time.perf_counter()
        outputs.append(bank.swiglu_routed(0,ids,x,scores))
        times.append((time.perf_counter()-t)*1e3)
    results.append(outputs)
    print(f"{label} real top-10 SwiGLU: median={np.median(times):.3f}ms p95={np.percentile(times,95):.3f}ms",flush=True)
for arm in results[1:]:
    for ref, got in zip(results[0], arm):
        rel=np.linalg.norm(ref-got)/np.linalg.norm(ref)
        assert rel<1e-5,rel
print("PASS: all 12 changing expert sets agree (relative L2 < 1e-5)")
