"""Is ANE decode cost flat across the 32 padded lanes, on a REAL 27B MLP?

Decode pads T=1 up to the hardware's minimum width of 32, so each step computes
a full 32-token batch and discards 31/32 of the result. If cost is flat in T,
those lanes are free capacity -- and speculative decoding can fill them.
"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import importlib.util
spec = importlib.util.spec_from_file_location("ane_serve", os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "ane_serve.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

M = os.environ.get("Q38_MODEL", os.path.expanduser(
    "~/.lmstudio/models/Qwen/Qwen3.8-27B"))
import glob
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
def load(name):
    f = os.path.join(M, idx[name])
    import struct
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        info = hdr[name]; off = info["data_offsets"]
        fh.seek(8 + n + off[0]); raw = fh.read(off[1]-off[0])
    a = np.frombuffer(raw, np.uint16).astype(np.uint32)
    return (a << 16).view(np.float32).reshape(info["shape"])

pre = "model.language_model.layers.0.mlp."
try:
    g = load(pre+"gate_proj.weight"); u = load(pre+"up_proj.weight"); d = load(pre+"down_proj.weight")
except KeyError:
    pre = "model.layers.0.mlp."
    g = load(pre+"gate_proj.weight"); u = load(pre+"up_proj.weight"); d = load(pre+"down_proj.weight")
print(f"MLP shapes  gate{g.shape} down{d.shape}")

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

eng = AneEngine()
blk = m.AneDenseMLP(eng, E, _iosurface_view, g, u, d, 32, 4)
print(f"baked {blk.nbytes/1e9:.2f} GB int4, ANE width S={blk.S}")

# reference: the same MLP on the GPU, bf16
import mlx.core as mx, mlx.nn as nn
G, U, D = mx.array(g), mx.array(u), mx.array(d)
mx.eval(G, U, D)
def gpu(x):
    a = nn.silu(x @ G.T) * (x @ U.T)
    return a @ D.T

print(f"\n{'S':>6} {'ANE ms':>9} {'GPU ms':>9} {'ANE us/tok':>11} {'GPU us/tok':>11} {'ANE TFLOP/s':>12} {'speedup':>8}")
FL = 3 * 2 * blk.H * blk.I          # gate+up+down, 2 flops per MAC
for S in (32, 64, 128, 256, 512, 1024, 2048):
    try:
        b = m.AneDenseMLP(eng, E, _iosurface_view, g, u, d, S, 4)
    except Exception as ex:
        print(f"{S:>6}  compile failed: {type(ex).__name__}"); continue
    xf = np.random.randn(S, blk.H).astype(np.float32) * 0.02
    xm = mx.array(xf).astype(mx.bfloat16)
    for _ in range(2): b(xf)
    N = 20 if S <= 512 else 6
    t0 = time.perf_counter()
    for _ in range(N): b(xf)
    ane = (time.perf_counter()-t0)/N*1e3
    for _ in range(2): mx.eval(gpu(xm))
    t0 = time.perf_counter()
    for _ in range(N): mx.eval(gpu(xm))
    gms = (time.perf_counter()-t0)/N*1e3
    tf = FL * S / (ane*1e-3) / 1e12
    print(f"{S:>6} {ane:>9.3f} {gms:>9.3f} {ane/S*1e3:>11.1f} {gms/S*1e3:>11.1f} {tf:>12.2f} {gms/ane:>8.2f}x")
    del b
