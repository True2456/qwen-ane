"""GPU counterpart to ane_saturate.py: same 27B MLP, same widths, pinned.

Reports achieved TFLOP/s with wall-clock windows so a concurrent powermetrics
sample can be attributed. Runs bf16 and 4-bit so the ANE's int4 weights have a
like-for-like opponent on memory traffic as well as arithmetic.
"""
import os, sys, time, json, struct, numpy as np
import mlx.core as mx, mlx.nn as nn

M = os.environ.get("Q38_MODEL", os.path.expanduser(
    "~/.lmstudio/models/Qwen/Qwen3.8-27B"))
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
def load(name):
    with open(os.path.join(M, idx[name]), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n)); info = hdr[name]; o = info["data_offsets"]
        fh.seek(8 + n + o[0]); raw = fh.read(o[1]-o[0])
    a = np.frombuffer(raw, np.uint16).astype(np.uint32)
    return (a << 16).view(np.float32).reshape(info["shape"])
pre = "model.language_model.layers.0.mlp."
if pre+"gate_proj.weight" not in idx: pre = "model.layers.0.mlp."
g, u, d = (load(pre+k+"_proj.weight") for k in ("gate", "up", "down"))
I, H = g.shape
FL = 3 * 2 * H * I
SECS = float(os.environ.get("SECS", "12"))

G, U, D = (mx.array(w).astype(mx.bfloat16) for w in (g, u, d))
mx.eval(G, U, D)
qG, qU, qD = (mx.quantize(w, bits=4, group_size=64) for w in (G, U, D))
mx.eval(qG, qU, qD)

def f16(x):
    return (nn.silu(x @ G.T) * (x @ U.T)) @ D.T
def f4(x):
    a = nn.silu(mx.quantized_matmul(x, *qG, transpose=True, bits=4, group_size=64)) \
        * mx.quantized_matmul(x, *qU, transpose=True, bits=4, group_size=64)
    return mx.quantized_matmul(a, *qD, transpose=True, bits=4, group_size=64)

print(f"# MLP gate{g.shape} down{d.shape}   {SECS:.0f}s per config")
print(f"{'cfg':>10} {'S':>6} {'iters/s':>9} {'ms each':>9} {'TFLOP/s':>9}   window")
for tag, fn in (("bf16", f16), ("int4", f4)):
    for S in (32, 128, 512):
        x = mx.random.normal((S, H)).astype(mx.bfloat16); mx.eval(x)
        for _ in range(3): mx.eval(fn(x))
        t0 = time.perf_counter(); n = 0; sw = time.time()
        while time.perf_counter() - t0 < SECS:
            mx.eval(fn(x)); n += 1
        el = time.perf_counter() - t0; ew = time.time()
        print(f"{tag:>10} {S:>6} {n/el:>9.1f} {el/n*1e3:>9.3f} "
              f"{FL*S*n/el/1e12:>9.2f}   {sw:.3f}-{ew:.3f}", flush=True)
