"""Saturation test: drive the ANE flat out with minimal host work.

The A/B in section 32 ran the ANE at ~53% duty behind a Python dispatch loop,
so its 0.72 W average says little about what the part draws when busy. This
loops one baked 27B MLP back-to-back for a fixed wall-clock window at several
program widths, reporting achieved TFLOP/s so it can be paired with a
concurrent powermetrics sample.
"""
import os, sys, time, json, struct, numpy as np
sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import importlib.util
spec = importlib.util.spec_from_file_location("ane_serve", os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "ane_serve.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

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

eng = AneEngine()
SECS = float(os.environ.get("SECS", "12"))
FL = 3 * 2 * g.shape[1] * g.shape[0]      # flops per token through the MLP

print(f"# MLP gate{g.shape} down{d.shape}   {SECS:.0f}s per width")
print(f"{'S':>6} {'dispatch/s':>11} {'ms each':>9} {'TFLOP/s':>9}   window")
for S in (32, 128, 512):
    b = m.AneDenseMLP(eng, E, _iosurface_view, g, u, d, S, 4)
    xf = (np.random.randn(S, b.H) * 0.02).astype(np.float32)
    for _ in range(3): b(xf)
    t0 = time.perf_counter(); n = 0
    start_wall = time.time()
    while time.perf_counter() - t0 < SECS:
        b(xf); n += 1
    el = time.perf_counter() - t0
    end_wall = time.time()
    print(f"{S:>6} {n/el:>11.1f} {el/n*1e3:>9.3f} {FL*S*n/el/1e12:>9.2f}   "
          f"{start_wall:.3f}-{end_wall:.3f}", flush=True)
    del b
