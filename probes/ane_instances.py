"""Does kANEFAneInstanceHint give real parallelism, or is there one engine?

Compiles the same 27B MLP twice under different instance hints, times each
alone, then runs both concurrently from two threads. Concurrent ~= max(t1,t2)
means two engines; concurrent ~= t1+t2 means one engine and the hint is advisory.
"""
import os, sys, time, json, struct, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import importlib.util
spec = importlib.util.spec_from_file_location("ane_serve",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "ane_serve.py"))
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
# AneDenseMLP does not expose instance_hint, so patch the engine default per build
import functools
orig_cm = eng.compile_multiproc
def cm_with_hint(hint):
    def f(*a, **k):
        k["instance_hint"] = hint
        return orig_cm(*a, **k)
    return f

blocks = []
for hint in (1, 2):
    eng.compile_multiproc = cm_with_hint(hint)
    b = m.AneDenseMLP(eng, E, _iosurface_view, g, u, d, 32, 4)
    blocks.append((hint, b))
    print(f"  built MLP with instance_hint={hint}")
eng.compile_multiproc = orig_cm

x = (np.random.randn(32, blocks[0][1].H) * 0.02).astype(np.float32)
def bench(b, n=40):
    for _ in range(3): b(x)
    t0 = time.perf_counter()
    for _ in range(n): b(x)
    return (time.perf_counter() - t0) / n * 1e3

t1 = bench(blocks[0][1]); t2 = bench(blocks[1][1])
print(f"\n  hint=1 alone            {t1:.3f} ms")
print(f"  hint=2 alone            {t2:.3f} ms")

pool = ThreadPoolExecutor(max_workers=2)
N = 40
for _ in range(3):
    list(pool.map(lambda b: b(x), [blocks[0][1], blocks[1][1]]))
t0 = time.perf_counter()
for _ in range(N):
    f1 = pool.submit(blocks[0][1], x)
    f2 = pool.submit(blocks[1][1], x)
    f1.result(); f2.result()
conc = (time.perf_counter() - t0) / N * 1e3
print(f"  both concurrently       {conc:.3f} ms   (per pair)")
print(f"  serialized would be     {t1+t2:.3f} ms")
print(f"  one engine would be     {max(t1,t2):.3f} ms if truly parallel")
print(f"\n  => {'PARALLEL' if conc < 0.7*(t1+t2) else 'SERIALIZED'} "
      f"(speedup {(t1+t2)/conc:.2f}x over serial)")
