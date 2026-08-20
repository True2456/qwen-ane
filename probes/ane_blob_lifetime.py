"""Does a loaded ANE program still work after its on-disk weights are deleted?

Each program writes its blobs to a temp dir that stays on disk for the process
lifetime. If the ANE copies them into its own memory at load, the page cache
holding those files is pure waste -- deleting them would roughly halve the
memory a build consumes.
"""
import os, sys, shutil, subprocess, numpy as np, importlib.util
sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
spec = importlib.util.spec_from_file_location("ane_serve",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "ane_serve.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

def free_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for l in out.splitlines():
        if "Pages free" in l:
            return int(l.split(":")[1].strip().rstrip(".")) * 16384 / 1e9
    return 0.0

H, I = 5120, 17408
rng = np.random.default_rng(0)
g = (rng.standard_normal((I, H)) * 0.02).astype(np.float32)
u = (rng.standard_normal((I, H)) * 0.02).astype(np.float32)
d = (rng.standard_normal((H, I)) * 0.02).astype(np.float32)
eng = AneEngine()
f0 = free_gb()
b = m.AneDenseMLP(eng, E, _iosurface_view, g, u, d, 32, 4)
f1 = free_gb()
print(f"  blobs on disk      {b.nbytes/1e9:.2f} GB")
print(f"  free before/after  {f0:.1f} -> {f1:.1f} GB   (consumed {f0-f1:.2f} GB, "
      f"{(f0-f1)/(b.nbytes/1e9):.1f}x the blob size)")

x = (rng.standard_normal((32, H)) * 0.02).astype(np.float32)
y1 = b(x)
loc = E._desc(E._msg(b.prog.model, "localModelPath"))
print(f"  model dir          {loc}")
sz = 0
if loc and os.path.isdir(loc):
    for r, _, fs in os.walk(loc):
        for fn in fs:
            sz += os.path.getsize(os.path.join(r, fn))
print(f"  on-disk size       {sz/1e9:.2f} GB")
shutil.rmtree(loc, ignore_errors=True)
print(f"  deleted            {not os.path.exists(loc)}")
f2 = free_gb()
y2 = b(x)
same = np.abs(y1 - y2).max()
print(f"  free after delete  {f2:.1f} GB")
print(f"  output after delete: max|diff| = {same:.6f}  -> "
      f"{'STILL CORRECT' if same < 1e-3 else 'BROKEN'}")
