#!/usr/bin/env python3
"""Did macOS 27.0 (26A428) open compiler/Exclave surfaces that beta blocked?

Each risky call is a child process. Does not try SIP/firmware/unsigned HWX.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

FRAMEWORK = "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler"


def _run_child(tag: str, fn_src: str, timeout: float = 30.0) -> None:
    script = f"""
import ctypes, sys
{fn_src}
"""
    p = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (p.stdout + p.stderr).strip().replace("\n", " | ")[:500]
    state = "ok" if p.returncode == 0 else f"exit {p.returncode}"
    if p.returncode < 0:
        state = f"SIGNAL {-p.returncode}"
    print(f"  [{tag}] {state}  {out}")


def probe_create_model_dictionary() -> None:
    print("\n== _ANECCreateModelDictionary (was NULL on (0,0)) ==")
    cases = {
        "0args": r"""
lib = ctypes.CDLL("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler")
fn = getattr(lib, "_ANECCreateModelDictionary")
fn.restype = ctypes.c_void_p
fn.argtypes = []
r = fn()
print(f"ptr={r}")
""",
        "1null": r"""
lib = ctypes.CDLL("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler")
fn = getattr(lib, "_ANECCreateModelDictionary")
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_void_p]
r = fn(None)
print(f"ptr={r}")
""",
        "2null": r"""
lib = ctypes.CDLL("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler")
fn = getattr(lib, "_ANECCreateModelDictionary")
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
r = fn(None, None)
print(f"ptr={r}")
""",
        "h17_null": r"""
import ctypes
from ctypes import util
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
cf.CFStringCreateWithCString.restype = ctypes.c_void_p
cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
arch = cf.CFStringCreateWithCString(None, b"h17", 0x08000100)
lib = ctypes.CDLL("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler")
fn = getattr(lib, "_ANECCreateModelDictionary")
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
r = fn(arch, None)
print(f"ptr={r}")
""",
        "3null": r"""
lib = ctypes.CDLL("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler")
fn = getattr(lib, "_ANECCreateModelDictionary")
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
r = fn(None, None, None)
print(f"ptr={r}")
""",
    }
    for tag, src in cases.items():
        _run_child(tag, src)


def probe_mutable_weight() -> None:
    print("\n== _ANECGetMutableWeight / Info / OperationInfo (NULL args) ==")
    for name, nargs in [
        ("_ANECGetMutableWeight", 1),
        ("_ANECGetMutableWeight", 2),
        ("_ANECGetMutableWeightInfo", 1),
        ("_ANECGetMutableWeightInfo", 2),
        ("_ANECGetMutableOperationInfo", 1),
        ("_ANECGetMutableOperationInfo", 2),
        ("_ANECValidateMutableProcedureInfo", 1),
        ("_ANECValidateMutableProcedureInfo", 2),
    ]:
        args = ", ".join(["None"] * nargs)
        src = f"""
import ctypes
lib = ctypes.CDLL("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler")
fn = getattr(lib, "{name}")
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_void_p] * {nargs}
r = fn({args})
print(f"ptr={{r}}")
"""
        _run_child(f"{name}/{nargs}", src)


def probe_coreai_ane_linear() -> None:
    print("\n== Core AI tiny Linear → Neural Engine (verifyBundle / Exclave) ==")
    src = r"""
import asyncio, os, tempfile, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from coreai_torch import TorchConverter, get_decomp_table
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(32, 32, 1, bias=False)
    def forward(self, x):
        return self.c(x)

m = M().eval().half()
x = torch.randn(1, 32, 1, 32, dtype=torch.float16)
ep = torch.export.export(m, args=(x,)).run_decompositions(get_decomp_table())
prog = TorchConverter().add_exported_program(ep, input_names=["x"], output_names=["y"]).to_coreai()
prog.optimize()
tmp = Path(tempfile.mkdtemp(prefix="gm_lin_"))
out = tmp / "lin.aimodel"
prog.save_asset(out)
kinds = {str(k): k for k in ComputeUnitKind.available_kinds()}
ane = kinds.get("Neural Engine")
if ane is None:
    print("no Neural Engine kind")
    raise SystemExit(0)
spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
async def go():
    t0 = time.perf_counter()
    mm = await AIModel.load(str(out), specialization_options=spec)
    fn = mm.load_function("main")
    y = await fn({"x": NDArray(x.numpy())})
    got = np.asarray(y["y"].numpy())
    print(f"ANE load+eval {(time.perf_counter()-t0)*1e3:.1f} ms  y{got.shape} finite={np.isfinite(got).all()}")
asyncio.run(go())
"""
    _run_child("tiny-conv-ane", src, timeout=120)


def probe_gathermm_ane() -> None:
    print("\n== GatherMM tiny E=8 on Neural Engine (subprocess; ANE abort kills child) ==")
    src = r"""
import asyncio, tempfile, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from coreai_torch import TorchConverter, get_decomp_table
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
from coreai_torch.composite_ops import GatherMM
from coreai_torch import ExternalizeSpec

class OneSwitch(nn.Module):
    def __init__(self):
        super().__init__()
        self.gather_mm = GatherMM(num_batch_axes=1)
        self.weight = nn.Parameter(torch.randn(1, 8, 32, 64, dtype=torch.float16))
    def forward(self, x, indices):
        wt = self.weight.transpose(-1, -2)
        return self.gather_mm(x, wt, rhs_indices=indices)

m = OneSwitch().eval().half()
x = torch.randn(1, 1, 1, 64, dtype=torch.float16)
ids = torch.randint(0, 8, (1, 2), dtype=torch.int32).to(torch.uint16)
prog = TorchConverter().add_pytorch_module(
    m, export_fn=lambda module: torch.export.export(module, args=(x, ids)).run_decompositions(get_decomp_table()),
    externalize_modules=[ExternalizeSpec(GatherMM, "gather_mm", ["num_batch_axes"])],
    input_names=["x","ids"], output_names=["y"]).to_coreai()
prog.optimize()
tmp = Path(tempfile.mkdtemp(prefix="gm_gmm_"))
out = tmp / "gmm.aimodel"
prog.save_asset(out)
kinds = {str(k): k for k in ComputeUnitKind.available_kinds()}
ane = kinds["Neural Engine"]
spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
async def go():
    t0 = time.perf_counter()
    mm = await AIModel.load(str(out), specialization_options=spec)
    fn = mm.load_function("main")
    feeds = {"x": NDArray(x.numpy()), "ids": NDArray(ids.numpy())}
    y = await fn(feeds)
    got = np.asarray(y["y"].numpy())
    for _ in range(3):
        await fn(feeds)
    t1 = time.perf_counter()
    n = 10
    for _ in range(n):
        await fn(feeds)
    ms = (time.perf_counter()-t1)/n*1e3
    print(f"ANE-preferred GatherMM (placement unverified) load {(time.perf_counter()-t0):.2f}s  y{got.shape} {ms:.2f} ms finite={np.isfinite(got).all()}")
asyncio.run(go())
"""
    _run_child("gathermm-ane-tiny", src, timeout=180)


def probe_espresso_gather_conv() -> None:
    print("\n== espresso gather_nd + inner_product fused (compile only) ==")
    src = r"""
import json, os, sys
from pathlib import Path
os.environ["Q38_ANE_REUSE_COMPILED"] = "0"
sys.path.insert(0, str(Path("/Users/true/Desktop/LLM - Reap/ane-port")))
import coremltools as ct
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from runtime.ane_lookup import _ane_compile_espresso

# table [8, 32, 64] gather axis0 k=2 -> [2,32,64]; flatten isn't in NN builder easily.
# Try gather then inner_product of x[64] with gathered rows... inner_product wants 2D.
builder = NeuralNetworkBuilder(
    [("table", datatypes.Array(8, 64)), ("ids", datatypes.Array(2)), ("x", datatypes.Array(1, 64))],
    [("y", None)],
    disable_rank5_shape_mapping=True,
)
builder.add_gather(name="g", input_names=["table","ids"], output_name="w", axis=0)
builder.add_inner_product(
    name="ip",
    W=None,
    b=None,
    input_channels=64,
    output_channels=2,
    has_bias=False,
    input_name="x",
    output_name="y",
)
# That inner_product still uses const W — not gathered. Check whether add_inner_product
# can take gathered W as input: NeuralNetwork inner_product weight is baked.
# Alternative: batched matmul via add_batched_mat_mul
print("nn inner_product cannot bind live W; trying batched_mat_mul")
"""
    # Real fused graph via batched matmul if available
    src = r"""
import json, os, sys
from pathlib import Path
os.environ["Q38_ANE_REUSE_COMPILED"] = "0"
sys.path.insert(0, str(Path("/Users/true/Desktop/LLM - Reap/ane-port")))
import coremltools as ct
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from runtime.ane_lookup import _ane_compile_espresso

builder = NeuralNetworkBuilder(
    [("table", datatypes.Array(8, 32, 64)), ("ids", datatypes.Array(2)), ("x", datatypes.Array(1, 1, 64))],
    [("y", None)],
    disable_rank5_shape_mapping=True,
)
builder.add_gather(name="g", input_names=["table","ids"], output_name="w", axis=0)
# w: [2,32,64], x: [1,1,64] -> batched matmul x @ w^T per expert? 
# add_batched_mat_mul(x, w) if ranks match.
try:
    builder.add_batched_mat_mul(name="bmm", input_names=["w","x"], output_name="y", transpose_a=False, transpose_b=True)
except Exception as e:
    print(f"add_batched_mat_mul failed: {type(e).__name__}: {e}")
    raise
spec = builder.spec
spec.description.input[1].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32
ml = ct.models.MLModel(spec, compute_units=ct.ComputeUnit.CPU_AND_NE)
root = Path(ml.get_compiled_model_path())
net = (root / "model.espresso.net").read_text()
shape = json.loads((root / "model.espresso.shape").read_text())
weights = (root / "model.espresso.weights").read_bytes()
print(f"espresso authored layers={[l.get('type') for l in json.loads(net).get('layers', json.loads(net) if False else [])][:12]}")
# shape file is the oracle if json layers fail
print("shape", {k: shape[k] for k in list(shape)[:12]})
_ane_compile_espresso(net, shape, weights)
print("ANE compile+load OK")
"""
    _run_child("gather+bmm", src, timeout=120)


def main() -> int:
    import platform
    print(f"python {sys.version.split()[0]}  mac {platform.mac_ver()[0]}")
    print(f"exe {sys.executable}")
    probe_create_model_dictionary()
    probe_mutable_weight()
    probe_coreai_ane_linear()
    probe_gathermm_ane()
    probe_espresso_gather_conv()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
