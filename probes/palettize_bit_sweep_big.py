#!/usr/bin/env python3
"""Bit-width sweep on an ANE-RESIDENT model (compute-plan verified)."""
import numpy as np, torch, torch.nn as nn, shutil, os, json, time
import coremltools as ct
from coremltools.optimize.coreml import (OptimizationConfig,
                                         palettize_weights, OpPalettizerConfig)
from coremltools.models.compute_plan import MLComputePlan

CCH, H, W, K, LAYERS = 64, 128, 128, 3, 5
rng = np.random.default_rng(42)
ws = [ (rng.normal(0, 0.05, (CCH, CCH if i else 3, K, K))).astype(np.float32)
       for i in range(LAYERS) ]
x0 = rng.normal(0, 0.2, (1, 3, H, W)).astype(np.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        first = nn.Conv2d(3, CCH, (K, K), padding=1, bias=False)
        with torch.no_grad():
            first.weight.copy_(torch.from_numpy(
                rng.normal(0, 0.05, (CCH, 3, K, K)).astype(np.float32)) / np.sqrt(3))
        self.convs.append(first)
        for i in range(LAYERS - 1):
            c = nn.Conv2d(CCH, CCH, (K, K), padding=1, bias=False)
            with torch.no_grad():  # residual-friendly scaling
                c.weight.copy_(torch.from_numpy(ws[i+1]) * 0.05)
            self.convs.append(c)
    def forward(self, x):
        x = self.convs[0](x)
        for c in self.convs[1:]: x = x + c(x)   # residual keeps activations O(1)
        return x

traced = torch.jit.trace(Net().eval(), torch.from_numpy(x0))
OUT = "/tmp/ane-palettized"
base_pkg = f"{OUT}/big_base.mlpackage"
if os.path.exists(base_pkg): shutil.rmtree(base_pkg)
base = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=tuple(x0.shape))],
                  minimum_deployment_target=ct.target.iOS18,
                  convert_to="mlprogram")
base.save(base_pkg)

def load(pkg, units): return ct.models.MLModel(pkg, compute_units=units)

# verify ANE residency of the BASELINE first
compiled = base_pkg.replace(".mlpackage", ".mlmodelc")
os.system(f"xcrun coremlcompiler compile {base_pkg} {compiled} >/dev/null 2>&1")
plan = MLComputePlan.load_from_path(f"{compiled}/{os.path.basename(compiled)}")
fn = list(plan.model_structure.program.functions.values())[0]
devs = set()
for op in fn.block.operations:
    if op.operator_name.endswith("conv"):
        d = plan.get_compute_device_usage_for_mlprogram_operation(op)
        devs.add(type(d.preferred_compute_device).__name__)
print("baseline preferred devices:", devs)
assert any("NeuralEngine" in d for d in devs), "model not ANE-resident - increase size"

xbig = {"x": x0}
y_ref_cpu = list(load(base_pkg, ct.ComputeUnit.CPU_ONLY).predict(xbig).values())[0]
y_ref_ane = list(load(base_pkg, ct.ComputeUnit.CPU_AND_NE).predict(xbig).values())[0]
den = np.max(np.abs(y_ref_cpu)) + 1e-9
print(f"fp16 baseline: ANE-vs-CPU exec_err="
      f"{np.max(np.abs(y_ref_ane-y_ref_cpu))/den:.5f}")

res = {}
for nbits in (4, 2, 1):
    # NOTE: enable_per_channel_scale emits constexpr_blockwise_shift_scale
    # which crashes Apple's own MPSGraph verifier on this OS - avoided here.
    cfg = OptimizationConfig(global_config=OpPalettizerConfig(
        mode="kmeans", nbits=nbits, weight_threshold=1))
    try:
        comp = palettize_weights(base, cfg)
    except Exception as e:
        print(f"p{nbits}: palettize failed {e}"); continue
    pkg = f"{OUT}/big_p{nbits}.mlpackage"
    if os.path.exists(pkg): shutil.rmtree(pkg)
    comp.save(pkg)
    y_cpu = list(load(pkg, ct.ComputeUnit.CPU_ONLY).predict(xbig).values())[0]
    y_ane = list(load(pkg, ct.ComputeUnit.CPU_AND_NE).predict(xbig).values())[0]
    qerr = float(np.max(np.abs(y_cpu - y_ref_cpu)) / den)
    eerr = float(np.max(np.abs(y_ane - y_cpu)) / den)
    res[nbits] = {"quant_err": qerr, "exec_err": eerr}
    print(f"p{nbits}{'(pcs)' if nbits<=2 else '    '}: quant_err={qerr:.4f} "
          f"exec_err={eerr:.5f}")
json.dump(res, open(f"{OUT}/big_sweep.json","w"), indent=1)
