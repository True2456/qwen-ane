#!/usr/bin/env python3
"""Bit-width sweep for palettized conv weights on Apple Neural Engine.

For nbits in {8(ref),4,2,1}: palettize the same conv, run on ANE via
CoreML (CPU_AND_NE), and measure:
  - exec_err : ANE output vs the SAME compressed model on CPU
               (= pure execution correctness, quant noise excluded)
  - quant_err: compressed-model output vs original fp16-weight output
               (= the coherence cost of the bits themselves)
"""
import numpy as np, torch, torch.nn as nn, shutil, os, json
import coremltools as ct
from coremltools.optimize.coreml import (OptimizationConfig,
                                         palettize_weights,
                                         OpPalettizerConfig)

C, S, K = 64, 32, 4
rng = np.random.default_rng(42)
w_np = rng.normal(0, 0.15, (C, 1, K, 1)).astype(np.float32)
x_np = rng.normal(0, 0.2, (1, 1, S, 1)).astype(np.float32)

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, C, (K, 1), bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(w_np))
    def forward(self, x): return self.conv(x)

traced = torch.jit.trace(Conv().eval(), torch.from_numpy(x_np))
base = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=(1,1,S,1))],
                  minimum_deployment_target=ct.target.iOS17,
                  convert_to="mlprogram")

_tmp_pkg = "/tmp/ane-palettized/_base.mlpackage"

def run_on(m, units):
    path = m
    if not isinstance(m, str):
        if os.path.exists(_tmp_pkg): shutil.rmtree(_tmp_pkg)
        m.save(_tmp_pkg); path = _tmp_pkg
    mm = ct.models.MLModel(path, compute_units=units)
    return list(mm.predict({"x": x_np}).values())[0]

out = {}
fp16_ane = run_on(base, ct.ComputeUnit.CPU_AND_NE)
fp16_cpu = run_on(base, ct.ComputeUnit.CPU_ONLY)
out["exec_err_fp16"] = float(np.max(np.abs(fp16_ane - fp16_cpu)) /
                             (np.max(np.abs(fp16_cpu)) + 1e-9))

for nbits in (4, 2, 1):
    for pcs in ((False,) if nbits > 1 else (False, True)):
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(
            mode="kmeans", nbits=nbits, weight_threshold=1,
            enable_per_channel_scale=pcs))
        comp = palettize_weights(base, cfg)
        pkg = f"/tmp/ane-palettized/sweep_p{nbits}{'_pcs' if pcs else ''}.mlpackage"
        if os.path.exists(pkg): shutil.rmtree(pkg)
        comp.save(pkg)
        y_cpu = run_on(pkg, ct.ComputeUnit.CPU_ONLY)     # compressed ground truth
        y_ane = run_on(pkg, ct.ComputeUnit.CPU_AND_NE)   # hardware execution
        tag = f"p{nbits}{'_pcs' if pcs else ''}"
        out[tag] = {
            "exec_err": float(np.max(np.abs(y_ane - y_cpu)) / (np.max(np.abs(y_cpu)) + 1e-9)),
            "quant_err": float(np.max(np.abs(y_cpu - fp16_cpu)) / (np.max(np.abs(fp16_cpu)) + 1e-9)),
            "sample_out": float(y_cpu.flatten()[0]),
            "ane_sample_out": float(y_ane.flatten()[0]),
        }
        print(f"{tag:8s} exec_err={out[tag]['exec_err']:.4f} "
              f"quant_err={out[tag]['quant_err']:.4f} "
              f"out0 cpu={out[tag]['sample_out']:+.4f} ane={out[tag]['ane_sample_out']:+.4f}")

print(f"fp16    exec_err={out['exec_err_fp16']:.5f}")
json.dump(out, open("/tmp/ane-palettized/bit_sweep.json", "w"), indent=1)
