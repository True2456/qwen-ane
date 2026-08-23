#!/usr/bin/env python3
"""Offline (build-time) palettized-conv .mlmodelc generator.

Produces artifacts that carry SUB-FP16 packed weights into the ANE via the
protobuf/mlprogram path - the path text-MIL cannot express. Run once; the
resulting packages are loaded natively at inference (no Python).
"""
import os, shutil, numpy as np, torch, torch.nn as nn
import coremltools as ct
from coremltools.optimize.coreml import (OptimizationConfig, palettize_weights,
                                         OpPalettizerConfig)

C, S, K = 64, 32, 4
OUT = "/tmp/ane-palettized"
rng = np.random.default_rng(42)
w_np = rng.normal(0, 0.15, (C, 1, K, 1)).astype(np.float32)  # torch Conv2d layout [C,1,K,1]
b_np = np.zeros(C, dtype=np.float32)
np.savez("/tmp/ane-palett-ref.npz", w=w_np, C=C, S=S, K=K)

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, C, (K, 1), bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(w_np))
    def forward(self, x):
        return self.conv(x)

model = Conv().eval()
traced = torch.jit.trace(model, torch.from_numpy(rng.normal(0, .2, (1, 1, S, 1)).astype(np.float32)))
mlmodel = ct.convert(traced,
                     inputs=[ct.TensorType(name="x", shape=(1, 1, S, 1))],
                     minimum_deployment_target=ct.target.iOS17,
                     convert_to="mlprogram")

for nbits in (4, 2):
    cfg = OptimizationConfig(global_config=OpPalettizerConfig(
        mode="kmeans", nbits=nbits, weight_threshold=1))
    comp = palettize_weights(mlmodel, cfg)
    pkg = f"{OUT}/conv_p{nbits}.mlpackage"
    if os.path.exists(pkg): shutil.rmtree(pkg)
    comp.save(pkg)
    print(f"saved {pkg}")
print("saved under", OUT)

# ---- emit native-side verification artifacts ----
import numpy as _np
x = _np.random.default_rng(7).normal(0, 0.2, (1, 1, S, 1)).astype(_np.float16)
x.astype(_np.float16).tofile("/tmp/ane-palettized/x_in.bin")
for nbits in (4, 2):
    m = ct.models.MLModel(f"{OUT}/conv_p{nbits}.mlpackage")
    y = m.predict({"x": x.astype(_np.float32)})["var_" + str(m.output_description.name if False else 0)] \
        if False else list(m.predict({"x": x.astype(_np.float32)}).values())[0]
    _np.asarray(y, dtype=_np.float16).tofile(f"{OUT}/y_ref_p{nbits}.bin")
    print(f"ref p{nbits}: out shape {y.shape}, sample {float(y.flatten()[0]):+.4f}")
print("native artifacts written")
