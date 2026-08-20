#!/usr/bin/env python3
"""Validate overflow-safe RMSNorm for three independent ANE decode lanes."""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runtime.q38_ane_engine import AneEngine, _iosurface_view, _BUILD_INFO


C, S, ACTIVE = 5120, 32, 3
eng = AneEngine()
rng = np.random.default_rng(7)
x = (rng.standard_normal((C, ACTIVE)) * np.array([1.0, 9.0, 45.0])).astype(np.float32)
x[0] = (1776.0, -428.0, 52.0)
w = (1.0 + rng.standard_normal(C) * 0.1).astype(np.float32)


def lane_block(lane: int) -> str:
    p = f"l{lane}_"
    return f'''    tensor<fp16, [1, {C}, 1, 1]> {p}x = slice_by_index(begin=tensor<int32, [4]>([0,0,0,{lane}]), end=tensor<int32, [4]>([1,{C},1,{lane+1}]), x=x)[name=string("{p}x")];
    tensor<fp16, [1, 1, 1, {C}]> {p}f = reshape(shape=tensor<int32, [4]>([1,1,1,{C}]), x={p}x)[name=string("{p}f")];
    tensor<fp16, [1, 1, 1, {C}]> {p}a = abs(x={p}f)[name=string("{p}a")];
    tensor<fp16, [1, 1, 1, 1]> {p}m0 = reduce_max(axes=tensor<int32, [1]>([3]), keep_dims=bool(true), x={p}a)[name=string("{p}m0")];
    tensor<fp16, [1, 1, 1, 1]> {p}m = maximum(x={p}m0, y=fp16(0x1.064p-10))[name=string("{p}m")];
    tensor<fp16, [1, 1, 1, {C}]> {p}s0 = real_div(x={p}f, y={p}m)[name=string("{p}s0")];
    tensor<fp16, [1, 1, 1, {C}]> {p}s = mul(x={p}s0, y=fp16(0x1p+6))[name=string("{p}s")];
    tensor<fp16, [1, 1, 1, {C}]> {p}sq = mul(x={p}s, y={p}s)[name=string("{p}sq")];
    tensor<fp16, [1, 1, 1, 1]> {p}ms = reduce_mean(axes=tensor<int32, [1]>([3]), keep_dims=bool(true), x={p}sq)[name=string("{p}ms")];
    tensor<fp16, [1, 1, 1, 1]> {p}er0 = real_div(x=fp16(0x1.064p-10), y={p}m)[name=string("{p}er0")];
    tensor<fp16, [1, 1, 1, 1]> {p}er = mul(x={p}er0, y=fp16(0x1p+6))[name=string("{p}er")];
    tensor<fp16, [1, 1, 1, 1]> {p}e2 = mul(x={p}er, y={p}er)[name=string("{p}e2")];
    tensor<fp16, [1, 1, 1, 1]> {p}mse = add(x={p}ms, y={p}e2)[name=string("{p}mse")];
    tensor<fp16, [1, 1, 1, 1]> {p}sd = sqrt(x={p}mse)[name=string("{p}sd")];
    tensor<fp16, [1, 1, 1, 1]> {p}d0 = mul(x={p}m, y={p}sd)[name=string("{p}d0")];
    tensor<fp16, [1, 1, 1, 1]> {p}d = mul(x={p}d0, y=fp16(0x1p-6))[name=string("{p}d")];
    tensor<fp16, [1, {C}, 1, 1]> {p}u = real_div(x={p}x, y={p}d)[name=string("{p}u")];
    tensor<fp16, [1, {C}, 1, 1]> {p}n = mul(x={p}u, y=nw)[name=string("{p}n")];
    tensor<fp16, [1, {C}, 1, {S}]> {p}p = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,0,0,0,{lane},{S-lane-1}]), x={p}n)[name=string("{p}p")];'''


body = "\n".join(lane_block(i) for i in range(ACTIVE))
body += '\n    tensor<fp16, [1, %d, 1, %d]> a01 = add(x=l0_p, y=l1_p)[name=string("a01")];' % (C, S)
body += '\n    tensor<fp16, [1, %d, 1, %d]> y = add(x=a01, y=l2_p)[name=string("y")];' % (C, S)
mil = f'''program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<fp16, [1, {C}, 1, 1]> nw = const()[name=string("nw"), val=tensor<fp16, [1, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/norm.bin"), offset=uint64(64)))];
{body}
  }} -> (y);
}}
// pure_ane_rmsnorm_active3
'''
capture = io.StringIO()
with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
    prog = eng.compile_multiproc(mil, {"norm.bin": w.astype(np.float16).tobytes()}, C, C, S)
if prog is None:
    raise RuntimeError("compile failed\n" + "\n".join(capture.getvalue().splitlines()[-12:]))
eng._ensure_io(prog)
with _iosurface_view(prog._in_surf, (C, S), np.float16) as dst:
    dst[:] = 0
    dst[:, :ACTIVE] = x.astype(np.float16)
if not eng.submit(prog):
    raise RuntimeError("submission failed")
with _iosurface_view(prog._out_surf, (C, S), np.float16) as src:
    got = np.array(src[:, :ACTIVE], np.float32)
ref = x / np.sqrt(np.mean(x*x, axis=0, keepdims=True) + 1e-6) * w[:, None]
for lane in range(ACTIVE):
    rel = np.max(np.abs(got[:, lane] - ref[:, lane])) / np.max(np.abs(ref[:, lane]))
    print(f"lane={lane} max={np.max(np.abs(x[:, lane])):.1f} rel={rel:.6g}")
    if not np.isfinite(got[:, lane]).all() or rel > 3e-3:
        raise RuntimeError(f"lane {lane} RMSNorm mismatch {rel}")
print("ANE_RMSNORM_LANES=PASS active=3")
