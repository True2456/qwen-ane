"""Test Qwen3.8's real-shape causal depthwise conv1d on the ANE.

Qwen3.8 uses C=10240, kernel=4, groups=C.  Decode supplies three cached
samples plus the current projection.  A custom left pad keeps the IOSurface
width at 32; the integrated runtime can put the live window at the right edge
and select its final output lane.
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view


eng = AneEngine()
S, K = 32, 4


def run(C, activation="none"):
    rng = np.random.default_rng(2000 + C)
    w = rng.normal(0, 0.15, (C, 1, 1, K)).astype(np.float16)
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<fp16, [{C}, 1, 1, {K}]> w = const()[name=string("w"), val=tensor<fp16, [{C}, 1, 1, {K}]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<int32, [2]> strides = const()[name=string("strides"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [2]> dil = const()[name=string("dil"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pad = const()[name=string("pad"), val=tensor<int32, [4]>([0,0,{K-1},0])];
    tensor<fp16, [1, {C}, 1, {S}]> c = conv(dilations=dil, groups=int32({C}), pad=pad, pad_type=string("custom"), strides=strides, weight=w, x=x)[name=string("c")];
    {('tensor<fp16, [1, %d, 1, %d]> sg = sigmoid(x=c)[name=string("sg")];' % (C, S)) if activation == 'sigmoid' else ''}
    {('tensor<fp16, [1, %d, 1, %d]> nc = mul(x=c, y=fp16(-0x1p+0))[name=string("nc")];' % (C, S)) if activation == 'exp' else ''}
    {('tensor<fp16, [1, %d, 1, %d]> ex = exp(x=nc)[name=string("ex")];' % (C, S)) if activation == 'exp' else ''}
    {('tensor<fp16, [1, %d, 1, %d]> den = add(x=ex, y=fp16(0x1p+0))[name=string("den")];' % (C, S)) if activation == 'exp' else ''}
    tensor<fp16, [1, {C}, 1, {S}]> y = {('mul(x=c, y=sg)' if activation == 'sigmoid' else ('real_div(x=c, y=den)' if activation == 'exp' else 'mul(x=c, y=fp16(0x1p+0))'))}[name=string("y")];
  }} -> (y);
}}
// qwen38_gdn_depthwise_C{C}
'''
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        prog = eng.compile_multiproc(mil, {"w.bin": w.tobytes()}, C, C, S)
    if prog is None:
        detail = next((s for s in capture.getvalue().splitlines()
                       if "Error" in s or "FAILED" in s), "")
        print(f"  C={C:<5} COMPILE FAILED {detail[:120]}")
        return False

    x = rng.normal(0, 0.2, (C, S)).astype(np.float32)
    xp = np.pad(x, ((0, 0), (K - 1, 0)))
    ref = np.empty_like(x)
    wf = w.astype(np.float32).reshape(C, K)
    for t in range(S):
        ref[:, t] = np.sum(xp[:, t:t+K] * wf, axis=1)
    if activation != "none":
        ref = ref / (1.0 + np.exp(-ref))

    eng._ensure_io(prog)
    with _iosurface_view(prog._in_surf, (C, S), np.float16) as dst:
        dst[:] = x.astype(np.float16)
    for _ in range(3):
        if not eng.submit(prog):
            print(f"  C={C:<5} SUBMIT FAILED")
            return False
    n = 40
    t0 = time.perf_counter()
    for _ in range(n):
        eng.submit(prog)
    ms = (time.perf_counter() - t0) * 1e3 / n
    with _iosurface_view(prog._out_surf, (C, S), np.float16) as src:
        got = np.array(src, np.float32)
    abs_err = np.max(np.abs(got - ref))
    rel = abs_err / (np.max(np.abs(ref)) + 1e-9)
    if rel >= 5e-3:
        # A diagnostic for undocumented kernel/pad ordering.  It makes a
        # wrong-but-close result actionable instead of merely reporting FAIL.
        candidates = {}
        for side, pad_width in (("left", (K-1, 0)), ("right", (0, K-1))):
            xx = np.pad(x, ((0, 0), pad_width))
            for order, ww in (("same", wf), ("flipped", wf[:, ::-1])):
                rr = np.empty_like(x)
                for t in range(S):
                    rr[:, t] = np.sum(xx[:, t:t+K] * ww, axis=1)
                if activation != "none":
                    rr = rr / (1.0 + np.exp(-rr))
                candidates[f"{side}/{order}"] = np.max(np.abs(got-rr)) / (np.max(np.abs(rr))+1e-9)
        best = min(candidates, key=candidates.get)
        per_col = np.max(np.abs(got - ref), axis=0)
        worst_col = int(np.argmax(per_col))
        interior = np.max(np.abs(got[:, K-1:] - ref[:, K-1:])) / (np.max(np.abs(ref[:, K-1:])) + 1e-9)
        print(f"    closest convention: {best} rel={candidates[best]:.4g}; "
              f"interior rel={interior:.4g}, worst lane={worst_col}")
    ok = np.isfinite(got).all() and rel < 5e-3
    label = "conv" if activation == "none" else f"conv+{activation}"
    print(f"  C={C:<5} {label:<9} {'OK' if ok else 'WRONG':<5} rel={rel:.4g} "
          f"abs={abs_err:.4g} {ms:.3f} ms")
    return ok


print("Qwen3.8 GDN depthwise causal conv1d + SiLU")
results = [run(64, "none")]
run(64, "sigmoid")  # documents the inaccurate shortcut; not a required path
results.append(run(64, "exp"))
results.append(run(10240, "none"))
run(10240, "sigmoid")
results.append(run(10240, "exp"))
print("PASS" if all(results) else "FAIL")
