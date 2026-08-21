"""Which construct in the chunked GDN graph kills ANECompilerService?

`probes/ane_gdn_scan64.py --algorithm chunk` fails at every token count, T=2
included, with `connection to service named com.apple.ANECompilerService`.
That is the compiler process dying rather than a MIL rejection, and the graph
uses only ops docs/ANE-REFERENCE.md lists as working, so the cause is a shape
or a combination rather than an op.

Each case below is the smallest graph containing one suspect construct. A case
that reports SERVICE DIED is the one to avoid; InvalidMILProgram is an ordinary
rejection and much less interesting.
"""
import contextlib, io, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine

H, T, D = 48, 8, 128            # Qwen GDN geometry
eng = AneEngine()


def build(label, body, cin, cout, width):
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {cin}, 1, {width}]> x) {{
{body}
  }} -> (y);
}}
'''
    cap = io.StringIO()
    err = None
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        try:
            p = eng.compile_multiproc(mil, {}, cin, cout, width)
        except Exception as ex:      # noqa: BLE001
            p, err = None, ex
    text = cap.getvalue()
    if p is not None:
        return "OK"
    if "ANECompilerService" in text or "4097" in text:
        return "SERVICE DIED"
    if "InvalidMILProgram" in text:
        return "InvalidMILProgram"
    return (text.strip().splitlines() or ["unknown"])[-1][:60]


cases = {}

# 1. a real axis permutation, swapping the head and token axes
cases["transpose perm=[0,2,1,3]"] = build(
    "t", f'''    tensor<fp16,[1,{H},{T},{D}]> r = reshape(shape=tensor<int32,[4]>([1,{H},{T},{D}]), x=x)[name=string("r")];
    tensor<fp16,[1,{T},{H},{D}]> t = transpose(perm=tensor<int32,[4]>([0,2,1,3]), x=r)[name=string("t")];
    tensor<fp16,[1,{H*T},1,{D}]> y = reshape(shape=tensor<int32,[4]>([1,{H*T},1,{D}]), x=t)[name=string("y")];''',
    H*T, H*T, D)

# 2. the last-two-axis transpose the working attention core already uses
cases["transpose perm=[0,1,3,2]"] = build(
    "t2", f'''    tensor<fp16,[1,{H},{T},{D}]> r = reshape(shape=tensor<int32,[4]>([1,{H},{T},{D}]), x=x)[name=string("r")];
    tensor<fp16,[1,{H},{D},{T}]> t = transpose(perm=tensor<int32,[4]>([0,1,3,2]), x=r)[name=string("t")];
    tensor<fp16,[1,{H*D},1,{T}]> y = reshape(shape=tensor<int32,[4]>([1,{H*D},1,{T}]), x=t)[name=string("y")];''',
    H*T, H*D, D)

# 3. matmul batched over all 48 heads (the core batches over 4)
cases[f"matmul batched over {H} heads"] = build(
    "mm", f'''    tensor<fp16,[1,{H},{T},{D}]> r = reshape(shape=tensor<int32,[4]>([1,{H},{T},{D}]), x=x)[name=string("r")];
    tensor<fp16,[1,{H},{T},{T}]> m = matmul(transpose_x=bool(false), transpose_y=bool(true), x=r, y=r)[name=string("m")];
    tensor<fp16,[1,{H*T},1,{T}]> y = reshape(shape=tensor<int32,[4]>([1,{H*T},1,{T}]), x=m)[name=string("y")];''',
    H*T, H*T, D)

# 4. same, batched over 4 heads, which the attention core proves works
cases["matmul batched over 4 heads"] = build(
    "mm4", f'''    tensor<fp16,[1,4,{T},{D}]> r = reshape(shape=tensor<int32,[4]>([1,4,{T},{D}]), x=x)[name=string("r")];
    tensor<fp16,[1,4,{T},{T}]> m = matmul(transpose_x=bool(false), transpose_y=bool(true), x=r, y=r)[name=string("m")];
    tensor<fp16,[1,{4*T},1,{T}]> y = reshape(shape=tensor<int32,[4]>([1,{4*T},1,{T}]), x=m)[name=string("y")];''',
    4*T, 4*T, D)

# 5. transpose_x=true, used to accumulate the chunk's state contribution
cases["matmul transpose_x=true"] = build(
    "mmt", f'''    tensor<fp16,[1,{H},{T},{D}]> r = reshape(shape=tensor<int32,[4]>([1,{H},{T},{D}]), x=x)[name=string("r")];
    tensor<fp16,[1,{H},{D},{D}]> m = matmul(transpose_x=bool(true), transpose_y=bool(false), x=r, y=r)[name=string("m")];
    tensor<fp16,[1,{H*D},1,{D}]> y = reshape(shape=tensor<int32,[4]>([1,{H*D},1,{D}]), x=m)[name=string("y")];''',
    H*T, H*D, D)

print(f"Qwen GDN geometry H={H} T={T} D={D}\n")
print(f"  {'construct':>32}  result")
for k, v in cases.items():
    print(f"  {k:>32}  {v}")
