"""Does the ANE support concat along the channel axis, and how wide can output go?"""
import os, sys, io, contextlib, numpy as np
sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view
eng = AneEngine()
S = 32
def build(name, C, body, out_c):
    mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
{body}
  }} -> (y);
}}
"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        prog = eng.compile_multiproc(mil, {}, C, out_c, S)
    print(f"  {name:38} {'OK' if prog else 'COMPILE FAILED'}")
    return prog



S_ = S
print("pad+add concat emulation, varying total width (H=5120 + P):")
for P in (1024, 2048, 3072, 4096, 6144, 10240, 16480):
    C, OUT = 5120, 5120 + P
    build(f"H=5120 P={P} -> {OUT}", C,
      f'    tensor<int32, [8]> pa = const()[name=string("pa"), val=tensor<int32, [8]>([0,0,0,{P},0,0,0,0])];\n'
      f'    tensor<int32, [8]> pb = const()[name=string("pb"), val=tensor<int32, [8]>([0,0,{C},0,0,0,0,0])];\n'
      f'    tensor<fp16, [1, {P}, 1, {S_}]> t = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{P},1,{S_}]), x=x)[name=string("t")];\n'
      f'    tensor<fp16, [1, {OUT}, 1, {S_}]> ya = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pa, x=x)[name=string("ya")];\n'
      f'    tensor<fp16, [1, {OUT}, 1, {S_}]> yb = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pb, x=t)[name=string("yb")];\n'
      f'    tensor<fp16, [1, {OUT}, 1, {S_}]> y = add(x=ya, y=yb)[name=string("y")];', OUT)
