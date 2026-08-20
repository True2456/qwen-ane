"""Can the ANE reduce across the CHANNEL axis? RMSNorm needs it.

Layer-level fusion (out_proj + residual + RMSNorm + MLP in one program) hinges
on computing mean(x^2) over C in the [1, C, 1, S] planar layout. Tries several
MIL formulations and checks each against numpy.
"""
import os, sys, io, contextlib, numpy as np
sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

eng = AneEngine()
C, S, eps = 512, 32, 1e-6
rng = np.random.default_rng(0)
x = (rng.standard_normal((C, S)) * 0.5).astype(np.float32)
w = (rng.standard_normal(C) * 0.3 + 1.0).astype(np.float32)
ref = (x / np.sqrt((x**2).mean(axis=0, keepdims=True) + eps)) * w[:, None]

WB = {"w.bin": w.astype(np.float16).tobytes(),
      "ones.bin": np.full((1, C), 1.0/C, np.float16).tobytes()}
DECL = (f'    tensor<fp16, [1, {C}, 1, 1]> gw = const()[name=string("gw"), '
        f'val=tensor<fp16, [1, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), '
        f'offset=uint64(64)))];')

ONES = (f'    tensor<fp16, [1, {C}, 1, 1]> ow = const()[name=string("ow"), '
        f'val=tensor<fp16, [1, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/ones.bin"), '
        f'offset=uint64(64)))];')
CONVC = ('    string pt = const()[name=string("pt"), val=string("valid")];\n'
         '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
         '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
         '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
         '    int32 gr = const()[name=string("gr"), val=int32(1)];')

VARIANTS = {
"conv_sum_rms": f'''
{ONES}
{CONVC}
    tensor<fp16, [1, {C}, 1, {S}]> sq = mul(x=x, y=x)[name=string("sq")];
    tensor<fp16, [1, 1, 1, {S}]> ms = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=ow, x=sq)[name=string("ms")];
    fp16 ep = const()[name=string("ep"), val=fp16(0x1.0p-20)];
    tensor<fp16, [1, 1, 1, {S}]> msa = add(x=ms, y=ep)[name=string("msa")];
    tensor<fp16, [1, 1, 1, {S}]> rs = rsqrt(x=msa)[name=string("rs")];
    tensor<fp16, [1, {C}, 1, {S}]> nx = mul(x=x, y=rs)[name=string("nx")];
    tensor<fp16, [1, {C}, 1, {S}]> y = mul(x=nx, y=gw)[name=string("y")];''',
"conv_sum_sqrt_div": f'''
{ONES}
{CONVC}
    tensor<fp16, [1, {C}, 1, {S}]> sq = mul(x=x, y=x)[name=string("sq")];
    tensor<fp16, [1, 1, 1, {S}]> ms = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=ow, x=sq)[name=string("ms")];
    fp16 ep = const()[name=string("ep"), val=fp16(0x1.0p-20)];
    tensor<fp16, [1, 1, 1, {S}]> msa = add(x=ms, y=ep)[name=string("msa")];
    tensor<fp16, [1, 1, 1, {S}]> sd = sqrt(x=msa)[name=string("sd")];
    tensor<fp16, [1, {C}, 1, {S}]> nx = real_div(x=x, y=sd)[name=string("nx")];
    tensor<fp16, [1, {C}, 1, {S}]> y = mul(x=nx, y=gw)[name=string("y")];''',
"reduce_mean_axis1": f'''
    tensor<fp16, [1, {C}, 1, {S}]> sq = mul(x=x, y=x)[name=string("sq")];
    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([1])];
    tensor<fp16, [1, 1, 1, {S}]> ms = reduce_mean(axes=ax, keep_dims=bool(true), x=sq)[name=string("ms")];
    fp16 ep = const()[name=string("ep"), val=fp16(0x1.0p-20)];
    tensor<fp16, [1, 1, 1, {S}]> msa = add(x=ms, y=ep)[name=string("msa")];
    tensor<fp16, [1, 1, 1, {S}]> rs = rsqrt(x=msa)[name=string("rs")];
    tensor<fp16, [1, {C}, 1, {S}]> nx = mul(x=x, y=rs)[name=string("nx")];
    tensor<fp16, [1, {C}, 1, {S}]> y = mul(x=nx, y=gw)[name=string("y")];''',
"l2_norm": f'''
    tensor<fp16, [1, {C}, 1, {S}]> nx = l2_norm(axes=const()[name=string("ax"), val=tensor<int32, [1]>([1])], epsilon=fp16(0x1.0p-20), x=x)[name=string("nx")];
    tensor<fp16, [1, {C}, 1, {S}]> y = mul(x=nx, y=gw)[name=string("y")];''',
"layer_norm_axis1": f'''
    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([1])];
    tensor<fp16, [1, {C}, 1, {S}]> nx = layer_norm(axes=ax, epsilon=fp16(0x1.0p-20), x=x)[name=string("nx")];
    tensor<fp16, [1, {C}, 1, {S}]> y = mul(x=nx, y=gw)[name=string("y")];''',
}

for name, body in VARIANTS.items():
    mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
{DECL}
{body}
  }} -> (y);
}}
"""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            prog = eng.compile_multiproc(mil, dict(WB), C, C, S)
    except Exception as exc:
        print(f"  {name:22} EXCEPTION {type(exc).__name__}: {exc}"); continue
    if prog is None:
        print(f"  {name:22} COMPILE FAILED"); continue
    eng._ensure_io(prog)
    with _iosurface_view(prog._in_surf, (C, S), np.float16) as d:
        d[:] = 0; d[:, :] = x.astype(np.float16)
    if not eng.submit(prog, procedure_index=0):
        print(f"  {name:22} SUBMIT FAILED"); continue
    with _iosurface_view(prog._out_surf, (C, S), np.float16) as o:
        y = np.array(o, np.float32)
    rel = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-9)
    print(f"  {name:22} {'OK  ' if rel < 5e-2 else 'WRONG'} rel={rel:.4f}")
