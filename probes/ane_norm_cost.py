"""Which op in the fused RMSNorm block is slow at real width (H=5120)?"""
import os, sys, io, time, contextlib, numpy as np
sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

eng = AneEngine()
H, S = 5120, 32
CONVC = ('    string pt = const()[name=string("pt"), val=string("valid")];\n'
         '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
         '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
         '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
         '    int32 gr = const()[name=string("gr"), val=int32(1)];')

def build(name, body, blobs, out_c):
    mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> x) {{
{CONVC}
{body}
  }} -> (y);
}}
"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        prog = eng.compile_multiproc(mil, blobs, H, out_c, S)
    if prog is None:
        print(f"  {name:26} COMPILE FAILED"); return
    eng._ensure_io(prog)
    xv = (np.random.randn(H, S) * 0.3).astype(np.float16)
    with _iosurface_view(prog._in_surf, (H, S), np.float16) as d: d[:] = xv
    for _ in range(3): eng.submit(prog, procedure_index=0)
    t0 = time.perf_counter(); N = 50
    for _ in range(N): eng.submit(prog, procedure_index=0)
    print(f"  {name:26} {(time.perf_counter()-t0)/N*1e3:8.3f} ms")

ones1 = {"on.bin": np.full((1, H), 1.0/H, np.float16).tobytes()}
D1 = (f'    tensor<fp16, [1, {H}, 1, 1]> onw = const()[name=string("onw"), '
      f'val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/on.bin"), offset=uint64(64)))];')
W = 64
onesW = {"on.bin": np.full((W, H), 1.0/H, np.float16).tobytes()}
DW = (f'    tensor<fp16, [{W}, {H}, 1, 1]> onw = const()[name=string("onw"), '
      f'val=tensor<fp16, [{W}, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/on.bin"), offset=uint64(64)))];')

build("passthrough", f'    tensor<fp16, [1, {H}, 1, {S}]> y = mul(x=x, y=x)[name=string("y")];', {}, H)
build("conv_to_1ch_only", D1 + f'''
    tensor<fp16, [1, {H}, 1, {S}]> sq = mul(x=x, y=x)[name=string("sq")];
    tensor<fp16, [1, 1, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=onw, x=sq)[name=string("y")];''', dict(ones1), 1)
build("full_rmsnorm_1ch", D1 + f'''
    tensor<fp16, [1, {H}, 1, {S}]> sq = mul(x=x, y=x)[name=string("sq")];
    tensor<fp16, [1, 1, 1, {S}]> ms = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=onw, x=sq)[name=string("ms")];
    fp16 ep = const()[name=string("ep"), val=fp16(0x1.0p-20)];
    tensor<fp16, [1, 1, 1, {S}]> msa = add(x=ms, y=ep)[name=string("msa")];
    tensor<fp16, [1, 1, 1, {S}]> sd = sqrt(x=msa)[name=string("sd")];
    tensor<fp16, [1, {H}, 1, {S}]> y = real_div(x=x, y=sd)[name=string("y")];''', dict(ones1), H)
build(f"full_rmsnorm_{W}ch_slice", DW + f'''
    tensor<fp16, [1, {H}, 1, {S}]> sq = mul(x=x, y=x)[name=string("sq")];
    tensor<fp16, [1, {W}, 1, {S}]> msw = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=onw, x=sq)[name=string("msw")];
    tensor<fp16, [1, 1, 1, {S}]> ms = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,1,1,{S}]), x=msw)[name=string("ms")];
    fp16 ep = const()[name=string("ep"), val=fp16(0x1.0p-20)];
    tensor<fp16, [1, 1, 1, {S}]> msa = add(x=ms, y=ep)[name=string("msa")];
    tensor<fp16, [1, 1, 1, {S}]> sd = sqrt(x=msa)[name=string("sd")];
    tensor<fp16, [1, {H}, 1, {S}]> y = real_div(x=x, y=sd)[name=string("y")];''', dict(onesW), H)
