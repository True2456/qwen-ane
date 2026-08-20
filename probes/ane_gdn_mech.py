"""The three mechanisms a GDN step needs in the [1, H*Dk, 1, Dv] layout.

  1. grouped conv, Dk->1 per head : the sum over Dk (kv_mem, y)
  2. grouped conv, 1->Dk per head : broadcasting delta back over Dk
  3. width-1 slice broadcast       : k, q, decay, beta against a Dv-wide tensor
"""
import os, sys, io, contextlib, numpy as np
sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view
eng = AneEngine()
H, Dk, Dv = 48, 128, 128
HK = H * Dk

HDR = ('    string pt = const()[name=string("pt"), val=string("valid")];\n'
       '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
       '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
       '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];')

def run(name, C, S, out_c, body, blobs, check=None):
    mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
{HDR}
{body}
  }} -> (y);
}}
"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        prog = eng.compile_multiproc(mil, blobs, C, out_c, S)
    if prog is None:
        print(f"  {name:42} COMPILE FAILED"); return
    if check is None:
        print(f"  {name:42} OK (compile)"); return
    eng._ensure_io(prog)
    xin, ref = check
    with _iosurface_view(prog._in_surf, (C, S), np.float16) as d:
        d[:] = 0; d[:, :] = xin.astype(np.float16)
    if not eng.submit(prog, procedure_index=0):
        print(f"  {name:42} SUBMIT FAILED"); return
    with _iosurface_view(prog._out_surf, (out_c, S), np.float16) as o:
        y = np.array(o, np.float32)
    rel = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-9)
    print(f"  {name:42} {'OK  ' if rel < 2e-2 else 'WRONG'} rel={rel:.4f}")

# 1. grouped conv Dk -> 1 per head  (sum over Dk)
w = np.ones((H, Dk, 1, 1), np.float16)
x1 = (np.random.default_rng(0).standard_normal((HK, Dv)) * 0.1).astype(np.float32)
ref1 = x1.reshape(H, Dk, Dv).sum(1)
run("grouped conv Dk->1 (sum over Dk)", HK, Dv, H,
    f'    int32 gr = const()[name=string("gr"), val=int32({H})];\n'
    f'    tensor<fp16, [{H}, {Dk}, 1, 1]> gw = const()[name=string("gw"), val=tensor<fp16, [{H}, {Dk}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/g.bin"), offset=uint64(64)))];\n'
    f'    tensor<fp16, [1, {H}, 1, {Dv}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)[name=string("y")];',
    {"g.bin": w.tobytes()}, check=(x1, ref1))

# 2. grouped conv 1 -> Dk per head  (replicate delta over Dk)
w2 = np.ones((HK, 1, 1, 1), np.float16)
x2 = (np.random.default_rng(1).standard_normal((H, Dv)) * 0.1).astype(np.float32)
ref2 = np.repeat(x2, Dk, axis=0)
run("grouped conv 1->Dk (replicate over Dk)", H, Dv, HK,
    f'    int32 gr = const()[name=string("gr"), val=int32({H})];\n'
    f'    tensor<fp16, [{HK}, 1, 1, 1]> gw = const()[name=string("gw"), val=tensor<fp16, [{HK}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/g.bin"), offset=uint64(64)))];\n'
    f'    tensor<fp16, [1, {HK}, 1, {Dv}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)[name=string("y")];',
    {"g.bin": w2.tobytes()}, check=(x2, ref2))

# 3. width-1 slice broadcast against a Dv-wide tensor.
#    Input and output must share a width here: _ensure_io sizes the output
#    surface from seq_len, so a program whose output is narrower than its input
#    reads past the end and fails with status 0x1d. Broadcast column 0.
x3 = (np.random.default_rng(2).standard_normal((HK, Dv)) * 0.1).astype(np.float32)
ref3 = x3 * x3[:, 0:1]
run("width-1 slice broadcast (k/q/decay)", HK, Dv, HK,
    f'    tensor<fp16, [1, {HK}, 1, 1]> b = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{HK},1,1]), x=x)[name=string("b")];\n'
    f'    tensor<fp16, [1, {HK}, 1, {Dv}]> y = mul(x=x, y=b)[name=string("y")];',
    {}, check=(x3, ref3))
