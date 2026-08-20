"""A full gated-delta step on the ANE, checked against the numpy reference.

Layout: state[h, dv, dk] -> channel h*Dk+dk, width dv. In that layout
  * the sum over Dk is a grouped conv (groups=H, Dk->1)
  * broadcasting delta back over Dk is a grouped conv (groups=H, 1->Dk)
  * k, q, decay, beta are width-1 columns broadcast across Dv
All six tensors ride in on one surface; y and the new state leave on two.
"""
import os, sys, io, ctypes, contextlib, numpy as np
sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (AneEngine, _iosurface_view, _load_iosurface,
                                    _iosurface_alloc_size, _create_iosurface,
                                    _msg, _sel, _desc, _objc, _cls, _nsarray,
                                    _nsnumber_int)

H, Dk, Dv = 48, 128, 128
# Row stride is width*2 bytes and every working program so far has been a
# multiple of 64. Dv+3 = 131 gives 262 bytes; pad the input to 160 (320 bytes)
# and keep the extra columns unused.
HK, W = H * Dk, 160
CIN = HK + 2 * H
eng = AneEngine()

blobs = {"gsum.bin": np.ones((H, Dk, 1, 1), np.float16).tobytes(),
         "grep.bin": np.ones((HK, 1, 1, 1), np.float16).tobytes()}
def sl(nm, src, c0, c1, w0, w1):
    return (f'    tensor<fp16, [1, {c1-c0}, 1, {w1-w0}]> {nm} = slice_by_index('
            f'begin=tensor<int32, [4]>([0,{c0},0,{w0}]), '
            f'end=tensor<int32, [4]>([1,{c1},1,{w1}]), x={src})[name=string("{nm}")];')

mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {CIN}, 1, {W}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 g1 = const()[name=string("g1"), val=int32(1)];
    int32 gh = const()[name=string("gh"), val=int32({H})];
    tensor<fp16, [{H}, {Dk}, 1, 1]> gsum = const()[name=string("gsum"), val=tensor<fp16, [{H}, {Dk}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/gsum.bin"), offset=uint64(64)))];
    tensor<fp16, [{HK}, 1, 1, 1]> grep = const()[name=string("grep"), val=tensor<fp16, [{HK}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/grep.bin"), offset=uint64(64)))];
{sl("stt", "x", 0, HK, 0, Dv)}
{sl("dcy", "x", 0, HK, Dv, Dv+1)}
{sl("kk",  "x", 0, HK, Dv+1, Dv+2)}
{sl("qq",  "x", 0, HK, Dv+2, Dv+3)}
{sl("vv",  "x", HK, HK+H, 0, Dv)}
{sl("bta", "x", HK+H, HK+2*H, 0, 1)}
    tensor<fp16, [1, {HK}, 1, {Dv}]> s1 = mul(x=stt, y=dcy)[name=string("s1")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> sk = mul(x=s1, y=kk)[name=string("sk")];
    tensor<fp16, [1, {H}, 1, {Dv}]> kvm = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sk)[name=string("kvm")];
    tensor<fp16, [1, {H}, 1, {Dv}]> dlt = sub(x=vv, y=kvm)[name=string("dlt")];
    tensor<fp16, [1, {H}, 1, {Dv}]> dbt = mul(x=dlt, y=bta)[name=string("dbt")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> dup = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=dbt)[name=string("dup")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> upd = mul(x=dup, y=kk)[name=string("upd")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> s2 = add(x=s1, y=upd)[name=string("s2")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> sq = mul(x=s2, y=qq)[name=string("sq")];
    tensor<fp16, [1, {H}, 1, {Dv}]> y = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sq)[name=string("y")];
  }} -> (y, s2);
}}
"""
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    prog = eng.compile_multiproc(mil, blobs, CIN, H, W)
if prog is None:
    print("COMPILE FAILED"); print("\n".join(buf.getvalue().strip().splitlines()[-5:])); sys.exit()
print(f"  compiled: in [{CIN},{W}] -> y [{H},{Dv}] + state [{HK},{Dv}]")

_load_iosurface()
sin = _create_iosurface(_iosurface_alloc_size(CIN * W))
sy  = _create_iosurface(_iosurface_alloc_size(H * Dv))
ss  = _create_iosurface(_iosurface_alloc_size(HK * Dv))
inner = _msg(prog.model, "model") or prog.model
import re
desc = _desc(_msg(inner, "description"))
chans = [int(c) for c, _, _ in re.findall(
    r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";', desc, re.S)]
print("  output symbol order (channels):", chans)
surf_for = {H: sy, HK: ss}
IS = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)
def wrap(s):
    return IS(("objc_msgSend", _objc))(_msg(_cls("_ANEIOSurfaceObject"), "alloc"),
        _sel("initWithIOSurface:startOffset:shouldRetain:"), s, _nsnumber_int(0), True)
F = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
req = F(("objc_msgSend", _objc))(
    _msg(_cls("_ANERequest"), "alloc"),
    _sel("initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:"
         "perfStats:procedureIndex:sharedEvents:transactionHandle:"),
    _nsarray([wrap(sin)]), _nsarray([_nsnumber_int(0)]),
    _nsarray([wrap(surf_for[c]) for c in chans]),
    _nsarray([_nsnumber_int(i) for i in range(len(chans))]),
    None, None, _nsnumber_int(0), None, None)
print(f"  request: {'built' if req else 'FAILED'}")

rng = np.random.default_rng(0)
state = (rng.standard_normal((H, Dv, Dk)) * 0.1).astype(np.float32)
k = (rng.standard_normal((H, Dk)) * 0.2).astype(np.float32)
q = (rng.standard_normal((H, Dk)) * 0.2).astype(np.float32)
v = (rng.standard_normal((H, Dv)) * 0.2).astype(np.float32)
decay = rng.uniform(0.8, 1.0, (H,)).astype(np.float32)
beta = rng.uniform(0.2, 0.8, (H,)).astype(np.float32)

xin = np.zeros((CIN, W), np.float32)
xin[:HK, :Dv] = state.transpose(0, 2, 1).reshape(HK, Dv)   # [h,dk,dv]
xin[:HK, Dv] = np.repeat(decay, Dk)
xin[:HK, Dv+1] = k.reshape(-1)
xin[:HK, Dv+2] = q.reshape(-1)
xin[HK:HK+H, :Dv] = v
xin[HK+H:HK+2*H, 0] = beta
with _iosurface_view(sin, (CIN, W), np.float16) as d:
    d[:] = xin.astype(np.float16)
Eval = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
err = ctypes.c_void_p(0)
ok = Eval(("objc_msgSend", _objc))(prog.model, _sel("evaluateWithQoS:options:request:error:"),
                                   21, prog._compile_opts, req, ctypes.byref(err))
if not ok:
    print(f"  evaluate FAILED: {(_desc(err.value) if err.value else '')[:150]}"); sys.exit()
with _iosurface_view(sy, (H, Dv), np.float16) as o: y_ane = np.array(o, np.float32)
with _iosurface_view(ss, (HK, Dv), np.float16) as o: s_ane = np.array(o, np.float32)

# reference
s = state * decay[:, None, None]
kv = (s * k[:, None, :]).sum(-1)
dlt = (v - kv) * beta[:, None]
s = s + k[:, None, :] * dlt[..., None]
y_ref = (s * q[:, None, :]).sum(-1)
s_ref = s.transpose(0, 2, 1).reshape(HK, Dv)
ry = np.abs(y_ane - y_ref).max() / (np.abs(y_ref).max() + 1e-9)
rs = np.abs(s_ane - s_ref).max() / (np.abs(s_ref).max() + 1e-9)
print(f"  y     rel {ry:.4f}  {'OK' if ry < 3e-2 else 'WRONG'}")
print(f"  state rel {rs:.4f}  {'OK' if rs < 3e-2 else 'WRONG'}")
