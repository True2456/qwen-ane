"""Probe zero-copy persistent GDN state with ping-pong IOSurfaces.

The server currently copies/transposes the recurrent state through NumPy after
every step.  A second ordinary MIL activation input is rejected, so this probe
uses one combined surface.  The recurrence runs across a padded width of 160
(the first 128 lanes are real state), so its state output has exactly the same
row stride as the prefix of the next input:

    request A: combined0 -> y + combined1[:state_rows]
    request B: combined1 -> y + combined0[:state_rows]

The attempted stride-compatible state packing is currently rejected by the
compiler.  The file is retained as a concrete negative result and starting
point for the next layout attempt.
"""
import contextlib
import ctypes
import io
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view


H, DK, DV = 48, 128, 128
HK = H * DK
CIN = HK + 2 * H
W = 160
eng = AneEngine()


def sl(nm, c0, c1, w0, w1):
    return (f'    tensor<fp16, [1, {c1-c0}, 1, {w1-w0}]> {nm} = '
            f'slice_by_index(begin=tensor<int32, [4]>([0,{c0},0,{w0}]), '
            f'end=tensor<int32, [4]>([1,{c1},1,{w1}]), x=x)'
            f'[name=string("{nm}")];')


blobs = {
    "sum.bin": np.ones((H, DK, 1, 1), np.float16).tobytes(),
    "rep.bin": np.ones((HK, 1, 1, 1), np.float16).tobytes(),
    "pack.bin": np.pad(np.eye(DV, dtype=np.float16), ((0, 0), (0, W-DV))).tobytes(),
}
mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {CIN}, 1, {W}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gh = const()[name=string("gh"), val=int32({H})];
    tensor<fp16, [{H}, {DK}, 1, 1]> gsum = const()[name=string("gsum"), val=tensor<fp16, [{H}, {DK}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/sum.bin"), offset=uint64(64)))];
    tensor<fp16, [{HK}, 1, 1, 1]> grep = const()[name=string("grep"), val=tensor<fp16, [{HK}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/rep.bin"), offset=uint64(64)))];
    tensor<fp16, [{DV}, {W}]> pack = const()[name=string("pack"), val=tensor<fp16, [{DV}, {W}]>(BLOBFILE(path=string("@model_path/weights/pack.bin"), offset=uint64(64)))];
{sl("state", 0, HK, 0, DV)}
{sl("dcy", 0, HK, DV, DV+1)}
{sl("kk", 0, HK, DV+1, DV+2)}
{sl("qq", 0, HK, DV+2, DV+3)}
{sl("vv", HK, HK+H, 0, DV)}
{sl("bta", HK+H, HK+2*H, 0, 1)}
    tensor<fp16, [1, {HK}, 1, {DV}]> s1 = mul(x=state, y=dcy)[name=string("s1")];
    tensor<fp16, [1, {HK}, 1, {DV}]> sk = mul(x=s1, y=kk)[name=string("sk")];
    tensor<fp16, [1, {H}, 1, {DV}]> kvm = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sk)[name=string("kvm")];
    tensor<fp16, [1, {H}, 1, {DV}]> delta = sub(x=vv, y=kvm)[name=string("delta")];
    tensor<fp16, [1, {H}, 1, {DV}]> db = mul(x=delta, y=bta)[name=string("db")];
    tensor<fp16, [1, {HK}, 1, {DV}]> dup = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=db)[name=string("dup")];
    tensor<fp16, [1, {HK}, 1, {DV}]> ku = mul(x=dup, y=kk)[name=string("ku")];
    tensor<fp16, [1, {HK}, 1, {DV}]> new_state = add(x=s1, y=ku)[name=string("new_state")];
    tensor<fp16, [1, {HK}, 1, {DV}]> sq = mul(x=new_state, y=qq)[name=string("sq")];
    tensor<fp16, [1, {H}, 1, {DV}]> y = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sq)[name=string("y")];
    tensor<fp16, [{HK}, {DV}]> ns2 = reshape(shape=tensor<int32, [2]>([{HK},{DV}]), x=new_state)[name=string("ns2")];
    tensor<fp16, [{HK}, {W}]> sw2 = matmul(transpose_x=bool(false), transpose_y=bool(false), x=ns2, y=pack)[name=string("sw2")];
    tensor<fp16, [1, {HK}, 1, {W}]> state_out = reshape(shape=tensor<int32, [4]>([1,{HK},1,{W}]), x=sw2)[name=string("state_out")];
  }} -> (y, state_out);
}}
// qwen38_gdn_persistent_pingpong
'''

capture = io.StringIO()
with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
    prog = eng.compile_multiproc(mil, blobs, CIN, H, W)
if prog is None:
    print("GDN persistent state: COMPILE FAILED")
    print("\n".join(capture.getvalue().splitlines()[-5:]))
    raise SystemExit(1)

E._load_iosurface()
combined0 = E._create_iosurface(E._iosurface_alloc_size(CIN * W))
combined1 = E._create_iosurface(E._iosurface_alloc_size(CIN * W))
y_surf = E._create_iosurface(E._iosurface_alloc_size(H * DV))
if not all((combined0, combined1, y_surf)):
    raise RuntimeError("IOSurface allocation failed")

inner = E._msg(prog.model, "model") or prog.model
desc = E._desc(E._msg(inner, "description"))
out_chans = [int(c) for c, _, _ in re.findall(
    r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";',
    desc, re.S)]
print(f"GDN persistent state: compiled, outputs={out_chans}")


def make_request(sin, sout):
    in_obj = E._wrap_iosurface(sin)
    # new_state has HK*W elements and lands at byte offset zero in the larger
    # combined surface.  The untouched tail contains v/beta host inputs.
    out_objs = [E._wrap_iosurface(y_surf if c == H else sout) for c in out_chans]
    F = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
    return F(("objc_msgSend", E._objc))(
        E._msg(E._cls("_ANERequest"), "alloc"),
        E._sel("initWithInputs:inputIndices:outputs:outputIndices:"
               "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
               "transactionHandle:"),
        E._nsarray([in_obj]), E._nsarray([E._nsnumber_int(0)]),
        E._nsarray(out_objs),
        E._nsarray([E._nsnumber_int(i) for i in range(len(out_objs))]),
        None, None, E._nsnumber_int(0), None, None)


req01, req10 = make_request(combined0, combined1), make_request(combined1, combined0)
if not req01 or not req10:
    raise RuntimeError("ping-pong request creation failed")
Eval = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p))
evaluate = Eval(("objc_msgSend", E._objc))


def submit(req):
    err = ctypes.c_void_p(0)
    ok = evaluate(prog.model, E._sel("evaluateWithQoS:options:request:error:"),
                  21, prog._compile_opts, req, ctypes.byref(err))
    if not ok:
        raise RuntimeError(E._desc(err.value) if err.value else "evaluate failed")


def write_params(surface, q, k, v, decay, beta):
    # Deliberately do not touch [:HK]: that is the persistent state.
    with _iosurface_view(surface, (CIN, W), np.float16) as b:
        b[:HK, DV] = np.repeat(decay, DK)
        b[:HK, DV+1] = k.reshape(-1)
        b[:HK, DV+2] = q.reshape(-1)
        b[HK:HK+H, :DV] = v
        b[HK:HK+H, DV:] = 0
        b[HK+H:, 0] = beta


def reference(state, q, k, v, decay, beta):
    state = state * decay[:, None, None]
    mem = np.sum(state * k[:, None, :], axis=-1)
    delta = (v - mem) * beta[:, None]
    state = state + k[:, None, :] * delta[:, :, None]
    y = np.sum(state * q[:, None, :], axis=-1)
    return y, state


rng = np.random.default_rng(42)
initial = rng.normal(0, 0.02, (H, DV, DK)).astype(np.float32)
with _iosurface_view(combined0, (CIN, W), np.float16) as dst:
    dst[:] = 0
    dst[:HK, :DV] = initial.transpose(0, 2, 1).reshape(HK, DV).astype(np.float16)
with _iosurface_view(combined1, (CIN, W), np.float16) as dst:
    dst[:] = 0

ref_state = initial
for step, (src, req) in enumerate(((combined0, req01), (combined1, req10))):
    q = rng.normal(0, 0.05, (H, DK)).astype(np.float32)
    k = rng.normal(0, 0.05, (H, DK)).astype(np.float32)
    v = rng.normal(0, 0.05, (H, DV)).astype(np.float32)
    decay = rng.uniform(0.94, 0.999, H).astype(np.float32)
    beta = rng.uniform(0.05, 0.4, H).astype(np.float32)
    write_params(src, q, k, v, decay, beta)
    submit(req)
    ref_y, ref_state = reference(ref_state, q, k, v, decay, beta)
    with _iosurface_view(y_surf, (H, DV), np.float16) as out:
        got_y = np.array(out, np.float32)
    rel_y = np.max(np.abs(got_y-ref_y)) / (np.max(np.abs(ref_y))+1e-9)
    print(f"  step {step+1}: y rel={rel_y:.4g}")

with _iosurface_view(combined0, (CIN, W), np.float16) as src:
    got_state = np.array(src[:HK, :DV], np.float32).reshape(H, DK, DV).transpose(0, 2, 1)
rel_state = np.max(np.abs(got_state-ref_state)) / (np.max(np.abs(ref_state))+1e-9)

# Pure recurrence timing: state ping-pongs in IOSurface memory.  Parameter
# writes are intentionally outside this timing and state is never read.
for _ in range(4):
    submit(req01); submit(req10)
n = 50
t0 = time.perf_counter()
for _ in range(n):
    submit(req01); submit(req10)
ms = (time.perf_counter() - t0) * 1e3 / (2*n)
ok = rel_state < 3e-3
print(f"  state after two zero-copy steps: rel={rel_state:.4g}")
print(f"  pure ping-pong recurrence: {ms:.3f} ms/step")
print("PASS" if ok else "FAIL")
