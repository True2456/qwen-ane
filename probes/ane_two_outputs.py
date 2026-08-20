"""Can one ANE program return TWO output tensors?

If so, the fused layer can emit `out` [H] and the next layer's projection [P]
separately, dodging the ~9216-channel pad+add width cap -- which is what stands
between the current build and putting every projection on the ANE.
"""
import os, sys, io, ctypes, contextlib, numpy as np
sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (AneEngine, _msg, _sel, _desc, _objc, _cls,
                                    _nsarray, _nsnumber_int, _indexset_to_nsarray,
                                    _iosurface_view, _load_iosurface,
                                    _iosurface_alloc_size, _create_iosurface)

eng = AneEngine()
C, A, B, S = 5120, 5120, 16480, 32
rng = np.random.default_rng(0)
wa = (rng.standard_normal((A, C)) * 0.02).astype(np.float32)
wb = (rng.standard_normal((B, C)) * 0.02).astype(np.float32)

def q4(W):
    """Returns blobs AND the dequantized weight -- the reference must use the
    weights the ANE actually holds, or int4 error masquerades as a wiring bug."""
    sc = np.abs(W).max(axis=1, keepdims=True) / 7
    q = np.clip(np.rint(W / np.where(sc == 0, 1, sc)), -8, 7).astype(np.int8)
    f = (q.reshape(-1).astype(np.uint8) & 0x0F)
    deq = q.astype(np.float32) * sc.astype(np.float16).astype(np.float32)
    return (f[0::2] | (f[1::2] << 4)).tobytes(), sc.astype(np.float16).tobytes(), deq

blobs = {}
blobs["a.bin"], blobs["as.bin"], dqa = q4(wa)
blobs["b.bin"], blobs["bs.bin"], dqb = q4(wb)
def decl(nm, O):
    return (f'    tensor<int4, [{O}, {C}, 1, 1]> {nm}q = const()[name=string("{nm}q"), val=tensor<int4, [{O}, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/{nm}.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{O}, 1, 1, 1]> {nm}s = const()[name=string("{nm}s"), val=tensor<fp16, [{O}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/{nm}s.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{O}, {C}, 1, 1]> {nm}w = constexpr_blockwise_shift_scale(data={nm}q, scale={nm}s)[name=string("{nm}d")];')

mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
{decl("a", A)}
{decl("b", B)}
    tensor<fp16, [1, {A}, 1, {S}]> y1 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=aw, x=x)[name=string("y1")];
    tensor<fp16, [1, {B}, 1, {S}]> y2 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=bw, x=x)[name=string("y2")];
  }} -> (y1, y2);
}}
"""
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    prog = eng.compile_multiproc(mil, blobs, C, A, S)
if prog is None:
    print("  two-output MIL: COMPILE FAILED")
    print("\n".join(buf.getvalue().strip().splitlines()[-4:])); sys.exit()
print(f"  two-output MIL: COMPILED  (outputs {A} and {B} channels)")

inner = _msg(prog.model, "model") or prog.model
Sym = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong)
send = Sym(("objc_msgSend", _objc))
print("  output symbols:", _desc(_indexset_to_nsarray(
    send(inner, _sel("outputSymbolIndicesForProcedureIndex:"), 0))).replace("\n", " "))

_load_iosurface()
in_surf = _create_iosurface(_iosurface_alloc_size(C * S))
o1 = _create_iosurface(_iosurface_alloc_size(A * S))
o2 = _create_iosurface(_iosurface_alloc_size(B * S))
surf_cls = _cls("_ANEIOSurfaceObject")
IS = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)
def wrap(s):
    return IS(("objc_msgSend", _objc))(_msg(surf_cls, "alloc"),
        _sel("initWithIOSurface:startOffset:shouldRetain:"), s, _nsnumber_int(0), True)

SEL = _sel("initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:"
           "perfStats:procedureIndex:sharedEvents:transactionHandle:")
F = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
req = F(("objc_msgSend", _objc))(
    _msg(_cls("_ANERequest"), "alloc"), SEL,
    _nsarray([wrap(in_surf)]), _nsarray([_nsnumber_int(0)]),
    _nsarray([wrap(o1), wrap(o2)]),
    _nsarray([_nsnumber_int(0), _nsnumber_int(1)]),
    None, None, _nsnumber_int(0), None, None)
print(f"  request with 2 outputs: {'built' if req else 'FAILED'}")
if not req: sys.exit()

x = (rng.standard_normal((S, C)) * 0.1).astype(np.float32)
with _iosurface_view(in_surf, (C, S), np.float16) as d:
    d[:] = x.T.astype(np.float16)
Eval = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
err = ctypes.c_void_p(0)
ok = Eval(("objc_msgSend", _objc))(prog.model, _sel("evaluateWithQoS:options:request:error:"),
                                   21, prog._compile_opts, req, ctypes.byref(err))
if not ok:
    print(f"  evaluate FAILED: {(_desc(err.value) if err.value else '')[:160]}"); sys.exit()
with _iosurface_view(o1, (A, S), np.float16) as s1:
    y1 = np.array(s1.T, np.float32)
with _iosurface_view(o2, (B, S), np.float16) as s2:
    y2 = np.array(s2.T, np.float32)
r1, r2 = x @ dqa.T, x @ dqb.T
e1 = np.abs(y1 - r1).max() / (np.abs(r1).max() + 1e-9)
e2 = np.abs(y2 - r2).max() / (np.abs(r2).max() + 1e-9)
print(f"  output 1 [{A}]: rel {e1:.4f}  {'OK' if e1 < 5e-2 else 'WRONG'}")
print(f"  output 2 [{B}]: rel {e2:.4f}  {'OK' if e2 < 5e-2 else 'WRONG'}")
