"""Where does the fused layer's 30 ms come from? Build it up piece by piece."""
import os, sys, io, time, contextlib, numpy as np
sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

eng = AneEngine()
H, I, Dc, S = 5120, 17408, 6144, 32
IN = Dc + H
CONVC = ('    string pt = const()[name=string("pt"), val=string("valid")];\n'
         '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
         '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
         '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
         '    int32 gr = const()[name=string("gr"), val=int32(1)];')

def q4(W):
    Wf = np.asarray(W, np.float32); hi = 7
    sc = np.abs(Wf).max(axis=1, keepdims=True) / hi
    q = np.clip(np.rint(Wf / np.where(sc == 0, 1, sc)), -8, 7).astype(np.int8)
    f = (q.reshape(-1).astype(np.uint8) & 0x0F)
    return (f[0::2] | (f[1::2] << 4)).tobytes(), sc.astype(np.float16).tobytes()

def decl(nm, O, In):
    return (f'    tensor<int4, [{O}, {In}, 1, 1]> {nm}q = const()[name=string("{nm}q"), val=tensor<int4, [{O}, {In}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/{nm}.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{O}, 1, 1, 1]> {nm}sc = const()[name=string("{nm}sc"), val=tensor<fp16, [{O}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/{nm}s.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{O}, {In}, 1, 1]> {nm}w = constexpr_blockwise_shift_scale(data={nm}q, scale={nm}sc)[name=string("{nm}dq")];')

rng = np.random.default_rng(0)
gu = rng.standard_normal((2*I, H)).astype(np.float32) * 0.02
dn = rng.standard_normal((H, I)).astype(np.float32) * 0.02
ow = rng.standard_normal((H, Dc)).astype(np.float32) * 0.02
B = {}
B["gu.bin"], B["gus.bin"] = q4(gu)
B["dn.bin"], B["dns.bin"] = q4(dn)
B["o.bin"],  B["os.bin"]  = q4(ow)
B["on.bin"] = np.full((1, H), 1.0/H, np.float16).tobytes()

MLPBODY = f'''
    tensor<fp16, [1, {2*I}, 1, {S}]> c = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=guw, x=NIN)[name=string("gu")];
    tensor<fp16, [1, {I}, 1, {S}]> g0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{I},1,{S}]), x=c)[name=string("g0")];
    tensor<fp16, [1, {I}, 1, {S}]> u0 = slice_by_index(begin=tensor<int32, [4]>([0,{I},0,0]), end=tensor<int32, [4]>([1,{2*I},1,{S}]), x=c)[name=string("u0")];
    tensor<fp16, [1, {I}, 1, {S}]> sg = sigmoid(x=g0)[name=string("sg")];
    tensor<fp16, [1, {I}, 1, {S}]> si = mul(x=g0, y=sg)[name=string("si")];
    tensor<fp16, [1, {I}, 1, {S}]> ac = mul(x=si, y=u0)[name=string("ac")];
    tensor<fp16, [1, {H}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=dnw, x=ac)[name=string("dn")];'''

def run(name, in_c, body, keys):
    blobs = {k: B[k] for k in keys}
    mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {in_c}, 1, {S}]> x) {{
{CONVC}
{chr(10).join(decl(n, *d) for n, d in keys_decl.get(name, []))}
{body}
  }} -> (y);
}}
"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        prog = eng.compile_multiproc(mil, blobs, in_c, H, S)
    if prog is None:
        print(f"  {name:34} COMPILE FAILED"); return
    eng._ensure_io(prog)
    with _iosurface_view(prog._in_surf, (in_c, S), np.float16) as d:
        d[:] = (rng.standard_normal((in_c, S)) * 0.3).astype(np.float16)
    for _ in range(3): eng.submit(prog, procedure_index=0)
    t0 = time.perf_counter(); N = 20
    for _ in range(N): eng.submit(prog, procedure_index=0)
    print(f"  {name:34} {(time.perf_counter()-t0)/N*1e3:8.3f} ms")

keys_decl = {
 "mlp_only (baseline)":        [("gu", (2*I, H)), ("dn", (H, I))],
 "mlp, wide input + slice":    [("gu", (2*I, H)), ("dn", (H, I))],
 "outproj + add + mlp":        [("o", (H, Dc)), ("gu", (2*I, H)), ("dn", (H, I))],
}
run("mlp_only (baseline)", H, MLPBODY.replace("NIN", "x"),
    ["gu.bin","gus.bin","dn.bin","dns.bin"])
run("mlp, wide input + slice", IN, f'''
    tensor<fp16, [1, {H}, 1, {S}]> res = slice_by_index(begin=tensor<int32, [4]>([0,{Dc},0,0]), end=tensor<int32, [4]>([1,{IN},1,{S}]), x=x)[name=string("res")];
''' + MLPBODY.replace("NIN", "res"), ["gu.bin","gus.bin","dn.bin","dns.bin"])
run("outproj + add + mlp", IN, f'''
    tensor<fp16, [1, {Dc}, 1, {S}]> core = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{Dc},1,{S}]), x=x)[name=string("core")];
    tensor<fp16, [1, {H}, 1, {S}]> res = slice_by_index(begin=tensor<int32, [4]>([0,{Dc},0,0]), end=tensor<int32, [4]>([1,{IN},1,{S}]), x=x)[name=string("res")];
    tensor<fp16, [1, {H}, 1, {S}]> r = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=ow_, x=core)[name=string("outp")];
    tensor<fp16, [1, {H}, 1, {S}]> h = add(x=res, y=r)[name=string("h")];
'''.replace("ow_", "ow") + MLPBODY.replace("NIN", "h"),
    ["o.bin","os.bin","gu.bin","gus.bin","dn.bin","dns.bin"])
