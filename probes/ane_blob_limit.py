"""Does the 0x50004 load failure track blob COUNT rather than bytes?

Same total weight bytes per program, split across K separate blob files.
If the reachable program count falls as K rises, packing every program's
weights into one file with offsets is the fix.
"""
import os, sys, io, contextlib, numpy as np
sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine

S, In, TOTAL_O = 32, 5120, 16384

def make(eng, K):
    """One program whose weights are split across K conv blobs."""
    O = TOTAL_O // K
    blobs, decl, body, prev = {}, [], [], "x"
    for i in range(K):
        W = (np.random.default_rng(i).standard_normal((O, In)) * 0.02).astype(np.float32)
        sc = np.abs(W).max(axis=1, keepdims=True) / 7
        q = np.clip(np.rint(W / np.where(sc == 0, 1, sc)), -8, 7).astype(np.int8)
        f = (q.reshape(-1).astype(np.uint8) & 0x0F)
        blobs[f"w{i}.bin"] = (f[0::2] | (f[1::2] << 4)).tobytes()
        blobs[f"w{i}s.bin"] = sc.astype(np.float16).tobytes()
        decl.append(
            f'    tensor<int4, [{O}, {In}, 1, 1]> q{i} = const()[name=string("q{i}"), val=tensor<int4, [{O}, {In}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w{i}.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{O}, 1, 1, 1]> s{i} = const()[name=string("s{i}"), val=tensor<fp16, [{O}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w{i}s.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{O}, {In}, 1, 1]> w{i} = constexpr_blockwise_shift_scale(data=q{i}, scale=s{i})[name=string("d{i}")];')
        body.append(f'    tensor<fp16, [1, {O}, 1, {S}]> c{i} = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w{i}, x=x)[name=string("c{i}")];')
    # sum the pieces so every blob is live
    acc = "c0"
    for i in range(1, K):
        body.append(f'    tensor<fp16, [1, {O}, 1, {S}]> a{i} = add(x={acc}, y=c{i})[name=string("a{i}")];')
        acc = f"a{i}"
    body.append(f'    tensor<fp16, [1, {O}, 1, {S}]> y = identity(x={acc})[name=string("y")];')
    mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {In}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
{chr(10).join(decl)}
{chr(10).join(body)}
  }} -> (y);
}}
"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        prog = eng.compile_multiproc(mil, blobs, In, O, S)
    return prog, sum(len(b) for b in blobs.values())

for K in (2, 6, 12):
    eng = AneEngine()
    keep, total = [], 0
    while len(keep) < 260:
        prog, nb = make(eng, K)
        if prog is None:
            break
        eng._ensure_io(prog)
        keep.append(prog); total += nb
    why = "hit cap" if len(keep) >= 260 else "LOAD FAILED"
    print(f"  {2*K:>3} blobs/program: {len(keep):>4} programs, {total/1e9:6.2f} GB  ({why})",
          flush=True)
    del keep, eng
