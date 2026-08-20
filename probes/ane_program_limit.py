"""Is the ~126 resident-program ceiling a COUNT limit or a MEMORY limit?

Compiles programs of a chosen weight size until one fails, reporting both the
count reached and the total bytes resident. If tiny programs also stop near 126
the limit is a count; if they go far past it, the limit is memory and shrinking
per-program blobs would buy more slots.
"""
import os, sys, io, contextlib, numpy as np
sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine

S = 32
def make(eng, O, In, tag, uniq=None):
    # uniq varies the MIL TEXT, not just the weights. Without this every
    # program hashes to the same compiled model (ANECCompile is content
    # addressed) and the ANE hands back one resident program over and over --
    # which is how an earlier run of this probe "reached" 500.
    O = O if uniq is None else O - uniq          # every program a distinct shape
    W = (np.random.default_rng(O + In).standard_normal((O, In)) * 0.02).astype(np.float32)
    hi = 7
    sc = np.abs(W).max(axis=1, keepdims=True) / hi
    q = np.clip(np.rint(W / np.where(sc == 0, 1, sc)), -8, 7).astype(np.int8)
    f = (q.reshape(-1).astype(np.uint8) & 0x0F)
    blobs = {"w.bin": (f[0::2] | (f[1::2] << 4)).tobytes(),
             "ws.bin": sc.astype(np.float16).tobytes()}
    decl = (f'    tensor<int4, [{O}, {In}, 1, 1]> wq = const()[name=string("wq"), val=tensor<int4, [{O}, {In}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{O}, 1, 1, 1]> wsc = const()[name=string("wsc"), val=tensor<fp16, [{O}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/ws.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{O}, {In}, 1, 1]> w = constexpr_blockwise_shift_scale(data=wq, scale=wsc)[name=string("dq")];')
    mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {In}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
{decl}
    tensor<fp16, [1, {O}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("y")];
  }} -> (y);
}}
"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return eng.compile_multiproc(mil, blobs, In, O, S), sum(len(b) for b in blobs.values())

WITH_IO = os.environ.get("WITH_IO", "1") == "1"
print(f"  (_ensure_io per program: {WITH_IO})")
for tag, O, In, cap in (("large [16384,5120]", 16384, 5120, 300),):
    eng = AneEngine()
    keep, total = [], 0
    while len(keep) < cap:
        prog, nb = make(eng, O, In, tag, uniq=len(keep))
        if len(keep) and len(keep) % 25 == 0 and prog is not None:
            print(f"    ... {len(keep)} programs, {total/1e9:.1f} GB", flush=True)
        if prog is None:
            break
        if WITH_IO and not eng._ensure_io(prog):
            print(f"    _ensure_io FAILED at program {len(keep)+1}")
            break
        keep.append(prog); total += nb
    why = "hit my cap" if len(keep) >= cap else "COMPILE FAILED"
    print(f"  {tag:20} stopped at {len(keep):>4} programs, {total/1e9:6.2f} GB "
          f"resident  ({why})", flush=True)
    del keep, eng
