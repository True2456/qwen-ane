"""The model's real projections at production precision, and why down_proj dies.

ane_peak_tflops.py measured the real projections in fp16 and found them all at
~33% of peak, with mlp down_proj at 5%.  Production ships int4, so that table
was the wrong precision to draw conclusions from.  This probe re-measures at
int4/int8, verifies that the fast int4 path is arithmetically correct (a fast
wrong kernel would explain the same numbers), and tests whether splitting the
17408-input down_proj along input channels recovers throughput.
"""
import contextlib, io, os, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, AneDynamicLinear, _iosurface_view

PEAK_FP16 = 21.0
eng = AneEngine()
rng = np.random.default_rng(0)


def quant(W, bits):
    hi = (1 << (bits - 1)) - 1
    s = np.abs(W).max(axis=1, keepdims=True) / hi
    s = np.where(s == 0, 1, s)
    q = np.clip(np.rint(W / s), -hi - 1, hi).astype(np.int8)
    return q, s.astype(np.float16), q * s.astype(np.float32)


def decl(M, H, bits, name="w"):
    if bits == 16:
        return (f'    tensor<fp16, [{M}, {H}, 1, 1]> {name} = const()[name=string("{name}"), '
                f'val=tensor<fp16, [{M}, {H}, 1, 1]>(BLOBFILE('
                f'path=string("@model_path/weights/{name}.bin"), offset=uint64(64)))];')
    return (
        f'    tensor<int{bits}, [{M}, {H}, 1, 1]> {name}q = const()[name=string("{name}q"), '
        f'val=tensor<int{bits}, [{M}, {H}, 1, 1]>(BLOBFILE('
        f'path=string("@model_path/weights/{name}.bin"), offset=uint64(64)))];\n'
        f'    tensor<fp16, [{M}, 1, 1, 1]> {name}s = const()[name=string("{name}s"), '
        f'val=tensor<fp16, [{M}, 1, 1, 1]>(BLOBFILE('
        f'path=string("@model_path/weights/{name}s.bin"), offset=uint64(64)))];\n'
        f'    tensor<fp16, [{M}, {H}, 1, 1]> {name} = constexpr_blockwise_shift_scale('
        f'data={name}q, scale={name}s)[name=string("{name}dq")];')


def blobs_for(W, bits, name="w"):
    if bits == 16:
        return {f"{name}.bin": W.astype(np.float16).tobytes()}, W.astype(np.float32)
    q, s, deq = quant(W, bits)
    if bits == 4:
        n = q.reshape(-1).astype(np.uint8) & 0x0F
        payload = (n[0::2] | (n[1::2] << 4)).tobytes()
    else:
        payload = q.tobytes()
    return {f"{name}.bin": payload, f"{name}s.bin": s.tobytes()}, deq


CONV = ('    string pt = const()[name=string("pt"), val=string("valid")];\n'
        '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
        '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
        '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
        '    int32 gr = const()[name=string("gr"), val=int32(1)];')


def build(M, H, S, bits):
    W = rng.standard_normal((M, H)).astype(np.float32) * 0.02
    b, deq = blobs_for(W, bits)
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> x) {{
{decl(M, H, bits)}
{CONV}
    tensor<fp16, [1, {M}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd,
        pad_type=pt, strides=st, weight=w, x=x)[name=string("mm")];
  }} -> (y);
}}
'''
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return eng.compile_multiproc(mil, b, H, M, S), deq
        except Exception:
            return None, None


def build_split(M, H, S, bits, parts):
    """down_proj as `parts` convs over disjoint input channels, summed."""
    if H % parts:
        return None, None
    c = H // parts
    W = rng.standard_normal((M, H)).astype(np.float32) * 0.02
    b, decls, terms, deq = {}, [], [], np.zeros((M, H), np.float32)
    for i in range(parts):
        bi, di = blobs_for(W[:, i*c:(i+1)*c], bits, name=f"w{i}")
        b.update(bi); deq[:, i*c:(i+1)*c] = di
        decls.append(decl(M, c, bits, name=f"w{i}"))
        decls.append(
            f'    tensor<int32, [4]> b{i} = const()[name=string("b{i}"), '
            f'val=tensor<int32, [4]>([0,{i*c},0,0])];\n'
            f'    tensor<int32, [4]> e{i} = const()[name=string("e{i}"), '
            f'val=tensor<int32, [4]>([1,{(i+1)*c},1,{S}])];\n'
            f'    tensor<fp16, [1, {c}, 1, {S}]> x{i} = slice_by_index(begin=b{i}, '
            f'end=e{i}, x=x)[name=string("sl{i}")];\n'
            f'    tensor<fp16, [1, {M}, 1, {S}]> p{i} = conv(dilations=dl, groups=gr, '
            f'pad=pd, pad_type=pt, strides=st, weight=w{i}, x=x{i})[name=string("c{i}")];')
        terms.append(f"p{i}")
    acc = terms[0]
    for i in range(1, parts):
        nxt = f"acc{i}"
        decls.append(f'    tensor<fp16, [1, {M}, 1, {S}]> {nxt} = add(x={acc}, '
                     f'y={terms[i]})[name=string("{nxt}")];')
        acc = nxt
    decls.append(f'    tensor<fp16, [1, {M}, 1, {S}]> y = identity(x={acc})[name=string("id")];')
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> x) {{
{CONV}
{chr(10).join(decls)}
  }} -> (y);
}}
'''
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return eng.compile_multiproc(mil, b, H, M, S), deq
        except Exception:
            return None, None


def measure(p, M, H, S, deq=None, n=11):
    eng._ensure_io(p)
    x = np.ascontiguousarray(rng.standard_normal((H, S)).astype(np.float16))
    with _iosurface_view(p._in_surf, (H, S), np.float16) as dst:
        np.copyto(dst, x)
    eng.submit(p, procedure_index=0)
    rel = None
    with _iosurface_view(p._out_surf, (M, S), np.float16) as o:
        got = np.array(o, np.float32)
    if not np.abs(got).max() > 0:
        return None, None
    if deq is not None:
        ref = deq @ x.astype(np.float32)
        rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        eng.submit(p, procedure_index=0)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], rel


if __name__ == "__main__":
    REAL = (
        ("mlp gate+up  [34816,5120]", 34816, 5120),
        ("mlp down     [5120,17408]", 5120, 17408),
        ("gdn in_proj  [16480,5120]", 16480, 5120),
        ("attn qkv     [14336,5120]", 14336, 5120),
        ("lm_head/4    [62080,5120]", 62080, 5120),
    )

    print("1. Real projections at S=512, TFLOP/s by weight precision")
    print(f"  {'projection':>28} {'fp16':>8} {'int8':>8} {'int4':>8}   {'int4 ms':>8}")
    for label, M, H in REAL:
        S, cells, ms4 = 512, "", None
        for bits in (16, 8, 4):
            p, _ = build(M, H, S, bits)
            if p is None:
                cells += f"{'rej':>8}"; continue
            ms, _ = measure(p, M, H, S)
            if ms is None:
                cells += f"{'ZERO':>8}"
            else:
                cells += f"{2*M*H*S/(ms/1000)/1e12:>8.1f}"
                if bits == 4:
                    ms4 = ms
            del p
        tail = f"{ms4:>8.2f}" if ms4 else f"{'--':>8}"
        print(f"  {label:>28}{cells}   {tail}", flush=True)

    print("\n2. Is the fast int4 path arithmetically correct? (max rel vs dequant ref)")
    for label, M, H in (("attn qkv", 14336, 5120), ("mlp down", 5120, 17408)):
        for bits in (16, 4):
            p, deq = build(M, H, 512, bits)
            if p is None:
                print(f"  {label:>10} int{bits:<3} rejected"); continue
            ms, rel = measure(p, M, H, 512, deq=deq, n=3)
            print(f"  {label:>10} int{bits:<3} rel={rel:.2e}  {ms:.2f} ms", flush=True)
            del p

    print("\n3. down_proj [5120,17408]: does splitting input channels recover it?")
    for bits in (16, 4):
        for parts in (1, 2, 4, 8):
            M, H, S = 5120, 17408, 512
            p, deq = (build(M, H, S, bits) if parts == 1
                      else build_split(M, H, S, bits, parts))
            if p is None:
                print(f"  int{bits:<3} parts={parts:<2} rejected"); continue
            ms, rel = measure(p, M, H, S, deq=deq, n=5)
            if ms is None:
                print(f"  int{bits:<3} parts={parts:<2} ZERO"); del p; continue
            tf = 2*M*H*S/(ms/1000)/1e12
            print(f"  int{bits:<3} parts={parts:<2} {ms:>8.2f} ms  {tf:>5.1f} TFLOP/s "
                  f"({100*tf/PEAK_FP16:>2.0f}% peak)  rel={rel:.2e}", flush=True)
            del p
