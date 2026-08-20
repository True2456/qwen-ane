"""Where is the ANE's real arithmetic ceiling, and how much of it do we use?

docs/PERFORMANCE.md reports "~10 TFLOP/s, flat from S=64 upward" and treats it
as the hardware ceiling.  That figure came from exactly one shape: the 27B
MLP.  M5 Max's ANE is specified at 42 TOPS INT8, which is ~21 TFLOP/s of fp16
MACs if INT8 runs a 2x dual lane.  Three separable questions:

  1. Is ~10 TFLOP/s a hardware ceiling, or a property of that MLP's shape?
  2. Does weight precision change *arithmetic* throughput, or only memory?
     constexpr_blockwise_shift_scale may simply materialise fp16 weights, in
     which case int4/int8 buy footprint and nothing else.
  3. Which of the model's real projections starve the array, and by how much?

Timing only.  Random weights are correct for this purpose and deliberately not
checked for numerics -- see docs/ANE-REFERENCE.md on why random weights lie
about accuracy but not about speed.
"""
import contextlib, io, os, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, AneDynamicLinear, _iosurface_view

# M5 Max ANE: 16 cores, 42 TOPS INT8 (Apple / Wikipedia M5 family table).
# TOPS counts 2 ops per MAC, so 42 TOPS = 21e12 INT8 MACs/s.  A dual-lane
# INT8/FP16 array does half that in fp16 -> ~21 TFLOP/s fp16, ~42 TOPS int8.
PEAK_FP16 = 21.0
PEAK_INT8 = 42.0

eng = AneEngine()
rng = np.random.default_rng(0)


def build(M, H, S, bits=16):
    """One baked [M,H] conv at width S.  Returns a compiled program or None."""
    W = rng.standard_normal((M, H)) * 0.02
    if bits == 16:
        blobs = {"w.bin": W.astype(np.float16).tobytes()}
        wdecl = (f'    tensor<fp16, [{M}, {H}, 1, 1]> w = const()[name=string("w"), '
                 f'val=tensor<fp16, [{M}, {H}, 1, 1]>(BLOBFILE('
                 f'path=string("@model_path/weights/w.bin"), offset=uint64(64)))];')
    else:
        hi = (1 << (bits - 1)) - 1
        s = np.abs(W).max(axis=1, keepdims=True) / hi
        q = np.clip(np.rint(W / np.where(s == 0, 1, s)), -hi - 1, hi).astype(np.int8)
        if bits == 4:
            n = q.reshape(-1).astype(np.uint8) & 0x0F
            payload = (n[0::2] | (n[1::2] << 4)).tobytes()
        else:
            payload = q.tobytes()
        blobs = {"w.bin": payload, "s.bin": s.astype(np.float16).tobytes()}
        wdecl = (
            f'    tensor<int{bits}, [{M}, {H}, 1, 1]> wq = const()[name=string("wq"), '
            f'val=tensor<int{bits}, [{M}, {H}, 1, 1]>(BLOBFILE('
            f'path=string("@model_path/weights/w.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{M}, 1, 1, 1]> sc = const()[name=string("sc"), '
            f'val=tensor<fp16, [{M}, 1, 1, 1]>(BLOBFILE('
            f'path=string("@model_path/weights/s.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{M}, {H}, 1, 1]> w = constexpr_blockwise_shift_scale('
            f'data=wq, scale=sc)[name=string("dq")];')
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> x) {{
{wdecl}
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {M}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd,
        pad_type=pt, strides=st, weight=w, x=x)[name=string("mm")];
  }} -> (y);
}}
'''
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return eng.compile_multiproc(mil, blobs, H, M, S)
        except Exception:
            return None


def run(p, M, H, S, n=15):
    """Median ms for one dispatch, plus a nonzero check (ANE returns silent 0s)."""
    eng._ensure_io(p)
    x = np.ascontiguousarray(rng.standard_normal((H, S)).astype(np.float16))
    with _iosurface_view(p._in_surf, (H, S), np.float16) as dst:
        np.copyto(dst, x)
    eng.submit(p, procedure_index=0)
    with _iosurface_view(p._out_surf, (M, S), np.float16) as o:
        if not np.abs(o).max() > 0:
            return None
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        eng.submit(p, procedure_index=0)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def tflops(M, H, S, ms):
    return 2 * M * H * S / (ms / 1000) / 1e12


print(f"M5 Max ANE reference peak: {PEAK_INT8:.0f} TOPS int8 "
      f"= ~{PEAK_FP16:.0f} TFLOP/s fp16-equivalent\n")

# ---------------------------------------------------------------- 1. shape
print("1. Is ~10 TFLOP/s a ceiling?  fp16 conv, TFLOP/s (% of fp16 peak)")
WIDTHS = (4096, 8192, 16384, 32768)
for H in (2048, 5120):
    print(f"\n  H={H}")
    print("     S  " + "".join(f"{('M=%d' % m):>16}" for m in WIDTHS))
    for S in (64, 128, 256, 512, 1024):
        row = ""
        for M in WIDTHS:
            p = build(M, H, S)
            if p is None:
                row += f"{'rejected':>16}"
                continue
            ms = run(p, M, H, S)
            if ms is None:
                row += f"{'ZERO':>16}"
                continue
            tf = tflops(M, H, S, ms)
            row += f"{tf:>10.1f} ({100*tf/PEAK_FP16:>2.0f}%)"
            del p
        print(f"  {S:>4}  " + row, flush=True)

# ------------------------------------------------------- 2. weight precision
print("\n2. Does weight precision change ARITHMETIC throughput, or only bytes?")
print(f"  {'shape':>22} {'fp16':>10} {'int8':>10} {'int4':>10}   verdict")
for (M, H, S) in ((8192, 5120, 256), (16384, 5120, 512), (32768, 2048, 512)):
    got = {}
    for bits in (16, 8, 4):
        p = build(M, H, S, bits=bits)
        ms = run(p, M, H, S) if p is not None else None
        got[bits] = tflops(M, H, S, ms) if ms else None
        del p
    vals = [got[b] for b in (16, 8, 4) if got[b]]
    spread = (max(vals) / min(vals) - 1) * 100 if len(vals) > 1 else 0
    verdict = ("same path (<10% spread): weights dequantise to fp16"
               if spread < 10 else f"DIFFERS by {spread:.0f}%")
    cells = "".join(f"{got[b]:>10.1f}" if got[b] else f"{'--':>10}"
                    for b in (16, 8, 4))
    print(f"  M={M:<6} H={H:<5} S={S:<4}{cells}   {verdict}", flush=True)

# ------------------------------------------- 3. the model's own projections
print("\n3. The 27B model's real projections, fp16, at prefill width S=512")
REAL = (
    ("mlp gate+up  [34816,5120]", 34816, 5120),
    ("mlp down     [5120,17408]", 5120, 17408),
    ("gdn in_proj  [16480,5120]", 16480, 5120),
    ("attn qkv     [14336,5120]", 14336, 5120),
    ("lm_head/4    [62080,5120]", 62080, 5120),
)
print(f"  {'projection':>28} {'ms':>8} {'TFLOP/s':>9} {'% peak':>7}")
for label, M, H in REAL:
    S = 512
    p = build(M, H, S)
    if p is None:
        print(f"  {label:>28} {'rejected':>8}")
        continue
    ms = run(p, M, H, S)
    if ms is None:
        print(f"  {label:>28} {'ZERO':>8}")
        continue
    tf = tflops(M, H, S, ms)
    print(f"  {label:>28} {ms:>8.3f} {tf:>9.1f} {100*tf/PEAK_FP16:>6.0f}%", flush=True)
    del p
