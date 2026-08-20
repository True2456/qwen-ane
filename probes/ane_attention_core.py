"""Prove the Qwen3.8 full-attention core on the ANE at decode shapes.

The model has 24 query heads, 4 KV heads, and head_dim=256.  KV heads are
repeated six times before this probe; that repeat can be removed in an
integrated program by slicing/broadcasting the four source heads.

Everything is packed into one IOSurface because the small engine wrapper only
binds one input.  The MIL program performs both dynamic matmuls and the stable
softmax.  A host-provided additive mask makes a fixed context bucket usable at
any logical cache length.
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view


eng = AneEngine()
H, HKV, D = 24, 4, 256


def _softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _mil(L, mode):
    # Preserve grouped-query attention instead of physically repeating each
    # KV head six times: q becomes [HKV, H/HKV, D], while K/V are
    # [HKV, L, D].  This is both exact and small enough for long contexts.
    C = H + 2 * HKV * L + 1
    q0, q1 = 0, H
    k0, k1 = q1, q1 + HKV * L
    v0, v1 = k1, k1 + HKV * L
    m0 = v1
    if mode == "direct":
        sm = (
            f'    tensor<fp16, [1, {H}, 1, {L}]> p = '
            'softmax(axis=int32(-1), x=scores)[name=string("p")];'
        )
    else:
        sm = f'''    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([3])];
    tensor<fp16, [1, {H}, 1, 1]> mx = reduce_max(axes=ax, keep_dims=bool(true), x=scores)[name=string("mx")];
    tensor<fp16, [1, {H}, 1, {L}]> centered = sub(x=scores, y=mx)[name=string("centered")];
    tensor<fp16, [1, {H}, 1, {L}]> ex = exp(x=centered)[name=string("ex")];
    tensor<fp16, [1, {H}, 1, 1]> den = reduce_sum(axes=ax, keep_dims=bool(true), x=ex)[name=string("den")];
    tensor<fp16, [1, {H}, 1, {L}]> p = real_div(x=ex, y=den)[name=string("p")];'''
    return C, f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {D}]> x) {{
    tensor<fp16, [1, {H}, 1, {D}]> q4 = slice_by_index(begin=tensor<int32, [4]>([0,{q0},0,0]), end=tensor<int32, [4]>([1,{q1},1,{D}]), x=x)[name=string("q4")];
    tensor<fp16, [1, {HKV*L}, 1, {D}]> kf = slice_by_index(begin=tensor<int32, [4]>([0,{k0},0,0]), end=tensor<int32, [4]>([1,{k1},1,{D}]), x=x)[name=string("kf")];
    tensor<fp16, [1, {HKV*L}, 1, {D}]> vf = slice_by_index(begin=tensor<int32, [4]>([0,{v0},0,0]), end=tensor<int32, [4]>([1,{v1},1,{D}]), x=x)[name=string("vf")];
    tensor<fp16, [1, 1, 1, {L}]> mask = slice_by_index(begin=tensor<int32, [4]>([0,{m0},0,0]), end=tensor<int32, [4]>([1,{m0+1},1,{L}]), x=x)[name=string("mask")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,{HKV},{H//HKV},{D}]), x=q4)[name=string("q")];
    tensor<fp16, [1, {HKV}, {L}, {D}]> k = reshape(shape=tensor<int32, [4]>([1,{HKV},{L},{D}]), x=kf)[name=string("k")];
    tensor<fp16, [1, {HKV}, {L}, {D}]> v = reshape(shape=tensor<int32, [4]>([1,{HKV},{L},{D}]), x=vf)[name=string("v")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {L}]> rawg = matmul(transpose_x=bool(false), transpose_y=bool(true), x=q, y=k)[name=string("rawg")];
    tensor<fp16, [1, {H}, 1, {L}]> raw = reshape(shape=tensor<int32, [4]>([1,{H},1,{L}]), x=rawg)[name=string("raw")];
    fp16 sc = const()[name=string("sc"), val=fp16(0x1.0p-4)];
    tensor<fp16, [1, {H}, 1, {L}]> scaled = mul(x=raw, y=sc)[name=string("scaled")];
    tensor<fp16, [1, {H}, 1, {L}]> scores = add(x=scaled, y=mask)[name=string("scores")];
{sm}
    tensor<fp16, [1, {HKV}, {H//HKV}, {L}]> pg = reshape(shape=tensor<int32, [4]>([1,{HKV},{H//HKV},{L}]), x=p)[name=string("pg")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {D}]> yg = matmul(transpose_x=bool(false), transpose_y=bool(false), x=pg, y=v)[name=string("yg")];
    tensor<fp16, [1, {H}, 1, {D}]> y = reshape(shape=tensor<int32, [4]>([1,{H},1,{D}]), x=yg)[name=string("y")];
  }} -> (y);
}}
// qwen38_attention_L{L}_{mode}
'''


def _chunked_mil(L, block=256):
    """Exact attention using online-softmax statistics over <=256-token blocks."""
    assert L % block == 0
    C = H + 2 * HKV * L + 1
    q0, q1 = 0, H
    k0, k1 = q1, q1 + HKV * L
    v0, v1 = k1, k1 + HKV * L
    m0 = v1
    chunks = L // block
    body = [f'''    tensor<fp16, [1, {H}, 1, {D}]> q4 = slice_by_index(begin=tensor<int32, [4]>([0,{q0},0,0]), end=tensor<int32, [4]>([1,{q1},1,{D}]), x=x)[name=string("q4")];
    tensor<fp16, [1, 1, 1, {L}]> mask = slice_by_index(begin=tensor<int32, [4]>([0,{m0},0,0]), end=tensor<int32, [4]>([1,{m0+1},1,{L}]), x=x)[name=string("mask")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,{HKV},{H//HKV},{D}]), x=q4)[name=string("q")];
    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([3])];
    fp16 sc = const()[name=string("sc"), val=fp16(0x1.0p-4)];''']
    for i in range(chunks):
        a, b = i * block, (i + 1) * block
        kr0, kr1 = k0 + i * HKV * block, k0 + (i + 1) * HKV * block
        vr0, vr1 = v0 + i * HKV * block, v0 + (i + 1) * HKV * block
        body.append(f'''    tensor<fp16, [1, {HKV*block}, 1, {D}]> kf{i} = slice_by_index(begin=tensor<int32, [4]>([0,{kr0},0,0]), end=tensor<int32, [4]>([1,{kr1},1,{D}]), x=x)[name=string("kf{i}")];
    tensor<fp16, [1, {HKV*block}, 1, {D}]> vf{i} = slice_by_index(begin=tensor<int32, [4]>([0,{vr0},0,0]), end=tensor<int32, [4]>([1,{vr1},1,{D}]), x=x)[name=string("vf{i}")];
    tensor<fp16, [1, {HKV}, {block}, {D}]> k{i} = reshape(shape=tensor<int32, [4]>([1,{HKV},{block},{D}]), x=kf{i})[name=string("k{i}")];
    tensor<fp16, [1, {HKV}, {block}, {D}]> v{i} = reshape(shape=tensor<int32, [4]>([1,{HKV},{block},{D}]), x=vf{i})[name=string("v{i}")];
    tensor<fp16, [1, 1, 1, {block}]> mask{i} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,{a}]), end=tensor<int32, [4]>([1,1,1,{b}]), x=mask)[name=string("mask{i}")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {block}]> raw{i} = matmul(transpose_x=bool(false), transpose_y=bool(true), x=q, y=k{i})[name=string("raw{i}")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {block}]> scaled{i} = mul(x=raw{i}, y=sc)[name=string("scaled{i}")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {block}]> score{i} = add(x=scaled{i}, y=mask{i})[name=string("score{i}")];
    tensor<fp16, [1, {HKV}, {H//HKV}, 1]> max{i} = reduce_max(axes=ax, keep_dims=bool(true), x=score{i})[name=string("max{i}")];''')
    body.append('    tensor<fp16, [1, 4, 6, 1]> gmax0 = mul(x=max0, y=fp16(0x1p+0))[name=string("gmax0")];')
    for i in range(1, chunks):
        body.append(f'    tensor<fp16, [1, 4, 6, 1]> gmax{i} = maximum(x=gmax{i-1}, y=max{i})[name=string("gmax{i}")];')
    gm = f"gmax{chunks-1}"
    for i in range(chunks):
        body.append(f'''    tensor<fp16, [1, {HKV}, {H//HKV}, {block}]> centered{i} = sub(x=score{i}, y={gm})[name=string("centered{i}")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {block}]> ex{i} = exp(x=centered{i})[name=string("ex{i}")];
    tensor<fp16, [1, {HKV}, {H//HKV}, 1]> den{i} = reduce_sum(axes=ax, keep_dims=bool(true), x=ex{i})[name=string("den{i}")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {D}]> num{i} = matmul(transpose_x=bool(false), transpose_y=bool(false), x=ex{i}, y=v{i})[name=string("num{i}")];''')
    body.append('    tensor<fp16, [1, 4, 6, 1]> dent0 = mul(x=den0, y=fp16(0x1p+0))[name=string("dent0")];')
    body.append(f'    tensor<fp16, [1, 4, 6, {D}]> numt0 = mul(x=num0, y=fp16(0x1p+0))[name=string("numt0")];')
    for i in range(1, chunks):
        body.append(f'    tensor<fp16, [1, 4, 6, 1]> dent{i} = add(x=dent{i-1}, y=den{i})[name=string("dent{i}")];')
        body.append(f'    tensor<fp16, [1, 4, 6, {D}]> numt{i} = add(x=numt{i-1}, y=num{i})[name=string("numt{i}")];')
    body.append(f'''    tensor<fp16, [1, {HKV}, {H//HKV}, {D}]> yg = real_div(x=numt{chunks-1}, y=dent{chunks-1})[name=string("yg")];
    tensor<fp16, [1, {H}, 1, {D}]> y = reshape(shape=tensor<int32, [4]>([1,{H},1,{D}]), x=yg)[name=string("y")];''')
    return C, f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {D}]> x) {{
{os.linesep.join(body)}
  }} -> (y);
}}
// qwen38_attention_L{L}_chunked{block}
'''


def _stream_mil(B=256):
    """One reusable context-chunk program returning value, max, and exp sum."""
    C = H + 2 * HKV * B + 1
    k0, k1 = H, H + HKV * B
    v0, v1 = k1, k1 + HKV * B
    m0 = v1
    return C, f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {D}]> x) {{
    tensor<fp16, [1, {H}, 1, {D}]> q4 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{H},1,{D}]), x=x)[name=string("q4")];
    tensor<fp16, [1, {HKV*B}, 1, {D}]> kf = slice_by_index(begin=tensor<int32, [4]>([0,{k0},0,0]), end=tensor<int32, [4]>([1,{k1},1,{D}]), x=x)[name=string("kf")];
    tensor<fp16, [1, {HKV*B}, 1, {D}]> vf = slice_by_index(begin=tensor<int32, [4]>([0,{v0},0,0]), end=tensor<int32, [4]>([1,{v1},1,{D}]), x=x)[name=string("vf")];
    tensor<fp16, [1, 1, 1, {B}]> mask = slice_by_index(begin=tensor<int32, [4]>([0,{m0},0,0]), end=tensor<int32, [4]>([1,{m0+1},1,{B}]), x=x)[name=string("mask")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,{HKV},{H//HKV},{D}]), x=q4)[name=string("q")];
    tensor<fp16, [1, {HKV}, {B}, {D}]> k = reshape(shape=tensor<int32, [4]>([1,{HKV},{B},{D}]), x=kf)[name=string("k")];
    tensor<fp16, [1, {HKV}, {B}, {D}]> v = reshape(shape=tensor<int32, [4]>([1,{HKV},{B},{D}]), x=vf)[name=string("v")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {B}]> raw = matmul(transpose_x=bool(false), transpose_y=bool(true), x=q, y=k)[name=string("raw")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {B}]> scaled = mul(x=raw, y=fp16(0x1.0p-4))[name=string("scaled")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {B}]> score = add(x=scaled, y=mask)[name=string("score")];
    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([3])];
    tensor<fp16, [1, {HKV}, {H//HKV}, 1]> mx = reduce_max(axes=ax, keep_dims=bool(true), x=score)[name=string("mx")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {B}]> centered = sub(x=score, y=mx)[name=string("centered")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {B}]> ex = exp(x=centered)[name=string("ex")];
    tensor<fp16, [1, {HKV}, {H//HKV}, 1]> den = reduce_sum(axes=ax, keep_dims=bool(true), x=ex)[name=string("den")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {D}]> num = matmul(transpose_x=bool(false), transpose_y=bool(false), x=ex, y=v)[name=string("num")];
    tensor<fp16, [1, {HKV}, {H//HKV}, {D}]> yg = real_div(x=num, y=den)[name=string("yg")];
    tensor<fp16, [1, {H}, 1, {D}]> y0 = reshape(shape=tensor<int32, [4]>([1,{H},1,{D}]), x=yg)[name=string("y0")];
    tensor<fp16, [1, {H}, 1, 1]> mx4 = reshape(shape=tensor<int32, [4]>([1,{H},1,1]), x=mx)[name=string("mx4")];
    tensor<fp16, [1, {H}, 1, 1]> dn4 = reshape(shape=tensor<int32, [4]>([1,{H},1,1]), x=den)[name=string("dn4")];
    tensor<fp16, [1, {H}, 1, {D}]> zero = mul(x=q4, y=fp16(0x0p+0))[name=string("zero")];
    tensor<fp16, [1, {H}, 1, {D}]> one = add(x=zero, y=fp16(0x1p+0))[name=string("one")];
    tensor<fp16, [1, {H}, 1, {D}]> mw = mul(x=mx4, y=one)[name=string("mw")];
    tensor<fp16, [1, {H}, 1, {D}]> dw = mul(x=dn4, y=one)[name=string("dw")];
    tensor<int32, [8]> py = const()[name=string("py"), val=tensor<int32, [8]>([0,0,0,{2*H},0,0,0,0])];
    tensor<int32, [8]> pm = const()[name=string("pm"), val=tensor<int32, [8]>([0,0,{H},{H},0,0,0,0])];
    tensor<int32, [8]> pd = const()[name=string("pd"), val=tensor<int32, [8]>([0,0,{2*H},0,0,0,0,0])];
    tensor<fp16, [1, {3*H}, 1, {D}]> yp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=py, x=y0)[name=string("yp")];
    tensor<fp16, [1, {3*H}, 1, {D}]> mp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pm, x=mw)[name=string("mp")];
    tensor<fp16, [1, {3*H}, 1, {D}]> dp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pd, x=dw)[name=string("dp")];
    tensor<fp16, [1, {3*H}, 1, {D}]> ym = add(x=yp, y=mp)[name=string("ym")];
    tensor<fp16, [1, {3*H}, 1, {D}]> y = add(x=ym, y=dp)[name=string("y")];
  }} -> (y);
}}
// qwen38_attention_stream_B{B}
'''


def run_streamed(L, valid=None, B=256):
    assert L % B == 0
    valid = valid or L - 7
    C, mil = _stream_mil(B)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        prog = eng.compile_multiproc(mil, {}, C, 3 * H, D)
    if prog is None:
        detail = next((s for s in capture.getvalue().splitlines()
                       if "Error" in s or "FAILED" in s), "")
        print(f"  L={L:<4} streamed   COMPILE FAILED {detail[:120]}")
        return False
    rng = np.random.default_rng(3000 + L)
    q = rng.normal(0, 0.35, (H, 1, D)).astype(np.float32)
    k = rng.normal(0, 0.35, (HKV, L, D)).astype(np.float32)
    v = rng.normal(0, 0.35, (HKV, L, D)).astype(np.float32)
    mask = np.zeros(L, np.float32); mask[valid:] = -1e4
    qg = q.reshape(HKV, H // HKV, D)
    scores = (qg @ k.swapaxes(-1, -2)) * (D ** -0.5) + mask
    ref = (_softmax(scores) @ v).reshape(H, D)
    eng._ensure_io(prog)
    ys, ms, ds = [], [], []
    t0 = time.perf_counter()
    for a in range(0, L, B):
        packed = np.zeros((C, D), np.float16)
        packed[:H] = q.reshape(H, D)
        p = H
        packed[p:p+HKV*B] = k[:, a:a+B].reshape(HKV*B, D); p += HKV*B
        packed[p:p+HKV*B] = v[:, a:a+B].reshape(HKV*B, D); p += HKV*B
        packed[p, :B] = mask[a:a+B].astype(np.float16)
        with _iosurface_view(prog._in_surf, (C, D), np.float16) as dst:
            dst[:] = packed
        if not eng.submit(prog):
            print(f"  L={L:<4} streamed   SUBMIT FAILED")
            return False
        with _iosurface_view(prog._out_surf, (3*H, D), np.float16) as src:
            out = np.array(src, np.float32)
        ys.append(out[:H]); ms.append(out[H:2*H, 0]); ds.append(out[2*H:, 0])
    ms, ds = np.stack(ms), np.stack(ds)
    gm = np.max(ms, axis=0)
    weights = ds * np.exp(ms - gm)
    got = sum(y * w[:, None] for y, w in zip(ys, weights)) / np.sum(weights, axis=0)[:, None]
    ms_call = (time.perf_counter() - t0) * 1e3 / (L // B)
    abs_err = np.max(np.abs(got - ref))
    rel = abs_err / (np.max(np.abs(ref)) + 1e-9)
    ok = np.isfinite(got).all() and rel < 8e-3
    print(f"  L={L:<4} streamed   {'OK' if ok else 'WRONG':<5} "
          f"valid={valid:<4} rel={rel:.4g} abs={abs_err:.4g} {ms_call:.3f} ms/chunk")
    return ok


def run(L, mode, valid=None):
    valid = valid or max(1, L - 7)
    C, mil = _chunked_mil(L) if mode == "chunked" else _mil(L, mode)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        prog = eng.compile_multiproc(mil, {}, C, H, D)
    if prog is None:
        detail = next((s for s in capture.getvalue().splitlines()
                       if "Error" in s or "FAILED" in s), "")
        print(f"  L={L:<4} {mode:<10} COMPILE FAILED {detail[:120]}")
        return False

    rng = np.random.default_rng(1000 + L)
    q = rng.normal(0, 0.35, (H, 1, D)).astype(np.float32)
    k = rng.normal(0, 0.35, (HKV, L, D)).astype(np.float32)
    v = rng.normal(0, 0.35, (HKV, L, D)).astype(np.float32)
    mask = np.zeros((L,), np.float32)
    mask[valid:] = -1e4
    qg = q.reshape(HKV, H // HKV, D)
    scores = (qg @ k.swapaxes(-1, -2)) * (D ** -0.5) + mask
    ref = (_softmax(scores) @ v).reshape(H, 1, D)

    packed = np.zeros((C, D), np.float16)
    packed[:H] = q.reshape(H, D)
    p = H
    if mode == "chunked":
        block = 256
        kk = np.concatenate([k[:, a:a+block].reshape(HKV*block, D)
                             for a in range(0, L, block)], axis=0)
        vv = np.concatenate([v[:, a:a+block].reshape(HKV*block, D)
                             for a in range(0, L, block)], axis=0)
        packed[p:p + HKV*L] = kk; p += HKV*L
        packed[p:p + HKV*L] = vv; p += HKV*L
    else:
        packed[p:p + HKV*L] = k.reshape(HKV*L, D); p += HKV*L
        packed[p:p + HKV*L] = v.reshape(HKV*L, D); p += HKV*L
    packed[p, :L] = mask.astype(np.float16)

    eng._ensure_io(prog)
    with _iosurface_view(prog._in_surf, (C, D), np.float16) as dst:
        dst[:] = packed
    for _ in range(3):
        if not eng.submit(prog):
            print(f"  L={L:<4} {mode:<10} SUBMIT FAILED")
            return False
    n = 30 if L <= 128 else 10
    t0 = time.perf_counter()
    for _ in range(n):
        eng.submit(prog)
    ms = (time.perf_counter() - t0) * 1e3 / n
    with _iosurface_view(prog._out_surf, (H, D), np.float16) as src:
        got = np.array(src, np.float32).reshape(H, 1, D)
    abs_err = np.max(np.abs(got - ref))
    rel = abs_err / (np.max(np.abs(ref)) + 1e-9)
    ok = np.isfinite(got).all() and rel < 2e-2
    print(f"  L={L:<4} {mode:<10} {'OK' if ok else 'WRONG':<5} "
          f"valid={valid:<4} rel={rel:.4g} abs={abs_err:.4g} {ms:.3f} ms")
    return ok


print("Qwen3.8 attention core: dynamic QK^T -> stable softmax -> PV")
results = [run(32, "direct"), run(32, "decomposed"),
           run(128, "direct"), run(256, "direct")]
# Direct attention has a measured 256-token reduction/matmul ceiling.  Keep the
# failing call visible, then prove the exact online-softmax replacement.
run(512, "direct")
run(512, "chunked")  # documents the single-program compiler-complexity ceiling
results.extend([run_streamed(512), run_streamed(1024)])
print("PASS" if all(results) else "FAIL")
