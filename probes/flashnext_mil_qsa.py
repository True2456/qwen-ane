"""QSA attention in MIL for decode (slot 0 live), verified vs FlashNextQSADecode.

Decode makes this far simpler than the Core AI graph, which unrolls a per-head
einsum over 24 heads. With one live query slot:

  * group the queries as [1, HKV, HQ/HKV, HD] and matmul against
    [1, HKV, Kv, HD]^T -> exact grouped-query attention with **no head
    expansion** and no 5-D tensors;
  * the mask collapses to [1, 1, 1, Kv], broadcast over the group axis.

Verified against `FlashNextQSADecode` at slot 0.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "probes"))

import runtime.q38_ane_engine as E  # noqa: E402
from runtime.q38_ane_engine import _iosurface_view  # noqa: E402
from ane_w8a8_projection import eng  # noqa: E402
from export_flashnext_coreai import (  # noqa: E402
    FlashNextQSADecode, _load_layer, H, QSA_HQ, QSA_HKV, QSA_HD, QSA_ROTARY,
    QSA_MASK, SEQ_DEFAULT,
)

S = SEQ_DEFAULT
G = QSA_HQ // QSA_HKV          # 12 query heads per kv head
HALF = QSA_ROTARY // 2
KVC = QSA_HKV * QSA_HD
MAXS = 32
KV = MAXS + S                  # cache plus the S new slots (slot 0 live).
# Every I/O last dim must be a multiple of 32: an IOSurface row is padded to
# 64 bytes, so a KV of 33 makes the host and the ANE disagree on the stride.
Bd: list[str] = []


def em(x):
    Bd.append("    " + x)


def sl(nm, src, dims, beg, end):
    em(f'tensor<int32, [4]> {nm}b = const()[name=string("{nm}b"), val=tensor<int32, [4]>([{",".join(map(str,beg))}])];')
    em(f'tensor<int32, [4]> {nm}e = const()[name=string("{nm}e"), val=tensor<int32, [4]>([{",".join(map(str,end))}])];')
    em(f'tensor<fp16, [{", ".join(map(str,dims))}]> {nm} = slice_by_index(x={src}, begin={nm}b, end={nm}e, begin_mask=mm, end_mask=mm)[name=string("{nm}")];')


def rsh(nm, src, dims):
    em(f'tensor<int32, [4]> {nm}s = const()[name=string("{nm}s"), val=tensor<int32, [4]>([{",".join(map(str,dims))}])];')
    em(f'tensor<fp16, [{", ".join(map(str,dims))}]> {nm} = reshape(x={src}, shape={nm}s)[name=string("{nm}")];')


def build(offs):
    Bd.clear()
    em('tensor<bool, [4]> mm = const()[name=string("mm"), val=tensor<bool, [4]>([false,false,false,false])];')
    em('fp16 eps = const()[name=string("eps"), val=fp16(0.000001)];')
    em(f'fp16 scl = const()[name=string("scl"), val=fp16({QSA_HD ** -0.5})];')
    em('fp16 mh = const()[name=string("mh"), val=fp16(-0.5)];')
    em('fp16 nho = const()[name=string("nho"), val=fp16(-1.0)];')
    em('fp16 one = const()[name=string("one"), val=fp16(1.0)];')
    em('tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([-1])];')
    em('bool kd = const()[name=string("kd"), val=bool(true)];')
    em('tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];')
    em('tensor<int32, [4]> pc = const()[name=string("pc"), val=tensor<int32, [4]>([0,3,2,1])];')
    em('string pt = const()[name=string("pt"), val=string("valid")];')
    em('tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];')
    em('tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];')
    em('tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];')
    em('int32 gr = const()[name=string("gr"), val=int32(1)];')
    for nm, o, ci, co in (("WQ", offs["q"], H, QSA_HQ * QSA_HD),
                          ("WG", offs["g"], H, QSA_HQ * QSA_HD),
                          ("WK", offs["k"], H, KVC), ("WV", offs["v"], H, KVC),
                          ("WO", offs["o"], QSA_HQ * QSA_HD, H)):
        em(f'tensor<fp16, [{co}, {ci}, 1, 1]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [{co}, {ci}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({o})))];')
    for nm, o, n in (("QN", offs["qn"], QSA_HD), ("KN", offs["kn"], QSA_HD)):
        em(f'tensor<fp16, [1, 1, 1, {n}]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [1, 1, 1, {n}]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({o})))];')
    # projections, slot 0 only
    sl("h0", "a_h", (1, H, 1, 1), (0, 0, 0, 0), (1, H, 1, 1))
    em(f'tensor<fp16, [1, {QSA_HQ * QSA_HD}, 1, 1]> qflat = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WQ, x=h0)[name=string("qflat")];')
    em(f'tensor<fp16, [1, {QSA_HQ * QSA_HD}, 1, 1]> gflat = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WG, x=h0)[name=string("gflat")];')
    em(f'tensor<fp16, [1, {KVC}, 1, 1]> kp = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WK, x=h0)[name=string("kp")];')
    em(f'tensor<fp16, [1, {KVC}, 1, 1]> vp = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WV, x=h0)[name=string("vp")];')
    rsh("qraw", "qflat", (1, QSA_HQ, 1, QSA_HD))
    rsh("kraw", "kp", (1, QSA_HKV, 1, QSA_HD))
    rsh("vraw", "vp", (1, QSA_HKV, 1, QSA_HD))
    for nm, src, hn, wn in (("qn", "qraw", QSA_HQ, "QN"), ("kn", "kraw", QSA_HKV, "KN")):
        em(f'tensor<fp16, [1, {hn}, 1, {QSA_HD}]> {nm}2 = mul(x={src}, y={src})[name=string("{nm}2")];')
        em(f'tensor<fp16, [1, {hn}, 1, 1]> {nm}m = reduce_mean(x={nm}2, axes=ax, keep_dims=kd)[name=string("{nm}m")];')
        em(f'tensor<fp16, [1, {hn}, 1, 1]> {nm}e = add(x={nm}m, y=eps)[name=string("{nm}e")];')
        em(f'tensor<fp16, [1, {hn}, 1, 1]> {nm}r = pow(x={nm}e, y=mh)[name=string("{nm}r")];')
        em(f'tensor<fp16, [1, {hn}, 1, {QSA_HD}]> {nm}n = mul(x={src}, y={nm}r)[name=string("{nm}n")];')
        em(f'tensor<fp16, [1, {hn}, 1, {QSA_HD}]> {nm}w = mul(x={nm}n, y={wn})[name=string("{nm}w")];')
    # RoPE on the first `rotary` dims
    sl("cs0", "b_cos", (1, HALF, 1, 1), (0, 0, 0, 0), (1, HALF, 1, 1))
    sl("sn0", "c_sin", (1, HALF, 1, 1), (0, 0, 0, 0), (1, HALF, 1, 1))
    em(f'tensor<fp16, [1, 1, 1, {HALF}]> cc = transpose(x=cs0, perm=pc)[name=string("cc")];')
    em(f'tensor<fp16, [1, 1, 1, {HALF}]> ss = transpose(x=sn0, perm=pc)[name=string("ss")];')
    for nm, hn in (("qnw", QSA_HQ), ("knw", QSA_HKV)):
        sl(f"{nm}A", nm, (1, hn, 1, HALF), (0, 0, 0, 0), (1, hn, 1, HALF))
        sl(f"{nm}B", nm, (1, hn, 1, HALF), (0, 0, 0, HALF), (1, hn, 1, QSA_ROTARY))
        sl(f"{nm}R", nm, (1, hn, 1, QSA_HD - QSA_ROTARY), (0, 0, 0, QSA_ROTARY), (1, hn, 1, QSA_HD))
        em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}ac = mul(x={nm}A, y=cc)[name=string("{nm}ac")];')
        em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}bs = mul(x={nm}B, y=ss)[name=string("{nm}bs")];')
        em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}p1 = sub(x={nm}ac, y={nm}bs)[name=string("{nm}p1")];')
        em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}bc = mul(x={nm}B, y=cc)[name=string("{nm}bc")];')
        em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}as = mul(x={nm}A, y=ss)[name=string("{nm}as")];')
        em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}p2 = add(x={nm}bc, y={nm}as)[name=string("{nm}p2")];')
        em(f'tensor<fp16, [1, {hn}, 1, {QSA_HD}]> {nm}f = concat(values=({nm}p1, {nm}p2, {nm}R), axis=int32(-1), interleave=bool(false))[name=string("{nm}f")];')
    # new_k / new_v back to BC1S for the host cache
    # last-dim-1 OUTPUT surfaces come back zero (same constraint as the
    # last-dim-1 input that failed at submit); widen them and read slot 0.
    rsh("nk1", "knwf", (1, KVC, 1, 1))
    rsh("nv1", "vraw", (1, KVC, 1, 1))
    em(f'tensor<int32, [4]> rq = const()[name=string("rq"), val=tensor<int32, [4]>([1,1,1,{S}])];')
    em(f'tensor<fp16, [1, {KVC}, 1, {S}]> y_newk = tile(x=nk1, reps=rq)[name=string("y_newk")];')
    em(f'tensor<fp16, [1, {KVC}, 1, {S}]> z_newv = tile(x=nv1, reps=rq)[name=string("z_newv")];')
    # keys/values: cache (1,KVC,1,MAXS) + this token
    em(f'tensor<fp16, [1, {KVC}, 1, {KV}]> kall = concat(values=(d_kc, y_newk), axis=int32(-1), interleave=bool(false))[name=string("kall")];')
    em(f'tensor<fp16, [1, {KVC}, 1, {KV}]> vall = concat(values=(e_vc, z_newv), axis=int32(-1), interleave=bool(false))[name=string("vall")];')
    rsh("kh", "kall", (1, QSA_HKV, QSA_HD, KV))
    rsh("vh", "vall", (1, QSA_HKV, QSA_HD, KV))
    em(f'tensor<fp16, [1, {QSA_HKV}, {KV}, {QSA_HD}]> kt = transpose(x=kh, perm=pm)[name=string("kt")];')
    em(f'tensor<fp16, [1, {QSA_HKV}, {KV}, {QSA_HD}]> vt = transpose(x=vh, perm=pm)[name=string("vt")];')
    # grouped query attention without expanding heads
    rsh("qgrp", "qnwf", (1, QSA_HKV, G, QSA_HD))
    em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> sc0 = matmul(x=qgrp, y=kt, transpose_x=bool(false), transpose_y=bool(true))[name=string("sc0")];')
    em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> sc1 = mul(x=sc0, y=scl)[name=string("sc1")];')
    em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> sc2 = add(x=sc1, y=f_mask)[name=string("sc2")];')
    em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> pr = softmax(x=sc2, axis=int32(-1))[name=string("pr")];')
    em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {QSA_HD}]> ov = matmul(x=pr, y=vt, transpose_x=bool(false), transpose_y=bool(false))[name=string("ov")];')
    rsh("ovh", "ov", (1, QSA_HQ, 1, QSA_HD))
    rsh("oflat", "ovh", (1, QSA_HQ * QSA_HD, 1, 1))
    # graw is a strided slice (axis-2 index 1 of the q/gate pair); reshaping it
    # directly is "The strides of the Reshape is not valid". Materialize first.
    em(f'tensor<fp16, [1, {QSA_HQ * QSA_HD}, 1, 1]> gs = sigmoid(x=gflat)[name=string("gs")];')
    em(f'tensor<fp16, [1, {QSA_HQ * QSA_HD}, 1, 1]> og = mul(x=oflat, y=gs)[name=string("og")];')
    em(f'tensor<int32, [4]> rp = const()[name=string("rp"), val=tensor<int32, [4]>([1,1,1,{S}])];')
    em(f'tensor<fp16, [1, {QSA_HQ * QSA_HD}, 1, {S}]> ox = tile(x=og, reps=rp)[name=string("ox")];')
    em(f'tensor<fp16, [1, {H}, 1, {S}]> x_out = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WO, x=ox)[name=string("x_out")];')
    return (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
            f"  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> a_h, "
            f"tensor<fp16, [1, {HALF}, 1, {S}]> b_cos, "
            f"tensor<fp16, [1, {HALF}, 1, {S}]> c_sin, "
            f"tensor<fp16, [1, {KVC}, 1, {MAXS}]> d_kc, "
            f"tensor<fp16, [1, {KVC}, 1, {MAXS}]> e_vc, "
            f"tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> f_mask) {{\n"
            + "\n".join(Bd) + f"\n  }} -> (x_out, y_newk, z_newv);\n}}\n")


def main() -> None:
    loader, w = _load_layer(3)
    m = FlashNextQSADecode().eval().half()
    m.load_from_layer(w)
    pk = E._BlobPacker()
    offs = {}
    # q_proj packs [q | gate] per head; split so neither needs a strided slice
    qw = m.q_proj.op.weight.detach().float().numpy().reshape(QSA_HQ, 2 * QSA_HD, H)
    import os as _os
    _sw = _os.environ.get("QSA_SWAP") == "1"
    _a, _b = (qw[:, QSA_HD:], qw[:, :QSA_HD]) if _sw else (qw[:, :QSA_HD], qw[:, QSA_HD:])
    offs["q"] = pk.append(np.ascontiguousarray(
        _a.reshape(QSA_HQ * QSA_HD, H, 1, 1)).astype(np.float16).tobytes()) + 64
    offs["g"] = pk.append(np.ascontiguousarray(
        _b.reshape(QSA_HQ * QSA_HD, H, 1, 1)).astype(np.float16).tobytes()) + 64
    for key, mod, co, ci in (("k", m.k_proj, KVC, H), ("v", m.v_proj, KVC, H),
                             ("o", m.o_proj, H, QSA_HQ * QSA_HD)):
        wt = mod.op.weight.detach().float().numpy().reshape(co, ci, 1, 1)
        offs[key] = pk.append(wt.astype(np.float16).tobytes()) + 64
    offs["qn"] = pk.append(m.q_norm.detach().float().numpy().reshape(1, 1, 1, QSA_HD).astype(np.float16).tobytes()) + 64
    offs["kn"] = pk.append(m.k_norm.detach().float().numpy().reshape(1, 1, 1, QSA_HD).astype(np.float16).tobytes()) + 64
    files = {"weight_data.bin": pk.getvalue()}

    rng = np.random.default_rng(17)
    h = np.zeros((H, S), np.float16)
    h[:, :1] = (rng.standard_normal((H, 1)) * 0.1).astype(np.float16)
    kc = np.ascontiguousarray((rng.standard_normal((KVC, MAXS)) * 0.05).astype(np.float16))
    vc = np.ascontiguousarray((rng.standard_normal((KVC, MAXS)) * 0.05).astype(np.float16))
    cos = np.zeros((HALF, S), np.float16); sin = np.zeros((HALF, S), np.float16)
    cos[:, 0] = np.cos(np.arange(HALF) * 0.01).astype(np.float16)
    sin[:, 0] = np.sin(np.arange(HALF) * 0.01).astype(np.float16)
    off = 5
    mask_ref = np.full((1, MAXS + S, 1, S), QSA_MASK, np.float16)
    mask_ref[:, :off, :, :1] = 0
    mask_ref[:, MAXS:MAXS + 1, :, :1] = 0
    # MIL mask: [1, HKV, G, KV] over cache 0..MAXS-1 plus this token
    mil_mask = np.full((1, QSA_HKV, G, KV), QSA_MASK, np.float16)
    mil_mask[:, :, :, :off] = 0
    mil_mask[:, :, :, MAXS] = 0   # only new slot 0 is live

    import torch as T
    with T.no_grad():
        r_out, r_nk, r_nv = m(T.from_numpy(h).reshape(1, H, 1, S),
                              T.from_numpy(kc).reshape(1, KVC, 1, MAXS),
                              T.from_numpy(vc).reshape(1, KVC, 1, MAXS),
                              T.from_numpy(cos).reshape(1, HALF, 1, S),
                              T.from_numpy(sin).reshape(1, HALF, 1, S),
                              T.from_numpy(mask_ref))
    r_out = r_out.float().numpy().reshape(H, S)[:, :1]
    r_nk = r_nk.float().numpy().reshape(KVC, S)[:, :1]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(build(offs), files, H, H, S,
                                      raw_weight_files=frozenset(files))
        except Exception as exc:  # noqa: BLE001
            p = None; buf.write(str(exc))
    if p is None:
        hit = [l for l in buf.getvalue().splitlines() if "rror" in l or "nvalid" in l]
        print(f"  COMPILE FAILED {(hit[-1] if hit else '')[:130]}")
        return
    print("  compiled")
    p.input_elems = [H * S, HALF * S, HALF * S, KVC * MAXS, KVC * MAXS,
                     QSA_HKV * G * KV]
    p.output_elems = [H * S, KVC * S, KVC * S]
    if not eng._ensure_io(p):
        print("  IO alloc failed"); return
    for surf, val in zip(p._in_surfs, (h, cos, sin, kc, vc,
                                       np.ascontiguousarray(mil_mask.reshape(QSA_HKV, G, KV)))):
        with _iosurface_view(surf, val.shape, np.float16) as d:
            np.copyto(d, val)
    if not eng.submit(p, procedure_index=0):
        print("  submit failed"); return
    with _iosurface_view(p._out_surfs[0], (H, S), np.float16) as o:
        g_out = np.array(o, np.float32)[:, :1]
    with _iosurface_view(p._out_surfs[1], (KVC, S), np.float16) as o:
        g_nk = np.array(o, np.float32)[:, :1]
    with _iosurface_view(p._out_surfs[2], (KVC, S), np.float16) as o:
        g_nv = np.array(o, np.float32)[:, :1]
    r_nv = r_nv.float().numpy().reshape(KVC, S)[:, :1]
    def _r(a, b_):
        return float(np.linalg.norm(a - b_) / max(np.linalg.norm(b_), 1e-12))
    print(f"    surf1 vs new_k {_r(g_nk, r_nk):.5f}  vs new_v {_r(g_nk, r_nv):.5f}")
    print(f"    surf2 vs new_k {_r(g_nv, r_nk):.5f}  vs new_v {_r(g_nv, r_nv):.5f}")
    kp_ref = (m.k_proj.op.weight.detach().float().numpy().reshape(KVC, H)
              @ h[:, :1].astype(np.float32))
    print(f"    surf1 vs raw k_proj (no norm/rope) {_r(g_nk, kp_ref):.5f}")

    def rel(a, b_):
        return float(np.linalg.norm(a - b_) / max(np.linalg.norm(b_), 1e-12))
    print(f"  QSA MIL: out rel {rel(g_out, r_out):.5f}   new_k rel {rel(g_nk, r_nk):.5f}",
          flush=True)
    import time
    for _ in range(5):
        eng.submit(p, procedure_index=0)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); eng.submit(p, procedure_index=0); ts.append(time.perf_counter() - t0)
    print(f"  QSA MIL: {np.median(ts)*1e3:.3f} ms  (Core AI QSA in-loop ~2.96 ms/layer)",
          flush=True)
    loader.close()


if __name__ == "__main__":
    main()
