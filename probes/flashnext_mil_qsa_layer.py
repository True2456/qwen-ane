"""The QSA layer MIL programs — the sibling of `flashnext_mil_layer`.

The front program supplies the attention mixed/inject inputs and indexer
projection. The main program runs grouped-query attention over a host-held KV
window, recombine, the second mixer, and the shared expert. Use MilQsaLayer to
stage both programs; running this module invokes the checked NumPy oracle.

Decode collapses the attention. With one live query slot the Core AI graph's
per-head einsum over 24 heads becomes a single grouped matmul:

    q  [1, HKV, G, HD]  x  k [1, HKV, KV, HD]^T  ->  [1, HKV, G, KV]

with no head expansion and no rank-5 tensor. The mask broadcasts over the group
axis. KV is `KVM + S`, and every I/O last dim must stay a multiple of 32: an
IOSurface row is padded to 64 bytes, so a 33-wide key axis makes the host and
the ANE disagree about the stride and the result is silently wrong.

Inputs (alphabetical, which is how surfaces bind):
    a_x      [1, HC_W, 1, S]        residual stream
    b_cos    [1, ROT/2, 1, S]       RoPE
    c_sin    [1, ROT/2, 1, S]
    d_hcn    [1, 640, 1, 32]        attn hc_n | mlp hc_n
    e_kc     [1, KVC, 1, KVM]       selected keys
    f_vc     [1, KVC, 1, KVM]       selected values
    g_mask   [1, 1, G*K, KVM + S]  broadcasts over the kv-head axis
    h_mixed  [1, H, 1, S]          required front-program mixed input
    i_inj    [1, HC, 1, S]         required front-program inject input
Outputs (alphabetical):
    t_newk   [1, KVC, 1, S]         slot 0 live, tiled (a last-dim-1 output
    u_shared [1, H, 1, S]           surface comes back zero)
    v_mixed  [1, H, 1, S]
    w_hyper  [1, HC_W, 1, S]
    x_inj    [1, HC, 1, S]
    y_newv   [1, KVC, 1, S]
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import runtime.q38_ane_engine as E  # noqa: E402
from ane_w8a8_projection import eng  # noqa: E402
from export_flashnext_coreai import (  # noqa: E402
    H, I, HC, HC_W, SEQ_DEFAULT, QSA_HQ, QSA_HKV, QSA_HD, QSA_ROTARY,
)
import flashnext_mil_layer as ML  # noqa: E402
from flashnext_pure_step import MIX_H  # noqa: E402

S = SEQ_DEFAULT
G = QSA_HQ // QSA_HKV
HALF = QSA_ROTARY // 2
KVC = QSA_HKV * QSA_HD
QW = QSA_HQ * QSA_HD

LAYER = [3]


#: One loader over the bf16 checkpoint, shared by every layer build.
#:
#: The shared expert is baked from the 4-bit checkpoint's dequantised 8-bit
#: copy, which is 0.9% away from the bf16 source by weight and moves the
#: layer's shared output by 1.7%. Taking it from bf16 instead is measurably
#: *worse*: 1.852451 against 1.845533 scoring 1024 tokens after a 512-token
#: prefill. The quantised checkpoint is calibrated and the raw source is not,
#: and the calibrated copy evidently suits the 4-bit routed bank it is added
#: to. Off by default; MIL_SHARED_FROM_BF16=1 switches it back.
_BF16 = [None]


def bf16_layer(layer: int):
    """Layer tensors straight from the bf16 checkpoint, or None."""
    if os.environ.get("MIL_SHARED_FROM_BF16", "0") != "1":
        return None
    if _BF16[0] is None:
        try:
            from tools.flashnext_reference import FlashNextLoader
            from export_flashnext_coreai import BASE as _B
            _BF16[0] = FlashNextLoader(str(_B))
        except Exception:  # noqa: BLE001
            _BF16[0] = False
    if not _BF16[0]:
        return None
    try:
        w = _BF16[0].layer(int(layer))
        return getattr(w, "tensors", w)
    except Exception:  # noqa: BLE001
        return None
KVM = [256]          # key window width; must be a multiple of 32
KTOK = [1]           # live token slots; must divide S
INT8 = [True]

#: Set to a truthy placeholder to make `build_layer` return its pieces
#: instead of compiling. See `build_program_multi`.
CAPTURE: list[dict | None] = [None]


def _em(line: str) -> None:
    ML.B.append("    " + line)


def _sl4(nm, src, beg, end, dims):
    _em(f'tensor<int32, [4]> {nm}b = const()[name=string("{nm}b"), val=tensor<int32, [4]>([{",".join(map(str,beg))}])];')
    _em(f'tensor<int32, [4]> {nm}e = const()[name=string("{nm}e"), val=tensor<int32, [4]>([{",".join(map(str,end))}])];')
    _em(f'tensor<fp16, [{", ".join(map(str,dims))}]> {nm} = slice_by_index(x={src}, begin={nm}b, end={nm}e, begin_mask=mm, end_mask=mm)[name=string("{nm}")];')


def _rsh(nm, src, dims):
    _em(f'tensor<int32, [4]> {nm}s = const()[name=string("{nm}s"), val=tensor<int32, [4]>([{",".join(map(str,dims))}])];')
    _em(f'tensor<fp16, [{", ".join(map(str,dims))}]> {nm} = reshape(x={src}, shape={nm}s)[name=string("{nm}")];')


def _proj(nm, offs, key, ci, co, src, out):
    """One projection, int8 with a per-output-channel scale or plain fp16."""
    if INT8[0]:
        _em(f'tensor<int8, [{co}, {ci}, 1, 1]> {nm}d = const()[name=string("{nm}d"), val=tensor<int8, [{co}, {ci}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({offs[key + "_d"]})))];')
        _em(f'tensor<fp16, [{co}, 1, 1, 1]> {nm}s = const()[name=string("{nm}s"), val=tensor<fp16, [{co}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_scale.bin"), offset=uint64({offs[key + "_s"]})))];')
        _em(f'tensor<fp16, [{co}, {ci}, 1, 1]> {nm} = constexpr_blockwise_shift_scale(data={nm}d, scale={nm}s)[name=string("{nm}")];')
    else:
        _em(f'tensor<fp16, [{co}, {ci}, 1, 1]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [{co}, {ci}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({offs[key + "_d"]})))];')
    _em(f'tensor<fp16, [1, {co}, 1, {src[1]}]> {out} = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight={nm}, x={src[0]})[name=string("{out}")];')


def build_mil(offs):
    M = KVM[0]
    KV = M + S
    ML.B.clear()
    _em('tensor<bool, [4]> mm = const()[name=string("mm"), val=tensor<bool, [4]>([false,false,false,false])];')
    for c, v in (("eps", 0.000001), ("ivh", 0.25), ("hlf", 0.5), ("two", 2.0),
                 ("mh", -0.5), ("nho", -1.0), ("scl", QSA_HD ** -0.5)):
        _em(f'fp16 {c} = const()[name=string("{c}"), val=fp16({v})];')
    _em('tensor<int32, [1]> ac = const()[name=string("ac"), val=tensor<int32, [1]>([1])];')
    _em('tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([-1])];')
    _em('bool kd = const()[name=string("kd"), val=bool(true)];')
    _em('tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];')
    _em('tensor<int32, [4]> pc = const()[name=string("pc"), val=tensor<int32, [4]>([0,3,2,1])];')
    _em('tensor<int32, [4]> pr = const()[name=string("pr"), val=tensor<int32, [4]>([0,2,3,1])];')
    _em('string pt = const()[name=string("pt"), val=string("valid")];')
    _em('tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];')
    _em('tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];')
    _em('tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];')
    _em('int32 gr = const()[name=string("gr"), val=int32(1)];')
    _em(f'tensor<int32, [4]> hs = const()[name=string("hs"), val=tensor<int32, [4]>([1,{HC_W},1,1])];')
    _sl4("hcm", "d_hcn", (0, 320, 0, 0), (1, 640, 1, 32), (1, 320, 1, 32))
    _em(f'tensor<fp16, [1, {HC_W}, 1, 1]> hcnM = reshape(x=hcm, shape=hs)[name=string("hcnM")];')
    # The attention mixer runs in the front program, whose output the indexer
    # also needs; taking it as an input keeps the total ANE work the same and
    # removes 13 MB of weights from this graph.
    amixed, ainj = "h_mixed", "i_inj"

    # --- QSA core over the K live slots
    kt_ = KTOK[0]
    _sl4("h0", amixed, (0, 0, 0, 0), (1, H, 1, kt_), (1, H, 1, kt_))
    _proj("WQ", offs, "q", H, QW, ("h0", kt_), "qflat")
    _proj("WG", offs, "g", H, QW, ("h0", kt_), "gflat")
    _proj("WK", offs, "k", H, KVC, ("h0", kt_), "kp")
    _proj("WV", offs, "v", H, KVC, ("h0", kt_), "vp")
    for nm, o, n in (("QN", offs["qn"], QSA_HD), ("KN", offs["kn"], QSA_HD)):
        _em(f'tensor<fp16, [1, 1, 1, {n}]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [1, 1, 1, {n}]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({o})))];')
    # (1, heads*hd, 1, k) -> (1, heads, k, hd): reshape splits the channel into
    # head-major rows, the transpose puts the token axis where the RMS and the
    # attention matmul expect it.
    for nm, src, hn in (("qraw", "qflat", QSA_HQ), ("kraw", "kp", QSA_HKV),
                        ("vraw", "vp", QSA_HKV)):
        _rsh(f"{nm}c", src, (1, hn, QSA_HD, kt_))
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {QSA_HD}]> {nm} = transpose(x={nm}c, perm=pm)[name=string("{nm}")];')
    for nm, src, hn, wn in (("qn", "qraw", QSA_HQ, "QN"), ("kn", "kraw", QSA_HKV, "KN")):
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {QSA_HD}]> {nm}2 = mul(x={src}, y={src})[name=string("{nm}2")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, 1]> {nm}m = reduce_mean(x={nm}2, axes=ax, keep_dims=kd)[name=string("{nm}m")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, 1]> {nm}e2 = add(x={nm}m, y=eps)[name=string("{nm}e2")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, 1]> {nm}r = pow(x={nm}e2, y=mh)[name=string("{nm}r")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {QSA_HD}]> {nm}n = mul(x={src}, y={nm}r)[name=string("{nm}n")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {QSA_HD}]> {nm}w = mul(x={nm}n, y={wn})[name=string("{nm}w")];')
    _sl4("cs0", "b_cos", (0, 0, 0, 0), (1, HALF, 1, kt_), (1, HALF, 1, kt_))
    _sl4("sn0", "c_sin", (0, 0, 0, 0), (1, HALF, 1, kt_), (1, HALF, 1, kt_))
    _em(f'tensor<fp16, [1, 1, {kt_}, {HALF}]> cc2 = transpose(x=cs0, perm=pr)[name=string("cc2")];')
    _em(f'tensor<fp16, [1, 1, {kt_}, {HALF}]> ss2 = transpose(x=sn0, perm=pr)[name=string("ss2")];')
    for nm, hn in (("qnw", QSA_HQ), ("knw", QSA_HKV)):
        _sl4(f"{nm}A", nm, (0, 0, 0, 0), (1, hn, kt_, HALF), (1, hn, kt_, HALF))
        _sl4(f"{nm}B", nm, (0, 0, 0, HALF), (1, hn, kt_, QSA_ROTARY), (1, hn, kt_, HALF))
        _sl4(f"{nm}R", nm, (0, 0, 0, QSA_ROTARY), (1, hn, kt_, QSA_HD), (1, hn, kt_, QSA_HD - QSA_ROTARY))
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {HALF}]> {nm}ac = mul(x={nm}A, y=cc2)[name=string("{nm}ac")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {HALF}]> {nm}bs = mul(x={nm}B, y=ss2)[name=string("{nm}bs")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {HALF}]> {nm}p1 = sub(x={nm}ac, y={nm}bs)[name=string("{nm}p1")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {HALF}]> {nm}bc = mul(x={nm}B, y=cc2)[name=string("{nm}bc")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {HALF}]> {nm}as = mul(x={nm}A, y=ss2)[name=string("{nm}as")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {HALF}]> {nm}p2 = add(x={nm}bc, y={nm}as)[name=string("{nm}p2")];')
        _em(f'tensor<fp16, [1, {hn}, {kt_}, {QSA_HD}]> {nm}f = concat(values=({nm}p1, {nm}p2, {nm}R), axis=int32(-1), interleave=bool(false))[name=string("{nm}f")];')
    # back to BC1S for the host cache, widened to S: a last-dim-k output is fine
    # but the host reads a fixed S-wide surface, and slots past k are masked.
    _em(f'tensor<fp16, [1, {QSA_HKV}, {QSA_HD}, {kt_}]> nkt = transpose(x=knwf, perm=pm)[name=string("nkt")];')
    _rsh("nk1", "nkt", (1, KVC, 1, kt_))
    _em(f'tensor<fp16, [1, {QSA_HKV}, {QSA_HD}, {kt_}]> nvt = transpose(x=vraw, perm=pm)[name=string("nvt")];')
    _rsh("nv1", "nvt", (1, KVC, 1, kt_))
    _em(f'tensor<int32, [4]> rq = const()[name=string("rq"), val=tensor<int32, [4]>([1,1,1,{S // kt_}])];')
    _em(f'tensor<fp16, [1, {KVC}, 1, {S}]> t_newk = tile(x=nk1, reps=rq)[name=string("t_newk")];')
    _em(f'tensor<fp16, [1, {KVC}, 1, {S}]> y_newv = tile(x=nv1, reps=rq)[name=string("y_newv")];')
    _em(f'tensor<fp16, [1, {KVC}, 1, {KV}]> kall = concat(values=(e_kc, t_newk), axis=int32(-1), interleave=bool(false))[name=string("kall")];')
    _em(f'tensor<fp16, [1, {KVC}, 1, {KV}]> vall = concat(values=(f_vc, y_newv), axis=int32(-1), interleave=bool(false))[name=string("vall")];')
    _rsh("kh", "kall", (1, QSA_HKV, QSA_HD, KV))
    _rsh("vh", "vall", (1, QSA_HKV, QSA_HD, KV))
    _em(f'tensor<fp16, [1, {QSA_HKV}, {KV}, {QSA_HD}]> kt = transpose(x=kh, perm=pm)[name=string("kt")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {KV}, {QSA_HD}]> vt2 = transpose(x=vh, perm=pm)[name=string("vt2")];')
    # (1, HQ, k, HD) -> (1, HKV, G*k, HD): row g*k+t, so the mask is shared
    # across the G query heads of one kv head and varies only with the token.
    _rsh("qgrp", "qnwf", (1, QSA_HKV, G * kt_, QSA_HD))
    GK = G * kt_
    _em(f'tensor<fp16, [1, {QSA_HKV}, {GK}, {KV}]> sc0 = matmul(x=qgrp, y=kt, transpose_x=bool(false), transpose_y=bool(true))[name=string("sc0")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {GK}, {KV}]> sc1 = mul(x=sc0, y=scl)[name=string("sc1")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {GK}, {KV}]> sc2 = add(x=sc1, y=g_mask)[name=string("sc2")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {GK}, {KV}]> pr2 = softmax(x=sc2, axis=int32(-1))[name=string("pr2")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {GK}, {QSA_HD}]> ov = matmul(x=pr2, y=vt2, transpose_x=bool(false), transpose_y=bool(false))[name=string("ov")];')
    _rsh("ovh", "ov", (1, QSA_HQ, kt_, QSA_HD))
    _em(f'tensor<fp16, [1, {QSA_HQ}, {QSA_HD}, {kt_}]> ovt = transpose(x=ovh, perm=pm)[name=string("ovt")];')
    _rsh("oflat", "ovt", (1, QW, 1, kt_))
    _em(f'tensor<fp16, [1, {QW}, 1, {kt_}]> gs = sigmoid(x=gflat)[name=string("gs")];')
    _em(f'tensor<fp16, [1, {QW}, 1, {kt_}]> og = mul(x=oflat, y=gs)[name=string("og")];')
    _em(f'tensor<fp16, [1, {QW}, 1, {S}]> ox = tile(x=og, reps=rq)[name=string("ox")];')
    _proj("WO", offs, "o", QW, H, ("ox", S), "attn")

    for i in range(HC):
        _sl4(f"ij{i}", ainj, (0, i, 0, 0), (1, i + 1, 1, S), (1, 1, 1, S))
        _em(f'tensor<fp16, [1, {H}, 1, {S}]> rc{i} = mul(x=attn, y=ij{i})[name=string("rc{i}")];')
    _em(f'tensor<fp16, [1, {HC_W}, 1, {S}]> rcat = concat(values=(rc0, rc1, rc2, rc3), axis=int32(1), interleave=bool(false))[name=string("rcat")];')
    _em(f'tensor<fp16, [1, {HC_W}, 1, {S}]> w_hyper = add(x=a_x, y=rcat)[name=string("w_hyper")];')

    ML.mixer("M", "w_hyper", "hcnM", offs["mlp"], "v_mixed", "x_inj")

    for nm, o, ci, co in (("SG", offs["sh_gate"], H, I), ("SU", offs["sh_up"], H, I),
                          ("SD", offs["sh_down"], I, H), ("SS", offs["sh_sg"], H, 1)):
        _em(f'tensor<fp16, [{co}, {ci}, 1, 1]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [{co}, {ci}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({o})))];')
    _em(f'tensor<fp16, [1, {I}, 1, {S}]> shg = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=SG, x=v_mixed)[name=string("shg")];')
    _em(f'tensor<fp16, [1, {I}, 1, {S}]> shs = sigmoid(x=shg)[name=string("shs")];')
    _em(f'tensor<fp16, [1, {I}, 1, {S}]> she = mul(x=shg, y=shs)[name=string("she")];')
    _em(f'tensor<fp16, [1, {I}, 1, {S}]> shu = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=SU, x=v_mixed)[name=string("shu")];')
    _em(f'tensor<fp16, [1, {I}, 1, {S}]> shm = mul(x=she, y=shu)[name=string("shm")];')
    _em(f'tensor<fp16, [1, {H}, 1, {S}]> shd = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=SD, x=shm)[name=string("shd")];')
    _em(f'tensor<fp16, [1, 1, 1, {S}]> shk = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=SS, x=v_mixed)[name=string("shk")];')
    _em(f'tensor<fp16, [1, 1, 1, {S}]> shq = sigmoid(x=shk)[name=string("shq")];')
    _em(f'tensor<fp16, [1, {H}, 1, {S}]> u_shared = mul(x=shd, y=shq)[name=string("u_shared")];')

    return (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
            f"  func main<ios18>(tensor<fp16, [1, {HC_W}, 1, {S}]> a_x, "
            f"tensor<fp16, [1, {HALF}, 1, {S}]> b_cos, "
            f"tensor<fp16, [1, {HALF}, 1, {S}]> c_sin, "
            f"tensor<fp16, [1, 640, 1, 32]> d_hcn, "
            f"tensor<fp16, [1, {KVC}, 1, {M}]> e_kc, "
            f"tensor<fp16, [1, {KVC}, 1, {M}]> f_vc, "
            f"tensor<fp16, [1, 1, {G * KTOK[0]}, {KV}]> g_mask, "
            f"tensor<fp16, [1, {H}, 1, {S}]> h_mixed, "
            f"tensor<fp16, [1, {HC}, 1, {S}]> i_inj) {{\n"
            + "\n".join(ML.B) +
            f"\n  }} -> (t_newk, u_shared, v_mixed, w_hyper, x_inj, y_newv);\n}}\n")


def _pack_proj(dp, sp, offs, key, wt, co, ci):
    wt = np.ascontiguousarray(np.asarray(wt, np.float32).reshape(co, ci))
    if INT8[0]:
        q, sc = E.quantize_linear_int8(wt)
        offs[key + "_d"] = dp.append(q.reshape(co, ci, 1, 1).tobytes()) + 64
        offs[key + "_s"] = sp.append(np.asarray(sc, np.float16).reshape(co, 1, 1, 1).tobytes()) + 64
    else:
        offs[key + "_d"] = dp.append(wt.astype(np.float16).reshape(co, ci, 1, 1).tobytes()) + 64


def build_layer(w, ref, qsa):
    """`ref` is the hyper-mixer carrier (attn/mlp); `qsa` a loaded FlashNextQSADecode."""
    dp, sp = E._BlobPacker(), E._BlobPacker()
    offs = {}
    qw = qsa.q_proj.op.weight.detach().float().numpy().reshape(QSA_HQ, 2 * QSA_HD, H)
    _pack_proj(dp, sp, offs, "q", qw[:, :QSA_HD], QW, H)
    _pack_proj(dp, sp, offs, "g", qw[:, QSA_HD:], QW, H)
    _pack_proj(dp, sp, offs, "k", qsa.k_proj.op.weight.detach().numpy(), KVC, H)
    _pack_proj(dp, sp, offs, "v", qsa.v_proj.op.weight.detach().numpy(), KVC, H)
    _pack_proj(dp, sp, offs, "o", qsa.o_proj.op.weight.detach().numpy(), H, QW)
    offs["qn"] = dp.append(qsa.q_norm.detach().float().numpy().reshape(1, 1, 1, QSA_HD).astype(np.float16).tobytes()) + 64
    offs["kn"] = dp.append(qsa.k_norm.detach().float().numpy().reshape(1, 1, 1, QSA_HD).astype(np.float16).tobytes()) + 64
    for key, mod in (("mlp", ref.mlp),):
        d = {}
        for nm, m2, co, ci in (("down", mod.down, MIX_H, HC_W), ("up", mod.up, HC_W, MIX_H),
                               ("inj", mod.inj, HC, HC_W)):
            wt = m2.op.weight.detach().float().numpy().reshape(co, ci, 1, 1)
            d[nm] = dp.append(wt.astype(np.float16).tobytes()) + 64
        offs[key] = d
    from runtime.expert_bank import Mlx4ExpertBank, MLX4_DEFAULT, MlxSafe
    bank = Mlx4ExpertBank(MLX4_DEFAULT)
    sg_, su_, sd_ = bank.shared_fp32(LAYER[0])
    # The shared expert runs on every token in every layer. Reading it from
    # the 4-bit checkpoint's dequantised 8-bit copy costs 0.9% relative error
    # against the bf16 source, which is right there in the layer weights, and
    # the port's whole per-layer error is only about 3%. Prefer bf16.
    _tens = bf16_layer(LAYER[0]) or getattr(w, "tensors", w)

    _bf16_shared = os.environ.get("MIL_SHARED_FROM_BF16", "0") == "1"

    def _shared(key, fallback):
        hit = _tens.get(f"mlp.shared_expert.{key}.weight") if _bf16_shared \
            else None
        return np.asarray(hit, np.float32) if hit is not None else fallback

    sg_ = _shared('gate_proj', sg_)
    su_ = _shared('up_proj', su_)
    sd_ = _shared('down_proj', sd_)
    src = MlxSafe(MLX4_DEFAULT)
    _sg_key = "mlp.shared_expert_gate.weight"
    sgate_ = (np.asarray(_tens[_sg_key], np.float32)
              if (_bf16_shared and _sg_key in _tens) else
              np.asarray(src.f32(
                  f"model.layers.{LAYER[0]}.mlp.shared_expert_gate.weight"),
                  np.float32)).reshape(1, H)
    src.close()
    for key, wt, co, ci in (("sh_gate", sg_, I, H), ("sh_up", su_, I, H),
                            ("sh_down", sd_, H, I), ("sh_sg", sgate_, 1, H)):
        offs[key] = dp.append(
            np.ascontiguousarray(wt).astype(np.float16).reshape(co, ci, 1, 1).tobytes()) + 64
    files = {"weight_data.bin": dp.getvalue()}
    if sp.getvalue():
        files["weight_scale.bin"] = sp.getvalue()
    hcn = np.concatenate([ref.attn.hc_n.detach().float().numpy().reshape(320, 32),
                          ref.mlp.hc_n.detach().float().numpy().reshape(320, 32)], axis=0)
    src_mil = build_mil(offs)
    if CAPTURE[0] is not None:
        CAPTURE[0] = {
            "mil": src_mil, "files": files,
            "hcn": np.ascontiguousarray(hcn.astype(np.float16)),
        }
        return None, None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            prog = eng.compile_multiproc(src_mil, files, HC_W, H, S,
                                         raw_weight_files=frozenset(files))
        except Exception as exc:  # noqa: BLE001
            prog = None
            buf.write(str(exc))
    if prog is None:
        hit = [l for l in buf.getvalue().splitlines() if "rror" in l or "nvalid" in l]
        raise RuntimeError(f"QSA MIL layer {LAYER[0]} failed: {(hit[-1] if hit else '')[:200]}")
    M = KVM[0]
    prog.input_elems = [HC_W * S, HALF * S, HALF * S, 640 * 32,
                        KVC * M, KVC * M, G * KTOK[0] * (M + S),
                        H * S, HC * S]
    prog.output_elems = [KVC * S, H * S, H * S, HC_W * S, HC * S, KVC * S]
    if not eng._ensure_io(prog):
        raise RuntimeError("QSA MIL: IO alloc failed")
    return prog, np.ascontiguousarray(hcn.astype(np.float16))


# --- the front program: attn mixer + the indexer's projection -----------------
#
# The indexer picks this layer's keys from the mixed state, and that has to
# happen before the layer's own ANE call, so the host used to recompute the
# mixer in fp32 NumPy: 1.2 ms a layer, 14 ms a speculative pass, on top of the
# copy the big graph already computes internally. Splitting the mixer into its
# own program moves that work to the ANE and lets the big graph take `mixed`
# as an input instead of recomputing it, so the total ANE work barely changes.
#
# Outputs (alphabetical): x_mixed [1, H, 1, S], y_inj [1, HC, 1, S],
# z_qk [1, IDX_W, 1, S] — the indexer's raw q|k, which the host then norms,
# RoPEs and pools exactly as before.

IDX_W = 640          # (indexer_n_heads + indexer_kv_heads) * indexer_head_dim


def build_front(offs):
    ML.B.clear()
    _em('tensor<bool, [4]> mm = const()[name=string("mm"), val=tensor<bool, [4]>([false,false,false,false])];')
    for c, v in (("eps", 0.000001), ("ivh", 0.25), ("hlf", 0.5), ("two", 2.0),
                 ("mh", -0.5)):
        _em(f'fp16 {c} = const()[name=string("{c}"), val=fp16({v})];')
    _em('tensor<int32, [1]> ac = const()[name=string("ac"), val=tensor<int32, [1]>([1])];')
    _em('bool kd = const()[name=string("kd"), val=bool(true)];')
    _em('string pt = const()[name=string("pt"), val=string("valid")];')
    _em('tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];')
    _em('tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];')
    _em('tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];')
    _em('int32 gr = const()[name=string("gr"), val=int32(1)];')
    _em(f'tensor<int32, [4]> hs = const()[name=string("hs"), val=tensor<int32, [4]>([1,{HC_W},1,1])];')
    _sl4("hca", "b_hcn", (0, 0, 0, 0), (1, 320, 1, 32), (1, 320, 1, 32))
    _em(f'tensor<fp16, [1, {HC_W}, 1, 1]> hcnA = reshape(x=hca, shape=hs)[name=string("hcnA")];')
    ML.mixer("A", "a_x", "hcnA", offs["attn"], "x_mixed", "y_inj")
    _em(f'tensor<fp16, [{IDX_W}, {H}, 1, 1]> WX = const()[name=string("WX"), val=tensor<fp16, [{IDX_W}, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({offs["idx"]})))];')
    _em(f'tensor<fp16, [1, {IDX_W}, 1, {S}]> z_qk = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WX, x=x_mixed)[name=string("z_qk")];')
    return (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
            f"  func main<ios18>(tensor<fp16, [1, {HC_W}, 1, {S}]> a_x, "
            f"tensor<fp16, [1, 640, 1, 32]> b_hcn) {{\n"
            + "\n".join(ML.B) +
            f"\n  }} -> (x_mixed, y_inj, z_qk);\n}}\n")


def build_front_layer(w, ref):
    dp, sp = E._BlobPacker(), E._BlobPacker()
    offs = {}
    d = {}
    for nm, m2, co, ci in (("down", ref.attn.down, MIX_H, HC_W),
                           ("up", ref.attn.up, HC_W, MIX_H),
                           ("inj", ref.attn.inj, HC, HC_W)):
        wt = m2.op.weight.detach().float().numpy().reshape(co, ci, 1, 1)
        d[nm] = dp.append(wt.astype(np.float16).tobytes()) + 64
    offs["attn"] = d
    qk = np.ascontiguousarray(
        np.asarray(w["self_attn.indexer.index_qk_proj.weight"], np.float32))
    offs["idx"] = dp.append(
        qk.reshape(IDX_W, H, 1, 1).astype(np.float16).tobytes()) + 64
    hcn = np.concatenate([ref.attn.hc_n.detach().float().numpy().reshape(320, 32),
                          np.zeros((320, 32), np.float32)], axis=0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            prog = eng.compile_multiproc(build_front(offs),
                                         {"weight_data.bin": dp.getvalue()},
                                         HC_W, H, S,
                                         raw_weight_files=frozenset({"weight_data.bin"}))
        except Exception as exc:  # noqa: BLE001
            prog = None
            buf.write(str(exc))
    if prog is None:
        hit = [l for l in buf.getvalue().splitlines() if "rror" in l or "nvalid" in l]
        raise RuntimeError(f"QSA front {LAYER[0]} failed: {(hit[-1] if hit else '')[:200]}")
    prog.input_elems = [HC_W * S, 640 * 32]
    prog.output_elems = [H * S, HC * S, IDX_W * S]
    if not eng._ensure_io(prog):
        raise RuntimeError("QSA front: IO alloc failed")
    return prog, np.ascontiguousarray(hcn.astype(np.float16))


class _Ref:
    """The two hyper mixers of one layer, which is all `build_layer` needs."""

    def __init__(self, w):
        from flashnext_pure_step import FlashNextGatedMix
        self.attn = FlashNextGatedMix().eval().half()
        self.attn.load(w, "attn_hyper_connection")
        self.mlp = FlashNextGatedMix().eval().half()
        self.mlp.load(w, "mlp_hyper_connection")



def build_program_multi(w, ref, qsa, specs):
    """One program with multiple procedures sharing baked weights.

    specs: list of (k, m) tuples (or int k using current KVM[0]).
    Returns (prog, hcn) or (None, None).
    """
    caught = []
    for spec in specs:
        if isinstance(spec, (list, tuple)):
            k, m = int(spec[0]), int(spec[1])
        else:
            k, m = int(spec), KVM[0]
        KTOK[0] = k
        KVM[0] = m
        CAPTURE[0] = {}
        try:
            build_layer(w, ref, qsa)
            got = CAPTURE[0]
        finally:
            CAPTURE[0] = None
        if not got or not got.get("mil"):
            return None, None
        caught.append((k, m, got))

    def body(text, name):
        i = text.index("  func main<ios18>")
        j = text.rstrip().rindex("}")
        return text[i:j].replace("func main<ios18>", f"func {name}<ios18>", 1)

    text = (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
            + "".join(body(c["mil"], f"procedure{n:03d}")
                      for n, (_, _, c) in enumerate(caught))
            + "}\n")
    # Weights are identical across widths and key lengths for a given layer
    files = max((c["files"] for _, _, c in caught),
                key=lambda f: sum(len(v) for v in f.values()))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            prog = eng.compile_multiproc(text, files, HC_W, H, S,
                                         raw_weight_files=frozenset(files))
        except Exception as exc:  # noqa: BLE001
            prog = None
            buf.write(str(exc))
    if prog is None:
        if os.environ.get("MIL_VERBOSE") == "2":
            print(buf.getvalue())
        elif os.environ.get("MIL_VERBOSE"):
            hit = [l for l in buf.getvalue().splitlines()
                   if "rror" in l or "nvalid" in l]
            print("  MIL QSA multi: " + (hit[-1] if hit else buf.getvalue()[-400:]))
        return None, None

    M = caught[0][1]
    input_elems = [HC_W * S, HALF * S, HALF * S, 640 * 32, KVC * M, KVC * M]
    mask_indices = []
    for n, (k, m, _) in enumerate(caught):
        mask_indices.append(len(input_elems))
        input_elems.append(G * k * (m + S))
    h_mixed_idx = len(input_elems)
    input_elems.append(H * S)
    i_inj_idx = len(input_elems)
    input_elems.append(HC * S)

    prog.input_elems = input_elems
    prog.proc_in_map = {}
    prog.proc_k = {}
    for n, (k, _, _) in enumerate(caught):
        prog.proc_in_map[n] = [0, 1, 2, 3, 4, 5, mask_indices[n], h_mixed_idx, i_inj_idx]
        prog.proc_k[n] = k

    prog.output_elems = [KVC * S, H * S, H * S, HC_W * S, HC * S, KVC * S]
    if not eng._ensure_io(prog):
        return None, None
    return prog, caught[0][2]["hcn"]


def main() -> None:
    """Use the checked oracle, including the front program's required feeds."""
    from mil_qsa_k_check import main as check
    check()


if __name__ == "__main__":
    main()
