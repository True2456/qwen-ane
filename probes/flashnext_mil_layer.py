"""The whole GDN layer in one MIL program, int8 projections.

Composes the three verified stages: hyper mixer (stage 3), front (stage 1),
GDN core (stage 2), recombine, second mixer. Diffed against `MultiTokenStep`
at k=1, which is the graph decode runs today.

Inputs (alphabetical, which is how surfaces bind):
    a_x      [1, HC_W, 1, S]      residual stream
    b_conv   [1, 3*QKV, 1, S]     conv cache
    c_param  [1, 3*HV, 1, DK]     gamma | dt | norm_w
    d_hcn    [1, 640, 1, 32]      attn hc_n | mlp hc_n
    e_state  [1, HV, DV, DK]      recurrent state
Outputs (alphabetical):
    v_mixed  [1, H, 1, S]
    w_hyper  [1, HC_W, 1, S]
    x_inj    [1, HC, 1, S]
    y_conv   [1, 3*QKV, 1, S]
    z_state  [1, HV, DV, DK]
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
    _load_layer, H, I, HC, HC_W, HK, HV, DK, DV, QKV, GDN_Y, IN_O, SEQ_DEFAULT,
)
from flashnext_pure_step import FlashNextGatedMix, MIX_H  # noqa: E402
from flashnext_multitoken_step import MultiTokenStep  # noqa: E402

S = SEQ_DEFAULT
NH = QKV // DK
B: list[str] = []


def emit(line: str) -> None:
    B.append("    " + line)


def sl(name, src, c0, c1, oc, d2=1, d3=S, o2=1, o3=None):
    o3 = S if o3 is None else o3
    emit(f'tensor<int32, [4]> {name}b = const()[name=string("{name}b"), val=tensor<int32, [4]>([0,{c0},0,0])];')
    emit(f'tensor<int32, [4]> {name}e = const()[name=string("{name}e"), val=tensor<int32, [4]>([1,{c1},{d2},{d3}])];')
    emit(f'tensor<fp16, [1, {oc}, {o2}, {o3}]> {name} = slice_by_index(x={src}, begin={name}b, end={name}e, begin_mask=mm, end_mask=mm)[name=string("{name}")];')


def mixer(pfx, src, hcn_var, offs, out_mixed, out_inj):
    for i in range(HC):
        sl(f"{pfx}g{i}", src, i * H, (i + 1) * H, H)
        emit(f'tensor<fp16, [1, {H}, 1, {S}]> {pfx}p{i} = mul(x={pfx}g{i}, y={pfx}g{i})[name=string("{pfx}p{i}")];')
        emit(f'tensor<fp16, [1, 1, 1, {S}]> {pfx}m{i} = reduce_mean(x={pfx}p{i}, axes=ac, keep_dims=kd)[name=string("{pfx}m{i}")];')
        emit(f'tensor<fp16, [1, 1, 1, {S}]> {pfx}e{i} = add(x={pfx}m{i}, y=eps)[name=string("{pfx}e{i}")];')
        emit(f'tensor<fp16, [1, 1, 1, {S}]> {pfx}r{i} = pow(x={pfx}e{i}, y=mh)[name=string("{pfx}r{i}")];')
        emit(f'tensor<fp16, [1, {H}, 1, {S}]> {pfx}n{i} = mul(x={pfx}g{i}, y={pfx}r{i})[name=string("{pfx}n{i}")];')
    emit(f'tensor<fp16, [1, {HC_W}, 1, {S}]> {pfx}nc = concat(values=({pfx}n0, {pfx}n1, {pfx}n2, {pfx}n3), axis=int32(1), interleave=bool(false))[name=string("{pfx}nc")];')
    emit(f'tensor<fp16, [1, {HC_W}, 1, {S}]> {pfx}nn = mul(x={pfx}nc, y={hcn_var})[name=string("{pfx}nn")];')
    for nm, o, ci, co in ((f"{pfx}WD", offs["down"], HC_W, MIX_H),
                          (f"{pfx}WU", offs["up"], MIX_H, HC_W),
                          (f"{pfx}WI", offs["inj"], HC_W, HC)):
        emit(f'tensor<fp16, [{co}, {ci}, 1, 1]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [{co}, {ci}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({o})))];')
    emit(f'tensor<fp16, [1, {MIX_H}, 1, {S}]> {pfx}dv = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight={pfx}WD, x={pfx}nn)[name=string("{pfx}dv")];')
    emit(f'tensor<fp16, [1, {MIX_H}, 1, {S}]> {pfx}dn = mul(x={pfx}dv, y=ivh)[name=string("{pfx}dn")];')
    emit(f'tensor<fp16, [1, {MIX_H}, 1, {S}]> {pfx}hf = mul(x={pfx}dn, y=hlf)[name=string("{pfx}hf")];')
    emit(f'tensor<fp16, [1, {MIX_H}, 1, {S}]> {pfx}tt = tanh(x={pfx}hf)[name=string("{pfx}tt")];')
    emit(f'tensor<fp16, [1, {MIX_H}, 1, {S}]> {pfx}hm = mul(x={pfx}hf, y={pfx}tt)[name=string("{pfx}hm")];')
    emit(f'tensor<fp16, [1, {MIX_H}, 1, {S}]> {pfx}gt = add(x={pfx}hf, y={pfx}hm)[name=string("{pfx}gt")];')
    emit(f'tensor<fp16, [1, {HC_W}, 1, {S}]> {pfx}uv = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight={pfx}WU, x={pfx}gt)[name=string("{pfx}uv")];')
    emit(f'tensor<fp16, [1, {HC_W}, 1, {S}]> {pfx}mw = sigmoid(x={pfx}uv)[name=string("{pfx}mw")];')
    for i in range(HC):
        sl(f"{pfx}w{i}", f"{pfx}mw", i * H, (i + 1) * H, H)
        sl(f"{pfx}q{i}", f"{pfx}nn", i * H, (i + 1) * H, H)
        emit(f'tensor<fp16, [1, {H}, 1, {S}]> {pfx}t{i} = mul(x={pfx}w{i}, y={pfx}q{i})[name=string("{pfx}t{i}")];')
    emit(f'tensor<fp16, [1, {H}, 1, {S}]> {pfx}s1 = add(x={pfx}t0, y={pfx}t1)[name=string("{pfx}s1")];')
    emit(f'tensor<fp16, [1, {H}, 1, {S}]> {pfx}s2 = add(x={pfx}s1, y={pfx}t2)[name=string("{pfx}s2")];')
    emit(f'tensor<fp16, [1, {H}, 1, {S}]> {pfx}s3 = add(x={pfx}s2, y={pfx}t3)[name=string("{pfx}s3")];')
    emit(f'tensor<fp16, [1, {H}, 1, {S}]> {out_mixed} = mul(x={pfx}s3, y=ivh)[name=string("{out_mixed}")];')
    emit(f'tensor<fp16, [1, {HC}, 1, {S}]> {pfx}iv = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight={pfx}WI, x={pfx}nn)[name=string("{pfx}iv")];')
    emit(f'tensor<fp16, [1, {HC}, 1, {S}]> {pfx}ih = mul(x={pfx}iv, y=ivh)[name=string("{pfx}ih")];')
    emit(f'tensor<fp16, [1, {HC}, 1, {S}]> {pfx}ig = sigmoid(x={pfx}ih)[name=string("{pfx}ig")];')
    emit(f'tensor<fp16, [1, {HC}, 1, {S}]> {out_inj} = mul(x={pfx}ig, y=two)[name=string("{out_inj}")];')


def build_mil(offs):
    B.clear()
    for c, v in (("mm", 'tensor<bool, [4]>([false,false,false,false])'),):
        emit(f'tensor<bool, [4]> {c} = const()[name=string("{c}"), val={v}];')
    for c, v in (("eps", 0.000001), ("ivh", 0.25), ("hlf", 0.5), ("two", 2.0),
                 ("mh", -0.5), ("nho", -1.0), ("qsc", DK ** -0.5)):
        emit(f'fp16 {c} = const()[name=string("{c}"), val=fp16({v})];')
    emit('tensor<int32, [1]> ac = const()[name=string("ac"), val=tensor<int32, [1]>([1])];')
    emit('tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([-1])];')
    emit('bool kd = const()[name=string("kd"), val=bool(true)];')
    emit('tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];')
    emit('string pt = const()[name=string("pt"), val=string("valid")];')
    emit('tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];')
    emit('tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];')
    emit('tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];')
    emit('int32 gr = const()[name=string("gr"), val=int32(1)];')
    emit(f'int32 gq = const()[name=string("gq"), val=int32({QKV})];')
    # hc_n carriers
    emit(f'tensor<int32, [4]> hs = const()[name=string("hs"), val=tensor<int32, [4]>([1,{HC_W},1,1])];')
    sl("hca", "d_hcn", 0, 320, 320, 1, 32, 1, 32)
    sl("hcm", "d_hcn", 320, 640, 320, 1, 32, 1, 32)
    emit(f'tensor<fp16, [1, {HC_W}, 1, 1]> hcnA = reshape(x=hca, shape=hs)[name=string("hcnA")];')
    emit(f'tensor<fp16, [1, {HC_W}, 1, 1]> hcnM = reshape(x=hcm, shape=hs)[name=string("hcnM")];')

    mixer("A", "a_x", "hcnA", offs["attn"], "amixed", "ainj")

    # --- front: in_proj int8 + 4-tap depthwise conv
    emit(f'tensor<int8, [{IN_O}, {H}, 1, 1]> ipd = const()[name=string("ipd"), val=tensor<int8, [{IN_O}, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({offs["in_d"]})))];')
    emit(f'tensor<fp16, [{IN_O}, 1, 1, 1]> ips = const()[name=string("ips"), val=tensor<fp16, [{IN_O}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_scale.bin"), offset=uint64({offs["in_s"]})))];')
    emit(f'tensor<fp16, [{IN_O}, {H}, 1, 1]> WI2 = constexpr_blockwise_shift_scale(data=ipd, scale=ips)[name=string("WI2")];')
    emit(f'tensor<fp16, [1, {IN_O}, 1, {S}]> yin = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WI2, x=amixed)[name=string("yin")];')
    sl("fqkv", "yin", 0, QKV, QKV)
    sl("frest", "yin", QKV, IN_O, IN_O - QKV)
    sl("fc0", "b_conv", 0, QKV, QKV, 1, 1, 1, 1)
    sl("fc1", "b_conv", QKV, 2 * QKV, QKV, 1, 1, 1, 1)
    sl("fc2", "b_conv", 2 * QKV, 3 * QKV, QKV, 1, 1, 1, 1)
    emit(f'tensor<fp16, [1, {QKV}, 1, {S + 3}]> fseq = concat(values=(fc0, fc1, fc2, fqkv), axis=int32(-1), interleave=bool(false))[name=string("fseq")];')
    emit(f'tensor<fp16, [{QKV}, 1, 1, 4]> TAP = const()[name=string("TAP"), val=tensor<fp16, [{QKV}, 1, 1, 4]>(BLOBFILE(path=string("@model_path/weights/weight_scale.bin"), offset=uint64({offs["taps"]})))];')
    emit(f'tensor<fp16, [1, {QKV}, 1, {S}]> fpre = conv(dilations=dl, groups=gq, pad=pd, pad_type=pt, strides=st, weight=TAP, x=fseq)[name=string("fpre")];')
    sl("fs12", "b_conv", QKV, 3 * QKV, 2 * QKV)
    emit(f'tensor<fp16, [1, {3 * QKV}, 1, {S}]> y_conv = concat(values=(fs12, fqkv), axis=int32(1), interleave=bool(false))[name=string("y_conv")];')
    emit(f'tensor<fp16, [1, {IN_O}, 1, {S}]> fyin = concat(values=(fpre, frest), axis=int32(1), interleave=bool(false))[name=string("fyin")];')

    # --- GDN core (slot 0)
    sl("one", "fyin", 0, IN_O, IN_O, 1, 1, 1, 1)
    sl("qk1", "one", 0, QKV, QKV, 1, 1, 1, 1)
    emit(f'tensor<int32, [4]> rs = const()[name=string("rs"), val=tensor<int32, [4]>([1,{NH},1,{DK}])];')
    emit(f'tensor<fp16, [1, {NH}, 1, {DK}]> pk = reshape(x=qk1, shape=rs)[name=string("pk")];')
    emit(f'tensor<fp16, [1, {NH}, 1, {DK}]> hx = mul(x=pk, y=hlf)[name=string("hx")];')
    emit(f'tensor<fp16, [1, {NH}, 1, {DK}]> th = tanh(x=hx)[name=string("th")];')
    emit(f'tensor<fp16, [1, {NH}, 1, {DK}]> hmm = mul(x=hx, y=th)[name=string("hmm")];')
    emit(f'tensor<fp16, [1, {NH}, 1, {DK}]> cc = add(x=hx, y=hmm)[name=string("cc")];')
    sl("qq", "cc", 0, HK, HK, 1, DK, 1, DK)
    sl("kk", "cc", HK, 2 * HK, HK, 1, DK, 1, DK)
    sl("vv", "cc", 2 * HK, NH, HV, 1, DK, 1, DK)
    for nm, src, scale in (("qn", "qq", True), ("kn", "kk", False)):
        emit(f'tensor<fp16, [1, {HK}, 1, {DK}]> {nm}2 = mul(x={src}, y={src})[name=string("{nm}2")];')
        emit(f'tensor<fp16, [1, {HK}, 1, 1]> {nm}s = reduce_sum(x={nm}2, axes=ax, keep_dims=kd)[name=string("{nm}s")];')
        emit(f'tensor<fp16, [1, {HK}, 1, 1]> {nm}e = add(x={nm}s, y=eps)[name=string("{nm}e")];')
        emit(f'tensor<fp16, [1, {HK}, 1, 1]> {nm}r = pow(x={nm}e, y=mh)[name=string("{nm}r")];')
        emit(f'tensor<fp16, [1, {HK}, 1, {DK}]> {nm}m = mul(x={src}, y={nm}r)[name=string("{nm}m")];')
        last = f"{nm}m"
        if scale:
            emit(f'tensor<fp16, [1, {HK}, 1, {DK}]> {nm}g = mul(x={nm}m, y=qsc)[name=string("{nm}g")];')
            last = f"{nm}g"
        emit(f'tensor<fp16, [1, {HK}, 3, {DK}]> {nm}c = concat(values=({last}, {last}, {last}), axis=int32(2), interleave=bool(false))[name=string("{nm}c")];')
        emit(f'tensor<int32, [4]> {nm}sh = const()[name=string("{nm}sh"), val=tensor<int32, [4]>([1,{HV},1,{DK}])];')
        emit(f'tensor<fp16, [1, {HV}, 1, {DK}]> {nm}48 = reshape(x={nm}c, shape={nm}sh)[name=string("{nm}48")];')
    sl("z1", "one", QKV, QKV + GDN_Y, GDN_Y, 1, 1, 1, 1)
    emit(f'tensor<int32, [4]> zs = const()[name=string("zs"), val=tensor<int32, [4]>([1,{HV},1,{DK}])];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DK}]> zz = reshape(x=z1, shape=zs)[name=string("zz")];')
    sl("b1", "one", QKV + GDN_Y, QKV + GDN_Y + HV, HV, 1, 1, 1, 1)
    sl("a1", "one", QKV + GDN_Y + HV, IN_O, HV, 1, 1, 1, 1)
    sl("gam", "c_param", 0, HV, HV, 1, DK, 1, DK)
    sl("dtb", "c_param", HV, 2 * HV, HV, 1, DK, 1, DK)
    sl("nw", "c_param", 2 * HV, 3 * HV, HV, 1, DK, 1, DK)
    emit(f'tensor<fp16, [1, {HV}, 1, {DK}]> ad = add(x=a1, y=dtb)[name=string("ad")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DK}]> an = mul(x=ad, y=nho)[name=string("an")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DK}]> sg = sigmoid(x=an)[name=string("sg")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DK}]> dec = pow(x=sg, y=gam)[name=string("dec")];')
    emit(f'tensor<fp16, [1, {HV}, 1, 1]> bet = sigmoid(x=b1)[name=string("bet")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, {DK}]> st1 = mul(x=e_state, y=dec)[name=string("st1")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, {DK}]> sk = mul(x=st1, y=kn48)[name=string("sk")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, 1]> mem = reduce_sum(x=sk, axes=ax, keep_dims=kd)[name=string("mem")];')
    emit(f'tensor<fp16, [1, {HV}, {DK}, 1]> vt = transpose(x=vv, perm=pm)[name=string("vt")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, 1]> df = sub(x=vt, y=mem)[name=string("df")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, 1]> dl2 = mul(x=df, y=bet)[name=string("dl2")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, {DK}]> dk2 = mul(x=dl2, y=kn48)[name=string("dk2")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, {DK}]> z_state = add(x=st1, y=dk2)[name=string("z_state")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, {DK}]> sq = mul(x=z_state, y=qn48)[name=string("sq")];')
    emit(f'tensor<fp16, [1, {HV}, {DV}, 1]> yv = reduce_sum(x=sq, axes=ax, keep_dims=kd)[name=string("yv")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DV}]> yt = transpose(x=yv, perm=pm)[name=string("yt")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DV}]> y2 = mul(x=yt, y=yt)[name=string("y2")];')
    emit(f'tensor<fp16, [1, {HV}, 1, 1]> ym = reduce_mean(x=y2, axes=ax, keep_dims=kd)[name=string("ym")];')
    emit(f'tensor<fp16, [1, {HV}, 1, 1]> ye = add(x=ym, y=eps)[name=string("ye")];')
    emit(f'tensor<fp16, [1, {HV}, 1, 1]> yr = pow(x=ye, y=mh)[name=string("yr")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DV}]> yn2 = mul(x=yt, y=yr)[name=string("yn2")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DV}]> yw = mul(x=yn2, y=nw)[name=string("yw")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DV}]> zg = sigmoid(x=zz)[name=string("zg")];')
    emit(f'tensor<fp16, [1, {HV}, 1, {DV}]> yo = mul(x=yw, y=zg)[name=string("yo")];')
    emit(f'tensor<int32, [4]> fs = const()[name=string("fs"), val=tensor<int32, [4]>([1,{GDN_Y},1,1])];')
    emit(f'tensor<fp16, [1, {GDN_Y}, 1, 1]> yf = reshape(x=yo, shape=fs)[name=string("yf")];')
    emit(f'tensor<int32, [4]> rp = const()[name=string("rp"), val=tensor<int32, [4]>([1,1,1,{S}])];')
    emit(f'tensor<fp16, [1, {GDN_Y}, 1, {S}]> yx = tile(x=yf, reps=rp)[name=string("yx")];')
    emit(f'tensor<int8, [{H}, {GDN_Y}, 1, 1]> opd = const()[name=string("opd"), val=tensor<int8, [{H}, {GDN_Y}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({offs["out_d"]})))];')
    emit(f'tensor<fp16, [{H}, 1, 1, 1]> ops = const()[name=string("ops"), val=tensor<fp16, [{H}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_scale.bin"), offset=uint64({offs["out_s"]})))];')
    emit(f'tensor<fp16, [{H}, {GDN_Y}, 1, 1]> WO = constexpr_blockwise_shift_scale(data=opd, scale=ops)[name=string("WO")];')
    emit(f'tensor<fp16, [1, {H}, 1, {S}]> attn = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WO, x=yx)[name=string("attn")];')

    # --- recombine: hyper + concat([attn * inj_i])
    for i in range(HC):
        sl(f"ij{i}", "ainj", i, i + 1, 1)
        emit(f'tensor<fp16, [1, {H}, 1, {S}]> rc{i} = mul(x=attn, y=ij{i})[name=string("rc{i}")];')
    emit(f'tensor<fp16, [1, {HC_W}, 1, {S}]> rcat = concat(values=(rc0, rc1, rc2, rc3), axis=int32(1), interleave=bool(false))[name=string("rcat")];')
    emit(f'tensor<fp16, [1, {HC_W}, 1, {S}]> w_hyper = add(x=a_x, y=rcat)[name=string("w_hyper")];')

    mixer("M", "w_hyper", "hcnM", offs["mlp"], "v_mixed", "x_inj")

    # --- shared expert: down(silu(gate(x)) * up(x)) * sigmoid(sgate(x))
    for nm, o, ci, co in (("SG", offs["sh_gate"], H, I), ("SU", offs["sh_up"], H, I),
                          ("SD", offs["sh_down"], I, H), ("SS", offs["sh_sg"], H, 1)):
        emit(f'tensor<fp16, [{co}, {ci}, 1, 1]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [{co}, {ci}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({o})))];')
    emit(f'tensor<fp16, [1, {I}, 1, {S}]> shg = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=SG, x=v_mixed)[name=string("shg")];')
    emit(f'tensor<fp16, [1, {I}, 1, {S}]> shs = sigmoid(x=shg)[name=string("shs")];')
    emit(f'tensor<fp16, [1, {I}, 1, {S}]> she = mul(x=shg, y=shs)[name=string("she")];')
    emit(f'tensor<fp16, [1, {I}, 1, {S}]> shu = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=SU, x=v_mixed)[name=string("shu")];')
    emit(f'tensor<fp16, [1, {I}, 1, {S}]> shm = mul(x=she, y=shu)[name=string("shm")];')
    emit(f'tensor<fp16, [1, {H}, 1, {S}]> shd = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=SD, x=shm)[name=string("shd")];')
    emit(f'tensor<fp16, [1, 1, 1, {S}]> shk = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=SS, x=v_mixed)[name=string("shk")];')
    emit(f'tensor<fp16, [1, 1, 1, {S}]> shq = sigmoid(x=shk)[name=string("shq")];')
    emit(f'tensor<fp16, [1, {H}, 1, {S}]> u_shared = mul(x=shd, y=shq)[name=string("u_shared")];')

    return (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
            f"  func main<ios18>(tensor<fp16, [1, {HC_W}, 1, {S}]> a_x, "
            f"tensor<fp16, [1, {3 * QKV}, 1, {S}]> b_conv, "
            f"tensor<fp16, [1, {3 * HV}, 1, {DK}]> c_param, "
            f"tensor<fp16, [1, 640, 1, 32]> d_hcn, "
            f"tensor<fp16, [1, {HV}, {DV}, {DK}]> e_state) {{\n"
            + "\n".join(B) +
            f"\n  }} -> (u_shared, v_mixed, w_hyper, x_inj, y_conv, z_state);\n}}\n")


def build_layer(w, ref):
    conn = ref.gdn
    dp, sp = E._BlobPacker(), E._BlobPacker()
    offs = {}
    ip = conn.front.in_proj.op.weight.detach().float().numpy().reshape(IN_O, H)
    q, sc = E.quantize_linear_int8(np.ascontiguousarray(ip))
    offs["in_d"] = dp.append(q.reshape(IN_O, H, 1, 1).tobytes()) + 64
    offs["in_s"] = sp.append(np.asarray(sc, np.float16).reshape(IN_O, 1, 1, 1).tobytes()) + 64
    op = conn.tail.out_proj.op.weight.detach().float().numpy().reshape(H, GDN_Y)
    q2, sc2 = E.quantize_linear_int8(np.ascontiguousarray(op))
    offs["out_d"] = dp.append(q2.reshape(H, GDN_Y, 1, 1).tobytes()) + 64
    offs["out_s"] = sp.append(np.asarray(sc2, np.float16).reshape(H, 1, 1, 1).tobytes()) + 64
    taps = np.concatenate([getattr(conn.front, f"tap{i}").detach().float().numpy().reshape(QKV, 1)
                           for i in range(4)], axis=1)
    offs["taps"] = sp.append(np.ascontiguousarray(taps.astype(np.float16).reshape(QKV, 1, 1, 4)).tobytes()) + 64
    for key, mod in (("attn", ref.attn), ("mlp", ref.mlp)):
        d = {}
        for nm, m2, co, ci in (("down", mod.down, MIX_H, HC_W), ("up", mod.up, HC_W, MIX_H),
                               ("inj", mod.inj, HC, HC_W)):
            wt = m2.op.weight.detach().float().numpy().reshape(co, ci, 1, 1)
            d[nm] = dp.append(wt.astype(np.float16).tobytes()) + 64
        offs[key] = d
    from runtime.expert_bank import Mlx4ExpertBank, MLX4_DEFAULT, MlxSafe
    bank = Mlx4ExpertBank(MLX4_DEFAULT)
    sg_, su_, sd_ = bank.shared_fp32(LAYER[0])
    src = MlxSafe(MLX4_DEFAULT)
    sgate_ = np.asarray(src.f32(f"model.layers.{LAYER[0]}.mlp.shared_expert_gate.weight"),
                        np.float32).reshape(1, H)
    src.close()
    for key, wt, co, ci in (("sh_gate", sg_, I, H), ("sh_up", su_, I, H),
                            ("sh_down", sd_, H, I), ("sh_sg", sgate_, 1, H)):
        offs[key] = dp.append(
            np.ascontiguousarray(wt).astype(np.float16).reshape(co, ci, 1, 1).tobytes()) + 64
    files = {"weight_data.bin": dp.getvalue(), "weight_scale.bin": sp.getvalue()}
    param = np.concatenate([
        np.repeat(conn.gamma.detach().float().numpy().reshape(HV, 1), DK, 1),
        np.repeat(conn.dt.detach().float().numpy().reshape(HV, 1), DK, 1),
        conn.tail.norm_w.detach().float().numpy().reshape(HV, DV)], axis=0)
    hcn = np.concatenate([ref.attn.hc_n.detach().float().numpy().reshape(320, 32),
                          ref.mlp.hc_n.detach().float().numpy().reshape(320, 32)], axis=0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            prog = eng.compile_multiproc(build_mil(offs), files, HC_W, H, S,
                                         raw_weight_files=frozenset(files))
        except Exception:
            prog = None
    if prog is None:
        return None
    prog.input_elems = [HC_W * S, 3 * QKV * S, 3 * HV * DK, 640 * 32, HV * DV * DK]
    prog.output_elems = [H * S, H * S, HC_W * S, HC * S, 3 * QKV * S, HV * DV * DK]
    eng._ensure_io(prog)
    return prog, np.ascontiguousarray(param.astype(np.float16)), np.ascontiguousarray(hcn.astype(np.float16))


LAYER = [0]


def main() -> None:
    rng = np.random.default_rng(12)
    x = np.zeros((HC_W, S), np.float16)
    x[:, :1] = (rng.standard_normal((HC_W, 1)) * 0.05).astype(np.float16)
    conv = np.ascontiguousarray((rng.standard_normal((3 * QKV, S)) * 0.02).astype(np.float16))
    state = np.ascontiguousarray((rng.standard_normal((HV, DV, DK)) * 0.02).astype(np.float16))
    param = np.concatenate([
        np.repeat(conn.gamma.detach().float().numpy().reshape(HV, 1), DK, 1),
        np.repeat(conn.dt.detach().float().numpy().reshape(HV, 1), DK, 1),
        conn.tail.norm_w.detach().float().numpy().reshape(HV, DV)], axis=0)
    hcn = np.concatenate([ref.attn.hc_n.detach().float().numpy().reshape(320, 32),
                          ref.mlp.hc_n.detach().float().numpy().reshape(320, 32)], axis=0)

    with torch.no_grad():
        r = ref(torch.from_numpy(x).reshape(1, HC_W, 1, S),
                torch.from_numpy(conv).reshape(1, 3 * QKV, 1, S),
                torch.from_numpy(state).reshape(1, HV, DV, DK))
    r_mixed = r[0].float().numpy().reshape(H, S)[:, :1]
    xr = r[0].float().numpy().reshape(H, S)[:, :1].T
    gg = xr @ sg_.T
    r_shared = (((gg / (1 + np.exp(-gg))) * (xr @ su_.T)) @ sd_.T
                / (1 + np.exp(-(xr @ sgate_.T)))).T
    r_state = r[3].float().numpy().reshape(HV, DV, DK)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(build_mil(offs), files, HC_W, H, S,
                                      raw_weight_files=frozenset(files))
        except Exception as exc:  # noqa: BLE001
            p = None
            buf.write(str(exc))
    if p is None:
        hit = [l for l in buf.getvalue().splitlines() if "rror" in l or "nvalid" in l]
        print(f"  COMPILE FAILED {(hit[-1] if hit else '')[:130]}")
        return
    print("  compiled")
    p.input_elems = [HC_W * S, 3 * QKV * S, 3 * HV * DK, 640 * 32, HV * DV * DK]
    p.output_elems = [H * S, H * S, HC_W * S, HC * S, 3 * QKV * S, HV * DV * DK]
    if not eng._ensure_io(p):
        print("  IO alloc failed")
        return
    for surf, val in zip(p._in_surfs, (x, conv, np.ascontiguousarray(param.astype(np.float16)),
                                       np.ascontiguousarray(hcn.astype(np.float16)), state)):
        with _iosurface_view(surf, val.shape, np.float16) as d:
            np.copyto(d, val)
    if not eng.submit(p, procedure_index=0):
        print("  submit failed")
        return
    with _iosurface_view(p._out_surfs[1], (H, S), np.float16) as o:
        g_mixed = np.array(o, np.float32)[:, :1]
    with _iosurface_view(p._out_surfs[0], (H, S), np.float16) as o:
        g_shared = np.array(o, np.float32)[:, :1]
    with _iosurface_view(p._out_surfs[5], (HV, DV, DK), np.float16) as o:
        g_state = np.array(o, np.float32)

    def rel(a, b_):
        return float(np.linalg.norm(a - b_) / max(np.linalg.norm(b_), 1e-12))
    print(f"  FULL LAYER: mixed rel {rel(g_mixed, r_mixed):.5f}   "
          f"state rel {rel(g_state, r_state):.5f}   "
          f"shared rel {rel(g_shared, r_shared):.5f}", flush=True)
    import time
    for _ in range(5):
        eng.submit(p, procedure_index=0)
    ts = []
    for _ in range(25):
        t0 = time.perf_counter()
        eng.submit(p, procedure_index=0)
        ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts)) * 1e3
    print(f"  MIL int8 layer: {ms:.3f} ms   "
          f"(Core AI fp16 pure_step k=1: 1.938 ms, with shared expert 2.020)",
          flush=True)
    loader.close()


if __name__ == "__main__":
    main()
