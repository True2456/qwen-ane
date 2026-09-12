"""The whole QSA layer in one MIL program — the sibling of `flashnext_mil_layer`.

Same shape as the GDN layer: hyper mixer, core, recombine, second mixer, shared
expert. Only the core differs — grouped-query attention over a host-held KV
window instead of the gated delta recurrence.

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
    g_mask   [1, 1, 1, KVM + S]     broadcasts over head and group
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
KVM = [256]          # key window width; must be a multiple of 32
INT8 = [True]


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
    _em('string pt = const()[name=string("pt"), val=string("valid")];')
    _em('tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];')
    _em('tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];')
    _em('tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];')
    _em('int32 gr = const()[name=string("gr"), val=int32(1)];')
    _em(f'tensor<int32, [4]> hs = const()[name=string("hs"), val=tensor<int32, [4]>([1,{HC_W},1,1])];')
    _sl4("hca", "d_hcn", (0, 0, 0, 0), (1, 320, 1, 32), (1, 320, 1, 32))
    _sl4("hcm", "d_hcn", (0, 320, 0, 0), (1, 640, 1, 32), (1, 320, 1, 32))
    _em(f'tensor<fp16, [1, {HC_W}, 1, 1]> hcnA = reshape(x=hca, shape=hs)[name=string("hcnA")];')
    _em(f'tensor<fp16, [1, {HC_W}, 1, 1]> hcnM = reshape(x=hcm, shape=hs)[name=string("hcnM")];')

    ML.mixer("A", "a_x", "hcnA", offs["attn"], "amixed", "ainj")

    # --- QSA core, slot 0 live
    _sl4("h0", "amixed", (0, 0, 0, 0), (1, H, 1, 1), (1, H, 1, 1))
    _proj("WQ", offs, "q", H, QW, ("h0", 1), "qflat")
    _proj("WG", offs, "g", H, QW, ("h0", 1), "gflat")
    _proj("WK", offs, "k", H, KVC, ("h0", 1), "kp")
    _proj("WV", offs, "v", H, KVC, ("h0", 1), "vp")
    for nm, o, n in (("QN", offs["qn"], QSA_HD), ("KN", offs["kn"], QSA_HD)):
        _em(f'tensor<fp16, [1, 1, 1, {n}]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [1, 1, 1, {n}]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({o})))];')
    _rsh("qraw", "qflat", (1, QSA_HQ, 1, QSA_HD))
    _rsh("kraw", "kp", (1, QSA_HKV, 1, QSA_HD))
    for nm, src, hn, wn in (("qn", "qraw", QSA_HQ, "QN"), ("kn", "kraw", QSA_HKV, "KN")):
        _em(f'tensor<fp16, [1, {hn}, 1, {QSA_HD}]> {nm}2 = mul(x={src}, y={src})[name=string("{nm}2")];')
        _em(f'tensor<fp16, [1, {hn}, 1, 1]> {nm}m = reduce_mean(x={nm}2, axes=ax, keep_dims=kd)[name=string("{nm}m")];')
        _em(f'tensor<fp16, [1, {hn}, 1, 1]> {nm}e2 = add(x={nm}m, y=eps)[name=string("{nm}e2")];')
        _em(f'tensor<fp16, [1, {hn}, 1, 1]> {nm}r = pow(x={nm}e2, y=mh)[name=string("{nm}r")];')
        _em(f'tensor<fp16, [1, {hn}, 1, {QSA_HD}]> {nm}n = mul(x={src}, y={nm}r)[name=string("{nm}n")];')
        _em(f'tensor<fp16, [1, {hn}, 1, {QSA_HD}]> {nm}w = mul(x={nm}n, y={wn})[name=string("{nm}w")];')
    _sl4("cs0", "b_cos", (0, 0, 0, 0), (1, HALF, 1, 1), (1, HALF, 1, 1))
    _sl4("sn0", "c_sin", (0, 0, 0, 0), (1, HALF, 1, 1), (1, HALF, 1, 1))
    _em(f'tensor<fp16, [1, 1, 1, {HALF}]> cc2 = transpose(x=cs0, perm=pc)[name=string("cc2")];')
    _em(f'tensor<fp16, [1, 1, 1, {HALF}]> ss2 = transpose(x=sn0, perm=pc)[name=string("ss2")];')
    for nm, hn in (("qnw", QSA_HQ), ("knw", QSA_HKV)):
        _sl4(f"{nm}A", nm, (0, 0, 0, 0), (1, hn, 1, HALF), (1, hn, 1, HALF))
        _sl4(f"{nm}B", nm, (0, 0, 0, HALF), (1, hn, 1, QSA_ROTARY), (1, hn, 1, HALF))
        _sl4(f"{nm}R", nm, (0, 0, 0, QSA_ROTARY), (1, hn, 1, QSA_HD), (1, hn, 1, QSA_HD - QSA_ROTARY))
        _em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}ac = mul(x={nm}A, y=cc2)[name=string("{nm}ac")];')
        _em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}bs = mul(x={nm}B, y=ss2)[name=string("{nm}bs")];')
        _em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}p1 = sub(x={nm}ac, y={nm}bs)[name=string("{nm}p1")];')
        _em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}bc = mul(x={nm}B, y=cc2)[name=string("{nm}bc")];')
        _em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}as = mul(x={nm}A, y=ss2)[name=string("{nm}as")];')
        _em(f'tensor<fp16, [1, {hn}, 1, {HALF}]> {nm}p2 = add(x={nm}bc, y={nm}as)[name=string("{nm}p2")];')
        _em(f'tensor<fp16, [1, {hn}, 1, {QSA_HD}]> {nm}f = concat(values=({nm}p1, {nm}p2, {nm}R), axis=int32(-1), interleave=bool(false))[name=string("{nm}f")];')
    _rsh("nk1", "knwf", (1, KVC, 1, 1))
    _rsh("nv1", "vp", (1, KVC, 1, 1))
    _em(f'tensor<int32, [4]> rq = const()[name=string("rq"), val=tensor<int32, [4]>([1,1,1,{S}])];')
    _em(f'tensor<fp16, [1, {KVC}, 1, {S}]> t_newk = tile(x=nk1, reps=rq)[name=string("t_newk")];')
    _em(f'tensor<fp16, [1, {KVC}, 1, {S}]> y_newv = tile(x=nv1, reps=rq)[name=string("y_newv")];')
    _em(f'tensor<fp16, [1, {KVC}, 1, {KV}]> kall = concat(values=(e_kc, t_newk), axis=int32(-1), interleave=bool(false))[name=string("kall")];')
    _em(f'tensor<fp16, [1, {KVC}, 1, {KV}]> vall = concat(values=(f_vc, y_newv), axis=int32(-1), interleave=bool(false))[name=string("vall")];')
    _rsh("kh", "kall", (1, QSA_HKV, QSA_HD, KV))
    _rsh("vh", "vall", (1, QSA_HKV, QSA_HD, KV))
    _em(f'tensor<fp16, [1, {QSA_HKV}, {KV}, {QSA_HD}]> kt = transpose(x=kh, perm=pm)[name=string("kt")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {KV}, {QSA_HD}]> vt2 = transpose(x=vh, perm=pm)[name=string("vt2")];')
    _rsh("qgrp", "qnwf", (1, QSA_HKV, G, QSA_HD))
    _em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> sc0 = matmul(x=qgrp, y=kt, transpose_x=bool(false), transpose_y=bool(true))[name=string("sc0")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> sc1 = mul(x=sc0, y=scl)[name=string("sc1")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> sc2 = add(x=sc1, y=g_mask)[name=string("sc2")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {KV}]> pr = softmax(x=sc2, axis=int32(-1))[name=string("pr")];')
    _em(f'tensor<fp16, [1, {QSA_HKV}, {G}, {QSA_HD}]> ov = matmul(x=pr, y=vt2, transpose_x=bool(false), transpose_y=bool(false))[name=string("ov")];')
    _rsh("ovh", "ov", (1, QSA_HQ, 1, QSA_HD))
    _rsh("oflat", "ovh", (1, QW, 1, 1))
    _em(f'tensor<fp16, [1, {QW}, 1, 1]> gs = sigmoid(x=gflat)[name=string("gs")];')
    _em(f'tensor<fp16, [1, {QW}, 1, 1]> og = mul(x=oflat, y=gs)[name=string("og")];')
    _em(f'tensor<int32, [4]> rp = const()[name=string("rp"), val=tensor<int32, [4]>([1,1,1,{S}])];')
    _em(f'tensor<fp16, [1, {QW}, 1, {S}]> ox = tile(x=og, reps=rp)[name=string("ox")];')
    _proj("WO", offs, "o", QW, H, ("ox", S), "attn")

    for i in range(HC):
        _sl4(f"ij{i}", "ainj", (0, i, 0, 0), (1, i + 1, 1, S), (1, 1, 1, S))
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
            f"tensor<fp16, [1, 1, 1, {KV}]> g_mask) {{\n"
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
    files = {"weight_data.bin": dp.getvalue()}
    if sp.getvalue():
        files["weight_scale.bin"] = sp.getvalue()
    hcn = np.concatenate([ref.attn.hc_n.detach().float().numpy().reshape(320, 32),
                          ref.mlp.hc_n.detach().float().numpy().reshape(320, 32)], axis=0)
    src_mil = build_mil(offs)
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
                        KVC * M, KVC * M, M + S]
    prog.output_elems = [KVC * S, H * S, H * S, HC_W * S, HC * S, KVC * S]
    if not eng._ensure_io(prog):
        raise RuntimeError("QSA MIL: IO alloc failed")
    return prog, np.ascontiguousarray(hcn.astype(np.float16))


class _Ref:
    """The two hyper mixers of one layer, which is all `build_layer` needs."""

    def __init__(self, w):
        from flashnext_pure_step import FlashNextGatedMix
        self.attn = FlashNextGatedMix().eval().half()
        self.attn.load(w, "attn_hyper_connection")
        self.mlp = FlashNextGatedMix().eval().half()
        self.mlp.load(w, "mlp_hyper_connection")


def main() -> None:
    import time

    import torch as T
    from export_flashnext_coreai import _load_layer, FlashNextQSADecode, QSA_MASK
    from flashnext_pure_step import _recombine
    from runtime.expert_bank import Mlx4ExpertBank, MLX4_DEFAULT, MlxSafe

    li = LAYER[0]
    M = KVM[0]
    loader, w = _load_layer(li)
    ref = _Ref(w)
    qsa = FlashNextQSADecode(max_s=M).eval().half()
    qsa.load_from_layer(w)

    rng = np.random.default_rng(12)
    x = np.zeros((HC_W, S), np.float16)
    x[:, :1] = (rng.standard_normal((HC_W, 1)) * 0.05).astype(np.float16)
    kc = np.ascontiguousarray((rng.standard_normal((KVC, M)) * 0.05).astype(np.float16))
    vc = np.ascontiguousarray((rng.standard_normal((KVC, M)) * 0.05).astype(np.float16))
    cos = np.zeros((HALF, S), np.float16)
    sin = np.zeros((HALF, S), np.float16)
    cos[:, 0] = np.cos(np.arange(HALF) * 0.01).astype(np.float16)
    sin[:, 0] = np.sin(np.arange(HALF) * 0.01).astype(np.float16)
    off = 37
    mask_ref = np.full((1, M + S, 1, S), QSA_MASK, np.float16)
    mask_ref[:, :off, :, :1] = 0
    mask_ref[:, M:M + 1, :, :1] = 0
    mil_mask = np.full(M + S, QSA_MASK, np.float16)
    mil_mask[:off] = 0
    mil_mask[M] = 0

    with T.no_grad():
        xt = T.from_numpy(x).reshape(1, HC_W, 1, S)
        mixed_a, hyper_a, inj_a = ref.attn(xt)
        att, r_nk, r_nv = qsa(mixed_a, T.from_numpy(kc).reshape(1, KVC, 1, M),
                              T.from_numpy(vc).reshape(1, KVC, 1, M),
                              T.from_numpy(cos).reshape(1, HALF, 1, S),
                              T.from_numpy(sin).reshape(1, HALF, 1, S),
                              T.from_numpy(mask_ref))
        hyper2 = _recombine(att, hyper_a, inj_a)
        mixed_m, _, inj_m = ref.mlp(hyper2)
    r_mixed = mixed_m.float().numpy().reshape(H, S)[:, :1]
    r_hyper = hyper2.float().numpy().reshape(HC_W, S)[:, :1]
    r_nk = r_nk.float().numpy().reshape(KVC, S)[:, :1]
    r_nv = r_nv.float().numpy().reshape(KVC, S)[:, :1]
    bank = Mlx4ExpertBank(MLX4_DEFAULT)
    sg_, su_, sd_ = bank.shared_fp32(li)
    src = MlxSafe(MLX4_DEFAULT)
    sgate_ = np.asarray(src.f32(f"model.layers.{li}.mlp.shared_expert_gate.weight"),
                        np.float32).reshape(1, H)
    src.close()
    xr = r_mixed.T
    gg = xr @ sg_.T
    r_shared = (((gg / (1 + np.exp(-gg))) * (xr @ su_.T)) @ sd_.T
                / (1 + np.exp(-(xr @ sgate_.T)))).T

    prog, hcn = build_layer(w, ref, qsa)
    loader.close()
    print("  compiled")
    for surf, val in zip(prog._in_surfs, (x, cos, sin, hcn, kc, vc,
                                          mil_mask)):
        with E._iosurface_view(surf, val.shape, np.float16) as d:
            np.copyto(d, val)
    if not eng.submit(prog, procedure_index=0):
        print("  submit failed")
        return
    got = {}
    for idx, nm, shape in ((0, "new_k", (KVC, S)), (1, "shared", (H, S)),
                           (2, "mixed", (H, S)), (3, "hyper", (HC_W, S)),
                           (5, "new_v", (KVC, S))):
        with E._iosurface_view(prog._out_surfs[idx], shape, np.float16) as o:
            got[nm] = np.array(o, np.float32)[:, :1]

    def rel(a, b_):
        return float(np.linalg.norm(a - b_) / max(np.linalg.norm(b_), 1e-12))
    print(f"  QSA FULL LAYER (int8={INT8[0]}, m={M}): "
          f"mixed {rel(got['mixed'], r_mixed):.5f}  "
          f"hyper {rel(got['hyper'], r_hyper):.5f}  "
          f"shared {rel(got['shared'], r_shared):.5f}  "
          f"new_k {rel(got['new_k'], r_nk):.5f}  "
          f"new_v {rel(got['new_v'], r_nv):.5f}", flush=True)
    for _ in range(5):
        eng.submit(prog, procedure_index=0)
    ts = []
    for _ in range(25):
        t0 = time.perf_counter()
        eng.submit(prog, procedure_index=0)
        ts.append(time.perf_counter() - t0)
    print(f"  QSA MIL layer: {float(np.median(ts)) * 1e3:.3f} ms  "
          f"(Core AI folded qsa_step k=1: ~2.96 ms)", flush=True)


if __name__ == "__main__":
    import os
    if os.environ.get("QSA_M"):
        KVM[0] = int(os.environ["QSA_M"])
    if os.environ.get("QSA_INT8") == "0":
        INT8[0] = False
    main()
