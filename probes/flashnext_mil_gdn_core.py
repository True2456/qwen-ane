"""Stage 2: the GDN core in MIL — tanh-SiLU, L2 norms, decay, the S=1
recurrence, and out_proj at int8. Verified against `Connected`/`Production`.

Adaptations forced by the MIL surface, all verified individually first:
  * `rsqrt` is InvalidMILProgram -> `pow(x, -0.5)`
  * rank-4 **consts** are rejected -> `gamma`, `dt` and `norm_w` arrive as one
    runtime input row block; `pow` with a runtime tensor exponent is fine
  * `repeat_interleave` 16->48 has no op -> concat on axis 2 then reshape
  * multi-IO surfaces bind in ALPHABETICAL symbol order
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
    _load_layer, H, HK, HV, DK, DV, QKV, GDN_Y, IN_O, SEQ_DEFAULT,
)
from flashnext_connected_gdn import Connected  # noqa: E402

S = SEQ_DEFAULT
NH = QKV // DK          # 80 packed heads out of in_proj


def _mil(q8: bool, d_off: int, s_off: int) -> str:
    W = (f'    tensor<int8, [{H}, {GDN_Y}, 1, 1]> wd = const()[name=string("wd"), '
         f'val=tensor<int8, [{H}, {GDN_Y}, 1, 1]>(BLOBFILE('
         f'path=string("@model_path/weights/weight_data.bin"), offset=uint64({d_off})))];\n'
         f'    tensor<fp16, [{H}, 1, 1, 1]> ws = const()[name=string("ws"), '
         f'val=tensor<fp16, [{H}, 1, 1, 1]>(BLOBFILE('
         f'path=string("@model_path/weights/weight_scale.bin"), offset=uint64({s_off})))];\n'
         f'    tensor<fp16, [{H}, {GDN_Y}, 1, 1]> WO = constexpr_blockwise_shift_scale('
         f'data=wd, scale=ws)[name=string("WO")];\n') if q8 else (
         f'    tensor<fp16, [{H}, {GDN_Y}, 1, 1]> WO = const()[name=string("WO"), '
         f'val=tensor<fp16, [{H}, {GDN_Y}, 1, 1]>(BLOBFILE('
         f'path=string("@model_path/weights/weight_data.bin"), offset=uint64({d_off})))];\n')

    def sl(name, src, c0, c1, d2, d3, oc, o2=1, o3=1):
        return (f'    tensor<int32, [4]> {name}b = const()[name=string("{name}b"), '
                f'val=tensor<int32, [4]>([0,{c0},0,0])];\n'
                f'    tensor<int32, [4]> {name}e = const()[name=string("{name}e"), '
                f'val=tensor<int32, [4]>([1,{c1},{d2},{d3}])];\n'
                f'    tensor<fp16, [1, {oc}, {o2}, {o3}]> {name} = slice_by_index('
                f'x={src}, begin={name}b, end={name}e, begin_mask=mm, end_mask=mm)'
                f'[name=string("{name}")];\n')

    body = (
        '    tensor<bool, [4]> mm = const()[name=string("mm"), val=tensor<bool, [4]>([false,false,false,false])];\n'
        '    fp16 eps = const()[name=string("eps"), val=fp16(0.000001)];\n'
        '    fp16 half = const()[name=string("half"), val=fp16(0.5)];\n'
        f'    fp16 qsc = const()[name=string("qsc"), val=fp16({DK ** -0.5})];\n'
        '    fp16 nho = const()[name=string("nho"), val=fp16(-1.0)];\n'
        '    fp16 mh = const()[name=string("mh"), val=fp16(-0.5)];\n'
        '    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([-1])];\n'
        '    bool kd = const()[name=string("kd"), val=bool(true)];\n'
        '    tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];\n'
        '    string pt = const()[name=string("pt"), val=string("valid")];\n'
        '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
        '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
        '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
        '    int32 gr = const()[name=string("gr"), val=int32(1)];\n'
    )
    body += sl("one", "a_pre", 0, IN_O, 1, 1, IN_O)
    body += sl("qk1", "one", 0, QKV, 1, 1, QKV)
    body += (f'    tensor<int32, [4]> rs = const()[name=string("rs"), val=tensor<int32, [4]>([1,{NH},1,{DK}])];\n'
             f'    tensor<fp16, [1, {NH}, 1, {DK}]> pk = reshape(x=qk1, shape=rs)[name=string("pk")];\n'
             f'    tensor<fp16, [1, {NH}, 1, {DK}]> hx = mul(x=pk, y=half)[name=string("hx")];\n'
             f'    tensor<fp16, [1, {NH}, 1, {DK}]> th = tanh(x=hx)[name=string("th")];\n'
             f'    tensor<fp16, [1, {NH}, 1, {DK}]> hm = mul(x=hx, y=th)[name=string("hm")];\n'
             f'    tensor<fp16, [1, {NH}, 1, {DK}]> cc = add(x=hx, y=hm)[name=string("cc")];\n')
    body += sl("qq", "cc", 0, HK, 1, DK, HK, 1, DK)
    body += sl("kk", "cc", HK, 2 * HK, 1, DK, HK, 1, DK)
    body += sl("vv", "cc", 2 * HK, NH, 1, DK, HV, 1, DK)
    for nm, src, scale in (("qn", "qq", True), ("kn", "kk", False)):
        body += (f'    tensor<fp16, [1, {HK}, 1, {DK}]> {nm}2 = mul(x={src}, y={src})[name=string("{nm}2")];\n'
                 f'    tensor<fp16, [1, {HK}, 1, 1]> {nm}s = reduce_sum(x={nm}2, axes=ax, keep_dims=kd)[name=string("{nm}s")];\n'
                 f'    tensor<fp16, [1, {HK}, 1, 1]> {nm}e = add(x={nm}s, y=eps)[name=string("{nm}e")];\n'
                 f'    tensor<fp16, [1, {HK}, 1, 1]> {nm}r = pow(x={nm}e, y=mh)[name=string("{nm}r")];\n'
                 f'    tensor<fp16, [1, {HK}, 1, {DK}]> {nm}m = mul(x={src}, y={nm}r)[name=string("{nm}m")];\n')
        last = f"{nm}m"
        if scale:
            body += (f'    tensor<fp16, [1, {HK}, 1, {DK}]> {nm}g = mul(x={nm}m, y=qsc)[name=string("{nm}g")];\n')
            last = f"{nm}g"
        body += (f'    tensor<fp16, [1, {HK}, 3, {DK}]> {nm}c = concat(values=({last}, {last}, {last}), axis=int32(2), interleave=bool(false))[name=string("{nm}c")];\n'
                 f'    tensor<int32, [4]> {nm}sh = const()[name=string("{nm}sh"), val=tensor<int32, [4]>([1,{HV},1,{DK}])];\n'
                 f'    tensor<fp16, [1, {HV}, 1, {DK}]> {nm}48 = reshape(x={nm}c, shape={nm}sh)[name=string("{nm}48")];\n')
    body += sl("z1", "one", QKV, QKV + GDN_Y, 1, 1, GDN_Y)
    body += (f'    tensor<int32, [4]> zs = const()[name=string("zs"), val=tensor<int32, [4]>([1,{HV},1,{DK}])];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DK}]> zz = reshape(x=z1, shape=zs)[name=string("zz")];\n')
    body += sl("b1", "one", QKV + GDN_Y, QKV + GDN_Y + HV, 1, 1, HV)
    body += sl("a1", "one", QKV + GDN_Y + HV, IN_O, 1, 1, HV)
    body += sl("gam", "b_param", 0, HV, 1, DK, HV, 1, DK)
    body += sl("dtb", "b_param", HV, 2 * HV, 1, DK, HV, 1, DK)
    body += sl("nw", "b_param", 2 * HV, 3 * HV, 1, DK, HV, 1, DK)
    body += (f'    tensor<fp16, [1, {HV}, 1, {DK}]> ad = add(x=a1, y=dtb)[name=string("ad")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DK}]> an = mul(x=ad, y=nho)[name=string("an")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DK}]> sg = sigmoid(x=an)[name=string("sg")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DK}]> dec = pow(x=sg, y=gam)[name=string("dec")];\n'
             f'    tensor<fp16, [1, {HV}, 1, 1]> bet = sigmoid(x=b1)[name=string("bet")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, {DK}]> st1 = mul(x=c_state, y=dec)[name=string("st1")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, {DK}]> sk = mul(x=st1, y=kn48)[name=string("sk")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, 1]> mem = reduce_sum(x=sk, axes=ax, keep_dims=kd)[name=string("mem")];\n'
             f'    tensor<fp16, [1, {HV}, {DK}, 1]> vt = transpose(x=vv, perm=pm)[name=string("vt")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, 1]> df = sub(x=vt, y=mem)[name=string("df")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, 1]> dl2 = mul(x=df, y=bet)[name=string("dl2")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, {DK}]> dk2 = mul(x=dl2, y=kn48)[name=string("dk2")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, {DK}]> z_state = add(x=st1, y=dk2)[name=string("z_state")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, {DK}]> sq = mul(x=z_state, y=qn48)[name=string("sq")];\n'
             f'    tensor<fp16, [1, {HV}, {DV}, 1]> yv = reduce_sum(x=sq, axes=ax, keep_dims=kd)[name=string("yv")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DV}]> yt = transpose(x=yv, perm=pm)[name=string("yt")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DV}]> y2 = mul(x=yt, y=yt)[name=string("y2")];\n'
             f'    tensor<fp16, [1, {HV}, 1, 1]> ym = reduce_mean(x=y2, axes=ax, keep_dims=kd)[name=string("ym")];\n'
             f'    tensor<fp16, [1, {HV}, 1, 1]> ye = add(x=ym, y=eps)[name=string("ye")];\n'
             f'    tensor<fp16, [1, {HV}, 1, 1]> yr = pow(x=ye, y=mh)[name=string("yr")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DV}]> yn2 = mul(x=yt, y=yr)[name=string("yn2")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DV}]> yw = mul(x=yn2, y=nw)[name=string("yw")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DV}]> zg = sigmoid(x=zz)[name=string("zg")];\n'
             f'    tensor<fp16, [1, {HV}, 1, {DV}]> yo = mul(x=yw, y=zg)[name=string("yo")];\n'
             f'    tensor<int32, [4]> fs = const()[name=string("fs"), val=tensor<int32, [4]>([1,{GDN_Y},1,1])];\n'
             f'    tensor<fp16, [1, {GDN_Y}, 1, 1]> yf = reshape(x=yo, shape=fs)[name=string("yf")];\n'
             f'    tensor<int32, [4]> rp = const()[name=string("rp"), val=tensor<int32, [4]>([1,1,1,{S}])];\n'
             f'    tensor<fp16, [1, {GDN_Y}, 1, {S}]> yx = tile(x=yf, reps=rp)[name=string("yx")];\n')
    body += W
    body += (f'    tensor<fp16, [1, {H}, 1, {S}]> y_attn = conv(dilations=dl, groups=gr, pad=pd, '
             f'pad_type=pt, strides=st, weight=WO, x=yx)[name=string("y_attn")];\n')
    return (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
            f"  func main<ios18>(tensor<fp16, [1, {IN_O}, 1, {S}]> a_pre, "
            f"tensor<fp16, [1, {3 * HV}, 1, {DK}]> b_param, "
            f"tensor<fp16, [1, {HV}, {DV}, {DK}]> c_state) {{\n"
            f"{body}  }} -> (y_attn, z_state);\n}}\n")


def main() -> None:
    loader, w = _load_layer(0)
    conn = Connected(w).eval().half()
    Wo = conn.tail.out_proj.op.weight.detach().float().numpy().reshape(H, GDN_Y)
    gam = conn.gamma.detach().float().numpy().reshape(HV, 1)
    dtb = conn.dt.detach().float().numpy().reshape(HV, 1)
    nw = conn.tail.norm_w.detach().float().numpy().reshape(HV, DV)
    param = np.concatenate([np.repeat(gam, DK, 1), np.repeat(dtb, DK, 1), nw], axis=0)

    rng = np.random.default_rng(4)
    yin = np.ascontiguousarray((rng.standard_normal((IN_O, S)) * 0.15).astype(np.float16))
    state = np.ascontiguousarray((rng.standard_normal((HV, DV, DK)) * 0.02).astype(np.float16))

    # torch reference: Connected minus the front
    import torch as T
    with T.no_grad():
        yt = T.from_numpy(yin).reshape(1, IN_O, 1, S)
        one = yt[..., :1]
        packed = one[:, :QKV].reshape(1, NH, 1, DK)
        hx = packed * conn.half_coeff
        c = hx + hx * T.tanh(hx)
        q, k, v = c[:, :HK], c[:, HK:2 * HK], c[:, 2 * HK:]
        q = q * T.rsqrt((q * q).sum(-1, keepdim=True) + conn.eps) * conn.qscale
        k = k * T.rsqrt((k * k).sum(-1, keepdim=True) + conn.eps)
        q = q.repeat_interleave(HV // HK, dim=1)
        k = k.repeat_interleave(HV // HK, dim=1)
        z = one[:, QKV:QKV + GDN_Y].reshape(1, HV, 1, DK)
        b = one[:, QKV + GDN_Y:QKV + GDN_Y + HV]
        a = one[:, QKV + GDN_Y + HV:]
        decay = T.pow(T.sigmoid(-(a + conn.dt)), conn.gamma).expand(1, HV, 1, DK)
        beta = T.sigmoid(b).expand(1, HV, 1, DK)
        params = T.cat((q, k, v, decay, beta, z), dim=2)
        attn_ref, st_ref = conn.tail(params, T.from_numpy(state).reshape(1, HV, DV, DK))
    attn_ref = attn_ref.float().numpy().reshape(H, S)[:, :1]
    st_ref = st_ref.float().numpy().reshape(HV, DV, DK)

    for q8 in (False, True):
        if q8:
            qd, sc = E.quantize_linear_int8(np.ascontiguousarray(Wo))
            dp = E._BlobPacker(); d_off = dp.append(qd.reshape(H, GDN_Y, 1, 1).tobytes())
            sp = E._BlobPacker(); s_off = sp.append(np.asarray(sc, np.float16).reshape(H, 1, 1, 1).tobytes())
            files = {"weight_data.bin": dp.getvalue(), "weight_scale.bin": sp.getvalue()}
        else:
            dp = E._BlobPacker(); d_off = dp.append(Wo.astype(np.float16).reshape(H, GDN_Y, 1, 1).tobytes())
            s_off = 0
            files = {"weight_data.bin": dp.getvalue()}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                p = eng.compile_multiproc(_mil(q8, d_off + 64, s_off + 64), files,
                                          IN_O, H, S, raw_weight_files=frozenset(files))
            except Exception as exc:  # noqa: BLE001
                p = None; buf.write(str(exc))
        tag = "int8" if q8 else "fp16"
        if p is None:
            hit = [l for l in buf.getvalue().splitlines() if "rror" in l or "nvalid" in l]
            print(f"  {tag}: COMPILE FAILED {(hit[-1] if hit else '')[:110]}")
            continue
        p.input_elems = [IN_O * S, 3 * HV * DK, HV * DV * DK]
        p.output_elems = [H * S, HV * DV * DK]
        if not eng._ensure_io(p):
            print(f"  {tag}: IO alloc failed"); continue
        for surf, val in zip(p._in_surfs, (yin, np.ascontiguousarray(param.astype(np.float16)), state)):
            with _iosurface_view(surf, val.shape, np.float16) as dst:
                np.copyto(dst, val)
        if not eng.submit(p, procedure_index=0):
            print(f"  {tag}: submit failed"); continue
        with _iosurface_view(p._out_surfs[0], (H, S), np.float16) as o:
            g_attn = np.array(o, np.float32)[:, :1]
        with _iosurface_view(p._out_surfs[1], (HV, DV, DK), np.float16) as o:
            g_st = np.array(o, np.float32)

        def rel(x, y):
            return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-12))
        print(f"  {tag}: attn rel {rel(g_attn, attn_ref):.5f}   state rel {rel(g_st, st_ref):.5f}",
              flush=True)
        del p
    loader.close()


if __name__ == "__main__":
    main()
