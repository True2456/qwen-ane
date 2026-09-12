"""Stage 1 of the MIL GDN layer: in_proj (int8) + the 4-tap depthwise conv.

Verified against `FlashNextFront`, which is what the Core AI graph runs today.

Two things shape the MIL here:

* rank-4 **elementwise** consts are InvalidMILProgram, but rank-4 blob consts
  consumed by `conv` are fine. The 4-tap is `s0*t0 + s1*t1 + s2*t2 + qkv*t3`
  over consecutive token slots, i.e. a depthwise conv of kernel 4 along the
  last axis, so it is expressed as `conv(groups=QKV)` rather than 4 muls.
* multi-IO surfaces bind in **alphabetical symbol order**, so outputs are named
  so that sorted order is the order we read them.
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
    FlashNextFront, _load_layer, H, QKV, IN_O, SEQ_DEFAULT,
)

S = SEQ_DEFAULT


def _mil(quantized: bool, tap_file: str, tap_off: int,
         d_off: int, s_off: int) -> str:
    """in_proj then the 4-tap depthwise conv. Outputs: a_pre, b_pack."""
    wdecl = (
        f'    tensor<int8, [{IN_O}, {H}, 1, 1]> wd = const()[name=string("wd"), '
        f'val=tensor<int8, [{IN_O}, {H}, 1, 1]>(BLOBFILE('
        f'path=string("@model_path/weights/weight_data.bin"), offset=uint64({d_off})))];\n'
        f'    tensor<fp16, [{IN_O}, 1, 1, 1]> ws = const()[name=string("ws"), '
        f'val=tensor<fp16, [{IN_O}, 1, 1, 1]>(BLOBFILE('
        f'path=string("@model_path/weights/weight_scale.bin"), offset=uint64({s_off})))];\n'
        f'    tensor<fp16, [{IN_O}, {H}, 1, 1]> W = constexpr_blockwise_shift_scale('
        f'data=wd, scale=ws)[name=string("W")];\n'
    ) if quantized else (
        f'    tensor<fp16, [{IN_O}, {H}, 1, 1]> W = const()[name=string("W"), '
        f'val=tensor<fp16, [{IN_O}, {H}, 1, 1]>(BLOBFILE('
        f'path=string("@model_path/weights/weight_data.bin"), offset=uint64({d_off})))];\n'
    )
    return f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> a_h, tensor<fp16, [1, {3 * QKV}, 1, {S}]> b_cp) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    int32 gq = const()[name=string("gq"), val=int32({QKV})];
{wdecl}    tensor<fp16, [1, {IN_O}, 1, {S}]> yin = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=a_h)[name=string("yin")];
    tensor<int32, [4]> qb = const()[name=string("qb"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [4]> qe = const()[name=string("qe"), val=tensor<int32, [4]>([1,{QKV},1,{S}])];
    tensor<bool, [4]> m0 = const()[name=string("m0"), val=tensor<bool, [4]>([false,false,false,false])];
    tensor<fp16, [1, {QKV}, 1, {S}]> qkv = slice_by_index(x=yin, begin=qb, end=qe, begin_mask=m0, end_mask=m0)[name=string("qkv")];
    tensor<int32, [4]> rb = const()[name=string("rb"), val=tensor<int32, [4]>([0,{QKV},0,0])];
    tensor<int32, [4]> re = const()[name=string("re"), val=tensor<int32, [4]>([1,{IN_O},1,{S}])];
    tensor<fp16, [1, {IN_O - QKV}, 1, {S}]> rest = slice_by_index(x=yin, begin=rb, end=re, begin_mask=m0, end_mask=m0)[name=string("rest")];
    tensor<int32, [4]> c0b = const()[name=string("c0b"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [4]> c0e = const()[name=string("c0e"), val=tensor<int32, [4]>([1,{QKV},1,1])];
    tensor<fp16, [1, {QKV}, 1, 1]> c0 = slice_by_index(x=b_cp, begin=c0b, end=c0e, begin_mask=m0, end_mask=m0)[name=string("c0")];
    tensor<int32, [4]> c1b = const()[name=string("c1b"), val=tensor<int32, [4]>([0,{QKV},0,0])];
    tensor<int32, [4]> c1e = const()[name=string("c1e"), val=tensor<int32, [4]>([1,{2 * QKV},1,1])];
    tensor<fp16, [1, {QKV}, 1, 1]> c1 = slice_by_index(x=b_cp, begin=c1b, end=c1e, begin_mask=m0, end_mask=m0)[name=string("c1")];
    tensor<int32, [4]> c2b = const()[name=string("c2b"), val=tensor<int32, [4]>([0,{2 * QKV},0,0])];
    tensor<int32, [4]> c2e = const()[name=string("c2e"), val=tensor<int32, [4]>([1,{3 * QKV},1,1])];
    tensor<fp16, [1, {QKV}, 1, 1]> c2 = slice_by_index(x=b_cp, begin=c2b, end=c2e, begin_mask=m0, end_mask=m0)[name=string("c2")];
    tensor<fp16, [1, {QKV}, 1, {S + 3}]> seq = concat(values=(c0, c1, c2, qkv), axis=int32(-1), interleave=bool(false))[name=string("seq")];
    tensor<fp16, [{QKV}, 1, 1, 4]> T = const()[name=string("T"), val=tensor<fp16, [{QKV}, 1, 1, 4]>(BLOBFILE(path=string("@model_path/weights/{tap_file}"), offset=uint64({tap_off})))];
    tensor<fp16, [1, {QKV}, 1, {S}]> y_pre = conv(dilations=dl, groups=gq, pad=pd, pad_type=pt, strides=st, weight=T, x=seq)[name=string("y_pre")];
    tensor<int32, [4]> sb = const()[name=string("sb"), val=tensor<int32, [4]>([0,{QKV},0,0])];
    tensor<int32, [4]> se = const()[name=string("se"), val=tensor<int32, [4]>([1,{3 * QKV},1,{S}])];
    tensor<fp16, [1, {2 * QKV}, 1, {S}]> s12 = slice_by_index(x=b_cp, begin=sb, end=se, begin_mask=m0, end_mask=m0)[name=string("s12")];
    tensor<fp16, [1, {3 * QKV}, 1, {S}]> z_pack = concat(values=(s12, qkv), axis=int32(1), interleave=bool(false))[name=string("z_pack")];
  }} -> (y_pre, z_pack);
}}
"""


def main() -> None:
    loader, w = _load_layer(0)
    front = FlashNextFront().eval().half()
    front.load_from_layer(w)

    W = front.in_proj.op.weight.detach().float().numpy().reshape(IN_O, H)
    taps = np.concatenate([
        front.tap0.detach().float().numpy().reshape(QKV, 1),
        front.tap1.detach().float().numpy().reshape(QKV, 1),
        front.tap2.detach().float().numpy().reshape(QKV, 1),
        front.tap3.detach().float().numpy().reshape(QKV, 1)], axis=1)
    # _BlobPacker, not the gist _chunk helper: _chunk omits the payload SIZE
    # at chunk byte 8, which constexpr_blockwise_shift_scale requires. That is
    # why every int8 variant of this graph was InvalidMILProgram.
    tap_bytes = np.ascontiguousarray(
        taps.astype(np.float16).reshape(QKV, 1, 1, 4)).tobytes()

    rng = np.random.default_rng(9)
    h = np.ascontiguousarray((rng.standard_normal((H, S)) * 0.1).astype(np.float16))
    cp = np.ascontiguousarray((rng.standard_normal((3 * QKV, S)) * 0.05).astype(np.float16))
    with torch.no_grad():
        ry, rp = front(torch.from_numpy(h).reshape(1, H, 1, S),
                       torch.from_numpy(cp).reshape(1, 3 * QKV, 1, S))
    ry = ry.float().numpy().reshape(IN_O, S)
    rp = rp.float().numpy().reshape(3 * QKV, S)

    for quantized in (False, True):
        if quantized:
            q, sc = E.quantize_linear_int8(np.ascontiguousarray(W))
            dp = E._BlobPacker()
            d_off = dp.append(q.reshape(IN_O, H, 1, 1).tobytes())
            sp = E._BlobPacker()
            s_off = sp.append(np.asarray(sc, np.float16).reshape(IN_O, 1, 1, 1).tobytes())
            t_off = sp.append(tap_bytes)
            files = {"weight_data.bin": dp.getvalue(),
                     "weight_scale.bin": sp.getvalue()}
            tap_file, tap_off = "weight_scale.bin", t_off + 64
        else:
            dp = E._BlobPacker()
            d_off = dp.append(W.astype(np.float16).reshape(IN_O, H, 1, 1).tobytes())
            tp = E._BlobPacker()
            t_off = tp.append(tap_bytes)
            files = {"weight_data.bin": dp.getvalue(), "taps.bin": tp.getvalue()}
            tap_file, tap_off = "taps.bin", t_off + 64
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                p = eng.compile_multiproc(_mil(quantized, tap_file, tap_off, d_off + 64, s_off + 64 if quantized else 0), files, H, QKV, S,
                                          raw_weight_files=frozenset(files))
            except Exception as exc:  # noqa: BLE001
                p = None
                buf.write(str(exc))
        tag = "int8" if quantized else "fp16"
        if p is None:
            hit = [l for l in buf.getvalue().splitlines() if "rror" in l or "nvalid" in l]
            print(f"  {tag}: COMPILE FAILED {(hit[-1] if hit else '')[:110]}")
            continue
        # alphabetical symbol order: inputs a_h then b_cp, outputs y_pre then z_pack
        p.input_elems = [H * S, 3 * QKV * S]
        p.output_elems = [QKV * S, 3 * QKV * S]
        if not eng._ensure_io(p):
            print(f"  {tag}: IO alloc failed")
            continue
        import os
        order = os.environ.get("MIL_IN_ORDER", "ab")
        vals = (h, cp) if order == "ab" else (cp, h)
        for surf, val in zip(p._in_surfs, vals):
            with _iosurface_view(surf, val.shape, np.float16) as dst:
                np.copyto(dst, val)
        if not eng.submit(p, procedure_index=0):
            print(f"  {tag}: submit failed")
            continue
        with _iosurface_view(p._out_surfs[0], (QKV, S), np.float16) as o:
            got_pre = np.array(o, np.float32)
        with _iosurface_view(p._out_surfs[1], (3 * QKV, S), np.float16) as o:
            got_pack = np.array(o, np.float32)

        def rel(a, b):
            return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))

        # FlashNextFront applies the same cache columns to every slot; a real
        # conv slides its window, so only slot 0 is comparable.
        print(f"  {tag}: conv_pre[slot0] rel {rel(got_pre[:, :1], ry[:QKV, :1]):.5f}   "
              f"new_pack rel {rel(got_pack, rp):.5f}   "
              f"new_pack[slot0] rel {rel(got_pack[:, :1], rp[:, :1]):.5f}", flush=True)
        del p
    loader.close()


if __name__ == "__main__":
    main()
