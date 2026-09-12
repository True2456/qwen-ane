"""Stage 3: the hyper-connection mixer in MIL, verified vs FlashNextGatedMix.

Two things differ from the GDN core:
  * the grouped RMS reduces over the **channel** axis (4 branches of H), not
    the last axis;
  * `hc_n` is a rank-4 const, which is rejected, so it arrives as a runtime
    input shaped `[1, 640, 1, 32]` and is reshaped to `[1, HC_W, 1, 1]`
    in-graph (a last-dim-1 *input* fails at submit; a last-dim-1 intermediate
    is fine).
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
from export_flashnext_coreai import _load_layer, H, HC, HC_W, SEQ_DEFAULT  # noqa: E402
from flashnext_pure_step import FlashNextGatedMix, MIX_H  # noqa: E402

S = SEQ_DEFAULT


def _mil(offs: dict) -> str:
    b = [
        '    tensor<bool, [4]> mm = const()[name=string("mm"), val=tensor<bool, [4]>([false,false,false,false])];',
        '    fp16 eps = const()[name=string("eps"), val=fp16(0.000001)];',
        '    fp16 ivh = const()[name=string("ivh"), val=fp16(0.25)];',
        '    fp16 hlf = const()[name=string("hlf"), val=fp16(0.5)];',
        '    fp16 two = const()[name=string("two"), val=fp16(2.0)];',
        '    fp16 mh = const()[name=string("mh"), val=fp16(-0.5)];',
        '    tensor<int32, [1]> ac = const()[name=string("ac"), val=tensor<int32, [1]>([1])];',
        '    bool kd = const()[name=string("kd"), val=bool(true)];',
        '    string pt = const()[name=string("pt"), val=string("valid")];',
        '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];',
        '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];',
        '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];',
        '    int32 gr = const()[name=string("gr"), val=int32(1)];',
    ]

    def sl(name, src, c0, c1, oc, last=S):
        b.append(f'    tensor<int32, [4]> {name}b = const()[name=string("{name}b"), val=tensor<int32, [4]>([0,{c0},0,0])];')
        b.append(f'    tensor<int32, [4]> {name}e = const()[name=string("{name}e"), val=tensor<int32, [4]>([1,{c1},1,{last}])];')
        b.append(f'    tensor<fp16, [1, {oc}, 1, {last}]> {name} = slice_by_index(x={src}, begin={name}b, end={name}e, begin_mask=mm, end_mask=mm)[name=string("{name}")];')

    # grouped RMS over 4 branches, reducing the channel axis
    for i in range(HC):
        sl(f"g{i}", "a_x", i * H, (i + 1) * H, H)
        b.append(f'    tensor<fp16, [1, {H}, 1, {S}]> p{i} = mul(x=g{i}, y=g{i})[name=string("p{i}")];')
        b.append(f'    tensor<fp16, [1, 1, 1, {S}]> m{i} = reduce_mean(x=p{i}, axes=ac, keep_dims=kd)[name=string("m{i}")];')
        b.append(f'    tensor<fp16, [1, 1, 1, {S}]> e{i} = add(x=m{i}, y=eps)[name=string("e{i}")];')
        b.append(f'    tensor<fp16, [1, 1, 1, {S}]> r{i} = pow(x=e{i}, y=mh)[name=string("r{i}")];')
        b.append(f'    tensor<fp16, [1, {H}, 1, {S}]> n{i} = mul(x=g{i}, y=r{i})[name=string("n{i}")];')
    b.append(f'    tensor<fp16, [1, {HC_W}, 1, {S}]> nc = concat(values=(n0, n1, n2, n3), axis=int32(1), interleave=bool(false))[name=string("nc")];')
    b.append(f'    tensor<int32, [4]> hs = const()[name=string("hs"), val=tensor<int32, [4]>([1,{HC_W},1,1])];')
    b.append(f'    tensor<fp16, [1, {HC_W}, 1, 1]> hcn = reshape(x=b_hcn, shape=hs)[name=string("hcn")];')
    b.append(f'    tensor<fp16, [1, {HC_W}, 1, {S}]> nn = mul(x=nc, y=hcn)[name=string("nn")];')
    for nm, o, ci, co in (("WD", offs["down"], HC_W, MIX_H),
                          ("WU", offs["up"], MIX_H, HC_W),
                          ("WI", offs["inj"], HC_W, HC)):
        b.append(f'    tensor<fp16, [{co}, {ci}, 1, 1]> {nm} = const()[name=string("{nm}"), val=tensor<fp16, [{co}, {ci}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({o})))];')
    b.append(f'    tensor<fp16, [1, {MIX_H}, 1, {S}]> dv = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WD, x=nn)[name=string("dv")];')
    b.append(f'    tensor<fp16, [1, {MIX_H}, 1, {S}]> dn = mul(x=dv, y=ivh)[name=string("dn")];')
    b.append(f'    tensor<fp16, [1, {MIX_H}, 1, {S}]> hf = mul(x=dn, y=hlf)[name=string("hf")];')
    b.append(f'    tensor<fp16, [1, {MIX_H}, 1, {S}]> tt = tanh(x=hf)[name=string("tt")];')
    b.append(f'    tensor<fp16, [1, {MIX_H}, 1, {S}]> hm = mul(x=hf, y=tt)[name=string("hm")];')
    b.append(f'    tensor<fp16, [1, {MIX_H}, 1, {S}]> gt = add(x=hf, y=hm)[name=string("gt")];')
    b.append(f'    tensor<fp16, [1, {HC_W}, 1, {S}]> uv = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WU, x=gt)[name=string("uv")];')
    b.append(f'    tensor<fp16, [1, {HC_W}, 1, {S}]> mw = sigmoid(x=uv)[name=string("mw")];')
    for i in range(HC):
        sl(f"w{i}", "mw", i * H, (i + 1) * H, H)
        sl(f"q{i}", "nn", i * H, (i + 1) * H, H)
        b.append(f'    tensor<fp16, [1, {H}, 1, {S}]> t{i} = mul(x=w{i}, y=q{i})[name=string("t{i}")];')
    b.append(f'    tensor<fp16, [1, {H}, 1, {S}]> s1 = add(x=t0, y=t1)[name=string("s1")];')
    b.append(f'    tensor<fp16, [1, {H}, 1, {S}]> s2 = add(x=s1, y=t2)[name=string("s2")];')
    b.append(f'    tensor<fp16, [1, {H}, 1, {S}]> s3 = add(x=s2, y=t3)[name=string("s3")];')
    b.append(f'    tensor<fp16, [1, {H}, 1, {S}]> y_mixed = mul(x=s3, y=ivh)[name=string("y_mixed")];')
    b.append(f'    tensor<fp16, [1, {HC}, 1, {S}]> iv = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=WI, x=nn)[name=string("iv")];')
    b.append(f'    tensor<fp16, [1, {HC}, 1, {S}]> ih = mul(x=iv, y=ivh)[name=string("ih")];')
    b.append(f'    tensor<fp16, [1, {HC}, 1, {S}]> ig = sigmoid(x=ih)[name=string("ig")];')
    b.append(f'    tensor<fp16, [1, {HC}, 1, {S}]> x_inj = mul(x=ig, y=two)[name=string("x_inj")];')
    return (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
            f"  func main<ios18>(tensor<fp16, [1, {HC_W}, 1, {S}]> a_x, "
            f"tensor<fp16, [1, 320, 1, 32]> b_hcn) {{\n" + "\n".join(b) +
            f"\n  }} -> (x_inj, y_mixed);\n}}\n")


def main() -> None:
    loader, w = _load_layer(0)
    mix = FlashNextGatedMix()
    mix.load(w, "attn_hyper_connection")
    mix = mix.eval().half()

    packer = E._BlobPacker()
    offs = {}
    for nm, mod, co, ci in (("down", mix.down, MIX_H, HC_W),
                            ("up", mix.up, HC_W, MIX_H),
                            ("inj", mix.inj, HC, HC_W)):
        wt = mod.op.weight.detach().float().numpy().reshape(co, ci, 1, 1)
        offs[nm] = packer.append(wt.astype(np.float16).tobytes()) + 64
    files = {"weight_data.bin": packer.getvalue()}

    rng = np.random.default_rng(6)
    x = np.ascontiguousarray((rng.standard_normal((HC_W, S)) * 0.1).astype(np.float16))
    hcn = np.ascontiguousarray(
        mix.hc_n.detach().float().numpy().reshape(320, 32).astype(np.float16))
    with torch.no_grad():
        r_mixed, _, r_inj = mix(torch.from_numpy(x).reshape(1, HC_W, 1, S))
    r_mixed = r_mixed.float().numpy().reshape(H, S)
    r_inj = r_inj.float().numpy().reshape(HC, S)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(_mil(offs), files, HC_W, H, S,
                                      raw_weight_files=frozenset(files))
        except Exception as exc:  # noqa: BLE001
            p = None
            buf.write(str(exc))
    if p is None:
        hit = [l for l in buf.getvalue().splitlines() if "rror" in l or "nvalid" in l]
        print(f"  COMPILE FAILED {(hit[-1] if hit else '')[:120]}")
        return
    p.input_elems = [HC_W * S, 320 * 32]
    p.output_elems = [HC * S, H * S]
    if not eng._ensure_io(p):
        print("  IO alloc failed")
        return
    for surf, val in zip(p._in_surfs, (x, hcn)):
        with _iosurface_view(surf, val.shape, np.float16) as d:
            np.copyto(d, val)
    if not eng.submit(p, procedure_index=0):
        print("  submit failed")
        return
    with _iosurface_view(p._out_surfs[0], (HC, S), np.float16) as o:
        g_inj = np.array(o, np.float32)
    with _iosurface_view(p._out_surfs[1], (H, S), np.float16) as o:
        g_mixed = np.array(o, np.float32)

    def rel(a, b_):
        return float(np.linalg.norm(a - b_) / max(np.linalg.norm(b_), 1e-12))

    print(f"  mixer: mixed rel {rel(g_mixed, r_mixed):.5f}   inj rel {rel(g_inj, r_inj):.5f}",
          flush=True)
    loader.close()


if __name__ == "__main__":
    main()
