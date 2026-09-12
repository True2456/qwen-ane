#!/usr/bin/env python3
"""W8A8 accuracy on a two-projection chain, routed around the multi-blob bug.

ane_w8a8_accuracy.py packs every const tensor into one weight.bin. That is
poisoned: _make_blob's payload-offset field (header byte 80) is file-absolute
(0x80), not chunk-relative, so only the FIRST chunk in a packed bin reads back
correctly and every later tensor is garbage. Splitting into several BLOBFILEs
is also out -- more than one blob file per program fails at
"verifyBundleAtPath: invalid model" (Code 10) at any size.

This probe therefore uses exactly ONE blob chunk in the whole program. The
first projection's weight is a RUNTIME INPUT, packed into the tail of the
input surface ([1, ic, 1, seq+oc], sliced with slice_by_size, reshaped and
transposed into matmul) -- the mechanism validated in this repo at rel err
5.4e-4 (docs/HWX-ISA-SPEC.md section 9, test [10]; MIL spelling from
probes/mm_5120_5120_512.mil and probes/test_ane_prefill_mm.cpp).

Graph:  x -> matmul(runtime W1) -> [quantize/dequantize] -> conv(const W2) -> y

Only W2 is a const tensor. Its per-output-channel fp16 scale is an INLINE MIL
literal, not a second blob chunk, so the blob holds one tensor and one chunk.

Real weights: Qwen3.8-Flash-Next layer 0, in_proj_z [6144, 2560] then
out_proj [2560, 6144] -- 2560 -> 6144 -> 2560.

Arms (all vs the same fp32 numpy reference w2 @ (w1 @ x)):
  fp16             fp16 W1 (surface), fp16 W2 (const), fp16 activations
  w8a16-pc         int8 per-output-channel weights, fp16 activations
  w8a8-pc-global   + q/dq activations at the gist's single scale 0.125
  w8a8-pc-calib    + q/dq activations at abs(intermediate).max()/127
  w8a8-pc-actchan  + q/dq at one scale per intermediate channel (axis=1)

Each arm also prints an fp32 numpy simulation of the same arithmetic. Device
and simulation agreeing to 3 digits is what makes the error columns
trustworthy; a divergence means the graph is not computing what the arm says.

Caveat this design cannot avoid: W1 rides the input surface, which is fp16, so
its int8 arms carry the int8 *rounding error* (host-side fp16(q*scale)) but
the second-projection compute is a genuine int8 const conv while the first is
an fp16 matmul on already-dequantized weights.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# The engine's compile cache can serve poisoned artifacts from earlier runs of
# the multi-blob experiments; never reuse unless the caller insists.
os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

MODEL = Path(os.environ.get(
    "FLASH_NEXT", "/Users/true/models/Qwen3.8-Flash-Next"))
SHARD = MODEL / "model-00001-of-00131.safetensors"
PREFIX = "model.language_model.layers.0.linear_attn."

eng = AneEngine()
rng = np.random.default_rng(0)


def read_safetensors(path: Path, names: list[str]) -> dict:
    """Verbatim from probes/ane_w8a8_accuracy.py (BF16 -> fp32 by bit shift)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k in names:
            meta = hdr[k]
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            if meta["dtype"] == "BF16":
                bits = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                out[k] = bits.view(np.float32).reshape(meta["shape"])
            else:
                out[k] = np.frombuffer(raw, np.float16).reshape(
                    meta["shape"]).astype(np.float32)
        return out


def quant_per_channel(w: np.ndarray):
    """int8 + one fp16 scale per output row (the only granularity accepted)."""
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
    q = np.clip(np.rint(w / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return q, scale


def dequant_per_channel(w: np.ndarray) -> np.ndarray:
    q, s = quant_per_channel(w)
    return (q.astype(np.float32) * s.astype(np.float32)).astype(np.float32)


_CONV_PRE = (
    '    string pt = const()[name=string("pt"), val=string("valid")];\n'
    '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '    int32 gr = const()[name=string("gr"), val=int32(1)];'
)


def _dyn_mm_lines(ic: int, oc: int, S: int, out_name: str) -> str:
    """x[1,ic,1,S+oc] -> act + tail weight -> matmul -> out_name[1,oc,1,S].

    Exact spelling of the validated dynamic weight-as-input kernel.
    """
    return "\n".join([
        f'    tensor<int32, [4]> ba = const()[name=string("ba"), val=tensor<int32, [4]>([0,0,0,0])];',
        f'    tensor<int32, [4]> sa = const()[name=string("sa"), val=tensor<int32, [4]>([1,{ic},1,{S}])];',
        f'    tensor<fp16, [1,{ic},1,{S}]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string("act")];',
        f'    tensor<int32, [4]> bw = const()[name=string("bw"), val=tensor<int32, [4]>([0,0,0,{S}])];',
        f'    tensor<int32, [4]> sw = const()[name=string("sw"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];',
        f'    tensor<fp16, [1,{ic},1,{oc}]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string("wt")];',
        f'    tensor<int32, [4]> ra = const()[name=string("ra"), val=tensor<int32, [4]>([1,1,{ic},{S}])];',
        f'    tensor<fp16, [1,1,{ic},{S}]> a2 = reshape(shape=ra,x=act)[name=string("a2")];',
        f'    tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];',
        f'    tensor<fp16, [1,1,{S},{ic}]> a3 = transpose(perm=pm,x=a2)[name=string("a3")];',
        f'    tensor<int32, [4]> rw = const()[name=string("rw"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];',
        f'    tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw,x=wt)[name=string("W")];',
        f'    bool bF = const()[name=string("bF"), val=bool(false)];',
        f'    tensor<fp16, [1,1,{S},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string("yh")];',
        f'    tensor<fp16, [1,1,{oc},{S}]> yt = transpose(perm=pm,x=yh)[name=string("yt")];',
        f'    tensor<int32, [4]> ro = const()[name=string("ro"), val=tensor<int32, [4]>([1,{oc},1,{S}])];',
        f'    tensor<fp16, [1,{oc},1,{S}]> {out_name} = reshape(shape=ro,x=yt)[name=string("{out_name}")];',
    ])


def _compile(mil: str, blob: bytes | None, ic: int, oc: int, surf_w: int):
    """compile_multiproc with stdout/stderr captured; returns (prog, tail)."""
    weights = {}
    raw = frozenset()
    if blob is not None:
        weights = {"w.bin": blob}
        raw = frozenset({"w.bin"})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            # input_dim x seq_len sizes the input IOSurface; the surface is a
            # flat 1-row buffer, so passing seq_len = S + oc gives the tail
            # room. The output surface is oversized by the same factor, which
            # is harmless.
            p = eng.compile_multiproc(mil, weights, ic, oc, surf_w,
                                      raw_weight_files=raw)
        except Exception as exc:  # noqa: BLE001 - want the message verbatim
            print(f"exception: {exc}")
            p = None
    txt = buf.getvalue().strip()
    tail = "\n".join(txt.splitlines()[-3:]) if txt else ""
    return p, tail


def build_dyn_mm_only(ic: int, oc: int, S: int):
    """Harness check: the validated single dynamic matmul, no consts, no blob."""
    mil = (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
           f"  func main<ios18>(tensor<fp16, [1, {ic}, 1, {S + oc}]> x) {{\n"
           f"{_dyn_mm_lines(ic, oc, S, 'y')}\n  }} -> (y);\n}}\n")
    return _compile(mil, None, ic, oc, S + oc)


def build_chain(w1, w2, S, arm, act_scale):
    """x -> matmul(runtime W1) -> [q/dq] -> conv(const W2) -> y."""
    ic, mid = w1.shape[1], w1.shape[0]
    oc = w2.shape[0]
    lines = [_dyn_mm_lines(ic, mid, S, "m0")]
    prev = "m0"

    if arm.startswith("w8a8"):
        if isinstance(act_scale, np.ndarray):
            # Per-activation-channel scale: rank-1 scale const + axis=1.
            # A rank-4 [1,C,1,1] scale on quantize is InvalidMILProgram; the
            # rank-1 + axis spelling is the one that compiles.
            svals = ",".join(f"{float(v):.6g}" for v in act_scale.reshape(-1))
            lines.append(
                f'    tensor<fp16, [{mid}]> q_scale = const()[name=string("q_scale"), '
                f'val=tensor<fp16, [{mid}]>([{svals}])];\n'
                f'    int32 q_axis = const()[name=string("q_axis"), val=int32(1)];\n'
                f'    string q_dtype = const()[name=string("q_dtype"), val=string("int8")];\n'
                f'    tensor<int8, [1, {mid}, 1, {S}]> q0 = quantize(input=m0, '
                f'output_dtype=q_dtype, scale=q_scale, axis=q_axis)[name=string("q0")];\n'
                f'    tensor<fp16, [1, {mid}, 1, {S}]> d0 = dequantize(input=q0, '
                f'scale=q_scale, axis=q_axis)[name=string("d0")];')
        else:
            lines.append(
                f'    fp16 q_scale = const()[name=string("q_scale"), val=fp16({act_scale!r})];\n'
                f'    string q_dtype = const()[name=string("q_dtype"), val=string("int8")];\n'
                f'    fp16 dq_scale = const()[name=string("dq_scale"), val=fp16({act_scale!r})];\n'
                f'    tensor<int8, [1, {mid}, 1, {S}]> q0 = quantize(input=m0, '
                f'output_dtype=q_dtype, scale=q_scale)[name=string("q0")];\n'
                f'    tensor<fp16, [1, {mid}, 1, {S}]> d0 = dequantize(input=q0, '
                f'scale=dq_scale)[name=string("d0")];')
        prev = "d0"

    if arm == "fp16":
        blob = E._make_blob(w2.astype(np.float16).tobytes())
        lines.append(
            f'    tensor<fp16, [{oc}, {mid}, 1, 1]> W2 = const()'
            f'[name=string("W2"), val=tensor<fp16, [{oc}, {mid}, 1, 1]>'
            f'(BLOBFILE(path=string("@model_path/weights/w.bin"), '
            f'offset=uint64(64)))];')
    else:
        q, sc = quant_per_channel(w2)
        blob = E._make_blob(q.tobytes())
        # Per-channel scale as an INLINE literal: a second blob chunk would be
        # read at the wrong offset, a second blob FILE fails bundle verify.
        # A rank-4 inline literal MUST be nested ([[[v]]] per element); the
        # flat form that works for tensor<int32,[4]> is InvalidMILProgram here.
        vals = ",".join(f"[[[{float(v):.6g}]]]" for v in sc.reshape(-1))
        lines.append(
            f'    tensor<int8, [{oc}, {mid}, 1, 1]> W2q = const()'
            f'[name=string("W2q"), val=tensor<int8, [{oc}, {mid}, 1, 1]>'
            f'(BLOBFILE(path=string("@model_path/weights/w.bin"), '
            f'offset=uint64(64)))];\n'
            f'    tensor<fp16, [{oc}, 1, 1, 1]> W2s = const()'
            f'[name=string("W2s"), val=tensor<fp16, [{oc}, 1, 1, 1]>([{vals}])];\n'
            f'    tensor<fp16, [{oc}, {mid}, 1, 1]> W2 = '
            f'constexpr_blockwise_shift_scale(data=W2q, scale=W2s)'
            f'[name=string("W2dq")];')

    lines.append(
        f'    tensor<fp16, [1, {oc}, 1, {S}]> y = conv(dilations=dl, groups=gr, '
        f'pad=pd, pad_type=pt, strides=st, weight=W2, x={prev})[name=string("y")];')

    mil = (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
           f"  func main<ios18>(tensor<fp16, [1, {ic}, 1, {S + mid}]> x) {{\n"
           f"{_CONV_PRE}\n" + "\n".join(lines) + "\n  } -> (y);\n}\n")
    return _compile(mil, blob, ic, oc, S + mid)


def run(p, x, wt, ic, mid, S, out_rows):
    """Write [act | packed weight] into the input surface, submit, read out.

    Tail packing: matmul computes yh[s,n] = sum_k act[k,s] * W[k,n], so the
    tail region viewed as [ic, mid] must hold w1.T (W[k,n] = w1[n,k]).
    """
    eng._ensure_io(p)
    surf_w = S + mid
    with _iosurface_view(p._in_surf, (ic, surf_w), np.float16) as dst:
        dst[:, :S] = x.astype(np.float16)
        dst[:, S:] = wt.astype(np.float16)
    eng.submit(p, procedure_index=0)
    with _iosurface_view(p._out_surf, (out_rows, S), np.float16) as o:
        return np.array(o, np.float32)


def rel_l2(got, ref):
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))


def simulate(arm, x, w1_deq, w2_deq, w1, w2, act_scale):
    """fp32 numpy model of the same arithmetic the arm asks the ANE for.

    If the device column tracks this column, the quantization error is real
    and the harness is sound; a divergence means the graph is not computing
    what the arm claims.
    """
    if arm == "fp16":
        return w2 @ (w1 @ x)
    m = w1_deq @ x
    if arm.startswith("w8a8"):
        s = act_scale
        if isinstance(s, np.ndarray):
            s = s.astype(np.float32).reshape(-1, 1)
        m = np.clip(np.rint(m / s), -127, 127) * s
    return w2_deq @ m


def main() -> None:
    S = int(os.environ.get("W8A8_S", "256"))
    if not SHARD.exists():
        print(f"missing {SHARD}")
        return
    t = read_safetensors(SHARD, [PREFIX + "in_proj_z.weight",
                                 PREFIX + "out_proj.weight"])
    w1 = t[PREFIX + "in_proj_z.weight"]      # [6144, 2560]
    w2 = t[PREFIX + "out_proj.weight"]       # [2560, 6144]
    ic, mid, oc = w1.shape[1], w1.shape[0], w2.shape[0]
    print(f"real layer-0 weights: in_proj_z {w1.shape} -> out_proj {w2.shape}, "
          f"S={S}, one const tensor, W1 as runtime input")
    print(f"input surface [1,{ic},1,{S}+{mid}], reuse_compiled="
          f"{os.environ['Q38_ANE_REUSE_COMPILED']}\n")

    # ---- harness check 0: the validated dynamic matmul on its own ----------
    xs = rng.standard_normal((ic, S)).astype(np.float32) * 0.1
    p0, tail0 = build_dyn_mm_only(ic, mid, S)
    if p0 is None:
        print(f"[check] dynamic matmul REJECTED: {tail0}\n")
    else:
        got = run(p0, xs, w1.T, ic, mid, S, mid)
        print(f"[check] dynamic matmul alone (runtime fp16 W1): "
              f"rel L2 {rel_l2(got, w1 @ xs):.4e}   "
              f"(spec test [10] reports 5.4e-4)")
        del p0
    print()

    cases = {}
    base = rng.standard_normal((ic, S)).astype(np.float32) * 0.1
    cases["iid gaussian"] = base
    outl = base.copy()
    picks = rng.choice(outl.shape[0], 16, replace=False)
    outl[picks] *= 20.0
    cases["16 input-channel outliers x20"] = outl

    # The two cases above put the outliers in the INPUT channels. w1 mixes all
    # 2560 of them, so the tensor that actually gets quantized (the [6144, S]
    # intermediate) comes out close to gaussian either way -- which is not the
    # situation a per-tensor activation scale fails in. This third case drives
    # 16 chosen INTERMEDIATE channels to ~20x by adding multiples of the
    # corresponding rows of w1 to x, which is the outlier shape a real LLM
    # activation has at the point of quantization.
    rows = w1[rng.choice(mid, 16, replace=False)]          # [16, ic]
    g = rng.standard_normal((16, S)).astype(np.float32)
    bump = rows.T @ g
    alpha = 20.0 * np.abs(w1 @ base).max() / np.abs(w1 @ bump).max()
    cases["16 intermediate-channel outliers x20"] = base + alpha * bump

    w1_deq = dequant_per_channel(w1)
    w2_deq = dequant_per_channel(w2)

    for name, x in cases.items():
        ref = w2 @ (w1 @ x)
        m = w1 @ x
        calib = float(np.abs(m).max() / 127.0)
        chan = np.maximum(np.abs(m).max(axis=1) / 127.0, 1e-7).astype(np.float16)
        p999 = float(np.percentile(np.abs(m), 99.9))
        print(f"== {name}   intermediate |max|={np.abs(m).max():.2f}  "
              f"p99.9={p999:.2f}  peak/p99.9={np.abs(m).max()/p999:.1f}x  "
              f"calibrated act scale={calib:.5f}")
        print(f"   {'arm':>19} {'ANE rel L2':>12} {'numpy sim':>12} {'ms':>8}")
        for arm, sc in (("fp16", None), ("w8a16-pc", None),
                        ("w8a8-pc-global", 0.125), ("w8a8-pc-calib", calib),
                        ("w8a8-pc-actchan", chan)):
            sim = simulate(arm, x, w1_deq, w2_deq, w1, w2, sc)
            p, tail = build_chain(w1, w2, S, arm, sc)
            if p is None:
                print(f"   {arm:>19}     REJECTED  {tail}")
                continue
            wt = (w1.T if arm == "fp16" else w1_deq.T)
            got = run(p, x, wt, ic, mid, S, oc)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                eng.submit(p, procedure_index=0)
                ts.append((time.perf_counter() - t0) * 1e3)
            del p
            ts.sort()
            print(f"   {arm:>19} {rel_l2(got, ref):>12.4e} "
                  f"{rel_l2(sim, ref):>12.4e} {ts[2]:>8.3f}")
        print()


if __name__ == "__main__":
    main()
