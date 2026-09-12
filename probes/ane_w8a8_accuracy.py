#!/usr/bin/env python3
"""Is the W8A8 chain accurate enough to serve, on real Flash-Next weights?

ane_w8a8_projection.py established the speed (2.1-3.7x). It used random int8
weights, one scalar weight scale per conv, and one hardcoded activation scale,
so it made no accuracy claim. This probe closes that.

Real weights: Qwen3.8-Flash-Next layer 0, in_proj_z [6144, 2560] followed by
out_proj [2560, 6144]. Two consecutive projections that chain exactly, so the
quantize/dequantize pair sits where it would in a fused layer.

Four arms, all compared against an fp32 numpy reference on the same inputs:

  fp16          fp16 weights, fp16 activations
  w8a16-pc      int8 weights, per-output-channel scales, fp16 activations
  w8a8-global   per-channel weights + the gist's single hardcoded act scale
  w8a8-calib    per-channel weights + an act scale calibrated from the data

Activations are tested twice: iid gaussian, and with outlier channels injected.
Real LLM activations have outlier channels; iid inputs understate the clipping
that a per-tensor activation scale causes, so the outlier row is the honest one.
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

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

MODEL = Path(os.environ.get(
    "FLASH_NEXT", "/Users/true/models/Qwen3.8-Flash-Next"))
SHARD = MODEL / "model-00001-of-00131.safetensors"
PREFIX = "model.language_model.layers.0.linear_attn."

eng = AneEngine()
rng = np.random.default_rng(0)


def read_safetensors(path: Path, names: list[str]) -> dict:
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


def _chunk(payload: bytes, marker: int) -> bytes:
    """Engine blob layout (_make_blob) for every arm.

    Two traps here, both of which produce a program that compiles and returns
    plausible-looking garbage:

    1. A hand-rolled 64+64 header omits the payload-size and 0x80
       payload-offset fields ANECCompile needs, so consts read the wrong bytes.
    2. The gist writes its dtype marker at header byte 10. In _make_blob's
       layout that byte is INSIDE the payload-size field at 72-75, so setting
       it corrupts the size. _make_blob needs no marker; do not add one.
    """
    del marker  # see note
    return E._make_blob(payload)


def quant_per_channel(w: np.ndarray):
    """int8 with one fp16 scale per output row, the only granularity accepted."""
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
    q = np.clip(np.rint(w / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return q, scale


_PRE = (
    '    string pt = const()[name=string("pt"), val=string("valid")];\n'
    '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '    int32 gr = const()[name=string("gr"), val=int32(1)];'
)


def build(w1, w2, S, arm, act_scale):
    """Two-conv chain: x -> conv(w1) -> [q/dq] -> conv(w2) -> y.

    Every tensor goes in ONE weight.bin, wrapped by _make_blob and addressed by
    offset. Measured on this OS: a program with more than one blob FILE fails
    at `verifyBundleAtPath: invalid model` regardless of size (a 1 MB
    512-512-512 chain fails; a single 31.5 MB conv is fine). Chain depth is not
    the problem -- an 8-conv chain in one file compiles. Do not split files.
    """
    chunks, decls = [], []

    def add(payload: bytes) -> int:
        base = sum(len(c) for c in chunks)
        chunks.append(E._make_blob(payload))
        return base + 64          # MIL offset lands on the DEADBEEF header

    for i, w in enumerate((w1, w2)):
        out_dim, in_dim = w.shape
        if arm == "fp16":
            off = add(w.astype(np.float16).tobytes())
            decls.append(
                f'    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W{i} = const()'
                f'[name=string("W{i}"), val=tensor<fp16, [{out_dim}, {in_dim}, 1, 1]>'
                f'(BLOBFILE(path=string("@model_path/weights/w.bin"), '
                f'offset=uint64({off})))];')
        else:
            # constexpr_affine_dequantize takes a SCALAR scale only; a
            # per-channel scale tensor is InvalidMILProgram. Per-output-channel
            # scales come from constexpr_blockwise_shift_scale instead.
            q, sc = quant_per_channel(w)
            qoff = add(q.tobytes())
            soff = add(sc.astype(np.float16).tobytes())
            decls.append(
                f'    tensor<int8, [{out_dim}, {in_dim}, 1, 1]> W{i}q = const()'
                f'[name=string("W{i}q"), val=tensor<int8, [{out_dim}, {in_dim}, 1, 1]>'
                f'(BLOBFILE(path=string("@model_path/weights/w.bin"), '
                f'offset=uint64({qoff})))];\n'
                f'    tensor<fp16, [{out_dim}, 1, 1, 1]> W{i}s = const()'
                f'[name=string("W{i}s"), val=tensor<fp16, [{out_dim}, 1, 1, 1]>'
                f'(BLOBFILE(path=string("@model_path/weights/w.bin"), '
                f'offset=uint64({soff})))];\n'
                f'    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W{i} = '
                f'constexpr_blockwise_shift_scale(data=W{i}q, scale=W{i}s)'
                f'[name=string("W{i}dq")];')

    m1, m2 = w1.shape[0], w2.shape[0]
    lines = [decls[0]]
    lines.append(
        f'    tensor<fp16, [1, {m1}, 1, {S}]> c0 = conv(dilations=dl, groups=gr, '
        f'pad=pd, pad_type=pt, strides=st, weight=W0, x=x)[name=string("c0")];')
    prev = "c0"
    if arm.startswith("w8a8"):
        lines.append(
            f'    fp16 q_scale = const()[name=string("q_scale"), val=fp16({act_scale!r})];\n'
            f'    string q_dtype = const()[name=string("q_dtype"), val=string("int8")];\n'
            f'    fp16 dq_scale = const()[name=string("dq_scale"), val=fp16({act_scale!r})];')
        lines.append(
            f'    tensor<int8, [1, {m1}, 1, {S}]> q0 = quantize(input=c0, '
            f'output_dtype=q_dtype, scale=q_scale)[name=string("q0")];')
        lines.append(
            f'    tensor<fp16, [1, {m1}, 1, {S}]> d0 = dequantize(input=q0, '
            f'scale=dq_scale)[name=string("d0")];')
        prev = "d0"
    lines.append(decls[1])
    lines.append(
        f'    tensor<fp16, [1, {m2}, 1, {S}]> y = conv(dilations=dl, groups=gr, '
        f'pad=pd, pad_type=pt, strides=st, weight=W1, x={prev})[name=string("y")];')

    mil = (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
           f"  func main<ios18>(tensor<fp16, [1, {w1.shape[1]}, 1, {S}]> x) {{\n"
           f"{_PRE}\n" + "\n".join(lines) + "\n  } -> (y);\n}\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(
                mil, {"w.bin": b"".join(chunks)}, w1.shape[1], m2, S,
                raw_weight_files=frozenset({"w.bin"}))
        except Exception:
            p = None
    return p, "\n".join(buf.getvalue().strip().splitlines()[-2:])


def run(p, x, M, S):
    eng._ensure_io(p)
    with _iosurface_view(p._in_surf, x.shape, np.float16) as dst:
        np.copyto(dst, x.astype(np.float16))
    eng.submit(p, procedure_index=0)
    with _iosurface_view(p._out_surf, (M, S), np.float16) as o:
        return np.array(o, np.float32)


def rel_l2(got, ref):
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))


def main() -> None:
    S = int(os.environ.get("W8A8_S", "256"))
    if not SHARD.exists():
        print(f"missing {SHARD}"); return
    t = read_safetensors(SHARD, [PREFIX + "in_proj_z.weight",
                                 PREFIX + "out_proj.weight"])
    w1 = t[PREFIX + "in_proj_z.weight"]
    w2 = t[PREFIX + "out_proj.weight"]
    print(f"real layer-0 weights: in_proj_z {w1.shape} -> out_proj {w2.shape}, S={S}\n")

    cases = {}
    base = rng.standard_normal((w1.shape[1], S)).astype(np.float32) * 0.1
    cases["iid gaussian"] = base
    out = base.copy()
    picks = rng.choice(out.shape[0], 16, replace=False)
    out[picks] *= 20.0
    cases["16 outlier channels x20"] = out

    for name, x in cases.items():
        ref = w2 @ (w1 @ x)
        mid = w1 @ x
        calib = float(np.abs(mid).max() / 127.0)
        print(f"== {name}   intermediate |max|={np.abs(mid).max():.2f}  "
              f"calibrated act scale={calib:.5f}")
        print(f"   {'arm':>13} {'rel L2 vs fp32':>15} {'ms':>8}")
        for arm, sc in (("fp16", None), ("w8a16-pc", None),
                        ("w8a8-pc-global", 0.125), ("w8a8-pc-calib", calib)):
            p, tail = build(w1, w2, S, arm, sc)
            if p is None:
                print(f"   {arm:>13}       REJECTED  {tail[:36]}")
                continue
            got = run(p, x, w2.shape[0], S)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                eng.submit(p, procedure_index=0)
                ts.append((time.perf_counter() - t0) * 1e3)
            del p
            ts.sort()
            print(f"   {arm:>13} {rel_l2(got, ref):>15.4e} {ts[2]:>8.3f}")
        print()


if __name__ == "__main__":
    main()
