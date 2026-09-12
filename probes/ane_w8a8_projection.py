#!/usr/bin/env python3
"""Does the Anemll W8A8 result survive real projection shapes?

The gist (gist.github.com/Anemll/49e219448ad350ef67ff4bfdcb9ebd8c) measures
36.8 TOPS on this M5 Max with square 512-channel weights reused across 4096
spatial positions. Our projections are the opposite regime: large output
dimension, weights read once per S-tile, spatial = [1, S].

So this probe rebuilds the same three arms at OUR layout and dims:

  fp16   fp16 weights, fp16 activations                      (control)
  w8a16  int8 weights via constexpr_affine_dequantize        (today's spelling)
  w8a8   int8 weights + quantize/dequantize between convs    (the gist recipe)

All three are a chain of `depth` convs inside ONE program, because the W8A8
saving is L2 traffic BETWEEN tiles -- a single-conv program is fp16-in/fp16-out
and can never show it. That is also why probes/ane_int8_tops_probe.py found
nothing: it tested int8 at the func signature, which ANECCompile rejects, and
it tested one conv.
"""
from __future__ import annotations

import contextlib
import io
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

eng = AneEngine()
rng = np.random.default_rng(0)

# Each conv gets scale 1/(4*sqrt(in_dim)) so a deep random chain is roughly
# magnitude-preserving; without that the activations either overflow fp16 or
# quantize to all-zero and the arms stop being comparable.
WMAX = 4
ACT_SCALE = 2.0 ** -10      # activations land near 0.1, so ~100 int8 levels


def _wscale(in_dim: int) -> float:
    return 1.0 / (WMAX * (in_dim ** 0.5))


def _chunk(payload: bytes, dtype_marker: int) -> bytes:
    """Gist-compatible milinternal chunk: 64B outer + 64B header + payload.

    MIL ``offset`` points at the DEADBEEF header; tensor bytes are at +64.
    """
    outer = bytearray(64)
    struct.pack_into("<II", outer, 0, 1, 2)
    hdr = bytearray(64)
    struct.pack_into("<I", hdr, 0, 0xDEADBEEF)
    struct.pack_into("<I", hdr, 4, 1)
    hdr[10] = dtype_marker
    return bytes(outer) + bytes(hdr) + payload


def _blob(shapes: list[tuple[int, int]], arm: str):
    """One weight.bin holding every conv's weights; returns bytes + offsets.

    Per-channel arms (`w8c`, `w4c`) also append an fp16 scale chunk per conv,
    because `constexpr_blockwise_shift_scale` rejects a scalar scale. Offsets
    become (data, scale) pairs for those arms.
    """
    chunks, offsets = [], []
    for out_dim, in_dim in shapes:
        base = sum(len(c) for c in chunks)
        q = rng.integers(-WMAX, WMAX + 1, (out_dim, in_dim)).astype(np.int8)
        if arm == "fp16":
            w = (q * _wscale(in_dim)).astype(np.float16)
            chunks.append(_chunk(w.tobytes(), 0x04))
            offsets.append(base + 64)
        elif arm in ("w8c", "w4c"):
            w = (q * _wscale(in_dim)).astype(np.float32)
            if arm == "w4c":
                packed, scale = E.quantize_linear_int4(w)
                chunks.append(_chunk(packed.tobytes(), 0x09))
            else:
                packed, scale = E.quantize_linear_int8(w)
                chunks.append(_chunk(packed.tobytes(), 0x08))
            s_base = base + len(chunks[-1])
            chunks.append(_chunk(
                np.asarray(scale, np.float16).reshape(out_dim, 1, 1, 1).tobytes(), 0x04))
            offsets.append((base + 64, s_base + 64))
        else:
            chunks.append(_chunk(q.tobytes(), 0x08))
            offsets.append(base + 64)
    return b"".join(chunks), offsets


_PREAMBLE = (
    '    string pt = const()[name=string("pt"), val=string("valid")];\n'
    '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '    int32 gr = const()[name=string("gr"), val=int32(1)];\n'
    f'    fp16 q_scale = const()[name=string("q_scale"), val=fp16({ACT_SCALE})];\n'
    '    string q_dtype = const()[name=string("q_dtype"), val=string("int8")];\n'
    f'    fp16 dq_scale = const()[name=string("dq_scale"), val=fp16({ACT_SCALE})];'
)


def build(shapes: list[tuple[int, int]], S: int, arm: str):
    """Compile a fused conv chain. shapes[i] = (out_dim, in_dim), chained."""
    blob, offs = _blob(shapes, arm)
    lines, prev = [], "x"
    for i, ((out_dim, in_dim), off) in enumerate(zip(shapes, offs)):
        if arm in ("w8c", "w4c"):
            dt = "int4" if arm == "w4c" else "int8"
            d_off, s_off = off
            lines.append(
                f'    tensor<{dt}, [{out_dim}, {in_dim}, 1, 1]> D{i} = const()'
                f'[name=string("D{i}"), val=tensor<{dt}, [{out_dim}, {in_dim}, 1, 1]>'
                f'(BLOBFILE(path=string("@model_path/weights/weight.bin"), '
                f'offset=uint64({d_off})))];')
            lines.append(
                f'    tensor<fp16, [{out_dim}, 1, 1, 1]> S{i} = const()'
                f'[name=string("S{i}"), val=tensor<fp16, [{out_dim}, 1, 1, 1]>'
                f'(BLOBFILE(path=string("@model_path/weights/weight.bin"), '
                f'offset=uint64({s_off})))];')
            lines.append(
                f'    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W{i} = '
                f'constexpr_blockwise_shift_scale(data=D{i}, scale=S{i})'
                f'[name=string("W{i}")];')
        elif arm == "fp16":
            lines.append(
                f'    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W{i} = const()'
                f'[name=string("W{i}"), val=tensor<fp16, [{out_dim}, {in_dim}, 1, 1]>'
                f'(BLOBFILE(path=string("@model_path/weights/weight.bin"), '
                f'offset=uint64({off})))];')
        else:
            lines.append(
                f'    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W{i} = '
                f'constexpr_affine_dequantize()[axis=int32(0), name=string("W{i}"), '
                f'quantized_data=tensor<int8, [{out_dim}, {in_dim}, 1, 1]>'
                f'(BLOBFILE(path=string("@model_path/weights/weight.bin"), '
                f'offset=uint64({off}))), scale=fp16({_wscale(in_dim)!r}), '
                f'zero_point=int8(0)];')
        lines.append(
            f'    tensor<fp16, [1, {out_dim}, 1, {S}]> c{i} = conv(dilations=dl, '
            f'groups=gr, pad=pd, pad_type=pt, strides=st, weight=W{i}, x={prev})'
            f'[name=string("c{i}")];')
        prev = f"c{i}"
        if arm == "w8a8" and i < len(shapes) - 1:
            lines.append(
                f'    tensor<int8, [1, {out_dim}, 1, {S}]> q{i} = quantize(input=c{i}, '
                f'output_dtype=q_dtype, scale=q_scale)[name=string("q{i}")];')
            lines.append(
                f'    tensor<fp16, [1, {out_dim}, 1, {S}]> d{i} = dequantize(input=q{i}, '
                f'scale=dq_scale)[name=string("d{i}")];')
            prev = f"d{i}"
    H = shapes[0][1]
    M = shapes[-1][0]
    mil = (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
           f"  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> x) {{\n"
           f"{_PREAMBLE}\n" + "\n".join(lines) + f"\n  }} -> ({prev});\n}}\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(mil, {"weight.bin": blob}, H, M, S,
                                      raw_weight_files=frozenset({"weight.bin"}))
        except Exception:
            p = None
    tail = "\n".join(buf.getvalue().strip().splitlines()[-2:])
    return p, tail


def measure(p, H, M, S, n=9):
    eng._ensure_io(p)
    x = np.ascontiguousarray((rng.standard_normal((H, S)) * 0.1).astype(np.float16))
    with _iosurface_view(p._in_surf, (H, S), np.float16) as dst:
        np.copyto(dst, x)
    eng.submit(p, procedure_index=0)
    with _iosurface_view(p._out_surf, (M, S), np.float16) as o:
        got = np.array(o, np.float32)
    finite = bool(np.isfinite(got).all() and np.abs(got).max() > 0)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        eng.submit(p, procedure_index=0)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], finite


# Real Qwen3.8-Flash-Next text-config dims (hidden 2560).
FAMILIES = {
    # gated delta net: in_proj 2560 -> 16384 (q,k = 16*128 each; v,z = 48*128
    # each), out_proj 6144 -> 2560. Chained as a repeating fused block.
    "gdn pair 2560<->16384": [(16384, 2560), (2560, 16384)],
    # square reference at hidden width, isolates depth from shape change
    "square 2560": [(2560, 2560)],
    # one routed expert: gate_up 2560 -> 2*640, then down back to 2560. The
    # probe omits SwiGLU, so down is measured with 1280 inputs rather than the
    # 640 it sees in the model; its weight bytes are therefore 2x the real one.
    "expert 2560<->1280": [(1280, 2560), (2560, 1280)],
}


def main() -> None:
    depth = int(os.environ.get("W8A8_DEPTH", "8"))
    widths = [int(v) for v in os.environ.get("W8A8_S", "128,256,512").split(",")]
    print(f"M5 Max / h17 / macOS 27  depth={depth} convs per program\n")
    for fam, block in FAMILIES.items():
        reps = max(1, depth // len(block))
        shapes = (block * reps)
        # a chain needs in_dim[i+1] == out_dim[i]; the blocks above already do
        print(f"== {fam}   {len(shapes)} convs")
        print(f"   {'S':>5} {'arm':>7} {'ms':>9} {'TOPS':>8} {'vs fp16':>8}  finite")
        for S in widths:
            gop = sum(2.0 * o * i * S for o, i in shapes) / 1e9
            base = None
            for arm in [a for a in os.environ.get(
                    "W8A8_ARMS", "fp16,w8a16,w8a8").split(",") if a]:
                p, tail = build(shapes, S, arm)
                if p is None:
                    print(f"   {S:>5} {arm:>7}   REJECTED  {tail[:40]}")
                    continue
                ms, finite = measure(p, shapes[0][1], shapes[-1][0], S)
                del p
                tops = gop / ms if ms else 0.0
                if arm == "fp16":
                    base = tops
                ratio = f"{tops/base:.2f}x" if base else "-"
                print(f"   {S:>5} {arm:>7} {ms:>8.3f} {tops:>8.2f} {ratio:>8}  {finite}")
        print()


if __name__ == "__main__":
    main()
