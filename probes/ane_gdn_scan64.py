"""Qualify exact multi-token Qwen3.8 GDN prefill formulations on the ANE.

``affine`` is an associative dense-transition scan, ``chunk`` is Qwen's
lower-triangular token-space algorithm, and the default ``fused`` path places
the exact ordered recurrence inside one ANE evaluation. The first two expose
private-compiler boundaries; fused is the compiler-safe deployed formulation.

Inputs come from the real layer-0 int4 projection and causal convolution. The
probe compares every output and the final state against both a NumPy sequential
oracle and the production resident ANE recurrence. Arbitrary prefix-state
continuation is covered so independently compiled blocks can chain and resume
the production prefix cache.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import os
import re
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.pure_ane import (  # noqa: E402
    AneDriver,
    AneGdnConv,
    AneGdnRecurrence,
    AneNormProjection,
    Checkpoint,
    StandaloneTokenizer,
    assert_standalone,
)


H = 48
D = 128


def _slice(name: str, source: str, c0: int, c1: int,
           shape: str) -> str:
    return (
        f'    tensor<fp16, [{shape}]> {name} = slice_by_index('
        f'begin=tensor<int32, [4]>([0,{c0},0,0]), '
        f'end=tensor<int32, [4]>([1,{c1},1,{D}]), x={source})'
        f'[name=string("{name}")];'
    )


def scan_mil(module, tokens: int) -> tuple[str, int, int]:
    if tokens < 1 or tokens & (tokens - 1):
        raise ValueError("scan token count must be a power of two")
    nh = tokens * H
    channels = 5 * nh
    output_channels = nh + H * D
    lines = [f'''program(1.3)
{module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {channels}, 1, {D}]> x) {{
    tensor<fp16, [1, 1, {D}, {D}]> ident = const()[name=string("ident"), val=tensor<fp16, [1, 1, {D}, {D}]>(BLOBFILE(path=string("@model_path/weights/identity.bin"), offset=uint64(64)))];
{_slice("qrow", "x", 0, nh, f"1, {nh}, 1, {D}")}
{_slice("krow", "x", nh, 2*nh, f"1, {nh}, 1, {D}")}
{_slice("vrow", "x", 2*nh, 3*nh, f"1, {nh}, 1, {D}")}
{_slice("drow", "x", 3*nh, 4*nh, f"1, {nh}, 1, {D}")}
{_slice("brow", "x", 4*nh, 5*nh, f"1, {nh}, 1, {D}")}
    tensor<fp16, [1, {nh}, 1, 1]> decay = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{nh},1,1]), x=drow)[name=string("decay")];
    tensor<fp16, [1, {nh}, 1, 1]> beta = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{nh},1,1]), x=brow)[name=string("beta")];
    tensor<fp16, [1, {nh}, {D}, 1]> kcol = reshape(shape=tensor<int32, [4]>([1,{nh},{D},1]), x=krow)[name=string("kcol")];
    tensor<fp16, [1, {nh}, {D}, 1]> vcol = reshape(shape=tensor<int32, [4]>([1,{nh},{D},1]), x=vrow)[name=string("vcol")];
    tensor<fp16, [1, {nh}, {D}, 1]> qcol = reshape(shape=tensor<int32, [4]>([1,{nh},{D},1]), x=qrow)[name=string("qcol")];
    tensor<fp16, [1, {nh}, {D}, {D}]> kk = matmul(transpose_x=bool(false), transpose_y=bool(false), x=kcol, y=krow)[name=string("kk")];
    tensor<fp16, [1, {nh}, {D}, {D}]> bkk = mul(x=kk, y=beta)[name=string("bkk")];
    tensor<fp16, [1, {nh}, {D}, {D}]> im = sub(x=ident, y=bkk)[name=string("im")];
    tensor<fp16, [1, {nh}, {D}, {D}]> m0 = mul(x=im, y=decay)[name=string("m0")];
    tensor<fp16, [1, {nh}, {D}, {D}]> vk = matmul(transpose_x=bool(false), transpose_y=bool(false), x=vcol, y=krow)[name=string("vk")];
    tensor<fp16, [1, {nh}, {D}, {D}]> b0 = mul(x=vk, y=beta)[name=string("b0")];''']

    mcur, bcur = "m0", "b0"
    offset = 1
    stage = 0
    while offset < tokens:
        cut = offset * H
        tail = nh - cut
        mn = f"m{stage+1}"
        bn = f"b{stage+1}"
        lines.append(f'''    tensor<fp16, [1, {tail}, {D}, {D}]> ml{stage} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{tail},{D},{D}]), x={mcur})[name=string("ml{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> mr{stage} = slice_by_index(begin=tensor<int32, [4]>([0,{cut},0,0]), end=tensor<int32, [4]>([1,{nh},{D},{D}]), x={mcur})[name=string("mr{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> bl{stage} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{tail},{D},{D}]), x={bcur})[name=string("bl{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> br{stage} = slice_by_index(begin=tensor<int32, [4]>([0,{cut},0,0]), end=tensor<int32, [4]>([1,{nh},{D},{D}]), x={bcur})[name=string("br{stage}")];
    tensor<fp16, [1, {cut}, {D}, {D}]> me{stage} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{cut},{D},{D}]), x={mcur})[name=string("me{stage}")];
    tensor<fp16, [1, {cut}, {D}, {D}]> be{stage} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{cut},{D},{D}]), x={bcur})[name=string("be{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> mt{stage} = matmul(transpose_x=bool(false), transpose_y=bool(false), x=ml{stage}, y=mr{stage})[name=string("mt{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> bt0_{stage} = matmul(transpose_x=bool(false), transpose_y=bool(false), x=bl{stage}, y=mr{stage})[name=string("bt0_{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> bt{stage} = add(x=bt0_{stage}, y=br{stage})[name=string("bt{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> mep{stage} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,{tail},0,0,0,0]), x=me{stage})[name=string("mep{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> mtp{stage} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{cut},0,0,0,0,0]), x=mt{stage})[name=string("mtp{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> {mn} = add(x=mep{stage}, y=mtp{stage})[name=string("{mn}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> bep{stage} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,{tail},0,0,0,0]), x=be{stage})[name=string("bep{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> btp{stage} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{cut},0,0,0,0,0]), x=bt{stage})[name=string("btp{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> {bn} = add(x=bep{stage}, y=btp{stage})[name=string("{bn}")];''')
        mcur, bcur = mn, bn
        stage += 1
        offset *= 2

    lines.append(f'''    tensor<fp16, [1, {nh}, {D}, 1]> y0 = matmul(transpose_x=bool(false), transpose_y=bool(false), x={bcur}, y=qcol)[name=string("y0")];
    tensor<fp16, [1, {nh}, {D}, 1]> y64 = mul(x=y0, y=fp16(0x1p+6))[name=string("y64")];
    tensor<fp16, [1, {nh}, 1, {D}]> yr = transpose(perm=tensor<int32, [4]>([0,1,3,2]), x=y64)[name=string("yr")];
    tensor<fp16, [1, {H}, {D}, {D}]> sf = slice_by_index(begin=tensor<int32, [4]>([0,{nh-H},0,0]), end=tensor<int32, [4]>([1,{nh},{D},{D}]), x={bcur})[name=string("sf")];
    tensor<fp16, [1, {H}, {D}, {D}]> sft = transpose(perm=tensor<int32, [4]>([0,1,3,2]), x=sf)[name=string("sft")];
    tensor<fp16, [1, {H*D}, 1, {D}]> sr = reshape(shape=tensor<int32, [4]>([1,{H*D},1,{D}]), x=sft)[name=string("sr")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> yp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,{H*D},0,0,0,0]), x=yr)[name=string("yp")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> sp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{nh},0,0,0,0,0]), x=sr)[name=string("sp")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> y = add(x=yp, y=sp)[name=string("y")];
  }} -> (y);
}}
// qwen38_gdn_affine_scan_h{H}_n{tokens}
''')
    return "\n".join(lines), channels, output_channels


class AneGdnAffineScan:
    def __init__(self, driver: AneDriver, tokens: int):
        self.driver = driver
        self.tokens = tokens
        self.nh = tokens * H
        mil, channels, output_channels = scan_mil(driver.module, tokens)
        blobs = {"identity.bin": np.eye(D, dtype=np.float16).tobytes()}
        capture = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(
                mil, blobs, channels, output_channels, D
            )
        self.compile_seconds = time.perf_counter() - started
        if self.program is None:
            detail = "\n".join(capture.getvalue().strip().splitlines()[-12:])
            raise RuntimeError(f"ANE GDN scan compile failed:\n{detail}")
        driver.engine._ensure_io(self.program)
        self.channels = channels
        self.output_channels = output_channels

    def load(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
             decay: np.ndarray, beta: np.ndarray) -> None:
        shape = (self.tokens, H, D)
        for name, value in (("q", q), ("k", k), ("v", v)):
            if value.shape != shape:
                raise ValueError(f"invalid {name} shape {value.shape}")
        if decay.shape != (self.tokens, H) or beta.shape != (self.tokens, H):
            raise ValueError("invalid gate shapes")
        nh = self.nh
        with self.driver.view(
            self.program._in_surf, (self.channels, D), np.float16
        ) as dst:
            dst[:] = 0
            dst[:nh] = q.reshape(nh, D)
            dst[nh:2*nh] = k.reshape(nh, D)
            dst[2*nh:3*nh] = v.reshape(nh, D)
            dst[3*nh:4*nh, 0] = decay.reshape(nh)
            dst[4*nh:5*nh, 0] = beta.reshape(nh)

    def run_loaded(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE GDN scan evaluation failed")
        nh = self.nh
        with self.driver.view(
            self.program._out_surf, (self.output_channels, D), np.float16
        ) as src:
            y = np.array(src[:nh], np.float32).reshape(self.tokens, H, D)
            state = np.array(src[nh:], np.float32).reshape(H, D, D)
        return y, state


def chunk_mil(module, tokens: int) -> tuple[str, int, int]:
    """Build the exact zero-initial-state Qwen chunked delta-rule graph."""
    if tokens < 2 or tokens & (tokens - 1):
        raise ValueError("chunk token count must be a power of two >= 2")
    nh = tokens * H
    channels = 5 * nh
    output_channels = nh + H * D
    lines = [f'''program(1.3)
{module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {channels}, 1, {D}]> x) {{
    tensor<fp16, [1, 1, {tokens}, {tokens}]> lower = const()[name=string("lower"), val=tensor<fp16, [1, 1, {tokens}, {tokens}]>(BLOBFILE(path=string("@model_path/weights/lower.bin"), offset=uint64(64)))];
    tensor<fp16, [1, 1, {tokens}, {tokens}]> strict = const()[name=string("strict"), val=tensor<fp16, [1, 1, {tokens}, {tokens}]>(BLOBFILE(path=string("@model_path/weights/strict.bin"), offset=uint64(64)))];
    tensor<fp16, [1, 1, {tokens}, {tokens}]> eye = const()[name=string("eye"), val=tensor<fp16, [1, 1, {tokens}, {tokens}]>(BLOBFILE(path=string("@model_path/weights/eye.bin"), offset=uint64(64)))];
{_slice("qrow_c", "x", 0, nh, f"1, {nh}, 1, {D}")}
{_slice("krow_c", "x", nh, 2*nh, f"1, {nh}, 1, {D}")}
{_slice("vrow_c", "x", 2*nh, 3*nh, f"1, {nh}, 1, {D}")}
{_slice("grow_c", "x", 3*nh, 4*nh, f"1, {nh}, 1, {D}")}
{_slice("brow_c", "x", 4*nh, 5*nh, f"1, {nh}, 1, {D}")}
    tensor<fp16, [1, {nh}, 1, 1]> gflat = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{nh},1,1]), x=grow_c)[name=string("gflat")];
    tensor<fp16, [1, {nh}, 1, 1]> bflat = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{nh},1,1]), x=brow_c)[name=string("bflat")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},{D}]), x=qrow_c)[name=string("q")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> k = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},{D}]), x=krow_c)[name=string("k")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> v = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},{D}]), x=vrow_c)[name=string("v")];
    tensor<fp16, [1, {H}, {tokens}, 1]> g = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},1]), x=gflat)[name=string("g")];
    tensor<fp16, [1, {H}, {tokens}, 1]> beta = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},1]), x=bflat)[name=string("beta")];
    tensor<fp16, [1, {H}, 1, {tokens}]> gt = transpose(perm=tensor<int32, [4]>([0,1,3,2]), x=g)[name=string("gt")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> gd = sub(x=g, y=gt)[name=string("gd")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> de = exp(x=gd)[name=string("de")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> dm = mul(x=de, y=lower)[name=string("dm")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> kb = mul(x=k, y=beta)[name=string("kb")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> vb = mul(x=v, y=beta)[name=string("vb")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> kk = matmul(transpose_x=bool(false), transpose_y=bool(true), x=kb, y=k)[name=string("kk")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> kkd = mul(x=kk, y=dm)[name=string("kkd")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> nk = mul(x=kkd, y=fp16(-0x1p+0))[name=string("nk")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> a0 = mul(x=nk, y=strict)[name=string("a0")];''']

    current = "a0"
    for i in range(1, tokens):
        if i == 1:
            # The i=1 correction is a 1x1 matmul, which the compiler rejects
            # with InvalidMILProgram -- and since every token count runs this
            # iteration, it is why the chunk graph failed at every size, T=2
            # included. At i=1 the product is a per-head scalar, so `mul` is
            # exact and accepted.
            lines.append(f'''    tensor<fp16, [1, {H}, 1, 1]> row1 = slice_by_index(begin=tensor<int32, [4]>([0,0,1,0]), end=tensor<int32, [4]>([1,{H},2,1]), x={current})[name=string("row1")];
    tensor<fp16, [1, {H}, 1, 1]> sub1 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{H},1,1]), x={current})[name=string("sub1")];
    tensor<fp16, [1, {H}, 1, 1]> corr1 = mul(x=row1, y=sub1)[name=string("corr1")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> cp1 = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,0,1,{tokens-2},0,{tokens-1}]), x=corr1)[name=string("cp1")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> a1 = add(x={current}, y=cp1)[name=string("a1")];''')
            current = "a1"
            continue
        lines.append(f'''    tensor<fp16, [1, {H}, 1, {i}]> row{i} = slice_by_index(begin=tensor<int32, [4]>([0,0,{i},0]), end=tensor<int32, [4]>([1,{H},{i+1},{i}]), x={current})[name=string("row{i}")];
    tensor<fp16, [1, {H}, {i}, {i}]> sub{i} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{H},{i},{i}]), x={current})[name=string("sub{i}")];
    tensor<fp16, [1, {H}, 1, {i}]> corr{i} = matmul(transpose_x=bool(false), transpose_y=bool(false), x=row{i}, y=sub{i})[name=string("corr{i}")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> cp{i} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,0,{i},{tokens-i-1},0,{tokens-i}]), x=corr{i})[name=string("cp{i}")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> a{i} = add(x={current}, y=cp{i})[name=string("a{i}")];''')
        current = f"a{i}"

    lines.append(f'''    tensor<fp16, [1, {H}, {tokens}, {tokens}]> ai = add(x={current}, y=eye)[name=string("ai")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> u = matmul(transpose_x=bool(false), transpose_y=bool(false), x=ai, y=vb)[name=string("u")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> qk = matmul(transpose_x=bool(false), transpose_y=bool(true), x=q, y=k)[name=string("qk")];
    tensor<fp16, [1, {H}, {tokens}, {tokens}]> qkd = mul(x=qk, y=dm)[name=string("qkd")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> yo = matmul(transpose_x=bool(false), transpose_y=bool(false), x=qkd, y=u)[name=string("yo")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> y64 = mul(x=yo, y=fp16(0x1p+6))[name=string("y64")];
    tensor<fp16, [1, {tokens}, {H}, {D}]> yt = transpose(perm=tensor<int32, [4]>([0,2,1,3]), x=y64)[name=string("yt")];
    tensor<fp16, [1, {nh}, 1, {D}]> yr = reshape(shape=tensor<int32, [4]>([1,{nh},1,{D}]), x=yt)[name=string("yr")];
    tensor<fp16, [1, {H}, 1, 1]> glast = slice_by_index(begin=tensor<int32, [4]>([0,0,{tokens-1},0]), end=tensor<int32, [4]>([1,{H},{tokens},1]), x=g)[name=string("glast")];
    tensor<fp16, [1, {H}, 1, {tokens}]> kgd0 = sub(x=glast, y=gt)[name=string("kgd0")];
    tensor<fp16, [1, {H}, 1, {tokens}]> kgd = exp(x=kgd0)[name=string("kgd")];
    tensor<fp16, [1, {H}, {tokens}, 1]> kgdt = transpose(perm=tensor<int32, [4]>([0,1,3,2]), x=kgd)[name=string("kgdt")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> kw = mul(x=k, y=kgdt)[name=string("kw")];
    tensor<fp16, [1, {H}, {D}, {D}]> sf = matmul(transpose_x=bool(true), transpose_y=bool(false), x=kw, y=u)[name=string("sf")];
    tensor<fp16, [1, {H}, {D}, {D}]> sft = transpose(perm=tensor<int32, [4]>([0,1,3,2]), x=sf)[name=string("sft")];
    tensor<fp16, [1, {H*D}, 1, {D}]> sr = reshape(shape=tensor<int32, [4]>([1,{H*D},1,{D}]), x=sft)[name=string("sr")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> yp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,{H*D},0,0,0,0]), x=yr)[name=string("yp")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> sp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{nh},0,0,0,0,0]), x=sr)[name=string("sp")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> y = add(x=yp, y=sp)[name=string("y")];
  }} -> (y);
}}
// qwen38_gdn_chunk_h{H}_n{tokens}
''')
    return "\n".join(lines), channels, output_channels


class AneGdnChunk(AneGdnAffineScan):
    """Real-shape Qwen chunk graph with the probe's packed IOSurface API."""

    def __init__(self, driver: AneDriver, tokens: int):
        self.driver = driver
        self.tokens = tokens
        self.nh = tokens * H
        mil, channels, output_channels = chunk_mil(driver.module, tokens)
        blobs = {
            "lower.bin": np.tril(np.ones((tokens, tokens), np.float16)).tobytes(),
            "strict.bin": np.tril(
                np.ones((tokens, tokens), np.float16), -1
            ).tobytes(),
            "eye.bin": np.eye(tokens, dtype=np.float16).tobytes(),
        }
        capture = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(
                mil, blobs, channels, output_channels, D
            )
        self.compile_seconds = time.perf_counter() - started
        if self.program is None:
            detail = "\n".join(capture.getvalue().strip().splitlines()[-12:])
            raise RuntimeError(f"ANE GDN chunk compile failed:\n{detail}")
        driver.engine._ensure_io(self.program)
        self.channels = channels
        self.output_channels = output_channels

    def load(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
             decay: np.ndarray, beta: np.ndarray) -> None:
        shape = (self.tokens, H, D)
        for name, value in (("q", q), ("k", k), ("v", v)):
            if value.shape != shape:
                raise ValueError(f"invalid {name} shape {value.shape}")
        if decay.shape != (self.tokens, H) or beta.shape != (self.tokens, H):
            raise ValueError("invalid gate shapes")
        cumulative_log_decay = np.cumsum(
            np.log(decay.astype(np.float32)), axis=0
        ).astype(np.float16)
        nh = self.nh
        with self.driver.view(
            self.program._in_surf, (self.channels, D), np.float16
        ) as dst:
            dst[:] = 0
            dst[:nh] = q.transpose(1, 0, 2).reshape(nh, D)
            dst[nh:2*nh] = k.transpose(1, 0, 2).reshape(nh, D)
            dst[2*nh:3*nh] = v.transpose(1, 0, 2).reshape(nh, D)
            dst[3*nh:4*nh, 0] = cumulative_log_decay.transpose(1, 0).reshape(nh)
            dst[4*nh:5*nh, 0] = beta.transpose(1, 0).reshape(nh)


def unrolled_mil(module, tokens: int) -> tuple[str, int, int]:
    """Fuse exact recurrent steps into one compiler-safe ANE evaluation."""
    if tokens < 2 or tokens & (tokens - 1):
        raise ValueError("unrolled token count must be a power of two >= 2")
    nh = tokens * H
    hk = H * D
    channels = 5 * nh + 2 * H + hk
    output_channels = nh
    lines = [f'''program(1.3)
{module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {channels}, 1, {D}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gh = const()[name=string("gh"), val=int32({H})];
    tensor<fp16, [{H}, {D}, 1, 1]> gsum = const()[name=string("gsum"), val=tensor<fp16, [{H}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/sum.bin"), offset=uint64(64)))];
    tensor<fp16, [{H}, {D}, 1, 1]> gmean = const()[name=string("gmean"), val=tensor<fp16, [{H}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/mean.bin"), offset=uint64(64)))];
    tensor<fp16, [{hk}, 1, 1, 1]> grep = const()[name=string("grep"), val=tensor<fp16, [{hk}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/repeat.bin"), offset=uint64(64)))];
{_slice("qrow_u", "x", 0, nh, f"1, {nh}, 1, {D}")}
{_slice("krow_u", "x", nh, 2*nh, f"1, {nh}, 1, {D}")}
{_slice("vrow_u", "x", 2*nh, 3*nh, f"1, {nh}, 1, {D}")}
{_slice("arow_u", "x", 3*nh, 4*nh, f"1, {nh}, 1, {D}")}
{_slice("brow_u", "x", 4*nh, 5*nh, f"1, {nh}, 1, {D}")}
{_slice("alog_u", "x", 5*nh, 5*nh+H, f"1, {H}, 1, {D}")}
{_slice("dtbias_u", "x", 5*nh+H, 5*nh+2*H, f"1, {H}, 1, {D}")}
{_slice("state_u", "x", 5*nh+2*H, 5*nh+2*H+hk, f"1, {hk}, 1, {D}")}
    tensor<fp16, [1, {nh}, 1, 1]> aflat = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{nh},1,1]), x=arow_u)[name=string("aflat")];
    tensor<fp16, [1, {nh}, 1, 1]> bflat = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{nh},1,1]), x=brow_u)[name=string("bflat")];
    tensor<fp16, [1, {H}, 1, 1]> alog = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{H},1,1]), x=alog_u)[name=string("alog")];
    tensor<fp16, [1, {H}, 1, 1]> dtbias = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{H},1,1]), x=dtbias_u)[name=string("dtbias")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},{D}]), x=qrow_u)[name=string("q")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> k = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},{D}]), x=krow_u)[name=string("k")];
    tensor<fp16, [1, {H}, {tokens}, {D}]> v = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},{D}]), x=vrow_u)[name=string("v")];
    tensor<fp16, [1, {H}, {tokens}, 1]> avec = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},1]), x=aflat)[name=string("avec")];
    tensor<fp16, [1, {H}, {tokens}, 1]> beta = reshape(shape=tensor<int32, [4]>([1,{H},{tokens},1]), x=bflat)[name=string("beta")];''']

    state = ""
    outputs = []
    for i in range(tokens):
        lines.append(f'''    tensor<fp16, [1, {H}, 1, {D}]> qt{i} = slice_by_index(begin=tensor<int32, [4]>([0,0,{i},0]), end=tensor<int32, [4]>([1,{H},{i+1},{D}]), x=q)[name=string("qt{i}")];
    tensor<fp16, [1, {H}, 1, {D}]> kt{i} = slice_by_index(begin=tensor<int32, [4]>([0,0,{i},0]), end=tensor<int32, [4]>([1,{H},{i+1},{D}]), x=k)[name=string("kt{i}")];
    tensor<fp16, [1, {H}, 1, {D}]> vt{i} = slice_by_index(begin=tensor<int32, [4]>([0,0,{i},0]), end=tensor<int32, [4]>([1,{H},{i+1},{D}]), x=v)[name=string("vt{i}")];
    tensor<fp16, [1, {H}, 1, 1]> at{i} = slice_by_index(begin=tensor<int32, [4]>([0,0,{i},0]), end=tensor<int32, [4]>([1,{H},{i+1},1]), x=avec)[name=string("at{i}")];
    tensor<fp16, [1, {H}, 1, 1]> bl{i} = slice_by_index(begin=tensor<int32, [4]>([0,0,{i},0]), end=tensor<int32, [4]>([1,{H},{i+1},1]), x=beta)[name=string("bl{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> qr{i} = reshape(shape=tensor<int32, [4]>([1,{hk},1,1]), x=qt{i})[name=string("qr{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> kr{i} = reshape(shape=tensor<int32, [4]>([1,{hk},1,1]), x=kt{i})[name=string("kr{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> q8_{i} = mul(x=qr{i}, y=fp16(0x1p+4))[name=string("q8_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> qsq_{i} = mul(x=q8_{i}, y=q8_{i})[name=string("qsq_{i}")];
    tensor<fp16, [1, {H}, 1, 1]> qms_{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gmean, x=qsq_{i})[name=string("qms_{i}")];
    tensor<fp16, [1, {H}, 1, 1]> qmse_{i} = add(x=qms_{i}, y=fp16(0x1.0c8p-12))[name=string("qmse_{i}")];
    tensor<fp16, [1, {H}, 1, 1]> qsd_{i} = sqrt(x=qmse_{i})[name=string("qsd_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> qsdr_{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=qsd_{i})[name=string("qsdr_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> qu_{i} = real_div(x=q8_{i}, y=qsdr_{i})[name=string("qu_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> qf{i} = mul(x=qu_{i}, y=fp16(0x1p-1))[name=string("qf{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> k8_{i} = mul(x=kr{i}, y=fp16(0x1p+4))[name=string("k8_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ksq_{i} = mul(x=k8_{i}, y=k8_{i})[name=string("ksq_{i}")];
    tensor<fp16, [1, {H}, 1, 1]> kms_{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gmean, x=ksq_{i})[name=string("kms_{i}")];
    tensor<fp16, [1, {H}, 1, 1]> kmse_{i} = add(x=kms_{i}, y=fp16(0x1.0c8p-12))[name=string("kmse_{i}")];
    tensor<fp16, [1, {H}, 1, 1]> ksd_{i} = sqrt(x=kmse_{i})[name=string("ksd_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ksdr_{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=ksd_{i})[name=string("ksdr_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ku_{i} = real_div(x=k8_{i}, y=ksdr_{i})[name=string("ku_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> kf{i} = mul(x=ku_{i}, y=fp16(0x1.6ap-4))[name=string("kf{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> aa{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=at{i})[name=string("aa{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> dtc{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=dtbias)[name=string("dtc{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> alr{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=alog)[name=string("alr{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ap{i} = add(x=aa{i}, y=dtc{i})[name=string("ap{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> pos{i} = relu(x=ap{i})[name=string("pos{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ab{i} = abs(x=ap{i})[name=string("ab{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> nab{i} = mul(x=ab{i}, y=fp16(-0x1p+0))[name=string("nab{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> tt{i} = exp(x=nab{i})[name=string("tt{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> hp5_{i} = mul(x=tt{i}, y=fp16(-0x1.84p-6))[name=string("hp5_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ha4_{i} = add(x=hp5_{i}, y=fp16(0x1.9acp-4))[name=string("ha4_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> hm4_{i} = mul(x=ha4_{i}, y=tt{i})[name=string("hm4_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ha3_{i} = add(x=hm4_{i}, y=fp16(-0x1.ab4p-3))[name=string("ha3_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> hm3_{i} = mul(x=ha3_{i}, y=tt{i})[name=string("hm3_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ha2_{i} = add(x=hm3_{i}, y=fp16(0x1.4c4p-2))[name=string("ha2_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> hm2_{i} = mul(x=ha2_{i}, y=tt{i})[name=string("hm2_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ha1_{i} = add(x=hm2_{i}, y=fp16(-0x1.ff4p-2))[name=string("ha1_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> hm1_{i} = mul(x=ha1_{i}, y=tt{i})[name=string("hm1_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> ha0_{i} = add(x=hm1_{i}, y=fp16(0x1p+0))[name=string("ha0_{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> tail{i} = mul(x=tt{i}, y=ha0_{i})[name=string("tail{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> soft{i} = add(x=pos{i}, y=tail{i})[name=string("soft{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> alx{i} = exp(x=alr{i})[name=string("alx{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> asp{i} = mul(x=alx{i}, y=soft{i})[name=string("asp{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> nasp{i} = mul(x=asp{i}, y=fp16(-0x1p+0))[name=string("nasp{i}")];
    tensor<fp16, [1, {hk}, 1, 1]> dr{i} = exp(x=nasp{i})[name=string("dr{i}")];
    tensor<fp16, [1, {H}, 1, 1]> nb{i} = mul(x=bl{i}, y=fp16(-0x1p+0))[name=string("nb{i}")];
    tensor<fp16, [1, {H}, 1, 1]> enb{i} = exp(x=nb{i})[name=string("enb{i}")];
    tensor<fp16, [1, {H}, 1, 1]> bden{i} = add(x=enb{i}, y=fp16(0x1p+0))[name=string("bden{i}")];
    tensor<fp16, [1, {H}, 1, 1]> bt{i} = real_div(x=fp16(0x1p+0), y=bden{i})[name=string("bt{i}")];''')
        previous = "state_u" if i == 0 else state
        lines.append(f'''    tensor<fp16, [1, {hk}, 1, {D}]> sd{i} = mul(x={previous}, y=dr{i})[name=string("sd{i}")];
    tensor<fp16, [1, {hk}, 1, {D}]> sk{i} = mul(x=sd{i}, y=kf{i})[name=string("sk{i}")];
    tensor<fp16, [1, {H}, 1, {D}]> mem{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sk{i})[name=string("mem{i}")];
    tensor<fp16, [1, {H}, 1, {D}]> dv{i} = sub(x=vt{i}, y=mem{i})[name=string("dv{i}")];
    tensor<fp16, [1, {H}, 1, {D}]> db{i} = mul(x=dv{i}, y=bt{i})[name=string("db{i}")];
    tensor<fp16, [1, {hk}, 1, {D}]> du{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=db{i})[name=string("du{i}")];
    tensor<fp16, [1, {hk}, 1, {D}]> up{i} = mul(x=du{i}, y=kf{i})[name=string("up{i}")];
    tensor<fp16, [1, {hk}, 1, {D}]> s{i+1} = add(x=sd{i}, y=up{i})[name=string("s{i+1}")];''')
        state = f"s{i+1}"
        outputs.append(state)

    padded = []
    for i, token_state in enumerate(outputs):
        lines.append(f'''    tensor<fp16, [1, {hk}, 1, {D}]> sq{i} = mul(x={token_state}, y=qf{i})[name=string("sq{i}")];
    tensor<fp16, [1, {H}, 1, {D}]> yo{i} = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sq{i})[name=string("yo{i}")];
    tensor<fp16, [1, {nh}, 1, {D}]> yp{i} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{i*H},{(tokens-i-1)*H},0,0,0,0]), x=yo{i})[name=string("yp{i}")];''')
        padded.append(f"yp{i}")
    ysum = padded[0]
    for i, output in enumerate(padded[1:], 1):
        lines.append(f'    tensor<fp16, [1, {nh}, 1, {D}]> ys{i} = add(x={ysum}, y={output})[name=string("ys{i}")];')
        ysum = f"ys{i}"
    lines.append(f'''    tensor<fp16, [1, {H}, {D}, {D}]> sf0 = reshape(shape=tensor<int32, [4]>([1,{H},{D},{D}]), x={state})[name=string("sf0")];
    tensor<fp16, [1, {H}, {D}, {D}]> sft = transpose(perm=tensor<int32, [4]>([0,1,3,2]), x=sf0)[name=string("sft")];
    tensor<fp16, [1, {nh}, 1, {D}]> y = identity(x={ysum})[name=string("y")];
    tensor<fp16, [1, {H}, {D}, {D}]> sout = identity(x=sft)[name=string("sout")];
  }} -> (y, sout);
}}
// qwen38_gdn_unrolled_h{H}_n{tokens}
''')
    return "\n".join(lines), channels, output_channels


class AneGdnUnrolled(AneGdnAffineScan):
    """One-dispatch, exact sequential GDN prefill block."""

    def __init__(self, driver: AneDriver, tokens: int):
        self.driver = driver
        self.tokens = tokens
        self.nh = tokens * H
        mil, channels, output_channels = unrolled_mil(driver.module, tokens)
        blobs = {
            "sum.bin": np.ones((H, D, 1, 1), np.float16).tobytes(),
            "mean.bin": np.full((H, D, 1, 1), 1.0/D, np.float16).tobytes(),
            "repeat.bin": np.ones((H*D, 1, 1, 1), np.float16).tobytes(),
        }
        capture = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(
                mil, blobs, channels, output_channels, D
            )
        self.compile_seconds = time.perf_counter() - started
        if self.program is None:
            detail = "\n".join(capture.getvalue().strip().splitlines()[-12:])
            raise RuntimeError(f"ANE GDN fused block compile failed:\n{detail}")
        driver.engine._ensure_io(self.program)
        self.channels = channels
        self.output_channels = output_channels
        self.state_surface = driver.module._create_iosurface(
            driver.module._iosurface_alloc_size(H * D * D)
        )
        if not self.state_surface:
            raise RuntimeError("ANE GDN fused state IOSurface allocation failed")
        inner = driver.module._msg(
            self.program.model, "model"
        ) or self.program.model
        desc = driver.module._desc(driver.module._msg(inner, "description"))
        output_channels = [int(ch) for ch, _, _ in re.findall(
            r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";',
            desc, re.S
        )]
        if sorted(output_channels) != sorted((self.nh, H)):
            raise RuntimeError(
                f"unexpected fused GDN outputs {output_channels}"
            )
        surfaces = {self.nh: self.program._out_surf, H: self.state_surface}
        init = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
        self.request = init(("objc_msgSend", driver.module._objc))(
            driver.module._msg(driver.module._cls("_ANERequest"), "alloc"),
            driver.module._sel(
                "initWithInputs:inputIndices:outputs:outputIndices:"
                "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                "transactionHandle:"
            ),
            driver.module._nsarray([
                driver.module._wrap_iosurface(self.program._in_surf)
            ]),
            driver.module._nsarray([driver.module._nsnumber_int(0)]),
            driver.module._nsarray([
                driver.module._wrap_iosurface(surfaces[c])
                for c in output_channels
            ]),
            driver.module._nsarray([
                driver.module._nsnumber_int(i)
                for i in range(len(output_channels))
            ]),
            None, None, driver.module._nsnumber_int(0), None, None
        )
        if not self.request:
            raise RuntimeError("ANE GDN fused request creation failed")
        self._Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )

    def load(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
             a: np.ndarray, beta_logits: np.ndarray,
             a_log: np.ndarray, dt_bias: np.ndarray,
             initial_state: np.ndarray | None = None) -> None:
        if q.shape != (self.tokens, 16, D) or k.shape != q.shape:
            raise ValueError(f"invalid raw q/k shapes {q.shape}/{k.shape}")
        if v.shape != (self.tokens, H, D):
            raise ValueError(f"invalid v shape {v.shape}")
        if a.shape != (self.tokens, H) or beta_logits.shape != a.shape:
            raise ValueError("invalid gate-logit shapes")
        if a_log.shape != (H,) or dt_bias.shape != (H,):
            raise ValueError("invalid gate-parameter shapes")
        if initial_state is None:
            initial_state = np.zeros((H, D, D), np.float16)
        if initial_state.shape != (H, D, D):
            raise ValueError(f"invalid initial state {initial_state.shape}")
        nh = self.nh
        with self.driver.view(
            self.program._in_surf, (self.channels, D), np.float16
        ) as dst:
            dst[:] = 0
            q48 = np.repeat(q.astype(np.float16, copy=False), 3, axis=1)
            k48 = np.repeat(k.astype(np.float16, copy=False), 3, axis=1)
            dst[:nh] = q48.transpose(1, 0, 2).reshape(nh, D)
            dst[nh:2*nh] = k48.transpose(1, 0, 2).reshape(nh, D)
            dst[2*nh:3*nh] = v.transpose(1, 0, 2).reshape(nh, D)
            dst[3*nh:4*nh, 0] = a.transpose(1, 0).reshape(nh)
            dst[4*nh:5*nh, 0] = beta_logits.transpose(1, 0).reshape(nh)
            dst[5*nh:5*nh+H, 0] = a_log
            dst[5*nh+H:5*nh+2*H, 0] = dt_bias
            dst[5*nh+2*H:] = initial_state.astype(
                np.float16, copy=False
            ).transpose(0, 2, 1).reshape(H*D, D)

    def run_loaded(self) -> tuple[np.ndarray, np.ndarray]:
        error = ctypes.c_void_p(0)
        ok = self._Eval(("objc_msgSend", self.driver.module._objc))(
            self.program.model,
            self.driver.module._sel("evaluateWithQoS:options:request:error:"),
            21, self.program._compile_opts, self.request, ctypes.byref(error)
        )
        if not ok:
            detail = self.driver.module._desc(error.value) \
                if error.value else "unknown"
            raise RuntimeError(f"ANE GDN fused evaluation failed: {detail}")
        with self.driver.view(
            self.program._out_surf, (self.nh, D), np.float16
        ) as ysrc, self.driver.view(
            self.state_surface, (H, D, D), np.float16
        ) as ssrc:
            y = np.array(ysrc, np.float32).reshape(self.tokens, H, D)
            state = np.array(ssrc, np.float32)
        return y, state


def actual_inputs(driver: AneDriver, checkpoint: Checkpoint,
                  token_ids: list[int], bits: int) -> tuple[np.ndarray, ...]:
    prefix = "model.language_model.layers.0"
    names = [f"{prefix}.linear_attn.{name}.weight" for name in
             ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
    head = AneNormProjection(
        driver, checkpoint, f"{prefix}.input_layernorm.weight", names,
        bits=bits, tag="scan64_layer0_head", active_lanes=3
    )
    conv = AneGdnConv(
        driver, checkpoint, f"{prefix}.linear_attn.conv1d.weight"
    )
    raw_q, raw_k, values, avec, bvec = [], [], [], [], []
    for start in range(0, len(token_ids), 3):
        ids = token_ids[start:start+3]
        hidden = np.stack([checkpoint.embedding(t) for t in ids], axis=1)
        projection = head(hidden)
        activated = conv(projection[:10240])
        if projection.ndim == 1:
            projection = projection[:, None]
            activated = activated[:, None]
        for lane in range(len(ids)):
            raw_q.append(activated[:2048, lane].reshape(16, D))
            raw_k.append(activated[2048:4096, lane].reshape(16, D))
            values.append(activated[4096:10240, lane].reshape(H, D))
            bvec.append(projection[16384:16432, lane])
            avec.append(projection[16432:16480, lane])
    return tuple(np.asarray(x, np.float16) for x in
                 (raw_q, raw_k, values, avec, bvec))


def scan_inputs(raw_q: np.ndarray, raw_k: np.ndarray, values: np.ndarray,
                avec: np.ndarray, bvec: np.ndarray,
                a_log: np.ndarray, dt_bias: np.ndarray) -> tuple[np.ndarray, ...]:
    q = raw_q.astype(np.float32)
    k = raw_k.astype(np.float32)
    qn = q / np.sqrt(np.mean(q*q, axis=-1, keepdims=True) + 1e-6) / D
    kn = k / np.sqrt(np.mean(k*k, axis=-1, keepdims=True) + 1e-6) / np.sqrt(D)
    q48 = np.repeat(qn, 3, axis=1).astype(np.float16)
    k48 = np.repeat(kn, 3, axis=1).astype(np.float16)
    decay = np.exp(
        -np.exp(a_log.astype(np.float32))[None, :]
        * np.logaddexp(avec.astype(np.float32)
                       + dt_bias.astype(np.float32)[None, :], 0.0)
    ).astype(np.float16)
    beta = (1.0 / (1.0 + np.exp(-bvec.astype(np.float32)))).astype(np.float16)
    return q48, k48, values.astype(np.float16), decay, beta


def numpy_sequential(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                     decay: np.ndarray, beta: np.ndarray,
                     initial_state: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((H, D, D), np.float32) if initial_state is None \
        else initial_state.astype(np.float32, copy=True)
    outputs = []
    for position in range(q.shape[0]):
        state *= decay[position, :, None, None].astype(np.float32)
        memory = np.sum(state * k[position, :, None, :].astype(np.float32), axis=-1)
        delta = (v[position].astype(np.float32) - memory) \
                * beta[position, :, None].astype(np.float32)
        state += delta[:, :, None] * k[position, :, None, :].astype(np.float32)
        outputs.append(64.0 * np.sum(
            state * q[position, :, None, :].astype(np.float32), axis=-1
        ))
    return np.asarray(outputs), state


def relative(got: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(got-expected)) /
                 (np.max(np.abs(expected)) + 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get(
        "Q38_MODEL", "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B"))
    parser.add_argument("--engine-path", default=os.environ.get(
        "Q38_ANE_ENGINE", ROOT))
    parser.add_argument("--tokens", type=int, choices=(2,4,8,16,32,64), default=64)
    parser.add_argument("--bits", type=int, choices=(4,8,16), default=4)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--prefix-tokens", type=int, default=0)
    parser.add_argument(
        "--algorithm", choices=("fused", "chunk", "affine"), default="fused"
    )
    args = parser.parse_args()
    if args.prefix_tokens and args.algorithm != "fused":
        parser.error("--prefix-tokens is currently supported by fused only")

    assert_standalone("GDN scan probe startup")
    checkpoint = Checkpoint(args.model)
    tokenizer = StandaloneTokenizer(args.model)
    seed_text = (
        "Implement a persistent inference server, explain each optimization, "
        "and verify every numerical result before changing the runtime. " * 16
    )
    total_tokens = args.prefix_tokens + args.tokens
    token_ids = tokenizer.encode(seed_text)[:total_tokens]
    if len(token_ids) != total_tokens:
        raise RuntimeError("probe prompt did not produce enough tokens")
    driver = AneDriver(args.engine_path)
    raw_q, raw_k, values, avec, bvec = actual_inputs(
        driver, checkpoint, token_ids, args.bits
    )
    prefix = "model.language_model.layers.0.linear_attn"
    a_log = checkpoint.tensor(f"{prefix}.A_log", np.float16)
    dt_bias = checkpoint.tensor(f"{prefix}.dt_bias", np.float16)
    all_q, all_k, all_v, all_decay, all_beta = scan_inputs(
        raw_q, raw_k, values, avec, bvec, a_log, dt_bias
    )
    prefix = args.prefix_tokens
    initial_state = None
    if prefix:
        _, initial_state = numpy_sequential(
            all_q[:prefix], all_k[:prefix], all_v[:prefix],
            all_decay[:prefix], all_beta[:prefix]
        )
    q, k, v = all_q[prefix:], all_k[prefix:], all_v[prefix:]
    decay, beta = all_decay[prefix:], all_beta[prefix:]
    raw_q, raw_k, values = raw_q[prefix:], raw_k[prefix:], values[prefix:]
    avec, bvec = avec[prefix:], bvec[prefix:]
    expected_y, expected_state = numpy_sequential(
        q, k, v, decay, beta, initial_state
    )

    recurrence = AneGdnRecurrence(driver)
    state = recurrence.new_state()

    def sequential_ane() -> tuple[np.ndarray, np.ndarray]:
        saved = np.zeros((H*D, D), np.float16) if initial_state is None \
            else initial_state.astype(np.float16).transpose(0, 2, 1).reshape(
                H*D, D
            )
        recurrence.restore(state, saved)
        ys = []
        for position in range(args.tokens):
            ys.append(recurrence(
                state, raw_q[position], raw_k[position], values[position],
                avec[position], bvec[position], a_log, dt_bias
            ).astype(np.float32))
        return np.asarray(ys), recurrence.materialize(state)

    seq_y, seq_state = sequential_ane()
    implementation = {
        "fused": AneGdnUnrolled,
        "chunk": AneGdnChunk,
        "affine": AneGdnAffineScan,
    }[args.algorithm]
    scan = implementation(driver, args.tokens)
    if args.algorithm == "fused":
        scan.load(
            raw_q, raw_k, values, avec, bvec, a_log, dt_bias, initial_state
        )
    else:
        scan.load(q, k, v, decay, beta)
    scan_y, scan_state = scan.run_loaded()

    scan_y_rel = relative(scan_y, expected_y)
    scan_state_rel = relative(scan_state, expected_state)
    seq_y_rel = relative(seq_y, expected_y)
    seq_state_rel = relative(seq_state, expected_state)
    fused_vs_step_y = relative(scan_y, seq_y)
    fused_vs_step_state = relative(scan_state, seq_state)

    # Warm both paths, then include the production state copies/parameter
    # writes for sequential timing. Scan timing uses an already-loaded input so
    # the large transform graph itself is visible separately from host loading.
    scan.run_loaded()
    sequential_ane()
    scan_ms = []
    seq_ms = []
    for _ in range(args.runs):
        started = time.perf_counter();scan.run_loaded()
        scan_ms.append((time.perf_counter()-started)*1e3)
        started = time.perf_counter();sequential_ane()
        seq_ms.append((time.perf_counter()-started)*1e3)

    scan_median = statistics.median(scan_ms)
    seq_median = statistics.median(seq_ms)
    # The deployable graph must reproduce the already-qualified production ANE
    # recurrence. Long fp16 state chains can diverge from the float32 oracle
    # while both ANE paths remain identical, so hidden-state oracle error is
    # reported but is not by itself an integration failure.
    passed = (scan_y_rel < 0.04 and fused_vs_step_y < 0.005
              and fused_vs_step_state < 0.005)
    faster = scan_median < seq_median
    print(f"GDN_BLOCK algorithm={args.algorithm} tokens={args.tokens} "
          f"prefix_tokens={args.prefix_tokens} "
          f"heads={H} dim={D} int{args.bits}")
    print(f"compile_seconds={scan.compile_seconds:.3f}")
    print(f"scan_y_relative_error={scan_y_rel:.6g}")
    print(f"scan_state_relative_error={scan_state_rel:.6g}")
    print(f"sequential_y_relative_error={seq_y_rel:.6g}")
    print(f"sequential_state_relative_error={seq_state_rel:.6g}")
    print(f"fused_vs_step_y_relative_error={fused_vs_step_y:.6g}")
    print(f"fused_vs_step_state_relative_error={fused_vs_step_state:.6g}")
    print(f"scan_median_ms={scan_median:.3f}")
    print(f"sequential_median_ms={seq_median:.3f}")
    print(f"speedup={seq_median/scan_median:.3f}x")
    label = args.algorithm.upper()
    print(f"GDN_{label}_NUMERICS=" + ("PASS" if passed else "FAIL"))
    print(f"GDN_{label}_INTEGRATION="
          + ("GO" if passed and faster else "NO_GO"))
    assert_standalone("GDN scan probe completion")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
