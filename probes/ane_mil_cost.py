#!/usr/bin/env python3
"""Measure the shape-dependent cost of small text-MIL programs on the ANE.

This deliberately times only ``AneEngine.submit``.  Compilation, IOSurface
allocation and host copies happen before the timed region.  The harness uses
the same ``compile_multiproc`` path as the Flash-Next layer, so the results are
about the program the private compiler actually emits rather than a Core ML
proxy.

Examples::

    ~/.rindi/venvs/coreai/bin/python probes/ane_mil_cost.py --suite core
    ~/.rindi/venvs/coreai/bin/python probes/ane_mil_cost.py --suite layout \
        --ops square,tanh,reduce_mean --json /tmp/ane-layout.json
    ~/.rindi/venvs/coreai/bin/python probes/ane_mil_cost.py --suite conv

``fp16`` is a real fp16 data path. ``int8_qdq`` keeps fp16 external surfaces
and inserts the supported int8 quantize/dequantize pair around the measured
chain.  ``fp32_io`` and ``int8_io`` intentionally try those signature dtypes;
on current firmware their compile failures are useful measurements too.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime.q38_ane_engine as E  # noqa: E402
from runtime.q38_ane_engine import AneEngine, _iosurface_view  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    op: str
    shape: tuple[int, ...]
    dtype: str = "fp16"
    axis: int | None = None
    depth: int = 8
    layout: str = "plain"
    out_channels: int | None = None
    parts: int = 1


def _shape(s: tuple[int, ...]) -> str:
    return ", ".join(map(str, s))


def _tensor(dtype: str, shape: tuple[int, ...]) -> str:
    return f"tensor<{dtype}, [{_shape(shape)}]>"


def _full_shape(shape: tuple[int, ...], axis: int, value: int = 1) -> tuple[int, ...]:
    out = list(shape)
    out[axis % len(shape)] = value
    return tuple(out)


def _op_chain(case: Case, src: str, dtype: str) -> tuple[list[str], str, tuple[int, ...]]:
    """Return MIL lines, final symbol and its shape."""
    lines: list[str] = []
    cur = src
    shp = case.shape
    t = _tensor(dtype, shp)
    axis = case.axis if case.axis is not None else 1

    if case.op == "reshape_roundtrip":
        if len(shp) != 4:
            raise ValueError("reshape_roundtrip needs rank 4")
        n = int(np.prod(shp))
        alt = (1, shp[1] * shp[2], 1, shp[3])
        lines += [
            f'    tensor<int32, [4]> ash = const()[name=string("ash"), '
            f'val=tensor<int32, [4]>([{_shape(alt)}])];',
            f'    tensor<int32, [4]> osh = const()[name=string("osh"), '
            f'val=tensor<int32, [4]>([{_shape(shp)}])];',
        ]
        for i in range(case.depth):
            lines.append(f'    {_tensor(dtype, alt)} a{i} = reshape(x={cur}, shape=ash)'
                         f'[name=string("a{i}")];')
            lines.append(f'    {t} z{i} = reshape(x=a{i}, shape=osh)[name=string("z{i}")];')
            cur = f"z{i}"
        assert n == int(np.prod(alt))
        return lines, cur, shp

    if case.op == "transpose_roundtrip":
        rank = len(shp)
        a, b = axis % rank, (axis + 1) % rank
        perm = list(range(rank))
        perm[a], perm[b] = perm[b], perm[a]
        tsh = tuple(shp[i] for i in perm)
        inv = [perm.index(i) for i in range(rank)]
        lines += [
            f'    tensor<int32, [{rank}]> pm = const()[name=string("pm"), '
            f'val=tensor<int32, [{rank}]>([{_shape(tuple(perm))}])];',
            f'    tensor<int32, [{rank}]> pi = const()[name=string("pi"), '
            f'val=tensor<int32, [{rank}]>([{_shape(tuple(inv))}])];',
        ]
        for i in range(case.depth):
            lines.append(f'    {_tensor(dtype, tsh)} a{i} = transpose(x={cur}, perm=pm)'
                         f'[name=string("a{i}")];')
            lines.append(f'    {t} z{i} = transpose(x=a{i}, perm=pi)[name=string("z{i}")];')
            cur = f"z{i}"
        return lines, cur, shp

    if case.op in {"reduce_mean", "reduce_sum", "reduce_max"}:
        rsh = _full_shape(shp, axis)
        lines += [
            f'    tensor<int32, [1]> ax = const()[name=string("ax"), '
            f'val=tensor<int32, [1]>([{axis}])];',
            '    bool kd = const()[name=string("kd"), val=bool(true)];',
        ]
        for i in range(case.depth):
            lines.append(f'    {_tensor(dtype, rsh)} r{i} = {case.op}(x={cur}, axes=ax, '
                         f'keep_dims=kd)[name=string("r{i}")];')
            # Broadcast back so every repetition has the same full-sized input
            # and the external output keeps a safe, 32-wide last dimension.
            lines.append(f'    {t} z{i} = add(x={cur}, y=r{i})[name=string("z{i}")];')
            cur = f"z{i}"
        return lines, cur, shp

    unary = {"abs", "relu", "sigmoid", "tanh"}
    for i in range(case.depth):
        out = f"z{i}"
        if case.op in unary:
            lines.append(f'    {t} {out} = {case.op}(x={cur})[name=string("{out}")];')
        elif case.op == "square":
            lines.append(f'    {t} {out} = mul(x={cur}, y={cur})[name=string("{out}")];')
        elif case.op == "add_scalar":
            lines.append(f'    {t} {out} = add(x={cur}, y=eps)[name=string("{out}")];')
        elif case.op == "pow":
            lines.append(f'    {t} {out} = pow(x={cur}, y=ph)[name=string("{out}")];')
        else:
            raise ValueError(f"unknown op {case.op}")
        cur = out
    return lines, cur, shp


def _build_elementwise(case: Case) -> tuple[str, dict[str, bytes], int, int]:
    sig_dtype = {"fp16": "fp16", "int8_qdq": "fp16", "fp32_io": "fp32",
                 "int8_io": "int8"}[case.dtype]
    lines = [
        '    fp16 eps = const()[name=string("eps"), val=fp16(0.0009765625)];',
        '    fp16 ph = const()[name=string("ph"), val=fp16(-0.5)];',
    ]
    src = "x"
    work_dtype = sig_dtype
    if case.dtype == "int8_qdq":
        lines += [
            '    fp16 qs = const()[name=string("qs"), val=fp16(0.015625)];',
            '    string qdt = const()[name=string("qdt"), val=string("int8")];',
            f'    {_tensor("int8", case.shape)} qin = quantize(input=x, '
            'output_dtype=qdt, scale=qs)[name=string("qin")];',
            f'    {_tensor("fp16", case.shape)} din = dequantize(input=qin, scale=qs)'
            '[name=string("din")];',
        ]
        src, work_dtype = "din", "fp16"

    if case.op == "group_rms":
        if case.shape != (1, 10240, 1, 32) or work_dtype != "fp16":
            raise ValueError("group_rms is defined for fp16 [1,10240,1,32]")
        lines += [
            '    tensor<int32, [1]> gac = const()[name=string("gac"), '
            'val=tensor<int32, [1]>([1])];',
            '    tensor<int32, [1]> gah = const()[name=string("gah"), '
            'val=tensor<int32, [1]>([2])];',
            '    bool gkd = const()[name=string("gkd"), val=bool(true)];',
            '    tensor<bool, [4]> gmm = const()[name=string("gmm"), '
            'val=tensor<bool, [4]>([false,false,false,false])];',
            '    tensor<int32, [4]> gfold = const()[name=string("gfold"), '
            'val=tensor<int32, [4]>([1,4,2560,32])];',
            '    tensor<int32, [4]> gunfold = const()[name=string("gunfold"), '
            'val=tensor<int32, [4]>([1,10240,1,32])];',
        ]
        cur = src
        for d in range(case.depth):
            if case.layout == "group_unrolled":
                vals = []
                for j in range(4):
                    b, e = j * 2560, (j + 1) * 2560
                    lines += [
                        f'    tensor<fp16, [1,2560,1,32]> g{d}s{j} = slice_by_index('
                        f'x={cur}, begin=tensor<int32, [4]>([0,{b},0,0]), '
                        f'end=tensor<int32, [4]>([1,{e},1,32]), begin_mask=gmm, '
                        f'end_mask=gmm)[name=string("g{d}s{j}")];',
                        f'    tensor<fp16, [1,2560,1,32]> g{d}q{j} = mul('
                        f'x=g{d}s{j}, y=g{d}s{j})[name=string("g{d}q{j}")];',
                        f'    tensor<fp16, [1,1,1,32]> g{d}m{j} = reduce_mean('
                        f'x=g{d}q{j}, axes=gac, keep_dims=gkd)[name=string("g{d}m{j}")];',
                        f'    tensor<fp16, [1,1,1,32]> g{d}e{j} = add('
                        f'x=g{d}m{j}, y=eps)[name=string("g{d}e{j}")];',
                        f'    tensor<fp16, [1,1,1,32]> g{d}r{j} = pow('
                        f'x=g{d}e{j}, y=ph)[name=string("g{d}r{j}")];',
                        f'    tensor<fp16, [1,2560,1,32]> g{d}n{j} = mul('
                        f'x=g{d}s{j}, y=g{d}r{j})[name=string("g{d}n{j}")];',
                    ]
                    vals.append(f"g{d}n{j}")
                cur = f"g{d}out"
                lines.append(f'    tensor<fp16, [1,10240,1,32]> {cur} = concat('
                             f'values=({", ".join(vals)}), axis=int32(1), '
                             f'interleave=bool(false))[name=string("{cur}")];')
            elif case.layout == "group_folded":
                lines += [
                    f'    tensor<fp16, [1,4,2560,32]> g{d}f = reshape('
                    f'x={cur}, shape=gfold)[name=string("g{d}f")];',
                    f'    tensor<fp16, [1,4,2560,32]> g{d}q = mul('
                    f'x=g{d}f, y=g{d}f)[name=string("g{d}q")];',
                    f'    tensor<fp16, [1,4,1,32]> g{d}m = reduce_mean('
                    f'x=g{d}q, axes=gah, keep_dims=gkd)[name=string("g{d}m")];',
                    f'    tensor<fp16, [1,4,1,32]> g{d}e = add('
                    f'x=g{d}m, y=eps)[name=string("g{d}e")];',
                    f'    tensor<fp16, [1,4,1,32]> g{d}r = pow('
                    f'x=g{d}e, y=ph)[name=string("g{d}r")];',
                    f'    tensor<fp16, [1,4,2560,32]> g{d}n = mul('
                    f'x=g{d}f, y=g{d}r)[name=string("g{d}n")];',
                    f'    tensor<fp16, [1,10240,1,32]> g{d}out = reshape('
                    f'x=g{d}n, shape=gunfold)[name=string("g{d}out")];',
                ]
                cur = f"g{d}out"
            else:
                raise ValueError(f"unknown group_rms layout {case.layout}")
        body, out, out_shape = [], cur, case.shape
    else:
        body, out, out_shape = _op_chain(case, src, work_dtype)
    lines += body
    if case.dtype == "int8_qdq":
        lines += [
            f'    {_tensor("int8", out_shape)} qout = quantize(input={out}, '
            'output_dtype=qdt, scale=qs)[name=string("qout")];',
            f'    {_tensor("fp16", out_shape)} y = dequantize(input=qout, scale=qs)'
            '[name=string("y")];',
        ]
        out = "y"

    mil = (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
           f"  func main<ios18>({_tensor(sig_dtype, case.shape)} x) {{\n"
           + "\n".join(lines) + f"\n  }} -> ({out});\n}}\n"
           + f"// ane_mil_cost {case.name}\n")
    return mil, {}, int(np.prod(case.shape)), int(np.prod(out_shape))


def _quant_weight(fmt: str, weight: np.ndarray,
                  suffix: str = "") -> tuple[str, dict[str, bytes]]:
    o, i = weight.shape
    wn, wqn, wsn = f"W{suffix}", f"Wq{suffix}", f"Ws{suffix}"
    if fmt == "fp16":
        p = E._BlobPacker()
        off = p.append(weight.astype(np.float16).reshape(o, i, 1, 1).tobytes()) + 64
        fn = f"w{suffix}.bin"
        decl = (f'    tensor<fp16, [{o}, {i}, 1, 1]> {wn} = const()'
                f'[name=string("{wn}"), val=tensor<fp16, [{o}, {i}, 1, 1]>'
                f'(BLOBFILE(path=string("@model_path/weights/{fn}"), offset=uint64({off})))];')
        return decl, {fn: p.getvalue()}
    p, s = E._BlobPacker(), E._BlobPacker()
    if fmt == "int8":
        q, sc = E.quantize_linear_int8(weight)
        data = q.reshape(o, i, 1, 1).tobytes()
        typ, ish = "int8", i
    elif fmt == "int4":
        q, sc = E.quantize_linear_int4(weight)
        data = q.tobytes()
        typ, ish = "int4", i
    else:
        raise ValueError(fmt)
    od = p.append(data) + 64
    os_ = s.append(np.asarray(sc, np.float16).reshape(o, 1, 1, 1).tobytes()) + 64
    fd, fs = f"wd{suffix}.bin", f"ws{suffix}.bin"
    decl = (f'    tensor<{typ}, [{o}, {ish}, 1, 1]> {wqn} = const()'
            f'[name=string("{wqn}"), val=tensor<{typ}, [{o}, {ish}, 1, 1]>'
            f'(BLOBFILE(path=string("@model_path/weights/{fd}"), offset=uint64({od})))];\n'
            f'    tensor<fp16, [{o}, 1, 1, 1]> {wsn} = const()'
            f'[name=string("{wsn}"), val=tensor<fp16, [{o}, 1, 1, 1]>'
            f'(BLOBFILE(path=string("@model_path/weights/{fs}"), offset=uint64({os_})))];\n'
            f'    tensor<fp16, [{o}, {i}, 1, 1]> {wn} = '
            f'constexpr_blockwise_shift_scale(data={wqn}, scale={wsn})'
            f'[name=string("{wn}")];')
    return decl, {fd: p.getvalue(), fs: s.getvalue()}


def _build_conv(case: Case) -> tuple[str, dict[str, bytes], int, int]:
    if len(case.shape) != 4 or case.shape[0] != 1 or case.shape[2] != 1:
        raise ValueError("conv cases require [1,C,1,S]")
    c, s = case.shape[1], case.shape[3]
    out_c = case.out_channels or c
    if case.depth > 1 and out_c != c:
        raise ValueError("non-square conv cases require depth=1")
    # Deterministic dense weights avoid a giant identity matrix while keeping
    # every output live.  Values are small enough for deep chains.
    rng = np.random.default_rng(c)
    w = np.ascontiguousarray(
        (rng.standard_normal((out_c, c), dtype=np.float32) / np.float32(np.sqrt(c)))
        .astype(np.float32))
    if out_c % case.parts:
        raise ValueError("out_channels must divide parts")
    if case.parts > 1 and case.depth != 1:
        raise ValueError("split conv requires depth=1")
    decls, files = [], {}
    for j, part in enumerate(np.split(w, case.parts, axis=0)):
        suffix = str(j) if case.parts > 1 else ""
        decl, part_files = _quant_weight(case.dtype, np.ascontiguousarray(part), suffix)
        decls.append(decl)
        files.update(part_files)
    lines = [
        '    string pt = const()[name=string("pt"), val=string("valid")];',
        '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];',
        '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];',
        '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];',
        '    int32 gr = const()[name=string("gr"), val=int32(1)];',
        *decls,
    ]
    if case.parts > 1:
        po = out_c // case.parts
        for j in range(case.parts):
            lines.append(f'    tensor<fp16, [1, {po}, 1, {s}]> p{j} = conv('
                         f'dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, '
                         f'weight=W{j}, x=x)[name=string("p{j}")];')
        vals = ", ".join(f"p{j}" for j in range(case.parts))
        lines.append(f'    tensor<fp16, [1, {out_c}, 1, {s}]> z0 = concat('
                     f'values=({vals}), axis=int32(1), interleave=bool(false))'
                     '[name=string("z0")];')
        cur = "z0"
    else:
        cur = "x"
        for i in range(case.depth):
            lines.append(f'    tensor<fp16, [1, {out_c}, 1, {s}]> z{i} = conv('
                         f'dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, '
                         f'weight=W, x={cur})[name=string("z{i}")];')
            cur = f"z{i}"
    mil = (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
           f"  func main<ios18>(tensor<fp16, [1, {c}, 1, {s}]> x) {{\n"
           + "\n".join(lines) + f"\n  }} -> ({cur});\n}}\n"
           + f"// ane_mil_cost {case.name}\n")
    return mil, files, c * s, out_c * s


def _layout_cases(ops: Iterable[str], depth: int) -> list[Case]:
    # Constant 327,680 elements.  Only the split between channel C and height
    # H changes.  Width remains the production decode tile width of 32.
    ch = ((10240, 1), (5120, 2), (2560, 4), (1280, 8), (640, 16),
          (320, 32), (160, 64), (40, 256), (4, 2560))
    return [Case(f"layout_{op}_c{c}_h{h}", op, (1, c, h, 32), axis=1,
                 depth=depth, layout=f"C={c},H={h}")
            for op in ops for c, h in ch]


def _rank_cases(ops: Iterable[str], depth: int) -> list[Case]:
    shapes = ((2560, 32), (1, 2560, 32), (1, 2560, 1, 32),
              (1, 1, 2560, 1, 32))
    return [Case(f"rank{len(sh)}_{op}", op, sh, axis=sh.index(2560), depth=depth,
                 layout=f"rank{len(sh)}") for op in ops for sh in shapes]


def _channel_cases(ops: Iterable[str], depth: int) -> list[Case]:
    return [Case(f"channels_{op}_c{c}", op, (1, c, 1, 32), axis=1,
                 depth=depth, layout=f"C={c},H=1")
            for op in ops for c in (1, 4, 8, 16, 32, 48, 64, 128, 320,
                                    640, 1280, 2560, 5120, 10240)]


def _axis_cases(depth: int) -> list[Case]:
    shapes = ((1, 2560, 4, 32), (1, 4, 2560, 32), (1, 32, 4, 2560))
    return [Case(f"axis{ax}_{i}_reduce_mean", "reduce_mean", sh, axis=ax,
                 depth=depth, layout=f"shape={sh},axis={ax}")
            for i, sh in enumerate(shapes) for ax in range(1, 4)]


def _dtype_cases(ops: Iterable[str], depth: int) -> list[Case]:
    return [Case(f"dtype_{dt}_{op}", op, (1, 2560, 1, 32), dtype=dt,
                 axis=1, depth=depth, layout=dt)
            for op in ops for dt in ("fp16", "int8_qdq", "fp32_io", "int8_io")]


def _conv_cases(depth: int) -> list[Case]:
    square = [Case(f"conv_c{c}_{dt}", "conv", (1, c, 1, 32), dtype=dt,
                   depth=depth, layout=f"I=O={c}")
              for c in (256, 1024, 2560) for dt in ("fp16", "int8", "int4")]
    projection = [Case(f"conv_i{i}_o{o}_{dt}", "conv", (1, i, 1, 32),
                       dtype=dt, depth=1, layout=f"I={i},O={o}", out_channels=o)
                  for i, o in ((2560, 16480), (6144, 2560))
                  for dt in ("fp16", "int8", "int4")]
    split = [Case(f"conv_i{i}_o{o}_{dt}_p{parts}", "conv", (1, i, 1, 32),
                  dtype=dt, depth=1, layout=f"I={i},O={o},parts={parts}",
                  out_channels=o, parts=parts)
             for i, o in ((2560, 16480), (6144, 2560))
             for dt in ("int8",) for parts in (2, 4, 8)]
    return square + projection + split


def make_cases(suite: str, ops: list[str], depth: int) -> list[Case]:
    if suite == "layout":
        return _layout_cases(ops, depth)
    if suite == "rank":
        return _rank_cases(ops, depth)
    if suite == "channels":
        return _channel_cases(ops, depth)
    if suite == "axis":
        return _axis_cases(depth)
    if suite == "dtype":
        return _dtype_cases(ops, depth)
    if suite == "conv":
        return _conv_cases(max(1, min(depth, 8)))
    if suite == "core":
        grouped = [Case("group_rms_unrolled", "group_rms", (1, 10240, 1, 32),
                        depth=max(1, depth // 2), layout="group_unrolled"),
                   Case("group_rms_folded", "group_rms", (1, 10240, 1, 32),
                        depth=max(1, depth // 2), layout="group_folded")]
        return (grouped + _layout_cases(ops, depth) + _rank_cases(ops, depth) +
                _channel_cases(ops, depth) +
                _axis_cases(depth) + _dtype_cases(ops, depth))
    raise ValueError(suite)


def run_case(eng: AneEngine, case: Case, warmup: int, repeats: int) -> dict:
    build = _build_conv if case.op == "conv" else _build_elementwise
    try:
        mil, weights, in_elems, out_elems = build(case)
    except Exception as exc:  # noqa: BLE001
        return {**dataclasses.asdict(case), "status": "build_failed", "error": repr(exc)}
    capture = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        try:
            p = eng.compile_multiproc(mil, weights, in_elems, out_elems, 1,
                                      raw_weight_files=frozenset(weights))
        except Exception as exc:  # noqa: BLE001
            p = None
            capture.write(f"{exc!r}\n")
    compile_ms = (time.perf_counter() - t0) * 1e3
    if p is None:
        lines = [x for x in capture.getvalue().splitlines() if x.strip()]
        return {**dataclasses.asdict(case), "status": "compile_failed",
                "compile_ms": compile_ms, "error": " | ".join(lines[-4:])}
    p.input_elems = [in_elems]
    p.output_elems = [out_elems]
    if not eng._ensure_io(p):
        return {**dataclasses.asdict(case), "status": "io_failed", "compile_ms": compile_ms}

    sig_dt = np.int8 if case.dtype == "int8_io" else (
        np.float32 if case.dtype == "fp32_io" else np.float16)
    rng = np.random.default_rng(zlib.crc32(case.name.encode()))
    x = rng.uniform(0.125, 0.75, in_elems).astype(sig_dt)
    with _iosurface_view(p._in_surf, (in_elems,), sig_dt) as dst:
        np.copyto(dst, x)
    for _ in range(warmup):
        if not eng.submit(p):
            return {**dataclasses.asdict(case), "status": "submit_failed",
                    "compile_ms": compile_ms}
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        ok = eng.submit(p)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
        if not ok:
            return {**dataclasses.asdict(case), "status": "submit_failed",
                    "compile_ms": compile_ms}
    med = statistics.median(samples)
    return {**dataclasses.asdict(case), "status": "ok", "compile_ms": compile_ms,
            "median_ms": med, "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90)), "min_ms": min(samples),
            "per_repeat_us": med * 1000 / case.depth, "samples_ms": samples}


def _system_value(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=("core", "layout", "rank", "channels", "axis", "dtype", "conv"),
                    default="core")
    ap.add_argument("--ops", default="square,tanh,reduce_mean",
                    help="comma-separated elementwise/reduction op kinds")
    ap.add_argument("--depth", type=int, default=8, help="operations per timed program")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=41)
    ap.add_argument("--filter", default="", help="run only case names containing this text")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.depth < 1 or args.warmup < 0 or args.repeats < 1:
        ap.error("depth/repeats must be positive and warmup nonnegative")

    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    cases = make_cases(args.suite, ops, args.depth)
    if args.filter:
        cases = [c for c in cases if args.filter in c.name]
    eng = AneEngine()
    if not eng.available:
        raise SystemExit("AneEngine unavailable")
    chip = _system_value("sysctl", "-n", "machdep.cpu.brand_string")
    os_build = _system_value("sw_vers", "-buildVersion")
    print(f"hardware={chip} ({platform.machine()}) macOS={platform.mac_ver()[0]} "
          f"build={os_build} "
          f"suite={args.suite} depth={args.depth} warmup={args.warmup} "
          f"repeats={args.repeats} cache={os.environ.get('Q38_ANE_REUSE_COMPILED', '1')}")
    results = []
    for i, case in enumerate(cases, 1):
        result = run_case(eng, case, args.warmup, args.repeats)
        results.append(result)
        if result["status"] == "ok":
            print(f"[{i:03d}/{len(cases):03d}] {case.name:42s} "
                  f"{result['median_ms']:8.4f} ms  {result['per_repeat_us']:8.2f} us/repeat "
                  f"p10..p90={result['p10_ms']:.4f}..{result['p90_ms']:.4f}", flush=True)
        else:
            print(f"[{i:03d}/{len(cases):03d}] {case.name:42s} {result['status']} "
                  f"{result.get('error', '')[-180:]}", flush=True)
    if args.json:
        payload = {"metadata": {"chip": chip, "machine": platform.machine(),
                                "macos": platform.mac_ver()[0], "os_build": os_build,
                                "suite": args.suite, "depth": args.depth,
                                "warmup": args.warmup, "repeats": args.repeats},
                   "results": results}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")
    failed = sum(r["status"] != "ok" for r in results)
    print(f"done: {len(results) - failed} ok, {failed} failed")


if __name__ == "__main__":
    main()
