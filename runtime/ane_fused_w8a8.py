# SPDX-License-Identifier: Apache-2.0
"""Fused W8A8 projection programs for Flash-Next linear-attention.

The 30 TOPS figure is not a property of int8 weights on a single conv. It is
int8 activations *between* convs inside one compiled program, at S >= 256:

    fp16(in) -> conv(int8 W) -> quantize(int8) -> dequantize(fp16) -> conv ...

Recipe and numbers: docs/W8A8-PROJECTIONS.md, probes/ane_flashnext_block.py.

This module compiles the *correct* linear-attn dataflow at that occupancy:

    x -> conv(in_proj_fused) -> slice qkv/z/b/a
       -> qdq(qkv) -> depthwise causal conv k=4

Decode at S=32 cannot see the 30 TOPS multiplier (the table's width threshold).
Prefill tiles should be TILE wide.
"""
from __future__ import annotations

import contextlib
import ctypes
import io
import struct
import time
from dataclasses import dataclass

import numpy as np

from runtime.q38_ane_engine import (
    _BUILD_INFO,
    AneEngine,
    _cls,
    _create_iosurface,
    _desc,
    _indexset_to_nsarray,
    _iosurface_alloc_size,
    _iosurface_view,
    _load_iosurface,
    _msg,
    _nsarray,
    _nsarray_strings,
    _nsnumber_int,
    _nsstring,
    _objc,
    _sel,
)

BLOB = "@model_path/weights/weight.bin"
TILE = 256

_PRE = (
    '    string pt = const()[name=string("pt"), val=string("valid")];\n'
    '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '    int32 gr = const()[name=string("gr"), val=int32(1)];\n'
    '    int32 ax = const()[name=string("ax"), val=int32(1)];\n'
    '    string q_dtype = const()[name=string("q_dtype"), val=string("int8")];'
)


def f16lit(v: float) -> str:
    v = float(np.float16(v))
    s = f"{v:.10g}"
    return f"{v:.12f}" if ("e" in s or "E" in s) else s


def quant_per_channel(w: np.ndarray):
    flat = w.reshape(w.shape[0], -1)
    scale = np.abs(flat).max(axis=1, keepdims=True) / 127.0
    scale = np.maximum(scale, 6e-8).astype(np.float16)
    q = np.clip(np.rint(flat / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scale


def act_scale_from_weight(w: np.ndarray, margin: float = 8.0) -> np.ndarray:
    """Per-channel activation scale assuming RMS(x) ~ 1 after the mixer.

    RMS(Wx) ~ ||W[c]||_2. Prefill tiles see 6-8 sigma peaks; the probe's 2x
    margin was on a measured per-channel *max*, not on RMS, so 2x here
    under-clips (absmax ~9 vs int8 range ~2.8). 8x RMS / 127 is the same
    1.5-2x-over-max rule with a Gaussian-peak stand-in for the missing
    calibration draw.
    """
    row = np.linalg.norm(np.asarray(w, np.float32).reshape(w.shape[0], -1), axis=1)
    return np.maximum(margin * row / 127.0, 6e-8).astype(np.float32)


class Bank:
    def __init__(self):
        self._payloads: list[bytes] = []

    def add(self, payload: bytes) -> int:
        self._payloads.append(payload)
        return len(self._payloads) - 1

    def pack(self) -> tuple[bytes, list[int]]:
        n = len(self._payloads)
        head = bytearray(64)
        struct.pack_into("<II", head, 0, 1, 2)
        table = bytearray(64 * n)
        arena = bytearray()
        arena_base = 64 + 64 * n
        offsets = []
        for k, payload in enumerate(self._payloads):
            pos = arena_base + len(arena)
            struct.pack_into("<I", table, 64 * k + 0, 0xDEADBEEF)
            struct.pack_into("<I", table, 64 * k + 4, 1)
            struct.pack_into("<I", table, 64 * k + 8, len(payload))
            struct.pack_into("<I", table, 64 * k + 16, pos)
            offsets.append(64 + 64 * k)
            arena += payload
            arena += b"\0" * (-len(payload) % 64)
        return bytes(head) + bytes(table) + bytes(arena), offsets


class Prog:
    def __init__(self, in_dim: int, seq: int):
        self.in_dim = in_dim
        self.seq = seq
        self.bank = Bank()
        self.lines: list = []
        self.pending: list = []
        self.outputs: list[tuple[str, int]] = []

    def weight_fp16(self, name: str, w: np.ndarray) -> str:
        shape = list(w.shape) + [1] * (4 - w.ndim)
        idx = self.bank.add(np.ascontiguousarray(w.astype(np.float16)).tobytes())
        sh = ", ".join(str(d) for d in shape)
        self.pending.append(
            (f'    tensor<fp16, [{sh}]> {name} = const()[name=string("{name}"), '
             f'val=tensor<fp16, [{sh}]>(BLOBFILE(path=string("{BLOB}"), '
             f'offset=uint64({{{idx}}})))];', ))
        self.lines.append(("PEND", len(self.pending) - 1))
        return name

    def weight_int8(self, name: str, w: np.ndarray) -> str:
        q, sc = quant_per_channel(w)
        shape = list(w.shape) + [1] * (4 - w.ndim)
        sh = ", ".join(str(d) for d in shape)
        O = w.shape[0]
        di = self.bank.add(np.ascontiguousarray(q).tobytes())
        si = self.bank.add(np.ascontiguousarray(sc.astype(np.float16)).tobytes())
        self.pending.append(
            (f'    tensor<int8, [{sh}]> {name}q = const()[name=string("{name}q"), '
             f'val=tensor<int8, [{sh}]>(BLOBFILE(path=string("{BLOB}"), '
             f'offset=uint64({{{di}}})))];\n'
             f'    tensor<fp16, [{O}, 1, 1, 1]> {name}s = const()'
             f'[name=string("{name}s"), val=tensor<fp16, [{O}, 1, 1, 1]>'
             f'(BLOBFILE(path=string("{BLOB}"), offset=uint64({{{si}}})))];\n'
             f'    tensor<fp16, [{sh}]> {name} = constexpr_blockwise_shift_scale('
             f'data={name}q, scale={name}s)[name=string("{name}dq")];', ))
        self.lines.append(("PEND", len(self.pending) - 1))
        return name

    def conv(self, name: str, x: str, w: str, out_ch: int) -> str:
        self.lines.append(
            f'    tensor<fp16, [1, {out_ch}, 1, {self.seq}]> {name} = conv('
            f'dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, '
            f'weight={w}, x={x})[name=string("{name}")];')
        return name

    def sigmoid(self, name: str, x: str, ch: int) -> str:
        self.lines.append(
            f'    tensor<fp16, [1, {ch}, 1, {self.seq}]> {name} = '
            f'sigmoid(x={x})[name=string("{name}")];')
        return name

    def mul(self, name: str, a: str, b: str, ch: int) -> str:
        self.lines.append(
            f'    tensor<fp16, [1, {ch}, 1, {self.seq}]> {name} = '
            f'mul(x={a}, y={b})[name=string("{name}")];')
        return name

    def add(self, name: str, a: str, b: str, ch: int) -> str:
        self.lines.append(
            f'    tensor<fp16, [1, {ch}, 1, {self.seq}]> {name} = '
            f'add(x={a}, y={b})[name=string("{name}")];')
        return name

    def slice_ch(self, name: str, x: str, c0: int, c1: int) -> str:
        self.lines.append(
            f'    tensor<fp16, [1, {c1 - c0}, 1, {self.seq}]> {name} = '
            f'slice_by_index(begin=tensor<int32, [4]>([0,{c0},0,0]), '
            f'end=tensor<int32, [4]>([1,{c1},1,{self.seq}]), x={x})'
            f'[name=string("{name}")];')
        return name

    def dwconv_causal(self, name: str, x: str, w: str, ch: int, k: int) -> str:
        self.lines.append(
            f'    int32 dwg{name} = const()[name=string("dwg{name}"), '
            f'val=int32({ch})];\n'
            f'    string dwpt{name} = const()[name=string("dwpt{name}"), '
            f'val=string("custom")];\n'
            f'    tensor<int32, [4]> dwpd{name} = const()'
            f'[name=string("dwpd{name}"), val=tensor<int32, [4]>'
            f'([0,0,{k - 1},0])];\n'
            f'    tensor<fp16, [1, {ch}, 1, {self.seq}]> {name} = conv('
            f'dilations=dl, groups=dwg{name}, pad=dwpd{name}, '
            f'pad_type=dwpt{name}, strides=st, weight={w}, x={x})'
            f'[name=string("{name}")];')
        return name

    def qdq(self, name: str, x: str, ch: int, scale: np.ndarray) -> str:
        lit = ",".join(f16lit(v) for v in np.asarray(scale).reshape(-1))
        self.lines.append(
            f'    tensor<fp16, [{ch}]> {name}s = const()[name=string("{name}s"), '
            f'val=tensor<fp16, [{ch}]>([{lit}])];\n'
            f'    tensor<int8, [1, {ch}, 1, {self.seq}]> {name}q = quantize('
            f'axis=ax, input={x}, output_dtype=q_dtype, scale={name}s)'
            f'[name=string("{name}q")];\n'
            f'    tensor<fp16, [1, {ch}, 1, {self.seq}]> {name} = dequantize('
            f'axis=ax, input={name}q, scale={name}s)[name=string("{name}")];')
        return name

    def out(self, name: str, ch: int) -> None:
        self.outputs.append((name, ch))

    def mil(self) -> tuple[str, bytes]:
        packed, offsets = self.bank.pack()
        body = []
        for entry in self.lines:
            if isinstance(entry, tuple) and entry[0] == "PEND":
                body.append(self.pending[entry[1]][0].format(*offsets))
            else:
                body.append(entry)
        outs = ", ".join(n for n, _ in self.outputs)
        mil = (f"program(1.3)\n{_BUILD_INFO}\n{{\n"
               f"  func main<ios18>(tensor<fp16, [1, {self.in_dim}, 1, "
               f"{self.seq}]> x) {{\n{_PRE}\n" + "\n".join(body)
               + f"\n  }} -> ({outs});\n}}\n")
        return mil, packed


def _ane_output_symbols(model) -> list[str]:
    ForKey = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p)
    attrs = _msg(model, "modelAttributes")
    desc = ForKey(("objc_msgSend", _objc))(
        attrs, _sel("objectForKey:"), _nsstring("ANEFModelDescription"))
    if not desc:
        return []
    arr = ForKey(("objc_msgSend", _objc))(
        desc, _sel("objectForKey:"),
        _nsstring("kANEFModelOutputSymbolsArrayKey"))
    return [s.split("@")[0] for s in _nsarray_strings(arr)] if arr else []


class MultiOut:
    """One input, N outputs, bound in compiled symbol order (not MIL order)."""

    def __init__(self, prog, in_dim: int, seq: int,
                 outs: list[tuple[str, int]]):
        self.prog, self.in_dim, self.seq = prog, in_dim, seq
        self.declared = outs
        by_mil = {n: c for n, c in outs}
        _load_iosurface()
        bound = _ane_output_symbols(prog.model)
        if not bound or sorted(bound) != sorted(by_mil):
            bound = list(by_mil)
        self.bound = bound
        self.ch_of = by_mil
        self.in_surf = _create_iosurface(_iosurface_alloc_size(in_dim * seq))
        self.out_surfs = [_create_iosurface(_iosurface_alloc_size(by_mil[n] * seq))
                          for n in bound]
        surf_cls = _cls("_ANEIOSurfaceObject")
        IS = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)
        sel = _sel("initWithIOSurface:startOffset:shouldRetain:")

        def wrap(s):
            return IS(("objc_msgSend", _objc))(
                _msg(surf_cls, "alloc"), sel, s, _nsnumber_int(0), True)

        F = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
        self.req = F(("objc_msgSend", _objc))(
            _msg(_cls("_ANERequest"), "alloc"),
            _sel("initWithInputs:inputIndices:outputs:outputIndices:"
                 "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                 "transactionHandle:"),
            _nsarray([wrap(self.in_surf)]), _nsarray([_nsnumber_int(0)]),
            _nsarray([wrap(s) for s in self.out_surfs]),
            _nsarray([_nsnumber_int(i) for i in range(len(self.bound))]),
            None, None, _nsnumber_int(0), None, None)
        self.last_ms = 0.0

    def submit(self) -> bool:
        Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        err = ctypes.c_void_p(0)
        ok = Eval(("objc_msgSend", _objc))(
            self.prog.model, _sel("evaluateWithQoS:options:request:error:"),
            21, self.prog._compile_opts, self.req, ctypes.byref(err))
        if not ok:
            self.err = (_desc(err.value) if err.value else "unknown")[:200]
        return bool(ok)

    def run(self, x: np.ndarray) -> dict[str, np.ndarray] | None:
        """x planar [in_dim, seq] -> {mil_name: [ch, seq]} fp32."""
        with _iosurface_view(self.in_surf, x.shape, np.float16) as dst:
            np.copyto(dst, x.astype(np.float16))
        t0 = time.perf_counter()
        if not self.submit():
            return None
        self.last_ms = (time.perf_counter() - t0) * 1e3
        got = {}
        for surf, name in zip(self.out_surfs, self.bound):
            with _iosurface_view(surf, (self.ch_of[name], self.seq),
                                 np.float16) as o:
                got[name] = np.array(o, np.float32)
        return got


@dataclass
class FusedLinearAttn:
    runner: MultiOut
    in_dim: int
    seq: int
    qkv_dim: int
    z_dim: int
    flops: int

    def run_seq(self, mixed: np.ndarray) -> dict[str, np.ndarray]:
        """mixed [S, H] -> dict of [ch, S] (unpadded)."""
        mixed = mixed.reshape(1, -1) if mixed.ndim == 1 else np.asarray(mixed, np.float32)
        S = mixed.shape[0]
        buf = np.zeros((self.in_dim, self.seq), np.float32)
        buf[:, :S] = mixed.T
        got = self.runner.run(buf)
        if got is None:
            raise RuntimeError("fused linear-attn W8A8 submit failed")
        return {k: v[:, :S] for k, v in got.items()}

    def tops(self) -> float:
        if self.runner.last_ms <= 0:
            return 0.0
        return self.flops / (self.runner.last_ms * 1e-3) / 1e12


def in_proj_dw_flops(out_dim: int, in_dim: int, qkv_dim: int, seq: int,
                     k: int = 4) -> int:
    return 2 * out_dim * in_dim * seq + 2 * qkv_dim * k * seq


def compile_in_proj_dw(
    engine: AneEngine,
    fused: np.ndarray,
    conv1d: np.ndarray,
    *,
    seq: int = TILE,
    qkv_dim: int | None = None,
    z_dim: int = 6144,
    arm: str = "fp16",
) -> FusedLinearAttn | None:
    """Bake in_proj_fused + causal dwconv as one W8A8 program.

    `fused` is [qkv+z+b+a, H] in that row order (same as generate's concat).
    `conv1d` is [qkv_dim, 4] or [qkv_dim, 1, 4].
    """
    fused = np.ascontiguousarray(fused, dtype=np.float32)
    conv = np.ascontiguousarray(conv1d, dtype=np.float32)
    if conv.ndim == 3:
        conv = conv.reshape(conv.shape[0], -1)
    Cq = int(qkv_dim or conv.shape[0])
    H = int(fused.shape[1])
    O = int(fused.shape[0])
    Hb = 48
    Ha = 48
    if O != Cq + z_dim + Hb + Ha:
        # still compile; slices follow the concat layout we were given
        pass
    c0, c1, c2, c3 = 0, Cq, Cq + z_dim, Cq + z_dim + Hb

    p = Prog(H, seq)
    wconst = p.weight_int8 if arm != "fp16" else p.weight_fp16
    wconst("Win", fused)
    p.conv("yin", "x", "Win", O)
    p.slice_ch("y_qkv", "yin", c0, c1)
    p.slice_ch("y_z", "yin", c1, c2)
    p.slice_ch("y_b", "yin", c2, c3)
    p.slice_ch("y_a", "yin", c3, O)
    src = "y_qkv"
    if arm == "w8a8":
        src = p.qdq("dq", "y_qkv", Cq, act_scale_from_weight(fused[:Cq]))
    kern = conv.reshape(Cq, 1, 1, -1)
    wconst("Wc", kern)
    p.dwconv_causal("y_dw", src, "Wc", Cq, kern.shape[-1])
    p.out("y_qkv", Cq)
    p.out("y_z", z_dim)
    p.out("y_b", Hb)
    p.out("y_a", Ha)
    p.out("y_dw", Cq)

    mil, packed = p.mil()
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        try:
            prog = engine.compile_multiproc(
                mil, {"weight.bin": packed}, H,
                max(c for _, c in p.outputs), seq,
                raw_weight_files=frozenset({"weight.bin"}))
        except Exception as exc:  # noqa: BLE001
            cap.write(f"exception: {exc!r}\n")
            prog = None
    if prog is None:
        tail = " | ".join(cap.getvalue().strip().splitlines()[-4:])
        print(f"  fused w8a8 compile failed: {tail}", flush=True)
        return None
    runner = MultiOut(prog, H, seq, p.outputs)
    return FusedLinearAttn(
        runner=runner,
        in_dim=H,
        seq=seq,
        qkv_dim=Cq,
        z_dim=z_dim,
        flops=in_proj_dw_flops(O, H, Cq, seq, kern.shape[-1]),
    )
