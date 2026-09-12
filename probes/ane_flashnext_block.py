#!/usr/bin/env python3
"""How large a FUSED, correct, W8A8 Neural Engine program can one
Qwen3.8-Flash-Next linear-attention layer's projections be compiled into?

Prior probes each proved one piece and none of them proved the composition:

  * ``ane_w8a8_projection.py``   speed of an int8 chain, random weights, no
                                 accuracy claim.
  * ``ane_w8a8_accuracy.py``     accuracy of int8 weights on the real layer-0
                                 in_proj_z -> out_proj pair, but with a SCALAR
                                 activation scale and (because of the
                                 ``_make_blob`` aliasing bug) effectively one
                                 usable const per program.
  * ``ane_blob_header_recover.py`` the container fix that lets many tensors
                                 live in one ``weight.bin``.
  * ``ane_act_quant_error.py``   the only per-channel activation scale spelling
                                 ANECCompile accepts (rank-1 + explicit axis).
  * ``ane_two_outputs.py``       an ``_ANERequest`` with two output surfaces.

This probe composes all five into one program and grows it a stage at a time,
scoring every stage against an fp32 numpy reference on the same inputs:

  S1  in_proj_z -> [per-channel q/dq] -> out_proj          2 int8 convs
  S2  + in_proj_qkv as a second program output              3 int8 convs
  S3  + in_proj_a and in_proj_b as outputs 3 and 4          5 int8 convs
  S4  + the depthwise causal conv1d (k=4, groups=10240) on the qkv output

Every stage runs three arms so the int8 cost is separable from the harness:

  fp16      fp16 weights, fp16 activations           expect ~4e-3
  w8a16     int8 per-output-channel weights, fp16 activations
  w8a8      + int8 per-channel activations at every conv-to-conv boundary

If the fp16 arm does not land near 4e-3 the harness is wrong and every other
number in the table is meaningless; ``main`` says so explicitly rather than
reporting them quietly.

Activation scales are calibrated on a SEPARATE draw from the one scored. An
oracle scale taken from the tensor being measured understates clipping, and
that is exactly the error W8A8 is being judged on.

Run with ``Q38_ANE_REUSE_COMPILED=0``: the compiler cache is content-addressed
and will serve an artifact built from an earlier, wrong version of this file.

Env: ``BLOCK_S`` (default 256; W8A8 only pays at S >= 256), ``BLOCK_STAGES``
(comma list, default ``1,2,3,4``), ``BLOCK_ARMS``, ``BLOCK_ACT``
(``iid`` / ``outlier`` / ``both``), ``BLOCK_ERRLINES``.
"""
from __future__ import annotations

import contextlib
import ctypes
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (
    AneEngine, _cls, _create_iosurface, _desc, _indexset_to_nsarray,
    _iosurface_alloc_size, _iosurface_view, _load_iosurface, _msg, _nsarray,
    _nsarray_strings, _nsnumber_int, _nsstring, _objc, _sel,
)

MODEL = Path(os.environ.get("FLASH_NEXT", "/Users/true/models/Qwen3.8-Flash-Next"))
SHARD = MODEL / "model-00001-of-00131.safetensors"
PREFIX = "model.language_model.layers.0.linear_attn."

BLOB = "@model_path/weights/weight.bin"
_PRE = (
    '    string pt = const()[name=string("pt"), val=string("valid")];\n'
    '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '    int32 gr = const()[name=string("gr"), val=int32(1)];\n'
    '    int32 ax = const()[name=string("ax"), val=int32(1)];\n'
    '    string q_dtype = const()[name=string("q_dtype"), val=string("int8")];'
)

eng = AneEngine()


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def read_safetensors(path: Path, names: list[str]) -> dict:
    """Verbatim from ane_w8a8_accuracy.py: BF16 -> fp32 by a 16-bit shift."""
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
    """int8 + one fp16 scale per OUTPUT row.

    Per-output-channel is the floor this compiler accepts: a scalar scale is
    rejected by constexpr_blockwise_shift_scale, and a finer (group-64) scale
    is rejected too. `w` is flattened to [O, -1] so a depthwise [C, 1, K]
    kernel quantizes per channel with the same helper.
    """
    flat = w.reshape(w.shape[0], -1)
    scale = np.abs(flat).max(axis=1, keepdims=True) / 127.0
    # Floor BEFORE the fp16 cast: a dead or tiny channel otherwise underflows
    # fp16 to 0.0 and the divide below yields NaN (6e-8 = smallest fp16
    # subnormal that still inverts finitely; same floor as act_scale).
    scale = np.maximum(scale, 6e-8).astype(np.float16)
    q = np.clip(np.rint(flat / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scale


def dequant(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The matmul the ANE actually holds. References must use THIS, not w,
    when the question is 'did the hardware compute what I asked for'."""
    return q.reshape(q.shape[0], -1).astype(np.float32) * scale.astype(np.float32)


# ---------------------------------------------------------------------------
# container: one blob FILE, many tensors (ane_blob_header_recover.py WINNER)
# ---------------------------------------------------------------------------

class Bank:
    """Packs every const of a program into ONE weight.bin.

    More than one blob file in a program is rejected at
    ``verifyBundleAtPath: invalid model`` at any size, so this is not an
    optimisation. Three fields are load bearing: uint32 1 / uint32 2 at file
    bytes 0 and 4, 0xDEADBEEF at each chunk header (which is what the MIL
    offset points at), and the payload's FILE-ABSOLUTE offset at chunk byte 16
    -- ``_make_blob`` hardcodes 0x80 there, which aliases every tensor past the
    first onto tensor 0. The payload size at chunk byte 8 is required by
    constexpr_blockwise_shift_scale specifically.
    """

    def __init__(self):
        self._payloads: list[bytes] = []

    def add(self, payload: bytes) -> int:
        """Returns the tensor's index; call ``offsets()`` for MIL offsets."""
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


# ---------------------------------------------------------------------------
# MIL fragments
# ---------------------------------------------------------------------------

def f16lit(v: float) -> str:
    """MIL fp16 literal; exponent notation is not worth risking in the parser."""
    v = float(np.float16(v))
    s = f"{v:.10g}"
    return f"{v:.12f}" if ("e" in s or "E" in s) else s


class Prog:
    """Accumulates MIL declarations, a weight bank, and an output list."""

    def __init__(self, in_dim: int, seq: int):
        self.in_dim = in_dim
        self.seq = seq
        self.bank = Bank()
        self.lines: list[str] = []
        self.pending: list = []          # (kind, name, ...) resolved on emit
        self.outputs: list[tuple[str, int]] = []

    # -- consts ------------------------------------------------------------
    def weight_fp16(self, name: str, w: np.ndarray) -> str:
        shape = list(w.shape) + [1] * (4 - w.ndim)
        idx = self.bank.add(np.ascontiguousarray(
            w.astype(np.float16)).tobytes())
        sh = ", ".join(str(d) for d in shape)
        self.pending.append(
            (f'    tensor<fp16, [{sh}]> {name} = const()[name=string("{name}"), '
             f'val=tensor<fp16, [{sh}]>(BLOBFILE(path=string("{BLOB}"), '
             f'offset=uint64({{{idx}}})))];', ))
        self.lines.append(("PEND", len(self.pending) - 1))
        return name

    def weight_int8(self, name: str, w: np.ndarray) -> str:
        """int8 data + per-output-channel fp16 scale [O,1,1,1].

        constexpr_affine_dequantize accepts only a SCALAR scale and is the
        wrong node here; constexpr_blockwise_shift_scale is the one that takes
        a per-channel scale tensor.
        """
        q, sc = quant_per_channel(w)
        shape = list(w.shape) + [1] * (4 - w.ndim)
        sh = ", ".join(str(d) for d in shape)
        O = w.shape[0]
        di = self.bank.add(np.ascontiguousarray(q).tobytes())
        si = self.bank.add(np.ascontiguousarray(
            sc.astype(np.float16)).tobytes())
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

    # -- ops ---------------------------------------------------------------
    def conv(self, name: str, x: str, w: str, out_ch: int) -> str:
        self.lines.append(
            f'    tensor<fp16, [1, {out_ch}, 1, {self.seq}]> {name} = conv('
            f'dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, '
            f'weight={w}, x={x})[name=string("{name}")];')
        return name

    def dwconv_causal(self, name: str, x: str, w: str, ch: int, k: int) -> str:
        """Depthwise causal conv along the S axis of [1, C, 1, S].

        groups == C, weight [C, 1, 1, K], and the causal left pad K-1 goes in
        the W-before slot of MIL's [H_before, H_after, W_before, W_after].
        """
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
        """Per-channel int8 activation round trip.

        Only rank-1 scale + explicit axis compiles; a rank-4 [1,C,1,1] scale
        with no axis is InvalidMILProgram. The scale is an INLINE literal, not
        a blob const -- the single permitted blob file is already full of
        weights, and inline rank-1 literals may be flat (rank-4 literals would
        have to be nested to element rank).
        """
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

    # -- emit --------------------------------------------------------------
    def mil(self) -> tuple[str, bytes]:
        packed, offsets = self.bank.pack()
        body = []
        for entry in self.lines:
            if isinstance(entry, tuple) and entry[0] == "PEND":
                tmpl = self.pending[entry[1]][0]
                body.append(tmpl.format(*offsets))
            else:
                body.append(entry)
        outs = ", ".join(n for n, _ in self.outputs)
        mil = (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
               f"  func main<ios18>(tensor<fp16, [1, {self.in_dim}, 1, "
               f"{self.seq}]> x) {{\n{_PRE}\n" + "\n".join(body)
               + f"\n  }} -> ({outs});\n}}\n")
        return mil, packed


# ---------------------------------------------------------------------------
# compile + multi-output submit
# ---------------------------------------------------------------------------

def compile_quiet(prog: Prog):
    mil, packed = prog.mil()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(
                mil, {"weight.bin": packed}, prog.in_dim,
                max(c for _, c in prog.outputs), prog.seq,
                raw_weight_files=frozenset({"weight.bin"}))
        except Exception as exc:                     # noqa: BLE001
            buf.write(f"exception: {exc!r}\n")
            p = None
    keep = 2 if p is not None else int(os.environ.get("BLOCK_ERRLINES", "3"))
    tail = " | ".join(buf.getvalue().strip().splitlines()[-keep:])
    return p, tail, len(packed), mil


def _ane_output_symbols(model) -> list[str]:
    """Compiled output names, in the order ANE binds output indices.

    Output twin of the engine's ``_ane_input_symbols``. Entries arrive as
    ``y_a@output``; the suffix is stripped. The order is alphabetical, NOT
    the MIL return order.
    """
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


class Runner:
    """One input surface, N output surfaces, one reusable _ANERequest.

    ``AneEngine._ensure_io``/``_ensure_request`` wire exactly one output
    surface, so a multi-output program has to build its own request. This is
    the shape proven in ane_two_outputs.py, generalised to N.

    Surfaces are bound in the model's compiled output-symbol order, not the
    MIL return order: a 4-output program binds (y_a, y_b, y_out, y_qkv), and
    a surface array in MIL order puts y_out's [2560,S] into y_a's [48,S]
    slot, which eval reports as Code=42 "IOSurface smaller than the model
    expects". Readback is mapped back to MIL order by name.
    """

    _MIL_NAME = {"out_proj": "y_out", "qkv": "y_qkv", "a": "y_a",
                 "b": "y_b", "conv1d": "y_dw"}

    def __init__(self, prog, in_dim: int, seq: int,
                 outs: list[tuple[str, int]]):
        self.prog, self.in_dim, self.seq = prog, in_dim, seq
        self.declared = outs
        by_mil = {self._MIL_NAME.get(n, n): c for n, c in outs}
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

    def output_symbols(self) -> str:
        inner = _msg(self.prog.model, "model") or self.prog.model
        Sym = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_ulonglong)
        idx = Sym(("objc_msgSend", _objc))(
            inner, _sel("outputSymbolIndicesForProcedureIndex:"), 0)
        return _desc(_indexset_to_nsarray(idx)).replace("\n", " ")

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

    def run(self, x: np.ndarray):
        """x planar [in_dim, seq] fp32 -> list of [ch, seq] fp32, MIL order."""
        with _iosurface_view(self.in_surf, x.shape, np.float16) as dst:
            np.copyto(dst, x.astype(np.float16))
        if not self.submit():
            return None
        got = {}
        for surf, name in zip(self.out_surfs, self.bound):
            with _iosurface_view(surf, (self.ch_of[name], self.seq),
                                 np.float16) as o:
                got[name] = np.array(o, np.float32)
        return [got[self._MIL_NAME.get(n, n)] for n, _ in self.declared]

    def timed(self, n: int = 7) -> float:
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.submit()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]


def rel_l2(got, ref) -> float:
    if got is None:
        return float("nan")
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------

def act_scale(a: np.ndarray) -> np.ndarray:
    """Per-channel int8 activation scale, floored off zero.

    6e-8 is the smallest fp16 subnormal that still inverts finitely; a dead
    channel with scale 0 would produce NaN in the quantize.
    """
    return np.maximum(np.abs(a).max(axis=1) / 127.0, 6e-8).astype(np.float32)


def sim_qdq(a: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """numpy model of the per-channel quantize/dequantize pair."""
    s = np.float32(np.float16(sc))[:, None]
    q = np.clip(np.rint(a / s), -128, 127)
    return (q * s).astype(np.float32)


def causal_dw(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """y[c,t] = sum_j k[c,j] * x[c, t-(K-1)+j], zero-padded on the left."""
    C, S = x.shape
    K = k.shape[-1]
    pad = np.zeros((C, S + K - 1), np.float32)
    pad[:, K - 1:] = x
    y = np.zeros((C, S), np.float32)
    for j in range(K):
        y += k[:, j:j + 1] * pad[:, j:j + S]
    return y


# ---------------------------------------------------------------------------
# stage construction
# ---------------------------------------------------------------------------

def build_stage(stage: int, arm: str, W: dict, S: int, scales: dict):
    """One program for `stage` in `arm`. Returns (Prog, ref_fn).

    `ref_fn(x)` produces the fp32 numpy reference list matching the program's
    output order, computed with the weights the ANE actually holds.
    """
    H = W["z"].shape[1]
    p = Prog(H, S)
    intn = (lambda n, w: p.weight_int8(n, w)) if arm != "fp16" else \
           (lambda n, w: p.weight_fp16(n, w))

    # what the hardware holds, per arm
    def held(w):
        if arm == "fp16":
            return w.astype(np.float16).astype(np.float32)
        q, sc = quant_per_channel(w)
        return dequant(q, sc)

    Hz, Ho = W["z"].shape[0], W["out"].shape[0]
    hz, ho = held(W["z"]), held(W["out"])

    intn("Wz", W["z"])
    p.conv("cz", "x", "Wz", Hz)
    mid = "cz"
    if arm == "w8a8":
        mid = p.qdq("dz", "cz", Hz, scales["z"])
    intn("Wo", W["out"])
    p.conv("y_out", mid, "Wo", Ho)
    p.out("y_out", Ho)

    def ref(x):
        m = hz @ x
        if arm == "w8a8":
            m = sim_qdq(m, scales["z"])
        return [ho @ m]

    if stage == 1:
        return p, ref

    Hq = W["qkv"].shape[0]
    hq = held(W["qkv"])
    intn("Wq", W["qkv"])
    p.conv("y_qkv", "x", "Wq", Hq)
    if stage == 4:
        # depthwise causal conv1d consumes qkv; qkv itself stays an output so
        # a rejection here is attributable to the dw node and nothing else.
        pass
    p.out("y_qkv", Hq)

    def ref2(x):
        return ref(x) + [hq @ x]

    if stage == 2:
        return p, ref2

    Ha, Hb = W["a"].shape[0], W["b"].shape[0]
    ha, hb = held(W["a"]), held(W["b"])
    intn("Wa", W["a"])
    p.conv("y_a", "x", "Wa", Ha)
    p.out("y_a", Ha)
    intn("Wb", W["b"])
    p.conv("y_b", "x", "Wb", Hb)
    p.out("y_b", Hb)

    def ref3(x):
        return ref2(x) + [ha @ x, hb @ x]

    if stage == 3:
        return p, ref3

    # stage 4: depthwise causal conv1d over the qkv stream
    kern = W["conv1d"].reshape(W["conv1d"].shape[0], 1, 1, -1)
    hk = held(kern.reshape(kern.shape[0], -1)).reshape(kern.shape[0], -1)
    src = "y_qkv"
    if arm == "w8a8":
        src = p.qdq("dq", "y_qkv", Hq, scales["qkv"])
    intn("Wc", kern)
    p.dwconv_causal("y_dw", src, "Wc", Hq, kern.shape[-1])
    p.out("y_dw", Hq)

    def ref4(x):
        outs = ref3(x)
        s = hq @ x
        if arm == "w8a8":
            s = sim_qdq(s, scales["qkv"])
        return outs + [causal_dw(s, hk)]

    return p, ref4


STAGE_LABEL = {
    1: "S1 z->qdq->out_proj                (2 convs)",
    2: "S2 + in_proj_qkv                   (3 convs)",
    3: "S3 + in_proj_a, in_proj_b          (5 convs)",
    4: "S4 + depthwise causal conv1d k=4   (6 convs)",
}
OUT_NAMES = {1: ["out_proj"], 2: ["out_proj", "qkv"],
             3: ["out_proj", "qkv", "a", "b"],
             4: ["out_proj", "qkv", "a", "b", "conv1d"]}


def main() -> None:
    if not eng._available:
        print("AneEngine unavailable")
        return
    if not SHARD.exists():
        print(f"missing {SHARD}")
        return
    S = int(os.environ.get("BLOCK_S", "256"))
    stages = [int(s) for s in os.environ.get("BLOCK_STAGES", "1,2,3,4").split(",")]
    arms = os.environ.get("BLOCK_ARMS", "fp16,w8a16,w8a8").split(",")
    act_mode = os.environ.get("BLOCK_ACT", "outlier")

    names = {"z": "in_proj_z.weight", "out": "out_proj.weight",
             "qkv": "in_proj_qkv.weight", "a": "in_proj_a.weight",
             "b": "in_proj_b.weight", "conv1d": "conv1d.weight"}
    t = read_safetensors(SHARD, [PREFIX + v for v in names.values()])
    W = {k: t[PREFIX + v] for k, v in names.items()}
    H = W["z"].shape[1]

    print(f"Qwen3.8-Flash-Next layer 0 linear_attn, S={S}, "
          f"Q38_ANE_REUSE_COMPILED={os.environ.get('Q38_ANE_REUSE_COMPILED')}")
    for k in ("z", "out", "qkv", "a", "b", "conv1d"):
        print(f"  {names[k]:<22} {tuple(W[k].shape)}")

    rng = np.random.default_rng(0)

    def draw(seed):
        r = np.random.default_rng(seed)
        x = r.standard_normal((H, S)).astype(np.float32) * 0.1
        if act_mode == "outlier":
            x[r.choice(H, 16, replace=False)] *= 20.0
        return x

    # Activation scales come from a SEPARATE calibration draw. Calibrating on
    # the scored tensor is an oracle and hides the clipping W8A8 is judged on.
    xc = draw(101)
    scales = {
        "z": act_scale(W["z"] @ xc),
        "qkv": act_scale(W["qkv"] @ xc),
    }
    x = draw(7)
    print(f"\nactivations: {act_mode}   |x|max={np.abs(x).max():.2f}   "
          f"act scales calibrated on an independent draw "
          f"(z max {scales['z'].max():.4f}, qkv max {scales['qkv'].max():.4f})")

    rows = []
    for stage in stages:
        print(f"\n== {STAGE_LABEL[stage]}")
        print(f"   {'arm':>7} {'weight.bin':>11} {'ms':>8}   per-output rel L2 "
              f"vs fp32 ({', '.join(OUT_NAMES[stage])})")
        for arm in arms:
            p, tail, nbytes, mil = build_stage_and_compile(stage, arm, W, S, scales)
            if p is None:
                print(f"   {arm:>7} {nbytes / 1e6:>10.1f}M   REJECTED  {tail[:150]}")
                rows.append((stage, arm, None, None, tail))
                continue
            prog, refs = p
            r = Runner(prog, H, S, prog_outs(stage, W))
            got = r.run(x)
            if got is None:
                print(f"   {arm:>7} {nbytes / 1e6:>10.1f}M   EVAL FAILED  "
                      f"{getattr(r, 'err', '')}")
                rows.append((stage, arm, None, None, "eval failed"))
                del prog
                continue
            ref = refs(x)
            errs = [rel_l2(g, rr) for g, rr in zip(got, ref)]
            ms = r.timed()
            print(f"   {arm:>7} {nbytes / 1e6:>10.1f}M {ms:>8.3f}   "
                  + "  ".join(f"{n}={e:.3e}" for n, e in
                              zip(OUT_NAMES[stage], errs)))
            rows.append((stage, arm, errs, ms, ""))
            del prog

    print("\n" + "=" * 78)
    fp16_s1 = next((r[2][0] for r in rows
                    if r[0] == 1 and r[1] == "fp16" and r[2]), None)
    if fp16_s1 is None:
        print("fp16 CONTROL DID NOT RUN -- no other number here is trustworthy.")
    elif not (1e-3 < fp16_s1 < 1.5e-2):
        print(f"fp16 CONTROL IS {fp16_s1:.3e}, NOT near 4e-3: THE HARNESS IS "
              f"WRONG and every other number above is meaningless.")
    else:
        print(f"fp16 control at stage 1 = {fp16_s1:.3e} (near the expected "
              f"4e-3): harness believed good.")


def prog_outs(stage, W):
    """Output channel list per stage, in program order."""
    o = [("out_proj", W["out"].shape[0])]
    if stage >= 2:
        o.append(("qkv", W["qkv"].shape[0]))
    if stage >= 3:
        o += [("a", W["a"].shape[0]), ("b", W["b"].shape[0])]
    if stage >= 4:
        o.append(("conv1d", W["qkv"].shape[0]))
    return o


def build_stage_and_compile(stage, arm, W, S, scales):
    prog, ref = build_stage(stage, arm, W, S, scales)
    p, tail, nbytes, mil = compile_quiet(prog)
    if p is None:
        return None, tail, nbytes, mil
    return (p, ref), tail, nbytes, mil


if __name__ == "__main__":
    main()
