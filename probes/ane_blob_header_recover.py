#!/usr/bin/env python3
"""Which weight-blob container spelling lets ONE MIL program read SEVERAL
tensors from ONE packed weight.bin?

Two measured facts frame this probe:

  * A program with more than one blob FILE is rejected at
    ``verifyBundleAtPath: invalid model`` (Code 10) at any size, so everything
    has to go in a single packed file.
  * Concatenating ``runtime.q38_ane_engine._make_blob`` chunks and addressing
    each by byte offset reads the FIRST tensor correctly and garbage for every
    later one.

``_make_blob`` emits 64 bytes of FILE header (uint32 1, uint32 2) followed by
64 bytes of CHUNK header (DEADBEEF, version, payload size, ... , uint32 0x80 at
header byte 80 = chunk byte 16) and the payload at 128. Concatenating N of them
therefore repeats the file header N times AND writes 0x80 -- an absolute file
offset of 128, i.e. chunk 0's payload -- into every chunk. Every later chunk
resolves its payload back into the first chunk. That is the hypothesis under
test.

The known-good counter-example is ``buildWeightBlobInt8`` in the ANE gist
(inmem_peak_int8.m): ONE 64-byte file header, then per-chunk 64-byte headers
with DEADBEEF and NO payload-offset field at all, payload immediately after the
chunk header, MIL ``offset`` pointing at the chunk start. It packs 128 tensors
in one file and runs correctly.

Method: a two-conv fp16 chain, 512 -> 512 -> 512 at S=64, with hand-picked
small-integer weights, packed into one weight.bin. Every variant is scored
three ways so a misread is attributed to a specific tensor rather than to the
aggregate:

  T0    one-conv program reading chunk 0 only
  T1    one-conv program reading chunk 1 only, out of the SAME packed file
  CHAIN two-conv program reading both

T1 also gets an alias check: its output is compared against ``W0 @ x``, which
is what comes back if chunk 1 resolved its payload into chunk 0.

Result: the fix is one uint32. See ``pack_weight_bin`` below -- the payload
pointer at chunk byte 16 has to be the chunk's own offset ABSOLUTE FROM FILE
START, and the payload size at chunk byte 8 has to be right or
``constexpr_blockwise_shift_scale`` will not compile. Verified with 8 fp16
tensors and 8 int8-plus-per-channel-scale tensors in one file.

Run with ``Q38_ANE_REUSE_COMPILED=0``; the compiler cache is content-addressed
on the MIL text plus weights and will happily serve a poisoned artifact from an
earlier run of this probe.

Env: ``BLOB_DIM`` ``BLOB_S`` ``BLOB_N`` (stage 2 tensor count) ``BLOB_NCONV``
(stage 3 conv count) ``BLOB_SWEEP_ONLY`` ``BLOB_SKIP_SWEEP``
``BLOB_SCALE_VARIANT`` ``BLOB_ERRLINES``.
"""
from __future__ import annotations

import contextlib
import io
import os
import struct
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

eng = AneEngine()

DIM = int(os.environ.get("BLOB_DIM", "512"))
S = int(os.environ.get("BLOB_S", "64"))

# fp16 dtype marker in the gist's chunk-header byte 10 slot; int8 is 0x08.
MARK_FP16 = 0x04
MARK_INT8 = 0x08


# ---------------------------------------------------------------------------
# THE ANSWER
#
# Measured winner, distilled from the sweep below. Three fields are load
# bearing and nothing else is:
#
#   * uint32 1 / uint32 2 at FILE bytes 0 and 4, once for the whole file.
#     Zero them and every const is rejected (`InvalidMILProgram`), which is
#     also why an arena with no leading file header fails.
#   * 0xDEADBEEF at the start of each 64-byte chunk header, which is what the
#     MIL ``offset`` points at.
#   * uint32 at chunk byte 16 = the chunk's payload offset ABSOLUTE FROM FILE
#     START. This is the whole bug: ``_make_blob`` hardcodes 0x80 there, so
#     every chunk past the first resolves its payload into chunk 0.
#
# Plus one field that is conditionally required:
#
#   * uint32 payload size at chunk byte 8. A plain fp16/int8 ``const`` reads
#     correctly without it, but ``constexpr_blockwise_shift_scale`` (and hence
#     any per-output-channel dequant) fails to compile unless it is right.
#     Always write it. The gist's dtype marker at chunk byte 10 lands inside
#     this field and breaks exactly that node -- do not write a marker.
#
# The chunk version field at byte 4 is NOT load bearing (variant m passes with
# it zeroed); it is written anyway to match what the ANE runtime expects.
# ---------------------------------------------------------------------------

def pack_weight_bin(payloads):
    """Pack raw tensor payloads into one ANE weight.bin.

    Returns ``(file_bytes, offsets)`` where ``offsets[k]`` is the value to put
    in tensor k's MIL ``BLOBFILE(path=..., offset=uint64(offsets[k]))``.

    Payloads are 64-byte aligned in the arena, so this is also correct for a
    per-channel scale tensor whose payload is not a multiple of 64.
    """
    n = len(payloads)
    head = bytearray(64)
    struct.pack_into("<II", head, 0, 1, 2)      # file magic, required
    table = bytearray(64 * n)
    arena = bytearray()
    arena_base = 64 + 64 * n
    offsets = []
    for k, payload in enumerate(payloads):
        pos = arena_base + len(arena)
        struct.pack_into("<I", table, 64 * k + 0, 0xDEADBEEF)
        struct.pack_into("<I", table, 64 * k + 4, 1)            # version
        struct.pack_into("<I", table, 64 * k + 8, len(payload))  # size
        struct.pack_into("<I", table, 64 * k + 16, pos)          # ABSOLUTE
        offsets.append(64 + 64 * k)
        arena += payload
        arena += b"\0" * (-len(payload) % 64)
    return bytes(head) + bytes(table) + bytes(arena), offsets


# ---------------------------------------------------------------------------
# container variants
#
# Each builder takes [(payload_bytes, dtype_marker), ...] and returns
# (packed_file_bytes, [mil_offset_per_tensor]).
# ---------------------------------------------------------------------------

def pack_makeblob_concat(items):
    """BASELINE: today's spelling. N x _make_blob concatenated.

    Repeats the 64-byte file header per chunk and leaves the payload-offset
    field at the absolute value 0x80 in every chunk.
    """
    chunks, offs = [], []
    for payload, _mark in items:
        base = sum(len(c) for c in chunks)
        chunks.append(E._make_blob(payload))
        offs.append(base + 64)          # MIL offset lands on DEADBEEF
    return b"".join(chunks), offs


def pack_gist(items, *, marker=False, size_field=False, payload_off=None,
              file_magic=True, version=1):
    """Gist layout: ONE file header, then 64-byte chunk headers.

      file[0:4]    uint32 1
      file[4:8]    uint32 2
      file[8:64]   zeros
      chunk[0:4]   DEADBEEF
      chunk[4:8]   uint32 1        (version)
      chunk[8:12]  payload size    (only if size_field)
      chunk[10]    dtype marker    (only if marker; overlaps the size field,
                                    which is why the two are mutually
                                    exclusive in the variants below)
      chunk[16:20] payload offset  (only if payload_off is not None)
      chunk[64:]   payload

    ``payload_off`` selects what goes in the 0x80 slot: None omits it (gist),
    "abs" writes the chunk's own payload offset from file start, "rel" writes
    64, "zero" writes 0.
    """
    head = bytearray(64)
    if file_magic:
        struct.pack_into("<II", head, 0, 1, 2)
    body = bytearray()
    offs = []
    for payload, mark in items:
        base = 64 + len(body)
        hdr = bytearray(64)
        struct.pack_into("<II", hdr, 0, 0xDEADBEEF, version)
        if size_field:
            struct.pack_into("<I", hdr, 8, len(payload))
        if marker:
            hdr[10] = mark
        if payload_off == "abs":
            struct.pack_into("<I", hdr, 16, base + 64)
        elif payload_off == "rel":
            struct.pack_into("<I", hdr, 16, 64)
        elif payload_off == "zero":
            struct.pack_into("<I", hdr, 16, 0)
        body += hdr + payload
        offs.append(base)
    return bytes(head) + bytes(body), offs


def pack_makeblob_abs(items):
    """_make_blob layout per chunk, 0x80 field rewritten to the chunk's own
    absolute payload offset."""
    chunks, offs = [], []
    for payload, _mark in items:
        base = sum(len(c) for c in chunks)
        blob = bytearray(E._make_blob(payload))
        struct.pack_into("<I", blob, 80, base + 128)
        chunks.append(bytes(blob))
        offs.append(base + 64)
    return b"".join(chunks), offs


def pack_split_arena(items, *, size_field=True, file_header=True):
    """All 64-byte chunk headers first, then every payload in one arena.

    If the 0x80 slot is genuinely a file-absolute payload pointer then the
    payload does not have to sit behind its own header at all, and a header
    table plus a payload arena is legal. That is the cleanest possible proof of
    the addressing mechanism, and it is also the layout you want in practice
    because the arena can be page-aligned independently of the headers.
    """
    head = bytearray(64) if file_header else bytearray()
    if file_header:
        struct.pack_into("<II", head, 0, 1, 2)
    n = len(items)
    hdr_table = bytearray()
    arena = bytearray()
    arena_base = len(head) + 64 * n
    offs = []
    for payload, _mark in items:
        hdr = bytearray(64)
        struct.pack_into("<II", hdr, 0, 0xDEADBEEF, 1)
        if size_field:
            struct.pack_into("<I", hdr, 8, len(payload))
        struct.pack_into("<I", hdr, 16, arena_base + len(arena))
        offs.append(len(head) + len(hdr_table))
        hdr_table += hdr
        arena += payload
    return bytes(head) + bytes(hdr_table) + bytes(arena), offs


def pack_makeblob_relzero(items, value):
    """_make_blob layout per chunk with the 0x80 field forced to ``value``
    (64 = chunk-relative payload start, 0 = absent)."""
    chunks, offs = [], []
    for payload, _mark in items:
        base = sum(len(c) for c in chunks)
        blob = bytearray(E._make_blob(payload))
        struct.pack_into("<I", blob, 80, value)
        chunks.append(bytes(blob))
        offs.append(base + 64)
    return b"".join(chunks), offs


VARIANTS = {
    # name                        builder
    "baseline_makeblob_concat": pack_makeblob_concat,
    "a_gist_exact":             lambda it: pack_gist(it),
    "b_gist_dtype_marker":      lambda it: pack_gist(it, marker=True),
    "c_makeblob_abs_payoff":    pack_makeblob_abs,
    "d_makeblob_payoff_64":     lambda it: pack_makeblob_relzero(it, 64),
    "e_makeblob_payoff_0":      lambda it: pack_makeblob_relzero(it, 0),
    "f_gist_size_field":        lambda it: pack_gist(it, size_field=True),
    "g_gist_size_abs_payoff":   lambda it: pack_gist(it, size_field=True,
                                                     payload_off="abs"),
    # field-isolation arms, added after c/g both passed: c and g have the SAME
    # chunk header and differ only in whether the 64-byte file header is
    # repeated, so the remaining questions are which chunk fields are load
    # bearing and whether the payload has to sit behind its header.
    "h_abs_payoff_no_size":     lambda it: pack_gist(it, payload_off="abs"),
    "i_no_file_header":         lambda it: pack_split_arena(it,
                                                            file_header=False),
    "j_header_table_arena":     lambda it: pack_split_arena(it),
    "k_abs_payoff_marker":      lambda it: pack_gist(it, marker=True,
                                                     payload_off="abs"),
    # `i` failed with two confounds at once (no file header AND chunk 0 at MIL
    # offset 0). These split them: `l` keeps 64 leading bytes but zeroes the
    # 1/2 magic, `m` zeroes the chunk version field.
    "l_zeroed_file_header":     lambda it: pack_gist(it, payload_off="abs",
                                                     file_magic=False),
    "m_zero_chunk_version":     lambda it: pack_gist(it, payload_off="abs",
                                                     version=0),
    "WINNER_pack_weight_bin":   lambda it: pack_weight_bin([p for p, _ in it]),
}

# Arms that read every fp16 tensor correctly but still fail the int8 +
# per-output-channel-scale arm, because they do not carry a correct payload
# size at chunk byte 8: h omits it, k overwrites its byte 10 with a dtype
# marker. Kept as variants so the size field's role stays visible.
NO_SIZE_FIELD = {"h_abs_payoff_no_size", "k_abs_payoff_marker"}


# ---------------------------------------------------------------------------
# MIL
# ---------------------------------------------------------------------------

_PRE = (
    '    string pt = const()[name=string("pt"), val=string("valid")];\n'
    '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '    int32 gr = const()[name=string("gr"), val=int32(1)];'
)
BLOB = "@model_path/weights/weight.bin"


def _fp16_const(idx, out_dim, in_dim, off):
    return (f'    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W{idx} = const()'
            f'[name=string("W{idx}"), val=tensor<fp16, [{out_dim}, {in_dim}, 1, 1]>'
            f'(BLOBFILE(path=string("{BLOB}"), offset=uint64({off})))];')


def _int8_const(idx, out_dim, in_dim, doff, soff):
    """int8 data + per-output-channel fp16 scale.

    constexpr_affine_dequantize takes a SCALAR scale only, so per-channel
    scales have to go through constexpr_blockwise_shift_scale.
    """
    return (f'    tensor<int8, [{out_dim}, {in_dim}, 1, 1]> W{idx}q = const()'
            f'[name=string("W{idx}q"), val=tensor<int8, [{out_dim}, {in_dim}, 1, 1]>'
            f'(BLOBFILE(path=string("{BLOB}"), offset=uint64({doff})))];\n'
            f'    tensor<fp16, [{out_dim}, 1, 1, 1]> W{idx}s = const()'
            f'[name=string("W{idx}s"), val=tensor<fp16, [{out_dim}, 1, 1, 1]>'
            f'(BLOBFILE(path=string("{BLOB}"), offset=uint64({soff})))];\n'
            f'    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W{idx} = '
            f'constexpr_blockwise_shift_scale(data=W{idx}q, scale=W{idx}s)'
            f'[name=string("W{idx}dq")];')


def build_chain(decls, dims, seq):
    """decls[k] declares W{k}; dims = [in, out0, out1, ...]."""
    lines = []
    prev = "x"
    for k, decl in enumerate(decls):
        lines.append(decl)
        name = "y" if k == len(decls) - 1 else f"c{k}"
        lines.append(
            f'    tensor<fp16, [1, {dims[k + 1]}, 1, {seq}]> {name} = conv('
            f'dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, '
            f'weight=W{k}, x={prev})[name=string("{name}")];')
        prev = name
    return (f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
            f"  func main<ios18>(tensor<fp16, [1, {dims[0]}, 1, {seq}]> x) {{\n"
            f"{_PRE}\n" + "\n".join(lines) + "\n  } -> (y);\n}\n")


def compile_quiet(mil, packed, in_dim, out_dim, seq):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(
                mil, {"weight.bin": packed}, in_dim, out_dim, seq,
                raw_weight_files=frozenset({"weight.bin"}))
        except Exception as exc:                    # noqa: BLE001
            buf.write(f"exception: {exc!r}\n")
            p = None
    keep = 2 if p is not None else int(os.environ.get("BLOB_ERRLINES", "2"))
    tail = " | ".join(buf.getvalue().strip().splitlines()[-keep:])
    return p, tail


def run(p, x, out_dim, seq):
    """x is planar [in, seq] fp32; returns [out, seq] fp32."""
    eng._ensure_io(p)
    with _iosurface_view(p._in_surf, x.shape, np.float16) as dst:
        np.copyto(dst, x.astype(np.float16))
    if not eng.submit(p, procedure_index=0):
        return None
    with _iosurface_view(p._out_surf, (out_dim, seq), np.float16) as o:
        return np.array(o, np.float32)


def rel_l2(got, ref):
    if got is None:
        return float("nan")
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))


# ---------------------------------------------------------------------------
# known weights
# ---------------------------------------------------------------------------

def known_weight(idx, out_dim, in_dim):
    """Small distinct integers, a different pattern per tensor.

    Integers keep the fp16 store exact, so any deviation past ~1e-3 is a
    container bug and not rounding. The patterns are mutually uncorrelated, so
    reading tensor j's bytes for tensor k shows up as rel L2 near 1.
    """
    o = np.arange(out_dim)[:, None]
    i = np.arange(in_dim)[None, :]
    a, b, m = (7, 13, 5), (3, 11, 7), (5, 17, 9)
    p = (a, b, m, (11, 23, 11))[idx % 4]
    w = ((o * p[0] + i * p[1] + idx * 31) % p[2]).astype(np.float32) - (p[2] // 2)
    return w / 8.0          # keep the chain's magnitude in fp16 range


def quant_per_channel(w):
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
    q = np.clip(np.rint(w / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return q, scale


# ---------------------------------------------------------------------------
# stage 1: variant sweep on two fp16 tensors
# ---------------------------------------------------------------------------

def sweep(rng):
    w0 = known_weight(0, DIM, DIM)
    w1 = known_weight(1, DIM, DIM)
    x = rng.standard_normal((DIM, S)).astype(np.float32) * 0.1

    ref0 = w0 @ x
    ref1 = w1 @ x
    ref_chain = w1 @ (w0 @ x)

    items = [(w0.astype(np.float16).tobytes(), MARK_FP16),
             (w1.astype(np.float16).tobytes(), MARK_FP16)]

    print(f"stage 1: 2 fp16 tensors [{DIM},{DIM}] in one weight.bin, S={S}")
    print(f"  payload {DIM * DIM * 2} B each (64-byte multiple, so alignment "
          f"is not a confound)")
    print(f"  {'variant':<26} {'T0':>10} {'T1':>10} {'T1=W0?':>10} "
          f"{'CHAIN':>10}   verdict")
    rows = []
    for name, builder in VARIANTS.items():
        packed, offs = builder(items)
        res = {}
        # T0 / T1: one conv, same packed file, different offset.
        for k, (off, ref) in enumerate(((offs[0], ref0), (offs[1], ref1))):
            mil = build_chain([_fp16_const(0, DIM, DIM, off)], [DIM, DIM], S)
            p, tail = compile_quiet(mil, packed, DIM, DIM, S)
            if p is None:
                res[f"T{k}"] = ("REJECT", tail)
                continue
            got = run(p, x, DIM, S)
            res[f"T{k}"] = (rel_l2(got, ref), got)
            del p
        # alias check: did chunk 1 read chunk 0's payload?
        alias = float("nan")
        t1 = res.get("T1", (None, None))
        if not isinstance(t1[0], str) and t1[1] is not None:
            alias = rel_l2(t1[1], ref0)
        # CHAIN
        mil = build_chain([_fp16_const(0, DIM, DIM, offs[0]),
                           _fp16_const(1, DIM, DIM, offs[1])],
                          [DIM, DIM, DIM], S)
        p, tail = compile_quiet(mil, packed, DIM, DIM, S)
        if p is None:
            res["CHAIN"] = ("REJECT", tail)
        else:
            res["CHAIN"] = (rel_l2(run(p, x, DIM, S), ref_chain), None)
            del p

        def fmt(v):
            return f"{v:>10}" if isinstance(v, str) else f"{v:>10.3e}"

        e0 = res["T0"][0]
        e1 = res["T1"][0]
        ec = res["CHAIN"][0]
        ok = (not isinstance(e0, str) and not isinstance(e1, str)
              and not isinstance(ec, str)
              and e0 < 5e-3 and e1 < 5e-3 and ec < 5e-3)
        verdict = "PASS" if ok else "fail"
        if not ok and not isinstance(e1, str) and alias < 5e-3:
            verdict = "fail (T1 aliased to chunk 0)"
        print(f"  {name:<26} {fmt(e0)} {fmt(e1)} "
              f"{'   -      ' if alias != alias else f'{alias:>10.3e}'} "
              f"{fmt(ec)}   {verdict}")
        for key in ("T0", "T1", "CHAIN"):
            if isinstance(res[key][0], str):
                print(f"      {key} rejected: {res[key][1][:110]}")
        rows.append((name, ok))
    return [n for n, ok in rows if ok]


# ---------------------------------------------------------------------------
# stage 2: does the winner scale to 4 fp16 tensors?
# ---------------------------------------------------------------------------

def scale_fp16(name, builder, rng, n=4):
    ws = [known_weight(k, DIM, DIM) for k in range(n)]
    x = rng.standard_normal((DIM, S)).astype(np.float32) * 0.1
    items = [(w.astype(np.float16).tobytes(), MARK_FP16) for w in ws]
    packed, offs = builder(items)
    print(f"\nstage 2: {n} fp16 tensors in one weight.bin "
          f"({len(packed) / 1e6:.2f} MB), variant {name}")

    allok = True
    for k in range(n):
        mil = build_chain([_fp16_const(0, DIM, DIM, offs[k])], [DIM, DIM], S)
        p, tail = compile_quiet(mil, packed, DIM, DIM, S)
        if p is None:
            print(f"  T{k} REJECT {tail[:100]}")
            allok = False
            continue
        err = rel_l2(run(p, x, DIM, S), ws[k] @ x)
        del p
        mark = "ok" if err < 5e-3 else "BAD"
        print(f"  T{k} isolated read  rel L2 {err:.3e}  {mark}")
        allok &= err < 5e-3

    decls = [_fp16_const(k, DIM, DIM, offs[k]) for k in range(n)]
    mil = build_chain(decls, [DIM] * (n + 1), S)
    p, tail = compile_quiet(mil, packed, DIM, DIM, S)
    ref = x
    for w in ws:
        ref = w @ ref
    if p is None:
        print(f"  {n}-conv chain REJECT {tail[:100]}")
        allok = False
    else:
        err = rel_l2(run(p, x, DIM, S), ref)
        del p
        print(f"  {n}-conv chain     rel L2 {err:.3e}  "
              f"{'ok' if err < 5e-2 else 'BAD'}")
        allok &= err < 5e-2
    return allok


# ---------------------------------------------------------------------------
# stage 3: int8 data + per-output-channel fp16 scale, in the same file
# ---------------------------------------------------------------------------

def scale_int8(name, builder, rng, nconv=2):
    ws = [known_weight(k, DIM, DIM) for k in range(nconv)]
    x = rng.standard_normal((DIM, S)).astype(np.float32) * 0.1
    items, deq = [], []
    for w in ws:
        q, sc = quant_per_channel(w)
        items.append((q.tobytes(), MARK_INT8))
        items.append((sc.astype(np.float16).tobytes(), MARK_FP16))
        deq.append(q.astype(np.float32) * sc.astype(np.float32))
    packed, offs = builder(items)
    print(f"\nstage 3: {nconv} x (int8 data + per-channel fp16 scale [O,1,1,1]) "
          f"= {len(items)} tensors in one weight.bin, variant {name}")
    print(f"  scale payload is {DIM * 2} B (still a 64-byte multiple); "
          f"consumed by constexpr_blockwise_shift_scale")

    allok = True
    for k in range(nconv):
        mil = build_chain(
            [_int8_const(0, DIM, DIM, offs[2 * k], offs[2 * k + 1])],
            [DIM, DIM], S)
        p, tail = compile_quiet(mil, packed, DIM, DIM, S)
        if p is None:
            print(f"  conv{k} isolated REJECT {tail[:100]}")
            allok = False
            continue
        err = rel_l2(run(p, x, DIM, S), deq[k] @ x)
        del p
        print(f"  conv{k} isolated (data+scale)  rel L2 {err:.3e}  "
              f"{'ok' if err < 5e-3 else 'BAD'}")
        allok &= err < 5e-3

    decls = [_int8_const(k, DIM, DIM, offs[2 * k], offs[2 * k + 1])
             for k in range(nconv)]
    mil = build_chain(decls, [DIM] * (nconv + 1), S)
    p, tail = compile_quiet(mil, packed, DIM, DIM, S)
    ref = x
    for w in deq:
        ref = w @ ref
    if p is None:
        print(f"  {nconv}-conv int8 chain REJECT {tail[:100]}")
        allok = False
    else:
        err = rel_l2(run(p, x, DIM, S), ref)
        del p
        print(f"  {nconv}-conv int8 chain          rel L2 {err:.3e}  "
              f"{'ok' if err < 5e-2 else 'BAD'}")
        allok &= err < 5e-2
    return allok


def main() -> None:
    if not eng._available:
        print("AneEngine unavailable")
        return
    print(f"Q38_ANE_REUSE_COMPILED={os.environ.get('Q38_ANE_REUSE_COMPILED')}\n")
    only = os.environ.get("BLOB_SCALE_VARIANT")
    if os.environ.get("BLOB_SKIP_SWEEP") == "1":
        winners = [only] if only else list(VARIANTS)
    else:
        winners = sweep(np.random.default_rng(0))
        print(f"\nvariants that read both tensors correctly: {winners or 'NONE'}")
        if os.environ.get("BLOB_SWEEP_ONLY") == "1":
            return
        # The sweep only scores fp16 consts. Scale-test the canonical packer
        # plus one size-field-less arm, so the int8 stage shows the size field
        # being required rather than just asserting it.
        winners = [only] if only else (
            [n for n in ("WINNER_pack_weight_bin", "h_abs_payoff_no_size")
             if n in winners])
    for name in winners:
        builder = VARIANTS[name]
        n4 = scale_fp16(name, builder, np.random.default_rng(1),
                        n=int(os.environ.get("BLOB_N", "4")))
        i8 = scale_int8(name, builder, np.random.default_rng(2),
                        nconv=int(os.environ.get("BLOB_NCONV", "2")))
        nf = int(os.environ.get("BLOB_N", "4"))
        print(f"\n{name}: {nf}x fp16 {'OK' if n4 else 'FAILED'}, "
              f"int8+per-channel-scale {'OK' if i8 else 'FAILED'}")


if __name__ == "__main__":
    main()
