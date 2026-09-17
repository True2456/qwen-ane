# SPDX-License-Identifier: Apache-2.0
"""q38_ane_engine: Independent ANE engine for M5 Max.

Drives the Apple Neural Engine through AppleNeuralEngine.framework's
private _ANEInMemoryModel API. No CoreML, no coremltools.

Architecture (from ground truth capture, §4/§11 of the ANE doc):
  1. Emit MIL text for a 1×1 conv (= linear layer), fp16, ios18 opset
  2. Write weight blobs to a temp dir with 64-byte headers
  3. Build _ANEInMemoryModelDescriptor with modelWithMILText:weights:optionsPlist:
  4. Compile, load via _ANEInMemoryModel
  5. Allocate IOSurface-backed buffers shared between ANE and Metal
  6. Submit via _ANERequest with procedureIndex
  7. Synchronise with Metal via shared events

The key finding from §11: the weights dict passed to modelWithMILText: uses
the full BLOBFILE path string as the key, and the value is a dict with
{"data": NSData, "offset": NSNumber}. The MIL text references these via
@model_path/weights/filename.bin, but the framework resolves them through
the weights dict, not the filesystem.

Standalone compile needs the same on-disk layout the ANE compiler expects *before*
compileWithQoS: hashed temp dir with model.mil + weights/*.bin, nil
optionsPlist, and MIL 1x1-conv programs. saveModelFiles is the espresso
path and must not be used for MIL. Blob headers are 128 bytes with
payload at offset 0x80.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import logging
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# ObjC runtime bridge
# ============================================================================

_objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")

# Configure function signatures
for _name, _res, _args in [
    ("objc_getClass", ctypes.c_void_p, [ctypes.c_char_p]),
    ("object_getClass", ctypes.c_void_p, [ctypes.c_void_p]),
    ("sel_registerName", ctypes.c_void_p, [ctypes.c_char_p]),
    ("class_getClassMethod", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
    ("class_getInstanceMethod", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
    ("method_getImplementation", ctypes.c_void_p, [ctypes.c_void_p]),
    ("method_setImplementation", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
    ("class_getName", ctypes.c_char_p, [ctypes.c_void_p]),
]:
    _f = getattr(_objc, _name)
    _f.restype = _res
    _f.argtypes = _args


def _sel(name: str) -> ctypes.c_void_p:
    return _objc.sel_registerName(name.encode())


_MSGSEND_ADDR = ctypes.cast(_objc.objc_msgSend, ctypes.c_void_p).value


def _objc_call(restype, argtypes, obj, sel_name, *args):
    """objc_msgSend with a fresh prototype (does not share ``_msg`` argtypes).

    ``_msg`` writes ``_objc.objc_msgSend.argtypes``. A later
    ``CFUNCTYPE(...)(("objc_msgSend", _objc))`` reuses that prototype, so a
    leftover ``POINTER`` type makes the next integer argument raise
    ``expected LP_c_void_p instance instead of int``. Binding the function
    address avoids that.
    """
    proto = ctypes.CFUNCTYPE(
        restype, ctypes.c_void_p, ctypes.c_void_p, *tuple(argtypes)
    )
    return proto(_MSGSEND_ADDR)(obj, _sel(sel_name), *args)


def _as_void_p(p) -> ctypes.c_void_p:
    if p is None:
        return ctypes.c_void_p(0)
    if isinstance(p, ctypes.c_void_p):
        return p
    return ctypes.c_void_p(int(p))


def _cls(name: str) -> ctypes.c_void_p:
    return _objc.objc_getClass(name.encode())


def _msg(obj, sel_name: str, *args,
         restype=ctypes.c_void_p,
         argtypes: list | None = None):
    """Send an ObjC message."""
    f = _objc.objc_msgSend
    f.restype = restype
    base = [ctypes.c_void_p, ctypes.c_void_p]
    f.argtypes = base + (argtypes or [])
    return f(obj, _sel(sel_name), *args)


def _nsdata(data: bytes) -> ctypes.c_void_p:
    """Create NSData from Python bytes (no-copy, caller must keep bytes alive)."""
    NSData = _cls("NSData")
    f = _objc.objc_msgSend
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_bool]
    buf = ctypes.create_string_buffer(data)
    # dataWithBytes:length: (copies the data)
    f2 = _objc.objc_msgSend
    f2.restype = ctypes.c_void_p
    f2.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_ulonglong]
    return f2(NSData, _sel("dataWithBytes:length:"),
              ctypes.cast(buf, ctypes.c_void_p), len(data))


def _nsstring(s: str) -> ctypes.c_void_p:
    """Create NSString from Python str."""
    NSString = _cls("NSString")
    f = _objc.objc_msgSend
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    return f(NSString, _sel("stringWithUTF8String:"), s.encode())


def _nsnumber_int(val: int) -> ctypes.c_void_p:
    """Create NSNumber from int."""
    NSNumber = _cls("NSNumber")
    f = _objc.objc_msgSend
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
    return f(NSNumber, _sel("numberWithInt:"), val)


def _nsnumber_uint(val: int) -> ctypes.c_void_p:
    """Create NSNumber; small ints match standard NSConstantIntegerNumber."""
    return _nsnumber_int(val)


def _nsdict_pairs(pairs: list) -> ctypes.c_void_p:
    """NSDictionary from a list of (key, value) ObjC pointers."""
    NSDictionary = _cls("NSDictionary")
    keys_arr = (ctypes.c_void_p * len(pairs))()
    vals_arr = (ctypes.c_void_p * len(pairs))()
    for i, (k, v) in enumerate(pairs):
        keys_arr[i] = k
        vals_arr[i] = v
    f = _objc.objc_msgSend
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.POINTER(ctypes.c_void_p),
                  ctypes.POINTER(ctypes.c_void_p),
                  ctypes.c_ulonglong]
    return f(NSDictionary, _sel("dictionaryWithObjects:forKeys:count:"),
             vals_arr, keys_arr, len(pairs))


def _nsdict(d: dict) -> ctypes.c_void_p:
    """Create NSDictionary from Python dict of ObjC objects."""
    return _nsdict_pairs(list(d.items()))


def _nsarray(objs: list) -> ctypes.c_void_p:
    """Create NSArray from a list of ObjC objects."""
    NSArray = _cls("NSArray")
    arr = (ctypes.c_void_p * len(objs))(*objs)
    return _msg(
        NSArray,
        "arrayWithObjects:count:",
        arr,
        len(objs),
        argtypes=[ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulonglong],
    )


def _nsnumber_ulong(val: int) -> ctypes.c_void_p:
    NSNumber = _cls("NSNumber")
    f = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong
    )(("objc_msgSend", _objc))
    return f(NSNumber, _sel("numberWithUnsignedLong:"), val)


# ============================================================================
# IOSurface helpers (ANE I/O is a 1-row byte blob, not a real image)
# ============================================================================

_iosurf = None
_cf = None


def _load_iosurface():
    global _iosurf, _cf
    if _iosurf is not None:
        return
    _iosurf = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/IOSurface.framework/IOSurface"
    )
    _cf = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    _iosurf.IOSurfaceCreate.restype = ctypes.c_void_p
    _iosurf.IOSurfaceCreate.argtypes = [ctypes.c_void_p]
    _iosurf.IOSurfaceLock.restype = ctypes.c_int
    _iosurf.IOSurfaceLock.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
    ]
    _iosurf.IOSurfaceUnlock.restype = ctypes.c_int
    _iosurf.IOSurfaceUnlock.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
    ]
    _iosurf.IOSurfaceGetBaseAddress.restype = ctypes.c_void_p
    _iosurf.IOSurfaceGetBaseAddress.argtypes = [ctypes.c_void_p]
    _iosurf.IOSurfaceGetAllocSize.restype = ctypes.c_size_t
    _iosurf.IOSurfaceGetAllocSize.argtypes = [ctypes.c_void_p]
    _cf.CFRetain.restype = ctypes.c_void_p
    _cf.CFRetain.argtypes = [ctypes.c_void_p]


def _iosurf_key(name: str) -> ctypes.c_void_p:
    return ctypes.c_void_p.in_dll(_iosurf, name)


def _iosurface_alloc_size(n_fp16: int) -> int:
    """ANE buffer requirement: max(64KiB, round_up_64KiB(n_fp16 * 2))."""
    n = (n_fp16 * 2 + 0xFFFF) & ~0xFFFF
    return max(n, 0x10000)


def _create_iosurface(nbytes: int) -> ctypes.c_void_p:
    _load_iosurface()
    props = _nsdict_pairs([
        (_iosurf_key("kIOSurfaceWidth"), _nsnumber_ulong(nbytes)),
        (_iosurf_key("kIOSurfaceHeight"), _nsnumber_int(1)),
        (_iosurf_key("kIOSurfaceBytesPerElement"), _nsnumber_int(1)),
        (_iosurf_key("kIOSurfaceBytesPerRow"), _nsnumber_ulong(nbytes)),
        (_iosurf_key("kIOSurfaceAllocSize"), _nsnumber_ulong(nbytes)),
        (_iosurf_key("kIOSurfacePixelFormat"), _nsnumber_int(0)),
    ])
    surf = _iosurf.IOSurfaceCreate(props)
    if surf:
        _cf.CFRetain(surf)
    return surf


def _iosurface_rw(surf, data: bytes | None = None, *, read_n: int | None = None) -> bytes | None:
    seed = ctypes.c_uint32(0)
    _iosurf.IOSurfaceLock(surf, 0, ctypes.byref(seed))
    try:
        base = _iosurf.IOSurfaceGetBaseAddress(surf)
        size = int(_iosurf.IOSurfaceGetAllocSize(surf))
        if data is not None:
            if len(data) > size:
                raise ValueError(f"IOSurface write {len(data)} > alloc {size}")
            ctypes.memmove(base, data, len(data))
            return None
        n = size if read_n is None else read_n
        return ctypes.string_at(base, n)
    finally:
        _iosurf.IOSurfaceUnlock(surf, 0, ctypes.byref(seed))


_FP16_MAX = float(np.finfo(np.float16).max)
_FP16_MIN = -_FP16_MAX


def _copyto_fp16(dst: np.ndarray, src: np.ndarray) -> None:
    """Copy into an fp16 IOSurface view, saturating out-of-range float32 values.

    ANE surfaces are fp16, but hybrid MLP hands them activations from bf16/fp32
    GPU paths where gate/up tails and swiglu outputs can exceed ~65504. A plain
    ``copyto(..., casting=\"unsafe\")`` raises RuntimeWarning and stores inf.
    """
    if src.dtype == np.float16:
        np.copyto(dst, src)
        return
    arr = np.asarray(src, dtype=np.float32)
    if arr.size:
        finite = np.isfinite(arr)
        if finite.all():
            peak = float(np.max(np.abs(arr)))
            if peak <= _FP16_MAX:
                np.copyto(dst, arr, casting="unsafe")
                return
        arr = np.clip(
            np.nan_to_num(arr, nan=0.0, posinf=_FP16_MAX, neginf=_FP16_MIN),
            _FP16_MIN,
            _FP16_MAX,
        )
    np.copyto(dst, arr, casting="unsafe")


@contextlib.contextmanager
def _iosurface_view(surf, shape, dtype=np.float16):
    """Map an IOSurface's memory as a writable numpy view — no copy.

    _iosurface_rw round-trips through bytes: memmove from a bytes object in,
    string_at out. With the transpose and dtype conversions around it that was
    ~8 full copies of the activation per call, which measured as ~86% of the
    wall time of an ANE evaluate (9.42 ms for 1.3 ms of actual compute).

    The surface must stay locked while the view is alive, hence the context
    manager. Do not let the array escape the block.
    """
    seed = ctypes.c_uint32(0)
    _iosurf.IOSurfaceLock(surf, 0, ctypes.byref(seed))
    try:
        base = _iosurf.IOSurfaceGetBaseAddress(surf)
        size = int(_iosurf.IOSurfaceGetAllocSize(surf))
        count = int(np.prod(shape))
        need = count * np.dtype(dtype).itemsize
        if need > size:
            raise ValueError(f"view {need} bytes > IOSurface alloc {size}")
        raw = (ctypes.c_uint8 * size).from_address(base)
        yield np.frombuffer(raw, dtype=dtype, count=count).reshape(shape)
    finally:
        _iosurf.IOSurfaceUnlock(surf, 0, ctypes.byref(seed))


def _desc(obj) -> str:
    """Get ObjC object description as Python string."""
    if not obj:
        return "(nil)"
    d = _msg(obj, "description")
    if not d:
        return "(no description)"
    cs = _msg(d, "UTF8String", restype=ctypes.c_char_p)
    return cs.decode(errors="replace") if cs else "(encoding error)"


# ============================================================================
# Weight blob format
# ============================================================================

def _indexset_to_nsarray(idxset) -> ctypes.c_void_p:
    """NSIndexSet -> NSArray of NSNumber.

    inputSymbolIndicesForProcedureIndex: returns an NSIndexSet, but
    _ANERequest wants an NSArray and subscripts it, so passing the set
    straight through raises
    -[NSIndexSet objectAtIndexedSubscript:] unrecognized selector.
    """
    if not idxset:
        return None
    count = _objc_call(ctypes.c_ulonglong, (), idxset, "count")
    if not count:
        return None
    cur = _objc_call(ctypes.c_ulonglong, (), idxset, "firstIndex")
    out = []
    for _ in range(int(count)):
        out.append(_nsnumber_int(int(cur)))
        cur = _objc_call(
            ctypes.c_ulonglong,
            [ctypes.c_ulonglong],
            idxset,
            "indexGreaterThanIndex:",
            cur,
        )
    return _nsarray(out)


def _make_blob(data: bytes) -> bytes:
    """Wrap raw weight data in the milinternal blob layout expected by the ANE.

    Layout from ``AneLinearModel::Impl`` (calloc payload+0x80) and a live
    ``qwen35_ane_compile_linear`` capture:

      0-3:    uint32 1
      4-7:    uint32 2
      8-63:   zeros
      64-67:  DEADBEEF
      68-71:  uint32 1 (version)
      72-75:  payload size as uint32
      76-79:  zeros
      80-83:  uint32 0x80 (payload file offset)
      84-127: zeros
      128+:   payload (no extra alignment padding)

    MIL ``offset=uint64(64)`` lands on the DEADBEEF sub-header; the 0x80
    field tells ANECCompile where the tensor bytes start.

    NOTE: the byte-80 field is FILE-ABSOLUTE, so the hardcoded 0x80 is only
    correct for a blob that sits at file offset 0 — i.e. one tensor per file.
    Do NOT concatenate these chunks by hand; use ``_BlobPacker``, which
    relocates the pointer per chunk.
    """
    header = bytearray(128)
    struct.pack_into("<II", header, 0, 1, 2)
    struct.pack_into("<IIII", header, 64, 0xDEADBEEF, 1, len(data), 0)
    struct.pack_into("<I", header, 80, 0x80)
    return bytes(header) + data


class _BlobPacker:
    """Accumulate several milinternal tensors into one packed weight file.

    ``_make_blob`` alone cannot be concatenated: its payload pointer at header
    byte 80 is a FILE-ABSOLUTE offset hardcoded to 0x80, so every chunk past
    the first would claim its payload lives at file offset 128 — chunk 0's
    payload. Everything then silently reads the FIRST tensor. This rewrites
    that field to each chunk's own absolute payload offset, which is the same
    spelling ``tools/pure_ane.py``'s ``append_blob`` uses and the layout
    ``probes/ane_blob_header_recover.py`` measured as correct.

    The other load-bearing fields already come from ``_make_blob``: uint32
    1 / uint32 2 at file bytes 0 and 4 (without them every const is rejected
    with ``InvalidMILProgram``), 0xDEADBEEF at chunk byte 0, and the payload
    SIZE at chunk byte 8 — the size is specifically required by
    ``constexpr_blockwise_shift_scale``, i.e. by any per-output-channel
    dequant, and omitting it gives ``_ANECompiler Code=1``.

    ``append`` returns the chunk's START offset. The MIL ``BLOBFILE`` offset
    is that plus 64, since it must point at the DEADBEEF header rather than at
    the repeated file header. Chunks are padded to 64 bytes so a tensor whose
    payload is not a multiple of 64 (a per-channel scale, say) cannot push the
    next chunk header off alignment.
    """

    __slots__ = ("_parts", "_size")

    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self._size = 0

    def append(self, payload: bytes) -> int:
        """Append one tensor; return its chunk start offset in the file."""
        start = self._size
        payload_offset = start + 128
        if payload_offset > 0xFFFFFFFF:
            raise ValueError(
                f"packed weight file exceeds the 32-bit blob payload pointer "
                f"({payload_offset} > 4 GiB); split the bank"
            )
        blob = bytearray(_make_blob(payload))
        struct.pack_into("<I", blob, 80, payload_offset)
        self._parts.append(bytes(blob))
        self._size += len(blob)
        pad = -self._size % 64
        if pad:
            self._parts.append(b"\0" * pad)
            self._size += pad
        return start

    def __len__(self) -> int:
        return self._size

    def getvalue(self) -> bytes:
        return b"".join(self._parts)


# ============================================================================
# MIL text generation
# ============================================================================

_BUILD_INFO = (
    '[buildInfo = dict<string, string>({{'
    '"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, '
    '{"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]'
)


def generate_linear_mil(input_dim: int, output_dim: int, seq_len: int,
                        *, quantized: bool = True) -> str:
    """Generate MIL text for a single linear layer (as 1×1 conv).
    
    If quantized=True, uses int8 data + fp16 per-channel scale (matching §11).
    If quantized=False, uses fp16 weights directly.
    """
    if quantized:
        return f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {input_dim}, 1, {seq_len}]> x) {{
    tensor<int8, [{output_dim}, {input_dim}, 1, 1]> wd = const()[name=string("wd"), val=tensor<int8, [{output_dim}, {input_dim}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64(64)))];
    tensor<fp16, [{output_dim}, 1, 1, 1]> ws = const()[name=string("ws"), val=tensor<fp16, [{output_dim}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_scale.bin"), offset=uint64(64)))];
    tensor<fp16, [{output_dim}, {input_dim}, 1, 1]> w = constexpr_blockwise_shift_scale(data=wd, scale=ws)[name=string("dequant")];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {output_dim}, 1, {seq_len}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
  }} -> (y);
}}
"""
    else:
        return f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {input_dim}, 1, {seq_len}]> x) {{
    tensor<fp16, [{output_dim}, {input_dim}, 1, 1]> w = const()[name=string("w"), val=tensor<fp16, [{output_dim}, {input_dim}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64(64)))];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {output_dim}, 1, {seq_len}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
  }} -> (y);
}}
"""


def generate_dynamic_linear_mil(input_dim: int, output_dim: int, seq_len: int) -> str:
    """MIL for a 1×1 conv whose weight is a live feature-map input.

    ``wimg`` is ANE NCHW ``[1, O, 1, I]`` (planar ``[O, I]`` fp16), reshaped
    to conv weight ``[O, I, 1, 1]``. Compiled symbol order is typically
    ``(wimg, x)`` — bind IO from ``kANEFModelInputSymbolsArrayKey``, not
    MIL argument order.
    """
    return f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {input_dim}, 1, {seq_len}]> x, tensor<fp16, [1, {output_dim}, 1, {input_dim}]> wimg) {{
    tensor<int32, [4]> shp = const()[name=string("shp"), val=tensor<int32, [4]>([{output_dim}, {input_dim}, 1, 1])];
    tensor<fp16, [{output_dim}, {input_dim}, 1, 1]> w = reshape(shape=shp, x=wimg)[name=string("wr")];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {output_dim}, 1, {seq_len}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
  }} -> (y);
}}
// dynlin_{output_dim}x{input_dim}_S{seq_len}_v1
"""


INT4_GROUP = 64


def quantize_linear_int8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel int8: data [O, I] + fp16 scale [O]."""
    assert weight.ndim == 2 and weight.dtype == np.float32
    scales = np.abs(weight).max(axis=1, keepdims=True).astype(np.float32)
    scales = np.where(scales == 0, 1.0, scales)
    w_int8 = np.clip(np.round(weight / scales * 127.0), -128, 127).astype(np.int8)
    scale_fp16 = (scales.squeeze() / 127.0).astype(np.float16)
    return w_int8, scale_fp16


def quantize_linear_int4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ANE-native int4 per-channel (packed 2/byte, low nibble first).

    gs=64 *blockwise* scale ``[O, I/64, 1, 1]`` compiles only when I=64
    (one group). At 27B widths it is ``InvalidMILProgram``. Per-channel
    ``[O, 1, 1, 1]`` compiles a full ``[17408, 5120]`` gate on this machine.
    """
    assert weight.ndim == 2 and weight.dtype == np.float32
    out_dim, in_dim = weight.shape
    if in_dim % 2:
        raise ValueError(f"int4 pack needs even input_dim, got {in_dim}")
    scales = np.abs(weight).max(axis=1, keepdims=True).astype(np.float32)
    scales = np.where(scales == 0, 1.0, scales)
    q = np.clip(np.round(weight / scales * 7.0), -8, 7).astype(np.int8)
    packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
    return packed, (scales.squeeze() / 7.0).astype(np.float16)


def resolve_ane_format(quantized: bool = True, weight_format: str | None = None) -> str:
    if weight_format:
        fmt = str(weight_format).strip().lower()
        if fmt in {"int8", "fp16", "int4", "int4_gs64"}:
            if fmt == "int4_gs64":
                return "int4"
            return fmt
        raise ValueError(f"unknown ANE weight_format {weight_format!r}")
    return "int8" if quantized else "fp16"


def pack_procedure_weight_bin(
    weights: list[np.ndarray],
    *,
    quantized: bool = True,
    weight_format: str | None = None,
) -> tuple[bytes, list[tuple[int, int]] | list[int]]:
    """Pack many linear weights into one ``weight.bin`` for procedure banks.

    int8 / int4: ``[(data_off, scale_off), ...]`` per procedure.
    fp16: ``[weight_off, ...]``. Offsets are chunk STARTS; the caller adds 64
    to land on each blob's DEADBEEF header (see generate_procedure_bank_mil).

    Every tensor goes through ``_BlobPacker``, which fixes up the per-chunk
    file-absolute payload pointer. Packing raw ``_make_blob`` output here used
    to alias every procedure onto the first tensor.
    """
    fmt = resolve_ane_format(quantized, weight_format)
    packer = _BlobPacker()
    if fmt == "fp16":
        fp16_offsets: list[int] = []
        for weight in weights:
            o, i = weight.shape
            fp16_offsets.append(
                packer.append(weight.astype(np.float16).reshape(o, i, 1, 1).tobytes())
            )
        return packer.getvalue(), fp16_offsets

    offsets: list[tuple[int, int]] = []
    for weight in weights:
        o, i = weight.shape
        if fmt == "int4":
            packed, scale_fp16 = quantize_linear_int4(weight)
            data_bytes = packed.reshape(o, i // 2, 1, 1).tobytes()
        else:
            packed, scale_fp16 = quantize_linear_int8(weight)
            data_bytes = packed.reshape(o, i, 1, 1).tobytes()
        # constexpr_blockwise_shift_scale rejects a scalar scale const: it has
        # to be a rank-matched [O,1,1,1] tensor, which is what the MIL in
        # generate_procedure_bank_mil declares.
        scale_bytes = scale_fp16.reshape(o, 1, 1, 1).tobytes()
        data_off = packer.append(data_bytes)
        scale_off = packer.append(scale_bytes)
        offsets.append((data_off, scale_off))
    return packer.getvalue(), offsets


def generate_procedure_bank_mil(
    input_dim: int,
    output_dim: int,
    seq_len: int,
    offsets: list[tuple[int, int]] | list[int],
    *,
    quantized: bool = True,
    weight_format: str | None = None,
) -> str:
    """Multi-procedure MIL: ``procedure000`` .. ``procedureNNN`` in one program."""
    fmt = resolve_ane_format(quantized, weight_format)
    blob = "@model_path/weights/weight.bin"
    lines = [
        "program(1.3)",
        _BUILD_INFO,
        "{",
    ]
    for idx, off in enumerate(offsets):
        name = f"procedure{idx:03d}"
        if fmt == "fp16":
            w_off = int(off)
            lines.extend(
                [
                    f"  func {name}<ios18>(tensor<fp16, [1, {input_dim}, 1, {seq_len}]> x) {{",
                    f"    tensor<fp16, [{output_dim}, {input_dim}, 1, 1]> w = const()"
                    f'[name=string("w{idx}"), val=tensor<fp16, [{output_dim}, {input_dim}, 1, 1]>'
                    f'(BLOBFILE(path=string("{blob}"), offset=uint64({w_off + 64})))];',
                ]
            )
        else:
            data_off, scale_off = off  # type: ignore[misc]
            if fmt == "int4":
                data_ty = f"int4, [{output_dim}, {input_dim}, 1, 1]"
                scale_ty = f"fp16, [{output_dim}, 1, 1, 1]"
            else:
                data_ty = f"int8, [{output_dim}, {input_dim}, 1, 1]"
                scale_ty = f"fp16, [{output_dim}, 1, 1, 1]"
            lines.extend(
                [
                    f"  func {name}<ios18>(tensor<fp16, [1, {input_dim}, 1, {seq_len}]> x) {{",
                    f"    tensor<{data_ty}> wd = const()"
                    f'[name=string("wd{idx}"), val=tensor<{data_ty}>'
                    f'(BLOBFILE(path=string("{blob}"), offset=uint64({data_off + 64})))];',
                    f"    tensor<{scale_ty}> ws = const()"
                    f'[name=string("ws{idx}"), val=tensor<{scale_ty}>'
                    f'(BLOBFILE(path=string("{blob}"), offset=uint64({scale_off + 64})))];',
                    f"    tensor<fp16, [{output_dim}, {input_dim}, 1, 1]> w = "
                    f"constexpr_blockwise_shift_scale(data=wd, scale=ws)[name=string(\"dq{idx}\")];",
                ]
            )
        lines.extend(
            [
                '    string pt = const()[name=string("pt"), val=string("valid")];',
                '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];',
                '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];',
                '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];',
                '    int32 gr = const()[name=string("gr"), val=int32(1)];',
                f"    tensor<fp16, [1, {output_dim}, 1, {seq_len}]> y = "
                f"conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)"
                f'[name=string("conv{idx}")];',
                "  } -> (y);",
            ]
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_swiglu_mil(input_dim: int, hidden_dim: int, seq_len: int) -> str:
    """Generate MIL text for a SwiGLU MLP (gate_proj + up_proj + silu + mul + down_proj)."""
    return f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {input_dim}, 1, {seq_len}]> x) {{
    tensor<fp16, [{hidden_dim}, {input_dim}, 1, 1]> gw = const()[name=string("gw"), val=tensor<fp16, [{hidden_dim}, {input_dim}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/gate.bin"), offset=uint64(64)))];
    tensor<fp16, [{hidden_dim}, {input_dim}, 1, 1]> uw = const()[name=string("uw"), val=tensor<fp16, [{hidden_dim}, {input_dim}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/up.bin"), offset=uint64(64)))];
    tensor<fp16, [{input_dim}, {hidden_dim}, 1, 1]> dw = const()[name=string("dw"), val=tensor<fp16, [{input_dim}, {hidden_dim}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/down.bin"), offset=uint64(64)))];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)[name=string("gate")];
    tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> sig = sigmoid(x=gate)[name=string("sigmoid")];
    tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> silu = mul(x=gate, y=sig)[name=string("silu")];
    tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=uw, x=x)[name=string("up")];
    tensor<fp16, [1, {hidden_dim}, 1, {seq_len}]> act = mul(x=silu, y=up)[name=string("swiglu")];
    tensor<fp16, [1, {input_dim}, 1, {seq_len}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=dw, x=act)[name=string("down")];
  }} -> (y);
}}
"""


# ============================================================================
# ANE Engine
# ============================================================================

@dataclass
class AneProgram:
    """A compiled and loaded ANE program for one fixed-shape linear."""
    model: ctypes.c_void_p
    input_dim: int
    output_dim: int
    seq_len: int
    weight_dequant: np.ndarray  # [O, I] fp32 — the matmul ANE actually runs
    num_procedures: int = 1
    _keep_alive: list = field(default_factory=list, repr=False)
    _compile_opts: ctypes.c_void_p = field(default=None, repr=False)
    _in_surf: ctypes.c_void_p = field(default=None, repr=False)
    _out_surf: ctypes.c_void_p = field(default=None, repr=False)
    #: Multi-IO: element counts per input / output symbol, in compiled symbol
    #: order. Empty means the single-surface path. A GDN layer needs 3 in and
    #: 6 out, and _ANERequest already takes NSArrays, so only the allocation
    #: and wrapping were single-tensor.
    input_elems: list = field(default_factory=list, repr=False)
    output_elems: list = field(default_factory=list, repr=False)
    _in_surfs: list = field(default_factory=list, repr=False)
    _out_surfs: list = field(default_factory=list, repr=False)
    _request: ctypes.c_void_p = field(default=None, repr=False)
    _proc_idx: int = field(default=-1, repr=False)


class AneEngine:
    """Independent ANE engine for Apple Silicon.
    
    Usage:
        engine = AneEngine()
        program = engine.compile_linear(weight_fp32, seq_len=1024)
        # ... submit and synchronize
    """

    def __init__(self):
        self._fw = None
        self._available = False
        # Aggregated startup telemetry for the private compiler/loader path.
        # A "compile" API call is not necessarily compiler work: recent ANE
        # builds retain content-addressed compiled artifacts across processes.
        self.compile_metrics = {
            "calls": 0,
            "cache_probe_hits": 0,
            "cache_load_hits": 0,
            "source_bytes": 0,
            "descriptor_seconds": 0.0,
            "cache_probe_seconds": 0.0,
            "materialize_seconds": 0.0,
            "compile_seconds": 0.0,
            "load_seconds": 0.0,
        }
        self._submit_q = None
        self._init_framework()

    def _init_framework(self):
        """Load AppleNeuralEngine.framework and verify required classes."""
        try:
            self._fw = ctypes.cdll.LoadLibrary(
                "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                "AppleNeuralEngine"
            )
            ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/Foundation.framework/Foundation"
            )
        except OSError:
            logger.error("AppleNeuralEngine.framework could not be loaded")
            return

        required = [
            "_ANEInMemoryModelDescriptor",
            "_ANEInMemoryModel",
            "_ANERequest",
            "_ANEIOSurfaceObject",
        ]
        for name in required:
            if not _cls(name):
                logger.error(f"Required class {name} not found")
                return

        self._available = True
        logger.info("ANE engine initialized, framework loaded")

    @property
    def available(self) -> bool:
        return self._available

    def _evaluate_blocking(self, program: "AneProgram",
                           procedure_index: int = 0) -> bool:
        """Synchronous `evaluateWithQoS:` — does not build the request."""
        err_ptr = ctypes.c_void_p(0)
        Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p))
        ok = Eval(("objc_msgSend", _objc))(
            program.model, _sel("evaluateWithQoS:options:request:error:"),
            21, program._compile_opts, program._request, ctypes.byref(err_ptr))
        if not ok:
            logger.error("submit: evaluate FAILED: %s",
                         _desc(err_ptr.value) if err_ptr.value else "unknown")
        return bool(ok)

    def submit(self, program: "AneProgram", procedure_index: int = 0) -> bool:
        """Run a loaded program, optionally selecting a procedure.

        _ensure_io builds the request once with procedureIndex hardcoded to 0.
        A multi-procedure program (one per expert) needs the index chosen per
        call, and _ANERequest takes it at construction, so switching means
        rebuilding the request. The IOSurfaces are reused, so this is object
        churn only -- no weight movement, which is the point of baking.

        Blocks until the ANE finishes. Use `submit_async` to overlap with GPU
        work on another thread.
        """
        if not self._ensure_io(program):
            return False
        if not self._ensure_request(program, procedure_index):
            return False
        return self._evaluate_blocking(program, procedure_index)

    def submit_async(self, program: "AneProgram", procedure_index: int = 0):
        """Queue an evaluate on the ANE worker thread.

        Returns an `AneFuture`. The call itself does not wait. Completion is
        the private `_ANERequest` handler when it attaches, otherwise the
        worker seeing `evaluateWithQoS:` return. MLX must not share this
        thread: `mx.eval` belongs on the caller.
        """
        from runtime.ane_async import AneFuture, AneSubmitQueue

        if not self._ensure_io(program) or not self._ensure_request(
                program, procedure_index):
            fut = AneFuture()
            fut._finish(False)
            return fut
        if self._submit_q is None:
            self._submit_q = AneSubmitQueue(self)
        attach = os.environ.get("FLASHNEXT_ANE_HANDLER", "0") not in (
            "0", "false", "")
        return self._submit_q.submit(program, procedure_index,
                                     attach_handler=attach)

    def compile_multiproc(
        self,
        mil_text: str,
        weights: dict,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        *,
        instance_hint: int = 0,
        raw_weight_files: frozenset[str] | None = None,
    ) -> Optional["AneProgram"]:
        """Compile raw multi-procedure MIL with prebuilt weight blobs.

        compile_linear() owns MIL generation and emits a single ``func main``.
        A baked-expert MoE layer needs many ``func procedureNNN`` in one
        program, selected at submit time by procedureIndex, so the caller
        supplies both the MIL and the weights dict.

        ``weights`` maps blob filename -> bytes; the BLOBFILE paths in the MIL
        must be "@model_path/weights/<name>".

        Pass ``raw_weight_files`` for packed banks (e.g. ``weight.bin``) whose
        bytes already contain per-tensor milinternal blob headers.
        """
        if not self._available:
            return None

        call_started = time.perf_counter()
        raw_weight_files = raw_weight_files or frozenset()
        keep_alive = []
        entries = {}
        for name, data in weights.items():
            if name in raw_weight_files:
                blob = data
            else:
                blob = _make_blob(data)
            keep_alive.append(blob)
            entries[_nsstring(f"@model_path/weights/{name}")] = _nsdict({
                _nsstring("data"): _nsdata(blob),
                _nsstring("offset"): _nsnumber_uint(0),
            })
        weights_dict = _nsdict(entries)

        # Hoist every helper call: _nsdata/_cls/_sel each mutate
        # _objc.objc_msgSend.argtypes for their own signature, so calling them
        # inside the argument list clobbers the 5-arg signature set just above
        # and the dispatch segfaults.
        desc_cls = _cls("_ANEInMemoryModelDescriptor")
        desc_sel = _sel("modelWithMILText:weights:optionsPlist:")
        mil_data = _nsdata(mil_text.encode("utf-8"))

        f = _objc.objc_msgSend
        f.restype = ctypes.c_void_p
        f.argtypes = [ctypes.c_void_p] * 5
        descriptor = f(desc_cls, desc_sel, mil_data, weights_dict, None)
        if not descriptor:
            logger.error("multiproc descriptor creation failed (nil)")
            return None

        model = _msg(_cls("_ANEInMemoryModel"), "inMemoryModelWithDescriptor:",
                     descriptor, argtypes=[ctypes.c_void_p])
        if not model:
            logger.error("multiproc in-memory model creation failed (nil)")
            return None

        descriptor_seconds = time.perf_counter() - call_started

        copts = {}
        if instance_hint > 0:
            copts[_nsstring("kANEFAneInstanceHint")] = _nsnumber_int(instance_hint)
        copts[_nsstring("kANEFProcedureVariantHint")] = _nsnumber_int(1)
        # The ANE wires a loaded model's pages by default, so a large build hits
        # "Program load failure (0x50004)" long before RAM is exhausted (measured
        # ~5x the blob size in wired headroom). Setting this to 0 lets the
        # kernel page model memory out. Q38_ANE_KEEP_WIRED=1 restores the
        # default; unset means unwired.
        _kw = os.environ.get("Q38_ANE_KEEP_WIRED")
        if _kw is not None:
            copts[_nsstring("kANEFKeepModelMemoryWiredKey")] = _nsnumber_int(int(_kw))
        opts_dict = _nsdict(copts)

        # `_ANEInMemoryModel` has a persistent content-addressed compiler
        # cache, but the old path unconditionally rewrote model.mil/weights and
        # invoked the compiler.  On a hit, load the cached artifact directly.
        # If that private behavior changes, the normal materialize+compile path
        # below remains an automatic fallback.
        probe_started = time.perf_counter()
        Exists = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        cache_hit = bool(Exists(("objc_msgSend", _objc))(
            model, _sel("compiledModelExists")))
        probe_seconds = time.perf_counter() - probe_started
        allow_reuse = os.environ.get("Q38_ANE_REUSE_COMPILED", "1") != "0"
        loaded_from_cache = False
        load_seconds = 0.0
        if cache_hit and allow_reuse:
            err_ptr = ctypes.c_void_p(0)
            Load = ctypes.CFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
            load_started = time.perf_counter()
            loaded_from_cache = bool(Load(("objc_msgSend", _objc))(
                model, _sel("loadWithQoS:options:error:"), 21, opts_dict,
                ctypes.byref(err_ptr)))
            load_seconds += time.perf_counter() - load_started
            if not loaded_from_cache:
                logger.info("cached ANE artifact did not load; recompiling: %s",
                            _desc(err_ptr.value) if err_ptr.value else "unknown")

        # Same on-disk materialisation compile_linear relies on: the compiler
        # reads model.mil + weights/ from localModelPath, not from the dict.
        materialize_seconds = compile_seconds = 0.0
        if not loaded_from_cache:
            materialize_started = time.perf_counter()
            local = _desc(_msg(model, "localModelPath"))
            if local and local != "(nil)":
                shutil.rmtree(local, ignore_errors=True)
                wdir = os.path.join(local, "weights")
                os.makedirs(wdir, exist_ok=True)
                with open(os.path.join(local, "model.mil"), "wb") as fh:
                    fh.write(mil_text.encode("utf-8"))
                for name, data in weights.items():
                    payload = data if name in raw_weight_files else _make_blob(data)
                    with open(os.path.join(wdir, name), "wb") as fh:
                        fh.write(payload)
            materialize_seconds = time.perf_counter() - materialize_started

            err_ptr = ctypes.c_void_p(0)
            Compile = ctypes.CFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
            compile_started = time.perf_counter()
            compiled = bool(Compile(("objc_msgSend", _objc))(
                    model, _sel("compileWithQoS:options:error:"), 21, opts_dict,
                    ctypes.byref(err_ptr)))
            compile_seconds = time.perf_counter() - compile_started
            if not compiled:
                logger.error("multiproc compile FAILED: %s",
                             _desc(err_ptr.value) if err_ptr.value else "unknown")
                return None

            err_ptr = ctypes.c_void_p(0)
            Load = ctypes.CFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
            load_started = time.perf_counter()
            loaded = bool(Load(("objc_msgSend", _objc))(
                    model, _sel("loadWithQoS:options:error:"), 21, opts_dict,
                    ctypes.byref(err_ptr)))
            load_seconds += time.perf_counter() - load_started
            if not loaded:
                logger.error("multiproc load FAILED: %s",
                             _desc(err_ptr.value) if err_ptr.value else "unknown")
                return None

        prog = AneProgram(model=model, input_dim=input_dim,
                          output_dim=output_dim, seq_len=seq_len,
                          weight_dequant=np.empty((0, 0), np.float32))
        prog._compile_opts = opts_dict
        # The weight blobs exist to be handed to the compiler. Once the model
        # is loaded they are also on disk under localModelPath, and holding
        # them costs 5 GB of Foundation objects across 48 layers.
        drop = os.environ.get("Q38_ANE_KEEP_WEIGHT_BLOBS") != "1"
        prog._keep_alive = [] if drop else keep_alive
        prog._compile_cache_hit = loaded_from_cache
        prog._compile_timings = {
            "descriptor_seconds": descriptor_seconds,
            "cache_probe_seconds": probe_seconds,
            "materialize_seconds": materialize_seconds,
            "compile_seconds": compile_seconds,
            "load_seconds": load_seconds,
        }
        metrics = self.compile_metrics
        metrics["calls"] += 1
        metrics["cache_probe_hits"] += int(cache_hit)
        metrics["cache_load_hits"] += int(loaded_from_cache)
        metrics["source_bytes"] += sum(len(x) for x in weights.values())
        for key, value in prog._compile_timings.items():
            metrics[key] += value
        return prog

    def compile_procedure_bank(
        self,
        weights: list[np.ndarray],
        seq_len: int,
        *,
        instance_hint: int = 1,
        quantized: bool = True,
        weight_format: str | None = None,
    ) -> list[AneProgram]:
        """Compile ``weights`` into one or more multi-procedure banks.

        All matrices must share ``(output_dim, input_dim)``. On ANE load
        failure the list is bisected recursively so each resident model stays
        under the ~4 GiB mapped-blob window.
        """
        if not self._available or not weights:
            return []
        fmt = resolve_ane_format(quantized, weight_format)

        def _try_one(chunk: list[np.ndarray]) -> Optional[AneProgram]:
            w0 = chunk[0]
            if w0.ndim != 2 or w0.dtype != np.float32:
                raise ValueError(f"expected float32 [O,I], got {w0.shape} {w0.dtype}")
            output_dim, input_dim = w0.shape
            for w in chunk[1:]:
                if w.shape != (output_dim, input_dim):
                    raise ValueError(
                        f"procedure bank shape mismatch {w.shape} != {(output_dim, input_dim)}"
                    )
            weight_bin, offsets = pack_procedure_weight_bin(
                chunk, quantized=quantized, weight_format=fmt
            )
            mil_text = generate_procedure_bank_mil(
                input_dim,
                output_dim,
                seq_len,
                offsets,
                quantized=quantized,
                weight_format=fmt,
            )
            prog = self.compile_multiproc(
                mil_text,
                {"weight.bin": weight_bin},
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                instance_hint=instance_hint,
                raw_weight_files=frozenset({"weight.bin"}),
            )
            if prog is None and instance_hint:
                prog = self.compile_multiproc(
                    mil_text,
                    {"weight.bin": weight_bin},
                    input_dim=input_dim,
                    output_dim=output_dim,
                    seq_len=seq_len,
                    instance_hint=0,
                    raw_weight_files=frozenset({"weight.bin"}),
                )
            if prog is not None:
                prog.num_procedures = len(chunk)
            return prog

        def _compile_recursive(chunk: list[np.ndarray]) -> list[AneProgram]:
            if not chunk:
                return []
            prog = _try_one(chunk)
            if prog is not None:
                return [prog]
            if len(chunk) == 1:
                logger.error("procedure bank failed for single weight %s", chunk[0].shape)
                return []
            mid = len(chunk) // 2
            logger.warning(
                "procedure bank load failed for %d weights; splitting %d + %d",
                len(chunk),
                mid,
                len(chunk) - mid,
            )
            return _compile_recursive(chunk[:mid]) + _compile_recursive(chunk[mid:])

        return _compile_recursive(weights)

    def compile_linear(
        self,
        weight: np.ndarray,
        seq_len: int,
        *,
        instance_hint: int = 0,
        quantized: bool = True,
        keep_weight_dequant: bool = True,
    ) -> Optional[AneProgram]:
        """Compile a linear layer onto the ANE.
        
        Args:
            weight: float32 weight matrix [output_dim, input_dim]
            seq_len: fixed sequence length
            instance_hint: ANE instance hint (1-4, or 0 for unpinned)
            quantized: if True, re-quantize to int8 per-channel
                       if False, convert to fp16 directly
            keep_weight_dequant: if False, drop the fp32 reference copy after
                compile (hybrid banks with dozens of programs would otherwise
                pin gigabytes of host RAM). Default True preserves the engine
                self-test path.
        
        Returns:
            AneProgram or None on failure
        """
        if not self._available:
            return None

        assert weight.ndim == 2, f"Weight must be rank-2, got {weight.ndim}"
        assert weight.dtype == np.float32, f"Weight must be float32, got {weight.dtype}"

        output_dim, input_dim = weight.shape
        keep_alive = []

        # Generate MIL
        mil_text = generate_linear_mil(input_dim, output_dim, seq_len,
                                       quantized=quantized)

        # Prepare weight blobs
        if quantized:
            # Per-channel int8 quantisation (matching §11 capture)
            scales = np.abs(weight).max(axis=1, keepdims=True).astype(np.float32)
            scales = np.where(scales == 0, 1.0, scales)  # avoid div by zero
            w_int8 = np.clip(np.round(weight / scales * 127.0), -128, 127).astype(np.int8)
            # Scale stored as fp16 per output channel
            scale_fp16 = (scales.squeeze() / 127.0).astype(np.float16)
            weight_dequant = w_int8.astype(np.float32) * scale_fp16.astype(np.float32)[:, None]

            # Reshape for ANE [O, I, 1, 1] layout
            w_data = w_int8.reshape(output_dim, input_dim, 1, 1).tobytes()
            s_data = scale_fp16.reshape(output_dim, 1, 1, 1).tobytes()

            weight_data_blob = _make_blob(w_data)
            weight_scale_blob = _make_blob(s_data)
            keep_alive.extend([weight_data_blob, weight_scale_blob])

            weights_dict = _nsdict({
                _nsstring("@model_path/weights/weight_data.bin"): _nsdict({
                    _nsstring("data"): _nsdata(weight_data_blob),
                    _nsstring("offset"): _nsnumber_uint(0),
                }),
                _nsstring("@model_path/weights/weight_scale.bin"): _nsdict({
                    _nsstring("data"): _nsdata(weight_scale_blob),
                    _nsstring("offset"): _nsnumber_uint(0),
                }),
            })
        else:
            # Direct fp16
            w_fp16 = weight.astype(np.float16)
            w_data = w_fp16.reshape(output_dim, input_dim, 1, 1).tobytes()
            weight_data_blob = _make_blob(w_data)
            keep_alive.append(weight_data_blob)
            weight_dequant = w_fp16.astype(np.float32)

            weights_dict = _nsdict({
                _nsstring("@model_path/weights/weight_data.bin"): _nsdict({
                    _nsstring("data"): _nsdata(weight_data_blob),
                    _nsstring("offset"): _nsnumber_uint(0),
                }),
            })

        mil_data = _nsdata(mil_text.encode("utf-8"))
        # Pass nil, not empty NSData. Empty plist trips InvalidCompilationParam.
        opts_data = None

        # Create descriptor
        desc_cls = _cls("_ANEInMemoryModelDescriptor")
        f = _objc.objc_msgSend
        f.restype = ctypes.c_void_p
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        descriptor = f(desc_cls,
                       _sel("modelWithMILText:weights:optionsPlist:"),
                       mil_data, weights_dict, opts_data)
        if not descriptor:
            logger.error("ANE descriptor creation failed (nil)")
            return None
        logger.info(f"Descriptor created: {_desc(descriptor)[:200]}")

        # Create in-memory model
        model_cls = _cls("_ANEInMemoryModel")
        model = _msg(model_cls, "inMemoryModelWithDescriptor:",
                     descriptor,
                     argtypes=[ctypes.c_void_p])
        if not model:
            logger.error("ANE in-memory model creation failed (nil)")
            return None
        logger.info(f"Model created: {_desc(model)[:200]}")

        ident = _desc(_msg(model, "hexStringIdentifier"))
        local = _desc(_msg(model, "localModelPath"))
        logger.info("ident=%s local=%s", ident[:120], local[:200])

        # Write model.mil + weights/*.bin into the hashed temp dir
        # *before* compileWithQoS. saveModelFiles is the espresso path
        # (net.plist + data) and produces InvalidMILProgram for MIL models.
        if local and local != "(nil)":
            # Delete the hashed dir first so espresso leftovers cannot
            # shadow model.mil, then write model.mil + weights/*.bin.
            shutil.rmtree(local, ignore_errors=True)
            wdir = os.path.join(local, "weights")
            os.makedirs(wdir, exist_ok=True)
            with open(os.path.join(local, "model.mil"), "wb") as f:
                f.write(mil_text.encode("utf-8"))
            with open(os.path.join(wdir, "weight_data.bin"), "wb") as f:
                f.write(weight_data_blob)
            if quantized:
                with open(os.path.join(wdir, "weight_scale.bin"), "wb") as f:
                    f.write(weight_scale_blob)
            logger.info("pre-wrote model.mil + weights/ under %s", local)

        # Build compile options
        compile_opts = {}
        if instance_hint > 0:
            compile_opts[_nsstring("kANEFAneInstanceHint")] = _nsnumber_int(instance_hint)
        compile_opts[_nsstring("kANEFProcedureVariantHint")] = _nsnumber_int(1)
        opts_dict = _nsdict(compile_opts) if compile_opts else _msg(_cls("NSDictionary"), "dictionary")

        # Compile. QoS is unsigned int (encoding I), 21 = QOS_CLASS_USER_INITIATED.
        err_ptr = ctypes.c_void_p(0)
        Compile = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        ok = Compile(("objc_msgSend", _objc))(
            model, _sel("compileWithQoS:options:error:"),
            21, opts_dict, ctypes.byref(err_ptr))
        if not ok:
            err_desc = _desc(err_ptr.value) if err_ptr.value else "unknown"
            logger.error(f"ANE compilation FAILED: {err_desc}")
            return None
        logger.info("ANE compilation succeeded!")

        # Load
        err_ptr = ctypes.c_void_p(0)
        Load = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        ok = Load(("objc_msgSend", _objc))(
            model, _sel("loadWithQoS:options:error:"),
            21, opts_dict, ctypes.byref(err_ptr))
        if not ok:
            err_desc = _desc(err_ptr.value) if err_ptr.value else "unknown"
            logger.error(f"ANE load FAILED: {err_desc}")
            return None
        logger.info("ANE model loaded!")

        if not keep_weight_dequant:
            weight_dequant = np.empty((0, 0), dtype=np.float32)

        return AneProgram(
            model=model,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            weight_dequant=weight_dequant,
            _keep_alive=keep_alive,
            _compile_opts=opts_dict,
        )

    def _ensure_io(self, program: AneProgram) -> bool:
        """Allocate IOSurfaces once per program.

        With ``input_elems`` / ``output_elems`` set, allocates one surface per
        symbol instead of a single in/out pair.
        """
        if program.input_elems or program.output_elems:
            if program._in_surfs and program._out_surfs:
                return True
            _load_iosurface()
            ins, outs = [], []
            for n in (program.input_elems or [program.input_dim * program.seq_len]):
                surf = _create_iosurface(_iosurface_alloc_size(int(n)))
                if not surf:
                    logger.error("multi-IO input surface alloc failed (%d elems)", n)
                    return False
                ins.append(surf)
            for n in (program.output_elems or [program.output_dim * program.seq_len]):
                surf = _create_iosurface(_iosurface_alloc_size(int(n)))
                if not surf:
                    logger.error("multi-IO output surface alloc failed (%d elems)", n)
                    return False
                outs.append(surf)
            program._in_surfs, program._out_surfs = ins, outs
            program._in_surf, program._out_surf = ins[0], outs[0]
            program._keep_alive.extend(ins + outs)
            return True
        if program._in_surf and program._out_surf:
            return True
        _load_iosurface()
        in_bytes = _iosurface_alloc_size(program.input_dim * program.seq_len)
        out_bytes = _iosurface_alloc_size(program.output_dim * program.seq_len)
        in_surf = _create_iosurface(in_bytes)
        out_surf = _create_iosurface(out_bytes)
        if not in_surf or not out_surf:
            logger.error("IOSurface allocation failed (%s, %s)", in_surf, out_surf)
            return False
        program._in_surf = in_surf
        program._out_surf = out_surf
        program._keep_alive.extend([in_surf, out_surf])
        return True

    def _ensure_request(self, program: AneProgram, procedure_index: int) -> bool:
        """Build or rebuild ``_ANERequest`` for ``procedure_index``."""
        if not self._ensure_io(program):
            return False
        if program._request is not None and program._proc_idx == int(procedure_index):
            return True

        surf_cls = _cls("_ANEIOSurfaceObject")
        init_sel = _sel("initWithIOSurface:startOffset:shouldRetain:")
        off0 = _nsnumber_int(0)
        InitSurf = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)

        def wrap(surf):
            raw = _msg(surf_cls, "alloc")
            return InitSurf(("objc_msgSend", _objc))(raw, init_sel, surf, off0, True)

        in_list = program._in_surfs or [program._in_surf]
        out_list = program._out_surfs or [program._out_surf]
        # Procedures in one program can declare different numbers of outputs.
        # The request pairs surfaces with the procedure's symbol indices by
        # position, so a procedure with fewer outputs has to be handed the
        # surfaces it actually writes, not the front of the list.
        proc_map = getattr(program, "proc_out_map", None)
        if proc_map and int(procedure_index) in proc_map:
            out_list = [out_list[i] for i in proc_map[int(procedure_index)]]
        proc_in = getattr(program, "proc_in_map", None)
        if proc_in and int(procedure_index) in proc_in:
            in_list = [in_list[i] for i in proc_in[int(procedure_index)]]
        in_objs = [wrap(s_) for s_ in in_list]
        out_objs = [wrap(s_) for s_ in out_list]
        if not all(in_objs) or not all(out_objs):
            logger.error("IOSurface wrap failed")
            return False
        in_obj, out_obj = in_objs[0], out_objs[0]

        inner = _msg(program.model, "model") or program.model
        SymSel = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong
        )
        send_sym = SymSel(("objc_msgSend", _objc))
        in_idx = send_sym(inner, _sel("inputSymbolIndicesForProcedureIndex:"), int(procedure_index))
        out_idx = send_sym(inner, _sel("outputSymbolIndicesForProcedureIndex:"), int(procedure_index))
        in_idx = _indexset_to_nsarray(in_idx)
        out_idx = _indexset_to_nsarray(out_idx)
        if not in_idx or not out_idx:
            in_idx = out_idx = _nsarray([_nsnumber_int(0)])

        proc_num = _nsnumber_int(int(procedure_index))
        req_cls = _cls("_ANERequest")
        raw = _msg(req_cls, "alloc")
        InitReq = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
        request = InitReq(("objc_msgSend", _objc))(
            raw,
            _sel(
                "initWithInputs:inputIndices:outputs:outputIndices:"
                "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                "transactionHandle:"
            ),
            _nsarray(in_objs),
            in_idx,
            _nsarray(out_objs),
            out_idx,
            None,
            None,
            proc_num,
            None,
            None,
        )
        if not request:
            logger.error("request creation failed (procedure=%d)", procedure_index)
            return False
        program._request = request
        program._proc_idx = int(procedure_index)
        program._keep_alive.extend(
            in_objs + out_objs + [in_idx, out_idx, proc_num, request])
        return True

    def evaluate(
        self,
        program: AneProgram,
        x: np.ndarray,
        *,
        procedure_index: int = 0,
        planar: bool = False,
        as_float32: bool = True,
    ) -> Optional[np.ndarray]:
        """Run the compiled linear: x is [seq, in] fp32/fp16, returns [seq, out] fp32.

        ANE layout is planar CHW fp16: [C, S] contiguous. This does not touch
        q38's generate path.
        """
        if not self._ensure_request(program, procedure_index):
            return None
        want = ((program.input_dim, program.seq_len) if planar
                else (program.seq_len, program.input_dim))
        if x.shape != want:
            raise ValueError(f"x shape {x.shape} != {want} (planar={planar})")
        # ANE wants planar [C, S]. Write the transpose straight into the
        # surface: one strided copy, no intermediate allocations.
        with _iosurface_view(
            program._in_surf, (program.input_dim, program.seq_len), np.float16
        ) as dst:
            if planar:
                _copyto_fp16(dst, x)
            else:
                _copyto_fp16(dst, x.T)

        err_ptr = ctypes.c_void_p(0)
        Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        ok = Eval(("objc_msgSend", _objc))(
            program.model,
            _sel("evaluateWithQoS:options:request:error:"),
            21,
            program._compile_opts,
            program._request,
            ctypes.byref(err_ptr),
        )
        if not ok:
            logger.error(
                "ANE evaluate FAILED: %s",
                _desc(err_ptr.value) if err_ptr.value else "unknown",
            )
            return None

        with _iosurface_view(
            program._out_surf, (program.output_dim, program.seq_len), np.float16
        ) as src:
            if planar:
                # Caller wants ANE layout and fp16: single flat copy.
                return src.copy()
            # [C, S] -> [S, C]; one strided copy, and fp16 is kept unless the
            # caller asks otherwise (astype(float32) used to double the data).
            out = np.empty((program.seq_len, program.output_dim), np.float16)
            np.copyto(out, src.T)
            return out.astype(np.float32) if as_float32 else out


def _nsarray_strings(arr) -> list[str]:
    if not arr:
        return []
    n = int(_objc_call(ctypes.c_ulonglong, (), arr, "count") or 0)
    out = []
    for i in range(n):
        obj = _objc_call(
            ctypes.c_void_p, [ctypes.c_ulonglong], arr, "objectAtIndex:", i
        )
        cs = (
            _objc_call(ctypes.c_char_p, (), obj, "UTF8String") if obj else None
        )
        out.append(cs.decode() if cs else "")
    return out


def _ane_input_symbols(model) -> list[str]:
    """Compiled live-input names, in the order ANE binds IO indices."""
    attrs = _objc_call(ctypes.c_void_p, (), model, "modelAttributes")
    if not attrs:
        return []
    desc = _objc_call(
        ctypes.c_void_p,
        [ctypes.c_void_p],
        attrs,
        "objectForKey:",
        _nsstring("ANEFModelDescription"),
    )
    if not desc:
        return []
    arr = _objc_call(
        ctypes.c_void_p,
        [ctypes.c_void_p],
        desc,
        "objectForKey:",
        _nsstring("kANEFModelInputSymbolsArrayKey"),
    )
    return _nsarray_strings(arr)


def _wrap_iosurface(surf) -> ctypes.c_void_p:
    raw = _objc_call(ctypes.c_void_p, (), _cls("_ANEIOSurfaceObject"), "alloc")
    return _objc_call(
        ctypes.c_void_p,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool],
        raw,
        "initWithIOSurface:startOffset:shouldRetain:",
        _as_void_p(surf),
        _nsnumber_int(0),
        True,
    )


@dataclass
class AneDynamicLinear:
    """One compiled ANE 1×1-conv with a live weight IOSurface.

    Compile once per ``(input_dim, output_dim, seq_len)``. Page a new
    ``[O, I]`` matrix into the weight surface and ``evaluate`` without
    recompiling. Does not touch q38 generate.
    """
    model: ctypes.c_void_p
    input_dim: int
    output_dim: int
    seq_len: int
    input_symbols: list[str]
    _compile_opts: ctypes.c_void_p = field(default=None, repr=False)
    _x_surf: ctypes.c_void_p = field(default=None, repr=False)
    _w_surf: ctypes.c_void_p = field(default=None, repr=False)
    _y_surf: ctypes.c_void_p = field(default=None, repr=False)
    _request: ctypes.c_void_p = field(default=None, repr=False)
    _keep_alive: list = field(default_factory=list, repr=False)

    @classmethod
    def compile(
        cls,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        *,
        instance_hint: int = 0,
    ) -> Optional["AneDynamicLinear"]:
        engine = AneEngine()
        if not engine.available:
            return None

        mil_text = generate_dynamic_linear_mil(input_dim, output_dim, seq_len)
        mil_data = _nsdata(mil_text.encode("utf-8"))
        empty = _msg(_cls("NSDictionary"), "dictionary")
        Make = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        )
        descriptor = Make(("objc_msgSend", _objc))(
            _cls("_ANEInMemoryModelDescriptor"),
            _sel("modelWithMILText:weights:optionsPlist:"),
            mil_data, empty, None,
        )
        if not descriptor:
            logger.error("ANE dynamic descriptor creation failed")
            return None

        InMem = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )
        model = InMem(("objc_msgSend", _objc))(
            _cls("_ANEInMemoryModel"),
            _sel("inMemoryModelWithDescriptor:"),
            descriptor,
        )
        if not model:
            logger.error("ANE dynamic in-memory model creation failed")
            return None

        local = _desc(_msg(model, "localModelPath"))
        if local and local != "(nil)":
            shutil.rmtree(local, ignore_errors=True)
            os.makedirs(os.path.join(local, "weights"), exist_ok=True)
            with open(os.path.join(local, "model.mil"), "wb") as f:
                f.write(mil_text.encode("utf-8"))

        compile_opts: dict = {
            _nsstring("kANEFProcedureVariantHint"): _nsnumber_int(1),
        }
        if instance_hint > 0:
            compile_opts[_nsstring("kANEFAneInstanceHint")] = _nsnumber_int(
                instance_hint
            )
        opts_dict = _nsdict(compile_opts)

        err_ptr = ctypes.c_void_p(0)
        Compile = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        ok = Compile(("objc_msgSend", _objc))(
            model, _sel("compileWithQoS:options:error:"),
            21, opts_dict, ctypes.byref(err_ptr),
        )
        if not ok:
            logger.error(
                "ANE dynamic compile FAILED: %s",
                _desc(err_ptr.value) if err_ptr.value else "unknown",
            )
            return None

        err_ptr = ctypes.c_void_p(0)
        Load = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        ok = Load(("objc_msgSend", _objc))(
            model, _sel("loadWithQoS:options:error:"),
            21, opts_dict, ctypes.byref(err_ptr),
        )
        if not ok:
            logger.error(
                "ANE dynamic load FAILED: %s",
                _desc(err_ptr.value) if err_ptr.value else "unknown",
            )
            return None

        symbols = _ane_input_symbols(model)
        logger.info("ANE dynamic linear loaded symbols=%s", symbols)
        prog = cls(
            model=model,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            input_symbols=symbols,
            _compile_opts=opts_dict,
            _keep_alive=[mil_data, empty, descriptor, opts_dict],
        )
        if not prog._ensure_io():
            return None
        return prog

    def _ensure_io(self) -> bool:
        if self._request:
            return True
        _load_iosurface()
        x_surf = _create_iosurface(
            _iosurface_alloc_size(self.input_dim * self.seq_len)
        )
        w_surf = _create_iosurface(
            _iosurface_alloc_size(self.output_dim * self.input_dim)
        )
        y_surf = _create_iosurface(
            _iosurface_alloc_size(self.output_dim * self.seq_len)
        )
        if not x_surf or not w_surf or not y_surf:
            logger.error("ANE dynamic IOSurface allocation failed")
            return False

        x_obj = _wrap_iosurface(x_surf)
        w_obj = _wrap_iosurface(w_surf)
        y_obj = _wrap_iosurface(y_surf)
        by_name = {"x": x_obj, "wimg": w_obj}
        missing = [s for s in self.input_symbols if s not in by_name]
        if missing:
            logger.error("ANE dynamic unexpected input symbols %s", self.input_symbols)
            return False
        in_objs = [by_name[s] for s in self.input_symbols]
        in_idx = _nsarray([_nsnumber_int(i) for i in range(len(in_objs))])
        out_idx = _nsarray([_nsnumber_int(0)])

        InitReq = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
        )
        raw = _msg(_cls("_ANERequest"), "alloc")
        request = InitReq(("objc_msgSend", _objc))(
            raw,
            _sel(
                "initWithInputs:inputIndices:outputs:outputIndices:"
                "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                "transactionHandle:"
            ),
            _nsarray(in_objs),
            in_idx,
            _nsarray([y_obj]),
            out_idx,
            None, None, _nsnumber_int(0), None, None,
        )
        if not request:
            logger.error("ANE dynamic request creation failed")
            return False
        self._x_surf = x_surf
        self._w_surf = w_surf
        self._y_surf = y_surf
        self._request = request
        self._keep_alive.extend(
            [x_obj, w_obj, y_obj, in_idx, out_idx, request]
        )
        return True

    def write_weight(self, weight: np.ndarray) -> None:
        """Pack ``weight`` [O, I] into the live wimg surface (planar [O, I] fp16)."""
        if weight.shape != (self.output_dim, self.input_dim):
            raise ValueError(
                f"weight shape {weight.shape} != "
                f"({self.output_dim}, {self.input_dim})"
            )
        # One in-place strided copy instead of astype + ascontiguousarray +
        # tobytes + memmove.
        with _iosurface_view(
            self._w_surf, (self.output_dim, self.input_dim), np.float16
        ) as dst:
            _copyto_fp16(dst, weight)

    def submit(self) -> bool:
        """Run the program with the input surface as-is; no staging, no read-back.

        For chaining: layer N's output surface is memcpy'd into layer N+1's
        input surface, so activations never round-trip through numpy. evaluate()
        always writes the input, which makes that impossible to measure.
        """
        err_ptr = ctypes.c_void_p(0)
        Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        ok = Eval(("objc_msgSend", _objc))(
            self.model,
            _sel("evaluateWithQoS:options:request:error:"),
            21,
            self._compile_opts,
            self._request,
            ctypes.byref(err_ptr),
        )
        if not ok:
            logger.error("ANE submit FAILED: %s",
                         _desc(err_ptr.value) if err_ptr.value else "unknown")
        return bool(ok)

    def evaluate(
        self,
        x: np.ndarray,
        weight: np.ndarray | None = None,
        *,
        planar: bool = False,
        as_float32: bool = True,
    ) -> Optional[np.ndarray]:
        """Run the compiled linear. ``x`` is [S, I]; ``weight`` is [O, I] or None.

        If ``weight`` is omitted, the current weight IOSurface is reused
        (page into it with ``write_weight`` first).
        """
        want = (self.input_dim, self.seq_len) if planar else (self.seq_len, self.input_dim)
        if x.shape != want:
            raise ValueError(
                f"x shape {x.shape} != {want} (planar={planar})"
            )
        if weight is not None:
            self.write_weight(weight)
        # Write the transpose straight into the surface: one strided copy,
        # no astype/ascontiguousarray/tobytes chain.
        with _iosurface_view(
            self._x_surf, (self.input_dim, self.seq_len), np.float16
        ) as dst:
            _copyto_fp16(dst, x if planar else x.T)

        err_ptr = ctypes.c_void_p(0)
        Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        ok = Eval(("objc_msgSend", _objc))(
            self.model,
            _sel("evaluateWithQoS:options:request:error:"),
            21,
            self._compile_opts,
            self._request,
            ctypes.byref(err_ptr),
        )
        if not ok:
            logger.error(
                "ANE dynamic evaluate FAILED: %s",
                _desc(err_ptr.value) if err_ptr.value else "unknown",
            )
            return None
        with _iosurface_view(
            self._y_surf, (self.output_dim, self.seq_len), np.float16
        ) as src:
            if planar:
                return src.copy()
            out = np.empty((self.seq_len, self.output_dim), np.float16)
            np.copyto(out, src.T)
            return out.astype(np.float32) if as_float32 else out


# ============================================================================
# Test harness
# ============================================================================

def main():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    engine = AneEngine()
    if not engine.available:
        print("ANE engine not available")
        return

    # Test with a small linear layer (128x64, matching the captured MIL)
    print("\n=== Test 1: Small linear (128x64, S=1024) ===")
    W = np.random.randn(128, 64).astype(np.float32)
    prog = engine.compile_linear(W, seq_len=1024, instance_hint=1, quantized=True)
    print(f"Result: {'SUCCESS' if prog else 'FAILED'}")

    if not prog:
        # Try without quantization
        print("\n=== Test 2: Small linear fp16 (128x64, S=1024) ===")
        prog = engine.compile_linear(W, seq_len=1024, instance_hint=1, quantized=False)
        print(f"Result: {'SUCCESS' if prog else 'FAILED'}")

    if not prog:
        # Try without instance hint
        print("\n=== Test 3: No instance hint ===")
        prog = engine.compile_linear(W, seq_len=1024, instance_hint=0, quantized=True)
        print(f"Result: {'SUCCESS' if prog else 'FAILED'}")

    if prog:
        print(f"\nCompiled program: {prog.input_dim}x{prog.output_dim}, S={prog.seq_len}")
        rng = np.random.default_rng(0)
        x = rng.standard_normal((prog.seq_len, prog.input_dim)).astype(np.float32)
        y = engine.evaluate(prog, x)
        if y is None:
            print("Evaluate: FAILED")
            return
        y_ref = x @ prog.weight_dequant.T
        err = np.max(np.abs(y - y_ref))
        rel = err / (np.max(np.abs(y_ref)) + 1e-6)
        print(f"Evaluate: y{tuple(y.shape)} max_abs_err={err:.5f} rel={rel:.5f}")
        print("OK" if rel < 5e-2 else "MISMATCH (still compiled; check layout)")


if __name__ == "__main__":
    main()
