#!/usr/bin/env python3
"""Standalone, GPU-free Qwen3.8 inference through AppleNeuralEngine.framework.

This module intentionally does not import MLX, oMLX, PyTorch, Core ML, or
Transformers.  The checkpoint reader memory-maps safetensors directly; the
tokenizer uses the small Rust ``tokenizers`` package; model arithmetic is
submitted to the ANE through the project's private-framework driver.

The complete 64-layer fp16 decode loop is implemented.  Smaller smoke commands
remain available to validate each real-shape component independently.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import io
import json
import os
import re
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

try:
    from compression import zstd as _zstd
except ImportError:  # Python <3.14: retain the uncompressed cache format.
    _zstd = None


FORBIDDEN_COMPUTE_MODULES = ("mlx", "torch", "coremltools")

# The ANE driver (``runtime.q38_ane_engine``) is vendored at the repository
# root.  ``Q38_ANE_ENGINE`` points at another checkout instead.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def assert_standalone(where: str = "runtime") -> None:
    """Fail closed if a GPU-capable model framework entered this process."""
    loaded = sorted(
        name for name in sys.modules
        if name == FORBIDDEN_COMPUTE_MODULES
        or name.startswith(tuple(f"{m}." for m in FORBIDDEN_COMPUTE_MODULES))
    )
    if loaded:
        roots = sorted({name.split(".", 1)[0] for name in loaded})
        raise RuntimeError(
            f"pure ANE invariant failed at {where}: loaded {', '.join(roots)}"
        )


@dataclass(frozen=True)
class TensorInfo:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


class SafeTensorFile:
    """Minimal read-only safetensors reader, including native BF16 support.

    NumPy has no portable BF16 dtype.  Reading the raw uint16 payload and
    placing its bits in the high half of a uint32 converts BF16 to float32
    exactly, with no ML framework involved.
    """

    _DTYPES = {
        "F16": (np.dtype("<f2"), 2),
        "F32": (np.dtype("<f4"), 4),
        "BF16": (np.dtype("<u2"), 2),
        "I8": (np.dtype("i1"), 1),
        "U8": (np.dtype("u1"), 1),
        "I16": (np.dtype("<i2"), 2),
        "I32": (np.dtype("<i4"), 4),
        "I64": (np.dtype("<i8"), 8),
    }

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        with self.path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors header: {self.path}")
            header_len = struct.unpack("<Q", raw)[0]
            header = json.loads(f.read(header_len))
        self.data_offset = 8 + header_len
        self.header = {k: v for k, v in header.items() if k != "__metadata__"}

    def info(self, name: str) -> TensorInfo:
        meta = self.header[name]
        dtype = meta["dtype"]
        if dtype not in self._DTYPES:
            raise TypeError(f"unsupported safetensors dtype {dtype} for {name}")
        a, b = (int(x) for x in meta["data_offsets"])
        shape = tuple(int(x) for x in meta["shape"])
        expected = int(np.prod(shape, dtype=np.int64)) * self._DTYPES[dtype][1]
        if b - a != expected:
            raise ValueError(
                f"invalid byte extent for {name}: got {b-a}, expected {expected}"
            )
        return TensorInfo(name, self.path, dtype, shape,
                          self.data_offset + a, b - a)

    def mmap(self, name: str) -> np.memmap:
        info = self.info(name)
        dtype = self._DTYPES[info.dtype][0]
        return np.memmap(info.path, mode="r", dtype=dtype,
                         offset=info.offset, shape=info.shape, order="C")

    def array(self, name: str, dtype=np.float32) -> np.ndarray:
        info = self.info(name)
        raw = self.mmap(name)
        if info.dtype == "BF16":
            u32 = raw.astype(np.uint32) << np.uint32(16)
            out = u32.view(np.float32)
        else:
            out = np.asarray(raw)
        return np.asarray(out, dtype=dtype)

    def row(self, name: str, index: int, dtype=np.float16) -> np.ndarray:
        info = self.info(name)
        if len(info.shape) != 2:
            raise ValueError(f"row lookup requires a matrix, got {info.shape}")
        if not 0 <= index < info.shape[0]:
            raise IndexError(index)
        raw = self.mmap(name)[index]
        if info.dtype == "BF16":
            out = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
        else:
            out = np.asarray(raw)
        return np.asarray(out, dtype=dtype)


class Checkpoint:
    """Indexed, lazy Qwen checkpoint with no model-framework dependency."""

    EMBEDDING_NAMES = (
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "model.word_embeddings.weight",
    )
    SHIFTED_NORM_SUFFIXES = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )

    def __init__(self, model_dir: str | os.PathLike[str], *,
                 shifted_norms: bool = True):
        # Qwen3.5/3.8 store RMSNorm weights as deltas from one; architectures
        # with plain norm weights (BailingMoeV3) must pass shifted_norms=False.
        self.shifted_norms = shifted_norms
        self.path = Path(model_dir).expanduser().resolve()
        with (self.path / "config.json").open() as f:
            self.config = json.load(f)
        index_path = self.path / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open() as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            one = self.path / "model.safetensors"
            if not one.exists():
                raise FileNotFoundError("no safetensors checkpoint found")
            sf = SafeTensorFile(one)
            self.weight_map = {name: one.name for name in sf.header}
        self._files: dict[str, SafeTensorFile] = {}
        self.quant_cache_dir: Path | None = None
        self.quant_cache_stats = {
            "hits": 0, "misses": 0, "bytes_read": 0, "bytes_written": 0,
            "disk_bytes_read": 0, "disk_bytes_written": 0,
            "read_seconds": 0.0, "quantize_seconds": 0.0,
            "compress_seconds": 0.0, "write_seconds": 0.0,
        }
        self.embedding_name = next(
            (n for n in self.EMBEDDING_NAMES if n in self.weight_map), None
        )
        if self.embedding_name is None:
            candidates = [n for n in self.weight_map if n.endswith("embed_tokens.weight")]
            if len(candidates) != 1:
                raise KeyError(f"could not identify embedding tensor: {candidates}")
            self.embedding_name = candidates[0]

    def configure_quant_cache(self, root: str | os.PathLike[str] | None) -> None:
        """Select a versioned persistent cache for prebaked ANE weight blobs."""
        if not root:
            self.quant_cache_dir = None
            return
        index = self.path / "model.safetensors.index.json"
        identity = hashlib.sha256()
        identity.update(b"q38-pure-ane-quant-cache-v2-zstd\0")
        identity.update(str(self.path).encode())
        for path in (self.path / "config.json", index):
            if path.exists():
                stat = path.stat()
                identity.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        self.quant_cache_dir = (
            Path(root).expanduser().resolve() / identity.hexdigest()[:20]
        )

    def _file(self, name: str) -> SafeTensorFile:
        filename = self.weight_map[name]
        if filename not in self._files:
            self._files[filename] = SafeTensorFile(self.path / filename)
        return self._files[filename]

    def info(self, name: str) -> TensorInfo:
        return self._file(name).info(name)

    def tensor(self, name: str, dtype=np.float32) -> np.ndarray:
        # Qwen3.5/3.8 stores standard RMSNorm weights as deltas from one.
        # Framework loaders call this "sanitize", add the implicit one while
        # the tensor is still BF16, and therefore round the sum back to BF16.
        # Reproducing only the addition in fp32/fp16 changes every decoder norm
        # by as much as half a BF16 ulp and compounds noticeably over 64 layers.
        # GDN's `.linear_attn.norm.weight` is deliberately not in this list.
        mtp_norm = (
            name.startswith("mtp.") and "norm" in name.lower()
            and len(self.info(name).shape) == 1
        )
        if self.shifted_norms and (name.endswith(self.SHIFTED_NORM_SUFFIXES)
                                   or mtp_norm):
            out = self._file(name).array(name, np.float32)
            out = np.asarray(out + np.float32(1.0), np.float32)
            # IEEE round-to-nearest-even float32 -> bfloat16 -> float32.
            words = out.view(np.uint32)
            bias = np.uint32(0x7fff) + ((words >> np.uint32(16)) & np.uint32(1))
            out = ((words + bias) & np.uint32(0xffff0000)).view(np.float32)
            return np.asarray(out, dtype=dtype)
        return self._file(name).array(name, dtype)

    def tensor_row_blocks(self, name: str, rows: int = 512, *,
                          start: int = 0, end: int | None = None) -> Iterator[np.ndarray]:
        """Yield float32 row blocks without materializing a whole matrix."""
        sf = self._file(name)
        info = sf.info(name)
        if len(info.shape) != 2:
            raise ValueError(f"row blocks require a matrix, got {info.shape}")
        end = info.shape[0] if end is None else end
        if not 0 <= start <= end <= info.shape[0]:
            raise ValueError(f"invalid row range {start}:{end} for {info.shape}")
        raw = sf.mmap(name)
        for a in range(start, end, rows):
            block = raw[a:min(a + rows, end)]
            if info.dtype == "BF16":
                block = (block.astype(np.uint32) << np.uint32(16)).view(np.float32)
            yield np.asarray(block, np.float32)

    def embedding(self, token_id: int) -> np.ndarray:
        return self._file(self.embedding_name).row(
            self.embedding_name, token_id, np.float16
        )

    def names(self, prefix: str = "") -> Iterator[str]:
        yield from sorted(n for n in self.weight_map if n.startswith(prefix))

    @property
    def architecture(self) -> dict[str, int | float | str]:
        c = self.config
        text = c.get("text_config", c)
        layer_types = text.get("layer_types", [])
        return {
            "model_type": text.get("model_type", c.get("model_type", "unknown")),
            "hidden_size": int(text["hidden_size"]),
            "intermediate_size": int(text["intermediate_size"]),
            "num_hidden_layers": int(text["num_hidden_layers"]),
            "vocab_size": int(text["vocab_size"]),
            "full_attention_layers": sum(x == "full_attention" for x in layer_types),
            "linear_attention_layers": sum(x != "full_attention" for x in layer_types),
            "rms_norm_eps": float(text.get("rms_norm_eps", 1e-6)),
        }


class StandaloneTokenizer:
    """Tokenizer-only dependency; it cannot schedule CPU/GPU tensor work."""

    def __init__(self, model_dir: str | os.PathLike[str]):
        from tokenizers import Tokenizer
        self.impl = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return self.impl.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.impl.decode(ids, skip_special_tokens=False)


def _dense_decl(name: str, out_dim: int, in_dim: int, bits: int) -> str:
    if bits == 16:
        return (
            f'    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> {name}w = const()'
            f'[name=string("{name}w"), val=tensor<fp16, '
            f'[{out_dim}, {in_dim}, 1, 1]>(BLOBFILE(path=string('
            f'"@model_path/weights/{name}.bin"), offset=uint64(64)))];'
        )
    return f'''    tensor<int{bits}, [{out_dim}, {in_dim}, 1, 1]> {name}q = const()[name=string("{name}q"), val=tensor<int{bits}, [{out_dim}, {in_dim}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/{name}.bin"), offset=uint64(64)))];
    tensor<fp16, [{out_dim}, 1, 1, 1]> {name}sc = const()[name=string("{name}sc"), val=tensor<fp16, [{out_dim}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/{name}s.bin"), offset=uint64(64)))];
    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> {name}w = constexpr_blockwise_shift_scale(data={name}q, scale={name}sc)[name=string("{name}dq")];'''


def _packed_split_down_projection(module, blobs: dict[str, bytes],
                                  out_dim: int, in_dim: int, width: int,
                                  bits: int, parts: int) -> tuple[str, str, frozenset[str]]:
    """Build one or a packed input-channel-split ``down_proj``.

    Splitting happens *after* row-wise quantization.  Every slice therefore
    retains the exact original int4/int8 values and row scale; the only
    numerical change is fp16 partial-sum order.  All slice tensors are packed
    into one milinternal file because the ANE compiler rejects models with
    more than 16 weight-file entries.
    """
    if parts == 1:
        body = (
            f'    tensor<fp16, [1, {out_dim}, 1, {width}]> mlp = '
            'conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, '
            'strides=st, weight=dw, x=act)[name=string("mlp")];'
        )
        return _dense_decl("d", out_dim, in_dim, bits), body, frozenset()
    if parts != 4:
        raise ValueError("down_proj parts must be 1 or 4")
    if in_dim % parts or (bits == 4 and (in_dim // parts) % 2):
        raise ValueError(f"cannot split input dimension {in_dim} into {parts}")

    data = blobs.pop("d.bin")
    scales = blobs.pop("ds.bin", None)
    part_in = in_dim // parts
    if bits == 16:
        rows = np.frombuffer(data, dtype=np.float16).reshape(out_dim, in_dim)
        payloads = [np.ascontiguousarray(
            rows[:, part * part_in:(part + 1) * part_in]
        ).tobytes() for part in range(parts)]
    elif bits == 8:
        rows = np.frombuffer(data, dtype=np.int8).reshape(out_dim, in_dim)
        payloads = [np.ascontiguousarray(
            rows[:, part * part_in:(part + 1) * part_in]
        ).tobytes() for part in range(parts)]
    elif bits == 4:
        packed_in = in_dim // 2
        packed_part = part_in // 2
        rows = np.frombuffer(data, dtype=np.uint8).reshape(out_dim, packed_in)
        payloads = [np.ascontiguousarray(
            rows[:, part * packed_part:(part + 1) * packed_part]
        ).tobytes() for part in range(parts)]
    else:
        raise ValueError(f"unsupported down_proj precision int{bits}")
    if bits != 16 and scales is None:
        raise ValueError("quantized down_proj has no row scales")

    raw = bytearray()

    def append_blob(payload: bytes) -> int:
        start = len(raw)
        blob = bytearray(module._make_blob(payload))
        payload_offset = start + 128
        if payload_offset > 0xffffffff:
            raise RuntimeError("packed down_proj exceeds 32-bit blob offsets")
        struct.pack_into("<I", blob, 80, payload_offset)
        raw.extend(blob)
        return start + 64

    decls: list[str] = []
    convs: list[str] = []
    path = '@model_path/weights/down.bin'
    for part, payload in enumerate(payloads):
        data_offset = append_blob(payload)
        name = f"dp{part}"
        if bits == 16:
            decls.append(
                f'    tensor<fp16, [{out_dim}, {part_in}, 1, 1]> {name}w = const()'
                f'[name=string("{name}w"), val=tensor<fp16, '
                f'[{out_dim}, {part_in}, 1, 1]>(BLOBFILE(path=string("{path}"), '
                f'offset=uint64({data_offset})))];'
            )
        else:
            scale_offset = append_blob(scales)  # type: ignore[arg-type]
            decls.append(f'''    tensor<int{bits}, [{out_dim}, {part_in}, 1, 1]> {name}q = const()[name=string("{name}q"), val=tensor<int{bits}, [{out_dim}, {part_in}, 1, 1]>(BLOBFILE(path=string("{path}"), offset=uint64({data_offset})))];
    tensor<fp16, [{out_dim}, 1, 1, 1]> {name}sc = const()[name=string("{name}sc"), val=tensor<fp16, [{out_dim}, 1, 1, 1]>(BLOBFILE(path=string("{path}"), offset=uint64({scale_offset})))];
    tensor<fp16, [{out_dim}, {part_in}, 1, 1]> {name}w = constexpr_blockwise_shift_scale(data={name}q, scale={name}sc)[name=string("{name}dq")];''')
        begin = part * part_in
        end = begin + part_in
        convs.append(f'''    tensor<fp16, [1, {part_in}, 1, {width}]> {name}x = slice_by_index(begin=tensor<int32, [4]>([0,{begin},0,0]), end=tensor<int32, [4]>([1,{end},1,{width}]), x=act)[name=string("{name}x")];
    tensor<fp16, [1, {out_dim}, 1, {width}]> {name}y = conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, strides=st, weight={name}w, x={name}x)[name=string("{name}y")];''')

    convs.extend((
        f'    tensor<fp16, [1, {out_dim}, 1, {width}]> dp01 = add(x=dp0y, y=dp1y)[name=string("dp01")];',
        f'    tensor<fp16, [1, {out_dim}, 1, {width}]> dp23 = add(x=dp2y, y=dp3y)[name=string("dp23")];',
        f'    tensor<fp16, [1, {out_dim}, 1, {width}]> mlp = add(x=dp01, y=dp23)[name=string("mlp")];',
    ))
    blobs["down.bin"] = bytes(raw)
    return "\n".join(decls), "\n".join(convs), frozenset({"down.bin"})


def _stable_rms_block(source: str, output: str, channels: int, width: int,
                      weight: str, prefix: str,
                      active_lanes: int = 1) -> str:
    """MIL for overflow-safe RMSNorm when every decode lane is identical.

    Channel-axis reductions are rejected by the current ANE compiler.  Slice
    one of the 32 duplicate decode lanes, reshape channels onto the supported
    width axis, and reduce there.  Dividing by the dynamic maximum before the
    square prevents fp16 overflow from Qwen's increasingly large residual
    outliers.  ``max(sqrt(eps), max(abs(x)))`` also lets us include epsilon
    without ever squaring that maximum.
    """
    if not 1 <= active_lanes <= width:
        raise ValueError(f"invalid active RMS lanes {active_lanes}/{width}")
    if active_lanes > 1:
        return _stable_rms_vectorized(
            source, output, channels, width, weight, prefix
        )
    eps_root = float(np.float16(1e-3)).hex()
    p = prefix
    return f'''    tensor<fp16, [1, {channels}, 1, 1]> {p}x0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{channels},1,1]), x={source})[name=string("{p}x0")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}flat = reshape(shape=tensor<int32, [4]>([1,1,1,{channels}]), x={p}x0)[name=string("{p}flat")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}abs = abs(x={p}flat)[name=string("{p}abs")];
    tensor<fp16, [1, 1, 1, 1]> {p}max0 = reduce_max(axes=tensor<int32, [1]>([3]), keep_dims=bool(true), x={p}abs)[name=string("{p}max0")];
    tensor<fp16, [1, 1, 1, 1]> {p}max = maximum(x={p}max0, y=fp16({eps_root}))[name=string("{p}max")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}scaled0 = real_div(x={p}flat, y={p}max)[name=string("{p}scaled0")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}scaled = mul(x={p}scaled0, y=fp16(0x1p+6))[name=string("{p}scaled")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}sq = mul(x={p}scaled, y={p}scaled)[name=string("{p}sq")];
    tensor<fp16, [1, 1, 1, 1]> {p}ms = reduce_mean(axes=tensor<int32, [1]>([3]), keep_dims=bool(true), x={p}sq)[name=string("{p}ms")];
    tensor<fp16, [1, 1, 1, 1]> {p}er0 = real_div(x=fp16({eps_root}), y={p}max)[name=string("{p}er0")];
    tensor<fp16, [1, 1, 1, 1]> {p}er = mul(x={p}er0, y=fp16(0x1p+6))[name=string("{p}er")];
    tensor<fp16, [1, 1, 1, 1]> {p}e2 = mul(x={p}er, y={p}er)[name=string("{p}e2")];
    tensor<fp16, [1, 1, 1, 1]> {p}mse = add(x={p}ms, y={p}e2)[name=string("{p}mse")];
    tensor<fp16, [1, 1, 1, 1]> {p}sd = sqrt(x={p}mse)[name=string("{p}sd")];
    tensor<fp16, [1, 1, 1, 1]> {p}den0 = mul(x={p}max, y={p}sd)[name=string("{p}den0")];
    tensor<fp16, [1, 1, 1, 1]> {p}den = mul(x={p}den0, y=fp16(0x1p-6))[name=string("{p}den")];
    tensor<fp16, [1, {channels}, 1, {width}]> {p}unit = real_div(x={source}, y={p}den)[name=string("{p}unit")];
    tensor<fp16, [1, {channels}, 1, {width}]> {output} = mul(x={p}unit, y={weight})[name=string("{output}")];'''


def _stable_rms_vectorized(source: str, output: str, channels: int,
                           width: int, weight: str, prefix: str) -> str:
    """Overflow-safe independent RMSNorm for every physical lane at once.

    Transposing ``[channels, lanes]`` to ``[lanes, channels]`` moves the
    otherwise compiler-rejected channel reduction onto the supported width
    axis. The resulting per-lane denominators transpose back to width scalars
    and broadcast over the original tensor. Zero-filled inactive lanes remain
    zero, so one graph serves decode and wide prefill without lane-unrolled
    work or a second set of weight programs.
    """
    eps_root = float(np.float16(1e-3)).hex()
    p = prefix
    return f'''    tensor<fp16, [1, {width}, 1, {channels}]> {p}xt = transpose(perm=tensor<int32, [4]>([0,3,2,1]), x={source})[name=string("{p}xt")];
    tensor<fp16, [1, {width}, 1, {channels}]> {p}abs = abs(x={p}xt)[name=string("{p}abs")];
    tensor<fp16, [1, {width}, 1, 1]> {p}max0 = reduce_max(axes=tensor<int32, [1]>([3]), keep_dims=bool(true), x={p}abs)[name=string("{p}max0")];
    tensor<fp16, [1, {width}, 1, 1]> {p}max = maximum(x={p}max0, y=fp16({eps_root}))[name=string("{p}max")];
    tensor<fp16, [1, {width}, 1, {channels}]> {p}scaled0 = real_div(x={p}xt, y={p}max)[name=string("{p}scaled0")];
    tensor<fp16, [1, {width}, 1, {channels}]> {p}scaled = mul(x={p}scaled0, y=fp16(0x1p+6))[name=string("{p}scaled")];
    tensor<fp16, [1, {width}, 1, {channels}]> {p}sq = mul(x={p}scaled, y={p}scaled)[name=string("{p}sq")];
    tensor<fp16, [1, {width}, 1, 1]> {p}ms = reduce_mean(axes=tensor<int32, [1]>([3]), keep_dims=bool(true), x={p}sq)[name=string("{p}ms")];
    tensor<fp16, [1, {width}, 1, 1]> {p}er0 = real_div(x=fp16({eps_root}), y={p}max)[name=string("{p}er0")];
    tensor<fp16, [1, {width}, 1, 1]> {p}er = mul(x={p}er0, y=fp16(0x1p+6))[name=string("{p}er")];
    tensor<fp16, [1, {width}, 1, 1]> {p}e2 = mul(x={p}er, y={p}er)[name=string("{p}e2")];
    tensor<fp16, [1, {width}, 1, 1]> {p}mse = add(x={p}ms, y={p}e2)[name=string("{p}mse")];
    tensor<fp16, [1, {width}, 1, 1]> {p}sd = sqrt(x={p}mse)[name=string("{p}sd")];
    tensor<fp16, [1, {width}, 1, 1]> {p}den0 = mul(x={p}max, y={p}sd)[name=string("{p}den0")];
    tensor<fp16, [1, {width}, 1, 1]> {p}den1 = mul(x={p}den0, y=fp16(0x1p-6))[name=string("{p}den1")];
    tensor<fp16, [1, 1, 1, {width}]> {p}den = transpose(perm=tensor<int32, [4]>([0,3,2,1]), x={p}den1)[name=string("{p}den")];
    tensor<fp16, [1, {channels}, 1, {width}]> {p}unit = real_div(x={source}, y={p}den)[name=string("{p}unit")];
    tensor<fp16, [1, {channels}, 1, {width}]> {output} = mul(x={p}unit, y={weight})[name=string("{output}")];'''


def _stable_rms_lanes(source: str, output: str, channels: int, width: int,
                      weight: str, prefix: str, active_lanes: int) -> str:
    """Overflow-safe independent RMSNorm for a small number of real lanes.

    The compiler rejects the natural channel-axis reduction at H=5120.  Each
    live lane is therefore sliced to width one, reshaped so channels become the
    supported width axis, normalized with dynamic max scaling, padded back to
    its original lane, and summed.  Unused physical lanes remain zero.
    """
    eps_root = float(np.float16(1e-3)).hex()
    blocks = []
    padded = []
    for lane in range(active_lanes):
        p = f"{prefix}l{lane}_"
        padded.append(f"{p}pad")
        blocks.append(f'''    tensor<fp16, [1, {channels}, 1, 1]> {p}x = slice_by_index(begin=tensor<int32, [4]>([0,0,0,{lane}]), end=tensor<int32, [4]>([1,{channels},1,{lane+1}]), x={source})[name=string("{p}x")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}flat = reshape(shape=tensor<int32, [4]>([1,1,1,{channels}]), x={p}x)[name=string("{p}flat")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}abs = abs(x={p}flat)[name=string("{p}abs")];
    tensor<fp16, [1, 1, 1, 1]> {p}max0 = reduce_max(axes=tensor<int32, [1]>([3]), keep_dims=bool(true), x={p}abs)[name=string("{p}max0")];
    tensor<fp16, [1, 1, 1, 1]> {p}max = maximum(x={p}max0, y=fp16({eps_root}))[name=string("{p}max")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}scaled0 = real_div(x={p}flat, y={p}max)[name=string("{p}scaled0")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}scaled = mul(x={p}scaled0, y=fp16(0x1p+6))[name=string("{p}scaled")];
    tensor<fp16, [1, 1, 1, {channels}]> {p}sq = mul(x={p}scaled, y={p}scaled)[name=string("{p}sq")];
    tensor<fp16, [1, 1, 1, 1]> {p}ms = reduce_mean(axes=tensor<int32, [1]>([3]), keep_dims=bool(true), x={p}sq)[name=string("{p}ms")];
    tensor<fp16, [1, 1, 1, 1]> {p}er0 = real_div(x=fp16({eps_root}), y={p}max)[name=string("{p}er0")];
    tensor<fp16, [1, 1, 1, 1]> {p}er = mul(x={p}er0, y=fp16(0x1p+6))[name=string("{p}er")];
    tensor<fp16, [1, 1, 1, 1]> {p}e2 = mul(x={p}er, y={p}er)[name=string("{p}e2")];
    tensor<fp16, [1, 1, 1, 1]> {p}mse = add(x={p}ms, y={p}e2)[name=string("{p}mse")];
    tensor<fp16, [1, 1, 1, 1]> {p}sd = sqrt(x={p}mse)[name=string("{p}sd")];
    tensor<fp16, [1, 1, 1, 1]> {p}den0 = mul(x={p}max, y={p}sd)[name=string("{p}den0")];
    tensor<fp16, [1, 1, 1, 1]> {p}den = mul(x={p}den0, y=fp16(0x1p-6))[name=string("{p}den")];
    tensor<fp16, [1, {channels}, 1, 1]> {p}unit = real_div(x={p}x, y={p}den)[name=string("{p}unit")];
    tensor<fp16, [1, {channels}, 1, 1]> {p}norm = mul(x={p}unit, y={weight})[name=string("{p}norm")];
    tensor<fp16, [1, {channels}, 1, {width}]> {p}pad = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,0,0,0,{lane},{width-lane-1}]), x={p}norm)[name=string("{p}pad")];''')
    current = padded[0]
    for lane, name in enumerate(padded[1:], 1):
        target = output if lane == len(padded)-1 else f"{prefix}sum{lane}"
        blocks.append(
            f'    tensor<fp16, [1, {channels}, 1, {width}]> {target} = '
            f'add(x={current}, y={name})[name=string("{target}")];'
        )
        current = target
    if active_lanes == 1:
        blocks.append(
            f'    tensor<fp16, [1, {channels}, 1, {width}]> {output} = '
            f'identity(x={current})[name=string("{output}")];'
        )
    return "\n".join(blocks)


def _lane_matrix(value: np.ndarray, rows: int, max_lanes: int) -> tuple[np.ndarray, int]:
    """Normalize a vector or [rows, lanes] matrix to the latter shape."""
    a = np.asarray(value, dtype=np.float16)
    if a.ndim == 1:
        if a.shape != (rows,):
            raise ValueError(f"expected ({rows},), got {a.shape}")
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] != rows or not 1 <= a.shape[1] <= max_lanes:
        raise ValueError(
            f"expected ({rows}, 1..{max_lanes}) lanes, got {a.shape}"
        )
    return a, a.shape[1]


def _restore_lane_rank(value: np.ndarray, lanes: int) -> np.ndarray:
    return value[:, 0] if lanes == 1 else value


def _quantize_matrix(checkpoint: Checkpoint, tensor_name: str,
                     blob_name: str, bits: int, rows: int = 512, *,
                     row_start: int = 0,
                     row_end: int | None = None) -> dict[str, bytes]:
    """Create ANE row-wise weight blobs with bounded temporary memory."""
    info = checkpoint.info(tensor_name)
    if len(info.shape) != 2:
        raise ValueError(f"linear weight is not a matrix: {tensor_name} {info.shape}")
    full_out, in_dim = info.shape
    row_end = full_out if row_end is None else row_end
    if not 0 <= row_start <= row_end <= full_out:
        raise ValueError(f"invalid quantization rows {row_start}:{row_end}")
    out_dim = row_end - row_start
    cache_key = None
    data_path = scale_path = None
    if checkpoint.quant_cache_dir is not None:
        source_stat = info.path.stat()
        cache_key = hashlib.sha256(
            (f"q38-quant-v2-zstd\0{tensor_name}\0{bits}\0{row_start}:{row_end}\0"
             f"{info.dtype}\0{info.shape}\0{info.offset}:{info.nbytes}\0"
             f"{source_stat.st_size}:{source_stat.st_mtime_ns}").encode()
        ).hexdigest()
        suffix = ".zst" if _zstd is not None else ".raw"
        data_path = checkpoint.quant_cache_dir / f"{cache_key}.data{suffix}"
        scale_path = checkpoint.quant_cache_dir / f"{cache_key}.scales{suffix}"
        data_size = out_dim * in_dim * (2 if bits == 16 else 1) // (2 if bits == 4 else 1)
        scale_size = 0 if bits == 16 else out_dim * 2
        if (data_path.is_file()
                and (scale_size == 0 or scale_path.is_file())):
            started = time.perf_counter()
            try:
                encoded_data = data_path.read_bytes()
                data = (_zstd.decompress(encoded_data) if _zstd is not None
                        else encoded_data)
                if len(data) != data_size:
                    raise ValueError("cached tensor payload size mismatch")
                blobs = {f"{blob_name}.bin": data}
                disk_bytes = len(encoded_data)
                if scale_size:
                    encoded_scale = scale_path.read_bytes()
                    scale = (_zstd.decompress(encoded_scale) if _zstd is not None
                             else encoded_scale)
                    if len(scale) != scale_size:
                        raise ValueError("cached tensor scale size mismatch")
                    blobs[f"{blob_name}s.bin"] = scale
                    disk_bytes += len(encoded_scale)
            except Exception:
                # A partial/obsolete entry is a normal cache miss. Atomic
                # replacement below repairs it without trusting corrupt data.
                pass
            else:
                stats = checkpoint.quant_cache_stats
                stats["hits"] += 1
                stats["bytes_read"] += sum(len(x) for x in blobs.values())
                stats["disk_bytes_read"] += disk_bytes
                stats["read_seconds"] += time.perf_counter() - started
                return blobs

    checkpoint.quant_cache_stats["misses"] += 1
    quantize_started = time.perf_counter()
    if bits == 16:
        packed = np.empty((out_dim, in_dim), np.float16)
        scales = None
    elif bits == 4:
        if in_dim % 2:
            raise ValueError("int4 packing requires an even input dimension")
        packed = np.empty(out_dim * in_dim // 2, np.uint8)
        scales = np.empty((out_dim, 1), np.float16)
    elif bits == 8:
        packed = np.empty(out_dim * in_dim, np.int8)
        scales = np.empty((out_dim, 1), np.float16)
    else:
        raise ValueError("bits must be 4, 8, or 16")
    hi = (1 << (bits - 1)) - 1 if bits != 16 else 0
    r0 = 0
    for block in checkpoint.tensor_row_blocks(
        tensor_name, rows, start=row_start, end=row_end
    ):
        n = block.shape[0]
        if bits == 16:
            packed[r0:r0+n] = block.astype(np.float16)
        else:
            scale = np.abs(block).max(axis=1, keepdims=True) / hi
            np.divide(block, np.where(scale == 0, 1, scale), out=block)
            np.rint(block, out=block)
            np.clip(block, -hi - 1, hi, out=block)
            q = block.astype(np.int8)
            if bits == 4:
                nibble = q.reshape(-1).astype(np.uint8) & 0x0f
                packed[r0*in_dim//2:(r0+n)*in_dim//2] = (
                    nibble[0::2] | (nibble[1::2] << 4)
                )
            else:
                packed[r0*in_dim:(r0+n)*in_dim] = q.reshape(-1)
            scales[r0:r0+n] = scale.astype(np.float16)
        r0 += n
    blobs = {f"{blob_name}.bin": packed.tobytes()}
    if scales is not None:
        blobs[f"{blob_name}s.bin"] = scales.tobytes()
    checkpoint.quant_cache_stats["quantize_seconds"] += (
        time.perf_counter() - quantize_started
    )
    if data_path is not None:
        compress_started = time.perf_counter()
        encoded = []
        for name in (f"{blob_name}.bin", f"{blob_name}s.bin"):
            if name in blobs:
                payload = blobs[name]
                encoded.append((_zstd.compress(payload, level=1)
                                if _zstd is not None else payload))
        checkpoint.quant_cache_stats["compress_seconds"] += (
            time.perf_counter() - compress_started
        )
        needed = sum(len(x) for x in encoded)
        # Compressed int4 tensors measured 70.3% of their raw size. Keep 8 GB
        # after a compressed write (20 GB for the uncompressed fallback).
        # Existing entries remain readable below this threshold.
        minimum_free = 8_000_000_000 if _zstd is not None else 20_000_000_000
        parent = checkpoint.quant_cache_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free >= needed + minimum_free:
            checkpoint.quant_cache_dir.mkdir(parents=True, exist_ok=True)
            write_started = time.perf_counter()
            targets = [(data_path, encoded[0])]
            if scales is not None:
                targets.append((scale_path, encoded[1]))
            for target, payload in targets:
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                with temporary.open("wb") as stream:
                    stream.write(payload)
                os.replace(temporary, target)
            checkpoint.quant_cache_stats["bytes_written"] += sum(
                len(x) for x in blobs.values()
            )
            checkpoint.quant_cache_stats["disk_bytes_written"] += needed
            checkpoint.quant_cache_stats["write_seconds"] += (
                time.perf_counter() - write_started
            )
    return blobs


class AneDriver:
    """Thin import boundary around the independent private ANE driver."""

    def __init__(self, engine_path: str):
        # Explicitly request pageable model memory. Without the key, this OS
        # build fails around 7.2 GB of resident int4 blobs with load 0x50004.
        os.environ.setdefault("Q38_ANE_KEEP_WIRED", "0")
        path = str(Path(engine_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
        import runtime.q38_ane_engine as engine_module
        self.module = engine_module
        self.engine = engine_module.AneEngine()
        self.view = engine_module._iosurface_view
        if not self.engine.available:
            raise RuntimeError("Apple Neural Engine private runtime is unavailable")
        assert_standalone("ANE driver initialization")

    def discard_compiler_files(self, program: object) -> None:
        """Drop rebuildable compiler files and copied payloads after load."""
        import gc, shutil
        try:
            local = self.module._desc(
                self.module._msg(program.model, "localModelPath")
            )
            if local and local != "(nil)" and (Path(local) / "model.mil").exists():
                shutil.rmtree(local)
        except OSError:
            pass
        # compile_multiproc retains the source bytes defensively. A loaded
        # _ANEInMemoryModel owns its compiled representation; live submission
        # remains bit-identical after releasing these Python copies. Keeping
        # them would duplicate the complete fp16 checkpoint in process RSS.
        keep=getattr(program,"_keep_alive",None)
        if isinstance(keep,list):
            keep[:]=[x for x in keep if not isinstance(x,(bytes,bytearray,memoryview))]
        gc.collect()


class AneNormProjection:
    """RMSNorm plus a fused set of input projections in one ANE program."""

    def __init__(self, driver: AneDriver, checkpoint: Checkpoint,
                 norm_name: str, projection_names: list[str], *,
                 bits: int = 4, width: int = 32, tag: str = "head",
                 norm_scale: float = 64.0, active_lanes: int = 1):
        self.driver = driver
        self.width = max(32, width)
        self.active_lanes = active_lanes
        norm = checkpoint.tensor(norm_name, np.float16).reshape(-1)
        self.hidden = norm.size
        infos = [checkpoint.info(name) for name in projection_names]
        if any(info.shape[1] != self.hidden for info in infos):
            raise ValueError("projection input dimensions do not match RMSNorm")
        self.spans, out0 = [], 0
        for name, info in zip(projection_names, infos):
            self.spans.append((name, out0, out0 + info.shape[0]))
            out0 += info.shape[0]
        self.output = out0

        # The fused matrix is represented by concatenated row blocks.  Quantize
        # each source independently, then concatenate payloads and scales; row-
        # wise quantization makes this exactly equivalent to one large matrix.
        parts = [_quantize_matrix(checkpoint, name, "p", bits)
                 for name in projection_names]
        blobs = {"p.bin": b"".join(part["p.bin"] for part in parts)}
        if bits != 16:
            blobs["ps.bin"] = b"".join(part["ps.bin"] for part in parts)
        blobs["norm.bin"] = norm.tobytes()

        H, O, S = self.hidden, self.output, self.width
        decl = _dense_decl("p", O, H, bits)
        norm_body = _stable_rms_block(
            "x", "norm", H, S, "nw", "nr", active_lanes
        )
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {H}, 1, 1]> nw = const()[name=string("nw"), val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/norm.bin"), offset=uint64(64)))];
{decl}
{norm_body}
    tensor<fp16, [1, {O}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=pw, x=norm)[name=string("{tag}")];
  }} -> (y);
}}
// pure_ane_{tag}_{H}_{O}_int{bits}
'''
        capture = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(
                mil, blobs, H, O, S
            )
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-8:])
            raise RuntimeError(f"ANE norm+projection compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        self.nbytes = sum(len(x) for x in blobs.values())
        self.compile_seconds = time.time() - t0
        assert_standalone("norm+projection compile")

    def __call__(self, hidden: np.ndarray) -> np.ndarray:
        hidden, lanes = _lane_matrix(hidden, self.hidden, self.active_lanes)
        with self.driver.view(
            self.program._in_surf, (self.hidden, self.width), np.float16
        ) as dst:
            dst[:] = 0
            if self.active_lanes == 1:
                # The legacy one-lane graph broadcasts lane zero internally.
                dst[:] = hidden[:, :1]
            else:
                dst[:, :lanes] = hidden
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE norm+projection submission failed")
        with self.driver.view(
            self.program._out_surf, (self.output, self.width), np.float16
        ) as src:
            out = np.array(src[:, :lanes], dtype=np.float16)
        assert_standalone("norm+projection dispatch")
        return _restore_lane_rank(out, lanes)


class AneGdnConv:
    """Real-shape K=4 causal depthwise convolution plus precise SiLU on ANE."""

    def __init__(self, driver: AneDriver, checkpoint: Checkpoint,
                 weight_name: str, width: int = 32):
        self.driver = driver
        self.width = max(32, width)
        weight = checkpoint.tensor(weight_name, np.float16)
        if weight.ndim != 3 or weight.shape[1:] != (1, 4):
            raise ValueError(f"unexpected GDN conv weight {weight.shape}")
        self.channels = weight.shape[0]
        self.weight = weight[:, 0, :].astype(np.float32)
        self.cache = np.zeros((self.channels, 3), np.float16)
        blobs = {"conv.bin": weight.reshape(self.channels, 1, 1, 4).tobytes()}
        C, S = self.channels, self.width
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<fp16, [{C}, 1, 1, 4]> w = const()[name=string("w"), val=tensor<fp16, [{C}, 1, 1, 4]>(BLOBFILE(path=string("@model_path/weights/conv.bin"), offset=uint64(64)))];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,3,0])];
    tensor<fp16, [1, {C}, 1, {S}]> c = conv(dilations=dl, groups=int32({C}), pad=pd, pad_type=string("custom"), strides=st, weight=w, x=x)[name=string("causal")];
    tensor<fp16, [1, {C}, 1, {S}]> nc = mul(x=c, y=fp16(-0x1p+0))[name=string("nc")];
    tensor<fp16, [1, {C}, 1, {S}]> ex = exp(x=nc)[name=string("ex")];
    tensor<fp16, [1, {C}, 1, {S}]> den = add(x=ex, y=fp16(0x1p+0))[name=string("den")];
    tensor<fp16, [1, {C}, 1, {S}]> y = real_div(x=c, y=den)[name=string("silu")];
  }} -> (y);
}}
// pure_ane_gdn_conv_C{C}
'''
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(mil, blobs, C, C, S)
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-8:])
            raise RuntimeError(f"ANE GDN convolution compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        assert_standalone("GDN convolution compile")

    def reset(self) -> None:
        self.cache[:] = 0

    def __call__(self, qkv: np.ndarray) -> np.ndarray:
        current, lanes = _lane_matrix(qkv, self.channels, self.width-3)
        with self.driver.view(
            self.program._in_surf, (self.channels, self.width), np.float16
        ) as dst:
            dst[:] = 0
            dst[:, :3] = self.cache
            dst[:, 3:3+lanes] = current
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE GDN convolution submission failed")
        with self.driver.view(
            self.program._out_surf, (self.channels, self.width), np.float16
        ) as src:
            out = np.array(src[:, 3:3+lanes], dtype=np.float16)
        history = np.concatenate((self.cache, current), axis=1)
        self.cache[:] = history[:, -3:]
        assert_standalone("GDN convolution dispatch")
        return _restore_lane_rank(out, lanes)


@dataclass
class GdnState:
    surface: object
    request: object


def _bind_secondary_output(driver:AneDriver,program,channels:int,width:int=32):
    # `width` is the program's compiled width. It was hardcoded at 32, which
    # silently under-allocates the secondary surface for wider programs and
    # fails evaluation rather than the binding.
    E=driver.module;E._load_iosurface()
    secondary=E._create_iosurface(E._iosurface_alloc_size(channels*width))
    inner=E._msg(program.model,"model") or program.model
    desc=E._desc(E._msg(inner,"description"))
    outputs=[(int(c),n) for c,_,n in re.findall(
        r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";',
        desc,re.S
    )]
    if len(outputs)!=2:raise RuntimeError(f"unexpected secondary outputs {outputs}")
    def surf(output):
        c,name=output
        # Chained projection outputs have a different channel count, which is
        # more robust than relying on the compiler-preserved MIL name. Legacy
        # same-shape norm outputs still use the y2 name discriminator.
        is_secondary=(c==channels if channels!=program.output_dim
                      else name.startswith("y2@"))
        return secondary if is_secondary else program._out_surf
    init=ctypes.CFUNCTYPE(*([ctypes.c_void_p]*12))
    req=init(("objc_msgSend",E._objc))(E._msg(E._cls("_ANERequest"),"alloc"),E._sel("initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),E._nsarray([E._wrap_iosurface(program._in_surf)]),E._nsarray([E._nsnumber_int(0)]),E._nsarray([E._wrap_iosurface(surf(output)) for output in outputs]),E._nsarray([E._nsnumber_int(i) for i in range(2)]),None,None,E._nsnumber_int(0),None,None)
    return secondary,req


def _submit_bound(driver:AneDriver,program,request):
    E=driver.module;err=ctypes.c_void_p(0)
    Eval=ctypes.CFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint,ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p))
    ok=Eval(("objc_msgSend",E._objc))(program.model,E._sel("evaluateWithQoS:options:request:error:"),21,program._compile_opts,request,ctypes.byref(err))
    if not ok:raise RuntimeError("bound two-output evaluation failed")


class AneGdnRecurrence:
    """Shared, state-resident Q/K norm + gated-delta decode program.

    Q/K RMSNorm, A_log exponentiation, fp16-safe softplus, decay, beta, the
    recurrent update, and the output contraction are all model arithmetic and
    therefore all live in this ANE graph.  The host only repeats head records
    and copies bytes into fixed IOSurface fields.
    """

    def __init__(self, driver: AneDriver, heads: int = 48,
                 key_dim: int = 128, value_dim: int = 128):
        self.driver = driver
        self.E = driver.module
        self.H, self.Dk, self.Dv = heads, key_dim, value_dim
        self.HK = heads * key_dim
        self.CIN = self.HK + 2 * heads
        self.W = ((value_dim + 5 + 31) // 32) * 32
        H, Dk, Dv, HK, CIN, W = (
            self.H, self.Dk, self.Dv, self.HK, self.CIN, self.W
        )
        blobs = {
            "sum.bin": np.ones((H, Dk, 1, 1), np.float16).tobytes(),
            "mean.bin": np.full((H, Dk, 1, 1), 1.0/Dk, np.float16).tobytes(),
            "repeat.bin": np.ones((HK, 1, 1, 1), np.float16).tobytes(),
        }

        def sl(name: str, c0: int, c1: int, w0: int, w1: int) -> str:
            return (
                f'    tensor<fp16, [1, {c1-c0}, 1, {w1-w0}]> {name} = '
                f'slice_by_index(begin=tensor<int32, [4]>([0,{c0},0,{w0}]), '
                f'end=tensor<int32, [4]>([1,{c1},1,{w1}]), x=x)'
                f'[name=string("{name}")];'
            )

        mil = f'''program(1.3)
{self.E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {CIN}, 1, {W}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gh = const()[name=string("gh"), val=int32({H})];
    tensor<fp16, [{H}, {Dk}, 1, 1]> gsum = const()[name=string("gsum"), val=tensor<fp16, [{H}, {Dk}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/sum.bin"), offset=uint64(64)))];
    tensor<fp16, [{H}, {Dk}, 1, 1]> gmean = const()[name=string("gmean"), val=tensor<fp16, [{H}, {Dk}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/mean.bin"), offset=uint64(64)))];
    tensor<fp16, [{HK}, 1, 1, 1]> grep = const()[name=string("grep"), val=tensor<fp16, [{HK}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/repeat.bin"), offset=uint64(64)))];
{sl("stt", 0, HK, 0, Dv)}
{sl("kraw", 0, HK, Dv, Dv+1)}
{sl("qraw", 0, HK, Dv+1, Dv+2)}
{sl("aa", 0, HK, Dv+2, Dv+3)}
{sl("dtc", 0, HK, Dv+3, Dv+4)}
{sl("alog", 0, HK, Dv+4, Dv+5)}
{sl("vv", HK, HK+H, 0, Dv)}
{sl("bb", HK+H, HK+2*H, 0, 1)}
    tensor<fp16, [1, {HK}, 1, 1]> k8 = mul(x=kraw, y=fp16(0x1p+4))[name=string("k8")];
    tensor<fp16, [1, {HK}, 1, 1]> ksq = mul(x=k8, y=k8)[name=string("ksq")];
    tensor<fp16, [1, {H}, 1, 1]> kms = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gmean, x=ksq)[name=string("kms")];
    tensor<fp16, [1, {H}, 1, 1]> kmse = add(x=kms, y=fp16(0x1.0c8p-12))[name=string("kmse")];
    tensor<fp16, [1, {H}, 1, 1]> ksd = sqrt(x=kmse)[name=string("ksd")];
    tensor<fp16, [1, {HK}, 1, 1]> ksdr = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=ksd)[name=string("ksdr")];
    tensor<fp16, [1, {HK}, 1, 1]> kunit = real_div(x=k8, y=ksdr)[name=string("kunit")];
    tensor<fp16, [1, {HK}, 1, 1]> kk = mul(x=kunit, y=fp16(0x1.6ap-4))[name=string("kk")];
    tensor<fp16, [1, {HK}, 1, 1]> q8 = mul(x=qraw, y=fp16(0x1p+4))[name=string("q8")];
    tensor<fp16, [1, {HK}, 1, 1]> qsq0 = mul(x=q8, y=q8)[name=string("qsq0")];
    tensor<fp16, [1, {H}, 1, 1]> qms = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gmean, x=qsq0)[name=string("qms")];
    tensor<fp16, [1, {H}, 1, 1]> qmse = add(x=qms, y=fp16(0x1.0c8p-12))[name=string("qmse")];
    tensor<fp16, [1, {H}, 1, 1]> qsd = sqrt(x=qmse)[name=string("qsd")];
    tensor<fp16, [1, {HK}, 1, 1]> qsdr = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=qsd)[name=string("qsdr")];
    tensor<fp16, [1, {HK}, 1, 1]> qunit = real_div(x=q8, y=qsdr)[name=string("qunit")];
    tensor<fp16, [1, {HK}, 1, 1]> qq = mul(x=qunit, y=fp16(0x1p-1))[name=string("qq")];
    tensor<fp16, [1, {HK}, 1, 1]> ap = add(x=aa, y=dtc)[name=string("ap")];
    tensor<fp16, [1, {HK}, 1, 1]> pos = relu(x=ap)[name=string("pos")];
    tensor<fp16, [1, {HK}, 1, 1]> ab = abs(x=ap)[name=string("ab")];
    tensor<fp16, [1, {HK}, 1, 1]> nab = mul(x=ab, y=fp16(-0x1p+0))[name=string("nab")];
    tensor<fp16, [1, {HK}, 1, 1]> tt = exp(x=nab)[name=string("tt")];
    tensor<fp16, [1, {HK}, 1, 1]> hp5 = mul(x=tt, y=fp16(-0x1.84p-6))[name=string("hp5")];
    tensor<fp16, [1, {HK}, 1, 1]> ha4 = add(x=hp5, y=fp16(0x1.9acp-4))[name=string("ha4")];
    tensor<fp16, [1, {HK}, 1, 1]> hm4 = mul(x=ha4, y=tt)[name=string("hm4")];
    tensor<fp16, [1, {HK}, 1, 1]> ha3 = add(x=hm4, y=fp16(-0x1.ab4p-3))[name=string("ha3")];
    tensor<fp16, [1, {HK}, 1, 1]> hm3 = mul(x=ha3, y=tt)[name=string("hm3")];
    tensor<fp16, [1, {HK}, 1, 1]> ha2 = add(x=hm3, y=fp16(0x1.4c4p-2))[name=string("ha2")];
    tensor<fp16, [1, {HK}, 1, 1]> hm2 = mul(x=ha2, y=tt)[name=string("hm2")];
    tensor<fp16, [1, {HK}, 1, 1]> ha1 = add(x=hm2, y=fp16(-0x1.ff4p-2))[name=string("ha1")];
    tensor<fp16, [1, {HK}, 1, 1]> hm1 = mul(x=ha1, y=tt)[name=string("hm1")];
    tensor<fp16, [1, {HK}, 1, 1]> ha0 = add(x=hm1, y=fp16(0x1p+0))[name=string("ha0")];
    tensor<fp16, [1, {HK}, 1, 1]> tail = mul(x=tt, y=ha0)[name=string("tail")];
    tensor<fp16, [1, {HK}, 1, 1]> soft = add(x=pos, y=tail)[name=string("soft")];
    tensor<fp16, [1, {HK}, 1, 1]> avec = exp(x=alog)[name=string("avec")];
    tensor<fp16, [1, {HK}, 1, 1]> asp = mul(x=avec, y=soft)[name=string("asp")];
    tensor<fp16, [1, {HK}, 1, 1]> nasp = mul(x=asp, y=fp16(-0x1p+0))[name=string("nasp")];
    tensor<fp16, [1, {HK}, 1, 1]> dcy = exp(x=nasp)[name=string("dcy")];
    tensor<fp16, [1, {H}, 1, 1]> nb = mul(x=bb, y=fp16(-0x1p+0))[name=string("nb")];
    tensor<fp16, [1, {H}, 1, 1]> enb = exp(x=nb)[name=string("enb")];
    tensor<fp16, [1, {H}, 1, 1]> bden = add(x=enb, y=fp16(0x1p+0))[name=string("bden")];
    tensor<fp16, [1, {H}, 1, 1]> bta = real_div(x=fp16(0x1p+0), y=bden)[name=string("bta")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> s1 = mul(x=stt, y=dcy)[name=string("s1")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> sk = mul(x=s1, y=kk)[name=string("sk")];
    tensor<fp16, [1, {H}, 1, {Dv}]> kvm = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sk)[name=string("kvm")];
    tensor<fp16, [1, {H}, 1, {Dv}]> dlt = sub(x=vv, y=kvm)[name=string("dlt")];
    tensor<fp16, [1, {H}, 1, {Dv}]> dbt = mul(x=dlt, y=bta)[name=string("dbt")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> dup = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=dbt)[name=string("dup")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> upd = mul(x=dup, y=kk)[name=string("upd")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> s2 = add(x=s1, y=upd)[name=string("s2")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> sq = mul(x=s2, y=qq)[name=string("sq")];
    tensor<fp16, [1, {H}, 1, {Dv}]> y = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sq)[name=string("y")];
  }} -> (y, s2);
}}
// pure_ane_gdn_norm_gates_recurrence
'''
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(mil, blobs, CIN, H, W)
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE GDN recurrence compile failed:\n{tail}")
        self.E._load_iosurface()
        self.input_surface = self.E._create_iosurface(
            self.E._iosurface_alloc_size(CIN * W)
        )
        self.y_surface = self.E._create_iosurface(
            self.E._iosurface_alloc_size(H * Dv)
        )
        inner = self.E._msg(self.program.model, "model") or self.program.model
        desc = self.E._desc(self.E._msg(inner, "description"))
        self.output_channels = [int(ch) for ch, _, _ in re.findall(
            r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";',
            desc, re.S
        )]
        if sorted(self.output_channels) != sorted((H, HK)):
            raise RuntimeError(f"unexpected recurrence outputs {self.output_channels}")
        self._Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        assert_standalone("GDN recurrence compile")

    def new_state(self) -> GdnState:
        surface = self.E._create_iosurface(
            self.E._iosurface_alloc_size(self.HK * self.Dv)
        )
        if not surface:
            raise RuntimeError("GDN state IOSurface allocation failed")
        with self.driver.view(surface, (self.HK, self.Dv), np.float16) as dst:
            dst[:] = 0
        surf_for = {self.H: self.y_surface, self.HK: surface}
        init = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
        request = init(("objc_msgSend", self.E._objc))(
            self.E._msg(self.E._cls("_ANERequest"), "alloc"),
            self.E._sel("initWithInputs:inputIndices:outputs:outputIndices:"
                        "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                        "transactionHandle:"),
            self.E._nsarray([self.E._wrap_iosurface(self.input_surface)]),
            self.E._nsarray([self.E._nsnumber_int(0)]),
            self.E._nsarray([self.E._wrap_iosurface(surf_for[c])
                             for c in self.output_channels]),
            self.E._nsarray([self.E._nsnumber_int(i)
                             for i in range(len(self.output_channels))]),
            None, None, self.E._nsnumber_int(0), None, None
        )
        if not request:
            raise RuntimeError("GDN state request creation failed")
        return GdnState(surface, request)

    def materialize(self, state: GdnState) -> np.ndarray:
        with self.driver.view(
            state.surface, (self.HK, self.Dv), np.float16
        ) as src:
            return np.array(src, np.float32).reshape(
                self.H, self.Dk, self.Dv
            ).transpose(0, 2, 1)

    def snapshot(self, state: GdnState) -> np.ndarray:
        """Copy the compact recurrent state for speculative rollback."""
        with self.driver.view(
            state.surface, (self.HK, self.Dv), np.float16
        ) as src:
            return np.array(src, np.float16)

    def restore(self, state: GdnState, saved: np.ndarray) -> None:
        if saved.shape != (self.HK, self.Dv) or saved.dtype != np.float16:
            raise ValueError(f"invalid GDN snapshot {saved.shape}/{saved.dtype}")
        with self.driver.view(
            state.surface, (self.HK, self.Dv), np.float16
        ) as dst:
            dst[:] = saved

    def __call__(self, state: GdnState, q: np.ndarray, k: np.ndarray,
                 v: np.ndarray, a: np.ndarray, beta_logits: np.ndarray,
                 a_log: np.ndarray, dt_bias: np.ndarray) -> np.ndarray:
        if q.shape != (16, self.Dk) or k.shape != (16, self.Dk):
            raise ValueError(f"unexpected GDN q/k shapes {q.shape}/{k.shape}")
        if v.shape != (self.H, self.Dv):
            raise ValueError(f"unexpected GDN v shape {v.shape}")
        q48 = np.repeat(q.astype(np.float16, copy=False), 3, axis=0)
        k48 = np.repeat(k.astype(np.float16, copy=False), 3, axis=0)
        with self.driver.view(
            state.surface, (self.HK, self.Dv), np.float16
        ) as state_src, self.driver.view(
            self.input_surface, (self.CIN, self.W), np.float16
        ) as dst:
            dst[:] = 0
            dst[:self.HK, :self.Dv] = state_src
            dst[:self.HK, self.Dv] = k48.reshape(-1)
            dst[:self.HK, self.Dv+1] = q48.reshape(-1)
            dst[:self.HK, self.Dv+2] = np.repeat(a, self.Dk)
            dst[:self.HK, self.Dv+3] = np.repeat(dt_bias, self.Dk)
            dst[:self.HK, self.Dv+4] = np.repeat(a_log, self.Dk)
            dst[self.HK:self.HK+self.H, :self.Dv] = v
            dst[self.HK+self.H:self.HK+2*self.H, 0] = beta_logits
        error = ctypes.c_void_p(0)
        ok = self._Eval(("objc_msgSend", self.E._objc))(
            self.program.model,
            self.E._sel("evaluateWithQoS:options:request:error:"),
            21, self.program._compile_opts, state.request, ctypes.byref(error)
        )
        if not ok:
            detail = self.E._desc(error.value) if error.value else "unknown"
            raise RuntimeError(f"GDN recurrence evaluate failed: {detail}")
        with self.driver.view(
            self.y_surface, (self.H, self.Dv), np.float16
        ) as src:
            out = np.array(src, np.float16)
        assert_standalone("GDN recurrence dispatch")
        return out


class AneGdnTail:
    """GDN gated norm through residual MLP, fused into one ANE program."""

    def __init__(self, driver: AneDriver, checkpoint: Checkpoint,
                 layer: int, bits: int = 4, width: int = 32,
                 next_norm_name: str | None = None,
                 next_projection_names: list[str] | None = None,
                 active_lanes: int = 1,
                 down_proj_parts: int = 1):
        self.driver = driver
        self.layer = layer
        self.bits = bits
        self.width = max(32, width)
        self.active_lanes = active_lanes
        p = f"model.language_model.layers.{layer}"
        names = {
            "o": f"{p}.linear_attn.out_proj.weight",
            "g": f"{p}.mlp.gate_proj.weight",
            "u": f"{p}.mlp.up_proj.weight",
            "d": f"{p}.mlp.down_proj.weight",
        }
        oi, gi, ui, di = (checkpoint.info(names[k]) for k in ("o", "g", "u", "d"))
        self.H, self.Dc = oi.shape
        self.I = gi.shape[0]
        if oi.shape[1] != 6144 or gi.shape != ui.shape \
                or gi.shape[1] != self.H or di.shape != (self.H, self.I):
            raise ValueError(f"unexpected layer {layer} tail shapes")
        self.input = 2*self.Dc + self.H
        H, Dc, I, S = self.H, self.Dc, self.I, self.width
        self.down_proj_parts = down_proj_parts

        blobs = {}
        for key in ("o", "d"):
            blobs.update(_quantize_matrix(checkpoint, names[key], key, bits))
        gp = _quantize_matrix(checkpoint, names["g"], "gu", bits)
        up = _quantize_matrix(checkpoint, names["u"], "gu", bits)
        blobs["gu.bin"] = gp["gu.bin"] + up["gu.bin"]
        if bits != 16:
            blobs["gus.bin"] = gp["gus.bin"] + up["gus.bin"]
        gated_norm = checkpoint.tensor(
            f"{p}.linear_attn.norm.weight", np.float16
        )
        if gated_norm.shape != (128,):
            raise ValueError(f"unexpected gated norm {gated_norm.shape}")
        blobs["gn.bin"] = np.tile(gated_norm, 48).astype(np.float16).tobytes()
        blobs["gmean.bin"] = np.full((48, 128), 1/128, np.float16).tobytes()
        blobs["grep.bin"] = np.ones((Dc, 1), np.float16).tobytes()
        blobs["pn.bin"] = checkpoint.tensor(
            f"{p}.post_attention_layernorm.weight", np.float16
        ).tobytes()
        if (next_norm_name is None)!=(next_projection_names is None):
            raise ValueError("next norm and projection must be supplied together")
        self.has_next=next_norm_name is not None
        self.next_output=0
        if self.has_next:
            blobs["next.bin"]=checkpoint.tensor(next_norm_name,np.float16).tobytes()
            next_infos=[checkpoint.info(name) for name in next_projection_names]
            if any(info.shape[1]!=H for info in next_infos):
                raise ValueError("next projection input dimensions do not match")
            self.next_output=sum(info.shape[0] for info in next_infos)
            parts=[_quantize_matrix(checkpoint,name,"np",bits)
                   for name in next_projection_names]
            blobs["np.bin"]=b"".join(part["np.bin"] for part in parts)
            if bits!=16:
                blobs["nps.bin"]=b"".join(part["nps.bin"] for part in parts)
        next_decl=(f'''    tensor<fp16, [1, {H}, 1, 1]> nnw = const()[name=string("nnw"), val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/next.bin"), offset=uint64(64)))];
{_dense_decl("np",self.next_output,H,bits)}''' if self.has_next else '')
        post_norm = _stable_rms_block(
            "h", "hn", H, S, "pn", f"hp{layer}_", active_lanes
        )
        next_norm = _stable_rms_block(
            "o0", "nn", H, S, "nnw", f"np{layer}_", active_lanes
        )
        next_body=(f'''    tensor<fp16, [1, {H}, 1, {S}]> y = identity(x=o0)[name=string("y")];
{next_norm}
    tensor<fp16, [1, {self.next_output}, 1, {S}]> y2 = conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, strides=st, weight=npw, x=nn)[name=string("y2")];''' if self.has_next else '')
        down_decl, down_body, raw_weight_files = _packed_split_down_projection(
            driver.module, blobs, H, I, S, bits, down_proj_parts
        )
        decl = "\n".join((
            _dense_decl("o", H, Dc, bits),
            _dense_decl("gu", 2*I, H, bits),
            down_decl,
        ))
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {self.input}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 g1 = const()[name=string("g1"), val=int32(1)];
    int32 g48 = const()[name=string("g48"), val=int32(48)];
{decl}
{next_decl}
    tensor<fp16, [1, {Dc}, 1, 1]> gn = const()[name=string("gn"), val=tensor<fp16, [1, {Dc}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/gn.bin"), offset=uint64(64)))];
    tensor<fp16, [48, 128, 1, 1]> gm = const()[name=string("gm"), val=tensor<fp16, [48, 128, 1, 1]>(BLOBFILE(path=string("@model_path/weights/gmean.bin"), offset=uint64(64)))];
    tensor<fp16, [{Dc}, 1, 1, 1]> gr = const()[name=string("gr"), val=tensor<fp16, [{Dc}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/grep.bin"), offset=uint64(64)))];
    tensor<fp16, [1, {H}, 1, 1]> pn = const()[name=string("pn"), val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/pn.bin"), offset=uint64(64)))];
    tensor<fp16, [1, {Dc}, 1, {S}]> core = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{Dc},1,{S}]), x=x)[name=string("core")];
    tensor<fp16, [1, {Dc}, 1, {S}]> z = slice_by_index(begin=tensor<int32, [4]>([0,{Dc},0,0]), end=tensor<int32, [4]>([1,{2*Dc},1,{S}]), x=x)[name=string("z")];
    tensor<fp16, [1, {H}, 1, {S}]> residual = slice_by_index(begin=tensor<int32, [4]>([0,{2*Dc},0,0]), end=tensor<int32, [4]>([1,{self.input},1,{S}]), x=x)[name=string("residual")];
    tensor<fp16, [1, {Dc}, 1, {S}]> csq = mul(x=core, y=core)[name=string("csq")];
    tensor<fp16, [1, 48, 1, {S}]> cms = conv(dilations=dl, groups=g48, pad=pd, pad_type=pt, strides=st, weight=gm, x=csq)[name=string("cms")];
    tensor<fp16, [1, 48, 1, {S}]> cmse = add(x=cms, y=fp16(0x1.0c8p-8))[name=string("cmse")];
    tensor<fp16, [1, 48, 1, {S}]> csd = sqrt(x=cmse)[name=string("csd")];
    tensor<fp16, [1, {Dc}, 1, {S}]> csdr = conv(dilations=dl, groups=g48, pad=pd, pad_type=pt, strides=st, weight=gr, x=csd)[name=string("csdr")];
    tensor<fp16, [1, {Dc}, 1, {S}]> cn = real_div(x=core, y=csdr)[name=string("cn")];
    tensor<fp16, [1, {Dc}, 1, {S}]> cnw = mul(x=cn, y=gn)[name=string("cnw")];
    tensor<fp16, [1, {Dc}, 1, {S}]> nz = mul(x=z, y=fp16(-0x1p+0))[name=string("nz")];
    tensor<fp16, [1, {Dc}, 1, {S}]> ez = exp(x=nz)[name=string("ez")];
    tensor<fp16, [1, {Dc}, 1, {S}]> zd = add(x=ez, y=fp16(0x1p+0))[name=string("zd")];
    tensor<fp16, [1, {Dc}, 1, {S}]> zs = real_div(x=z, y=zd)[name=string("zs")];
    tensor<fp16, [1, {Dc}, 1, {S}]> gated = mul(x=cnw, y=zs)[name=string("gated")];
    tensor<fp16, [1, {H}, 1, {S}]> attn = conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, strides=st, weight=ow, x=gated)[name=string("out_proj")];
    tensor<fp16, [1, {H}, 1, {S}]> h = add(x=residual, y=attn)[name=string("h")];
{post_norm}
    tensor<fp16, [1, {2*I}, 1, {S}]> gu = conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, strides=st, weight=guw, x=hn)[name=string("gu")];
    tensor<fp16, [1, {I}, 1, {S}]> gate = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{I},1,{S}]), x=gu)[name=string("gate")];
    tensor<fp16, [1, {I}, 1, {S}]> up = slice_by_index(begin=tensor<int32, [4]>([0,{I},0,0]), end=tensor<int32, [4]>([1,{2*I},1,{S}]), x=gu)[name=string("up")];
    tensor<fp16, [1, {I}, 1, {S}]> ng = mul(x=gate, y=fp16(-0x1p+0))[name=string("ng")];
    tensor<fp16, [1, {I}, 1, {S}]> eg = exp(x=ng)[name=string("eg")];
    tensor<fp16, [1, {I}, 1, {S}]> gd = add(x=eg, y=fp16(0x1p+0))[name=string("gd")];
    tensor<fp16, [1, {I}, 1, {S}]> gs = real_div(x=gate, y=gd)[name=string("gs")];
    tensor<fp16, [1, {I}, 1, {S}]> act = mul(x=gs, y=up)[name=string("act")];
{down_body}
    tensor<fp16, [1, {H}, 1, {S}]> {'o0' if self.has_next else 'y'} = add(x=h, y=mlp)[name=string("{'o0' if self.has_next else 'y'}")];
{next_body}
  }} -> ({'y, y2' if self.has_next else 'y'});
}}
// pure_ane_gdn_tail_layer{layer}_int{bits}_down{down_proj_parts}
'''
        capture = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(
                mil, blobs, self.input, H, S,
                raw_weight_files=raw_weight_files
            )
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE GDN tail compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        if self.has_next:self.next_surface,self.request=_bind_secondary_output(driver,self.program,self.next_output,self.width)
        self.nbytes = sum(len(x) for x in blobs.values())
        self.compile_seconds = time.time() - t0
        assert_standalone("GDN tail compile")

    def __call__(self, core: np.ndarray, z: np.ndarray,
                 residual: np.ndarray) -> np.ndarray:
        core, lanes = _lane_matrix(core, self.Dc, self.active_lanes)
        z, zlanes = _lane_matrix(z, self.Dc, self.active_lanes)
        residual, rlanes = _lane_matrix(residual, self.H, self.active_lanes)
        if zlanes != lanes or rlanes != lanes:
            raise ValueError("GDN tail lane counts differ")
        with self.driver.view(
            self.program._in_surf, (self.input, self.width), np.float16
        ) as dst:
            dst[:] = 0
            dst[:self.Dc, :lanes] = core
            dst[self.Dc:2*self.Dc, :lanes] = z
            dst[2*self.Dc:, :lanes] = residual
        if self.has_next:_submit_bound(self.driver,self.program,self.request)
        elif not self.driver.engine.submit(self.program, procedure_index=0):raise RuntimeError("ANE GDN tail submission failed")
        with self.driver.view(
            self.program._out_surf, (self.H, self.width), np.float16
        ) as src:
            out = np.array(src[:, :lanes], np.float16)
        if self.has_next:
            with self.driver.view(self.next_surface,(self.next_output,self.width),np.float16) as src:nxt=np.array(src[:,:lanes],np.float16)
        assert_standalone("GDN tail dispatch")
        out=_restore_lane_rank(out,lanes)
        if self.has_next:nxt=_restore_lane_rank(nxt,lanes)
        return (out,nxt) if self.has_next else out


class AneAttentionPrepare:
    """Per-layer Q/K RMSNorm and partial non-traditional RoPE on ANE."""

    def __init__(self, driver: AneDriver, checkpoint: Checkpoint,
                 layer: int, width: int = 32):
        self.driver = driver
        self.layer = layer
        self.width = max(32, width)
        self.Hq, self.Hkv, self.D, self.R = 24, 4, 256, 64
        self.Q, self.K = self.Hq*self.D, self.Hkv*self.D
        # q, k, then full per-channel cos/sin arrays for both.  Supplying the
        # position table as data avoids compiler-rejected rank-axis padding.
        self.input = 3 * (self.Q + self.K)
        self.output = self.Q + self.K
        p = f"model.language_model.layers.{layer}.self_attn"
        qn = checkpoint.tensor(f"{p}.q_norm.weight", np.float16)
        kn = checkpoint.tensor(f"{p}.k_norm.weight", np.float16)
        qperm = np.zeros((self.Q, self.D), np.float16)
        kperm = np.zeros((self.K, self.D), np.float16)
        for head in range(self.Hq):
            base = head*self.D
            for d in range(32):
                qperm[base+d, d+32] = -1
                qperm[base+d+32, d] = 1
        for head in range(self.Hkv):
            base = head*self.D
            for d in range(32):
                kperm[base+d, d+32] = -1
                kperm[base+d+32, d] = 1
        blobs = {
            "qn.bin": np.tile(qn, self.Hq).astype(np.float16).tobytes(),
            "kn.bin": np.tile(kn, self.Hkv).astype(np.float16).tobytes(),
            "qm.bin": np.full((self.Hq, self.D), 1/self.D, np.float16).tobytes(),
            "km.bin": np.full((self.Hkv, self.D), 1/self.D, np.float16).tobytes(),
            "qr.bin": np.ones((self.Q, 1), np.float16).tobytes(),
            "kr.bin": np.ones((self.K, 1), np.float16).tobytes(),
            "qp.bin": qperm.tobytes(),
            "kp.bin": kperm.tobytes(),
        }
        Q, K, Hq, Hk, D, R, S = (
            self.Q, self.K, self.Hq, self.Hkv, self.D, self.R, self.width
        )
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {self.input}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    tensor<fp16, [1, {Q}, 1, 1]> qnw = const()[name=string("qnw"), val=tensor<fp16, [1, {Q}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/qn.bin"), offset=uint64(64)))];
    tensor<fp16, [1, {K}, 1, 1]> knw = const()[name=string("knw"), val=tensor<fp16, [1, {K}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/kn.bin"), offset=uint64(64)))];
    tensor<fp16, [{Hq}, {D}, 1, 1]> qmw = const()[name=string("qmw"), val=tensor<fp16, [{Hq}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/qm.bin"), offset=uint64(64)))];
    tensor<fp16, [{Hk}, {D}, 1, 1]> kmw = const()[name=string("kmw"), val=tensor<fp16, [{Hk}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/km.bin"), offset=uint64(64)))];
    tensor<fp16, [{Q}, 1, 1, 1]> qrw = const()[name=string("qrw"), val=tensor<fp16, [{Q}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/qr.bin"), offset=uint64(64)))];
    tensor<fp16, [{K}, 1, 1, 1]> krw = const()[name=string("krw"), val=tensor<fp16, [{K}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/kr.bin"), offset=uint64(64)))];
    tensor<fp16, [{Q}, {D}, 1, 1]> qpw = const()[name=string("qpw"), val=tensor<fp16, [{Q}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/qp.bin"), offset=uint64(64)))];
    tensor<fp16, [{K}, {D}, 1, 1]> kpw = const()[name=string("kpw"), val=tensor<fp16, [{K}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/kp.bin"), offset=uint64(64)))];
    tensor<fp16, [1, {Q}, 1, {S}]> q0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{Q},1,{S}]), x=x)[name=string("q0")];
    tensor<fp16, [1, {K}, 1, {S}]> k0 = slice_by_index(begin=tensor<int32, [4]>([0,{Q},0,0]), end=tensor<int32, [4]>([1,{Q+K},1,{S}]), x=x)[name=string("k0")];
    tensor<fp16, [1, {Q}, 1, {S}]> qc = slice_by_index(begin=tensor<int32, [4]>([0,{Q+K},0,0]), end=tensor<int32, [4]>([1,{2*Q+K},1,{S}]), x=x)[name=string("qc")];
    tensor<fp16, [1, {Q}, 1, {S}]> qs = slice_by_index(begin=tensor<int32, [4]>([0,{2*Q+K},0,0]), end=tensor<int32, [4]>([1,{3*Q+K},1,{S}]), x=x)[name=string("qs")];
    tensor<fp16, [1, {K}, 1, {S}]> kc = slice_by_index(begin=tensor<int32, [4]>([0,{3*Q+K},0,0]), end=tensor<int32, [4]>([1,{3*Q+2*K},1,{S}]), x=x)[name=string("kc")];
    tensor<fp16, [1, {K}, 1, {S}]> ks = slice_by_index(begin=tensor<int32, [4]>([0,{3*Q+2*K},0,0]), end=tensor<int32, [4]>([1,{3*Q+3*K},1,{S}]), x=x)[name=string("ks")];
    tensor<fp16, [1, {Q}, 1, {S}]> q8 = mul(x=q0, y=fp16(0x1p+3))[name=string("q8")];
    tensor<fp16, [1, {Q}, 1, {S}]> qsq = mul(x=q8, y=q8)[name=string("qsq")];
    tensor<fp16, [1, {Hq}, 1, {S}]> qms = conv(dilations=dl, groups=int32({Hq}), pad=pd, pad_type=pt, strides=st, weight=qmw, x=qsq)[name=string("qms")];
    tensor<fp16, [1, {Hq}, 1, {S}]> qmse = add(x=qms, y=fp16(0x1.0c8p-14))[name=string("qmse")];
    tensor<fp16, [1, {Hq}, 1, {S}]> qsd = sqrt(x=qmse)[name=string("qsd")];
    tensor<fp16, [1, {Q}, 1, {S}]> qsdr = conv(dilations=dl, groups=int32({Hq}), pad=pd, pad_type=pt, strides=st, weight=qrw, x=qsd)[name=string("qsdr")];
    tensor<fp16, [1, {Q}, 1, {S}]> qu = real_div(x=q8, y=qsdr)[name=string("qu")];
    tensor<fp16, [1, {Q}, 1, {S}]> qn = mul(x=qu, y=qnw)[name=string("qn")];
    tensor<fp16, [1, {K}, 1, {S}]> k8 = mul(x=k0, y=fp16(0x1p+3))[name=string("k8")];
    tensor<fp16, [1, {K}, 1, {S}]> ksq = mul(x=k8, y=k8)[name=string("ksq")];
    tensor<fp16, [1, {Hk}, 1, {S}]> kms = conv(dilations=dl, groups=int32({Hk}), pad=pd, pad_type=pt, strides=st, weight=kmw, x=ksq)[name=string("kms")];
    tensor<fp16, [1, {Hk}, 1, {S}]> kmse = add(x=kms, y=fp16(0x1.0c8p-14))[name=string("kmse")];
    tensor<fp16, [1, {Hk}, 1, {S}]> ksd = sqrt(x=kmse)[name=string("ksd")];
    tensor<fp16, [1, {K}, 1, {S}]> ksdr = conv(dilations=dl, groups=int32({Hk}), pad=pd, pad_type=pt, strides=st, weight=krw, x=ksd)[name=string("ksdr")];
    tensor<fp16, [1, {K}, 1, {S}]> ku = real_div(x=k8, y=ksdr)[name=string("ku")];
    tensor<fp16, [1, {K}, 1, {S}]> kn = mul(x=ku, y=knw)[name=string("kn")];
    tensor<fp16, [1, {Q}, 1, {S}]> qswap = conv(dilations=dl, groups=int32({Hq}), pad=pd, pad_type=pt, strides=st, weight=qpw, x=qn)[name=string("qswap")];
    tensor<fp16, [1, {K}, 1, {S}]> kswap = conv(dilations=dl, groups=int32({Hk}), pad=pd, pad_type=pt, strides=st, weight=kpw, x=kn)[name=string("kswap")];
    tensor<fp16, [1, {Q}, 1, {S}]> qcos = mul(x=qn, y=qc)[name=string("qcos")];
    tensor<fp16, [1, {Q}, 1, {S}]> qsin = mul(x=qswap, y=qs)[name=string("qsin")];
    tensor<fp16, [1, {Q}, 1, {S}]> qrot = add(x=qcos, y=qsin)[name=string("qrot")];
    tensor<fp16, [1, {K}, 1, {S}]> kcos = mul(x=kn, y=kc)[name=string("kcos")];
    tensor<fp16, [1, {K}, 1, {S}]> ksin = mul(x=kswap, y=ks)[name=string("ksin")];
    tensor<fp16, [1, {K}, 1, {S}]> krot = add(x=kcos, y=ksin)[name=string("krot")];
    tensor<int32, [8]> pq = const()[name=string("pq"), val=tensor<int32, [8]>([0,0,0,{K},0,0,0,0])];
    tensor<int32, [8]> pk = const()[name=string("pk"), val=tensor<int32, [8]>([0,0,{Q},0,0,0,0,0])];
    tensor<fp16, [1, {Q+K}, 1, {S}]> qfp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pq, x=qrot)[name=string("qfp")];
    tensor<fp16, [1, {Q+K}, 1, {S}]> kfp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pk, x=krot)[name=string("kfp")];
    tensor<fp16, [1, {Q+K}, 1, {S}]> y = add(x=qfp, y=kfp)[name=string("y")];
  }} -> (y);
}}
// pure_ane_attention_prepare_layer{layer}
'''
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(
                mil, blobs, self.input, self.output, S
            )
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE attention prepare compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        assert_standalone("attention prepare compile")

    @staticmethod
    def rope_table(position: int) -> tuple[np.ndarray, np.ndarray]:
        inv = 1.0 / (10_000_000.0 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))
        angle = position * inv
        return np.cos(angle).astype(np.float16), np.sin(angle).astype(np.float16)

    def __call__(self, q: np.ndarray, k: np.ndarray,
                 position: int) -> tuple[np.ndarray, np.ndarray]:
        if q.shape != (self.Hq, self.D) or k.shape != (self.Hkv, self.D):
            raise ValueError(f"invalid attention q/k {q.shape}/{k.shape}")
        co, si = self.rope_table(position)
        qcos = np.ones((self.Hq, self.D), np.float16)
        qsin = np.zeros((self.Hq, self.D), np.float16)
        kcos = np.ones((self.Hkv, self.D), np.float16)
        ksin = np.zeros((self.Hkv, self.D), np.float16)
        qcos[:, :32] = co; qcos[:, 32:64] = co
        qsin[:, :32] = si; qsin[:, 32:64] = si
        kcos[:, :32] = co; kcos[:, 32:64] = co
        ksin[:, :32] = si; ksin[:, 32:64] = si
        with self.driver.view(
            self.program._in_surf, (self.input, self.width), np.float16
        ) as dst:
            dst[:self.Q] = q.reshape(-1).astype(np.float16)[:, None]
            dst[self.Q:self.Q+self.K] = k.reshape(-1).astype(np.float16)[:, None]
            dst[self.Q+self.K:2*self.Q+self.K] = qcos.reshape(-1)[:, None]
            dst[2*self.Q+self.K:3*self.Q+self.K] = qsin.reshape(-1)[:, None]
            dst[3*self.Q+self.K:3*self.Q+2*self.K] = kcos.reshape(-1)[:, None]
            dst[3*self.Q+2*self.K:] = ksin.reshape(-1)[:, None]
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE attention prepare submission failed")
        with self.driver.view(
            self.program._out_surf, (self.output, self.width), np.float16
        ) as src:
            out = np.array(src[:, 0], np.float16)
        return out[:self.Q].reshape(self.Hq, self.D), out[self.Q:].reshape(self.Hkv, self.D)


class AneAttentionPrepareMatmul:
    """Compiler-safe two-program attention preprocessing.

    Program one applies learned per-head RMSNorm. Program two applies RoPE as
    a dynamic 256x256 matrix shared by all heads. The complete rotation table
    is precomputed before decode, so the host only selects and copies a table
    row while both normalization and rotation arithmetic execute on ANE.
    """

    def __init__(self, driver: AneDriver, checkpoint: Checkpoint,
                 layer: int, width: int = 32, positions: int = 256):
        self.driver = driver
        self.layer = layer
        self.width = max(32, width)
        self.Hq, self.Hkv, self.D = 24, 4, 256
        self.Q, self.K = self.Hq*self.D, self.Hkv*self.D
        self.flat = self.Q + self.K
        p = f"model.language_model.layers.{layer}.self_attn"
        qn = checkpoint.tensor(f"{p}.q_norm.weight", np.float16)
        kn = checkpoint.tensor(f"{p}.k_norm.weight", np.float16)
        blobs = {
            "qn.bin": np.tile(qn, self.Hq).astype(np.float16).tobytes(),
            "kn.bin": np.tile(kn, self.Hkv).astype(np.float16).tobytes(),
            "qm.bin": np.full((self.Hq, self.D), 1/self.D, np.float16).tobytes(),
            "km.bin": np.full((self.Hkv, self.D), 1/self.D, np.float16).tobytes(),
            "qr.bin": np.ones((self.Q, 1), np.float16).tobytes(),
            "kr.bin": np.ones((self.K, 1), np.float16).tobytes(),
        }
        Q, K, Hq, Hk, D, S = self.Q, self.K, self.Hq, self.Hkv, self.D, self.width
        norm_mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {Q+K}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    tensor<fp16, [1, {Q}, 1, 1]> qnw = const()[name=string("qnw"), val=tensor<fp16, [1, {Q}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/qn.bin"), offset=uint64(64)))];
    tensor<fp16, [1, {K}, 1, 1]> knw = const()[name=string("knw"), val=tensor<fp16, [1, {K}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/kn.bin"), offset=uint64(64)))];
    tensor<fp16, [{Hq}, {D}, 1, 1]> qmw = const()[name=string("qmw"), val=tensor<fp16, [{Hq}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/qm.bin"), offset=uint64(64)))];
    tensor<fp16, [{Hk}, {D}, 1, 1]> kmw = const()[name=string("kmw"), val=tensor<fp16, [{Hk}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/km.bin"), offset=uint64(64)))];
    tensor<fp16, [{Q}, 1, 1, 1]> qrw = const()[name=string("qrw"), val=tensor<fp16, [{Q}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/qr.bin"), offset=uint64(64)))];
    tensor<fp16, [{K}, 1, 1, 1]> krw = const()[name=string("krw"), val=tensor<fp16, [{K}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/kr.bin"), offset=uint64(64)))];
    tensor<fp16, [1, {Q}, 1, {S}]> q0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{Q},1,{S}]), x=x)[name=string("q0")];
    tensor<fp16, [1, {K}, 1, {S}]> k0 = slice_by_index(begin=tensor<int32, [4]>([0,{Q},0,0]), end=tensor<int32, [4]>([1,{Q+K},1,{S}]), x=x)[name=string("k0")];
    tensor<fp16, [1, {Q}, 1, {S}]> q8 = mul(x=q0, y=fp16(0x1p+3))[name=string("q8")];
    tensor<fp16, [1, {Q}, 1, {S}]> qsq = mul(x=q8, y=q8)[name=string("qsq")];
    tensor<fp16, [1, {Hq}, 1, {S}]> qms = conv(dilations=dl, groups=int32({Hq}), pad=pd, pad_type=pt, strides=st, weight=qmw, x=qsq)[name=string("qms")];
    tensor<fp16, [1, {Hq}, 1, {S}]> qmse = add(x=qms, y=fp16(0x1.0c8p-14))[name=string("qmse")];
    tensor<fp16, [1, {Hq}, 1, {S}]> qsd = sqrt(x=qmse)[name=string("qsd")];
    tensor<fp16, [1, {Q}, 1, {S}]> qsdr = conv(dilations=dl, groups=int32({Hq}), pad=pd, pad_type=pt, strides=st, weight=qrw, x=qsd)[name=string("qsdr")];
    tensor<fp16, [1, {Q}, 1, {S}]> qu = real_div(x=q8, y=qsdr)[name=string("qu")];
    tensor<fp16, [1, {Q}, 1, {S}]> qnorm = mul(x=qu, y=qnw)[name=string("qnorm")];
    tensor<fp16, [1, {K}, 1, {S}]> k8 = mul(x=k0, y=fp16(0x1p+3))[name=string("k8")];
    tensor<fp16, [1, {K}, 1, {S}]> ksq = mul(x=k8, y=k8)[name=string("ksq")];
    tensor<fp16, [1, {Hk}, 1, {S}]> kms = conv(dilations=dl, groups=int32({Hk}), pad=pd, pad_type=pt, strides=st, weight=kmw, x=ksq)[name=string("kms")];
    tensor<fp16, [1, {Hk}, 1, {S}]> kmse = add(x=kms, y=fp16(0x1.0c8p-14))[name=string("kmse")];
    tensor<fp16, [1, {Hk}, 1, {S}]> ksd = sqrt(x=kmse)[name=string("ksd")];
    tensor<fp16, [1, {K}, 1, {S}]> ksdr = conv(dilations=dl, groups=int32({Hk}), pad=pd, pad_type=pt, strides=st, weight=krw, x=ksd)[name=string("ksdr")];
    tensor<fp16, [1, {K}, 1, {S}]> ku = real_div(x=k8, y=ksdr)[name=string("ku")];
    tensor<fp16, [1, {K}, 1, {S}]> knorm = mul(x=ku, y=knw)[name=string("knorm")];
    tensor<int32, [8]> pq = const()[name=string("pq"), val=tensor<int32, [8]>([0,0,0,{K},0,0,0,0])];
    tensor<int32, [8]> pk = const()[name=string("pk"), val=tensor<int32, [8]>([0,0,{Q},0,0,0,0,0])];
    tensor<fp16, [1, {Q+K}, 1, {S}]> qp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pq, x=qnorm)[name=string("qp")];
    tensor<fp16, [1, {Q+K}, 1, {S}]> kp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pk, x=knorm)[name=string("kp")];
    tensor<fp16, [1, {Q+K}, 1, {S}]> y = add(x=qp, y=kp)[name=string("y")];
  }} -> (y);
}}
// pure_ane_attention_norm_layer{layer}
'''
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.norm_program = driver.engine.compile_multiproc(
                norm_mil, blobs, self.flat, self.flat, S
            )
        if self.norm_program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE attention norm compile failed:\n{tail}")
        driver.engine._ensure_io(self.norm_program)

        C = self.Hq + self.Hkv + self.D
        rope_mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {D}]> x) {{
    tensor<fp16, [1, {Hq}, 1, {D}]> q0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{Hq},1,{D}]), x=x)[name=string("q0")];
    tensor<fp16, [1, {Hk}, 1, {D}]> k0 = slice_by_index(begin=tensor<int32, [4]>([0,{Hq},0,0]), end=tensor<int32, [4]>([1,{Hq+Hk},1,{D}]), x=x)[name=string("k0")];
    tensor<fp16, [1, {D}, 1, {D}]> r0 = slice_by_index(begin=tensor<int32, [4]>([0,{Hq+Hk},0,0]), end=tensor<int32, [4]>([1,{C},1,{D}]), x=x)[name=string("r0")];
    tensor<fp16, [1, 1, {Hq}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,1,{Hq},{D}]), x=q0)[name=string("q")];
    tensor<fp16, [1, 1, {Hk}, {D}]> k = reshape(shape=tensor<int32, [4]>([1,1,{Hk},{D}]), x=k0)[name=string("k")];
    tensor<fp16, [1, 1, {D}, {D}]> r = reshape(shape=tensor<int32, [4]>([1,1,{D},{D}]), x=r0)[name=string("r")];
    tensor<fp16, [1, 1, {Hq}, {D}]> qr = matmul(transpose_x=bool(false), transpose_y=bool(true), x=q, y=r)[name=string("qr")];
    tensor<fp16, [1, 1, {Hk}, {D}]> kr = matmul(transpose_x=bool(false), transpose_y=bool(true), x=k, y=r)[name=string("kr")];
    tensor<fp16, [1, {Hq}, 1, {D}]> qf = reshape(shape=tensor<int32, [4]>([1,{Hq},1,{D}]), x=qr)[name=string("qf")];
    tensor<fp16, [1, {Hk}, 1, {D}]> kf = reshape(shape=tensor<int32, [4]>([1,{Hk},1,{D}]), x=kr)[name=string("kf")];
    tensor<int32, [8]> pq = const()[name=string("pq"), val=tensor<int32, [8]>([0,0,0,{Hk},0,0,0,0])];
    tensor<int32, [8]> pk = const()[name=string("pk"), val=tensor<int32, [8]>([0,0,{Hq},0,0,0,0,0])];
    tensor<fp16, [1, {Hq+Hk}, 1, {D}]> qp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pq, x=qf)[name=string("qp")];
    tensor<fp16, [1, {Hq+Hk}, 1, {D}]> kp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pk, x=kf)[name=string("kp")];
    tensor<fp16, [1, {Hq+Hk}, 1, {D}]> y = add(x=qp, y=kp)[name=string("y")];
  }} -> (y);
}}
// pure_ane_rope_dynamic
'''
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.rope_program = driver.engine.compile_multiproc(
                rope_mil, {}, C, self.Hq+self.Hkv, D
            )
        if self.rope_program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE RoPE compile failed:\n{tail}")
        driver.engine._ensure_io(self.rope_program)

        identity = np.eye(self.D, dtype=np.float16)
        self.rotations = np.empty((positions, self.D, self.D), np.float16)
        for position in range(positions):
            co, si = AneAttentionPrepare.rope_table(position)
            r = identity.copy()
            for d in range(32):
                r[d, d] = co[d]; r[d, d+32] = -si[d]
                r[d+32, d] = si[d]; r[d+32, d+32] = co[d]
            self.rotations[position] = r
        assert_standalone("attention norm and RoPE compile")

    def __call__(self, q: np.ndarray, k: np.ndarray,
                 position: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= position < len(self.rotations):
            raise RuntimeError(f"RoPE position {position} exceeds precomputed table")
        with self.driver.view(
            self.norm_program._in_surf, (self.flat, self.width), np.float16
        ) as dst:
            dst[:self.Q] = q.reshape(-1).astype(np.float16)[:, None]
            dst[self.Q:] = k.reshape(-1).astype(np.float16)[:, None]
        if not self.driver.engine.submit(self.norm_program, procedure_index=0):
            raise RuntimeError("ANE attention norm submission failed")
        with self.driver.view(
            self.norm_program._out_surf, (self.flat, self.width), np.float16
        ) as src:
            norm = np.array(src[:, 0], np.float16)
        C = self.Hq + self.Hkv + self.D
        with self.driver.view(
            self.rope_program._in_surf, (C, self.D), np.float16
        ) as dst:
            dst[:self.Hq] = norm[:self.Q].reshape(self.Hq, self.D)
            dst[self.Hq:self.Hq+self.Hkv] = norm[self.Q:].reshape(self.Hkv, self.D)
            dst[self.Hq+self.Hkv:] = self.rotations[position]
        if not self.driver.engine.submit(self.rope_program, procedure_index=0):
            raise RuntimeError("ANE RoPE submission failed")
        with self.driver.view(
            self.rope_program._out_surf, (self.Hq+self.Hkv, self.D), np.float16
        ) as src:
            out = np.array(src, np.float16)
        return out[:self.Hq], out[self.Hq:]


class _AneHeadRmsNorm:
    def __init__(self, driver: AneDriver, weight: np.ndarray,
                 heads: int, dim: int = 256, width: int = 32, tag: str = "q"):
        self.driver, self.heads, self.dim = driver, heads, dim
        self.channels, self.width = heads*dim, max(32, width)
        C, H, D, S = self.channels, heads, dim, self.width
        blobs = {
            "n.bin": np.tile(weight.astype(np.float16), H).tobytes(),
            "m.bin": np.full((H, D), 1/D, np.float16).tobytes(),
            "r.bin": np.ones((C, 1), np.float16).tobytes(),
        }
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    tensor<fp16, [1, {C}, 1, 1]> nw = const()[name=string("nw"), val=tensor<fp16, [1, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/n.bin"), offset=uint64(64)))];
    tensor<fp16, [{H}, {D}, 1, 1]> mw = const()[name=string("mw"), val=tensor<fp16, [{H}, {D}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/m.bin"), offset=uint64(64)))];
    tensor<fp16, [{C}, 1, 1, 1]> rw = const()[name=string("rw"), val=tensor<fp16, [{C}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/r.bin"), offset=uint64(64)))];
    tensor<fp16, [1, {C}, 1, {S}]> x8 = mul(x=x, y=fp16(0x1p+3))[name=string("x8")];
    tensor<fp16, [1, {C}, 1, {S}]> sq = mul(x=x8, y=x8)[name=string("sq")];
    tensor<fp16, [1, {H}, 1, {S}]> ms = conv(dilations=dl, groups=int32({H}), pad=pd, pad_type=pt, strides=st, weight=mw, x=sq)[name=string("ms")];
    tensor<fp16, [1, {H}, 1, {S}]> mse = add(x=ms, y=fp16(0x1.0c8p-14))[name=string("mse")];
    tensor<fp16, [1, {H}, 1, {S}]> sd = sqrt(x=mse)[name=string("sd")];
    tensor<fp16, [1, {C}, 1, {S}]> sdr = conv(dilations=dl, groups=int32({H}), pad=pd, pad_type=pt, strides=st, weight=rw, x=sd)[name=string("sdr")];
    tensor<fp16, [1, {C}, 1, {S}]> unit = real_div(x=x8, y=sdr)[name=string("unit")];
    tensor<fp16, [1, {C}, 1, {S}]> y = mul(x=unit, y=nw)[name=string("y")];
  }} -> (y);
}}
// pure_ane_{tag}_head_norm
'''
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(mil, blobs, C, C, S)
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-8:])
            raise RuntimeError(f"ANE {tag} head norm compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.driver.view(
            self.program._in_surf, (self.channels, self.width), np.float16
        ) as dst:
            dst[:] = x.reshape(-1).astype(np.float16)[:, None]
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE head norm submission failed")
        with self.driver.view(
            self.program._out_surf, (self.channels, self.width), np.float16
        ) as src:
            return np.array(src[:, 0], np.float16).reshape(self.heads, self.dim)


class _AneRopeMatmul:
    def __init__(self, driver: AneDriver, positions: int = 256):
        self.driver = driver
        self.Hq, self.Hk, self.D = 24, 4, 256
        Hq, Hk, D = self.Hq, self.Hk, self.D
        C = Hq + Hk + D
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {D}]> x) {{
    tensor<fp16, [1, {Hq}, 1, {D}]> q0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{Hq},1,{D}]), x=x)[name=string("q0")];
    tensor<fp16, [1, {Hk}, 1, {D}]> k0 = slice_by_index(begin=tensor<int32, [4]>([0,{Hq},0,0]), end=tensor<int32, [4]>([1,{Hq+Hk},1,{D}]), x=x)[name=string("k0")];
    tensor<fp16, [1, {D}, 1, {D}]> r0 = slice_by_index(begin=tensor<int32, [4]>([0,{Hq+Hk},0,0]), end=tensor<int32, [4]>([1,{C},1,{D}]), x=x)[name=string("r0")];
    tensor<fp16, [1, 1, {Hq}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,1,{Hq},{D}]), x=q0)[name=string("q")];
    tensor<fp16, [1, 1, {Hk}, {D}]> k = reshape(shape=tensor<int32, [4]>([1,1,{Hk},{D}]), x=k0)[name=string("k")];
    tensor<fp16, [1, 1, {D}, {D}]> r = reshape(shape=tensor<int32, [4]>([1,1,{D},{D}]), x=r0)[name=string("r")];
    tensor<fp16, [1, 1, {Hq}, {D}]> qr = matmul(transpose_x=bool(false), transpose_y=bool(true), x=q, y=r)[name=string("qr")];
    tensor<fp16, [1, 1, {Hk}, {D}]> kr = matmul(transpose_x=bool(false), transpose_y=bool(true), x=k, y=r)[name=string("kr")];
    tensor<fp16, [1, {Hq}, 1, {D}]> qf = reshape(shape=tensor<int32, [4]>([1,{Hq},1,{D}]), x=qr)[name=string("qf")];
    tensor<fp16, [1, {Hk}, 1, {D}]> kf = reshape(shape=tensor<int32, [4]>([1,{Hk},1,{D}]), x=kr)[name=string("kf")];
    tensor<int32, [8]> pq = const()[name=string("pq"), val=tensor<int32, [8]>([0,0,0,{Hk},0,0,0,0])];
    tensor<int32, [8]> pk = const()[name=string("pk"), val=tensor<int32, [8]>([0,0,{Hq},0,0,0,0,0])];
    tensor<fp16, [1, {Hq+Hk}, 1, {D}]> qp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pq, x=qf)[name=string("qp")];
    tensor<fp16, [1, {Hq+Hk}, 1, {D}]> kp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=pk, x=kf)[name=string("kp")];
    tensor<fp16, [1, {Hq+Hk}, 1, {D}]> y = add(x=qp, y=kp)[name=string("y")];
  }} -> (y);
}}
// pure_ane_rope_matmul
'''
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(mil, {}, C, Hq+Hk, D)
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-8:])
            raise RuntimeError(f"ANE RoPE matmul compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        identity = np.eye(D, dtype=np.float16)
        self.rotations = np.empty((positions, D, D), np.float16)
        for position in range(positions):
            co, si = AneAttentionPrepare.rope_table(position)
            r = identity.copy()
            for d in range(32):
                r[d, d] = co[d]; r[d, d+32] = -si[d]
                r[d+32, d] = si[d]; r[d+32, d+32] = co[d]
            self.rotations[position] = r

    def __call__(self, q: np.ndarray, k: np.ndarray,
                 position: int) -> tuple[np.ndarray, np.ndarray]:
        C = self.Hq + self.Hk + self.D
        with self.driver.view(self.program._in_surf, (C, self.D), np.float16) as dst:
            dst[:self.Hq] = q
            dst[self.Hq:self.Hq+self.Hk] = k
            dst[self.Hq+self.Hk:] = self.rotations[position]
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE RoPE matmul submission failed")
        with self.driver.view(
            self.program._out_surf, (self.Hq+self.Hk, self.D), np.float16
        ) as src:
            out = np.array(src, np.float16)
        return out[:self.Hq], out[self.Hq:]


class AneAttentionPrepareSplit:
    def __init__(self, driver: AneDriver, checkpoint: Checkpoint,
                 layer: int, positions: int = 256,
                 rope: _AneRopeMatmul | None = None):
        p = f"model.language_model.layers.{layer}.self_attn"
        self.qnorm = _AneHeadRmsNorm(
            driver, checkpoint.tensor(f"{p}.q_norm.weight", np.float16), 24, tag="q"
        )
        self.knorm = _AneHeadRmsNorm(
            driver, checkpoint.tensor(f"{p}.k_norm.weight", np.float16), 4, tag="k"
        )
        self.rope = rope if rope is not None else _AneRopeMatmul(driver, positions)
        assert_standalone("split attention preparation compile")

    def __call__(self, q: np.ndarray, k: np.ndarray,
                 position: int) -> tuple[np.ndarray, np.ndarray]:
        return self.rope(self.qnorm(q), self.knorm(k), position)


class AneAttentionPrepareDynamic:
    """One shared ANE program for layer-dynamic Q/K norm weights and RoPE."""
    def __init__(self,driver:AneDriver,positions:int=256):
        self.driver=driver;self.Hq=24;self.Hk=4;self.D=256;C=24+4+2+256;D=256
        self.positions=positions
        mil=f'''program(1.3)
{driver.module._BUILD_INFO}
{{ func main<ios18>(tensor<fp16,[1,{C},1,{D}]> x) {{
 tensor<fp16,[1,24,1,256]> q0=slice_by_index(begin=tensor<int32,[4]>([0,0,0,0]),end=tensor<int32,[4]>([1,24,1,256]),x=x)[name=string("q0")]; tensor<fp16,[1,4,1,256]> k0=slice_by_index(begin=tensor<int32,[4]>([0,24,0,0]),end=tensor<int32,[4]>([1,28,1,256]),x=x)[name=string("k0")]; tensor<fp16,[1,1,1,256]> qw=slice_by_index(begin=tensor<int32,[4]>([0,28,0,0]),end=tensor<int32,[4]>([1,29,1,256]),x=x)[name=string("qw")]; tensor<fp16,[1,1,1,256]> kw=slice_by_index(begin=tensor<int32,[4]>([0,29,0,0]),end=tensor<int32,[4]>([1,30,1,256]),x=x)[name=string("kw")]; tensor<fp16,[1,256,1,256]> r0=slice_by_index(begin=tensor<int32,[4]>([0,30,0,0]),end=tensor<int32,[4]>([1,{C},1,256]),x=x)[name=string("r0")];
 tensor<int32,[1]> ax=const()[name=string("ax"),val=tensor<int32,[1]>([3])]; tensor<fp16,[1,24,1,256]> q8=mul(x=q0,y=fp16(0x1p+3))[name=string("q8")]; tensor<fp16,[1,24,1,256]> qsq=mul(x=q8,y=q8)[name=string("qsq")]; tensor<fp16,[1,24,1,1]> qsum=reduce_sum(axes=ax,keep_dims=bool(true),x=qsq)[name=string("qsum")]; tensor<fp16,[1,24,1,1]> qms=mul(x=qsum,y=fp16(0x1p-8))[name=string("qms")]; tensor<fp16,[1,24,1,1]> qe=add(x=qms,y=fp16(0x1.0c8p-14))[name=string("qe")]; tensor<fp16,[1,24,1,1]> qsd=sqrt(x=qe)[name=string("qsd")]; tensor<fp16,[1,24,1,256]> qu=real_div(x=q8,y=qsd)[name=string("qu")]; tensor<fp16,[1,24,1,256]> qn=mul(x=qu,y=qw)[name=string("qn")];
 tensor<fp16,[1,4,1,256]> k8=mul(x=k0,y=fp16(0x1p+3))[name=string("k8")]; tensor<fp16,[1,4,1,256]> ksq=mul(x=k8,y=k8)[name=string("ksq")]; tensor<fp16,[1,4,1,1]> ksum=reduce_sum(axes=ax,keep_dims=bool(true),x=ksq)[name=string("ksum")]; tensor<fp16,[1,4,1,1]> kms=mul(x=ksum,y=fp16(0x1p-8))[name=string("kms")]; tensor<fp16,[1,4,1,1]> ke=add(x=kms,y=fp16(0x1.0c8p-14))[name=string("ke")]; tensor<fp16,[1,4,1,1]> ksd=sqrt(x=ke)[name=string("ksd")]; tensor<fp16,[1,4,1,256]> ku=real_div(x=k8,y=ksd)[name=string("ku")]; tensor<fp16,[1,4,1,256]> kn=mul(x=ku,y=kw)[name=string("kn")];
 tensor<fp16,[1,1,24,256]> q4=reshape(shape=tensor<int32,[4]>([1,1,24,256]),x=qn)[name=string("q4")]; tensor<fp16,[1,1,4,256]> k4=reshape(shape=tensor<int32,[4]>([1,1,4,256]),x=kn)[name=string("k4")]; tensor<fp16,[1,1,256,256]> r=reshape(shape=tensor<int32,[4]>([1,1,256,256]),x=r0)[name=string("r")]; tensor<fp16,[1,1,24,256]> qr=matmul(transpose_x=bool(false),transpose_y=bool(true),x=q4,y=r)[name=string("qr")]; tensor<fp16,[1,1,4,256]> kr=matmul(transpose_x=bool(false),transpose_y=bool(true),x=k4,y=r)[name=string("kr")]; tensor<fp16,[1,24,1,256]> qf=reshape(shape=tensor<int32,[4]>([1,24,1,256]),x=qr)[name=string("qf")]; tensor<fp16,[1,4,1,256]> kf=reshape(shape=tensor<int32,[4]>([1,4,1,256]),x=kr)[name=string("kf")]; tensor<int32,[8]> pq=const()[name=string("pq"),val=tensor<int32,[8]>([0,0,0,4,0,0,0,0])]; tensor<int32,[8]> pk=const()[name=string("pk"),val=tensor<int32,[8]>([0,0,24,0,0,0,0,0])]; tensor<fp16,[1,28,1,256]> qp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=pq,x=qf)[name=string("qp")]; tensor<fp16,[1,28,1,256]> kp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=pk,x=kf)[name=string("kp")]; tensor<fp16,[1,28,1,256]> y=add(x=qp,y=kp)[name=string("y")]; }} -> (y); }}'''
        self.program=driver.engine.compile_multiproc(mil,{},C,28,D)
        if self.program is None:raise RuntimeError("dynamic attention preparation failed")
        driver.engine._ensure_io(self.program)
        # A dense RoPE matrix is 128 KiB.  Precomputing one for every context
        # position would consume 32 GiB at the checkpoint's 262k-token limit.
        # Decode only needs the positions in the current (up to three-lane)
        # batch, and the same positions are reused by all attention layers.
        self.identity=np.eye(256,dtype=np.float16)
        self.rotations:dict[int,np.ndarray]={}
        self.rotation_order:list[int]=[]
    def _rotation(self,pos:int)->np.ndarray:
        if not 0<=pos<self.positions:
            raise RuntimeError(
                f"RoPE position {pos} exceeds configured context {self.positions}"
            )
        cached=self.rotations.get(pos)
        if cached is not None:return cached
        co,si=AneAttentionPrepare.rope_table(pos);r=self.identity.copy()
        for d in range(32):
            r[d,d]=co[d];r[d,d+32]=-si[d]
            r[d+32,d]=si[d];r[d+32,d+32]=co[d]
        self.rotations[pos]=r;self.rotation_order.append(pos)
        if len(self.rotation_order)>8:
            old=self.rotation_order.pop(0);del self.rotations[old]
        return r
    def run(self,q,k,pos,qw,kw):
        with self.driver.view(self.program._in_surf,(286,256),np.float16) as d:d[:24]=q;d[24:28]=k;d[28]=qw;d[29]=kw;d[30:]=self._rotation(pos)
        if not self.driver.engine.submit(self.program):raise RuntimeError("dynamic attention prepare submit failed")
        with self.driver.view(self.program._out_surf,(28,256),np.float16) as o:y=np.array(o,np.float16)
        return y[:24],y[24:]


class _DynamicPrepareLayer:
    def __init__(self,shared,qw,kw):self.shared=shared;self.qw=qw;self.kw=kw
    def __call__(self,q,k,pos):return self.shared.run(q,k,pos,self.qw,self.kw)


class AneAttentionCore:
    """Shared exact grouped-query attention for a fixed 256-token cache."""

    def __init__(self, driver: AneDriver, length: int = 256):
        self.driver = driver
        self.Hq, self.Hkv, self.D, self.L = 24, 4, 256, length
        H, K, D, L = self.Hq, self.Hkv, self.D, self.L
        self.input = H + 2*K*L + 1
        C = self.input
        k0, k1 = H, H + K*L
        v0, v1 = k1, k1 + K*L
        m0 = v1
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {D}]> x) {{
    tensor<fp16, [1, {H}, 1, {D}]> q4 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{H},1,{D}]), x=x)[name=string("q4")];
    tensor<fp16, [1, {K*L}, 1, {D}]> kf = slice_by_index(begin=tensor<int32, [4]>([0,{k0},0,0]), end=tensor<int32, [4]>([1,{k1},1,{D}]), x=x)[name=string("kf")];
    tensor<fp16, [1, {K*L}, 1, {D}]> vf = slice_by_index(begin=tensor<int32, [4]>([0,{v0},0,0]), end=tensor<int32, [4]>([1,{v1},1,{D}]), x=x)[name=string("vf")];
    tensor<fp16, [1, 1, 1, {L}]> mask = slice_by_index(begin=tensor<int32, [4]>([0,{m0},0,0]), end=tensor<int32, [4]>([1,{m0+1},1,{L}]), x=x)[name=string("mask")];
    tensor<fp16, [1, {K}, {H//K}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,{K},{H//K},{D}]), x=q4)[name=string("q")];
    tensor<fp16, [1, {K}, {L}, {D}]> k = reshape(shape=tensor<int32, [4]>([1,{K},{L},{D}]), x=kf)[name=string("k")];
    tensor<fp16, [1, {K}, {L}, {D}]> v = reshape(shape=tensor<int32, [4]>([1,{K},{L},{D}]), x=vf)[name=string("v")];
    tensor<fp16, [1, {K}, {H//K}, {L}]> rawg = matmul(transpose_x=bool(false), transpose_y=bool(true), x=q, y=k)[name=string("rawg")];
    tensor<fp16, [1, {H}, 1, {L}]> raw = reshape(shape=tensor<int32, [4]>([1,{H},1,{L}]), x=rawg)[name=string("raw")];
    tensor<fp16, [1, {H}, 1, {L}]> scaled = mul(x=raw, y=fp16(0x1p-4))[name=string("scaled")];
    tensor<fp16, [1, {H}, 1, {L}]> scores = add(x=scaled, y=mask)[name=string("scores")];
    tensor<fp16, [1, {H}, 1, {L}]> prob = softmax(axis=int32(-1), x=scores)[name=string("prob")];
    tensor<fp16, [1, {K}, {H//K}, {L}]> pg = reshape(shape=tensor<int32, [4]>([1,{K},{H//K},{L}]), x=prob)[name=string("pg")];
    tensor<fp16, [1, {K}, {H//K}, {D}]> yg = matmul(transpose_x=bool(false), transpose_y=bool(false), x=pg, y=v)[name=string("yg")];
    tensor<fp16, [1, {H}, 1, {D}]> y = reshape(shape=tensor<int32, [4]>([1,{H},1,{D}]), x=yg)[name=string("y")];
  }} -> (y);
}}
// pure_ane_attention_L{L}
'''
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(mil, {}, C, H, D)
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE attention core compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        self.keys = np.zeros((K, L, D), np.float16)
        self.values = np.zeros((K, L, D), np.float16)
        self.offset = 0
        assert_standalone("attention core compile")

    def reset(self) -> None:
        self.keys[:] = 0
        self.values[:] = 0
        self.offset = 0

    def fork_cache(self) -> "AneAttentionCore":
        """Share the loaded program while giving another layer its own KV cache."""
        other=object.__new__(AneAttentionCore)
        other.driver=self.driver
        other.Hq,other.Hkv,other.D,other.L=(self.Hq,self.Hkv,self.D,self.L)
        other.input=self.input; other.program=self.program
        other.keys=np.zeros_like(self.keys); other.values=np.zeros_like(self.values)
        other.offset=0
        return other

    def __call__(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.offset >= self.L:
            raise RuntimeError(f"attention cache exceeds direct limit {self.L}")
        self.keys[:, self.offset] = k
        self.values[:, self.offset] = v
        valid = self.offset + 1
        with self.driver.view(
            self.program._in_surf, (self.input, self.D), np.float16
        ) as dst:
            dst[:] = 0
            dst[:self.Hq] = q
            p = self.Hq
            dst[p:p+self.Hkv*self.L] = self.keys.reshape(-1, self.D); p += self.Hkv*self.L
            dst[p:p+self.Hkv*self.L] = self.values.reshape(-1, self.D); p += self.Hkv*self.L
            dst[p, valid:self.L] = np.float16(-1e4)
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE attention submission failed")
        with self.driver.view(
            self.program._out_surf, (self.Hq, self.D), np.float16
        ) as src:
            out = np.array(src, np.float16)
        self.offset = valid
        assert_standalone("attention dispatch")
        return out


class AneAttentionCombine:
    """Merge two online-softmax accumulators entirely on the ANE."""

    def __init__(self,driver:AneDriver):
        self.driver=driver;self.H,self.D=24,256;H,D=self.H,self.D
        C=6*H
        mil=f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16,[1,{C},1,{D}]> x) {{
    tensor<fp16,[1,{H},1,{D}]> ya=slice_by_index(begin=tensor<int32,[4]>([0,0,0,0]),end=tensor<int32,[4]>([1,{H},1,{D}]),x=x)[name=string("ya")];
    tensor<fp16,[1,{H},1,{D}]> ma=slice_by_index(begin=tensor<int32,[4]>([0,{H},0,0]),end=tensor<int32,[4]>([1,{2*H},1,{D}]),x=x)[name=string("ma")];
    tensor<fp16,[1,{H},1,{D}]> da=slice_by_index(begin=tensor<int32,[4]>([0,{2*H},0,0]),end=tensor<int32,[4]>([1,{3*H},1,{D}]),x=x)[name=string("da")];
    tensor<fp16,[1,{H},1,{D}]> yb=slice_by_index(begin=tensor<int32,[4]>([0,{3*H},0,0]),end=tensor<int32,[4]>([1,{4*H},1,{D}]),x=x)[name=string("yb")];
    tensor<fp16,[1,{H},1,{D}]> mb=slice_by_index(begin=tensor<int32,[4]>([0,{4*H},0,0]),end=tensor<int32,[4]>([1,{5*H},1,{D}]),x=x)[name=string("mb")];
    tensor<fp16,[1,{H},1,{D}]> db=slice_by_index(begin=tensor<int32,[4]>([0,{5*H},0,0]),end=tensor<int32,[4]>([1,{6*H},1,{D}]),x=x)[name=string("db")];
    tensor<fp16,[1,{H},1,{D}]> gm=maximum(x=ma,y=mb)[name=string("gm")];
    tensor<fp16,[1,{H},1,{D}]> dma=sub(x=ma,y=gm)[name=string("dma")];
    tensor<fp16,[1,{H},1,{D}]> dmb=sub(x=mb,y=gm)[name=string("dmb")];
    tensor<fp16,[1,{H},1,{D}]> ea=exp(x=dma)[name=string("ea")];
    tensor<fp16,[1,{H},1,{D}]> eb=exp(x=dmb)[name=string("eb")];
    tensor<fp16,[1,{H},1,{D}]> wa=mul(x=da,y=ea)[name=string("wa")];
    tensor<fp16,[1,{H},1,{D}]> wb=mul(x=db,y=eb)[name=string("wb")];
    tensor<fp16,[1,{H},1,{D}]> den=add(x=wa,y=wb)[name=string("den")];
    tensor<fp16,[1,{H},1,{D}]> na=mul(x=ya,y=wa)[name=string("na")];
    tensor<fp16,[1,{H},1,{D}]> nb=mul(x=yb,y=wb)[name=string("nb")];
    tensor<fp16,[1,{H},1,{D}]> num=add(x=na,y=nb)[name=string("num")];
    tensor<fp16,[1,{H},1,{D}]> y=real_div(x=num,y=den)[name=string("y")];
    tensor<int32,[8]> py=const()[name=string("py"),val=tensor<int32,[8]>([0,0,0,{2*H},0,0,0,0])];
    tensor<int32,[8]> pm=const()[name=string("pm"),val=tensor<int32,[8]>([0,0,{H},{H},0,0,0,0])];
    tensor<int32,[8]> pd=const()[name=string("pd"),val=tensor<int32,[8]>([0,0,{2*H},0,0,0,0,0])];
    tensor<fp16,[1,{3*H},1,{D}]> yp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=py,x=y)[name=string("yp")];
    tensor<fp16,[1,{3*H},1,{D}]> mp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=pm,x=gm)[name=string("mp")];
    tensor<fp16,[1,{3*H},1,{D}]> dp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=pd,x=den)[name=string("dp")];
    tensor<fp16,[1,{3*H},1,{D}]> ym=add(x=yp,y=mp)[name=string("ym")];
    tensor<fp16,[1,{3*H},1,{D}]> out=add(x=ym,y=dp)[name=string("out")];
  }} -> (out);
}}
// pure_ane_attention_online_softmax_combine
'''
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
            self.program=driver.engine.compile_multiproc(mil,{},C,3*H,D)
        if self.program is None:
            tail="\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE attention combine compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)

    def __call__(self,left:np.ndarray,right:np.ndarray)->np.ndarray:
        if left.shape!=(3*self.H,self.D) or right.shape!=left.shape:
            raise ValueError(f"invalid online-softmax states {left.shape}/{right.shape}")
        with self.driver.view(
            self.program._in_surf,(6*self.H,self.D),np.float16
        ) as dst:
            dst[:3*self.H]=left;dst[3*self.H:]=right
        if not self.driver.engine.submit(self.program):
            raise RuntimeError("ANE attention combine submission failed")
        with self.driver.view(
            self.program._out_surf,(3*self.H,self.D),np.float16
        ) as src:
            return np.array(src,np.float16)


class AneAttentionStreamGroup:
    """Scan 2-32 independent 256-token blocks in one ANE submission."""

    DEN_SCALE=8192

    def __init__(self,driver:AneDriver,blocks:int,block:int=256):
        if blocks not in (2,4,8,16,32):
            raise ValueError(f"unsupported attention stream group {blocks}")
        self.driver=driver;self.N=blocks;self.B=block
        self.H,self.K,self.D=24,4,256
        N,B,H,K,D=self.N,self.B,self.H,self.K,self.D
        G=N*K;HH=N*H
        self.input=HH+2*G*B+G;C=self.input
        k0,k1=HH,HH+G*B;v0,v1=k1,k1+G*B;m0=v1
        mil=f'''program(1.3)
{driver.module._BUILD_INFO}
{{ func main<ios18>(tensor<fp16,[1,{C},1,{D}]> x) {{
 tensor<fp16,[1,{HH},1,{D}]> qf=slice_by_index(begin=tensor<int32,[4]>([0,0,0,0]),end=tensor<int32,[4]>([1,{HH},1,{D}]),x=x)[name=string("qf")];
 tensor<fp16,[1,{G*B},1,{D}]> kf=slice_by_index(begin=tensor<int32,[4]>([0,{k0},0,0]),end=tensor<int32,[4]>([1,{k1},1,{D}]),x=x)[name=string("kf")];
 tensor<fp16,[1,{G*B},1,{D}]> vf=slice_by_index(begin=tensor<int32,[4]>([0,{v0},0,0]),end=tensor<int32,[4]>([1,{v1},1,{D}]),x=x)[name=string("vf")];
 tensor<fp16,[1,{G},1,{B}]> mask=slice_by_index(begin=tensor<int32,[4]>([0,{m0},0,0]),end=tensor<int32,[4]>([1,{m0+G},1,{B}]),x=x)[name=string("mask")];
 tensor<fp16,[1,{G},{H//K},{D}]> q=reshape(shape=tensor<int32,[4]>([1,{G},{H//K},{D}]),x=qf)[name=string("q")];
 tensor<fp16,[1,{G},{B},{D}]> k=reshape(shape=tensor<int32,[4]>([1,{G},{B},{D}]),x=kf)[name=string("k")];
 tensor<fp16,[1,{G},{B},{D}]> v=reshape(shape=tensor<int32,[4]>([1,{G},{B},{D}]),x=vf)[name=string("v")];
 tensor<fp16,[1,{G},{H//K},{B}]> raw=matmul(transpose_x=bool(false),transpose_y=bool(true),x=q,y=k)[name=string("raw")];
 tensor<fp16,[1,{G},{H//K},{B}]> scaled=mul(x=raw,y=fp16(0x1p-4))[name=string("scaled")];
 tensor<fp16,[1,{G},{H//K},{B}]> score=add(x=scaled,y=mask)[name=string("score")];
 tensor<int32,[1]> ax3=const()[name=string("ax3"),val=tensor<int32,[1]>([3])];
 tensor<fp16,[1,{G},{H//K},1]> mx=reduce_max(axes=ax3,keep_dims=bool(true),x=score)[name=string("mx")];
 tensor<fp16,[1,{G},{H//K},{B}]> centered=sub(x=score,y=mx)[name=string("centered")];
 tensor<fp16,[1,{G},{H//K},{B}]> ex=exp(x=centered)[name=string("ex")];
 tensor<fp16,[1,{G},{H//K},1]> den=reduce_sum(axes=ax3,keep_dims=bool(true),x=ex)[name=string("den")];
 tensor<fp16,[1,{G},{H//K},{D}]> num=matmul(transpose_x=bool(false),transpose_y=bool(false),x=ex,y=v)[name=string("num")];
 tensor<fp16,[1,{G},{H//K},{D}]> yn=real_div(x=num,y=den)[name=string("yn")];
 tensor<fp16,[1,{N},{H},{D}]> yb=reshape(shape=tensor<int32,[4]>([1,{N},{H},{D}]),x=yn)[name=string("yb")];
 tensor<fp16,[1,{N},{H},1]> mb=reshape(shape=tensor<int32,[4]>([1,{N},{H},1]),x=mx)[name=string("mb")];
 tensor<fp16,[1,{N},{H},1]> db=reshape(shape=tensor<int32,[4]>([1,{N},{H},1]),x=den)[name=string("db")];
 tensor<int32,[1]> ax1=const()[name=string("ax1"),val=tensor<int32,[1]>([1])];
 tensor<fp16,[1,1,{H},1]> gm=reduce_max(axes=ax1,keep_dims=bool(true),x=mb)[name=string("gm")];
 tensor<fp16,[1,{N},{H},1]> dm=sub(x=mb,y=gm)[name=string("dm")];
 tensor<fp16,[1,{N},{H},1]> ew=exp(x=dm)[name=string("ew")];
 tensor<fp16,[1,{N},{H},1]> weights=mul(x=db,y=ew)[name=string("weights")];
 tensor<fp16,[1,1,{H},1]> dt=reduce_sum(axes=ax1,keep_dims=bool(true),x=weights)[name=string("dt")];
 tensor<fp16,[1,{N},{H},{D}]> yw=mul(x=yb,y=weights)[name=string("yw")];
 tensor<fp16,[1,1,{H},{D}]> nt=reduce_sum(axes=ax1,keep_dims=bool(true),x=yw)[name=string("nt")];
 tensor<fp16,[1,1,{H},{D}]> yt=real_div(x=nt,y=dt)[name=string("yt")];
 tensor<fp16,[1,{H},1,{D}]> y=reshape(shape=tensor<int32,[4]>([1,{H},1,{D}]),x=yt)[name=string("y")];
 tensor<fp16,[1,{H},1,1]> gm4=reshape(shape=tensor<int32,[4]>([1,{H},1,1]),x=gm)[name=string("gm4")];
 tensor<fp16,[1,{H},1,1]> dt4=reshape(shape=tensor<int32,[4]>([1,{H},1,1]),x=dt)[name=string("dt4")];
 tensor<fp16,[1,{H},1,{D}]> zero=mul(x=y,y=fp16(0x0p+0))[name=string("zero")];
 tensor<fp16,[1,{H},1,{D}]> one=add(x=zero,y=fp16(0x1p+0))[name=string("one")];
 tensor<fp16,[1,{H},1,{D}]> mw=mul(x=gm4,y=one)[name=string("mw")];
 tensor<fp16,[1,{H},1,1]> ds=mul(x=dt4,y=fp16(0x1p-13))[name=string("ds")];
 tensor<fp16,[1,{H},1,{D}]> dw=mul(x=ds,y=one)[name=string("dw")];
 tensor<int32,[8]> py=const()[name=string("py"),val=tensor<int32,[8]>([0,0,0,{2*H},0,0,0,0])];
 tensor<int32,[8]> pm=const()[name=string("pm"),val=tensor<int32,[8]>([0,0,{H},{H},0,0,0,0])];
 tensor<int32,[8]> pd=const()[name=string("pd"),val=tensor<int32,[8]>([0,0,{2*H},0,0,0,0,0])];
 tensor<fp16,[1,{3*H},1,{D}]> yp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=py,x=y)[name=string("yp")];
 tensor<fp16,[1,{3*H},1,{D}]> mp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=pm,x=mw)[name=string("mp")];
 tensor<fp16,[1,{3*H},1,{D}]> dp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=pd,x=dw)[name=string("dp")];
 tensor<fp16,[1,{3*H},1,{D}]> ym=add(x=yp,y=mp)[name=string("ym")];
 tensor<fp16,[1,{3*H},1,{D}]> out=add(x=ym,y=dp)[name=string("out")];
 }} -> (out); }}
// pure_ane_attention_stream_blocks{N}
'''
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
            self.program=driver.engine.compile_multiproc(mil,{},C,3*H,D)
        if self.program is None:
            tail="\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE {N}-block attention compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)

    def __call__(self,q:np.ndarray,keys:np.ndarray,values:np.ndarray,
                 last_valid:int)->np.ndarray:
        expected=(self.N,self.K,self.B,self.D)
        if keys.shape!=expected or values.shape!=expected:
            raise ValueError(f"invalid block group {keys.shape}/{values.shape}")
        with self.driver.view(
            self.program._in_surf,(self.input,self.D),np.float16
        ) as dst:
            dst[:]=0;dst[:self.N*self.H]=np.tile(q,(self.N,1))
            p=self.N*self.H;nkv=self.N*self.K*self.B
            dst[p:p+nkv]=keys.reshape(-1,self.D);p+=nkv
            dst[p:p+nkv]=values.reshape(-1,self.D);p+=nkv
            dst[p+(self.N-1)*self.K:p+self.N*self.K,
                last_valid:self.B]=np.float16(-1e4)
        if not self.driver.engine.submit(self.program):
            raise RuntimeError(f"ANE {self.N}-block attention submission failed")
        with self.driver.view(
            self.program._out_surf,(3*self.H,self.D),np.float16
        ) as src:
            return np.array(src,np.float16)


class AneLongContextAttentionCore:
    """Exact causal attention beyond 256 tokens using reusable ANE blocks.

    The direct softmax program is retained for the first block so existing
    short-context output remains bit-for-bit stable.  Later positions stream
    all populated 256-token KV blocks through the ANE and merge their online
    softmax states with a second weight-free ANE program.  The CPU only moves
    buffers and schedules dispatches; it performs no attention arithmetic.
    """

    def __init__(self,driver:AneDriver,length:int,block:int=256,
                 group_sizes:tuple[int,...]=(4,16,32)):
        if length<=block:
            raise ValueError("long-context attention requires length > block")
        self.driver=driver;self.Hq,self.Hkv,self.D=24,4,256
        self.L,self.B=length,block
        self.capacity=-(-length//block)*block
        direct=AneAttentionCore(driver,block)
        self.direct_program=direct.program;self.direct_input=direct.input
        H,K,D,B=self.Hq,self.Hkv,self.D,self.B
        self.input=H+2*K*B+1;C=self.input
        k0,k1=H,H+K*B;v0,v1=k1,k1+K*B;m0=v1
        mil=f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16,[1,{C},1,{D}]> x) {{
    tensor<fp16,[1,{H},1,{D}]> q4=slice_by_index(begin=tensor<int32,[4]>([0,0,0,0]),end=tensor<int32,[4]>([1,{H},1,{D}]),x=x)[name=string("q4")];
    tensor<fp16,[1,{K*B},1,{D}]> kf=slice_by_index(begin=tensor<int32,[4]>([0,{k0},0,0]),end=tensor<int32,[4]>([1,{k1},1,{D}]),x=x)[name=string("kf")];
    tensor<fp16,[1,{K*B},1,{D}]> vf=slice_by_index(begin=tensor<int32,[4]>([0,{v0},0,0]),end=tensor<int32,[4]>([1,{v1},1,{D}]),x=x)[name=string("vf")];
    tensor<fp16,[1,1,1,{B}]> mask=slice_by_index(begin=tensor<int32,[4]>([0,{m0},0,0]),end=tensor<int32,[4]>([1,{m0+1},1,{B}]),x=x)[name=string("mask")];
    tensor<fp16,[1,{K},{H//K},{D}]> q=reshape(shape=tensor<int32,[4]>([1,{K},{H//K},{D}]),x=q4)[name=string("q")];
    tensor<fp16,[1,{K},{B},{D}]> k=reshape(shape=tensor<int32,[4]>([1,{K},{B},{D}]),x=kf)[name=string("k")];
    tensor<fp16,[1,{K},{B},{D}]> v=reshape(shape=tensor<int32,[4]>([1,{K},{B},{D}]),x=vf)[name=string("v")];
    tensor<fp16,[1,{K},{H//K},{B}]> raw=matmul(transpose_x=bool(false),transpose_y=bool(true),x=q,y=k)[name=string("raw")];
    tensor<fp16,[1,{K},{H//K},{B}]> scaled=mul(x=raw,y=fp16(0x1.0p-4))[name=string("scaled")];
    tensor<fp16,[1,{K},{H//K},{B}]> score=add(x=scaled,y=mask)[name=string("score")];
    tensor<int32,[1]> ax=const()[name=string("ax"),val=tensor<int32,[1]>([3])];
    tensor<fp16,[1,{K},{H//K},1]> mx=reduce_max(axes=ax,keep_dims=bool(true),x=score)[name=string("mx")];
    tensor<fp16,[1,{K},{H//K},{B}]> centered=sub(x=score,y=mx)[name=string("centered")];
    tensor<fp16,[1,{K},{H//K},{B}]> ex=exp(x=centered)[name=string("ex")];
    tensor<fp16,[1,{K},{H//K},1]> den=reduce_sum(axes=ax,keep_dims=bool(true),x=ex)[name=string("den")];
    tensor<fp16,[1,{K},{H//K},{D}]> num=matmul(transpose_x=bool(false),transpose_y=bool(false),x=ex,y=v)[name=string("num")];
    tensor<fp16,[1,{K},{H//K},{D}]> yg=real_div(x=num,y=den)[name=string("yg")];
    tensor<fp16,[1,{H},1,{D}]> y0=reshape(shape=tensor<int32,[4]>([1,{H},1,{D}]),x=yg)[name=string("y0")];
    tensor<fp16,[1,{H},1,1]> mx4=reshape(shape=tensor<int32,[4]>([1,{H},1,1]),x=mx)[name=string("mx4")];
    tensor<fp16,[1,{H},1,1]> dn4=reshape(shape=tensor<int32,[4]>([1,{H},1,1]),x=den)[name=string("dn4")];
    tensor<fp16,[1,{H},1,{D}]> zero=mul(x=q4,y=fp16(0x0p+0))[name=string("zero")];
    tensor<fp16,[1,{H},1,{D}]> one=add(x=zero,y=fp16(0x1p+0))[name=string("one")];
    tensor<fp16,[1,{H},1,{D}]> mw=mul(x=mx4,y=one)[name=string("mw")];
    tensor<fp16,[1,{H},1,1]> ds=mul(x=dn4,y=fp16(0x1p-13))[name=string("ds")];
    tensor<fp16,[1,{H},1,{D}]> dw=mul(x=ds,y=one)[name=string("dw")];
    tensor<int32,[8]> py=const()[name=string("py"),val=tensor<int32,[8]>([0,0,0,{2*H},0,0,0,0])];
    tensor<int32,[8]> pm=const()[name=string("pm"),val=tensor<int32,[8]>([0,0,{H},{H},0,0,0,0])];
    tensor<int32,[8]> pd=const()[name=string("pd"),val=tensor<int32,[8]>([0,0,{2*H},0,0,0,0,0])];
    tensor<fp16,[1,{3*H},1,{D}]> yp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=py,x=y0)[name=string("yp")];
    tensor<fp16,[1,{3*H},1,{D}]> mp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=pm,x=mw)[name=string("mp")];
    tensor<fp16,[1,{3*H},1,{D}]> dp=pad(mode=string("constant"),constant_val=fp16(0x0p+0),pad=pd,x=dw)[name=string("dp")];
    tensor<fp16,[1,{3*H},1,{D}]> ym=add(x=yp,y=mp)[name=string("ym")];
    tensor<fp16,[1,{3*H},1,{D}]> out=add(x=ym,y=dp)[name=string("out")];
  }} -> (out);
}}
// pure_ane_attention_stream_B{B}
'''
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
            self.program=driver.engine.compile_multiproc(mil,{},C,3*H,D)
        if self.program is None:
            tail="\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE streamed attention compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        self.combine=AneAttentionCombine(driver)
        self.groups={n:AneAttentionStreamGroup(driver,n,B)
                     for n in group_sizes}
        self.programs=[self.direct_program,self.program,self.combine.program]
        self.programs.extend(group.program for group in self.groups.values())
        self.blocks=self.capacity//B
        # Block-major makes every 256-token K/V submission contiguous.  At
        # 256K this avoids materializing a strided copy for every KV head and
        # leaves untouched future blocks as lazily committed virtual memory.
        self.keys=np.zeros((self.blocks,K,B,D),np.float16)
        self.values=np.zeros((self.blocks,K,B,D),np.float16)
        self.offset=0
        assert_standalone("long-context attention compile")

    def reset(self)->None:
        # Stale entries are masked and then overwritten as offset advances, so
        # resetting a 17 GB aggregate 256K cache must not zero all its pages.
        self.offset=0

    def fork_cache(self)->"AneLongContextAttentionCore":
        other=object.__new__(AneLongContextAttentionCore)
        for name in ("driver","Hq","Hkv","D","L","B","capacity","blocks",
                     "direct_program","direct_input","input","program",
                     "combine","groups","programs"):
            setattr(other,name,getattr(self,name))
        # np.zeros_like eagerly faults these multi-GiB arrays on macOS. Fresh
        # calloc-backed arrays preserve sparse virtual allocation until a KV
        # block is actually populated.
        other.keys=np.zeros(self.keys.shape,self.keys.dtype)
        other.values=np.zeros(self.values.shape,self.values.dtype)
        other.offset=0
        return other

    def _run_direct(self,q:np.ndarray,valid:int)->np.ndarray:
        with self.driver.view(
            self.direct_program._in_surf,(self.direct_input,self.D),np.float16
        ) as dst:
            dst[:]=0;dst[:self.Hq]=q;p=self.Hq
            dst[p:p+self.Hkv*self.B]=self.keys[0].reshape(-1,self.D);p+=self.Hkv*self.B
            dst[p:p+self.Hkv*self.B]=self.values[0].reshape(-1,self.D);p+=self.Hkv*self.B
            dst[p,valid:self.B]=np.float16(-1e4)
        if not self.driver.engine.submit(self.direct_program):
            raise RuntimeError("ANE direct attention submission failed")
        with self.driver.view(
            self.direct_program._out_surf,(self.Hq,self.D),np.float16
        ) as src:
            return np.array(src,np.float16)

    def _run_block(self,q:np.ndarray,block_index:int,valid:int)->np.ndarray:
        with self.driver.view(
            self.program._in_surf,(self.input,self.D),np.float16
        ) as dst:
            dst[:]=0;dst[:self.Hq]=q;p=self.Hq
            dst[p:p+self.Hkv*self.B]=self.keys[block_index].reshape(-1,self.D);p+=self.Hkv*self.B
            dst[p:p+self.Hkv*self.B]=self.values[block_index].reshape(-1,self.D);p+=self.Hkv*self.B
            dst[p,valid:self.B]=np.float16(-1e4)
        if not self.driver.engine.submit(self.program):
            raise RuntimeError("ANE streamed attention submission failed")
        with self.driver.view(
            self.program._out_surf,(3*self.Hq,self.D),np.float16
        ) as src:
            return np.array(src,np.float16)

    def _run_group(self,q:np.ndarray,block_index:int,count:int,
                   last_valid:int)->np.ndarray:
        if count==1:return self._run_block(q,block_index,last_valid)
        return self.groups[count](
            q,self.keys[block_index:block_index+count],
            self.values[block_index:block_index+count],last_valid
        )

    def __call__(self,q:np.ndarray,k:np.ndarray,v:np.ndarray)->np.ndarray:
        if self.offset>=self.L:
            raise RuntimeError(f"attention cache exceeds configured limit {self.L}")
        block_index,token_index=divmod(self.offset,self.B)
        self.keys[block_index,:,token_index]=k
        self.values[block_index,:,token_index]=v
        valid=self.offset+1
        if valid<=self.B:
            out=self._run_direct(q,valid)
        else:
            state=None;block_index=0
            remaining=-(-valid//self.B)
            group_order=sorted(self.groups,reverse=True)+[1]
            while remaining:
                count=next(n for n in group_order if n<=remaining)
                reaches_end=block_index+count== -(-valid//self.B)
                last_valid=(valid-(block_index+count-1)*self.B
                            if reaches_end else self.B)
                current=self._run_group(q,block_index,count,last_valid)
                state=current if state is None else self.combine(state,current)
                block_index+=count;remaining-=count
            assert state is not None
            out=state[:self.Hq]
        self.offset=valid
        assert_standalone("long-context attention dispatch")
        return out


class AneAttentionTail:
    """Attention output gate through residual MLP in one ANE program."""

    def __init__(self, driver: AneDriver, checkpoint: Checkpoint,
                 layer: int, bits: int = 4, width: int = 32,
                 next_norm_name: str | None = None,
                 next_projection_names: list[str] | None = None,
                 active_lanes: int = 1,
                 prefix: str | None = None,
                 tag: str | None = None,
                 down_proj_parts: int = 1):
        self.driver, self.layer, self.bits = driver, layer, bits
        self.width = max(32, width)
        self.active_lanes = active_lanes
        p = prefix or f"model.language_model.layers.{layer}"
        label = tag or f"layer{layer}"
        names = {
            "o": f"{p}.self_attn.o_proj.weight",
            "g": f"{p}.mlp.gate_proj.weight",
            "u": f"{p}.mlp.up_proj.weight",
            "d": f"{p}.mlp.down_proj.weight",
        }
        oi, gi, ui, di = (checkpoint.info(names[k]) for k in ("o", "g", "u", "d"))
        self.H, self.Dc = oi.shape
        self.I = gi.shape[0]
        self.input = 2*self.Dc + self.H
        H, Dc, I, S = self.H, self.Dc, self.I, self.width
        self.down_proj_parts = down_proj_parts
        blobs = {}
        for key in ("o", "d"):
            blobs.update(_quantize_matrix(checkpoint, names[key], key, bits))
        gp = _quantize_matrix(checkpoint, names["g"], "gu", bits)
        up = _quantize_matrix(checkpoint, names["u"], "gu", bits)
        blobs["gu.bin"] = gp["gu.bin"] + up["gu.bin"]
        if bits != 16:
            blobs["gus.bin"] = gp["gus.bin"] + up["gus.bin"]
        blobs["pn.bin"] = checkpoint.tensor(
            f"{p}.post_attention_layernorm.weight", np.float16
        ).tobytes()
        if (next_norm_name is None)!=(next_projection_names is None):
            raise ValueError("next norm and projection must be supplied together")
        self.has_next=next_norm_name is not None
        self.next_output=0
        if self.has_next:
            blobs["next.bin"]=checkpoint.tensor(next_norm_name,np.float16).tobytes()
            next_infos=[checkpoint.info(name) for name in next_projection_names]
            if any(info.shape[1]!=H for info in next_infos):
                raise ValueError("next projection input dimensions do not match")
            self.next_output=sum(info.shape[0] for info in next_infos)
            parts=[_quantize_matrix(checkpoint,name,"np",bits)
                   for name in next_projection_names]
            blobs["np.bin"]=b"".join(part["np.bin"] for part in parts)
            if bits!=16:
                blobs["nps.bin"]=b"".join(part["nps.bin"] for part in parts)
        next_decl=(f'''    tensor<fp16, [1, {H}, 1, 1]> nnw = const()[name=string("nnw"), val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/next.bin"), offset=uint64(64)))];
{_dense_decl("np",self.next_output,H,bits)}''' if self.has_next else '')
        post_norm = _stable_rms_block(
            "h", "hn", H, S, "pn", f"ha{label}_", active_lanes
        )
        next_norm = _stable_rms_block(
            "o0", "nn", H, S, "nnw", f"na{label}_", active_lanes
        )
        next_body=(f'''    tensor<fp16, [1, {H}, 1, {S}]> y = identity(x=o0)[name=string("y")];
{next_norm}
    tensor<fp16, [1, {self.next_output}, 1, {S}]> y2 = conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, strides=st, weight=npw, x=nn)[name=string("y2")];''' if self.has_next else '')
        down_decl, down_body, raw_weight_files = _packed_split_down_projection(
            driver.module, blobs, H, I, S, bits, down_proj_parts
        )
        decl = "\n".join((
            _dense_decl("o", H, Dc, bits),
            _dense_decl("gu", 2*I, H, bits),
            down_decl,
        ))
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {self.input}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 g1 = const()[name=string("g1"), val=int32(1)];
{decl}
{next_decl}
    tensor<fp16, [1, {H}, 1, 1]> pn = const()[name=string("pn"), val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/pn.bin"), offset=uint64(64)))];
    tensor<fp16, [1, {Dc}, 1, {S}]> core = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{Dc},1,{S}]), x=x)[name=string("core")];
    tensor<fp16, [1, {Dc}, 1, {S}]> gate0 = slice_by_index(begin=tensor<int32, [4]>([0,{Dc},0,0]), end=tensor<int32, [4]>([1,{2*Dc},1,{S}]), x=x)[name=string("gate0")];
    tensor<fp16, [1, {H}, 1, {S}]> residual = slice_by_index(begin=tensor<int32, [4]>([0,{2*Dc},0,0]), end=tensor<int32, [4]>([1,{self.input},1,{S}]), x=x)[name=string("residual")];
    tensor<fp16, [1, {Dc}, 1, {S}]> ng0 = mul(x=gate0, y=fp16(-0x1p+0))[name=string("ng0")];
    tensor<fp16, [1, {Dc}, 1, {S}]> eg0 = exp(x=ng0)[name=string("eg0")];
    tensor<fp16, [1, {Dc}, 1, {S}]> gd0 = add(x=eg0, y=fp16(0x1p+0))[name=string("gd0")];
    tensor<fp16, [1, {Dc}, 1, {S}]> gated = real_div(x=core, y=gd0)[name=string("gated")];
    tensor<fp16, [1, {H}, 1, {S}]> attn = conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, strides=st, weight=ow, x=gated)[name=string("out_proj")];
    tensor<fp16, [1, {H}, 1, {S}]> h = add(x=residual, y=attn)[name=string("h")];
{post_norm}
    tensor<fp16, [1, {2*I}, 1, {S}]> gu = conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, strides=st, weight=guw, x=hn)[name=string("gu")];
    tensor<fp16, [1, {I}, 1, {S}]> gate = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{I},1,{S}]), x=gu)[name=string("gate")];
    tensor<fp16, [1, {I}, 1, {S}]> up = slice_by_index(begin=tensor<int32, [4]>([0,{I},0,0]), end=tensor<int32, [4]>([1,{2*I},1,{S}]), x=gu)[name=string("up")];
    tensor<fp16, [1, {I}, 1, {S}]> ng = mul(x=gate, y=fp16(-0x1p+0))[name=string("ng")];
    tensor<fp16, [1, {I}, 1, {S}]> eg = exp(x=ng)[name=string("eg")];
    tensor<fp16, [1, {I}, 1, {S}]> gd = add(x=eg, y=fp16(0x1p+0))[name=string("gd")];
    tensor<fp16, [1, {I}, 1, {S}]> gs = real_div(x=gate, y=gd)[name=string("gs")];
    tensor<fp16, [1, {I}, 1, {S}]> act = mul(x=gs, y=up)[name=string("act")];
{down_body}
    tensor<fp16, [1, {H}, 1, {S}]> {'o0' if self.has_next else 'y'} = add(x=h, y=mlp)[name=string("{'o0' if self.has_next else 'y'}")];
{next_body}
  }} -> ({'y, y2' if self.has_next else 'y'});
}}
// pure_ane_attention_tail_{label}_int{bits}_down{down_proj_parts}
'''
        capture = io.StringIO(); t0 = time.time()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(
                mil, blobs, self.input, H, S,
                raw_weight_files=raw_weight_files
            )
        if self.program is None:
            tail = "\n".join(capture.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"ANE attention tail compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        if self.has_next:self.next_surface,self.request=_bind_secondary_output(driver,self.program,self.next_output,self.width)
        self.nbytes = sum(len(x) for x in blobs.values())
        self.compile_seconds = time.time()-t0
        assert_standalone("attention tail compile")

    def __call__(self, core: np.ndarray, gate: np.ndarray,
                 residual: np.ndarray) -> np.ndarray:
        core=np.asarray(core,np.float16)
        if core.ndim==3:
            core=core.reshape(-1,core.shape[-1])
        elif core.ndim==2 and core.shape==(24,256):
            core=core.reshape(-1)
        core,lanes=_lane_matrix(core,self.Dc,self.active_lanes)
        gate=np.asarray(gate,np.float16)
        if gate.ndim==3:gate=gate.reshape(-1,gate.shape[-1])
        elif gate.ndim==2 and gate.shape==(24,256):gate=gate.reshape(-1)
        gate,glanes=_lane_matrix(gate,self.Dc,self.active_lanes)
        residual,rlanes=_lane_matrix(residual,self.H,self.active_lanes)
        if glanes!=lanes or rlanes!=lanes:
            raise ValueError("attention tail lane counts differ")
        with self.driver.view(
            self.program._in_surf, (self.input, self.width), np.float16
        ) as dst:
            dst[:]=0
            dst[:self.Dc,:lanes]=core
            dst[self.Dc:2*self.Dc,:lanes]=gate
            dst[2*self.Dc:,:lanes]=residual
        if self.has_next:_submit_bound(self.driver,self.program,self.request)
        elif not self.driver.engine.submit(self.program, procedure_index=0):raise RuntimeError("ANE attention tail submission failed")
        with self.driver.view(
            self.program._out_surf, (self.H, self.width), np.float16
        ) as src:
            out = np.array(src[:, :lanes], np.float16)
        if self.has_next:
            with self.driver.view(self.next_surface,(self.next_output,self.width),np.float16) as src:nxt=np.array(src[:,:lanes],np.float16)
        assert_standalone("attention tail dispatch")
        out=_restore_lane_rank(out,lanes)
        if self.has_next:nxt=_restore_lane_rank(nxt,lanes)
        return (out,nxt) if self.has_next else out


class AneFinalHead:
    """Final RMSNorm and vocabulary projection, chunked across four programs."""

    def __init__(self, driver: AneDriver, checkpoint: Checkpoint,
                 bits: int = 4, chunks: int = 4, width: int = 32,
                 active_lanes: int = 1,
                 head_name: str = "lm_head.weight",
                 norm_name: str = "model.language_model.norm.weight"):
        self.driver, self.bits = driver, bits
        self.width = max(32, width)
        self.active_lanes = active_lanes
        info = checkpoint.info(head_name)
        self.V, self.H = info.shape
        self.norm = checkpoint.tensor(norm_name, np.float16)
        step = -(-self.V // chunks)
        self.programs, self.spans, self.nbytes = [], [], 0
        H, S = self.H, self.width
        for ci, v0 in enumerate(range(0, self.V, step)):
            v1 = min(self.V, v0+step)
            O = v1-v0
            blobs = _quantize_matrix(
                checkpoint, head_name, "h", bits,
                row_start=v0, row_end=v1
            )
            decl = _dense_decl("h", O, H, bits)
            norm_body = _stable_rms_block(
                "hidden", "n", H, S, "nw", f"fh{ci}_", active_lanes
            )
            mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {2*H}, 1, {S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 g1 = const()[name=string("g1"), val=int32(1)];
{decl}
    tensor<fp16, [1, {H}, 1, {S}]> hidden = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{H},1,{S}]), x=x)[name=string("hidden")];
    tensor<fp16, [1, {H}, 1, 1]> nw = slice_by_index(begin=tensor<int32, [4]>([0,{H},0,0]), end=tensor<int32, [4]>([1,{2*H},1,1]), x=x)[name=string("nw")];
{norm_body}
    tensor<fp16, [1, {O}, 1, {S}]> y = conv(dilations=dl, groups=g1, pad=pd, pad_type=pt, strides=st, weight=hw, x=n)[name=string("head{ci}")];
  }} -> (y);
}}
// pure_ane_final_head_{ci}_int{bits}
'''
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                program=driver.engine.compile_multiproc(mil,blobs,2*H,O,S)
            if program is None:
                tail="\n".join(capture.getvalue().strip().splitlines()[-8:])
                raise RuntimeError(f"ANE final head chunk {ci} failed:\n{tail}")
            driver.engine._ensure_io(program)
            self.programs.append(program); self.spans.append((v0,v1))
            self.nbytes += sum(len(x) for x in blobs.values())
            driver.discard_compiler_files(program)
        assert_standalone("final norm and head compile")

    def __call__(self, hidden: np.ndarray,
                 norm_weight: np.ndarray | None = None) -> np.ndarray:
        hidden,lanes=_lane_matrix(hidden,self.H,self.active_lanes)
        nw=self.norm if norm_weight is None else np.asarray(norm_weight,np.float16)
        if nw.shape!=(self.H,):raise ValueError(f"bad final norm shape {nw.shape}")
        logits=np.empty((self.V,lanes),np.float32)
        for program,(v0,v1) in zip(self.programs,self.spans):
            with self.driver.view(
                program._in_surf,(2*self.H,self.width),np.float16
            ) as dst:
                dst[:]=0;dst[:self.H,:lanes]=hidden;dst[self.H:,0]=nw
            if not self.driver.engine.submit(program,procedure_index=0):
                raise RuntimeError("ANE final head submission failed")
            with self.driver.view(
                program._out_surf,(v1-v0,self.width),np.float16
            ) as src:
                logits[v0:v1]=src[:,:lanes]
        assert_standalone("final norm and head dispatch")
        return _restore_lane_rank(logits,lanes)


class AneMtpFusion:
    """MTP embedding/hidden RMSNorm and protected fp16 fusion projection."""

    def __init__(self,driver:AneDriver,checkpoint:Checkpoint,
                 width:int=32,active_lanes:int=3):
        self.driver=driver;self.width=max(32,width);self.active_lanes=active_lanes
        self.H=checkpoint.info("mtp.pre_fc_norm_hidden.weight").shape[0]
        H,S=self.H,self.width
        fc=checkpoint.tensor("mtp.fc.weight",np.float16)
        if fc.shape!=(H,2*H):raise ValueError(f"unexpected MTP fc {fc.shape}")
        blobs={
            "fce.bin":np.ascontiguousarray(fc[:,:H]).tobytes(),
            "fch.bin":np.ascontiguousarray(fc[:,H:]).tobytes(),
            "en.bin":checkpoint.tensor(
                "mtp.pre_fc_norm_embedding.weight",np.float16
            ).tobytes(),
            "hn.bin":checkpoint.tensor(
                "mtp.pre_fc_norm_hidden.weight",np.float16
            ).tobytes(),
        }
        en=_stable_rms_block("e","enorm",H,S,"enw","mtpe_",active_lanes)
        hn=_stable_rms_block("h","hnorm",H,S,"hnw","mtph_",active_lanes)
        mil=f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16,[1,{2*H},1,{S}]> x) {{
    string pt=const()[name=string("pt"),val=string("valid")]; tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])]; tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,0,0])]; tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])]; int32 g=const()[name=string("g"),val=int32(1)];
    tensor<fp16,[{H},{H},1,1]> few=const()[name=string("few"),val=tensor<fp16,[{H},{H},1,1]>(BLOBFILE(path=string("@model_path/weights/fce.bin"),offset=uint64(64)))];
    tensor<fp16,[{H},{H},1,1]> fhw=const()[name=string("fhw"),val=tensor<fp16,[{H},{H},1,1]>(BLOBFILE(path=string("@model_path/weights/fch.bin"),offset=uint64(64)))];
    tensor<fp16,[1,{H},1,1]> enw=const()[name=string("enw"),val=tensor<fp16,[1,{H},1,1]>(BLOBFILE(path=string("@model_path/weights/en.bin"),offset=uint64(64)))];
    tensor<fp16,[1,{H},1,1]> hnw=const()[name=string("hnw"),val=tensor<fp16,[1,{H},1,1]>(BLOBFILE(path=string("@model_path/weights/hn.bin"),offset=uint64(64)))];
    tensor<fp16,[1,{H},1,{S}]> e=slice_by_index(begin=tensor<int32,[4]>([0,0,0,0]),end=tensor<int32,[4]>([1,{H},1,{S}]),x=x)[name=string("e")];
    tensor<fp16,[1,{H},1,{S}]> h=slice_by_index(begin=tensor<int32,[4]>([0,{H},0,0]),end=tensor<int32,[4]>([1,{2*H},1,{S}]),x=x)[name=string("h")];
{en}
{hn}
    tensor<fp16,[1,{H},1,{S}]> ef=conv(dilations=dl,groups=g,pad=pd,pad_type=pt,strides=st,weight=few,x=enorm)[name=string("ef")];
    tensor<fp16,[1,{H},1,{S}]> hf=conv(dilations=dl,groups=g,pad=pd,pad_type=pt,strides=st,weight=fhw,x=hnorm)[name=string("hf")];
    tensor<fp16,[1,{H},1,{S}]> y=add(x=ef,y=hf)[name=string("y")];
  }} -> (y);
}}
// pure_ane_mtp_fusion_fp16
'''
        cap=io.StringIO()
        with contextlib.redirect_stdout(cap),contextlib.redirect_stderr(cap):
            self.program=driver.engine.compile_multiproc(mil,blobs,2*H,H,S)
        if self.program is None:
            raise RuntimeError("MTP fusion compile failed:\n"+"\n".join(cap.getvalue().splitlines()[-10:]))
        driver.engine._ensure_io(self.program);self.nbytes=sum(map(len,blobs.values()))

    def __call__(self,embedding:np.ndarray,hidden:np.ndarray)->np.ndarray:
        embedding,lanes=_lane_matrix(embedding,self.H,self.active_lanes)
        hidden,hlanes=_lane_matrix(hidden,self.H,self.active_lanes)
        if hlanes!=lanes:raise ValueError("MTP fusion lane counts differ")
        with self.driver.view(self.program._in_surf,(2*self.H,self.width),np.float16) as d:
            d[:]=0;d[:self.H,:lanes]=embedding;d[self.H:,:lanes]=hidden
        if not self.driver.engine.submit(self.program):raise RuntimeError("MTP fusion submit failed")
        with self.driver.view(self.program._out_surf,(self.H,self.width),np.float16) as o:
            out=np.array(o[:,:lanes],np.float16)
        return _restore_lane_rank(out,lanes)


class PureAneMtp:
    """The checkpoint's single MTP decoder layer, entirely on the ANE."""

    def __init__(self,driver:AneDriver,checkpoint:Checkpoint,
                 projection:_BankProjection,prepare:_DynamicPrepareLayer,
                 core:AneAttentionCore,final_head:AneFinalHead,bits:int,
                 active_lanes:int=3,down_proj_parts:int=1):
        self.driver=driver;self.checkpoint=checkpoint;self.projection=projection
        self.prepare=prepare;self.core=core;self.final_head=final_head
        self.active_lanes=active_lanes
        self.fusion=AneMtpFusion(driver,checkpoint,active_lanes=active_lanes)
        self.tail=AneAttentionTail(
            driver,checkpoint,0,bits=bits,active_lanes=active_lanes,
            prefix="mtp.layers.0",tag="mtp",
            down_proj_parts=down_proj_parts
        )
        self.norm=checkpoint.tensor("mtp.norm.weight",np.float16)
        self.nbytes=self.fusion.nbytes+self.tail.nbytes

    def snapshot(self)->int:return self.core.offset
    def restore(self,offset:int)->None:self.core.offset=offset

    def step_many(self,hidden:np.ndarray,token_ids:list[int],*,
                  project_logits:bool=True)->tuple[np.ndarray | None,np.ndarray]:
        if not 1<=len(token_ids)<=self.active_lanes:raise ValueError("bad MTP batch")
        embeds=np.stack([self.checkpoint.embedding(t) for t in token_ids],axis=1)
        hidden,lanes=_lane_matrix(hidden,5120,self.active_lanes)
        if lanes!=len(token_ids):raise ValueError("MTP hidden/token lane mismatch")
        fused=self.fusion(embeds,hidden)
        fused,_=_lane_matrix(fused,5120,self.active_lanes)
        projection=self.projection(fused)
        projection,_=_lane_matrix(projection,14336,self.active_lanes)
        cores=np.empty((6144,lanes),np.float16);gates=np.empty((6144,lanes),np.float16)
        for lane in range(lanes):
            qp=projection[:12288,lane].reshape(24,512)
            q,gate=qp[:,:256],qp[:,256:]
            k=projection[12288:13312,lane].reshape(4,256)
            v=projection[13312:14336,lane].reshape(4,256)
            q,k=self.prepare(q,k,self.core.offset)
            cores[:,lane]=self.core(q,k,v).reshape(-1);gates[:,lane]=gate.reshape(-1)
        out=self.tail(cores,gates,fused)
        out,_=_lane_matrix(out,5120,self.active_lanes)
        logits=None
        if project_logits:
            logits=self.final_head(out,self.norm)
            logits,_=_lane_matrix(logits,self.final_head.V,self.active_lanes)
        return logits,out

    def step(self,hidden:np.ndarray,token_id:int)->tuple[np.ndarray,np.ndarray]:
        logits,out=self.step_many(hidden,[token_id])
        assert logits is not None
        return logits[:,0],out[:,0]


class AneNormProjectionBank:
    """Many same-shape normalized projections as procedures in one model."""
    def __init__(self,driver:AneDriver,checkpoint:Checkpoint,
                 specs:list[tuple[str,list[str],float]],bits:int,tag:str):
        self.driver=driver; self.width=32; self.bits=bits
        first=[checkpoint.info(n) for n in specs[0][1]]
        self.H=first[0].shape[1]; self.O=sum(x.shape[0] for x in first)
        self.spans=[]; o=0
        for n,info in zip(specs[0][1],first):
            self.spans.append((n,o,o+info.shape[0]));o+=info.shape[0]
        blobs={"mean.bin":np.full((1,self.H),1/self.H,np.float16).tobytes()}
        funcs=[]; self.nbytes=0
        for i,(norm_name,proj_names,scale) in enumerate(specs):
            parts=[_quantize_matrix(checkpoint,n,f"p{i}",bits) for n in proj_names]
            blobs[f"p{i}.bin"]=b"".join(x[f"p{i}.bin"] for x in parts)
            if bits!=16: blobs[f"p{i}s.bin"]=b"".join(x[f"p{i}s.bin"] for x in parts)
            blobs[f"n{i}.bin"]=checkpoint.tensor(norm_name,np.float16).tobytes()
            decl=_dense_decl(f"p{i}",self.O,self.H,bits)
            sh=float(np.float16(scale)).hex(); eh=float(np.float16(1e-6*scale*scale)).hex()
            fn=f'''  func procedure{i:03d}<ios18>(tensor<fp16, [1, {self.H}, 1, 32]> x) {{
    string pt=const()[name=string("pt"),val=string("valid")]; tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])]; tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,0,0])]; tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])]; int32 g=const()[name=string("g"),val=int32(1)];
{decl}
    tensor<fp16,[1,{self.H},1,1]> nw=const()[name=string("nw"),val=tensor<fp16,[1,{self.H},1,1]>(BLOBFILE(path=string("@model_path/weights/n{i}.bin"),offset=uint64(64)))]; tensor<fp16,[1,{self.H},1,1]> mw=const()[name=string("mw"),val=tensor<fp16,[1,{self.H},1,1]>(BLOBFILE(path=string("@model_path/weights/mean.bin"),offset=uint64(64)))];
    tensor<fp16,[1,{self.H},1,32]> xs=mul(x=x,y=fp16({sh}))[name=string("xs")]; tensor<fp16,[1,{self.H},1,32]> sq=mul(x=xs,y=xs)[name=string("sq")]; tensor<fp16,[1,1,1,32]> ms=conv(dilations=dl,groups=g,pad=pd,pad_type=pt,strides=st,weight=mw,x=sq)[name=string("ms")]; tensor<fp16,[1,1,1,32]> me=add(x=ms,y=fp16({eh}))[name=string("me")]; tensor<fp16,[1,1,1,32]> sd=sqrt(x=me)[name=string("sd")]; tensor<fp16,[1,{self.H},1,32]> u=real_div(x=xs,y=sd)[name=string("u")]; tensor<fp16,[1,{self.H},1,32]> n=mul(x=u,y=nw)[name=string("n")]; tensor<fp16,[1,{self.O},1,32]> y=conv(dilations=dl,groups=g,pad=pd,pad_type=pt,strides=st,weight=p{i}w,x=n)[name=string("y")];
  }} -> (y);'''
            funcs.append(fn.replace('name=string("',f'name=string("p{i}_'))
        mil=f'''program(1.3)\n{driver.module._BUILD_INFO}\n{{\n{chr(10).join(funcs)}\n}}\n// pure_{tag}_bank'''
        cap=io.StringIO()
        with contextlib.redirect_stdout(cap),contextlib.redirect_stderr(cap):
            self.program=driver.engine.compile_multiproc(mil,blobs,self.H,self.O,32)
        if self.program is None: raise RuntimeError(f"{tag} projection bank failed:\n"+"\n".join(cap.getvalue().splitlines()[-8:]))
        driver.engine._ensure_io(self.program); self.nbytes=sum(len(x) for x in blobs.values())
    def run(self,index:int,hidden:np.ndarray)->np.ndarray:
        with self.driver.view(self.program._in_surf,(self.H,32),np.float16) as d:d[:]=hidden[:,None]
        if not self.driver.engine.submit(self.program,procedure_index=index):raise RuntimeError("projection bank submit failed")
        with self.driver.view(self.program._out_surf,(self.O,32),np.float16) as o:return np.array(o[:,0],np.float16)


class _BankProjection:
    def __init__(self,bank:AneNormProjectionBank,index:int):self.bank=bank;self.index=index;self.nbytes=0
    def __call__(self,x):return self.bank.run(self.index,x)


class AneLinearProjectionBank:
    """Projection-only procedure bank using the driver's proven blob layout."""
    def __init__(self,driver,checkpoint,projection_sets,bits,tag,width=32):
        if bits not in (4,8,16):
            raise ValueError("pure procedure banks require int4, int8, or fp16")
        # width is the compiled program width. A width-64 dispatch was measured
        # to cost the same as width-32 (docs/OPTIMIZATIONS.md), so batched
        # callers can halve their dispatch count for free. Default stays 32.
        if width%32:raise ValueError("bank width must be a multiple of 32")
        self.driver=driver;self.width=width;self.active_lanes=3
        infos=[checkpoint.info(n) for n in projection_sets[0]]
        self.H=infos[0].shape[1];self.O=sum(x.shape[0] for x in infos)
        raw=bytearray();offsets=[];self.nbytes=0
        E=driver.module
        def append_blob(payload:bytes)->int:
            """Append a milinternal tensor with a file-absolute payload offset."""
            start=len(raw)
            blob=bytearray(E._make_blob(payload))
            payload_offset=start+128
            if payload_offset > 0xffffffff:
                raise RuntimeError(f"{tag} bank exceeds 32-bit blob offset range")
            struct.pack_into("<I",blob,80,payload_offset)
            raw.extend(blob)
            return start
        for names in projection_sets:
            parts=[_quantize_matrix(checkpoint,n,"p",bits) for n in names]
            data=b"".join(x["p.bin"] for x in parts)
            doff=append_blob(data)
            if bits==16:
                offsets.append(doff);self.nbytes+=len(data)
            else:
                scale=b"".join(x["ps.bin"] for x in parts)
                soff=append_blob(scale)
                offsets.append((doff,soff));self.nbytes+=len(data)+len(scale)
        mil=E.generate_procedure_bank_mil(
            self.H,self.O,self.width,offsets,quantized=bits!=16,
            weight_format="fp16" if bits==16 else f"int{bits}"
        )
        self.program=driver.engine.compile_multiproc(mil,{"weight.bin":bytes(raw)},self.H,self.O,self.width,raw_weight_files=frozenset({"weight.bin"}))
        if self.program is None:raise RuntimeError(f"{tag} linear projection bank failed")
        driver.engine._ensure_io(self.program)
    def run(self,index,hidden):
        hidden,lanes=_lane_matrix(hidden,self.H,self.active_lanes)
        with self.driver.view(self.program._in_surf,(self.H,self.width),np.float16) as d:
            d[:]=0;d[:,:lanes]=hidden
        if not self.driver.engine.submit(self.program,procedure_index=index):raise RuntimeError("linear projection bank submit failed")
        with self.driver.view(self.program._out_surf,(self.O,self.width),np.float16) as o:
            out=np.array(o[:,:lanes],np.float16)
        return _restore_lane_rank(out,lanes)


class AneGdnConvBank:
    def __init__(self,driver:AneDriver,checkpoint:Checkpoint,layers:list[int]):
        self.driver=driver;self.C=10240;self.width=32;blobs={};funcs=[]
        for i,layer in enumerate(layers):
            w=checkpoint.tensor(f"model.language_model.layers.{layer}.linear_attn.conv1d.weight",np.float16)
            blobs[f"w{i}.bin"]=w.reshape(self.C,1,1,4).tobytes()
            fn=f'''  func procedure{i:03d}<ios18>(tensor<fp16,[1,{self.C},1,32]> x) {{ tensor<fp16,[{self.C},1,1,4]> w=const()[name=string("w"),val=tensor<fp16,[{self.C},1,1,4]>(BLOBFILE(path=string("@model_path/weights/w{i}.bin"),offset=uint64(64)))]; tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])]; tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])]; tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,3,0])]; tensor<fp16,[1,{self.C},1,32]> c=conv(dilations=dl,groups=int32({self.C}),pad=pd,pad_type=string("custom"),strides=st,weight=w,x=x)[name=string("c")]; tensor<fp16,[1,{self.C},1,32]> nc=mul(x=c,y=fp16(-0x1p+0))[name=string("nc")]; tensor<fp16,[1,{self.C},1,32]> ex=exp(x=nc)[name=string("ex")]; tensor<fp16,[1,{self.C},1,32]> den=add(x=ex,y=fp16(0x1p+0))[name=string("den")]; tensor<fp16,[1,{self.C},1,32]> y=real_div(x=c,y=den)[name=string("y")]; }} -> (y);'''
            funcs.append(fn.replace('name=string("',f'name=string("c{i}_'))
        mil=f'''program(1.3)\n{driver.module._BUILD_INFO}\n{{\n{chr(10).join(funcs)}\n}}''';cap=io.StringIO()
        with contextlib.redirect_stdout(cap),contextlib.redirect_stderr(cap):self.program=driver.engine.compile_multiproc(mil,blobs,self.C,self.C,32)
        if self.program is None:raise RuntimeError("GDN conv bank failed:\n"+"\n".join(cap.getvalue().splitlines()[-8:]))
        driver.engine._ensure_io(self.program)
    def run(self,index,qkv,cache):
        with self.driver.view(self.program._in_surf,(self.C,32),np.float16) as d:d[:]=0;d[:,:3]=cache;d[:,3]=qkv
        if not self.driver.engine.submit(self.program,procedure_index=index):raise RuntimeError("conv bank submit failed")
        with self.driver.view(self.program._out_surf,(self.C,32),np.float16) as o:y=np.array(o[:,3],np.float16)
        cache[:,:2]=cache[:,1:3];cache[:,2]=qkv;return y


class _BankGdnConv:
    def __init__(self,bank,index):self.bank=bank;self.index=index;self.cache=np.zeros((10240,3),np.float16)
    def __call__(self,x):return self.bank.run(self.index,x,self.cache)


@dataclass
class _PureGdnLayer:
    head: AneNormProjection | None
    projection_output: int
    conv: AneGdnConv
    state: GdnState
    tail: AneGdnTail
    a_log: np.ndarray
    dt_bias: np.ndarray


@dataclass
class _PureAttentionLayer:
    head: AneNormProjection | None
    projection_output: int
    prepare: _DynamicPrepareLayer
    core: AneAttentionCore
    tail: AneAttentionTail


@dataclass
class PureAneSnapshot:
    steps: int
    gdn_conv: list[np.ndarray]
    gdn_state: list[np.ndarray]
    attention_offsets: list[int]
    last_hidden: np.ndarray | None
    mtp_attention_offset: int | None


class PureAneRuntime:
    """Standalone 64-layer Qwen3.8 decode scheduler; no framework fallback."""

    def __init__(self, checkpoint: Checkpoint, engine_path: str,
                 bits: int = 4, context: int = 256, mtp_draft: int = 0,
                 bake_cache: str | os.PathLike[str] | None = None,
                 profile_decode: bool = False,
                 down_proj_parts: int = 4):
        startup_started = time.perf_counter()
        checkpoint.configure_quant_cache(bake_cache)
        text=checkpoint.config.get("text_config",checkpoint.config)
        model_context=int(text.get("max_position_embeddings",context))
        if context<256:
            raise ValueError("pure ANE context must be at least 256 tokens")
        if context>model_context:
            raise ValueError(
                f"requested context {context} exceeds checkpoint limit {model_context}"
            )
        if context>256 and bits==16 and mtp_draft:
            raise ValueError(
                "fp16 long context plus MTP needs 129 distinct in-memory "
                "models on the current loader path; use int4/int8, disable "
                "MTP, or use context 256"
            )
        self.checkpoint, self.bits = checkpoint, bits
        self.context=context
        if down_proj_parts not in (1,4):
            raise ValueError("down_proj parts must be 1 or 4")
        self.down_proj_parts=down_proj_parts
        # Prefill batch width. Programs compile at width 32, and a width-32
        # dispatch is measured to cost the same as one carrying a single lane,
        # so lanes below 32 leave the rest of the dispatch idle. Raising this
        # enlarges the per-lane RMSNorm MIL, so it is a parameter, not a
        # constant. Q38_ANE_LANES overrides for measurement.
        # Program width. Blocks default to 32; a width-64 dispatch is measured
        # to cost the same, and AneGdnConv reserves 3 columns for convolution
        # history, so width 32 caps the batch at 29 lanes and AneGdnUnrolled's
        # power-of-two requirement pins it to 16. Width 64 lifts the cap to 61
        # and makes 32 lanes legal, halving the prefill passes.
        self.program_width = int(os.environ.get("Q38_ANE_WIDTH", "32"))
        if self.program_width % 32:
            raise ValueError("program width must be a multiple of 32")
        default_lanes = 16 if self.program_width == 32 else 32
        self.active_lanes = int(
            os.environ.get("Q38_ANE_LANES", str(default_lanes)))
        if self.active_lanes > self.program_width - 3:
            raise ValueError(
                f"{self.active_lanes} lanes exceeds the convolution limit "
                f"{self.program_width - 3} at width {self.program_width}")
        # Speculative draft width, so the maximum draft depth is this minus
        # one. It was 3, capping drafting at depth 2, but the verify pass
        # batches up to active_lanes positions and a marginal candidate costs
        # ~23 ms against 273 ms for the first. Measured over five varied
        # prompts: depth 2 gives 3.78 tok/s, depth 3 gives 3.98, depth 4 falls
        # back to 3.68 as sequential drafting cost overtakes acceptance.
        self.mtp_lanes = int(os.environ.get("Q38_ANE_MTP_LANES", "5"))
        if not 0 <= mtp_draft < self.mtp_lanes:
            raise ValueError(f"MTP draft must be 0..{self.mtp_lanes-1}")
        self.mtp_draft=mtp_draft
        self.profile_enabled=profile_decode
        self._profile_phase="idle"
        self._profile_request:dict[str,dict]={}
        self._profile_cumulative:dict[str,dict]={}
        self.steps = 0
        self.last_hidden: np.ndarray | None = None
        self.driver = AneDriver(engine_path)
        self.recurrence = AneGdnRecurrence(self.driver)
        self.driver.discard_compiler_files(self.recurrence.program)
        # One shared, weight-free exact 16-position recurrence graph. Importing
        # lazily avoids a module cycle in the standalone qualification probe;
        # this class is framework-free and uses the same local ANE driver.
        from probes.ane_gdn_scan64 import AneGdnUnrolled
        self.prefill_recurrence = AneGdnUnrolled(self.driver, self.active_lanes)
        self.driver.discard_compiler_files(self.prefill_recurrence.program)
        shared_prepare = AneAttentionPrepareDynamic(self.driver, context)
        self.driver.discard_compiler_files(shared_prepare.program)
        types = text.get("layer_types") or [
            "full_attention" if (i+1)%4==0 else "linear_attention"
            for i in range(int(text["num_hidden_layers"]))
        ]
        gidx=[i for i,k in enumerate(types) if k!="full_attention"]
        aidx=[i for i,k in enumerate(types) if k=="full_attention"]
        def pspec(layer,attention):
            p=f"model.language_model.layers.{layer}"
            if attention:
                ns=[f"{p}.self_attn.{x}.weight" for x in ("q_proj","k_proj","v_proj")]
            else:
                ns=[f"{p}.linear_attn.{x}.weight" for x in ("in_proj_qkv","in_proj_z","in_proj_b","in_proj_a")]
            return (f"{p}.input_layernorm.weight",ns,64.0 if layer==0 else 2.0)
        # Target input projections for layers 1..63 are chained into the
        # preceding tail. This removes 63 dispatches and the two target
        # projection-bank programs. MTP still needs its own draft projection.
        mtp_bank=None
        if mtp_draft:
            mtp_names=[
                f"mtp.layers.0.self_attn.{x}.weight"
                for x in ("q_proj","k_proj","v_proj")
            ]
            mtp_bank=AneLinearProjectionBank(
                self.driver,checkpoint,[mtp_names],bits,"mtp_head"
            )
            # the bank defaults to 3 active lanes; the MTP path drafts mtp_lanes
            mtp_bank.active_lanes=self.mtp_lanes
            self.driver.discard_compiler_files(mtp_bank.program)
        if context<=256:
            shared_core=AneAttentionCore(self.driver,context)
        else:
            # Stay within the 127 distinct-model budget observed on this
            # process's _ANEInMemoryModel loader path. Missing
            # group sizes fall back to repeated one-block scans, so this only
            # changes dispatch efficiency, never supported context or math.
            group_sizes=(() if bits==16 else ((32,) if mtp_draft else (4,16,32)))
            shared_core=AneLongContextAttentionCore(
                self.driver,context,group_sizes=group_sizes
            )
        attention_programs=getattr(shared_core,"programs",[shared_core.program])
        for program in attention_programs:
            self.driver.discard_compiler_files(program)
        l0spec=pspec(0,False)
        layer0_head=AneNormProjection(
            self.driver,checkpoint,l0spec[0],l0spec[1],bits=bits,
            tag="layer0_head",norm_scale=64,active_lanes=self.active_lanes,
            width=self.program_width
        )
        self.driver.discard_compiler_files(layer0_head.program)
        self.layers: list[_PureGdnLayer | _PureAttentionLayer] = []
        self.program_count = (4+len(attention_programs)+
                              (1 if mtp_bank is not None else 0))
        self.blob_bytes = (layer0_head.nbytes+
                           (mtp_bank.nbytes if mtp_bank is not None else 0))
        layer_started=time.perf_counter()
        for layer,kind in enumerate(types):
            p=f"model.language_model.layers.{layer}"
            current_names=pspec(layer,kind=="full_attention")[1]
            projection_output=sum(checkpoint.info(name).shape[0]
                                  for name in current_names)
            head=layer0_head if layer==0 else None
            if layer+1<len(types):
                next_attention=types[layer+1]=="full_attention"
                next_norm,next_names,_=pspec(layer+1,next_attention)
            else:
                next_norm=next_names=None
            if kind=="full_attention":
                prepare=_DynamicPrepareLayer(
                    shared_prepare,
                    checkpoint.tensor(f"{p}.self_attn.q_norm.weight",np.float16),
                    checkpoint.tensor(f"{p}.self_attn.k_norm.weight",np.float16),
                )
                core=shared_core.fork_cache()
                tail=AneAttentionTail(
                    self.driver,checkpoint,layer,bits=bits,next_norm_name=next_norm,
                    next_projection_names=next_names,
                    active_lanes=self.active_lanes,
                    down_proj_parts=down_proj_parts,
                    width=self.program_width
                )
                self.layers.append(_PureAttentionLayer(
                    head,projection_output,prepare,core,tail
                ))
                self.driver.discard_compiler_files(tail.program)
                self.program_count += 1
                self.blob_bytes += tail.nbytes
            else:
                conv=AneGdnConv(
                    self.driver,checkpoint,f"{p}.linear_attn.conv1d.weight",
                    width=self.program_width
                )
                state=self.recurrence.new_state()
                tail=AneGdnTail(
                    self.driver,checkpoint,layer,bits=bits,next_norm_name=next_norm,
                    next_projection_names=next_names,
                    active_lanes=self.active_lanes,
                    down_proj_parts=down_proj_parts,
                    width=self.program_width
                )
                al=checkpoint.tensor(f"{p}.linear_attn.A_log",np.float16)
                dt=checkpoint.tensor(f"{p}.linear_attn.dt_bias",np.float16)
                self.layers.append(_PureGdnLayer(
                    head,projection_output,conv,state,tail,al,dt
                ))
                for prog in (conv.program,tail.program):
                    self.driver.discard_compiler_files(prog)
                self.program_count += 2
                self.blob_bytes += tail.nbytes
            print(f"  pure bake layer {layer+1:02d}/64 {kind:<16} "
                  f"{self.blob_bytes/1e9:.2f}GB "
                  f"{time.perf_counter()-layer_started:.1f}s",flush=True)
        self.final_head=AneFinalHead(
            self.driver,checkpoint,bits=bits,chunks=4,
            active_lanes=self.active_lanes,width=self.program_width
        )
        self.program_count += len(self.final_head.programs)
        self.blob_bytes += self.final_head.nbytes
        self.mtp:PureAneMtp | None=None
        if mtp_draft:
            assert mtp_bank is not None
            mtpp="mtp.layers.0.self_attn"
            mtp_prepare=_DynamicPrepareLayer(
                shared_prepare,
                checkpoint.tensor(f"{mtpp}.q_norm.weight",np.float16),
                checkpoint.tensor(f"{mtpp}.k_norm.weight",np.float16),
            )
            self.mtp=PureAneMtp(
                self.driver,checkpoint,_BankProjection(mtp_bank,0),
                mtp_prepare,shared_core.fork_cache(),self.final_head,bits,
                self.mtp_lanes,down_proj_parts
            )
            for prog in (self.mtp.fusion.program,self.mtp.tail.program):
                self.driver.discard_compiler_files(prog)
            self.program_count+=2;self.blob_bytes+=self.mtp.nbytes
        cache_capacity=getattr(shared_core,"capacity",context)
        cache_count=len(aidx)+(1 if self.mtp is not None else 0)
        self.kv_cache_bytes=(cache_count*2*shared_core.Hkv*cache_capacity*
                             shared_core.D*np.dtype(np.float16).itemsize)
        assert_standalone("full model bake")
        total_seconds = time.perf_counter() - startup_started
        engine_metrics = dict(self.driver.engine.compile_metrics)
        engine_accounted = sum(engine_metrics[key] for key in (
            "descriptor_seconds", "cache_probe_seconds", "materialize_seconds",
            "compile_seconds", "load_seconds"
        ))
        self.startup_metrics = {
            "total_seconds": total_seconds,
            "artifact_cache_hit_rate": (
                engine_metrics["cache_load_hits"] / engine_metrics["calls"]
                if engine_metrics["calls"] else 0.0
            ),
            "driver": engine_metrics,
            "quant_cache": dict(checkpoint.quant_cache_stats),
            "host_prepare_and_io_seconds": max(0.0, total_seconds-engine_accounted),
            "bake_cache": (str(checkpoint.quant_cache_dir)
                           if checkpoint.quant_cache_dir else None),
            "down_proj_parts": down_proj_parts,
        }
        print(f"PURE_ANE_BAKE=PASS programs={self.program_count} "
              f"blobs={self.blob_bytes/1e9:.2f}GB context={self.context} "
              f"down_proj_parts={down_proj_parts} "
              f"kv_capacity={self.kv_cache_bytes/1e9:.2f}GB "
              f"seconds={total_seconds:.1f} "
              f"compiled={engine_metrics['calls']-engine_metrics['cache_load_hits']} "
              f"cache_loaded={engine_metrics['cache_load_hits']}",
              flush=True)

    def snapshot(self) -> PureAneSnapshot:
        """Capture all mutable target state before speculative verification."""
        gconv=[];gstate=[];attention=[]
        for layer in self.layers:
            if isinstance(layer,_PureGdnLayer):
                gconv.append(layer.conv.cache.copy())
                gstate.append(self.recurrence.snapshot(layer.state))
            else:
                attention.append(layer.core.offset)
        return PureAneSnapshot(
            self.steps,gconv,gstate,attention,
            None if self.last_hidden is None else self.last_hidden.copy(),
            None if self.mtp is None else self.mtp.snapshot(),
        )

    def restore(self,saved:PureAneSnapshot)->None:
        gi=ai=0
        for layer in self.layers:
            if isinstance(layer,_PureGdnLayer):
                layer.conv.cache[:]=saved.gdn_conv[gi]
                self.recurrence.restore(layer.state,saved.gdn_state[gi]);gi+=1
            else:
                layer.core.offset=saved.attention_offsets[ai];ai+=1
        self.steps=saved.steps
        self.last_hidden=(None if saved.last_hidden is None else
                          saved.last_hidden.copy())
        if self.mtp is not None and saved.mtp_attention_offset is not None:
            self.mtp.restore(saved.mtp_attention_offset)

    def reset(self) -> None:
        """Reset all per-sequence state without recompiling any ANE program.

        Long-context KV arrays are intentionally not zeroed: every valid slot
        is overwritten before use and attention masks entries beyond ``offset``.
        This keeps a 256K reset effectively constant-time and preserves sparse
        virtual-memory allocation.
        """
        zero_state=np.zeros((self.recurrence.HK,self.recurrence.Dv),np.float16)
        for layer in self.layers:
            if isinstance(layer,_PureGdnLayer):
                layer.conv.reset()
                self.recurrence.restore(layer.state,zero_state)
            else:
                layer.core.reset()
        if self.mtp is not None:
            self.mtp.core.reset()
        self.steps=0
        self.last_hidden=None
        assert_standalone("sequence reset")

    def begin_profile(self) -> None:
        """Reset the opt-in production-loop profile for one request."""
        self._profile_request={}
        self._profile_phase="idle"

    @staticmethod
    def _merge_profile(target:dict,phase:str,lanes:int,total_ns:int,
                       operations:dict[str,list[int]])->None:
        bucket=target.setdefault(phase,{"batches":0,"tokens":0,
                                        "batch_ns":0,"operations":{}})
        bucket["batches"]+=1;bucket["tokens"]+=lanes
        bucket["batch_ns"]+=total_ns
        for name,(elapsed,calls) in operations.items():
            op=bucket["operations"].setdefault(name,{"ns":0,"calls":0})
            op["ns"]+=elapsed;op["calls"]+=calls

    def _record_profile(self,phase:str,lanes:int,total_ns:int,
                        operations:dict[str,list[int]])->None:
        self._merge_profile(self._profile_request,phase,lanes,total_ns,operations)
        self._merge_profile(self._profile_cumulative,phase,lanes,total_ns,operations)

    @staticmethod
    def _format_profile(source:dict)->dict:
        result={}
        for phase,bucket in source.items():
            total_ns=bucket["batch_ns"]
            accounted=sum(x["ns"] for x in bucket["operations"].values())
            operations={
                name:{"seconds":value["ns"]/1e9,"calls":value["calls"],
                      "milliseconds_per_call":(value["ns"]/1e6/value["calls"]
                                               if value["calls"] else 0.0),
                      "percent_of_batch":(100*value["ns"]/total_ns
                                          if total_ns else 0.0)}
                for name,value in bucket["operations"].items()
            }
            residual=max(0,total_ns-accounted)
            operations["host_scheduler_unattributed"]={
                "seconds":residual/1e9,"calls":bucket["batches"],
                "milliseconds_per_call":(residual/1e6/bucket["batches"]
                                           if bucket["batches"] else 0.0),
                "percent_of_batch":(100*residual/total_ns if total_ns else 0.0),
            }
            result[phase]={"batches":bucket["batches"],
                           "tokens":bucket["tokens"],
                           "model_seconds":total_ns/1e9,
                           "milliseconds_per_token":(total_ns/1e6/bucket["tokens"]
                                                     if bucket["tokens"] else 0.0),
                           "operations":operations}
        return result

    def profile_snapshot(self)->dict|None:
        if not self.profile_enabled:return None
        return {"request":self._format_profile(self._profile_request),
                "cumulative":self._format_profile(self._profile_cumulative)}

    def step_many(self,token_ids:list[int])->tuple[np.ndarray,np.ndarray]:
        """Advance 1-3 causal positions, batching all learned weight matmuls."""
        if not 1<=len(token_ids)<=self.active_lanes:
            raise ValueError(f"step_many supports 1..{self.active_lanes} tokens")
        assert_standalone("token batch start")
        lanes=len(token_ids)
        profiling=self.profile_enabled
        operations:dict[str,list[int]]={}
        batch_started=time.perf_counter_ns() if profiling else 0
        def record(name:str,started:int,calls:int=1)->None:
            elapsed=time.perf_counter_ns()-started
            value=operations.setdefault(name,[0,0])
            value[0]+=elapsed;value[1]+=calls
        started=time.perf_counter_ns() if profiling else 0
        hidden=np.stack([self.checkpoint.embedding(t) for t in token_ids],axis=1)
        if profiling:record("embedding",started,lanes)
        normalized=None
        for index,layer in enumerate(self.layers):
            if index==0:
                if layer.head is None:
                    raise RuntimeError("layer zero is missing its input projection")
                started=time.perf_counter_ns() if profiling else 0
                projection=layer.head(hidden)
                if profiling:record("projection_head",started)
            else:
                if normalized is None:
                    raise RuntimeError(f"layer {index} is missing chained projection")
                projection=normalized
            projection,_=_lane_matrix(
                projection,layer.projection_output,self.active_lanes
            )
            if isinstance(layer,_PureGdnLayer):
                started=time.perf_counter_ns() if profiling else 0
                activated=layer.conv(projection[:10240])
                if profiling:record("gdn_conv",started)
                activated,_=_lane_matrix(activated,10240,self.active_lanes)
                if lanes == self.active_lanes:
                    started=time.perf_counter_ns() if profiling else 0
                    raw_q=np.stack([
                        activated[:2048,lane].reshape(16,128)
                        for lane in range(lanes)
                    ])
                    raw_k=np.stack([
                        activated[2048:4096,lane].reshape(16,128)
                        for lane in range(lanes)
                    ])
                    values=np.stack([
                        activated[4096:10240,lane].reshape(48,128)
                        for lane in range(lanes)
                    ])
                    initial=self.recurrence.materialize(layer.state)
                    self.prefill_recurrence.load(
                        raw_q,raw_k,values,
                        projection[16432:16480,:lanes].T,
                        projection[16384:16432,:lanes].T,
                        layer.a_log,layer.dt_bias,initial
                    )
                    block,state=self.prefill_recurrence.run_loaded()
                    self.recurrence.restore(
                        layer.state,
                        state.astype(np.float16).transpose(0,2,1).reshape(
                            self.recurrence.HK,self.recurrence.Dv
                        )
                    )
                    if profiling:record("gdn_recurrence",started)
                    cores=block.reshape(lanes,6144).T.astype(
                        np.float16,copy=False
                    )
                else:
                    cores=np.empty((6144,lanes),np.float16)
                    for lane in range(lanes):
                        q=activated[:2048,lane].reshape(16,128)
                        k=activated[2048:4096,lane].reshape(16,128)
                        v=activated[4096:10240,lane].reshape(48,128)
                        started=time.perf_counter_ns() if profiling else 0
                        core=self.recurrence(
                            layer.state,q,k,v,projection[16432:16480,lane],
                            projection[16384:16432,lane],layer.a_log,layer.dt_bias
                        )
                        if profiling:record("gdn_recurrence",started)
                        cores[:,lane]=core.reshape(-1)
                started=time.perf_counter_ns() if profiling else 0
                result=layer.tail(
                    cores,projection[10240:16384],hidden
                )
                if profiling:record("gdn_tail",started)
            else:
                cores=np.empty((6144,lanes),np.float16)
                gates=np.empty((6144,lanes),np.float16)
                for lane in range(lanes):
                    qp=projection[:12288,lane].reshape(24,512)
                    q,gate=qp[:,:256],qp[:,256:]
                    k=projection[12288:13312,lane].reshape(4,256)
                    v=projection[13312:14336,lane].reshape(4,256)
                    position=layer.core.offset
                    started=time.perf_counter_ns() if profiling else 0
                    q,k=layer.prepare(q,k,position)
                    if profiling:record("attention_prepare",started)
                    started=time.perf_counter_ns() if profiling else 0
                    cores[:,lane]=layer.core(q,k,v).reshape(-1)
                    if profiling:record("attention_core",started)
                    gates[:,lane]=gate.reshape(-1)
                started=time.perf_counter_ns() if profiling else 0
                result=layer.tail(cores,gates,hidden)
                if profiling:record("attention_tail",started)
            if isinstance(result,tuple):hidden,normalized=result
            else:hidden=result;normalized=None
            hidden,_=_lane_matrix(hidden,5120,self.active_lanes)
            if normalized is not None:
                if index+1>=len(self.layers):
                    raise RuntimeError("final layer unexpectedly returned a projection")
                normalized,_=_lane_matrix(
                    normalized,self.layers[index+1].projection_output,
                    self.active_lanes
                )
            trace_step=int(os.environ.get("PURE_ANE_TRACE_STEP","0") or 0)
            if (os.environ.get("PURE_ANE_TRACE") == "1" or
                    trace_step == self.steps+1):
                trace=hidden[:,-1].astype(np.float32)
                print(
                    f"L{index:02d} mean={float(trace.mean()):.7g} "
                    f"rms={float(np.sqrt(np.mean(trace*trace))):.7g} "
                    f"min={float(trace.min()):.7g} max={float(trace.max()):.7g} "
                    f"first={trace[:4].tolist()}",flush=True
                )
        started=time.perf_counter_ns() if profiling else 0
        logits=self.final_head(hidden)
        if profiling:record("final_head",started)
        logits,_=_lane_matrix(logits,self.final_head.V,self.active_lanes)
        self.steps += lanes
        self.last_hidden=hidden[:,-1].copy()
        if profiling:
            self._record_profile(self._profile_phase,lanes,
                                 time.perf_counter_ns()-batch_started,operations)
        assert_standalone("token batch end")
        return logits,hidden

    def step(self, token_id: int) -> np.ndarray:
        logits,_=self.step_many([token_id])
        return logits[:,0]

    def generate(self, tokenizer: StandaloneTokenizer, prompt: str,
                 max_tokens: int, *,
                 on_token: Callable[[int], bool | None] | None = None,
                 stop_token_ids: set[int] | None = None,
                 token_selector: Callable[[np.ndarray,list[int]],int] | None = None,
                 prefilled_tokens: int = 0,
                 prefill_logits: np.ndarray | None = None,
                 on_prefill: Callable[[np.ndarray],None] | None = None,
                 ) -> tuple[list[int], float]:
        """Prefill and decode a sequence, optionally reporting tokens live.

        ``on_token`` runs after a token is committed and can return ``False``
        to stop. A custom ``token_selector`` enables CPU-side sampling; MTP is
        used only for the deterministic greedy path because speculative
        acceptance for sampling requires coupled RNG distributions.
        """
        ids=tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt tokenized to nothing")
        if max_tokens<1:
            raise ValueError("max_tokens must be at least 1")
        if len(ids)+max_tokens>self.context:
            raise ValueError(
                f"prompt plus generation ({len(ids)+max_tokens}) exceeds "
                f"configured context {self.context}"
            )
        if not 0<=prefilled_tokens<=len(ids):
            raise ValueError(f"invalid prefilled token count {prefilled_tokens}")
        if prefilled_tokens and self.steps!=prefilled_tokens:
            raise ValueError(
                f"restored runtime has {self.steps} steps, expected {prefilled_tokens}"
            )
        if prefilled_tokens==len(ids) and prefill_logits is None:
            raise ValueError("an exact prefix hit requires cached logits")
        cached_last_hidden=(None if self.last_hidden is None else
                            self.last_hidden.copy())
        logits=(None if prefill_logits is None else
                np.asarray(prefill_logits,dtype=np.float16))
        prompt_hidden=[]
        start=time.perf_counter()
        self._profile_phase="prefill"
        for start_at in range(prefilled_tokens,len(ids),self.active_lanes):
            chunk=ids[start_at:start_at+self.active_lanes]
            batch_logits,batch_hidden=self.step_many(chunk)
            logits=batch_logits[:,-1]
            prompt_hidden.extend(batch_hidden[:,i].copy() for i in range(len(chunk)))
            if os.environ.get("PURE_ANE_TRACE_PREFIX") == "1":
                print(f"prefix={start_at+len(chunk)} token={chunk[-1]} "
                      f"next={int(np.argmax(logits))}",flush=True)
        use_mtp=self.mtp is not None and token_selector is None
        self._profile_phase="mtp_prefill"
        if use_mtp and prefilled_tokens==0 and len(ids)>1:
            history=np.stack(prompt_hidden[:-1],axis=1);next_ids=ids[1:]
            for start_at in range(0,len(next_ids),self.mtp_lanes):
                chunk=next_ids[start_at:start_at+self.mtp_lanes]
                self.mtp.step_many(
                    history[:,start_at:start_at+len(chunk)],chunk,
                    project_logits=False
                )
        elif use_mtp and prefilled_tokens and prompt_hidden:
            if cached_last_hidden is None:
                raise ValueError("MTP prefix resume requires cached last hidden state")
            history=np.stack([cached_last_hidden]+prompt_hidden[:-1],axis=1)
            next_ids=ids[prefilled_tokens:]
            for start_at in range(0,len(next_ids),self.mtp_lanes):
                chunk=next_ids[start_at:start_at+self.mtp_lanes]
                self.mtp.step_many(
                    history[:,start_at:start_at+len(chunk)],chunk,
                    project_logits=False
                )
        if on_prefill is not None:
            on_prefill(logits)
        self._profile_phase="decode"
        def select(current:np.ndarray,generated:list[int])->int:
            if token_selector is None:return int(np.argmax(current))
            return int(token_selector(current,generated))
        if use_mtp:
            last_prompt_hidden=(prompt_hidden[-1] if prompt_hidden else
                                self.last_hidden)
            if last_prompt_hidden is None:
                raise ValueError("MTP generation requires the last prompt hidden state")
            return self._generate_mtp(
                int(np.argmax(logits)),last_prompt_hidden,max_tokens,start,
                on_token=on_token,stop_token_ids=stop_token_ids
            )
        generated=[]
        for index in range(max_tokens):
            token=select(logits,generated)
            if stop_token_ids and token in stop_token_ids:
                break
            generated.append(token)
            if on_token is not None and on_token(token) is False:
                break
            if index+1<max_tokens:
                logits=self.step(token)
        return generated,time.perf_counter()-start

    def _generate_mtp(self,cur:int,last_hidden:np.ndarray,max_tokens:int,
                      start:float,*,
                      on_token:Callable[[int],bool | None] | None=None,
                      stop_token_ids:set[int] | None=None
                      )->tuple[list[int],float]:
        assert self.mtp is not None
        generated=[];cycles=accepted=0
        def commit(tokens:list[int])->bool:
            for token in tokens:
                if len(generated)>=max_tokens:return False
                if stop_token_ids and token in stop_token_ids:return False
                generated.append(token)
                if on_token is not None and on_token(token) is False:return False
            return True
        if not commit([cur]):
            return generated,time.perf_counter()-start
        while len(generated)<max_tokens:
            mtp_saved=self.mtp.snapshot();drafts=[];dh=last_hidden;dtok=cur
            remaining=max_tokens-len(generated)
            draft_count=min(self.mtp_draft,max(0,remaining-1))
            for _ in range(draft_count):
                dlogits,dh=self.mtp.step(dh,dtok)
                dtok=int(np.argmax(dlogits));drafts.append(dtok)
            target_saved=self.snapshot()
            verify_logits,verify_hidden=self.step_many([cur]+drafts)
            preds=[int(x) for x in np.argmax(verify_logits,axis=0)]
            n_ok=0
            for wanted,actual in zip(drafts,preds):
                if wanted!=actual:break
                n_ok+=1
            if n_ok==len(drafts):
                additions=drafts+[preds[-1]]
                accepted+=min(len(additions),max_tokens-len(generated))
                cur=preds[-1];last_hidden=verify_hidden[:,-1]
            else:
                self.restore(target_saved);self.mtp.restore(mtp_saved)
                keep=drafts[:n_ok];fix=preds[n_ok]
                replay=[cur]+keep
                _,replay_hidden=self.step_many(replay)
                actual_next=keep+[fix]
                self.mtp.step_many(
                    replay_hidden,actual_next,project_logits=False
                )
                additions=actual_next
                accepted+=min(len(additions),max_tokens-len(generated))
                cur=fix;last_hidden=replay_hidden[:,-1]
            cycles+=1
            if not commit(additions):break
        elapsed=time.perf_counter()-start
        print(f"PURE_ANE_MTP=PASS draft={self.mtp_draft} cycles={cycles} "
              f"accepted_per_cycle={accepted/max(1,cycles):.3f}",flush=True)
        return generated[:max_tokens],elapsed


def pure_infer(checkpoint: Checkpoint, engine_path: str, prompt: str,
               tokens: int, bits: int, raw_prompt: bool,
               verify_reference: bool = False,mtp_draft:int = 0,
               context:int = 256,down_proj_parts:int = 4) -> None:
    if bits < 16:
        qualification = (
            "matches the first 4 MLX reference tokens and remains coherent "
            "in the 32-token check" if bits == 4 else
            "matches the first 16 MLX reference tokens and answers correctly "
            "in the 32-token check"
        )
        print(
            f"NOTICE: int{bits} passes component numerics, {qualification}; "
            "broader benchmark quality is not yet qualified.",
            flush=True,
        )
    tokenizer=StandaloneTokenizer(checkpoint.path)
    formatted=(prompt if raw_prompt else
               f"<|im_start|>user\n{prompt}<|im_end|>\n"
               f"<|im_start|>assistant\n")
    runtime=PureAneRuntime(
        checkpoint,engine_path,bits=bits,context=context,mtp_draft=mtp_draft,
        down_proj_parts=down_proj_parts
    )
    generated,seconds=runtime.generate(tokenizer,formatted,tokens)
    text=tokenizer.decode(generated)
    if verify_reference:
        expected=[248068,198,760,1156]
        if raw_prompt or prompt != "Reply with exactly: OK" \
                or tokens < len(expected):
            raise ValueError(
                "--verify-reference requires the default templated prompt, "
                "and at least four generated tokens"
            )
        if generated[:len(expected)] != expected:
            raise RuntimeError(
                f"pure ANE reference mismatch: got {generated[:4]}, "
                f"expected {expected}"
            )
        print("PURE_ANE_REFERENCE=PASS tokens=[248068,198,760,1156]")
    print(f"PURE_ANE_EXECUTION=PASS prompt_tokens={len(tokenizer.encode(formatted))} "
          f"generated={len(generated)} seconds={seconds:.3f} "
          f"tok_per_s={len(generated)/seconds:.3f}")
    print(f"token_ids={generated}")
    print(text)
    assert_standalone("inference completion")


def loader_smoke(checkpoint: Checkpoint, text: str) -> None:
    tok = StandaloneTokenizer(checkpoint.path)
    ids = tok.encode(text)
    if not ids:
        raise RuntimeError("tokenizer returned no tokens")
    row = checkpoint.embedding(ids[0])
    H = int(checkpoint.architecture["hidden_size"])
    if row.shape != (H,) or row.dtype != np.float16 or not np.isfinite(row).all():
        raise RuntimeError(f"bad embedding row: {row.shape} {row.dtype}")
    assert_standalone("loader smoke test")
    print(f"tokens={ids}")
    print(f"first_embedding=shape{row.shape} dtype={row.dtype} "
          f"range=[{float(row.min()):.5g},{float(row.max()):.5g}]")
    print("PURE_BACKEND_GUARD=PASS (no MLX/PyTorch/CoreML modules loaded)")


def projection_smoke(checkpoint: Checkpoint, engine_path: str,
                     token_id: int, bits: int) -> None:
    """Compile and execute layer 0's real RMSNorm+GDN projection head."""
    prefix = "model.language_model.layers.0"
    names = [
        f"{prefix}.linear_attn.in_proj_qkv.weight",
        f"{prefix}.linear_attn.in_proj_z.weight",
        f"{prefix}.linear_attn.in_proj_b.weight",
        f"{prefix}.linear_attn.in_proj_a.weight",
    ]
    driver = AneDriver(engine_path)
    program = AneNormProjection(
        driver, checkpoint, f"{prefix}.input_layernorm.weight", names,
        bits=bits, tag="layer0_gdn_head"
    )
    hidden = checkpoint.embedding(token_id)
    # The private runtime can return stale/uninitialised output on the first
    # evaluation after load on some OS builds.  Warm the exact same surfaces
    # before timing and validating the result.
    program(hidden)
    program(hidden)
    t0 = time.perf_counter()
    out = program(hidden)
    elapsed = (time.perf_counter() - t0) * 1e3
    if out.shape != (16480,) or not np.isfinite(out).all():
        raise RuntimeError(f"invalid layer-0 output: {out.shape}")
    # This is a development-only oracle, never part of decode.  It proves the
    # private compiler did not silently accept a numerically broken graph.
    x32 = hidden.astype(np.float32)
    nw = checkpoint.tensor(f"{prefix}.input_layernorm.weight")
    normed = x32 / np.sqrt(np.mean(x32 * x32) + 1e-6) * nw
    errors, offset = [], 0
    for name in names:
        weight = checkpoint.tensor(name)
        ref = weight @ normed
        got = out[offset:offset + ref.size].astype(np.float32)
        errors.append(float(np.max(np.abs(got - ref)) /
                            (np.max(np.abs(ref)) + 1e-9)))
        offset += ref.size
    limit = 0.015 if bits == 16 else (0.20 if bits == 4 else 0.04)
    if max(errors) >= limit:
        raise RuntimeError(f"layer-0 numerical validation failed: {errors}")
    print(f"compiled=layer0_rmsnorm+qkv/z/b/a int{bits} "
          f"blob={program.nbytes/1e6:.1f}MB time={program.compile_seconds:.2f}s")
    print(f"dispatch={elapsed:.3f}ms output=shape{out.shape} "
          f"range=[{float(out.min()):.5g},{float(out.max()):.5g}]")
    print("span_relative_errors=" + ",".join(f"{x:.5g}" for x in errors))
    print("PURE_ANE_LAYER0_HEAD=PASS")
    assert_standalone("projection smoke test")


def gdn_conv_smoke(checkpoint: Checkpoint, engine_path: str,
                   token_id: int, bits: int) -> None:
    """Run two dependent layer-0 projection→conv steps entirely on ANE."""
    prefix = "model.language_model.layers.0"
    projection_names = [
        f"{prefix}.linear_attn.in_proj_qkv.weight",
        f"{prefix}.linear_attn.in_proj_z.weight",
        f"{prefix}.linear_attn.in_proj_b.weight",
        f"{prefix}.linear_attn.in_proj_a.weight",
    ]
    driver = AneDriver(engine_path)
    head = AneNormProjection(
        driver, checkpoint, f"{prefix}.input_layernorm.weight",
        projection_names, bits=bits, tag="layer0_gdn_head"
    )
    conv = AneGdnConv(
        driver, checkpoint, f"{prefix}.linear_attn.conv1d.weight"
    )
    history = np.zeros((conv.channels, 3), np.float32)
    errors, elapsed = [], []
    for tid in (token_id, token_id + 1):
        projection = head(checkpoint.embedding(tid))
        qkv = projection[:conv.channels].astype(np.float32)
        window = np.column_stack((history, qkv))
        raw = np.sum(window * conv.weight, axis=1)
        ref = raw / (1.0 + np.exp(-raw))
        t0 = time.perf_counter()
        got = conv(projection[:conv.channels]).astype(np.float32)
        elapsed.append((time.perf_counter() - t0) * 1e3)
        errors.append(float(np.max(np.abs(got - ref)) /
                            (np.max(np.abs(ref)) + 1e-9)))
        history[:, :2] = history[:, 1:3]
        history[:, 2] = qkv
    if max(errors) >= 0.006:
        raise RuntimeError(f"dependent GDN conv validation failed: {errors}")
    print(f"dependent_steps=2 projection=int{bits} conv=C{conv.channels}/K4")
    print("conv_relative_errors=" + ",".join(f"{x:.5g}" for x in errors))
    print("conv_dispatch_ms=" + ",".join(f"{x:.3f}" for x in elapsed))
    print("PURE_ANE_GDN_PROJECTION_CONV=PASS")
    assert_standalone("dependent GDN convolution smoke test")


def gdn_recurrence_smoke(checkpoint: Checkpoint, engine_path: str,
                         token_id: int, bits: int) -> None:
    """Validate two full projection→conv→norm/gates/recurrence steps."""
    prefix = "model.language_model.layers.0"
    projection_names = [
        f"{prefix}.linear_attn.in_proj_qkv.weight",
        f"{prefix}.linear_attn.in_proj_z.weight",
        f"{prefix}.linear_attn.in_proj_b.weight",
        f"{prefix}.linear_attn.in_proj_a.weight",
    ]
    driver = AneDriver(engine_path)
    head = AneNormProjection(
        driver, checkpoint, f"{prefix}.input_layernorm.weight",
        projection_names, bits=bits, tag="layer0_gdn_head"
    )
    conv = AneGdnConv(
        driver, checkpoint, f"{prefix}.linear_attn.conv1d.weight"
    )
    recurrence = AneGdnRecurrence(driver)
    slot = recurrence.new_state()
    a_log = checkpoint.tensor(f"{prefix}.linear_attn.A_log", np.float16)
    dt_bias = checkpoint.tensor(f"{prefix}.linear_attn.dt_bias", np.float16)
    state_ref = np.zeros((48, 128, 128), np.float32)
    y_errors, state_errors, elapsed = [], [], []
    for tid in (token_id, token_id + 1):
        projection = head(checkpoint.embedding(tid))
        activated = conv(projection[:10240]).astype(np.float32)
        q = activated[:2048].reshape(16, 128)
        k = activated[2048:4096].reshape(16, 128)
        v = activated[4096:10240].reshape(48, 128)
        b = projection[16384:16432].astype(np.float32)
        a = projection[16432:16480].astype(np.float32)
        qn = q / np.sqrt(np.mean(q*q, axis=1, keepdims=True) + 1e-6) / 128.0
        kn = k / np.sqrt(np.mean(k*k, axis=1, keepdims=True) + 1e-6) / np.sqrt(128.0)
        q48 = np.repeat(qn, 3, axis=0)
        k48 = np.repeat(kn, 3, axis=0)
        decay = np.exp(-np.exp(a_log.astype(np.float32)) *
                       np.logaddexp(a + dt_bias.astype(np.float32), 0.0))
        beta = 1.0 / (1.0 + np.exp(-b))
        state_ref *= decay[:, None, None]
        memory = np.sum(state_ref * k48[:, None, :], axis=-1)
        delta = (v - memory) * beta[:, None]
        state_ref += delta[:, :, None] * k48[:, None, :]
        # The ANE carries the recurrence output at 64x to preserve the tiny
        # values that BF16 represents but native fp16 contractions otherwise
        # quantize away.  The following gated RMSNorm absorbs this scale while
        # using epsilon*64^2, so model semantics are unchanged.
        y_ref = 64.0 * np.sum(state_ref * q48[:, None, :], axis=-1)
        t0 = time.perf_counter()
        got = recurrence(slot, q, k, v, a, b, a_log, dt_bias).astype(np.float32)
        elapsed.append((time.perf_counter() - t0) * 1e3)
        got_state = recurrence.materialize(slot)
        y_errors.append(float(np.max(np.abs(got-y_ref)) /
                              (np.max(np.abs(y_ref)) + 1e-9)))
        state_errors.append(float(np.max(np.abs(got_state-state_ref)) /
                                  (np.max(np.abs(state_ref)) + 1e-9)))
    if max(y_errors + state_errors) >= 0.012:
        raise RuntimeError(
            f"dependent GDN recurrence failed: y={y_errors} state={state_errors}"
        )
    print(f"dependent_steps=2 projection=int{bits} conv+qk_norm+gates+recurrence=ANE")
    print("y_relative_errors=" + ",".join(f"{x:.5g}" for x in y_errors))
    print("state_relative_errors=" + ",".join(f"{x:.5g}" for x in state_errors))
    print("recurrence_ms=" + ",".join(f"{x:.3f}" for x in elapsed))
    print("PURE_ANE_GDN_CORE=PASS")
    assert_standalone("dependent GDN recurrence smoke test")


def gdn_layer_smoke(checkpoint: Checkpoint, engine_path: str,
                    token_id: int, bits: int,
                    down_proj_parts: int = 1,width: int = 32) -> None:
    """Execute one complete Qwen decoder layer with ANE model arithmetic."""
    p = "model.language_model.layers.0"
    names = [f"{p}.linear_attn.{x}.weight" for x in
             ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
    driver = AneDriver(engine_path)
    head = AneNormProjection(
        driver, checkpoint, f"{p}.input_layernorm.weight", names,
        bits=bits, tag="layer0_gdn_head"
    )
    conv = AneGdnConv(driver, checkpoint, f"{p}.linear_attn.conv1d.weight")
    recurrence = AneGdnRecurrence(driver)
    slot = recurrence.new_state()
    tail = AneGdnTail(
        driver, checkpoint, 0, bits=bits,
        down_proj_parts=down_proj_parts,width=width
    )
    hidden = checkpoint.embedding(token_id)
    times = {}
    t0 = time.perf_counter(); projection = head(hidden); times["head"] = time.perf_counter()-t0
    t0 = time.perf_counter(); activated = conv(projection[:10240]); times["conv"] = time.perf_counter()-t0
    q = activated[:2048].reshape(16, 128)
    k = activated[2048:4096].reshape(16, 128)
    v = activated[4096:].reshape(48, 128)
    z = projection[10240:16384]
    b = projection[16384:16432]
    a = projection[16432:16480]
    al = checkpoint.tensor(f"{p}.linear_attn.A_log", np.float16)
    dt = checkpoint.tensor(f"{p}.linear_attn.dt_bias", np.float16)
    t0 = time.perf_counter(); core = recurrence(slot, q, k, v, a, b, al, dt); times["recurrence"] = time.perf_counter()-t0
    t0 = time.perf_counter(); got = tail(core.reshape(-1), z, hidden); times["tail"] = time.perf_counter()-t0
    if got.shape != (5120,) or not np.isfinite(got).all():
        raise RuntimeError(f"invalid full GDN layer output {got.shape}")

    # Development oracle for the tail.  Upstream values are the actual ANE
    # outputs, so this specifically validates gated norm + projections + MLP.
    cf = core.astype(np.float32).reshape(48, 128) / 64.0
    zw = z.astype(np.float32).reshape(48, 128)
    gn = checkpoint.tensor(f"{p}.linear_attn.norm.weight")
    cn = cf / np.sqrt(np.mean(cf*cf, axis=1, keepdims=True) + 1e-6) * gn
    gated = cn * (zw / (1.0 + np.exp(-zw)))
    h = hidden.astype(np.float32) + checkpoint.tensor(
        f"{p}.linear_attn.out_proj.weight"
    ) @ gated.reshape(-1)
    pn = checkpoint.tensor(f"{p}.post_attention_layernorm.weight")
    hn = h / np.sqrt(np.mean(h*h) + 1e-6) * pn
    gate = checkpoint.tensor(f"{p}.mlp.gate_proj.weight") @ hn
    up = checkpoint.tensor(f"{p}.mlp.up_proj.weight") @ hn
    act = (gate / (1.0 + np.exp(-gate))) * up
    ref = h + checkpoint.tensor(f"{p}.mlp.down_proj.weight") @ act
    rel = float(np.max(np.abs(got.astype(np.float32)-ref)) /
                (np.max(np.abs(ref)) + 1e-9))
    limit = 0.04 if bits == 16 else (0.40 if bits == 4 else 0.12)
    if rel >= limit:
        raise RuntimeError(
            f"full GDN layer tail validation failed: rel={rel} "
            f"got=[{float(got.min())},{float(got.max())}] "
            f"ref=[{float(ref.min())},{float(ref.max())}]"
        )
    tail_samples=[]
    for _ in range(9):
        started=time.perf_counter();tail(core.reshape(-1),z,hidden)
        tail_samples.append((time.perf_counter()-started)*1e3)
    print(f"layer=0 complete_gdn int{bits} down_proj_parts={down_proj_parts} "
          f"width={width} blob={tail.nbytes/1e6:.1f}MB "
          f"compile={tail.compile_seconds:.2f}s")
    print("dispatch_ms=" + ",".join(
        f"{name}:{value*1e3:.3f}" for name, value in times.items()))
    print(f"tail_relative_error={rel:.5g} output_range="
          f"[{float(got.min()):.5g},{float(got.max()):.5g}]")
    print(f"tail_median_ms={float(np.median(tail_samples)):.3f}")
    print("PURE_ANE_COMPLETE_GDN_LAYER=PASS")
    assert_standalone("complete GDN layer smoke test")


def attention_core_smoke(checkpoint: Checkpoint, engine_path: str,
                         token_id: int, bits: int) -> None:
    """Validate layer-3 projection, Q/K norm+RoPE, cache, and attention."""
    layer = 3
    p = f"model.language_model.layers.{layer}"
    names = [f"{p}.self_attn.{x}.weight" for x in ("q_proj", "k_proj", "v_proj")]
    driver = AneDriver(engine_path)
    head = AneNormProjection(
        driver, checkpoint, f"{p}.input_layernorm.weight", names,
        bits=bits, tag="layer3_attention_head"
    )
    shared_prepare = AneAttentionPrepareDynamic(driver)
    prepare = _DynamicPrepareLayer(
        shared_prepare,
        checkpoint.tensor(f"{p}.self_attn.q_norm.weight", np.float16),
        checkpoint.tensor(f"{p}.self_attn.k_norm.weight", np.float16),
    )
    core = AneAttentionCore(driver)
    qnw = checkpoint.tensor(f"{p}.self_attn.q_norm.weight")
    knw = checkpoint.tensor(f"{p}.self_attn.k_norm.weight")
    q_errors, k_errors, y_errors, elapsed = [], [], [], []
    keys, values = [], []
    for position, tid in enumerate((token_id, token_id + 1)):
        projection = head(checkpoint.embedding(tid))
        qp = projection[:12288].reshape(24, 512).astype(np.float32)
        q, gate = qp[:, :256], qp[:, 256:]
        k = projection[12288:13312].reshape(4, 256).astype(np.float32)
        v = projection[13312:14336].reshape(4, 256).astype(np.float32)
        qn = q / np.sqrt(np.mean(q*q, axis=1, keepdims=True)+1e-6) * qnw
        kn = k / np.sqrt(np.mean(k*k, axis=1, keepdims=True)+1e-6) * knw
        co, si = AneAttentionPrepare.rope_table(position)
        qref, kref = qn.copy(), kn.copy()
        qa, qb = qn[:, :32].copy(), qn[:, 32:64].copy()
        ka, kb = kn[:, :32].copy(), kn[:, 32:64].copy()
        qref[:, :32] = qa*co - qb*si; qref[:, 32:64] = qa*si + qb*co
        kref[:, :32] = ka*co - kb*si; kref[:, 32:64] = ka*si + kb*co
        qgot, kgot = prepare(q, k, position)
        q_errors.append(float(np.max(np.abs(qgot-qref)) /
                              (np.max(np.abs(qref))+1e-9)))
        k_errors.append(float(np.max(np.abs(kgot-kref)) /
                              (np.max(np.abs(kref))+1e-9)))
        keys.append(kgot.astype(np.float32)); values.append(v)
        ks = np.stack(keys, axis=1); vs = np.stack(values, axis=1)
        qgroup = qgot.astype(np.float32).reshape(4, 6, 256)
        scores = np.matmul(qgroup, ks.swapaxes(-1, -2)) / 16.0
        scores -= scores.max(axis=-1, keepdims=True)
        prob = np.exp(scores); prob /= prob.sum(axis=-1, keepdims=True)
        yref = np.matmul(prob, vs).reshape(24, 256)
        t0 = time.perf_counter(); ygot = core(qgot, kgot, v); elapsed.append(time.perf_counter()-t0)
        y_errors.append(float(np.max(np.abs(ygot-yref)) /
                              (np.max(np.abs(yref))+1e-9)))
        if gate.shape != (24, 256):
            raise RuntimeError("attention gate split is wrong")
    if max(q_errors+k_errors+y_errors) >= 0.025:
        raise RuntimeError(
            f"attention core validation failed q={q_errors} k={k_errors} y={y_errors}"
        )
    print(f"dependent_steps=2 layer=3 projection=int{bits} qk_norm+rope+attention=ANE")
    print("q_relative_errors=" + ",".join(f"{x:.5g}" for x in q_errors))
    print("k_relative_errors=" + ",".join(f"{x:.5g}" for x in k_errors))
    print("attention_relative_errors=" + ",".join(f"{x:.5g}" for x in y_errors))
    print("attention_ms=" + ",".join(f"{x*1e3:.3f}" for x in elapsed))
    print("PURE_ANE_ATTENTION_CORE=PASS")
    assert_standalone("attention core smoke test")


def long_attention_core_smoke(engine_path:str,context:int=512,
                              valid:int=257)->None:
    """Validate the integrated streamed core and ANE online-softmax merge."""
    if context<=256:
        raise ValueError("long attention smoke requires --context over 256")
    if not 257<=valid<=context:
        raise ValueError("long attention --valid must be 257..context")
    driver=AneDriver(engine_path);core=AneLongContextAttentionCore(driver,context)
    rng=np.random.default_rng(20_480+valid)
    q=rng.normal(0,.3,(24,256)).astype(np.float16)
    k=rng.normal(0,.3,(4,valid,256)).astype(np.float16)
    v=rng.normal(0,.3,(4,valid,256)).astype(np.float16)
    for block,start in enumerate(range(0,valid-1,core.B)):
        count=min(core.B,valid-1-start)
        core.keys[block,:,:count]=k[:,start:start+count]
        core.values[block,:,:count]=v[:,start:start+count]
    core.offset=valid-1
    got=core(q,k[:,-1],v[:,-1]).astype(np.float32)
    qg=q.astype(np.float32).reshape(4,6,256)
    scores=np.matmul(qg,k.astype(np.float32).swapaxes(-1,-2))*.0625
    scores-=scores.max(axis=-1,keepdims=True)
    prob=np.exp(scores);prob/=prob.sum(axis=-1,keepdims=True)
    ref=np.matmul(prob,v.astype(np.float32)).reshape(24,256)
    rel=float(np.max(np.abs(got-ref))/(np.max(np.abs(ref))+1e-9))
    if not np.isfinite(got).all() or rel>=2e-2:
        raise RuntimeError(f"long-context attention validation failed rel={rel}")
    print(f"PURE_ANE_LONG_ATTENTION=PASS context={context} valid={valid} "
          f"relative_error={rel:.6g} programs={len(core.programs)}")
    assert_standalone("long attention core smoke test")


def attention_layer_smoke(checkpoint: Checkpoint, engine_path: str,
                          token_id: int, bits: int,
                          down_proj_parts: int = 1,width: int = 32) -> None:
    """Execute a complete full-attention decoder layer on ANE."""
    layer = 3
    p = f"model.language_model.layers.{layer}"
    names = [f"{p}.self_attn.{x}.weight" for x in ("q_proj", "k_proj", "v_proj")]
    driver = AneDriver(engine_path)
    head = AneNormProjection(
        driver, checkpoint, f"{p}.input_layernorm.weight", names,
        bits=bits, tag="layer3_attention_head"
    )
    shared_prepare = AneAttentionPrepareDynamic(driver)
    prepare = _DynamicPrepareLayer(
        shared_prepare,
        checkpoint.tensor(f"{p}.self_attn.q_norm.weight", np.float16),
        checkpoint.tensor(f"{p}.self_attn.k_norm.weight", np.float16),
    )
    core_program = AneAttentionCore(driver)
    tail = AneAttentionTail(
        driver, checkpoint, layer, bits=bits,
        down_proj_parts=down_proj_parts,width=width
    )
    hidden = checkpoint.embedding(token_id)
    times = {}
    t0=time.perf_counter(); projection=head(hidden); times["head"]=time.perf_counter()-t0
    qp = projection[:12288].reshape(24,512)
    q, gate = qp[:,:256], qp[:,256:]
    k = projection[12288:13312].reshape(4,256)
    v = projection[13312:14336].reshape(4,256)
    t0=time.perf_counter(); q,k=prepare(q,k,0); times["norm_rope"]=time.perf_counter()-t0
    t0=time.perf_counter(); core=core_program(q,k,v); times["attention"]=time.perf_counter()-t0
    t0=time.perf_counter(); got=tail(core,gate,hidden); times["tail"]=time.perf_counter()-t0
    if got.shape != (5120,) or not np.isfinite(got).all():
        raise RuntimeError("invalid complete attention layer output")
    cf=core.astype(np.float32).reshape(-1); gf=gate.astype(np.float32).reshape(-1)
    gated=cf/(1.0+np.exp(-gf))
    h=hidden.astype(np.float32)+checkpoint.tensor(f"{p}.self_attn.o_proj.weight")@gated
    pn=checkpoint.tensor(f"{p}.post_attention_layernorm.weight")
    hn=h/np.sqrt(np.mean(h*h)+1e-6)*pn
    mg=checkpoint.tensor(f"{p}.mlp.gate_proj.weight")@hn
    mu=checkpoint.tensor(f"{p}.mlp.up_proj.weight")@hn
    act=mg/(1.0+np.exp(-mg))*mu
    ref=h+checkpoint.tensor(f"{p}.mlp.down_proj.weight")@act
    rel=float(np.max(np.abs(got.astype(np.float32)-ref))/(np.max(np.abs(ref))+1e-9))
    limit=0.04 if bits==16 else (0.40 if bits==4 else 0.12)
    if rel>=limit:
        raise RuntimeError(f"attention tail validation failed rel={rel}")
    tail_samples=[]
    for _ in range(9):
        started=time.perf_counter();tail(core,gate,hidden)
        tail_samples.append((time.perf_counter()-started)*1e3)
    print(f"layer=3 complete_attention int{bits} "
          f"down_proj_parts={down_proj_parts} width={width} "
          f"blob={tail.nbytes/1e6:.1f}MB "
          f"compile={tail.compile_seconds:.2f}s")
    print("dispatch_ms="+",".join(f"{k}:{v*1e3:.3f}" for k,v in times.items()))
    print(f"tail_relative_error={rel:.5g} output_range="
          f"[{float(got.min()):.5g},{float(got.max()):.5g}]")
    print(f"tail_median_ms={float(np.median(tail_samples)):.3f}")
    print("PURE_ANE_COMPLETE_ATTENTION_LAYER=PASS")
    assert_standalone("complete attention layer smoke test")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get(
        "Q38_MODEL", "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B"))
    p.add_argument("--engine-path", default=os.environ.get(
        "Q38_ANE_ENGINE", _REPO_ROOT))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    smoke = sub.add_parser("loader-smoke")
    smoke.add_argument("--text", default="Hello from the pure ANE backend")
    proj = sub.add_parser("projection-smoke")
    proj.add_argument("--token-id", type=int, default=9419)
    proj.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    gconv = sub.add_parser("gdn-conv-smoke")
    gconv.add_argument("--token-id", type=int, default=9419)
    gconv.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    grec = sub.add_parser("gdn-recurrence-smoke")
    grec.add_argument("--token-id", type=int, default=9419)
    grec.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    glayer = sub.add_parser("gdn-layer-smoke")
    glayer.add_argument("--token-id", type=int, default=9419)
    glayer.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    glayer.add_argument("--down-proj-parts",type=int,choices=(1,4),default=1)
    glayer.add_argument("--width",type=int,choices=(32,64),default=32)
    acore = sub.add_parser("attention-core-smoke")
    acore.add_argument("--token-id", type=int, default=9419)
    acore.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    along = sub.add_parser("attention-long-smoke")
    along.add_argument("--context",type=int,default=512)
    along.add_argument("--valid",type=int,default=257,
                       help="populated positions to validate")
    alayer = sub.add_parser("attention-layer-smoke")
    alayer.add_argument("--token-id", type=int, default=9419)
    alayer.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    alayer.add_argument("--down-proj-parts",type=int,choices=(1,4),default=1)
    alayer.add_argument("--width",type=int,choices=(32,64),default=32)
    infer = sub.add_parser("infer")
    infer.add_argument("--prompt", default="Reply with exactly: OK")
    infer.add_argument("--prompt-file",
                       help="read the prompt from a UTF-8 file (overrides --prompt)")
    infer.add_argument("--tokens", type=int, default=4)
    infer.add_argument("--bits", type=int, choices=(4,8,16), default=16)
    infer.add_argument("--raw-prompt", action="store_true")
    infer.add_argument("--verify-reference", action="store_true")
    infer.add_argument("--mtp-draft",type=int,choices=(0,1,2),default=0,
                       help="pure-ANE MTP speculative draft depth")
    infer.add_argument("--context",type=int,default=256,
                       help="KV-cache capacity; values over 256 use exact streamed ANE attention")
    infer.add_argument("--down-proj-parts",type=int,choices=(1,4),default=4,
                       help="input-channel partitions for packed ANE down_proj")
    args = p.parse_args()

    assert_standalone("startup")
    checkpoint = Checkpoint(args.model)
    if args.command == "inspect":
        print(json.dumps(checkpoint.architecture, indent=2))
        print(f"tensors={len(checkpoint.weight_map)} shards={len(set(checkpoint.weight_map.values()))}")
        print(f"embedding={checkpoint.embedding_name} {checkpoint.info(checkpoint.embedding_name).shape}")
        assert_standalone("inspection")
    elif args.command == "loader-smoke":
        loader_smoke(checkpoint, args.text)
    elif args.command == "projection-smoke":
        projection_smoke(checkpoint, args.engine_path, args.token_id, args.bits)
    elif args.command == "gdn-conv-smoke":
        gdn_conv_smoke(checkpoint, args.engine_path, args.token_id, args.bits)
    elif args.command == "gdn-recurrence-smoke":
        gdn_recurrence_smoke(checkpoint, args.engine_path, args.token_id, args.bits)
    elif args.command == "gdn-layer-smoke":
        gdn_layer_smoke(checkpoint, args.engine_path, args.token_id, args.bits,
                        args.down_proj_parts,args.width)
    elif args.command == "attention-core-smoke":
        attention_core_smoke(checkpoint, args.engine_path, args.token_id, args.bits)
    elif args.command == "attention-long-smoke":
        long_attention_core_smoke(args.engine_path,args.context,args.valid)
    elif args.command == "attention-layer-smoke":
        attention_layer_smoke(checkpoint, args.engine_path, args.token_id,
                              args.bits,args.down_proj_parts,args.width)
    elif args.command == "infer":
        prompt=(Path(args.prompt_file).read_text(encoding="utf-8")
                if args.prompt_file else args.prompt)
        pure_infer(checkpoint,args.engine_path,prompt,args.tokens,
                   args.bits,args.raw_prompt,args.verify_reference,args.mtp_draft,
                   args.context,args.down_proj_parts)


if __name__ == "__main__":
    main()
