#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ane_hwx_inspector.py - Extract, dissect, and parse .hwx binary containers on Apple Silicon (M5 Max)."""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import sys
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (
    AneEngine,
    _desc,
    _msg,
    _iosurface_view,
    _nsdict,
    _objc,
    _sel,
)


def generate_valid_linear_mil(in_dim: int, out_dim: int, seq_len: int) -> str:
    """Generate known-valid MIL text for 1x1 linear convolution."""
    return f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {in_dim}, 1, {seq_len}]> x) {{
    tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> w = const()[name=string("w"), val=tensor<fp16, [{out_dim}, {in_dim}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<int32, [2]> strides = const()[name=string("strides"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [2]> dil = const()[name=string("dil"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pad = const()[name=string("pad"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<fp16, [1, {out_dim}, 1, {seq_len}]> y = conv(dilations=dil, groups=int32(1), pad=pad, pad_type=string("valid"), strides=strides, weight=w, x=x)[name=string("y")];
  }} -> (y);
}}
"""


def generate_valid_conv1d_mil(channels: int, seq_len: int, kernel_size: int = 4) -> str:
    """Generate known-valid MIL text for depthwise causal 1d convolution."""
    return f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {channels}, 1, {seq_len}]> x) {{
    tensor<fp16, [{channels}, 1, 1, {kernel_size}]> w = const()[name=string("w"), val=tensor<fp16, [{channels}, 1, 1, {kernel_size}]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<int32, [2]> strides = const()[name=string("strides"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [2]> dil = const()[name=string("dil"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pad = const()[name=string("pad"), val=tensor<int32, [4]>([0,0,{kernel_size-1},0])];
    tensor<fp16, [1, {channels}, 1, {seq_len}]> c = conv(dilations=dil, groups=int32({channels}), pad=pad, pad_type=string("custom"), strides=strides, weight=w, x=x)[name=string("c")];
    tensor<fp16, [1, {channels}, 1, {seq_len}]> y = mul(x=c, y=fp16(0x1p+0))[name=string("y")];
  }} -> (y);
}}
"""


def compile_and_extract(
    engine: AneEngine,
    mil_text: str,
    weight_bytes: bytes,
    in_dim: int,
    out_dim: int,
    seq_len: int,
    tag: str,
    save_dir: Path,
) -> tuple[Path, bytes]:
    """Compile MIL text, extract the generated model.hwx package, and save it."""
    os.environ["Q38_ANE_REUSE_COMPILED"] = "0"
    prog = engine.compile_multiproc(
        mil_text,
        {"w.bin": weight_bytes},
        in_dim,
        out_dim,
        seq_len,
    )
    assert prog is not None, f"Failed to compile {tag}"

    local_path = _desc(_msg(prog.model, "localModelPath"))
    assert os.path.exists(local_path), f"localModelPath does not exist: {local_path}"

    hwx_path = Path(local_path) / "model.hwx"
    if not hwx_path.exists():
        found = list(Path(local_path).glob("**/*.hwx"))
        if found:
            hwx_path = found[0]

    assert hwx_path.exists(), f"model.hwx not found in {local_path}"
    hwx_bytes = hwx_path.read_bytes()

    dest_dir = save_dir / tag
    shutil.rmtree(dest_dir, ignore_errors=True)
    shutil.copytree(local_path, dest_dir)
    print(f"  ✓ [{tag}] Saved package ({len(hwx_bytes)} B hwx) to: {dest_dir}")

    return dest_dir / "model.hwx", hwx_bytes


def dissect_hwx(data: bytes, tag: str):
    """Detailed structural dissection of the ANE .hwx binary container."""
    print("\n" + "=" * 70)
    print(f"  DISSECTING .HWX: {tag} ({len(data)} bytes / 0x{len(data):x})")
    print("=" * 70)

    # 1. First 64 bytes hex dump
    print("\n--- [1] Header Block (0x0000 - 0x0040) ---")
    for i in range(0, min(64, len(data)), 16):
        chunk = data[i:i+16]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  0x{i:04x}: {hex_str:<48}  |{ascii_str}|")

    # 2. Extract header fields
    magic = data[0:4]
    magic_str = "".join(chr(b) if 32 <= b < 127 else "." for b in magic)
    print(f"\n  Magic / Identifier: 0x{struct.unpack('<I', magic)[0]:08x} ('{magic_str}')")

    # Scan for architecture signature
    for tag_candidate in [b"H17P", b"H16P", b"H13P", b"HWX0", b"ANEF", b"AIR0", b"MIL0"]:
        idx = data.find(tag_candidate)
        if idx != -1:
            print(f"  Target Architecture Tag: '{tag_candidate.decode()}' at 0x{idx:04x}")

    # 3. Look for 32-bit section headers / pointers
    print("\n--- [2] Header 32-bit Words (Offsets & Table Pointers) ---")
    num_words = min(32, len(data) // 4)
    for w in range(num_words):
        val = struct.unpack("<I", data[w*4:(w+1)*4])[0]
        # Check if val looks like an offset within the file
        notes = []
        if val == len(data):
            notes.append("== Total File Size")
        elif 0 < val < len(data):
            notes.append(f"-> Offset 0x{val:04x}")
            # Peek at destination
            if val + 4 <= len(data):
                dest_w = struct.unpack("<I", data[val:val+4])[0]
                dest_str = "".join(chr(b) if 32 <= b < 127 else "." for b in data[val:val+4])
                notes.append(f"[Val at dest: 0x{dest_w:08x} '{dest_str}']")
        note_str = " (" + ", ".join(notes) + ")" if notes else ""
        print(f"  Word {w:2d} (0x{w*4:02x}): 0x{val:08x} ({val:8d}){note_str}")


def main():
    print("=" * 70)
    print("  Apple Neural Engine .hwx Binary Container Inspector (M5 Max)")
    print("=" * 70)

    engine = AneEngine()
    if not engine.available:
        print("ERROR: ANE framework not available!")
        sys.exit(1)

    out_dir = _REPO_ROOT / "probes" / "captured_hwx"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        # (tag, kind, in_dim, out_dim, seq_len)
        ("linear_fp16_64x64_s32", "linear", 64, 64, 32),
        ("linear_fp16_64x64_s64", "linear", 64, 64, 64),
        ("linear_fp16_128x64_s32", "linear", 64, 128, 32),
        ("conv1d_fp16_c64_s32", "conv1d", 64, 64, 32),
    ]

    rng = np.random.default_rng(42)
    for tag, kind, in_dim, out_dim, seq_len in test_cases:
        if kind == "linear":
            w = rng.normal(0, 0.1, (out_dim, in_dim, 1, 1)).astype(np.float16)
            mil = generate_valid_linear_mil(in_dim, out_dim, seq_len)
        else:
            w = rng.normal(0, 0.1, (in_dim, 1, 1, 4)).astype(np.float16)
            mil = generate_valid_conv1d_mil(in_dim, seq_len, 4)

        hwx_path, hwx_bytes = compile_and_extract(
            engine, mil, w.tobytes(), in_dim, out_dim, seq_len, tag, out_dir
        )
        dissect_hwx(hwx_bytes, tag)

    print("\n" + "=" * 70)
    print("  PHASE 1 COMPLETE: Extracted and analyzed 4 micro-kernel .hwx containers")
    print("=" * 70)


if __name__ == "__main__":
    main()
