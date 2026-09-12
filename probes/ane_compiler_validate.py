#!/usr/bin/env python3
"""ANECompiler C-API, the live half: initialize the layer descriptors the MIL
frontend refuses (TopK / Sort / Gather / SDPA / RingBufferWriter) and push
them at the unit validators, one crash-isolated subprocess per op.

Rung ladder per op: symbol exists -> DescInitialize runs -> validator
created -> validator verdict (accept=1 / reject=0 / crash). "Validator
accepts" is not "executes"; _ANECCreateModelDictionary + _ANECCompile are
the next rungs and are out of scope here.

Subcommands:
  strings            literals referenced by the (shared) validator code
  init               hexdump default descriptors after *LayerDescInitialize
  validate OP        one op per process; exit 0/1/2 = accept/reject/crash
  validate-all       parent driver: forks one `validate OP` child per op

Needs capstone for `strings` only. Everything crashes safely except `init`,
which calls nothing more exotic than a memset thunk.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ane_compiler_api import (  # noqa: E402
    load_framework, exports, read, disassemble, extent, cfstring_at, md,
)

OPS = ["TopK", "Sort", "Gather", "SDPA", "RingBufferWriter", "ArgMinMax",
       "MatrixMult", "RMSNorm"]
DESC_SIZE = 0x400

_libc = ctypes.CDLL(None)


def sym(base: int, table: dict[str, int], name: str) -> int:
    off = table.get("_ANEC" + name) or table.get(name)
    if off is None:
        raise KeyError(name)
    return base + off


# ------------------------------------------------------------------- strings

def follow_branch(base: int, table: dict[str, int], name: str) -> int | None:
    """Final unconditional-branch target of a validator thunk."""
    target = None
    for ins in disassemble(base, table["_ANECValidate" + name + "Layer"], 512):
        if ins.mnemonic == "b":
            target = int(ins.op_str.lstrip("#"), 0)
    return target


def cmd_strings(base: int, table: dict[str, int]) -> None:
    import capstone  # noqa: F401  (fail loudly here, not later)
    seen = set()
    for op in OPS + ["ArgMinMax", "GlobalArgMinMax"]:
        tgt = follow_branch(base, table, op)
        if tgt is None:
            continue
        if tgt in seen:
            continue
        seen.add(tgt)
        rel = tgt - base
        print(f"\n== shared impl reached from {op} validator: +0x{rel:X}")
        code = read(tgt, 2048)
        pages: dict[str, int] = {}
        for ins in md().disasm(code, tgt):
            if ins.mnemonic == "adrp":
                reg, imm = [x.strip() for x in ins.op_str.split(",", 1)]
                pages[reg] = int(imm.lstrip("#"), 0)
            elif ins.mnemonic == "add" and "#" in ins.op_str:
                parts = [x.strip() for x in ins.op_str.split(",")]
                if len(parts) == 3 and parts[1] in pages:
                    try:
                        addr = pages[parts[1]] + int(parts[2].lstrip("#"), 0)
                    except ValueError:
                        continue
                    cf = cfstring_at(addr)
                    if cf:
                        print(f"  cfstr {cf!r}")
                        continue
                    try:
                        raw = read(addr, 200)
                    except Exception:
                        continue
                    s = raw.split(b"\0", 1)[0]
                    if len(s) >= 6 and all(32 <= c < 127 for c in s):
                        print(f"  cstr  {s.decode()!r}")


# ---------------------------------------------------------------------- init

def hexdump_nonzero(buf: bytes, base: int, label: str) -> None:
    print(f"  -- {label}")
    w = memoryview(buf)
    for off in range(0, len(buf), 8):
        word = int.from_bytes(w[off:off + 8], "little")
        if word == 0:
            continue
        note = ""
        if 0x1_0000_0000 <= word < 0x8000_0000_0000:
            cf = cfstring_at(word)
            if cf:
                note = f'  -> CFString "{cf}"'
        print(f"    +0x{off:03x}: 0x{word:016x}{note}")


def cmd_init(base: int, table: dict[str, int]) -> None:
    for op in ["TensorDims", "TensorDesc", "KernelSize", "Padding"] + OPS:
        name = op + ("LayerDescInitialize" if op in OPS else "Initialize")
        fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
            sym(base, table, name))
        buf = ctypes.create_string_buffer(DESC_SIZE)
        fn(buf)
        hexdump_nonzero(buf.raw, base, name)


# ------------------------------------------------------------------ validate

def make_validator(base: int, table: dict[str, int]) -> int:
    """_ANECUnitValidatorCreate(NULL, CFSTR("h17"), &out) -> status 0.

    Recovered by fuzz: arg0 unused, arg1 a CFString arch property (exactly
    "h17" on M5 Max; "H17C" is rejected with status 1), arg2 the out-pointer.
    All-NULL gives status 1; a zeroed garbage arg1 segfaults (treated as a
    CFType and retained).
    """
    cf = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                             ctypes.c_uint32]
    arch = cf.CFStringCreateWithCString(None, b"h17", 0x08000100)
    out = ctypes.create_string_buffer(16)
    create = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p)(
        sym(base, table, "UnitValidatorCreate"))
    status = create(None, arch, ctypes.cast(out, ctypes.c_void_p))
    validator = int.from_bytes(out.raw[:8], "little")
    if status != 0 or validator < 0x10000:
        raise RuntimeError(f"validator create failed: status={status:#x}")
    return validator


def child_validate(op: str) -> int:
    """Runs INSIDE the crash-isolated child. 0 accept, 1 reject, 2 crash-ish."""
    base = load_framework()
    table = exports()
    buf = ctypes.create_string_buffer(DESC_SIZE)
    if "_ANEC" + op + "LayerDescInitialize" in table:
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
            sym(base, table, op + "LayerDescInitialize"))(buf)
    else:
        print(f"{op}: no DescInitialize export; zeroed desc")

    validator = make_validator(base, table)
    print(f"{op}: validator @ {validator:#x}")

    if "_ANECValidate" + op + "Layer" not in table:
        print(f"{op}: no Validate export (folded into another validator?)")
        return 1
    validate = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p)(
        sym(base, table, "Validate" + op + "Layer"))
    r = validate(validator, buf, None, None, None, None)
    print(f"{op}: Validate -> {r}")
    return 0 if r == 1 else 1


def cmd_validate_all() -> None:
    me = os.path.abspath(__file__)
    for op in OPS:
        p = subprocess.run([sys.executable, me, "validate", op],
                           capture_output=True, text=True, timeout=120)
        state = {0: "ACCEPT", 1: "REJECT"}.get(p.returncode, "CRASH/ABORT")
        print(f"{op:<18} {state} (exit {p.returncode})")
        for line in (p.stdout + p.stderr).strip().splitlines()[-4:]:
            print(f"    {line}")


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "validate":
        return child_validate(argv[1])
    base = load_framework()
    table = exports()
    if argv and argv[0] == "strings":
        cmd_strings(base, table)
    elif argv and argv[0] == "init":
        cmd_init(base, table)
    elif argv and argv[0] == "validate-all":
        cmd_validate_all()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
