#!/usr/bin/env python3
"""ANECompiler.framework low-level API probe (disassembly / struct recovery half).

The framework lives only in the dyld shared cache, so there is no file on disk
for objdump/otool to open.  Instead we dlopen it into this process, locate the
loaded mach-o header via the dyld image APIs, add the export offsets reported by
`dyld_info -exports`, and disassemble the bytes in memory with capstone.

Subcommands:
  exports                 list the 140 exported symbols (offset + name)
  dis SYM [SYM ...]       disassemble a symbol until its final ret
  init-sizes              recover the struct size each *LayerDescInitialize memsets
  scan-strings SYM        pull the ASCII literals a validator references (adrp/add pairs)

Companion file: probes/ane_compiler_api.m does the live validator calls.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import subprocess
import sys

FRAMEWORK = "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler"

try:
    import capstone
except ImportError:  # pragma: no cover
    capstone = None


# ---------------------------------------------------------------- image lookup

libc = ctypes.CDLL(None)
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.restype = ctypes.c_char_p
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
libc._dyld_get_image_header.restype = ctypes.c_void_p
libc._dyld_get_image_header.argtypes = [ctypes.c_uint32]


def load_framework() -> int:
    """dlopen ANECompiler and return the load address of its mach header."""
    handle = ctypes.CDLL(FRAMEWORK, mode=ctypes.RTLD_GLOBAL)
    assert handle is not None
    for i in range(libc._dyld_image_count()):
        name = libc._dyld_get_image_name(i)
        if name and b"ANECompiler" in name:
            return int(libc._dyld_get_image_header(i))
    raise SystemExit("ANECompiler dlopened but not found in the dyld image list")


def exports() -> dict[str, int]:
    out = subprocess.run(
        ["dyld_info", "-exports", FRAMEWORK],
        capture_output=True, text=True, check=True,
    ).stdout
    table: dict[str, int] = {}
    for line in out.splitlines():
        m = re.match(r"\s+0x([0-9A-Fa-f]+)\s+(\S+)\s*$", line)
        if m:
            table[m.group(2)] = int(m.group(1), 16)
    return table


# ------------------------------------------------------------------ disassembly

def read(addr: int, n: int) -> bytes:
    return ctypes.string_at(addr, n)


def extent(sym: str, table: dict[str, int], cap: int = 1 << 18) -> int:
    """Byte length of a function, taken as the gap to the next exported symbol.

    The validators are laid out contiguously in __text, so this is exact for
    them and merely an upper bound elsewhere.
    """
    offs = sorted(table.values())
    off = table[sym]
    i = offs.index(off)
    nxt = offs[i + 1] if i + 1 < len(offs) else off + cap
    return min(nxt - off, cap)


def md():
    if capstone is None:
        raise SystemExit("pip install capstone (use a venv) to disassemble")
    m = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_LITTLE_ENDIAN)
    m.detail = True
    return m


def disassemble(base: int, offset: int, limit: int = 4096):
    """Disassemble from base+offset, stopping after the ret that closes the frame."""
    code = read(base + offset, limit)
    insns = []
    for ins in md().disasm(code, base + offset):
        insns.append(ins)
        if ins.mnemonic in ("ret", "retab", "retaa", "brk"):
            break
    return insns


def print_dis(base: int, sym: str, table: dict[str, int], limit: int = 4096):
    off = table.get(sym)
    if off is None:
        print(f"!! {sym} not exported")
        return
    print(f"\n===== {sym}  (+0x{off:X})")
    for ins in disassemble(base, off, limit):
        print(f"  {ins.address - base:#010x}  {ins.mnemonic:<10} {ins.op_str}")


# --------------------------------------------------------------- init-size scan

MEMSET_HINT = re.compile(r"^#(0x[0-9a-f]+|\d+)$")


def init_sizes(base: int, table: dict[str, int]):
    """Every *LayerDescInitialize is a tiny thunk; recover the size it clears.

    The shipping pattern is `mov w1, #0 ; mov w2, #<size> ; b _memset` (or an
    inlined stp/str sequence).  We report the immediate handed to memset, plus
    any constant stores that follow, which are the version/magic defaults.
    """
    rows = []
    for sym, off in sorted(table.items(), key=lambda kv: kv[1]):
        if "DescInitialize" not in sym and "Initialize" not in sym:
            continue
        insns = disassemble(base, off, 256)
        size = None
        stores = []
        for ins in insns:
            if ins.mnemonic in ("mov", "movz", "orr") and ins.op_str.startswith(("w2,", "x2,")):
                m = re.search(r"#(0x[0-9a-fA-F]+|\d+)", ins.op_str)
                if m:
                    size = int(m.group(1), 0)
            if ins.mnemonic in ("str", "strb", "strh", "stp"):
                stores.append(f"{ins.mnemonic} {ins.op_str}")
        rows.append((sym, off, size, len(insns), stores))
    w = max(len(r[0]) for r in rows)
    for sym, off, size, n, stores in rows:
        s = f"{size} (0x{size:X})" if size else "?"
        print(f"{sym:<{w}}  +0x{off:06X}  memset_size={s:<14} insns={n:<4}")
        for st in stores[:6]:
            print(f"{'':<{w}}    {st}")


# ------------------------------------------------------------- string scanning

def cfstring_at(addr: int) -> str | None:
    """Decode a constant CFString (isa, flags, char*, length)."""
    try:
        w = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint64))
        flags, ptr, length = w[1], w[2], w[3]
        if not (0 < length < 4096) or ptr < 0x1000:
            return None
        raw = read(ptr, length)
        if all(32 <= c < 127 for c in raw):
            return raw.decode()
    except Exception:
        return None
    return None


def scan_strings(base: int, sym: str, table: dict[str, int]):
    """Collect literals reachable from adrp/add pairs inside one function.

    Validators build rejection messages, and pick enum values, out of constant
    C strings and constant CFStrings, so this reads out the constraints they
    enforce without having to follow the control flow.
    """
    off = table.get(sym)
    if off is None:
        print(f"!! {sym} not exported")
        return
    n = extent(sym, table)
    code = read(base + off, n)
    pages: dict[str, int] = {}
    seen: list[tuple[str, str]] = []

    def note(kind: str, txt: str):
        if (kind, txt) not in seen:
            seen.append((kind, txt))

    for ins in md().disasm(code, base + off):
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
                    note("cfstr", cf)
                    continue
                try:
                    raw = read(addr, 300)
                except Exception:
                    continue
                s = raw.split(b"\0", 1)[0]
                if len(s) >= 4 and all(32 <= c < 127 for c in s):
                    note("cstr", s.decode())
    print(f"\n===== literals in {sym} (+0x{off:X}, {n} bytes)")
    for kind, s in seen:
        print(f"  {kind:<6} {s!r}")


# ----------------------------------------------------------------------- driver

def main(argv: list[str]) -> int:
    base = load_framework()
    table = exports()
    print(f"# ANECompiler @ {base:#x}, {len(table)} exports", file=sys.stderr)

    if not argv or argv[0] == "exports":
        for sym, off in sorted(table.items(), key=lambda kv: kv[1]):
            print(f"0x{off:08X}  {sym}")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "dis":
        for sym in rest:
            print_dis(base, sym if sym.startswith("_") else "_" + sym, table)
    elif cmd == "init-sizes":
        init_sizes(base, table)
    elif cmd == "scan-strings":
        for sym in rest:
            scan_strings(base, sym if sym.startswith("_") else "_" + sym, table)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
