#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""rindi_mem.py - what a running Rindi process actually costs in memory.

macOS reports at least three different numbers for the same process and none
of them is self-explanatory:

  ps RSS            undercounts badly - IOAccelerator mappings barely appear
  phys_footprint    what Activity Monitor shows; EXCLUDES clean file-backed
                    pages, so the mmap'd weights are invisible in it
  vmmap resident    footprint + those file-backed pages = the honest total

This prints all three, says which is which, and breaks the total down by
where it actually went.

Usage:  tools/rindi_mem.py [pid | process-name-fragment]
        (with no argument, finds a running rindi-fm / rindi-server)
"""

import re
import subprocess
import sys

SIZE = re.compile(r"^\d+(\.\d+)?[KMGT]?$")


def to_bytes(token: str) -> float:
    """'34.4G' -> bytes. Bare integers are counts, not sizes."""
    unit = token[-1]
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}.get(unit)
    if scale is None:
        return float(token)
    return float(token[:-1]) * scale


def human(value: float) -> str:
    for unit, scale in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20)):
        if abs(value) >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value / 1024:.0f} KB"


def find_pid(hint: str | None) -> int:
    if hint and hint.isdigit():
        return int(hint)
    pattern = hint or "rindi-fm|rindi-server"
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    if not pids:
        sys.exit(f"no running process matching {pattern!r}")
    return pids[-1]


def parse_vmmap(pid: int):
    """Returns (footprint_bytes, {region_name: {field: bytes}})."""
    out = subprocess.run(["vmmap", "--summary", str(pid)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"vmmap failed for pid {pid}:\n{out.stderr.strip()}")

    footprint = 0.0
    regions: dict[str, dict[str, float]] = {}
    fields = ("virtual", "resident", "dirty", "swapped",
              "volatile", "nonvol", "empty", "count")

    for line in out.stdout.splitlines():
        if line.startswith("Physical footprint:"):
            footprint = to_bytes(line.split()[-1])
            continue
        # Stop before "TOTAL, minus reserved VM space" and the MALLOC ZONE
        # table that follows it; both repeat names with different meanings.
        # Match the MALLOC ZONE table's own header only - region rows carry a
        # trailing "see MALLOC ZONE table below" comment that must not match.
        if line.startswith("TOTAL, minus") or line.startswith("MALLOC ZONE"):
            break

        tokens = line.split()
        if len(tokens) < 9:
            continue
        # Walk left-to-right until the first size-looking token; everything
        # before it is the region name, the next eight are its columns.
        start = next((i for i, tok in enumerate(tokens) if SIZE.match(tok)), None)
        if start is None or start == 0 or len(tokens) < start + 8:
            continue
        name = " ".join(tokens[:start])
        values = tokens[start:start + 8]
        if not all(SIZE.match(v) for v in values):
            continue
        regions.setdefault(name, {f: to_bytes(v) for f, v in zip(fields, values)})

    return footprint, regions


def system_summary() -> str:
    out = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("PhysMem:"):
            return line.strip()
    return "PhysMem: unavailable"


def main() -> None:
    hint = sys.argv[1] if len(sys.argv) > 1 else None
    pid = find_pid(hint)
    footprint, regions = parse_vmmap(pid)

    def field(name: str, key: str = "resident") -> float:
        return regions.get(name, {}).get(key, 0.0)

    accel = field("IOAccelerator (graphics)") + field("IOAccelerator")
    mapped = field("mapped file")
    reserved = regions.get("Neural Engine (reserved)", {}).get("virtual", 0.0)
    mapped_virtual = regions.get("mapped file", {}).get("virtual", 0.0)
    swapped = field("IOAccelerator (graphics)", "swapped")
    heap = sum(values["resident"] for name, values in regions.items()
               if name.startswith("Malloc"))
    total = field("TOTAL")
    accel_regions = int(regions.get("IOAccelerator (graphics)", {}).get("count", 0))

    ps = subprocess.run(["ps", "-o", "rss=,comm=", "-p", str(pid)],
                        capture_output=True, text=True).stdout.split(None, 1)
    rss = float(ps[0]) * 1024 if ps else 0.0
    name = ps[1].strip().split("/")[-1] if len(ps) > 1 else "?"

    print(f"\n{name} [{pid}]\n")
    print("  the three numbers macOS will give you")
    print(f"    ps RSS                      {human(rss):>10}   undercounts IOAccelerator")
    print(f"    phys_footprint              {human(footprint):>10}   <- Activity Monitor shows this")
    print(f"    vmmap resident              {human(total):>10}   <- the honest total\n")
    other = total - accel - mapped - heap
    print("  where it went")
    print(f"    ANE + GPU buffers           {human(accel):>10}   IOAccelerator, {accel_regions} regions")
    print(f"    mmap'd weights (resident)   {human(mapped):>10}   of {human(mapped_virtual)} mapped")
    print(f"    heap                        {human(heap):>10}")
    print(f"    everything else             {human(other):>10}")
    # Never leave a mystery bucket: name what is actually in it.
    counted = {"IOAccelerator (graphics)", "IOAccelerator", "mapped file", "TOTAL"}
    rest = [(n, v["resident"]) for n, v in regions.items()
            if n not in counted and not n.startswith("Malloc")
            and "reserved" not in n and v["resident"] > 0]
    for region_name, size in sorted(rest, key=lambda kv: -kv[1])[:4]:
        print(f"        {region_name[:26]:<26}{human(size):>10}")
    if swapped:
        print(f"    (of which swapped out)      {human(swapped):>10}")
    if reserved:
        print(f"    reserved address space      {human(reserved):>10}   no physical cost\n")
    else:
        print()
    gap = total - footprint
    if gap > 0:
        print(f"  Activity Monitor is under by {human(gap)}: phys_footprint excludes clean")
        print("  file-backed pages, so the mmap'd weight files do not appear in it.\n")
    print(f"  system: {system_summary()}\n")


if __name__ == "__main__":
    main()
