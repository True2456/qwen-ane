"""Attribute live host allocations. RSS and ``o.base is None`` both lie here.

The previous FLASHNEXT_MEM_REPORT scan skipped every numpy view, so the 5 GB
of mixer/shared copies (views of the uint32 shift buffers) reported as empty.
Walk roots, include views, torch, and large bytes objects.
"""
from __future__ import annotations

import gc
import os
import sys
from collections import Counter, defaultdict


def _root_array(arr):
    seen = set()
    cur = arr
    while getattr(cur, "base", None) is not None:
        ident = id(cur.base)
        if ident in seen:
            break
        seen.add(ident)
        cur = cur.base
    return cur


def _nbytes(obj) -> int:
    n = getattr(obj, "nbytes", None)
    if n is not None:
        return int(n)
    n = getattr(obj, "nelement", None)
    el = getattr(obj, "element_size", None)
    if callable(n) and callable(el):
        try:
            return int(n()) * int(el())
        except Exception:
            return 0
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    return 0


def scan(min_bytes: int = 4 << 20) -> dict:
    """Return a dict of category -> [(label, bytes)] plus totals.

    Numpy views are billed to their root allocation so a 12 MB mixer is not
    counted twice as view + uint32 base.
    """
    gc.collect()
    np = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    roots: dict[int, tuple[object, int]] = {}
    views_of: dict[int, int] = defaultdict(int)
    torch_items: list[tuple[str, int]] = []
    blob_items: list[tuple[str, int]] = []

    for o in gc.get_objects():
        try:
            if np is not None and isinstance(o, np.ndarray):
                n = int(o.nbytes)
                if n < min_bytes:
                    continue
                root = _root_array(o)
                rid = id(root)
                if o is not root:
                    views_of[rid] += 1
                if rid not in roots or n > roots[rid][1]:
                    roots[rid] = (root, int(getattr(root, "nbytes", n)))
                continue
            if torch is not None and isinstance(o, torch.Tensor):
                n = _nbytes(o)
                if n >= min_bytes:
                    torch_items.append(
                        (f"torch {tuple(o.shape)} {o.dtype} {o.device}", n))
                continue
            if isinstance(o, (bytes, bytearray, memoryview)) and len(o) >= min_bytes:
                blob_items.append((type(o).__name__, len(o)))
        except Exception:
            continue

    np_items = []
    np_total = 0
    for root, n in roots.values():
        try:
            label = f"numpy {tuple(getattr(root, 'shape', ()))} {root.dtype}"
        except Exception:
            label = f"numpy {type(root).__name__}"
        np_items.append((label, n))
        np_total += n
    np_items.sort(key=lambda kv: -kv[1])
    torch_items.sort(key=lambda kv: -kv[1])
    blob_items.sort(key=lambda kv: -kv[1])
    return {
        "numpy_total": np_total,
        "numpy": np_items,
        "numpy_view_roots": len(views_of),
        "torch_total": sum(n for _, n in torch_items),
        "torch": torch_items,
        "bytes_total": sum(n for _, n in blob_items),
        "bytes": blob_items,
    }


def host_layer_breakdown(host_layers) -> list[tuple[str, int]]:
    """Bytes held on each HostLayer field, regardless of gc tracking."""
    rows = []
    tally = Counter()
    for i, hl in host_layers.items():
        for pack_name in ("attn", "mlp"):
            pack = getattr(hl, pack_name, None)
            if pack is None:
                continue
            for field in ("hc_n", "down_w", "up_w", "inj_w"):
                a = getattr(pack, field, None)
                n = _nbytes(a) if a is not None else 0
                if n:
                    tally[f"mixer {pack_name}.{field}"] += n
        moe = getattr(hl, "moe", None)
        if moe is None:
            continue
        for field in (
            "router_w", "gu", "dn", "shared_gate", "shared_up", "shared_down",
            "shared_sgate", "gu_buf", "dn_buf", "gu_f16", "dn_f16",
            "gate_w", "up_w", "down_w",
        ):
            a = getattr(moe, field, None)
            n = _nbytes(a) if a is not None else 0
            if n:
                tally[f"moe.{field}"] += n
        resident = getattr(moe, "_resident", None)
        if resident is not None:
            n = int(getattr(resident, "nbytes", 0) or 0)
            if n:
                tally["moe._resident (mlx)"] += n
    rows = list(tally.items())
    rows.sort(key=lambda kv: -kv[1])
    return rows


def print_report(host_layers=None, file=None) -> None:
    out = file or sys.stdout
    info = scan()
    gb = 1024 ** 3

    def line(msg: str) -> None:
        print(msg, file=out, flush=True)

    line(f"  host arrays over 4 MB (roots, including views): "
         f"{info['numpy_total'] / gb:.2f} GB  "
         f"({info['numpy_view_roots']} roots have views)")
    tally = Counter()
    for label, n in info["numpy"]:
        tally[label] += n
    for label, n in tally.most_common(15):
        line(f"    {n / gb:7.3f} GB  {label}")
    if info["torch"]:
        line(f"  torch tensors over 4 MB: {info['torch_total'] / gb:.2f} GB")
        for label, n in info["torch"][:10]:
            line(f"    {n / gb:7.3f} GB  {label}")
    if info["bytes"]:
        line(f"  bytes/bytearray over 4 MB: {info['bytes_total'] / gb:.2f} GB")
        for label, n in info["bytes"][:8]:
            line(f"    {n / gb:7.3f} GB  {label}")
    if host_layers:
        rows = host_layer_breakdown(host_layers)
        total = sum(n for _, n in rows)
        line(f"  HostLayer fields: {total / gb:.2f} GB")
        for label, n in rows:
            if n >= 1 << 20:
                line(f"    {n / gb:7.3f} GB  {label}")
    pid = os.getpid()
    line(f"  pid {pid}  (footprint -p {pid} for IOAccelerator/Foundation)")
