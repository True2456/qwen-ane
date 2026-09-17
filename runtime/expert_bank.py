"""MLX 4-bit routed-expert library for host MoE.

Reads ``models/Qwen3.8-Flash-Next-MLX-4bit/model.safetensors``
(affine gs=64, bits=4 on ``switch_mlp.{gate,up,down}_proj``). GDN/QSA stay
the fp16 ANE ``.aimodel``s. Shared expert is 8-bit in that checkpoint (tiny).
Does not touch ``ngram_embedding``. Does not write the BF16 tree.

Default decode is the **fp16 hot store** (``artifacts/experts_f16`` +
in-process LRU). This 4-bit bank is **opt-in**: ``FLASHNEXT_MOE=hybrid``
keeps it as a compact RAM store and dequants only new top-10 into the
packed SwiGLU workspace, then host GEMV. ``FLASHNEXT_MOE=q4gemv`` restores
fused 4-bit C GEMV (0.805 tok/s regression). Hybrid first-touch dequant
against a cache-cold 60 GB bank lost the 4-token race (0.447 vs 1.11 tok/s).

This is **mlx-lm 4-bit**, not Core AI palettize of GDN/QSA Conv2d (rel 0.31).
Greedy ``"The"`` on MLX 4-bit is ``220, 17, 15, 15`` (``The 2000…``). Token 4
``15`` vs BF16 ``16`` is the known 4-bit digit, not a bug.
"""
from __future__ import annotations

import ctypes
import fcntl
import json
import mmap
import os
import struct
import subprocess
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Darwin fcntl.h. F_NOCACHE keeps the kernel from retaining file pages in the
# unified buffer cache after a read; without it, copying a 71 GB safetensors
# into MLX arrays double-occupies RAM for the whole load.
_F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)


def suppress_file_cache(fh) -> None:
    """Stop the kernel caching this fd's pages. FLASHNEXT_FILE_CACHE=1 restores."""
    if os.environ.get("FLASHNEXT_FILE_CACHE", "").strip() in ("1", "true", "TRUE"):
        return
    try:
        fcntl.fcntl(fh.fileno(), _F_NOCACHE, 1)
    except Exception:
        pass


def drop_mmap_range(mm: mmap.mmap, offset: int, length: int) -> None:
    """Give the kernel back pages we have already copied out of ``mm``."""
    if length <= 0 or os.environ.get("FLASHNEXT_FILE_CACHE", "").strip() in (
            "1", "true", "TRUE"):
        return
    page = mmap.PAGESIZE
    start = offset & ~(page - 1)
    extra = offset - start
    try:
        mm.madvise(mmap.MADV_DONTNEED, start, int(length) + extra)
    except Exception:
        pass

H = 2560
I = 640
E = 512
GS = 64
BITS_EXPERT = 4
BITS_SHARED = 8
GU_SHAPE = (2 * I, H)
DN_SHAPE = (H, I)

MLX4_DEFAULT = Path.home() / "models" / "Qwen3.8-Flash-Next-MLX-4bit" / "model.safetensors"
ART_DEFAULT = Path(__file__).resolve().parents[1] / "artifacts" / "experts"

_U32 = np.uint32
_SHIFTS4 = np.arange(8, dtype=_U32) * _U32(4)
_SHIFTS8 = np.arange(4, dtype=_U32) * _U32(8)
_MASK4 = _U32(0xF)
_MASK8 = _U32(0xFF)

_DTYPE = {
    "BF16": np.uint16, "F16": np.float16, "F32": np.float32,
    "U32": np.uint32, "I32": np.int32, "U8": np.uint8, "I8": np.int8,
}
_ITEM = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1, "I8": 1}


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(u16, dtype=np.uint16)
    return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)


def unpack_codes(packed: np.ndarray, bits: int) -> np.ndarray:
    """uint32 packed (low bits first) → uint32 codes, last dim expanded."""
    p = np.ascontiguousarray(packed, dtype=_U32)
    if bits == 4:
        q = (p[..., None] >> _SHIFTS4) & _MASK4
    elif bits == 8:
        q = (p[..., None] >> _SHIFTS8) & _MASK8
    else:
        raise ValueError(f"bits={bits}")
    return q.reshape(*p.shape[:-1], p.shape[-1] * (32 // bits))


def dequant_affine(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                   dest: np.ndarray, bits: int = 4, gs: int = GS) -> np.ndarray:
    """Affine dequant into ``dest`` (out, in) fp32. scales/biases (out, n_g) fp32."""
    q = unpack_codes(packed, bits).astype(np.float32, copy=False)
    out, inn = q.shape
    n_g = inn // gs
    q = q.reshape(out, n_g, gs)
    sc = np.asarray(scales, np.float32).reshape(out, n_g, 1)
    bi = np.asarray(biases, np.float32).reshape(out, n_g, 1)
    np.multiply(q, sc, out=q)
    q += bi
    dest[...] = q.reshape(out, inn)
    return dest


class MlxSafe:
    """Read-only mmap of one safetensors file. Supports U32 packed weights."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        suppress_file_cache(self._fh)
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n))
        self.data_start = 8 + n
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return (k for k in self.header if k != "__metadata__")

    def meta(self, key: str) -> dict:
        return self.header[key]

    def byte_range(self, key: str, expert: int | None = None) -> tuple[int, int]:
        """File-absolute (offset, length) of one tensor's payload in the mmap."""
        m = self.header[key]
        off0, off1 = m["data_offsets"]
        if expert is not None:
            inner = int(np.prod(m["shape"][1:]))
            stride = inner * _ITEM[m["dtype"]]
            off0 = off0 + int(expert) * stride
            off1 = off0 + stride
        start = self.data_start + off0
        return start, off1 - off0

    def drop_pages(self, key: str, expert: int | None = None) -> None:
        start, n = self.byte_range(key, expert)
        drop_mmap_range(self._mm, start, n)

    def drop_all_pages(self) -> None:
        drop_mmap_range(self._mm, 0, self._mm.size())

    def raw(self, key: str, expert: int | None = None) -> np.ndarray:
        m = self.header[key]
        dt = m["dtype"]
        shape = list(m["shape"])
        off0, off1 = m["data_offsets"]
        item = _ITEM[dt]
        if expert is not None:
            inner = int(np.prod(shape[1:]))
            stride = inner * item
            off0 = off0 + int(expert) * stride
            off1 = off0 + stride
            shape = shape[1:]
        buf = self._mm[self.data_start + off0: self.data_start + off1]
        arr = np.frombuffer(buf, dtype=_DTYPE[dt])
        return arr.reshape(shape)

    def f32(self, key: str, expert: int | None = None) -> np.ndarray:
        a = self.raw(key, expert)
        dt = self.header[key]["dtype"]
        if dt == "BF16":
            return _bf16_to_f32(a)
        return np.asarray(a, np.float32)

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass


def _layer_prefix(i: int) -> str:
    return f"model.layers.{i}.mlp.switch_mlp"


class LayerQ:
    """One layer's routed experts: packed uint32 + fp32 scales/biases."""

    __slots__ = (
        "gate_w", "gate_s", "gate_b",
        "up_w", "up_s", "up_b",
        "down_w", "down_s", "down_b",
        "ram",
    )

    def nbytes(self) -> int:
        n = 0
        for k in self.__slots__:
            if k == "ram":
                continue
            a = getattr(self, k, None)
            if a is not None:
                n += a.nbytes
        return n


class Mlx4ExpertBank:
    """48-layer MLX 4-bit expert library. Mmap + optional RAM copy of packed W."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else MLX4_DEFAULT
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.shard = MlxSafe(self.path)
        self._layers: dict[int, LayerQ] = {}
        self.bits = BITS_EXPERT
        self.gs = GS
        self.shared_bits = BITS_SHARED
        self.hits = 0
        self.bank_hits = 0
        self.misses = 0
        self.lru_hits = 0
        self.last_dequant = 0
        self.last_reuse = 0
        self.last_lru = 0
        self._ram = False
        self._shared: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._sb_cache: dict[tuple[int, int], tuple] = {}

    def _q(self, layer: int) -> LayerQ:
        layer = int(layer)
        got = self._layers.get(layer)
        if got is not None:
            return got
        p = _layer_prefix(layer)
        L = LayerQ()
        L.ram = False
        L.gate_w = self.shard.raw(f"{p}.gate_proj.weight")
        L.up_w = self.shard.raw(f"{p}.up_proj.weight")
        L.down_w = self.shard.raw(f"{p}.down_proj.weight")
        # scales/biases stay BF16 mmap; convert per expert on GEMV (not 15 GB fp32).
        L.gate_s = self.shard.raw(f"{p}.gate_proj.scales")
        L.gate_b = self.shard.raw(f"{p}.gate_proj.biases")
        L.up_s = self.shard.raw(f"{p}.up_proj.scales")
        L.up_b = self.shard.raw(f"{p}.up_proj.biases")
        L.down_s = self.shard.raw(f"{p}.down_proj.scales")
        L.down_b = self.shard.raw(f"{p}.down_proj.biases")
        self._layers[layer] = L
        return L

    @staticmethod
    def _sb(s: np.ndarray, b: np.ndarray, e: int) -> tuple[np.ndarray, np.ndarray]:
        se, be = s[int(e)], b[int(e)]
        if se.dtype == np.float32:
            return np.ascontiguousarray(se), np.ascontiguousarray(be)
        return _bf16_to_f32(se), _bf16_to_f32(be)

    def _cached_sb(self, layer: int, e: int) -> tuple:
        key = (int(layer), int(e))
        trip = self._sb_cache.get(key)
        if trip is not None:
            return trip
        L = self._q(layer)
        trip = (
            self._sb(L.gate_s, L.gate_b, e),
            self._sb(L.up_s, L.up_b, e),
            self._sb(L.down_s, L.down_b, e),
        )
        self._sb_cache[key] = trip
        return trip

    def load_ram(self, layers: range | list[int] | None = None,
                 progress=True) -> int:
        """Copy packed uint32 expert W into process RAM (scales stay mmap BF16)."""
        ids = list(layers) if layers is not None else list(range(48))
        n = 0
        t0 = time.perf_counter()
        for i, li in enumerate(ids):
            L = self._q(li)
            if not L.ram:
                try:
                    L.gate_w = np.ascontiguousarray(L.gate_w)
                    L.up_w = np.ascontiguousarray(L.up_w)
                    L.down_w = np.ascontiguousarray(L.down_w)
                    L.ram = True
                except MemoryError:
                    print("    mlx4 packed copy MemoryError — leaving remaining layers mmap",
                          flush=True)
                    break
            n += L.gate_w.nbytes + L.up_w.nbytes + L.down_w.nbytes
            if progress and ((i + 1) % 8 == 0 or i == len(ids) - 1):
                print(
                    f"    mlx4 packed L{li:02d}  {n / 1e9:.1f} GB RAM  "
                    f"{time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
        self._ram = True
        return n

    def prepare(self, n_layers: int = 48) -> int:
        """Mmap all layer packed W, then copy into RAM (the 4-bit library)."""
        t0 = time.perf_counter()
        for i in range(n_layers):
            self._q(i)
        n = self.load_ram(range(n_layers))
        print(
            f"    mlx4 switch_mlp {n_layers} layers  packed RAM {n / 1e9:.1f} GB  "
            f"{time.perf_counter() - t0:.1f}s  source={self.path.name}",
            flush=True,
        )
        return n

    def disk_bytes(self) -> int:
        return int(self.path.stat().st_size)

    def ram_bytes(self) -> int:
        # raw() currently owns copied bytes, including BF16 scale/bias arrays.
        # Include those and converted metadata, not just packed weight codes.
        n = sum(L.nbytes() for L in self._layers.values())
        n += sum(a.nbytes for trip in self._sb_cache.values() for pair in trip for a in pair)
        n += sum(a.nbytes for trip in self._shared.values() for a in trip)
        return n

    def dequant_expert(self, layer: int, e_id: int,
                       gu: np.ndarray, dn: np.ndarray) -> None:
        """Affine gs=64 4-bit → fp32 packed SwiGLU (gate||up, down). C kernel."""
        L = self._q(layer)
        e = int(e_id)
        (gs, gb), (us, ub), (ds, db) = self._cached_sb(layer, e)
        gate = gu[:I]
        up = gu[I:]
        if not (gate.flags.c_contiguous and up.flags.c_contiguous and dn.flags.c_contiguous):
            raise ValueError("dequant dest must be C-contiguous fp32")
        _dequant_q4(L.gate_w[e], gs, gb, gate, I, H)
        _dequant_q4(L.up_w[e], us, ub, up, I, H)
        _dequant_q4(L.down_w[e], ds, db, dn, H, I)

    def gather_f32(self, layer: int, eids, gu_buf: np.ndarray,
                   dn_buf: np.ndarray) -> None:
        """Dequant every id (no slot reuse). Prefer ``fill_slots`` for decode."""
        ids = [int(e) for e in eids]
        for e in ids:
            self._cached_sb(layer, e)
        def _one(j: int) -> None:
            self.dequant_expert(layer, ids[j], gu_buf[j], dn_buf[j])
        if len(ids) <= 1:
            for j in range(len(ids)):
                _one(j)
        else:
            list(_dequant_pool().map(_one, range(len(ids))))
        self.bank_hits += len(ids)
        self.last_dequant = len(ids)
        self.last_reuse = 0

    def fill_slots(self, layer: int, eids, gu_buf: np.ndarray, dn_buf: np.ndarray,
                   slot_ids: list[int]) -> list[int]:
        """Dequant only experts not already in the packed top-k workspace.

        Jaccard ~0.3 vs the previous token → typically ~7 new / ~3 reused.
        Reused slots stay; permutations memcpy; misses C-dequant from the
        4-bit RAM library. Never 4-bit GEMV.
        """
        new_ids = [int(e) for e in eids]
        k = len(new_ids)
        if len(slot_ids) != k:
            slot_ids = [-1] * k
        old_pos = {e: j for j, e in enumerate(slot_ids) if e >= 0}
        overwrite = [j for j in range(k) if slot_ids[j] != new_ids[j]]
        src_needed = {old_pos[new_ids[j]] for j in overwrite if new_ids[j] in old_pos}
        snap: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for src in src_needed:
            if src in overwrite:
                snap[src] = (np.copy(gu_buf[src]), np.copy(dn_buf[src]))
        jobs: list[int] = []
        reuse = 0
        for j, e in enumerate(new_ids):
            if slot_ids[j] == e:
                reuse += 1
                continue
            if e in old_pos:
                src = old_pos[e]
                g, d = snap[src] if src in snap else (gu_buf[src], dn_buf[src])
                np.copyto(gu_buf[j], g)
                np.copyto(dn_buf[j], d)
                reuse += 1
            else:
                jobs.append(j)
        self._q(layer)
        for j in jobs:
            self._cached_sb(layer, new_ids[j])
        def _one(j: int) -> None:
            self.dequant_expert(layer, new_ids[j], gu_buf[j], dn_buf[j])
        if len(jobs) == 1:
            _one(jobs[0])
        elif jobs:
            list(_dequant_pool().map(_one, jobs))
        self.hits += reuse
        self.bank_hits += len(jobs)
        self.last_reuse = reuse
        self.last_dequant = len(jobs)
        return new_ids

    def shared_fp32(self, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dequant tiny 8-bit shared expert (gate, up, down). Cached per layer."""
        layer = int(layer)
        hit = self._shared.get(layer)
        if hit is not None:
            return hit
        p = f"model.layers.{layer}.mlp.shared_expert"
        g = np.empty((I, H), np.float32)
        u = np.empty((I, H), np.float32)
        d = np.empty((H, I), np.float32)
        dequant_affine(
            self.shard.raw(f"{p}.gate_proj.weight"),
            self.shard.f32(f"{p}.gate_proj.scales"),
            self.shard.f32(f"{p}.gate_proj.biases"),
            g, bits=BITS_SHARED,
        )
        dequant_affine(
            self.shard.raw(f"{p}.up_proj.weight"),
            self.shard.f32(f"{p}.up_proj.scales"),
            self.shard.f32(f"{p}.up_proj.biases"),
            u, bits=BITS_SHARED,
        )
        dequant_affine(
            self.shard.raw(f"{p}.down_proj.weight"),
            self.shard.f32(f"{p}.down_proj.scales"),
            self.shard.f32(f"{p}.down_proj.biases"),
            d, bits=BITS_SHARED,
        )
        hit = (
            np.ascontiguousarray(g),
            np.ascontiguousarray(u),
            np.ascontiguousarray(d),
        )
        self._shared[layer] = hit
        return hit

    def swiglu_routed(self, layer: int, eids, x: np.ndarray, scores: np.ndarray,
                      scratch: _SwiGLUScratch | None = None) -> np.ndarray:
        """Top-k scored SwiGLU from 4-bit GEMV. Decode default is fill_slots + host GEMV."""
        L = self._q(layer)
        x = np.ascontiguousarray(x, np.float32).reshape(-1)
        sc = np.ascontiguousarray(scores, np.float32).reshape(-1)
        buf = scratch if scratch is not None else _SwiGLUScratch()
        if os.environ.get("FLASHNEXT_Q4_PARALLEL", "1") != "0":
            ids = [int(e) for e in eids]
            if len(ids) > 10:
                raise ValueError("native workspace supports at most ten experts")
            for j, e in enumerate(ids):
                (gs, gb), (us, ub), (ds, db) = self._cached_sb(layer, e)
                arrays = (L.gate_w[e], gs, gb, L.up_w[e], us, ub, L.down_w[e], ds, db)
                buf.ptrs[j] = [a.ctypes.data for a in arrays]
            _q4lib().affine_q4_swiglu_topk(
                buf.ptrs.ctypes.data, x.ctypes.data, sc.ctypes.data,
                buf.outputs.ctypes.data, len(ids), H, I, GS,
            )
            self.bank_hits += len(ids)
            return buf.outputs[:len(ids)].sum(axis=0)
        routed = np.zeros(H, np.float32)
        for j, e in enumerate(eids):
            e = int(e)
            (gs, gb), (us, ub), (ds, db) = self._cached_sb(layer, e)
            _gemv_q4(L.gate_w[e], gs, gb, x, buf.yg, I, H)
            _gemv_q4(L.up_w[e], us, ub, x, buf.yu, I, H)
            np.multiply(silu_np(buf.yg), buf.yu, out=buf.act)
            buf.act *= sc[j]
            _gemv_q4(L.down_w[e], ds, db, buf.act, buf.yd, H, I)
            routed += buf.yd
            self.bank_hits += 1
        return routed

    def close(self) -> None:
        self.shard.close()
        self._layers.clear()
        self._shared.clear()


_Q4LIB = None
_DYLIB = Path(__file__).with_name("q4_gemv.dylib")
_CSRC = Path(__file__).with_name("q4_gemv.c")


def _q4lib():
    global _Q4LIB
    if _Q4LIB is not None:
        return _Q4LIB
    if _CSRC.is_file() and (not _DYLIB.is_file() or _CSRC.stat().st_mtime_ns > _DYLIB.stat().st_mtime_ns):
        with tempfile.TemporaryDirectory(dir=_DYLIB.parent) as build:
            dest = Path(build) / _DYLIB.name
            subprocess.check_call(
                ["cc", "-O3", "-ffast-math", "-fPIC", "-shared", "-o", str(dest), str(_CSRC)]
            )
            os.replace(dest, _DYLIB)
    lib = ctypes.CDLL(str(_DYLIB))
    lib.affine_q4_gemv.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.affine_q8_gemv.argtypes = lib.affine_q4_gemv.argtypes
    lib.affine_q4_dequant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.affine_q4_gemv.restype = None
    lib.affine_q4_swiglu_topk.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4
    lib.affine_q4_swiglu_topk.restype = None
    lib.affine_q8_gemv.restype = None
    lib.affine_q4_dequant.restype = None
    _Q4LIB = lib
    return lib


_DEQUANT_POOL: ThreadPoolExecutor | None = None


def _dequant_pool() -> ThreadPoolExecutor:
    global _DEQUANT_POOL
    if _DEQUANT_POOL is None:
        n = os.environ.get("FLASHNEXT_DEQUANT_THREADS", "8").strip() or "8"
        _DEQUANT_POOL = ThreadPoolExecutor(
            max_workers=max(1, int(n)), thread_name_prefix="q4deq",
        )
    return _DEQUANT_POOL


def _as_c(a: np.ndarray, dt) -> np.ndarray:
    if a.dtype == dt and a.flags.c_contiguous:
        return a
    return np.ascontiguousarray(a, dt)


def _dequant_q4(packed, scales, biases, dest, out_n: int, in_n: int) -> None:
    """Write affine-dequantized (out_n, in_n) fp32 into dest. dest must be the buffer."""
    lib = _q4lib()
    packed = _as_c(packed, np.uint32)
    scales = _as_c(scales, np.float32)
    biases = _as_c(biases, np.float32)
    if dest.dtype != np.float32 or not dest.flags.c_contiguous:
        raise ValueError("affine_q4_dequant dest must be contiguous fp32")
    if dest.size != int(out_n) * int(in_n):
        raise ValueError(f"dequant dest size {dest.size} != {out_n}*{in_n}")
    lib.affine_q4_dequant(
        packed.ctypes.data, scales.ctypes.data, biases.ctypes.data,
        dest.ctypes.data, int(out_n), int(in_n), GS,
    )


def _gemv_q4(packed, scales, biases, x, y, out_n: int, in_n: int) -> None:
    lib = _q4lib()
    packed = _as_c(packed, np.uint32)
    scales = _as_c(scales, np.float32)
    biases = _as_c(biases, np.float32)
    lib.affine_q4_gemv(
        packed.ctypes.data, scales.ctypes.data, biases.ctypes.data,
        x.ctypes.data, y.ctypes.data, int(out_n), int(in_n), GS,
    )


def silu_np(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0))))


class _SwiGLUScratch:
    __slots__ = ("yg", "yu", "act", "yd", "ptrs", "outputs")

    def __init__(self):
        self.yg = np.empty(I, np.float32)
        self.yu = np.empty(I, np.float32)
        self.act = np.empty(I, np.float32)
        self.yd = np.empty(H, np.float32)
        self.ptrs = np.empty((10,9), np.uintp)
        self.outputs = np.empty((10,H), np.float32)


def expert_mlx4_bank(path: Path | None = None) -> Mlx4ExpertBank:
    global _BANK
    if _BANK is None:
        _BANK = Mlx4ExpertBank(path=path)
    return _BANK


_BANK: Mlx4ExpertBank | None = None
