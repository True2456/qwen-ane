#!/usr/bin/env python3
"""Convert one Flash-Next (qwen4_exp) GDN layer to Core AI `.aimodel`.

Authoring is Conv2d / BC1S so the Neural Engine compiler sees convolutions,
not Linear-decomposed GEMMs. Weights come from the BF16 base via read-only
mmap; nothing is written under /Users/true/models/Qwen3.8-Flash-Next.

Stages (run in order):
  smoke   tiny random Conv2d SwiGLU, S=32
  body    layer-0 in_proj + out_proj + pinned-10 SwiGLU + shared
  mixers  body + attn hyper-connection mix (input C=10240)
  gdn_core  prepared-input recurrence, last dim 128
  decode    fused S=1 GDN + MoE (one graph; ANE rel ~0.09)
  split     full GDN layer: host attn-mix | ANE front | host prep |
            ANE GDN-only | host recombine | host MLP-mix | host top-10
            route+pack | ANE scored SwiGLU | host recombine.
  qsa       export FlashNextQSADecode for layer 3, max_S=32; vs numpy
            full_attention_layer; GPU then ANE. Wires mix→QSA→recombine
            →MLP mix→numpy MoE→recombine (first generate bakes MoE later).
  layers    export front+gdn-only for every GDN layer and QSA for every
            full_attention layer (same shapes as L0/L3; weights baked).
            `--reuse` skips assets that already exist. `--layers 0,1,3`
            limits the set. No ANE bench — generate specializes on load.
  generate  48-layer greedy decode. Host embed + mixer + lm_head.
            ANE: `pure_step` (1 submit: attn mix + connected GDN + recombine +
            MLP mix) when `flashnext_pure_step_L*.aimodel` exist;
            else connected GDN (1 submit) when all 36 prod graphs exist;
            else GDN front+gdn-only (2 submits) + QSA. `--host-front` is a
            1-submit split fallback (CPU S=1 front; FLASHNEXT_FRONT=mlx uses GPU).
            FLASHNEXT_CONNECTED=0 forces the 2-submit split even if connected
            assets exist.
            Host/GPU: 512-way router + fp16 hot expert store
            (artifacts/experts_f16 + in-process LRU; BF16 mmap on miss).
            FLASHNEXT_MOE=hybrid is 4-bit RAM bank → dequant top-10 →
            packed GEMV (opt-in; first-touch lost the 4-token race).
            FLASHNEXT_MOE=q4gemv is parallel native CPU INT4 SwiGLU.
            FLASHNEXT_COMPACT_GDN=1 uses compact single-token ANE tails.
            PLE zeros without ngram_index.json.
  host_test CPU-only HostPrep + hyper mix (no checkpoint, no ANE)
  fuse      GDN 2-submit → 1: fused front||gdn (host-prep as inputs, no SiLU),
            shared-spec pair entrypoints, host S=1 front + existing gdn-only.
            See scripts/export_flashnext_gdn_fuse.py.

  ~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py smoke
  ~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py body
  ~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py qsa
  ~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py generate --tokens 4
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import mmap
import os
import shutil
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import host_fastpath

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BASE = Path("/Users/true/models/Qwen3.8-Flash-Next")
OUT_DIR = ROOT / "artifacts" / "coreai"

H = 2560
I = 640
K_PIN = 10
HC = 4
HC_W = HC * H  # 10240
HV, HK, DV, DK = 48, 16, 128, 128
GDN_Y = HV * DV  # 6144
QKV = HK * DK + HK * DK + HV * DV  # 10240
IN_O = QKV + GDN_Y + HV + HV  # qkv + z + b + a
SEQ_DEFAULT = 32


def _fp16(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x, np.float16)))


def _conv_w(linear_oi: np.ndarray) -> torch.Tensor:
    """Linear [O, I] -> Conv2d [O, I, 1, 1]."""
    w = _fp16(linear_oi)
    return w.view(w.shape[0], w.shape[1], 1, 1)


class Conv1x1(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.op = nn.Conv2d(cin, cout, kernel_size=1, bias=False)

    def set_w(self, linear_oi: np.ndarray) -> None:
        self.op.weight.data.copy_(_conv_w(linear_oi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """BC1S RMSNorm over channels (dim=1). eps is a fp16 buffer, not a Python float."""
    ms = (x * x).mean(dim=1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * weight


class SmokeSwiGLU(nn.Module):
    """Minimal ANE-shaped MLP: (1, C, 1, S) in/out."""

    def __init__(self, cin: int = 256, hidden: int = 512):
        super().__init__()
        self.gate = Conv1x1(cin, hidden)
        self.up = Conv1x1(cin, hidden)
        self.down = Conv1x1(hidden, cin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(F.silu(self.gate(x)) * self.up(x))


class PinnedSwiGLU(nn.Module):
    """Dense stand-in for top-10: stacked gate/up + concatenated down / K."""

    def __init__(self):
        super().__init__()
        self.gate = Conv1x1(H, K_PIN * I)
        self.up = Conv1x1(H, K_PIN * I)
        self.down = Conv1x1(K_PIN * I, H)
        self.sh_gate = Conv1x1(H, I)
        self.sh_up = Conv1x1(H, I)
        self.sh_down = Conv1x1(I, H)
        self.sgate = Conv1x1(H, 1)

    def load_from_layer(self, w: dict) -> None:
        gu = w["mlp.experts.gate_up_proj"]
        dn = w["mlp.experts.down_proj"]
        gates, ups, downs = [], [], []
        for e in range(K_PIN):
            slab = np.asarray(gu[e], np.float32)
            gates.append(slab[:I])
            ups.append(slab[I:])
            downs.append(np.asarray(dn[e], np.float32))
        self.gate.set_w(np.concatenate(gates, axis=0))
        self.up.set_w(np.concatenate(ups, axis=0))
        # Concatenate expert downs on the reduction axis and average (pin proxy).
        self.down.set_w(np.concatenate(downs, axis=1) / float(K_PIN))
        self.sh_gate.set_w(w["mlp.shared_expert.gate_proj.weight"])
        self.sh_up.set_w(w["mlp.shared_expert.up_proj.weight"])
        self.sh_down.set_w(w["mlp.shared_expert.down_proj.weight"])
        sg = np.asarray(w["mlp.shared_expert_gate.weight"], np.float32).reshape(1, -1)
        self.sgate.set_w(sg)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        routed = self.down(F.silu(self.gate(h)) * self.up(h))
        shared = self.sh_down(F.silu(self.sh_gate(h)) * self.sh_up(h))
        return routed + torch.sigmoid(self.sgate(h)) * shared


class RoutedSwiGLU(nn.Module):
    """Top-10 SwiGLU. Routed expert weights are inputs; shared is baked per layer.

    scores is (1, K, 1, S), already softmax-topk renormalized. Each expert's I
    channels are scaled by its score before the concatenated down-proj, which is
    sum_k score_k * down_k(silu(gate_k(h))*up_k(h)).
    """

    def __init__(self):
        super().__init__()
        self.sh_gate = Conv1x1(H, I)
        self.sh_up = Conv1x1(H, I)
        self.sh_down = Conv1x1(I, H)
        self.sgate = Conv1x1(H, 1)

    def load_shared(self, w: dict) -> None:
        self.sh_gate.set_w(w["mlp.shared_expert.gate_proj.weight"])
        self.sh_up.set_w(w["mlp.shared_expert.up_proj.weight"])
        self.sh_down.set_w(w["mlp.shared_expert.down_proj.weight"])
        sg = np.asarray(w["mlp.shared_expert_gate.weight"], np.float32).reshape(1, -1)
        self.sgate.set_w(sg)

    def forward(self, h, gate_w, up_w, down_w, scores):
        act = F.silu(F.conv2d(h, gate_w)) * F.conv2d(h, up_w)
        B, _, _, S = h.shape
        act = act.reshape(B, K_PIN, I, S) * scores.reshape(B, K_PIN, 1, S)
        act = act.reshape(B, K_PIN * I, 1, S)
        routed = F.conv2d(act, down_w)
        shared = self.sh_down(F.silu(self.sh_gate(h)) * self.sh_up(h))
        return routed + torch.sigmoid(self.sgate(h)) * shared


class RoutedSwiGLUBaked(nn.Module):
    """Same math as RoutedSwiGLU, but routed weights are Parameters.

    Fallback if Core AI rejects conv-weight inputs. Host packs the token's
    top-10 into the Parameters before export (token-specific graph).
    """

    def __init__(self):
        super().__init__()
        self.gate = Conv1x1(H, K_PIN * I)
        self.up = Conv1x1(H, K_PIN * I)
        self.down = Conv1x1(K_PIN * I, H)
        self.sh_gate = Conv1x1(H, I)
        self.sh_up = Conv1x1(H, I)
        self.sh_down = Conv1x1(I, H)
        self.sgate = Conv1x1(H, 1)

    def load_shared(self, w: dict) -> None:
        self.sh_gate.set_w(w["mlp.shared_expert.gate_proj.weight"])
        self.sh_up.set_w(w["mlp.shared_expert.up_proj.weight"])
        self.sh_down.set_w(w["mlp.shared_expert.down_proj.weight"])
        sg = np.asarray(w["mlp.shared_expert_gate.weight"], np.float32).reshape(1, -1)
        self.sgate.set_w(sg)

    def load_routed(self, gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray) -> None:
        self.gate.op.weight.data.copy_(_fp16(gate_w.reshape(K_PIN * I, H, 1, 1)))
        self.up.op.weight.data.copy_(_fp16(up_w.reshape(K_PIN * I, H, 1, 1)))
        self.down.op.weight.data.copy_(_fp16(down_w.reshape(H, K_PIN * I, 1, 1)))

    def forward(self, h, scores):
        act = F.silu(self.gate(h)) * self.up(h)
        B, _, _, S = h.shape
        act = act.reshape(B, K_PIN, I, S) * scores.reshape(B, K_PIN, 1, S)
        act = act.reshape(B, K_PIN * I, 1, S)
        routed = self.down(act)
        shared = self.sh_down(F.silu(self.sh_gate(h)) * self.sh_up(h))
        return routed + torch.sigmoid(self.sgate(h)) * shared


def _moe_device() -> torch.device:
    """CPU batched SwiGLU by default. MPS launch/sync was 20–45 ms/layer.

    FLASHNEXT_MOE_MPS=1 enables GPU experts (still LRU-cached). Router stays
    numpy — a 1.3M-FLOP GEMM is slower on MPS once you pay .cpu().
    """
    if os.environ.get("FLASHNEXT_MOE_MPS", "").strip() in ("1", "true", "TRUE"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _moe_decode_mode() -> str:
    """Default ``fp16``: artifacts/experts_f16 + in-process LRU (~1.56 tok/s).

    ``FLASHNEXT_MOE=hybrid`` is the 4-bit RAM library → dequant top-10 →
    packed GEMV (opt-in; 0.447 tok/s on 4-token — first-touch dequant).
    ``FLASHNEXT_MOE=q4gemv`` is native vectorized, parallel CPU INT4 SwiGLU.
    ``FLASHNEXT_MOE=mlxresident`` is resident quantized GPU MoE, an explicit
    hybrid diagnostic; requires MLX in this Python environment.
    ``FLASHNEXT_MOE_BF16=1`` / ``FLASHNEXT_MOE=bf16`` are aliases for fp16.
    """
    if os.environ.get("FLASHNEXT_MOE_BF16", "").strip().lower() in ("1", "true"):
        return "fp16"
    raw = os.environ.get("FLASHNEXT_MOE", "fp16").strip().lower() or "fp16"
    if raw in ("bf16", "f16", "fp16"):
        return "fp16"
    if raw in ("q4gemv", "q4", "gemv4"):
        return "q4gemv"
    if raw == "mlxresident":
        return "mlxresident"
    if raw in ("hybrid", "mlx4", "q4dequant"):
        return "hybrid"
    return "fp16"


def _willneed_expert(slab, e_id: int) -> None:
    """Prefault one expert slab in the safetensors mmap (MADV_WILLNEED)."""
    try:
        meta = slab.shard.meta(slab.key)
        off0, _ = meta["data_offsets"]
        inner = int(np.prod(meta["shape"][1:]))
        stride = inner * 2
        start = slab.shard.data_start + off0 + int(e_id) * stride
        slab.shard._mm.madvise(mmap.MADV_WILLNEED, start, stride)
    except Exception:
        pass


_FN16_MAGIC = b"FN16"


def _as_shaped_f16(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    a = np.ascontiguousarray(arr, dtype=np.float16)
    if a.shape != shape:
        a = np.ascontiguousarray(a.T)
    if a.shape != shape:
        a = np.ascontiguousarray(a.reshape(shape))
    return a


def _bf16_bits_to_f16(u16: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Owned BF16 bits → contiguous fp16. One convert; do not keep fp32."""
    src = np.array(u16, dtype=np.uint16, copy=True, order="C").reshape(-1)
    out = torch.from_numpy(src).view(torch.bfloat16).to(torch.float16).numpy()
    return _as_shaped_f16(out, shape)


def _bf16_into_f32(raw, dest: np.ndarray, want: tuple[int, ...]) -> None:
    """BF16 mmap bits → dest fp32. Widen in place into the GEMV workspace."""
    src = np.asarray(raw, np.uint16)
    if src.size != dest.size:
        raise ValueError(f"bf16 size {src.size} != dest {dest.size}")
    if src.shape == want or src.shape == dest.shape:
        bits = dest.reshape(-1).view(np.uint32)
        np.copyto(bits, np.reshape(src, -1))
        bits <<= np.uint32(16)
        return
    tmp = np.empty(src.shape, np.float32)
    bits = tmp.reshape(-1).view(np.uint32)
    np.copyto(bits, np.reshape(src, -1))
    bits <<= np.uint32(16)
    np.copyto(dest, np.ascontiguousarray(tmp.T))


def _raw_into_f32(raw, dt: str, dest: np.ndarray, want: tuple[int, ...]) -> None:
    if dt == "BF16":
        _bf16_into_f32(raw, dest, want)
        return
    if dt == "F16":
        np.copyto(dest, _as_shaped_f16(raw, want))
        return
    a = np.asarray(raw, np.float32)
    if a.shape != want:
        a = np.ascontiguousarray(a.T)
    np.copyto(dest, a)


class ExpertHotStore:
    """Process-RAM fp16 expert cache keyed by (layer, expert_id), plus disk.

    Holds packed SwiGLU ``gate_up`` (gate||up) and ``down`` as fp16. Shared
    experts stay as static fp32 on HostMoE. Miss = one BF16→fp16 convert then
    cache. Hit = return the resident fp16 (HostMoE memcpy + fp32 GEMV).

    Disk: ``ane-port/artifacts/experts_f16/Lxx/eeee.bin`` with magic ``FN16``
    + raw fp16 bits (legacy files are BF16 bits, migrated on flush). Never
    materializes 512×48. Never writes the BF16 tree.
    """

    E = 512
    GU_SHAPE = (2 * I, H)
    DN_SHAPE = (H, I)
    GU_ELEMS = 2 * I * H
    DN_ELEMS = H * I
    SLAB_ELEMS = GU_ELEMS + DN_ELEMS
    SLAB_BYTES = SLAB_ELEMS * 2
    ART = ROOT / "artifacts" / "experts_f16"

    def __init__(self, max_ram: int | None = None):
        env_n = (os.environ.get("FLASHNEXT_EXPERT_RAM", "").strip()
                 or os.environ.get("FLASHNEXT_EXPERT_F32", "").strip())
        # fp16 is half the old fp32 LRU: 4096 rows ≈ 40 GB, same cap as 2048 fp32.
        self.max_ram = int(env_n) if env_n else (max_ram if max_ram is not None else 4096)
        self.max_f32 = self.max_ram
        self._f16: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._disk_ok: set[tuple[int, int]] = set()
        self._fn16: set[tuple[int, int]] = set()
        self._ok: dict[int, np.ndarray] = {}
        self.hits = 0
        self.disk_hits = 0
        self.misses = 0
        self.ART.mkdir(parents=True, exist_ok=True)
        self.mlx4 = None
        self.resident_models = {}
        self.bits = 16
        mlx4_path = os.environ.get("FLASHNEXT_MLX4", "").strip()
        from runtime.expert_bank import MLX4_DEFAULT, Mlx4ExpertBank
        p = Path(mlx4_path) if mlx4_path else MLX4_DEFAULT
        if p.is_file() and _moe_decode_mode() in ("hybrid", "q4gemv"):
            self.mlx4 = Mlx4ExpertBank(p)
            self.bits = 4
        if _moe_decode_mode() == "mlxresident":
            self.bits = 4
        elif self.mlx4 is None:
            self._scan_disk()

    def counts(self) -> tuple[int, int, int]:
        if self.mlx4 is not None:
            return self.mlx4.hits, self.mlx4.bank_hits, self.mlx4.misses
        return self.hits, self.disk_hits, self.misses

    def ram_bytes(self) -> int:
        if self.resident_models:
            return sum(m.nbytes for m in self.resident_models.values())
        if self.mlx4 is not None:
            return self.mlx4.ram_bytes()
        return len(self._f16) * self.SLAB_BYTES

    def _scan_disk(self) -> None:
        if not self.ART.is_dir():
            return
        for d in self.ART.glob("L[0-9][0-9]"):
            try:
                layer = int(d.name[1:])
            except ValueError:
                continue
            for p in d.glob("*.bin"):
                try:
                    eid = int(p.stem)
                    sz = p.stat().st_size
                except ValueError:
                    continue
                if sz == 4 + self.SLAB_BYTES:
                    key = (layer, eid)
                    self._disk_ok.add(key)
                    self._fn16.add(key)
                elif sz == self.SLAB_BYTES:
                    self._disk_ok.add((layer, eid))

    def _layer_ok(self, layer: int) -> np.ndarray:
        bit = self._ok.get(layer)
        if bit is None:
            bit = np.zeros(self.E, np.uint8)
            self._ok[layer] = bit
        return bit

    def resident(self, layer: int, e_id: int) -> bool:
        key = (int(layer), int(e_id))
        return key in self._f16 or key in self._disk_ok

    def _path(self, layer: int, e_id: int) -> Path:
        return self.ART / f"L{layer:02d}" / f"{e_id:04d}.bin"

    def _evict(self) -> None:
        while len(self._f16) >= self.max_ram:
            self._f16.popitem(last=False)

    def _put(self, key: tuple[int, int], g: np.ndarray, d: np.ndarray) -> None:
        self._evict()
        self._f16[key] = (g, d)
        self._layer_ok(key[0])[key[1]] = 1

    def _load_disk(self, layer: int, e_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        path = self._path(layer, e_id)
        try:
            with open(path, "rb") as fh:
                blob = fh.read()
        except OSError:
            return None
        if blob.startswith(_FN16_MAGIC):
            b = np.frombuffer(blob, dtype=np.float16, offset=4)
            if b.size != self.SLAB_ELEMS:
                return None
            g = np.ascontiguousarray(b[: self.GU_ELEMS].reshape(self.GU_SHAPE))
            d = np.ascontiguousarray(b[self.GU_ELEMS:].reshape(self.DN_SHAPE))
            return g, d
        u = np.frombuffer(blob, dtype=np.uint16)
        if u.size != self.SLAB_ELEMS:
            return None
        g = _bf16_bits_to_f16(u[: self.GU_ELEMS], self.GU_SHAPE)
        d = _bf16_bits_to_f16(u[self.GU_ELEMS:], self.DN_SHAPE)
        return g, d

    def _from_mmap(self, e_id: int, slab_gu, slab_dn) -> tuple[np.ndarray, np.ndarray]:
        g = np.empty(self.GU_SHAPE, np.float32)
        d = np.empty(self.DN_SHAPE, np.float32)
        self._mmap_into(e_id, slab_gu, slab_dn, g, d)
        return np.ascontiguousarray(g, np.float16), np.ascontiguousarray(d, np.float16)

    def _mmap_into(self, e_id: int, slab_gu, slab_dn, g_f32: np.ndarray, d_f32: np.ndarray) -> None:
        raw, dt = slab_gu.shard.raw(slab_gu.key, expert=e_id)
        _raw_into_f32(raw, dt, g_f32, self.GU_SHAPE)
        raw, dt = slab_dn.shard.raw(slab_dn.key, expert=e_id)
        _raw_into_f32(raw, dt, d_f32, self.DN_SHAPE)

    def ensure(self, layer: int, e_id: int, slab_gu, slab_dn) -> tuple[np.ndarray, np.ndarray]:
        return self.ensure_many(layer, (e_id,), slab_gu, slab_dn)[0]

    def ensure_many(self, layer: int, eids, slab_gu, slab_dn) -> list[tuple[np.ndarray, np.ndarray]]:
        """Prefault mmap misses, then convert each once into the fp16 LRU."""
        layer = int(layer)
        ids = [int(e) for e in eids]
        out: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(ids)
        miss_i: list[int] = []
        for i, e_id in enumerate(ids):
            key = (layer, e_id)
            hit = self._f16.get(key)
            if hit is not None:
                self._f16.move_to_end(key)
                self.hits += 1
                out[i] = hit
                continue
            if key in self._disk_ok:
                disk = self._load_disk(layer, e_id)
                if disk is not None:
                    self.disk_hits += 1
                    self._put(key, disk[0], disk[1])
                    out[i] = disk
                    continue
                self._disk_ok.discard(key)
                self._fn16.discard(key)
            miss_i.append(i)
        for i in miss_i:
            _willneed_expert(slab_gu, ids[i])
            _willneed_expert(slab_dn, ids[i])
        for i in miss_i:
            e_id = ids[i]
            g, d = self._from_mmap(e_id, slab_gu, slab_dn)
            self.misses += 1
            self._put((layer, e_id), g, d)
            out[i] = (g, d)
        return out  # type: ignore[return-value]

    def gather_f32(self, layer: int, eids, slab_gu, slab_dn,
                   gu_buf: np.ndarray, dn_buf: np.ndarray) -> None:
        """Write top-k fp32 packed slabs. Miss converts into the workspace once."""
        ids = [int(e) for e in eids]
        self.gather_slots(
            layer, ids, list(range(len(ids))), slab_gu, slab_dn, gu_buf, dn_buf,
        )

    def gather_slots(self, layer: int, eids, dest, slab_gu, slab_dn,
                     gu_buf: np.ndarray, dn_buf: np.ndarray) -> None:
        """Write listed experts into ``dest`` slots of the packed fp32 workspace."""
        if self.mlx4 is not None:
            tmp_gu = np.empty((len(eids),) + self.GU_SHAPE, np.float32)
            tmp_dn = np.empty((len(eids),) + self.DN_SHAPE, np.float32)
            self.mlx4.gather_f32(layer, eids, tmp_gu, tmp_dn)
            for src, j in enumerate(dest):
                np.copyto(gu_buf[int(j)], tmp_gu[src])
                np.copyto(dn_buf[int(j)], tmp_dn[src])
            return
        layer = int(layer)
        ids = [int(e) for e in eids]
        slots = [int(j) for j in dest]
        miss: list[int] = []
        for src, (e_id, j) in enumerate(zip(ids, slots)):
            key = (layer, e_id)
            hit = self._f16.get(key)
            if hit is not None:
                self._f16.move_to_end(key)
                np.copyto(gu_buf[j], hit[0])
                np.copyto(dn_buf[j], hit[1])
                self.hits += 1
                continue
            if key in self._disk_ok:
                disk = self._load_disk(layer, e_id)
                if disk is not None:
                    self.disk_hits += 1
                    self._put(key, disk[0], disk[1])
                    np.copyto(gu_buf[j], disk[0])
                    np.copyto(dn_buf[j], disk[1])
                    continue
                self._disk_ok.discard(key)
                self._fn16.discard(key)
            miss.append(src)
        for src in miss:
            _willneed_expert(slab_gu, ids[src])
            _willneed_expert(slab_dn, ids[src])
        for src in miss:
            e_id = ids[src]
            j = slots[src]
            self._mmap_into(e_id, slab_gu, slab_dn, gu_buf[j], dn_buf[j])
            self._put(
                (layer, e_id),
                np.ascontiguousarray(gu_buf[j], np.float16),
                np.ascontiguousarray(dn_buf[j], np.float16),
            )
            self.misses += 1

    def flush_disk(self) -> int:
        """Write RAM fp16 rows to compact FN16 files. No-op for MLX 4-bit bank."""
        if self.mlx4 is not None:
            return 0
        n = 0
        payload_elems = self.SLAB_ELEMS
        for (layer, e_id), (g, d) in list(self._f16.items()):
            key = (layer, e_id)
            path = self._path(layer, e_id)
            if key in self._fn16 and path.is_file():
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                body = np.empty(payload_elems, np.float16)
                body[: self.GU_ELEMS] = np.ascontiguousarray(g, np.float16).reshape(-1)
                body[self.GU_ELEMS:] = np.ascontiguousarray(d, np.float16).reshape(-1)
                with open(tmp, "wb") as fh:
                    fh.write(_FN16_MAGIC)
                    body.tofile(fh)
                os.replace(tmp, path)
                self._disk_ok.add(key)
                self._fn16.add(key)
                n += 1
            except Exception:
                pass
        return n

    def warm_ram_from_disk(self, limit: int | None = None) -> int:
        """Load compact files into the fp16 LRU so decode misses are RAM memcpy."""
        if self.mlx4 is not None:
            return self.mlx4.prepare(48)
        n = 0
        keys = list(self._disk_ok)
        if limit is not None:
            keys = keys[: int(limit)]
        for layer, e_id in keys:
            if len(self._f16) >= self.max_ram:
                break
            key = (layer, e_id)
            if key in self._f16:
                continue
            pair = self._load_disk(layer, e_id)
            if pair is None:
                continue
            self._put(key, pair[0], pair[1])
            n += 1
        return n


_F16_STORE: ExpertHotStore | None = None


def expert_f16_store() -> ExpertHotStore:
    global _F16_STORE
    if _F16_STORE is None:
        _F16_STORE = ExpertHotStore()
    return _F16_STORE


class HostMoE:
    """512-way router + top-10 SwiGLU + shared expert on host/GPU.

    Default (fp16): compact ``artifacts/experts_f16`` + in-process LRU,
    sticky packed fp32 slots (skip copyto for resident eids) + host GEMV
    (~1.6 ms/layer). ``FLASHNEXT_MOE=hybrid`` uses the
    mlx-lm 4-bit RAM bank and dequants new top-10 into the packed
    workspace (opt-in). ``FLASHNEXT_MOE=q4gemv`` is fused 4-bit C GEMV
    (regression). Shared expert is 8-bit in the MLX-4bit checkpoint when
    that bank is loaded. Conv2d pack buffers remain for the ANE split probe.
    """

    def __init__(self, w, seq: int = SEQ_DEFAULT, device: torch.device | None = None,
                 store: ExpertHotStore | None = None):
        self.layer = int(getattr(w, "index", 0))
        self.store = store if store is not None else expert_f16_store()
        self.router_w = np.ascontiguousarray(np.asarray(w["mlp.gate.weight"], np.float32))
        self.gu = w["mlp.experts.gate_up_proj"]
        self.dn = w["mlp.experts.down_proj"]
        self.seq = seq
        sg = np.asarray(w["mlp.shared_expert.gate_proj.weight"], np.float32)
        su = np.asarray(w["mlp.shared_expert.up_proj.weight"], np.float32)
        sd = np.asarray(w["mlp.shared_expert.down_proj.weight"], np.float32)
        sgt = np.asarray(w["mlp.shared_expert_gate.weight"], np.float32).reshape(1, -1)
        self.shared_gate = np.ascontiguousarray(sg)
        self.shared_up = np.ascontiguousarray(su)
        self.shared_down = np.ascontiguousarray(sd)
        self.shared_sgate = np.ascontiguousarray(sgt)
        if getattr(self.store, "mlx4", None) is not None:
            sg, su, sd = self.store.mlx4.shared_fp32(self.layer)
            self.shared_gate = sg
            self.shared_up = su
            self.shared_down = sd
        self._resident = None
        if _moe_decode_mode() == "mlxresident":
            from runtime.flashnext_mlx_moe import ResidentMoe
            from runtime.expert_bank import Mlx4ExpertBank, MLX4_DEFAULT
            path = Path(os.environ.get("FLASHNEXT_MLX4") or MLX4_DEFAULT)
            bank = Mlx4ExpertBank(path)
            sg, su, sd = bank.shared_fp32(self.layer)
            self._resident = ResidentMoe(self.layer, (sg, su, sd, self.shared_sgate), path)
            bank.shard.close()
            self.store.resident_models[self.layer] = self._resident
            self.gu_buf = self.dn_buf = np.empty(0, np.float32)
            self.inds = np.zeros((1, K_PIN), np.int32)
            self.scores = np.zeros((1, K_PIN, 1, seq), np.float16)
            self.last_ms = {"route": 0.0, "gather": 0.0, "gemm": 0.0}
            return
        from runtime.expert_bank import _SwiGLUScratch
        self._q4_scratch = _SwiGLUScratch()
        self._moe_mode = _moe_decode_mode()
        self._slot_ids = [-1] * K_PIN
        self._eid_slot: dict[int, int] = {}
        self.last_reuse = 0
        self.last_copy = K_PIN
        self.gate_w = np.empty((K_PIN * I, H, 1, 1), np.float16)
        self.up_w = np.empty((K_PIN * I, H, 1, 1), np.float16)
        self.down_w = np.empty((H, K_PIN * I, 1, 1), np.float16)
        if getattr(self.store, "mlx4", None) is not None:
            # Persist top-10 as fp16 (4.7 GB / 48 layers). GEMV uses a shared
            # 196 MB fp32 scratch so the 60 GB 4-bit bank is not crowded out.
            self.gu_f16 = np.empty((K_PIN, 2 * I, H), np.float16)
            self.dn_f16 = np.empty((K_PIN, H, I), np.float16)
            self.gu_buf, self.dn_buf = self._fp32_scratch()
        else:
            self.gu_f16 = None
            self.dn_f16 = None
            self.gu_buf = np.empty((K_PIN, 2 * I, H), np.float32)
            self.dn_buf = np.empty((K_PIN, H, I), np.float32)
        self._gu_y = np.empty((K_PIN, 2 * I), np.float32)
        self._act = np.empty((K_PIN, I), np.float32)
        self._dn_y = np.empty(H, np.float32)
        self._routed = np.empty(H, np.float32)
        self._sc = np.empty((K_PIN, 1), np.float32)
        self.scores = np.zeros((1, K_PIN, 1, seq), np.float16)
        self.inds = np.zeros((1, K_PIN), np.int32)
        self.device = device if device is not None else _moe_device()
        self._gu_mps: dict[int, torch.Tensor] = {}
        self._dn_mps: dict[int, torch.Tensor] = {}
        self._use_mps = self.device.type == "mps"
        if self._use_mps:
            self.sh_gate_t = torch.from_numpy(self.shared_gate).to(self.device)
            self.sh_up_t = torch.from_numpy(self.shared_up).to(self.device)
            self.sh_down_t = torch.from_numpy(self.shared_down).to(self.device)
            self.sh_sgate_t = torch.from_numpy(self.shared_sgate).to(self.device)
        self.last_ms = {"route": 0.0, "gather": 0.0, "gemm": 0.0}

    _GU_F32: np.ndarray | None = None
    _DN_F32: np.ndarray | None = None

    @classmethod
    def _fp32_scratch(cls) -> tuple[np.ndarray, np.ndarray]:
        if cls._GU_F32 is None:
            cls._GU_F32 = np.empty((K_PIN, 2 * I, H), np.float32)
            cls._DN_F32 = np.empty((K_PIN, H, I), np.float32)
        return cls._GU_F32, cls._DN_F32

    def _gu_np(self, e_id: int) -> np.ndarray:
        g, _ = self.store.ensure(self.layer, int(e_id), self.gu, self.dn)
        return g

    def _dn_np(self, e_id: int) -> np.ndarray:
        _, d = self.store.ensure(self.layer, int(e_id), self.gu, self.dn)
        return d

    def _f32_dev(self, cache: dict[int, torch.Tensor], arr: np.ndarray, e_id: int) -> torch.Tensor:
        t = cache.get(int(e_id))
        if t is None:
            t = torch.from_numpy(np.ascontiguousarray(arr, np.float32)).to(self.device)
            cache[int(e_id)] = t
        return t

    def _gu_dev(self, e_id: int) -> torch.Tensor:
        return self._f32_dev(self._gu_mps, self._gu_np(e_id), e_id)

    def _dn_dev(self, e_id: int) -> torch.Tensor:
        return self._f32_dev(self._dn_mps, self._dn_np(e_id), e_id)

    def _route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = x @ self.router_w.T
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp((logits - m).astype(np.float64))
        probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
        inds = np.argpartition(probs, -K_PIN, axis=-1)[:, -K_PIN:]
        scores = np.take_along_axis(probs, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        self.inds[0] = inds[0]
        self.scores[...] = 0
        self.scores[0, :, 0, 0] = scores[0]
        return inds, scores

    def apply(self, h_bsh: np.ndarray) -> np.ndarray:
        """Top-10 SwiGLU + shared. Default: fp16 LRU gather + packed GEMV."""
        x = np.ascontiguousarray(np.asarray(h_bsh, np.float32).reshape(-1, H))
        t0 = time.perf_counter()
        inds, scores = self._route(x)
        t1 = time.perf_counter()
        if self._resident is not None:
            y, ms = self._resident.apply(x, inds[0], scores[0])
            self.last_ms = {"route": (t1-t0)*1e3, "gather": 0.0, "gemm": ms}
            return y.reshape(h_bsh.shape)
        if self._use_mps:
            y = self._apply_mps(x, inds, scores)
            t2 = time.perf_counter()
            self.last_ms = {"route": (t1 - t0) * 1e3, "gather": 0.0, "gemm": (t2 - t1) * 1e3}
            return y.reshape(h_bsh.shape[0], h_bsh.shape[1], H)
        x0 = x[0]
        t_g0 = time.perf_counter()
        if self.store.mlx4 is not None and self._moe_mode == "q4gemv":
            routed = self.store.mlx4.swiglu_routed(
                self.layer, inds[0], x0, scores[0], scratch=self._q4_scratch
            )
            t_g1 = time.perf_counter()
            sg = HostPrep._silu(x @ self.shared_gate.T)
            su = x @ self.shared_up.T
            shared = (sg * su) @ self.shared_down.T
            sgate = 1.0 / (1.0 + np.exp(-np.clip(x @ self.shared_sgate.T, -80, 80)))
            y = routed + (sgate * shared)[0]
            t3 = time.perf_counter()
            self.last_ms = {
                "route": (t1 - t0) * 1e3,
                "gather": 0.0,
                "gemm": (t3 - t_g0) * 1e3,
            }
            return y.reshape(h_bsh.shape[0], h_bsh.shape[1], H)
        self._pack_top10(inds[0])
        t_g1 = time.perf_counter()
        np.matmul(self.gu_buf, x0, out=self._gu_y)
        if self.store.mlx4 is None:
            sc_map = {int(e): float(s) for e, s in zip(inds[0], scores[0])}
            for j, e in enumerate(self._slot_ids):
                self._sc[j, 0] = sc_map[e]
        else:
            self._sc[:, 0] = scores[0]
        np.multiply(HostPrep._silu(self._gu_y[:, :I]), self._gu_y[:, I:], out=self._act)
        np.multiply(self._act, self._sc, out=self._act)
        self._routed.fill(0)
        for j in range(K_PIN):
            np.dot(self.dn_buf[j], self._act[j], out=self._dn_y)
            self._routed += self._dn_y
        t2 = time.perf_counter()
        sg = HostPrep._silu(x @ self.shared_gate.T)
        su = x @ self.shared_up.T
        shared = (sg * su) @ self.shared_down.T
        sgate = 1.0 / (1.0 + np.exp(-np.clip(x @ self.shared_sgate.T, -80, 80)))
        y = self._routed + (sgate * shared)[0]
        t3 = time.perf_counter()
        self.last_ms = {
            "route": (t1 - t0) * 1e3,
            "gather": (t_g1 - t_g0) * 1e3,
            "gemm": ((t2 - t_g1) + (t3 - t2)) * 1e3,
        }
        return y.reshape(h_bsh.shape[0], h_bsh.shape[1], H)

    def _pack_fp16_slots(self, eids) -> None:
        """Sticky packed fp32 slots: skip fp16→fp32 copyto for resident eids."""
        new_ids = [int(e) for e in eids]
        old_pos = self._eid_slot
        occupied: set[int] = set()
        slot_of: dict[int, int] = {}
        for e in new_ids:
            j = old_pos.get(e)
            if j is not None and j not in occupied:
                slot_of[e] = j
                occupied.add(j)
                key = (self.layer, e)
                if key in self.store._f16:
                    self.store._f16.move_to_end(key)
                    self.store.hits += 1
        free = [j for j in range(K_PIN) if j not in occupied]
        need = [e for e in new_ids if e not in slot_of]
        if need:
            self.store.gather_slots(
                self.layer, need, free, self.gu, self.dn, self.gu_buf, self.dn_buf,
            )
            for e, j in zip(need, free):
                slot_of[e] = j
        slots = [-1] * K_PIN
        for e, j in slot_of.items():
            slots[j] = e
        self._slot_ids = slots
        self._eid_slot = slot_of
        self.last_reuse = len(new_ids) - len(need)
        self.last_copy = len(need)

    def _pack_top10(self, eids) -> None:
        """Fill packed fp32 SwiGLU scratch. mlx4: dequant only new experts."""
        if self.store.mlx4 is None:
            self._pack_fp16_slots(eids)
            return
        from runtime.expert_bank import _dequant_pool

        gu32, dn32 = self._fp32_scratch()
        self.gu_buf, self.dn_buf = gu32, dn32
        bank = self.store.mlx4
        new_ids = [int(e) for e in eids]
        old = self._slot_ids
        old_pos = {e: j for j, e in enumerate(old) if e >= 0}
        jobs: list[int] = []
        reuse = 0
        for j, e in enumerate(new_ids):
            if e in old_pos:
                src = old_pos[e]
                np.copyto(gu32[j], self.gu_f16[src])
                np.copyto(dn32[j], self.dn_f16[src])
                reuse += 1
            else:
                jobs.append(j)
        bank._q(self.layer)
        for j in jobs:
            bank._cached_sb(self.layer, new_ids[j])

        def _one(j: int) -> None:
            bank.dequant_expert(self.layer, new_ids[j], gu32[j], dn32[j])

        if len(jobs) == 1:
            _one(jobs[0])
        elif jobs:
            list(_dequant_pool().map(_one, jobs))
        for j, e in enumerate(new_ids):
            if old[j] != e:
                np.copyto(self.gu_f16[j], gu32[j])
                np.copyto(self.dn_f16[j], dn32[j])
        self._slot_ids = new_ids
        bank.hits += reuse
        bank.bank_hits += len(jobs)
        bank.last_reuse = reuse
        bank.last_dequant = len(jobs)
        self.last_reuse = reuse
        self.last_copy = len(jobs)

    def _apply_mps(self, x: np.ndarray, inds: np.ndarray, scores: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(x[0]).to(self.device)
        gu_t = torch.stack([self._gu_dev(int(e)) for e in inds[0]], dim=0)
        dn_t = torch.stack([self._dn_dev(int(e)) for e in inds[0]], dim=0)
        gu = torch.matmul(gu_t, xt)
        gate, up = gu[:, :I], gu[:, I:]
        sc = torch.from_numpy(np.ascontiguousarray(scores[0])).to(self.device)
        act = torch.nn.functional.silu(gate) * up * sc[:, None]
        routed = torch.matmul(dn_t, act[:, :, None]).squeeze(-1).sum(0)
        sg = torch.nn.functional.silu(torch.nn.functional.linear(xt, self.sh_gate_t))
        su = torch.nn.functional.linear(xt, self.sh_up_t)
        shared = torch.nn.functional.linear(sg * su, self.sh_down_t)
        sgate = torch.sigmoid(torch.nn.functional.linear(xt, self.sh_sgate_t))
        y = routed + (sgate * shared).reshape(-1)
        return y.detach().cpu().numpy()

    def route_and_pack(self, h_bsh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(h_bsh, np.float32).reshape(-1, H)
        inds, scores = self._route(x)
        self.pack_ids(inds[0])
        return inds, scores

    def pack_ids(self, inds: np.ndarray) -> None:
        """Copy the given expert ids into the packed Conv2d buffers."""
        self.inds[0] = np.asarray(inds, np.int32).reshape(K_PIN)
        slabs = self.store.ensure_many(self.layer, self.inds[0], self.gu, self.dn)
        for j, (slab, d) in enumerate(slabs):
            sl = slice(j * I, (j + 1) * I)
            self.gate_w[sl, :, 0, 0] = slab[:I]
            self.up_w[sl, :, 0, 0] = slab[I:]
            self.down_w[:, sl, 0, 0] = d

    def mass_on(self, h_bsh: np.ndarray, baked: np.ndarray) -> tuple[float, np.ndarray]:
        """Softmax mass on a frozen expert set; scores in baked order, slot 0."""
        x = np.asarray(h_bsh, np.float32).reshape(-1, H)
        logits = x @ self.router_w.T
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp((logits - m).astype(np.float64))
        probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)[0]
        baked = np.asarray(baked, np.int32).reshape(-1)
        mass = float(probs[baked].sum())
        sc = probs[baked].astype(np.float32)
        s = float(sc.sum()) + np.float32(1e-12)
        self.scores[...] = 0
        self.scores[0, : baked.size, 0, 0] = sc / s
        return mass, self.scores


MOE_MASS_THRESH = 0.85


def _packed_swiglu(w, host: HostMoE, h_bsh: np.ndarray) -> np.ndarray:
    """10-expert SwiGLU + shared from already-packed fp16 slabs. Host/GPU-cheap."""
    x = np.asarray(h_bsh, np.float32).reshape(-1, H)
    g = x @ host.gate_w[:, :, 0, 0].T.astype(np.float32)
    u = x @ host.up_w[:, :, 0, 0].T.astype(np.float32)
    act = HostPrep._silu(g) * u
    sc = host.scores[0, :, 0, 0].astype(np.float32)[:, None]
    act = (act.reshape(K_PIN, I) * sc).reshape(1, K_PIN * I)
    routed = act @ host.down_w[:, :, 0, 0].T.astype(np.float32)
    sg = HostPrep._silu(x @ host.shared_gate.T)
    su = x @ host.shared_up.T
    shared = (sg * su) @ host.shared_down.T
    gate = 1.0 / (1.0 + np.exp(-np.clip(x @ host.shared_sgate.T, -80, 80)))
    y = routed + gate * shared
    return y.reshape(h_bsh.shape[0], h_bsh.shape[1], H)


class FlashNextBody(nn.Module):
    """Layer body without GDN recurrence: in_proj -> proxy y -> out_proj + pinned MoE.

    Proxy GDN takes the first 6144 channels of in_proj (value-shaped) as y.
    Real recurrence is the `gdn` stage.
    """

    def __init__(self):
        super().__init__()
        self.in_proj = Conv1x1(H, IN_O)
        self.out_proj = Conv1x1(GDN_Y, H)
        self.moe = PinnedSwiGLU()
        self.post_n = nn.Parameter(torch.ones(1, H, 1, 1, dtype=torch.float16))
        self.eps = nn.Buffer(torch.tensor(1e-6, dtype=torch.float16), persistent=False)

    def load_from_layer(self, w: dict) -> None:
        in_proj = np.concatenate(
            [
                np.asarray(w["linear_attn.in_proj_qkv.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_z.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_b.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_a.weight"], np.float32),
            ],
            axis=0,
        )
        self.in_proj.set_w(in_proj)
        self.out_proj.set_w(w["linear_attn.out_proj.weight"])
        pn = w.get("post_attention_layernorm.weight") if hasattr(w, "get") else None
        if pn is None or np.asarray(pn).size != H:
            pn = np.ones(H, np.float32)
        self.post_n.data.copy_(_fp16(np.asarray(pn, np.float32).reshape(1, H, 1, 1)))
        self.moe.load_from_layer(w)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        yin = self.in_proj(h)
        gdn_y = yin[:, :GDN_Y]
        attn = self.out_proj(gdn_y)
        n = _rms(h, self.post_n, self.eps)
        return attn + self.moe(n)


class FlashNextMixers(nn.Module):
    """Attn hyper-connection mix (10240 -> 2560) then body. Recombine left to host."""

    def __init__(self):
        super().__init__()
        self.inv_hc = nn.Buffer(torch.tensor(0.25, dtype=torch.float16), persistent=False)
        self.mix_down = Conv1x1(HC_W, 320)
        self.mix_up = Conv1x1(320, HC_W)
        self.body = FlashNextBody()

    def load_from_layer(self, w: dict) -> None:
        self.mix_down.set_w(w["attn_hyper_connection.input_mix_weight_down.weight"])
        self.mix_up.set_w(w["attn_hyper_connection.input_mix_weight_up.weight"])
        self.body.load_from_layer(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.mix_up(F.silu(self.mix_down(x) * self.inv_hc))
        h = (
            mixed[:, 0:H]
            + mixed[:, H : 2 * H]
            + mixed[:, 2 * H : 3 * H]
            + mixed[:, 3 * H : 4 * H]
        ) * self.inv_hc
        return self.body(h)


class FlashNextGDN(nn.Module):
    """Decode-width GDN step: conv state + SSM state as readonly I/O, S=32 graph.

    Prefill-style unrolled recurrence over the last axis. Host concatenates the
    3-tap conv delay into the qkv stream before the call (we take S tokens of
    already-convolved qkv from in_proj, then depthwise k=4 inside).
    """

    def __init__(self, seq: int):
        super().__init__()
        self.seq = seq
        self.repeat = HV // HK
        self.inv_hc = nn.Buffer(torch.tensor(0.25, dtype=torch.float16), persistent=False)
        self.inv_sqrt_dk = nn.Buffer(
            torch.tensor(DK ** -0.5, dtype=torch.float16), persistent=False
        )
        self.eps = nn.Buffer(torch.tensor(1e-6, dtype=torch.float16), persistent=False)
        self.mix_down = Conv1x1(HC_W, 320)
        self.mix_up = Conv1x1(320, HC_W)
        self.in_qkv = Conv1x1(H, QKV)
        self.in_z = Conv1x1(H, GDN_Y)
        self.in_b = Conv1x1(H, HV)
        self.in_a = Conv1x1(H, HV)
        self.dw = nn.Conv2d(QKV, QKV, kernel_size=(1, 4), groups=QKV, bias=False)
        self.out_proj = Conv1x1(GDN_Y, H)
        self.norm_w = nn.Parameter(torch.ones(1, GDN_Y, 1, 1, dtype=torch.float16))
        self.A_log = nn.Parameter(torch.zeros(HV, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(HV, dtype=torch.float16))
        self.moe = PinnedSwiGLU()
        self.post_n = nn.Parameter(torch.ones(1, H, 1, 1, dtype=torch.float16))

    def load_from_layer(self, w: dict) -> None:
        self.mix_down.set_w(w["attn_hyper_connection.input_mix_weight_down.weight"])
        self.mix_up.set_w(w["attn_hyper_connection.input_mix_weight_up.weight"])
        self.in_qkv.set_w(w["linear_attn.in_proj_qkv.weight"])
        self.in_z.set_w(w["linear_attn.in_proj_z.weight"])
        self.in_b.set_w(w["linear_attn.in_proj_b.weight"])
        self.in_a.set_w(w["linear_attn.in_proj_a.weight"])
        cw = np.asarray(w["linear_attn.conv1d.weight"], np.float32)
        # HF depthwise conv1d: (C, 1, K) -> Conv2d (C, 1, 1, K)
        if cw.ndim == 3:
            cw = cw[:, :, None, :] if cw.shape[-1] == 4 else cw.reshape(QKV, 1, 1, 4)
        self.dw.weight.data.copy_(_fp16(cw).view(QKV, 1, 1, 4))
        self.out_proj.set_w(w["linear_attn.out_proj.weight"])
        nw = np.asarray(w["linear_attn.norm.weight"], np.float32)
        # GDN RMS is per value-head over Dv, then flatten. Store as (1, GDN_Y, 1, 1)
        # by repeating the per-head weight across Dv if it is [HV] or using as-is if [GDN_Y].
        if nw.size == HV:
            nw = np.repeat(nw, DV).reshape(1, GDN_Y, 1, 1)
        elif nw.size == DV:
            nw = np.tile(nw, HV).reshape(1, GDN_Y, 1, 1)
        else:
            nw = nw.reshape(1, -1, 1, 1)
        self.norm_w.data.copy_(_fp16(nw))
        self.A_log.data.copy_(_fp16(w["linear_attn.A_log"]).reshape(HV))
        self.dt_bias.data.copy_(_fp16(w["linear_attn.dt_bias"]).reshape(HV))
        pn = np.asarray(w.get("post_attention_layernorm.weight", np.ones(H)), np.float32)
        self.post_n.data.copy_(_fp16(pn.reshape(1, H, 1, 1)))
        self.moe.load_from_layer(w)

    def _l2(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, heads, dim, S) — norm over dim
        ms = (x * x).mean(dim=2, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)

    def forward(self, x: torch.Tensor, conv_st: torch.Tensor, ssm: torch.Tensor):
        """x (1, 10240, 1, S); conv_st (1, QKV, 1, 3); ssm (1, HV, DV, DK).

        Returns (h_out, new_conv, new_ssm). h_out is mixed-hidden + attn + moe,
        still C=2560 — host does hyper recombine.
        """
        mixed = self.mix_up(F.silu(self.mix_down(x) * self.inv_hc))
        h = (
            mixed[:, 0:H]
            + mixed[:, H : 2 * H]
            + mixed[:, 2 * H : 3 * H]
            + mixed[:, 3 * H : 4 * H]
        ) * self.inv_hc
        qkv = self.in_qkv(h)
        z = self.in_z(h)
        b = self.in_b(h)
        a = self.in_a(h)
        stream = torch.cat([conv_st, qkv], dim=-1)
        conv_pre = self.dw(stream)[..., : self.seq]
        conv_out = F.silu(conv_pre)
        new_conv = stream[..., -3:]

        q_raw = conv_out[:, : HK * DK]
        k_raw = conv_out[:, HK * DK : 2 * HK * DK]
        v = conv_out[:, 2 * HK * DK :]
        B = x.shape[0]
        S = self.seq
        q = q_raw.reshape(B, HK, DK, S)
        k = k_raw.reshape(B, HK, DK, S)
        v = v.reshape(B, HV, DV, S)
        q = self._l2(q) * self.inv_sqrt_dk
        k = self._l2(k)
        q = q.repeat_interleave(self.repeat, dim=1)
        k = k.repeat_interleave(self.repeat, dim=1)

        beta = torch.sigmoid(b.reshape(B, HV, 1, S))
        # softplus(a + dt_bias); keep dtype by avoiding Python floats
        aa = a.reshape(B, HV, 1, S) + self.dt_bias.view(1, HV, 1, 1)
        dt = F.softplus(aa)
        decay = torch.exp(-torch.exp(self.A_log).view(1, HV, 1, 1) * dt)

        state = ssm
        ys = []
        for t in range(S):
            kt = k[:, :, :, t]
            vt = v[:, :, :, t]
            qt = q[:, :, :, t]
            gt = decay[:, :, 0, t]
            bt = beta[:, :, 0, t]
            state = state * gt[:, :, None, None]
            kv_mem = (state * kt[:, :, None, :]).sum(dim=-1)
            delta = (vt - kv_mem) * bt[:, :, None]
            state = state + delta[:, :, :, None] * kt[:, :, None, :]
            ys.append((state * qt[:, :, None, :]).sum(dim=-1))
        y = torch.stack(ys, dim=-1)
        y = y.reshape(B, GDN_Y, 1, S)
        gated = _rms(y, self.norm_w, self.eps) * torch.sigmoid(z)
        attn = self.out_proj(gated)
        n = _rms(h, self.post_n, self.eps)
        hout = attn + self.moe(n)
        return hout, new_conv, state


class GdnCore(nn.Module):
    """Prepared-input gated-delta recurrence. Last dim is Dk=128 (ANE-aligned).

    decay/beta live in column 0 of their blocks so no fp16 exp/softplus
    is compiled into the graph — host prepares those scalars.
    """

    def __init__(self, tokens: int):
        super().__init__()
        self.tokens = tokens

    def forward(self, xin: torch.Tensor) -> torch.Tensor:
        n, h, d = self.tokens, HV, DK
        nh = n * h
        q = xin[0:nh].reshape(n, h, d)
        k = xin[nh : 2 * nh].reshape(n, h, d)
        v = xin[2 * nh : 3 * nh].reshape(n, h, d)
        decay = xin[3 * nh : 4 * nh, 0].reshape(n, h)
        beta = xin[4 * nh : 5 * nh, 0].reshape(n, h)
        state = xin[5 * nh :].reshape(h, d, d)
        outputs = []
        for token in range(n):
            state = state * decay[token, :, None, None]
            memory = (state * k[token, :, :, None]).sum(dim=1)
            delta = (v[token] - memory) * beta[token, :, None]
            state = state + k[token, :, :, None] * delta[:, None, :]
            outputs.append((state * q[token, :, :, None]).sum(dim=1))
        y = torch.stack(outputs, dim=0).reshape(n * h, d)
        return torch.cat((y, state.reshape(h * d, d)), dim=0)


class FlashNextDecodeS1(nn.Module):
    """One GDN decode step + pinned MoE, authored for Neural Engine.

    S=1 math, width-32 I/O. decay is a host input (no exp/softplus in-graph).
    Depthwise conv is 4 channel-wise taps so last dim stays 32.
    conv_pack is [s_{t-3}, s_{t-2}, s_{t-1}] stacked on channels, each (QKV, 32).
    """

    def __init__(self):
        super().__init__()
        self.repeat = HV // HK
        self.inv_sqrt_dk = nn.Buffer(
            torch.tensor(DK ** -0.5, dtype=torch.float16), persistent=False
        )
        self.eps = nn.Buffer(torch.tensor(1e-6, dtype=torch.float16), persistent=False)
        self.in_qkv = Conv1x1(H, QKV)
        self.in_z = Conv1x1(H, GDN_Y)
        self.in_b = Conv1x1(H, HV)
        self.tap0 = nn.Parameter(torch.ones(1, QKV, 1, 1, dtype=torch.float16))
        self.tap1 = nn.Parameter(torch.ones(1, QKV, 1, 1, dtype=torch.float16))
        self.tap2 = nn.Parameter(torch.ones(1, QKV, 1, 1, dtype=torch.float16))
        self.tap3 = nn.Parameter(torch.ones(1, QKV, 1, 1, dtype=torch.float16))
        self.out_proj = Conv1x1(GDN_Y, H)
        self.norm_w = nn.Parameter(torch.ones(1, HV, 1, DV, dtype=torch.float16))
        self.post_n = nn.Parameter(torch.ones(1, H, 1, 1, dtype=torch.float16))
        self.moe = PinnedSwiGLU()

    def load_from_layer(self, w) -> None:
        self.in_qkv.set_w(w["linear_attn.in_proj_qkv.weight"])
        self.in_z.set_w(w["linear_attn.in_proj_z.weight"])
        self.in_b.set_w(w["linear_attn.in_proj_b.weight"])
        cw = np.asarray(w["linear_attn.conv1d.weight"], np.float32).reshape(QKV, 4)
        self.tap0.data.copy_(_fp16(cw[:, 0]).view(1, QKV, 1, 1))
        self.tap1.data.copy_(_fp16(cw[:, 1]).view(1, QKV, 1, 1))
        self.tap2.data.copy_(_fp16(cw[:, 2]).view(1, QKV, 1, 1))
        self.tap3.data.copy_(_fp16(cw[:, 3]).view(1, QKV, 1, 1))
        self.out_proj.set_w(w["linear_attn.out_proj.weight"])
        nw = np.asarray(w["linear_attn.norm.weight"], np.float32)
        if nw.size == HV * DV:
            nw = nw.reshape(1, HV, 1, DV)
        elif nw.size == DV:
            nw = np.broadcast_to(nw.reshape(1, 1, 1, DV), (1, HV, 1, DV)).copy()
        elif nw.size == HV:
            nw = np.broadcast_to(nw.reshape(1, HV, 1, 1), (1, HV, 1, DV)).copy()
        else:
            nw = np.ones((1, HV, 1, DV), np.float32)
        self.norm_w.data.copy_(_fp16(nw))
        pn = w.get("post_attention_layernorm.weight") if hasattr(w, "get") else None
        if pn is None or np.asarray(pn).size != H:
            pn = np.ones(H, np.float32)
        self.post_n.data.copy_(_fp16(np.asarray(pn, np.float32).reshape(1, H, 1, 1)))
        self.moe.load_from_layer(w)

    def _l2_last(self, x: torch.Tensor) -> torch.Tensor:
        acc = (x * x).sum(dim=-1, keepdim=True)
        return x * torch.rsqrt(acc + self.eps)

    def forward(self, h, conv_pack, ssm, decay):
        qkv = self.in_qkv(h)
        z = self.in_z(h)
        b = self.in_b(h)
        s0, s1, s2 = conv_pack.chunk(3, dim=1)
        conv_out = F.silu(s0 * self.tap0 + s1 * self.tap1 + s2 * self.tap2 + qkv * self.tap3)
        new_pack = torch.cat([s1, s2, qkv], dim=1)

        B, _, _, S = h.shape
        q = conv_out[:, : HK * DK].reshape(B, HK, DK, S).permute(0, 1, 3, 2)
        k = conv_out[:, HK * DK : 2 * HK * DK].reshape(B, HK, DK, S).permute(0, 1, 3, 2)
        v = conv_out[:, 2 * HK * DK :].reshape(B, HV, DV, S).permute(0, 1, 3, 2)
        q = self._l2_last(q) * self.inv_sqrt_dk
        k = self._l2_last(k)
        q = q.repeat_interleave(self.repeat, dim=1)
        k = k.repeat_interleave(self.repeat, dim=1)

        qt = q[:, :, 0, :]
        kt = k[:, :, 0, :]
        vt = v[:, :, 0, :]
        gt = decay[:, :, 0, 0]
        bt = torch.sigmoid(b[:, :, 0, 0])
        state = ssm * gt[:, :, None, None]
        kv_mem = (state * kt[:, :, None, :]).sum(dim=-1)
        delta = (vt - kv_mem) * bt[:, :, None]
        state = state + delta[:, :, :, None] * kt[:, :, None, :]
        ytok = (state * qt[:, :, None, :]).sum(dim=-1)

        y = torch.zeros(B, HV, S, DV, dtype=h.dtype, device=h.device)
        y[:, :, 0, :] = ytok
        ms = (y * y).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(ms + self.eps) * self.norm_w
        z4 = z.reshape(B, HV, DV, S).permute(0, 1, 3, 2)
        gated = y * torch.sigmoid(z4)
        g_bc = gated.permute(0, 1, 3, 2).reshape(B, GDN_Y, 1, S)
        attn = self.out_proj(g_bc)
        n = _rms(h, self.post_n, self.eps)
        return attn + self.moe(n), new_pack, state


class FlashNextFront(nn.Module):
    """ANE graph 1: in_proj + 4-tap depthwise conv. Host does SiLU / l2 / decay."""

    def __init__(self):
        super().__init__()
        self.in_proj = Conv1x1(H, IN_O)
        self.tap0 = nn.Parameter(torch.ones(1, QKV, 1, 1, dtype=torch.float16))
        self.tap1 = nn.Parameter(torch.ones(1, QKV, 1, 1, dtype=torch.float16))
        self.tap2 = nn.Parameter(torch.ones(1, QKV, 1, 1, dtype=torch.float16))
        self.tap3 = nn.Parameter(torch.ones(1, QKV, 1, 1, dtype=torch.float16))

    def load_from_layer(self, w) -> None:
        in_proj = np.concatenate(
            [
                np.asarray(w["linear_attn.in_proj_qkv.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_z.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_b.weight"], np.float32),
                np.asarray(w["linear_attn.in_proj_a.weight"], np.float32),
            ],
            axis=0,
        )
        self.in_proj.set_w(in_proj)
        cw = np.asarray(w["linear_attn.conv1d.weight"], np.float32).reshape(QKV, 4)
        self.tap0.data.copy_(_fp16(cw[:, 0]).view(1, QKV, 1, 1))
        self.tap1.data.copy_(_fp16(cw[:, 1]).view(1, QKV, 1, 1))
        self.tap2.data.copy_(_fp16(cw[:, 2]).view(1, QKV, 1, 1))
        self.tap3.data.copy_(_fp16(cw[:, 3]).view(1, QKV, 1, 1))

    def forward(self, h, conv_pack):
        yin = self.in_proj(h)
        qkv = yin[:, :QKV]
        rest = yin[:, QKV:]
        s0, s1, s2 = conv_pack.chunk(3, dim=1)
        conv_pre = s0 * self.tap0 + s1 * self.tap1 + s2 * self.tap2 + qkv * self.tap3
        new_pack = torch.cat([s1, s2, qkv], dim=1)
        return torch.cat([conv_pre, rest], dim=1), new_pack


class FlashNextGdnOnly(nn.Module):
    """ANE graph 2: prepared S=1 GDN + out_proj. MoE runs after the MLP mix."""

    def __init__(self):
        super().__init__()
        self.eps = nn.Buffer(torch.tensor(1e-6, dtype=torch.float16), persistent=False)
        self.out_proj = Conv1x1(GDN_Y, H)
        self.norm_w = nn.Parameter(torch.ones(1, HV, 1, DV, dtype=torch.float16))

    def load_from_layer(self, w) -> None:
        self.out_proj.set_w(w["linear_attn.out_proj.weight"])
        nw = np.asarray(w["linear_attn.norm.weight"], np.float32)
        if nw.size == HV * DV:
            nw = nw.reshape(1, HV, 1, DV)
        elif nw.size == DV:
            nw = np.broadcast_to(nw.reshape(1, 1, 1, DV), (1, HV, 1, DV)).copy()
        elif nw.size == HV:
            nw = np.broadcast_to(nw.reshape(1, HV, 1, 1), (1, HV, 1, DV)).copy()
        else:
            nw = np.ones((1, HV, 1, DV), np.float32)
        self.norm_w.data.copy_(_fp16(nw))

    def forward(self, q, k, v, decay, beta, state, z):
        qt = q[:, :, 0, :]
        kt = k[:, :, 0, :]
        vt = v[:, :, 0, :]
        gt = decay[:, :, 0, 0]
        bt = beta[:, :, 0, 0]
        state = state * gt[:, :, None, None]
        kv_mem = (state * kt[:, :, None, :]).sum(dim=-1)
        delta = (vt - kv_mem) * bt[:, :, None]
        state = state + delta[:, :, :, None] * kt[:, :, None, :]
        ytok = (state * qt[:, :, None, :]).sum(dim=-1)
        y = torch.zeros_like(q)
        y[:, :, 0, :] = ytok
        ms = (y * y).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(ms + self.eps) * self.norm_w
        B, _, _, S = z.shape
        z4 = z.reshape(B, HV, DV, S).permute(0, 1, 3, 2)
        gated = y * torch.sigmoid(z4)
        g_bc = gated.permute(0, 1, 3, 2).reshape(B, GDN_Y, 1, S)
        return self.out_proj(g_bc), state


class FlashNextGdnTail(nn.Module):
    """Legacy fused GDN+MoE (wrong residual). Kept for old artifacts."""

    def __init__(self):
        super().__init__()
        self.eps = nn.Buffer(torch.tensor(1e-6, dtype=torch.float16), persistent=False)
        self.out_proj = Conv1x1(GDN_Y, H)
        self.norm_w = nn.Parameter(torch.ones(1, HV, 1, DV, dtype=torch.float16))
        self.post_n = nn.Parameter(torch.ones(1, H, 1, 1, dtype=torch.float16))
        self.moe = PinnedSwiGLU()

    def load_from_layer(self, w) -> None:
        self.out_proj.set_w(w["linear_attn.out_proj.weight"])
        nw = np.asarray(w["linear_attn.norm.weight"], np.float32)
        if nw.size == HV * DV:
            nw = nw.reshape(1, HV, 1, DV)
        elif nw.size == DV:
            nw = np.broadcast_to(nw.reshape(1, 1, 1, DV), (1, HV, 1, DV)).copy()
        elif nw.size == HV:
            nw = np.broadcast_to(nw.reshape(1, HV, 1, 1), (1, HV, 1, DV)).copy()
        else:
            nw = np.ones((1, HV, 1, DV), np.float32)
        self.norm_w.data.copy_(_fp16(nw))
        pn = w.get("post_attention_layernorm.weight") if hasattr(w, "get") else None
        if pn is None or np.asarray(pn).size != H:
            pn = np.ones(H, np.float32)
        self.post_n.data.copy_(_fp16(np.asarray(pn, np.float32).reshape(1, H, 1, 1)))
        self.moe.load_from_layer(w)

    def forward(self, q, k, v, decay, beta, state, h, z):
        qt = q[:, :, 0, :]
        kt = k[:, :, 0, :]
        vt = v[:, :, 0, :]
        gt = decay[:, :, 0, 0]
        bt = beta[:, :, 0, 0]
        state = state * gt[:, :, None, None]
        kv_mem = (state * kt[:, :, None, :]).sum(dim=-1)
        delta = (vt - kv_mem) * bt[:, :, None]
        state = state + delta[:, :, :, None] * kt[:, :, None, :]
        ytok = (state * qt[:, :, None, :]).sum(dim=-1)
        y = torch.zeros_like(q)
        y[:, :, 0, :] = ytok
        ms = (y * y).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(ms + self.eps) * self.norm_w
        B, _, _, S = h.shape
        z4 = z.reshape(B, HV, DV, S).permute(0, 1, 3, 2)
        gated = y * torch.sigmoid(z4)
        g_bc = gated.permute(0, 1, 3, 2).reshape(B, GDN_Y, 1, S)
        attn = self.out_proj(g_bc)
        n = _rms(h, self.post_n, self.eps)
        return attn + self.moe(n), state


class HostPrep:
    """fp32 SiLU / l2 / head-repeat / decay between the two ANE graphs.

    Writes into preallocated fp16 buffers so the split runner can wrap them
    as NDArray without a torch round-trip (that was ~1 ms of the host time).
    Front graph still does in_proj + 4-tap; SiLU stays here.
    """

    def __init__(self, w, seq: int = SEQ_DEFAULT):
        self.A_log = np.asarray(w["linear_attn.A_log"], np.float32).reshape(1, HV, 1, 1)
        self.dt_bias = np.asarray(w["linear_attn.dt_bias"], np.float32).reshape(1, HV, 1, 1)
        self.inv_sqrt = np.float32(DK ** -0.5)
        self.eps = np.float32(1e-6)
        self.seq = seq
        self.q = np.empty((1, HV, seq, DK), np.float16)
        self.k = np.empty((1, HV, seq, DK), np.float16)
        self.v = np.empty((1, HV, seq, DV), np.float16)
        self.decay = np.empty((1, HV, 1, seq), np.float16)
        self.beta = np.empty((1, HV, 1, seq), np.float16)
        self.z = np.empty((1, GDN_Y, 1, seq), np.float16)

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return host_fastpath.silu(x)

    def _l2(self, x: np.ndarray) -> np.ndarray:
        acc = np.sum(x * x, axis=-1, keepdims=True)
        return x * np.reciprocal(np.sqrt(acc + self.eps))

    def __call__(self, yin: np.ndarray):
        """yin: (1, IN_O, 1, S) from the front graph (conv_pre, not SiLU).

        GDN-only reads slot 0, but the compiled S=32 graph still sees pad
        columns, so this prepares the full last dim of yin.
        """
        return host_fastpath.run_prep(self, yin)


class HostMixPack:
    """Cached 10240→320→10240 mixer weights (hc_norm stored as w−1)."""

    __slots__ = ("hc_n", "down_w", "up_w", "inj_w", "_fp")

    def __init__(self, w: dict, prefix: str):
        self.hc_n = (np.asarray(w[f"{prefix}.hc_norm.weight"], np.float32) + 1.0).reshape(HC, H)
        self.down_w = np.ascontiguousarray(
            np.asarray(w[f"{prefix}.input_mix_weight_down.weight"], np.float32)
        )
        self.up_w = np.ascontiguousarray(
            np.asarray(w[f"{prefix}.input_mix_weight_up.weight"], np.float32)
        )
        inj_key = f"{prefix}.block_inject_weight.weight"
        self.inj_w = (
            np.ascontiguousarray(np.asarray(w[inj_key], np.float32))
            if inj_key in w else None
        )
        self._fp = host_fastpath.MixScratch(1, int(self.down_w.shape[0]))


class HostLayer:
    """Per-layer host mixers + packed MoE. Built once; reused every token."""

    __slots__ = ("attn", "mlp", "moe")

    def __init__(self, w, seq: int = SEQ_DEFAULT, device: torch.device | None = None,
                 store: ExpertHotStore | None = None):
        self.attn = HostMixPack(w, "attn_hyper_connection")
        self.mlp = HostMixPack(w, "mlp_hyper_connection")
        self.moe = HostMoE(w, seq=seq, device=device, store=store)


def host_gated_residual_cached(x_hc: np.ndarray, pack: HostMixPack, eps: float = 1e-6):
    """Exact 4-branch hyper mix using already-converted weights."""
    return host_fastpath.gated_residual(x_hc, pack, eps=eps)


def lm_logits(hidden: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Host lm_head: (T, vocab) or (vocab,) for T=1. weight is (vocab, H)."""
    return host_fastpath.lm_logits(hidden, weight)


def host_gated_residual(x_hc: np.ndarray, w: dict, *, prefix: str,
                        eps: float = 1e-6):
    """Exact 4-branch hyper mix on the host. x_hc is BC1S (1, 10240, 1, S).

    Matches tools/flashnext_reference.py gated_residual: grouped RMS, silu(down/hc),
    sigmoid(up), mean over hc, inj = 2*sigmoid(raw/hc). No ANE — mixers drifted
    when compiled.
    """
    return host_gated_residual_cached(x_hc, HostMixPack(w, prefix), eps=eps)


def host_recombine(out_h: np.ndarray, hyper_input: np.ndarray, inj: np.ndarray) -> np.ndarray:
    """Broadcast attn/MLP output onto the 4 residual branches: h + out * inj."""
    return host_fastpath.recombine(out_h, hyper_input, inj)


def _bsh_to_bc1s(x: np.ndarray) -> np.ndarray:
    """(B, S, C) -> (B, C, 1, S)."""
    x = np.asarray(x, np.float32)
    return np.transpose(x, (0, 2, 1))[:, :, None, :]


def _bc1s_to_bsh(x: np.ndarray) -> np.ndarray:
    """(B, C, 1, S) -> (B, S, C)."""
    x = np.asarray(x, np.float32)
    return np.transpose(x[:, :, 0, :], (0, 2, 1))


def _pad32_into(x: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Write padded BC1S fp16 into `out` (token stays in slot 0)."""
    out.fill(0)
    x = np.asarray(x)
    s = min(int(x.shape[-1]), out.shape[-1])
    out[..., :s] = np.asarray(x[..., :s], np.float16)
    return out


def _pad32(x: np.ndarray, seq: int = SEQ_DEFAULT) -> np.ndarray:
    """BC1S last-dim pad to `seq`, token stays in slot 0. Returns contiguous fp16."""
    x = np.asarray(x)
    out = np.zeros((x.shape[0], x.shape[1], 1, seq), np.float16)
    return _pad32_into(x, out)


def _conv_to_pack_into(conv: np.ndarray, out: np.ndarray) -> np.ndarray:
    """GDNState.conv (1, 3, QKV) → front conv_pack, in place."""
    out.fill(0)
    c = np.asarray(conv, np.float16)
    for i in range(3):
        out[0, i * QKV : (i + 1) * QKV, 0, :] = c[0, i, :, None]
    return out


def _rel(g, r, name: str) -> float:
    g = np.asarray(g, np.float32)
    r = np.asarray(r, np.float32)
    d = float(np.abs(g - r).max() / (np.abs(r).max() + 1e-9))
    print(f"  {name:28s} rel={d:.4f}  g{tuple(g.shape)} r{tuple(r.shape)}")
    return d


QSA_HQ, QSA_HKV, QSA_HD = 24, 2, 256
QSA_ROTARY = 64
QSA_MASK = np.float16(-40000.0)
MLX_GREEDY_THE = (760, (220, 17, 15, 15))  # "The" → " 2000…"


class FlashNextQSADecode(nn.Module):
    """S=1 GQA for every-4th full_attention layer. Dense (context ≤ indexer budget).

    Neural Engine: per-head einsum, readonly KV I/O, RoPE cos/sin as 4D inputs,
    mask -40000 not -inf, every I/O last dim ≥ 32. q_proj is 2*head_dim
    (value + sigmoid gate). Indexer stays on the host.
    """

    def __init__(self, max_s: int = 32):
        super().__init__()
        self.max_s = max_s
        self.hq, self.hkv, self.hd = QSA_HQ, QSA_HKV, QSA_HD
        self.rotary = QSA_ROTARY
        self.rot_half = self.rotary // 2
        self.group = self.hq // self.hkv
        self.eps = nn.Buffer(torch.tensor(1e-6, dtype=torch.float16), persistent=False)
        self.scale = nn.Buffer(
            torch.tensor(self.hd ** -0.5, dtype=torch.float16), persistent=False
        )
        self.q_proj = Conv1x1(H, self.hq * 2 * self.hd)
        self.k_proj = Conv1x1(H, self.hkv * self.hd)
        self.v_proj = Conv1x1(H, self.hkv * self.hd)
        self.o_proj = Conv1x1(self.hq * self.hd, H)
        # HF q_norm/k_norm are (head_dim,) stored as w-1; broadcast over heads.
        self.q_norm = nn.Parameter(torch.ones(1, 1, 1, self.hd, dtype=torch.float16))
        self.k_norm = nn.Parameter(torch.ones(1, 1, 1, self.hd, dtype=torch.float16))

    def load_from_layer(self, w) -> None:
        self.q_proj.set_w(w["self_attn.q_proj.weight"])
        self.k_proj.set_w(w["self_attn.k_proj.weight"])
        self.v_proj.set_w(w["self_attn.v_proj.weight"])
        self.o_proj.set_w(w["self_attn.o_proj.weight"])
        qn = np.asarray(w["self_attn.q_norm.weight"], np.float32).reshape(-1) + 1.0
        kn = np.asarray(w["self_attn.k_norm.weight"], np.float32).reshape(-1) + 1.0
        self.q_norm.data.copy_(_fp16(qn.reshape(1, 1, 1, self.hd)))
        self.k_norm.data.copy_(_fp16(kn.reshape(1, 1, 1, self.hd)))

    def _rms_last(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ms = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * weight

    def forward(self, h, k_cache, v_cache, cos, sin, mask):
        """h (1,H,1,S); caches (1, hkv*hd, 1, max_S); cos/sin (1, rotary/2, 1, S);
        mask (1, max_S+S, 1, S) with -40000 on masked keys.
        Returns (out, new_k, new_v) — host writes new_k/new_v[:, :, :, :1] into cache.
        """
        B, _, _, S = h.shape
        qg = self.q_proj(h).reshape(B, self.hq, 2 * self.hd, S).permute(0, 1, 3, 2)
        q, gate = qg[..., : self.hd], qg[..., self.hd :]
        k = self.k_proj(h).reshape(B, self.hkv, self.hd, S).permute(0, 1, 3, 2)
        v = self.v_proj(h).reshape(B, self.hkv, self.hd, S).permute(0, 1, 3, 2)
        q = self._rms_last(q, self.q_norm)
        k = self._rms_last(k, self.k_norm)
        # cos/sin: (1, half, 1, S) → (1, 1, S, half) broadcasts over heads
        c = cos.permute(0, 2, 3, 1)
        s = sin.permute(0, 2, 3, 1)
        q0, q1 = q[..., : self.rot_half], q[..., self.rot_half : self.rotary]
        k0, k1 = k[..., : self.rot_half], k[..., self.rot_half : self.rotary]
        q_rot = torch.cat((q0 * c - q1 * s, q1 * c + q0 * s), dim=-1)
        k_rot = torch.cat((k0 * c - k1 * s, k1 * c + k0 * s), dim=-1)
        q = torch.cat((q_rot, q[..., self.rotary :]), dim=-1)
        k = torch.cat((k_rot, k[..., self.rotary :]), dim=-1)
        new_k = k.permute(0, 1, 3, 2).reshape(B, self.hkv * self.hd, 1, S)
        new_v = v.permute(0, 1, 3, 2).reshape(B, self.hkv * self.hd, 1, S)
        k_full = torch.cat([k_cache, new_k], dim=-1)
        v_full = torch.cat([v_cache, new_v], dim=-1)
        keys = k_full.reshape(B, self.hkv, self.hd, 1, -1)
        values = v_full.reshape(B, self.hkv, self.hd, 1, -1)
        outs = []
        for head in range(self.hq):
            kv_i = head // self.group
            qh = q[:, head : head + 1, :, :].permute(0, 3, 1, 2)  # (B, hd, 1, S)
            kh = keys[:, kv_i]  # (B, hd, 1, Kv)
            vh = values[:, kv_i]
            kh_t = kh.permute(0, 3, 2, 1)  # (B, Kv, 1, hd)
            vh_t = vh.permute(0, 3, 2, 1)
            attn = torch.einsum("bchq,bkhc->bkhq", qh, kh_t) * self.scale
            attn = torch.softmax(attn + mask, dim=1)
            outs.append(torch.einsum("bkhq,bkhc->bchq", attn, vh_t))
        y = torch.cat(outs, dim=1)
        gate_bc = gate.permute(0, 1, 3, 2).reshape(B, self.hq * self.hd, 1, S)
        return self.o_proj(y * torch.sigmoid(gate_bc)), new_k, new_v


def _load_layer(i: int):
    """Read-only mmap. Caller must keep the loader alive until experts are copied."""
    from tools.flashnext_reference import FlashNextLoader

    assert BASE.is_dir(), BASE
    assert BASE.resolve() != ROOT.resolve()
    loader = FlashNextLoader(str(BASE))
    return loader, loader.layer(i)


def _load_layer0():
    return _load_layer(0)


def _rope_cos_sin(pos: int, seq: int = SEQ_DEFAULT, theta: float = 10_000_000.0):
    """RoPE frequencies as BC1S with last dim ≥ 32. Channels = rotary/2 = 32."""
    half = QSA_ROTARY // 2
    idx = np.arange(0, QSA_ROTARY, 2, dtype=np.float32)
    inv = 1.0 / (theta ** (idx / np.float32(QSA_ROTARY)))
    freqs = np.float32(pos) * inv
    cos = np.zeros((1, half, 1, seq), np.float16)
    sin = np.zeros((1, half, 1, seq), np.float16)
    c = np.cos(freqs).astype(np.float16)
    s = np.sin(freqs).astype(np.float16)
    cos[0, :, 0, :] = c[:, None]
    sin[0, :, 0, :] = s[:, None]
    return cos, sin


def _qsa_mask(offset: int, seq: int = SEQ_DEFAULT, max_s: int = SEQ_DEFAULT) -> np.ndarray:
    """(1, max_S+S, 1, S): 0 on past keys 0..offset-1 and the new token at max_S."""
    kv = max_s + seq
    mask = np.full((1, kv, 1, seq), QSA_MASK, np.float16)
    if offset > 0:
        mask[:, :offset, :, :1] = 0
    mask[:, max_s : max_s + 1, :, :1] = 0
    return mask


def _qsa_feeds(h_bsh: np.ndarray, cache, seq: int = SEQ_DEFAULT):
    """Pack numpy mixed (1,1,H) + AttnCache into QSA I/O. Token lives in slot 0."""
    kv_c = QSA_HKV * QSA_HD
    bufs = (
        np.zeros((1, H, 1, seq), np.float16),
        np.zeros((1, kv_c, 1, seq), np.float16),
        np.zeros((1, kv_c, 1, seq), np.float16),
        np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16),
        np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16),
        np.full((1, seq + seq, 1, seq), QSA_MASK, np.float16),
    )
    return _qsa_feeds_into(h_bsh, cache, seq, bufs)


def _qsa_feeds_into(h_bsh, cache, seq, bufs):
    h, k_cache, v_cache, cos, sin, mask = bufs
    _pad32_into(_bsh_to_bc1s(h_bsh), h)
    k_cache.fill(0)
    v_cache.fill(0)
    off = int(cache.offset)
    kv_c = QSA_HKV * QSA_HD
    if off:
        k = np.asarray(cache.keys[:, :off], np.float16).reshape(QSA_HKV, off, QSA_HD)
        v = np.asarray(cache.values[:, :off], np.float16).reshape(QSA_HKV, off, QSA_HD)
        k_cache[0, :, 0, :off] = k.transpose(0, 2, 1).reshape(kv_c, off)
        v_cache[0, :, 0, :off] = v.transpose(0, 2, 1).reshape(kv_c, off)
    half = QSA_ROTARY // 2
    idx = np.arange(0, QSA_ROTARY, 2, dtype=np.float32)
    inv = 1.0 / (10_000_000.0 ** (idx / np.float32(QSA_ROTARY)))
    freqs = np.float32(off) * inv
    c = np.cos(freqs).astype(np.float16)
    s = np.sin(freqs).astype(np.float16)
    cos.fill(0)
    sin.fill(0)
    cos[0, :, 0, :] = c[:, None]
    sin[0, :, 0, :] = s[:, None]
    mask.fill(QSA_MASK)
    if off > 0:
        mask[:, :off, :, :1] = 0
    mask[:, seq : seq + 1, :, :1] = 0
    return h, k_cache, v_cache, cos, sin, mask


def _conv_to_pack(conv: np.ndarray, seq: int = SEQ_DEFAULT) -> np.ndarray:
    """GDNState.conv (1, 3, QKV) → front conv_pack (1, 3*QKV, 1, seq)."""
    pack = np.zeros((1, 3 * QKV, 1, seq), np.float16)
    return _conv_to_pack_into(conv, pack)


def _pack_to_conv(pack: np.ndarray) -> np.ndarray:
    """new_pack slot 0 → GDNState.conv (1, 3, QKV)."""
    p = np.asarray(pack, np.float32)
    out = np.empty((1, 3, QKV), np.float32)
    for i in range(3):
        out[0, i] = p[0, i * QKV : (i + 1) * QKV, 0, 0]
    return out


def _moe_after_attn(w, attn_bc1s: np.ndarray, hyper: np.ndarray, inj: np.ndarray,
                    layout=None, host: HostMoE | None = None):
    """Host MLP mix + packed/numpy MoE + recombine. attn_bc1s is (1, H, 1, 1)."""
    from tools.flashnext_reference import moe_layer

    post = host_recombine(attn_bc1s, hyper, inj)
    mixed2, hyper2, inj2 = host_gated_residual(
        post, w, prefix="mlp_hyper_connection"
    )
    mixed_bsh = _bc1s_to_bsh(mixed2)[:, :1, :]
    if host is not None:
        host.route_and_pack(mixed_bsh)
        moe = _packed_swiglu(w, host, mixed_bsh)
    else:
        moe = moe_layer(w, mixed_bsh, layout=layout)
    hc = host_recombine(_bsh_to_bc1s(moe), hyper2, inj2)
    return _bc1s_to_bsh(hc)[:, :1, :], mixed_bsh


def _decode_ids(ids: list[int]) -> str:
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(BASE / "tokenizer.json"))
        return tok.decode(ids)
    except Exception:
        return ""


def _ngram_index_exists() -> bool:
    return (BASE / "ngram_index.json").is_file()


def _export(model: nn.Module, example: tuple, names: tuple[list[str], list[str]], tag: str) -> Path:
    from coreai_torch import TorchConverter, get_decomp_table

    model.eval().half()
    t0 = time.perf_counter()
    exported = torch.export.export(model, args=example)
    exported = exported.run_decompositions(get_decomp_table())
    print(f"  torch.export {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    program = (
        TorchConverter()
        .add_exported_program(exported, input_names=names[0], output_names=names[1])
        .to_coreai()
    )
    program.optimize()
    print(f"  coreai convert+optimize {time.perf_counter() - t0:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"flashnext_{tag}.aimodel"
    if out.exists():
        shutil.rmtree(out)
    program.save_asset(out)
    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"  saved {out} ({mb:.1f} MB)")
    return out


async def _bench(path: Path, feeds: dict, ref: dict, seq: int,
                units: tuple[str, ...] = ("gpu", "ane", "cpu")) -> None:
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    kind_by_label = {"cpu": "CPU", "ane": "Neural Engine", "gpu": "GPU"}
    kinds = {str(k): k for k in ComputeUnitKind.available_kinds()}
    print("  available:", list(kinds))
    for label in units:
        key = kind_by_label[label]
        kind = kinds.get(key)
        if kind is None:
            continue
        spec = SpecializationOptions.from_preferred_compute_unit_kind(kind)
        t0 = time.perf_counter()
        try:
            mm = await AIModel.load(str(path), specialization_options=spec)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] LOAD FAILED: {exc}")
            continue
        load_s = time.perf_counter() - t0
        fn = mm.load_function("main")
        ndfeeds = {k: NDArray(v) for k, v in feeds.items()}
        try:
            out = await fn(ndfeeds)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] RUN FAILED ({load_s:.1f}s load): {exc}")
            continue
        rels = []
        for name, r in ref.items():
            g = np.asarray(out[name].numpy(), dtype=np.float32)
            rr = np.asarray(r, dtype=np.float32)
            if not np.isfinite(g).all() or not np.isfinite(rr).all():
                rels.append(f"{name} rel=nan shape={g.shape}")
            else:
                rel = float(np.abs(g - rr).max() / (np.abs(rr).max() + 1e-9))
                rels.append(f"{name} rel={rel:.4f} shape={g.shape}")
        await fn(ndfeeds)
        runs = 8 if label != "cpu" else 3
        t1 = time.perf_counter()
        for _ in range(runs):
            await fn(ndfeeds)
        ms = (time.perf_counter() - t1) * 1e3 / runs
        tok_s = 1000.0 / ms if seq else 0.0
        print(
            f"  [{label}] load {load_s:.1f}s  {ms:.2f} ms/eval  "
            f"~{tok_s:.1f} tok/s if 1 tok/eval  | " + "  ".join(rels)
        )


def stage_smoke(seq: int, skip_bench: bool) -> None:
    torch.manual_seed(0)
    m = SmokeSwiGLU().eval().half()
    x = (torch.randn(1, 256, 1, seq) * 0.02).half()
    with torch.no_grad():
        y = m(x)
    path = _export(m, (x,), (["x"], ["y"]), f"smoke_s{seq}")
    if skip_bench:
        return
    asyncio.run(
        _bench(
            path,
            {"x": np.ascontiguousarray(x.numpy())},
            {"y": y.float().numpy()},
            seq,
        )
    )


def stage_body(seq: int, skip_bench: bool, mixers: bool) -> None:
    print("loading layer 0 from BF16 base (read-only mmap)…")
    loader, w = _load_layer0()
    if mixers:
        m = FlashNextMixers().eval().half()
        m.load_from_layer(w)
        cin = HC_W
        tag = f"mixers_L0_s{seq}"
    else:
        m = FlashNextBody().eval().half()
        m.load_from_layer(w)
        cin = H
        tag = f"body_L0_s{seq}"
    loader.close()
    x = (torch.randn(1, cin, 1, seq) * 0.02).half()
    with torch.no_grad():
        y = m(x)
    path = _export(m, (x,), (["x"], ["y"]), tag)
    if skip_bench:
        return
    asyncio.run(
        _bench(
            path,
            {"x": np.ascontiguousarray(x.numpy())},
            {"y": y.float().numpy()},
            seq,
        )
    )


def stage_gdn(seq: int, skip_bench: bool) -> None:
    print("loading layer 0 from BF16 base (read-only mmap)…")
    loader, w = _load_layer0()
    m = FlashNextGDN(seq).eval().half()
    m.load_from_layer(w)
    loader.close()
    x = (torch.randn(1, HC_W, 1, seq) * 0.02).half()
    conv = torch.zeros(1, QKV, 1, 3, dtype=torch.float16)
    ssm = torch.zeros(1, HV, DV, DK, dtype=torch.float16)
    with torch.no_grad():
        y, nc, ns = m(x, conv, ssm)
    path = _export(
        m,
        (x, conv, ssm),
        (["x", "conv_st", "ssm"], ["y", "new_conv", "new_ssm"]),
        f"gdn_L0_s{seq}",
    )
    if skip_bench:
        return
    asyncio.run(
        _bench(
            path,
            {
                "x": np.ascontiguousarray(x.numpy()),
                "conv_st": np.ascontiguousarray(conv.numpy()),
                "ssm": np.ascontiguousarray(ssm.numpy()),
            },
            {
                "y": y.float().numpy(),
                "new_conv": nc.float().numpy(),
                "new_ssm": ns.float().numpy(),
            },
            seq,
        )
    )


def stage_gdn_core(seq: int, skip_bench: bool) -> None:
    torch.manual_seed(7)
    rows = 5 * seq * HV + HV * DV
    xin = (torch.randn(rows, DK) * 0.02).half()
    nh = seq * HV
    xin[3 * nh : 4 * nh, 0] = 0.99
    xin[4 * nh : 5 * nh, 0] = 0.5
    m = GdnCore(seq).half().eval()
    with torch.no_grad():
        y = m(xin)
    path = _export(m, (xin,), (["xin"], ["y_state"]), f"gdn_core_s{seq}")
    if skip_bench:
        return
    asyncio.run(
        _bench(
            path,
            {"xin": np.ascontiguousarray(xin.numpy())},
            {"y_state": y.float().numpy()},
            seq,
        )
    )


def stage_decode(seq: int, skip_bench: bool) -> None:
    print("loading layer 0 from BF16 base (read-only mmap)…")
    loader, w = _load_layer0()
    m = FlashNextDecodeS1().eval().half()
    m.load_from_layer(w)
    loader.close()
    h = (torch.randn(1, H, 1, seq) * 0.5).half()
    conv_pack = (torch.randn(1, 3 * QKV, 1, seq) * 0.02).half()
    ssm = (torch.randn(1, HV, DV, DK) * 0.01).half()
    decay = torch.full((1, HV, 1, seq), 0.99).half()
    with torch.no_grad():
        y, npk, ns = m(h, conv_pack, ssm, decay)
    path = _export(
        m,
        (h, conv_pack, ssm, decay),
        (["h", "conv_pack", "ssm", "decay"], ["y", "new_pack", "new_ssm"]),
        f"decode_s1_L0_s{seq}",
    )
    if skip_bench:
        return
    asyncio.run(
        _bench(
            path,
            {
                "h": np.ascontiguousarray(h.numpy()),
                "conv_pack": np.ascontiguousarray(conv_pack.numpy()),
                "ssm": np.ascontiguousarray(ssm.numpy()),
                "decay": np.ascontiguousarray(decay.numpy()),
            },
            {
                "y": y.float().numpy(),
                "new_pack": npk.float().numpy(),
                "new_ssm": ns.float().numpy(),
            },
            seq,
            units=("gpu", "ane"),
        )
    )


def _maybe_export(model, example, names, tag: str, reuse: bool) -> Path:
    out = OUT_DIR / f"flashnext_{tag}.aimodel"
    if reuse and out.exists():
        print(f"  reuse {out}")
        return out
    return _export(model, example, names, tag)


def _gdn_asset_paths(layer: int, seq: int) -> tuple[Path, Path]:
    return (
        OUT_DIR / f"flashnext_split_front_L{layer}_s{seq}.aimodel",
        OUT_DIR / f"flashnext_split_gdnonly_L{layer}_s{seq}.aimodel",
    )


def _connected_gdn_path(layer: int) -> Path:
    return OUT_DIR / f"flashnext_connected_gdn_prod_L{layer}.aimodel"


def _pure_step_path(layer: int) -> Path:
    return OUT_DIR / f"flashnext_pure_step_L{layer}.aimodel"


def _qsa_asset_path(layer: int, seq: int) -> Path:
    return OUT_DIR / f"flashnext_qsa_L{layer}_s{seq}.aimodel"


def _moe_asset_path(layer: int, seq: int) -> Path:
    return OUT_DIR / f"flashnext_moe_decode_L{layer}_s{seq}.aimodel"


def _moe_inds_path(layer: int, seq: int) -> Path:
    return OUT_DIR / f"flashnext_moe_inds_L{layer}_s{seq}.npy"


def stage_layers(seq: int, reuse: bool = True, only: list[int] | None = None) -> None:
    """Bake one front+gdn-only (or QSA) .aimodel per decoder layer."""
    from tools.flashnext_reference import AttnCache, FlashNextLoader

    loader = FlashNextLoader(str(BASE))
    types = list(loader.text_config["layer_types"])
    want = set(only) if only else set(range(len(types)))
    front = FlashNextFront().eval().half()
    gdn = FlashNextGdnOnly().eval().half()
    qsa = FlashNextQSADecode(max_s=seq).eval().half()
    h32 = torch.zeros(1, H, 1, seq, dtype=torch.float16)
    pack32 = torch.zeros(1, 3 * QKV, 1, seq, dtype=torch.float16)
    qt = torch.zeros(1, HV, seq, DK, dtype=torch.float16)
    kt = torch.zeros(1, HV, seq, DK, dtype=torch.float16)
    vt = torch.zeros(1, HV, seq, DV, dtype=torch.float16)
    decay_t = torch.zeros(1, HV, 1, seq, dtype=torch.float16)
    beta_t = torch.zeros(1, HV, 1, seq, dtype=torch.float16)
    ssm32 = torch.zeros(1, HV, DV, DK, dtype=torch.float16)
    z_t = torch.zeros(1, GDN_Y, 1, seq, dtype=torch.float16)
    cache = AttnCache.empty(QSA_HKV, QSA_HD, max_len=seq)
    qsa_np = _qsa_feeds(np.zeros((1, 1, H), np.float32), cache, seq)
    qsa_ex = tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in qsa_np)
    n_ok = n_skip = 0
    t_all = time.perf_counter()
    for i, lt in enumerate(types):
        if i not in want:
            continue
        t0 = time.perf_counter()
        w = loader.layer(i)
        if lt == "linear_attention":
            p_front, p_gdn = _gdn_asset_paths(i, seq)
            if reuse and p_front.is_dir() and p_gdn.is_dir():
                print(f"  L{i:02d} GDN reuse", flush=True)
                n_skip += 1
                continue
            front.load_from_layer(w)
            gdn.load_from_layer(w)
            _maybe_export(
                front, (h32, pack32),
                (["h", "conv_pack"], ["yin", "new_pack"]),
                f"split_front_L{i}_s{seq}", reuse,
            )
            _maybe_export(
                gdn, (qt, kt, vt, decay_t, beta_t, ssm32, z_t),
                (["q", "k", "v", "decay", "beta", "state", "z"], ["attn", "new_ssm"]),
                f"split_gdnonly_L{i}_s{seq}", reuse,
            )
        else:
            p_qsa = _qsa_asset_path(i, seq)
            if reuse and p_qsa.is_dir():
                print(f"  L{i:02d} QSA reuse", flush=True)
                n_skip += 1
                continue
            qsa.load_from_layer(w)
            _maybe_export(
                qsa, qsa_ex,
                (["h", "k_cache", "v_cache", "cos", "sin", "mask"],
                 ["out", "new_k", "new_v"]),
                f"qsa_L{i}_s{seq}", reuse,
            )
        n_ok += 1
        print(f"  L{i:02d} {lt} exported in {time.perf_counter() - t0:.1f}s", flush=True)
    loader.close()
    print(
        f"  layers done  exported={n_ok} reused={n_skip}  "
        f"{time.perf_counter() - t_all:.1f}s"
    )


def stage_split(seq: int, skip_bench: bool, reuse: bool = False,
               weight_inputs: bool = False) -> None:
    """Full GDN layer: host mixers + ANE front/GDN/MoE, vs numpy decoder_layer."""
    from tools.flashnext_reference import (
        decoder_layer,
        gated_residual,
        linear_attention_layer,
        moe_layer,
    )

    print("loading layer 0 from BF16 base (read-only mmap)…")
    loader, w = _load_layer0()
    front = FlashNextFront().eval().half()
    front.load_from_layer(w)
    gdn = FlashNextGdnOnly().eval().half()
    gdn.load_from_layer(w)
    moe = RoutedSwiGLU().eval().half()
    moe.load_shared(w)
    host_moe = HostMoE(w, seq=seq)
    prep = HostPrep(w, seq=seq)

    rng = np.random.default_rng(0)
    x_bsh = (rng.standard_normal((1, 1, HC_W)) * 0.1).astype(np.float32)
    x_bc1s = _bsh_to_bc1s(x_bsh)

    print("  host mixers vs numpy gated_residual")
    mixed_h, hyper_h, inj_h = host_gated_residual(
        x_bc1s, w, prefix="attn_hyper_connection"
    )
    mixed_r, _, inj_r = gated_residual(w, "attn_hyper_connection", x_bsh, True)
    _rel(_bc1s_to_bsh(mixed_h), mixed_r, "attn mix")
    _rel(np.transpose(inj_h[:, :, 0, :], (0, 2, 1)), inj_r, "attn inj")

    print("  numpy linear_attention + decoder_layer (real top-10)")
    attn_r, st_r = linear_attention_layer(w, mixed_r, None)
    out_full, _ = decoder_layer(w, x_bsh)

    h32 = torch.from_numpy(_pad32(mixed_h, seq))
    pack32 = torch.zeros(1, 3 * QKV, 1, seq, dtype=torch.float16)
    ssm32 = torch.zeros(1, HV, DV, DK, dtype=torch.float16)
    with torch.no_grad():
        yin_t, new_pack_t = front(h32, pack32)
        q, k, v, decay, beta, z = prep(np.asarray(yin_t.detach().numpy()))
        qt = torch.from_numpy(np.ascontiguousarray(q))
        kt = torch.from_numpy(np.ascontiguousarray(k))
        vt = torch.from_numpy(np.ascontiguousarray(v))
        decay_t = torch.from_numpy(np.ascontiguousarray(decay))
        beta_t = torch.from_numpy(np.ascontiguousarray(beta))
        z_t = torch.from_numpy(np.ascontiguousarray(z))
        attn_t, ns_t = gdn(qt, kt, vt, decay_t, beta_t, ssm32, z_t)
    attn0 = np.asarray(attn_t[..., :1].detach().numpy(), np.float32)
    print("  fp16 GDN vs numpy linear_attention")
    _rel(_bc1s_to_bsh(attn0), attn_r, "pt GDN vs numpy")

    post_attn = host_recombine(attn0, hyper_h, inj_h)
    mixed2, hyper2, inj2 = host_gated_residual(
        post_attn, w, prefix="mlp_hyper_connection"
    )
    mixed2_bsh = _bc1s_to_bsh(mixed2)
    t_pack = time.perf_counter()
    inds, scores_np = host_moe.route_and_pack(mixed2_bsh)
    pack_ms = (time.perf_counter() - t_pack) * 1e3
    print(f"  route+pack {pack_ms:.2f} ms  experts {inds[0].tolist()}")

    moe_r = moe_layer(w, mixed2_bsh)
    h_moe = torch.from_numpy(_pad32(mixed2, seq))
    gate_t = torch.from_numpy(np.ascontiguousarray(host_moe.gate_w))
    up_t = torch.from_numpy(np.ascontiguousarray(host_moe.up_w))
    down_t = torch.from_numpy(np.ascontiguousarray(host_moe.down_w))
    scores_t = torch.from_numpy(np.ascontiguousarray(host_moe.scores))
    with torch.no_grad():
        moe_t = moe(h_moe, gate_t, up_t, down_t, scores_t)
    moe0 = np.asarray(moe_t[..., :1].detach().numpy(), np.float32)
    print("  fp16 routed MoE vs numpy moe_layer")
    _rel(_bc1s_to_bsh(moe0), moe_r, "pt routed vs numpy moe")
    proxy_hc = host_recombine(moe0, hyper2, inj2)
    print("  routed proxy layer vs numpy decoder_layer")
    _rel(_bc1s_to_bsh(proxy_hc), out_full, "routed vs decoder_layer")

    loader.close()

    p_front = _maybe_export(
        front,
        (h32, pack32),
        (["h", "conv_pack"], ["yin", "new_pack"]),
        f"split_front_L0_s{seq}",
        reuse,
    )
    p_gdn = _maybe_export(
        gdn,
        (qt, kt, vt, decay_t, beta_t, ssm32, z_t),
        (["q", "k", "v", "decay", "beta", "state", "z"], ["attn", "new_ssm"]),
        f"split_gdnonly_L0_s{seq}",
        reuse,
    )
    # Weight-as-input convs work (GPU rel 0.0005) but page ~100 MB/token (~380 ms).
    # Bake the token's top-10 into Parameters; scores stay an ANE input.
    if weight_inputs:
        p_moe = _maybe_export(
            moe,
            (h_moe, gate_t, up_t, down_t, scores_t),
            (["h", "gate_w", "up_w", "down_w", "scores"], ["y"]),
            f"split_moe_routed_L0_s{seq}",
            reuse,
        )
    else:
        baked = RoutedSwiGLUBaked().eval().half()
        baked.load_shared(w)
        baked.load_routed(host_moe.gate_w, host_moe.up_w, host_moe.down_w)
        with torch.no_grad():
            moe_t = baked(h_moe, scores_t)
        moe0 = np.asarray(moe_t[..., :1].detach().numpy(), np.float32)
        proxy_hc = host_recombine(moe0, hyper2, inj2)
        p_moe = _maybe_export(
            baked,
            (h_moe, scores_t),
            (["h", "scores"], ["y"]),
            f"split_moe_baked_L0_s{seq}",
            reuse,
        )
        moe = baked
    if skip_bench:
        return

    print("  GPU probe of routed MoE (before ANE)…")
    moe_feeds = {"h": np.ascontiguousarray(h_moe.numpy())}
    moe_ref = {"y": moe_t.float().numpy()}
    if weight_inputs:
        moe_feeds.update({
            "gate_w": np.ascontiguousarray(host_moe.gate_w),
            "up_w": np.ascontiguousarray(host_moe.up_w),
            "down_w": np.ascontiguousarray(host_moe.down_w),
            "scores": np.ascontiguousarray(host_moe.scores),
        })
    else:
        moe_feeds["scores"] = np.ascontiguousarray(host_moe.scores)
    asyncio.run(_bench(p_moe, moe_feeds, moe_ref, seq, units=("gpu",)))

    async def pipeline() -> None:
        from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

        ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
        spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
        print("  loading front + gdn-only + moe on Neural Engine…")
        t0 = time.perf_counter()
        m_front = await AIModel.load(str(p_front), specialization_options=spec)
        m_gdn = await AIModel.load(str(p_gdn), specialization_options=spec)
        m_moe = await AIModel.load(str(p_moe), specialization_options=spec)
        fn_f = m_front.load_function("main")
        fn_g = m_gdn.load_function("main")
        fn_m = m_moe.load_function("main")
        print(f"  loaded in {time.perf_counter() - t0:.1f}s")

        pack_nd = NDArray(np.ascontiguousarray(pack32.numpy()))
        ssm_nd = NDArray(np.ascontiguousarray(ssm32.numpy()))

        async def run_once():
            t_mix = time.perf_counter()
            mixed, hyper, inj = host_gated_residual(
                x_bc1s, w, prefix="attn_hyper_connection"
            )
            h_nd = NDArray(_pad32(mixed, seq))
            t_mix = time.perf_counter() - t_mix

            t1 = time.perf_counter()
            out_f = await fn_f({"h": h_nd, "conv_pack": pack_nd})
            t_front = time.perf_counter() - t1

            t1 = time.perf_counter()
            prep(np.asarray(out_f["yin"].numpy()))
            t_prep = time.perf_counter() - t1

            feeds_g = {
                "q": NDArray(prep.q),
                "k": NDArray(prep.k),
                "v": NDArray(prep.v),
                "decay": NDArray(prep.decay),
                "beta": NDArray(prep.beta),
                "state": ssm_nd,
                "z": NDArray(prep.z),
            }
            t1 = time.perf_counter()
            out_g = await fn_g(feeds_g)
            t_gdn = time.perf_counter() - t1

            t1 = time.perf_counter()
            attn0_a = np.asarray(out_g["attn"].numpy(), np.float32)[..., :1]
            post = host_recombine(attn0_a, hyper, inj)
            mixed2_a, hyper2_a, inj2_a = host_gated_residual(
                post, w, prefix="mlp_hyper_connection"
            )
            h_moe_nd = NDArray(_pad32(mixed2_a, seq))
            t_mlp = time.perf_counter() - t1

            feeds_m = {"h": h_moe_nd, "scores": NDArray(host_moe.scores)}
            if weight_inputs:
                feeds_m["gate_w"] = NDArray(host_moe.gate_w)
                feeds_m["up_w"] = NDArray(host_moe.up_w)
                feeds_m["down_w"] = NDArray(host_moe.down_w)
            t1 = time.perf_counter()
            out_m = await fn_m(feeds_m)
            t_moe = time.perf_counter() - t1

            t1 = time.perf_counter()
            moe0_a = np.asarray(out_m["y"].numpy(), np.float32)[..., :1]
            hc_a = host_recombine(moe0_a, hyper2_a, inj2_a)
            t_rec = time.perf_counter() - t1
            return out_f, out_g, out_m, attn0_a, hc_a, (t_mix, t_front, t_prep, t_gdn, t_mlp, t_moe, t_rec)

        out_f, out_g, out_m, attn0_a, hc_a, _ = await run_once()
        print("  ANE vs fp16 pytorch")
        _rel(np.asarray(out_f["yin"].numpy()), yin_t.float().numpy(), "ANE front")
        _rel(np.asarray(out_f["new_pack"].numpy()), new_pack_t.float().numpy(), "ANE new_pack")
        _rel(np.asarray(out_g["attn"].numpy())[..., :1], attn_t.float().numpy()[..., :1], "ANE gdn attn[0]")
        _rel(np.asarray(out_g["new_ssm"].numpy()), ns_t.float().numpy(), "ANE new_ssm")
        _rel(np.asarray(out_m["y"].numpy())[..., :1], moe_t.float().numpy()[..., :1], "ANE moe[0]")
        print("  ANE vs numpy")
        _rel(_bc1s_to_bsh(attn0_a), attn_r, "ANE GDN vs numpy")
        _rel(_bc1s_to_bsh(hc_a), _bc1s_to_bsh(proxy_hc), "ANE layer vs routed")
        _rel(_bc1s_to_bsh(hc_a), out_full, "ANE layer vs decoder_layer")

        await run_once()
        runs = 8
        acc = np.zeros(7, np.float64)
        for _ in range(runs):
            _, _, _, _, _, ts = await run_once()
            acc += ts
        acc = acc / runs * 1e3
        labels = ("mix", "front", "prep", "gdn", "mlp-mix", "moe", "recombine")
        parts = "  ".join(f"{n} {t:.2f}" for n, t in zip(labels, acc))
        total = float(acc.sum())
        print(
            f"  {parts}  ms  | total {total:.2f} ms/layer  "
            f"route+pack {pack_ms:.2f} ms  "
            f"→ {1000.0 / (total + pack_ms):.1f} tok/s with pack  "
            f"| x48 ≈ {(total + pack_ms) * 48:.0f} ms/tok "
            f"({1000.0 / ((total + pack_ms) * 48):.1f} tok/s)  "
            f"[weight_inputs={weight_inputs}]"
        )

    asyncio.run(pipeline())


def stage_host_test(seq: int) -> None:
    """CPU-only: HostPrep buffers + hyper mix/recombine. No checkpoint, no ANE."""
    rng = np.random.default_rng(0)
    fake = {
        "linear_attn.A_log": rng.standard_normal(HV).astype(np.float32) * 0.1 - 1.0,
        "linear_attn.dt_bias": rng.standard_normal(HV).astype(np.float32) * 0.1,
    }
    prep = HostPrep(fake, seq=seq)
    yin = (rng.standard_normal((1, IN_O, 1, seq)) * 0.3).astype(np.float16)
    t0 = time.perf_counter()
    q, k, v, decay, beta, z = prep(yin)
    dt = (time.perf_counter() - t0) * 1e3
    qn = np.sqrt(np.sum(q.astype(np.float32) ** 2, axis=-1))
    print(f"  HostPrep {dt:.2f} ms (cold)  q {q.shape} k {k.shape} v {v.shape}")
    print(f"  q l2 (want ~{DK**-0.5:.4f} after scale): mean={qn.mean():.4f}  "
          f"decay[{decay.min():.3f},{decay.max():.3f}]  beta[{beta.min():.3f},{beta.max():.3f}]")
    assert q.shape == (1, HV, seq, DK)
    assert 0.0 < float(decay.min()) <= float(decay.max()) <= 1.0
    assert 0.0 < float(beta.min()) <= float(beta.max()) <= 1.0
    q2, _, _, _, _, _ = prep(yin)
    assert q2 is q

    x1 = np.asarray(yin, np.float32)
    conv = HostPrep._silu(x1[:, :QKV])
    B, _, _, S1 = x1.shape
    q_ref = conv[:, : HK * DK].reshape(B, HK, DK, S1).transpose(0, 1, 3, 2)
    k_ref = conv[:, HK * DK : 2 * HK * DK].reshape(B, HK, DK, S1).transpose(0, 1, 3, 2)
    v_ref = conv[:, 2 * HK * DK :].reshape(B, HV, DV, S1).transpose(0, 1, 3, 2)
    q_ref = np.repeat(prep._l2(q_ref) * prep.inv_sqrt, HV // HK, axis=1).astype(np.float16)
    k_ref = np.repeat(prep._l2(k_ref), HV // HK, axis=1).astype(np.float16)
    v_ref = v_ref.astype(np.float16)
    z_ref = x1[:, QKV : QKV + GDN_Y].astype(np.float16)
    b = x1[:, QKV + GDN_Y : QKV + GDN_Y + HV]
    a = x1[:, QKV + GDN_Y + HV :]
    beta_ref = (1.0 / (1.0 + np.exp(-np.clip(b, -80, 80)))).astype(np.float16)
    dtv = np.logaddexp(0.0, a + prep.dt_bias)
    decay_ref = np.exp(-np.exp(prep.A_log) * dtv).astype(np.float16)
    d_q = _rel(q, q_ref, "prep q")
    d_k = _rel(k, k_ref, "prep k")
    d_v = _rel(v, v_ref, "prep v")
    d_z = _rel(z, z_ref, "prep z")
    d_b = _rel(beta, beta_ref, "prep beta")
    d_d = _rel(decay, decay_ref, "prep decay")
    assert max(d_q, d_k, d_v, d_z, d_b, d_d) < 1e-5

    from tools.flashnext_reference import LayerWeights, gated_residual

    mix_w = {
        "attn_hyper_connection.hc_norm.weight": np.ones(HC_W, np.float32),
        "attn_hyper_connection.input_mix_weight_down.weight":
            (rng.standard_normal((320, HC_W)) * 0.02).astype(np.float32),
        "attn_hyper_connection.input_mix_weight_up.weight":
            (rng.standard_normal((HC_W, 320)) * 0.02).astype(np.float32),
        "attn_hyper_connection.block_inject_weight.weight":
            (rng.standard_normal((HC, HC_W)) * 0.02).astype(np.float32),
    }
    x_hc = (rng.standard_normal((1, HC_W, 1, seq)) * 0.1).astype(np.float32)
    mixed, hyper, inj = host_gated_residual(x_hc, mix_w, prefix="attn_hyper_connection")
    out = (rng.standard_normal((1, H, 1, seq)) * 0.1).astype(np.float32)
    rec = host_recombine(out, hyper, inj)
    print(f"  hyper mixed {mixed.shape} inj {inj.shape} recombine {rec.shape}")
    assert mixed.shape == (1, H, 1, seq)
    assert inj.shape == (1, HC, 1, seq)
    assert rec.shape == (1, HC_W, 1, seq)
    lw = LayerWeights(
        0, "linear_attention", mix_w,
        {"hc_count": HC, "hidden_size": H, "rms_norm_eps": 1e-6},
    )
    mixed_r, _, inj_r = gated_residual(lw, "attn_hyper_connection", _bc1s_to_bsh(x_hc), True)
    d_mix = _rel(_bc1s_to_bsh(mixed), mixed_r, "host vs ref mix")
    d_inj = _rel(np.transpose(inj[:, :, 0, :], (0, 2, 1)), inj_r, "host vs ref inj")
    assert d_mix < 1e-5 and d_inj < 1e-5

    x_s1 = np.ascontiguousarray(x_hc[..., :1])
    mixed1, _, inj1 = host_gated_residual(x_s1, mix_w, prefix="attn_hyper_connection")
    mixed1_r, _, inj1_r = gated_residual(lw, "attn_hyper_connection", _bc1s_to_bsh(x_s1), True)
    d_m1 = _rel(_bc1s_to_bsh(mixed1), mixed1_r, "host vs ref mix S=1")
    d_i1 = _rel(np.transpose(inj1[:, :, 0, :], (0, 2, 1)), inj1_r, "host vs ref inj S=1")
    assert d_m1 < 1e-5 and d_i1 < 1e-5

    pack = HostMixPack(mix_w, "attn_hyper_connection")

    def _ms(fn, n=80, warmup=16) -> float:
        for _ in range(warmup):
            fn()
        t1 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t1) / n * 1e3

    ms_prep = _ms(lambda: prep(yin))
    ms_mix1 = _ms(lambda: host_gated_residual_cached(x_s1, pack))
    ms_mix32 = _ms(lambda: host_gated_residual_cached(x_hc, pack))
    print(
        f"  bench  HostPrep S={seq} {ms_prep:.3f} ms  "
        f"x36={ms_prep * 36:.1f} ms/tok"
    )
    print(
        f"  bench  mixer S=1 {ms_mix1:.3f} ms  x96={ms_mix1 * 96:.1f} ms/tok  "
        f"mixer S={seq} {ms_mix32:.3f} ms"
    )
    print("  QSA: run `qsa` to export FlashNextQSADecode (layer 3, max_S=32)")
    print("  host_test ok")


def stage_qsa(seq: int, skip_bench: bool, reuse: bool = False) -> None:
    """Export layer-3 QSA, compare vs numpy, then mix→QSA→recombine→numpy MoE."""
    from tools.flashnext_reference import (
        AttnCache,
        decoder_layer,
        full_attention_layer,
        moe_layer,
    )

    print("loading layer 3 (full_attention) from BF16 base (read-only mmap)…")
    print(f"  ngram_index.json exists={_ngram_index_exists()} "
          f"(mlx-lm zeros fallback; PLE is layer 1 / ple_layer_ids=[2])")
    loader, w = _load_layer(3)
    assert w.layer_type == "full_attention", w.layer_type
    m = FlashNextQSADecode(max_s=seq).eval().half()
    m.load_from_layer(w)

    rng = np.random.default_rng(0)
    x_bsh = (rng.standard_normal((1, 1, HC_W)) * 0.1).astype(np.float32)
    mixed_h, hyper_h, inj_h = host_gated_residual(
        _bsh_to_bc1s(x_bsh), w, prefix="attn_hyper_connection"
    )
    mixed_bsh = _bc1s_to_bsh(mixed_h)[:, :1, :]

    cache_np = AttnCache.empty(QSA_HKV, QSA_HD, max_len=max(8, seq))
    attn_r = full_attention_layer(w, mixed_bsh, cache_np)
    print("  fp16 QSA vs numpy full_attention_layer")
    cache_in = AttnCache.empty(QSA_HKV, QSA_HD, max_len=max(8, seq))
    h, k_cache, v_cache, cos, sin, mask = _qsa_feeds(mixed_bsh, cache_in, seq)
    ht = torch.from_numpy(np.ascontiguousarray(h))
    kct = torch.from_numpy(np.ascontiguousarray(k_cache))
    vct = torch.from_numpy(np.ascontiguousarray(v_cache))
    cost = torch.from_numpy(np.ascontiguousarray(cos))
    sint = torch.from_numpy(np.ascontiguousarray(sin))
    maskt = torch.from_numpy(np.ascontiguousarray(mask))
    with torch.no_grad():
        out_t, nk_t, nv_t = m(ht, kct, vct, cost, sint, maskt)
    attn0 = np.asarray(out_t[..., :1].detach().numpy(), np.float32)
    _rel(_bc1s_to_bsh(attn0), attn_r, "pt QSA vs numpy")
    nk0 = np.asarray(nk_t[..., :1].detach().numpy(), np.float32).reshape(QSA_HKV, QSA_HD)
    _rel(nk0, cache_np.keys[:, 0], "pt new_k vs numpy k")

    post = host_recombine(attn0, hyper_h, inj_h)
    mixed2, hyper2, inj2 = host_gated_residual(
        post, w, prefix="mlp_hyper_connection"
    )
    moe_r = moe_layer(w, _bc1s_to_bsh(mixed2)[:, :1, :])
    hc = host_recombine(_bsh_to_bc1s(moe_r), hyper2, inj2)
    cache_layer = AttnCache.empty(QSA_HKV, QSA_HD, max_len=max(8, seq))
    out_full, _ = decoder_layer(w, x_bsh, attn_cache=cache_layer)
    print("  QSA + numpy MoE vs numpy decoder_layer")
    _rel(_bc1s_to_bsh(hc)[:, :1, :], out_full, "qsa residual vs decoder")

    example = (ht, kct, vct, cost, sint, maskt)
    names = (
        ["h", "k_cache", "v_cache", "cos", "sin", "mask"],
        ["out", "new_k", "new_v"],
    )
    if os.environ.get("FLASHNEXT_SKIP_EXPORT"):
        print("  FLASHNEXT_SKIP_EXPORT=1 — not writing .aimodel")
        loader.close()
        return
    path = _maybe_export(m, example, names, f"qsa_L3_s{seq}", reuse)
    loader.close()
    if skip_bench:
        return
    feeds = {
        "h": np.ascontiguousarray(h),
        "k_cache": np.ascontiguousarray(k_cache),
        "v_cache": np.ascontiguousarray(v_cache),
        "cos": np.ascontiguousarray(cos),
        "sin": np.ascontiguousarray(sin),
        "mask": np.ascontiguousarray(mask),
    }
    ref = {
        "out": out_t.float().numpy(),
        "new_k": nk_t.float().numpy(),
        "new_v": nv_t.float().numpy(),
    }
    print("  GPU probe of QSA (before ANE)…")
    units = tuple(
        u.strip() for u in os.environ.get("FLASHNEXT_QSA_UNITS", "gpu,ane").split(",")
        if u.strip()
    )
    asyncio.run(_bench(path, feeds, ref, seq, units=units or ("gpu", "ane")))


def _numpy_greedy(seq: int, max_new: int, prompt_ids: list[int], moe_hook=None):
    """48-layer numpy decode. moe_hook(w, hidden, layout) may wrap moe_layer."""
    import tools.flashnext_reference as ref

    orig = ref.moe_layer
    if moe_hook is not None:
        ref.moe_layer = moe_hook
    loader = ref.FlashNextLoader(str(BASE))
    cfg = loader.text_config
    n_layers = int(cfg["num_hidden_layers"])
    layer_cache: dict[int, object] = {}
    layout = None
    gdn_state = [None] * n_layers
    attn_state = [None] * n_layers
    for i, lt in enumerate(cfg["layer_types"]):
        if lt == "linear_attention":
            gdn_state[i] = ref.zero_gdn_state(cfg, 1)
        else:
            attn_state[i] = ref.AttnCache.empty(
                int(cfg["num_key_value_heads"]),
                int(cfg["head_dim"]),
                max_len=max(seq, len(prompt_ids) + max_new + 8),
            )

    def layer_w(i: int):
        nonlocal layout
        w = layer_cache.get(i)
        if w is None:
            w = loader.layer(i)
            layer_cache[i] = w
            if layout is None:
                layout = ref.infer_expert_layout(w, verbose=True)
        return w

    generated: list[int] = []
    cur = list(prompt_ids)
    t_all = time.perf_counter()
    try:
        for step in range(max_new):
            hidden = ref.real_embedding_hidden(loader, [cur[-1]])
            t_tok = time.perf_counter()
            for i in range(n_layers):
                w = layer_w(i)
                if w.layer_type == "linear_attention":
                    hidden, gdn_state[i] = ref.decoder_layer(
                        w, hidden, state=gdn_state[i], layout=layout,
                        use_mlx_l2_eps=True,
                    )
                else:
                    hidden, attn_state[i] = ref.decoder_layer(
                        w, hidden, layout=layout, use_mlx_l2_eps=True,
                        attn_cache=attn_state[i],
                    )
            mixed = ref.mixer_hidden(loader, hidden)
            logits = ref.lm_logits(loader, mixed).reshape(-1)
            nxt = int(np.argmax(logits))
            generated.append(nxt)
            cur.append(nxt)
            print(
                f"  step {step + 1}/{max_new}  id={nxt}  "
                f"{_decode_ids([nxt])!r}  {time.perf_counter() - t_tok:.2f}s  "
                f"{generated}",
                flush=True,
            )
    finally:
        ref.moe_layer = orig
        loader.close()
    print(f"  numpy greedy {_decode_ids(cur)!r}  "
          f"{time.perf_counter() - t_all:.1f}s")
    return generated


def stage_stickiness(seq: int, max_new: int, prompt_ids: list[int]) -> None:
    """Prove GDN vs QSA expert stickiness on the BF16 numpy stack."""
    import tools.flashnext_reference as ref

    box: dict = {"cur": []}
    steps: list[list[dict]] = []
    orig = ref.moe_layer

    def hooked(w, hidden, stats=ref._NULL_STATS, layout=None, norm_topk_prob=True):
        x = np.asarray(hidden, np.float32).reshape(-1, H)
        rw = np.asarray(w["mlp.gate.weight"], np.float32)
        logits = x @ rw.T
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp((logits - m).astype(np.float64))
        probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)[0]
        inds = np.argpartition(probs, -K_PIN)[-K_PIN:]
        inds = inds[np.argsort(-probs[inds])]
        box["cur"].append({
            "i": int(w.index),
            "type": w.layer_type,
            "inds": inds.astype(np.int32),
            "probs": probs,
        })
        return orig(w, hidden, stats, layout, norm_topk_prob)

    def stepping(w, hidden, stats=ref._NULL_STATS, layout=None, norm_topk_prob=True):
        if int(w.index) == 0:
            if box["cur"]:
                steps.append(box["cur"])
            box["cur"] = []
        return hooked(w, hidden, stats, layout, norm_topk_prob)

    generated = _numpy_greedy(seq, max_new, prompt_ids, moe_hook=stepping)
    if box["cur"]:
        steps.append(box["cur"])
    if not steps:
        print("  no router records")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rec in steps[0]:
        np.save(_moe_inds_path(rec["i"], seq), rec["inds"])
    print(f"  saved token-0 top-10 ids under {OUT_DIR}/flashnext_moe_inds_L*_s{seq}.npy")

    t0 = steps[0]
    print(f"  {'tok':>3}  {'mean Jaccard':>12} {'mean mass@t0':>12}   "
          f"GDN jac/mass     QSA jac/mass     "
          f"GDN≥{MOE_MASS_THRESH}  QSA≥{MOE_MASS_THRESH}")
    for t, recs in enumerate(steps):
        by_i = {r["i"]: r for r in recs}
        jacs, masses, gdn_j, gdn_m, qsa_j, qsa_m = [], [], [], [], [], []
        gdn_hit = qsa_hit = gdn_n = qsa_n = 0
        for r0 in t0:
            r = by_i[r0["i"]]
            a, b = set(r0["inds"].tolist()), set(r["inds"].tolist())
            jac = len(a & b) / max(len(a | b), 1)
            mass = float(r["probs"][r0["inds"]].sum())
            jacs.append(jac)
            masses.append(mass)
            if r0["type"] == "linear_attention":
                gdn_j.append(jac)
                gdn_m.append(mass)
                gdn_n += 1
                gdn_hit += int(mass >= MOE_MASS_THRESH)
            else:
                qsa_j.append(jac)
                qsa_m.append(mass)
                qsa_n += 1
                qsa_hit += int(mass >= MOE_MASS_THRESH)
        print(
            f"  {t:3d}  {np.mean(jacs):12.3f} {np.mean(masses):12.3f}   "
            f"{np.mean(gdn_j):.3f}/{np.mean(gdn_m):.3f}        "
            f"{np.mean(qsa_j):.3f}/{np.mean(qsa_m):.3f}        "
            f"{gdn_hit}/{gdn_n}          {qsa_hit}/{qsa_n}",
            flush=True,
        )
    print("  vs previous token (not vs t0):")
    for t in range(1, len(steps)):
        gdn_j, qsa_j = [], []
        prev = {r["i"]: r for r in steps[t - 1]}
        for r in steps[t]:
            a = set(prev[r["i"]]["inds"].tolist())
            b = set(r["inds"].tolist())
            jac = len(a & b) / max(len(a | b), 1)
            (gdn_j if r["type"] == "linear_attention" else qsa_j).append(jac)
        print(
            f"    tok {t - 1}->{t}  GDN Jaccard {np.mean(gdn_j):.3f}  "
            f"QSA Jaccard {np.mean(qsa_j):.3f}",
            flush=True,
        )
    print(f"  generated ids={generated}  text={_decode_ids(prompt_ids + generated)!r}")


def stage_moe(seq: int, reuse: bool, prompt_ids: list[int]) -> None:
    """Bake per-layer scored MoE from token-0 top-10 (stickiness npy or 1-token)."""
    from tools.flashnext_reference import FlashNextLoader

    missing = [i for i in range(48) if not _moe_inds_path(i, seq).is_file()]
    if missing:
        print(f"  missing inds for {len(missing)} layers — running 1 numpy token")
        stage_stickiness(seq, 1, prompt_ids)
    loader = FlashNextLoader(str(BASE))
    n_layers = int(loader.text_config["num_hidden_layers"])
    baked = RoutedSwiGLUBaked().eval().half()
    h32 = torch.zeros(1, H, 1, seq, dtype=torch.float16)
    scores_t = torch.zeros(1, K_PIN, 1, seq, dtype=torch.float16)
    n_ok = 0
    t_all = time.perf_counter()
    for i in range(n_layers):
        p = _moe_asset_path(i, seq)
        ip = _moe_inds_path(i, seq)
        if reuse and p.is_dir() and ip.is_file():
            print(f"  L{i:02d} MoE reuse", flush=True)
            n_ok += 1
            continue
        w = loader.layer(i)
        inds = np.load(ip)
        host = HostMoE(w, seq=seq)
        host.pack_ids(inds)
        baked.load_shared(w)
        baked.load_routed(host.gate_w, host.up_w, host.down_w)
        _maybe_export(
            baked, (h32, scores_t),
            (["h", "scores"], ["y"]),
            f"moe_decode_L{i}_s{seq}",
            reuse=False,
        )
        n_ok += 1
        print(f"  L{i:02d} MoE baked experts {inds.tolist()}", flush=True)
    loader.close()
    print(f"  moe bake {n_ok} layers  {time.perf_counter() - t_all:.1f}s")


def stage_generate(seq: int, max_new: int, prompt_ids: list[int],
                   no_ane: bool = False, host_front: bool = False,
                   ppl_ids: list[int] | None = None) -> None:
    """48-layer greedy decode. ANE every layer that has a compiled asset."""
    if not prompt_ids or max_new < 1:
        raise ValueError("generate requires a nonempty prompt and max_new >= 1")
    # Multi-token QSA graphs are `indexer_budget` wide (default 2048). That is
    # the number of keys per submit, not max context: the host KV cache grows
    # and the indexer gathers a budget-sized subset. Single-token graphs still
    # bake max_S=seq and really do cap the context.
    _pk = int(os.environ.get("FLASHNEXT_PREFILL_K", "0") or 0)
    _pm = int(os.environ.get("FLASHNEXT_PREFILL_MAX_S", "2048"))
    _wide = (OUT_DIR / f"flashnext_multitoken_qsa_L3_m{_pm}.aimodel").is_dir()
    # MIL QSA is built per rung at run time, so it has no baked max_S to cap.
    _wide = _wide or os.environ.get("FLASHNEXT_MIL_QSA", "0") not in ("0", "false", "")
    _pk = _pk or (1 if _wide else 0)
    _host_ctx = len(prompt_ids) + max_new
    if not no_ane and not (_pk and _wide) and _host_ctx - 2 > seq:
        raise ValueError(
            f"prompt + generation ({_host_ctx}) exceeds the compiled single-token "
            f"QSA cache ({seq}); set FLASHNEXT_PREFILL_K and export "
            f"flashnext_multitoken_qsa_L*_m{_pm}.aimodel for long context")
    from tools.flashnext_reference import (
        AttnCache,
        LayerWeights,
        decoder_layer,
        gated_residual,
        real_embedding_hidden,
        zero_gdn_state,
    )

    prompt_id, mlx_prefix = MLX_GREEDY_THE
    ple_enabled = os.environ.get("FLASHNEXT_PLE", "0") == "1"
    print("  PLE: checkpoint-derived SSD rows" if ple_enabled else
          "  PLE: zero-table baseline (FLASHNEXT_PLE=1 enables real SSD rows)")
    if len(prompt_ids) > 16:
        print(f"  prompt n={len(prompt_ids)} ids={prompt_ids[:8]}…{prompt_ids[-4:]}  "
              f"max_new={max_new}  MLX 4-bit greedy after {prompt_id}: {list(mlx_prefix)}…")
    else:
        print(f"  prompt ids={prompt_ids}  max_new={max_new}  "
              f"MLX 4-bit greedy after {prompt_id}: {list(mlx_prefix)}…")
    text = _decode_ids(prompt_ids) if len(prompt_ids) <= 32 else ""
    if text:
        print(f"  prompt text={text!r}")

    from tools.flashnext_reference import FlashNextLoader

    loader = FlashNextLoader(str(BASE))
    cfg = loader.text_config
    n_layers = int(cfg["num_hidden_layers"])
    types = list(cfg["layer_types"])
    layer_cache: dict[int, object] = {}
    layout = None
    gdn_state = [None] * n_layers
    attn_state = [None] * n_layers
    for i, lt in enumerate(types):
        if lt == "linear_attention":
            gdn_state[i] = zero_gdn_state(cfg, 1)
        else:
            attn_state[i] = AttnCache.empty(
                int(cfg["num_key_value_heads"]),
                int(cfg["head_dim"]),
                max_len=max(seq, len(prompt_ids) + max_new + 8),
                dtype=np.float16,
            )

    def layer_w(i: int):
        nonlocal layout
        w = layer_cache.get(i)
        if w is None:
            from tools.flashnext_reference import infer_expert_layout

            t0 = time.perf_counter()
            w = loader.layer(i)
            layer_cache[i] = w
            if layout is None:
                layout = infer_expert_layout(w, verbose=True)
            print(f"    mmap L{i} {w.layer_type} {time.perf_counter() - t0:.1f}s",
                  flush=True)
        return w

    def numpy_layer(i: int, hidden):
        w = layer_w(i)
        if w.layer_type == "linear_attention":
            hidden, gdn_state[i] = decoder_layer(
                w, hidden, state=gdn_state[i], layout=layout,
                use_mlx_l2_eps=True,
            )
        else:
            hidden, attn_state[i] = decoder_layer(
                w, hidden, layout=layout, use_mlx_l2_eps=True,
                attn_cache=attn_state[i],
            )
        return hidden

    generated: list[int] = []
    cur = list(prompt_ids)
    gdn_assets: dict[int, tuple[Path, Path]] = {}
    connected_assets: dict[int, Path] = {}
    pure_assets: dict[int, Path] = {}
    compact_gdn = os.environ.get("FLASHNEXT_COMPACT_GDN", "0") == "1"
    want_connected = os.environ.get("FLASHNEXT_CONNECTED", "1").strip().lower() not in (
        "0", "false", "no",
    )
    want_pure = os.environ.get("FLASHNEXT_PURE", "1").strip().lower() not in (
        "0", "false", "no",
    )
    qsa_assets: dict[int, Path] = {}
    n_gdn = sum(1 for lt in types if lt == "linear_attention")
    if not no_ane:
        for i, lt in enumerate(types):
            if lt == "linear_attention":
                pp = _pure_step_path(i)
                if want_pure and pp.is_dir():
                    pure_assets[i] = pp
                pc = _connected_gdn_path(i)
                if pc.is_dir():
                    connected_assets[i] = pc
                pf, pg = _gdn_asset_paths(i, seq)
                if compact_gdn:
                    pg = OUT_DIR / f"flashnext_compact_tail_L{i}.aimodel"
                    if not pg.is_dir():
                        raise FileNotFoundError(f"missing compact tail: {pg}; run probes/flashnext_compact_tail.py --export-all")
                if pg.is_dir() and (pf.is_dir() or host_front):
                    gdn_assets[i] = (pf, pg)
            else:
                pq = _qsa_asset_path(i, seq)
                if pq.is_dir():
                    qsa_assets[i] = pq
        if pure_assets:
            for i in pure_assets:
                connected_assets.pop(i, None)
                gdn_assets.pop(i, None)
        if want_connected and len(connected_assets) == n_gdn - len(pure_assets) and n_gdn:
            gdn_assets = {}
        else:
            if connected_assets and want_connected and not pure_assets:
                print(
                    f"  connected GDN incomplete ({len(connected_assets)}/{n_gdn}); "
                    f"keeping 2-submit split",
                    flush=True,
                )
            if not pure_assets:
                connected_assets = {}
    host_front_decode = (
        (not no_ane) and host_front and bool(gdn_assets)
        and not connected_assets and not pure_assets
    )
    moe_dev = _moe_device()
    moe_mode = _moe_decode_mode()
    if moe_mode == "fp16":
        moe_label = "host fp16 hot store (artifacts/experts_f16 LRU)"
    elif moe_mode == "mlxresident":
        moe_label = "resident quantized MLX GPU MoE (ANE attention; explicit diagnostic)"
    elif moe_mode == "q4gemv":
        moe_label = "parallel native CPU INT4 SwiGLU (FLASHNEXT_MOE=q4gemv)"
    else:
        moe_label = "hybrid 4-bit store → dequant top-10 → packed fp32 GEMV (FLASHNEXT_MOE=hybrid)"
    if pure_assets:
        gdn_submit = f"1 (pure_step {len(pure_assets)}/{n_gdn})"
    elif connected_assets:
        gdn_submit = "1 (connected GDN)"
    elif host_front_decode:
        gdn_submit = "1 (host front)"
    else:
        gdn_submit = "2 (ANE front)"
    print(
        f"  ANE GDN={len(pure_assets) + len(connected_assets) or len(gdn_assets)}  "
        f"QSA={len(qsa_assets)}  "
        f"GDN submits={gdn_submit}  "
        f"MoE={moe_label} ({moe_dev})  FLASHNEXT_MOE={moe_mode}"
        if (pure_assets or connected_assets or gdn_assets or qsa_assets) else
        "  ANE: off (numpy 48-layer). run `layers` then generate"
    )
    print(
        f"  context host={_host_ctx}  QSA graph width={_pm if (_pk and _wide) else seq}  "
        f"prefill_k={_pk}  (graph width is indexer budget, not max context)",
        flush=True,
    )

    mix_prefix = "model.language_model.hyper_connection_mixer"
    mix_tensors = {
        "hyper_connection_mixer.hc_norm.weight":
            loader.get(f"{mix_prefix}.hc_norm.weight"),
        "hyper_connection_mixer.input_mix_weight_down.weight":
            loader.get(f"{mix_prefix}.input_mix_weight_down.weight"),
        "hyper_connection_mixer.input_mix_weight_up.weight":
            loader.get(f"{mix_prefix}.input_mix_weight_up.weight"),
    }
    mix_w = LayerWeights(
        index=-1, layer_type="mixer", tensors=mix_tensors,
        config=cfg, loader=loader,
    )
    head_mode = os.environ.get("FLASHNEXT_HEAD", "cpu").strip().lower()
    q_head = None
    if head_mode == "mlx":
        from runtime.flashnext_mlx_head import QuantizedHead
        q_head = QuantizedHead()
        print(f"  lm_head: quantized GPU head {q_head.bits}-bit "
              f"group {q_head.group_size}  {q_head.nbytes / 1e9:.2f} GB", flush=True)
    print("  caching lm_head…", flush=True)
    t_lm = time.perf_counter()
    lm_w = (np.zeros((1, H), np.float32) if q_head is not None
            else np.ascontiguousarray(loader.get("lm_head.weight"), dtype=np.float32))
    print(f"  lm_head {lm_w.shape} {lm_w.nbytes / 1e9:.2f} GB  "
          f"{time.perf_counter() - t_lm:.1f}s", flush=True)

    async def run() -> None:
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
        from runtime.coreai_surfaces import SurfacePool, wrap_ndarray

        keep = []
        fn_front: dict[int, object] = {}
        fn_gdn: dict[int, object] = {}
        fn_connected: dict[int, object] = {}
        fn_pure: dict[int, object] = {}
        fn_qsa: dict[int, object] = {}
        fn_multi: dict[int, object] = {}
        fn_qsa_multi: dict[int, object] = {}
        fn_qsa_step: dict[int, object] = {}
        fn_qsa_step1: dict[int, object] = {}
        mil_gdn: dict[int, object] = {}
        mil_qsa: dict[int, object] = {}
        # A second, wider set of the same graphs, used only to walk the prompt.
        # Decode wants a narrow graph because a block confirms two or three
        # tokens; prefill wants the widest that compiles, because every slot
        # carries a real token and the ~1.1 ms a submit costs amortises. The
        # two sets are independent programs and coexist: 132 of them load.
        mil_gdn_pf: dict[int, object] = {}
        mil_qsa_pf: dict[int, object] = {}
        prefill_mil_k = int(os.environ.get("FLASHNEXT_PREFILL_MIL_K", "0") or 0)
        # Which set the block path is currently driving. Swapped for the
        # prompt and swapped back before the first drafted block.
        _active = {"gdn": mil_gdn, "qsa": mil_qsa}
        use_mil = os.environ.get("FLASHNEXT_MIL_GDN", "0") not in ("0", "false", "")
        use_mil_qsa = os.environ.get("FLASHNEXT_MIL_QSA", "0") not in ("0", "false", "")
        _idx_host = os.environ.get("FLASHNEXT_IDX_HOST", "0") not in (
            "0", "false", "")
        # Speculation width: the MIL graphs are baked at this many live slots
        # and are used for plain decode too, which just leaves the extra slots
        # unread. One set of programs, not two — the ANE ceiling is ~80.
        spec_k = int(os.environ.get("FLASHNEXT_SPEC", "0") or 0)
        spec_kv_base: dict[int, int] = {}
        drafter = None
        fn_qsa_rung: dict[tuple, dict] = {}
        qsa_rungs = [int(v) for v in os.environ.get(
            "FLASHNEXT_QSA_RUNGS", "256").split(",") if v.strip()]
        qsa_rungs = sorted(set(qsa_rungs + [int(os.environ.get(
            "FLASHNEXT_PREFILL_MAX_S", "2048"))]))
        prefill_k = int(os.environ.get("FLASHNEXT_PREFILL_K", "0") or 0)
        prefill_max_s = int(os.environ.get("FLASHNEXT_PREFILL_MAX_S", "2048"))
        prep: dict[int, HostPrep] = {}
        host_layers: dict[int, HostLayer] = {}
        host_fronts: dict[int, object] = {}
        if pure_assets or connected_assets or gdn_assets or qsa_assets:
            from export_flashnext_gdn_fuse import HostFront
            if os.environ.get("FLASHNEXT_FRONT", "cpu") == "mlx":
                from export_flashnext_gdn_fuse import MlxFront as HostFront

            ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
            spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
            if os.environ.get("FLASHNEXT_COREAI_DEBUG", "0") == "1":
                spec = spec.with_debug(enabled=True)
            t0 = time.perf_counter()

            async def _load_fn(path: Path, label: str):
                if os.environ.get("FLASHNEXT_ANE_WORKER", "0") == "1":
                    from runtime.coreai_worker import WorkerAIModel
                    m = await WorkerAIModel.load(path)
                    return m, m.load_function("main")
                last = None
                for attempt in range(2):
                    try:
                        m = await AIModel.load(str(path), specialization_options=spec)
                        fn = m.load_function("main")
                        return m, fn
                    except Exception as exc:  # noqa: BLE001
                        last = exc
                        print(f"    retry {attempt + 1} {label}: {exc}", flush=True)
                        await asyncio.sleep(0.5 * (attempt + 1))
                raise last

            if use_mil_qsa and qsa_assets:
                # MIL int8 QSA replaces the folded Core AI qsa_step, 3.1x on the
                # layer at m=256 and 2.2x at m=2048. Same exclusivity rule as
                # the GDN side: one path or the other, never both resident.
                from runtime.mil_qsa_backend import MilQsaLayer
                from flashnext_mil_qsa_layer import _Ref as _QRef
                t_mq = time.perf_counter()
                mq_rungs = sorted(set(qsa_rungs))
                for i in sorted(qsa_assets):
                    lw = layer_w(i)
                    try:
                        qd = FlashNextQSADecode(max_s=max(mq_rungs)).eval().half()
                        qd.load_from_layer(lw)
                        mil_qsa[i] = MilQsaLayer(i, lw, _QRef(lw), qd, mq_rungs,
                                                 k=max(1, spec_k))
                    except Exception as exc:  # noqa: BLE001
                        print(f"  MIL QSA L{i} failed ({exc}); falling back to Core AI",
                              flush=True)
                        mil_qsa.clear()
                        break
                if mil_qsa:
                    print(f"  MIL int8 QSA: {len(mil_qsa)} layers, rungs "
                          f"{mq_rungs} in {time.perf_counter() - t_mq:.1f}s", flush=True)
                    qsa_assets.clear()
            for i, pq in sorted(qsa_assets.items()):
                print(f"  loading ANE QSA L{i}…", flush=True)
                try:
                    m_q, fn_q = await _load_fn(pq, f"QSA L{i}")
                except Exception as exc:  # noqa: BLE001
                    if os.environ.get("FLASHNEXT_STRICT_LOAD", "0") == "1":
                        raise RuntimeError(f"Required QSA L{i} failed to load") from exc
                    print(f"  QSA L{i} failed ({exc}); numpy fallback", flush=True)
                    continue
                keep.append(m_q)
                fn_qsa[i] = fn_q

            if use_mil:
                # MIL int8 GDN replaces pure_step; they are never both resident
                # (ANE ceiling ~80 programs). 1.60x on the layer, rel 0.027.
                from runtime.mil_gdn_backend import MilGdnLayer
                from flashnext_multitoken_step import MultiTokenStep as _MTS
                t_mil = time.perf_counter()
                for i in sorted(pure_assets):
                    lw = layer_w(i)
                    try:
                        mil_gdn[i] = MilGdnLayer(
                            i, lw, _MTS(lw, 1).eval().half(),
                            k=max(1, spec_k),
                            prefill_k=(prefill_mil_k
                                       if prefill_mil_k > max(1, spec_k) else 0))
                    except Exception as exc:  # noqa: BLE001
                        print(f"  MIL L{i} failed ({exc}); falling back to Core AI",
                              flush=True)
                        mil_gdn.clear()
                        break
                    if (i + 1) % 8 == 0:
                        print(f"    MIL GDN {len(mil_gdn)}/{len(pure_assets)} "
                              f"{time.perf_counter() - t_mil:.0f}s", flush=True)
                if mil_gdn:
                    print(f"  MIL int8 GDN: {len(mil_gdn)} layers in "
                          f"{time.perf_counter() - t_mil:.1f}s", flush=True)
                if mil_gdn and mil_qsa and prefill_mil_k > max(1, spec_k):
                    t_pf = time.perf_counter()
                    try:
                        # The GDN layers already carry the wide unroll as a
                        # second procedure of the same program, so there is
                        # nothing more to build and no weights to duplicate.
                        for i in sorted(mil_gdn):
                            mil_gdn_pf[i] = mil_gdn[i]
                        # Prefill only ever sees the widest rung, so it needs
                        # one program a layer plus the front, not the ladder.
                        for i in sorted(mil_qsa):
                            lw = layer_w(i)
                            qd = FlashNextQSADecode(
                                max_s=max(mq_rungs)).eval().half()
                            qd.load_from_layer(lw)
                            mil_qsa_pf[i] = MilQsaLayer(
                                i, lw, _QRef(lw), qd, [max(mq_rungs)],
                                k=prefill_mil_k)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  MIL prefill k={prefill_mil_k} failed ({exc}); "
                              f"prompt walks the decode graphs", flush=True)
                        mil_gdn_pf.clear()
                        mil_qsa_pf.clear()
                    if mil_gdn_pf:
                        print(f"  MIL prefill k={prefill_mil_k}: "
                              f"{len(mil_gdn_pf)} GDN (a second procedure of "
                              f"the decode program) + {len(mil_qsa_pf)} QSA "
                              f"in {time.perf_counter() - t_pf:.1f}s", flush=True)
                if mil_gdn:
                    pure_assets.clear()
            for i, pp in sorted(list(pure_assets.items())):
                print(f"  loading ANE pure_step L{i} (mix+GDN+mlp mix)…", flush=True)
                try:
                    m_p, fn_p = await _load_fn(pp, f"pure L{i}")
                except Exception as exc:  # noqa: BLE001
                    if os.environ.get("FLASHNEXT_STRICT_LOAD", "0") == "1":
                        raise RuntimeError(f"Required pure_step L{i} failed to load") from exc
                    print(
                        f"  pure_step L{i} failed ({exc}); falling back to connected GDN",
                        flush=True,
                    )
                    pure_assets.pop(i, None)
                    pc = _connected_gdn_path(i)
                    if pc.is_dir():
                        connected_assets[i] = pc
                    continue
                keep.append(m_p)
                fn_pure[i] = fn_p
                layer_w(i)
                for key in list(layer_cache[i].tensors):
                    if key.startswith("linear_attn."):
                        del layer_cache[i].tensors[key]
            for i, pc in sorted(list(connected_assets.items())):
                print(f"  loading ANE connected GDN L{i} (1-submit)…", flush=True)
                try:
                    m_c, fn_c = await _load_fn(pc, f"connected L{i}")
                except Exception as exc:  # noqa: BLE001
                    if os.environ.get("FLASHNEXT_STRICT_LOAD", "0") == "1":
                        raise RuntimeError(f"Required connected GDN L{i} failed to load") from exc
                    print(f"  connected L{i} failed ({exc}); numpy fallback", flush=True)
                    connected_assets.pop(i, None)
                    continue
                keep.append(m_c)
                fn_connected[i] = fn_c
                layer_w(i)
                for key in list(layer_cache[i].tensors):
                    if key.startswith("linear_attn."):
                        del layer_cache[i].tensors[key]
            for i, (pf, pg) in sorted(gdn_assets.items()):
                print(
                    f"  loading ANE GDN L{i} "
                    f"{'gdn-only' if host_front_decode else 'front+gdn'}…",
                    flush=True,
                )
                if not host_front_decode:
                    m_f = await AIModel.load(str(pf), specialization_options=spec)
                    keep.append(m_f)
                    fn_front[i] = m_f.load_function("main")
                m_g = await AIModel.load(str(pg), specialization_options=spec)
                keep.append(m_g)
                fn_gdn[i] = m_g.load_function("main")
                prep[i] = HostPrep(layer_w(i), seq=seq)
                if host_front_decode:
                    host_fronts[i] = HostFront(layer_w(i))
                for key in list(layer_cache[i].tensors):
                    if key.startswith("linear_attn."):
                        del layer_cache[i].tensors[key]
            print(
                f"  ANE loaded {len(fn_pure)} pure_step + {len(fn_connected)} connected + "
                f"{len(fn_gdn)} GDN + {len(fn_qsa)} QSA in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

            # Multi-token graphs for chunked prefill. Same input/output names
            # and shapes as pure_step, so SurfacePool feeds them unchanged; the
            # only additions are k real token slots and a "shared" output.
            if prefill_k:
                t_m = time.perf_counter()
                gdn_ids = sorted(fn_pure) or sorted(mil_gdn)
                qsa_ids = sorted(fn_qsa) or sorted(mil_qsa)
                for i in gdn_ids:
                    pm = OUT_DIR / f"flashnext_multitoken_step_k{prefill_k}_L{i}.aimodel"
                    if not pm.is_dir():
                        print(f"  prefill: missing {pm.name}; falling back to serial",
                              flush=True)
                        fn_multi.clear()
                        break
                    mm, fm = await _load_fn(pm, f"multi L{i}")
                    keep.append(mm)
                    fn_multi[i] = fm
                if fn_multi:
                    # Folded QSA (mixers inside the graph) when exported: it
                    # costs +0.55 ms on the ANE and removes two host mixer
                    # passes and two recombines per layer per chunk.
                    for i in qsa_ids:
                        # Folded FP16 mixers change greedy near-ties (including
                        # the required "The 2016–17" prefix). Keep bare QSA as
                        # the correctness default; folding is experimental.
                        if os.environ.get("FLASHNEXT_QSA_FOLDED", "0") == "0":
                            break
                        pf = (OUT_DIR
                              / f"flashnext_qsa_step_k{prefill_k}_L{i}_m{prefill_max_s}.aimodel")
                        if not pf.is_dir():
                            fn_qsa_step.clear()
                            break
                        mm, fm = await _load_fn(pf, f"qsaStep L{i}")
                        keep.append(mm)
                        fn_qsa_step[i] = fm
                if fn_qsa_step:
                    # A narrow rung matters more than the token count: the
                    # graph's K/V inputs are max_S wide and wrap_ndarray copies
                    # them per layer per submit, so a 2048 rung moves 4 MB/layer
                    # whatever the real context is. Pick the smallest that fits.
                    # ANE resources run out around ~80 resident models, so
                    # load only what each path uses: k=1 at every rung for
                    # decode, and the chunk width only at the widest rung.
                    # (prefill_max_s, prefill_k) is already loaded above.
                    fn_qsa_rung[(prefill_max_s, prefill_k)] = fn_qsa_step
                    for rung in qsa_rungs:
                        for kk in (1,):
                            d = {}
                            for i in qsa_ids:
                                pr = (OUT_DIR
                                      / f"flashnext_qsa_step_k{kk}_L{i}_m{rung}.aimodel")
                                if not pr.is_dir():
                                    d.clear()
                                    break
                                mm, fm = await _load_fn(pr, f"qsa k{kk} m{rung} L{i}")
                                keep.append(mm)
                                d[i] = fm
                            if d:
                                fn_qsa_rung[(rung, kk)] = d
                    if fn_qsa_rung:
                        print("  QSA rungs: " + ", ".join(
                            f"m{r}/k{k}" for (r, k) in sorted(fn_qsa_rung)), flush=True)
                if fn_multi and not fn_qsa_step:
                    for i in qsa_ids:
                        pm = (OUT_DIR
                              / f"flashnext_multitoken_qsa_L{i}_m{prefill_max_s}.aimodel")
                        if not pm.is_dir():
                            print(f"  prefill: missing {pm.name}; falling back to serial",
                                  flush=True)
                            fn_multi.clear()
                            fn_qsa_multi.clear()
                            break
                        mm, fm = await _load_fn(pm, f"multiQSA L{i}")
                        keep.append(mm)
                        fn_qsa_multi[i] = fm
                if fn_multi:
                    print(f"  prefill k={prefill_k}: {len(fn_multi)} GDN + "
                          f"{len(fn_qsa_step) or len(fn_qsa_multi)} QSA "
                          f"({'folded' if fn_qsa_step else 'bare'}, "
                          f"max_S={prefill_max_s}) "
                          f"in {time.perf_counter() - t_m:.1f}s", flush=True)
        t_h = time.perf_counter()
        store = expert_f16_store()
        n_warm = 0
        warm_env = os.environ.get("FLASHNEXT_EXPERT_WARM", "1").strip().lower()
        if warm_env not in ("0", "false", "no"):
            n_warm = store.warm_ram_from_disk()
        t_warm = time.perf_counter() - t_h
        for i in range(n_layers):
            if i not in host_layers:
                host_layers[i] = HostLayer(layer_w(i), seq=seq, device=moe_dev, store=store)
                # The ANE assets already own attention weights. HostPrep and
                # HostFront retain the arrays they use; don't also retain
                # the loader's large original projection matrices.
                if i in fn_gdn or i in fn_qsa or i in fn_connected or i in fn_pure:
                    tensors = layer_cache[i].tensors
                    drop = ("linear_attn.", "self_attn.")
                    if i in fn_pure:
                        drop = drop + ("attn_hyper_connection.", "mlp_hyper_connection.")
                    for key in list(tensors):
                        # The QSA indexer weights live under self_attn. but are
                        # host-side (chunked prefill selects keys with them), so
                        # they must survive the drop. ~6.5 MB per QSA layer.
                        if key.startswith(drop) and ".indexer." not in key:
                            del tensors[key]
                if moe_mode == "mlxresident" and (i + 1) % 8 == 0:
                    print(f"    resident GPU MoE {i+1}/{n_layers}: {store.ram_bytes()/1e9:.2f} GB", flush=True)
        ws = 0
        if store.mlx4 is not None:
            for hl in host_layers.values():
                ws += hl.moe.gu_f16.nbytes + hl.moe.dn_f16.nbytes
            gu32, dn32 = HostMoE._fp32_scratch()
            gu32.fill(0)
            dn32.fill(0)
        else:
            for hl in host_layers.values():
                ws += hl.moe.gu_buf.nbytes + hl.moe.dn_buf.nbytes
        print(
            f"  host mixers+MoE ready {len(host_layers)} layers  "
            f"{time.perf_counter() - t_h:.1f}s  device={'MLX GPU' if moe_mode == 'mlxresident' else moe_dev}  "
            f"experts bits={store.bits}  mode={_moe_decode_mode()}  "
            f"{('resident mlx4' if moe_mode == 'mlxresident' else 'mlx4 ' + str(store.mlx4.path) if store.mlx4 is not None else store.ART)}  "
            f"warmed={n_warm} in {t_warm:.1f}s  "
            f"ram={store.ram_bytes() / 1e9:.2f} GB  packed_ws={ws / 1e9:.2f} GB fp16",
            flush=True,
        )

        # Persistent host numpy + ping-ponged conv/ssm NDArrays (runtime/coreai_surfaces.py).
        pool = SurfacePool(seq)
        t_nd = time.perf_counter()
        for i in fn_pure:
            pool.add_gdn(i)
        for i in fn_connected:
            pool.add_gdn(i)
        for i in fn_gdn:
            pool.add_gdn(i, prep[i])
        for i in fn_qsa:
            pool.add_qsa(i)
        print(
            f"  persistent ANE surfaces GDN={len(pool.gdn_layers)} "
            f"QSA={len(pool.qsa_layers)}  pure={len(fn_pure)}  "
            f"connected={len(fn_connected)}  "
            f"{time.perf_counter() - t_nd:.1f}s",
            flush=True,
        )
        ple_rows = None
        ple_layers = {}
        if ple_enabled:
            from runtime.flashnext_ngram import NgramRows, CpuPLE
            ple_rows = NgramRows(BASE)
            ple_layers = {int(i)-1: CpuPLE(ple_rows, int(i)-1) for i in cfg["ple_layer_ids"]}
            print(f"  PLE: {ple_rows.total} rows x {ple_rows.dim}, pread selected rows only", flush=True)

        def apply_moe(i, attn0, hyper, inj, timers):
            hl = host_layers[i]
            t0 = time.perf_counter()
            post = host_recombine(attn0, hyper, inj)
            mixed2, hyper2, inj2 = host_gated_residual_cached(post, hl.mlp)
            t1 = time.perf_counter()
            mixed_bsh = _bc1s_to_bsh(mixed2)[:, :1, :]
            moe = hl.moe.apply(mixed_bsh)
            t2 = time.perf_counter()
            hc = host_recombine(_bsh_to_bc1s(moe), hyper2, inj2)
            timers["mlp_mix"] += t1 - t0
            timers["moe"] += t2 - t1
            timers["moe_route"] += hl.moe.last_ms["route"]
            timers["moe_gather"] += hl.moe.last_ms["gather"]
            timers["moe_gemm"] += hl.moe.last_ms["gemm"]
            timers["moe_slot_hit"] += getattr(hl.moe, "last_reuse", 0)
            timers["moe_slot_copy"] += getattr(hl.moe, "last_copy", K_PIN)
            timers["recombine"] += time.perf_counter() - t2
            return _bc1s_to_bsh(hc)[:, :1, :]

        def apply_moe_premix(i, mixed0, hyper2, inj2, timers):
            """MoE + recombine when mixers already ran in the ANE pure_step graph."""
            hl = host_layers[i]
            t0 = time.perf_counter()
            mixed_bsh = _bc1s_to_bsh(mixed0)[:, :1, :]
            moe = hl.moe.apply(mixed_bsh)
            t1 = time.perf_counter()
            hc = host_recombine(_bsh_to_bc1s(moe), hyper2, inj2)
            timers["moe"] += t1 - t0
            timers["moe_route"] += hl.moe.last_ms["route"]
            timers["moe_gather"] += hl.moe.last_ms["gather"]
            timers["moe_gemm"] += hl.moe.last_ms["gemm"]
            timers["moe_slot_hit"] += getattr(hl.moe, "last_reuse", 0)
            timers["moe_slot_copy"] += getattr(hl.moe, "last_copy", K_PIN)
            timers["recombine"] += time.perf_counter() - t1
            return _bc1s_to_bsh(hc)[:, :1, :]

        # Shared by chunked prefill and decode: the indexer state must carry
        # across the boundary, and the 2048-wide QSA buffers are reused.
        prefill_timers: list = []
        qsa_idx: dict[int, object] = {}
        qsa_idx_state: dict[int, object] = {}
        qsa_bufs_multi: dict[str, np.ndarray] = {}
        _qsa_any = fn_qsa_step or fn_qsa_multi or mil_qsa
        if _qsa_any:
            from runtime.flashnext_indexer import QSAIndexer, IndexerState, clip_to_budget
            kv_c_m = QSA_HKV * QSA_HD
            for i in _qsa_any:
                qsa_idx[i] = QSAIndexer(layer_w(i), cfg)
                qsa_idx_state[i] = IndexerState()
            # Per-layer K/V surfaces so the sub-budget case can be updated
            # incrementally: only one key changes per decode step, but a full
            # re-gather copies the whole prefix (~1.2 ms/layer, 14 ms/token).
            qsa_kv_layer = {
                i: [np.zeros((1, kv_c_m, 1, prefill_max_s), np.float16),
                    np.zeros((1, kv_c_m, 1, prefill_max_s), np.float16), 0]
                for i in _qsa_any
            }
            qsa_rung_bufs: dict[int, dict] = {}
            mil_qsa_ms = {"mix": 0.0, "index": 0.0, "feed": 0.0, "ane": 0.0,
                          "moe": 0.0, "recombine": 0.0}
            mil_qsa_inv = (1.0 / (10_000_000.0 ** (
                np.arange(0, QSA_ROTARY, 2, dtype=np.float32)
                / np.float32(QSA_ROTARY)))).astype(np.float32)

            def _bufs_for(m: int) -> dict:
                b = qsa_rung_bufs.get(m)
                if b is None:
                    b = {
                        "h": np.zeros((1, H, 1, seq), np.float16),
                        "k": np.zeros((1, kv_c_m, 1, m), np.float16),
                        "v": np.zeros((1, kv_c_m, 1, m), np.float16),
                        "cos": np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16),
                        "sin": np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16),
                        "mask": np.full((1, m + seq, 1, seq), QSA_MASK, np.float16),
                        "inv": (1.0 / (10_000_000.0 ** (
                            np.arange(0, QSA_ROTARY, 2, dtype=np.float32)
                            / np.float32(QSA_ROTARY)))).astype(np.float32),
                        "layer_kv": {},
                    }
                    qsa_rung_bufs[m] = b
                return b

            qsa_bufs_multi = {
                "layer_kv": {},
                "h": np.zeros((1, H, 1, seq), np.float16),
                "k": np.zeros((1, kv_c_m, 1, prefill_max_s), np.float16),
                "v": np.zeros((1, kv_c_m, 1, prefill_max_s), np.float16),
                "cos": np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16),
                "sin": np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16),
                "k_scratch": np.zeros((1, kv_c_m, 1, prefill_max_s), np.float16),
                "v_scratch": np.zeros((1, kv_c_m, 1, prefill_max_s), np.float16),
                "mask": np.full((1, prefill_max_s + seq, 1, seq), QSA_MASK, np.float16),
                "inv": (1.0 / (10_000_000.0 ** (
                    np.arange(0, QSA_ROTARY, 2, dtype=np.float32)
                    / np.float32(QSA_ROTARY)))).astype(np.float32),
            }

        async def qsa_multi_attn(i: int, mixed_bsh: np.ndarray, n: int) -> np.ndarray:
            """One QSA layer over n token slots against the 2048-wide graph.

            Used by prefill with n = k and by decode with n = 1, which is what
            lifts the 32-token context cap: the single-token graphs are baked at
            max_S=32, these are not. Keys past the indexer budget come from the
            selector; below it the whole prefix is fed.
            """
            m = prefill_max_s
            kv_c = QSA_HKV * QSA_HD
            b = qsa_bufs_multi
            cache = attn_state[i]
            off = int(cache.offset)
            sel = qsa_idx[i].update_and_select(
                np.asarray(mixed_bsh, np.float32).reshape(-1, H), off, qsa_idx_state[i])
            if sel is None and off > m:
                raise RuntimeError(
                    f"QSA dense prefix ({off}) exceeds graph width {m}; "
                    f"indexer must select past the budget")
            keep = (np.asarray(sel, np.int64) if sel is not None
                    else np.arange(off, dtype=np.int64))
            keep = clip_to_budget(keep, off, m)
            nsel = int(keep.size)
            lkv = b["layer_kv"].get(i)
            if lkv is None:
                lkv = [np.zeros((1, kv_c, 1, m), np.float16),
                       np.zeros((1, kv_c, 1, m), np.float16), 0]
                b["layer_kv"][i] = lkv
            lk, lv, filled = lkv
            if sel is None and filled <= off and nsel == off:
                # Sub-budget: the selection is the whole prefix in order, so
                # only the keys added since last time need copying.
                if off > filled:
                    idx = np.arange(filled, off, dtype=np.int64)
                    lk[0, :, 0, filled:off] = cache.keys[:, idx].transpose(0, 2, 1).reshape(kv_c, idx.size)
                    lv[0, :, 0, filled:off] = cache.values[:, idx].transpose(0, 2, 1).reshape(kv_c, idx.size)
                    lkv[2] = off
                b["k"], b["v"] = lk, lv
            else:
                b["k"] = b["k_scratch"]
                b["v"] = b["v_scratch"]
                b["k"].fill(0)
                b["v"].fill(0)
                lkv[2] = 0
                if nsel:
                    b["k"][0, :, 0, :nsel] = cache.keys[:, keep].transpose(0, 2, 1).reshape(kv_c, nsel)
                    b["v"][0, :, 0, :nsel] = cache.values[:, keep].transpose(0, 2, 1).reshape(kv_c, nsel)
            b["h"].fill(0)
            bc = _bsh_to_bc1s(np.asarray(mixed_bsh, np.float32))
            b["h"][..., :n] = np.asarray(bc[..., :n], np.float16)
            b["cos"].fill(0)
            b["sin"].fill(0)
            if n:
                pos = (np.arange(n, dtype=np.float32) + np.float32(off))[:, None] * b["inv"][None, :]
                b["cos"][0, :, 0, :n] = np.cos(pos).T.astype(np.float16)
                b["sin"][0, :, 0, :n] = np.sin(pos).T.astype(np.float16)
            b["mask"].fill(QSA_MASK)
            if nsel:
                b["mask"][:, :nsel, :, :n] = 0
            for slot in range(n):
                b["mask"][:, m:m + slot + 1, :, slot] = 0
            feed = {"h": wrap_ndarray(b["h"]), "k_cache": wrap_ndarray(b["k"]),
                    "v_cache": wrap_ndarray(b["v"]), "cos": wrap_ndarray(b["cos"]),
                    "sin": wrap_ndarray(b["sin"]), "mask": wrap_ndarray(b["mask"])}
            out_q = await fn_qsa_multi[i](feed)
            attn = np.array(out_q.pop("out").numpy(), np.float32, copy=True)[..., :n]
            nk = np.array(out_q.pop("new_k").numpy(), np.float32, copy=True)[0, :, 0, :n]
            nv = np.array(out_q.pop("new_v").numpy(), np.float32, copy=True)[0, :, 0, :n]
            out_q.clear()
            del out_q
            cache.keys[:, off:off + n] = nk.reshape(QSA_HKV, QSA_HD, n).transpose(0, 2, 1)
            cache.values[:, off:off + n] = nv.reshape(QSA_HKV, QSA_HD, n).transpose(0, 2, 1)
            cache.offset = off + n
            return attn

        def _qsa_kv_feed(i: int, mixed_bsh: np.ndarray, n: int, m: int, b: dict):
            """Indexer selection + the five KV/RoPE/mask inputs. Returns offset."""
            kv_c = QSA_HKV * QSA_HD
            cache = attn_state[i]
            off = int(cache.offset)
            sel = qsa_idx[i].update_and_select(
                np.asarray(mixed_bsh, np.float32).reshape(-1, H), off, qsa_idx_state[i])
            if sel is None and off > m:
                raise RuntimeError(
                    f"QSA dense prefix ({off}) exceeds graph width {m}; "
                    f"indexer must select past the budget")
            keep = (np.asarray(sel, np.int64) if sel is not None
                    else np.arange(off, dtype=np.int64))
            keep = clip_to_budget(keep, off, m)
            nsel = int(keep.size)
            b["k"].fill(0)
            b["v"].fill(0)
            if nsel:
                b["k"][0, :, 0, :nsel] = cache.keys[:, keep].transpose(0, 2, 1).reshape(kv_c, nsel)
                b["v"][0, :, 0, :nsel] = cache.values[:, keep].transpose(0, 2, 1).reshape(kv_c, nsel)
            b["cos"].fill(0)
            b["sin"].fill(0)
            if n:
                pos = (np.arange(n, dtype=np.float32) + np.float32(off))[:, None] * b["inv"][None, :]
                b["cos"][0, :, 0, :n] = np.cos(pos).T.astype(np.float16)
                b["sin"][0, :, 0, :n] = np.sin(pos).T.astype(np.float16)
            b["mask"].fill(QSA_MASK)
            if nsel:
                b["mask"][:, :nsel, :, :n] = 0
            for slot in range(n):
                b["mask"][:, m:m + slot + 1, :, slot] = 0
            return off

        def _qsa_store_kv(i: int, out_q, n: int, off: int) -> None:
            cache = attn_state[i]
            nk = np.array(out_q.pop("new_k").numpy(), np.float32, copy=True)[0, :, 0, :n]
            nv = np.array(out_q.pop("new_v").numpy(), np.float32, copy=True)[0, :, 0, :n]
            cache.keys[:, off:off + n] = nk.reshape(QSA_HKV, QSA_HD, n).transpose(0, 2, 1)
            cache.values[:, off:off + n] = nv.reshape(QSA_HKV, QSA_HD, n).transpose(0, 2, 1)
            cache.offset = off + n

        async def qsa_step_layer(i: int, hidden: np.ndarray, n: int,
                                 bc1s: bool = False) -> np.ndarray:
            """A QSA layer whose recombine and MLP mix live in the ANE graph.

            The host still runs the attention mix, because the indexer selects
            keys from it and that has to happen before the ANE call. What the
            fold removes is the recombine and the second mixer — two numpy
            passes over a 10240-wide stream per layer per chunk.
            """
            hl = host_layers[i]
            need = int(attn_state[i].offset) + n
            m = next((r for r in qsa_rungs
                      if r >= need and (r, n if n == 1 else prefill_k) in fn_qsa_rung),
                     prefill_max_s)
            fns = fn_qsa_rung.get((m, n if n == 1 else prefill_k)) or fn_qsa_step
            b = _bufs_for(m)
            pt = prefill_timers[-1] if prefill_timers else None
            _t0 = time.perf_counter()
            x_bc = (np.asarray(hidden, np.float32) if bc1s
                    else _bsh_to_bc1s(np.asarray(hidden, np.float32)))
            mixed, _, _ = host_gated_residual_cached(x_bc, hl.attn)
            _t1 = time.perf_counter()
            if pt is not None:
                pt["mixers"] += _t1 - _t0
            off = _qsa_kv_feed(i, _bc1s_to_bsh(mixed), n, m, b)
            _t2 = time.perf_counter()
            if pt is not None:
                pt["kvfeed"] += _t2 - _t1
            xb = np.zeros((1, HC_W, 1, seq), np.float16)
            xb[..., :n] = np.asarray(x_bc[..., :n], np.float16)
            feed = {"x": wrap_ndarray(xb), "k_cache": wrap_ndarray(b["k"]),
                    "v_cache": wrap_ndarray(b["v"]), "cos": wrap_ndarray(b["cos"]),
                    "sin": wrap_ndarray(b["sin"]), "mask": wrap_ndarray(b["mask"])}
            _t3 = time.perf_counter()
            out_q = await fns[i](feed)
            _qsa_store_kv(i, out_q, n, off)
            _t4 = time.perf_counter()
            if pt is not None:
                pt["ane_qsa"] += _t4 - _t3
            mixed2 = np.array(out_q.pop("mixed").numpy(), np.float32, copy=True)[..., :n]
            hyper2 = np.array(out_q.pop("hyper").numpy(), np.float32, copy=True)[..., :n]
            inj2 = np.array(out_q.pop("inj").numpy(), np.float32, copy=True)[..., :n]
            shared = np.array(out_q.pop("shared").numpy(), np.float32, copy=True)[..., :n]
            out_q.clear()
            del out_q
            mixed_bsh2 = _bc1s_to_bsh(mixed2)
            _t5 = time.perf_counter()
            inds, sc = hl.moe._route(np.asarray(mixed_bsh2, np.float32).reshape(-1, H))
            routed = hl.moe._resident.routed_multi(mixed_bsh2, inds, sc)
            _t6 = time.perf_counter()
            if pt is not None:
                pt["take"] += _t5 - _t4
                pt["moe"] += _t6 - _t5
            y = routed + _bc1s_to_bsh(shared)
            hc = host_recombine(_bsh_to_bc1s(y), hyper2, inj2)
            out = hc if bc1s else _bc1s_to_bsh(hc)
            if pt is not None:
                pt["recombine"] += time.perf_counter() - _t6
            return out

        def mil_qsa_step_layer(i: int, hidden: np.ndarray, bc1s: bool = False,
                               n: int = 1, commit: bool = True, until: str = "full"):
            """One QSA layer on the MIL int8 backend over n live token slots.

            The host still runs the attention mixer, because the indexer picks
            keys from it and that has to happen before the ANE call; the graph
            runs its own copy. What the graph folds in is everything after the
            selection: attention, recombine, the MLP mixer and the shared
            expert. `commit=False` leaves the cache offset for the caller to
            rewind, which is how a partly accepted speculative block unwinds.
            `until="ane"` returns (mixed, hyper, inj, shared) BC1S so the
            caller can overlap GPU MoE with the next ANE submit.
            """
            hl = host_layers[i]
            lay = _active["qsa"][i]
            cache = attn_state[i]
            tq = mil_qsa_ms
            _t0 = time.perf_counter()
            off = int(cache.offset)
            m = lay.rung_for(off + n)
            x_bc = (np.asarray(hidden, np.float32) if bc1s
                    else _bsh_to_bc1s(np.asarray(hidden, np.float32)))
            xb = np.zeros((HC_W, seq), np.float16)
            xb[:, :n] = np.asarray(x_bc[0, :, 0, :n], np.float16)
            a_mixed, a_inj, a_qk = lay.front(xb, n)
            _t1 = time.perf_counter()
            tq["mix"] += _t1 - _t0
            if _idx_host:
                # A/B for long context: project the indexer's q and k on the
                # host in fp32 from the ANE's mixed hidden, instead of taking
                # the int8 projection the front program already computed.
                sel = qsa_idx[i].update_and_select(
                    np.ascontiguousarray(a_mixed.T),
                    off, qsa_idx_state[i])
            else:
                sel = qsa_idx[i].update_and_select(
                    None, off, qsa_idx_state[i],
                    projected=qsa_idx[i].split_qk(a_qk))
            keep = (np.asarray(sel, np.int64) if sel is not None
                    else np.arange(off, dtype=np.int64))
            keep = clip_to_budget(keep, off, m)
            nsel = int(keep.size)
            _t2 = time.perf_counter()
            tq["index"] += _t2 - _t1
            kv_c = QSA_HKV * QSA_HD
            ks = (cache.keys[:, keep].transpose(0, 2, 1).reshape(kv_c, nsel)
                  if nsel else np.zeros((kv_c, 0), np.float16))
            vs = (cache.values[:, keep].transpose(0, 2, 1).reshape(kv_c, nsel)
                  if nsel else np.zeros((kv_c, 0), np.float16))
            pos = (np.arange(n, dtype=np.float32) + np.float32(off))[:, None] \
                * mil_qsa_inv[None, :]
            cos_b = np.zeros((QSA_ROTARY // 2, seq), np.float16)
            sin_b = np.zeros((QSA_ROTARY // 2, seq), np.float16)
            cos_b[:, :n] = np.cos(pos).T.astype(np.float16)
            sin_b[:, :n] = np.sin(pos).T.astype(np.float16)
            _t3 = time.perf_counter()
            tq["feed"] += _t3 - _t2
            m_mix, m_hyp, m_inj, m_sh, nk, nv = lay(
                xb, np.asarray(ks, np.float16), np.asarray(vs, np.float16),
                cos_b, sin_b, nsel, m, n=n,
                mixed=np.asarray(a_mixed, np.float16),
                inj=np.asarray(a_inj, np.float16))
            _t4 = time.perf_counter()
            tq["ane"] += _t4 - _t3
            cache.keys[:, off:off + n] = nk.reshape(QSA_HKV, QSA_HD, n).transpose(0, 2, 1)
            cache.values[:, off:off + n] = nv.reshape(QSA_HKV, QSA_HD, n).transpose(0, 2, 1)
            cache.offset = off + n
            if until == "ane":
                return m_mix, m_hyp, m_inj, m_sh
            mixed_bsh2 = _bc1s_to_bsh(m_mix)
            inds, sc = hl.moe._route(np.asarray(mixed_bsh2, np.float32).reshape(-1, H))
            routed = hl.moe._resident.routed_multi(mixed_bsh2, inds, sc)
            y = routed + _bc1s_to_bsh(m_sh)
            _t5 = time.perf_counter()
            tq["moe"] += _t5 - _t4
            hc = host_recombine(_bsh_to_bc1s(y), m_hyp, m_inj)
            tq["recombine"] += time.perf_counter() - _t5
            return hc if bc1s else _bc1s_to_bsh(hc)

        async def prefill_chunked(ids: list[int], k: int) -> int:
            """Run prompt tokens through the k-token graphs, k slots per submit.

            Replaces the serial prefix prefill, which walks the decode loop once
            per prompt token at ~220 ms each. The multi-token and single-token
            graphs share the recurrent state, conv cache and KV cache formats,
            so decode picks up from here unchanged.

            Only whole chunks of k go through here. The recurrence is unrolled
            to exactly k steps, so a short chunk would still run k of them and
            advance the state over zero-padding. The remainder stays serial.
            Returns the number of tokens consumed.
            """
            full = (len(ids) // k) * k
            if full == 0:
                return 0
            pt = {"ane_gdn": 0.0, "ane_qsa": 0.0, "take": 0.0, "route": 0.0,
                  "moe": 0.0, "recombine": 0.0, "mixers": 0.0, "indexer": 0.0,
                  "kvfeed": 0.0, "embed": 0.0, "ple": 0.0}
            prefill_timers.append(pt)
            from runtime.flashnext_indexer import QSAIndexer, IndexerState

            idx = {}
            idx_state = {}
            for i in fn_qsa_multi:
                idx[i] = QSAIndexer(layer_w(i), cfg)
                idx_state[i] = IndexerState()
            kv_c = QSA_HKV * QSA_HD
            m = prefill_max_s
            h_buf = np.zeros((1, H, 1, seq), np.float16)
            kc = np.zeros((1, kv_c, 1, m), np.float16)
            vc = np.zeros((1, kv_c, 1, m), np.float16)
            cos_b = np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16)
            sin_b = np.zeros((1, QSA_ROTARY // 2, 1, seq), np.float16)
            mask_b = np.full((1, m + seq, 1, seq), QSA_MASK, np.float16)
            inv = 1.0 / (10_000_000.0 ** (
                np.arange(0, QSA_ROTARY, 2, dtype=np.float32) / np.float32(QSA_ROTARY)))

            pos = 0
            t_pf0 = time.perf_counter()
            n_chunks = full // k
            chunk_i = 0
            while pos < full:
                chunk = ids[pos:pos + k]
                n = len(chunk)
                _t0 = time.perf_counter()
                hidden = real_embedding_hidden(loader, chunk)
                hidden_bc = _bsh_to_bc1s(np.asarray(hidden, np.float32))
                pt["embed"] += time.perf_counter() - _t0
                pool.x_hc.fill(0)
                for i in range(n_layers):
                    if i in ple_layers:
                        _t0 = time.perf_counter()
                        hidden = _bc1s_to_bsh(hidden_bc)
                        for j in range(n):
                            hidden[:, j:j + 1, :] = ple_layers[i].step(
                                hidden[:, j:j + 1, :], chunk[j])
                        hidden_bc = _bsh_to_bc1s(np.asarray(hidden, np.float32))
                        pt["ple"] += time.perf_counter() - _t0
                    if i in fn_multi:
                        _t0 = time.perf_counter()
                        # slots n..seq-1 stay zero for the whole run, so the
                        # buffer does not need re-zeroing every layer
                        pool.x_hc[..., :n] = np.asarray(hidden_bc[..., :n], np.float16)
                        out = await fn_multi[i](pool.pure_feeds(i))
                        pool.accept_connected(i, out)
                        _t1 = time.perf_counter()
                        pt["ane_gdn"] += _t1 - _t0
                        mixed = np.array(out.pop("mixed").numpy(), np.float32, copy=True)[..., :n]
                        hyper = np.array(out.pop("hyper").numpy(), np.float32, copy=True)[..., :n]
                        inj = np.array(out.pop("inj").numpy(), np.float32, copy=True)[..., :n]
                        shared = np.array(out.pop("shared").numpy(), np.float32, copy=True)[..., :n]
                        out.clear()
                        del out
                        mixed_bsh = _bc1s_to_bsh(mixed)
                        _t2 = time.perf_counter()
                        pt["take"] += _t2 - _t1
                        hl = host_layers[i]
                        inds, sc = hl.moe._route(
                            np.asarray(mixed_bsh, np.float32).reshape(-1, H))
                        _t3 = time.perf_counter()
                        pt["route"] += _t3 - _t2
                        routed = hl.moe._resident.routed_multi(mixed_bsh, inds, sc)
                        _t4 = time.perf_counter()
                        pt["moe"] += _t4 - _t3
                        y = routed + _bc1s_to_bsh(shared)
                        hidden_bc = host_recombine(_bsh_to_bc1s(y), hyper, inj)
                        pt["recombine"] += time.perf_counter() - _t4
                    elif i in fn_qsa_step:
                        hidden_bc = await qsa_step_layer(i, hidden_bc, n, bc1s=True)
                    elif i in fn_qsa_multi:
                        mixed, hyper, inj = host_gated_residual_cached(
                            _bsh_to_bc1s(np.asarray(hidden, np.float32)), host_layers[i].attn)
                        mixed_bsh = _bc1s_to_bsh(mixed)
                        attn = await qsa_multi_attn(i, mixed_bsh, n)
                        post = host_recombine(attn, hyper, inj)
                        mixed2, hyper2, inj2 = host_gated_residual_cached(
                            post, host_layers[i].mlp)
                        mixed_bsh2 = _bc1s_to_bsh(mixed2)
                        hl = host_layers[i]
                        inds, sc = hl.moe._route(
                            np.asarray(mixed_bsh2, np.float32).reshape(-1, H))
                        routed = hl.moe._resident.routed_multi(mixed_bsh2, inds, sc)
                        xx = np.asarray(mixed_bsh2, np.float32).reshape(-1, H)
                        sg = HostPrep._silu(xx @ hl.moe.shared_gate.T)
                        sh = (sg * (xx @ hl.moe.shared_up.T)) @ hl.moe.shared_down.T
                        gate = 1.0 / (1.0 + np.exp(-(xx @ hl.moe.shared_sgate.T)))
                        y = routed + (sh * gate).reshape(routed.shape)
                        hc = host_recombine(_bsh_to_bc1s(y), hyper2, inj2)
                        hidden = _bc1s_to_bsh(hc)
                    else:
                        raise RuntimeError(f"prefill: layer {i} has no multi-token graph")
                pos += n
                chunk_i += 1
                if chunk_i % 2 == 0:
                    gc.collect()
                if n_chunks >= 8 and (
                        chunk_i == 1 or chunk_i == n_chunks
                        or chunk_i % max(1, n_chunks // 16) == 0):
                    elapsed = time.perf_counter() - t_pf0
                    print(f"    prefill {pos}/{full}  {elapsed:.1f}s  "
                          f"{pos / max(elapsed, 1e-9):.1f} tok/s  "
                          f"{elapsed / pos * 1e3:.1f} ms/token", flush=True)
            return full

        # Slots past the block width stay zero for the whole run, so the
        # staging buffer is allocated and zeroed once rather than 36 times a
        # pass.
        _spec_xb = np.zeros((1, HC_W, 1, seq), np.float16)
        spec_pos = [[0, 0] for _ in range(max(1, spec_k))]
        _spec_n1 = os.environ.get("FLASHNEXT_SPEC_N1") == "1"
        spec_ms = {"embed": 0.0, "gdn_stage": 0.0,
                   "gdn_ane": 0.0, "gdn_route": 0.0, "gdn_moe": 0.0,
                   "gdn_rec": 0.0, "qsa": 0.0, "head": 0.0, "commit": 0.0,
                   "overlap": 0.0}
        _spec_pipe = os.environ.get("FLASHNEXT_PIPE", "0") not in ("0", "false", "")

        def _moe_from_ane(hl, m_mix, m_hyp, m_inj, m_sh, _b=None):
            mixed_bsh = _bc1s_to_bsh(m_mix)
            t0 = time.perf_counter()
            inds, sc = hl.moe._route(
                np.asarray(mixed_bsh, np.float32).reshape(-1, H))
            t1 = time.perf_counter()
            routed = hl.moe._resident.routed_multi(mixed_bsh, inds, sc)
            y = routed + _bc1s_to_bsh(m_sh)
            t2 = time.perf_counter()
            hid_h = _bc1s_to_bsh(host_recombine(_bsh_to_bc1s(y), m_hyp, m_inj))
            t3 = time.perf_counter()
            if _b is not None:
                _b["gdn_route"] += t1 - t0
                _b["gdn_moe"] += t2 - t1
                _b["gdn_rec"] += t3 - t2
            return hid_h

        def _block_forward_serial(ids):
            n = len(ids)
            _b = spec_ms
            _t = time.perf_counter()
            hid = real_embedding_hidden(loader, list(ids))
            _b["embed"] += time.perf_counter() - _t
            for i in range(n_layers):
                if i in ple_layers:
                    for j in range(n):
                        hid[:, j:j + 1, :] = ple_layers[i].step(hid[:, j:j + 1, :], ids[j])
                hl = host_layers[i]
                if i in mil_gdn:
                    _t = time.perf_counter()
                    bc = _bsh_to_bc1s(np.asarray(hid, np.float32))
                    _spec_xb[..., :n] = np.asarray(bc[..., :n], np.float16)
                    _ts = time.perf_counter()
                    m_mix, m_hyp, m_inj, m_sh = _active["gdn"][i](_spec_xb, n=n)
                    _t2 = time.perf_counter()
                    _b["gdn_stage"] += _ts - _t
                    _b["gdn_ane"] += _t2 - _ts
                    hid = _moe_from_ane(hl, m_mix, m_hyp, m_inj, m_sh, _b)
                elif i in mil_qsa:
                    _t = time.perf_counter()
                    hid = mil_qsa_step_layer(i, hid, n=n, commit=False)
                    _b["qsa"] += time.perf_counter() - _t
                else:
                    raise RuntimeError(f"speculation: layer {i} has no K-slot graph")
            _t = time.perf_counter()
            mixed = gated_residual(mix_w, "hyper_connection_mixer", hid, False)
            lg = q_head(mixed) if q_head is not None else lm_logits(mixed, lm_w)
            _b["head"] += time.perf_counter() - _t
            return hid, np.asarray(lg, np.float32).reshape(n, -1)

        def _block_forward_pipe(ids):
            """Two-half wavefront: GPU MoE of half h overlaps ANE of the next half.

            ANE(i, h1); GPU(i,h1) || ANE(i,h2); GPU(i,h2) || ANE(i+1,h1).
            GDN recurrent state is fenced into the input surface between
            halves of the same layer. QSA stays one K-slot submit (attention
            across the block) and its MoE is split so it overlaps the next
            GDN half.
            """
            n = len(ids)
            n0 = n // 2
            cuts = ((0, n0), (n0, n))
            _b = spec_ms
            _t = time.perf_counter()
            hid = real_embedding_hidden(loader, list(ids))
            _b["embed"] += time.perf_counter() - _t
            pending = None  # (layer_i, lo, hi, mix, hyp, inj, sh)

            def run_pending_gpu():
                nonlocal pending, hid
                if pending is None:
                    return
                i, lo, hi, mix, hyp, inj, sh = pending
                hid[:, lo:hi, :] = _moe_from_ane(
                    host_layers[i], mix, hyp, inj, sh, _b)
                pending = None

            def gdn_ane_half(i, h, *, async_):
                lo, hi = cuts[h]
                w = hi - lo
                _spec_xb.fill(0)
                bc = _bsh_to_bc1s(np.asarray(hid[:, lo:hi, :], np.float32))
                _spec_xb[..., :w] = np.asarray(bc[..., :w], np.float16)
                t0 = time.perf_counter()
                mil_gdn[i].run(_spec_xb, w, async_=async_)
                if not async_:
                    _b["gdn_ane"] += time.perf_counter() - t0
                return t0

            def gdn_finish_half(i, h, *, fence):
                lo, hi = cuts[h]
                w = hi - lo
                t0 = time.perf_counter()
                out = mil_gdn[i].finish(w, offset=lo, fence=fence)
                _b["gdn_ane"] += time.perf_counter() - t0
                return out

            def ple_range(lo, hi):
                if i_ple not in ple_layers:
                    return
                for j in range(lo, hi):
                    hid[:, j:j + 1, :] = ple_layers[i_ple].step(
                        hid[:, j:j + 1, :], ids[j])

            for i in range(n_layers):
                i_ple = i
                if i in mil_gdn:
                    mil_gdn[i].begin_block(n)
                    # hid[h1] is already ready (embed, or GPU of the previous
                    # layer's h1). GPU of the previous h2 may still be pending
                    # — that is the GPU(i,h2) || ANE(i+1,h1) slot.
                    ple_range(*cuts[0])
                    have_gpu = pending is not None
                    t_sub = gdn_ane_half(i, 0, async_=have_gpu)
                    if have_gpu:
                        t_gpu = time.perf_counter()
                        run_pending_gpu()
                        gpu_dt = time.perf_counter() - t_gpu
                        ple_range(*cuts[1])
                    else:
                        ple_range(*cuts[1])
                        gpu_dt = 0.0
                    mix, hyp, inj, sh = gdn_finish_half(i, 0, fence=True)
                    if have_gpu:
                        _b["overlap"] += min(gpu_dt, time.perf_counter() - t_sub)
                    pending = (i, cuts[0][0], cuts[0][1], mix, hyp, inj, sh)
                    t_sub = gdn_ane_half(i, 1, async_=True)
                    t_gpu = time.perf_counter()
                    run_pending_gpu()
                    gpu_dt = time.perf_counter() - t_gpu
                    mix, hyp, inj, sh = gdn_finish_half(i, 1, fence=False)
                    _b["overlap"] += min(gpu_dt, time.perf_counter() - t_sub)
                    pending = (i, cuts[1][0], cuts[1][1], mix, hyp, inj, sh)
                elif i in mil_qsa:
                    run_pending_gpu()
                    ple_range(0, n)
                    _t = time.perf_counter()
                    mix, hyp, inj, sh = mil_qsa_step_layer(
                        i, hid, n=n, commit=False, until="ane")
                    _b["qsa"] += time.perf_counter() - _t
                    # Split the QSA MoE so GPU(h1) is done before the next
                    # GDN half, and GPU(h2) overlaps that GDN.
                    lo, hi = cuts[0]
                    pending = (i, lo, hi,
                               np.ascontiguousarray(mix[..., :n0]),
                               np.ascontiguousarray(hyp[..., :n0]),
                               np.ascontiguousarray(inj[..., :n0]),
                               np.ascontiguousarray(sh[..., :n0]))
                    run_pending_gpu()
                    lo, hi = cuts[1]
                    pending = (i, lo, hi,
                               np.ascontiguousarray(mix[..., n0:]),
                               np.ascontiguousarray(hyp[..., n0:]),
                               np.ascontiguousarray(inj[..., n0:]),
                               np.ascontiguousarray(sh[..., n0:]))
                else:
                    raise RuntimeError(f"speculation: layer {i} has no K-slot graph")
            run_pending_gpu()
            _t = time.perf_counter()
            mixed = gated_residual(mix_w, "hyper_connection_mixer", hid, False)
            lg = q_head(mixed) if q_head is not None else lm_logits(mixed, lm_w)
            _b["head"] += time.perf_counter() - _t
            return hid, np.asarray(lg, np.float32).reshape(n, -1)

        def _block_forward(ids):
            """One K-slot backbone pass. Returns (hidden BSH, logits per slot).

            No state is committed: speculation only knows how many tokens the
            block confirmed after it sees the logits, so every layer holds a
            state per prefix until `_block_commit` picks one.
            """
            n = len(ids)
            if _spec_pipe and n >= 2 and mil_gdn:
                return _block_forward_pipe(ids)
            return _block_forward_serial(ids)

        def _block_commit(j: int) -> None:
            """Adopt the prefix of length j + 1 across every layer."""
            _t = time.perf_counter()
            for i, lay in _active["gdn"].items():
                lay.commit(j)
            for i in mil_qsa:
                cache = attn_state[i]
                cache.offset = spec_kv_base[i] + j + 1
                qsa_idx_state[i].trim(cache.offset, qsa_idx[i].compress_ratio)
            spec_ms["commit"] += time.perf_counter() - _t

        _spec_eager = os.environ.get("FLASHNEXT_EAGER", "1") not in (
            "0", "false", "")

        async def evaluate_ppl(eval_ids) -> None:
            """Teacher-forced negative log likelihood over `eval_ids`.

            The same K-slot graphs the speculator verifies with, driven with
            the true continuation instead of drafts, so every slot is accepted
            and the measurement runs at the block rate rather than the token
            rate. Slot j predicts the token after ids[j], so a block of n slots
            scores n targets.
            """
            kk = spec_k
            tid = cur[-1]
            pos = 0
            total = 0.0
            n_scored = 0
            # Every slot but the first reuses the key set the block selected,
            # so a gap that lands only on the later slots means the shared
            # selection is the cost, not the arithmetic.
            per_slot = [[0.0, 0] for _ in range(max(1, kk))]
            t0 = time.perf_counter()
            while pos < len(eval_ids):
                n = min(kk, len(eval_ids) - pos)
                ids = [tid] + list(eval_ids[pos:pos + n - 1])
                for i in mil_qsa:
                    spec_kv_base[i] = int(attn_state[i].offset)
                _hid, lg = _block_forward(ids)
                lg = np.asarray(lg, np.float64)
                mx_ = lg.max(axis=-1, keepdims=True)
                lse = mx_[:, 0] + np.log(np.exp(lg - mx_).sum(axis=-1))
                for j in range(n):
                    d = lse[j] - lg[j, int(eval_ids[pos + j])]
                    total += d
                    per_slot[j][0] += d
                    per_slot[j][1] += 1
                n_scored += n
                _block_commit(n - 1)
                tid = int(eval_ids[pos + n - 1])
                pos += n
                if n_scored % 256 < kk:
                    el = time.perf_counter() - t0
                    print(f"    {n_scored}/{len(eval_ids)} tokens  "
                          f"ppl {np.exp(total / n_scored):.4f}  "
                          f"{n_scored / max(el, 1e-9):.1f} tok/s", flush=True)
            print("    per slot nll  " + "  ".join(
                f"s{j}={v / max(c, 1):.4f}" for j, (v, c) in enumerate(per_slot)
                if c), flush=True)
            from runtime.flashnext_indexer import STATS as _IST
            if _IST["calls"]:
                print(f"    indexer past budget: {_IST['calls']} calls, "
                      f"{_IST['none_past_budget']} fell back to recency, "
                      f"{_IST['lost_prefix']} had lost the prefix, "
                      f"{_IST['in_recent_2048'] / max(_IST['selected'], 1):.1%}"
                      f" of selected keys were in the last 2048",
                      flush=True)
            el = time.perf_counter() - t0
            print(f"  ppl over {n_scored} tokens: "
                  f"nll {total / max(n_scored, 1):.6f}  "
                  f"ppl {np.exp(total / max(n_scored, 1)):.4f}  "
                  f"in {el:.1f}s ({n_scored / max(el, 1e-9):.1f} tok/s)",
                  flush=True)

        async def speculate() -> None:
            """Draft with the MTP head, verify the whole block on the ANE."""
            nonlocal generated, cur
            kk = spec_k
            tid_l = cur[-1]
            chain = None
            blocks = accepted = proposed = 0
            t_draft = t_verify = t_eager = 0.0
            drafter_base = drafter.offset if drafter is not None else 0
            pending_drafts = None
            # Suffix matching over the prompt and everything generated so far.
            # The MTP head carries the first draft position; this carries the
            # rest whenever the text is repeating something it has already
            # seen, which is most of what a coding model emits.
            lookup = None
            if os.environ.get("FLASHNEXT_NGRAM", "1") not in ("0", "false", ""):
                from runtime.flashnext_ngram import ContextLookup
                lookup = ContextLookup()
                lookup.extend(cur)
            while len(generated) < max_new:
                _t0 = time.perf_counter()
                if pending_drafts is not None:
                    drafts = pending_drafts
                    pending_drafts = None
                else:
                    drafts = (drafter.draft(chain, tid_l, kk - 1, lookup)
                              if chain is not None and not _spec_n1
                              and drafter else [])
                t_draft += time.perf_counter() - _t0
                ids = [tid_l] + drafts
                proposed += len(drafts)
                for i in mil_qsa:
                    spec_kv_base[i] = int(attn_state[i].offset)
                _t1 = time.perf_counter()
                hid, lg = _block_forward(ids)
                preds = [int(v) for v in np.argmax(lg, axis=-1)]
                t_verify += time.perf_counter() - _t1
                blocks += 1
                m = 0
                while m < len(drafts) and preds[m] == drafts[m]:
                    m += 1
                for t in range(len(drafts)):
                    spec_pos[t][1] += 1
                    spec_pos[t][0] += int(preds[t] == drafts[t])
                accepted += m
                emit = drafts[:m] + [preds[m]]
                _before = len(cur)
                for tok in emit:
                    if len(generated) >= max_new:
                        break
                    generated.append(tok)
                    cur.append(tok)
                if lookup is not None:
                    lookup.extend(cur[_before:])
                tid_l = preds[m]
                chain = mx.array(np.ascontiguousarray(
                    np.asarray(hid, np.float32)[:, m:m + 1, :]))
                # Drafter positions past the accepted prefix were conditioned
                # on a token the backbone rejected. Trim first, then overlap
                # the next MTP chain with GDN/QSA commit (different engines).
                #
                # These rows are approximate in two ways and neither matters.
                # Step i writes the row for ids[i], so keeping m of them leaves
                # no row for the last accepted token, and every row pairs its
                # token with a state the drafter invented rather than the
                # backbone's. Keeping the extra row changed which drafts were
                # proposed and not how many were accepted (37/81 either way),
                # and replaying the accepted prefix against the backbone's own
                # hidden states was worse: 2.21 tokens a pass against 2.29, for
                # an extra MTP forward a block. The head's prediction is
                # carried by the front, not by its own short attention.
                if drafter is not None:
                    drafter.trim_to(drafter_base + m)
                    drafter_base = drafter.offset
                next_n = kk - 1
                do_eager = (_spec_eager and drafter is not None
                            and not _spec_n1 and next_n > 0
                            and len(generated) < max_new)
                commit_box = {}

                def _commit():
                    try:
                        _block_commit(m)
                    except Exception as exc:  # noqa: BLE001
                        commit_box["err"] = exc

                _t_c = time.perf_counter()
                if do_eager:
                    # MLX streams are thread-local; keep drafting on this
                    # thread and run GDN/QSA commit on another.
                    commit_th = threading.Thread(target=_commit, daemon=True)
                    commit_th.start()
                    pending_drafts = drafter.draft(chain, tid_l, next_n,
                                                   lookup)
                    commit_th.join()
                    if "err" in commit_box:
                        raise commit_box["err"]
                    t_eager += time.perf_counter() - _t_c
                else:
                    _block_commit(m)
            print(f"  speculation: {blocks} blocks, {len(generated)} tokens, "
                  f"{accepted}/{proposed} drafts accepted "
                  f"({accepted / max(proposed, 1):.0%}), "
                  f"{len(generated) / blocks:.2f} tokens/pass", flush=True)
            print(f"    draft {t_draft * 1e3:.0f} ms  "
                  f"verify {t_verify * 1e3:.0f} ms  "
                  f"eager-overlap {t_eager * 1e3:.0f} ms", flush=True)
            if lookup is not None:
                print(f"    context lookup fired {lookup.stats()} draft steps",
                      flush=True)
            print("    draft position top-1 match  " + "  ".join(
                f"d{t + 1}={a}/{b}" for t, (a, b) in enumerate(spec_pos) if b),
                flush=True)
            print("    per block ms  " + "  ".join(
                f"{k2}={v2 * 1e3 / max(blocks, 1):.1f}"
                for k2, v2 in spec_ms.items()), flush=True)
            print("    qsa(MIL) ms/block  " + "  ".join(
                f"{k2}={v2 * 1e3 / max(blocks, 1):.1f}"
                for k2, v2 in mil_qsa_ms.items()), flush=True)

        if spec_k > 1:
            if not (mil_gdn and mil_qsa and q_head is not None):
                raise RuntimeError("FLASHNEXT_SPEC needs MIL GDN + MIL QSA + "
                                   "FLASHNEXT_HEAD=mlx")
            import mlx.core as mx
            from runtime.flashnext_mtp import MtpDrafter
            _t = time.perf_counter()
            _a0 = mx.get_active_memory()
            drafter = None if os.environ.get("FLASHNEXT_NO_DRAFTER") == "1" \
                else MtpDrafter(q_head)
            _info = mx.metal.device_info()
            print(f"  MTP drafter loaded in {time.perf_counter() - _t:.1f}s  "
                  f"active {_a0 / 1e9:.1f} -> {mx.get_active_memory() / 1e9:.1f} GB  "
                  f"peak {mx.get_peak_memory() / 1e9:.1f} GB  "
                  f"max working set "
                  f"{_info['max_recommended_working_set_size'] / 1e9:.1f} GB",
                  flush=True)
            print(f"  ANE/GPU pipeline: "
                  f"{'on' if _spec_pipe else 'off'} "
                  f"(FLASHNEXT_PIPE, two micro-batches of {spec_k // 2}+"
                  f"{spec_k - spec_k // 2}); "
                  f"eager MTP {'on' if _spec_eager else 'off'} "
                  f"(FLASHNEXT_EAGER, draft || commit)",
                  flush=True)

        t_all = time.perf_counter()
        prefill_steps = len(prompt_ids) - 1
        pf_ids = list(prompt_ids)
        if fn_multi and prefill_steps > 0:
            done = await prefill_chunked(prompt_ids[:-1], prefill_k)
            if done:
                t_pf = time.perf_counter() - t_all
                if prefill_timers:
                    agg = {kk: sum(t[kk] for t in prefill_timers) * 1e3
                           for kk in prefill_timers[0]}
                    print("    prefill ms  " + "  ".join(
                        f"{kk}={vv:.0f}" for kk, vv in sorted(
                            agg.items(), key=lambda z: -z[1]) if vv >= 1), flush=True)
                print(f"  chunked prefill: {done} tokens in {t_pf:.3f}s "
                      f"({done / max(t_pf, 1e-9):.1f} tok/s, "
                      f"{t_pf / done * 1e3:.1f} ms/token); "
                      f"{prefill_steps - done} left serial", flush=True)
                pf_ids = prompt_ids[done:]
                prefill_steps = len(pf_ids) - 1
        t_decode = t_all
        if spec_k > 1:
            # Walk the prompt through the same K-slot graphs, full width. A
            # submit costs about 1.13 ms that does not depend on how many
            # slots carry a token, so feeding one token at a time paid it once
            # per token: 7.4 tok/s against 22 at K=4 and more at K=8. Every
            # slot holds a real token here, so committing the last one adopts
            # the whole chunk.
            if drafter is not None:
                drafter.reset()
            _pf_at = 0
            _pf_w = 1 if os.environ.get("FLASHNEXT_PREFILL_SERIAL") == "1" \
                else spec_k
            def _hand_over():
                """Move the prompt from the wide graphs to the decode ones.

                A GDN layer carries exactly the recurrent state and the conv
                window across a pass; the QSA cache and the indexer's blocks
                were host side all along.
                """
                nonlocal _pf_w
                if _active["gdn"] is not mil_gdn_pf:
                    return
                for i2, pf in mil_gdn_pf.items():
                    dec = mil_gdn[i2]
                    if dec is pf:
                        # Same program, same surfaces: the chunk's state and
                        # conv window are already where decode reads them.
                        dec.select(0)
                        continue
                    dec.set_state(pf.current_state())
                    dec._conv[:] = pf._conv
                    dec._conv_surface_current = False
                _active["gdn"], _active["qsa"] = mil_gdn, mil_qsa
                _pf_w = spec_k

            if mil_gdn_pf and _pf_w > 1 and prefill_steps >= prefill_mil_k:
                for lay_pf in mil_gdn_pf.values():
                    if hasattr(lay_pf, "select"):
                        lay_pf.select(1)
                _active["gdn"], _active["qsa"] = mil_gdn_pf, mil_qsa_pf
                _pf_w = prefill_mil_k
            while _pf_at < prefill_steps:
                # The wide graphs only export the last slot's state, so they
                # can only run chunks of exactly their width. The tail goes
                # through the decode graphs.
                if (_active["gdn"] is mil_gdn_pf
                        and prefill_steps - _pf_at < _pf_w):
                    _hand_over()
                n = min(_pf_w, prefill_steps - _pf_at)
                chunk = pf_ids[_pf_at:_pf_at + n]
                for i in mil_qsa:
                    spec_kv_base[i] = int(attn_state[i].offset)
                _pre_hid, _ = _block_forward(chunk)
                _block_commit(n - 1)
                # The drafter is conditioned on each position's hidden state
                # paired with the token that actually follows it, so it has to
                # walk the prompt too or it drafts from an empty cache.
                if drafter is not None:
                    nxt = pf_ids[_pf_at + 1:_pf_at + n + 1]
                    if nxt:
                        drafter.advance(
                            mx.array(np.ascontiguousarray(
                                np.asarray(_pre_hid, np.float32)[:, :len(nxt), :])),
                            [list(nxt)])
                _pf_at += n
            _hand_over()
            t_decode = time.perf_counter()
            if prefill_steps:
                print(f"  serial prefix prefill: {prefill_steps} tokens in "
                      f"{t_decode - t_all:.3f}s "
                      f"({prefill_steps / max(t_decode - t_all, 1e-9):.1f} tok/s)",
                      flush=True)
                print("    prefill ms/token  " + "  ".join(
                    f"{k2}={v2 * 1e3 / prefill_steps:.2f}"
                    for k2, v2 in spec_ms.items() if v2), flush=True)
                from runtime.mil_gdn_backend import TIMERS as _GT
                print("    prefill gdn call ms/token  " + "  ".join(
                    f"{k2}={v2 * 1e3 / prefill_steps:.2f}"
                    for k2, v2 in _GT.items()), flush=True)
                print("    prefill qsa ms/token  " + "  ".join(
                    f"{k2}={v2 * 1e3 / prefill_steps:.2f}"
                    for k2, v2 in mil_qsa_ms.items() if v2), flush=True)
            for _k in spec_ms:
                spec_ms[_k] = 0.0
            for _k in mil_qsa_ms:
                mil_qsa_ms[_k] = 0.0
            if ppl_ids:
                await evaluate_ppl(ppl_ids)
                return
            await speculate()
        else:
          for input_step in range(prefill_steps + max_new):
            step = input_step - prefill_steps
            tid = pf_ids[input_step] if input_step < len(pf_ids) else cur[-1]
            if step == 0:
                t_decode = time.perf_counter()
                if prefill_steps:
                    print(f"  serial prefix prefill: {prefill_steps} tokens in {t_decode-t_all:.3f}s", flush=True)
            hidden = real_embedding_hidden(loader, [tid])
            t_tok = time.perf_counter()
            store_c0 = store.counts()
            timers = {
                "attn_mix": 0.0, "ane": 0.0, "prep": 0.0, "host_front": 0.0,
                "qsa_feed": 0.0,
                "mlp_mix": 0.0, "moe": 0.0, "moe_route": 0.0, "moe_gather": 0.0,
                "moe_gemm": 0.0, "moe_slot_hit": 0, "moe_slot_copy": 0,
                "recombine": 0.0,
                "ple": 0.0, "ple_lookup": 0.0,
            }
            for i in range(n_layers):
                t1 = time.perf_counter()
                if i in ple_layers:
                    hidden = ple_layers[i].step(hidden, tid)
                    timers["ple"] += time.perf_counter()-t1
                    timers["ple_lookup"] += ple_layers[i].last_lookup_ms/1e3
                used = "numpy"
                hl = host_layers[i]
                if i in mil_gdn:
                    t_a = time.perf_counter()
                    xb = np.zeros((1, HC_W, 1, seq), np.float16)
                    bc = _bsh_to_bc1s(np.asarray(hidden[:, :1, :], np.float32))
                    xb[..., :1] = np.asarray(bc[..., :1], np.float16)
                    m_mix, m_hyp, m_inj, m_sh = mil_gdn[i](xb)
                    # host_recombine takes slot-0 tensors, as pool.take_pure
                    # hands it; the MIL graph returns all 32 slots.
                    m_hyp = np.ascontiguousarray(m_hyp[..., :1])
                    m_inj = np.ascontiguousarray(m_inj[..., :1])
                    timers["ane"] += time.perf_counter() - t_a
                    used = "MIL-int8"
                    t_m = time.perf_counter()
                    mixed_bsh = _bc1s_to_bsh(m_mix)[:, :1, :]
                    inds, sc = hl.moe._route(
                        np.asarray(mixed_bsh, np.float32).reshape(-1, H))
                    routed = hl.moe._resident.routed_multi(mixed_bsh, inds, sc)
                    y = routed + _bc1s_to_bsh(m_sh)[:, :1, :]
                    timers["moe"] += time.perf_counter() - t_m
                    t_r = time.perf_counter()
                    hc = host_recombine(_bsh_to_bc1s(y), m_hyp, m_inj)
                    hidden = _bc1s_to_bsh(hc)[:, :1, :]
                    timers["recombine"] += time.perf_counter() - t_r
                elif i in fn_pure:
                    t_a = time.perf_counter()
                    _pad32_into(_bsh_to_bc1s(hidden[:, :1, :]), pool.x_hc)
                    out_p = await fn_pure[i](pool.pure_feeds(i))
                    pool.accept_connected(i, out_p)
                    mixed0, hyper2, inj2 = pool.take_pure(out_p)
                    timers["ane"] += time.perf_counter() - t_a
                    used = "ANE-pure"
                    hidden = apply_moe_premix(i, mixed0, hyper2, inj2, timers)
                elif i in fn_connected:
                    t_m = time.perf_counter()
                    mixed, hyper, inj = host_gated_residual_cached(
                        _bsh_to_bc1s(hidden[:, :1, :]), hl.attn
                    )
                    timers["attn_mix"] += time.perf_counter() - t_m
                    t_a = time.perf_counter()
                    _pad32_into(mixed, pool.h)
                    out_c = await fn_connected[i](pool.connected_feeds(i))
                    pool.accept_connected(i, out_c)
                    attn0 = pool.take_attn(out_c["attn"])
                    timers["ane"] += time.perf_counter() - t_a
                    used = "ANE-GDN/conn"
                    hidden = apply_moe(i, attn0, hyper, inj, timers)
                elif i in fn_gdn:
                    t_m = time.perf_counter()
                    mixed, hyper, inj = host_gated_residual_cached(
                        _bsh_to_bc1s(hidden[:, :1, :]), hl.attn
                    )
                    timers["attn_mix"] += time.perf_counter() - t_m
                    if i in fn_front:
                        t_a = time.perf_counter()
                        _pad32_into(mixed, pool.h)
                        out_f = await fn_front[i](pool.front_feeds(i))
                        pool.accept_front(i, out_f)
                        t_front = time.perf_counter()
                        yin = out_f["yin"].numpy()
                        prep[i](yin[..., :1] if compact_gdn else yin)
                        t_prep = time.perf_counter()
                        out_g = await fn_gdn[i](pool.gdn_feeds(i, compact=compact_gdn))
                        pool.accept_gdn(i, out_g)
                        attn0 = pool.take_attn(out_g["attn"])
                        t_gdn = time.perf_counter()
                        timers["ane"] += (t_front - t_a) + (t_gdn - t_prep)
                        timers["prep"] += t_prep - t_front
                        used = "ANE-GDN/2sub"
                    else:
                        t_hf = time.perf_counter()
                        yin, new_conv = host_fronts[i].step(mixed, gdn_state[i].conv)
                        gdn_state[i].conv = new_conv
                        timers["host_front"] += time.perf_counter() - t_hf
                        t_p = time.perf_counter()
                        prep[i](yin if compact_gdn else _pad32(yin, seq))
                        t_prep = time.perf_counter()
                        timers["prep"] += t_prep - t_p
                        t_a = time.perf_counter()
                        out_g = await fn_gdn[i](pool.gdn_feeds(i, compact=compact_gdn))
                        pool.accept_gdn(i, out_g)
                        attn0 = pool.take_attn(out_g["attn"])
                        timers["ane"] += time.perf_counter() - t_a
                        used = "ANE-GDN/1sub"
                    hidden = apply_moe(i, attn0, hyper, inj, timers)
                elif i in mil_qsa:
                    t_a = time.perf_counter()
                    hidden = mil_qsa_step_layer(i, hidden[:, :1, :])
                    timers["ane"] += time.perf_counter() - t_a
                    used = "MIL-qsa"
                elif i in fn_qsa_step:
                    t_a = time.perf_counter()
                    hidden = await qsa_step_layer(i, hidden[:, :1, :], 1)
                    timers["ane"] += time.perf_counter() - t_a
                    used = "ANE-qsa-step"
                elif i in fn_qsa:
                    t_m = time.perf_counter()
                    mixed, hyper, inj = host_gated_residual_cached(
                        _bsh_to_bc1s(hidden[:, :1, :]), hl.attn
                    )
                    timers["attn_mix"] += time.perf_counter() - t_m
                    mixed_bsh = _bc1s_to_bsh(mixed)[:, :1, :]
                    if i in fn_qsa_multi:
                        # The single-token graphs are baked at max_S=32, which
                        # caps the whole context. Driving the 2048-wide graph
                        # with one slot lifts that at ~0.4 ms/layer.
                        t_a = time.perf_counter()
                        attn0 = await qsa_multi_attn(i, mixed_bsh, 1)
                        timers["ane"] += time.perf_counter() - t_a
                    else:
                        t_f = time.perf_counter()
                        _qsa_feeds_into(mixed_bsh, attn_state[i], seq, pool.qsa_bufs())
                        timers["qsa_feed"] += time.perf_counter() - t_f
                        t_a = time.perf_counter()
                        out_q = await fn_qsa[i](pool.qsa_feeds())
                        attn0 = pool.take_attn(out_q["out"])
                        nk, nv = pool.take_qsa_kv(out_q)
                        cache = attn_state[i]
                        cache.keys[:, cache.offset] = nk.reshape(QSA_HKV, QSA_HD)
                        cache.values[:, cache.offset] = nv.reshape(QSA_HKV, QSA_HD)
                        cache.offset += 1
                        timers["ane"] += time.perf_counter() - t_a
                    hidden = apply_moe(i, attn0, hyper, inj, timers)
                    used = "ANE-QSA/pack"
                else:
                    hidden = numpy_layer(i, hidden)
                if step == 0:
                    print(
                        f"    L{i:02d} {types[i]:18s} {used:8s} "
                        f"{(time.perf_counter() - t1) * 1e3:.0f} ms",
                        flush=True,
                    )
            if step < 0:
                continue
            t_head = time.perf_counter()
            mixed = gated_residual(mix_w, "hyper_connection_mixer", hidden, False)
            logits = q_head(mixed) if q_head is not None else lm_logits(mixed, lm_w)
            head_ms = (time.perf_counter() - t_head) * 1e3
            finite = bool(np.isfinite(logits).all() and np.isfinite(hidden).all())
            nxt = int(np.argmax(logits))
            top = np.argpartition(logits, -5)[-5:]
            top = top[np.argsort(-logits[top])]
            print(
                f"    hidden rms={float(np.sqrt(np.mean(hidden * hidden))):.4g}  "
                f"finite={finite}  logit[max]={float(logits[nxt]):.4g}  "
                f"top5={[(int(j), float(logits[j])) for j in top]}",
                flush=True,
            )
            generated.append(nxt)
            cur.append(nxt)
            dt = time.perf_counter() - t_tok
            piece = _decode_ids([nxt])
            print(
                f"  step {step + 1}/{max_new}  id={nxt}  {piece!r}  {dt:.2f}s  "
                f"ids so far {generated}",
                flush=True,
            )
            mixers_ms = (timers["attn_mix"] + timers["mlp_mix"]) * 1e3
            print(
                f"    timing ms  mixers={mixers_ms:.0f} "
                f"(attn={timers['attn_mix']*1e3:.0f} mlp={timers['mlp_mix']*1e3:.0f})  "
                f"ane+io={timers['ane']*1e3:.0f}  host_front={timers['host_front']*1e3:.0f}  "
                f"prep={timers['prep']*1e3:.0f}  "
                f"qsa_feed={timers['qsa_feed']*1e3:.0f}  "
                f"moe={timers['moe']*1e3:.0f} "
                f"(route={timers['moe_route']:.0f} gather={timers['moe_gather']:.0f} "
                f"gemm={timers['moe_gemm']:.0f} "
                f"slot_hit={int(timers['moe_slot_hit'])} "
                f"slot_copy={int(timers['moe_slot_copy'])})  "
                f"recombine={timers['recombine']*1e3:.0f}  lm_head={head_ms:.0f}  "
                f"ple={timers['ple']*1e3:.1f} (lookup={timers['ple_lookup']*1e3:.1f})",
                flush=True,
            )
            if mil_qsa:
                print("    qsa(MIL) ms  " + "  ".join(
                    f"{k}={v * 1e3:.1f}" for k, v in mil_qsa_ms.items()), flush=True)
                for k in mil_qsa_ms:
                    mil_qsa_ms[k] = 0.0
            h1, d1, m1 = store.counts()
            if moe_mode == "q4gemv":
                print(f"    native Q4 expert evaluations={d1-store_c0[1]} bank={store.ram_bytes()/1e9:.2f} GB", flush=True)
            elif moe_mode == "mlxresident":
                print(f"    resident GPU Q4 bank={store.ram_bytes()/1e9:.2f} GB", flush=True)
            elif store.mlx4 is not None:
                print(
                    f"    expert mlx4-hybrid  slot_hits={h1 - store_c0[0]}  "
                    f"dequant={d1 - store_c0[1]}  mmap_miss={m1 - store_c0[2]}  "
                    f"ram={store.ram_bytes() / 1e9:.2f} GB",
                    flush=True,
                )
            else:
                print(
                    f"    expert fp16  hits={h1 - store_c0[0]}  disk={d1 - store_c0[1]}  "
                    f"mmap_miss={m1 - store_c0[2]}  ram={store.ram_bytes() / 1e9:.2f} GB",
                    flush=True,
                )

        elapsed = time.perf_counter() - t_decode
        gen_text = _decode_ids(generated)
        full_text = _decode_ids(cur)
        print(f"  generated ids={generated}")
        print(f"  generated text={gen_text!r}")
        print(f"  full text={full_text!r}")
        print(f"  {elapsed:.2f}s  {len(generated) / elapsed:.3f} tok/s")
        st = expert_f16_store()
        t_fl = time.perf_counter()
        n_flush = st.flush_disk()
        if moe_mode == "mlxresident":
            print(f"  resident quantized GPU MoE: {st.ram_bytes()/1e9:.2f} GB; no expert dequant/pack", flush=True)
        elif moe_mode == "q4gemv":
            print(f"  native CPU INT4 MoE: {st.ram_bytes()/1e9:.2f} GB; no expanded expert matrices", flush=True)
        elif st.mlx4 is not None:
            print(
                f"  expert mlx4-hybrid  bits=4 gs=64  "
                f"slot_hits={st.mlx4.hits}  dequant={st.mlx4.bank_hits}  "
                f"path={st.mlx4.path}  ram={st.ram_bytes() / 1e9:.2f} GB  "
                f"layers={len(st.mlx4._layers)}  "
                f"mode={_moe_decode_mode()}  (no 4-bit GEMV, no BF16 mmap, no f16 flush)"
            )
        else:
            print(
                f"  expert store hits={st.hits}  disk_hits={st.disk_hits}  "
                f"mmap_misses={st.misses}  ram_fp16={len(st._f16)}  "
                f"{st.ram_bytes() / 1e9:.2f} GB  disk={len(st._disk_ok)}  flushed={n_flush} in "
                f"{time.perf_counter() - t_fl:.1f}s"
            )
        if ple_rows is not None:
            ple_rows.close()
        if prompt_ids == [prompt_id] and not ple_enabled:
            want = list(mlx_prefix[: len(generated)])
            match = generated[: len(want)] == want
            print(f"  vs MLX 4-bit prefix {want}: {'MATCH' if match else 'DIFF'}  "
                  f"(token 4 = 15 is the known 4-bit digit; BF16 is 16)")
            bf16_want = [220, 17, 15, 16][: len(generated)]
            print(
                f"  vs BF16 greedy {bf16_want}: "
                f"{'MATCH' if generated[: len(bf16_want)] == bf16_want else 'DIFF'}"
            )

    asyncio.run(run())
    loader.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=(
            "smoke", "body", "mixers", "gdn", "gdn_core", "decode", "split",
            "qsa", "layers", "stickiness", "moe", "generate", "host_test",
            "fuse",
        ),
    )
    p.add_argument("--seq", type=int, default=SEQ_DEFAULT)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--reuse", action="store_true", help="reuse existing .aimodel assets")
    p.add_argument(
        "--weight-inputs",
        action="store_true",
        help="page routed expert weights as ANE inputs (~380 ms/token; quality probe only)",
    )
    p.add_argument("--max-new", type=int, default=4, help="generate: greedy tokens to emit")
    p.add_argument("--no-ane", action="store_true", help="generate: numpy 48-layer only")
    p.add_argument(
        "--host-front",
        action="store_true",
        help="generate: S=1 front + ANE gdn-only (1 submit); FLASHNEXT_FRONT=mlx optionally runs the projection on GPU.",
    )
    p.add_argument(
        "--layers",
        default="",
        help="layers: comma-separated decoder indices (default: all 48)",
    )
    p.add_argument(
        "--ppl-file",
        default=None,
        help="generate: score this text file's tokens instead of generating, "
             "and report negative log likelihood and perplexity",
    )
    p.add_argument(
        "--ppl-tokens",
        type=int,
        default=2048,
        help="generate: how many tokens of --ppl-file to score",
    )
    p.add_argument(
        "--ppl-prefill",
        type=int,
        default=1,
        help="generate: how many leading tokens to prefill before scoring, "
             "so the prefill path itself can be scored",
    )
    p.add_argument(
        "--prompt-ids",
        default="760",
        help="generate: comma-separated prompt token ids (default 760 = The)",
    )
    p.add_argument(
        "--prompt-len",
        type=int,
        default=0,
        help="generate: pad --prompt-ids to this length with seeded random ids "
             "(for 8k/16k context timing; 0 = use ids as-is)",
    )
    p.add_argument(
        "--export-layers",
        action="store_true",
        help="fuse: after L0 bench, bake fused+pair for every GDN layer",
    )
    args = p.parse_args()
    if args.seq < 32 and args.stage not in ("host_test", "generate"):
        print(f"warning: ANE wants last dim >= 32; seq={args.seq} will pad badly")
    if args.stage == "host_test":
        print(f"stage=host_test seq={args.seq}  (no checkpoint, no ANE)")
        stage_host_test(args.seq)
        return
    print(f"stage={args.stage} seq={args.seq}  base={BASE}")
    if args.stage == "smoke":
        stage_smoke(args.seq, args.skip_bench)
    elif args.stage == "body":
        stage_body(args.seq, args.skip_bench, mixers=False)
    elif args.stage == "mixers":
        stage_body(args.seq, args.skip_bench, mixers=True)
    elif args.stage == "gdn_core":
        stage_gdn_core(args.seq, args.skip_bench)
    elif args.stage == "decode":
        stage_decode(args.seq, args.skip_bench)
    elif args.stage == "split":
        stage_split(args.seq, args.skip_bench, args.reuse, args.weight_inputs)
    elif args.stage == "qsa":
        stage_qsa(args.seq, args.skip_bench, args.reuse)
    elif args.stage == "layers":
        only = [int(t) for t in args.layers.split(",") if t.strip()] or None
        stage_layers(args.seq, reuse=args.reuse, only=only)
    elif args.stage == "stickiness":
        ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
        stage_stickiness(args.seq, args.max_new, ids)
    elif args.stage == "moe":
        ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
        stage_moe(args.seq, args.reuse, ids)
    elif args.stage == "generate":
        ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
        if args.prompt_len and args.prompt_len > len(ids):
            rng = np.random.default_rng(0)
            extra = rng.integers(0, 32_000, size=args.prompt_len - len(ids),
                                 dtype=np.int64)
            ids = ids + extra.tolist()
            print(f"  padded prompt to {len(ids)} tokens "
                  f"(prefix {ids[:len(ids) - extra.size]})", flush=True)
        ppl_ids = None
        if args.ppl_file:
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(str(BASE / "tokenizer.json"))
            text = Path(args.ppl_file).read_text()
            all_ids = tk.encode(text, add_special_tokens=False).ids
            pre = max(1, int(args.ppl_prefill))
            all_ids = all_ids[:args.ppl_tokens + pre]
            # The leading tokens are prefilled; everything after is scored.
            ids = all_ids[:pre]
            ppl_ids = all_ids[pre:]
            print(f"  ppl: {len(ppl_ids)} tokens from {args.ppl_file}")
            # Cache sizing follows max_new, and scoring walks the whole file.
            args.max_new = len(ppl_ids) + 8
        stage_generate(args.seq, args.max_new, ids, no_ane=args.no_ane,
                       host_front=args.host_front, ppl_ids=ppl_ids)
    elif args.stage == "fuse":
        from export_flashnext_gdn_fuse import stage_fuse
        stage_fuse(args.seq, args.skip_bench, args.reuse, args.export_layers)
    else:
        stage_gdn(args.seq, args.skip_bench)


if __name__ == "__main__":
    main()
