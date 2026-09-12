"""Persistent Core AI I/O for Flash-Next generate.

Core AI copies on ``NDArray(numpy)`` wrap, and ``NDArray.numpy()`` is a snapshot
that does **not** write back. Mutating a cached ``.numpy()`` view and resubmitting
the same NDArray therefore feeds stale zeros.

This pool:

* Reuses host numpy buffers for h, conv_pack, ssm, q/k/v/decay/beta/z, QSA
  caches, and scores (no ``np.empty`` per layer/token).
* Ping-pongs ANE output NDArrays for recurrent ``conv_pack`` / ``ssm`` so those
  two large tensors are not re-wrapped after the first submit.
* Re-wraps host-written tensors (h, q/k/v/decay/beta/z, QSA) at submit time —
  the only correct way to push new values into an NDArray.

Geometry matches ``scripts/export_flashnext_coreai.py``.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

H = 2560
HC = 4
HC_W = HC * H
HV, HK, DV, DK = 48, 16, 128, 128
GDN_Y = HV * DV
QKV = HK * DK + HK * DK + HV * DV
K_PIN = 10
QSA_HKV, QSA_HD = 2, 256
QSA_ROTARY = 64
QSA_MASK = np.float16(-40000.0)

_ND = None
_SK = None


def _want_iosurface() -> bool:
    return os.environ.get("FLASHNEXT_ANE_IOSURFACE", "1").strip() not in (
        "0", "false", "FALSE",
    )


def _runtime():
    global _ND, _SK
    if _ND is None:
        from coreai.runtime import NDArray, StorageKind

        _ND, _SK = NDArray, StorageKind
    return _ND, _SK


def wrap_ndarray(buf: np.ndarray, *, iosurface: bool | None = None):
    """Copy ``buf`` into a new NDArray.

    Recurrent ping-pong (conv/ssm) should pass ``iosurface=True`` so the ANE
    can keep those surfaces. Host-written inputs that are re-wrapped every
    submit must use ``iosurface=False`` (BYTES): a 8k prefill is ~50k wraps
    and IOSurface allocation dies around 3k
    (``Failed to allocate storage for NDArray ... sk: ioSurface``).
    """
    NDArray, StorageKind = _runtime()
    arr = np.ascontiguousarray(buf)
    # Default BYTES. IOSurface is only for recurrent ping-pong (explicit True)
    # and is still gated by FLASHNEXT_ANE_IOSURFACE.
    if iosurface is None:
        want = False
    else:
        want = bool(iosurface) and _want_iosurface()
    if want:
        try:
            return NDArray(arr, backing=StorageKind.IO_SURFACE)
        except Exception:
            pass
    return NDArray(arr)


def wrap_host(buf: np.ndarray):
    """Host-written input. Always BYTES — see wrap_ndarray."""
    return wrap_ndarray(buf, iosurface=False)


class SurfacePool:
    """Shared host numpy + per-layer recurrent NDArray ping-pong.

    One set of host buffers is reused across all 36 GDN + 12 QSA layers
    (decode is sequential). ``conv_pack`` / ``ssm`` NDArrays stay per-layer
    so ANE outputs feed the next token without a host round-trip.
    """

    __slots__ = (
        "seq", "h", "x_hc", "conv_pack", "ssm", "q", "k", "v", "decay", "beta", "z",
        "scores", "k_cache", "v_cache", "cos", "sin", "mask",
        "attn0", "hyper0", "inj0", "nk", "nv",
        "_conv_nd", "_ssm_nd",
        "_front", "_gdn", "_qsa",
        "gdn_layers", "qsa_layers",
        "compact_params",
    )

    def __init__(self, seq: int = 32):
        self.seq = int(seq)
        self.compact_params = np.empty((1, HV, 6, DK), np.float16)
        s = self.seq
        kv_c = QSA_HKV * QSA_HD
        self.h = np.zeros((1, H, 1, s), np.float16)
        self.x_hc = np.zeros((1, HC_W, 1, s), np.float16)
        self.conv_pack = np.zeros((1, 3 * QKV, 1, s), np.float16)
        self.ssm = np.zeros((1, HV, DV, DK), np.float16)
        self.q = np.empty((1, HV, s, DK), np.float16)
        self.k = np.empty((1, HV, s, DK), np.float16)
        self.v = np.empty((1, HV, s, DV), np.float16)
        self.decay = np.empty((1, HV, 1, s), np.float16)
        self.beta = np.empty((1, HV, 1, s), np.float16)
        self.z = np.empty((1, GDN_Y, 1, s), np.float16)
        self.scores = np.zeros((1, K_PIN, 1, s), np.float16)
        self.k_cache = np.zeros((1, kv_c, 1, s), np.float16)
        self.v_cache = np.zeros((1, kv_c, 1, s), np.float16)
        self.cos = np.zeros((1, QSA_ROTARY // 2, 1, s), np.float16)
        self.sin = np.zeros((1, QSA_ROTARY // 2, 1, s), np.float16)
        self.mask = np.full((1, s + s, 1, s), QSA_MASK, np.float16)
        self.attn0 = np.zeros((1, H, 1, 1), np.float32)
        self.hyper0 = np.zeros((1, HC_W, 1, 1), np.float32)
        self.inj0 = np.zeros((1, HC, 1, 1), np.float32)
        self.nk = np.zeros(kv_c, np.float32)
        self.nv = np.zeros(kv_c, np.float32)
        self._conv_nd: dict[int, Any] = {}
        self._ssm_nd: dict[int, Any] = {}
        self._front: dict[str, Any] = {"h": None, "conv_pack": None}
        self._gdn: dict[str, Any] = {
            "q": None, "k": None, "v": None, "decay": None,
            "beta": None, "state": None, "z": None,
        }
        self._qsa: dict[str, Any] = {
            "h": None, "k_cache": None, "v_cache": None,
            "cos": None, "sin": None, "mask": None,
        }
        self.gdn_layers: list[int] = []
        self.qsa_layers: list[int] = []

    def add_gdn(self, layer_i: int, prep=None) -> None:
        i = int(layer_i)
        self._conv_nd[i] = None
        self._ssm_nd[i] = None
        self.gdn_layers.append(i)
        if prep is not None:
            prep.q, prep.k, prep.v = self.q, self.k, self.v
            prep.decay, prep.beta, prep.z = self.decay, self.beta, self.z

    def add_qsa(self, layer_i: int) -> None:
        self.qsa_layers.append(int(layer_i))

    def qsa_bufs(self) -> tuple[np.ndarray, ...]:
        return (self.h, self.k_cache, self.v_cache, self.cos, self.sin, self.mask)

    def front_feeds(self, layer_i: int) -> dict[str, Any]:
        """Wrap host ``h``; reuse ping-ponged ``conv_pack`` NDArray when present."""
        i = int(layer_i)
        self._front["h"] = wrap_ndarray(self.h, iosurface=False)
        nd = self._conv_nd.get(i)
        if nd is None:
            nd = wrap_ndarray(self.conv_pack, iosurface=True)
            self._conv_nd[i] = nd
        self._front["conv_pack"] = nd
        return self._front

    def accept_front(self, layer_i: int, out: dict) -> None:
        nd = out.get("new_pack")
        if nd is not None:
            self._conv_nd[int(layer_i)] = nd

    def gdn_feeds(self, layer_i: int, compact: bool = False) -> dict[str, Any]:
        """Wrap HostPrep q/k/v/decay/beta/z; ping-pong ``state``."""
        i = int(layer_i)
        if compact:
            p = self.compact_params
            p[:, :, 0, :] = self.q[:, :, 0, :]
            p[:, :, 1, :] = self.k[:, :, 0, :]
            p[:, :, 2, :] = self.v[:, :, 0, :]
            p[:, :, 3, :] = self.decay[:, :, 0, :1]
            p[:, :, 4, :] = self.beta[:, :, 0, :1]
            p[:, :, 5, :] = self.z[..., 0].reshape(1, HV, DK)
            nd = self._ssm_nd.get(i)
            if nd is None:
                nd = wrap_ndarray(self.ssm, iosurface=True)
                self._ssm_nd[i] = nd
            return {"params": wrap_ndarray(p), "state": nd}
        self._gdn["q"] = wrap_ndarray(self.q)
        self._gdn["k"] = wrap_ndarray(self.k)
        self._gdn["v"] = wrap_ndarray(self.v)
        self._gdn["decay"] = wrap_ndarray(self.decay)
        self._gdn["beta"] = wrap_ndarray(self.beta)
        self._gdn["z"] = wrap_ndarray(self.z)
        nd = self._ssm_nd.get(i)
        if nd is None:
            nd = wrap_ndarray(self.ssm, iosurface=True)
            self._ssm_nd[i] = nd
        self._gdn["state"] = nd
        return self._gdn

    def accept_gdn(self, layer_i: int, out: dict) -> None:
        nd = out.get("new_ssm")
        if nd is not None:
            self._ssm_nd[int(layer_i)] = nd

    def connected_feeds(self, layer_i: int) -> dict[str, Any]:
        """One-submit connected GDN: wrap host ``h``; ping-pong ``conv`` + ``state``.

        Graph I/O names are ``h``, ``conv``, ``state`` / ``attn``, ``new_ssm``,
        ``new_conv`` (not the split ``conv_pack`` / ``new_pack`` pair).
        """
        i = int(layer_i)
        conv_nd = self._conv_nd.get(i)
        if conv_nd is None:
            conv_nd = wrap_ndarray(self.conv_pack, iosurface=True)
            self._conv_nd[i] = conv_nd
        ssm_nd = self._ssm_nd.get(i)
        if ssm_nd is None:
            ssm_nd = wrap_ndarray(self.ssm, iosurface=True)
            self._ssm_nd[i] = ssm_nd
        return {"h": wrap_ndarray(self.h, iosurface=False), "conv": conv_nd, "state": ssm_nd}

    def pure_feeds(self, layer_i: int) -> dict[str, Any]:
        """MLX-shaped layer step: 10240-d residual in; mixers live in the graph."""
        i = int(layer_i)
        conv_nd = self._conv_nd.get(i)
        if conv_nd is None:
            conv_nd = wrap_ndarray(self.conv_pack, iosurface=True)
            self._conv_nd[i] = conv_nd
        ssm_nd = self._ssm_nd.get(i)
        if ssm_nd is None:
            ssm_nd = wrap_ndarray(self.ssm, iosurface=True)
            self._ssm_nd[i] = ssm_nd
        return {"x": wrap_ndarray(self.x_hc, iosurface=False), "conv": conv_nd, "state": ssm_nd}

    def take_pure(self, out: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Slot-0 mixed / hyper / inj from a pure_step graph."""
        mixed = self.take_attn(out["mixed"])
        np.copyto(self.hyper0, out["hyper"].numpy()[..., :1], casting="unsafe")
        np.copyto(self.inj0, out["inj"].numpy()[..., :1], casting="unsafe")
        return mixed, self.hyper0, self.inj0

    def accept_connected(self, layer_i: int, out: dict) -> None:
        i = int(layer_i)
        nd = out.get("new_conv")
        if nd is not None:
            self._conv_nd[i] = nd
        nd = out.get("new_ssm")
        if nd is not None:
            self._ssm_nd[i] = nd

    def qsa_feeds(self) -> dict[str, Any]:
        self._qsa["h"] = wrap_ndarray(self.h, iosurface=False)
        self._qsa["k_cache"] = wrap_ndarray(self.k_cache, iosurface=False)
        self._qsa["v_cache"] = wrap_ndarray(self.v_cache, iosurface=False)
        self._qsa["cos"] = wrap_ndarray(self.cos, iosurface=False)
        self._qsa["sin"] = wrap_ndarray(self.sin, iosurface=False)
        self._qsa["mask"] = wrap_ndarray(self.mask, iosurface=False)
        return self._qsa

    def take_attn(self, out_nd) -> np.ndarray:
        """Copy token slot 0 of an ANE output into the reused fp32 ``attn0``."""
        src = out_nd.numpy() if hasattr(out_nd, "numpy") else np.asarray(out_nd)
        np.copyto(self.attn0, src[..., :1], casting="unsafe")
        return self.attn0

    def take_qsa_kv(self, out: dict) -> tuple[np.ndarray, np.ndarray]:
        nk = out["new_k"].numpy()
        nv = out["new_v"].numpy()
        np.copyto(self.nk, np.asarray(nk, np.float32)[0, :, 0, 0])
        np.copyto(self.nv, np.asarray(nv, np.float32)[0, :, 0, 0])
        return self.nk, self.nv
