"""Fast CPU kernels for Flash-Next host mixers, HostPrep, and lm_head.

Mixers: 2-D contiguous GEMM/GEMV with preallocated ``out=`` buffers.
HostPrep: vectorized SiLU / L2 / head-repeat / decay (no per-head Python loops).
lm_head: ``x @ W.T`` into a reused buffer (same layout as the previous generate path).

No MPS: tiny ops are launch-bound on GPU and previously flipped greedy tokens.
"""
from __future__ import annotations

import numpy as np

H = 2560
HC = 4
HC_W = HC * H  # 10240
HV, HK, DV, DK = 48, 16, 128, 128
GDN_Y = HV * DV  # 6144
QKV = HK * DK + HK * DK + HV * DV  # 10240
IN_O = QKV + GDN_Y + HV + HV
_REPEAT = HV // HK  # 3
_CLIP = 80.0

_logits_buf: np.ndarray | None = None


def silu(x: np.ndarray) -> np.ndarray:
    """Match export `HostPrep._silu` (clip-exp), not the reference branch sigmoid."""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -_CLIP, _CLIP))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -_CLIP, _CLIP)))


class MixScratch:
    """Reusable fp32 workspace for one `HostMixPack`."""

    __slots__ = ("s", "mix_h", "normed", "down", "up", "inj", "mixed", "inj_out")

    def __init__(self, s: int, mix_h: int):
        self.s = int(s)
        self.mix_h = int(mix_h)
        s, mh = self.s, self.mix_h
        self.normed = np.empty((1, HC, H, s), np.float32)
        self.down = np.empty((mh, s), np.float32)
        self.up = np.empty((HC_W, s), np.float32)
        self.inj = np.empty((HC, s), np.float32)
        self.mixed = np.empty((1, H, 1, s), np.float32)
        self.inj_out = np.empty((1, HC, 1, s), np.float32)


def _mix_scratch(pack, s: int) -> MixScratch:
    mh = int(pack.down_w.shape[0])
    sc = getattr(pack, "_fp", None)
    if sc is None or sc.s != s or sc.mix_h != mh:
        sc = MixScratch(s, mh)
        pack._fp = sc
    return sc


def gated_residual(x_hc: np.ndarray, pack, eps: float = 1e-6):
    """4-branch hyper mix. x_hc is BC1S ``(B, 10240, 1, S)``.

    `pack.hc_n` is already (stored + 1). Optional `pack.inj_w` (final mixer).
    Matches the previous host mixer: grouped RMS, silu(down/hc), sigmoid(up),
    mean over hc, inj = 2*sigmoid(raw/hc). GEMMs are 2-D ``W @ (C, S)``.
    """
    x = np.asarray(x_hc, np.float32)
    B, c, _, S = x.shape
    if c != HC_W:
        raise ValueError(f"expected C={HC_W}, got {c}")
    if B != 1:
        raise ValueError(f"host mixer expects B=1, got {B}")
    sc = _mix_scratch(pack, S)
    blocks = np.ascontiguousarray(x).reshape(1, HC, H, S)
    n = sc.normed
    inv = np.reciprocal(
        np.sqrt(np.mean(blocks * blocks, axis=2, keepdims=True) + np.float32(eps))
    )
    np.multiply(blocks, inv, out=n)
    n *= pack.hc_n[None, :, :, None]
    flat = n.reshape(HC_W, S)
    np.dot(pack.down_w, flat, out=sc.down)
    gate = silu(sc.down / HC)
    np.dot(pack.up_w, gate, out=sc.up)
    mw = sigmoid(sc.up).reshape(1, HC, H, S)
    np.mean(mw * n, axis=1, out=sc.mixed.reshape(1, H, S))
    if pack.inj_w is None:
        return sc.mixed, x, None
    np.dot(pack.inj_w, flat, out=sc.inj)
    sc.inj_out.reshape(HC, S)[:] = np.float32(2.0) * sigmoid(sc.inj / HC)
    return sc.mixed, x, sc.inj_out


def recombine(out_h: np.ndarray, hyper_input: np.ndarray, inj: np.ndarray) -> np.ndarray:
    """Broadcast attn/MLP output onto the 4 residual branches: h + out * inj."""
    out = np.asarray(out_h, np.float32)
    s = out.shape[-1]
    injection = out.reshape(1, 1, H, s) * np.asarray(inj, np.float32).reshape(1, HC, 1, s)
    return np.asarray(hyper_input, np.float32) + injection.reshape(1, HC_W, 1, s)


def _l2_last(x: np.ndarray, eps) -> np.ndarray:
    acc = np.sum(x * x, axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(acc + eps))


def run_prep(prep, yin: np.ndarray):
    """SiLU / L2 / head-repeat / decay into prep's fp16 buffers.

    Full last dim of yin (front-graph pad columns). Slot-0-only was ~16× faster
    isolated but flipped greedy token 2 (16 vs 17); ANE reads the padded S=32.
    """
    x = np.asarray(yin, np.float32)
    if x.ndim != 4 or x.shape[1] != IN_O:
        raise ValueError(f"yin expected (1, {IN_O}, 1, S), got {x.shape}")
    B, _, _, S = x.shape
    conv = silu(x[:, :QKV])
    z = x[:, QKV : QKV + GDN_Y]
    b = x[:, QKV + GDN_Y : QKV + GDN_Y + HV]
    a = x[:, QKV + GDN_Y + HV :]
    q = conv[:, : HK * DK].reshape(B, HK, DK, S).transpose(0, 1, 3, 2)
    k = conv[:, HK * DK : 2 * HK * DK].reshape(B, HK, DK, S).transpose(0, 1, 3, 2)
    v = conv[:, 2 * HK * DK :].reshape(B, HV, DV, S).transpose(0, 1, 3, 2)
    q = np.repeat(_l2_last(q, prep.eps) * prep.inv_sqrt, _REPEAT, axis=1)
    k = np.repeat(_l2_last(k, prep.eps), _REPEAT, axis=1)
    prep.q[:, :, :S, :] = q
    prep.k[:, :, :S, :] = k
    prep.v[:, :, :S, :] = v
    prep.z[..., :S] = z
    prep.beta[..., :S] = sigmoid(b)
    dt = np.logaddexp(0.0, a + prep.dt_bias)
    prep.decay[..., :S] = np.exp(-np.exp(prep.A_log) * dt)
    return prep.q, prep.k, prep.v, prep.decay, prep.beta, prep.z


def lm_logits(hidden: np.ndarray, weight: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """`(T, vocab)` as ``x @ W.T`` — same contraction the previous generate used."""
    global _logits_buf
    w = np.asarray(weight)
    if w.dtype != np.float32 or not w.flags.c_contiguous:
        w = np.ascontiguousarray(w, np.float32)
    x = np.ascontiguousarray(np.asarray(hidden, np.float32).reshape(-1, w.shape[1]))
    t, _ = x.shape
    v = w.shape[0]
    if out is None:
        if t == 1:
            if _logits_buf is None or _logits_buf.size != v:
                _logits_buf = np.empty(v, np.float32)
            out = _logits_buf
        else:
            out = np.empty((t, v), np.float32)
    dest = out.reshape(t, v)
    np.dot(x, w.T, out=dest)
    return dest.reshape(-1) if t == 1 else dest
