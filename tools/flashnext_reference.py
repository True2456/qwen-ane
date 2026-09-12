#!/usr/bin/env python3
"""CPU (numpy/fp32) reference for ONE full Qwen3.8-Flash-Next (`qwen4_exp`) layer.

Purpose
-------
Ground truth for Neural Engine ports of this architecture's pieces. Every stage
of the layer is computed in fp32 numpy from the BF16 checkpoint, and per-stage
activation statistics (mean / absmax / p99.9 of |x|) are dumped so int8 ANE
activation scales can be calibrated against real numbers instead of guesses.

Reference sources (NOT invented here)
-------------------------------------
Primary, and the one the code follows op-for-op:
    /Users/true/.mlx128/mlx-lm/mlx_lm/models/qwen4_exp.py
    /Users/true/.mlx128/mlx-lm/mlx_lm/models/gated_delta.py
    /Users/true/.mlx128/mlx-lm/mlx_lm/models/switch_layers.py
  This is an MLX port of HF transformers `modeling_qwen4_exp.py` /
  `modular_qwen4_exp.py` (huggingface/transformers PR #48337), per its own
  module docstring. `qwen4_exp` is exactly this model's `model_type`.

Corroborating, for the gated-delta recurrence and the q/k normalization only:
    /opt/homebrew/lib/python3.14/site-packages/transformers/models/qwen3_next/
        modeling_qwen3_next.py
  (transformers 5.14.1). `torch_recurrent_gated_delta_rule` at line ~456 and
  `Qwen3NextGatedDeltaNet.forward` at line ~660 agree with the MLX port
  op-for-op, including the decay-then-delta ordering, `beta = sigmoid(b)`,
  `g = -exp(A_log) * softplus(a + dt_bias)`, the l2norm of q/k, and the extra
  `1/sqrt(head_k_dim)` on q. Installed transformers (5.14.1) has NO
  models/qwen4_exp, so the MLX port is the only qwen4_exp-specific source.

Recurrence implemented (per value head h, timestep t; see report in module
`RECURRENCE_DOC`):
    qkv_t          = conv_silu(in_proj_qkv(x))_t                 # depthwise K=4
    q,k,v          = split(qkv_t, [16*128, 16*128, 48*128])
    q,k            = repeat_interleave(q,k, 48/16 = 3) over heads
    q_hat          = l2norm(q) / sqrt(128)
    k_hat          = l2norm(k)
    beta_t         = sigmoid(in_proj_b(x)_t)                     # [48]
    g_t            = exp(-exp(A_log) * softplus(in_proj_a(x)_t + dt_bias))
    S_t'           = g_t * S_{t-1}                               # S: [48,128v,128k]
    kv_mem         = S_t' @ k_hat_t                              # [128v]
    delta          = (v_t - kv_mem) * beta_t
    S_t            = S_t' + outer(delta, k_hat_t)
    y_t            = S_t @ q_hat_t                               # [128v]
    o_t            = norm.weight * rms(y_t) * sigmoid(z_t)       # z = in_proj_z(x)
    out_t          = out_proj(flatten(o_t))

Ownership: this file only. Does not read or write runtime/, probes/, docs/ or
any other tools/ file.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_MODEL = "/Users/true/models/Qwen3.8-Flash-Next"

RECURRENCE_DOC = """\
gated delta net (Qwen3-Next / qwen4_exp linear_attention), per value head h:

  state S in R^{Dv x Dk}, Dv = Dk = 128, 48 value heads, 16 key heads (share 3:1)

  input projections (all read the same normed/mixed hidden x_t in R^2560):
      qkv_t = W_qkv x_t                 W_qkv: [10240, 2560]
      z_t   = W_z   x_t                 W_z:   [ 6144, 2560]
      a_t   = W_a   x_t                 W_a:   [   48, 2560]
      b_t   = W_b   x_t                 W_b:   [   48, 2560]

  depthwise causal conv over the 10240 qkv channels, kernel 4, then SiLU:
      u_t[c] = silu( sum_{j=0..3} conv_w[c, j] * qkv_{t-3+j}[c] )
      (channels padded on the left by the 3-token conv state; zeros at BOS)

  split u_t into q (16x128), k (16x128), v (48x128); q,k repeat_interleave x3
  so all three index 48 heads.

      q_hat = q * rsqrt(sum(q^2) + 1e-6) * (1/sqrt(128))
      k_hat = k * rsqrt(sum(k^2) + 1e-6)

  gates:
      beta_t = sigmoid(b_t)                                  in (0,1),  [48]
      g_t    = exp( -exp(A_log) * softplus(a_t + dt_bias) )  in (0,1),  [48]

  recurrence (decay, then delta rule, then read out):
      S <- g_t * S
      m  = S k_hat_t                    (kv memory read, R^Dv)
      d  = (v_t - m) * beta_t           (delta, R^Dv)
      S <- S + d k_hat_t^T              (rank-1 write)
      y_t = S q_hat_t                   (R^Dv)

  output:
      o_t = norm_w * ( y_t * rsqrt(mean(y_t^2) + 1e-6) ) * sigmoid(z_t)
      out_t = W_out * flatten_48x128(o_t)     W_out: [2560, 6144]

  NOTE the gate is applied AFTER the RMS norm and the norm weight is used RAW
  (not (1 + w)); the hyper-connection hc_norm weights ARE (1 + w) shifted.
"""


# ---------------------------------------------------------------------------
# safetensors mmap loader (BF16 -> fp32, one layer at a time)
# ---------------------------------------------------------------------------
_DTYPE_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4,
                   "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def bf16_to_fp32(u16: np.ndarray) -> np.ndarray:
    """BF16 bit pattern (uint16) -> float32, exactly (shift left 16 bits)."""
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


class SafeShard:
    """One memory-mapped safetensors shard. Nothing is read until sliced."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n))
        self.data_start = 8 + n
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def meta(self, key: str) -> dict:
        return self.header[key]

    def raw(self, key: str, expert: Optional[int] = None) -> np.ndarray:
        """Return the raw stored array (no dtype conversion).

        `expert`, if given, slices only that leading-index slab out of the mmap,
        so a [512, ...] packed expert tensor never materializes in full.
        """
        m = self.header[key]
        dt = m["dtype"]
        shape = list(m["shape"])
        off0, off1 = m["data_offsets"]
        itemsize = _DTYPE_ITEMSIZE[dt]
        if expert is not None:
            inner = int(np.prod(shape[1:]))
            stride = inner * itemsize
            off0 = off0 + expert * stride
            off1 = off0 + stride
            shape = shape[1:]
        base = self.data_start
        buf = self._mm[base + off0: base + off1]
        np_dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
                 "F64": np.float64, "I64": np.int64, "I32": np.int32,
                 "I16": np.int16, "I8": np.int8, "U8": np.uint8,
                 "BOOL": np.bool_}[dt]
        arr = np.frombuffer(buf, dtype=np_dt)
        return arr.reshape(shape), dt

    def fp32(self, key: str, expert: Optional[int] = None) -> np.ndarray:
        arr, dt = self.raw(key, expert)
        if dt == "BF16":
            return bf16_to_fp32(arr)
        return arr.astype(np.float32)

    def f16(self, key: str, expert: Optional[int] = None) -> np.ndarray:
        """Saturating BF16/FP32 → fp16, no extra full-tensor materialize."""
        arr, dt = self.raw(key, expert)
        if dt == "F16":
            return np.ascontiguousarray(arr)
        if dt == "BF16":
            f32 = bf16_to_fp32(arr)
        else:
            f32 = arr.astype(np.float32)
        return np.clip(f32, -65504.0, 65504.0).astype(np.float16)

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()


class ExpertSlab:
    """Lazy per-expert view into a packed [E, ...] tensor. fp32 on demand."""

    def __init__(self, shard: SafeShard, key: str):
        self.shard = shard
        self.key = key
        self.shape = tuple(shard.meta(key)["shape"])
        self.dtype = shard.meta(key)["dtype"]
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_f16: Dict[int, np.ndarray] = {}

    def __getitem__(self, e: int) -> np.ndarray:
        w = self._cache.get(e)
        if w is None:
            w = self.shard.fp32(self.key, expert=e)
            self._cache[e] = w
        return w

    def f16(self, e: int) -> np.ndarray:
        w = self._cache_f16.get(e)
        if w is None:
            w = np.ascontiguousarray(self.shard.f16(self.key, expert=int(e)))
            self._cache_f16[e] = w
        return w

    def __repr__(self):
        return f"ExpertSlab({self.key}, shape={self.shape}, {self.dtype})"


class FlashNextLoader:
    """Pulls one layer's tensors by name out of the 131-shard checkpoint."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.root = Path(model_path)
        self.config = json.loads((self.root / "config.json").read_text())
        self.text_config = self.config.get("text_config", self.config)
        idx = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map: Dict[str, str] = idx["weight_map"]
        self._shards: Dict[str, SafeShard] = {}

    def _shard(self, fname: str) -> SafeShard:
        s = self._shards.get(fname)
        if s is None:
            s = SafeShard(self.root / fname)
            self._shards[fname] = s
        return s

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> np.ndarray:
        return self._shard(self.weight_map[key]).fp32(key)

    def shape_of(self, key: str) -> Tuple[int, ...]:
        s = self._shard(self.weight_map[key])
        return tuple(s.meta(key)["shape"])

    def slab(self, key: str) -> ExpertSlab:
        return ExpertSlab(self._shard(self.weight_map[key]), key)

    def layer(self, i: int, prefix: str = "model.language_model.layers") -> "LayerWeights":
        p = f"{prefix}.{i}."
        keys = [k for k in self.weight_map if k.startswith(p)]
        if not keys:
            raise KeyError(f"no tensors under {p}")
        w: Dict[str, Any] = {}
        for k in keys:
            short = k[len(p):]
            if "ngram_embedding.shard_" in short or short.endswith("ngram_embedding.weight"):
                continue
            if short in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                w[short] = self.slab(k)
            else:
                w[short] = self.get(k)
        lt = self.text_config["layer_types"][i]
        return LayerWeights(index=i, layer_type=lt, tensors=w,
                            config=self.text_config, loader=self)

    def close(self):
        for s in self._shards.values():
            s.close()
        self._shards.clear()


@dataclass
class LayerWeights:
    index: int
    layer_type: str
    tensors: Dict[str, Any]
    config: dict
    loader: Any = None

    def __getitem__(self, k: str):
        return self.tensors[k]

    def get(self, k, default=None):
        return self.tensors.get(k, default)

    def __contains__(self, k):
        return k in self.tensors

    def keys(self):
        return self.tensors.keys()


# ---------------------------------------------------------------------------
# elementwise / norm primitives (all fp32)
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable, branch-free
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    ex = np.exp(-np.abs(x, dtype=np.float32))
    out[pos] = 1.0 / (1.0 + ex[pos])
    out[~pos] = ex[~pos] / (1.0 + ex[~pos])
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return (x * sigmoid(x)).astype(np.float32)


def softplus(x: np.ndarray) -> np.ndarray:
    # log1p(exp(x)) computed as logaddexp(0, x): exact and overflow-free
    return np.logaddexp(np.float32(0.0), x).astype(np.float32)


def rms_norm(x: np.ndarray, weight: Optional[np.ndarray], eps: float) -> np.ndarray:
    """x * rsqrt(mean(x^2) + eps), optionally scaled by `weight` (last axis)."""
    x = x.astype(np.float32)
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * inv
    return y if weight is None else (y * weight).astype(np.float32)


def l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """FLA / transformers form: x * rsqrt(sum(x^2) + eps)."""
    x = x.astype(np.float32)
    return (x * (1.0 / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps))).astype(np.float32)


def l2norm_mlx(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """mlx-lm's algebraically-equivalent form: eps lands scaled by D."""
    d = x.shape[-1]
    return (rms_norm(x, None, eps) * (d ** -0.5)).astype(np.float32)


def grouped_rms_norm(x: np.ndarray, weight: np.ndarray, group_size: int,
                     eps: float) -> np.ndarray:
    """Normalize within contiguous groups, then scale by the FULL-length weight.

    `weight` here must already be (1 + stored_weight); see `hc_norm_weight`.
    """
    *lead, d = x.shape
    xg = x.reshape(*lead, d // group_size, group_size)
    xg = rms_norm(xg, None, eps)
    return (weight * xg.reshape(*lead, d)).astype(np.float32)


def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """y = x @ w.T for HF-stored [out, in] weights. Always fp32 math."""
    x = np.asarray(x, np.float32)
    w = np.asarray(w, np.float32)
    return (x @ w.T).astype(np.float32)


def hc_norm_weight(w: np.ndarray) -> np.ndarray:
    """hc_norm / q_norm / k_norm style weights are stored as (w - 1)."""
    return (w.astype(np.float32) + 1.0)


# ---------------------------------------------------------------------------
# activation statistics
# ---------------------------------------------------------------------------
class StatCollector:
    """Records shape / mean / absmax / p99.9(|x|) per named stage."""

    def __init__(self, enabled: bool = True, extra_percentiles=(99.0, 99.9, 99.99)):
        self.enabled = enabled
        self.extra = tuple(extra_percentiles)
        self.stages: List[dict] = []
        self._seen: set = set()

    def __call__(self, name: str, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        if name in self._seen:
            # keep stage names unique so JSON consumers can key on them
            i = 2
            while f"{name}#{i}" in self._seen:
                i += 1
            name = f"{name}#{i}"
        self._seen.add(name)
        a = np.asarray(x, dtype=np.float32)
        absa = np.abs(a).ravel()
        rec = {
            "stage": name,
            "shape": list(a.shape),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "absmax": float(absa.max()),
            "rms": float(np.sqrt(np.mean(a.astype(np.float64) ** 2))),
        }
        pcts = np.percentile(absa, self.extra)
        for p, v in zip(self.extra, np.atleast_1d(pcts)):
            rec[f"abs_p{p:g}".replace(".", "_")] = float(v)
        # int8 scale suggestions: symmetric per-tensor
        rec["int8_scale_absmax"] = rec["absmax"] / 127.0
        rec["int8_scale_p999"] = float(rec["abs_p99_9"]) / 127.0
        self.stages.append(rec)
        return x

    def to_json(self) -> dict:
        return {"stages": self.stages}

    def table(self) -> str:
        if not self.stages:
            return "(no stages recorded)"
        hdr = (f"{'stage':<34} {'shape':>22} {'mean':>12} {'absmax':>12} "
               f"{'p99.9|x|':>12} {'p99.99|x|':>12}")
        lines = [hdr, "-" * len(hdr)]
        for r in self.stages:
            lines.append(
                f"{r['stage']:<34} {str(tuple(r['shape'])):>22} "
                f"{r['mean']:>12.5g} {r['absmax']:>12.5g} "
                f"{r['abs_p99_9']:>12.5g} {r['abs_p99_99']:>12.5g}")
        return "\n".join(lines)


_NULL_STATS = StatCollector(enabled=False)


# ---------------------------------------------------------------------------
# hyper-connections (GatedResidual)
# ---------------------------------------------------------------------------
def gated_residual(w: LayerWeights, prefix: str, hyper_input: np.ndarray,
                   use_combine: bool = True, stats: StatCollector = _NULL_STATS):
    """4-branch hyper-connection mix.

    hyper_input: (B, S, hc*H). Returns (mixed (B,S,H), hyper_input, inj (B,S,hc)).
    """
    cfg = w.config
    hc = cfg["hc_count"]
    h = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]

    normed = grouped_rms_norm(hyper_input,
                              hc_norm_weight(w[f"{prefix}.hc_norm.weight"]),
                              group_size=h, eps=eps)
    stats(f"{prefix}.hc_norm", normed)

    down = linear(normed, w[f"{prefix}.input_mix_weight_down.weight"])
    stats(f"{prefix}.mix_down", down)
    gate = silu(down / hc)
    up = linear(gate, w[f"{prefix}.input_mix_weight_up.weight"])
    stats(f"{prefix}.mix_up_pre_sigmoid", up)
    mix_w = sigmoid(up)

    *lead, _ = normed.shape
    mixed = (mix_w.reshape(*lead, hc, h) * normed.reshape(*lead, hc, h)).mean(axis=-2)
    mixed = mixed.astype(np.float32)
    stats(f"{prefix}.mixed", mixed)

    if not use_combine:
        return mixed
    raw_inj = linear(normed, w[f"{prefix}.block_inject_weight.weight"])
    inj = (2.0 * sigmoid(raw_inj / hc)).astype(np.float32)
    stats(f"{prefix}.inject", inj)
    return mixed, hyper_input, inj


def recombine(out: np.ndarray, hyper_input: np.ndarray, inj: np.ndarray) -> np.ndarray:
    *lead, h = out.shape
    injection = out[..., None, :] * inj[..., :, None]
    return (hyper_input + injection.reshape(*lead, -1)).astype(np.float32)


def apply_rope(x: np.ndarray, pos: int, rotary_dim: int = 64,
               theta: float = 10_000_000.0) -> np.ndarray:
    """GPT-NeoX / MLX ``RoPE(..., traditional=False)`` on the first rotary dims."""
    x = np.asarray(x, np.float32)
    d = int(rotary_dim)
    half = d // 2
    idx = np.arange(0, d, 2, dtype=np.float32)
    inv = 1.0 / (theta ** (idx / d))
    freqs = float(pos) * inv
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    x1 = x[..., :half]
    x2 = x[..., half:d]
    out = x.copy()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:d] = x1 * sin + x2 * cos
    return out


@dataclass
class AttnCache:
    """Decode-step KV cache for one full_attention layer."""
    keys: np.ndarray
    values: np.ndarray
    offset: int = 0

    @classmethod
    def empty(cls, n_kv: int, head_dim: int, max_len: int = 256,
              dtype=np.float32) -> "AttnCache":
        return cls(
            np.zeros((n_kv, max_len, head_dim), dtype),
            np.zeros((n_kv, max_len, head_dim), dtype),
            0,
        )

    def reset(self) -> None:
        self.keys[:] = 0
        self.values[:] = 0
        self.offset = 0


def full_attention_layer(weights: LayerWeights, hidden: np.ndarray,
                         cache: AttnCache,
                         stats: StatCollector = _NULL_STATS) -> np.ndarray:
    """Dense GQA decode step (context ≤ indexer_budget). hidden is mixed (B,1,H)."""
    cfg = weights.config
    hq = int(cfg["num_attention_heads"])
    hkv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    eps = float(cfg["rms_norm_eps"])
    rotary = int(hd * float(cfg.get("partial_rotary_factor", 0.25)))
    theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", 10_000_000.0))
    B, S, _ = hidden.shape
    if B != 1 or S != 1:
        raise NotImplementedError("full_attention reference is a single-token step")
    if cache.offset >= cache.keys.shape[1]:
        raise RuntimeError(f"attention cache exceeds {cache.keys.shape[1]}")

    qg = linear(hidden, weights["self_attn.q_proj.weight"]).reshape(hq, 2 * hd)
    q, gate = qg[:, :hd], qg[:, hd:]
    k = linear(hidden, weights["self_attn.k_proj.weight"]).reshape(hkv, hd)
    v = linear(hidden, weights["self_attn.v_proj.weight"]).reshape(hkv, hd)
    q = rms_norm(q, hc_norm_weight(weights["self_attn.q_norm.weight"]), eps)
    k = rms_norm(k, hc_norm_weight(weights["self_attn.k_norm.weight"]), eps)
    q = apply_rope(q, cache.offset, rotary, theta)
    k = apply_rope(k, cache.offset, rotary, theta)
    stats("attn.q", q)
    stats("attn.k", k)

    cache.keys[:, cache.offset] = k
    cache.values[:, cache.offset] = v
    valid = cache.offset + 1
    k_all = cache.keys[:, :valid]
    v_all = cache.values[:, :valid]
    group = hq // hkv
    scale = hd ** -0.5
    qg = q.reshape(hkv, group, hd)
    scores = np.einsum("hgd,hld->hgl", qg, k_all) * scale
    scores = scores.reshape(hq, valid)
    m = scores.max(axis=-1, keepdims=True)
    prob = np.exp(scores - m)
    prob = prob / prob.sum(axis=-1, keepdims=True)
    y = np.einsum(
        "hgl,hld->hgd", prob.reshape(hkv, group, valid), v_all,
    ).reshape(hq, hd)
    gated = (y * sigmoid(gate)).reshape(1, 1, -1)
    out = linear(gated, weights["self_attn.o_proj.weight"])
    stats("attn.out", out)
    cache.offset = valid
    return out


def mixer_hidden(loader: "FlashNextLoader", hidden_hc: np.ndarray) -> np.ndarray:
    """Final hyper-connection mix (no extra RMSNorm) → (B, S, hidden)."""
    prefix = "model.language_model.hyper_connection_mixer"
    tensors = {
        "hyper_connection_mixer.hc_norm.weight":
            loader.get(f"{prefix}.hc_norm.weight"),
        "hyper_connection_mixer.input_mix_weight_down.weight":
            loader.get(f"{prefix}.input_mix_weight_down.weight"),
        "hyper_connection_mixer.input_mix_weight_up.weight":
            loader.get(f"{prefix}.input_mix_weight_up.weight"),
    }
    w = LayerWeights(index=-1, layer_type="mixer", tensors=tensors,
                     config=loader.text_config, loader=loader)
    mixed = gated_residual(w, "hyper_connection_mixer", hidden_hc, False)
    return mixed


def lm_logits(loader: "FlashNextLoader", mixed: np.ndarray) -> np.ndarray:
    w = loader.get("lm_head.weight")
    x = np.asarray(mixed, np.float32).reshape(-1, w.shape[1])
    return (x @ w.T).astype(np.float32)


# ---------------------------------------------------------------------------
# linear attention (gated delta net)
# ---------------------------------------------------------------------------
@dataclass
class GDNState:
    """Recurrent state carried between chunks/steps."""
    conv: np.ndarray   # (B, K-1, conv_dim)  last K-1 raw qkv projections
    ssm: np.ndarray    # (B, Hv, Dv, Dk)     fp32 delta-net memory

    def copy(self) -> "GDNState":
        return GDNState(self.conv.copy(), self.ssm.copy())


def gdn_geometry(cfg: dict) -> dict:
    hk = cfg["linear_num_key_heads"]
    hv = cfg["linear_num_value_heads"]
    dk = cfg["linear_key_head_dim"]
    dv = cfg["linear_value_head_dim"]
    key_dim = hk * dk
    value_dim = hv * dv
    return dict(num_k_heads=hk, num_v_heads=hv, head_k_dim=dk, head_v_dim=dv,
                key_dim=key_dim, value_dim=value_dim,
                conv_dim=2 * key_dim + value_dim,
                conv_kernel=cfg["linear_conv_kernel_dim"],
                repeat=hv // hk)


def zero_gdn_state(cfg: dict, batch: int = 1) -> GDNState:
    g = gdn_geometry(cfg)
    return GDNState(
        conv=np.zeros((batch, g["conv_kernel"] - 1, g["conv_dim"]), dtype=np.float32),
        ssm=np.zeros((batch, g["num_v_heads"], g["head_v_dim"], g["head_k_dim"]),
                     dtype=np.float32),
    )


def depthwise_causal_conv1d(x: np.ndarray, conv_state: np.ndarray,
                            weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """x: (B, S, C); conv_state: (B, K-1, C); weight: (C, 1, K) as stored in HF.

    Returns (out (B, S, C) pre-activation, new_conv_state (B, K-1, C)).
    Output t = sum_j w[c, j] * inp[t + j] over the left-padded stream, i.e.
    a causal depthwise convolution using tokens t-(K-1) .. t.
    """
    B, S, C = x.shape
    w = weight.reshape(C, -1).astype(np.float32)     # (C, K)
    K = w.shape[1]
    assert conv_state.shape == (B, K - 1, C), conv_state.shape
    stream = np.concatenate([conv_state, x], axis=1)  # (B, S+K-1, C)
    new_state = stream[:, -(K - 1):, :].copy()
    out = np.zeros((B, S, C), dtype=np.float32)
    for j in range(K):
        out += stream[:, j:j + S, :] * w[:, j]
    return out, new_state


def gated_delta_recurrence(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                           g: np.ndarray, beta: np.ndarray,
                           state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sequential gated delta rule.

    q, k: (B, S, Hv, Dk) already l2-normed/scaled and head-expanded
    v:    (B, S, Hv, Dv)
    g:    (B, S, Hv)   multiplicative decay in (0, 1)
    beta: (B, S, Hv)   write strength in (0, 1)
    state:(B, Hv, Dv, Dk)
    Returns y (B, S, Hv, Dv), final state.
    """
    B, S, Hv, Dk = q.shape
    Dv = v.shape[-1]
    S_mem = state.astype(np.float32, copy=True)
    y = np.empty((B, S, Hv, Dv), dtype=np.float32)
    for t in range(S):
        kt = k[:, t]                                   # (B, Hv, Dk)
        vt = v[:, t]                                   # (B, Hv, Dv)
        qt = q[:, t]
        S_mem *= g[:, t][:, :, None, None]
        kv_mem = np.einsum("bhvk,bhk->bhv", S_mem, kt, optimize=True)
        delta = (vt - kv_mem) * beta[:, t][:, :, None]
        S_mem += delta[..., :, None] * kt[..., None, :]
        y[:, t] = np.einsum("bhvk,bhk->bhv", S_mem, qt, optimize=True)
    return y, S_mem


def linear_attention_layer(weights: LayerWeights, hidden: np.ndarray,
                           state: Optional[GDNState] = None,
                           stats: StatCollector = _NULL_STATS,
                           use_mlx_l2_eps: bool = False,
                           ) -> Tuple[np.ndarray, GDNState]:
    """Full gated-delta-net linear_attention block.

    hidden: (B, S, hidden_size) -- the hyper-connection *mixed* output, NOT the
            raw hc-wide residual stream.
    state:  None for a fresh sequence (prefill), or a GDNState to continue from
            (single-token step or chunked prefill).
    Returns (out (B, S, hidden_size), new GDNState).
    """
    cfg = weights.config
    geo = gdn_geometry(cfg)
    B, S, _ = hidden.shape
    if state is None:
        state = zero_gdn_state(cfg, B)

    # -- 1. four input projections (all read the same hidden) ---------------
    qkv = linear(hidden, weights["linear_attn.in_proj_qkv.weight"])
    stats("gdn.in_proj_qkv", qkv)
    z = linear(hidden, weights["linear_attn.in_proj_z.weight"])
    stats("gdn.in_proj_z", z)
    a = linear(hidden, weights["linear_attn.in_proj_a.weight"])
    stats("gdn.in_proj_a", a)
    b = linear(hidden, weights["linear_attn.in_proj_b.weight"])
    stats("gdn.in_proj_b", b)
    z = z.reshape(B, S, geo["num_v_heads"], geo["head_v_dim"])

    # -- 2. depthwise causal conv (K=4) + SiLU ------------------------------
    conv_pre, conv_state = depthwise_causal_conv1d(
        qkv, state.conv, weights["linear_attn.conv1d.weight"])
    stats("gdn.conv1d_pre_silu", conv_pre)
    conv_out = silu(conv_pre)
    stats("gdn.conv1d_post_silu", conv_out)

    # -- 3. split / head reshape / normalize -------------------------------
    kd, vd = geo["key_dim"], geo["value_dim"]
    q_raw = conv_out[..., :kd].reshape(B, S, geo["num_k_heads"], geo["head_k_dim"])
    k_raw = conv_out[..., kd:2 * kd].reshape(B, S, geo["num_k_heads"], geo["head_k_dim"])
    v = conv_out[..., 2 * kd:].reshape(B, S, geo["num_v_heads"], geo["head_v_dim"])
    stats("gdn.q_raw", q_raw)
    stats("gdn.k_raw", k_raw)
    stats("gdn.v", v)

    nrm = l2norm_mlx if use_mlx_l2_eps else l2norm
    inv_scale = geo["head_k_dim"] ** -0.5
    q = nrm(q_raw) * inv_scale
    k = nrm(k_raw)
    stats("gdn.q_l2_scaled", q)
    stats("gdn.k_l2", k)

    r = geo["repeat"]
    if r > 1:                                   # repeat_interleave over heads
        q = np.repeat(q, r, axis=-2)
        k = np.repeat(k, r, axis=-2)

    # -- 4. gates from A_log / dt_bias / a / b -----------------------------
    A_log = weights["linear_attn.A_log"].astype(np.float32)
    dt_bias = weights["linear_attn.dt_bias"].astype(np.float32)
    beta = sigmoid(b.astype(np.float32))
    dt = softplus(a.astype(np.float32) + dt_bias)
    stats("gdn.softplus_dt", dt)
    g = np.exp(-np.exp(A_log) * dt).astype(np.float32)
    stats("gdn.decay_g", g)
    stats("gdn.beta", beta)

    # -- 5. recurrence ------------------------------------------------------
    y, ssm_state = gated_delta_recurrence(q, k, v, g, beta, state.ssm)
    stats("gdn.core_out", y)
    stats("gdn.ssm_state_final", ssm_state)

    # -- 6. RMS norm over head_v_dim (raw weight) + sigmoid output gate ----
    eps = cfg["rms_norm_eps"]
    nw = weights["linear_attn.norm.weight"].astype(np.float32)
    normed = rms_norm(y, nw, eps)
    stats("gdn.norm_out", normed)
    gate_act = sigmoid(z.astype(np.float32))
    stats("gdn.output_gate", gate_act)
    gated = (normed * gate_act).astype(np.float32)
    stats("gdn.gated", gated)

    # -- 7. out_proj --------------------------------------------------------
    out = linear(gated.reshape(B, S, -1), weights["linear_attn.out_proj.weight"])
    stats("gdn.out_proj", out)
    return out, GDNState(conv=conv_state, ssm=ssm_state)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------
def infer_expert_layout(weights: LayerWeights, verbose: bool = True) -> dict:
    """Determine the packed-expert layout FROM SHAPES, not by assumption."""
    cfg = weights.config
    E = cfg["num_experts"]
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]

    gu = weights["mlp.experts.gate_up_proj"]
    dn = weights["mlp.experts.down_proj"]
    gu_shape = tuple(gu.shape)
    dn_shape = tuple(dn.shape)

    notes = []
    # gate_up_proj
    if gu_shape == (E, 2 * I, H):
        gu_layout = "E_2I_H"      # [expert, out(=gate|up concat), in]
        notes.append(f"gate_up_proj {gu_shape} == (E={E}, 2*I={2*I}, H={H}) -> "
                     f"[expert, out, in]; gate = [:, :{I}, :], up = [:, {I}:, :]; "
                     f"y = x @ W.T")
    elif gu_shape == (E, H, 2 * I):
        gu_layout = "E_H_2I"
        notes.append(f"gate_up_proj {gu_shape} == (E={E}, H={H}, 2*I={2*I}) -> "
                     f"[expert, in, out]; y = x @ W")
    else:
        raise ValueError(f"unrecognized gate_up_proj shape {gu_shape} for "
                         f"E={E} H={H} I={I}")
    # down_proj
    if dn_shape == (E, H, I):
        dn_layout = "E_H_I"
        notes.append(f"down_proj    {dn_shape} == (E={E}, H={H}, I={I}) -> "
                     f"[expert, out, in]; y = h @ W.T")
    elif dn_shape == (E, I, H):
        dn_layout = "E_I_H"
        notes.append(f"down_proj    {dn_shape} == (E={E}, I={I}, H={H}) -> "
                     f"[expert, in, out]; y = h @ W")
    else:
        raise ValueError(f"unrecognized down_proj shape {dn_shape}")

    # gate/up split ORDER is not inferable from shape alone. Source:
    # mlx_lm/models/qwen4_exp.py Model.sanitize():
    #     out[base+"switch_mlp.gate_proj.weight"] = v[:, :I, :]
    #     out[base+"switch_mlp.up_proj.weight"]   = v[:, I:, :]
    notes.append("gate/up split order (gate first, then up) comes from "
                 "mlx_lm/models/qwen4_exp.py Model.sanitize(), not from shapes.")

    layout = dict(gate_up=gu_layout, down=dn_layout, E=E, H=H, I=I,
                  gate_up_shape=gu_shape, down_shape=dn_shape, notes=notes)
    if verbose:
        print("[expert layout inferred from shapes]")
        for n in notes:
            print("  " + n)
    return layout


def _expert_matmuls(w_gu: np.ndarray, w_dn: np.ndarray, x: np.ndarray,
                    layout: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One expert's SwiGLU MLP on x (..., H). Returns (gate, up, out)."""
    I = layout["I"]
    w_gu = np.asarray(w_gu, np.float32)
    w_dn = np.asarray(w_dn, np.float32)
    if layout["gate_up"] == "E_2I_H":
        gu = x @ w_gu.T
    else:
        gu = x @ w_gu
    gate, up = gu[..., :I], gu[..., I:]
    h = silu(gate) * up
    if layout["down"] == "E_H_I":
        out = h @ w_dn.T
    else:
        out = h @ w_dn
    return gate, up, out.astype(np.float32)


def moe_layer(weights: LayerWeights, hidden: np.ndarray,
              stats: StatCollector = _NULL_STATS,
              layout: Optional[dict] = None,
              norm_topk_prob: bool = True) -> np.ndarray:
    """Sparse MoE block: softmax router -> top-k renormalized -> experts,
    plus the sigmoid-gated shared expert.

    hidden: (B, S, hidden_size) -- the mlp hyper-connection *mixed* output.
    """
    cfg = weights.config
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    top_k = cfg["num_experts_per_tok"]
    if layout is None:
        layout = infer_expert_layout(weights, verbose=False)

    B, S, _ = hidden.shape
    x = hidden.reshape(-1, H)                                   # (T, H)

    # -- router ------------------------------------------------------------
    logits = linear(x, weights["mlp.gate.weight"])              # (T, E)
    stats("moe.router_logits", logits)
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp((logits - m).astype(np.float64))
    probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    stats("moe.router_probs", probs)

    inds = np.argpartition(probs, -top_k, axis=-1)[:, -top_k:]  # (T, k)
    scores = np.take_along_axis(probs, inds, axis=-1)
    stats("moe.topk_scores_raw", scores)
    if norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    stats("moe.topk_scores", scores)

    # -- routed experts ----------------------------------------------------
    gu_slab = weights["mlp.experts.gate_up_proj"]
    dn_slab = weights["mlp.experts.down_proj"]
    T = x.shape[0]
    routed = np.zeros((T, H), dtype=np.float32)
    all_gate, all_up, all_out = [], [], []
    # group tokens by expert so each expert slab is fetched once
    by_expert: Dict[int, List[Tuple[int, int]]] = {}
    for t in range(T):
        for j in range(top_k):
            by_expert.setdefault(int(inds[t, j]), []).append((t, j))
    for e_id in sorted(by_expert):
        rows = by_expert[e_id]
        ti = np.array([r[0] for r in rows])
        w_gu = gu_slab[e_id] if isinstance(gu_slab, ExpertSlab) else gu_slab[e_id]
        w_dn = dn_slab[e_id] if isinstance(dn_slab, ExpertSlab) else dn_slab[e_id]
        g_, u_, o_ = _expert_matmuls(w_gu, w_dn, x[ti], layout)
        all_gate.append(g_)
        all_up.append(u_)
        all_out.append(o_)
        sc = np.array([scores[t, j] for (t, j) in rows], dtype=np.float32)[:, None]
        np.add.at(routed, ti, o_ * sc)
    stats("moe.expert_gate_pre_silu", np.concatenate(all_gate, axis=0))
    stats("moe.expert_up", np.concatenate(all_up, axis=0))
    stats("moe.expert_down_out", np.concatenate(all_out, axis=0))
    stats("moe.routed_sum", routed)

    # -- shared expert -----------------------------------------------------
    sg = linear(x, weights["mlp.shared_expert.gate_proj.weight"])
    su = linear(x, weights["mlp.shared_expert.up_proj.weight"])
    stats("moe.shared_gate_pre_silu", sg)
    stats("moe.shared_up", su)
    sh = silu(sg) * su
    shared = linear(sh, weights["mlp.shared_expert.down_proj.weight"])
    stats("moe.shared_out", shared)
    sgate = sigmoid(linear(x, weights["mlp.shared_expert_gate.weight"]))
    stats("moe.shared_expert_gate", sgate)

    y = routed + sgate * shared
    stats("moe.out", y)
    return y.reshape(B, S, H)


# ---------------------------------------------------------------------------
# full decoder layer
# ---------------------------------------------------------------------------
def decoder_layer(weights: LayerWeights, hidden_hc: np.ndarray,
                  state: Optional[GDNState] = None,
                  stats: StatCollector = _NULL_STATS,
                  layout: Optional[dict] = None,
                  use_mlx_l2_eps: bool = False,
                  attn_cache: Optional[AttnCache] = None,
                  ) -> Tuple[np.ndarray, Any]:
    """One decoder layer (linear_attention or full_attention) + MoE.

    hidden_hc: (B, S, hc_count * hidden_size) residual stream.
    PLE is a no-op without the n-gram table (mlx-lm's fallback is zeros).
    """
    stats("layer.input_hc", hidden_hc)

    mixed, hyper_input, inj = gated_residual(
        weights, "attn_hyper_connection", hidden_hc, True, stats)
    extra: Any
    if weights.layer_type == "linear_attention":
        r, extra = linear_attention_layer(
            weights, mixed, state, stats, use_mlx_l2_eps=use_mlx_l2_eps)
    elif weights.layer_type == "full_attention":
        if attn_cache is None:
            raise ValueError("full_attention needs attn_cache")
        r = full_attention_layer(weights, mixed, attn_cache, stats)
        extra = attn_cache
    else:
        raise NotImplementedError(
            f"layer {weights.index} is {weights.layer_type}")
    hidden_hc = recombine(r, hyper_input, inj)
    stats("layer.post_attn_hc", hidden_hc)

    mixed, hyper_input, inj = gated_residual(
        weights, "mlp_hyper_connection", hidden_hc, True, stats)
    r = moe_layer(weights, mixed, stats, layout)
    hidden_hc = recombine(r, hyper_input, inj)
    stats("layer.output_hc", hidden_hc)
    return hidden_hc, extra


# ---------------------------------------------------------------------------
# self-consistency: prefill vs single-token step
# ---------------------------------------------------------------------------
def check_prefill_vs_step(weights: LayerWeights, hidden: np.ndarray,
                          use_mlx_l2_eps: bool = False) -> dict:
    """Run S tokens as one prefill, then as (S-1 prefill + 1 step); the last
    token's output and the final state must agree."""
    B, S, _ = hidden.shape
    assert S >= 2, "need at least 2 tokens"
    full_out, full_state = linear_attention_layer(
        weights, hidden, None, use_mlx_l2_eps=use_mlx_l2_eps)
    pre_out, pre_state = linear_attention_layer(
        weights, hidden[:, :S - 1], None, use_mlx_l2_eps=use_mlx_l2_eps)
    step_out, step_state = linear_attention_layer(
        weights, hidden[:, S - 1:S], pre_state, use_mlx_l2_eps=use_mlx_l2_eps)

    def rel(a, b):
        d = np.abs(a - b).max()
        s = max(float(np.abs(a).max()), 1e-30)
        return float(d), float(d / s)

    out_abs, out_rel = rel(full_out[:, S - 1:S], step_out)
    ssm_abs, ssm_rel = rel(full_state.ssm, step_state.ssm)
    conv_abs, conv_rel = rel(full_state.conv, step_state.conv)

    # also: token-by-token stepping for the whole sequence
    st = None
    outs = []
    for t in range(S):
        o, st = linear_attention_layer(weights, hidden[:, t:t + 1], st,
                                       use_mlx_l2_eps=use_mlx_l2_eps)
        outs.append(o)
    seq_out = np.concatenate(outs, axis=1)
    all_abs, all_rel = rel(full_out, seq_out)
    ssm2_abs, ssm2_rel = rel(full_state.ssm, st.ssm)

    res = dict(
        last_token_max_abs_diff=out_abs, last_token_max_rel_diff=out_rel,
        ssm_state_max_abs_diff=ssm_abs, ssm_state_max_rel_diff=ssm_rel,
        conv_state_max_abs_diff=conv_abs, conv_state_max_rel_diff=conv_rel,
        full_seq_step_max_abs_diff=all_abs, full_seq_step_max_rel_diff=all_rel,
        full_seq_step_ssm_max_rel_diff=ssm2_rel,
    )
    tol = 2e-4
    assert out_rel < tol, f"prefill vs step disagree on last token: {res}"
    assert ssm_rel < tol, f"prefill vs step disagree on ssm state: {res}"
    assert all_rel < tol, f"prefill vs per-token stepping disagree: {res}"
    res["passed"] = True
    res["tolerance"] = tol
    return res


# ---------------------------------------------------------------------------
# optional cross-check against mlx-lm's independent implementation
# ---------------------------------------------------------------------------
MLX_LM_PATH = "/Users/true/.mlx128/mlx-lm"


def crosscheck_mlx(weights: LayerWeights, hidden_hc: np.ndarray,
                   mlx_lm_path: str = MLX_LM_PATH,
                   full_layer: bool = True) -> dict:
    """Load the SAME bf16 layer-0 tensors into mlx-lm's own modules (a separate
    codebase and framework) and compare against this numpy implementation.

    Runs in fp32 with the ops-based (non-Metal-kernel) recurrence path so the
    comparison is arithmetic, not kernel-precision.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if mlx_lm_path not in sys.path:
        sys.path.insert(0, mlx_lm_path)
    import mlx.core as mx           # noqa: E402
    from mlx_lm.models import qwen4_exp as qx   # noqa: E402

    mx.set_default_device(mx.cpu)   # force the ops-based recurrence path

    cfg = qx.TextConfig.from_dict(dict(weights.config))
    layer = qx.DecoderLayer(cfg, weights.index)
    layer.linear_attn.use_kernel = False

    I = cfg.moe_intermediate_size
    NORM_PLUS1 = ("hc_norm.weight",)
    params: Dict[str, mx.array] = {}
    for k, v in weights.tensors.items():
        if k == "mlp.experts.gate_up_proj":
            arr = np.stack([v[e] for e in range(cfg.num_experts)]) \
                if isinstance(v, ExpertSlab) else np.asarray(v)
            params["mlp.switch_mlp.gate_proj.weight"] = mx.array(arr[:, :I, :])
            params["mlp.switch_mlp.up_proj.weight"] = mx.array(arr[:, I:, :])
            continue
        if k == "mlp.experts.down_proj":
            arr = np.stack([v[e] for e in range(cfg.num_experts)]) \
                if isinstance(v, ExpertSlab) else np.asarray(v)
            params["mlp.switch_mlp.down_proj.weight"] = mx.array(arr)
            continue
        a = np.asarray(v, dtype=np.float32)
        if k.endswith("conv1d.weight") and a.ndim == 3 and a.shape[-1] != 1:
            a = np.moveaxis(a, 2, 1)                     # (C,1,K) -> (C,K,1)
        if any(k.endswith(s) for s in NORM_PLUS1) and a.ndim == 1:
            a = a + 1.0
        params[k] = mx.array(a)

    layer.load_weights(list(params.items()), strict=False)
    layer.eval()

    h = mx.array(hidden_hc.astype(np.float32))
    res: Dict[str, Any] = {}

    def rel(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        d = float(np.abs(a - b).max())
        s = max(float(np.abs(a).max()), 1e-30)
        return dict(max_abs_diff=d, max_rel_diff=d / s,
                    rms_ref=float(np.sqrt(np.mean(a.astype(np.float64) ** 2))))

    # --- piece 1: attention hyper-connection ---
    m_mixed, m_hyper, m_inj = layer.attn_hyper_connection(h)
    n_mixed, n_hyper, n_inj = gated_residual(
        weights, "attn_hyper_connection", hidden_hc, True)
    res["attn_hyper_connection.mixed"] = rel(n_mixed, np.array(m_mixed))
    res["attn_hyper_connection.inject"] = rel(n_inj, np.array(m_inj))

    # --- piece 2: gated delta net (from the mlx-lm mixed input, fp32) ---
    mixed_np = np.array(m_mixed, dtype=np.float32)
    m_gdn = layer.linear_attn(m_mixed, None, None)
    # mlx-lm's l2 form has eps scaled by D; match it for an apples-to-apples
    # arithmetic comparison, and also report the HF-eps variant's drift.
    n_gdn_mlxeps, _ = linear_attention_layer(weights, mixed_np, None,
                                             use_mlx_l2_eps=True)
    n_gdn_hfeps, _ = linear_attention_layer(weights, mixed_np, None,
                                            use_mlx_l2_eps=False)
    res["linear_attn.out (mlx l2 eps)"] = rel(n_gdn_mlxeps, np.array(m_gdn))
    res["linear_attn.out (hf l2 eps)"] = rel(n_gdn_hfeps, np.array(m_gdn))
    res["l2norm_eps_variant_drift"] = rel(n_gdn_hfeps, n_gdn_mlxeps)

    # --- piece 3: MoE block ---
    h2 = qx._recombine(m_gdn, m_hyper, m_inj)
    m_mixed2, m_hyper2, m_inj2 = layer.mlp_hyper_connection(h2)
    m_moe = layer.mlp(m_mixed2)
    n_moe = moe_layer(weights, np.array(m_mixed2, dtype=np.float32))
    res["mlp (MoE) out"] = rel(n_moe, np.array(m_moe))

    if full_layer:
        m_layer = layer(h, None, mask=None, cache=None)
        n_layer, _ = decoder_layer(weights, hidden_hc, None,
                                   use_mlx_l2_eps=True)
        res["full layer output"] = rel(n_layer, np.array(m_layer))

    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def seeded_hidden(cfg: dict, batch: int, seq: int, seed: int,
                  scale: float = 0.02) -> np.ndarray:
    """Fixed seeded hc-wide residual stream, shaped like the real one.

    The real stream entering layer 0 is concat([embeds] * hc_count), so this
    generates one hidden_size vector per token and tiles it hc_count times.
    Scale 0.02 matches config initializer_range and the observed embedding RMS.
    """
    rng = np.random.default_rng(seed)
    h = cfg["hidden_size"]
    hc = cfg["hc_count"]
    embeds = rng.standard_normal((batch, seq, h)).astype(np.float32) * scale
    return np.concatenate([embeds] * hc, axis=-1)


def real_embedding_hidden(loader: FlashNextLoader, token_ids: List[int]) -> np.ndarray:
    """hc-wide stream from the checkpoint's real embedding rows."""
    key = "model.language_model.embed_tokens.weight"
    shard = loader._shard(loader.weight_map[key])
    meta = shard.meta(key)
    vocab, h = meta["shape"]
    off0, _ = meta["data_offsets"]
    rows = []
    for tid in token_ids:
        a = off0 + tid * h * 2
        buf = shard._mm[shard.data_start + a: shard.data_start + a + h * 2]
        rows.append(bf16_to_fp32(np.frombuffer(buf, dtype=np.uint16)))
    embeds = np.stack(rows)[None, ...]
    hc = loader.text_config["hc_count"]
    return np.concatenate([embeds] * hc, axis=-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="CPU fp32 reference for one Qwen3.8-Flash-Next layer")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.02,
                    help="stddev of the seeded synthetic embedding")
    ap.add_argument("--tokens", type=str, default=None,
                    help="comma-separated real token ids; uses the checkpoint's "
                         "embedding rows instead of a synthetic input")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write per-stage statistics to this JSON file")
    ap.add_argument("--json-stdout", action="store_true",
                    help="print the statistics JSON to stdout")
    ap.add_argument("--no-step-check", action="store_true")
    ap.add_argument("--check-mlx", action="store_true",
                    help="cross-check against mlx-lm's implementation")
    ap.add_argument("--mlx-l2-eps", action="store_true",
                    help="use mlx-lm's l2norm eps placement instead of HF's")
    args = ap.parse_args(argv)

    loader = FlashNextLoader(args.model)
    cfg = loader.text_config
    print(f"model      : {args.model}")
    print(f"model_type : {cfg['model_type']}  layers={cfg['num_hidden_layers']}  "
          f"hidden={cfg['hidden_size']}  hc_count={cfg['hc_count']}")
    print(f"layer {args.layer}    : {cfg['layer_types'][args.layer]}")

    w = loader.layer(args.layer)
    print(f"tensors    : {len(w.tensors)} loaded for layer {args.layer}")
    for k in sorted(w.tensors):
        v = w.tensors[k]
        sh = tuple(v.shape)
        kind = "lazy-slab" if isinstance(v, ExpertSlab) else "fp32"
        print(f"  {k:<52} {str(sh):<22} {kind}")

    layout = infer_expert_layout(w, verbose=True)

    ple_ids = cfg.get("ple_layer_ids") or []
    if (args.layer + 1) in ple_ids:
        print(f"WARNING: layer {args.layer} carries a PLE / n-gram embedding "
              f"(ple_layer_ids={ple_ids}); this reference does NOT implement it.")

    if args.tokens:
        ids = [int(t) for t in args.tokens.split(",") if t.strip()]
        hidden = real_embedding_hidden(loader, ids)
        print(f"input      : real embedding rows for tokens {ids}")
    else:
        hidden = seeded_hidden(cfg, args.batch, args.seq, args.seed, args.scale)
        print(f"input      : seeded normal(0, {args.scale}) x hc, "
              f"seed={args.seed}, shape={hidden.shape}")

    stats = StatCollector(enabled=True)
    attn_cache = None
    if w.layer_type == "full_attention":
        attn_cache = AttnCache.empty(
            int(cfg["num_key_value_heads"]), int(cfg["head_dim"]),
            max_len=max(8, hidden.shape[1]))
        # decode-step reference: last token only
        hidden = hidden[:, -1:, :]
    out, state = decoder_layer(w, hidden, None, stats, layout,
                               use_mlx_l2_eps=args.mlx_l2_eps,
                               attn_cache=attn_cache)
    print()
    print(stats.table())
    print()

    payload = {
        "model": args.model,
        "layer": args.layer,
        "layer_type": w.layer_type,
        "input": {
            "kind": "real_tokens" if args.tokens else "seeded_normal",
            "tokens": args.tokens,
            "seed": args.seed,
            "scale": args.scale,
            "shape": list(hidden.shape),
        },
        "expert_layout": {k: v for k, v in layout.items()
                          if k != "notes"} | {"notes": layout["notes"]},
        "l2norm_eps": "mlx" if args.mlx_l2_eps else "hf",
        "stats": stats.stages,
    }

    if not args.no_step_check and w.layer_type == "linear_attention":
        mixed, _, _ = gated_residual(w, "attn_hyper_connection", hidden, True)
        chk = check_prefill_vs_step(w, mixed, use_mlx_l2_eps=args.mlx_l2_eps)
        payload["prefill_vs_step"] = chk
        print("prefill vs single-token step (self-consistency):")
        for k, v in chk.items():
            print(f"  {k:<34} {v}")
        print()

    if args.check_mlx:
        print("cross-check vs mlx-lm (independent implementation, MLX/CPU fp32):")
        try:
            cc = crosscheck_mlx(w, hidden)
            payload["crosscheck_mlx"] = cc
            for k, v in cc.items():
                print(f"  {k:<34} max_abs={v['max_abs_diff']:.4g}  "
                      f"max_rel={v['max_rel_diff']:.4g}  ref_rms={v['rms_ref']:.4g}")
        except Exception as exc:   # noqa: BLE001
            payload["crosscheck_mlx_error"] = repr(exc)
            print(f"  FAILED: {exc!r}")
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json_out}")
    if args.json_stdout:
        print(json.dumps(payload, indent=1))

    loader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
