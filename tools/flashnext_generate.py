#!/usr/bin/env python3
"""Full GPU-free Qwen3.8-Flash-Next generate on the Apple Neural Engine.

48 decoder layers, both linear_attention and full_attention, MoE, embed, and
lm_head. Shared compiled programs (one dynamic linear per unique GEMM shape,
one GDN step, one GQA core) stay under the 127-model loader budget. Per-layer
weights are paged onto those surfaces; 512-expert MoE pages the top-10 slabs.

Short context (≤ indexer_budget 2048) uses dense GQA — the same path mlx-lm
takes before the QSA indexer starts sparsifying. PLE/n-gram is a zero table
here (no ngram_index.json on disk), matching mlx-lm's fallback.

No MLX / PyTorch / Core ML.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import resource
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("Q38_ANE_ENGINE", str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.flashnext_ane import (  # noqa: E402
    DK,
    DV,
    H_GDN,
    WIDTH,
    FlashNextGdn,
    _compile_linear,
    _eval_lane0,
)
from tools.flashnext_reference import (  # noqa: E402
    DEFAULT_MODEL,
    AttnCache,
    FlashNextLoader,
    apply_rope,
    decoder_layer,
    grouped_rms_norm,
    hc_norm_weight,
    infer_expert_layout,
    l2norm_mlx,
    lm_logits,
    mixer_hidden,
    real_embedding_hidden,
    recombine,
    rms_norm,
    sigmoid,
    silu,
    softplus,
    zero_gdn_state,
)
from tools.pure_ane import AneDriver, StandaloneTokenizer, assert_standalone  # noqa: E402
from runtime.ane_baked_moe import compile_stacked_swiglu  # noqa: E402
from runtime.q38_ane_engine import (  # noqa: E402
    AneDynamicLinear,
    AneEngine,
    _iosurface_view,
)
from runtime.ane_fused_w8a8 import (  # noqa: E402
    TILE as FUSED_TILE,
    compile_in_proj_dw,
)

logging.getLogger("runtime.q38_ane_engine").setLevel(logging.ERROR)

HQ, HKV, HD = 24, 2, 256
ROPE_DIM = 64
ROPE_THETA = 10_000_000.0
ATTN_LEN = 256
HEAD_CHUNKS = 4
_CONVERT_POOL = ThreadPoolExecutor(max_workers=16)


def embed_row(loader: FlashNextLoader, token_id: int) -> np.ndarray:
    key = "model.language_model.embed_tokens.weight"
    return loader._shard(loader.weight_map[key]).fp32(key, expert=int(token_id))


class DynBank:
    """One compiled ANE program per (out, in) shape; page weights at eval."""

    def __init__(self, seq_len: int = WIDTH):
        self.seq_len = int(seq_len)
        self._progs: dict[tuple[int, int], AneDynamicLinear] = {}
        self._xbuf: dict[tuple[int, int], np.ndarray] = {}

    def get(self, out_dim: int, in_dim: int) -> AneDynamicLinear:
        key = (int(out_dim), int(in_dim))
        prog = self._progs.get(key)
        if prog is None:
            t0 = time.perf_counter()
            prog = AneDynamicLinear.compile(in_dim, out_dim, self.seq_len)
            if prog is None:
                raise RuntimeError(
                    f"dynamic linear {out_dim}x{in_dim} S={self.seq_len} compile failed")
            print(f"  dyn {out_dim}x{in_dim} S={self.seq_len}  "
                  f"{time.perf_counter() - t0:.2f}s", flush=True)
            self._progs[key] = prog
            self._xbuf[key] = np.zeros((self.seq_len, in_dim), np.float32)
        return prog

    def release(self) -> int:
        n = len(self._progs)
        self._progs.clear()
        self._xbuf.clear()
        return n

    def eval(self, weight: np.ndarray, x: np.ndarray) -> np.ndarray:
        w = np.ascontiguousarray(weight)
        o, i = int(w.shape[0]), int(w.shape[1])
        prog = self.get(o, i)
        prog.write_weight(w)
        return self.eval_staged(o, i, x)

    def eval_staged(self, out_dim: int, in_dim: int, x: np.ndarray) -> np.ndarray:
        """Run after the caller already wrote the weight surface."""
        prog = self.get(out_dim, in_dim)
        xv = np.asarray(x, np.float16).reshape(-1)
        with _iosurface_view(prog._x_surf, (in_dim, prog.seq_len), np.float16) as dst:
            dst[:, 0] = xv
        if not prog.submit():
            raise RuntimeError(f"dynamic staged eval failed {out_dim}x{in_dim}")
        with _iosurface_view(prog._y_surf, (out_dim, prog.seq_len), np.float16) as src:
            return np.array(src[:, 0], np.float32)

    def eval_batch(self, weight: np.ndarray, x: np.ndarray) -> np.ndarray:
        """x [S, I] -> [S, O], S <= seq_len. Same weight paged once."""
        x = np.asarray(x, np.float32)
        if x.ndim == 1:
            return self.eval(weight, x).reshape(1, -1)
        S = int(x.shape[0])
        if S > self.seq_len:
            raise ValueError(f"batch {S} exceeds compiled S={self.seq_len}")
        w = np.ascontiguousarray(weight)
        o, i = int(w.shape[0]), int(w.shape[1])
        prog = self.get(o, i)
        prog.write_weight(w)
        with _iosurface_view(prog._x_surf, (i, prog.seq_len), np.float16) as dst:
            dst[:] = 0
            dst[:, :S] = np.asarray(x, np.float16).T
        if not prog.submit():
            raise RuntimeError(f"dynamic batch eval failed {o}x{i}")
        with _iosurface_view(prog._y_surf, (o, prog.seq_len), np.float16) as src:
            return np.array(src[:, :S].T, np.float32)


_SKIP_F16 = (
    "ngram_heads_offsets",
    "ngram_heads_vocab_sizes",
    "layer_multipliers",
)
_FP16_MAX = float(np.finfo(np.float16).max)


def _to_f16(w):
    from tools.flashnext_reference import ExpertSlab
    out = {}
    for k, v in w.tensors.items():
        if isinstance(v, ExpertSlab) or any(s in k for s in _SKIP_F16):
            out[k] = v
        elif isinstance(v, np.ndarray):
            arr = np.asarray(v, np.float32)
            if arr.size and float(np.max(np.abs(arr))) > _FP16_MAX:
                arr = np.clip(arr, -_FP16_MAX, _FP16_MAX)
            out[k] = np.ascontiguousarray(arr, dtype=np.float16)
        else:
            out[k] = v
    w.tensors = out
    return w


class LruSlab:
    def __init__(self, slab, cap: int = 256):
        self.slab = slab
        self.cap = cap
        self._d: OrderedDict[int, np.ndarray] = OrderedDict()

    def __getitem__(self, e: int) -> np.ndarray:
        e = int(e)
        cached = self._d.get(e)
        if cached is not None:
            self._d.move_to_end(e)
            return cached
        w = np.ascontiguousarray(self.slab.f16(e) if hasattr(self.slab, "f16")
                                 else self.slab[e], dtype=np.float16)
        self._d[e] = w
        if len(self._d) > self.cap:
            self._d.popitem(last=False)
        return w


class GqaCore:
    """Shared dense GQA, Flash-Next shapes (24q / 2kv / d=256), cache L=256."""

    def __init__(self, driver: AneDriver, length: int = ATTN_LEN):
        self.driver = driver
        self.Hq, self.Hkv, self.D, self.L = HQ, HKV, HD, length
        H, K, D, L = self.Hq, self.Hkv, self.D, self.L
        self.input = H + 2 * K * L + 1
        C = self.input
        k0, k1 = H, H + K * L
        v0, v1 = k1, k1 + K * L
        m0 = v1
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {D}]> x) {{
    tensor<fp16, [1, {H}, 1, {D}]> q4 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{H},1,{D}]), x=x)[name=string("q4")];
    tensor<fp16, [1, {K * L}, 1, {D}]> kf = slice_by_index(begin=tensor<int32, [4]>([0,{k0},0,0]), end=tensor<int32, [4]>([1,{k1},1,{D}]), x=x)[name=string("kf")];
    tensor<fp16, [1, {K * L}, 1, {D}]> vf = slice_by_index(begin=tensor<int32, [4]>([0,{v0},0,0]), end=tensor<int32, [4]>([1,{v1},1,{D}]), x=x)[name=string("vf")];
    tensor<fp16, [1, 1, 1, {L}]> mask = slice_by_index(begin=tensor<int32, [4]>([0,{m0},0,0]), end=tensor<int32, [4]>([1,{m0 + 1},1,{L}]), x=x)[name=string("mask")];
    tensor<fp16, [1, {K}, {H // K}, {D}]> q = reshape(shape=tensor<int32, [4]>([1,{K},{H // K},{D}]), x=q4)[name=string("q")];
    tensor<fp16, [1, {K}, {L}, {D}]> k = reshape(shape=tensor<int32, [4]>([1,{K},{L},{D}]), x=kf)[name=string("k")];
    tensor<fp16, [1, {K}, {L}, {D}]> v = reshape(shape=tensor<int32, [4]>([1,{K},{L},{D}]), x=vf)[name=string("v")];
    tensor<fp16, [1, {K}, {H // K}, {L}]> rawg = matmul(transpose_x=bool(false), transpose_y=bool(true), x=q, y=k)[name=string("rawg")];
    tensor<fp16, [1, {H}, 1, {L}]> raw = reshape(shape=tensor<int32, [4]>([1,{H},1,{L}]), x=rawg)[name=string("raw")];
    tensor<fp16, [1, {H}, 1, {L}]> scaled = mul(x=raw, y=fp16(0x1p-4))[name=string("scaled")];
    tensor<fp16, [1, {H}, 1, {L}]> scores = add(x=scaled, y=mask)[name=string("scores")];
    tensor<fp16, [1, {H}, 1, {L}]> prob = softmax(axis=int32(-1), x=scores)[name=string("prob")];
    tensor<fp16, [1, {K}, {H // K}, {L}]> pg = reshape(shape=tensor<int32, [4]>([1,{K},{H // K},{L}]), x=prob)[name=string("pg")];
    tensor<fp16, [1, {K}, {H // K}, {D}]> yg = matmul(transpose_x=bool(false), transpose_y=bool(false), x=pg, y=v)[name=string("yg")];
    tensor<fp16, [1, {H}, 1, {D}]> y = reshape(shape=tensor<int32, [4]>([1,{H},1,{D}]), x=yg)[name=string("y")];
  }} -> (y);
}}
// flashnext_gqa_L{L}
'''
        cap = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
            self.program = driver.engine.compile_multiproc(mil, {}, C, H, D)
        if self.program is None:
            tail = "\n".join(cap.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"GQA core compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        print(f"  compile gqa L={L}                         "
              f"{time.perf_counter() - t0:.2f}s", flush=True)

    def fork(self) -> "GqaCache":
        return GqaCache(self)


class GqaCache:
    def __init__(self, core: GqaCore):
        self.core = core
        self.keys = np.zeros((core.Hkv, core.L, core.D), np.float16)
        self.values = np.zeros((core.Hkv, core.L, core.D), np.float16)
        self.offset = 0

    def reset(self) -> None:
        self.keys[:] = 0
        self.values[:] = 0
        self.offset = 0

    def __call__(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        c = self.core
        if self.offset >= c.L:
            raise RuntimeError(f"attention cache exceeds {c.L}")
        self.keys[:, self.offset] = k
        self.values[:, self.offset] = v
        valid = self.offset + 1
        with c.driver.view(c.program._in_surf, (c.input, c.D), np.float16) as dst:
            dst[:] = 0
            dst[:c.Hq] = q
            p = c.Hq
            dst[p:p + c.Hkv * c.L] = self.keys.reshape(-1, c.D)
            p += c.Hkv * c.L
            dst[p:p + c.Hkv * c.L] = self.values.reshape(-1, c.D)
            p += c.Hkv * c.L
            dst[p, valid:c.L] = np.float16(-1e4)
        if not c.driver.engine.submit(c.program, procedure_index=0):
            raise RuntimeError("GQA submit failed")
        with c.driver.view(c.program._out_surf, (c.Hq, c.D), np.float16) as src:
            out = np.array(src, np.float32)
        self.offset = valid
        return out


def _tokens(hidden_hc: np.ndarray) -> np.ndarray:
    """Normalize residual stream to [S, hc*H]."""
    x = np.asarray(hidden_hc, np.float32)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 3:
        return x.reshape(x.shape[1], x.shape[-1]) if x.shape[0] == 1 else x.reshape(-1, x.shape[-1])
    return x


def _hyper(dyn: DynBank, w, prefix: str, hyper_input: np.ndarray,
           cfg: dict, combine: bool = True, *, batch: bool | None = None):
    h = int(cfg["hidden_size"])
    hc = int(cfg["hc_count"])
    eps = float(cfg["rms_norm_eps"])
    low = int(cfg["hc_lowrank"])
    seq = _tokens(hyper_input)
    S = seq.shape[0]
    if batch is None:
        batch = S > 1
    normed = grouped_rms_norm(
        seq, hc_norm_weight(w[f"{prefix}.hc_norm.weight"]),
        group_size=h, eps=eps,
    )
    fused = w.get(f"{prefix}.mix_down_inject.weight")
    gemm = dyn.eval_batch if batch else dyn.eval
    if fused is not None and combine:
        both = gemm(fused, normed if batch else normed.reshape(-1))
        if not batch:
            both = both.reshape(1, -1)
        down, raw_inj = both[:, :low], both[:, low:]
    else:
        down = gemm(w[f"{prefix}.input_mix_weight_down.weight"],
                    normed if batch else normed.reshape(-1))
        if not batch:
            down = down.reshape(1, -1)
        raw_inj = None
    gate = silu(down / hc)
    up = gemm(w[f"{prefix}.input_mix_weight_up.weight"], gate if batch else gate.reshape(-1))
    if not batch:
        up = up.reshape(1, -1)
    mix_w = sigmoid(up).reshape(S, hc, h)
    mixed = (mix_w * normed.reshape(S, hc, h)).mean(axis=1).astype(np.float32)
    if not combine:
        return mixed if batch else mixed.reshape(-1)
    if raw_inj is None:
        raw_inj = gemm(w[f"{prefix}.block_inject_weight.weight"],
                       normed if batch else normed.reshape(-1))
        if not batch:
            raw_inj = raw_inj.reshape(1, -1)
    inj = (2.0 * sigmoid(raw_inj / hc)).astype(np.float32)
    if not batch:
        return mixed.reshape(-1), hyper_input, inj.reshape(-1)
    return mixed, seq, inj


class PackedMoe:
    """Stage top-k experts into two stacked GEMMs (Ling strategy A).

    One gate_up [k*2I, H] and one down [H, k*I] instead of 2k separate
    evaluates. Same bytes moved, ~10x fewer ANE submits.
    """

    def __init__(self, dyn: DynBank, top_k: int, layout: dict, *,
                 n_experts: int = 512):
        self.dyn = dyn
        self.top_k = top_k
        self.layout = layout
        self.n_experts = int(n_experts)
        self.I = int(layout["I"])
        self.H = int(layout["H"])
        self.two_i = 2 * self.I
        self.gu_o = top_k * self.two_i
        self.dn_i = top_k * self.I
        self.counts = np.zeros(self.n_experts, np.int64)
        self.recent = np.zeros(self.n_experts, np.int64)
        self.baked = None
        self.record_route = False

    def __call__(self, hidden: np.ndarray, w) -> np.ndarray:
        x = np.asarray(hidden, np.float32).reshape(-1)
        if self.baked is not None:
            if self.record_route:
                self._record_cpu(x, w)
            y = self._eval_baked(x)
            if not self.baked.get("shared"):
                y = y + self._shared_dyn(x, w)
            return y
        logits = self.dyn.eval(w["mlp.gate.weight"], x)
        return self._finish(x, logits, w)

    def _record_cpu(self, x, w) -> None:
        g = np.asarray(w["mlp.gate.weight"], np.float32)
        logits = g @ np.asarray(x, np.float32).reshape(-1)
        m = float(np.max(logits))
        e = np.exp((logits - m).astype(np.float64))
        probs = (e / e.sum()).astype(np.float32)
        inds = np.argpartition(probs, -self.top_k)[-self.top_k:]
        self.recent[np.asarray(inds, np.int64)] += 1

    def _shared_dyn(self, x: np.ndarray, w) -> np.ndarray:
        gu_sh = w.get("mlp.shared_expert.gate_up_proj.weight")
        if gu_sh is not None:
            fused_sh = self.dyn.eval(gu_sh, x)
            shared_h = silu(fused_sh[:self.I]) * fused_sh[self.I:]
        else:
            sg = self.dyn.eval(w["mlp.shared_expert.gate_proj.weight"], x)
            su = self.dyn.eval(w["mlp.shared_expert.up_proj.weight"], x)
            shared_h = silu(sg) * su
        shared = self.dyn.eval(w["mlp.shared_expert.down_proj.weight"], shared_h)
        sgate = sigmoid(self.dyn.eval(w["mlp.shared_expert_gate.weight"], x))
        return sgate * shared

    def _eval_baked(self, x: np.ndarray) -> np.ndarray:
        runner = self.baked["runner"]
        buf = np.zeros((self.H, runner.seq), np.float32)
        xv = np.asarray(x, np.float32)
        if xv.ndim == 1:
            buf[:, 0] = xv
            got = runner.run(buf)
            if got is None:
                raise RuntimeError("baked swiglu evaluate failed")
            y = np.asarray(got["y"][:, 0], np.float32)
            if "sh" in got:
                sg = float(got["sg"][0, 0])
                y = y + (1.0 / (1.0 + np.exp(-sg))) * np.asarray(got["sh"][:, 0], np.float32)
            return y
        S = xv.shape[0]
        buf[:, :S] = xv.T
        got = runner.run(buf)
        if got is None:
            raise RuntimeError("baked swiglu batch evaluate failed")
        y = np.asarray(got["y"][:, :S].T, np.float32)
        if "sh" in got:
            sg = 1.0 / (1.0 + np.exp(-np.asarray(got["sg"][0, :S], np.float32)))
            y = y + sg[:, None] * np.asarray(got["sh"][:, :S].T, np.float32)
        return y

    def prefill(self, hidden: np.ndarray, w, dyn: DynBank) -> np.ndarray:
        """Router + shared expert at tile width; routed experts still per token."""
        x = np.asarray(hidden, np.float32)
        if x.ndim == 1:
            return self(x, w)
        logits = dyn.eval_batch(w["mlp.gate.weight"], x)
        routed = np.empty((x.shape[0], self.H), np.float32)
        for t in range(x.shape[0]):
            routed[t] = self._routed(x[t], logits[t], w)
        gu_sh = w.get("mlp.shared_expert.gate_up_proj.weight")
        if gu_sh is not None:
            fused_sh = dyn.eval_batch(gu_sh, x)
            shared_h = silu(fused_sh[:, :self.I]) * fused_sh[:, self.I:]
        else:
            sg = dyn.eval_batch(w["mlp.shared_expert.gate_proj.weight"], x)
            su = dyn.eval_batch(w["mlp.shared_expert.up_proj.weight"], x)
            shared_h = silu(sg) * su
        shared = dyn.eval_batch(w["mlp.shared_expert.down_proj.weight"], shared_h)
        sgate = sigmoid(dyn.eval_batch(w["mlp.shared_expert_gate.weight"], x))
        return routed + sgate * shared

    def _finish(self, x: np.ndarray, logits: np.ndarray, w) -> np.ndarray:
        routed = self._routed(x, logits, w)
        gu_sh = w.get("mlp.shared_expert.gate_up_proj.weight")
        if gu_sh is not None:
            fused_sh = self.dyn.eval(gu_sh, x)
            shared_h = silu(fused_sh[:self.I]) * fused_sh[self.I:]
        else:
            sg = self.dyn.eval(w["mlp.shared_expert.gate_proj.weight"], x)
            su = self.dyn.eval(w["mlp.shared_expert.up_proj.weight"], x)
            shared_h = silu(sg) * su
        shared = self.dyn.eval(w["mlp.shared_expert.down_proj.weight"], shared_h)
        sgate = sigmoid(self.dyn.eval(w["mlp.shared_expert_gate.weight"], x))
        return routed + sgate * shared

    def _routed(self, x: np.ndarray, logits: np.ndarray, w) -> np.ndarray:
        m = float(np.max(logits))
        e = np.exp((logits - m).astype(np.float64))
        probs = (e / e.sum()).astype(np.float32)
        if self.baked is not None:
            return self._routed_baked(x, probs)
        k = self.top_k
        inds = np.argpartition(probs, -k)[-k:]
        scores = probs[inds]
        scores = scores / scores.sum()
        self.counts[np.asarray(inds, np.int64)] += 1

        gu_slab = w["mlp.experts.gate_up_proj"]
        dn_slab = w["mlp.experts.down_proj"]
        eids = [int(e) for e in inds.tolist()]
        gu_futs = [_CONVERT_POOL.submit(gu_slab.__getitem__, e) for e in eids]
        dn_futs = [_CONVERT_POOL.submit(dn_slab.__getitem__, e) for e in eids]
        gu_parts = [f.result() for f in gu_futs]
        gu_prog = self.dyn.get(self.gu_o, self.H)
        with _iosurface_view(gu_prog._w_surf, (self.gu_o, self.H), np.float16) as dst:
            for j, slab in enumerate(gu_parts):
                if self.layout["gate_up"] != "E_2I_H":
                    slab = np.ascontiguousarray(slab.T, dtype=np.float16)
                dst[j * self.two_i:(j + 1) * self.two_i] = slab
        fused = self.dyn.eval_staged(self.gu_o, self.H, x)

        hcat = np.empty(self.dn_i, np.float32)
        for j, sc in enumerate(scores.tolist()):
            sl = fused[j * self.two_i:(j + 1) * self.two_i]
            hcat[j * self.I:(j + 1) * self.I] = sc * silu(sl[:self.I]) * sl[self.I:]

        dn_parts = [f.result() for f in dn_futs]
        dn_prog = self.dyn.get(self.H, self.dn_i)
        with _iosurface_view(dn_prog._w_surf, (self.H, self.dn_i), np.float16) as dst:
            for j, slab in enumerate(dn_parts):
                if self.layout["down"] != "E_H_I":
                    slab = np.ascontiguousarray(slab.T, dtype=np.float16)
                dst[:, j * self.I:(j + 1) * self.I] = slab
        return self.dyn.eval_staged(self.H, self.dn_i, hcat)

    def _routed_baked(self, x: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self._eval_baked(x)

    def bake(self, eng, w, k_keep: int, tag: str, counts=None) -> int:
        src = self.counts if counts is None else counts
        nz = np.flatnonzero(src)
        if nz.size == 0:
            if counts is not None:
                return 0
            raise RuntimeError(f"{tag}: no routed experts recorded during prefill")
        if nz.size <= k_keep:
            eids = [int(e) for e in sorted(nz.tolist())]
        else:
            eids = [int(e) for e in np.sort(np.argsort(src)[-k_keep:]).tolist()]
        if self.baked is not None:
            old = set(self.baked.get("eids") or [])
            if set(eids) <= old:
                return 0
        k = len(eids)
        gu_slab = w["mlp.experts.gate_up_proj"]
        dn_slab = w["mlp.experts.down_proj"]
        gu_parts = []
        for e in eids:
            slab = np.asarray(gu_slab[e], np.float32)
            if self.layout["gate_up"] != "E_2I_H":
                slab = np.ascontiguousarray(slab.T, dtype=np.float32)
            gu_parts.append(slab)
        gu = np.concatenate(gu_parts, axis=0).reshape(k, self.two_i, self.H)
        gate = np.ascontiguousarray(gu[:, :self.I, :].reshape(k * self.I, self.H))
        up = np.ascontiguousarray(gu[:, self.I:, :].reshape(k * self.I, self.H))
        dn_parts = []
        for e in eids:
            slab = np.asarray(dn_slab[e], np.float32)
            if self.layout["down"] != "E_H_I":
                slab = np.ascontiguousarray(slab.T, dtype=np.float32)
            dn_parts.append(slab)
        dn = np.ascontiguousarray(np.concatenate(dn_parts, axis=1) / float(k), np.float32)
        sh_gu = w.get("mlp.shared_expert.gate_up_proj.weight")
        kw = dict(seq=WIDTH, tag=tag)
        if sh_gu is not None:
            sh = np.asarray(sh_gu, np.float32)
            kw.update(
                shared_gate=sh[:self.I],
                shared_up=sh[self.I:],
                shared_down=np.asarray(w["mlp.shared_expert.down_proj.weight"], np.float32),
                sgate=np.asarray(w["mlp.shared_expert_gate.weight"], np.float32),
            )
        try:
            runner = compile_stacked_swiglu(eng, gate, up, dn, **kw)
        except RuntimeError as exc:
            if "shared_gate" not in kw:
                raise
            print(f"    {tag} shared fold failed ({exc}); retry routed-only",
                  flush=True)
            kw = dict(seq=WIDTH, tag=tag)
            runner = compile_stacked_swiglu(eng, gate, up, dn, **kw)
        self.baked = {
            "eids": eids,
            "runner": runner,
            "eng": eng,
            "shared": "shared_gate" in kw,
        }
        self.recent[:] = 0
        return k


def _eval_rows(eng, prog, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        return _eval_lane0(eng, prog, x)
    S = x.shape[0]
    buf = np.zeros((prog.seq_len, prog.input_dim), np.float32)
    buf[:S] = x
    y = eng.evaluate(prog, buf)
    if y is None:
        raise RuntimeError("ANE evaluate returned None")
    return np.asarray(y[:S], np.float32)


def _compile_linear_q(eng, weight: np.ndarray, tag: str, seq_len: int = WIDTH):
    """Constexpr int8 linear. Decode S=32 is latency-bound; int8 still halves blob bytes."""
    t0 = time.perf_counter()
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        prog = eng.compile_linear(
            np.ascontiguousarray(weight, dtype=np.float32),
            seq_len,
            quantized=True,
            keep_weight_dequant=False,
        )
    if prog is None:
        tail = "\n".join(cap.getvalue().strip().splitlines()[-8:])
        raise RuntimeError(f"ANE int8 compile failed for {tag}:\n{tail}")
    print(f"  bake {tag:<36} {tuple(weight.shape)}  "
          f"{time.perf_counter() - t0:.2f}s", flush=True)
    return prog


def cpu_conv_silu(qkv: np.ndarray, cache: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal K=4 depthwise conv + SiLU. Same arithmetic as FlashNextConv."""
    x = np.asarray(qkv, np.float32).reshape(-1)
    w = np.asarray(weight, np.float32).reshape(x.shape[0], 4)
    stream = np.concatenate([cache, x[:, None]], axis=1)
    y = (stream * w).sum(axis=1)
    new_cache = stream[:, -3:].astype(np.float16)
    return silu(y), new_cache


class LinearBlock:
    def __init__(self, driver: AneDriver, dyn: DynBank, gdn: FlashNextGdn,
                 moe: PackedMoe, weights, cfg: dict, *,
                 dyn_pre: DynBank | None = None, fused=None):
        self.dyn = dyn
        self.dyn_pre = dyn_pre or dyn
        self.gdn = gdn
        self.moe = moe
        self.w = weights
        self.cfg = cfg
        self.fused = fused
        self.baked_in = None
        self.baked_out = None
        self.conv_w = np.asarray(weights["linear_attn.conv1d.weight"], np.float16)
        self.conv_cache = np.zeros((int(self.conv_w.shape[0]), 3), np.float16)
        self.A_log = np.asarray(weights["linear_attn.A_log"], np.float32)
        self.dt_bias = np.asarray(weights["linear_attn.dt_bias"], np.float32)
        self.norm_w = np.asarray(weights["linear_attn.norm.weight"], np.float32)
        self.eps = float(cfg["rms_norm_eps"])
        self.state = None
        self.qkv_dim = int(self.conv_w.shape[0])
        self.z_dim = H_GDN * DV
        self.fused_ms = 0.0
        self.fused_tops = 0.0

    def reset(self) -> None:
        self.conv_cache[:] = 0
        self.state = self.gdn.new_state()

    def _gdn_from_proj(self, qkv, z, a, b, conv_out):
        z = z.reshape(H_GDN, DV)
        kd = 16 * DK
        q_raw = conv_out[:kd].reshape(16, DK)
        k_raw = conv_out[kd:2 * kd].reshape(16, DK)
        v = conv_out[2 * kd:].reshape(H_GDN, DV)
        inv = DK ** -0.5
        q48 = np.repeat(l2norm_mlx(q_raw) * inv, 3, axis=0)
        k48 = np.repeat(l2norm_mlx(k_raw), 3, axis=0)
        beta = sigmoid(b)
        decay = np.exp(-np.exp(self.A_log) * softplus(a + self.dt_bias)).astype(np.float32)
        y = self.gdn(self.state, q48, k48, v, decay, beta)
        gated = (rms_norm(y, self.norm_w, self.eps) * sigmoid(z)).reshape(-1)
        return gated

    def _attn(self, mixed: np.ndarray) -> np.ndarray:
        w = self.w
        fused = w.get("linear_attn.in_proj_fused.weight")
        if self.baked_in is not None:
            p = _eval_rows(self.baked_in[0], self.baked_in[1], mixed)
            if p.ndim == 1:
                qkv = p[:self.qkv_dim]
                z = p[self.qkv_dim:self.qkv_dim + self.z_dim]
                b = p[self.qkv_dim + self.z_dim:self.qkv_dim + self.z_dim + H_GDN]
                a = p[self.qkv_dim + self.z_dim + H_GDN:]
            else:
                qkv = p[:, :self.qkv_dim]
                z = p[:, self.qkv_dim:self.qkv_dim + self.z_dim]
                b = p[:, self.qkv_dim + self.z_dim:self.qkv_dim + self.z_dim + H_GDN]
                a = p[:, self.qkv_dim + self.z_dim + H_GDN:]
        elif fused is not None:
            p = self.dyn.eval(fused, mixed)
            qkv = p[:self.qkv_dim]
            z = p[self.qkv_dim:self.qkv_dim + self.z_dim]
            b = p[self.qkv_dim + self.z_dim:self.qkv_dim + self.z_dim + H_GDN]
            a = p[self.qkv_dim + self.z_dim + H_GDN:]
        else:
            qkv = self.dyn.eval(w["linear_attn.in_proj_qkv.weight"], mixed)
            z = self.dyn.eval(w["linear_attn.in_proj_z.weight"], mixed)
            a = self.dyn.eval(w["linear_attn.in_proj_a.weight"], mixed)
            b = self.dyn.eval(w["linear_attn.in_proj_b.weight"], mixed)
        conv_out, self.conv_cache = cpu_conv_silu(qkv, self.conv_cache, self.conv_w)
        gated = self._gdn_from_proj(qkv, z, a, b, conv_out)
        if self.baked_out is not None:
            return _eval_rows(self.baked_out[0], self.baked_out[1], gated)
        return self.dyn.eval(w["linear_attn.out_proj.weight"], gated)

    def step(self, hidden_hc: np.ndarray) -> np.ndarray:
        mixed, hyper_in, inj = _hyper(
            self.dyn, self.w, "attn_hyper_connection", hidden_hc, self.cfg)
        r = self._attn(mixed)
        hidden_hc = recombine(r.reshape(1, 1, -1), hyper_in, inj.reshape(1, 1, -1))
        mixed, hyper_in, inj = _hyper(
            self.dyn, self.w, "mlp_hyper_connection", hidden_hc, self.cfg)
        r = self.moe(mixed, self.w)
        return recombine(r.reshape(1, 1, -1), hyper_in, inj.reshape(1, 1, -1))

    def prefill(self, hidden_hc: np.ndarray) -> np.ndarray:
        dyn = self.dyn_pre
        mixed, hyper_in, inj = _hyper(
            dyn, self.w, "attn_hyper_connection", hidden_hc, self.cfg, batch=True)
        mixed = np.asarray(mixed, np.float32)
        if mixed.ndim == 1:
            mixed = mixed.reshape(1, -1)
        S = mixed.shape[0]
        w = self.w
        if self.fused is not None:
            got = self.fused.run_seq(mixed)
            self.fused_ms = self.fused.runner.last_ms
            self.fused_tops = self.fused.tops()
            qkv = got["y_qkv"].T
            z = got["y_z"].T
            a = got["y_a"].T
            b = got["y_b"].T
            conv_pre = got["y_dw"].T
            conv_out = silu(conv_pre)
            prev = np.zeros((self.qkv_dim, 3), np.float32)
            stream = np.concatenate([prev, qkv.T], axis=1)
            self.conv_cache = stream[:, -3:].astype(np.float16)
        else:
            fused_w = w.get("linear_attn.in_proj_fused.weight")
            if fused_w is not None:
                p = dyn.eval_batch(fused_w, mixed)
                qkv = p[:, :self.qkv_dim]
                z = p[:, self.qkv_dim:self.qkv_dim + self.z_dim]
                b = p[:, self.qkv_dim + self.z_dim:self.qkv_dim + self.z_dim + H_GDN]
                a = p[:, self.qkv_dim + self.z_dim + H_GDN:]
            else:
                qkv = dyn.eval_batch(w["linear_attn.in_proj_qkv.weight"], mixed)
                z = dyn.eval_batch(w["linear_attn.in_proj_z.weight"], mixed)
                a = dyn.eval_batch(w["linear_attn.in_proj_a.weight"], mixed)
                b = dyn.eval_batch(w["linear_attn.in_proj_b.weight"], mixed)
            conv_out = np.empty_like(qkv)
            for t in range(S):
                conv_out[t], self.conv_cache = cpu_conv_silu(
                    qkv[t], self.conv_cache, self.conv_w)
        gated = np.empty((S, int(w["linear_attn.out_proj.weight"].shape[1])), np.float32)
        for t in range(S):
            gated[t] = self._gdn_from_proj(qkv[t], z[t], a[t], b[t], conv_out[t])
        r = dyn.eval_batch(w["linear_attn.out_proj.weight"], gated)
        hidden_hc = recombine(r, hyper_in, inj)
        mixed, hyper_in, inj = _hyper(
            dyn, self.w, "mlp_hyper_connection", hidden_hc, self.cfg, batch=True)
        r = self.moe.prefill(mixed, self.w, dyn)
        return recombine(r, hyper_in, inj)


class FullAttnBlock:
    def __init__(self, dyn: DynBank, cache: GqaCache, moe: PackedMoe,
                 weights, cfg: dict, *, dyn_pre: DynBank | None = None):
        self.dyn = dyn
        self.dyn_pre = dyn_pre or dyn
        self.cache = cache
        self.moe = moe
        self.w = weights
        self.cfg = cfg
        self.eps = float(cfg["rms_norm_eps"])
        self.q_nw = hc_norm_weight(weights["self_attn.q_norm.weight"])
        self.k_nw = hc_norm_weight(weights["self_attn.k_norm.weight"])
        self.baked_q = None
        self.baked_kv = None
        self.baked_o = None

    def reset(self) -> None:
        self.cache.reset()

    def _attn(self, mixed: np.ndarray) -> np.ndarray:
        w = self.w
        if self.baked_q is not None:
            qg = _eval_rows(self.baked_q[0], self.baked_q[1], mixed).reshape(HQ, 2 * HD)
        else:
            qg = self.dyn.eval(w["self_attn.q_proj.weight"], mixed).reshape(HQ, 2 * HD)
        q, gate = qg[:, :HD], qg[:, HD:]
        kvw = w.get("self_attn.kv_proj.weight")
        if self.baked_kv is not None:
            kv = _eval_rows(self.baked_kv[0], self.baked_kv[1], mixed)
            k = kv[:HKV * HD].reshape(HKV, HD)
            v = kv[HKV * HD:].reshape(HKV, HD)
        elif kvw is not None:
            kv = self.dyn.eval(kvw, mixed)
            k = kv[:HKV * HD].reshape(HKV, HD)
            v = kv[HKV * HD:].reshape(HKV, HD)
        else:
            k = self.dyn.eval(w["self_attn.k_proj.weight"], mixed).reshape(HKV, HD)
            v = self.dyn.eval(w["self_attn.v_proj.weight"], mixed).reshape(HKV, HD)
        q = rms_norm(q, self.q_nw, self.eps)
        k = rms_norm(k, self.k_nw, self.eps)
        pos = self.cache.offset
        q = apply_rope(q, pos)
        k = apply_rope(k, pos)
        y = self.cache(q, k, v)
        gated = (y * sigmoid(gate)).reshape(-1)
        if self.baked_o is not None:
            return _eval_rows(self.baked_o[0], self.baked_o[1], gated)
        return self.dyn.eval(w["self_attn.o_proj.weight"], gated)

    def step(self, hidden_hc: np.ndarray) -> np.ndarray:
        mixed, hyper_in, inj = _hyper(
            self.dyn, self.w, "attn_hyper_connection", hidden_hc, self.cfg)
        r = self._attn(mixed)
        hidden_hc = recombine(r.reshape(1, 1, -1), hyper_in, inj.reshape(1, 1, -1))
        mixed, hyper_in, inj = _hyper(
            self.dyn, self.w, "mlp_hyper_connection", hidden_hc, self.cfg)
        r = self.moe(mixed, self.w)
        return recombine(r.reshape(1, 1, -1), hyper_in, inj.reshape(1, 1, -1))

    def prefill(self, hidden_hc: np.ndarray) -> np.ndarray:
        dyn = self.dyn_pre
        w = self.w
        mixed, hyper_in, inj = _hyper(
            dyn, w, "attn_hyper_connection", hidden_hc, self.cfg, batch=True)
        S = mixed.shape[0]
        qg = dyn.eval_batch(w["self_attn.q_proj.weight"], mixed).reshape(S, HQ, 2 * HD)
        q, gate = qg[:, :, :HD], qg[:, :, HD:]
        kvw = w.get("self_attn.kv_proj.weight")
        if kvw is not None:
            kv = dyn.eval_batch(kvw, mixed)
            k = kv[:, :HKV * HD].reshape(S, HKV, HD)
            v = kv[:, HKV * HD:].reshape(S, HKV, HD)
        else:
            k = dyn.eval_batch(w["self_attn.k_proj.weight"], mixed).reshape(S, HKV, HD)
            v = dyn.eval_batch(w["self_attn.v_proj.weight"], mixed).reshape(S, HKV, HD)
        y = np.empty((S, HQ, HD), np.float32)
        for t in range(S):
            qt = rms_norm(q[t], self.q_nw, self.eps)
            kt = rms_norm(k[t], self.k_nw, self.eps)
            pos = self.cache.offset
            qt = apply_rope(qt, pos)
            kt = apply_rope(kt, pos)
            y[t] = self.cache(qt, kt, v[t])
        gated = (y * sigmoid(gate)).reshape(S, -1)
        r = dyn.eval_batch(w["self_attn.o_proj.weight"], gated)
        hidden_hc = recombine(r, hyper_in, inj)
        mixed, hyper_in, inj = _hyper(
            dyn, w, "mlp_hyper_connection", hidden_hc, self.cfg, batch=True)
        r = self.moe.prefill(mixed, w, dyn)
        return recombine(r, hyper_in, inj)


class FinalHead:
    """RMSNorm of mixed hidden + chunked vocabulary projection on ANE."""

    def __init__(self, eng: AneEngine, weight: np.ndarray, chunks: int = HEAD_CHUNKS):
        self.eng = eng
        self.V, self.H = weight.shape
        step = -(-self.V // chunks)
        self.progs = []
        self.spans = []
        for v0 in range(0, self.V, step):
            v1 = min(self.V, v0 + step)
            tag = f"lm_head[{v0}:{v1}]"
            self.progs.append(_compile_linear(eng, weight[v0:v1], tag))
            self.spans.append((v0, v1))

    def __call__(self, hidden: np.ndarray) -> np.ndarray:
        logits = np.empty(self.V, np.float32)
        for prog, (v0, v1) in zip(self.progs, self.spans):
            logits[v0:v1] = _eval_lane0(self.eng, prog, hidden)
        return logits


class FlashNextCpuModel:
    """fp32-math / fp16-weight decode through all 48 layers. No ANE."""

    def __init__(self, model_path: str):
        self.loader = FlashNextLoader(model_path)
        self.cfg = self.loader.text_config
        self.hc = int(self.cfg["hc_count"])
        self.H = int(self.cfg["hidden_size"])
        self.n_layers = int(self.cfg["num_hidden_layers"])
        self.types = list(self.cfg["layer_types"])
        print("loading layers (cpu)", flush=True)
        layout = None
        self.weights = []
        for i, lt in enumerate(self.types):
            w = _to_f16(self.loader.layer(i))
            if layout is None:
                layout = infer_expert_layout(w, verbose=True)
            w.tensors["mlp.experts.gate_up_proj"] = LruSlab(
                w.tensors["mlp.experts.gate_up_proj"])
            w.tensors["mlp.experts.down_proj"] = LruSlab(
                w.tensors["mlp.experts.down_proj"])
            self.weights.append(w)
            print(f"  layer {i:02d} {lt}  "
                  f"rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f}",
                  flush=True)
        self.layout = layout
        self.reset()

    def reset(self) -> None:
        self.gdn_states = [
            zero_gdn_state(self.cfg) if lt == "linear_attention" else None
            for lt in self.types
        ]
        hkv = int(self.cfg["num_key_value_heads"])
        hd = int(self.cfg["head_dim"])
        self.attn_caches = [
            AttnCache.empty(hkv, hd, ATTN_LEN) if lt == "full_attention" else None
            for lt in self.types
        ]

    def step_token(self, token_id: int) -> np.ndarray:
        h = real_embedding_hidden(self.loader, [int(token_id)])
        for i, w in enumerate(self.weights):
            h, extra = decoder_layer(
                w, h, self.gdn_states[i], layout=self.layout,
                use_mlx_l2_eps=True, attn_cache=self.attn_caches[i],
            )
            if self.types[i] == "linear_attention":
                self.gdn_states[i] = extra
        mixed = mixer_hidden(self.loader, h)
        return lm_logits(self.loader, mixed).reshape(-1)

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        self.reset()
        h = real_embedding_hidden(self.loader, [int(t) for t in token_ids])
        for i, w in enumerate(self.weights):
            h, extra = decoder_layer(
                w, h, None, layout=self.layout,
                use_mlx_l2_eps=True, attn_cache=self.attn_caches[i],
            )
            if self.types[i] == "linear_attention":
                self.gdn_states[i] = extra
        mixed = mixer_hidden(self.loader, h)
        logits = lm_logits(self.loader, mixed)
        return np.asarray(logits, np.float32).reshape(-1, logits.shape[-1])[-1]


def generate_loop(model, token_ids: list[int], max_new: int,
                  eos_ids: set[int]) -> list[int]:
    model.reset()
    out = list(token_ids)
    t_pre = time.perf_counter()
    if hasattr(model, "prefill"):
        logits = model.prefill(token_ids)
        last_dt = time.perf_counter() - t_pre
        print(f"  prefill S={len(token_ids)}  argmax={int(np.argmax(logits))}  "
              f"{last_dt:.3f}s  ({len(token_ids) / max(last_dt, 1e-9):.2f} tok/s)",
              flush=True)
    else:
        logits = None
        last_dt = 0.0
        for i, tid in enumerate(token_ids):
            t1 = time.perf_counter()
            logits = model.step_token(tid)
            last_dt = time.perf_counter() - t1
            print(f"  prefill {i + 1}/{len(token_ids)} id={tid}  "
                  f"argmax={int(np.argmax(logits))}  {last_dt:.3f}s", flush=True)
        print(f"  prefill done  {time.perf_counter() - t_pre:.2f}s", flush=True)
    if getattr(model, "pin_k", 0):
        t_b = time.perf_counter()
        model.bake_pinned()
        print(f"  bake done  {time.perf_counter() - t_b:.1f}s", flush=True)
    mtp_k = int(getattr(model, "mtp_draft", 0) or 0)
    n = 0
    n_accept = 0
    n_cycle = 0
    decode_s = []
    last_dt = None
    while n < max_new:
        nxt = int(np.argmax(logits))
        if nxt in eos_ids:
            out.append(nxt)
            print(f"  gen {n + 1}/{max_new} id={nxt}  eos", flush=True)
            break
        if mtp_k >= 2 and hasattr(model, "snapshot") and (n + 1) < max_new:
            drafts = model.draft(nxt, min(mtp_k - 1, max_new - n - 1))
            model.snapshot()
            t1 = time.perf_counter()
            logits = model.step_token(nxt)
            last_dt = time.perf_counter() - t1
            got = [nxt]
            for d in drafts:
                if int(np.argmax(logits)) != int(d):
                    break
                t1 = time.perf_counter()
                logits = model.step_token(d)
                last_dt += time.perf_counter() - t1
                got.append(d)
            n_cycle += 1
            n_accept += len(got)
            decode_s.append(last_dt / max(len(got), 1))
            for tok in got:
                n += 1
                out.append(tok)
                print(f"  gen {n}/{max_new} id={tok}  "
                      f"{len(got) / max(last_dt, 1e-9):.3f} tok/s  "
                      f"{last_dt:.3f}s/cycle  accept={len(got)}/{1+len(drafts)}",
                      flush=True)
                if n >= max_new:
                    break
            continue
        out.append(nxt)
        n += 1
        if last_dt is None:
            print(f"  gen {n}/{max_new} id={nxt}  (prefill logits)", flush=True)
        else:
            decode_s.append(last_dt)
            print(f"  gen {n}/{max_new} id={nxt}  "
                  f"{1.0 / max(last_dt, 1e-9):.3f} tok/s  {last_dt:.3f}s/tok",
                  flush=True)
        if n >= max_new:
            break
        t1 = time.perf_counter()
        logits = model.step_token(nxt)
        last_dt = time.perf_counter() - t1
    if decode_s:
        med = float(np.median(decode_s))
        print(f"  warm decode  {1.0 / max(med, 1e-9):.2f} tok/s  "
              f"median {med:.3f}s/tok  n={len(decode_s)}", flush=True)
    if n_cycle:
        print(f"  mtp cycles={n_cycle}  mean_accept={n_accept / n_cycle:.2f}",
              flush=True)
    return out


def _cat_rows(*mats) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([np.asarray(m, np.float16) for m in mats], axis=0))


def _fuse_static(w, cfg: dict):
    """Concat linears that share an input so they are one ANE evaluate."""
    t = w.tensors
    if "linear_attn.in_proj_qkv.weight" in t:
        t["linear_attn.in_proj_fused.weight"] = _cat_rows(
            t["linear_attn.in_proj_qkv.weight"],
            t["linear_attn.in_proj_z.weight"],
            t["linear_attn.in_proj_b.weight"],
            t["linear_attn.in_proj_a.weight"],
        )
        for k in ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                  "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"):
            t.pop(k, None)
    if "self_attn.k_proj.weight" in t:
        t["self_attn.kv_proj.weight"] = _cat_rows(
            t["self_attn.k_proj.weight"], t["self_attn.v_proj.weight"])
        t.pop("self_attn.k_proj.weight", None)
        t.pop("self_attn.v_proj.weight", None)
    t["mlp.shared_expert.gate_up_proj.weight"] = _cat_rows(
        t["mlp.shared_expert.gate_proj.weight"],
        t["mlp.shared_expert.up_proj.weight"],
    )
    t.pop("mlp.shared_expert.gate_proj.weight", None)
    t.pop("mlp.shared_expert.up_proj.weight", None)
    for prefix in ("attn_hyper_connection", "mlp_hyper_connection"):
        t[f"{prefix}.mix_down_inject.weight"] = _cat_rows(
            t[f"{prefix}.input_mix_weight_down.weight"],
            t[f"{prefix}.block_inject_weight.weight"],
        )
        t.pop(f"{prefix}.input_mix_weight_down.weight", None)
        t.pop(f"{prefix}.block_inject_weight.weight", None)
    return w


class FlashNextAneModel:
    def __init__(self, model_path: str, *, fused: bool = True, fused_arm: str = "fp16",
                 pin_k: int = 0, reselect_every: int = 0, mtp_draft: int = 0):
        self.loader = FlashNextLoader(model_path)
        self.cfg = self.loader.text_config
        self.hc = int(self.cfg["hc_count"])
        self.H = int(self.cfg["hidden_size"])
        self.n_layers = int(self.cfg["num_hidden_layers"])
        self.types = list(self.cfg["layer_types"])
        self.pin_k = int(pin_k)
        self.reselect_every = int(reselect_every)
        self.mtp_draft = int(mtp_draft)
        self._decode_n = 0
        if self.pin_k:
            fused = False
        self.driver = AneDriver(str(ROOT))
        self.dyn = DynBank(WIDTH)
        self.dyn_pre = self.dyn if self.pin_k else DynBank(FUSED_TILE)
        self.fused_arm = fused_arm
        print("compiling shared ANE programs", flush=True)
        self.gdn = FlashNextGdn(self.driver)
        self.gqa = GqaCore(self.driver)

        print("loading layers", flush=True)
        layout = None
        self.blocks = []
        n_fused = 0
        for i, lt in enumerate(self.types):
            w = _fuse_static(_to_f16(self.loader.layer(i)), self.cfg)
            if layout is None:
                layout = infer_expert_layout(w, verbose=True)
            w.tensors["mlp.experts.gate_up_proj"] = LruSlab(
                w.tensors["mlp.experts.gate_up_proj"])
            w.tensors["mlp.experts.down_proj"] = LruSlab(
                w.tensors["mlp.experts.down_proj"])
            moe = PackedMoe(self.dyn, int(self.cfg["num_experts_per_tok"]), layout,
                            n_experts=int(self.cfg["num_experts"]))
            print(f"  layer {i:02d} {lt}", flush=True)
            fused_prog = None
            if fused and lt == "linear_attention":
                fw = w.tensors.get("linear_attn.in_proj_fused.weight")
                cw = w.tensors["linear_attn.conv1d.weight"]
                if fw is not None:
                    t0 = time.perf_counter()
                    fused_prog = compile_in_proj_dw(
                        self.driver.engine,
                        np.asarray(fw, np.float32),
                        np.asarray(cw, np.float32),
                        seq=FUSED_TILE,
                        qkv_dim=int(np.asarray(cw).reshape(np.asarray(cw).shape[0], -1).shape[0]),
                        z_dim=H_GDN * DV,
                        arm=fused_arm,
                    )
                    dt = time.perf_counter() - t0
                    if fused_prog is not None:
                        n_fused += 1
                        print(f"    fused {fused_arm} in_proj+dw S={FUSED_TILE}  {dt:.2f}s",
                              flush=True)
                    else:
                        print(f"    fused {fused_arm} skipped  {dt:.2f}s", flush=True)
            if lt == "linear_attention":
                self.blocks.append(
                    LinearBlock(self.driver, self.dyn, self.gdn, moe, w, self.cfg,
                                dyn_pre=self.dyn_pre, fused=fused_prog))
            elif lt == "full_attention":
                self.blocks.append(
                    FullAttnBlock(self.dyn, self.gqa.fork(), moe, w, self.cfg,
                                  dyn_pre=self.dyn_pre))
            else:
                raise NotImplementedError(lt)
            print(f"    ready rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f}",
                  flush=True)
        self._n_fused = n_fused
        print(f"  fused {fused_arm} programs: {n_fused}", flush=True)

        k = int(self.cfg["num_experts_per_tok"])
        I, H = int(layout["I"]), int(layout["H"])
        if not self.pin_k:
            print("warming packed expert GEMMs", flush=True)
            self.dyn.get(k * 2 * I, H)
            self.dyn.get(H, k * I)
        else:
            print("pin-k: skip packed-expert warmup (paged prefill compiles them)",
                  flush=True)

        mix_w = {
            "hyper_connection_mixer.hc_norm.weight":
                self.loader.get("model.language_model.hyper_connection_mixer.hc_norm.weight"),
            "hyper_connection_mixer.input_mix_weight_down.weight":
                self.loader.get("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"),
            "hyper_connection_mixer.input_mix_weight_up.weight":
                self.loader.get("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"),
        }
        self.mixer_w = mix_w
        print("loading lm_head ...", flush=True)
        head_w = self.loader.get("lm_head.weight")
        self.head = FinalHead(self.driver.engine, head_w)
        self._mix_prefix = "hyper_connection_mixer"
        self.reset()

    def bake_pinned(self) -> None:
        """Freeze per-layer prefill-union experts into constexpr int8 GEMMs."""
        if self.dyn_pre is self.dyn:
            n_pre = 0
        else:
            n_pre = self.dyn_pre.release()
        moe0 = self.blocks[0].moe
        dropped = 0
        for key in ((moe0.gu_o, moe0.H), (moe0.H, moe0.dn_i)):
            if self.dyn._progs.pop(key, None) is not None:
                self.dyn._xbuf.pop(key, None)
                dropped += 1
        print(f"pin-k={self.pin_k}: released {n_pre} prefill dyn + {dropped} "
              f"paged-expert programs, baking constexpr int8", flush=True)
        n_rebake = 0
        n_shared = 0
        for i, b in enumerate(self.blocks):
            b.moe.record_route = self.reselect_every > 0
            n_e = b.moe.bake(self.driver.engine, b.w, self.pin_k, f"L{i:02d}")
            n_union = int(np.count_nonzero(b.moe.counts))
            folded = bool(b.moe.baked and b.moe.baked.get("shared"))
            n_shared += int(folded)
            print(f"    layer {i:02d}  baked={n_e}  prefill_union={n_union}"
                  f"  shared={'fold' if folded else 'dyn'}", flush=True)
            if n_e:
                n_rebake += 1
        print(f"  FFN constexpr {n_rebake}/48  shared-fold {n_shared}/48 "
              f"(one SwiGLU eval/layer; GDN stays a shared recurrent program)",
              flush=True)
        reserved = 0
        seen_lin = seen_full = False
        for b in self.blocks:
            w = b.w
            keys = (
                "attn_hyper_connection.mix_down_inject.weight",
                "attn_hyper_connection.input_mix_weight_down.weight",
                "attn_hyper_connection.input_mix_weight_up.weight",
                "mlp_hyper_connection.mix_down_inject.weight",
                "mlp_hyper_connection.input_mix_weight_down.weight",
                "mlp_hyper_connection.input_mix_weight_up.weight",
            )
            if isinstance(b, LinearBlock):
                keys += ("linear_attn.in_proj_fused.weight", "linear_attn.out_proj.weight")
                seen_lin = True
            else:
                keys += ("self_attn.q_proj.weight", "self_attn.kv_proj.weight",
                         "self_attn.o_proj.weight")
                seen_full = True
            for key in keys:
                wt = w.get(key)
                if wt is None:
                    continue
                key2 = (int(wt.shape[0]), int(wt.shape[1]))
                if key2 not in self.dyn._progs:
                    self.dyn.get(*key2)
                    reserved += 1
            if seen_lin and seen_full:
                break
        for key, wt in self.mixer_w.items():
            if not hasattr(wt, "ndim") or int(getattr(wt, "ndim", 0)) != 2:
                continue
            key2 = (int(wt.shape[0]), int(wt.shape[1]))
            if key2 not in self.dyn._progs:
                self.dyn.get(*key2)
                reserved += 1
        print(f"  reserved {reserved} decode dyn shapes before extras "
              f"(bank={len(self.dyn._progs)})", flush=True)
        eng = self.driver.engine

        def _try(owner, attr, weight, tag) -> bool:
            if getattr(owner, attr) is not None or weight is None:
                return True
            try:
                setattr(owner, attr, (eng, _compile_linear_q(
                    eng, np.asarray(weight, np.float32), tag)))
                return True
            except RuntimeError as exc:
                print(f"    bake extras stopped at {tag}: {exc}", flush=True)
                return False

        extras = []
        for i, b in enumerate(self.blocks):
            if isinstance(b, LinearBlock):
                extras.append((b, "baked_out", b.w.tensors.get("linear_attn.out_proj.weight"),
                               f"L{i:02d}.out"))
        for i, b in enumerate(self.blocks):
            if isinstance(b, LinearBlock):
                extras.append((b, "baked_in", b.w.tensors.get("linear_attn.in_proj_fused.weight"),
                               f"L{i:02d}.in"))
        for i, b in enumerate(self.blocks):
            if isinstance(b, FullAttnBlock):
                extras.append((b, "baked_o", b.w.tensors.get("self_attn.o_proj.weight"),
                               f"L{i:02d}.o"))
        for i, b in enumerate(self.blocks):
            if isinstance(b, FullAttnBlock):
                extras.append((b, "baked_q", b.w.tensors.get("self_attn.q_proj.weight"),
                               f"L{i:02d}.q"))
        for i, b in enumerate(self.blocks):
            if isinstance(b, FullAttnBlock):
                extras.append((b, "baked_kv", b.w.get("self_attn.kv_proj.weight"),
                               f"L{i:02d}.kv"))
        n_extra = 0
        for owner, attr, weight, tag in extras:
            if not _try(owner, attr, weight, tag):
                break
            if getattr(owner, attr) is not None:
                n_extra += 1
        print(f"  baked extras {n_extra}/{len(extras)} "
              f"(out_proj, in_proj, then QSA o/q/kv until the 127 cap)",
              flush=True)
        return n_rebake

    def reset(self) -> None:
        for b in self.blocks:
            b.reset()
        self._decode_n = 0

    def snapshot(self):
        pack = []
        for b in self.blocks:
            if isinstance(b, LinearBlock):
                g = b.gdn
                with g.driver.view(b.state.surface, (g.HK, g.Dv), np.float16) as src:
                    st = np.array(src)
                pack.append(("L", st, np.array(b.conv_cache)))
            else:
                pack.append(("A", b.cache.keys.copy(), b.cache.values.copy(),
                             int(b.cache.offset)))
        return pack

    def restore(self, pack) -> None:
        for b, item in zip(self.blocks, pack):
            if item[0] == "L":
                g = b.gdn
                with g.driver.view(b.state.surface, (g.HK, g.Dv), np.float16) as dst:
                    np.copyto(dst, item[1])
                b.conv_cache = item[2]
            else:
                b.cache.keys[:] = item[1]
                b.cache.values[:] = item[2]
                b.cache.offset = item[3]

    def maybe_reselect(self) -> None:
        if not self.pin_k or self.reselect_every <= 0:
            return
        self._decode_n += 1
        if self._decode_n % self.reselect_every:
            return
        changed = 0
        skipped = 0
        for i, b in enumerate(self.blocks):
            try:
                n = b.moe.bake(self.driver.engine, b.w, self.pin_k, f"L{i:02d}r",
                               counts=b.moe.recent)
            except RuntimeError as exc:
                skipped += 1
                print(f"  reselect @ tok {self._decode_n}: compile failed on L{i:02d} "
                      f"(keeping previous bake): {exc}", flush=True)
                break
            if n:
                changed += 1
        if changed:
            print(f"  reselect @ tok {self._decode_n}: rebaked {changed} layers",
                  flush=True)
        elif skipped == 0:
            print(f"  reselect @ tok {self._decode_n}: pinned set unchanged",
                  flush=True)

    def step_token(self, token_id: int) -> np.ndarray:
        row = embed_row(self.loader, token_id)
        h = np.concatenate([row] * self.hc).reshape(1, 1, -1)
        for b in self.blocks:
            h = b.step(h)
        mixed = _hyper(self.dyn, self.mixer_w, self._mix_prefix, h, self.cfg,
                       combine=False)
        self._last_mixed = np.asarray(mixed, np.float32).reshape(-1)
        self._last_h = h
        self.maybe_reselect()
        return self.head(self._last_mixed)

    def _load_mtp_fc(self) -> None:
        if getattr(self, "_mtp_ready", False):
            return
        self._mtp_fc_h = np.asarray(self.loader.get("mtp.fc_hidden.weight"), np.float32)
        self._mtp_fc_e = np.asarray(self.loader.get("mtp.fc_embedding.weight"), np.float32)
        self._mtp_ready = True
        print(f"  mtp fc_hidden {self._mtp_fc_h.shape}  fc_embed {self._mtp_fc_e.shape}",
              flush=True)

    def draft(self, token_id: int, n: int) -> list[int]:
        """MTP front-end only (fc_embed + fc_hidden), shared lm_head. Tests verify."""
        self._load_mtp_fc()
        e = embed_row(self.loader, int(token_id)).astype(np.float32)
        h = np.asarray(self._last_mixed, np.float32).reshape(-1)
        if h.size != e.size:
            h = h.reshape(-1, e.size).mean(axis=0)
        x = self._mtp_fc_e @ e + self._mtp_fc_h @ h
        out = []
        for i in range(n):
            logits = self.head(x)
            t = int(np.argmax(logits))
            out.append(t)
            if i + 1 >= n:
                break
            e = embed_row(self.loader, t).astype(np.float32)
            x = self._mtp_fc_e @ e + self._mtp_fc_h @ x
        return out

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        self.reset()
        ids = [int(t) for t in token_ids]
        fused_ms, fused_tops = [], []
        h = None
        pos = 0
        tile = self.dyn_pre.seq_len
        while pos < len(ids):
            chunk = ids[pos:pos + tile]
            rows = np.stack([embed_row(self.loader, tid) for tid in chunk])
            h = np.concatenate([rows] * self.hc, axis=-1)
            for b in self.blocks:
                h = b.prefill(h)
                ms = getattr(b, "fused_ms", 0.0)
                if ms:
                    fused_ms.append(ms)
                    fused_tops.append(b.fused_tops)
            pos += len(chunk)
        mixed = _hyper(self.dyn_pre, self.mixer_w, self._mix_prefix, h[-1:],
                       self.cfg, combine=False, batch=True)
        self._last_mixed = np.asarray(mixed, np.float32).reshape(-1)
        self._last_h = h[-1:]
        logits = self.head(self._last_mixed)
        if fused_ms:
            fused_ms.sort()
            fused_tops.sort()
            print(f"  fused {self.fused_arm}  n={len(fused_ms)}  "
                  f"median {fused_ms[len(fused_ms)//2]:.2f} ms  "
                  f"{fused_tops[len(fused_tops)//2]:.1f} TOPS  "
                  f"(tile S={FUSED_TILE}, {self._n_fused} programs)",
                  flush=True)
        return logits


CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
IM_END, EOT = 248046, 248044


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="Reply with exactly: OK")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--ids", default=None,
                    help="comma-separated prompt token ids (skips chat template)")
    ap.add_argument("--raw-prompt", action="store_true",
                    help="encode --prompt as-is, no chat template")
    ap.add_argument("--cpu", action="store_true",
                    help="numpy fp32 reference decode (no ANE)")
    ap.add_argument("--no-fused", action="store_true",
                    help="skip fused in_proj+dw programs (paged prefill GEMMs only)")
    ap.add_argument("--w8a8", action="store_true",
                    help="int8 weights + int8 activations on the fused tile (30 TOPS recipe; greedy may drift)")
    ap.add_argument("--pin-k", type=int, default=0,
                    help="after prefill, bake the per-layer expert union (capped at K) as constexpr int8; skips fused in_proj")
    ap.add_argument("--reselect-every", type=int, default=0,
                    help="with --pin-k, rebake layers whose recent top-10 union changed every N decode tokens")
    ap.add_argument("--mtp-draft", type=int, default=0,
                    help="speculative depth k using MTP fc front-end + snapshot/verify (try 2)")
    args = ap.parse_args(argv)

    tok = StandaloneTokenizer(args.model)
    if args.ids:
        prompt_ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        text = args.prompt if args.raw_prompt else CHAT_TEMPLATE.format(
            prompt=args.prompt)
        prompt_ids = tok.encode(text)
        if not prompt_ids:
            raise SystemExit("empty prompt encoding")
    print(f"prompt ids: {prompt_ids}", flush=True)
    print(f"prompt text: {tok.decode(prompt_ids)!r}", flush=True)

    if not args.cpu:
        assert_standalone("flashnext generate startup")
        model = FlashNextAneModel(
            args.model, fused=not args.no_fused,
            fused_arm="w8a8" if args.w8a8 else "fp16",
            pin_k=args.pin_k,
            reselect_every=args.reselect_every,
            mtp_draft=args.mtp_draft)
    else:
        model = FlashNextCpuModel(args.model)

    eos_ids = {IM_END, EOT}
    ids = generate_loop(model, prompt_ids, args.tokens, eos_ids)
    gen = ids[len(prompt_ids):]
    print("ids", ids)
    print("generated", gen)
    print("text", tok.decode(ids))
    if not args.cpu:
        assert_standalone("flashnext generate completion")
        print(f"FLASHNEXT_ANE_EXECUTION=PASS generated={len(gen)}")
    else:
        print(f"FLASHNEXT_CPU_EXECUTION=PASS generated={len(gen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
