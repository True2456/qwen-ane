"""MTP drafter on the GPU, ported from the production Swift backend.

`Qwen4ExpMTPHead` in `Rindi-NativeMLX/Sources/RindiInference/Qwen4Exp.swift` is
two grouped RMS norms, two 2560x2560 projections, one full sparse-attention
decoder layer with its own 512-expert MoE, and a final mixer. Drafting one
token costs roughly one forty-eighth of a backbone pass, so a chain of four
drafts is cheap next to the block it proposes.

It runs on the GPU and stays there: the backbone's hidden state is the only
thing that crosses the NumPy boundary, once per block. The ANE never sees it.

Two details that silently produce a useless drafter rather than an error:

  * `pre_fc_norm_embedding` and `pre_fc_norm_hidden` are zero-centred like
    every other norm in this checkpoint, so their weights load as w + 1. The
    Swift loader calls this out for exactly this reason — miss it and the
    accept rate is zero with no other symptom.
  * `mtp_front_end` is `per_branch` here: `fc_hidden` applies to each of the
    four hyper-connection branches separately and the embedding broadcasts
    across them. The `mean` / `sum` / first-branch variants all produce
    plausible numbers and a drafter that never agrees with the backbone.

The MTP layer has no indexer weights in this checkpoint, so its attention is
dense over its own KV cache. Drafts are verified by the backbone regardless,
so the accept rate is the test that matters.
"""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

from runtime.expert_bank import MlxSafe, MLX4_DEFAULT

H = 2560
HC = 4
HC_W = H * HC
HC_LOWRANK = 320
HQ, HKV, HD = 24, 2, 256
ROTARY = 64
THETA = 10_000_000.0
K_PIN = 10
EPS = 1e-6
_PLUS_ONE = ("hc_norm.weight", "q_norm.weight", "k_norm.weight",
             "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")


def _silu(x):
    return x * mx.sigmoid(x)


def _grouped_rms(x, w, group: int):
    lead = list(x.shape[:-1])
    d = x.shape[-1]
    g = x.reshape(*lead, d // group, group)
    g = g * mx.rsqrt(mx.mean(g * g, axis=-1, keepdims=True) + EPS)
    return w * g.reshape(*lead, d)


class _Mix:
    """One hyper-connection mixer. `inj` is None for the head's final mixer."""

    def __init__(self, src, prefix: str, combine: bool):
        self.hc_n = mx.array(src.f32(f"{prefix}.hc_norm.weight") + 1.0)
        self.down = mx.array(src.f32(f"{prefix}.input_mix_weight_down.weight"))
        self.up = mx.array(src.f32(f"{prefix}.input_mix_weight_up.weight"))
        self.inj = (mx.array(src.f32(f"{prefix}.block_inject_weight.weight"))
                    if combine else None)

    def __call__(self, x_hc):
        n = _grouped_rms(x_hc, self.hc_n, H)
        w = mx.sigmoid((_silu((n @ self.down.T) * 0.25)) @ self.up.T)
        lead = list(n.shape[:-1])
        mixed = mx.mean(w.reshape(*lead, HC, H) * n.reshape(*lead, HC, H), axis=-2)
        if self.inj is None:
            return mixed, None
        return mixed, 2.0 * mx.sigmoid((n @ self.inj.T) * 0.25)


def _recombine(out_h, hyper, inj):
    lead = list(out_h.shape[:-1])
    parts = mx.expand_dims(out_h, -2) * mx.expand_dims(inj, -1)
    return hyper + parts.reshape(*lead, HC_W)


class _Q:
    """A quantized projection kept packed; only activations are materialized."""

    def __init__(self, src, prefix: str, in_dim: int):
        self.w = mx.array(src.raw(f"{prefix}.weight"))
        scales = src.f32(f"{prefix}.scales")
        self.s = mx.array(scales, dtype=mx.float16)
        self.b = mx.array(src.f32(f"{prefix}.biases"), dtype=mx.float16)
        # Bit width follows from the row: a packed row holds `in_dim` values in
        # `cols` uint32 words. Guessing it from the scale count is ambiguous —
        # 4-bit/group-128 and 8-bit/group-64 pack to the same size.
        self.bits = 32 * self.w.shape[-1] // int(in_dim)
        self.group = int(in_dim) // scales.shape[-1]

    def __call__(self, x):
        return mx.quantized_matmul(x, self.w, self.s, self.b, transpose=True,
                                   group_size=self.group, bits=self.bits)

    def rows(self, ids):
        """Dequantized rows — the embedding table is a lookup, not a matmul."""
        return mx.dequantize(mx.take(self.w, ids, axis=0),
                             mx.take(self.s, ids, axis=0),
                             mx.take(self.b, ids, axis=0),
                             group_size=self.group, bits=self.bits)


def _rope(x, offset: int):
    """Partial rotary over the first ROTARY dims, rotate-half convention."""
    l = x.shape[2]
    pos = mx.arange(offset, offset + l, dtype=mx.float32).reshape(-1, 1)
    inv = 1.0 / (THETA ** (mx.arange(0, ROTARY, 2, dtype=mx.float32) / ROTARY))
    f = pos * inv.reshape(1, -1)
    cos = mx.concatenate([mx.cos(f), mx.cos(f)], axis=-1).reshape(1, 1, l, ROTARY)
    sin = mx.concatenate([mx.sin(f), mx.sin(f)], axis=-1).reshape(1, 1, l, ROTARY)
    r, rest = x[..., :ROTARY], x[..., ROTARY:]
    half = ROTARY // 2
    rot = mx.concatenate([-r[..., half:], r[..., :half]], axis=-1)
    return mx.concatenate([r * cos + rot * sin, rest], axis=-1)


class _Embed:
    """Embedding rows dequantized from the mmap on demand.

    Keeping the packed table on the GPU costs 0.7 GB, which is 0.7 GB the
    expert bank does not get. A decode step needs one row.
    """

    def __init__(self, path, prefix: str = "model.embed_tokens"):
        self.src = MlxSafe(path)
        self.w = self.src.raw(f"{prefix}.weight")
        self.sc = self.src.raw(f"{prefix}.scales")
        self.bi = self.src.raw(f"{prefix}.biases")
        self.bits = 32 * self.w.shape[-1] // H
        self.group = H // self.sc.shape[-1]

    def close(self):
        self.src.close()

    def __call__(self, ids):
        rows = np.asarray(ids, np.int64).ravel()
        packed = self.w[rows]
        per = 32 // self.bits
        q = np.right_shift(
            packed[:, :, None].astype(np.uint32),
            (np.arange(per, dtype=np.uint32) * self.bits)[None, None, :])
        q = (q & np.uint32((1 << self.bits) - 1)).reshape(len(rows), -1)
        sc = _bf16(self.sc[rows])[:, :, None]
        bi = _bf16(self.bi[rows])[:, :, None]
        v = q.reshape(len(rows), -1, self.group).astype(np.float32) * sc + bi
        return v.reshape(len(rows), H)


def _bf16(a):
    if a.dtype == np.float32:
        return a
    if a.dtype.itemsize == 2 and a.dtype.kind == "V" or a.dtype == np.uint16:
        u = np.asarray(a, np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    return np.asarray(a, np.float32)


class MtpDrafter:
    """Chained greedy drafting off the backbone's hyper-connection state."""

    def __init__(self, head, path=MLX4_DEFAULT, max_kv: int = 4096,
                 routed: bool | None = None):
        # Every gigabyte here comes out of the expert bank's residency. The
        # bank is 69 GB on a 137 GB machine and the whole run falls off a cliff
        # when the total stops fitting, so the drafter is kept small: scales in
        # fp16, and its routed experts optional.
        if routed is None:
            routed = os.environ.get("FLASHNEXT_MTP_SHARED_ONLY") != "1"
        self.routed = bool(routed)
        src = MlxSafe(os.environ.get("FLASHNEXT_MLX4") or path)
        try:
            self.emb = _Embed(os.environ.get("FLASHNEXT_MLX4") or path)
            self.pre_e = mx.array(src.f32("mtp.pre_fc_norm_embedding.weight") + 1.0)
            self.pre_h = mx.array(src.f32("mtp.pre_fc_norm_hidden.weight") + 1.0)
            self.fc_e = mx.array(src.f32("mtp.fc_embedding.weight"))
            self.fc_h = mx.array(src.f32("mtp.fc_hidden.weight"))
            lp = "mtp.layers.0"
            self.amix = _Mix(src, f"{lp}.attn_hyper_connection", True)
            self.mmix = _Mix(src, f"{lp}.mlp_hyper_connection", True)
            self.out_mix = _Mix(src, "mtp.hyper_connection_mixer", False)
            self.q = _Q(src, f"{lp}.self_attn.q_proj", H)
            self.k = _Q(src, f"{lp}.self_attn.k_proj", H)
            self.v = _Q(src, f"{lp}.self_attn.v_proj", H)
            self.o = _Q(src, f"{lp}.self_attn.o_proj", HQ * HD)
            self.qn = mx.array(src.f32(f"{lp}.self_attn.q_norm.weight") + 1.0)
            self.kn = mx.array(src.f32(f"{lp}.self_attn.k_norm.weight") + 1.0)
            self.router = _Q(src, f"{lp}.mlp.gate", H)
            self.sh_g = _Q(src, f"{lp}.mlp.shared_expert.gate_proj", H)
            self.sh_u = _Q(src, f"{lp}.mlp.shared_expert.up_proj", H)
            self.sh_d = _Q(src, f"{lp}.mlp.shared_expert.down_proj", 640)
            self.sh_gate = _Q(src, f"{lp}.mlp.shared_expert_gate", H)
            # The MTP head's experts are 8-bit here while the backbone's are
            # 4-bit, so the packing is read off the shapes rather than assumed.
            self.experts = []
            self.expert_q = []
            for name, in_dim in ((("gate_proj", H), ("up_proj", H), ("down_proj", 640))
                                 if self.routed else ()):
                p = f"{lp}.mlp.switch_mlp.{name}"
                w = mx.array(src.raw(p + ".weight"))
                sc = mx.array(src.f32(p + ".scales"), dtype=mx.float16)
                bi = mx.array(src.f32(p + ".biases"), dtype=mx.float16)
                gs = in_dim // sc.shape[-1]
                bits = 32 * w.shape[-1] // in_dim
                if bits > 4:
                    # 8-bit drafter experts cost 2.5 GB. Requantizing to 4 bits
                    # halves that; the drafter only has to guess well enough to
                    # be accepted, and the backbone checks every token anyway.
                    w, sc, bi = mx.quantize(
                        mx.dequantize(w, sc, bi, group_size=gs, bits=bits),
                        group_size=64, bits=4)
                    sc, bi, gs, bits = sc.astype(mx.float16), bi.astype(mx.float16), 64, 4
                    mx.eval(w, sc, bi)
                self.experts.append((w, sc, bi))
                self.expert_q.append((gs, bits))
        finally:
            src.close()
        self.head = head
        self.nbytes = sum(a.nbytes for p_ in self.experts for a in p_)
        self.max_kv = int(max_kv)
        self.kc = None
        self.vc = None
        self.offset = 0
        mx.eval(self.pre_e, self.pre_h, self.fc_e, self.fc_h)

    def reset(self) -> None:
        self.kc = self.vc = None
        self.offset = 0

    def trim_to(self, n: int) -> None:
        """Drop cache rows past `n`, which is how a rejected draft is undone."""
        n = max(0, int(n))
        if self.kc is not None and n < self.offset:
            self.kc = self.kc[:, :, :n] if n else None
            self.vc = self.vc[:, :, :n] if n else None
        self.offset = n

    def _front(self, hidden_hc, token_ids):
        tok = np.asarray(token_ids, np.int64)
        e = _grouped_rms(mx.array(self.emb(tok).reshape(*tok.shape, H)), self.pre_e, H)
        e = e @ self.fc_e.T
        hh = _grouped_rms(hidden_hc, self.pre_h, H)
        lead = list(hh.shape[:-1])
        hh = hh.reshape(*lead, HC, H)
        y = (hh @ self.fc_h.T) + mx.expand_dims(e, -2)
        return y.reshape(*lead, HC_W)

    def _attn(self, x):
        b, l = x.shape[0], x.shape[1]
        qg = self.q(x).reshape(b, l, HQ, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:].reshape(b, l, HQ * HD)
        k = self.k(x).reshape(b, l, HKV, HD)
        v = self.v(x).reshape(b, l, HKV, HD)
        q = (self.qn * (q * mx.rsqrt(mx.mean(q * q, -1, keepdims=True) + EPS))).transpose(0, 2, 1, 3)
        k = (self.kn * (k * mx.rsqrt(mx.mean(k * k, -1, keepdims=True) + EPS))).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = _rope(q, self.offset)
        k = _rope(k, self.offset)
        if self.kc is None:
            self.kc, self.vc = k, v
        else:
            self.kc = mx.concatenate([self.kc, k], axis=2)
            self.vc = mx.concatenate([self.vc, v], axis=2)
        if self.kc.shape[2] > self.max_kv:
            self.kc = self.kc[:, :, -self.max_kv:]
            self.vc = self.vc[:, :, -self.max_kv:]
        self.offset += l
        mask = "causal" if l > 1 else None
        out = mx.fast.scaled_dot_product_attention(
            q, self.kc, self.vc, scale=HD ** -0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, l, HQ * HD)
        return self.o(out * mx.sigmoid(gate))

    def _moe(self, x):
        sg = _silu(self.sh_g(x))
        shared = self.sh_d(sg * self.sh_u(x)) * mx.sigmoid(self.sh_gate(x))
        if not self.routed:
            return shared
        logits = self.router(x).reshape(-1, 512)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        ids = mx.argpartition(-probs, K_PIN - 1, axis=-1)[:, :K_PIN]
        sc = mx.take_along_axis(probs, ids, axis=-1)
        sc = sc / mx.sum(sc, axis=-1, keepdims=True)
        k = ids.shape[0]
        t = mx.expand_dims(x.reshape(1, k, -1), (-2, -3))
        rid = ids.astype(mx.uint32).reshape(1, k, -1)

        def project(u, j):
            gs, bits = self.expert_q[j]
            return mx.gather_qmm(u, *self.experts[j], rhs_indices=rid,
                                 transpose=True, group_size=gs, bits=bits)

        g = project(t, 0)
        y = project(g * mx.sigmoid(g) * project(t, 1), 2)
        y = mx.sum(y.squeeze(-2) * sc.reshape(1, k, -1, 1), axis=-2)
        return y.reshape(x.shape) + shared

    def _layer(self, x_hc):
        mixed, inj = self.amix(x_hc)
        h = _recombine(self._attn(mixed), x_hc, inj)
        mixed2, inj2 = self.mmix(h)
        return _recombine(self._moe(mixed2), h, inj2)

    def step(self, hidden_hc, token_ids):
        """One MTP pass. Returns (logits, pre-mix state to chain from)."""
        pre = self._layer(self._front(hidden_hc, token_ids))
        mixed, _ = self.out_mix(pre)
        return self.head.logits_mx(mixed), pre

    def draft(self, hidden_hc, token_id: int, n: int, lookup=None) -> list[int]:
        """Chain `n` greedy drafts. Each one is a serialized round trip.

        `lookup` is an optional `ContextLookup`: when the drafted suffix has
        occurred before, the token that followed it then replaces the head's
        own argmax and is fed back into the chain. The head still runs every
        step, because the drafter's attention cache needs a row per drafted
        position for a partly accepted block to unwind.
        """
        out: list[int] = []
        h = hidden_hc
        tok = int(token_id)
        for _ in range(n):
            logits, h = self.step(h, [[tok]])
            nxt = mx.argmax(logits.reshape(-1))
            mx.eval(nxt, h)
            tok = int(nxt.item())
            if lookup is not None:
                hit = lookup.next_token(out)
                if hit is not None:
                    tok = int(hit)
            out.append(tok)
        return out

    def advance(self, hidden_hc, token_ids) -> None:
        """Replay confirmed tokens so the drafter's cache stays aligned."""
        self._layer(self._front(hidden_hc, token_ids))
