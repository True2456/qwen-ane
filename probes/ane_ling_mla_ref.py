"""Numpy oracle for Ling's MLA, and does the absorbed form equal the naive one?

Before any MIL is written, this pins down the two things most likely to be
silently wrong in an MLA port:

  1. RoPE is INTERLEAVED (GPT-J pairing). The modeling code's non-interleaved
     branch is literally `x = 1/0`, so there is no fallback -- and applying it
     as non-interleaved produces plausible but wrong output rather than a crash.
  2. The absorbed form must be algebraically identical to materializing K and V.
     Absorbed caches 512 latent + 64 k_pe per token per layer (1152 B) against
     60 KiB/token for materialized KV, so it is the only affordable option at
     long context -- but only if it is exact.

Runs against the real layer-3 weights in float64. No ANE involved.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
from pure_ling import load

MODEL = os.environ.get("Q38_LING_MODEL",
                       str(Path.home() / ".lmstudio/models/inclusionAI/Ling-3.0-tiny"))
LAYER, T = 3, 12

checkpoint, spec = load(MODEL)
n = spec.attention_names(LAYER)
W = {k: checkpoint.tensor(v, np.float64) for k, v in n.items()}
norms = spec.norm_names(LAYER)
in_norm = checkpoint.tensor(norms["input"], np.float64)

H, Dn, Dr, Dv = spec.heads, spec.qk_nope, spec.qk_rope, spec.v_head_dim
Dq = spec.qk_head_dim
scale = Dq ** -0.5
print(f"layer {LAYER}: H={H} qk_nope={Dn} qk_rope={Dr} v={Dv} qk_head_dim={Dq} "
      f"scale={scale:.7f}")
print(f"q_lora={spec.q_lora_rank} kv_lora={spec.kv_lora_rank} "
      f"theta={spec.rope_theta:g} interleave={spec.rope_interleave}\n")


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope_tables(positions):
    """BailingMoeV3RotaryEmbedding: head_dim forced to qk_rope, partial=1.0."""
    inv = 1.0 / (spec.rope_theta ** (np.arange(0, Dr, 2, dtype=np.float64) / Dr))
    f = positions[:, None] * inv[None, :]                 # [T, Dr/2]
    emb = np.concatenate([f, f], axis=-1)                 # [T, Dr]
    return np.cos(emb), np.sin(emb)


def apply_rope_interleave(x, cos, sin):
    """apply_rotary_pos_emb_interleave: de-interleave, then NeoX rotation.

    The checkpoint stores rope dims interleaved [x0,y0,x1,y1,...]; the reference
    reshapes to (...,Dr/2,2), swaps the last two axes to get [x0..,y0..], and
    only then does the standard half-rotation. The result stays de-interleaved,
    and k_pe goes through the same transform, so the dot products are consistent.
    """
    half = Dr // 2
    xd = x.reshape(*x.shape[:-1], half, 2).swapaxes(-1, -2).reshape(*x.shape)
    rot = np.concatenate([-xd[..., half:], xd[..., :half]], axis=-1)
    return xd * cos + rot * sin


def apply_rope_neox(x, cos, sin):
    """The WRONG one for this checkpoint: no de-interleave. Used to size the bug."""
    half = Dr // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


def project(x, rope_fn):
    """Everything up to the attention scores. x is [T, hidden], already normed."""
    q = rmsnorm(x @ W["q_a"].T, W["q_a_norm"], spec.rms_eps) @ W["q_b"].T
    q = q.reshape(T, H, Dq)
    q_nope, q_rope = q[..., :Dn], q[..., Dn:]
    c = x @ W["kv_a"].T
    k_lat, k_rope = c[..., :spec.kv_lora_rank], c[..., spec.kv_lora_rank:]
    cos, sin = rope_tables(np.arange(T, dtype=np.float64))
    q_rope = rope_fn(q_rope, cos[:, None, :], sin[:, None, :])
    k_rope = rope_fn(k_rope, cos, sin)                    # [T, Dr], shared (MQA)
    return q_nope, q_rope, k_lat, k_rope


def naive(x, rope_fn=apply_rope_interleave):
    """Materialize K and V per head, as the HF reference does."""
    q_nope, q_rope, k_lat, k_rope = project(x, rope_fn)
    kv = rmsnorm(k_lat, W["kv_a_norm"], spec.rms_eps) @ W["kv_b"].T
    kv = kv.reshape(T, H, Dn + Dv)
    k_nope, v = kv[..., :Dn], kv[..., Dn:]
    q = np.concatenate([q_nope, q_rope], -1)                       # [T,H,Dq]
    k = np.concatenate([k_nope, np.broadcast_to(k_rope[:, None, :],
                                                (T, H, Dr))], -1)
    s = np.einsum("thd,shd->hts", q, k) * scale
    s = np.where(np.tril(np.ones((T, T), bool))[None], s, -np.inf)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    return np.einsum("hts,shv->thv", a, v)


def absorbed(x, rope_fn=apply_rope_interleave):
    """Cache only the 512 latent and 64 k_pe; fold kv_b into q and the output."""
    q_nope, q_rope, k_lat, k_rope = project(x, rope_fn)
    kv_b = W["kv_b"].reshape(H, Dn + Dv, spec.kv_lora_rank)
    W_K, W_V = kv_b[:, :Dn, :], kv_b[:, Dn:, :]            # [H,Dn,512] [H,Dv,512]
    lat = rmsnorm(k_lat, W["kv_a_norm"], spec.rms_eps)     # [T, 512]
    q_abs = np.einsum("thn,hnl->thl", q_nope, W_K)         # [T,H,512]
    s = (np.einsum("thl,sl->hts", q_abs, lat)
         + np.einsum("thr,sr->hts", q_rope, k_rope)) * scale
    s = np.where(np.tril(np.ones((T, T), bool))[None], s, -np.inf)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    ctx = np.einsum("hts,sl->thl", a, lat)                 # [T,H,512]
    return np.einsum("thl,hvl->thv", ctx, W_V)


def tail(attn, x):
    gate = 1.0 / (1.0 + np.exp(-(x @ W["gate"].T)))        # head_wise, [T,H]
    return (attn * gate[..., None]).reshape(T, H * Dv) @ W["out"].T


rng = np.random.default_rng(0)
hidden = rng.standard_normal((T, spec.hidden)) * 0.6
x = rmsnorm(hidden, in_norm, spec.rms_eps)

a_naive, a_abs = naive(x), absorbed(x)
rel = np.abs(a_naive - a_abs).max() / np.abs(a_naive).max()
print(f"absorbed vs materialized KV : rel {rel:.3e}   "
      f"(must be ~float64 epsilon, not merely small)")

o_naive, o_abs = tail(a_naive, x), tail(a_abs, x)
rel_out = np.abs(o_naive - o_abs).max() / np.abs(o_naive).max()
print(f"through gate + dense        : rel {rel_out:.3e}")

wrong = tail(naive(x, apply_rope_neox), x)
rel_rope = np.abs(o_naive - wrong).max() / np.abs(o_naive).max()
cos_sim = float((o_naive * wrong).sum()
                / np.sqrt((o_naive ** 2).sum() * (wrong ** 2).sum()))
print(f"\nnon-interleaved RoPE instead: rel {rel_rope:.3e}, cosine {cos_sim:.4f}")
print("  -- plausible-looking output, so this cannot be caught by eyeballing")

cache_absorbed = (spec.kv_lora_rank + Dr) * 2
cache_naive = H * (Dq + Dv) * 2
print(f"\nKV per token per layer: absorbed {cache_absorbed} B, "
      f"materialized {cache_naive} B ({cache_naive/cache_absorbed:.1f}x)")
print(f"  over {len(spec.full_attention_layers)} MLA layers at 32K: "
      f"{cache_absorbed*len(spec.full_attention_layers)*32768/1e9:.2f} GB vs "
      f"{cache_naive*len(spec.full_attention_layers)*32768/1e9:.2f} GB")

ok = rel < 1e-12 and rel_out < 1e-12
print(f"\nANE_LING_MLA_REF={'PASS' if ok else 'FAIL'}")
