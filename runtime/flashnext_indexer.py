"""QSA sparse-attention indexer on the host.

The model never attends to more than `indexer_budget` (2048) keys. Past that it
pools keys into blocks of `compress_ratio` (4), scores the blocks against the
query, and gathers the top `budget // compress_ratio` blocks plus the ragged
tail. So a 256k context (`max_position_embeddings` 262144) needs a big *host*
KV cache and this selector — not a bigger ANE graph. The exported QSA graphs at
max_S=2048 are already the right shape for any context length.

Ported from the **production Swift backend**, `Rindi-NativeMLX`
`Sources/RindiInference/Qwen4Exp.swift` `Qwen4ExpIndexer.updateAndSelect`, not
from mlx_lm. The Python version is behind it in four ways that matter once
speculation and chunked prefill are in play: no `coversPrefix` guard, no
retained raw-key window for rollback, a block count derived from the offset
rather than from the array being partitioned, and no `L > 16` short-circuit.
mlx_lm's own multi-token branch also calls `mx.unique`, which does not exist in
this MLX build, so it has never run.
"""
from __future__ import annotations

import numpy as np


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    v = np.asarray(x, np.float32)
    ms = np.mean(v * v, axis=-1, keepdims=True)
    return (v * (1.0 / np.sqrt(ms + eps))) * np.asarray(weight, np.float32)


def _rope(x: np.ndarray, positions: np.ndarray, rotary_dim: int, theta: float) -> np.ndarray:
    """Non-traditional (NeoX) RoPE on the last axis, one position per row."""
    out = np.array(x, np.float32, copy=True)
    dim_idx = np.arange(0, rotary_dim, 2, dtype=np.float32)
    inv = 1.0 / (theta ** (dim_idx / np.float32(rotary_dim)))
    freqs = np.asarray(positions, np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)
    mid = rotary_dim // 2
    r = out[..., :rotary_dim]
    r_rot = np.concatenate([-r[..., mid:], r[..., :mid]], axis=-1)
    shape = [1] * (r.ndim - 2) + list(cos.shape)
    out[..., :rotary_dim] = r * cos.reshape(shape) + r_rot * sin.reshape(shape)
    return out


#: Rewind depth the retained raw keys can absorb, in tokens. Two rewinds have
#: to fit: a rejected draft chain (tens of tokens) and a snapshot trimmed back
#: to the recovery boundary (a further 64). Matches the Swift backend.
RETAINED_KEYS = 192


def clip_to_budget(keep: np.ndarray, offset: int, budget: int,
                   compress_ratio: int = 4) -> np.ndarray:
    """Fit selected keys into an ANE graph that is `budget` wide.

    The indexer (and mlx_lm) may return 512 blocks plus a ragged tail, which
    is 2049-2051 keys. ``keep[:budget]`` would drop the tail — the most recent
    tokens. Always keep the tail, then the most recent selected body keys.
    """
    keep = np.asarray(keep, np.int64)
    keep = keep[(keep >= 0) & (keep < offset)]
    if keep.size <= budget:
        return keep
    tail_start = (offset // compress_ratio) * compress_ratio
    is_tail = keep >= tail_start
    tail = keep[is_tail]
    body = keep[~is_tail]
    room = budget - int(tail.size)
    if room <= 0:
        return np.sort(tail)[-budget:]
    if body.size > room:
        body = np.sort(body)[-room:]
    return np.concatenate([body, tail])


class IndexerState:
    """Block keys plus a bounded window of raw keys, as the Swift cache keeps.

    The window exists so a speculative rewind into the middle of a block can
    rebuild that block. Keeping only the ragged tail — what mlx_lm does — makes
    the rewind destroy keys the block was pooled from, and it can never be
    rebuilt.
    """

    __slots__ = ("block_keys", "recent", "recent_start", "covers_prefix")

    def __init__(self) -> None:
        self.block_keys: np.ndarray | None = None   # (n_blocks, head_dim)
        self.recent: np.ndarray | None = None       # raw keys [recent_start, offset)
        self.recent_start: int = 0
        #: Block i must describe tokens [i*ratio, (i+1)*ratio). A gap would not
        #: weaken selection, it would point at the wrong tokens, so an index
        #: that loses a block stops being used rather than being extended.
        self.covers_prefix: bool = True

    def trim(self, offset: int, compress_ratio: int) -> None:
        """Rewind the window and the blocks to a confirmed boundary."""
        if self.recent is not None and self.recent_start + self.recent.shape[0] > offset:
            keep = offset - self.recent_start
            if keep > 0:
                self.recent = self.recent[:keep]
            else:
                self.recent = None
                self.recent_start = offset
        if self.block_keys is not None:
            valid = offset // compress_ratio
            if self.block_keys.shape[0] > valid:
                self.block_keys = self.block_keys[:valid] if valid > 0 else None
        if (self.block_keys.shape[0] if self.block_keys is not None else 0) \
                * compress_ratio < self.recent_start:
            self.covers_prefix = False


class QSAIndexer:
    """Host-side top-k block selection for one QSA layer."""

    def __init__(self, w, config):
        cfg = config
        self.n_heads = int(cfg.get("indexer_n_heads") or 4)
        self.kv_heads = int(cfg.get("indexer_kv_heads") or 1)
        self.head_dim = int(cfg.get("indexer_head_dim") or 128)
        self.budget = int(cfg.get("indexer_budget") or 2048)
        self.compress_ratio = int(cfg.get("indexer_compress_ratio") or 4)
        self.block_topk = self.budget // self.compress_ratio
        self.scale = self.head_dim ** -0.5
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.theta = float(cfg.get("rope_theta", 10_000_000.0))
        self.rotary_dim = int(int(cfg["head_dim"]) * float(cfg.get("partial_rotary_factor", 0.25)))
        p = "self_attn.indexer."
        self.qk = np.ascontiguousarray(np.asarray(w[p + "index_qk_proj.weight"], np.float32))
        self.q_norm = np.asarray(w[p + "q_layernorm.weight"], np.float32).reshape(-1)
        self.k_norm = np.asarray(w[p + "k_layernorm.weight"], np.float32).reshape(-1)

    def _project(self, x: np.ndarray):
        h = np.asarray(x, np.float32).reshape(-1, self.qk.shape[1])
        qk = h @ self.qk.T
        cut = self.n_heads * self.head_dim
        q = qk[:, :cut].reshape(-1, self.n_heads, self.head_dim)
        k = qk[:, cut:].reshape(-1, self.head_dim)
        return q, k

    #: Prefill chunks wider than this skip selection and use dense causal
    #: attention; only the block keys are updated, for later decode steps.
    DENSE_PREFILL_L = 16

    def update_and_select(self, x: np.ndarray, offset: int, state: IndexerState):
        """Returns selected token indices, or None while the context fits the budget."""
        q, k_raw = self._project(x)
        L = k_raw.shape[0]

        if not state.covers_prefix:
            if offset != 0:
                return None
            # A turn starting from zero rebuilds the whole index, so an earlier
            # gap no longer matters.
            state.block_keys = None
            state.recent = None
            state.recent_start = 0
            state.covers_prefix = True

        # A caller that moved the offset without going through trim().
        if state.block_keys is not None:
            valid = offset // self.compress_ratio
            if state.block_keys.shape[0] > valid:
                state.block_keys = state.block_keys[:valid] if valid > 0 else None

        # Extend the window. Decode adds one key, prefill adds the chunk.
        if state.recent is not None and state.recent_start + state.recent.shape[0] == offset:
            state.recent = np.concatenate([state.recent, k_raw], axis=0)
        else:
            state.recent = k_raw
            state.recent_start = offset

        # Pool every block the window completes, always starting at the first
        # position not yet blocked, so a block key can never be filed under an
        # index that does not describe it.
        if state.covers_prefix and state.recent is not None:
            held = 0 if state.block_keys is None else state.block_keys.shape[0]
            block_start = held * self.compress_ratio
            if block_start < state.recent_start:
                state.covers_prefix = False
            else:
                low = block_start - state.recent_start
                n_new = (state.recent.shape[0] - low) // self.compress_ratio
                if n_new > 0:
                    complete = state.recent[low: low + n_new * self.compress_ratio]
                    pooled = complete.reshape(n_new, self.compress_ratio,
                                              self.head_dim).mean(axis=1)
                    normed = _rms_norm(pooled, self.k_norm, self.eps)
                    pos = (np.arange(n_new) + held) * self.compress_ratio
                    blocks = _rope(normed, pos, self.rotary_dim, self.theta)
                    state.block_keys = (blocks if state.block_keys is None
                                        else np.concatenate([state.block_keys, blocks], axis=0))

        # Hold the window to its bound now that the blocks are built.
        if state.recent is not None and state.recent.shape[0] > RETAINED_KEYS:
            drop = state.recent.shape[0] - RETAINED_KEYS
            state.recent = state.recent[drop:]
            state.recent_start += drop

        if L > self.DENSE_PREFILL_L:
            return None

        total = offset + L
        if total <= self.budget or state.block_keys is None:
            return None
        # Both bounds come from the array about to be partitioned; deriving the
        # count from the offset instead lets them disagree, which aborts inside
        # argpartition rather than degrading the selection.
        held_blocks = state.block_keys.shape[0]
        n_blocks = min(total // self.compress_ratio, held_blocks)
        if n_blocks <= self.block_topk:
            return None

        qn = _rms_norm(q, self.q_norm, self.eps)                      # (L, heads, hd)
        qr = _rope(qn.transpose(1, 0, 2), np.arange(offset, offset + L),
                   self.rotary_dim, self.theta)                       # (heads, L, hd)
        keys = state.block_keys[:n_blocks]
        scores = np.maximum(qr @ keys.T, 0.0).sum(axis=0) * self.scale  # (L, n_blocks)
        tail_len = total - n_blocks * self.compress_ratio
        if L == 1:
            top = np.argpartition(scores[0], -self.block_topk)[-self.block_topk:]
        else:
            top = np.unique(
                np.argpartition(scores, -self.block_topk, axis=-1)[:, -self.block_topk:]
            )
            # L queries of top-512 can unique to more blocks than the ANE graph
            # can hold (budget / ratio). Keep the highest-scoring unique blocks.
            room_blocks = max(0, (self.budget - tail_len) // self.compress_ratio)
            if top.size > room_blocks:
                block_score = scores.max(axis=0)
                top = top[np.argsort(-block_score[top], kind="stable")][:room_blocks]
        idx = (top[:, None] * self.compress_ratio + np.arange(self.compress_ratio)).reshape(-1)
        if tail_len > 0:
            idx = np.concatenate([idx, np.arange(n_blocks * self.compress_ratio, total)])
        return idx.astype(np.int32)
