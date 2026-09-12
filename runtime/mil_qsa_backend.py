"""MIL int8 QSA layer backend — the sibling of `mil_gdn_backend`.

Replaces the Core AI folded `qsa_step` graph for the 12 sparse-attention
layers. Same contract: in goes the residual stream plus the host-selected KV
window, out come mixed / hyper / inj / shared / new_k / new_v, so decode can
swap one for the other without touching anything else.

Measured on layer 3 (`probes/flashnext_mil_qsa_layer.py`), k=1:

    key window    MIL int8      Core AI fp16
    256           0.941 ms      2.96 ms        3.1x
    512           0.963 ms
    2048          1.333 ms                     2.2x

    mixed rel 0.028   hyper 0.026   shared 0.042   new_k 0.014

Error matches the GDN MIL layer (mixed 0.027), against a shipping MLX 4-bit
build that measures 0.102 per tensor.

Two things the attention needs that the GDN layer did not:

  * every I/O last dim must be a multiple of 32. An IOSurface row is padded to
    64 bytes, so a key axis of `max_s + 1` makes the host and the ANE disagree
    about the stride, and the output is silently wrong rather than rejected.
    The key axis is therefore `m + S`, the same width the torch reference uses,
    with the S-1 dead slots masked off.
  * decode is one live query, so grouped-query attention is a single matmul of
    [1, HKV, G, HD] against [1, HKV, KV, HD]^T. No head expansion, no rank-5
    tensor, and the mask broadcasts from [1, 1, 1, KV].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts"), str(_ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runtime.q38_ane_engine import AneEngine, _iosurface_view

from export_flashnext_coreai import (  # noqa: E402
    H, HC, HC_W, SEQ_DEFAULT, QSA_HD, QSA_HKV, QSA_HQ, QSA_ROTARY, QSA_MASK,
)

S = SEQ_DEFAULT
HALF = QSA_ROTARY // 2
KVC = QSA_HKV * QSA_HD
G = QSA_HQ // QSA_HKV
IDX_W = 640


class MilQsaLayer:
    """One QSA layer, one program per key-window rung.

    Rungs cost a resident model each, and the ANE runs out near 80, so keep the
    list short: a narrow rung for early tokens and the widest one for the rest
    is enough, since the indexer caps selection at the budget anyway.
    """

    __slots__ = ("layer", "k", "_progs", "_hcn", "_eng", "_bufs", "_front")

    def __init__(self, layer: int, weights, ref, qsa, rungs, k: int = 1,
                 engine: AneEngine | None = None):
        import flashnext_mil_qsa_layer as QL
        self._eng = engine or QL.eng
        QL.LAYER[0] = int(layer)
        QL.KTOK[0] = int(k)
        self.k = int(k)
        self.layer = int(layer)
        self._progs: dict[int, object] = {}
        self._hcn = None
        for m in sorted(set(int(r) for r in rungs)):
            if m % 32:
                raise ValueError(f"QSA rung {m} is not a multiple of 32")
            QL.KVM[0] = m
            prog, hcn = QL.build_layer(weights, ref, qsa)
            self._progs[m] = prog
            self._hcn = hcn
        # One hc_norm row serves both programs: the front slices the attn half
        # and the big graph the mlp half.
        self._front, _ = QL.build_front_layer(weights, ref)
        self._bufs = {
            m: {"k": np.zeros((KVC, m), np.float16),
                "v": np.zeros((KVC, m), np.float16),
                "mask": np.full((G * self.k, m + S), QSA_MASK, np.float16)}
            for m in self._progs
        }

    @property
    def rungs(self) -> list[int]:
        return sorted(self._progs)

    def rung_for(self, need: int) -> int:
        return next((m for m in self.rungs if m >= need), self.rungs[-1])

    def front(self, x_bc1s, n: int):
        """Attention mixer plus the indexer's projection, on the ANE.

        Returns (mixed, inj, qk) over n slots. `mixed` and `inj` are handed
        straight back to `__call__`, so nothing is computed twice.
        """
        p = self._front
        x = np.asarray(x_bc1s, np.float16).reshape(HC_W, S)
        for surf, val in zip(p._in_surfs, (x, self._hcn)):
            with _iosurface_view(surf, val.shape, np.float16) as dst:
                np.copyto(dst, val)
        if not self._eng.submit(p, procedure_index=0):
            raise RuntimeError(f"QSA MIL front {self.layer}: submit failed")
        out = []
        for idx, c in ((0, H), (1, HC), (2, IDX_W)):
            with _iosurface_view(p._out_surfs[idx], (c, S), np.float16) as o:
                out.append(np.array(o[:, :n], np.float32))
        return out[0], out[1], out[2].T

    def __call__(self, x_bc1s, keys, values, cos, sin, nsel: int, m: int,
                 n: int | None = None, mixed=None, inj=None):
        """x is (1, HC_W, 1, S); keys/values (KVC, nsel); cos/sin (HALF, S).

        `nsel` selected keys sit at the head of the window and this pass's own
        tokens at m, m+1, ... Returns (mixed, hyper, inj, shared, new_k, new_v),
        the first four BC1S over n slots and the last two (KVC, n).
        """
        prog = self._progs[m]
        b = self._bufs[m]
        w = self.k if n is None else int(n)
        b["k"][:, :nsel] = keys
        b["k"][:, nsel:] = 0
        b["v"][:, :nsel] = values
        b["v"][:, nsel:] = 0
        # One mask row per (query head group, token): a token attends to the
        # selected prefix and to this block's own tokens up to itself.
        row = b["mask"][:self.k]
        row[:] = QSA_MASK
        row[:, :nsel] = 0
        for t in range(self.k):
            row[t, m:m + t + 1] = 0
        for g in range(1, G):
            b["mask"][g * self.k:(g + 1) * self.k] = row
        x = np.ascontiguousarray(np.asarray(x_bc1s, np.float16).reshape(HC_W, S))
        mx_ = np.zeros((H, S), np.float16)
        ij_ = np.zeros((HC, S), np.float16)
        mx_[:, :w] = mixed
        ij_[:, :w] = inj
        for surf, val in zip(prog._in_surfs, (x, cos, sin, self._hcn,
                                              b["k"], b["v"], b["mask"],
                                              mx_, ij_)):
            with _iosurface_view(surf, val.shape, np.float16) as dst:
                np.copyto(dst, val)
        if not self._eng.submit(prog, procedure_index=0):
            raise RuntimeError(f"QSA MIL layer {self.layer}: submit failed")
        # surfaces bind alphabetically: t_newk, u_shared, v_mixed, w_hyper,
        # x_inj, y_newv
        out = {}
        for idx, nm, shape in ((0, "new_k", (KVC, S)), (1, "shared", (H, S)),
                               (2, "mixed", (H, S)), (3, "hyper", (HC_W, S)),
                               (4, "inj", (HC, S)), (5, "new_v", (KVC, S))):
            with _iosurface_view(prog._out_surfs[idx], shape, np.float16) as o:
                out[nm] = np.array(o[:, :w], np.float32)
        def bc(nm, c):
            return out[nm].reshape(1, c, 1, w)
        return (bc("mixed", H), bc("hyper", HC_W), bc("inj", HC), bc("shared", H),
                out["new_k"], out["new_v"])


def build_layers(layer_indices, loader_fn, rungs, k: int = 1, engine=None):
    """Compile a MIL QSA layer per index. Returns {index: MilQsaLayer}."""
    from export_flashnext_coreai import FlashNextQSADecode
    from flashnext_mil_qsa_layer import _Ref

    out = {}
    for li in layer_indices:
        loader, w = loader_fn(li)
        try:
            qsa = FlashNextQSADecode(max_s=max(rungs)).eval().half()
            qsa.load_from_layer(w)
            out[li] = MilQsaLayer(li, w, _Ref(w), qsa, rungs, k, engine)
        finally:
            loader.close()
    return out
