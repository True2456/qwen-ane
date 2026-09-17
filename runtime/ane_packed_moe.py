#!/usr/bin/env python3
"""Packed-K AneDynamicLinear MoE (GPU-off).

Decode: one K=10 pair, page the current top-10 into wimg.
Prefill: one K=32 pair, used in 32-expert packs so a tile's union fits.

    FLASHNEXT_MOE=anepacked
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("Q38_ANE_ENGINE", str(ROOT))

from runtime.host_fastpath import silu  # noqa: E402
from runtime.q38_ane_engine import AneDynamicLinear  # noqa: E402

H = 2560
I = 640
K_PIN = 10
K_PRE = 32
S = 32


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


class PackedAneBank:
    """One compiled pair per shape, shared by all 48 layers."""

    _inst: PackedAneBank | None = None

    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.dec_gu = AneDynamicLinear.compile(H, K_PIN * 2 * I, S)
        self.dec_dn = AneDynamicLinear.compile(K_PIN * I, H, S)
        self.pre_gu = AneDynamicLinear.compile(H, K_PRE * 2 * I, S)
        self.pre_dn = AneDynamicLinear.compile(K_PRE * I, H, S)
        if None in (self.dec_gu, self.dec_dn, self.pre_gu, self.pre_dn):
            raise RuntimeError("PackedAneBank: AneDynamicLinear compile failed")
        print(
            f"  packed ANE MoE  K=10 + K=32  S={S}  "
            f"{time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        self.x_dec = np.zeros((S, H), np.float32)
        self.h_dec = np.zeros((S, K_PIN * I), np.float32)
        self.h_pre = np.zeros((S, K_PRE * I), np.float32)

    @classmethod
    def get(cls) -> PackedAneBank:
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst


class AnePackedMoe:
    """ResidentMoe-shaped wrapper: apply / routed_multi, ANE packed convs."""

    def __init__(self, host_moe) -> None:
        self.moe = host_moe
        self.layer = int(host_moe.layer)
        self.store = host_moe.store
        self.bank = PackedAneBank.get()
        # Per-layer host mirrors. ANE wimg is shared, so write_weight still
        # runs every layer; sticky only avoids store fetches.
        self.w_gu = np.zeros((K_PIN * 2 * I, H), np.float16)
        self.w_dn = np.zeros((H, K_PIN * I), np.float16)
        self._slot_ids = [-1] * K_PIN
        self.last_copy = K_PIN
        self.last_reuse = 0
        self.nbytes = self.w_gu.nbytes + self.w_dn.nbytes

    def _slabs(self, eids) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.store.ensure_many(
            self.layer, eids, self.moe.gu, self.moe.dn
        )

    def _pack10(self, eids: np.ndarray) -> int:
        ids = [int(e) for e in np.asarray(eids).reshape(-1)[:K_PIN]]
        have = {e: j for j, e in enumerate(self._slot_ids) if e >= 0}
        need = [e for e in ids if e not in have]
        if ids == self._slot_ids and not need:
            return 0
        slabs = {e: s for e, s in zip(need, self._slabs(need))} if need else {}
        gu, dn = self.w_gu, self.w_dn
        new_gu = np.empty_like(gu)
        new_dn = np.empty_like(dn)
        copies = 0
        for j, e in enumerate(ids):
            if e in slabs:
                g, d = slabs[e]
                new_gu[j * 2 * I:(j + 1) * 2 * I] = np.ascontiguousarray(g, np.float16)
                new_dn[:, j * I:(j + 1) * I] = np.ascontiguousarray(d, np.float16)
                copies += 1
            else:
                src = have[e]
                new_gu[j * 2 * I:(j + 1) * 2 * I] = gu[src * 2 * I:(src + 1) * 2 * I]
                new_dn[:, j * I:(j + 1) * I] = dn[:, src * I:(src + 1) * I]
        gu[:] = new_gu
        dn[:] = new_dn
        self._slot_ids = ids
        return copies

    def _shared(self, x: np.ndarray) -> np.ndarray:
        m = self.moe
        sg = silu(x @ m.shared_gate.T)
        su = x @ m.shared_up.T
        shared = (sg * su) @ m.shared_down.T
        return _sigmoid(x @ m.shared_sgate.T) * shared

    def _routed10(self, x1: np.ndarray, ids, scores) -> np.ndarray:
        copies = self._pack10(ids)
        self.last_copy = copies
        self.last_reuse = K_PIN - copies
        b = self.bank
        # Shared ANE surface — always page this layer's pack before eval.
        b.dec_gu.write_weight(self.w_gu)
        b.dec_dn.write_weight(self.w_dn)
        b.x_dec[:] = 0
        b.x_dec[0] = x1
        y_gu = b.dec_gu.evaluate(b.x_dec)
        if y_gu is None:
            raise RuntimeError("packed decode gate_up eval failed")
        g = y_gu[0].reshape(K_PIN, 2 * I)
        act = silu(g[:, :I]) * g[:, I:] * np.asarray(scores, np.float32).reshape(K_PIN, 1)
        b.h_dec[:] = 0
        b.h_dec[0] = act.reshape(K_PIN * I)
        y_dn = b.dec_dn.evaluate(b.h_dec)
        if y_dn is None:
            raise RuntimeError("packed decode down eval failed")
        return y_dn[0]

    def _routed_pack32(
        self,
        x_pad: np.ndarray,
        pack_ids: list[int],
        score_mat: np.ndarray,
        n: int,
    ) -> np.ndarray:
        k = len(pack_ids)
        if k < K_PRE:
            pack_ids = pack_ids + [pack_ids[0]] * (K_PRE - k)
            pad = np.zeros((score_mat.shape[0], K_PRE - k), np.float32)
            score_mat = np.concatenate([score_mat, pad], axis=1)
        slabs = self._slabs(pack_ids)
        gu = np.zeros((K_PRE * 2 * I, H), np.float16)
        dn = np.zeros((H, K_PRE * I), np.float16)
        for j, (g, d) in enumerate(slabs):
            gu[j * 2 * I:(j + 1) * 2 * I] = np.ascontiguousarray(g, np.float16)
            dn[:, j * I:(j + 1) * I] = np.ascontiguousarray(d, np.float16)
        b = self.bank
        b.pre_gu.write_weight(gu)
        b.pre_dn.write_weight(dn)
        y_gu = b.pre_gu.evaluate(x_pad)
        if y_gu is None:
            raise RuntimeError("packed prefill gate_up eval failed")
        g = y_gu[:n].reshape(n, K_PRE, 2 * I)
        act = silu(g[:, :, :I]) * g[:, :, I:] * score_mat[:, :, None]
        b.h_pre[:] = 0
        b.h_pre[:n] = act.reshape(n, K_PRE * I)
        y_dn = b.pre_dn.evaluate(b.h_pre)
        if y_dn is None:
            raise RuntimeError("packed prefill down eval failed")
        return y_dn[:n]

    def apply(self, x, ids, scores):
        t = time.perf_counter()
        x = np.ascontiguousarray(np.asarray(x, np.float32).reshape(-1, H))
        routed = self._routed10(x[0], ids, scores)
        y = routed + self._shared(x)[0]
        self.moe.last_copy = self.last_copy
        self.moe.last_reuse = self.last_reuse
        return y.reshape(1, -1), (time.perf_counter() - t) * 1e3

    def routed_multi(self, x_k, ids_k, scores_k, shared_k=None, hyp_k=None, inj_k=None):
        x = np.ascontiguousarray(np.asarray(x_k, np.float32).reshape(-1, H))
        ids = np.asarray(ids_k, np.int32).reshape(x.shape[0], -1)
        sc = np.asarray(scores_k, np.float32).reshape(x.shape[0], -1)
        n = x.shape[0]
        if n == 1:
            routed = self._routed10(x[0], ids[0], sc[0]).reshape(1, H)
        else:
            routed = np.zeros((n, H), np.float32)
            for lo in range(0, n, S):
                hi = min(n, lo + S)
                tile = x[lo:hi]
                t_ids = ids[lo:hi]
                t_sc = sc[lo:hi]
                union: list[int] = []
                seen: set[int] = set()
                for row in t_ids:
                    for e in row.tolist():
                        e = int(e)
                        if e not in seen:
                            seen.add(e)
                            union.append(e)
                nt = hi - lo
                x_pad = np.zeros((S, H), np.float32)
                x_pad[:nt] = tile
                for p0 in range(0, max(len(union), 1), K_PRE):
                    pack = union[p0:p0 + K_PRE]
                    sm = np.zeros((nt, len(pack)), np.float32)
                    index = {e: j for j, e in enumerate(pack)}
                    for ti, (erow, srow) in enumerate(zip(t_ids, t_sc)):
                        for e, s in zip(erow.tolist(), srow.tolist()):
                            j = index.get(int(e))
                            if j is not None:
                                sm[ti, j] += float(s)
                    routed[lo:hi] += self._routed_pack32(x_pad, pack, sm, nt)

        if shared_k is not None:
            y_tot = routed + np.asarray(shared_k, np.float32).reshape(n, -1)
        else:
            y_tot = routed + self._shared(x)

        if hyp_k is not None and inj_k is not None:
            inj = np.asarray(inj_k, np.float32).reshape(n, -1, 1)
            hyp = np.asarray(hyp_k, np.float32).reshape(n, -1)
            hc = inj.shape[1]
            y4 = y_tot.reshape(n, 1, H)
            injection = (inj * y4).reshape(n, hc * H)
            return (hyp + injection).reshape(1, n, -1)
        return y_tot.reshape(1, n, H)
