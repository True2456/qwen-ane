"""A QSA layer with its mixers folded in, matching what pure_step does for GDN.

The 36 GDN layers run attn mix + attention + recombine + mlp mix inside one ANE
graph. The 12 QSA layers do not: each pays two `host_gated_residual_cached`
passes and two `host_recombine` over a 10240-wide stream, in numpy, per token
batch. Measured over a 16-token prefill chunk that host work is ~8 ms/layer and
dominates the ~385 ms of host time per chunk.

This folds them in. Inputs and outputs match the multi-token QSA graph plus the
three mixer streams, so it drops into both prefill and decode. Verified against
the host mixer path over k slots.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "probes"))

from export_flashnext_coreai import (  # noqa: E402
    FlashNextQSADecode, _export, _load_layer, _bsh_to_bc1s, _bc1s_to_bsh,
    host_gated_residual_cached, host_recombine, HostMixPack, HostPrep,
    H, HC_W, QSA_HKV, QSA_HD, QSA_ROTARY, QSA_MASK, SEQ_DEFAULT,
)
from flashnext_pure_step import FlashNextGatedMix, _recombine  # noqa: E402
from flashnext_multitoken_step import SharedExpert  # noqa: E402
import flashnext_multitoken_qsa as Q  # noqa: E402

S = SEQ_DEFAULT
KV_C = QSA_HKV * QSA_HD


class QsaStep(nn.Module):
    """attn mix + QSA + recombine + mlp mix (+ shared expert) in one graph."""

    def __init__(self, w, shared=None):
        super().__init__()
        self.attn = FlashNextGatedMix()
        self.attn.load(w, "attn_hyper_connection")
        self.qsa = FlashNextQSADecode().eval().half()
        self.qsa.load_from_layer(w)
        self.mlp = FlashNextGatedMix()
        self.mlp.load(w, "mlp_hyper_connection")
        self.shared = None
        if shared is not None:
            self.shared = SharedExpert(w).eval().half()
            self.shared.load(*shared)

    def forward(self, x, k_cache, v_cache, cos, sin, mask):
        mixed, hyper, inj = self.attn(x)
        attn, new_k, new_v = self.qsa(mixed, k_cache, v_cache, cos, sin, mask)
        h = _recombine(attn, hyper, inj)
        mixed2, hyper2, inj2 = self.mlp(h)
        if self.shared is None:
            return mixed2, hyper2, inj2, new_k, new_v
        return mixed2, hyper2, inj2, new_k, new_v, self.shared(mixed2)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--max-s", type=int, default=2048)
    ap.add_argument("--offset", type=int, default=8)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--export-all", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    Q.MAX_S = a.max_s

    from runtime.expert_bank import Mlx4ExpertBank, MLX4_DEFAULT, MlxSafe

    bank = Mlx4ExpertBank(MLX4_DEFAULT)

    def shared_for(i):
        sg, su, sd = bank.shared_fp32(i)
        src = MlxSafe(MLX4_DEFAULT)
        try:
            sgate = np.asarray(
                src.f32(f"model.layers.{i}.mlp.shared_expert_gate.weight"), np.float32)
        finally:
            src.close()
        return (sg, su, sd, sgate)

    names_in = ["x", "k_cache", "v_cache", "cos", "sin", "mask"]
    names_out = ["mixed", "hyper", "inj", "new_k", "new_v", "shared"]

    def example(k):
        rng = np.random.default_rng(21)
        x = np.zeros((1, HC_W, 1, S), np.float16)
        x[..., :k] = (rng.standard_normal((1, HC_W, 1, k)) * 0.05).astype(np.float16)
        kh = (rng.standard_normal((a.max_s, KV_C)) * 0.05).astype(np.float16)
        vh = (rng.standard_normal((a.max_s, KV_C)) * 0.05).astype(np.float16)
        hs = np.zeros((1, k, H), np.float32)   # unused; feeds() only needs shapes
        _, kc, vc, cos, sin, mask = Q.feeds(hs, kh, vh, a.offset, k)
        return (x, kc, vc, cos, sin, mask)

    if a.export_all:
        k = a.steps[0]
        for i in [x for x in range(48) if x % 4 == 3]:
            out = Path(f"artifacts/coreai/flashnext_qsa_step_k{k}_L{i}_m{a.max_s}.aimodel")
            if out.is_dir():
                print(f"  L{i} exists, skipping", flush=True)
                continue
            ld, lw = _load_layer(i)
            m = QsaStep(lw, shared_for(i)).eval().half()
            _export(m, tuple(_t(v) for v in example(k)), (names_in, names_out),
                    f"qsa_step_k{k}_L{i}_m{a.max_s}")
            ld.close()
            print(f"  exported QSA step L{i} k={k} max_S={a.max_s}", flush=True)
        return

    ld, w = _load_layer(a.layer)
    sh = shared_for(a.layer)
    m = QsaStep(w, sh).eval().half()
    host_attn = HostMixPack(w, "attn_hyper_connection")
    host_mlp = HostMixPack(w, "mlp_hyper_connection")
    ref_qsa = FlashNextQSADecode().eval().half()
    ref_qsa.load_from_layer(w)

    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from runtime.coreai_surfaces import wrap_ndarray
    ane = [x for x in ComputeUnitKind.available_kinds() if str(x) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)

    for k in a.steps:
        ex = example(k)
        with torch.no_grad():
            got = m(*[_t(v) for v in ex])
        # host reference: mixers in numpy, same QSA module in the middle
        x32 = np.asarray(ex[0], np.float32)
        hm, hy, ij = host_gated_residual_cached(x32, host_attn)
        with torch.no_grad():
            attn, nk, nv = ref_qsa(_t(np.asarray(hm, np.float16)),
                                   *[_t(v) for v in ex[1:]])
        post = host_recombine(np.asarray(attn.float().numpy(), np.float32), hy, ij)
        m2, h2, i2 = host_gated_residual_cached(post, host_mlp)

        def rel(p, q):
            p = np.asarray(p, np.float64)
            q = np.asarray(q, np.float64)
            return float(np.linalg.norm(p - q) / max(np.linalg.norm(q), 1e-12))

        e_mixed = rel(got[0].float().numpy()[..., :k], m2[..., :k])
        e_hyper = rel(got[1].float().numpy()[..., :k], h2[..., :k])
        e_k = rel(got[3].float().numpy()[..., :k], nk.float().numpy()[..., :k])
        line = (f"  k={k:<3} vs host mixers: mixed {e_mixed:.5f}  hyper {e_hyper:.5f}  "
                f"new_k {e_k:.5f}")
        if a.bench:
            path = _export(m, tuple(_t(v) for v in ex), (names_in, names_out),
                           f"qsa_step_probe_k{k}_m{a.max_s}")
            model = await AIModel.load(str(path), specialization_options=spec)
            fn = model.load_function("main")
            feed = {n: wrap_ndarray(v) for n, v in zip(names_in, ex)}
            for _ in range(4):
                await fn(feed)
            ts = []
            for _ in range(a.reps):
                t = time.perf_counter()
                await fn(feed)
                ts.append(time.perf_counter() - t)
            ms = float(np.median(ts)) * 1e3
            line += f"   ANE {ms:6.3f} ms"
        print(line, flush=True)
    ld.close()


if __name__ == "__main__":
    asyncio.run(main())
