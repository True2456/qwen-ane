"""A full Flash-Next layer that consumes k real tokens per ANE submit.

Everything in `pure_step` except the recurrence already runs over all 32 slots,
and at the real layer size seq=32 is the *fast* shape (1.06 ms vs 2.20 ms at
seq=1). Today 31 of those slots carry padding. This fills the first k of them:

  * the 4-tap depthwise conv becomes a causal shift across slots, with the
    3-deep cache supplying only the tokens before the batch;
  * the GDN recurrence is unrolled k times (probes/flashnext_multistep_tail.py:
    k=8 costs 2.3x a single step, not 8x);
  * the mixers, recombine and out_proj were already per-slot.

Correctness is checked against k sequential single-token steps of the existing
verified path, which is what decode does today.
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
    FlashNextFront, FlashNextGdnOnly, Conv1x1, _export, _fp16,
    _load_layer0, _load_layer,
    H, I, HC_W, HK, HV, DK, DV, QKV, GDN_Y, SEQ_DEFAULT,
)
from flashnext_compact_tail import CompactTail  # noqa: E402
from flashnext_connected_gdn import Connected, Production  # noqa: E402
from flashnext_pure_step import FlashNextGatedMix, _recombine  # noqa: E402

S = SEQ_DEFAULT


class MultiTokenFront(nn.Module):
    """`FlashNextFront` with the depthwise conv shifted causally across slots.

    Slot i must see taps from tokens i-3..i. The single-token graph applies the
    same cached columns to every slot, which is only right for slot 0.
    """

    def __init__(self, front: FlashNextFront, steps: int):
        super().__init__()
        self.in_proj = front.in_proj
        self.tap0, self.tap1 = front.tap0, front.tap1
        self.tap2, self.tap3 = front.tap2, front.tap3
        self.steps = int(steps)

    def forward(self, h, conv_pack):
        yin = self.in_proj(h)
        qkv, rest = yin[:, :QKV], yin[:, QKV:]
        s0, s1, s2 = conv_pack.chunk(3, dim=1)
        # Three history columns then the batch: slot i of `seq` is token i-3.
        seq = torch.cat([s0[..., :1], s1[..., :1], s2[..., :1], qkv], dim=-1)
        conv_pre = (seq[..., 0:S] * self.tap0 + seq[..., 1:S + 1] * self.tap1
                    + seq[..., 2:S + 2] * self.tap2 + seq[..., 3:S + 3] * self.tap3)
        k = self.steps
        # Carry the last three consumed tokens, broadcast the way decode reads them.
        new_pack = torch.cat(
            [seq[..., k + i:k + i + 1].expand(1, QKV, 1, S) for i in range(3)], dim=1
        )
        return torch.cat([conv_pre, rest], dim=1), new_pack


class MultiTokenConnected(nn.Module):
    """`Connected` over k slots, threading one recurrent state through them."""

    def __init__(self, connected: Connected, steps: int):
        super().__init__()
        self.steps = int(steps)
        self.front = MultiTokenFront(connected.front, steps)
        self.tail = connected.tail
        for name in ("gamma", "dt", "eps", "qscale", "half_coeff"):
            self.register_buffer(name, getattr(connected, name))

    def _params(self, yin, i):
        one = yin[..., i:i + 1]
        packed = one[:, :QKV].reshape(1, QKV // DK, 1, DK)
        halfx = packed * self.half_coeff
        c = halfx + halfx * torch.tanh(halfx)
        q, k, v = c[:, :HK], c[:, HK:2 * HK], c[:, 2 * HK:]
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + self.eps) * self.qscale
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + self.eps)
        q = q.repeat_interleave(HV // HK, dim=1)
        k = k.repeat_interleave(HV // HK, dim=1)
        z = one[:, QKV:QKV + GDN_Y].reshape(1, HV, 1, DK)
        b = one[:, QKV + GDN_Y:QKV + GDN_Y + HV]
        a = one[:, QKV + GDN_Y + HV:]
        decay = torch.pow(torch.sigmoid(-(a + self.dt)), self.gamma).expand(1, HV, 1, DK)
        beta = torch.sigmoid(b).expand(1, HV, 1, DK)
        return torch.cat((q, k, v, decay, beta, z), dim=2)

    def forward(self, h, conv, state):
        yin, newconv = self.front(h, conv)
        ys = []
        for i in range(self.steps):
            p = self._params(yin, i)
            state = state * p[:, :, 3:4, :]
            memory = (state * p[:, :, 1:2, :]).sum(-1, keepdim=True)
            delta = (p[:, :, 2:3, :].transpose(-1, -2) - memory) * p[:, :, 4:5, :].transpose(-1, -2)
            state = state + delta * p[:, :, 1:2, :]
            y = (state * p[:, :, 0:1, :]).sum(-1, keepdim=True).transpose(-1, -2)
            y = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + self.tail.eps)
            y = y * self.tail.norm_w * torch.sigmoid(p[:, :, 5:6, :])
            ys.append(y.reshape(1, GDN_Y, 1, 1))
        y = torch.cat(ys, dim=-1)
        y = torch.cat([y, y[..., -1:].expand(1, GDN_Y, 1, S - self.steps)], dim=-1)
        return self.tail.out_proj(y), state, newconv


class SharedExpert(nn.Module):
    """The dense always-on expert, moved off the GPU.

    Only 9.8 MB of fp16 — 7% on top of a 136 MB layer graph, about 0.07 ms of
    ANE weight streaming. On the GPU it costs ~0.5 ms per layer (24 ms/token
    over 48) because each call pays the wake penalty of a device that just sat
    idle through an ANE submit. The routed experts stay on the GPU; only this
    fixed part moves.
    """

    def __init__(self, w):
        super().__init__()
        self.gate = Conv1x1(H, I)
        self.up = Conv1x1(H, I)
        self.down = Conv1x1(I, H)
        self.sgate = Conv1x1(H, 1)

    def load(self, gate, up, down, sgate) -> None:
        self.gate.set_w(gate)
        self.up.set_w(up)
        self.down.set_w(down)
        self.sgate.set_w(np.asarray(sgate, np.float32).reshape(1, H))

    def forward(self, x):
        g = self.gate(x)
        h = self.down(g * torch.sigmoid(g) * self.up(x))
        return h * torch.sigmoid(self.sgate(x))


class StatefulMultiTokenStep(nn.Module):
    """`MultiTokenStep` with the recurrent buffers as Core AI state.

    Core AI allocates a fresh IOSurface for every output of every submit and
    exposes no way to bind or release them, so a long prefill dies at ~64 chunks
    with `Failed to allocate storage for NDArray ... sk: ioSurface`. Buffers
    that `torch.export` sees mutated become stateful inputs, which Core AI then
    owns for the life of the function — no per-submit allocation, and no host
    round trip for conv/ssm either.
    """

    def __init__(self, w, steps: int, shared=None):
        super().__init__()
        self.inner = MultiTokenStep(w, steps, shared)
        self.register_buffer("conv", torch.zeros(1, 3 * QKV, 1, S, dtype=torch.float16))
        self.register_buffer("ssm", torch.zeros(1, HV, DV, DK, dtype=torch.float16))

    def forward(self, x):
        out = self.inner(x, self.conv, self.ssm)
        mixed2, hyper2, inj2, new_ssm, new_conv = out[:5]
        self.ssm.copy_(new_ssm)
        self.conv.copy_(new_conv)
        if len(out) > 5:
            return mixed2, hyper2, inj2, out[5]
        return mixed2, hyper2, inj2


class MultiTokenStep(nn.Module):
    """`FlashNextPureStep` consuming k tokens per submit."""

    def __init__(self, w, steps: int, shared=None):
        super().__init__()
        self.attn = FlashNextGatedMix()
        self.attn.load(w, "attn_hyper_connection")
        self.gdn = MultiTokenConnected(Connected(w).eval().half(), steps)
        self.mlp = FlashNextGatedMix()
        self.mlp.load(w, "mlp_hyper_connection")
        self.shared = None
        if shared is not None:
            self.shared = SharedExpert(w).eval().half()
            self.shared.load(*shared)

    def forward(self, x, conv, state):
        mixed, hyper, inj = self.attn(x)
        attn, state, conv = self.gdn(mixed, conv, state)
        h = _recombine(attn, hyper, inj)
        mixed2, hyper2, inj2 = self.mlp(h)
        if self.shared is None:
            return mixed2, hyper2, inj2, state, conv
        return mixed2, hyper2, inj2, state, conv, self.shared(mixed2)


def _single_reference(w, x, conv, state, steps: int):
    """k sequential single-token steps of the shipped path."""
    from flashnext_pure_step import FlashNextPureStep
    ref = FlashNextPureStep(w).eval().half()
    outs = []
    with torch.no_grad():
        for i in range(steps):
            xi = torch.zeros(1, HC_W, 1, S).half()
            xi[..., :1] = x[..., i:i + 1]
            m2, h2, i2, state, conv = ref(xi, conv, state)
            outs.append((m2[..., :1], h2[..., :1], i2[..., :1]))
    return outs, state, conv


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--shared", action="store_true",
                    help="fold the dense shared expert into the ANE graph")
    ap.add_argument("--export-all", action="store_true",
                    help="export every GDN layer at one k, no verify/bench")
    ap.add_argument("--name", default="multitoken_step",
                    help="asset basename")
    a = ap.parse_args()

    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from runtime.coreai_surfaces import wrap_ndarray

    ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)

    if a.export_all:
        steps = a.steps[0]
        from runtime.expert_bank import Mlx4ExpertBank, MLX4_DEFAULT, MlxSafe
        bank = Mlx4ExpertBank(MLX4_DEFAULT)
        gdn = [i for i in range(48) if i % 4 != 3]
        for i in gdn:
            out = Path(f"artifacts/coreai/flashnext_{a.name}_k{steps}_L{i}.aimodel")
            if out.is_dir():
                print(f"  L{i} exists, skipping", flush=True)
                continue
            ld, lw = _load_layer(i)
            shared_w = None
            if a.shared:
                sg, su, sd = bank.shared_fp32(i)
                src = MlxSafe(MLX4_DEFAULT)
                try:
                    sgate = np.asarray(
                        src.f32(f"model.layers.{i}.mlp.shared_expert_gate.weight"),
                        np.float32)
                finally:
                    src.close()
                shared_w = (sg, su, sd, sgate)
            torch.manual_seed(918)
            x = torch.zeros(1, HC_W, 1, S).half()
            x[..., :steps] = torch.randn(1, HC_W, 1, steps).half() * 0.05
            conv = torch.zeros(1, 3 * QKV, 1, S).half()
            state = torch.zeros(1, HV, DV, DK).half()
            m = MultiTokenStep(lw, steps, shared_w).eval().half()
            names = ["mixed", "hyper", "inj", "new_ssm", "new_conv"]
            if a.shared:
                names.append("shared")
            _export(m, (x, conv, state), (["x", "conv", "state"], names),
                    f"{a.name}_k{steps}_L{i}")
            ld.close()
            print(f"  exported L{i}", flush=True)
        return

    loader, w = (_load_layer0() if a.layer == 0 else _load_layer(a.layer))

    for steps in a.steps:
        torch.manual_seed(918)
        x = torch.zeros(1, HC_W, 1, S).half()
        x[..., :steps] = torch.randn(1, HC_W, 1, steps).half() * 0.05
        conv = torch.randn(1, 3 * QKV, 1, S).half() * 0.02
        state = torch.randn(1, HV, DV, DK).half() * 0.02

        shared_w = None
        if a.shared:
            from runtime.expert_bank import Mlx4ExpertBank, MLX4_DEFAULT, MlxSafe
            bank = Mlx4ExpertBank(MLX4_DEFAULT)
            sg, su, sd = bank.shared_fp32(a.layer)
            src = MlxSafe(MLX4_DEFAULT)
            try:
                sgate = np.asarray(
                    src.f32(f"model.layers.{a.layer}.mlp.shared_expert_gate.weight"),
                    np.float32)
            finally:
                src.close()
            shared_w = (sg, su, sd, sgate)

        m = MultiTokenStep(w, steps, shared_w).eval().half()
        with torch.no_grad():
            out_all = m(x, conv, state)
        mixed2, hyper2, inj2, ns, nc = out_all[:5]
        shared_out = out_all[5] if a.shared else None
        if shared_out is not None:
            xx = mixed2[..., :steps].reshape(1, H, steps).permute(0, 2, 1).float().numpy()
            g = xx @ shared_w[0].T
            ref_sh = ((g / (1 + np.exp(-g))) * (xx @ shared_w[1].T)) @ shared_w[2].T
            ref_sh = ref_sh / (1 + np.exp(-(xx @ shared_w[3].reshape(1, H).T)))
            got = shared_out[..., :steps].reshape(1, H, steps).permute(0, 2, 1).float().numpy()
            sh_rel = float(np.linalg.norm(got - ref_sh) / max(np.linalg.norm(ref_sh), 1e-12))
            print(f"  shared expert on ANE graph: rel {sh_rel:.5f}", flush=True)
        ref_outs, ref_state, ref_conv = _single_reference(w, x, conv, state, steps)

        def rel(p, q):
            p = np.asarray(p.float().numpy(), np.float64)
            q = np.asarray(q.float().numpy(), np.float64)
            return float(np.linalg.norm(p - q) / max(np.linalg.norm(q), 1e-12))

        y_err = max(rel(mixed2[..., i:i + 1], ref_outs[i][0]) for i in range(steps))
        s_err = rel(ns, ref_state)
        c_err = rel(nc[:, :QKV, :, :1], ref_conv[:, :QKV, :, :1])

        names = ["mixed", "hyper", "inj", "new_ssm", "new_conv"]
        if a.shared:
            names.append("shared")
        path = _export(m, (x, conv, state), (["x", "conv", "state"], names),
                       f"multitoken_step_k{steps}{'_shared' if a.shared else ''}")
        model = await AIModel.load(str(path), specialization_options=spec)
        fn = model.load_function("main")
        feeds = {"x": wrap_ndarray(x.numpy()), "conv": wrap_ndarray(conv.numpy()),
                 "state": wrap_ndarray(state.numpy())}
        for _ in range(5):
            await fn(feeds)
        ts = []
        for _ in range(a.reps):
            t = time.perf_counter()
            await fn(feeds)
            ts.append(time.perf_counter() - t)
        ms = float(np.median(ts)) * 1e3
        print(f"  k={steps:<2} {ms:6.3f} ms   {ms / steps:6.3f} ms/token   "
              f"vs {steps} single steps: mixed {y_err:.5f}  state {s_err:.5f}  conv {c_err:.5f}",
              flush=True)
    loader.close()


if __name__ == "__main__":
    asyncio.run(main())
