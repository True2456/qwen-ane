"""Does a k-step GDN recurrence cost k times a 1-step one?

The ANE graph already pays for 32 padded token slots, and a 136 MB projection
takes the same time at seq=1 and seq=32. If the recurrence tail is likewise
latency-bound rather than work-bound, then verifying k drafted tokens in one
submit costs little more than decoding one - which is what makes speculative
decoding worth building on this path.

Exports CompactTail unrolled k times: params carry 6 rows per step, the state
threads through, and the k outputs land in the first k of the 32 slots.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "probes"))

from export_flashnext_coreai import (  # noqa: E402
    FlashNextGdnOnly, _load_layer0, _export, HV, DK, DV, GDN_Y,
)
from flashnext_compact_tail import CompactTail  # noqa: E402


class MultiStepTail(torch.nn.Module):
    """CompactTail with the recurrence unrolled ``steps`` times."""

    def __init__(self, original, steps: int):
        super().__init__()
        self.steps = int(steps)
        self.out_proj = original.out_proj
        self.norm_w = original.norm_w
        self.register_buffer("eps", torch.tensor(1e-6, dtype=torch.float16))

    def forward(self, params, state):
        ys = []
        for i in range(self.steps):
            b = 6 * i
            q, k, v = params[:, :, b + 0:b + 1, :], params[:, :, b + 1:b + 2, :], params[:, :, b + 2:b + 3, :]
            decay, beta, z = params[:, :, b + 3:b + 4, :], params[:, :, b + 4:b + 5, :], params[:, :, b + 5:b + 6, :]
            state = state * decay
            memory = (state * k).sum(-1, keepdim=True)
            delta = (v.transpose(-1, -2) - memory) * beta.transpose(-1, -2)
            state = state + delta * k
            y = (state * q).sum(-1, keepdim=True).transpose(-1, -2)
            y = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + self.eps)
            y = y * self.norm_w * torch.sigmoid(z)
            ys.append(y.reshape(1, GDN_Y, 1, 1))
        y = torch.cat(ys, dim=-1) if len(ys) > 1 else ys[0]
        pad = 32 - y.shape[-1]
        if pad > 0:
            y = torch.cat([y, y[..., -1:].expand(1, GDN_Y, 1, pad)], dim=-1)
        return self.out_proj(y), state


def _params(steps: int):
    torch.manual_seed(915)
    p = torch.randn(1, HV, 6 * steps, DK).half() * .03
    for i in range(steps):
        p[:, :, 6 * i + 3, :] = .98
        p[:, :, 6 * i + 4, :] = .4
    return p


def _reference(single: CompactTail, params, state, steps: int):
    """Chain the verified 1-step tail ``steps`` times."""
    outs = []
    with torch.no_grad():
        for i in range(steps):
            out, state = single(params[:, :, 6 * i:6 * i + 6, :], state)
            outs.append(out[..., :1])
    return outs, state


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()

    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    from runtime.coreai_surfaces import wrap_ndarray
    import time

    loader, w = _load_layer0()
    old = FlashNextGdnOnly().eval().half()
    old.load_from_layer(w)
    single = CompactTail(old).eval().half()

    ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)

    for steps in a.steps:
        m = MultiStepTail(old, steps).eval().half()
        params = _params(steps)
        state = torch.randn(1, HV, DV, DK).half() * .02
        with torch.no_grad():
            out, ns = m(params, state)
        ref_outs, ref_state = _reference(single, params, state, steps)
        st_err = (ns - ref_state).abs().max().item()
        y_err = max((out[..., i:i + 1] - ref_outs[i]).abs().max().item() for i in range(steps))
        path = _export(m, (params, state), (["params", "state"], ["attn", "new_ssm"]),
                       f"multistep_tail_k{steps}")
        model = await AIModel.load(str(path), specialization_options=spec)
        fn = model.load_function("main")
        feeds = {"params": wrap_ndarray(params.numpy()), "state": wrap_ndarray(state.numpy())}
        for _ in range(5):
            await fn(feeds)
        ts = []
        for _ in range(a.reps):
            t = time.perf_counter()
            await fn(feeds)
            ts.append(time.perf_counter() - t)
        ms = float(np.median(ts)) * 1e3
        print(f"  k={steps:<2} {ms:6.3f} ms   {ms / steps:6.3f} ms/token   "
              f"params {params.numel() * 2 / 1024:.0f} KB   "
              f"vs chained 1-step: state {st_err:.5f}  y {y_err:.5f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
