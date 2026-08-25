#!/usr/bin/env python3
"""Export the shared Qwen3.8-27B GDN prefill recurrence to CoreAI.

The graph contains no learned weights, so one artifact is shared by all 48
linear-attention layers.  Its packed ABI is deliberately simple: token-major
q/k/v/decay/beta records followed by the incoming [H,K,V] recurrent state.
The single output appends the final state after all token outputs.

Run with the repository's CoreAI environment, for example:

    GDN_TOKENS=32 ~/.rindi/venvs/coreai/bin/python \
        scripts/export_gdn_prefill_recurrence.py
"""

import asyncio
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


H = 48
D = 128
TOKENS = int(os.environ.get("GDN_TOKENS", "32"))
OUT_DIR = Path.home() / ".rindi/aimodels"


class GdnPrefillRecurrence(nn.Module):
    """Exact sequential prepared-input gated-delta recurrence."""

    def __init__(self, tokens: int):
        super().__init__()
        self.tokens = tokens

    def forward(self, xin):
        # xin: [(5*N*H + H*D), D].  Scalar records use column zero.
        n = self.tokens
        nh = n * H
        q = xin[0:nh].reshape(n, H, D)
        k = xin[nh:2 * nh].reshape(n, H, D)
        v = xin[2 * nh:3 * nh].reshape(n, H, D)
        decay = xin[3 * nh:4 * nh, 0].reshape(n, H)
        beta = xin[4 * nh:5 * nh, 0].reshape(n, H)
        state = xin[5 * nh:].reshape(H, D, D)

        outputs = []
        for token in range(n):
            state = state * decay[token, :, None, None]
            memory = (state * k[token, :, :, None]).sum(dim=1)
            delta = (v[token] - memory) * beta[token, :, None]
            state = state + k[token, :, :, None] * delta[:, None, :]
            outputs.append((state * q[token, :, :, None]).sum(dim=1))

        y = torch.stack(outputs, dim=0).reshape(n * H, D)
        return torch.cat((y, state.reshape(H * D, D)), dim=0)


def reference(x: np.ndarray, tokens: int) -> np.ndarray:
    nh = tokens * H
    q = x[0:nh].reshape(tokens, H, D).astype(np.float32)
    k = x[nh:2 * nh].reshape(tokens, H, D).astype(np.float32)
    v = x[2 * nh:3 * nh].reshape(tokens, H, D).astype(np.float32)
    decay = x[3 * nh:4 * nh, 0].reshape(tokens, H).astype(np.float32)
    beta = x[4 * nh:5 * nh, 0].reshape(tokens, H).astype(np.float32)
    state = x[5 * nh:].reshape(H, D, D).astype(np.float32)
    ys = []
    for token in range(tokens):
        state = state * decay[token, :, None, None]
        memory = np.sum(state * k[token, :, :, None], axis=1)
        delta = (v[token] - memory) * beta[token, :, None]
        state = state + k[token, :, :, None] * delta[:, None, :]
        ys.append(np.sum(state * q[token, :, :, None], axis=1))
        state = state.astype(np.float16).astype(np.float32)
    return np.concatenate((np.asarray(ys).reshape(tokens * H, D),
                           state.reshape(H * D, D)), axis=0)


def main() -> None:
    if TOKENS < 2:
        raise SystemExit("GDN_TOKENS must be at least 2")
    from coreai_torch import TorchConverter, get_decomp_table

    rows = 5 * TOKENS * H + H * D
    torch.manual_seed(7)
    xin = (torch.randn(rows, D) * 0.02).half()
    # Keep the synthetic recurrence stable and representative.
    nh = TOKENS * H
    xin[3 * nh:4 * nh, 0] = 0.99
    xin[4 * nh:5 * nh, 0] = 0.5
    model = GdnPrefillRecurrence(TOKENS).half().eval()
    with torch.no_grad():
        expected = model(xin).float().numpy()

    print(f"exporting N={TOKENS}, input={rows}x{D}, output="
          f"{TOKENS * H + H * D}x{D}")
    started = time.perf_counter()
    exported = torch.export.export(model, args=(xin,))
    exported = exported.run_decompositions(get_decomp_table())
    program = (TorchConverter()
               .add_exported_program(exported, input_names=["xin"],
                                     output_names=["y_state"])
               .to_coreai())
    program.optimize()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"qwen38_gdn_prefill_recurrence_n{TOKENS}.aimodel"
    if out.exists():
        shutil.rmtree(out)
    program.save_asset(out)
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"saved {out} ({size_mb:.1f} MB) in {time.perf_counter()-started:.1f}s")

    async def verify() -> None:
        from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
        opts = SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.neural_engine())
        loaded = await AIModel.load(str(out), specialization_options=opts)
        fn = loaded.load_function("main")
        nd = NDArray(xin.numpy())
        got = np.asarray((await fn({"xin": nd}))["y_state"].numpy(),
                         dtype=np.float32)
        rel = float(np.max(np.abs(got - expected)) /
                    (np.max(np.abs(expected)) + 1e-9))
        await fn({"xin": nd})
        runs = int(os.environ.get("GDN_RUNS", "10"))
        started_run = time.perf_counter()
        for _ in range(runs):
            await fn({"xin": nd})
        ms = (time.perf_counter() - started_run) * 1e3 / runs
        print(f"ANE rel_vs_export={rel:.6f} {ms:.3f} ms/eval")

    if os.environ.get("GDN_SKIP_BENCH", "0") != "1":
        asyncio.run(verify())


if __name__ == "__main__":
    main()
