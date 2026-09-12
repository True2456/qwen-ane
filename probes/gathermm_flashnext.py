#!/usr/bin/env python3
"""Does Apple GatherMM actually export and run for Flash-Next MoE shapes?

Tiny first (does the op lower?), then one SwitchLinear at decode size
H=2560, I=640, K=10. GPU before ANE — ANE abort is process death.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class OneSwitch(nn.Module):
    """One GatherMM: y = x @ W[eids]^T, W is (1, E, out, in)."""

    def __init__(self, n_in: int, n_out: int, n_experts: int, k: int):
        super().__init__()
        from coreai_torch.composite_ops import GatherMM

        self.k = k
        self.gather_mm = GatherMM(num_batch_axes=1)
        self.weight = nn.Parameter(
            torch.randn(1, n_experts, n_out, n_in, dtype=torch.float16)
        )

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # x: (B*S, 1, 1, in)   indices: (B*S, k) uint16
        wt = self.weight.transpose(-1, -2)  # (1, E, in, out)
        y = self.gather_mm(x, wt, rhs_indices=indices)
        return y  # (1, B*S, k, 1, out)


def _ref(x, w, ids):
    """x (1, in), w (E, out, in), ids (k,) -> (k, out)"""
    rows = w[ids]  # (k, out, in)
    return np.einsum("koi,i->ko", rows, x)


def eager_check(n_in: int, n_out: int, n_experts: int, k: int) -> None:
    torch.manual_seed(0)
    m = OneSwitch(n_in, n_out, n_experts, k).eval().half()
    x = torch.randn(1, 1, 1, n_in, dtype=torch.float16)
    ids = torch.randint(0, n_experts, (1, k), dtype=torch.int32).to(torch.uint16)
    with torch.no_grad():
        y = m(x, ids)
    got = y.detach().float().numpy().reshape(k, n_out)
    expect = _ref(
        x.numpy().reshape(n_in).astype(np.float32),
        m.weight.detach().float().numpy()[0],
        ids.cpu().numpy().reshape(k).astype(np.int32),
    )
    rel = float(np.linalg.norm(got - expect) / (np.linalg.norm(expect) + 1e-12))
    print(f"  eager rel={rel:.3e}  y{got.shape}")


def export_and_run(n_in: int, n_out: int, n_experts: int, k: int, units: tuple[str, ...]) -> None:
    from coreai_torch import TorchConverter, get_decomp_table, ExternalizeSpec
    from coreai_torch.composite_ops import GatherMM
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    torch.manual_seed(0)
    m = OneSwitch(n_in, n_out, n_experts, k).eval().half()
    x = torch.randn(1, 1, 1, n_in, dtype=torch.float16)
    ids = torch.randint(0, n_experts, (1, k), dtype=torch.int32)
    ids = ids.to(torch.uint16)
    example = (x, ids)
    t0 = time.perf_counter()
    ep = torch.export.export(m, args=example)
    ep = ep.run_decompositions(get_decomp_table())
    print(f"  torch.export {time.perf_counter()-t0:.2f}s")
    t0 = time.perf_counter()
    prog = (
        TorchConverter()
        .add_pytorch_module(m, export_fn=lambda module: torch.export.export(
            module, args=example).run_decompositions(get_decomp_table()),
            externalize_modules=[ExternalizeSpec(GatherMM, "gather_mm", ["num_batch_axes"])],
            input_names=["x", "ids"], output_names=["y"])
        .to_coreai()
    )
    prog.optimize()
    print(f"  convert+optimize {time.perf_counter()-t0:.2f}s")
    tmp = Path(tempfile.mkdtemp(prefix="gathermm_"))
    out = tmp / "gathermm.aimodel"
    prog.save_asset(out)
    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"  asset {mb:.1f} MB  {out}")

    with torch.no_grad():
        ref = m(x, ids.to(torch.uint16) if False else ids.to(torch.int32))
    ref_np = ref.detach().float().numpy()
    feeds = {
        "x": NDArray(x.numpy()),
        "ids": NDArray(ids.numpy()),
    }
    kinds = {str(knd): knd for knd in ComputeUnitKind.available_kinds()}
    print("  available", list(kinds))

    async def _one(label: str, key: str) -> None:
        kind = kinds.get(key)
        if kind is None:
            print(f"  [{label}] skip (no {key})")
            return
        spec = SpecializationOptions.from_preferred_compute_unit_kind(kind)
        t0 = time.perf_counter()
        try:
            mm = await AIModel.load(str(out), specialization_options=spec)
        except Exception as exc:
            print(f"  [{label}] LOAD FAILED: {exc}")
            return
        fn = mm.load_function("main")
        y = (await fn(feeds))["y"]
        got = np.asarray(y.numpy(), dtype=np.float32)
        rel = float(
            np.linalg.norm(got.reshape(-1) - ref_np.reshape(-1))
            / (np.linalg.norm(ref_np) + 1e-12)
        )
        for _ in range(3):
            await fn(feeds)
        t1 = time.perf_counter()
        n = 10
        for _ in range(n):
            await fn(feeds)
        ms = (time.perf_counter() - t1) / n * 1e3
        print(
            f"  [{label}-preferred; placement not verified] load {time.perf_counter()-t0:.2f}s  "
            f"rel={rel:.3e}  {ms:.2f} ms  y{got.shape}"
        )

    async def _all() -> None:
        mapping = {"cpu": "CPU", "gpu": "GPU", "ane": "Neural Engine"}
        for label in units:
            await _one(label, mapping[label])

    asyncio.run(_all())
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("== eager tiny ==")
    eager_check(64, 32, 8, 2)
    print("\n== export tiny E=8 in=64 out=32 k=2 ==")
    try:
        export_and_run(64, 32, 8, 2, units=("gpu", "cpu", "ane"))
    except Exception as exc:
        import traceback
        print(f"  TINY EXPORT FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    print("\n== eager Flash-Next gate slice E=512 in=2560 out=640 k=10 ==")
    try:
        eager_check(2560, 640, 512, 10)
    except Exception as exc:
        print(f"  eager full FAILED: {exc}")

    print("\n== export Flash-Next gate E=64 (subset) in=2560 out=640 k=10 ==")
    try:
        export_and_run(2560, 640, 64, 10, units=("gpu", "cpu"))
    except Exception as exc:
        import traceback
        print(f"  FN EXPORT FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
