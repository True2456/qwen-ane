#!/usr/bin/env python3
"""P21: INT4-palettized GDN tail via coreai-opt (nn.Linear form).

Token-major layout [S, C+H] so palettizer sees standard Linear weights and the
ANE gets row-major activations. Pipeline: KMeansPalettizer(4-bit, group 32,
CoreAI backend) -> TorchConverter -> .aimodel -> verify vs fp32/fp16 reference
-> CPU/ANE benchmark.
"""
import asyncio, os, time
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCALE = float(os.environ.get("GDN_SCALE", "1.0"))
H = int(5120 * SCALE) // 64 * 64            # hidden
C = int(6144 * SCALE) // 64 * 64            # out_proj input width
I = int(17408 * SCALE) // 64 * 64           # MLP intermediate
S = int(os.environ.get("GDN_S", "32"))      # prefill chunk width
GROUP = int(os.environ.get("GDN_GROUP", "32"))
NBITS = int(os.environ.get("GDN_BITS", "4"))
OUT_DIR = Path.home() / ".rindi/aimodels"


class RMSNorm(nn.Module):
    """RMSNorm over last dim (token-major friendly)."""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.w


class GdnTail(nn.Module):
    """Fused tail, token-major: out_proj + residual + norm + SwiGLU + down +
    residual + norm + next-input projection."""
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(C, H, bias=False)
        self.post_norm = RMSNorm(H)
        self.gate_up = nn.Linear(H, 2 * I, bias=False)
        self.down = nn.Linear(I, H, bias=False)
        self.next_norm = RMSNorm(H)
        self.next_proj = nn.Linear(H, 2048, bias=False)

    def forward(self, xin):                       # xin: [S, C+H]
        core, res = torch.split(xin, [C, H], dim=-1)
        h = res + self.o_proj(core)               # [S, H]
        n = self.post_norm(h)
        g0, u0 = self.gate_up(n).chunk(2, dim=-1)
        m = self.down(torch.nn.functional.silu(g0) * u0)   # [S, H]
        y = h + m
        y2 = self.next_proj(self.next_norm(y))    # [S, 2048]
        return y, y2


def main():
    from coreai_torch import TorchConverter, get_decomp_table
    from coreai_opt.palettization import (
        KMeansPalettizer, KMeansPalettizerConfig, ModuleKMeansPalettizerConfig,
        PalettizationSpec)
    from coreai_opt.palettization.spec.granularity import (
        PerGroupedChannelGranularity)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    model = GdnTail().eval().half()
    xin = torch.randn(S, C + H).half()
    with torch.no_grad():
        ref_y, ref_y2 = model(xin)

    gran = PerGroupedChannelGranularity(axis=0, group_size=GROUP)
    lin_cfg = ModuleKMeansPalettizerConfig(module_state_spec={
        "weight": PalettizationSpec(n_bits=NBITS, granularity=gran)})
    cfg = KMeansPalettizerConfig(module_type_configs={nn.Linear: lin_cfg})

    t0 = time.perf_counter()
    eng = KMeansPalettizer(model, cfg)
    prepared = eng.prepare(example_inputs=(xin,))
    from coreai_opt.base_model_compressor import ExportBackend
    pal = eng.finalize(backend=ExportBackend.CoreAI)
    print(f"palettized+finalized in {time.perf_counter()-t0:.1f}s "
          f"(bits={NBITS}, group={GROUP})")
    with torch.no_grad():
        py_y, py_y2 = pal(xin)
    ryf, r2f = ref_y.float(), ref_y2.float()
    print(f"post-compress rel(y)={float((py_y.float()-ryf).abs().max()/ryf.abs().max()):.5f}")

    exp = torch.export.export(pal, args=(xin,))
    exp = exp.run_decompositions(get_decomp_table())
    prog = (TorchConverter()
            .add_exported_program(exp, input_names=["xin"], output_names=["y", "y2"])
            .to_coreai())
    prog.optimize()

    tag = f"int{NBITS}_g{GROUP}_tm_s{S}"
    out = OUT_DIR / f"gdn_tail_ip_{tag}.aimodel"
    prog.save_asset(out)
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"saved {out} ({mb:.1f} MB)")

    golden = {"x": xin, "y": ref_y, "y2": ref_y2}
    torch.save(golden, str(out) + ".golden.pt")

    async def bench():
        from coreai.runtime import AIModel, SpecializationOptions, ComputeUnitKind, NDArray
        for name, kind in (("cpu", ComputeUnitKind.cpu()),
                           ("ane", ComputeUnitKind.neural_engine())):
            so = SpecializationOptions.from_preferred_compute_unit_kind(kind)
            try:
                mm = await AIModel.load(str(out), specialization_options=so)
            except Exception as e:
                print(f"[{name}] LOAD FAILED: {str(e)[:200]}")
                continue
            fn = mm.load_function("main")
            r = await fn({"xin": NDArray(xin.numpy())})
            g1 = np.asarray(r["y"].numpy(), dtype=np.float32)
            g2 = np.asarray(r["y2"].numpy(), dtype=np.float32)
            ry = ref_y.float().numpy(); r2 = ref_y2.float().numpy()
            e1 = float(np.abs(g1 - ry).max() / np.abs(ry).max())
            e2 = float(np.abs(g2 - r2).max() / np.abs(r2).max())
            nd = NDArray(xin.numpy())
            await fn({"xin": nd})
            N = 100; t0 = time.perf_counter()
            for _ in range(N):
                await fn({"xin": nd})
            ms = (time.perf_counter() - t0) / N * 1e3
            flops = 2 * (C*H + H*2*I + I*H + H*2048) * S / 1e12
            print(f"[{name}] rel(y)={e1:.5f} rel(y2)={e2:.5f}  {ms:.2f} ms/chunk "
                  f"-> {flops/(ms/1e3):.1f} TFLOPS, {S/ms*1000:.0f} tok/s")

    asyncio.run(bench())


if __name__ == "__main__":
    main()
