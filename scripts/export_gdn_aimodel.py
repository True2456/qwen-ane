#!/usr/bin/env python3
"""export_gdn_aimodel.py - P15 step 1: export our GDN-style fused tail as .aimodel

Mirrors rindi_native_chain's tail graph (out_proj -> residual -> RMSNorm ->
gate/up -> SiLU*up -> down [+ next-input-layernorm projection]) as a PyTorch
module, converts via coreai-torch, optionally palettizes weights to INT4 via
coreai-opt, and saves a .aimodel artifact ready for AOT compilation:

    xcrun coreai-build compile gdn_tail.aimodel \
        --preferred-compute neural-engine --architecture h17

Run with the persistent venv after booting macOS 27:
    ~/.rindi/venvs/coreai/bin/python scripts/export_gdn_aimodel.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HIDDEN = 5120
CORE = 6144          # GDN out_proj input width (heads * v_dim)
INTERMEDIATE = 17408 # MLP intermediate (gate/up = 2x)
SEQ = 32             # prefill chunk width (29 live lanes padded)
OUT_DIR = Path.home() / ".rindi/aimodels"


class GdnTail(nn.Module):
    """Fused tail: out_proj + residual + RMSNorm + SwiGLU MLP (+ optional
    next-layer input projection), channel-major [C, S] like the engine."""

    def __init__(self, has_next: bool):
        super().__init__()
        self.has_next = has_next
        self.o_w = nn.Parameter(torch.randn(HIDDEN, CORE) * 0.02)
        self.post_norm_w = nn.Parameter(torch.ones(HIDDEN))
        self.gu_w = nn.Parameter(torch.randn(2 * INTERMEDIATE, HIDDEN) * 0.02)
        self.dn_w = nn.Parameter(torch.randn(HIDDEN, INTERMEDIATE) * 0.02)
        if has_next:
            self.il_w = nn.Parameter(torch.ones(HIDDEN))
            self.ip_w = nn.Parameter(torch.randn(2048, HIDDEN) * 0.02)

    def forward(self, xin):                      # xin: [core+hidden, seq]
        core, res = torch.split(xin, [CORE, HIDDEN], dim=0)   # [(core,S), (h,S)]
        h = res + self.o_w @ core                              # out_proj + resid
        sq = h * h
        ms = sq.mean(dim=0, keepdim=True)                      # [1, S]
        nx = h / torch.sqrt(ms + 1e-6)
        n = nx * self.post_norm_w.unsqueeze(1)
        c = self.gu_w @ n                                      # gate/up fused
        g0, u0 = torch.split(c, [INTERMEDIATE, INTERMEDIATE], dim=0)
        ac = (g0 * torch.sigmoid(g0)) * u0                     # SiLU(g)*u
        m = self.dn_w @ ac
        y = h + m
        if not self.has_next:
            return y
        nsd = torch.sqrt((y * y).mean(dim=0, keepdim=True) + 1e-6)
        nn_in = (y / nsd) * self.il_w.unsqueeze(1)
        y2 = self.ip_w @ nn_in                                 # next qkv-ish proj
        return y, y2


def main():
    from coreai_torch import TorchConverter, get_decomp_table

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    for has_next in (True, False):
        tag = "with_ip" if has_next else "final"
        model = GdnTail(has_next).eval()
        xin = torch.randn(CORE + HIDDEN, SEQ)

        exported = torch.export.export(model, args=(xin,))
        exported = exported.run_decompositions(get_decomp_table())

        out_names = ["y", "y2"] if has_next else ["y"]
        program = (
            TorchConverter()
            .add_exported_program(
                exported,
                input_names=["xin"],
                output_names=out_names,
            )
            .to_coreai()
        )
        program.optimize()

        out = OUT_DIR / f"gdn_tail_{tag}.aimodel"
        program.save_asset(out)
        size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
        print(f"saved {out} ({size_mb:.1f} MB, fp16)")

    print("\nnext: AOT-compile for ANE once coreai-build is available:")
    print("  xcrun coreai-build compile "
          f"{OUT_DIR}/gdn_tail_final.aimodel "
          "--preferred-compute neural-engine --architecture h17")


if __name__ == "__main__":
    sys.exit(main())
