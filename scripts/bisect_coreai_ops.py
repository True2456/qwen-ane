#!/usr/bin/env python3
"""bisect_coreai_ops.py - find which op stage diverges in CoreAI vs PyTorch.

Builds incrementally growing prefixes of the GDN tail graph, converts each,
runs on CPU, compares against PyTorch fp32. First divergent stage = culprit.
"""
import asyncio
import os
import torch
from torch import nn

SCALE = float(os.environ.get("GDN_SCALE", "0.125"))
H = max(int(5120 * SCALE) // 64 * 64, 64)
C = max(int(6144 * SCALE) // 64 * 64, 64)
I = max(int(17408 * SCALE) // 64 * 64, 64)
S = 32


def cmp(name, model, args, out_names):
    import coreai_torch
    from coreai.runtime import AIModel, NDArray, SpecializationOptions
    model = model.eval().half()
    args = tuple(a.half() for a in args)
    exported = torch.export.export(model, args=args)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    prog = (coreai_torch.TorchConverter()
            .add_exported_program(exported,
                                  input_names=[f"i{n}" for n in range(len(args))],
                                  output_names=out_names)
            .to_coreai())
    prog.optimize()
    from pathlib import Path
    path = Path(f"/tmp/bisect_{name}.aimodel")
    prog.save_asset(path)

    async def run():
        m = await AIModel.load(path, specialization_options=SpecializationOptions.cpu_only())
        fn = m.load_function("main")
        feeds = {f"i{n}": NDArray(a.numpy().astype(np.float16))
                 for n, a in enumerate(args)}
        outs = await fn(feeds)
        return [outs[n].numpy().astype(np.float32) for n in out_names]

    got = asyncio.run(run())

    with torch.no_grad():
        out_ref = model(*[a.half() for a in args])
    ref = out_ref if isinstance(out_ref, (list, tuple)) else [out_ref]
    ref = [t.float() for t in ref]
    worst_name, worst = "-", 0.0
    for n, (g, r) in enumerate(zip(got, ref)):
        rn = r.numpy()
        d = np.abs(g - rn).max()
        rel = d / (np.abs(rn).max() + 1e-9)
        if rel > worst:
            worst, worst_name = rel, out_names[n]
        try:
            corr = np.corrcoef(g.flatten(), rn.flatten())[0, 1] if g.size > 1 else 1.0
        except Exception:
            corr = float("nan")
            print(f"    SHAPE MISMATCH got={g.shape} ref={rn.shape}")
        print(f"  {name:<22} {out_names[n]:>3}: rel={rel:.4f} corr={corr:.5f}")
    status = "OK" if worst < 5e-2 else "DIVERGES"
    print(f"[{status}] {name} (worst {worst_name} rel={worst:.4f})\n")
    return worst < 5e-2


import numpy as np
from pathlib import Path


def main():
    torch.manual_seed(7)
    x = torch.randn(C, S)
    h_in = torch.randn(H, S)
    o_w = torch.randn(H, C) * 0.02
    pn = torch.ones(H)
    gu_w = torch.randn(2 * I, H) * 0.02
    dn_w = torch.randn(H, I) * 0.02

    # Stage 1: pure matmul
    class S1(nn.Module):
        def forward(self, core, w): return w @ core
    cmp("s1_matmul", S1(), (x, o_w), ["y"])

    # Stage 2: + residual + RMSNorm(mean over C)
    class S2(nn.Module):
        def forward(self, core, res, w, pn):
            h = res + w @ core
            ms = (h * h).mean(dim=0, keepdim=True)
            return (h / torch.sqrt(ms + 1e-6)) * pn.unsqueeze(1)
    cmp("s2_rmsnorm", S2(), (x, h_in, o_w, pn), ["y"])

    # Stage 3: gate/up split + SiLU
    class S3(nn.Module):
        def forward(self, n, gu_w):
            c = gu_w @ n
            g0, u0 = torch.split(c, [c.shape[0] // 2, c.shape[0] // 2], dim=0)
            return (g0 * torch.sigmoid(g0)) * u0
    n_in = torch.randn(H, S)
    cmp("s3_swiglu", S3(), (n_in, gu_w), ["y"])

    # Stage 4: down proj after swiglu
    class S4(nn.Module):
        def forward(self, ac, dn_w): return dn_w @ ac
    ac_in = torch.randn(I, S)
    cmp("s4_downproj", S4(), (ac_in, dn_w), ["y"])

    # Stage 5: sigmoid alone
    class S5(nn.Module):
        def forward(self, a): return torch.sigmoid(a)
    cmp("s5_sigmoid", S5(), (n_in,), ["y"])

    # Stage 6: TWO outputs - does name->value mapping survive?
    class S6(nn.Module):
        def __init__(self):
            super().__init__()
            super().__init__()
            self.a_w = nn.Parameter(torch.randn(64, H) * 0.02)
            self.b_w = nn.Parameter(torch.randn(128, H) * 0.02)
        def forward(self, n):
            ya = self.a_w @ n
            yb = self.b_w @ n
            return ya, yb
    m6 = S6()
    cmp("s6_two_out", m6.half(), (n_in,), ["ya", "yb"])

    # Stage 7: full tail WITH next proj (the actual artifact graph)
    class S7(nn.Module):
        def __init__(self):
            super().__init__()
            super().__init__()
            self.o_w = nn.Parameter(torch.randn(H, C) * 0.02)
            self.pn = nn.Parameter(torch.ones(H))
            self.gu_w = nn.Parameter(torch.randn(2 * I, H) * 0.02)
            self.dn_w = nn.Parameter(torch.randn(H, I) * 0.02)
            self.il_w = nn.Parameter(torch.ones(H))
            self.ip_w = nn.Parameter(torch.randn(int(2048 * SCALE), H) * 0.02)
        def forward(self, xin):
            core, res = torch.split(xin, [C, H], dim=0)
            h = res + self.o_w @ core
            ms = (h * h).mean(dim=0, keepdim=True)
            n = (h / torch.sqrt(ms + 1e-6)) * self.pn.unsqueeze(1)
            c = self.gu_w @ n
            g0, u0 = torch.split(c, [I, I], dim=0)
            ac = (g0 * torch.sigmoid(g0)) * u0
            m = self.dn_w @ ac
            y = h + m
            nsd = torch.sqrt((y * y).mean(dim=0, keepdim=True) + 1e-6)
            nn_in = (y / nsd) * self.il_w.unsqueeze(1)
            y2 = self.ip_w @ nn_in
            return y, y2
    xin = torch.randn(C + H, S)
    cmp("s7_full_tail", S7().half(), (xin,), ["y", "y2"])

    # Stage 8: SAME graph but through the save_asset -> disk -> AIModel.load
    # roundtrip (mirrors export_gdn_aimodel.py + run_gdn_aimodel.py).
    import coreai_torch
    from coreai.runtime import SpecializationOptions
    m8 = S7().half()
    exp8 = torch.export.export(m8, args=(xin,)).run_decompositions(
        coreai_torch.get_decomp_table())
    prog8 = (coreai_torch.TorchConverter()
             .add_exported_program(exp8, input_names=["xin"], output_names=["y", "y2"])
             .to_coreai())
    prog8.optimize()
    p8 = Path(f"/tmp/bisect_s8_roundtrip.aimodel")
    prog8.save_asset(p8)

    async def run8():
        from coreai.runtime import AIModel, NDArray
        mm = await AIModel.load(str(p8), specialization_options=SpecializationOptions.cpu_only())
        fn = mm.load_function("main")
        outs = await fn({"xin": NDArray(xin.float().numpy())})
        return [outs["y"].numpy().astype(np.float32),
                outs["y2"].numpy().astype(np.float32)]

    got8 = asyncio.run(run8())
    with torch.no_grad():
        r8 = [t.float() for t in m8(xin.half())]
    for nm, g, r in zip(("y", "y2"), got8, r8):
        rn = r.numpy()
        rel = np.abs(g - rn).max() / (np.abs(rn).max() + 1e-9)
        corr = np.corrcoef(g.flatten(), rn.flatten())[0, 1]
        print(f"  s8_roundtrip {nm}: rel={rel:.4f} corr={corr:.5f}")
    worst8 = max(np.abs(g - r).max() / (np.abs(r).max() + 1e-9) for g, r in zip(got8, r8))
    print(f"[{'OK' if worst8 < 5e-2 else 'DIVERGES'}] s8_roundtrip (worst rel={worst8:.4f})")


if __name__ == "__main__":
    main()
