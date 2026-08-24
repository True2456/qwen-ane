#!/usr/bin/env python3
"""ane_op_bisect.py - Find exactly which op breaks ANE compilation.

Starts from a known-working single matmul, adds ops one at a time,
reports which addition first triggers ANERegionFormationPass crash.
Once identified, we can restructure around it.
"""
import asyncio, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

H = 5120
C = 6144
I = 17408
S = 32

def try_compile_and_run(model, args, out_names, tag):
    """Try full pipeline: convert -> save -> load -> execute.
    Returns (ok, detail)."""
    from coreai_torch import TorchConverter, get_decomp_table
    from coreai.runtime import AIModel, NDArray, SpecializationOptions

    try:
        model = model.eval().half()
        exported = torch.export.export(model, args=args)
        exported = exported.run_decompositions(get_decomp_table())
        prog = (TorchConverter()
                .add_exported_program(exported,
                                      input_names=[f"i{n}" for n in range(len(args))],
                                      output_names=out_names)
                .to_coreai())
        prog.optimize()
        p = Path(f"/tmp/bisect_{tag}.aimodel")
        prog.save_asset(p)

        async def run():
            m = await AIModel.load(str(p))
            fn = m.load_function("main")
            feeds = {f"i{n}": NDArray(a.numpy()) for n, a in enumerate(args)}
            return await fn(feeds)
        out = asyncio.run(run())
        return True, out
    except Exception as e:
        return False, str(e)[:120]


def check(name, ok, detail):
    status = "OK" if ok else f"FAIL ({detail})"
    print(f"  {name:30s}: {status}", flush=True)
    return ok


def main():
    torch.manual_seed(42)
    core = torch.randn(C, S).half()
    res = torch.randn(H, S).half()
    o_w = torch.randn(H, C).half() * 0.02
    pn = torch.ones(H)
    gu_w = torch.randn(2*I, H).half() * 0.02
    dn_w = torch.randn(H, I).half() * 0.02

    # Stage 0: single dense
    class M0(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(o_w.clone())
        def forward(self, x): return self.w @ x
    m0 = M0()
    ok, d = try_compile_and_run(m0, (core,), ["y"], "s0")
    check("matmul alone", ok, d)
    if not ok: return

    # Stage 1: + residual add
    class M1(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(o_w.clone())
        def forward(self, x, r): return r + self.w @ x
    m1 = M1()
    ok, d = try_compile_and_run(m1, (core, res), ["y"], "s1")
    check("+ residual add", ok, d)
    if not ok: return

    # Stage 2: + mean over channel dim
    class M2(nn.Module):
        def forward(self, h):
            ms = (h * h).mean(dim=0, keepdim=True)
            return h / torch.sqrt(ms + 1e-6)
    m2 = M2()
    h_ref = res + o_w @ core
    ok, d = try_compile_and_run(m2, (h_ref,), ["y"], "s2")
    check("+ rmsnorm (mean/sqrt/div)", ok, d)
    if not ok: return

    # Stage 3: + sigmoid
    class S3(nn.Module):
        def forward(self, a): return torch.sigmoid(a)
    a_in = torch.randn(I, S).half()
    ok, d = try_compile_and_run(S3(), (a_in,), ["y"], "s3")
    check("+ sigmoid", ok, d)
    if not ok: return

    # Stage 4: full SwiGLU block (gate/up split + silu*up)
    class M4(nn.Module):
        def __init__(self):
            super().__init__()
            self.gu_w = nn.Parameter(gu_w.clone())
        def forward(self, n):
            c = self.gu_w @ n
            g0, u0 = torch.split(c, [I, I], dim=0)
            return (g0 * torch.sigmoid(g0)) * u0
    n_in = torch.randn(H, S).half()
    m4 = M4()
    ok, d = try_compile_and_run(m4, (n_in,), ["y"], "s4")
    check("+ SwiGLU (split+sigmoid+mul)", ok, d)
    if not ok: return

    # Stage 5: down projection after swiglu
    dn_w_t = torch.randn(H, I).half() * 0.02
    ac = torch.randn(I, S).half()
    class M5(nn.Module):
        def __init__(self):
            super().__init__()
            self.dn_w = nn.Parameter(dn_w.clone())
        def forward(self, ac): return self.dn_w @ ac
    m5 = M5()
    ok, d = try_compile_and_run(m5, (ac,), ["y"], "s5")
    check("+ down proj", ok, d)
    if not ok: return

    # Stage 6: FULL TAIL - everything combined
    class M6(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_w = nn.Parameter(o_w.clone())
            self.pn = nn.Parameter(pn.clone())
            self.gu_w = nn.Parameter(gu_w.clone())
            self.dn_w = nn.Parameter(dn_w.clone())
        def forward(self, core, res):
            h = res + self.o_w @ core
            ms = (h * h).mean(dim=0, keepdim=True)
            nx = h / torch.sqrt(ms + 1e-6)
            n = nx * self.pn.unsqueeze(1)
            c = self.gu_w @ n
            g0, u0 = torch.split(c, [I, I], dim=0)
            ac = (g0 * torch.sigmoid(g0)) * u0
            y = h + self.dn_w @ ac
            return y
    m6 = M6()
    ok, d = try_compile_and_run(m6, (core, res), ["y"], "s6")
    check("FULL TAIL", ok, d)
    if not ok: return

    # Stage 7: FULL TAIL + next-proj
    ip_w = torch.randn(2048, H).half() * 0.02
    il_w = torch.ones(H)
    class M7(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_w = nn.Parameter(o_w.clone())
            self.pn = nn.Parameter(pn.clone())
            self.gu_w = nn.Parameter(gu_w.clone())
            self.dn_w = nn.Parameter(dn_w.clone())
            self.il_w = nn.Parameter(il_w.clone())
            self.ip_w = nn.Parameter(ip_w.clone())
        def forward(self, core, res):
            h = res + self.o_w @ core
            ms = (h * h).mean(dim=0, keepdim=True)
            nx = h / torch.sqrt(ms + 1e-6)
            n = nx * self.pn.unsqueeze(1)
            c = self.gu_w @ n
            g0, u0 = torch.split(c, [I, I], dim=0)
            ac = (g0 * torch.sigmoid(g0)) * u0
            y = h + self.dn_w @ ac
            nsd = torch.sqrt((y*y).mean(0, keepdim=True) + 1e-6)
            y2 = self.ip_w @ ((y/nsd) * self.il_w.unsqueeze(1))
            return y, y2
    m7 = M7()
    ok, d = try_compile_and_run(m7, (core, res), ["y", "y2"], "s7")
    check("FULL TAIL + next-proj", ok, d)

if __name__ == "__main__":
    main()
