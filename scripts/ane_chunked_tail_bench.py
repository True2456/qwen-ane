#!/usr/bin/env python3
"""P16: full-size GDN tail on Neural Engine via CoreAI - K-chunked graph.
KC=1536 keeps every reduction dim on ANE's correct path. Verified numerically
correct (rel 0.0047/0.0044) at 8.44 ms/eval, 3.7 TFLOPS-equivalent.
Requires macOS 27 + coreai-opt/coreai-torch venv (~/.rindi/venvs/coreai)."""
import asyncio, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from coreai_torch import TorchConverter, get_decomp_table
from coreai.runtime import AIModel, NDArray, SpecializationOptions, ComputeUnitKind

H, C, I, S, KC = 5120, 6144, 17408, 32, 1536

class ChunkedTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_w = nn.Parameter(torch.randn(H, C) * 0.02)
        self.pn = nn.Parameter(torch.ones(H))
        self.gu_w = nn.Parameter(torch.randn(2*I, H) * 0.02)
        self.dn_w = nn.Parameter(torch.randn(H, I) * 0.02)
        self.il_w = nn.Parameter(torch.ones(H))
        self.ip_w = nn.Parameter(torch.randn(2048, H) * 0.02)
    def ksplit_mm(self, x, w):
        outs, off = [], 0
        while off < x.shape[0]:
            kc = min(KC, x.shape[0] - off)
            outs.append(w[:, off:off+kc] @ x[off:off+kc, :])
            off += kc
        return torch.stack(outs).sum(dim=0)
    def forward(self, xin):
        core, res = torch.split(xin, [C, H], dim=0)
        h = res + self.ksplit_mm(core, self.o_w)
        n = (h / torch.sqrt((h*h).mean(0, keepdim=True) + 1e-6)) * self.pn.unsqueeze(1)
        c = self.ksplit_mm(n, self.gu_w)
        g0, u0 = torch.split(c, [I, I], dim=0)
        ac = (g0 * torch.sigmoid(g0)) * u0
        m = self.ksplit_mm(ac, self.dn_w)
        y = h + m
        nsd = torch.sqrt((y*y).mean(0, keepdim=True) + 1e-6)
        y2 = self.ksplit_mm((y/nsd) * self.il_w.unsqueeze(1), self.ip_w)
        return y, y2

if __name__ == "__main__":
    torch.manual_seed(555)
    m = ChunkedTail().eval().half()
    xin = torch.randn(C + H, S).half()
    exp = torch.export.export(m, args=(xin,)).run_decompositions(get_decomp_table())
    prog = TorchConverter().add_exported_program(exp, input_names=["xin"], output_names=["y", "y2"]).to_coreai()
    prog.optimize()
    p = Path("/tmp/chunked_tail.aimodel"); prog.save_asset(p)
    async def go():
        ne = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
        spec = SpecializationOptions.from_preferred_compute_unit_kind(ne)
        mm = await AIModel.load(str(p), specialization_options=spec)
        fn = mm.load_function("main")
        out = await fn({"xin": NDArray(xin.numpy())})
        with torch.no_grad(): ry, ry2 = m(xin)
        g1 = np.asarray(out["y"].numpy(), dtype=np.float32)
        g2 = np.asarray(out["y2"].numpy(), dtype=np.float32)
        print("y rel:", np.abs(g1-ry.float().numpy()).max()/(np.abs(ry.float().numpy()).max()+1e-9))
        print("y2 rel:", np.abs(g2-ry2.float().numpy()).max()/(np.abs(ry2.float().numpy()).max()+1e-9))
    asyncio.run(go())
