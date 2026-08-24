import asyncio, sys, numpy as np, torch
from pathlib import Path
from torch import nn
from coreai_torch import TorchConverter, get_decomp_table
from coreai.runtime import AIModel, NDArray, SpecializationOptions

H, C, I, S = 640, 768, 2176, 32

class T(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_w = nn.Parameter(torch.randn(H, C) * 0.02)
        self.pn = nn.Parameter(torch.ones(H))
        self.gu_w = nn.Parameter(torch.randn(2 * I, H) * 0.02)
        self.dn_w = nn.Parameter(torch.randn(H, I) * 0.02)
    def forward(self, xin):
        core, res = torch.split(xin, [C, H], dim=0)
        h = res + self.o_w @ core
        n = (h / torch.sqrt((h*h).mean(0, keepdim=True) + 1e-6)) * self.pn.unsqueeze(1)
        c = self.gu_w @ n
        g0, u0 = torch.split(c, [I, I], dim=0)
        y = h + self.dn_w @ ((g0 * torch.sigmoid(g0)) * u0)
        return y

mode = sys.argv[1]
p = Path("/tmp/xproc.aimodel")
if mode == "make":
    torch.manual_seed(99)
    m = T().eval().half()
    x = torch.randn(C + H, S).half()
    exp = torch.export.export(m, args=(x,)).run_decompositions(get_decomp_table())
    prog = TorchConverter().add_exported_program(exp, input_names=["x"], output_names=["y"]).to_coreai()
    prog.optimize()
    prog.save_asset(p)
    torch.save({"x": x, "state": m.state_dict(), "y": m(x)}, "/tmp/xproc_ref.pt")
    print("made + same-process run:")
    from coreai.runtime import AIModel, NDArray
    async def go():
        mm = await AIModel.load(str(p), specialization_options=SpecializationOptions.cpu_only())
        fn = mm.load_function("main")
        out = await fn({"x": NDArray(x.numpy())})
        return np.asarray(out["y"].numpy(), dtype=np.float32)
    got = asyncio.run(go())
    ref = torch.load("/tmp/xproc_ref.pt", weights_only=False)["y"].detach().float().numpy()
    print("  same-proc rel =", float(np.abs(got-ref).max()/ (np.abs(ref).max()+1e-9)))
else:
    import numpy as _np
    d = torch.load("/tmp/xproc_ref.pt", weights_only=False)
    x = d["x"]; ref = d["y"].detach().float().numpy()
    from coreai.runtime import AIModel, NDArray
    async def go():
        mm = await AIModel.load(str(p), specialization_options=SpecializationOptions.cpu_only())
        fn = mm.load_function("main")
        out = await fn({"x": NDArray(x.numpy())})
        return np.asarray(out["y"].numpy(), dtype=np.float32)
    got = asyncio.run(go())
    print("  fresh-proc rel =", float(np.abs(got-ref).max()/ (np.abs(ref).max()+1e-9)))
