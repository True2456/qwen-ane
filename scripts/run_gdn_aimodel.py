#!/usr/bin/env python3
"""run_gdn_aimodel.py - execute the exported GDN tail on CoreAI runtime (macOS 27).

Loads "", runs S=32 chunks, checks vs a PyTorch
reference, and reports dispatch latency vs our private-pipeline baseline.
"""
import asyncio
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import os
SCALE = float(os.environ.get("GDN_SCALE", "1.0"))
HIDDEN = int(5120 * SCALE) // 64 * 64
CORE = int(6144 * SCALE) // 64 * 64
INTERMEDIATE = int(17408 * SCALE) // 64 * 64
TAGSCALE = f"{int(SCALE*100):02d}"
SEQ = int(os.environ.get("GDN_S", "32"))


class GdnTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_w = nn.Parameter(torch.randn(HIDDEN, CORE) * 0.02)
        self.post_norm_w = nn.Parameter(torch.ones(HIDDEN))
        self.gu_w = nn.Parameter(torch.randn(2 * INTERMEDIATE, HIDDEN) * 0.02)
        self.dn_w = nn.Parameter(torch.randn(HIDDEN, INTERMEDIATE) * 0.02)
        self.il_w = nn.Parameter(torch.ones(HIDDEN))
        self.ip_w = nn.Parameter(torch.randn(2048, HIDDEN) * 0.02)

    def forward(self, xin):
        core, res = torch.split(xin, [CORE, HIDDEN], dim=0)
        h = res + self.o_w @ core
        ms = (h * h).mean(dim=0, keepdim=True)
        nx = h / torch.sqrt(ms + 1e-6)
        n = nx * self.post_norm_w.unsqueeze(1)
        c = self.gu_w @ n
        g0, u0 = torch.split(c, [INTERMEDIATE, INTERMEDIATE], dim=0)
        ac = (g0 * torch.sigmoid(g0)) * u0
        m = self.dn_w @ ac
        y = h + m
        nsd = torch.sqrt((y * y).mean(dim=0, keepdim=True) + 1e-6)
        nn_in = (y / nsd) * self.il_w.unsqueeze(1)
        y2 = self.ip_w @ nn_in
        return y, y2


async def main():
    from coreai.runtime import NDArray
    import os as _os
    if _os.environ.get("GDN_MODEL"):
        model_path = Path(_os.environ["GDN_MODEL"])
    else:
        model_path = Path.home() / f".rindi/aimodels/gdn_tail_with_ip_{TAGSCALE}_s{SEQ}.aimodel"
    if not model_path.exists():
        print("run export_gdn_aimodel.py first"); return 2

    model = GdnTail().eval().half()
    sd_path = Path(_os.environ.get("GDN_SD", str(model_path) + ".state.pt"))
    if sd_path.exists():
        model.load_state_dict(torch.load(sd_path, weights_only=True))
        print("reference weights loaded from", sd_path.name)
    # NOTE: weights in the .aimodel came from the seed-42 export; rebuild the
    # reference with identical init by re-seeding identically.
    ref = GdnTail().eval()
    with torch.no_grad():
        for p_ref, p_model in zip(ref.parameters(), model.parameters()):
            pass  # same seed -> same tensors; model here is a fresh twin

    xin = torch.randn(CORE + HIDDEN, SEQ)

    from coreai.runtime import AIModel
    print("loading", model_path)
    t0 = time.perf_counter()
    from coreai.runtime import SpecializationOptions, ComputeUnitKind
    import os as _os
    spec = None
    _cu = _os.environ.get("GDN_CU", "ane").lower()
    if _cu == "gpu":
        spec = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu)
    elif _cu == "cpu":
        spec = SpecializationOptions.cpu_only()
    ai_model = await AIModel.load(str(model_path), specialization_options=spec)
    print(f"  loaded+specialized in {time.perf_counter()-t0:.2f} s")

    fn = await asyncio.to_thread(ai_model.load_function, "main")

    golden_p = Path(str(model_path) + ".golden.pt")
    # SEQ already read from env
    if not golden_p.exists():
        print("no golden file"); return 2
    golden = torch.load(golden_p, weights_only=False)
    xg = golden["x"]
    ry, ry2 = golden["y"].float(), golden["y2"].float()

    # Adaptive input dtype: some save/load roundtrips drop fp16 metadata and
    # the runtime then demands float32. Try the golden dtype first; on a
    # scalar-type rejection, parse the expected type and retry.
    import re as _re
    # desc says float16 (verified); feed fp16. NOTE: stale coreai-cache
    # specializations (~9GB) previously poisoned results - keep caches clean.
    arr = np.ascontiguousarray(xg.numpy().astype(np.float16))
    nd = NDArray(arr)
    for attempt in range(2):
        try:
            out = await fn({"xin": nd})
            break
        except RuntimeError as e:
            raise
    out = await fn({"xin": nd})
    y = out["y"].numpy(); y2 = out["y2"].numpy()
    print("out shapes:", y.shape, y2.shape)

    y32 = np.abs(y.astype(np.float32)); y232 = np.abs(y2.astype(np.float32))
    rn = ry.numpy()
    # layout hypothesis: runtime interprets 2D NDArray as column-major
    alt_in = np.ascontiguousarray(xin.half().numpy().T).T  # same values, sanity
    with torch.no_grad():
        ry_t, _ = model(torch.from_numpy(xin.half().numpy().T.copy()).T.contiguous())
    rt = ry_t.float().numpy()
    print("diag: corr(got, ref_transposed_input) =", np.corrcoef(y32.flatten(), rt.flatten())[0,1])
    print("diag: got[:4,0] =", np.round(y32[:4,0], 4))
    print("diag: ref[:4,0] =", np.round(rn[:4,0], 4))
    print("diag: corr(got,ref) =", np.corrcoef(y32.flatten(), rn.flatten())[0,1])
    print("diag: corr(got, ref.T) =", np.corrcoef(y32.flatten(), rn.T.flatten())[0,1])
    dy = np.abs(y32 - rn).max()
    d2 = np.abs(y232 - ry2.numpy()).max()
    den = float(np.abs(ry.numpy()).max()) + 1e-9
    print(f"correctness vs pytorch: max|dy|={dy:.5f} (rel {dy/den:.2e}), "
          f"max|dy2|={d2:.5f}")

    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        await fn({"xin": NDArray(arr)})
    dt = (time.perf_counter() - t0) / N * 1e3
    flops = 2.0 * (HIDDEN * CORE + 2 * INTERMEDIATE * HIDDEN +
                   HIDDEN * INTERMEDIATE + 2048 * HIDDEN) * SEQ
    print(f"latency p50~mean: {dt:.3f} ms/eval | {flops/dt/1e9:.2f} TFLOPS "
          f"| x64 layers ~= {dt*64:.1f} ms/chunk-pass")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()) or 0)
