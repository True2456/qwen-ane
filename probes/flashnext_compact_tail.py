"""ANE GDN tail with six compact parameter rows instead of padded tokens.

Separate artifact; does not replace production graphs. Checks nonzero state
and compares against the existing Torch GDN tail before exporting.
"""
import asyncio
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_flashnext_coreai import (
    FlashNextGdnOnly, _load_layer0, _load_layer, _export, _bench, HV, DK, DV, GDN_Y,
)


class CompactTail(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.out_proj = original.out_proj
        self.norm_w = original.norm_w
        self.register_buffer("eps", torch.tensor(1e-6, dtype=torch.float16))

    def forward(self, params, state):
        q, k, v = params[:, :, 0:1, :], params[:, :, 1:2, :], params[:, :, 2:3, :]
        decay, beta, z = params[:, :, 3:4, :], params[:, :, 4:5, :], params[:, :, 5:6, :]
        state = state * decay
        memory = (state * k).sum(-1, keepdim=True)
        delta = (v.transpose(-1, -2) - memory) * beta.transpose(-1, -2)
        state = state + delta * k
        y = (state * q).sum(-1, keepdim=True).transpose(-1, -2)
        y = y * torch.rsqrt((y*y).mean(-1, keepdim=True) + self.eps)
        y = y * self.norm_w * torch.sigmoid(z)
        # Repeat the useful output across aligned columns; no zero/scatter.
        y = y.reshape(1, GDN_Y, 1, 1).expand(1, GDN_Y, 1, 32)
        return self.out_proj(y), state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-all", action="store_true")
    args = parser.parse_args()
    loader, w = _load_layer0()
    old = FlashNextGdnOnly().eval().half()
    old.load_from_layer(w)
    m = CompactTail(old).eval().half()
    torch.manual_seed(915)
    params = torch.randn(1, HV, 6, DK).half() * .03
    params[:, :, 3, :] = .98
    params[:, :, 4, :] = .4
    state = torch.randn(1, HV, DV, DK).half() * .02
    q, k, v = [params[:, :, i:i+1, :].expand(1, HV, 32, DK).contiguous() for i in range(3)]
    decay = params[:, :, 3:4, 0:1].expand(1, HV, 1, 32).contiguous()
    beta = params[:, :, 4:5, 0:1].expand(1, HV, 1, 32).contiguous()
    z = params[:, :, 5:6, :].reshape(1, GDN_Y, 1, 1).expand(1, GDN_Y, 1, 32).contiguous()
    with torch.no_grad():
        expected, ss = old(q, k, v, decay, beta, state, z)
        out, ns = m(params, state)
    torch.testing.assert_close(ns, ss, rtol=.002, atol=.0001)
    torch.testing.assert_close(out[..., :1], expected[..., :1], rtol=.002, atol=.0001)
    print("PASS compact vs old tail; params bytes:", params.numel()*2, flush=True)
    path = _export(m, (params, state), (["params", "state"], ["attn", "new_ssm"]), "compact_tail_L0")
    asyncio.run(_bench(path, {"params": params.numpy(), "state": state.numpy()},
                       {"attn": out.numpy(), "new_ssm": ns.numpy()}, 1, units=("ane",)))
    async def recurrent_check():
        from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
        ane = next(k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine")
        model = await AIModel.load(str(path), specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(ane))
        fn = model.load_function("main")
        sref = state.clone()
        snd = NDArray(state.numpy())
        worst = 0.0
        for step in range(16):
            p = params.clone()
            p[:, :, :3, :] *= 1 + step / 16
            with torch.no_grad():
                yref, sref = m(p, sref)
            result = await fn({"params": NDArray(p.numpy()), "state": snd})
            snd = result["new_ssm"]
            yy, ss = result["attn"].numpy(), snd.numpy()
            for actual, expected in ((yy, yref.numpy()), (ss, sref.numpy())):
                rel = float(np.linalg.norm(actual.astype(np.float32)-expected)/np.linalg.norm(expected.astype(np.float32)))
                worst = max(worst, rel)
                assert np.isfinite(rel) and rel < .02, (step, rel)
        print(f"PASS: 16 dependent ANE steps, worst relative L2={worst:.6g}", flush=True)
    asyncio.run(recurrent_check())
    if args.export_all:
        for layer in range(48):
            if layer % 4 == 3:
                continue
            ll, ww = _load_layer(layer)
            original = FlashNextGdnOnly().eval().half()
            original.load_from_layer(ww)
            compact = CompactTail(original).eval().half()
            _export(compact, (params, state), (["params", "state"], ["attn", "new_ssm"]), f"compact_tail_L{layer}")
            ll.close()


if __name__ == "__main__":
    main()
