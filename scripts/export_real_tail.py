#!/usr/bin/env python3
"""P21: REAL-weights INT4 GDN tail export from Qwen3.8-27B bf16 safetensors.

Extracts layer-L o_proj/gate_proj/up_proj/down_proj, builds the token-major
fused tail (o_proj + residual + RMSNorm + SwiGLU + down), palettizes 4-bit
group-32 via coreai-opt, exports .aimodel, verifies vs fp16 reference,
benchmarks CPU vs ANE. Golden vectors dumped raw for the C++ probe.

Run: ~/.rindi/venvs/coreai/bin/python scripts/export_real_tail.py [layer]
"""
import asyncio, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors import safe_open

MODEL_DIR = Path.home() / ".lmstudio/models/Qwen/Qwen3.8-27B"
OUT_DIR = Path.home() / ".rindi/aimodels"
H, C, I = 5120, 6144, 17408
S = int(os.environ.get("GDN_S", "32"))
GROUP = int(os.environ.get("GDN_GROUP", "32"))
NBITS = int(os.environ.get("GDN_BITS", "4"))


def load_weight(layer: int, name: str, weight_map) -> torch.Tensor:
    key = f"model.language_model.layers.{layer}.{name}.weight"
    path = MODEL_DIR / weight_map[key]
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key)


class RealTail(nn.Module):
    """Token-major fused tail with real weights."""
    def __init__(self, o_w, gu_w, dn_w, norm_w):
        super().__init__()
        self.o_proj = nn.Linear(C, H, bias=False)
        self.o_proj.weight.data = o_w
        self.post_norm = nn.Parameter(norm_w.clone())
        self.gate_up = nn.Linear(H, 2 * I, bias=False)
        self.gate_up.weight.data = gu_w
        self.down = nn.Linear(I, H, bias=False)
        self.down.weight.data = dn_w

    def forward(self, xin):                        # xin: [S, C+H]
        core, res = torch.split(xin, [C, H], dim=-1)
        h = res + self.o_proj(core)
        # pure-fp16 graph: any fp32 cast op splits ANE regions (24ms regression)
        n = h / torch.sqrt((h * h).mean(-1, keepdim=True) + 1e-6) * self.post_norm
        g0, u0 = self.gate_up(n).chunk(2, dim=-1)
        y = h + self.down(torch.nn.functional.silu(g0) * u0)
        return y


def main():
    from coreai_torch import TorchConverter, get_decomp_table
    from coreai_opt.palettization import (
        KMeansPalettizer, KMeansPalettizerConfig, ModuleKMeansPalettizerConfig,
        PalettizationSpec)
    from coreai_opt.palettization.spec.granularity import (
        PerGroupedChannelGranularity)

    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    t0 = time.perf_counter()
    import json
    wm = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]

    # GDN layers project out via linear_attn.out_proj (same [5120,6144] shape);
    # full-attention layers use self_attn.o_proj.
    o_key = ("linear_attn.out_proj"
             if f"model.language_model.layers.{layer}.linear_attn.out_proj.weight" in wm
             else "self_attn.o_proj")
    o_w = load_weight(layer, o_key, wm)
    gate = load_weight(layer, "mlp.gate_proj", wm)
    up = load_weight(layer, "mlp.up_proj", wm)
    dn = load_weight(layer, "mlp.down_proj", wm)
    norm = load_weight(layer, "post_attention_layernorm", wm)
    print(f"[layer {layer}] out_proj via {o_key}: {o_w.shape}")
    print(f"loaded real weights in {time.perf_counter()-t0:.1f}s: "
          f"o{o_w.shape} gu{gate.shape}+{up.shape} dn{dn.shape}")

    model = RealTail(o_w.half(), torch.cat([gate, up]).half(), dn.half(),
                     norm.half()).eval()
    xin = torch.randn(S, C + H).half()
    with torch.no_grad():
        ref = model(xin)

    gran = PerGroupedChannelGranularity(axis=0, group_size=GROUP)
    pcs = os.environ.get("GDN_PCS", "1") == "1"
    lin_cfg = ModuleKMeansPalettizerConfig(module_state_spec={
        "weight": PalettizationSpec(n_bits=NBITS, granularity=gran,
                                    enable_per_channel_scale=pcs)})
    cfg = KMeansPalettizerConfig(module_type_configs={nn.Linear: lin_cfg})
    if os.environ.get("GDN_O_FP16", "0") == "1":
        # keep attention output projection in fp16 (residual-critical)
        cfg.set_module_name("o_proj", None)

    t0 = time.perf_counter()
    eng = KMeansPalettizer(model, cfg)
    eng.prepare(example_inputs=(xin,))
    from coreai_opt.base_model_compressor import ExportBackend
    pal = eng.finalize(backend=ExportBackend.CoreAI)
    print(f"palettized+finalized in {time.perf_counter()-t0:.1f}s")
    with torch.no_grad():
        py_y = pal(xin)
    rf = ref.float()
    print(f"post-compress rel(y)={float((py_y.float()-rf).abs().max()/rf.abs().max()):.5f}")

    exp = torch.export.export(pal, args=(xin,))
    exp = exp.run_decompositions(get_decomp_table())
    prog = (TorchConverter()
            .add_exported_program(exp, input_names=["xin"], output_names=["y"])
            .to_coreai())
    prog.optimize()

    tag = f"L{layer}_int{NBITS}_g{GROUP}_tm_s{S}"
    out = OUT_DIR / f"qwen38_27b_tail_{tag}.aimodel"
    if out.exists():
        import shutil; shutil.rmtree(out)
    prog.save_asset(out)
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"saved {out} ({mb:.1f} MB)")
    # golden for C++ probe (single output)
    xin.contiguous().numpy().tofile("/tmp/golden_rtm_x.f16")
    ref.numpy().tofile("/tmp/golden_rtm_y.f16")

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
            ry = ref.float().numpy()
            e1 = float(np.abs(g1 - ry).max() / np.abs(ry).max())
            nd = NDArray(xin.numpy())
            await fn({"xin": nd})
            N = 200; t0 = time.perf_counter()
            for _ in range(N):
                await fn({"xin": nd})
            ms = (time.perf_counter() - t0) / N * 1e3
            flops = 2 * (C*H + H*2*I + I*H) * S / 1e12
            print(f"[{name}] rel(y)={e1:.5f}  {ms:.2f} ms/chunk "
                  f"-> {flops/(ms/1e3):.1f} TFLOPS, {S/ms*1000:.0f} tok/s")

    asyncio.run(bench())


if __name__ == "__main__":
    main()
