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
    """Token-major fused tail with real weights. Optionally folds the NEXT
    layer's input projection: y2 = W_next @ rmsnorm(y)*il_{L+1}, matching the
    engine's fused-tail ABI (outputs y [S,H] and y2 [S,P])."""
    def __init__(self, o_w, gu_w, dn_w, norm_w, ip_w=None, il_w=None):
        super().__init__()
        self.o_proj = nn.Linear(C, H, bias=False)
        self.o_proj.weight.data = o_w
        self.post_norm = nn.Parameter(norm_w.clone())
        self.gate_up = nn.Linear(H, 2 * I, bias=False)
        self.gate_up.weight.data = gu_w
        self.down = nn.Linear(I, H, bias=False)
        self.down.weight.data = dn_w
        self.has_ip = ip_w is not None
        if self.has_ip:
            self.next_norm = nn.Parameter(il_w.clone())
            self.next_proj = nn.Linear(H, ip_w.shape[0], bias=False)
            self.next_proj.weight.data = ip_w

    def forward(self, xin):                        # xin: [S, C+H]
        core, res = torch.split(xin, [C, H], dim=-1)
        h = res + self.o_proj(core)
        # pure-fp16 graph: any fp32 cast op splits ANE regions (24ms regression)
        n = h / torch.sqrt((h * h).mean(-1, keepdim=True) + 1e-6) * self.post_norm
        g0, u0 = self.gate_up(n).chunk(2, dim=-1)
        y = h + self.down(torch.nn.functional.silu(g0) * u0)
        if not self.has_ip:
            return y
        # folded next-layer input path: rmsnorm(y)*il_{L+1} then in-projection
        ny = y / torch.sqrt((y * y).mean(-1, keepdim=True) + 1e-6) * self.next_norm
        return y, self.next_proj(ny)


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

    # Folded next-layer input projection: GDN -> qkv+z+b+a rows; attn -> q+k+v.
    nxt = layer + 1
    ip_w = il_w = None
    if nxt < 64:
        gdn = f"model.language_model.layers.{nxt}.linear_attn.in_proj_qkv.weight" in wm
        parts = (["linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                  "linear_attn.in_proj_b", "linear_attn.in_proj_a"] if gdn else
                 ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])
        mats = [load_weight(nxt, p, wm) for p in parts]
        ip_w = torch.cat(mats, dim=0)
        il_key = f"model.language_model.layers.{nxt}.input_layernorm.weight"
        with safe_open(MODEL_DIR / wm[il_key], framework="pt") as f:
            il_w = f.get_tensor(il_key)
        print(f"[layer {layer}] folds L{nxt} {'GDN' if gdn else 'attn'} "
              f"in-projection P={ip_w.shape[0]}")

    model = RealTail(o_w.half(), torch.cat([gate, up]).half(), dn.half(),
                     norm.half(), None if ip_w is None else ip_w.half(),
                     None if il_w is None else il_w.half()).eval()
    xin = torch.randn(S, C + H).half()
    with torch.no_grad():
        ref_pair = model(xin)
    ref = ref_pair[0] if isinstance(ref_pair, tuple) else ref_pair

    gran = PerGroupedChannelGranularity(axis=0, group_size=GROUP)
    pcs = os.environ.get("GDN_PCS", "1") == "1"
    lin_cfg = ModuleKMeansPalettizerConfig(module_state_spec={
        "weight": PalettizationSpec(n_bits=NBITS, granularity=gran,
                                    enable_per_channel_scale=pcs)})
    cfg = KMeansPalettizerConfig(module_type_configs={nn.Linear: lin_cfg})
    if os.environ.get("GDN_O_FP16", "0") == "1":
        # keep attention output projection in fp16 (residual-critical)
        cfg.set_module_name("o_proj", None)
    ip_bits = int(os.environ.get("GDN_IP_BITS", "0"))
    if ip_bits:
        cfg.set_module_name("next_proj", ModuleKMeansPalettizerConfig(
            module_state_spec={"weight": PalettizationSpec(
                n_bits=ip_bits, granularity=gran,
                enable_per_channel_scale=pcs)}))

    t0 = time.perf_counter()
    eng = KMeansPalettizer(model, cfg)
    eng.prepare(example_inputs=(xin,))
    from coreai_opt.base_model_compressor import ExportBackend
    pal = eng.finalize(backend=ExportBackend.CoreAI)
    print(f"palettized+finalized in {time.perf_counter()-t0:.1f}s")
    with torch.no_grad():
        pal_out = pal(xin)
    py_y = pal_out[0] if isinstance(pal_out, tuple) else pal_out
    rf = ref.float()
    print(f"post-compress rel(y)={float((py_y.float()-rf).abs().max()/rf.abs().max()):.5f}")

    exp = torch.export.export(pal, args=(xin,))
    exp = exp.run_decompositions(get_decomp_table())
    out_names = ["y", "y2"] if model.has_ip else ["y"]
    prog = (TorchConverter()
            .add_exported_program(exp, input_names=["xin"], output_names=out_names)
            .to_coreai())
    prog.optimize()

    tag = f"L{layer}_ip{1 if model.has_ip else 0}_int{NBITS}_g{GROUP}_tm_s{S}"
    out = OUT_DIR / f"qwen38_27b_tail_{tag}.aimodel"
    if out.exists():
        import shutil; shutil.rmtree(out)
    prog.save_asset(out)
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"saved {out} ({mb:.1f} MB)")
    # goldens for C++ probe
    xin.contiguous().numpy().tofile(f"/tmp/golden_L{layer}_x.f16")
    ref.contiguous().numpy().tofile(f"/tmp/golden_L{layer}_y.f16")
    if model.has_ip:
        ref_pair[1].contiguous().numpy().tofile(f"/tmp/golden_L{layer}_y2.f16")

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
            if model.has_ip:
                g2 = np.asarray(r["y2"].numpy(), dtype=np.float32)
                r2 = ref_pair[1].float().numpy()
                e1 = max(e1, float(np.abs(g2 - r2).max() / np.abs(r2).max()))
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
