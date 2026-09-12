"""MLX-shaped Flash-Next layer step on Core AI: mixers + connected GDN + recombine.

This is DecoderLayer._pure_step minus routed MoE. Attn mix, GDN, recombine, and
MLP mix live in one compiled graph so generate does not host-mix around GDN.
Routed experts stay off this graph (changing top-10 cannot be baked; weight
inputs page ~100 MB/token). Shared expert is not in this graph either — HostMoE
owns the full SparseMoeBlock.

SiLU in the mixer is F.silu (small 320-d op). GDN keeps tanh-SiLU. Do not put
F.silu on the GDN qkv path.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT))

from export_flashnext_coreai import (  # noqa: E402
    Conv1x1,
    H,
    HC,
    HC_W,
    HV,
    DK,
    DV,
    QKV,
    OUT_DIR,
    _bsh_to_bc1s,
    _export,
    _bench,
    _load_layer,
    _load_layer0,
    _fp16,
    host_gated_residual,
    host_recombine,
)
from flashnext_connected_gdn import Connected, Production, _l2  # noqa: E402
from tools.flashnext_reference import (  # noqa: E402
    linear_attention_layer,
    zero_gdn_state,
)

MIX_H = 320


class FlashNextGatedMix(nn.Module):
    """4-branch hyper mix matching host_fastpath.gated_residual / mlx GatedResidual.

    hc_norm is stored as (w−1); we load (w+1). Grouped RMS over each 2560-channel
    branch. inj is (1, 4, 1, S) — last dim is S=32, not 4.
    """

    def __init__(self):
        super().__init__()
        self.eps = nn.Buffer(torch.tensor(1e-6, dtype=torch.float16), persistent=False)
        self.inv_hc = nn.Buffer(torch.tensor(0.25, dtype=torch.float16), persistent=False)
        self.two = nn.Buffer(torch.tensor(2.0, dtype=torch.float16), persistent=False)
        self.half_coeff = nn.Buffer(torch.tensor(0.5, dtype=torch.float16), persistent=False)
        self.hc_n = nn.Parameter(torch.ones(1, HC_W, 1, 1, dtype=torch.float16))
        self.down = Conv1x1(HC_W, MIX_H)
        self.up = Conv1x1(MIX_H, HC_W)
        self.inj = Conv1x1(HC_W, HC)

    def load(self, w, prefix: str) -> None:
        hc = np.asarray(w[f"{prefix}.hc_norm.weight"], np.float32) + 1.0
        self.hc_n.data.copy_(_fp16(hc.reshape(1, HC_W, 1, 1)))
        self.down.set_w(w[f"{prefix}.input_mix_weight_down.weight"])
        self.up.set_w(w[f"{prefix}.input_mix_weight_up.weight"])
        self.inj.set_w(w[f"{prefix}.block_inject_weight.weight"])

    def _grouped_rms(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for i in range(HC):
            sl = x[:, i * H : (i + 1) * H]
            ms = (sl * sl).mean(dim=1, keepdim=True)
            parts.append(sl * torch.rsqrt(ms + self.eps))
        return torch.cat(parts, dim=1) * self.hc_n

    def forward(self, x: torch.Tensor):
        n = self._grouped_rms(x)
        down = self.down(n) * self.inv_hc
        half = down * self.half_coeff
        # tanh-SiLU: F.silu in a large GDN graph previously poisoned numerics.
        gate = half + half * torch.tanh(half)
        mw = torch.sigmoid(self.up(gate))
        acc = (
            mw[:, 0:H] * n[:, 0:H]
            + mw[:, H : 2 * H] * n[:, H : 2 * H]
            + mw[:, 2 * H : 3 * H] * n[:, 2 * H : 3 * H]
            + mw[:, 3 * H : 4 * H] * n[:, 3 * H : 4 * H]
        )
        mixed = acc * self.inv_hc
        inj = self.two * torch.sigmoid(self.inj(n) * self.inv_hc)
        return mixed, x, inj


def _recombine(out_h: torch.Tensor, hyper: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
    parts = [out_h * inj[:, i : i + 1] for i in range(HC)]
    return hyper + torch.cat(parts, dim=1)


class FlashNextPureStep(nn.Module):
    """Attn mix + connected GDN + recombine + MLP mix. MoE stays on the host."""

    def __init__(self, w):
        super().__init__()
        self.attn = FlashNextGatedMix()
        self.attn.load(w, "attn_hyper_connection")
        self.gdn = Production(Connected(w)).eval().half()
        self.mlp = FlashNextGatedMix()
        self.mlp.load(w, "mlp_hyper_connection")

    def forward(self, x, conv, state):
        mixed, hyper, inj = self.attn(x)
        attn, state, conv = self.gdn(mixed, conv, state)
        h = _recombine(attn, hyper, inj)
        mixed2, hyper2, inj2 = self.mlp(h)
        return mixed2, hyper2, inj2, state, conv


def _prod_path(layer: int) -> Path:
    return OUT_DIR / f"flashnext_pure_step_L{layer}.aimodel"


def _mixer_path() -> Path:
    return OUT_DIR / "flashnext_gated_mix_L0.aimodel"


def _hidden_bsh_from_hc(x_bc1s) -> np.ndarray:
    arr = x_bc1s.float().numpy() if hasattr(x_bc1s, "float") else np.asarray(x_bc1s, np.float32)
    return arr[:, :, 0, :].transpose(0, 2, 1)[:, :1, :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-all", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--mixer-only", action="store_true")
    ap.add_argument("--layers", default="", help="comma-separated GDN indices to (re)export")
    args = ap.parse_args()

    extra = [int(x) for x in args.layers.split(",") if x.strip()]
    if extra:
        torch.manual_seed(918)
        x = torch.zeros(1, HC_W, 1, 32).half()
        x[..., 0] = torch.randn(1, HC_W, 1).half() * 0.05
        conv = torch.zeros(1, 3 * QKV, 1, 32).half()
        state = torch.zeros(1, HV, DV, DK).half()
        for i in extra:
            ld, lw = _load_layer(i)
            m = FlashNextPureStep(lw).eval().half()
            _export(
                m, (x, conv, state),
                (["x", "conv", "state"], ["mixed", "hyper", "inj", "new_ssm", "new_conv"]),
                f"pure_step_L{i}",
            )
            ld.close()
            print(f"exported L{i}", flush=True)
        return

    loader, w = _load_layer0()
    mix = FlashNextGatedMix().eval().half()
    mix.load(w, "attn_hyper_connection")
    torch.manual_seed(918)
    x = torch.zeros(1, HC_W, 1, 32).half()
    x[..., 0] = torch.randn(1, HC_W, 1).half() * 0.05
    with torch.no_grad():
        mixed_t, hyper_t, inj_t = mix(x)
    mixed_h, hyper_h, inj_h = host_gated_residual(
        x.float().numpy(), w, prefix="attn_hyper_connection"
    )
    r_m = _l2(mixed_t.float().numpy()[..., :1], mixed_h[..., :1])
    r_i = _l2(inj_t.float().numpy()[..., :1], inj_h[..., :1])
    print(f"Torch mixer vs host: mixed relL2={r_m:.6g} inj relL2={r_i:.6g}", flush=True)
    assert r_m < 0.05 and r_i < 0.02, (r_m, r_i)
    print("PASS Torch mixer vs host_gated_residual", flush=True)

    mp = _mixer_path()
    if not args.reuse or not mp.is_dir():
        mp = _export(
            mix, (x,),
            (["x"], ["mixed", "hyper", "inj"]),
            "gated_mix_L0",
        )
    else:
        print(f"reuse {mp}", flush=True)
    asyncio.run(_bench(
        mp,
        {"x": x.numpy()},
        {"mixed": mixed_t.numpy(), "hyper": hyper_t.numpy(), "inj": inj_t.numpy()},
        1, units=("gpu", "ane"),
    ))
    if args.mixer_only:
        loader.close()
        return

    step = FlashNextPureStep(w).eval().half()
    conv = torch.zeros(1, 3 * QKV, 1, 32).half()
    state = torch.zeros(1, HV, DV, DK).half()
    with torch.no_grad():
        mixed2, hyper2, inj2, ns, nc = step(x, conv, state)
    print(
        f"Torch pure_step mixed2={tuple(mixed2.shape)} hyper2={tuple(hyper2.shape)} "
        f"inj2={tuple(inj2.shape)}",
        flush=True,
    )

    mixed_a, hyper_a, inj_a = host_gated_residual(
        x.float().numpy(), w, prefix="attn_hyper_connection"
    )
    cfg = loader.text_config
    mixed_s1, hyper_s1, inj_s1 = mixed_a[..., :1], hyper_a[..., :1], inj_a[..., :1]
    np_attn, st = linear_attention_layer(
        w, mixed_s1[:, :, 0, :].transpose(0, 2, 1),
        state=zero_gdn_state(cfg, 1), use_mlx_l2_eps=True,
    )
    with torch.no_grad():
        attn_t, _, _ = step.gdn(
            torch.from_numpy(np.asarray(mixed_a, np.float16)), conv, state
        )
    r_attn_np = _l2(attn_t.float().numpy()[..., :1], _bsh_to_bc1s(np_attn))
    print(f"  connected GDN vs numpy attn relL2={r_attn_np:.6g} (tanh-SiLU, expected ~0.005)", flush=True)
    hc = host_recombine(attn_t.float().numpy()[..., :1], hyper_s1, inj_s1)
    mixed_m, hyper_m, inj_m = host_gated_residual(hc, w, prefix="mlp_hyper_connection")
    r_mix = _l2(mixed2.float().numpy()[..., :1], mixed_m[..., :1])
    r_inj = _l2(inj2.float().numpy()[..., :1], inj_m[..., :1])
    print(
        f"Torch pure_step vs host mix + Torch GDN + host mlp: "
        f"mixed relL2={r_mix:.6g} inj relL2={r_inj:.6g}",
        flush=True,
    )
    assert r_mix < 0.03 and r_inj < 0.03, (r_mix, r_inj)
    print("PASS Torch pure_step vs split host mixers + connected GDN", flush=True)

    path = _prod_path(0)
    if not args.reuse or not path.is_dir():
        path = _export(
            step, (x, conv, state),
            (["x", "conv", "state"], ["mixed", "hyper", "inj", "new_ssm", "new_conv"]),
            "pure_step_L0",
        )
    else:
        print(f"reuse {path}", flush=True)
    asyncio.run(_bench(
        path,
        {"x": x.numpy(), "conv": conv.numpy(), "state": state.numpy()},
        {
            "mixed": mixed2.numpy(),
            "hyper": hyper2.numpy(),
            "inj": inj2.numpy(),
            "new_ssm": ns.numpy(),
            "new_conv": nc.numpy(),
        },
        1, units=("gpu", "ane"),
    ))

    async def check_ane():
        from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
        ane = next(k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine")
        mm = await AIModel.load(
            str(path),
            specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(ane),
        )
        fn = mm.load_function("main")
        got = await fn({"x": NDArray(x.numpy()), "conv": NDArray(conv.numpy()),
                        "state": NDArray(state.numpy())})
        rel = _l2(got["mixed"].numpy()[..., :1], mixed_m[..., :1])
        print(f"ANE pure_step vs numpy mixed relL2={rel:.6g}", flush=True)
        assert np.isfinite(rel) and rel < 0.04, rel
        print("PASS ANE pure_step vs numpy", flush=True)

    asyncio.run(check_ane())

    if args.export_all:
        types = loader.text_config["layer_types"]
        n = len(types)
        loader.close()
        for i in range(n):
            if types[i] != "linear_attention":
                continue
            dest = _prod_path(i)
            if args.reuse and dest.is_dir():
                print(f"reuse {dest.name}", flush=True)
                continue
            ld, lw = _load_layer(i)
            m = FlashNextPureStep(lw).eval().half()
            _export(
                m, (x, conv, state),
                (["x", "conv", "state"], ["mixed", "hyper", "inj", "new_ssm", "new_conv"]),
                f"pure_step_L{i}",
            )
            ld.close()
            print(f"exported L{i}", flush=True)
        return
    loader.close()


if __name__ == "__main__":
    main()
