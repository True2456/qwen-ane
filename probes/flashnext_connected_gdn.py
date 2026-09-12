"""Connected ANE front/preparation/compact GDN test, with real L0 weights.

SiLU in-graph is the tanh identity x/2 + (x/2)*tanh(x/2), not F.silu / exp.
Never replaces production assets until recurrent numpy PASS. Requires
numerical and device validation.
"""
import asyncio
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from export_flashnext_coreai import (
    FlashNextFront, FlashNextGdnOnly, _load_layer0, _load_layer, _export, _bench,
    HostPrep, H, HV, HK, DK, DV, QKV, GDN_Y, OUT_DIR, _pack_to_conv,
)
from flashnext_compact_tail import CompactTail
from tools.flashnext_reference import GDNState, linear_attention_layer, zero_gdn_state


class Connected(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.front = FlashNextFront().eval().half()
        self.front.load_from_layer(w)
        old = FlashNextGdnOnly().eval().half()
        old.load_from_layer(w)
        self.tail = CompactTail(old)
        self.register_buffer("gamma", torch.tensor(np.exp(w["linear_attn.A_log"])).half().reshape(1, HV, 1, 1))
        self.register_buffer("dt", torch.tensor(w["linear_attn.dt_bias"]).half().reshape(1, HV, 1, 1))
        self.register_buffer("eps", torch.tensor(1e-6).half())
        self.register_buffer("qscale", torch.tensor(DK ** -0.5).half())
        self.register_buffer("half_coeff", torch.tensor(0.5).half())

    def forward(self, h, conv, state):
        yin, newconv = self.front(h, conv)
        one = yin[..., :1]
        packed = one[:, :QKV].reshape(1, QKV // DK, 1, DK)
        halfx = packed * self.half_coeff
        # tanh-SiLU: x/2 + (x/2)*tanh(x/2). Do not use F.silu / exp / softplus.
        c = halfx + halfx * torch.tanh(halfx)
        q = c[:, :HK]
        k = c[:, HK:2 * HK]
        v = c[:, 2 * HK:]
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + self.eps) * self.qscale
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + self.eps)
        q = q.repeat_interleave(HV // HK, dim=1)
        k = k.repeat_interleave(HV // HK, dim=1)
        z = one[:, QKV:QKV + GDN_Y].reshape(1, HV, 1, DK)
        b = one[:, QKV + GDN_Y:QKV + GDN_Y + HV]
        a = one[:, QKV + GDN_Y + HV:]
        decay = torch.pow(torch.sigmoid(-(a + self.dt)), self.gamma).expand(1, HV, 1, DK)
        beta = torch.sigmoid(b).expand(1, HV, 1, DK)
        params = torch.cat((q, k, v, decay, beta, z), dim=2)
        attn, state = self.tail(params, state)
        return attn, state, newconv, params, yin, c


class Production(torch.nn.Module):
    def __init__(self, connected):
        super().__init__()
        self.block = connected

    def forward(self, h, conv, state):
        attn, state, conv, *_ = self.block(h, conv, state)
        return attn, state, conv


def _l2(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def _hidden_bsh(h_bc1s) -> np.ndarray:
    arr = h_bc1s.float().numpy() if hasattr(h_bc1s, "float") else np.asarray(h_bc1s, np.float32)
    return arr[:, :, 0, 0].reshape(1, 1, H)


def _np_state(conv_t, ssm_t) -> GDNState:
    conv_np = conv_t.numpy() if hasattr(conv_t, "numpy") else conv_t
    ssm_np = ssm_t.numpy() if hasattr(ssm_t, "numpy") else ssm_t
    return GDNState(
        conv=_pack_to_conv(conv_np).astype(np.float32),
        ssm=np.asarray(ssm_np, np.float32),
    )


def _prod_path(layer: int) -> Path:
    return OUT_DIR / f"flashnext_connected_gdn_prod_L{layer}.aimodel"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-all", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse existing prod L0; skip debug graph")
    args = ap.parse_args()
    loader, w = _load_layer0()
    m = Connected(w).eval().half()
    torch.manual_seed(918)
    h = torch.zeros(1, H, 1, 32).half()
    h[..., 0] = torch.randn(1, H, 1).half() * 0.1
    conv = torch.randn(1, 3 * QKV, 1, 32).half() * 0.02
    state = torch.randn(1, HV, DV, DK).half() * 0.02
    prep = HostPrep(w)
    with torch.no_grad():
        yin, _ = m.front(h, conv)
        prep(yin.numpy())
        params = np.stack((
            prep.q[:, :, 0], prep.k[:, :, 0], prep.v[:, :, 0],
            np.broadcast_to(prep.decay[:, :, 0, :1], (1, HV, DK)),
            np.broadcast_to(prep.beta[:, :, 0, :1], (1, HV, DK)),
            prep.z[..., 0].reshape(1, HV, DK),
        ), axis=2)
        ref, ss = m.tail(torch.from_numpy(params), state)
        out, ns, nc, pp, yy, cc = m(h, conv, state)
    rel = _l2(out.float().numpy(), ref.float().numpy())
    print(f"connected Torch vs compact-tail HostPrep: relative L2={rel:.6g}", flush=True)
    assert rel < 0.02, rel

    debug_path = OUT_DIR / "flashnext_connected_gdn_L0.aimodel"
    path = _prod_path(0)
    if not args.reuse or not debug_path.is_dir():
        debug_path = _export(
            m, (h, conv, state),
            (["h", "conv", "state"], ["attn", "new_ssm", "new_conv", "params", "yin", "activated"]),
            "connected_gdn_L0",
        )
        # GPU before ANE: new graphs can abort the process (MPS Code -19).
        asyncio.run(_bench(
            debug_path,
            {"h": h.numpy(), "conv": conv.numpy(), "state": state.numpy()},
            {"attn": out.numpy(), "new_ssm": ns.numpy(), "new_conv": nc.numpy(),
             "params": pp.numpy(), "yin": yy.numpy(), "activated": cc.numpy()},
            1, units=("gpu", "ane"),
        ))

        async def inspect():
            from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
            ane = next(k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine")
            mm = await AIModel.load(
                str(debug_path),
                specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(ane),
            )
            fn = mm.load_function("main")
            result = await fn({"h": NDArray(h.numpy()), "conv": NDArray(conv.numpy()), "state": NDArray(state.numpy())})
            actual = result["params"].numpy().astype(np.float32)
            expected = pp.float().numpy()
            raw = result["yin"].numpy().astype(np.float32)
            act = result["activated"].numpy().astype(np.float32)
            cref = cc.float().numpy()
            print(
                f"activation range ref=[{cref.min():.6g},{cref.max():.6g}] "
                f"zeros={np.mean(act == 0):.3f} "
                f"ref_subnormal={np.mean((np.abs(cref) < np.finfo(np.float16).tiny) & (cref != 0)):.3f}",
                flush=True,
            )
            golden = yy.float().numpy()
            a, b = raw[:, :QKV, ..., :1], golden[:, :QKV, ..., :1]
            print(f"conv_pre: relL2={_l2(a, b):.6g}", flush=True)
            for i, name in enumerate(("q", "k", "v", "decay", "beta", "z")):
                aa, bb = actual[:, :, i], expected[:, :, i]
                print(f"parameter {name}: relL2={_l2(aa, bb):.6g} max_abs={np.abs(aa - bb).max():.6g}", flush=True)
        asyncio.run(inspect())

    production = Production(m).eval().half()
    with torch.no_grad():
        y_prod, ns_prod, nc_prod = production(h, conv, state)
    if not args.reuse or not path.is_dir():
        path = _export(
            production, (h, conv, state),
            (["h", "conv", "state"], ["attn", "new_ssm", "new_conv"]),
            "connected_gdn_prod_L0",
        )
        asyncio.run(_bench(
            path,
            {"h": h.numpy(), "conv": conv.numpy(), "state": state.numpy()},
            {"attn": y_prod.numpy(), "new_ssm": ns_prod.numpy(), "new_conv": nc_prod.numpy()},
            1, units=("gpu",),
        ))
    else:
        print(f"reuse {path}", flush=True)

    async def check_recurrent():
        from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
        ane = next(k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine")
        mm = await AIModel.load(
            str(path),
            specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(ane),
        )
        fn = mm.load_function("main")
        cfg = loader.text_config
        # Zero conv/state + tiny h (scale 0.02) makes attn rms ~7e-4; relative
        # L2 then reports ~0.48 from ~2e-3 fp16 abs noise — false alarm.
        # Generate-scale h (rms ~0.045) + 16-step from a live pack is the real test.
        h1 = torch.zeros_like(h)
        torch.manual_seed(1)
        h1[..., 0] = torch.randn(1, H, 1).half()
        zc, zs = torch.zeros_like(conv), torch.zeros_like(state)
        with torch.no_grad():
            y1, _, _ = production(h1, zc, zs)
        g0 = await fn({"h": NDArray(h1.numpy()), "conv": NDArray(zc.numpy()), "state": NDArray(zs.numpy())})
        a0 = g0["attn"].numpy().astype(np.float32)
        t0 = y1.float().numpy()
        rel0 = _l2(a0, t0)
        rms0 = float(np.sqrt(np.mean(t0 * t0)))
        assert np.isfinite(rel0) and rel0 < 0.03, ("generate-start torch", rel0, rms0)
        print(f"PASS generate-start vs Torch: attn relL2={rel0:.6g} rms={rms0:.6g}", flush=True)

        np0, st0 = linear_attention_layer(
            w, _hidden_bsh(h1), state=zero_gdn_state(cfg, 1), use_mlx_l2_eps=False,
        )
        rel0n = _l2(a0[:, :, 0, 0], np0[0, 0])
        rel0s = _l2(g0["new_ssm"].numpy(), st0.ssm)
        print(
            f"  generate-start vs numpy linear_attention: attn relL2={rel0n:.6g} ssm relL2={rel0s:.6g}",
            flush=True,
        )
        assert np.isfinite(rel0n) and rel0n < 0.02, ("generate-start numpy attn", rel0n)
        assert np.isfinite(rel0s) and rel0s < 0.02, ("generate-start numpy ssm", rel0s)
        print("PASS generate-start vs numpy linear_attention", flush=True)

        sr, cr = state.clone(), conv.clone()
        sn, cn = NDArray(sr.numpy()), NDArray(cr.numpy())
        np_state = _np_state(conv, state)
        worst_t, worst_n, worst_n_ssm, worst_n_conv = 0.0, 0.0, 0.0, 0.0
        for step in range(16):
            hh = torch.zeros_like(h)
            hh[..., 0] = torch.randn(1, H, 1).half() * 0.1
            with torch.no_grad():
                yr, sr, cr = production(hh, cr, sr)
            got = await fn({"h": NDArray(hh.numpy()), "conv": cn, "state": sn})
            sn, cn = got["new_ssm"], got["new_conv"]
            for key, ref in (("attn", yr), ("new_ssm", sr), ("new_conv", cr)):
                actual = got[key].numpy().astype(np.float32)
                target = ref.float().numpy()
                rel = _l2(actual, target)
                worst_t = max(worst_t, rel)
                assert np.isfinite(rel) and rel < 0.03, (step, "torch", key, rel)
            np_out, np_state = linear_attention_layer(
                w, _hidden_bsh(hh), state=np_state, use_mlx_l2_eps=False,
            )
            attn_n = _l2(got["attn"].numpy()[:, :, 0, 0], np_out[0, 0])
            ssm_n = _l2(got["new_ssm"].numpy(), np_state.ssm)
            conv_n = _l2(_pack_to_conv(got["new_conv"].numpy()), np_state.conv)
            worst_n = max(worst_n, attn_n)
            worst_n_ssm = max(worst_n_ssm, ssm_n)
            worst_n_conv = max(worst_n_conv, conv_n)
            assert np.isfinite(attn_n) and attn_n < 0.02, (step, "numpy attn", attn_n)
            assert np.isfinite(ssm_n) and ssm_n < 0.02, (step, "numpy ssm", ssm_n)
            assert np.isfinite(conv_n) and conv_n < 0.02, (step, "numpy conv", conv_n)
        print(
            f"PASS production connected: 16 dependent ANE steps vs Torch worst relL2={worst_t:.6g}",
            flush=True,
        )
        print(
            f"PASS 16-step vs numpy linear_attention: "
            f"attn worst={worst_n:.6g} ssm worst={worst_n_ssm:.6g} conv worst={worst_n_conv:.6g}",
            flush=True,
        )

    asyncio.run(check_recurrent())
    if args.export_all:
        for layer in range(48):
            if layer % 4 == 3:
                continue
            dest = _prod_path(layer)
            if args.reuse and dest.is_dir():
                print(f"  L{layer:02d} connected reuse", flush=True)
                continue
            ll, ww = _load_layer(layer)
            model = Production(Connected(ww)).eval().half()
            _export(
                model, (h, conv, state),
                (["h", "conv", "state"], ["attn", "new_ssm", "new_conv"]),
                f"connected_gdn_prod_L{layer}",
            )
            ll.close()


if __name__ == "__main__":
    main()
