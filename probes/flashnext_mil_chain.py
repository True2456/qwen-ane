"""Does int8 error compound across stacked MIL layers?

One layer measures rel 0.027 against the Core AI graph. The question that
decides whether the port is usable is what N layers do, since a decode step
runs 36 of them and the recurrent state carries between tokens. Chains real
consecutive GDN layers in MIL and compares against `MultiTokenStep` chained
identically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "probes"))

import flashnext_mil_layer as ML  # noqa: E402
from runtime.q38_ane_engine import _iosurface_view  # noqa: E402
from ane_w8a8_projection import eng  # noqa: E402
from export_flashnext_coreai import (  # noqa: E402
    _load_layer, H, HC_W, HV, DK, DV, QKV, SEQ_DEFAULT,
)
from flashnext_multitoken_step import MultiTokenStep  # noqa: E402

S = SEQ_DEFAULT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--tokens", type=int, default=3)
    a = ap.parse_args()

    progs, refs, keep = [], [], []
    for li in a.layers:
        ML.LAYER[0] = li
        loader, w = _load_layer(li)
        ref = MultiTokenStep(w, 1).eval().half()
        built = ML.build_layer(w, ref)
        if built is None:
            print(f"  layer {li}: compile failed")
            return
        progs.append(built)
        refs.append(ref)
        keep.append(loader)
    print(f"  built {len(progs)} MIL int8 layers: {a.layers}", flush=True)

    rng = np.random.default_rng(21)
    # per-layer recurrent state, MIL and torch sides kept separate
    m_state = [np.ascontiguousarray(np.zeros((HV, DV, DK), np.float16)) for _ in progs]
    m_conv = [np.ascontiguousarray(np.zeros((3 * QKV, S), np.float16)) for _ in progs]
    t_state = [torch.zeros(1, HV, DV, DK).half() for _ in progs]
    t_conv = [torch.zeros(1, 3 * QKV, 1, S).half() for _ in progs]

    def rel(x, y):
        return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-12))

    for tok in range(a.tokens):
        x = np.zeros((HC_W, S), np.float16)
        x[:, :1] = (rng.standard_normal((HC_W, 1)) * 0.05).astype(np.float16)
        mx = np.ascontiguousarray(x)
        tx = torch.from_numpy(x).reshape(1, HC_W, 1, S)
        per_layer = []
        for i, (prog, param, hcn) in enumerate(progs):
            for surf, val in zip(prog._in_surfs, (mx, m_conv[i], param, hcn, m_state[i])):
                with _iosurface_view(surf, val.shape, np.float16) as d:
                    np.copyto(d, val)
            if not eng.submit(prog, procedure_index=0):
                print(f"  token {tok} layer {i}: submit failed")
                return
            with _iosurface_view(prog._out_surfs[2], (HC_W, S), np.float16) as o:
                hyper = np.array(o, np.float16)
            with _iosurface_view(prog._out_surfs[4], (3 * QKV, S), np.float16) as o:
                m_conv[i] = np.ascontiguousarray(np.array(o, np.float16))
            with _iosurface_view(prog._out_surfs[5], (HV, DV, DK), np.float16) as o:
                m_state[i] = np.ascontiguousarray(np.array(o, np.float16))
            # the residual stream handed to the next layer is `hyper`
            mx = np.ascontiguousarray(hyper)

            with torch.no_grad():
                r = refs[i](tx, t_conv[i], t_state[i])
            t_state[i], t_conv[i] = r[3], r[4]
            # torch reference recombines the same way inside the layer; its
            # stream for the next layer is the hyper output
            tx = r[1]
            per_layer.append(rel(np.asarray(mx, np.float32)[:, :1],
                                 tx.float().numpy().reshape(HC_W, S)[:, :1]))
        print(f"  token {tok}: cumulative rel after each layer  "
              + "  ".join(f"L{a.layers[j]} {v:.4f}" for j, v in enumerate(per_layer)),
              flush=True)
    for ld in keep:
        ld.close()


if __name__ == "__main__":
    main()
