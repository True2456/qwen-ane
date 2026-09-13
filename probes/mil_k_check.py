"""Diff the K-slot MIL GDN layer against MultiTokenStep over K real tokens.

The point of the unroll is speculation: one backbone pass has to produce a
hidden state per drafted token *and* the recurrent state at every prefix, so a
partially accepted block does not cost a second pass.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_flashnext_coreai import _load_layer, H, HC_W, HV, DV, DK, QKV, SEQ_DEFAULT
from flashnext_multitoken_step import MultiTokenStep
from runtime.mil_gdn_backend import MilGdnLayer
from runtime.q38_ane_engine import _iosurface_view

S = SEQ_DEFAULT


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    li = 0
    loader, w = _load_layer(li)
    ref = MultiTokenStep(w, k).eval().half()
    lay = MilGdnLayer(li, w, MultiTokenStep(w, 1).eval().half(), k=k, single_state=os.environ.get("MIL_K_SINGLE_STATE", "1") == "1")
    loader.close()

    rng = np.random.default_rng(12)
    x = np.zeros((HC_W, S), np.float16)
    x[:, :k] = (rng.standard_normal((HC_W, k)) * 0.05).astype(np.float16)
    conv = np.ascontiguousarray((rng.standard_normal((3 * QKV, S)) * 0.02).astype(np.float16))
    state = np.ascontiguousarray((rng.standard_normal((HV, DV, DK)) * 0.02).astype(np.float16))
    lay._conv[:] = conv[:, 0]
    lay.set_state(state)

    with torch.no_grad():
        r = ref(torch.from_numpy(x).reshape(1, HC_W, 1, S),
                torch.from_numpy(conv).reshape(1, 3 * QKV, 1, S),
                torch.from_numpy(state).reshape(1, HV, DV, DK))
    r_mixed = r[0].float().numpy().reshape(H, S)[:, :k]
    r_state = r[3].float().numpy().reshape(HV, DV, DK)
    r_conv = r[4].float().numpy().reshape(3 * QKV, S)[:, :1]

    g_mixed = lay(x.reshape(1, HC_W, 1, S), n=k)[0].reshape(H, k)
    g_state = lay.state_at(k - 1).astype(np.float32)
    lay.commit(k - 1)
    g_conv = lay._conv.astype(np.float32).reshape(-1, 1)

    def rel(a, b_):
        return float(np.linalg.norm(a - b_) / max(np.linalg.norm(b_), 1e-12))
    per = "  ".join(f"t{t}={rel(g_mixed[:, t], r_mixed[:, t]):.4f}" for t in range(k))
    print(f"  K={k}: mixed {per}   final state {rel(g_state, r_state):.5f}   "
          f"conv {rel(g_conv, r_conv):.5f}", flush=True)

    def invoke():
        # Reproduce the pre-optimization host staging in the timed region for
        # a controlled A/B without keeping that work in the production path.
        if os.environ.get("MIL_K_RESTAGE_INPUTS") == "1":
            with _iosurface_view(lay._prog._in_surfs[1], (3 * QKV, S),
                                 np.float16) as dst:
                dst[:, 0] = lay._conv
            for surf, value in zip(lay._prog._in_surfs[2:4],
                                   (lay._param, lay._hcn)):
                with _iosurface_view(surf, value.shape, np.float16) as dst:
                    np.copyto(dst, value)
        return lay(x.reshape(1, HC_W, 1, S), n=k)

    for _ in range(5):
        invoke()
    ts = []
    repeats = int(os.environ.get("MIL_K_REPEATS", "25"))
    for _ in range(repeats):
        t0 = time.perf_counter()
        invoke()
        ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts)) * 1e3
    print(f"  K={k}: {ms:.3f} ms/pass  =  {ms / k:.3f} ms/token", flush=True)

    if k > 4:
        loader, w = _load_layer(li)
        both = MilGdnLayer(li, w, MultiTokenStep(w, 1).eval().half(),
                           k=4, prefill_k=k, single_state=True)
        loader.close()
        both._conv[:] = conv[:, 0]
        both.set_state(state)
        both.select(1)
        wide_m = both(x.reshape(1, HC_W, 1, S), n=k)[0].reshape(H, k)
        both.commit(k - 1)
        wide_s = both.current_state().astype(np.float32)
        wide_c = both._conv.astype(np.float32)
        both.set_state(state)
        both._conv[:] = conv[:, 0]
        both._conv_surface_current = False
        both.select(0)
        parts = []
        for t in range(0, k, 4):
            xb = np.zeros((HC_W, S), np.float16)
            xb[:, :4] = x[:, t:t + 4]
            parts.append(both(xb.reshape(1, HC_W, 1, S), n=4)[0].reshape(H, 4))
            both.commit(3)
        nar_m = np.concatenate(parts, axis=1)
        nar_s = both.current_state().astype(np.float32)
        nar_c = both._conv.astype(np.float32)
        print(f"  K={k} vs 8xK=4: mixed {rel(wide_m, nar_m):.5f}  "
              f"state {rel(wide_s, nar_s):.5f}  conv {rel(wide_c, nar_c):.5f}",
              flush=True)
    if os.environ.get("MIL_K_SUBMIT_ONLY") == "1":
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            lay._eng.submit(lay._prog, procedure_index=0)
            ts.append(time.perf_counter() - t0)
        print(f"  K={k}: submit-only {float(np.median(ts)) * 1e3:.3f} ms",
              flush=True)


if __name__ == "__main__":
    main()
