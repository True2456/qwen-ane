"""Diff the K-slot MIL QSA layer against a numpy reference.

The probe that used to stand in for this (`flashnext_mil_qsa_layer.py` run as
main) never fed `h_mixed` / `i_inj` after the front-program split, so it
reported relative errors near 1.0 at every width, including K=1 where the
shipped decode path is known good. This one drives `MilQsaLayer` and a numpy
reference from the same weights, key cache, rotary table and mask.

Establish the oracle at K=1 (error band ~0.03, same as the GDN layer) before
trusting a number at 32.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_flashnext_coreai import (  # noqa: E402
    FlashNextQSADecode,
    H,
    HC,
    HC_W,
    QSA_HD,
    QSA_HKV,
    QSA_HQ,
    QSA_MASK,
    QSA_ROTARY,
    SEQ_DEFAULT,
    _load_layer,
    host_gated_residual,
    host_recombine,
)
from flashnext_mil_qsa_layer import _Ref  # noqa: E402
from runtime.expert_bank import MLX4_DEFAULT, Mlx4ExpertBank, MlxSafe  # noqa: E402
from runtime.host_fastpath import sigmoid, silu  # noqa: E402
from runtime.mil_qsa_backend import MilQsaLayer  # noqa: E402

S = SEQ_DEFAULT
G = QSA_HQ // QSA_HKV
HALF = QSA_ROTARY // 2
KVC = QSA_HKV * QSA_HD
EPS = 1e-6
SCALE = QSA_HD ** -0.5


def rel(a, b) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def _rms_last(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(ms + np.float32(EPS))) * weight


def _rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """NeoX RoPE on the last axis. x is (n, heads, HD); cos/sin (n, HALF)."""
    a = x[..., :HALF]
    b = x[..., HALF:QSA_ROTARY]
    rest = x[..., QSA_ROTARY:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    rot = np.concatenate([a * c - b * s, b * c + a * s], axis=-1)
    return np.concatenate([rot, rest], axis=-1)


def numpy_qsa(mixed, kc, vc, cos, sin, mask, n: int, m: int, w) -> tuple:
    """QSA core. mixed (H, n); kc/vc (KVC, m); cos/sin (HALF, S); mask (n, m+S).

    Matches `FlashNextQSADecode` and the MIL concat-of-tiled-new-keys layout,
    so a dead slot past n is a repeat of the live block and stays masked.
    """
    h = np.asarray(mixed, np.float32).reshape(H, -1)[:, :n].T
    wq = np.asarray(w["self_attn.q_proj.weight"], np.float32)
    wk = np.asarray(w["self_attn.k_proj.weight"], np.float32)
    wv = np.asarray(w["self_attn.v_proj.weight"], np.float32)
    wo = np.asarray(w["self_attn.o_proj.weight"], np.float32)
    qn = np.asarray(w["self_attn.q_norm.weight"], np.float32).reshape(-1) + 1.0
    kn = np.asarray(w["self_attn.k_norm.weight"], np.float32).reshape(-1) + 1.0

    qg = (h @ wq.T).reshape(n, QSA_HQ, 2 * QSA_HD)
    q, gate = qg[..., :QSA_HD], qg[..., QSA_HD:]
    k = (h @ wk.T).reshape(n, QSA_HKV, QSA_HD)
    v = (h @ wv.T).reshape(n, QSA_HKV, QSA_HD)
    q = _rms_last(q, qn)
    k = _rms_last(k, kn)
    c = np.asarray(cos, np.float32)[:, :n].T
    s = np.asarray(sin, np.float32)[:, :n].T
    q = _rope(q, c, s)
    k = _rope(k, c, s)

    new_k = np.ascontiguousarray(k.transpose(1, 2, 0).reshape(KVC, n))
    new_v = np.ascontiguousarray(v.transpose(1, 2, 0).reshape(KVC, n))
    if S % n == 0:
        k_tail = np.tile(new_k, (1, S // n))
        v_tail = np.tile(new_v, (1, S // n))
    else:
        k_tail = np.zeros((KVC, S), np.float32)
        v_tail = np.zeros((KVC, S), np.float32)
        k_tail[:, :n] = new_k
        v_tail[:, :n] = new_v
    k_full = np.concatenate([np.asarray(kc, np.float32), k_tail], axis=1)
    v_full = np.concatenate([np.asarray(vc, np.float32), v_tail], axis=1)
    kh = k_full.reshape(QSA_HKV, QSA_HD, m + S)
    vh = v_full.reshape(QSA_HKV, QSA_HD, m + S)
    mask_f = np.asarray(mask, np.float32)
    outs = np.empty((n, QSA_HQ, QSA_HD), np.float32)
    for head in range(QSA_HQ):
        kv_i = head // G
        scores = (q[:, head] @ kh[kv_i]) * np.float32(SCALE) + mask_f
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        outs[:, head] = p @ vh[kv_i].T
    gated = (outs * sigmoid(gate)).reshape(n, QSA_HQ * QSA_HD)
    attn = gated @ wo.T
    return attn.T, new_k, new_v


def numpy_layer(x_bc1s, kc, vc, cos, sin, mask, n, m, w, shared):
    mixed, hyper, inj = host_gated_residual(
        x_bc1s, w, prefix="attn_hyper_connection")
    attn, new_k, new_v = numpy_qsa(
        mixed[0, :, 0, :n], kc, vc, cos, sin, mask, n, m, w)
    h = host_recombine(attn.reshape(1, H, 1, n), hyper[..., :n], inj[..., :n])
    mixed2, hyper2, _inj2 = host_gated_residual(
        h, w, prefix="mlp_hyper_connection")
    xr = mixed2[0, :, 0, :n].T
    sg, su, sd, sgate = shared
    sh = (silu(xr @ sg.T) * (xr @ su.T)) @ sd.T * sigmoid(xr @ sgate.T)
    return {
        "mixed": mixed2[0, :, 0, :n],
        "hyper": hyper2[0, :, 0, :n],
        "shared": sh.T,
        "new_k": new_k,
        "new_v": new_v,
        "attn_mixed": np.asarray(mixed[0, :, 0, :n], np.float32),
        "attn_inj": np.asarray(inj[0, :, 0, :n], np.float32),
        "attn": attn,
    }


def shared_weights(layer: int):
    bank = Mlx4ExpertBank(MLX4_DEFAULT)
    sg, su, sd = bank.shared_fp32(layer)
    src = MlxSafe(MLX4_DEFAULT)
    try:
        sgate = np.asarray(
            src.f32(f"model.layers.{layer}.mlp.shared_expert_gate.weight"),
            np.float32).reshape(1, H)
    finally:
        src.close()
    return sg, su, sd, sgate


def torch_qsa_check(mixed, kc, vc, cos, sin, mask, n, m, qsa) -> dict:
    """Same feeds through `FlashNextQSADecode`. The numpy core has to match this."""
    h = np.zeros((1, H, 1, S), np.float16)
    h[0, :, 0, :n] = np.asarray(mixed, np.float16)
    mask_t = np.full((1, m + S, 1, S), QSA_MASK, np.float16)
    mask_t[0, :, 0, :n] = np.asarray(mask, np.float16).T
    with torch.no_grad():
        out, nk, nv = qsa(
            torch.from_numpy(h),
            torch.from_numpy(np.asarray(kc, np.float16).reshape(1, KVC, 1, m)),
            torch.from_numpy(np.asarray(vc, np.float16).reshape(1, KVC, 1, m)),
            torch.from_numpy(np.asarray(cos, np.float16).reshape(1, HALF, 1, S)),
            torch.from_numpy(np.asarray(sin, np.float16).reshape(1, HALF, 1, S)),
            torch.from_numpy(mask_t),
        )
    return {
        "attn": out.float().numpy().reshape(H, S)[:, :n],
        "new_k": nk.float().numpy().reshape(KVC, S)[:, :n],
        "new_v": nv.float().numpy().reshape(KVC, S)[:, :n],
    }


def make_feeds(k: int, m: int, rng: np.random.Generator, offset: int = 37):
    x = np.zeros((1, HC_W, 1, S), np.float32)
    x[0, :, 0, :k] = (rng.standard_normal((HC_W, k)) * 0.05).astype(np.float32)
    nsel = min(offset, m)
    kc = np.zeros((KVC, m), np.float16)
    vc = np.zeros((KVC, m), np.float16)
    kc[:, :nsel] = (rng.standard_normal((KVC, nsel)) * 0.05).astype(np.float16)
    vc[:, :nsel] = (rng.standard_normal((KVC, nsel)) * 0.05).astype(np.float16)
    cos = np.zeros((HALF, S), np.float16)
    sin = np.zeros((HALF, S), np.float16)
    inv = 1.0 / (10_000_000.0 ** (
        np.arange(0, QSA_ROTARY, 2, dtype=np.float32) / np.float32(QSA_ROTARY)))
    pos = (np.arange(k, dtype=np.float32) + np.float32(offset))[:, None] * inv
    cos[:, :k] = np.cos(pos).T.astype(np.float16)
    sin[:, :k] = np.sin(pos).T.astype(np.float16)
    mask = np.full((k, m + S), QSA_MASK, np.float16)
    mask[:, :nsel] = 0
    for t in range(k):
        mask[t, m:m + t + 1] = 0
    return x, kc, vc, cos, sin, mask, nsel


def run_k(k: int, m: int, layer: int, w, shared, qsa_torch, lay: MilQsaLayer) -> dict:
    rng = np.random.default_rng(12)
    x, kc, vc, cos, sin, mask, nsel = make_feeds(k, m, rng)
    ref = numpy_layer(x, kc, vc, cos, sin, mask, k, m, w, shared)

    torch_c = torch_qsa_check(
        ref["attn_mixed"], kc, vc, cos, sin, mask, k, m, qsa_torch)
    vs_torch = {
        "attn": rel(ref["attn"], torch_c["attn"]),
        "new_k": rel(ref["new_k"], torch_c["new_k"]),
        "new_v": rel(ref["new_v"], torch_c["new_v"]),
    }

    xb = np.zeros((HC_W, S), np.float16)
    xb[:, :k] = np.asarray(x[0, :, 0, :], np.float16)[:, :k]
    g_mix, g_hyp, _g_inj, g_sh, g_nk, g_nv = lay(
        xb, kc[:, :nsel], vc[:, :nsel], cos, sin, nsel, m, n=k,
        mixed=np.asarray(ref["attn_mixed"], np.float16),
        inj=np.asarray(ref["attn_inj"], np.float16),
    )
    got = {
        "mixed": g_mix.reshape(H, k),
        "hyper": g_hyp.reshape(HC_W, k),
        "shared": g_sh.reshape(H, k),
        "new_k": g_nk,
        "new_v": g_nv,
    }
    vs_mil = {nm: rel(got[nm], ref[nm]) for nm in
              ("mixed", "hyper", "shared", "new_k", "new_v")}
    per = {nm: [rel(got[nm][:, t], ref[nm][:, t]) for t in range(k)]
           for nm in ("mixed", "new_k")}
    return {"vs_torch": vs_torch, "vs_mil": vs_mil, "per": per}


def _report(tag: str, out: dict) -> None:
    k = len(out["per"]["mixed"])
    vt = out["vs_torch"]
    vm = out["vs_mil"]
    print(f"  {tag}: numpy vs torch  attn {vt['attn']:.5f}  "
          f"new_k {vt['new_k']:.5f}  new_v {vt['new_v']:.5f}", flush=True)
    mix_slots = "  ".join(f"t{t}={e:.4f}" for t, e in enumerate(out["per"]["mixed"]))
    nk_slots = "  ".join(f"t{t}={e:.4f}" for t, e in enumerate(out["per"]["new_k"]))
    print(f"  {tag}: mixed {mix_slots}", flush=True)
    print(f"  {tag}: new_k {nk_slots}", flush=True)
    print(f"  {tag}: MIL vs numpy  mixed {vm['mixed']:.5f}  "
          f"hyper {vm['hyper']:.5f}  shared {vm['shared']:.5f}  "
          f"new_k {vm['new_k']:.5f}  new_v {vm['new_v']:.5f}", flush=True)


def walk_compare(narrow: MilQsaLayer, wide: MilQsaLayer, k_wide: int, m: int,
                 w, shared, tokens: int) -> None:
    """Walk the same tokens one-by-one and in k-wide chunks; compare the cache."""
    rng = np.random.default_rng(3)
    x = np.zeros((1, HC_W, 1, tokens), np.float32)
    x[0, :, 0, :] = (rng.standard_normal((HC_W, tokens)) * 0.05).astype(np.float32)
    inv = 1.0 / (10_000_000.0 ** (
        np.arange(0, QSA_ROTARY, 2, dtype=np.float32) / np.float32(QSA_ROTARY)))

    def run(lay, width):
        ck = np.zeros((KVC, tokens), np.float16)
        cv = np.zeros((KVC, tokens), np.float16)
        last = None
        off = 0
        while off < tokens:
            n = width
            xb = np.zeros((1, HC_W, 1, S), np.float32)
            xb[..., :n] = x[..., off:off + n]
            cos = np.zeros((HALF, S), np.float16)
            sin = np.zeros((HALF, S), np.float16)
            pos = (np.arange(n, dtype=np.float32) + np.float32(off))[:, None] * inv
            cos[:, :n] = np.cos(pos).T.astype(np.float16)
            sin[:, :n] = np.sin(pos).T.astype(np.float16)
            mask = np.full((n, m + S), QSA_MASK, np.float16)
            nsel = min(off, m)
            mask[:, :nsel] = 0
            for t in range(n):
                mask[t, m:m + t + 1] = 0
            kc = np.zeros((KVC, m), np.float16)
            vc = np.zeros((KVC, m), np.float16)
            if nsel:
                kc[:, :nsel] = ck[:, :nsel]
                vc[:, :nsel] = cv[:, :nsel]
            ref = numpy_layer(xb, kc, vc, cos, sin, mask, n, m, w, shared)
            x16 = np.zeros((HC_W, S), np.float16)
            x16[:, :n] = np.asarray(xb[0, :, 0, :n], np.float16)
            g_mix, _, _, _, g_nk, g_nv = lay(
                x16, ck[:, :nsel], cv[:, :nsel], cos, sin, nsel, m, n=n,
                mixed=np.asarray(ref["attn_mixed"], np.float16),
                inj=np.asarray(ref["attn_inj"], np.float16))
            ck[:, off:off + n] = np.asarray(g_nk, np.float16)
            cv[:, off:off + n] = np.asarray(g_nv, np.float16)
            last = (g_mix.reshape(H, n), ref["mixed"], g_nk, ref["new_k"])
            off += n
        return ck, cv, last

    nck, ncv, nlast = run(narrow, 1)
    wck, wcv, wlast = run(wide, k_wide)
    print(f"  walk {tokens} tok k=1 vs k={k_wide}: "
          f"cache_k {rel(wck, nck):.5f}  cache_v {rel(wcv, ncv):.5f}  "
          f"last_tok mixed {rel(wlast[0][:, -1], nlast[0][:, -1]):.5f}  "
          f"last_chunk vs numpy {rel(wlast[0], wlast[1]):.5f}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("widths", nargs="*", type=int, default=None,
                    help="token widths to check (default: 1, or 1 4 8 16 32 with --all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--two-proc", action="store_true",
                    help="also compile k=4 + k=32 in one program and check proc 1")
    ap.add_argument("--walk", type=int, default=0,
                    help="walk this many tokens at k=1 vs the last width")
    args = ap.parse_args()
    widths = args.widths or ([1, 4, 8, 16, 32] if args.all else [1])
    m = int(args.m)
    if m % 32:
        raise SystemExit(f"rung {m} is not a multiple of 32")
    li = int(args.layer)

    loader, w = _load_layer(li)
    shared = shared_weights(li)
    qsa_torch = FlashNextQSADecode(max_s=m).eval().half()
    qsa_torch.load_from_layer(w)
    ref_mix = _Ref(w)
    layers = {}

    for k in widths:
        print(f"  compiling QSA L{li} k={k} m={m}", flush=True)
        lay = MilQsaLayer(li, w, ref_mix, qsa_torch, rungs=[m], k=k)
        layers[k] = lay
        _report(f"K={k}", run_k(k, m, li, w, shared, qsa_torch, lay))

    if args.two_proc:
        print(f"  compiling QSA L{li} two-proc k=4/32 m={m}", flush=True)
        two = MilQsaLayer(li, w, ref_mix, qsa_torch, rungs=[m], k=4, prefill_k=32)
        two.select(1)
        _report("two-proc k=32", run_k(32, m, li, w, shared, qsa_torch, two))

    if args.walk:
        if 1 not in layers:
            print("  compiling QSA L{li} k=1 for the walk", flush=True)
            layers[1] = MilQsaLayer(li, w, ref_mix, qsa_torch, rungs=[m], k=1)
        wide_k = widths[-1]
        walk_compare(layers[1], layers[wide_k], wide_k, m, w, shared, args.walk)

    loader.close()


if __name__ == "__main__":
    main()
