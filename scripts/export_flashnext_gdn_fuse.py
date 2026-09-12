#!/usr/bin/env python3
"""GDN submit fusion: cut front + gdn-only from 2 ANE submits to 1.

SiLU / L2 / decay / beta stay on the host (in-graph SiLU poisoned GDN, rel ~0.09).
That means a *connected* front→gdn graph is impossible: HostPrep sits between
them. This stage measures the honest alternatives:

  fused   one compiled graph with two *disconnected* subgraphs (front || gdn).
          Host-prep tensors are inputs. One ANE submit only if the host already
          has q/k/v/decay/beta/z — i.e. a CPU S=1 front clone ran first.
  pair    one .aimodel, two entrypoints (`front`, `gdn`), shared specialization.
          Still two submits; cheaper load; SurfacePool-style rewrap.
  host1   CPU S=1 front + existing ANE gdn-only. True 1-submit, no new graph
          math, front GEMM on host (S=1, no width-32 pad).

GPU then ANE. If ANE aborts the fused graph, stop that graph.

  ~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_gdn_fuse.py
  ~/.rindi/venvs/coreai/bin/python scripts/export_flashnext_coreai.py fuse
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_flashnext_coreai import (  # noqa: E402
    BASE,
    DK,
    DV,
    GDN_Y,
    H,
    HK,
    HV,
    IN_O,
    OUT_DIR,
    QKV,
    SEQ_DEFAULT,
    FlashNextFront,
    FlashNextGdnOnly,
    HostPrep,
    _bc1s_to_bsh,
    _bsh_to_bc1s,
    _gdn_asset_paths,
    _load_layer0,
    _pad32,
    _rel,
    host_gated_residual,
)

REL_BUDGET = 0.02


class FlashNextGdnFused(nn.Module):
    """Front and gdn-only in one module. HostPrep tensors are inputs.

    No SiLU / exp / softplus. Subgraphs do not consume each other's tensors, so
    the compiler cannot skip HostPrep — this is one *submit*, not one *math*
    pipeline.
    """

    def __init__(self):
        super().__init__()
        self.front = FlashNextFront()
        self.gdn = FlashNextGdnOnly()

    def load_from_layer(self, w) -> None:
        self.front.load_from_layer(w)
        self.gdn.load_from_layer(w)

    def forward(self, h, conv_pack, q, k, v, decay, beta, state, z):
        yin, new_pack = self.front(h, conv_pack)
        attn, new_ssm = self.gdn(q, k, v, decay, beta, state, z)
        return yin, new_pack, attn, new_ssm


class HostFront:
    """S=1 numpy clone of FlashNextFront. Writes yin / new_pack, no SiLU."""

    __slots__ = ("in_proj", "taps")

    def __init__(self, w):
        self.in_proj = np.ascontiguousarray(
            np.concatenate(
                [
                    np.asarray(w["linear_attn.in_proj_qkv.weight"], np.float32),
                    np.asarray(w["linear_attn.in_proj_z.weight"], np.float32),
                    np.asarray(w["linear_attn.in_proj_b.weight"], np.float32),
                    np.asarray(w["linear_attn.in_proj_a.weight"], np.float32),
                ],
                axis=0,
            )
        )
        self.taps = np.ascontiguousarray(
            np.asarray(w["linear_attn.conv1d.weight"], np.float32).reshape(QKV, 4)
        )

    def __call__(self, h_bc1s: np.ndarray, conv_pack: np.ndarray):
        """h (1,H,1,S) token in slot 0; conv_pack (1, 3*QKV, 1, S). Returns S=1 yin."""
        h = np.asarray(h_bc1s, np.float32)[0, :, 0, 0]
        pack = np.asarray(conv_pack, np.float32)
        s0 = pack[0, 0:QKV, 0, 0]
        s1 = pack[0, QKV : 2 * QKV, 0, 0]
        s2 = pack[0, 2 * QKV : 3 * QKV, 0, 0]
        yin = self._project(h)
        # Preserve the raw projection for the next convolution step. A view
        # aliases yin, whose first QKV entries are overwritten below.
        qkv = yin[:QKV].copy()
        t = self.taps
        yin[:QKV] = s0 * t[:, 0] + s1 * t[:, 1] + s2 * t[:, 2] + qkv * t[:, 3]
        new_pack = np.concatenate([s1, s2, qkv])
        yin_b = np.zeros((1, IN_O, 1, 1), np.float32)
        yin_b[0, :, 0, 0] = yin
        pack_b = np.zeros((1, 3 * QKV, 1, 1), np.float32)
        pack_b[0, :, 0, 0] = new_pack
        return yin_b, pack_b

    def _project(self, h):
        return self.in_proj @ h

    def step(self, h_bc1s: np.ndarray, conv_state: np.ndarray):
        """Decode step: conv_state is GDNState.conv (1, 3, QKV). Returns S=1 yin + new conv."""
        pack = np.zeros((1, 3 * QKV, 1, 1), np.float32)
        c = np.asarray(conv_state, np.float32)
        pack[0, 0:QKV, 0, 0] = c[0, 0]
        pack[0, QKV : 2 * QKV, 0, 0] = c[0, 1]
        pack[0, 2 * QKV : 3 * QKV, 0, 0] = c[0, 2]
        yin, new_pack = self(np.asarray(h_bc1s)[..., :1], pack)
        new_conv = np.empty_like(c)
        new_conv[0, 0] = new_pack[0, 0:QKV, 0, 0]
        new_conv[0, 1] = new_pack[0, QKV : 2 * QKV, 0, 0]
        new_conv[0, 2] = new_pack[0, 2 * QKV : 3 * QKV, 0, 0]
        return yin, new_conv


class MlxFront(HostFront):
    """Brief resident GPU projection; convolution and ANE recurrence unchanged."""
    __slots__ = ("_mx_w", "_bits")

    def __init__(self, w):
        import mlx.core as mx
        super().__init__(w)
        mx.set_wired_limit(min(80 * 1024**3, mx.device_info()["max_recommended_working_set_size"]))
        mx.set_cache_limit(256 * 1024**2)
        self._bits = int(os.environ.get("FLASHNEXT_FRONT_BITS", "32"))
        if self._bits not in (8, 32):
            raise ValueError("FLASHNEXT_FRONT_BITS must be 8 or 32")
        weight = mx.array(self.in_proj)
        self._mx_w = mx.quantize(weight, group_size=64, bits=8) if self._bits == 8 else weight
        if self._bits == 8:
            mx.eval(*self._mx_w)
        else:
            mx.eval(self._mx_w)
        self.in_proj = None

    def _project(self, h):
        import mlx.core as mx
        if self._bits == 8:
            y = mx.quantized_matmul(mx.array(h)[None], *self._mx_w,
                                    transpose=True, group_size=64, bits=8).reshape(-1)
        else:
            y = self._mx_w @ mx.array(h)
        mx.eval(y)
        return np.array(y)


def _export_ep(model, example, names, tag: str, *, entrypoint: str = "main") -> Path:
    from coreai_torch import TorchConverter, get_decomp_table

    model.eval().half()
    t0 = time.perf_counter()
    exported = torch.export.export(model, args=example).run_decompositions(get_decomp_table())
    print(f"  torch.export {tag} {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    program = (
        TorchConverter()
        .add_exported_program(
            exported,
            input_names=names[0],
            output_names=names[1],
            entrypoint_name=entrypoint,
        )
        .to_coreai()
    )
    program.optimize()
    print(f"  coreai convert+optimize {tag} {time.perf_counter() - t0:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"flashnext_{tag}.aimodel"
    if out.exists():
        shutil.rmtree(out)
    program.save_asset(out)
    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"  saved {out} ({mb:.1f} MB)")
    return out


def _export_pair(front, gdn, front_ex, gdn_ex, tag: str) -> Path:
    from coreai_torch import TorchConverter, get_decomp_table

    front.eval().half()
    gdn.eval().half()
    t0 = time.perf_counter()
    ep_f = torch.export.export(front, args=front_ex).run_decompositions(get_decomp_table())
    ep_g = torch.export.export(gdn, args=gdn_ex).run_decompositions(get_decomp_table())
    print(f"  torch.export pair {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    program = (
        TorchConverter()
        .add_exported_program(
            ep_f,
            input_names=["h", "conv_pack"],
            output_names=["yin", "new_pack"],
            entrypoint_name="front",
        )
        .add_exported_program(
            ep_g,
            input_names=["q", "k", "v", "decay", "beta", "state", "z"],
            output_names=["attn", "new_ssm"],
            entrypoint_name="gdn",
        )
        .to_coreai()
    )
    program.optimize()
    print(f"  coreai convert+optimize pair {time.perf_counter() - t0:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"flashnext_{tag}.aimodel"
    if out.exists():
        shutil.rmtree(out)
    program.save_asset(out)
    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"  saved {out} ({mb:.1f} MB)  functions=front,gdn")
    return out


def _fuse_paths(layer: int, seq: int) -> tuple[Path, Path]:
    return (
        OUT_DIR / f"flashnext_gdn_fuse_L{layer}_s{seq}.aimodel",
        OUT_DIR / f"flashnext_gdn_pair_L{layer}_s{seq}.aimodel",
    )


async def _load_kind(path: Path, label: str):
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

    kinds = {str(k): k for k in ComputeUnitKind.available_kinds()}
    key = {"gpu": "GPU", "ane": "Neural Engine", "cpu": "CPU"}[label]
    kind = kinds.get(key)
    if kind is None:
        print(f"  [{label}] not available")
        return None, None
    spec = SpecializationOptions.from_preferred_compute_unit_kind(kind)
    t0 = time.perf_counter()
    try:
        mm = await AIModel.load(str(path), specialization_options=spec)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] LOAD FAILED: {exc}")
        return None, None
    print(f"  [{label}] load {time.perf_counter() - t0:.1f}s  functions={mm.function_names}")
    return mm, mm.function_names


def _nd(feeds: dict):
    from coreai.runtime import NDArray

    return {k: NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}


async def _eval_fn(fn, feeds: dict, runs: int = 8) -> tuple[dict, float]:
    nd = _nd(feeds)
    out = await fn(nd)
    await fn(nd)
    t0 = time.perf_counter()
    for _ in range(runs):
        out = await fn(nd)
    ms = (time.perf_counter() - t0) * 1e3 / runs
    return out, ms


def _attn_rel(out_attn, ref_np, name: str) -> float:
    g = np.asarray(out_attn, np.float32)[..., :1]
    return _rel(_bc1s_to_bsh(g), ref_np, name)


async def _bench_units(path: Path, fn_name: str, feeds: dict, ref_attn, seq: int,
                       units: tuple[str, ...] = ("gpu", "ane")) -> dict[str, float]:
    """GPU then ANE. Returns {label: ms}. Stops the ANE attempt on run/load fail."""
    times: dict[str, float] = {}
    for label in units:
        mm, names = await _load_kind(path, label)
        if mm is None:
            if label == "ane":
                print("  [ane] abort/fail — stopping this graph")
            continue
        if fn_name not in names:
            print(f"  [{label}] missing function {fn_name!r} have {names}")
            continue
        fn = mm.load_function(fn_name)
        try:
            out, ms = await _eval_fn(fn, feeds)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] RUN FAILED: {exc}")
            if label == "ane":
                print("  [ane] abort/fail — stopping this graph")
            continue
        rel = _attn_rel(out["attn"].numpy(), ref_attn, f"[{label}] fused/gdn vs numpy")
        print(f"  [{label}] {ms:.2f} ms/eval  attn rel={rel:.4f}")
        times[label] = ms
        if label == "ane" and rel > REL_BUDGET:
            print(f"  [ane] rel {rel:.4f} > {REL_BUDGET} — do not ship this graph")
    return times


def stage_fuse(seq: int, skip_bench: bool, reuse: bool, export_layers: bool) -> None:
    from tools.flashnext_reference import linear_attention_layer

    print("loading layer 0 from BF16 base (read-only mmap)…")
    loader, w = _load_layer0()
    front = FlashNextFront().eval().half()
    front.load_from_layer(w)
    gdn = FlashNextGdnOnly().eval().half()
    gdn.load_from_layer(w)
    fused = FlashNextGdnFused().eval().half()
    fused.load_from_layer(w)
    prep = HostPrep(w, seq=seq)
    host_front = HostFront(w)

    rng = np.random.default_rng(0)
    x_bsh = (rng.standard_normal((1, 1, 4 * H)) * 0.1).astype(np.float32)
    x_bc1s = _bsh_to_bc1s(x_bsh)
    mixed_h, _, _ = host_gated_residual(x_bc1s, w, prefix="attn_hyper_connection")
    mixed_r = _bc1s_to_bsh(mixed_h)
    attn_r, _ = linear_attention_layer(w, mixed_r, None)

    h32 = torch.from_numpy(_pad32(mixed_h, seq))
    pack32 = torch.zeros(1, 3 * QKV, 1, seq, dtype=torch.float16)
    ssm32 = torch.zeros(1, HV, DV, DK, dtype=torch.float16)
    with torch.no_grad():
        yin_t, new_pack_t = front(h32, pack32)
        q, k, v, decay, beta, z = prep(np.asarray(yin_t.detach().numpy()))
        qt = torch.from_numpy(np.ascontiguousarray(q))
        kt = torch.from_numpy(np.ascontiguousarray(k))
        vt = torch.from_numpy(np.ascontiguousarray(v))
        decay_t = torch.from_numpy(np.ascontiguousarray(decay))
        beta_t = torch.from_numpy(np.ascontiguousarray(beta))
        z_t = torch.from_numpy(np.ascontiguousarray(z))
        attn_t, ns_t = gdn(qt, kt, vt, decay_t, beta_t, ssm32, z_t)
        fused_out = fused(h32, pack32, qt, kt, vt, decay_t, beta_t, ssm32, z_t)
    print("  fp16 pytorch GDN vs numpy linear_attention")
    _rel(_bc1s_to_bsh(np.asarray(attn_t[..., :1].detach().numpy(), np.float32)), attn_r, "pt GDN vs numpy")
    _rel(fused_out[0].float().numpy(), yin_t.float().numpy(), "fused.yin vs front")
    _rel(fused_out[2][..., :1].float().numpy(), attn_t[..., :1].float().numpy(), "fused.attn vs gdn")

    yin_h, _ = host_front(mixed_h[..., :1], np.zeros((1, 3 * QKV, 1, 1), np.float32))
    prep_h = HostPrep(w, seq=seq)
    qh, kh, vh, dh, bh, zh = prep_h(_pad32(yin_h, seq))
    print("  host S=1 front+prep vs pytorch front+prep")
    _rel(qh, q, "host q vs pt q")
    _rel(kh, k, "host k vs pt k")
    _rel(vh, v, "host v vs pt v")
    _rel(dh, decay, "host decay vs pt")
    _rel(yin_h, np.asarray(yin_t[..., :1].detach().numpy(), np.float32), "host yin vs pt yin")

    n_front = 8
    t0 = time.perf_counter()
    for _ in range(n_front):
        host_front(mixed_h[..., :1], np.zeros((1, 3 * QKV, 1, 1), np.float32))
    host_front_ms = (time.perf_counter() - t0) * 1e3 / n_front
    t0 = time.perf_counter()
    for _ in range(n_front):
        prep(_pad32(yin_h, seq))
    prep_ms = (time.perf_counter() - t0) * 1e3 / n_front
    print(f"  host front S=1 {host_front_ms:.2f} ms   HostPrep {prep_ms:.2f} ms")

    p_fuse, p_pair = _fuse_paths(0, seq)
    if reuse and p_fuse.is_dir():
        print(f"  reuse {p_fuse}")
    else:
        p_fuse = _export_ep(
            fused,
            (h32, pack32, qt, kt, vt, decay_t, beta_t, ssm32, z_t),
            (
                ["h", "conv_pack", "q", "k", "v", "decay", "beta", "state", "z"],
                ["yin", "new_pack", "attn", "new_ssm"],
            ),
            f"gdn_fuse_L0_s{seq}",
        )
    if reuse and p_pair.is_dir():
        print(f"  reuse {p_pair}")
    else:
        p_pair = _export_pair(
            front, gdn, (h32, pack32),
            (qt, kt, vt, decay_t, beta_t, ssm32, z_t),
            f"gdn_pair_L0_s{seq}",
        )
    loader.close()
    if skip_bench:
        return

    feeds_fused = {
        "h": np.ascontiguousarray(h32.numpy()),
        "conv_pack": np.ascontiguousarray(pack32.numpy()),
        "q": np.ascontiguousarray(q),
        "k": np.ascontiguousarray(k),
        "v": np.ascontiguousarray(v),
        "decay": np.ascontiguousarray(decay),
        "beta": np.ascontiguousarray(beta),
        "state": np.ascontiguousarray(ssm32.numpy()),
        "z": np.ascontiguousarray(z),
    }
    feeds_front = {"h": feeds_fused["h"], "conv_pack": feeds_fused["conv_pack"]}
    feeds_gdn = {n: feeds_fused[n] for n in ("q", "k", "v", "decay", "beta", "state", "z")}
    p_front_split, p_gdn_split = _gdn_asset_paths(0, seq)

    print("\n== pair two-function shared specialization GPU then ANE (known-good I/O) ==")
    pair_stats = asyncio.run(_bench_pair_pipeline(
        p_pair, p_front_split, p_gdn_split,
        feeds_front, feeds_gdn, prep, seq, attn_r, host_front_ms, prep_ms,
    ))

    print("\n== fused one-graph (front || gdn, host-prep as inputs) GPU then ANE ==")
    print("  ANE last: if this graph aborts, pair/host1 numbers above still stand")
    fused_ms = asyncio.run(_bench_units(p_fuse, "main", feeds_fused, attn_r, seq))
    _print_fused_verdict(fused_ms, host_front_ms, prep_ms, pair_stats)

    if export_layers:
        _export_remaining(seq, reuse=reuse)


async def _bench_pair_pipeline(
    p_pair: Path,
    p_front_split: Path,
    p_gdn_split: Path,
    feeds_front: dict,
    feeds_gdn: dict,
    prep: HostPrep,
    seq: int,
    attn_r: np.ndarray,
    host_front_ms: float,
    prep_ms: float,
) -> dict:
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    kinds = {str(k): k for k in ComputeUnitKind.available_kinds()}
    gpu = kinds.get("GPU")
    ane = kinds.get("Neural Engine")
    if gpu is None:
        print("  GPU missing — skip pair (ANE-first is unsafe)")
        return {}

    async def load(path: Path, kind, tag: str):
        spec = SpecializationOptions.from_preferred_compute_unit_kind(kind)
        t0 = time.perf_counter()
        mm = await AIModel.load(str(path), specialization_options=spec)
        print(f"  [{tag}] load {time.perf_counter() - t0:.1f}s  {mm.function_names}")
        return mm

    print("  GPU pair (before ANE)…")
    mm = await load(p_pair, gpu, "gpu pair")
    fn_f = mm.load_function("front")
    fn_g = mm.load_function("gdn")
    out_f, ms_f = await _eval_fn(fn_f, feeds_front)
    q, k, v, decay, beta, z = prep(np.asarray(out_f["yin"].numpy()))
    feeds_g = {
        "q": q, "k": k, "v": v, "decay": decay, "beta": beta,
        "state": feeds_gdn["state"], "z": z,
    }
    out_g, ms_g = await _eval_fn(fn_g, feeds_g)
    rel_gpu = _attn_rel(out_g["attn"].numpy(), attn_r, "[gpu] pair gdn vs numpy")
    print(f"  [gpu] pair sequential front {ms_f:.2f} + gdn {ms_g:.2f} = {ms_f + ms_g:.2f} ms  rel={rel_gpu:.4f}")

    if ane is None:
        print("  Neural Engine missing")
        return {}

    print("  ANE split baseline (two assets) + pair + persistent pipeline…")
    try:
        m_fs = await load(p_front_split, ane, "ane split-front") if p_front_split.is_dir() else None
        m_gs = await load(p_gdn_split, ane, "ane split-gdn") if p_gdn_split.is_dir() else None
        m_p = await load(p_pair, ane, "ane pair")
    except Exception as exc:  # noqa: BLE001
        print(f"  [ane] LOAD FAILED: {exc} — stopping")
        return {}

    fn_ps = (m_fs.load_function("main"), m_gs.load_function("main")) if m_fs and m_gs else None
    fn_pf = m_p.load_function("front")
    fn_pg = m_p.load_function("gdn")

    async def two_submit(fn_front, fn_gdn, persistent: bool) -> tuple[float, float, float, float]:
        from runtime.coreai_surfaces import wrap_ndarray

        # Mutating NDArray.numpy() and resubmitting feeds stale zeros (rel=1.0).
        # Match SurfacePool: re-wrap host-written tensors; ping-pong only ssm.
        ssm_nd = wrap_ndarray(feeds_gdn["state"])

        async def once():
            t0 = time.perf_counter()
            of = await fn_front({
                "h": wrap_ndarray(feeds_front["h"]),
                "conv_pack": wrap_ndarray(feeds_front["conv_pack"]),
            })
            t1 = time.perf_counter()
            prep(np.asarray(of["yin"].numpy()))
            t2 = time.perf_counter()
            og = await fn_gdn({
                "q": wrap_ndarray(prep.q),
                "k": wrap_ndarray(prep.k),
                "v": wrap_ndarray(prep.v),
                "decay": wrap_ndarray(prep.decay),
                "beta": wrap_ndarray(prep.beta),
                "state": ssm_nd,
                "z": wrap_ndarray(prep.z),
            })
            t3 = time.perf_counter()
            return of, og, (t1 - t0), (t2 - t1), (t3 - t2)

        of, og, *_ = await once()
        rel = _attn_rel(og["attn"].numpy(), attn_r, "ane vs numpy")
        await once()
        acc = np.zeros(3, np.float64)
        runs = 8
        for _ in range(runs):
            _, _, *ts = await once()
            acc += ts
        acc = acc / runs * 1e3
        return rel, float(acc[0]), float(acc[1]), float(acc[2])

    split_times = None
    if fn_ps:
        try:
            rel, fms, pms, gms = await two_submit(fn_ps[0], fn_ps[1], persistent=True)
            split_times = (rel, fms, pms, gms)
            print(
                f"  [ane] split+rewrap  front {fms:.2f}  prep {pms:.2f}  "
                f"gdn {gms:.2f}  total {fms + pms + gms:.2f} ms  rel={rel:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [ane] split pipeline FAILED: {exc}")

    try:
        rel, fms, pms, gms = await two_submit(fn_pf, fn_pg, persistent=True)
        print(
            f"  [ane] pair+rewrap  front {fms:.2f}  prep {pms:.2f}  "
            f"gdn {gms:.2f}  total {fms + pms + gms:.2f} ms  rel={rel:.4f}"
        )
        pair_total = fms + pms + gms
    except Exception as exc:  # noqa: BLE001
        print(f"  [ane] pair pipeline FAILED: {exc} — stopping")
        return {}

    # host S=1 front + existing gdn-only (true 1 ANE submit, no SiLU in graph)
    if m_gs is not None:
        fn_g_only = m_gs.load_function("main")
        gdn_feeds = {n: NDArray(np.ascontiguousarray(feeds_gdn[n])) for n in feeds_gdn}

        async def gdn_only_once():
            t0 = time.perf_counter()
            og = await fn_g_only(gdn_feeds)
            return og, time.perf_counter() - t0

        try:
            og, _ = await gdn_only_once()
            rel1 = _attn_rel(og["attn"].numpy(), attn_r, "[ane] gdn-only vs numpy")
            await gdn_only_once()
            acc = 0.0
            runs = 8
            for _ in range(runs):
                _, dt = await gdn_only_once()
                acc += dt
            gdn_only_ms = acc / runs * 1e3
            host1 = host_front_ms + prep_ms + gdn_only_ms
            print(
                f"  [ane] hostS1+gdn-only  host_front {host_front_ms:.2f}  "
                f"prep {prep_ms:.2f}  gdn {gdn_only_ms:.2f}  total {host1:.2f} ms  "
                f"rel={rel1:.4f}  (1 ANE submit)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [ane] gdn-only 1-submit FAILED: {exc}")
            host1 = None
            rel1 = None
            gdn_only_ms = None
    else:
        host1 = rel1 = gdn_only_ms = None

    print("\n== pair/host1 verdict ==")
    print("  connected one-graph without SiLU: impossible (HostPrep is between front and gdn)")
    print(f"  pair 2-submit rewrap {pair_total:.2f} ms  (shared specialization, still 2 submits)")
    if split_times:
        print(
            f"  split 2-submit rewrap {sum(split_times[1:]):.2f} ms  "
            f"rel={split_times[0]:.4f}"
        )
    if host1 is not None:
        print(
            f"  host S=1 front + ANE gdn-only {host1:.2f} ms  rel={rel1:.4f}  "
            f"→ 36 GDN submits not 72 if generate skips ANE front"
        )
        win = host1 < pair_total - 0.3
        print(
            f"  2→1 submit via host front + gdn-only: "
            f"{'faster' if win else 'not faster than pair 2-submit'} "
            f"(SiLU stays host; rel budget {REL_BUDGET})"
        )
    return {
        "pair_total": pair_total,
        "host1": host1,
        "host1_rel": rel1,
        "split_total": None if split_times is None else sum(split_times[1:]),
        "split_rel": None if split_times is None else split_times[0],
    }


def _print_fused_verdict(fused_ms: dict[str, float], host_front_ms: float,
                         prep_ms: float, pair_stats: dict | None) -> None:
    print("\n== fused verdict ==")
    fuse_ane = fused_ms.get("ane")
    fuse_gpu = fused_ms.get("gpu")
    if fuse_ane is not None:
        extra = host_front_ms + prep_ms
        print(
            f"  fused 1-submit ANE {fuse_ane:.2f} ms  GPU {fuse_gpu}  "
            f"(+ host front {host_front_ms:.2f} + prep {prep_ms:.2f} = "
            f"{fuse_ane + extra:.2f} ms if q/k/v come from a CPU clone)"
        )
        print(
            "  fused graph still needs pre-baked q/k/v — ANE front in that submit is "
            "redundant with the CPU clone, or you keep 2 submits"
        )
        if pair_stats and pair_stats.get("pair_total") is not None:
            print(
                f"  vs pair 2-submit {pair_stats['pair_total']:.2f} ms  "
                f"vs host1 {pair_stats.get('host1')}"
            )
    else:
        print("  fused 1-submit: ANE did not run (load/run fail or skipped)")


def _export_remaining(seq: int, reuse: bool) -> None:
    """Bake fused + pair assets for every linear_attention layer."""
    from tools.flashnext_reference import FlashNextLoader

    loader = FlashNextLoader(str(BASE))
    types = list(loader.text_config["layer_types"])
    fused = FlashNextGdnFused().eval().half()
    front = FlashNextFront().eval().half()
    gdn = FlashNextGdnOnly().eval().half()
    h32 = torch.zeros(1, H, 1, seq, dtype=torch.float16)
    pack32 = torch.zeros(1, 3 * QKV, 1, seq, dtype=torch.float16)
    qt = torch.zeros(1, HV, seq, DK, dtype=torch.float16)
    kt = torch.zeros(1, HV, seq, DK, dtype=torch.float16)
    vt = torch.zeros(1, HV, seq, DV, dtype=torch.float16)
    decay_t = torch.zeros(1, HV, 1, seq, dtype=torch.float16)
    beta_t = torch.zeros(1, HV, 1, seq, dtype=torch.float16)
    ssm32 = torch.zeros(1, HV, DV, DK, dtype=torch.float16)
    z_t = torch.zeros(1, GDN_Y, 1, seq, dtype=torch.float16)
    n_ok = n_skip = 0
    t_all = time.perf_counter()
    for i, lt in enumerate(types):
        if lt != "linear_attention":
            continue
        p_fuse, p_pair = _fuse_paths(i, seq)
        if reuse and p_fuse.is_dir() and p_pair.is_dir():
            print(f"  L{i:02d} fuse/pair reuse", flush=True)
            n_skip += 1
            continue
        t0 = time.perf_counter()
        w = loader.layer(i)
        fused.load_from_layer(w)
        front.load_from_layer(w)
        gdn.load_from_layer(w)
        if not (reuse and p_fuse.is_dir()):
            _export_ep(
                fused,
                (h32, pack32, qt, kt, vt, decay_t, beta_t, ssm32, z_t),
                (
                    ["h", "conv_pack", "q", "k", "v", "decay", "beta", "state", "z"],
                    ["yin", "new_pack", "attn", "new_ssm"],
                ),
                f"gdn_fuse_L{i}_s{seq}",
            )
        if not (reuse and p_pair.is_dir()):
            _export_pair(
                front, gdn, (h32, pack32),
                (qt, kt, vt, decay_t, beta_t, ssm32, z_t),
                f"gdn_pair_L{i}_s{seq}",
            )
        n_ok += 1
        print(f"  L{i:02d} fuse+pair {time.perf_counter() - t0:.1f}s", flush=True)
    loader.close()
    print(f"  remaining layers exported={n_ok} reused={n_skip}  {time.perf_counter() - t_all:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=SEQ_DEFAULT)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--reuse", action="store_true")
    p.add_argument(
        "--export-layers",
        action="store_true",
        help="after L0, bake fused+pair for every GDN layer",
    )
    args = p.parse_args()
    print(f"stage=fuse seq={args.seq}  base={BASE}")
    stage_fuse(args.seq, args.skip_bench, args.reuse, args.export_layers)


if __name__ == "__main__":
    main()
