#!/usr/bin/env python3
"""Pure-ANE MoE methods that are not AFM pin.

Pin covering-set is dead (flashnext_afm_cover). This times the remaining
GPU-free spellings at Flash-Next shapes:

  dyn-packed   AneDynamicLinear, K experts concatenated, page fp16 then eval
  dyn-serial   same engine, one expert at a time (10 submits / layer)
  int8-baked   constexpr int8 stacked GEMM (what pin would have been)
  onehot       baked Conv2d over E experts (dense-all / one-hot proxy)

Each arm is a child process so an ANE abort does not kill the suite.

    /Users/true/.rindi/venvs/coreai/bin/python -u probes/flashnext_pure_ane_moe.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("Q38_ANE_ENGINE", str(ROOT))
os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")

H, I, S = 2560, 640, 32
K_ROUTE = 10


def _median_ms(fn, n: int = 11, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def arm_dyn_packed(k: int) -> int:
    from runtime.q38_ane_engine import AneDynamicLinear

    rng = np.random.default_rng(0)
    o_gu, i_dn = k * 2 * I, k * I
    print(f"dyn-packed K={k}  gu=[{o_gu},{H}]  dn=[{H},{i_dn}]  S={S}", flush=True)
    t0 = time.perf_counter()
    gu = AneDynamicLinear.compile(H, o_gu, S)
    dn = AneDynamicLinear.compile(i_dn, H, S)
    print(f"  compile {time.perf_counter() - t0:.2f}s  "
          f"gu={'ok' if gu else 'FAIL'} dn={'ok' if dn else 'FAIL'}", flush=True)
    if gu is None or dn is None:
        return 1
    w_gu = rng.standard_normal((o_gu, H), np.float32).astype(np.float32) * 0.02
    w_dn = rng.standard_normal((H, i_dn), np.float32).astype(np.float32) * 0.02
    x = rng.standard_normal((S, H), np.float32).astype(np.float32) * 0.02
    h = rng.standard_normal((S, i_dn), np.float32).astype(np.float32) * 0.02
    t0 = time.perf_counter()
    gu.write_weight(w_gu)
    dn.write_weight(w_dn)
    print(f"  first page {(time.perf_counter() - t0) * 1e3:.1f} ms  "
          f"fp16={(o_gu * H + H * i_dn) * 2 / 1e6:.1f} MB", flush=True)
    if gu.evaluate(x) is None or dn.evaluate(h) is None:
        print("  first eval FAIL", flush=True)
        return 1

    def eval_only():
        gu.evaluate(x)
        dn.evaluate(h)

    def page_eval():
        gu.write_weight(w_gu)
        dn.write_weight(w_dn)
        gu.evaluate(x)
        dn.evaluate(h)

    e = _median_ms(eval_only)
    p = _median_ms(page_eval, n=7)
    bytes_fp16 = (o_gu * H + H * i_dn) * 2
    print(
        f"  eval-only {e:.2f} ms/layer  ({1000 / (48 * e):.2f} tok/s MoE)  "
        f"{bytes_fp16 / e / 1e6:.1f} GB/s",
        flush=True,
    )
    print(
        f"  page+eval {p:.2f} ms/layer  ({1000 / (48 * p):.2f} tok/s MoE)  "
        f"48L={48 * p:.0f} ms/tok",
        flush=True,
    )
    return 0


def arm_dyn_serial(k: int = K_ROUTE) -> int:
    from runtime.q38_ane_engine import AneDynamicLinear

    rng = np.random.default_rng(0)
    print(f"dyn-serial K={k}  one expert / submit  gu=[{2*I},{H}] dn=[{H},{I}]", flush=True)
    t0 = time.perf_counter()
    gu = AneDynamicLinear.compile(H, 2 * I, S)
    dn = AneDynamicLinear.compile(I, H, S)
    print(f"  compile {time.perf_counter() - t0:.2f}s", flush=True)
    if gu is None or dn is None:
        return 1
    weights = [
        (rng.standard_normal((2 * I, H), np.float32).astype(np.float32) * 0.02,
         rng.standard_normal((H, I), np.float32).astype(np.float32) * 0.02)
        for _ in range(k)
    ]
    x = rng.standard_normal((S, H), np.float32).astype(np.float32) * 0.02
    h = rng.standard_normal((S, I), np.float32).astype(np.float32) * 0.02
    gu.write_weight(weights[0][0])
    dn.write_weight(weights[0][1])
    if gu.evaluate(x) is None or dn.evaluate(h) is None:
        print("  first eval FAIL", flush=True)
        return 1

    def page_eval():
        for wg, wd in weights:
            gu.write_weight(wg)
            y = gu.evaluate(x)
            h[0, :I] = y[0, :I]
            dn.write_weight(wd)
            dn.evaluate(h)

    p = _median_ms(page_eval, n=5, warmup=1)
    print(
        f"  {k} experts page+eval {p:.2f} ms/layer  "
        f"({1000 / (48 * p):.2f} tok/s MoE)  48L={48 * p:.0f} ms/tok",
        flush=True,
    )
    return 0


def arm_int8_baked(k: int) -> int:
    from runtime.q38_ane_engine import AneEngine

    rng = np.random.default_rng(0)
    o_gu, i_dn = k * 2 * I, k * I
    print(f"int8-baked K={k}  gu=[{o_gu},{H}]  dn=[{H},{i_dn}]  S={S}", flush=True)
    eng = AneEngine()
    w_gu = rng.standard_normal((o_gu, H), np.float32).astype(np.float32) * 0.02
    w_dn = rng.standard_normal((H, i_dn), np.float32).astype(np.float32) * 0.02
    x = rng.standard_normal((S, H), np.float32).astype(np.float32) * 0.02
    h = rng.standard_normal((S, i_dn), np.float32).astype(np.float32) * 0.02
    t0 = time.perf_counter()
    gu = eng.compile_linear(w_gu, S, quantized=True, keep_weight_dequant=False)
    dn = eng.compile_linear(w_dn, S, quantized=True, keep_weight_dequant=False)
    print(f"  compile {time.perf_counter() - t0:.2f}s  "
          f"gu={'ok' if gu else 'FAIL'} dn={'ok' if dn else 'FAIL'}", flush=True)
    if gu is None or dn is None:
        return 1
    if eng.evaluate(gu, x) is None or eng.evaluate(dn, h) is None:
        print("  first eval FAIL", flush=True)
        return 1

    def eval_only():
        eng.evaluate(gu, x)
        eng.evaluate(dn, h)

    e = _median_ms(eval_only)
    bytes_i8 = o_gu * H + H * i_dn
    print(
        f"  eval {e:.2f} ms/layer  ({1000 / (48 * e):.2f} tok/s MoE)  "
        f"{bytes_i8 / e / 1e6:.1f} GB/s int8  48L={48 * e:.0f} ms/tok",
        flush=True,
    )
    return 0


def arm_decode_budget() -> int:
    """48-layer packed K=10 token vs the 4.7 tok/s bar (~213 ms).

    eval-only = weights already on the surface (the 2.64 ms number).
    fp16-page = memcpy already-fp16 slabs, no fp32 convert.
    submit     = no numpy read-back of Y (closer to a chained decode).
    """
    from runtime.q38_ane_engine import AneDynamicLinear, _iosurface_view

    k = K_ROUTE
    rng = np.random.default_rng(0)
    o_gu, i_dn = k * 2 * I, k * I
    print(f"decode-budget packed K={k}  48 layers  S={S}", flush=True)
    gu = AneDynamicLinear.compile(H, o_gu, S)
    dn = AneDynamicLinear.compile(i_dn, H, S)
    if gu is None or dn is None:
        print("  compile FAIL", flush=True)
        return 1
    w_gu = (rng.standard_normal((o_gu, H), np.float32) * 0.02).astype(np.float16)
    w_dn = (rng.standard_normal((H, i_dn), np.float32) * 0.02).astype(np.float16)
    x = (rng.standard_normal((S, H), np.float32) * 0.02).astype(np.float16)
    h = (rng.standard_normal((S, i_dn), np.float32) * 0.02).astype(np.float16)
    x32 = np.asarray(x, np.float32)
    h32 = np.asarray(h, np.float32)
    gu.write_weight(w_gu)
    dn.write_weight(w_dn)
    if gu.evaluate(x32) is None or dn.evaluate(h32) is None:
        print("  first eval FAIL", flush=True)
        return 1

    def _fill_x(prog, arr):
        with _iosurface_view(prog._x_surf, (prog.input_dim, prog.seq_len), np.float16) as dst:
            dst[:] = 0
            dst[:, 0] = np.asarray(arr, np.float16).reshape(-1)[:prog.input_dim]

    def eval_only_48():
        for _ in range(48):
            gu.evaluate(x32)
            dn.evaluate(h32)

    def fp16_page_eval_48():
        for _ in range(48):
            gu.write_weight(w_gu)
            dn.write_weight(w_dn)
            gu.evaluate(x32)
            dn.evaluate(h32)

    def submit_only_48():
        _fill_x(gu, x32[0])
        _fill_x(dn, h32[0])
        for _ in range(48):
            if not gu.submit() or not dn.submit():
                raise RuntimeError("submit failed")

    def fp16_page_submit_48():
        _fill_x(gu, x32[0])
        _fill_x(dn, h32[0])
        for _ in range(48):
            gu.write_weight(w_gu)
            dn.write_weight(w_dn)
            if not gu.submit() or not dn.submit():
                raise RuntimeError("submit failed")

    attn_ms = 84.0
    bar_ms = 1000.0 / 4.7
    print(f"  4.7 tok/s bar = {bar_ms:.0f} ms/tok  (attn assumed {attn_ms:.0f} ms)", flush=True)
    for name, fn in (
        ("eval-only 48L", eval_only_48),
        ("fp16-page+eval 48L", fp16_page_eval_48),
        ("submit-only 48L", submit_only_48),
        ("fp16-page+submit 48L", fp16_page_submit_48),
    ):
        ms = _median_ms(fn, n=5, warmup=1)
        tot = ms + attn_ms
        print(
            f"  {name:22s}  MoE {ms:7.1f} ms  +attn {tot:7.1f} ms  "
            f"{1000 / tot:4.2f} tok/s  "
            f"{'PASS' if tot <= bar_ms else 'under 4.7'}",
            flush=True,
        )
    return 0


def arm_wide_prefill() -> int:
    """Find the stacked-K compile cliff, then time N×K=32 as a 32-token bank."""
    from runtime.q38_ane_engine import AneDynamicLinear, AneEngine

    rng = np.random.default_rng(0)
    print("wide-prefill: compile cliff (dyn fp16, then int8 baked)", flush=True)
    for k in (36, 40, 48, 56, 64):
        o_gu, i_dn = k * 2 * I, k * I
        t0 = time.perf_counter()
        gu = AneDynamicLinear.compile(H, o_gu, S)
        dn = AneDynamicLinear.compile(i_dn, H, S) if gu is not None else None
        ok = gu is not None and dn is not None
        print(f"  dyn K={k:<3}  {time.perf_counter()-t0:.2f}s  "
              f"{'ok' if ok else 'FAIL'}", flush=True)
        del gu, dn

    eng = AneEngine()
    x = rng.standard_normal((S, H), np.float32).astype(np.float32) * 0.02
    for k in (32, 48, 64, 96):
        o_gu, i_dn = k * 2 * I, k * I
        w_gu = rng.standard_normal((o_gu, H), np.float32).astype(np.float32) * 0.02
        w_dn = rng.standard_normal((H, i_dn), np.float32).astype(np.float32) * 0.02
        h = rng.standard_normal((S, i_dn), np.float32).astype(np.float32) * 0.02
        t0 = time.perf_counter()
        gu = eng.compile_linear(w_gu, S, quantized=True, keep_weight_dequant=False)
        dn = eng.compile_linear(w_dn, S, quantized=True, keep_weight_dequant=False) if gu else None
        dt = time.perf_counter() - t0
        if gu is None or dn is None:
            print(f"  int8 K={k:<3}  compile FAIL  {dt:.2f}s", flush=True)
            continue
        if eng.evaluate(gu, x) is None or eng.evaluate(dn, h) is None:
            print(f"  int8 K={k:<3}  compile ok eval FAIL  {dt:.2f}s", flush=True)
            continue

        def once(gu=gu, dn=dn, h=h):
            eng.evaluate(gu, x)
            eng.evaluate(dn, h)

        e = _median_ms(once, n=5, warmup=1)
        print(
            f"  int8 K={k:<3}  compile {dt:.2f}s  eval {e:.2f} ms/layer  "
            f"32tok MoE={48*e:.0f} ms  ({32/(48*e/1e3):.1f} tok/s MoE)",
            flush=True,
        )
        del gu, dn

    print("wide-prefill: N× dyn K=32 (composed bank, eval-only)", flush=True)
    k = 32
    o_gu, i_dn = k * 2 * I, k * I
    gu = AneDynamicLinear.compile(H, o_gu, S)
    dn = AneDynamicLinear.compile(i_dn, H, S)
    if gu is None or dn is None:
        print("  K=32 compile FAIL", flush=True)
        return 1
    w_gu = (rng.standard_normal((o_gu, H), np.float32) * 0.02).astype(np.float16)
    w_dn = (rng.standard_normal((H, i_dn), np.float32) * 0.02).astype(np.float16)
    x32 = (rng.standard_normal((S, H), np.float32) * 0.02).astype(np.float32)
    h32 = (rng.standard_normal((S, i_dn), np.float32) * 0.02).astype(np.float32)
    gu.write_weight(w_gu)
    dn.write_weight(w_dn)
    gu.evaluate(x32)
    dn.evaluate(h32)
    for npack in (1, 2, 3, 4):
        def packs(n=npack):
            for _ in range(n):
                gu.evaluate(x32)
                dn.evaluate(h32)

        e = _median_ms(packs, n=5, warmup=1)
        experts = npack * 32
        print(
            f"  {npack}×K=32 ({experts} experts)  {e:.2f} ms/layer  "
            f"48L={48*e:.0f} ms / 32 tok  ({32/(48*e/1e3):.1f} tok/s MoE)",
            flush=True,
        )
    return 0


def arm_onehot(e: int) -> int:
    """Baked Conv2d over E experts — the ANE-legal spelling of dense-all."""
    import asyncio
    import tempfile

    import torch
    from torch import nn
    from coreai_torch import TorchConverter, get_decomp_table
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions, NDArray

    class Bank(nn.Module):
        def __init__(self):
            super().__init__()
            self.gu = nn.Conv2d(H, e * 2 * I, 1, bias=False)
            self.dn = nn.Conv2d(e * I, H, 1, bias=False)

        def forward(self, x, act):
            y = self.gu(x)
            return self.dn(act)

    print(f"onehot baked E={e}  gu C={e*2*I}  S={S}", flush=True)
    m = Bank().eval().half()
    x = torch.randn(1, H, 1, S, dtype=torch.float16)
    act = torch.randn(1, e * I, 1, S, dtype=torch.float16)
    t0 = time.perf_counter()
    ep = torch.export.export(m, (x, act)).run_decompositions(get_decomp_table())
    prog = (TorchConverter()
            .add_exported_program(ep, input_names=["x", "act"], output_names=["y"])
            .to_coreai())
    prog.optimize()
    tmp = Path(tempfile.mkdtemp(prefix="fn_onehot_")) / "m.aimodel"
    prog.save_asset(tmp)
    print(f"  convert {time.perf_counter() - t0:.1f}s", flush=True)

    async def run(kind_name: str):
        kinds = {str(k): k for k in ComputeUnitKind.available_kinds()}
        kind = kinds[kind_name]
        mm = await AIModel.load(
            str(tmp),
            specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(kind),
        )
        fn = mm.load_function("main")
        xn = NDArray(x.numpy())
        an = NDArray(act.numpy())
        await fn({"x": xn, "act": an})
        ts = []
        for _ in range(6):
            t = time.perf_counter()
            await fn({"x": xn, "act": an})
            ts.append((time.perf_counter() - t) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    gpu = asyncio.run(run("GPU"))
    print(f"  GPU {gpu:.2f} ms", flush=True)
    ane = asyncio.run(run("Neural Engine"))
    bytes_fp16 = (e * 2 * I * H + H * e * I) * 2
    print(
        f"  ANE {ane:.2f} ms/layer  ({1000 / (48 * ane):.2f} tok/s MoE)  "
        f"{bytes_fp16 / ane / 1e6:.1f} GB/s  48L={48 * ane:.0f} ms/tok",
        flush=True,
    )
    return 0


def _spawn(arm: str, extra: list[str]) -> int:
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--child", arm, *extra,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n== {arm} {' '.join(extra)} ==", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"  exit {r.returncode}", flush=True)
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--child", default="")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--e", type=int, default=32)
    p.add_argument("--only", default="",
                   help="comma list: dyn-packed,dyn-serial,int8-baked,onehot")
    args = p.parse_args()
    if args.child == "dyn-packed":
        return arm_dyn_packed(args.k)
    if args.child == "dyn-serial":
        return arm_dyn_serial(args.k)
    if args.child == "int8-baked":
        return arm_int8_baked(args.k)
    if args.child == "onehot":
        return arm_onehot(args.e)
    if args.child == "decode-budget":
        return arm_decode_budget()
    if args.child == "wide-prefill":
        return arm_wide_prefill()

    want = {x.strip() for x in args.only.split(",") if x.strip()} or {
        "dyn-packed", "dyn-serial", "int8-baked", "onehot",
    }
    rc = 0
    if "dyn-serial" in want:
        rc |= _spawn("dyn-serial", ["--k", "10"])
    if "dyn-packed" in want:
        for k in (10, 16, 32, 64):
            rc |= _spawn("dyn-packed", ["--k", str(k)])
    if "int8-baked" in want:
        for k in (10, 16):
            rc |= _spawn("int8-baked", ["--k", str(k)])
    if "onehot" in want:
        for e in (32, 64):
            rc |= _spawn("onehot", ["--e", str(e)])
    if "decode-budget" in want:
        rc |= _spawn("decode-budget", [])
    if "wide-prefill" in want:
        rc |= _spawn("wide-prefill", [])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
