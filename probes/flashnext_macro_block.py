#!/usr/bin/env python3
"""One Flash-Next linear-attention layer as a *single* constexpr ANE graph.

Not a GEMM RPC server. Load layer 0 from the BF16 base (read-only mmap),
requantize int8 per-channel into an ephemeral blob, compile:

  mixed[H] -> in_proj + out_proj(proxy GDN y) + stacked SwiGLU(+shared)

and time it at S=1, S=4 (hc_count), S=32.

Base / 4-bit / AWQ checkpoints are never opened for write. Any requant lives
in the compiled ANE weight blob only.

  Q38_ANE_REUSE_COMPILED=0 python3 -u probes/flashnext_macro_block.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("Q38_ANE_ENGINE", str(ROOT))
os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")
sys.path.insert(0, str(ROOT))

from runtime.ane_baked_moe import _swiglu  # noqa: E402
from runtime.ane_fused_w8a8 import MultiOut, Prog  # noqa: E402
from runtime.q38_ane_engine import AneEngine  # noqa: E402
from tools.flashnext_reference import FlashNextLoader  # noqa: E402

BASE = Path("/Users/true/models/Qwen3.8-Flash-Next")
MLX4 = Path("/Users/true/models/Qwen3.8-Flash-Next-MLX-4bit")
AWQ = Path("/Users/true/models/Qwen3.8-Flash-Next-MLX-AWQ-3b2b")
assert BASE.is_dir() and BASE.resolve() != ROOT.resolve()

H, I, K = 2560, 640, 10
QKV = 10240  # 16*128 + 16*128 + 48*128
GDN_Y = 48 * 128  # 6144
IN_O = QKV + 6144 + 48 + 48  # qkv + z + b + a = 16480


def _cat(*rows):
    return np.ascontiguousarray(
        np.concatenate([np.asarray(m, np.float32) for m in rows], axis=0)
    )


def load_layer0():
    """Read-only mmap of the BF16 base. Experts 0..K-1 for stacked width."""
    loader = FlashNextLoader(str(BASE))
    w = loader.layer(0)
    in_proj = _cat(
        w["linear_attn.in_proj_qkv.weight"],
        w["linear_attn.in_proj_z.weight"],
        w["linear_attn.in_proj_b.weight"],
        w["linear_attn.in_proj_a.weight"],
    )
    out_proj = np.asarray(w["linear_attn.out_proj.weight"], np.float32)
    gu = w["mlp.experts.gate_up_proj"]
    dn = w["mlp.experts.down_proj"]
    gu_parts = [np.asarray(gu[e], np.float32) for e in range(K)]
    stacked = np.concatenate(gu_parts, axis=0).reshape(K, 2 * I, H)
    gate = np.ascontiguousarray(stacked[:, :I, :].reshape(K * I, H))
    up = np.ascontiguousarray(stacked[:, I:, :].reshape(K * I, H))
    dn_parts = [np.asarray(dn[e], np.float32) for e in range(K)]
    down = np.ascontiguousarray(np.concatenate(dn_parts, axis=1) / float(K))
    sh = np.asarray(w["mlp.shared_expert.gate_proj.weight"], np.float32)
    su = np.asarray(w["mlp.shared_expert.up_proj.weight"], np.float32)
    sd = np.asarray(w["mlp.shared_expert.down_proj.weight"], np.float32)
    sg = np.asarray(w["mlp.shared_expert_gate.weight"], np.float32).reshape(1, -1)
    mix_d = np.asarray(w["attn_hyper_connection.input_mix_weight_down.weight"], np.float32)
    mix_u = np.asarray(w["attn_hyper_connection.input_mix_weight_up.weight"], np.float32)
    inj = np.asarray(w["attn_hyper_connection.block_inject_weight.weight"], np.float32)
    loader.close()
    return {
        "in_proj": in_proj, "out_proj": out_proj,
        "gate": gate, "up": up, "down": down,
        "sh": sh, "su": su, "sd": sd, "sgate": sg,
        "mix_down": mix_d, "mix_up": mix_u, "inject": inj,
    }


def compile_graph(eng, w, seq: int, *, mixers: bool, tag: str):
    """One input, one evaluate. mixers=True uses hc=4 input [10240]."""
    if mixers:
        in_dim = 10240
        p = Prog(in_dim, seq)
        both = _cat(w["mix_down"], w["inject"])
        p.weight_int8("Wmix", both)
        p.conv("both", "x", "Wmix", both.shape[0])
        p.slice_ch("down", "both", 0, w["mix_down"].shape[0])
        p.sigmoid("dsig", "down", w["mix_down"].shape[0])
        p.mul("dsilu", "down", "dsig", w["mix_down"].shape[0])
        p.weight_int8("Wup", w["mix_up"])
        p.conv("mixed", "dsilu", "Wup", w["mix_up"].shape[0])
        # mix_up is [10240, 320]; we need [2560] for the block. Take mean of 4
        # hc branches via a constexpr 2560x10240 average (ones/4), not a MIL reduce.
        avg = np.zeros((H, 10240), np.float32)
        for b in range(4):
            avg[:, b * H:(b + 1) * H] = 0.25 * np.eye(H, dtype=np.float32)
        p.weight_int8("Wavg", avg)
        p.conv("h", "mixed", "Wavg", H)
        hidden = "h"
    else:
        in_dim = H
        p = Prog(in_dim, seq)
        hidden = "x"

    p.weight_int8("Win", w["in_proj"])
    p.conv("yin", hidden, "Win", w["in_proj"].shape[0])
    p.slice_ch("gdn_proxy", "yin", 0, GDN_Y)
    p.weight_int8("Wout", w["out_proj"])
    p.conv("yattn", "gdn_proxy", "Wout", H)
    _swiglu(p, hidden, w["gate"], w["up"], w["down"], "", "yffn")
    _swiglu(p, hidden, w["sh"], w["su"], w["sd"], "s", "ysh")
    p.weight_int8("Wsg", w["sgate"])
    p.conv("sg", hidden, "Wsg", 1)
    p.out("yattn", H)
    p.out("yffn", H)
    p.out("ysh", H)
    p.out("sg", 1)

    mil, packed = p.mil()
    cap = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        try:
            prog = eng.compile_multiproc(
                mil, {"weight.bin": packed}, in_dim, H, seq,
                raw_weight_files=frozenset({"weight.bin"}))
        except Exception as exc:  # noqa: BLE001
            cap.write(f"exception: {exc!r}\n")
            prog = None
    dt = time.perf_counter() - t0
    if prog is None:
        tail = " | ".join(cap.getvalue().strip().splitlines()[-8:])
        return None, dt, tail, p
    runner = MultiOut(prog, in_dim, seq, p.outputs)
    compiled_s = getattr(prog, "seq_len", seq)
    return runner, dt, f"ok compiled_seq={compiled_s}", p


def time_runner(runner, in_dim, seq, n_warm=3, n_rep=11):
    x = np.random.default_rng(0).standard_normal((in_dim, seq), np.float32).astype(np.float32) * 0.02
    for _ in range(n_warm):
        got = runner.run(x)
        if got is None:
            return None, "evaluate returned None"
    ts = []
    for _ in range(n_rep):
        t = time.perf_counter()
        got = runner.run(x)
        ts.append((time.perf_counter() - t) * 1e3)
        if got is None:
            return None, "evaluate returned None mid-run"
    ts.sort()
    return ts[len(ts) // 2], ts


def try_linear_eval(eng, prog, seq, in_dim):
    x = np.random.default_rng(0).standard_normal((seq, in_dim), np.float32).astype(np.float32) * 0.02
    t0 = time.perf_counter()
    y = eng.evaluate(prog, x)
    dt = (time.perf_counter() - t0) * 1e3
    if y is None:
        return None, dt
    return np.asarray(y).shape, dt


def try_linear_s1(eng, weight, seq):
    cap = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        prog = eng.compile_linear(
            np.ascontiguousarray(weight, np.float32), seq,
            quantized=True, keep_weight_dequant=False)
    dt = time.perf_counter() - t0
    if prog is None:
        tail = " | ".join(cap.getvalue().strip().splitlines()[-6:])
        return None, dt, tail
    return prog, dt, f"ok seq_len={getattr(prog, 'seq_len', '?')} in={getattr(prog, 'input_dim', '?')}"


def main() -> int:
    print("weights (read-only):")
    print(f"  base  {BASE}   (this probe reads layer 0 only)")
    print(f"  q4    {MLX4}   (unused this run)")
    print(f"  awq   {AWQ}    (unused this run; ANE graph is bf16->int8 constexpr)")
    print("loading layer 0 from base (mmap, no writes)", flush=True)
    w = load_layer0()
    print(f"  in_proj {w['in_proj'].shape}  out {w['out_proj'].shape}  "
          f"swiglu gate {w['gate'].shape}  mix_down {w['mix_down'].shape}  "
          f"mix_up {w['mix_up'].shape}  inject {w['inject'].shape}", flush=True)

    eng = AneEngine()
    print("\n=== compile_linear S sweep (in_proj rows, int8) ===", flush=True)
    sample = w["in_proj"][:2560]
    for s in (1, 4, 8, 16, 32):
        prog, dt, msg = try_linear_s1(eng, sample, s)
        extra = ""
        if prog is not None:
            shp, ev = try_linear_eval(eng, prog, s, 2560)
            extra = f"  eval {ev:.2f} ms  y={shp}"
        print(f"  linear S={s:<3}  {dt:.2f}s  {msg}{extra}", flush=True)

    print("\n=== fused layer body (in_proj + out_proj + SwiGLU+shared) ===", flush=True)
    for mixers in (False, True):
        kind = "hc=4 mixers+body" if mixers else "body on mixed[2560]"
        print(f"\n-- {kind} --", flush=True)
        for s in (1, 4, 32):
            tag = f"{'mix' if mixers else 'body'}.S{s}"
            runner, dt, msg, p = compile_graph(eng, w, s, mixers=mixers, tag=tag)
            print(f"  compile S={s:<3}  {dt:.2f}s  {msg}", flush=True)
            if runner is None:
                continue
            in_dim = 10240 if mixers else H
            med, ts = time_runner(runner, in_dim, runner.seq)
            if med is None:
                err = getattr(runner, "err", ts)
                print(f"  eval    S={s:<3}  FAIL {err}", flush=True)
                if s < 32:
                    pad = MultiOut(runner.prog, in_dim, 32, p.outputs)
                    med, ts = time_runner(pad, in_dim, 32)
                    if med is None:
                        print(f"  eval    S={s:<3}  pad32 FAIL {getattr(pad, 'err', ts)}",
                              flush=True)
                        continue
                    print(f"  eval    S={s:<3}  pad32-surface median {med:.2f} ms  "
                          f"48 layers ~ {48 * med:.0f} ms/tok  "
                          f"({1000 / max(48 * med, 1e-6):.1f} tok/s body-only)",
                          flush=True)
                continue
            per48 = 48 * med
            print(f"  eval    S={s:<3}  median {med:.2f} ms  "
                  f"48 layers ~ {per48:.0f} ms/tok  ({1000 / max(per48, 1e-6):.1f} tok/s if this were the whole model)",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
