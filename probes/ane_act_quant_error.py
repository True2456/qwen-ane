#!/usr/bin/env python3
"""How much error does INT8 *activation* quantization add on the ANE?

`ane_w8a8_projection.py` established the speed (2.1-3.7x on real projection
shapes, see docs/W8A8-PROJECTIONS.md). It made no accuracy claim, and
`ane_w8a8_accuracy.py` could not close that gap because it needs two weight
tensors in one program: a program with more than one blob FILE fails
`verifyBundleAtPath: invalid model`, and a packed multi-tensor blob misreads
every tensor after the first.

This probe sidesteps that container blocker completely. The thing under test is
the `quantize`/`dequantize` pair, not the chain, so ONE const weight suffices:

    program A (reference):  x -> conv(W) -> y
    program B (quantized):  x -> conv(W) -> quantize -> dequantize -> y

Both hold exactly one blob const. A's output is the fp16 intermediate the
hardware really produces; B's output is that same intermediate pushed through
int8 and back. `B vs A` therefore isolates activation-quantization error on a
real intermediate distribution, and `A vs fp32` is the control that says
whether the harness is trustworthy at all.

Three questions:

1. Which activation scale minimises the error -- the published global 0.125, a
   max-calibrated scale, or a percentile scale that clips outliers to use more
   of the int8 range?
2. How much worse does it get when the input has outlier channels, which is
   what real LLM activations look like?
3. Does `quantize` accept a PER-CHANNEL scale? If it does, the outlier problem
   goes away and per-tensor calibration stops mattering. ANSWER: yes, but only
   as a RANK-1 scale with an explicit `axis`. See PC_SPELLINGS below.

Real weights: Qwen3.8-Flash-Next layer 0 `in_proj_z` [6144, 2560].

Two notes on reading the output.

* The fp16 control is 2.99e-3 on iid input and *falls* to ~1.6e-3 / ~9.3e-4
  when outlier channels are injected. That is not harness drift. The ANE conv's
  error is constant in ABSOLUTE terms (||err|| = 0.326 / 0.328 / 0.333) while
  the output norm grows 3.3x, so the ratio shrinks. Weight rounding is not
  involved: BF16 -> fp16 on this tensor is a 2.3e-8 relative change (4280 of
  15.7M entries differ, all subnormal), so essentially the whole control error
  is the conv datapath's own noise floor. Only the iid number is comparable to
  the established 2.99e-3, so that is what is gated; the other controls are
  reported with their absolute norms so the constancy is visible.
* The `vs ANE fp16` column, not `vs fp32`, is the activation-quantization
  error. `vs fp32` mixes in that constant fp16 floor, which shrinks as the
  distribution gets hotter, and so flatters the outlier rows.

A methodological point that changes what the outlier rows mean. Scaling INPUT
channels barely changes the shape of the tensor `quantize` actually sees: a
dense projection mixes all 2560 inputs, so the intermediate stays roughly
Gaussian and its max/rms only moves from 15.2 to ~13. That is why per-tensor
error is nearly identical across those three cases. Real LLM activation
outliers live in the tensor being quantized, i.e. in its OUTPUT channels, so
the last case injects them there directly, by adding a component along the
weight rows of chosen output channels. That case, not the input-channel ones,
is the realistic LLM test.

Run with Q38_ANE_REUSE_COMPILED=0. The engine's content-addressed compile cache
keys on the MIL text, and a poisoned artifact from an earlier session will be
served back and quietly invalidate every number below.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

MODEL = Path(os.environ.get("FLASH_NEXT", str(Path.home() / "models/Qwen3.8-Flash-Next")))
SHARD = MODEL / "model-00001-of-00131.safetensors"
WNAME = "model.language_model.layers.0.linear_attn.in_proj_z.weight"

eng = AneEngine()

_PRE = (
    '    string pt = const()[name=string("pt"), val=string("valid")];\n'
    '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '    int32 gr = const()[name=string("gr"), val=int32(1)];'
)

# Which per-channel activation-scale spellings ANECCompile takes, measured on
# macOS 27.0 build 26A428 / M5 Max h17. Only the rank-1 + explicit axis form is
# accepted; everything else is InvalidMILProgram. `probe_pc_spellings()`
# re-derives this table.
PC_SPELLINGS = ("rank1+axis", "rank4", "rank4+axis", "rank1")


def read_safetensors(path: Path, names: list[str]) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k in names:
            meta = hdr[k]
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            if meta["dtype"] == "BF16":
                bits = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                out[k] = bits.view(np.float32).reshape(meta["shape"])
            else:
                out[k] = np.frombuffer(raw, np.float16).reshape(
                    meta["shape"]).astype(np.float32)
        return out


def f16lit(v: float) -> str:
    """MIL fp16 literal. Exponent notation is not worth risking in the parser."""
    v = float(np.float16(v))
    s = f"{v:.10g}"
    return f"{v:.12f}" if ("e" in s or "E" in s) else s


def _qdq_lines(O: int, S: int, mode: str, scale, spelling: str) -> list[str]:
    """The quantize/dequantize pair. `spelling` only matters for per-channel.

    The per-channel scale goes in as an INLINE literal const, not a blob. Only
    one blob file is permitted per program and the weight already owns it; a
    second tensor packed into the same file reads back wrong.
    """
    out = ['    string q_dtype = const()[name=string("q_dtype"), '
           'val=string("int8")];']
    if mode == "scalar":
        out.append(f'    fp16 qs = const()[name=string("qs"), '
                   f'val=fp16({f16lit(scale)})];')
        axis = ""
    else:
        lit = ",".join(f16lit(v) for v in np.asarray(scale).reshape(-1))
        shape = f"[{O}]" if spelling.startswith("rank1") else f"[1, {O}, 1, 1]"
        out.append(f'    tensor<fp16, {shape}> qs = const()[name=string("qs"), '
                   f'val=tensor<fp16, {shape}>([{lit}])];')
        if spelling.endswith("axis"):
            out.append('    int32 ax = const()[name=string("ax"), '
                       'val=int32(1)];')
            axis = "axis=ax, "
        else:
            axis = ""
    out.append(f'    tensor<int8, [1, {O}, 1, {S}]> q = quantize({axis}input=c, '
               f'output_dtype=q_dtype, scale=qs)[name=string("q")];')
    out.append(f'    tensor<fp16, [1, {O}, 1, {S}]> y = dequantize({axis}input=q, '
               f'scale=qs)[name=string("y")];')
    return out


def build(W: np.ndarray, S: int, mode: str, scale=None,
          spelling: str = "rank1+axis") -> tuple:
    """One const weight, optionally one quantize/dequantize pair after the conv.

    mode "ref"     -> x -> conv -> y
    mode "scalar"  -> ... -> quantize(scalar scale) -> dequantize -> y
    mode "chan"    -> ... -> quantize(vector scale) -> dequantize -> y
    """
    O, C = W.shape
    body = [
        f'    tensor<fp16, [1, {O}, 1, {S}]> c = conv(dilations=dl, groups=gr, '
        f'pad=pd, pad_type=pt, strides=st, weight=W, x=x)[name=string("c")];'
    ]
    if mode == "ref":
        body.append(f'    tensor<fp16, [1, {O}, 1, {S}]> y = '
                    f'identity(x=c)[name=string("y")];')
    else:
        body += _qdq_lines(O, S, mode, scale, spelling)

    mil = (
        f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
        f"  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{\n"
        f'    tensor<fp16, [{O}, {C}, 1, 1]> W = const()[name=string("W"), '
        f'val=tensor<fp16, [{O}, {C}, 1, 1]>(BLOBFILE('
        f'path=string("@model_path/weights/w.bin"), offset=uint64(64)))];\n'
        f"{_PRE}\n" + "\n".join(body) + "\n  } -> (y);\n}\n"
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(
                mil, {"w.bin": W.astype(np.float16).tobytes()}, C, O, S)
        except Exception as exc:  # noqa: BLE001
            p = None
            print(f"exception: {exc}")
    return p, buf.getvalue().strip()


def run(p, x: np.ndarray, O: int, S: int) -> np.ndarray:
    eng._ensure_io(p)
    with _iosurface_view(p._in_surf, x.shape, np.float16) as dst:
        np.copyto(dst, x.astype(np.float16))
    eng.submit(p, procedure_index=0)
    with _iosurface_view(p._out_surf, (O, S), np.float16) as o:
        return np.array(o, np.float32)


def timed(p, n: int = 5) -> float:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        eng.submit(p, procedure_index=0)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))


def sim_qdq(mid: np.ndarray, scale) -> np.ndarray:
    """numpy model of the op pair, as a cross-check on what the ANE did.

    If the hardware and this disagree, the pair is not doing what the spelling
    says -- e.g. the compiler folded quantize/dequantize away, in which case
    the measured error would collapse onto the reference arm's.
    """
    s = np.asarray(scale, np.float32)
    if s.ndim:
        s = s.reshape(-1, 1)
    return (np.clip(np.rint(mid / s), -128, 127) * s
            ).astype(np.float16).astype(np.float32)


def scale_table(mid: np.ndarray) -> dict:
    a = np.abs(mid)
    return {
        "global 0.125": 0.125,
        "calib max/127": float(a.max() / 127.0),
        "pct 99.99": float(np.percentile(a, 99.99) / 127.0),
        "pct 99.9": float(np.percentile(a, 99.9) / 127.0),
        "pct 99": float(np.percentile(a, 99.0) / 127.0),
        "pct 95": float(np.percentile(a, 95.0) / 127.0),
    }


def best_scalar_alpha(mid: np.ndarray) -> tuple[float, float, float]:
    """Finest scalar-scale search, done in numpy because the sim is faithful.

    The hardware arms above agree with sim_qdq to three significant figures at
    every scale tested, so searching on hardware would spend a compile per
    point to learn nothing. Returns (alpha, scale, simulated rel err) where
    scale = alpha * max|mid| / 127; alpha < 1 clips outliers to buy resolution.
    """
    amax = float(np.abs(mid).max())
    best = (1.0, amax / 127.0, float("inf"))
    for alpha in np.geomspace(0.05, 1.0, 61):
        sc = alpha * amax / 127.0
        err = rel_l2(sim_qdq(mid, sc), mid)
        if err < best[2]:
            best = (float(alpha), sc, err)
    return best


def heavier_tail_extrapolation(mid: np.ndarray, discrepancy: float) -> None:
    """What per-tensor int8 costs if real intermediates are heavier-tailed.

    Everything measured above sits at max/rms 13-15, because a dense 2560->6144
    projection averages 2560 inputs and the central limit theorem flattens
    whatever structure the input had. Real LLM intermediates are reported far
    heavier-tailed than that, from trained structure this harness has no way to
    reproduce -- there is no real layer-0 activation here, only synthetic input.

    So this section is EXTRAPOLATION, not measurement. It scales a subset of
    the measured intermediate's output channels to manufacture the tail, and
    evaluates in numpy. That is defensible only because the simulator tracked
    the hardware to within `discrepancy` relative over every arm above; it is
    still a simulation and is labelled as one.
    """
    print(f"\n=== EXTRAPOLATION (numpy only; sim reproduced every measured "
          f"error above to {discrepancy:.1e} absolute)")
    print("    per-tensor int8 error if the intermediate had a heavier tail")
    print(f"    {'gain on 1% of chans':>20} {'max/rms':>9} "
          f"{'per-tensor best':>16} {'per-channel max':>16}")
    O = mid.shape[0]
    k = max(1, O // 100)
    picks = np.random.default_rng(11).choice(O, k, replace=False)
    for g in (1.0, 5.0, 20.0, 100.0):
        m = mid.copy()
        m[picks] *= g
        a = np.abs(m)
        _, _, per_tensor = best_scalar_alpha(m)
        pc = np.maximum(a.max(axis=1) / 127.0, 6e-8).astype(np.float32)
        per_chan = rel_l2(sim_qdq(m, pc), m)
        print(f"    {'x' + format(g, '.0f'):>20} "
              f"{a.max()/max(m.std(),1e-9):>9.1f} "
              f"{per_tensor:>16.4e} {per_chan:>16.4e}")


def probe_pc_spellings(W: np.ndarray, S: int, O: int) -> None:
    """Which per-channel scale spellings does ANECCompile accept?

    Compiled at the real shape so acceptance is not a small-shape artifact.
    """
    print("\n=== does quantize accept a per-CHANNEL activation scale?")
    vec = np.full(O, 0.01, np.float32)
    for sp in PC_SPELLINGS:
        shape = f"[{O}]" if sp.startswith("rank1") else f"[1,{O},1,1]"
        ax = "axis=1" if sp.endswith("axis") else "no axis"
        p, tail = build(W, S, "chan", vec, spelling=sp)
        if p is not None:
            print(f"    scale {shape:>12}, {ax:>7}: ACCEPTED")
            del p
            continue
        line = next((l.strip() for l in tail.splitlines()
                     if "InvalidMILProgram" in l or "FAILED" in l), tail[:120])
        print(f"    scale {shape:>12}, {ax:>7}: REJECTED -- "
              f"{line[:100]}")


def main() -> None:
    S = int(os.environ.get("ACT_S", "256"))
    if os.environ.get("Q38_ANE_REUSE_COMPILED") != "0":
        print("WARNING: set Q38_ANE_REUSE_COMPILED=0; cached artifacts mislead\n")
    if not SHARD.exists():
        print(f"missing {SHARD}")
        return

    W = read_safetensors(SHARD, [WNAME])[WNAME]
    O, C = W.shape
    # Stated so the control error cannot be blamed on weight rounding.
    wf16 = W.astype(np.float16).astype(np.float32)
    print(f"real layer-0 in_proj_z {W.shape}, S={S}, |W|max={np.abs(W).max():.4f}, "
          f"BF16->fp16 weight rounding {rel_l2(wf16, W):.1e} rel "
          f"({int((wf16 != W).sum())}/{W.size} entries, all subnormal)")

    # Feed exactly what the reference sees: round the input to fp16 up front so
    # input rounding is not silently charged to the quantizer. `cal` is an
    # independent draw of the same distribution, used for held-out calibration.
    rng = np.random.default_rng(0)

    def draw(k: int, gen) -> np.ndarray:
        z = (gen.standard_normal((C, S)).astype(np.float32) * 0.1)
        if k:
            z[gen.choice(C, k, replace=False)] *= 20.0
        return z.astype(np.float16).astype(np.float32)

    def draw_out_outlier(picks: np.ndarray, gen, boost: float = 20.0) -> np.ndarray:
        """Outliers in the tensor being QUANTIZED, not in the input.

        A dense projection averages 2560 inputs, so input-channel outliers do
        not survive into the intermediate as outliers. To put them where
        `quantize` sees them, add a random multiple of W[o, :] to the input for
        the chosen output channels: that lands on channel o as
        ||W[o]||^2 * coef, while the cross terms W[o'] . W[o] stay small.

        `picks` is passed in, not drawn here, so a held-out calibration draw
        has the SAME outlier channels and only fresh noise -- which is the real
        situation, since which channels blow up is a property of the weights.
        """
        rows = W[picks]                                     # [k, C]
        rn2 = np.maximum((rows ** 2).sum(1), 1e-12)          # [k]
        target = boost * 0.0870                              # ~boost x mid rms
        coef = (target / rn2)[:, None] * gen.standard_normal((len(picks), S))
        z = draw(0, gen) + (rows.T @ coef).astype(np.float32)
        return z.astype(np.float16).astype(np.float32)

    cases = {}
    for k in (0, 16, 64):
        label = "iid gaussian*0.1" if k == 0 else f"{k}/{C} in-channels x20"
        cases[label] = (draw(k, np.random.default_rng(0)),
                        draw(k, np.random.default_rng(1234)))
    picks = np.random.default_rng(7).choice(O, 16, replace=False)
    cases[f"16/{O} OUT-channels x20"] = (
        draw_out_outlier(picks, np.random.default_rng(0)),
        draw_out_outlier(picks, np.random.default_rng(1234)))

    ref_prog, tail = build(W, S, "ref", None)
    if ref_prog is None:
        print(f"reference program REJECTED:\n{tail}")
        return
    ref_ms = timed(ref_prog)

    probe_pc_spellings(W, S, O)

    best: dict[str, tuple] = {}
    disc: list[tuple[str, float]] = []
    iid_mid = None
    for name, (x, xcal) in cases.items():
        ref32 = W @ x                      # fp32 numpy truth
        mid = run(ref_prog, x, O, S)       # what the ANE really produced
        ctrl = rel_l2(mid, ref32)
        abs_err = float(np.linalg.norm(mid - ref32))
        a = np.abs(mid)
        perchan = a.max(axis=1)
        # Held-out intermediate, for calibration that has not seen the test data.
        mid_cal = run(ref_prog, xcal, O, S)

        print(f"\n=== {name}")
        print(f"    intermediate |max|={a.max():.4f} rms={mid.std():.4f} "
              f"max/rms={a.max()/max(mid.std(),1e-9):.1f}")
        print(f"    per-out-channel |max|: min={perchan.min():.4f} "
              f"med={np.median(perchan):.4f} max={perchan.max():.4f} "
              f"(spread {perchan.max()/max(perchan.min(),1e-9):.1f}x)")
        gate = "" if name.startswith("iid") else "  (not gated, see module docstring)"
        ok = "OK, near 3e-3" if 2e-3 <= ctrl <= 5e-3 else "OFF"
        print(f"    CONTROL x -> conv(W) -> y   vs fp32: {ctrl:.4e} "
              f"(abs {abs_err:.3f}, ||ref|| {np.linalg.norm(ref32):.1f})  "
              f"{ref_ms:.3f} ms  {ok}{gate}")
        if name.startswith("iid") and not 2e-3 <= ctrl <= 5e-3:
            print("    !! iid control is not near 3e-3: the harness is wrong "
                  "and nothing below means anything")
            return

        print(f"    {'act scale':>18} {'value':>10} {'vs fp32':>11} "
              f"{'vs ANE fp16':>12} {'numpy sim':>11} {'ms':>7}")
        rows = []

        def arm(label, mode, sc, sim_sc=None):
            p, tail = build(W, S, mode, sc)
            shown = f"{sc:>10.5f}" if mode == "scalar" else f"{'vector':>10}"
            if p is None:
                last = tail.splitlines()[-1][:44] if tail else "?"
                print(f"    {label:>18} {shown}   REJECTED  {last}")
                return None
            got = run(p, x, O, S)
            e32, emid = rel_l2(got, ref32), rel_l2(got, mid)
            simmed = sim_qdq(mid, sim_sc if sim_sc is not None else sc)
            esim = rel_l2(simmed, mid)
            # How closely the numpy model of quantize/dequantize reproduces the
            # hardware, in the quantity actually being reported. Collected so
            # the extrapolation section can say how far the simulator has
            # earned the right to be trusted. Point-wise hw-vs-sim distance is
            # the wrong statistic: at a degenerate scale like 0.125 both
            # outputs are near-destroyed and agree on the error to 5 digits
            # while differing 2.4e-2 from each other.
            disc.append((label, abs(emid - esim)))
            ms = timed(p)
            del p
            print(f"    {label:>18} {shown} {e32:>11.4e} {emid:>12.4e} "
                  f"{esim:>11.4e} {ms:>7.3f}")
            rows.append((label, e32, emid))
            return got

        for label, sc in scale_table(mid).items():
            arm(label, "scalar", sc)

        # Finest scalar point, found in numpy (validated against the arms above)
        # and then confirmed on hardware.
        alpha, sc_opt, sim_opt = best_scalar_alpha(mid)
        print(f"    -- numpy-optimal scalar clip alpha={alpha:.3f} "
              f"(scale {sc_opt:.5f}, sim {sim_opt:.4e}); confirming on ANE:")
        arm(f"opt scalar a={alpha:.2f}", "scalar", sc_opt)

        # Per-channel, the question that decides whether calibration matters.
        for label, vec in (("per-chan max/127", perchan / 127.0),
                           ("per-chan p99.9/127",
                            np.percentile(a, 99.9, axis=1) / 127.0)):
            vec = np.maximum(vec, 6e-8).astype(np.float32)
            got = arm(label, "chan", vec)
            if got is None:
                continue
            # A silent scalar fallback would look like per-channel until
            # compared against BOTH simulations.
            d_pc = rel_l2(got, sim_qdq(mid, vec))
            d_sc = rel_l2(got, sim_qdq(mid, float(a.max() / 127.0)))
            verdict = "per-channel confirmed" if d_pc < d_sc else "!! LOOKS SCALAR"
            print(f"      cross-check vs numpy: per-chan sim {d_pc:.4e}, "
                  f"scalar sim {d_sc:.4e} -> {verdict}")

        # Held-out calibration: scales from an independent draw. This is the
        # only deployable number; everything above is oracle-calibrated.
        #
        # A per-channel max taken on one draw UNDERestimates the next draw's max
        # for about half the channels, and an underestimated scale clips the
        # very largest values, which is the expensive kind of error. So sweep a
        # safety margin on top of the held-out scale and report the optimum.
        acal = np.abs(mid_cal)
        a_h, sc_h, _ = best_scalar_alpha(mid_cal)
        pc_cal = np.maximum(acal.max(axis=1) / 127.0, 6e-8).astype(np.float32)
        margins = (1.0, 1.25, 1.5, 2.0, 3.0)
        sims = [(m, rel_l2(sim_qdq(mid, pc_cal * m), mid)) for m in margins]
        m_best = min(sims, key=lambda t: t[1])[0]
        print("    -- held-out calibration (scales from an independent draw); "
              "per-chan margin sim: "
              + ", ".join(f"x{m}={e:.3e}" for m, e in sims))
        arm(f"heldout scalar a={a_h:.2f}", "scalar", sc_h)
        arm("heldout per-chan max", "chan", pc_cal)
        if m_best != 1.0:
            arm(f"heldout per-chan x{m_best}", "chan", pc_cal * m_best)

        if rows:
            best[name] = min(rows, key=lambda r: r[2])
        if iid_mid is None:
            iid_mid = mid

    print("\n=== best activation scale per distribution (by error vs ANE fp16)")
    for name, (label, e32, emid) in best.items():
        print(f"    {name:>26}  {label:<22}  vs fp32 {e32:.4e}  "
              f"vs ANE fp16 {emid:.4e}")

    worst = max(disc, key=lambda t: t[1])
    print(f"\n    numpy quantize/dequantize model vs hardware: the reported "
          f"error agrees to within {worst[1]:.1e} absolute over {len(disc)} "
          f"arms (worst at '{worst[0]}')")
    if iid_mid is not None:
        heavier_tail_extrapolation(iid_mid, worst[1])


if __name__ == "__main__":
    main()
