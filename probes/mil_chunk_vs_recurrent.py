#!/usr/bin/env python3
"""Compare chunked GDN arithmetic to the token-by-token recurrence.

Isolates the core (no projections). Prints per-slot output error and the
final-state error of each ANE chunk width against an ANE recurrent graph
on identical fp16 inputs. NumPy fp64/fp16 rows show how much of the gap
is the WY equations versus fp16/padding.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "probes"):
    sys.path.insert(0, str(path))

import flashnext_mil_layer as m
from flashnext_mil_chunk import gdn_chunk
from gdn_chunk_reference import chunk
from runtime.q38_ane_engine import _BlobPacker, _iosurface_view


H, C, D = 48, 32, 128


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def recurrent_numpy(q, k, v, gates, beta, state):
    """Token-by-token gated delta, matching MIL gdn_core (pre-RMS)."""
    s = np.array(state, copy=True)
    ys = []
    for t in range(q.shape[-2]):
        g = gates[..., t, None, None]
        b = beta[..., t, None, None]
        kk = k[..., t:t + 1, :]
        vv = v[..., t:t + 1, :]
        qq = q[..., t:t + 1, :]
        s = s * g
        mem = np.sum(s * kk, axis=-1, keepdims=True)
        delta = b * (np.swapaxes(vv, -1, -2) - mem)
        s = s + delta * kk
        y = np.sum(s * qq, axis=-1, keepdims=True)
        ys.append(np.swapaxes(y, -1, -2))
    return np.concatenate(ys, axis=-2), s


def pad_seq(arr, fill, seq_axis, live):
    pad_shape = list(arr.shape)
    pad_shape[seq_axis] = C - live
    pad = np.full(pad_shape, fill, dtype=arr.dtype)
    sl = [slice(None)] * arr.ndim
    sl[seq_axis] = slice(0, live)
    return np.concatenate([arr[tuple(sl)], pad], axis=seq_axis)


def pad_width(q, k, v, gates, beta, live):
    """Neutral pad used by the MIL emitter: gate 1, beta 0, zero q/k/v."""
    return (pad_seq(q, 0, -2, live), pad_seq(k, 0, -2, live),
            pad_seq(v, 0, -2, live), pad_seq(gates, 1, -1, live),
            pad_seq(beta, 0, -1, live))


def make_inputs(rng, case):
    q, k = [rng.normal(size=(1, H, C, D)).astype(np.float32) for _ in range(2)]
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(D)
    v = rng.normal(0, 0.02, size=q.shape).astype(np.float32)
    state = rng.normal(0, 0.02, size=(1, H, D, D)).astype(np.float32)
    gates = rng.uniform(0.1, 1, size=(1, H, C)).astype(np.float32)
    beta = rng.uniform(0, 1, size=gates.shape).astype(np.float32)
    if case == "correlated":
        k[:] = 1 / np.sqrt(D)
        q[:] = 1 / D
        gates[:] = 1
        beta[:] = 1
    return q, k, v, gates, beta, state


def numpy_rows(q, k, v, gates, beta, state):
    y_rec, s_rec = recurrent_numpy(q, k, v, gates, beta, state)
    y_ch, s_ch = chunk(q, k, v, gates, beta, state)
    print("numpy fp64  chunk vs recurrent  "
          f"y={rel(y_ch, y_rec):.8f}  state={rel(s_ch, s_rec):.8f}", flush=True)
    q16, k16, v16 = [a.astype(np.float16).astype(np.float32) for a in (q, k, v)]
    g16, b16, st16 = [a.astype(np.float16).astype(np.float32) for a in (gates, beta, state)]
    y_r16, s_r16 = recurrent_numpy(q16, k16, v16, g16, b16, st16)
    y_c16, s_c16 = chunk(q16, k16, v16, g16, b16, st16)
    print("numpy fp16-cast chunk vs recurrent  "
          f"y={rel(y_c16, y_r16):.8f}  state={rel(s_c16, s_r16):.8f}", flush=True)
    print("numpy fp16-cast chunk vs fp64 recurrent  "
          f"y={rel(y_c16, y_rec):.8f}  state={rel(s_c16, s_rec):.8f}", flush=True)
    for live in (4, 8, 16, 32):
        qp, kp, vp, gp, bp = pad_width(q16, k16, v16, g16, b16, live)
        y_pad, s_pad = chunk(qp, kp, vp, gp, bp, st16)
        y_nat, s_nat = chunk(q16[..., :live, :], k16[..., :live, :],
                             v16[..., :live, :], g16[..., :live],
                             b16[..., :live], st16)
        print(f"numpy fp16 pad {live}->32 vs native-{live}  "
              f"y={rel(y_pad[..., :live, :], y_nat):.8f}  "
              f"state={rel(s_pad, s_nat):.8f}", flush=True)
    return y_rec, s_rec


def compile_core(kind, tile=32, update_scale=64, q_scale=64):
    os.environ["MIL_GDN_CHUNK_TILE"] = str(tile)
    os.environ["MIL_GDN_CHUNK_UPDATE_SCALE"] = str(update_scale)
    os.environ["MIL_GDN_CHUNK_Q_SCALE"] = str(q_scale)
    m.B.clear()
    m.emit('tensor<bool, [4]> mm = const()[name=string("mm"), '
           'val=tensor<bool, [4]>([false,false,false,false])];')
    m.emit('tensor<int32, [4]> pm = const()[name=string("pm"), '
           'val=tensor<int32, [4]>([0,1,3,2])];')
    m.emit('fp16 nho = const()[name=string("nho"), val=fp16(-1)];')
    m.emit('tensor<int32, [1]> ax = const()[name=string("ax"), '
           'val=tensor<int32, [1]>([-1])];')
    m.emit('bool kd = const()[name=string("kd"), val=bool(true)];')

    def prepare(t):
        for name, src in (("qn48", "a_q"), ("kn48", "b_k"), ("vv", "c_v"),
                          ("dec", "d_g"), ("bet", "f_b")):
            width = 1 if name == "bet" else D
            m.sl4(f"g{t}{name}", src, (0, 0, t, 0),
                  (1, H, t + 1, width), (1, H, 1, width))
    m.gdn_prepare = prepare
    m.gdn_finish = lambda t: None
    pack = _BlobPacker()
    offs = {}
    for name, mask in (("lower", np.tril(np.ones((C, C)))),
                       ("strict", np.tril(np.ones((C, C)), -1)),
                       ("eye", np.eye(C))):
        offs["chunk_" + name] = pack.append(mask.astype(np.float16).tobytes()) + 64
    row, col = np.arange(C)[:, None], np.arange(C)[None, :]
    for stage in range(5):
        half = 1 << stage
        mask = ((row // (2 * half) == col // (2 * half))
                & ((row // half) % 2 == 1) & ((col // half) % 2 == 0))
        offs[f"chunk_block{stage}"] = pack.append(mask.astype(np.float16).tobytes()) + 64
    if kind == "recurrent":
        prev = "e_state"
        for t in range(C):
            m.gdn_core(t, prev, f"q_state{t:02d}")
            prev = f"q_state{t:02d}"
    else:
        gdn_chunk(m, C, offs)
    ys = ", ".join(f"g{t}yt" for t in range(C))
    m.emit(f'tensor<fp16, [1, {H}, {C}, {D}]> yout = concat(values=({ys}), '
           f'axis=int32(2), interleave=bool(false))[name=string("yout")];')
    names = ["q_state31", "yout"]
    sig = []
    for name in ("a_q", "b_k", "c_v", "d_g", "e_state", "f_b"):
        shape = (1, H, D, D) if name == "e_state" else (1, H, C, D)
        sig.append(f'tensor<fp16, [{", ".join(map(str, shape))}]> {name}')
    mil = (f"program(1.3)\n{m.E._BUILD_INFO}\n{{\n func main<ios18>("
           + ", ".join(sig) + ") {\n" + "\n".join(m.B)
           + f"\n }} -> ({', '.join(names)});\n}}")
    print(f"compile {kind} tile={tile} update={update_scale} q={q_scale}",
          flush=True)
    prog = m.eng.compile_multiproc(
        mil, {"weight_scale.bin": pack.getvalue()}, H * C, H * C, D,
        raw_weight_files=frozenset(["weight_scale.bin"]))
    if prog is None:
        raise RuntimeError(f"compile failed: {kind} tile={tile}")
    prog.input_elems = [H * C * D] * 4 + [H * D * D, H * C * D]
    # Surfaces bind in alphabetical output order: q_state31, yout.
    prog.output_elems = [H * D * D, H * C * D]
    m.eng._ensure_io(prog)
    return prog


def run(prog, arrays):
    for surf, arr in zip(prog._in_surfs, arrays):
        with _iosurface_view(surf, arr.shape, np.float16) as dst:
            np.copyto(dst, arr)
    for _ in range(4):
        assert m.eng.submit(prog, procedure_index=0)
    shapes = [(1, H, D, D), (1, H, C, D)]
    out = []
    for shape, surf in zip(shapes, prog._out_surfs):
        with _iosurface_view(surf, shape, np.float16) as src:
            out.append(np.array(src, np.float64))
    return out[1], out[0]  # y, state


def report(label, y, state, y_ref, s_ref):
    slots = "  ".join(f"t{t:02d}={rel(y[:, :, t], y_ref[:, :, t]):.6f}"
                      for t in range(C))
    print(f"{label}  state={rel(state, s_ref):.8f}", flush=True)
    print(f"  {slots}", flush=True)


def main():
    rng = np.random.default_rng(20260913)
    case = os.environ.get("MIL_CHUNK_VS_CASE", "random")
    q, k, v, gates, beta, state = make_inputs(rng, case)
    print(f"case={case}", flush=True)
    y_rec, s_rec = numpy_rows(q, k, v, gates, beta, state)
    arrays = [
        q, k, v,
        np.broadcast_to(gates[..., None], q.shape),
        state,
        np.broadcast_to(beta[..., None], q.shape),
    ]
    arrays = [np.ascontiguousarray(x, np.float16) for x in arrays]
    rec = compile_core("recurrent")
    y_ane_r, s_ane_r = run(rec, arrays)
    report("ANE recurrent vs numpy fp64", y_ane_r, s_ane_r, y_rec, s_rec)
    configs = [
        ("chunk", 4, 64, 64),
        ("chunk", 8, 64, 64),
        ("chunk", 16, 64, 64),
        ("chunk", 32, 64, 64),
        ("chunk", 32, 1, 64),
        ("chunk", 32, 64, 1),
        ("chunk", 4, 1, 64),
    ]
    for kind, tile, upd, qs in configs:
        prog = compile_core(kind, tile=tile, update_scale=upd, q_scale=qs)
        y, s = run(prog, arrays)
        label = f"ANE {kind} tile={tile} upd={upd} q={qs}"
        report(f"{label} vs ANE recurrent", y, s, y_ane_r, s_ane_r)
        report(f"{label} vs numpy fp64", y, s, y_rec, s_rec)


if __name__ == "__main__":
    main()
