#!/usr/bin/env python3
"""Can one program hold both unroll widths, so the weights are baked once?

Decode wants a k=4 graph and prefill wants k=32. Today those are two compiled
programs, and a program carries its own copy of the layer's baked weights.
`compile_multiproc` already puts many procedures in one program for the routed
experts, so the question is whether two procedures can differ in unroll depth
and in how many prefix states they export.

    ~/.rindi/venvs/coreai/bin/python probes/mil_two_proc_widths.py [narrow] [wide]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_flashnext_coreai import (_load_layer, H, HC, HC_W, HV, DV, DK,
                                     QKV, SEQ_DEFAULT)
from flashnext_multitoken_step import MultiTokenStep
import flashnext_mil_layer as ML
from runtime.q38_ane_engine import _iosurface_view

S = SEQ_DEFAULT


def capture(w, ref, k, single, slots=None):
    """Build the MIL text and weight blobs for one width, without compiling."""
    grabbed = {}
    real = ML.eng.compile_multiproc

    def fake(mil_text, files, *a, **kw):
        grabbed["mil"] = mil_text
        grabbed["files"] = files
        return None

    ML.eng.compile_multiproc = fake
    ML.K[0] = k
    ML.SINGLE_STATE[0] = single
    ML.EXPORT_SLOTS[0] = slots
    try:
        ML.build_layer(w, ref)
        grabbed["slots"] = ML.state_slots()
    finally:
        ML.eng.compile_multiproc = real
        ML.SINGLE_STATE[0] = False
        ML.EXPORT_SLOTS[0] = None
    return grabbed


def body(text: str, name: str) -> str:
    i = text.index("  func main<ios18>")
    j = text.rstrip().rindex("}")
    return text[i:j].replace("func main<ios18>", f"func {name}<ios18>", 1)


def main() -> None:
    narrow = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    wide = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    loader, w = _load_layer(0)
    ref = MultiTokenStep(w, 1).eval().half()
    ML.LAYER[0] = 0

    a = capture(w, ref, narrow, False)
    # The wide procedure exports only its end-of-chunk state; the request
    # hands it the surfaces it writes rather than the front of the list.
    b = capture(w, ref, wide, True)
    conn = ref.gdn
    param = np.concatenate([
        np.repeat(conn.gamma.detach().float().numpy().reshape(HV, 1), DK, 1),
        np.repeat(conn.dt.detach().float().numpy().reshape(HV, 1), DK, 1),
        conn.tail.norm_w.detach().float().numpy().reshape(HV, DV)], axis=0)
    hcn = np.concatenate([ref.attn.hc_n.detach().float().numpy().reshape(320, 32),
                          ref.mlp.hc_n.detach().float().numpy().reshape(320, 32)],
                         axis=0)
    loader.close()

    same = all(a["files"][k2][:len(a["files"][k2])] ==
               b["files"][k2][:len(a["files"][k2])] for k2 in a["files"])
    print(f"  narrow blobs are a prefix of the wide ones: {same}")
    for k2 in sorted(a["files"]):
        print(f"    {k2}: narrow {len(a['files'][k2]) / 1e6:.1f} MB, "
              f"wide {len(b['files'][k2]) / 1e6:.1f} MB")

    text = (f"program(1.3)\n{ML.E._BUILD_INFO}\n{{\n"
            + body(a["mil"], "procedure000")
            + body(b["mil"], "procedure001")
            + "}\n")
    n_out = max(len(a["slots"]), len(b["slots"]))
    print(f"  procedure000 k={narrow} exports {len(a['slots'])} states, "
          f"procedure001 k={wide} exports {len(b['slots'])}")

    t0 = time.perf_counter()
    prog = ML.eng.compile_multiproc(text, b["files"], HC_W, H, S,
                                    raw_weight_files=frozenset(b["files"]))
    if prog is None:
        print("  two-procedure compile FAILED")
        return
    prog.conv_out_width = ML.conv_cache_width()
    prog.input_elems = [HC_W * S, 3 * QKV * S, 3 * HV * DK, 640 * 32,
                        HV * DV * DK]
    prog.output_elems = ([HV * DV * DK] * n_out
                         + [H * S, H * S, HC_W * S, HC * S,
                            QKV * prog.conv_out_width])
    n_a, n_b = len(a["slots"]), len(b["slots"])
    tail = list(range(n_out, n_out + 5))
    prog.proc_out_map = {0: list(range(n_a)) + tail,
                         1: list(range(n_out - n_b, n_out)) + tail}
    if not ML.eng._ensure_io(prog):
        print("  IOSurface allocation FAILED")
        return
    print(f"  compiled and loaded in {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(12)
    x = np.zeros((HC_W, S), np.float16)
    x[:, :] = (rng.standard_normal((HC_W, S)) * 0.05).astype(np.float16)
    conv = (rng.standard_normal((3 * QKV,)) * 0.02).astype(np.float16)
    for idx, val in ((0, x), (1, np.zeros((3 * QKV, S), np.float16))):
        with _iosurface_view(prog._in_surfs[idx], val.shape, np.float16) as d:
            np.copyto(d, val)
    with _iosurface_view(prog._in_surfs[1], (3 * QKV, S), np.float16) as d:
        d[:, 0] = conv
    with _iosurface_view(prog._in_surfs[4], (HV, DV, DK), np.float16) as d:
        d[:] = 0
    for idx, val in ((2, param.astype(np.float16)), (3, hcn.astype(np.float16))):
        with _iosurface_view(prog._in_surfs[idx], val.shape, np.float16) as d:
            np.copyto(d, val)

    def med(fn, n=31):
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return sorted(ts)[len(ts) // 2] * 1e3

    for proc, k in ((0, narrow), (1, wide)):
        if not ML.eng.submit(prog, procedure_index=proc):
            print(f"  procedure{proc:03d} submit FAILED")
            continue
        with _iosurface_view(prog._out_surfs[n_out], (H, S), np.float16) as o:
            mixed = np.array(o[:, :4], np.float32)
        ms = med(lambda: ML.eng.submit(prog, procedure_index=proc))
        print(f"  procedure{proc:03d} k={k:<2} {ms:.3f} ms steady, "
              f"{ms / k:.3f} ms a token, "
              f"mixed[0,:4] {np.array2string(mixed[0], precision=4)}")
    both = med(lambda: (ML.eng.submit(prog, 0), ML.eng.submit(prog, 1)))
    print(f"  alternating both procedures {both:.3f} ms a pair "
          f"(the request is rebuilt on every switch)")


if __name__ == "__main__":
    main()
