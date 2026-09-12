#!/usr/bin/env python3
"""How wide can one GDN MIL submit go, and what does it cost per token?

Decode wants a narrow graph: a submit costs ~1.13 ms that does not scale with
tokens, plus ~0.146 ms a token, so K=4 is the right shape when only two or
three drafts are accepted. Prefill has no such limit — every slot carries a
real token — so the fixed cost should amortise until the graph stops
compiling. The emitter's sequence width is 32, which caps k there.

    ~/.rindi/venvs/coreai/bin/python probes/mil_wide_prefill.py [k ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_flashnext_coreai import _load_layer, HC_W, SEQ_DEFAULT
from flashnext_multitoken_step import MultiTokenStep
from runtime.mil_gdn_backend import MilGdnLayer

S = SEQ_DEFAULT
REPEAT = 31


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ks", type=int, nargs="*", default=[1,4,8,16,32])
    parser.add_argument("--compare", action="store_true",
                        help="alternate recurrent/chunk single-state graphs")
    parser.add_argument("--repeats", type=int, default=301)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    loader, w = _load_layer(0)
    ref = MultiTokenStep(w, 1).eval().half()
    rng = np.random.default_rng(7)
    x = np.zeros((1, HC_W, 1, S), np.float16)
    x[0, :, 0, :] = (rng.standard_normal((HC_W, S)) * 0.05).astype(np.float16)
    records = []
    print("K mode compile_s wall_ms submit_ms wall_ms/token", flush=True)
    for k in args.ks:
        modes = ["0", "1"] if args.compare else [os.environ.get("MIL_GDN_CHUNK", "0")]
        layers = {}
        for mode in modes:
            os.environ["MIL_GDN_CHUNK"] = mode
            t0 = time.perf_counter()
            lay = MilGdnLayer(0, w, ref, k=k, single_state=(args.compare or
                              os.environ.get("MIL_K_SINGLE_STATE") == "1"))
            layers[mode] = lay
            records.append(dict(k=k, mode=mode, compile_s=time.perf_counter()-t0,
                                wall_samples=[], submit_samples=[]))
        current = records[-len(modes):]
        for _ in range(args.warmup):
            for lay in layers.values(): lay(x,n=k)
        for repeat in range(args.repeats):
            # Reverse order each pair to reduce drift/order bias.
            for rec in current[::1 if repeat % 2 == 0 else -1]:
                lay = layers[rec["mode"]]
                t0=time.perf_counter(); lay(x,n=k)
                rec["wall_samples"].append((time.perf_counter()-t0)*1e3)
                t0=time.perf_counter()
                if not lay._eng.submit(lay._prog,procedure_index=0):
                    raise RuntimeError("submit failed")
                rec["submit_samples"].append((time.perf_counter()-t0)*1e3)
        for rec in current:
            for kind in ["wall", "submit"]:
                rec[kind+"_ms"] = float(np.median(rec[kind+"_samples"]))
            print(f'{k} {rec["mode"]} {rec["compile_s"]:.3f} '
                  f'{rec["wall_ms"]:.6f} {rec["submit_ms"]:.6f} '
                  f'{rec["wall_ms"]/k:.6f}', flush=True)
        layers.clear()
    fits = {}
    for mode in sorted(set(r["mode"] for r in records)):
        rows = [r for r in records if r["mode"]==mode]
        if len(rows)>1:
            fits[mode] = {}
            for kind in ["wall", "submit"]:
                slope,intercept = np.polyfit([r["k"] for r in rows],
                                             [r[kind+"_ms"] for r in rows],1)
                fits[mode][kind] = dict(slope_ms=float(slope),intercept_ms=float(intercept))
            print(f"mode={mode} fit={fits[mode]}",flush=True)
    if args.json:
        args.json.write_text(json.dumps(dict(records=records,fits=fits),indent=2)+"\n")
    loader.close()


if __name__ == "__main__":
    main()
