#!/usr/bin/env python3
"""Can a narrow decode set and a wide prefill set of GDN layers coexist?

Decode wants k=4: the fixed ~1.1 ms a submit is small against three drafts.
Prefill wants the widest graph that compiles, because every slot carries a
real token. Those are different programs, and the note in the exporter says
the ANE tops out near 80 resident models. This counts where it actually
stops.

    ~/.rindi/venvs/coreai/bin/python probes/mil_two_widths.py [narrow] [wide]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_flashnext_coreai import (_load_layer, HC_W, SEQ_DEFAULT,
                                     FlashNextQSADecode)
from flashnext_multitoken_step import MultiTokenStep
from runtime.mil_gdn_backend import MilGdnLayer
from runtime.mil_qsa_backend import MilQsaLayer
from flashnext_mil_qsa_layer import _Ref as _QRef

S = SEQ_DEFAULT
GDN_LAYERS = [i for i in range(48) if i % 4 != 3]
QSA_LAYERS = [i for i in range(48) if i % 4 == 3]


def main() -> None:
    narrow = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    wide = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    rng = np.random.default_rng(3)
    x = np.zeros((1, HC_W, S), np.float16)
    x = np.zeros((1, HC_W, 1, S), np.float16)
    x[0, :, 0, :] = (rng.standard_normal((HC_W, S)) * 0.05).astype(np.float16)
    kept = []
    t0 = time.perf_counter()
    for tag, k in (("decode", narrow), ("prefill", wide)):
        for n, i in enumerate(GDN_LAYERS):
            loader, w = _load_layer(i)
            try:
                lay = MilGdnLayer(i, w, MultiTokenStep(w, 1).eval().half(), k=k)
            except Exception as exc:  # noqa: BLE001
                print(f"  {tag} k={k} L{i} FAILED after {len(kept)} resident "
                      f"programs: {exc}", flush=True)
                return
            finally:
                loader.close()
            kept.append(lay)
            if (n + 1) % 12 == 0:
                print(f"  {tag} k={k}: {n + 1}/{len(GDN_LAYERS)} "
                      f"({len(kept)} resident, {time.perf_counter() - t0:.0f}s)",
                      flush=True)
    # A QSA layer is one program a key rung plus one front program. Decode
    # needs both rungs; prefill only ever sees the widest.
    for tag, k, rungs in (("decode", narrow, [256, 2048]),
                          ("prefill", wide, [2048])):
        for n, i in enumerate(QSA_LAYERS):
            loader, w = _load_layer(i)
            try:
                qd = FlashNextQSADecode(max_s=max(rungs)).eval().half()
                qd.load_from_layer(w)
                kept.append(MilQsaLayer(i, w, _QRef(w), qd, rungs, k=k))
            except Exception as exc:  # noqa: BLE001
                print(f"  qsa {tag} k={k} L{i} FAILED at {len(kept)} layer "
                      f"objects: {exc}", flush=True)
                return
            finally:
                loader.close()
            print(f"    qsa {tag} k={k} {n + 1}/{len(QSA_LAYERS)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"  {len(kept)} layer objects resident in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    for tag, lay in (("first", kept[0]), ("last", kept[-1])):
        t = time.perf_counter()
        lay(x, n=lay.k)
        print(f"  {tag} program (L{lay.layer} k={lay.k}) ran in "
              f"{(time.perf_counter() - t) * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
