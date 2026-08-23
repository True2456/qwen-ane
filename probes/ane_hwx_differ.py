#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ane_hwx_differ.py - Phase 3: Bitfield Mapping & Model Descriptor Analysis."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
import pprint
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (
    AneEngine,
    _desc,
    _msg,
    _objc,
    _sel,
    _cls,
)
from probes.ane_gdn_conv1d import run as compile_conv1d


def analyze_model_descriptors():
    print("=" * 70)
    print("  PHASE 3: ANE MODEL DESCRIPTOR & BITFIELD MAPPING (M5 Max)")
    print("=" * 70)

    # Compile micro models across different dimensions
    models_to_test = [
        ("C64_K4_S32", 64, 4, 32),
        ("C10240_K4_S32", 10240, 4, 32),
    ]

    for tag, C, K, S in models_to_test:
        print(f"\n{'='*30} [{tag}] {'='*30}")
        ret = compile_conv1d(C, "none")
        assert isinstance(ret, tuple) and ret[0] and ret[1] is not None
        ok, prog = ret

        ane_model = _msg(prog.model, "model")
        prog_handle = _msg(ane_model, "programHandle", restype=ctypes.c_ulonglong)
        inter_handle = _msg(ane_model, "intermediateBufferHandle", restype=ctypes.c_ulonglong)
        q_depth = _msg(ane_model, "queueDepth", restype=ctypes.c_uint)
        perf_mask = _msg(ane_model, "perfStatsMask", restype=ctypes.c_uint)

        print(f"  Program Handle:             0x{prog_handle:016x}")
        print(f"  Intermediate Buffer Handle: 0x{inter_handle:016x}")
        print(f"  Queue Depth:                {q_depth}")
        print(f"  Perf Stats Mask:            0x{perf_mask:x}")

        # Extract modelAttributes
        attrs = _msg(ane_model, "modelAttributes")
        desc_str = _desc(attrs)
        print("\n  [Model Attributes Dictionary]:")
        for line in desc_str.splitlines():
            print("   ", line)

        # Inspect procedure symbols and mappings
        sym_indices_sel = _sel("symbolIndicesForProcedureIndex:indexArrayKey:")
        in_syms = _msg(ane_model, "inputSymbolIndicesForProcedureIndex:", 0)
        out_syms = _msg(ane_model, "outputSymbolIndicesForProcedureIndex:", 0)
        print(f"\n  Input Symbol Indices (Proc 0):  {_desc(in_syms)}")
        print(f"  Output Symbol Indices (Proc 0): {_desc(out_syms)}")

    print("\n" + "=" * 70)
    print("  PHASE 3 COMPLETE: Descriptor Structure & Attributes Mapped")
    print("=" * 70)


if __name__ == "__main__":
    analyze_model_descriptors()
