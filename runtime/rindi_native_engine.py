# SPDX-License-Identifier: Apache-2.0
"""runtime/rindi_native_engine.py - Python ctypes bridge to librindi_native.dylib."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional
import numpy as np

_LIB_PATH = Path(__file__).resolve().parent / "librindi_native.dylib"


class RindiNativeEngine:
    """Zero-overhead native 64-layer ANE chain execution engine."""

    def __init__(self, hidden_dim: int = 5120, seq_len: int = 32):
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lib = ctypes.CDLL(str(_LIB_PATH))

        # Define ctypes signatures
        self.lib.rindi_engine_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self.lib.rindi_engine_create.restype = ctypes.c_void_p

        self.lib.rindi_engine_load_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
        self.lib.rindi_engine_load_layer.restype = ctypes.c_bool

        self.lib.rindi_engine_evaluate_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.rindi_engine_evaluate_step.restype = ctypes.c_bool

        self.lib.rindi_engine_get_last_latency_ms.argtypes = [ctypes.c_void_p]
        self.lib.rindi_engine_get_last_latency_ms.restype = ctypes.c_double

        self.lib.rindi_engine_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rindi_engine_destroy.restype = None

        self.handle = self.lib.rindi_engine_create(hidden_dim, seq_len)
        if not self.handle:
            raise RuntimeError("Failed to create native Rindi engine handle")

    def load_layer(self, layer_idx: int, package_path: str) -> bool:
        """Load a compiled ANE layer package into the native C++ chain."""
        return bool(self.lib.rindi_engine_load_layer(self.handle, layer_idx, package_path.encode("utf-8")))

    def evaluate_step(self, input_fp16: np.ndarray, output_fp16: np.ndarray) -> bool:
        """Run all 64 ANE layers in pure C with zero dispatch overhead."""
        in_ptr = input_fp16.ctypes.data_as(ctypes.c_void_p)
        out_ptr = output_fp16.ctypes.data_as(ctypes.c_void_p)
        return bool(self.lib.rindi_engine_evaluate_step(self.handle, in_ptr, out_ptr))

    def get_last_latency_ms(self) -> float:
        """Get latency of the last 64-layer ANE forward pass in ms."""
        return float(self.lib.rindi_engine_get_last_latency_ms(self.handle))

    def close(self):
        """Free native resources."""
        if self.handle:
            self.lib.rindi_engine_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
