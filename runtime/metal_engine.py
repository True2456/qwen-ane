# SPDX-License-Identifier: Apache-2.0
"""metal_engine.py - Python ctypes bridge for the high-performance Metal C engine."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DYLIB_PATH = _REPO_ROOT / "runtime" / "libmetal_engine.dylib"

_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not _DYLIB_PATH.exists():
            raise FileNotFoundError(
                f"Metal engine dylib not found at {_DYLIB_PATH}. Please compile runtime/metal_engine.m first."
            )
        _lib = ctypes.CDLL(str(_DYLIB_PATH))
        _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib: ctypes.CDLL) -> None:
    lib.metal_context_create.restype = ctypes.c_void_p
    lib.metal_context_create.argtypes = []

    lib.metal_context_destroy.restype = None
    lib.metal_context_destroy.argtypes = [ctypes.c_void_p]

    lib.metal_get_device_name.restype = ctypes.c_char_p
    lib.metal_get_device_name.argtypes = [ctypes.c_void_p]

    lib.metal_create_iosurface.restype = ctypes.c_void_p
    lib.metal_create_iosurface.argtypes = [ctypes.c_size_t]

    lib.metal_buffer_from_iosurface.restype = ctypes.c_void_p
    lib.metal_buffer_from_iosurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lib.metal_buffer_create.restype = ctypes.c_void_p
    lib.metal_buffer_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    lib.metal_iosurface_get_base_address.restype = ctypes.c_void_p
    lib.metal_iosurface_get_base_address.argtypes = [ctypes.c_void_p]

    lib.metal_iosurface_lock.restype = None
    lib.metal_iosurface_lock.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    lib.metal_iosurface_unlock.restype = None
    lib.metal_iosurface_unlock.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    lib.metal_buffer_release.restype = None
    lib.metal_buffer_release.argtypes = [ctypes.c_void_p]

    lib.metal_buffer_get_contents.restype = ctypes.c_void_p
    lib.metal_buffer_get_contents.argtypes = [ctypes.c_void_p]

    lib.metal_buffer_get_length.restype = ctypes.c_size_t
    lib.metal_buffer_get_length.argtypes = [ctypes.c_void_p]

    lib.metal_shared_event_create.restype = ctypes.c_void_p
    lib.metal_shared_event_create.argtypes = [ctypes.c_void_p]

    lib.metal_shared_event_release.restype = None
    lib.metal_shared_event_release.argtypes = [ctypes.c_void_p]

    lib.metal_shared_event_get_value.restype = ctypes.c_uint64
    lib.metal_shared_event_get_value.argtypes = [ctypes.c_void_p]

    lib.metal_shared_event_set_value.restype = None
    lib.metal_shared_event_set_value.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

    lib.metal_load_library_source.restype = ctypes.c_bool
    lib.metal_load_library_source.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]

    lib.metal_get_pipeline.restype = ctypes.c_void_p
    lib.metal_get_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    lib.metal_command_buffer_create.restype = ctypes.c_void_p
    lib.metal_command_buffer_create.argtypes = [ctypes.c_void_p]

    lib.metal_encode_signal_event.restype = None
    lib.metal_encode_signal_event.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]

    lib.metal_encode_wait_event.restype = None
    lib.metal_encode_wait_event.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]

    lib.metal_command_buffer_commit.restype = None
    lib.metal_command_buffer_commit.argtypes = [ctypes.c_void_p]

    lib.metal_command_buffer_wait.restype = None
    lib.metal_command_buffer_wait.argtypes = [ctypes.c_void_p]

    lib.metal_command_buffer_is_completed.restype = ctypes.c_bool
    lib.metal_command_buffer_is_completed.argtypes = [ctypes.c_void_p]

    lib.metal_dispatch_rmsnorm_fp16.restype = None
    lib.metal_dispatch_rmsnorm_fp16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]

    lib.metal_dispatch_moe_gather_fp16.restype = None
    lib.metal_dispatch_moe_gather_fp16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.metal_dispatch_moe_scatter_fp16.restype = None
    lib.metal_dispatch_moe_scatter_fp16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.metal_dispatch_layout_transform_fp16.restype = None
    lib.metal_dispatch_layout_transform_fp16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.metal_dispatch_gemm_fp16.restype = None
    lib.metal_dispatch_gemm_fp16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.metal_dispatch_argmax_fp16.restype = None
    lib.metal_dispatch_argmax_fp16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]


class MetalBuffer:
    """Zero-copy Metal buffer wrapper backed by Unified Memory / IOSurface."""

    def __init__(self, ctx_handle: int, buf_handle: int, iosurface_ref: Optional[int] = None):
        self._ctx = ctx_handle
        self._handle = buf_handle
        self._iosurface = iosurface_ref
        self._lib = _get_lib()

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def iosurface(self) -> Optional[int]:
        return self._iosurface

    @property
    def length(self) -> int:
        return self._lib.metal_buffer_get_length(self._handle)

    def numpy_view(self, shape: tuple[int, ...], dtype=np.float16) -> np.ndarray:
        """Create a zero-copy numpy array mapped over the buffer's unified memory."""
        if self._iosurface:
            ptr = self._lib.metal_iosurface_get_base_address(self._iosurface)
        else:
            ptr = self._lib.metal_buffer_get_contents(self._handle)
        if not ptr:
            raise RuntimeError("Cannot get buffer contents pointer")
        itemsize = np.dtype(dtype).itemsize
        total_items = int(np.prod(shape))
        if total_items * itemsize > self.length:
            raise ValueError(f"View size {total_items * itemsize} exceeds buffer length {self.length}")
        
        # Build numpy array from memory buffer directly
        array_type = ctypes.c_char * (total_items * itemsize)
        raw_buf = array_type.from_address(ptr)
        return np.frombuffer(raw_buf, dtype=dtype).reshape(shape)

    def __del__(self):
        if self._handle:
            self._lib.metal_buffer_release(self._handle)
            self._handle = 0


class MetalSharedEvent:
    """Hardware synchronization event across GPU and ANE."""

    def __init__(self, ctx_handle: int):
        self._lib = _get_lib()
        self._handle = self._lib.metal_shared_event_create(ctx_handle)
        if not self._handle:
            raise RuntimeError("Failed to create MTLSharedEvent")

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def value(self) -> int:
        return self._lib.metal_shared_event_get_value(self._handle)

    @value.setter
    def value(self, val: int) -> None:
        self._lib.metal_shared_event_set_value(self._handle, val)

    def __del__(self):
        if self._handle:
            self._lib.metal_shared_event_release(self._handle)
            self._handle = 0


class MetalEngine:
    """High-performance Metal runtime controller."""

    def __init__(self):
        self._lib = _get_lib()
        self._ctx = self._lib.metal_context_create()
        if not self._ctx:
            raise RuntimeError("Failed to initialize Metal Context (no Metal device found)")
        self.device_name = self._lib.metal_get_device_name(self._ctx).decode("utf-8")

    def create_iosurface_buffer(self, nbytes: int) -> MetalBuffer:
        """Allocate an IOSurface and bind it immediately to an MTLBuffer (zero-copy)."""
        surface = self._lib.metal_create_iosurface(nbytes)
        if not surface:
            raise RuntimeError(f"Failed to create IOSurface of {nbytes} bytes")
        buf = self._lib.metal_buffer_from_iosurface(self._ctx, surface)
        if not buf:
            raise RuntimeError("Failed to bind IOSurface to MTLBuffer")
        return MetalBuffer(self._ctx, buf, surface)

    def buffer_from_existing_iosurface(self, surface_ref: int) -> MetalBuffer:
        """Wrap an existing IOSurfaceRef (e.g. from ANE) in a zero-copy MTLBuffer."""
        buf = self._lib.metal_buffer_from_iosurface(self._ctx, surface_ref)
        if not buf:
            raise RuntimeError("Failed to wrap existing IOSurface in MTLBuffer")
        return MetalBuffer(self._ctx, buf, surface_ref)

    def create_buffer(self, nbytes: int) -> MetalBuffer:
        """Create a standard shared memory MTLBuffer."""
        buf = self._lib.metal_buffer_create(self._ctx, nbytes)
        if not buf:
            raise RuntimeError(f"Failed to create MTLBuffer of {nbytes} bytes")
        return MetalBuffer(self._ctx, buf)

    def create_shared_event(self) -> MetalSharedEvent:
        """Create an MTLSharedEvent for GPU-to-ANE hardware synchronization."""
        return MetalSharedEvent(self._ctx)

    def dispatch_rmsnorm(
        self,
        in_buf: MetalBuffer,
        weight_buf: MetalBuffer,
        out_buf: MetalBuffer,
        S: int,
        C: int,
        eps: float = 1e-6,
        signal_event: Optional[tuple[MetalSharedEvent, int]] = None,
        wait_event: Optional[tuple[MetalSharedEvent, int]] = None,
    ) -> None:
        cmd = self._lib.metal_command_buffer_create(self._ctx)
        if wait_event:
            ev, val = wait_event
            self._lib.metal_encode_wait_event(cmd, ev.handle, val)

        self._lib.metal_dispatch_rmsnorm_fp16(
            self._ctx, cmd, in_buf.handle, weight_buf.handle, out_buf.handle, S, C, eps
        )

        if signal_event:
            ev, val = signal_event
            self._lib.metal_encode_signal_event(cmd, ev.handle, val)

        self._lib.metal_command_buffer_commit(cmd)
        self._lib.metal_command_buffer_wait(cmd)

    def dispatch_moe_gather(
        self,
        src_activations: MetalBuffer,
        expert_indices: MetalBuffer,
        dst_gathered: MetalBuffer,
        num_tokens: int,
        top_k: int,
        hidden_dim: int,
    ) -> None:
        cmd = self._lib.metal_command_buffer_create(self._ctx)
        self._lib.metal_dispatch_moe_gather_fp16(
            self._ctx,
            cmd,
            src_activations.handle,
            expert_indices.handle,
            dst_gathered.handle,
            num_tokens,
            top_k,
            hidden_dim,
        )
        self._lib.metal_command_buffer_commit(cmd)
        self._lib.metal_command_buffer_wait(cmd)

    def dispatch_moe_scatter(
        self,
        expert_outputs: MetalBuffer,
        expert_weights: MetalBuffer,
        expert_indices: MetalBuffer,
        dst_combined: MetalBuffer,
        num_tokens: int,
        top_k: int,
        hidden_dim: int,
    ) -> None:
        cmd = self._lib.metal_command_buffer_create(self._ctx)
        self._lib.metal_dispatch_moe_scatter_fp16(
            self._ctx,
            cmd,
            expert_outputs.handle,
            expert_weights.handle,
            expert_indices.handle,
            dst_combined.handle,
            num_tokens,
            top_k,
            hidden_dim,
        )
        self._lib.metal_command_buffer_commit(cmd)
        self._lib.metal_command_buffer_wait(cmd)

    def dispatch_layout_transform(
        self,
        in_buf: MetalBuffer,
        out_buf: MetalBuffer,
        C: int,
        S: int,
        mode: int,  # 0: Linear -> ANE, 1: ANE -> Linear
    ) -> None:
        cmd = self._lib.metal_command_buffer_create(self._ctx)
        self._lib.metal_dispatch_layout_transform_fp16(
            self._ctx, cmd, in_buf.handle, out_buf.handle, C, S, mode
        )
        self._lib.metal_command_buffer_commit(cmd)
        self._lib.metal_command_buffer_wait(cmd)

    def dispatch_gemm(
        self,
        in_buf: MetalBuffer,
        weight_buf: MetalBuffer,
        out_buf: MetalBuffer,
        M: int,
        N: int,
        K: int,
    ) -> None:
        cmd = self._lib.metal_command_buffer_create(self._ctx)
        self._lib.metal_dispatch_gemm_fp16(
            self._ctx, cmd, in_buf.handle, weight_buf.handle, out_buf.handle, M, N, K
        )
        self._lib.metal_command_buffer_commit(cmd)
        self._lib.metal_command_buffer_wait(cmd)

    def dispatch_argmax(
        self,
        logits_buf: MetalBuffer,
        out_tokens_buf: MetalBuffer,
        B: int,
        V: int,
    ) -> None:
        cmd = self._lib.metal_command_buffer_create(self._ctx)
        self._lib.metal_dispatch_argmax_fp16(
            self._ctx, cmd, logits_buf.handle, out_tokens_buf.handle, B, V
        )
        self._lib.metal_command_buffer_commit(cmd)
        self._lib.metal_command_buffer_wait(cmd)

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.metal_context_destroy(self._ctx)
            self._ctx = 0
