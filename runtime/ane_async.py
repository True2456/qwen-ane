# SPDX-License-Identifier: Apache-2.0
"""Async ANE submit: private-framework completion + a dedicated evaluate thread.

`evaluateWithQoS:options:request:error:` blocks. `_ANERequest` exposes
`setCompletionHandler:` (block) and `sharedEvents` wrapping a Metal shared
event; neither makes the evaluate call return early on this OS (measured in
`probes/ane_async_api.py`). Overlap with MLX therefore needs the blocking
call on a different thread from `mx.eval`. ctypes releases the GIL around
`objc_msgSend`, so the two engines can run at once.

The completion handler is still attached when the block ABI is accepted: it
is the framework's completion signal, and `AneFuture.wait` prefers it.
"""
from __future__ import annotations

import ctypes
import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_BLOCK_IS_GLOBAL = 1 << 28
_keep_blocks: list = []


class _BlockDescriptor(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ulong), ("size", ctypes.c_ulong)]


class _BlockLiteral(ctypes.Structure):
    _fields_ = [
        ("isa", ctypes.c_void_p),
        ("flags", ctypes.c_int),
        ("reserved", ctypes.c_int),
        ("invoke", ctypes.c_void_p),
        ("descriptor", ctypes.POINTER(_BlockDescriptor)),
    ]


def _ns_concrete_global_block() -> ctypes.c_void_p:
    libc = ctypes.CDLL(None)
    return ctypes.c_void_p.in_dll(libc, "_NSConcreteGlobalBlock")


# void (^)(BOOL, NSError *)
_INVOKE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p)


def make_bool_error_block(callback):
    """ObjC global block. `callback(ok: bool, err_ptr)`."""

    @_INVOKE
    def invoke(_block, ok, err):
        callback(bool(ok), err)

    desc = _BlockDescriptor(0, ctypes.sizeof(_BlockLiteral))
    blk = _BlockLiteral(
        _ns_concrete_global_block(),
        _BLOCK_IS_GLOBAL,
        0,
        ctypes.cast(invoke, ctypes.c_void_p),
        ctypes.pointer(desc),
    )
    _keep_blocks.extend((invoke, desc, blk))
    return ctypes.addressof(blk)


class AneFuture:
    """Completion of one ANE evaluate."""

    __slots__ = ("_event", "_ok", "_err")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._ok = False
        self._err = None

    def _finish(self, ok: bool, err=None) -> None:
        self._ok = bool(ok)
        self._err = err
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        if not self._event.wait(timeout):
            raise TimeoutError("ANE submit_async timed out")
        return self._ok


class AneSubmitQueue:
    """One thread, one in-flight evaluate. Matches the single ane0 device."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="ane-submit", daemon=True
        )
        self._thread.start()
        self._handler_ok: Optional[bool] = None

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            program, procedure_index, fut, attach_handler = item
            try:
                if attach_handler and self._handler_ok is not False:
                    self._try_handler(program, fut)
                ok = self._engine._evaluate_blocking(program, procedure_index)
                if not fut._event.is_set():
                    fut._finish(ok)
            except Exception as exc:  # noqa: BLE001
                logger.exception("ANE worker evaluate failed")
                if not fut._event.is_set():
                    fut._finish(False, exc)

    def _try_handler(self, program, fut: AneFuture) -> None:
        if self._handler_ok is False or program._request is None:
            return
        try:
            from runtime.q38_ane_engine import _objc, _sel

            def on_done(ok, err):
                if not fut._event.is_set():
                    fut._finish(bool(ok), err)

            addr = make_bool_error_block(on_done)
            fn = _objc.objc_msgSend
            fn.restype = None
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            fn(program._request, _sel("setCompletionHandler:"),
               ctypes.c_void_p(addr))
            self._handler_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.info("setCompletionHandler: not usable (%s)", exc)
            self._handler_ok = False

    def submit(self, program, procedure_index: int = 0,
               attach_handler: bool = False) -> AneFuture:
        fut = AneFuture()
        self._q.put((program, int(procedure_index), fut, attach_handler))
        return fut

    def close(self) -> None:
        self._q.put(None)
