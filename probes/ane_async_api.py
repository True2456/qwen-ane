#!/usr/bin/env python3
"""Dump private ANE evaluate/async selectors and time ANE||GPU overlap.

Lab measurement, not a production path. Re-run whenever the framework changes.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
ctypes.cdll.LoadLibrary(
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
)

_objc.objc_getClass.restype = ctypes.c_void_p
_objc.objc_getClass.argtypes = [ctypes.c_char_p]
_objc.object_getClass.restype = ctypes.c_void_p
_objc.object_getClass.argtypes = [ctypes.c_void_p]
_objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
_objc.class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
_objc.method_getName.restype = ctypes.c_void_p
_objc.method_getName.argtypes = [ctypes.c_void_p]
_objc.sel_getName.restype = ctypes.c_char_p
_objc.sel_getName.argtypes = [ctypes.c_void_p]
_objc.class_getInstanceMethod.restype = ctypes.c_void_p
_objc.class_getInstanceMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_objc.class_getClassMethod.restype = ctypes.c_void_p
_objc.class_getClassMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_objc.sel_registerName.restype = ctypes.c_void_p
_objc.sel_registerName.argtypes = [ctypes.c_char_p]
_objc.method_getTypeEncoding.restype = ctypes.c_char_p
_objc.method_getTypeEncoding.argtypes = [ctypes.c_void_p]


def _methods(cls, class_methods: bool = False) -> list[tuple[str, str]]:
    target = _objc.object_getClass(cls) if class_methods else cls
    n = ctypes.c_uint(0)
    raw = _objc.class_copyMethodList(target, ctypes.byref(n))
    out = []
    for i in range(n.value):
        sel = _objc.method_getName(raw[i])
        name = _objc.sel_getName(sel).decode()
        enc = _objc.method_getTypeEncoding(raw[i])
        out.append((name, enc.decode() if enc else ""))
    return sorted(out)


def _has(cls, sel: str) -> str | None:
    s = _objc.sel_registerName(sel.encode())
    m = _objc.class_getInstanceMethod(cls, s)
    if not m:
        m = _objc.class_getClassMethod(cls, s)
    if not m:
        return None
    enc = _objc.method_getTypeEncoding(m)
    return enc.decode() if enc else ""


KEYS = (
    "evaluat", "async", "complet", "queue", "shared", "event", "signal",
    "wait", "enqueu", "processRequest", "chain", "handler", "block",
)


def dump_class(name: str) -> None:
    cls = _objc.objc_getClass(name.encode())
    if not cls:
        print(f"\n{name}: MISSING")
        return
    print(f"\n===== {name} =====")
    for label, class_methods in (("class", True), ("inst", False)):
        found = []
        for n, enc in _methods(cls, class_methods=class_methods):
            low = n.lower()
            if any(k.lower() in low for k in KEYS):
                found.append(f"  {label} {n}  {enc}")
        if found:
            print("\n".join(found))


def main() -> None:
    for name in (
        "_ANEInMemoryModel", "_ANERequest", "_ANEClient", "_ANEVirtualClient",
        "_ANEProgramForEvaluation", "_ANEChainingRequest", "_ANESharedEvents",
        "_ANESharedSignalEvent", "_ANESharedWaitEvent", "_ANEOutputSetEnqueue",
        "_ANEInputBuffersReady", "_ANEModel",
    ):
        dump_class(name)

    print("\n===== selector probe =====")
    cls = _objc.objc_getClass(b"_ANERequest")
    for sel in (
        "setCompletionHandler:", "completionHandler", "setQueueDepth:",
        "queueDepth", "sharedEvents", "setSharedEvents:",
        "evaluateAsyncWithQoS:options:request:completionHandler:",
        "evaluateWithQoS:options:request:completionHandler:",
    ):
        enc = _has(cls, sel)
        print(f"  _ANERequest {sel}: {enc!r}")
    mcls = _objc.objc_getClass(b"_ANEInMemoryModel")
    for sel in (
        "evaluateWithQoS:options:request:error:",
        "evaluateWithQoS:options:request:completionHandler:",
        "evaluateAsyncWithQoS:options:request:error:",
        "setQueueDepth:", "queueDepth",
        "doEvaluateDirectWithModel:options:request:qos:error:",
    ):
        enc = _has(mcls, sel)
        print(f"  _ANEInMemoryModel {sel}: {enc!r}")

    if os.environ.get("ANE_ASYNC_BENCH", "1") == "0":
        return

    import numpy as np
    import runtime.q38_ane_engine as E

    eng = E.AneEngine()
    rng = np.random.default_rng(0)
    W = rng.standard_normal((256, 256)).astype(np.float32)
    prog = eng.compile_linear(W, seq_len=32, instance_hint=1, quantized=True)
    if prog is None:
        print("compile_linear failed")
        return
    x = rng.standard_normal((32, 256)).astype(np.float32)
    y = eng.evaluate(prog, x)
    assert y is not None
    for _ in range(5):
        eng.submit(prog)

    def ane_work(n: int = 8) -> None:
        for _ in range(n):
            eng.submit(prog)

    try:
        import mlx.core as mx
        a = mx.random.normal((4096, 4096), dtype=mx.float16)
        b = mx.random.normal((4096, 4096), dtype=mx.float16)
        mx.eval(a, b)

        def gpu_work(n: int = 8) -> None:
            z = a
            for _ in range(n):
                z = z @ b
            mx.eval(z)
    except Exception as exc:  # noqa: BLE001
        print(f"MLX unavailable: {exc}")
        return

    def med(fn, reps: int = 7) -> float:
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    t_ane = med(lambda: ane_work(8))
    t_gpu = med(lambda: gpu_work(8))
    t_serial = med(lambda: (ane_work(8), gpu_work(8)))

    def overlap_threads() -> None:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(ane_work, 8)
            gpu_work(8)
            fa.result()

    t_thr = med(overlap_threads)

    def overlap_worker() -> None:
        done = threading.Event()
        err = []

        def run():
            try:
                ane_work(8)
            except Exception as e:  # noqa: BLE001
                err.append(e)
            finally:
                done.set()

        threading.Thread(target=run, daemon=True).start()
        gpu_work(8)
        done.wait()
        if err:
            raise err[0]

    t_w = med(overlap_worker)
    saved = t_serial - t_thr
    print("\n===== ANE || GPU overlap (small linear + 4096 GEMM) =====")
    print(f"  ANE only     {t_ane * 1e3:7.2f} ms")
    print(f"  GPU only     {t_gpu * 1e3:7.2f} ms")
    print(f"  serial sum   {t_serial * 1e3:7.2f} ms")
    print(f"  2-thread     {t_thr * 1e3:7.2f} ms   saved {saved * 1e3:.2f} ms  "
          f"({100 * saved / max(t_serial, 1e-9):.0f}% of serial)")
    print(f"  worker+main  {t_w * 1e3:7.2f} ms")
    print(f"  lower bound  {max(t_ane, t_gpu) * 1e3:7.2f} ms")


if __name__ == "__main__":
    main()
