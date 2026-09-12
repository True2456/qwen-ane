#!/usr/bin/env python3
"""Dump ANE shared-event / chaining selectors and time wait-before-evaluate.

Question: if evaluate is waiting on a Metal shared event, does the 1.2 ms GDN
fixed cost run during the wait (overlap with GPU) or after the signal?
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
ctypes.cdll.LoadLibrary(
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
)
try:
    ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/IOSurface.framework/IOSurface"
    )
except OSError:
    pass

_objc.objc_getClass.restype = ctypes.c_void_p
_objc.objc_getClass.argtypes = [ctypes.c_char_p]
_objc.object_getClass.restype = ctypes.c_void_p
_objc.object_getClass.argtypes = [ctypes.c_void_p]
_objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
_objc.class_copyMethodList.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
_objc.method_getName.restype = ctypes.c_void_p
_objc.method_getName.argtypes = [ctypes.c_void_p]
_objc.sel_getName.restype = ctypes.c_char_p
_objc.sel_getName.argtypes = [ctypes.c_void_p]
_objc.method_getTypeEncoding.restype = ctypes.c_char_p
_objc.method_getTypeEncoding.argtypes = [ctypes.c_void_p]
_objc.sel_registerName.restype = ctypes.c_void_p
_objc.sel_registerName.argtypes = [ctypes.c_char_p]
_objc.class_getInstanceMethod.restype = ctypes.c_void_p
_objc.class_getInstanceMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_objc.class_getClassMethod.restype = ctypes.c_void_p
_objc.class_getClassMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def _all_methods(cls, class_methods: bool = False):
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


def dump_all(name: str) -> None:
    cls = _objc.objc_getClass(name.encode())
    if not cls:
        print(f"\n{name}: MISSING")
        return
    print(f"\n===== {name} =====")
    for label, cm in (("class", True), ("inst", False)):
        ms = _all_methods(cls, cm)
        if not ms:
            print(f"  ({label} none)")
            continue
        for n, enc in ms:
            print(f"  {label} {n}  {enc}")


def _has(cls, sel: str):
    s = _objc.sel_registerName(sel.encode())
    m = _objc.class_getInstanceMethod(cls, s) or _objc.class_getClassMethod(cls, s)
    if not m:
        return None
    enc = _objc.method_getTypeEncoding(m)
    return enc.decode() if enc else ""


def dump_virtual_client_eval():
    cls = _objc.objc_getClass(b"_ANEVirtualClient")
    print("\n===== _ANEVirtualClient eval-like =====")
    if not cls:
        print("  MISSING")
        return
    for n, enc in _all_methods(cls, False) + _all_methods(cls, True):
        low = n.lower()
        if any(k in low for k in ("eval", "complet", "event", "enque", "chain",
                                  "async", "qos", "request")):
            print(f"  {n}  {enc}")


def try_wait_overlap():
    import numpy as np
    import runtime.q38_ane_engine as E

    eng = E.AneEngine()
    rng = np.random.default_rng(0)
    # Fat enough that a submit is ~1 ms, like a slice of GDN fixed cost.
    dim = 2048
    W = rng.standard_normal((dim, dim)).astype(np.float32)
    prog = eng.compile_linear(W, seq_len=32, instance_hint=1, quantized=True)
    if prog is None:
        print("compile_linear failed")
        return
    x = rng.standard_normal((32, dim)).astype(np.float32)
    y = eng.evaluate(prog, x)
    assert y is not None
    for _ in range(8):
        eng.submit(prog)

    def med(fn, reps=9):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    t_ane = med(lambda: eng.submit(prog))
    print(f"\n===== wait-event overlap =====")
    print(f"  ANE linear {dim}x{dim}  {t_ane * 1e3:.3f} ms")

    # Software barrier: start evaluate on worker, signal after `delay`.
    # If evaluate waits then runs, wall ≈ delay + t_ane.
    # If it overlaps setup with the delay, wall < delay + t_ane.
    delay = 0.005

    def delayed_submit():
        start = threading.Event()
        done = threading.Event()
        err = []

        def run():
            start.wait()
            try:
                eng.submit(prog)
            except Exception as exc:  # noqa: BLE001
                err.append(exc)
            finally:
                done.set()

        threading.Thread(target=run, daemon=True).start()
        t0 = time.perf_counter()
        start.set()
        time.sleep(delay)
        done.wait()
        if err:
            raise err[0]
        return time.perf_counter() - t0

    t_delay = med(delayed_submit)
    print(f"  sleep {delay * 1e3:.1f} ms then ANE on other thread (control)")
    print(f"    wall {t_delay * 1e3:.3f} ms   "
          f"delay+ane {(delay + t_ane) * 1e3:.3f} ms   "
          f"max(delay,ane) {max(delay, t_ane) * 1e3:.3f} ms")

    # Thread overlap without wait event: ANE || sleep, then we know threads work.
    def overlap_sleep():
        done = threading.Event()

        def run():
            eng.submit(prog)
            done.set()

        threading.Thread(target=run, daemon=True).start()
        time.sleep(delay)
        done.wait()

    t_ov = med(overlap_sleep)
    print(f"  ANE || sleep {delay * 1e3:.1f} ms")
    print(f"    wall {t_ov * 1e3:.3f} ms   "
          f"(expect ~max = {max(delay, t_ane) * 1e3:.3f} if overlap)")

    # Try wrapping a Metal shared event if the dylib is present.
    try:
        from runtime.metal_engine import MetalEngine
        me = MetalEngine()
        ev = me.create_shared_event()
        print(f"  MTLSharedEvent handle={ev.handle:#x} value={ev.value}")
        wait_cls = _objc.objc_getClass(b"_ANESharedWaitEvent")
        sig_cls = _objc.objc_getClass(b"_ANESharedSignalEvent")
        evs_cls = _objc.objc_getClass(b"_ANESharedEvents")
        ios_cls = _objc.objc_getClass(b"IOSurfaceSharedEvent")
        print(f"  _ANESharedWaitEvent={wait_cls} Signal={sig_cls} "
              f"Events={evs_cls} IOSurfaceSharedEvent={ios_cls}")
        for cls, name in (
            (wait_cls, "_ANESharedWaitEvent"),
            (sig_cls, "_ANESharedSignalEvent"),
            (evs_cls, "_ANESharedEvents"),
            (ios_cls, "IOSurfaceSharedEvent"),
        ):
            if not cls:
                continue
            print(f"  inits on {name}:")
            for n, enc in _all_methods(cls, True) + _all_methods(cls, False):
                if "init" in n.lower() or "event" in n.lower() or n.startswith("wait") or n.startswith("signal"):
                    print(f"    {n}  {enc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  Metal/shared-event wrap: {type(exc).__name__}: {exc}")

    vcls = _objc.objc_getClass(b"_ANEVirtualClient")
    if vcls:
        print("\n  trying doEvaluate…completionEvent on VirtualClient")
        for sel in (
            "sharedRemoteConnection",
            "new",
            "alloc",
        ):
            print(f"    {sel}: {_has(vcls, sel)!r}")


def main():
    for name in (
        "_ANESharedEvents", "_ANESharedSignalEvent", "_ANESharedWaitEvent",
        "_ANEChainingRequest", "_ANEOutputSetEnqueue", "_ANEInputBuffersReady",
        "IOSurfaceSharedEvent",
    ):
        dump_all(name)
    dump_virtual_client_eval()
    try_wait_overlap()


if __name__ == "__main__":
    main()
