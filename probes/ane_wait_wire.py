#!/usr/bin/env python3
"""Wire IOSurfaceSharedEvent wait/complete into _ANERequest and time it."""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (
    _cls, _msg, _nsarray, _nsnumber_int, _objc, _sel, _desc,
)

ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/IOSurface.framework/IOSurface"
)


def _cfn(restype, *argtypes):
    return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)


def make_ios_event(value=0):
    cls = _cls("IOSurfaceSharedEvent")
    raw = _msg(cls, "alloc")
    Init = _cfn(ctypes.c_void_p, ctypes.c_ulonglong)
    ev = Init(("objc_msgSend", _objc))(raw, _sel("initWithOptions:"), 0)
    if not ev:
        raise RuntimeError("IOSurfaceSharedEvent initWithOptions:0 failed")
    Set = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong)
    Set(("objc_msgSend", _objc))(ev, _sel("setSignaledValue:"), int(value))
    return ev


def signaled(ev):
    Get = ctypes.CFUNCTYPE(ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_void_p)
    return Get(("objc_msgSend", _objc))(ev, _sel("signaledValue"))


def set_signaled(ev, value):
    Set = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong)
    Set(("objc_msgSend", _objc))(ev, _sel("setSignaledValue:"), int(value))


def make_wait(ev, value=1):
    Wait = _cfn(ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_void_p)
    return Wait(("objc_msgSend", _objc))(
        _cls("_ANESharedWaitEvent"),
        _sel("waitEventWithValue:sharedEvent:"),
        ctypes.c_ulonglong(value), ev)


def make_shared_events(*, waits=None, signals=None):
    waits_a = _nsarray(waits) if waits else None
    sigs_a = _nsarray(signals) if signals else None
    Fn = _cfn(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    return Fn(("objc_msgSend", _objc))(
        _cls("_ANESharedEvents"),
        _sel("sharedEventsWithSignalEvents:waitEvents:"),
        sigs_a, waits_a)


def rebuild_request(eng, prog, shared_events):
    prog._request = None
    prog._proc_idx = -1
    ok = eng._ensure_request(prog, 0)
    if not ok:
        raise RuntimeError("ensure_request failed")
    # Rebuild with sharedEvents by calling init again.
    # Patch: setSharedEvents: if it exists, else rebuild.
    req = prog._request
    setter = _sel("setSharedEvents:")
    has = _objc.class_getInstanceMethod(_cls("_ANERequest"), setter)
    if has:
        Set = _cfn(None, ctypes.c_void_p)
        Set(("objc_msgSend", _objc))(req, setter, shared_events)
        print("  attached via setSharedEvents:")
        return
    raise RuntimeError("no setSharedEvents:")


def dump_client():
    cls = _cls("_ANEClient")
    print("\n===== _ANEClient eval-like =====")
    n = ctypes.c_uint(0)
    _objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
    _objc.class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    for target, label in ((_objc.object_getClass(cls), "class"), (cls, "inst")):
        raw = _objc.class_copyMethodList(target, ctypes.byref(n))
        names = []
        for i in range(n.value):
            sel = _objc.method_getName(raw[i])
            name = _objc.sel_getName(sel).decode()
            if any(k in name.lower() for k in ("eval", "enque", "complet", "event", "shared")):
                enc = _objc.method_getTypeEncoding(raw[i])
                names.append(f"  {label} {name}  {enc.decode() if enc else ''}")
        print("\n".join(names) if names else f"  ({label} none)")


def med(fn, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    dump_client()
    eng = E.AneEngine()
    rng = np.random.default_rng(0)
    dim = 2048
    W = rng.standard_normal((dim, dim)).astype(np.float32)
    prog = eng.compile_linear(W, seq_len=32, instance_hint=1, quantized=True)
    x = rng.standard_normal((32, dim)).astype(np.float32)
    assert eng.evaluate(prog, x) is not None
    for _ in range(8):
        eng.submit(prog)
    t_ane = med(lambda: eng.submit(prog))
    print(f"\nbaseline submit {t_ane * 1e3:.3f} ms")

    ev = make_ios_event(0)
    print(f"IOSurfaceSharedEvent signaledValue={signaled(ev)}")
    wait = make_wait(ev, 1)
    print(f"wait event {wait}")
    events = make_shared_events(waits=[wait])
    print(f"sharedEvents {events}")
    rebuild_request(eng, prog, events)
    # keep alive
    prog._keep_alive.extend([ev, wait, events])

    # If wait is honoured, submit without signaling should hang.
    print("submit with wait=1, event=0 (0.4s timeout)...")
    done = threading.Event()
    result = {}

    def run():
        t0 = time.perf_counter()
        try:
            ok = eng.submit(prog)
            result["ok"] = ok
        except Exception as exc:  # noqa: BLE001
            result["err"] = repr(exc)
        result["dt"] = time.perf_counter() - t0
        done.set()

    threading.Thread(target=run, daemon=True).start()
    if done.wait(0.4):
        print(f"  RETURNED in {result.get('dt', 0)*1e3:.3f} ms  {result}")
        print("  wait event is IGNORED by evaluateWithQoS")
    else:
        print("  still blocked after 0.4s — wait event is HONOURED")
        set_signaled(ev, 1)
        if done.wait(2.0):
            print(f"  unblocked after signal in {result.get('dt', 0)*1e3:.3f} ms  {result}")
        else:
            print("  STILL blocked after signal — abort")
            return

    # Pre-signaled wait=1, event=1: should be ~baseline
    set_signaled(ev, 1)
    t_pre = med(lambda: eng.submit(prog))
    print(f"pre-signaled wait  {t_pre * 1e3:.3f} ms")

    # Delayed signal overlap
    delay = 0.003

    def delayed():
        set_signaled(ev, 0)
        start = time.perf_counter()
        done2 = threading.Event()
        box = {}

        def run2():
            box["ok"] = eng.submit(prog)
            done2.set()

        threading.Thread(target=run2, daemon=True).start()
        time.sleep(delay)
        set_signaled(ev, 1)
        done2.wait()
        return time.perf_counter() - start

    # Only meaningful if wait is honoured. Try anyway.
    try:
        t_del = med(delayed)
        print(f"signal after {delay*1e3:.1f} ms  wall {t_del*1e3:.3f}  "
              f"delay+ane {(delay+t_ane)*1e3:.3f}  max {max(delay, t_ane)*1e3:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"delayed signal failed: {exc}")

    # VirtualClient doEvaluate + completionEvent
    print("\n===== VirtualClient doEvaluate completionEvent =====")
    vcls = _cls("_ANEVirtualClient")
    client = _msg(vcls, "new")
    print(f"client {client}")
    inner = _msg(prog.model, "model") or prog.model
    cev = make_ios_event(0)
    err = ctypes.c_void_p(0)
    Eval = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    t0 = time.perf_counter()
    try:
        ok = Eval(("objc_msgSend", _objc))(
            client, _sel("doEvaluateWithModel:options:request:qos:completionEvent:error:"),
            inner, prog._compile_opts, prog._request, 21, cev, ctypes.byref(err))
        dt = time.perf_counter() - t0
        print(f"  returned ok={ok} in {dt*1e3:.3f} ms  "
              f"completion signaled={signaled(cev)}  err={_desc(err.value) if err.value else None}")
    except Exception as exc:  # noqa: BLE001
        print(f"  doEvaluate crashed: {type(exc).__name__}: {exc}")

    # _ANEClient
    print("\n===== _ANEClient sharedConnection =====")
    ccls = _cls("_ANEClient")
    conn = _msg(ccls, "sharedConnection")
    print(f"sharedConnection {conn}")
    if conn:
        for sel in (
            "doEvaluateDirectWithModel:options:request:qos:error:",
            "doEvaluateWithModel:options:request:qos:completionEvent:error:",
            "doEnqueueSetsWithModel:outputSet:",
        ):
            m = _objc.class_getInstanceMethod(ccls, _sel(sel))
            enc = _objc.method_getTypeEncoding(m).decode() if m else None
            print(f"  {sel}: {enc}")


if __name__ == "__main__":
    main()
