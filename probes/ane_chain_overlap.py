#!/usr/bin/env python3
"""Can chaining / enqueueSets / buffersReady start ANE setup before inputs are ready?

Need: one mailbox, GPU running while ANE is already doing the 1.2 ms fixed cost.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (
    _BUILD_INFO, _MSGSEND_ADDR, _cls, _desc, _msg, _nsarray, _nsnumber_int,
    _objc, _objc_call, _sel,
)
from runtime.metal_engine import MetalEngine


def _enc(cls, sel):
    m = _objc.class_getInstanceMethod(cls, _sel(sel)) or _objc.class_getClassMethod(cls, _sel(sel))
    if not m:
        return None
    e = _objc.method_getTypeEncoding(m)
    return e.decode() if e else ""


def med(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def make_signal(mtl_ios, value=1, symbol=0, event_type=0):
    return _objc_call(
        ctypes.c_void_p,
        [ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_longlong, ctypes.c_void_p],
        _cls("_ANESharedSignalEvent"),
        "signalEventWithValue:symbolIndex:eventType:sharedEvent:",
        value, symbol, event_type, mtl_ios,
    )


def main():
    _objc.method_getTypeEncoding.restype = ctypes.c_char_p
    _objc.method_getTypeEncoding.argtypes = [ctypes.c_void_p]
    _objc.class_getInstanceMethod.restype = ctypes.c_void_p
    _objc.class_getClassMethod.restype = ctypes.c_void_p

    print("===== _ANERequest extras =====")
    req_cls = _cls("_ANERequest")
    n = ctypes.c_uint(0)
    _objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
    _objc.class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    _objc.method_getName.restype = ctypes.c_void_p
    _objc.sel_getName.restype = ctypes.c_char_p
    raw = _objc.class_copyMethodList(req_cls, ctypes.byref(n))
    for i in range(n.value):
        name = _objc.sel_getName(_objc.method_getName(raw[i])).decode()
        print(f"  {name}  {_enc(req_cls, name)}")

    eng = E.AneEngine()
    rng = np.random.default_rng(0)
    dim, seq = 4096, 32
    W = rng.standard_normal((dim, dim)).astype(np.float32)
    print(f"\ncompile_linear {dim}x{dim} …", flush=True)
    prog = eng.compile_linear(W, seq_len=seq, instance_hint=1, quantized=True)
    x = rng.standard_normal((seq, dim)).astype(np.float32)
    y = eng.evaluate(prog, x)
    assert y is not None
    for _ in range(8):
        eng.submit(prog)
    t_ane = med(lambda: eng.submit(prog))
    print(f"warm submit {t_ane * 1e3:.3f} ms")

    t_two = med(lambda: (eng.submit(prog), eng.submit(prog)))
    print(f"two submits {t_two * 1e3:.3f} ms  (2x={2 * t_ane * 1e3:.3f})")

    inner = _msg(prog.model, "model") or prog.model
    client = _msg(_cls("_ANEClient"), "sharedConnection")
    print(f"client {client}  inner {inner}")

    me = MetalEngine()
    ev = me.create_shared_event()
    mtl = ctypes.c_void_p(ev.handle)
    ios = _objc_call(ctypes.c_void_p, [], mtl, "IOSurfaceSharedEvent")
    sig = make_signal(ios, 1)
    print(f"signal {sig} {_desc(sig) if sig else None}")

    outset = _objc_call(
        ctypes.c_void_p,
        [ctypes.c_uint, ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_bool, ctypes.c_bool],
        _cls("_ANEOutputSetEnqueue"),
        "outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:",
        0, 0, 1, False, True,
    )
    print(f"outset {outset} {_desc(outset) if outset else None}")

    Eval = ctypes.CFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
    )

    def client_call(sel, model, arg):
        err = ctypes.c_void_p(0)
        t0 = time.perf_counter()
        ok = Eval(_MSGSEND_ADDR)(
            client, _sel(sel), model, arg, prog._compile_opts, 21, ctypes.byref(err))
        dt = time.perf_counter() - t0
        return ok, dt, _desc(err.value) if err.value else None

    print("\n===== enqueueSets =====")
    for model, label in ((inner, "inner"), (prog.model, "inmem")):
        ev.value = 0
        ok, dt, err = client_call(
            "enqueueSetsWithModel:outputSet:options:qos:error:", model, outset)
        print(f"  {label} ok={ok} {dt*1e3:.3f}ms err={err} signaled={ev.value}")

    print("\n===== doEnqueueSets =====")
    ev.value = 0
    ok, dt, err = client_call(
        "doEnqueueSetsWithModel:outputSet:options:qos:error:", inner, outset)
    print(f"  inner ok={ok} {dt*1e3:.3f}ms err={err} signaled={ev.value}")

    # chaining request — all args are ids
    print("\n===== prepareChaining =====")
    Chain = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
    # wrapping IOSurfaces already on request
    in_surfs = prog._in_surfs or [prog._in_surf]
    # chaining wants inputs array and outputSets
    # Use objects from existing request if we can get them
    for getter in ("inputs", "inputArray", "inputSurfaces"):
        print(f"  request {getter}: {_enc(req_cls, getter)}")

    chain_sel = (
        "chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:"
        "procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:"
    )
    print(f"  chain ctor {_enc(_cls('_ANEChainingRequest'), chain_sel)}")

    # Build with nils except outputSets and procedureIndex
    raw = _msg(_cls("_ANEChainingRequest"), "alloc")
    # class method might be easier
    creq = _objc_call(
        ctypes.c_void_p,
        [ctypes.c_void_p] * 9,
        _cls("_ANEChainingRequest"),
        chain_sel,
        None,  # inputs
        _nsarray([outset]),
        None, None,  # loopback ids
        _nsnumber_int(0),
        _nsarray([sig]) if sig else None,
        None,  # transaction
        None,  # fwEnqueueDelay
        None,  # memoryPoolId
    )
    print(f"  creq {creq} {_desc(creq) if creq else None}")
    if creq:
        valid = _objc_call(ctypes.c_bool, [], creq, "validate")
        print(f"  validate {valid}")
        ok, dt, err = client_call(
            "prepareChainingWithModel:options:chainingReq:qos:error:", inner, creq)
        print(f"  prepare inner ok={ok} {dt*1e3:.3f}ms err={err}")
        ok, dt, err = client_call(
            "doPrepareChainingWithModel:options:chainingReq:qos:error:", inner, creq)
        print(f"  doPrepare inner ok={ok} {dt*1e3:.3f}ms err={err}")

    print("\n===== buffersReady =====")
    ib = _objc_call(
        ctypes.c_void_p,
        [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong],
        _cls("_ANEInputBuffersReady"),
        "inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:",
        0, None, None, 0,
    )
    print(f"  ib {ib} {_desc(ib) if ib else None}")
    if ib:
        print(f"  ib.validate {_objc_call(ctypes.c_bool, [], ib, 'validate')}")
        ok, dt, err = client_call(
            "buffersReadyWithModel:inputBuffers:options:qos:error:", inner, ib)
        print(f"  buffersReady ok={ok} {dt*1e3:.3f}ms err={err}")

    # Two-proc MIL: does one program with two procedures share one load?
    print("\n===== two-proc sequential submit =====")
    C, S = 512, 32
    mil = (
        f"program(1.3)\n{_BUILD_INFO}\n{{\n"
        f"  func procedure000<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{\n"
        f'    tensor<fp16, [1, {C}, 1, {S}]> y = add(x=x, y=fp16(0x1p+0))[name=string("y0")];\n'
        f"  }} -> (y);\n"
        f"  func procedure001<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{\n"
        f'    tensor<fp16, [1, {C}, 1, {S}]> y = add(x=x, y=fp16(0x1p+1))[name=string("y1")];\n'
        f"  }} -> (y);\n}}\n"
    )
    p2 = eng.compile_multiproc(mil, {}, C, C, S)
    print(f"  two-proc compile {p2 is not None} num_proc={getattr(p2, 'num_procedures', None)}")
    if p2 is not None:
        xx = rng.standard_normal((S, C)).astype(np.float32)
        y0 = eng.evaluate(p2, xx, procedure_index=0)
        y1 = eng.evaluate(p2, xx, procedure_index=1)
        print(f"  y0 mean {float(np.mean(y0)) if y0 is not None else None}  "
              f"y1 mean {float(np.mean(y1)) if y1 is not None else None}")
        for _ in range(8):
            eng.submit(p2, 0)
            eng.submit(p2, 1)
        t0 = med(lambda: eng.submit(p2, 0))
        t1 = med(lambda: eng.submit(p2, 1))
        tb = med(lambda: (eng.submit(p2, 0), eng.submit(p2, 1)))
        print(f"  proc0 {t0*1e3:.3f} ms  proc1 {t1*1e3:.3f} ms  both {tb*1e3:.3f} ms")


if __name__ == "__main__":
    main()
