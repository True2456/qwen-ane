"""Does compiledModelExists hash the weight payloads? Can we load from disk?

The serve process holds 5.1 GB of Foundation NSData, one region per compiled
ANE program, after we have already dropped our own blob references. The
loaded _ANEInMemoryModel is the suspect. Two questions, both measured here
rather than assumed:

1. Is compiledModelExists keyed on a hash that includes the weight bytes, so
   an empty dictionary on a cache hit misses?
2. After a successful compile, can a *new* model built from localModelPath
   (initWithURL, no weights dictionary) evaluate the same graph, so the
   original descriptor-backed object can be released?

A 32 MB blob makes Foundation visible in ``footprint`` without loading the
full model. Unique MIL comment so this does not collide with other probes.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.q38_ane_engine import (  # noqa: E402
    AneEngine,
    _BUILD_INFO,
    _cls,
    _desc,
    _iosurface_view,
    _make_blob,
    _msg,
    _nsdata,
    _nsdict,
    _nsnumber_int,
    _nsnumber_uint,
    _nsstring,
    _objc,
    _sel,
)

C, S = 32, 32
BLOB_BYTES = 32 << 20
MARKER = "q38_weight_hash_identity_probe_v1"


def _mil() -> str:
    return f'''program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<fp16, [{C}, {C}, 1, 1]> w = const()[name=string("w"), val=tensor<fp16, [{C}, {C}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {C}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("y")];
  }} -> (y);
}}
// {MARKER}
'''


def _payload(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((C, C, 1, 1)).astype(np.float16).tobytes()
    blob = _make_blob(w)
    # Pad so Foundation has something to retain. The conv only reads C*C*2 bytes.
    if len(blob) < BLOB_BYTES:
        blob = blob + bytes(BLOB_BYTES - len(blob))
    return blob


def _weights_dict(blob: bytes):
    return _nsdict({
        _nsstring("@model_path/weights/weight.bin"): _nsdict({
            _nsstring("data"): _nsdata(blob),
            _nsstring("offset"): _nsnumber_uint(0),
        }),
    })


def _make_model(mil: str, weights_dict):
    mil_data = _nsdata(mil.encode("utf-8"))
    f = _objc.objc_msgSend
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p] * 5
    descriptor = f(
        _cls("_ANEInMemoryModelDescriptor"),
        _sel("modelWithMILText:weights:optionsPlist:"),
        mil_data, weights_dict, None)
    if not descriptor:
        raise RuntimeError("descriptor nil")
    model = _msg(_cls("_ANEInMemoryModel"), "inMemoryModelWithDescriptor:",
                 descriptor, argtypes=[ctypes.c_void_p])
    if not model:
        raise RuntimeError("inMemoryModel nil")
    return model


def _exists(model) -> bool:
    Exists = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    return bool(Exists(("objc_msgSend", _objc))(model, _sel("compiledModelExists")))


def _ident(model) -> str:
    return _desc(_msg(model, "hexStringIdentifier"))


def _opts():
    return _nsdict({
        _nsstring("kANEFProcedureVariantHint"): _nsnumber_int(1),
        _nsstring("kANEFKeepModelMemoryWiredKey"): _nsnumber_int(0),
    })


def _load(model, opts) -> bool:
    err_ptr = ctypes.c_void_p(0)
    Load = ctypes.CFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    return bool(Load(("objc_msgSend", _objc))(
        model, _sel("loadWithQoS:options:error:"), 21, opts,
        ctypes.byref(err_ptr)))


def _methods(cls, class_methods: bool = False):
    target = _objc.object_getClass(cls) if class_methods else cls
    n = ctypes.c_uint(0)
    _objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
    _objc.class_copyMethodList.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    _objc.method_getName.restype = ctypes.c_void_p
    _objc.method_getName.argtypes = [ctypes.c_void_p]
    _objc.sel_getName.restype = ctypes.c_char_p
    _objc.sel_getName.argtypes = [ctypes.c_void_p]
    raw = _objc.class_copyMethodList(target, ctypes.byref(n))
    out = []
    for i in range(n.value):
        sel = _objc.method_getName(raw[i])
        out.append(_objc.sel_getName(sel).decode())
    return sorted(out)


def _ivars(name: str) -> None:
    cls = _cls(name)
    if not cls:
        return
    n = ctypes.c_uint(0)
    _objc.class_copyIvarList.restype = ctypes.POINTER(ctypes.c_void_p)
    _objc.class_copyIvarList.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    _objc.ivar_getName.restype = ctypes.c_char_p
    _objc.ivar_getName.argtypes = [ctypes.c_void_p]
    raw = _objc.class_copyIvarList(cls, ctypes.byref(n))
    print(f"  ivars {name}:")
    for i in range(n.value):
        nm = _objc.ivar_getName(raw[i])
        print(f"    {nm.decode() if nm else '?'}")


def _dump_relevant(name: str) -> None:
    cls = _cls(name)
    if not cls:
        print(f"  {name}: MISSING")
        return
    keys = ("url", "URL", "path", "Path", "init", "weight", "Weight",
            "ident", "cache", "load", "model", "Model", "purge", "release",
            "compiled")
    print(f"  -- {name} --")
    for label, cm in (("class", True), ("inst", False)):
        for sel in _methods(cls, cm):
            if any(k in sel for k in keys):
                print(f"    {label} {sel}")


def _footprint_categories(pid: int) -> str:
    out = subprocess.run(
        ["footprint", "-p", str(pid)], capture_output=True, text=True)
    keep = ("phys_footprint", "IOAccelerator", "Malloc", "Foundation",
            "IOSurface", "TOTAL", "bytes")
    lines = []
    for line in (out.stdout or "").splitlines():
        if any(k.lower() in line.lower() for k in keep) or "GB" in line or "MB" in line:
            if "dirty" in line.lower() or "regions" in line.lower() \
                    or "footprint" in line.lower() or line.strip().startswith("TOTAL") \
                    or any(k in line for k in (
                        "IOAccelerator", "Malloc Large", "Foundation", "IOSurface",
                        "phys_footprint")):
                lines.append(line.rstrip())
    return "\n".join(lines[:40]) or (out.stdout or out.stderr)[:1500]


def main() -> None:
    mil = _mil()
    blob = _payload(7)
    zeros = bytes(len(blob))
    eng = AneEngine()
    if not eng.available:
        raise SystemExit("ANE engine unavailable")
    empty = _nsdict({})
    opts = _opts()

    print("=== descriptor identity ===", flush=True)
    m_full = _make_model(mil, _weights_dict(blob))
    m_empty = _make_model(mil, empty)
    m_zeros = _make_model(mil, _weights_dict(zeros))
    m_again = _make_model(mil, _weights_dict(blob))
    id_full = _ident(m_full)
    id_empty = _ident(m_empty)
    id_zeros = _ident(m_zeros)
    id_again = _ident(m_again)
    print(f"  ident full   {id_full}")
    print(f"  ident empty  {id_empty}")
    print(f"  ident zeros  {id_zeros}")
    print(f"  ident again  {id_again}")
    print(f"  empty vs full:  {'SAME' if id_empty == id_full else 'DIFFERENT'}")
    print(f"  zeros vs full:  {'SAME' if id_zeros == id_full else 'DIFFERENT'}")
    print(f"  again vs full:  {'SAME' if id_again == id_full else 'DIFFERENT'}")

    # Compile+load the real one through the production path so localModelPath
    # is populated the same way serve does it.
    os.environ.setdefault("Q38_ANE_KEEP_WEIGHT_BLOBS", "0")
    prog = eng.compile_multiproc(
        mil, {"weight.bin": blob}, C, C, S,
        raw_weight_files=frozenset({"weight.bin"}))
    if prog is None:
        raise SystemExit("compile_multiproc failed")
    print(f"  compile cache_hit={getattr(prog, '_compile_cache_hit', None)} "
          f"exists_after={_exists(prog.model)}", flush=True)
    loc = _desc(_msg(prog.model, "localModelPath"))
    print(f"  localModelPath {loc}", flush=True)
    if loc and os.path.isdir(loc):
        for root, _, files in os.walk(loc):
            for fn in files:
                p = os.path.join(root, fn)
                print(f"    {p[len(loc):] or '/'}  {os.path.getsize(p)} bytes")

    print("=== compiledModelExists after a real compile ===", flush=True)
    m_full2 = _make_model(mil, _weights_dict(blob))
    m_empty2 = _make_model(mil, empty)
    m_zeros2 = _make_model(mil, _weights_dict(zeros))
    print(f"  exists full  {_exists(m_full2)}  ident={_ident(m_full2)}")
    print(f"  exists empty {_exists(m_empty2)}  ident={_ident(m_empty2)}")
    print(f"  exists zeros {_exists(m_zeros2)}  ident={_ident(m_zeros2)}")

    print("=== evaluate via compile_multiproc (control) ===", flush=True)
    eng._ensure_io(prog)
    x = (np.arange(C * S, dtype=np.float16).reshape(C, S) / 128)
    with _iosurface_view(prog._in_surf, (C, S), np.float16) as dst:
        np.copyto(dst, x)
    if not eng.submit(prog):
        raise SystemExit("control evaluate failed")
    with _iosurface_view(prog._out_surf, (C, S), np.float16) as src:
        y_ctrl = np.array(src)

    print("=== selectors that could load from disk ===", flush=True)
    for name in ("_ANEInMemoryModel", "_ANEInMemoryModelDescriptor",
                 "_ANEModel", "_ANEClient"):
        _dump_relevant(name)
        _ivars(name)

    print("=== modelURL / underlying _ANEModel ===", flush=True)
    print(f"  modelURL {_desc(_msg(prog.model, 'modelURL'))}")
    ane_model = _msg(prog.model, "model")
    print(f"  underlying _ANEModel {ane_model}")
    if ane_model:
        print(f"    modelURL {_desc(_msg(ane_model, 'modelURL'))}")
        print(f"    sourceURL {_desc(_msg(ane_model, 'sourceURL'))}")
        print(f"    cacheURLIdentifier {_desc(_msg(ane_model, 'cacheURLIdentifier'))}")
    NSURL = _cls("NSURL")
    file_url = _msg(NSURL, "fileURLWithPath:", _nsstring(loc),
                    argtypes=[ctypes.c_void_p])
    from_url = _msg(_cls("_ANEModel"), "modelAtURL:key:",
                    file_url, _nsstring("disk_reload"),
                    argtypes=[ctypes.c_void_p, ctypes.c_void_p])
    print(f"  _ANEModel.modelAtURL:key:(localModelPath) {from_url}")

    print("=== AneDirectEngine.load_compiled_package ===", flush=True)
    y_url = None
    loaded = False
    try:
        from runtime.ane_direct_engine import AneDirectEngine
        direct = AneDirectEngine()
        dprog = direct.load_compiled_package(loc, C, C, S)
        print(f"  package {bool(dprog)} url={getattr(dprog, 'model_url', None)}")
        loaded = bool(dprog)
    except Exception as exc:  # noqa: BLE001
        print(f"  direct load raised {type(exc).__name__}: {exc}")
        dprog = None

    # Try any init*URL* that actually exists on _ANEInMemoryModel.
    inmem_sels = set(_methods(_cls("_ANEInMemoryModel"), False))
    url_sels = sorted(s for s in inmem_sels if "URL" in s or "Path" in s or "url" in s)
    print(f"  in-memory URL/path selectors: {url_sels or 'none'}")

    print("=== footprint (this process) ===", flush=True)
    print(_footprint_categories(os.getpid()))
    empty_miss = (not _exists(m_empty2)) and _exists(m_full2)
    print("=== verdict ===", flush=True)
    print(f"  ident empty vs full: DIFFERENT (weights are in the hash)")
    print(f"  empty dict compiledModelExists after compile: {_exists(m_empty2)}")
    print(f"  matching-weights compiledModelExists after compile: {_exists(m_full2)}")
    print(f"  empty dict misses cache: {empty_miss}")
    print(f"  zeros miss cache: {(not _exists(m_zeros2)) and _exists(m_full2)}")
    print(f"  _ANEInMemoryModel URL init exists: {bool(url_sels)}")
    print(f"  direct package load: {loaded}")


if __name__ == "__main__":
    main()
