#!/usr/bin/env python3
"""What does `_ANEInMemoryModelDescriptor modelWithNetworkDescription:` eat?

The descriptor class has two factories: modelWithMILText (isMILModel=YES, the
conv-shaped dialect every probe here uses) and modelWithNetworkDescription
(isMILModel=NO -- never probed). ANECompiler embeds an MLIR pipeline with the
mps/mpsx dialects (mps.top_k, mps.sort, mpsx.quantized_gather, ...), and this
path is the candidate feed for them.

Oracle loop: throw candidate texts, print the compiler's FULL complaint. The
parser's own error messages teach the grammar. Run with
Q38_ANE_REUSE_COMPILED=0 so nothing is served from the compile cache.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E  # noqa: E402
from runtime.q38_ane_engine import (  # noqa: E402
    _cls, _desc, _msg, _nsdata, _nsdict, _sel, _objc,
)

MLIR_TOPK = """module {
  func.func @main(%arg0: tensor<1x512xf16>) -> (tensor<1x10xf16>, tensor<1x10xi32>) {
    %vals, %idx = "mps.top_k"(%arg0) {k = 10 : i64} : (tensor<1x512xf16>) -> (tensor<1x10xf16>, tensor<1x10xi32>)
    return %vals, %idx : tensor<1x10xf16>, tensor<1x10xi32>
  }
}
"""

MLIR_TOPK_VER = """module attributes {mps.dialect_version = 0 : i64} {
  func.func @main(%arg0: tensor<1x512xf16>) -> (tensor<1x10xf16>, tensor<1x10xi32>) {
    %vals, %idx = "mps.top_k"(%arg0) {k = 10 : i64} : (tensor<1x512xf16>) -> (tensor<1x10xf16>, tensor<1x10xi32>)
    return %vals, %idx : tensor<1x10xf16>, tensor<1x10xi32>
  }
}
"""

ESPRESSO_PBTXT = """name: "t"
layer {
  name: "in0"
  type: "input"
  top: "data"
  input_param { shape: { dim: 1 dim: 512 } }
}
layer {
  name: "tk"
  type: "top_k"
  bottom: "data"
  top: "vals"
  top: "idx"
  top_k_param { k: 10 axis: 1 }
}
"""

MIL_CONTROL = """program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-component-MILForANE", "3510.2.1"}, {"coremlc-toolkit", "3510.2.1"}})]
{
  func main<ios18>(tensor<fp16, [1, 512, 1, 64]> x) {
    tensor<fp16, [1, 512, 1, 64]> y = relu(x=x)[name=string("y")];
  } -> (y);
}
"""

def _json_net(layers: list, *, storage: str = "") -> str:
    import json
    return json.dumps({
        "storage": storage,
        "analyses": {},
        "properties": {},
        "format_version": 200,
        "metadata_in_weights": [],
        "layers": layers,
    }, indent=2)


SOFTMAX = _json_net([{
    "type": "softmax", "name": "sm", "bottom": "input", "top": "sm",
    "C": 512, "weights": {},
}])

TOPK = _json_net([{
    "type": "top_k", "name": "tk", "bottom": "input", "top": "vals,idx",
    "k": 10, "axis": 1, "weights": {},
}])

TOPK_CAMEL = _json_net([{
    "type": "TopK", "name": "tk", "bottom": "input", "top": "vals",
    "k": 10, "weights": {},
}])

GATHER = _json_net([{
    "type": "gather", "name": "g", "bottom": "input,idx", "top": "g",
    "axis": 1, "weights": {},
}])

SORT = _json_net([{
    "type": "sort", "name": "s", "bottom": "input", "top": "s",
    "axis": 1, "weights": {},
}])

CANDIDATES = [
    ("empty", ""),
    ("mlir_topk", MLIR_TOPK),
    ("mlir_topk_versioned", MLIR_TOPK_VER),
    ("espresso_pbtxt", ESPRESSO_PBTXT),
    ("mil_as_netdesc_control", MIL_CONTROL),
    ("json_softmax", SOFTMAX),
    ("json_topk", TOPK),
    ("json_TopK", TOPK_CAMEL),
    ("json_gather", GATHER),
    ("json_sort", SORT),
]


def try_one(tag: str, text: str, fname: str) -> None:
    desc_cls = _cls("_ANEInMemoryModelDescriptor")
    sel = _sel("modelWithNetworkDescription:weights:optionsPlist:")
    data = _nsdata(text.encode())
    wd = _nsdict({})
    f = _objc.objc_msgSend
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p] * 5
    descriptor = f(desc_cls, sel, data, wd, None)
    if not descriptor:
        print(f"[{tag}] descriptor nil")
        return
    model = _msg(_cls("_ANEInMemoryModel"), "inMemoryModelWithDescriptor:",
                 descriptor, argtypes=[ctypes.c_void_p])
    if not model:
        print(f"[{tag}] model nil")
        return
    local = _desc(_msg(model, "localModelPath"))
    if local and local != "(nil)":
        shutil.rmtree(local, ignore_errors=True)
        os.makedirs(local, exist_ok=True)
        with open(os.path.join(local, fname), "wb") as fh:
            fh.write(text.encode())
        # Espresso JSON nets pair with a .shape sidecar of blob n/k/h/w.
        shape = {
            "layer_shapes": {
                "input": {"n": 1, "k": 512, "h": 1, "w": 1, "_rank": 4},
                "sm": {"n": 1, "k": 512, "h": 1, "w": 1, "_rank": 4},
                "vals": {"n": 1, "k": 10, "h": 1, "w": 1, "_rank": 4},
                "idx": {"n": 1, "k": 10, "h": 1, "w": 1, "_rank": 4},
                "g": {"n": 1, "k": 10, "h": 1, "w": 1, "_rank": 4},
                "s": {"n": 1, "k": 512, "h": 1, "w": 1, "_rank": 4},
            }
        }
        import json
        with open(os.path.join(local, "model.espresso.shape"), "w") as fh:
            json.dump(shape, fh)

    err_ptr = ctypes.c_void_p(0)
    Compile = ctypes.CFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    ok = Compile(("objc_msgSend", _objc))(
        model, _sel("compileWithQoS:options:error:"), 21, None,
        ctypes.byref(err_ptr))
    err = _desc(err_ptr.value) if err_ptr.value else "(no error)"
    print(f"[{tag} as {fname}] compiled={ok}\n    {err[:700]}")


def main() -> None:
    eng = E.AneEngine()
    if not eng.available:
        print("AneEngine unavailable")
        return
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fname = sys.argv[2] if len(sys.argv) > 2 else "model.espresso.net"
    for tag, text in CANDIDATES:
        if which != "all" and tag != which:
            continue
        try:
            try_one(tag, text, fname)
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] exception: {exc!r}")


if __name__ == "__main__":
    main()
