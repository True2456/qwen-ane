#!/usr/bin/env python3
"""Does the ANE text-MIL frontend (program 1.3 / ios18) accept the ops
Qwen3.8-Flash-Next routing needs, without dropping to the ANECompiler C API?

The ops on trial: ``top_k`` (expert router top-10 of 512), ``sort``,
``gather`` (expert weight/activation indexing), ``reduce_argmax`` and
``scaled_dot_product_attention``. Text MIL through the runtime's existing
compile path is the only integration surface the engine has, so "does it
compile" is the rung that matters; per the compiler probe's export table the
C layer underneath has LayerDesc + validator symbols for all of these
(_ANECTopKLayerDescInitialize, _ANECValidateTopKLayer, ...), so a rejection
here routes the work to probes/ane_compiler_api.py, not to /dev/null.

Compile-only: descriptor creation does not validate, compileWithQoS does.
Each arm prints ACCEPTED or the compiler's own rejection lines -- the error
text distinguishes "unknown op" (frontend has no such op) from shape/attr
complaints (op exists, spelling wrong).

Env: Q38_ANE_REUSE_COMPILED is forced 0 (a cached artifact from an older
spelling would turn a reject into a silent pass).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

os.environ["Q38_ANE_REUSE_COMPILED"] = "0"

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E  # noqa: E402

eng = E.AneEngine()

S = 64  # tiny; acceptance is decided at compile, not throughput

ARMS: list[tuple[str, str, str]] = [
    # (arm name, op body, return decl)
    # -- controls: these MUST compile or the harness/spelling is broken, and
    #    a rejection above means nothing.
    ("ctrl_relu",
     """    tensor<fp16, [1, 512, 1, SS]> r = relu(x=x)[name=string("r")];""",
     "r"),
    ("ctrl_softmax",
     """    int32 ax1 = const()[name=string("ax1"), val=int32(1)];
    tensor<fp16, [1, 512, 1, SS]> sm = softmax(x=x, axis=ax1)[name=string("sm")];""",
     "sm"),
    ("ctrl_reduce_max",
     """    int32 ax1 = const()[name=string("ax1"), val=int32(1)];
    bool kd = const()[name=string("kd"), val=bool(true)];
    tensor<fp16, [1, 1, 1, SS]> rm = reduce_max(x=x, axes=ax1, keep_dims=kd)[name=string("rm")];""",
     "rm"),
    ("ctrl_concat",
     """    tensor<fp16, [1, 512, 1, SS]> b = identity(x=x)[name=string("b")];
    int32 ax1 = const()[name=string("ax1"), val=int32(1)];
    bool ip = const()[name=string("ip"), val=bool(false)];
    tensor<fp16, [1, 1024, 1, SS]> cc = concat(values=(x, b), axis=ax1, interleave=ip)[name=string("cc")];""",
     "cc"),
    # -- ops on trial
    ("top_k",
     """    int32 kc = const()[name=string("kc"), val=int32(10)];
    int32 ax1 = const()[name=string("ax1"), val=int32(1)];
    bool dsc = const()[name=string("dsc"), val=bool(true)];
    tensor<fp16, [1, 10, 1, SS]> vals, tensor<int32, [1, 10, 1, SS]> idxs = top_k(x=x, k=kc, axis=ax1, descending=dsc)[name=string("tk")];""",
     "vals, idxs"),
    ("sort",
     """    int32 ax1 = const()[name=string("ax1"), val=int32(1)];
    bool dsc = const()[name=string("dsc"), val=bool(true)];
    tensor<fp16, [1, 512, 1, SS]> srt = sort(x=x, axis=ax1, descending=dsc)[name=string("srt")];""",
     "srt"),
    ("argsort",
     """    int32 ax1 = const()[name=string("ax1"), val=int32(1)];
    bool dsc = const()[name=string("dsc"), val=bool(true)];
    tensor<int32, [1, 512, 1, SS]> asrt = argsort(x=x, axis=ax1, descending=dsc)[name=string("asrt")];""",
     "asrt"),
    ("gather",
     """    int32 ax1 = const()[name=string("ax1"), val=int32(1)];
    tensor<int32, [1, 10, 1, 1]> gidx = const()[name=string("gidx"), val=tensor<int32, [1, 10, 1, 1]>([[[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]]])];
    tensor<fp16, [1, 10, 1, SS]> gat = gather(x=x, indices=gidx, axis=ax1)[name=string("gat")];""",
     "gat"),
    ("reduce_argmax",
     """    int32 ax1 = const()[name=string("ax1"), val=int32(1)];
    tensor<int32, [1, 1, 1, SS]> amx = reduce_argmax(x=x, axis=ax1)[name=string("amx")];""",
     "amx"),
    ("sdpa",
     """    tensor<fp16, [1, 512, 1, SS]> k2 = identity(x=x)[name=string("k2")];
    tensor<fp16, [1, 512, 1, SS]> v2 = identity(x=x)[name=string("v2")];
    tensor<fp16, [1, 512, 1, SS]> att = scaled_dot_product_attention(query=x, key=k2, value=v2)[name=string("att")];""",
     "att"),
]


def compile_arm(name: str, body: str, rets: str) -> tuple[bool, str]:
    mil = (
        f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
        f"  func main<ios18>(tensor<fp16, [1, 512, 1, {S}]> x) {{\n"
        + body.replace("SS", str(S))
        + f"\n  }} -> ({rets});\n}}\n"
    )
    buf = io.StringIO()
    ok = False
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(
                mil, {}, 512, 512, S, raw_weight_files=frozenset())
            ok = p is not None
            p = None
            del p
        except Exception as exc:  # noqa: BLE001
            buf.write(f"exception: {exc!r}\n")
    return ok, buf.getvalue().strip()


def main() -> None:
    if not eng._available:
        print("AneEngine unavailable")
        return
    print(f"ANE text-MIL op acceptance, S={S}, Q38_ANE_REUSE_COMPILED=0")
    for name, body, rets in ARMS:
        ok, tail = compile_arm(name, body, rets)
        if ok:
            print(f"  {name:<18} ACCEPTED")
        else:
            lines = [l for l in tail.splitlines() if l.strip()]
            keep = "\n    ".join(lines[-4:]) if lines else "(no compiler output)"
            print(f"  {name:<18} REJECTED\n    {keep}")


if __name__ == "__main__":
    main()
