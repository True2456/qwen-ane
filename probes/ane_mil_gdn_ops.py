#!/usr/bin/env python3
"""Which GDN ops compile in MIL? The go/no-go for porting the layer off Core AI.

Core AI cannot hand the ANE compressed weights (int8 fails ANE codegen,
palettize decompresses), so the 1.8x from int8 projections needs the private
MIL engine. The projection chain is already proven by ane_w8a8_projection.py.
What is NOT proven is the rest of a GDN layer, and the MIL notes record crashes
for `exp`/`softplus` in-graph and for a 2D (rows, 128) recurrence.

This compiles each risky construct on its own, smallest useful shape, so the
unknowns surface in seconds instead of after a 36-layer port. Compile only:
whether the ANE compiler accepts the spelling, not whether it is fast.

The decay term is the one with no obvious fallback:
`pow(sigmoid(-(a + dt)), gamma)` — if `pow` with a const exponent is rejected,
the rewrite goes through `exp`, which is a recorded crash.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E  # noqa: E402
from runtime.q38_ane_engine import AneEngine  # noqa: E402

eng = AneEngine()

HV, DK, DV = 48, 128, 128


def _try(label: str, in_shape, out_shape, body: str, out: str):
    """Compile one tiny program; report whether the ANE compiler took it."""
    ins = ", ".join(str(v) for v in in_shape)
    mil = (
        f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
        f"  func main<ios18>(tensor<fp16, [{ins}]> x) {{\n"
        f"{body}\n  }} -> ({out});\n}}\n"
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(mil, {}, in_shape[1], out_shape[1], in_shape[-1])
        except Exception as exc:  # noqa: BLE001
            p = None
            buf.write(str(exc))
    tail = buf.getvalue().strip().splitlines()
    note = ""
    if p is None:
        hits = [l for l in tail if ("rror" in l or "nvalid" in l or "FAIL" in l)]
        note = (hits[-1] if hits else (tail[-1] if tail else ""))[:150]
    ok = p is not None
    print(f"  {label:34s} {'OK  ' if ok else 'FAIL'}  {note}", flush=True)
    del p
    return ok


S = 32
V4 = (1, HV, DV, DK)          # recurrent state
P4 = (1, HV, 1, DK)           # one token of per-head params
C4 = (1, 2560, 1, S)          # BC1S activations


def main() -> None:
    print("MIL op support for a GDN layer (compile only)\n")

    _try("tanh, BC1S", C4, C4,
         '    tensor<fp16, [1, 2560, 1, %d]> y = tanh(x=x)[name=string("y")];' % S, "y")

    _try("sigmoid, BC1S", C4, C4,
         '    tensor<fp16, [1, 2560, 1, %d]> y = sigmoid(x=x)[name=string("y")];' % S, "y")

    # tanh-SiLU as the Core AI graph spells it: x/2 + (x/2)*tanh(x/2)
    _try("tanh-SiLU chain", C4, C4, f"""    fp16 half = const()[name=string("half"), val=fp16(0.5)];
    tensor<fp16, [1, 2560, 1, {S}]> h = mul(x=x, y=half)[name=string("h")];
    tensor<fp16, [1, 2560, 1, {S}]> t = tanh(x=h)[name=string("t")];
    tensor<fp16, [1, 2560, 1, {S}]> m = mul(x=h, y=t)[name=string("m")];
    tensor<fp16, [1, 2560, 1, {S}]> y = add(x=h, y=m)[name=string("y")];""", "y")

    # the decay term, the one with no cheap fallback
    _try("pow, const scalar exponent", P4, P4, f"""    fp16 g = const()[name=string("g"), val=fp16(0.7)];
    tensor<fp16, [1, {HV}, 1, {DK}]> s = sigmoid(x=x)[name=string("s")];
    tensor<fp16, [1, {HV}, 1, {DK}]> y = pow(x=s, y=g)[name=string("y")];""", "y")

    _try("pow, per-head const tensor", P4, P4, f"""    tensor<fp16, [1, {HV}, 1, 1]> g = const()[name=string("g"), val=tensor<fp16, [1, {HV}, 1, 1]>([{", ".join(["0.7"] * HV)}])];
    tensor<fp16, [1, {HV}, 1, {DK}]> s = sigmoid(x=x)[name=string("s")];
    tensor<fp16, [1, {HV}, 1, {DK}]> y = pow(x=s, y=g)[name=string("y")];""", "y")

    _try("exp (recorded crash)", P4, P4,
         f'    tensor<fp16, [1, {HV}, 1, {DK}]> y = exp(x=x)[name=string("y")];', "y")

    # recurrence pieces on the 4D state shape
    _try("reduce_sum last axis, 4D state", V4, (1, HV, DV, 1), f"""    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([-1])];
    bool kd = const()[name=string("kd"), val=bool(true)];
    tensor<fp16, [1, {HV}, {DV}, 1]> y = reduce_sum(x=x, axes=ax, keep_dims=kd)[name=string("y")];""",
         "y")

    _try("mul broadcast state x scalar", V4, V4, f"""    fp16 k = const()[name=string("k"), val=fp16(0.98)];
    tensor<fp16, [1, {HV}, {DV}, {DK}]> y = mul(x=x, y=k)[name=string("y")];""", "y")

    _try("mul state x per-head-row tensor", V4, V4, f"""    tensor<fp16, [1, {HV}, 1, {DK}]> k = const()[name=string("k"), val=tensor<fp16, [1, {HV}, 1, {DK}]>([{",".join(["0.02"] * (HV * DK))}])];
    tensor<fp16, [1, {HV}, {DV}, {DK}]> y = mul(x=x, y=k)[name=string("y")];""", "y")

    _try("pow(x, -0.5) as rsqrt", P4, P4, f"""    fp16 nh = const()[name=string("nh"), val=fp16(-0.5)];
    tensor<fp16, [1, {HV}, 1, {DK}]> y = pow(x=x, y=nh)[name=string("y")];""", "y")

    _try("rsqrt", P4, P4,
         f'    tensor<fp16, [1, {HV}, 1, {DK}]> y = rsqrt(x=x)[name=string("y")];', "y")

    _try("reduce_mean last axis", P4, (1, HV, 1, 1), f"""    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([-1])];
    bool kd = const()[name=string("kd"), val=bool(true)];
    tensor<fp16, [1, {HV}, 1, 1]> y = reduce_mean(x=x, axes=ax, keep_dims=kd)[name=string("y")];""",
         "y")

    _try("transpose 4D (-1,-2)", V4, (1, HV, DK, DV), f"""    tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];
    tensor<fp16, [1, {HV}, {DK}, {DV}]> y = transpose(x=x, perm=pm)[name=string("y")];""", "y")

    _try("slice last axis (token slot)", C4, (1, 2560, 1, 1), f"""    tensor<int32, [4]> b = const()[name=string("b"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [4]> e = const()[name=string("e"), val=tensor<int32, [4]>([1,2560,1,1])];
    tensor<bool, [4]> bm = const()[name=string("bm"), val=tensor<bool, [4]>([false,false,false,false])];
    tensor<bool, [4]> em = const()[name=string("em"), val=tensor<bool, [4]>([false,false,false,false])];
    tensor<fp16, [1, 2560, 1, 1]> y = slice_by_index(x=x, begin=b, end=e, begin_mask=bm, end_mask=em)[name=string("y")];""",
         "y")

    _try("concat on last axis", C4, (1, 2560, 1, 2 * S), f"""    tensor<fp16, [1, 2560, 1, {2*S}]> y = concat(values=(x, x), axis=int32(-1), interleave=bool(false))[name=string("y")];""",
         "y")


if __name__ == "__main__":
    main()
