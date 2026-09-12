"""Can the private MIL engine take more than one input and output surface?

`_ensure_io` allocated exactly one of each, which capped the MIL path at
single-tensor programs. A GDN layer needs 3 inputs (x, conv, state) and 6
outputs, so this was the blocker for porting a whole layer off Core AI.
`_ANERequest` already takes NSArrays and the loader already queries full symbol
index sets, so only allocation and wrapping were single-tensor.

Two inputs, two outputs, values checked against numpy.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))

import runtime.q38_ane_engine as E  # noqa: E402
from runtime.q38_ane_engine import _iosurface_view  # noqa: E402
from ane_w8a8_projection import eng  # noqa: E402

C, S = 64, 32


def main() -> None:
    mil = (
        f"program(1.3)\n{E._BUILD_INFO}\n{{\n"
        f"  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> a, "
        f"tensor<fp16, [1, {C}, 1, {S}]> b) {{\n"
        f'    tensor<fp16, [1, {C}, 1, {S}]> s = add(x=a, y=b)[name=string("s")];\n'
        f'    tensor<fp16, [1, {C}, 1, {S}]> d = mul(x=a, y=b)[name=string("d")];\n'
        f"  }} -> (s, d);\n}}\n"
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            p = eng.compile_multiproc(mil, {}, C, C, S)
        except Exception as exc:  # noqa: BLE001
            p = None
            buf.write(str(exc))
    if p is None:
        tail = [l for l in buf.getvalue().splitlines() if "rror" in l or "nvalid" in l]
        print(f"  compile FAILED: {(tail[-1] if tail else '')[:110]}")
        return
    print("  compile OK (2 in, 2 out)")

    p.input_elems = [C * S, C * S]
    p.output_elems = [C * S, C * S]
    rng = np.random.default_rng(0)
    a = np.ascontiguousarray((rng.standard_normal((C, S)) * 0.5).astype(np.float16))
    b = np.ascontiguousarray((rng.standard_normal((C, S)) * 0.5).astype(np.float16))
    if not eng._ensure_io(p):
        print("  IO alloc FAILED")
        return
    for surf, val in zip(p._in_surfs, (a, b)):
        with _iosurface_view(surf, (C, S), np.float16) as dst:
            np.copyto(dst, val)
    if not eng.submit(p, procedure_index=0):
        print("  submit FAILED")
        return
    got = []
    for surf in p._out_surfs:
        with _iosurface_view(surf, (C, S), np.float16) as o:
            got.append(np.array(o, np.float32))
    af, bf = a.astype(np.float32), b.astype(np.float32)
    refs = {"s (add)": af + bf, "d (mul)": af * bf}
    def r(g, ref):
        return float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-12))
    print("            surface0   surface1")
    for name, ref in refs.items():
        print(f"  {name:9s} {r(got[0], ref):9.6f}  {r(got[1], ref):9.6f}")
    print("  -> surfaces bind in ALPHABETICAL symbol order (d, s), "
          "not MIL return order (s, d)")


if __name__ == "__main__":
    main()
