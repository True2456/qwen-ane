"""Verify cross-process reuse of the private ANE compiled-model cache.

Run this file twice.  The first invocation may compile; the second must report
``cache_load_hits=1`` and still execute the identity graph correctly.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get(
    "Q38_ANE_ENGINE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runtime.q38_ane_engine import AneEngine, _BUILD_INFO, _iosurface_view


MIL = f'''program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, 32, 1, 32]> x) {{
    tensor<fp16, [1, 32, 1, 32]> y = identity(x=x)[name=string("y")];
  }} -> (y);
}}
// q38_cross_process_compile_cache_probe_v1
'''


def main() -> None:
    engine = AneEngine()
    program = engine.compile_multiproc(MIL, {}, 32, 32, 32)
    if program is None:
        raise RuntimeError("cache probe failed to compile/load")
    engine._ensure_io(program)
    expected = np.arange(32 * 32, dtype=np.float16).reshape(32, 32) / 128
    with _iosurface_view(program._in_surf, expected.shape, np.float16) as dst:
        np.copyto(dst, expected)
    if not engine.submit(program):
        raise RuntimeError("cache probe evaluation failed")
    with _iosurface_view(program._out_surf, expected.shape, np.float16) as src:
        error = float(np.max(np.abs(np.asarray(src) - expected)))
    result = dict(engine.compile_metrics)
    result["max_abs_error"] = error
    result["status"] = "PASS" if error == 0.0 else "FAIL"
    print(json.dumps(result, indent=2, sort_keys=True))
    if error != 0.0:
        raise RuntimeError(f"cached program output error {error}")


if __name__ == "__main__":
    main()
