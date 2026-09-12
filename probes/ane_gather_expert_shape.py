#!/usr/bin/env python3
"""Can espresso gather_nd compile the real expert slab, not 2560-d embed?

gate_up is [512, 1280, 2560], down is [512, 2560, 640]. Flattening that to
D=32768 died on ANEC_IR serialize — this probe checks whether that is a
width cliff, a rank-2 cliff, or just CoreML's 2D authoring.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _author_nd(table_shape: tuple[int, ...], k: int = 10):
    import coremltools as ct
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder

    ins = [("table", datatypes.Array(*table_shape)), ("ids", datatypes.Array(k))]
    builder = NeuralNetworkBuilder(ins, [("y", None)], disable_rank5_shape_mapping=True)
    builder.add_gather(
        name="lookup", input_names=["table", "ids"], output_name="y", axis=0
    )
    spec = builder.spec
    spec.description.input[1].type.multiArrayType.dataType = (
        ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32
    )
    ml = ct.models.MLModel(spec, compute_units=ct.ComputeUnit.CPU_AND_NE)
    root = Path(ml.get_compiled_model_path())
    net = (root / "model.espresso.net").read_text()
    shape = json.loads((root / "model.espresso.shape").read_text())
    weights = (root / "model.espresso.weights").read_bytes()
    try:
        obj = json.loads(net)
        obj.setdefault("properties", {})["ane_port_gather"] = f"t{table_shape}_k{k}"
        net = json.dumps(obj)
    except json.JSONDecodeError:
        pass
    return net, shape, weights, ml


def try_compile(tag: str, table_shape: tuple[int, ...]) -> None:
    from runtime.ane_lookup import _ane_compile_espresso

    t0 = time.perf_counter()
    net, shape, weights, ml = _author_nd(table_shape)
    t_auth = (time.perf_counter() - t0) * 1e3
    print(f"  [{tag}] authored {t_auth:.0f} ms  espresso.shape={shape}")
    t1 = time.perf_counter()
    _ane_compile_espresso(net, shape, weights)
    print(f"  [{tag}] espresso compile+load {(time.perf_counter()-t1)*1e3:.0f} ms")


def eval_shape(table_shape: tuple[int, ...], k: int = 10) -> None:
    from runtime.ane_lookup import AneGather

    rng = np.random.default_rng(0)
    table = rng.standard_normal(table_shape, dtype=np.float32)
    ids = rng.integers(0, table_shape[0], size=k, dtype=np.int32)
    expect = table[ids]
    print(f"  table {table.nbytes/1e6:.1f} MB  y {expect.nbytes/1e6:.1f} MB")
    g = AneGather(table_shape[0], k=k, table_shape=table_shape)
    t0 = time.perf_counter()
    g.write_table(table)
    print(f"  write_table {(time.perf_counter()-t0)*1e3:.1f} ms")
    y = g.gather(ids, as_float32=True)
    rel = float(np.linalg.norm(y - expect) / (np.linalg.norm(expect) + 1e-12))
    y2 = g.gather((ids + 1) % table_shape[0], as_float32=True)
    print(f"  first rel={rel:.3e}  ids vs ids+1 L2={np.linalg.norm(y-y2):.4g}")
    np.testing.assert_allclose(y, expect.astype(np.float16).astype(np.float32),
                               rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(y2, table[(ids + 1) % table_shape[0]].astype(np.float16),
                               rtol=1e-3, atol=1e-3)
    snapshot = g.gather(ids, as_float32=False)
    saved = snapshot.copy()
    g.gather((ids + 1) % table_shape[0], as_float32=False)
    np.testing.assert_array_equal(snapshot, saved)
    print("  PASS changed indices and retained fp16 result")
    t_host0 = time.perf_counter()
    for _ in range(5):
        _ = np.ascontiguousarray(table[ids], dtype=np.float32)
    host_ms = (time.perf_counter() - t_host0) / 5 * 1e3
    for _ in range(3):
        g.gather(ids, as_float32=False)
    t1 = time.perf_counter()
    n = 10
    for _ in range(n):
        g.gather(ids, as_float32=False)
    ane_ms = (time.perf_counter() - t1) / n * 1e3
    print(f"  host take+f32 {host_ms:.2f} ms   ANE ids+eval+fp16 snapshot {ane_ms:.2f} ms")


def main() -> int:
    cases = [
        ("embed 512x2560", (512, 2560)),
        ("D=16384", (64, 16384)),
        ("D=20480", (64, 20480)),
        ("D=24576", (64, 24576)),
        ("D=28672", (64, 28672)),
        ("D=32768", (64, 32768)),
        ("gu row 64x3276800 skip-alloc-check", (8, 32768)),
        ("3D mini 32x16x2560", (32, 16, 2560)),
        ("3D gu 64x1280x2560", (64, 1280, 2560)),
        ("3D gu 512x1280x2560", (512, 1280, 2560)),
        ("3D dn 512x2560x640", (512, 2560, 640)),
    ]
    which = [a for a in sys.argv[1:] if a != "eval"]
    do_eval = "eval" in sys.argv[1:]
    for tag, shape in cases:
        if which and not any(w in tag or w in str(shape) for w in which):
            continue
        print(f"\n== {tag} {shape} ==")
        try:
            if do_eval:
                eval_shape(shape)
            else:
                try_compile(tag, shape)
        except Exception as exc:
            msg = str(exc).replace("\n", " ")[:400]
            print(f"  FAIL {type(exc).__name__}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
