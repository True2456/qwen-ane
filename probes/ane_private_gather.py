#!/usr/bin/env python3
"""Execute ANE gather via the private lookup path.

Text MIL rejects `gather`. The compiler still has `_ANECGatherLayer` and
shipping ANE nets implement embedding lookup as espresso
`inner_product { is_lookup: 1 }` (see ane_embeddings.espresso.net).

This probe compiles that gather three ways and checks whether it actually
runs on Neural Engine:

  embedding_nd   CoreML NeuralNetwork embeddingND (lowers to lookup)
  mil_gather     CoreML MIL `mb.gather` along axis 0
  espresso       handmade espresso JSON `inner_product` is_lookup=1

MoE-shaped arm: vocab=512 experts, embedding=2560, k=10 indices.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _np_take(table_oc_v: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """table (embedding, vocab) + ids (...) -> (..., embedding)."""
    return np.take(table_oc_v.T, ids.astype(np.int64), axis=0)


def _print_plan(model, tag: str) -> None:
    try:
        from coremltools.models.compute_plan import MLComputePlan
        from coremltools.models.compute_device import (
            MLCPUComputeDevice,
            MLGPUComputeDevice,
            MLNeuralEngineComputeDevice,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [{tag}] compute plan import failed: {exc!r}")
        return
    path = model.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(path)
    struct = plan.model_structure

    def device_name(d) -> str:
        if d is None:
            return "None"
        t = type(d).__name__
        if "NeuralEngine" in t or "ANE" in t:
            return "NeuralEngine"
        if "GPU" in t:
            return "GPU"
        if "CPU" in t:
            return "CPU"
        return f"{t}:{d!r}"[:120]

    printed = 0
    nn = getattr(struct, "neuralnetwork", None) or getattr(struct, "neural_network", None)
    if nn is not None:
        for layer in nn.layers:
            usage = plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
            print(f"  [{tag}] nn {layer.type} {layer.name!r} usage={usage!r}")
            if usage is not None:
                pref = device_name(getattr(usage, "preferred_compute_device", None))
                supported = [device_name(x) for x in (getattr(usage, "supported_compute_devices", None) or [])]
                print(f"  [{tag}] preferred={pref} supported={supported}")
            printed += 1
    if getattr(struct, "program", None) is not None:
        fn = struct.program.functions.get("main")
        if fn is not None:
            for op in fn.block.operations:
                if op.operator_name in ("const", "cast", "identity", "reshape"):
                    continue
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                pref = device_name(usage.preferred_compute_device) if usage else "?"
                print(f"  [{tag}] mil {op.operator_name} preferred={pref}")
                printed += 1
                if printed >= 12:
                    break
    if printed == 0:
        print(f"  [{tag}] no plan ops (path={path})")


def run_embedding_nd(vocab: int, dim: int, k: int, *, nbits: int | None = None, cpu_only: bool = False) -> dict:
    import coremltools as ct
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder

    rng = np.random.default_rng(0)
    W = rng.standard_normal((dim, vocab), dtype=np.float32).astype(np.float32)
    ids = rng.integers(0, vocab, size=(k, 1), dtype=np.int32)
    expect = _np_take(W, np.squeeze(ids, axis=-1))

    builder = NeuralNetworkBuilder(
        [("ids", datatypes.Array(k, 1))],
        [("y", None)],
        disable_rank5_shape_mapping=True,
    )
    builder.add_embedding_nd(
        name="lookup",
        input_name="ids",
        output_name="y",
        vocab_size=vocab,
        embedding_size=dim,
        W=W,
    )
    spec = builder.spec
    spec.description.input[0].type.multiArrayType.dataType = (
        ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32
    )
    units = ct.ComputeUnit.CPU_ONLY if cpu_only else ct.ComputeUnit.CPU_AND_NE
    model = ct.models.MLModel(spec, compute_units=units)
    if nbits is not None:
        from coremltools.models.neural_network import quantization_utils
        model = quantization_utils.quantize_weights(model, nbits=nbits)
    tag = f"embedding_nd V={vocab} D={dim} K={k} nbits={nbits} cpu={cpu_only}"
    try:
        _print_plan(model, f"embedding_nd V={vocab} D={dim} K={k}")
    except Exception as exc:  # noqa: BLE001
        print(f"  embedding_nd plan failed: {exc!r}")

    # warmup + timed
    feed = {"ids": ids}
    for _ in range(3):
        out = model.predict(feed)["y"]
    t0 = time.perf_counter()
    for _ in range(20):
        out = model.predict(feed)["y"]
    ms = (time.perf_counter() - t0) / 20 * 1e3
    got = np.asarray(out, dtype=np.float32)
    raw_shape = tuple(got.shape)
    candidates = {
        "reshape": np.reshape(got, expect.shape) if got.size == expect.size else None,
        "T_reshape": np.reshape(got.T, expect.shape) if got.size == expect.size else None,
    }
    best_name, best_rel, best = "none", 1e9, got
    for name, arr in candidates.items():
        if arr is None:
            continue
        rel = float(np.linalg.norm(arr - expect) / (np.linalg.norm(expect) + 1e-12))
        if rel < best_rel:
            best_name, best_rel, best = name, rel, arr
    print(
        f"  embedding_nd: {ms:.3f} ms  rel={best_rel:.3e} ({best_name})  "
        f"raw{raw_shape} expect{expect.shape} mean={got.mean():.4f}/{expect.mean():.4f}"
    )
    if vocab <= 64 or best_rel > 1e-2:
        print(f"    got[:2,:4]={got.reshape(got.size)[:8]} expect[:2,:4]={expect.reshape(-1)[:8]}")
        try:
            dump_compiled_espresso(model, f"embedding_nd_{vocab}x{dim}")
        except Exception as exc:  # noqa: BLE001
            print(f"  dump compiled failed: {exc!r}")
    ok_tol = 5e-2 if nbits == 8 else 1e-3
    return {"ms": ms, "rel": best_rel, "ok": best_rel < ok_tol}


def run_mil_gather(vocab: int, dim: int, k: int) -> dict:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    rng = np.random.default_rng(1)
    table = rng.standard_normal((vocab, dim), dtype=np.float32)
    ids = rng.integers(0, vocab, size=(k,), dtype=np.int32)
    expect = table[ids]

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(vocab, dim), dtype=types.fp16),
            mb.TensorSpec(shape=(k,), dtype=types.int32),
        ]
    )
    def prog(x, idx):
        return mb.gather(x=x, indices=idx, axis=0)

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    _print_plan(model, f"mil_gather V={vocab} D={dim} K={k}")
    feed = {"x": table.astype(np.float16), "idx": ids}
    for _ in range(3):
        out = model.predict(feed)
    key = [k for k in out if k != "idx"][0] if "y" not in out else "y"
    # convert names: last output
    key = list(out.keys())[-1]
    t0 = time.perf_counter()
    for _ in range(20):
        got = np.asarray(model.predict(feed)[key], dtype=np.float32)
    ms = (time.perf_counter() - t0) / 20 * 1e3
    got = np.reshape(got, expect.shape)
    rel = float(np.linalg.norm(got - expect) / (np.linalg.norm(expect) + 1e-12))
    print(f"  mil_gather: {ms:.3f} ms  rel={rel:.3e}  out{got.shape} key={key}")
    return {"ms": ms, "rel": rel, "ok": rel < 2e-2}


def run_espresso_lookup() -> None:
    """Handmade espresso inner_product is_lookup through the private descriptor."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ane_netdesc_probe", _REPO / "probes" / "ane_netdesc_probe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import runtime.q38_ane_engine as E
    E.AneEngine()
    vocab, dim = 64, 64
    net = json.dumps({
        "storage": "",
        "analyses": {},
        "properties": {},
        "format_version": 200,
        "metadata_in_weights": [],
        "layers": [{
            "type": "inner_product",
            "name": "lookup",
            "bottom": "input",
            "top": "y",
            "is_lookup": 1,
            "nB": vocab,
            "nC": dim,
            "has_biases": 0,
            "has_relu": 0,
            "has_tanh": 0,
            "has_prelu": 0,
            "quantization_mode": 0,
            "nd_mode": True,
            "debug_info": "",
            "weights": {},
        }],
    })
    mod.try_one("json_lookup", net, "model.espresso.net")


def dump_compiled_espresso(model, tag: str) -> None:
    """If CoreML compiled an espresso net, print the lookup layer."""
    path = model.get_compiled_model_path()
    print(f"  [{tag}] compiled at {path}")
    for p in Path(path).rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(s in name for s in ("espresso", "ane", "hwx", "mil", "plist")):
            print(f"    {p.relative_to(path)}  {p.stat().st_size}B")
    for p in Path(path).rglob("*.espresso.net"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            print(f"    {p.name}: not json")
            continue
        types = {}
        lookup = []
        for L in d.get("layers", []):
            t = L.get("type")
            types[t] = types.get(t, 0) + 1
            if L.get("is_lookup") or t in ("gather", "gather_nd", "embedding"):
                lookup.append(
                    {
                        k: L[k]
                        for k in L
                        if k in (
                            "type",
                            "name",
                            "is_lookup",
                            "nB",
                            "nC",
                            "bottom",
                            "top",
                            "quantization_mode",
                        )
                    }
                )
        print(f"    espresso types {types}")
        print(f"    lookup layers {lookup}")
        if p.stat().st_size < 2000:
            print("    net", p.read_text().replace("\n", " ")[:400])


def main() -> int:
    print("== one-program live gather (AneLookup) ==")
    from runtime.ane_lookup import AneLookup, PagedLookup

    rng = np.random.default_rng(0)
    dim, bank, k = 2560, 64, 10
    W = rng.standard_normal((dim, 512), dtype=np.float32)
    ids = rng.integers(0, 512, size=k, dtype=np.int32)

    lu = AneLookup(vocab=bank, dim=dim, k=k)
    dump_compiled_espresso(lu._model, "live_gather")
    try:
        _print_plan(lu._model, "live_gather")
    except Exception as exc:  # noqa: BLE001
        print(f"  plan failed: {exc!r}")

    # page bank 0, then bank 1 — same compiled program
    lu.write_table(W[:, :bank])
    got0 = lu.gather(np.arange(k, dtype=np.int32))
    exp0 = W.T[np.arange(k)]
    rel0 = float(np.linalg.norm(got0 - exp0) / (np.linalg.norm(exp0) + 1e-12))
    lu.write_table(W[:, bank : 2 * bank])
    got1 = lu.gather(np.arange(k, dtype=np.int32))
    exp1 = W.T[bank + np.arange(k)]
    rel1 = float(np.linalg.norm(got1 - exp1) / (np.linalg.norm(exp1) + 1e-12))
    t0 = time.perf_counter()
    for _ in range(20):
        lu.gather(np.arange(k, dtype=np.int32))
    ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"  AneLookup page0 rel={rel0:.3e} page1 rel={rel1:.3e}  {ms:.3f} ms")

    print("\n== PagedLookup 512x2560 k=10 (one program) ==")
    paged = PagedLookup(W, k=k)
    t0 = time.perf_counter()
    for _ in range(5):
        got = paged.gather(ids)
    ms = (time.perf_counter() - t0) / 5 * 1e3
    expect = W.T[ids]
    rel = float(np.linalg.norm(got - expect) / (np.linalg.norm(expect) + 1e-12))
    print(f"  PagedLookup: {ms:.3f} ms  rel={rel:.3e}  programs=1")
    ok = rel0 < 1e-3 and rel1 < 1e-3 and rel < 1e-3
    print(f"\nok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
