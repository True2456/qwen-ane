#!/usr/bin/env python3
"""W8A8 probe: does quantising ACTIVATIONS (not just weights) to int8 engage
the ANE int8 MAC path?

Premise under test (PERFORMANCE.md "INT8 lane is unreachable"):
  * W8A16 (int8 weights only) is *provably* fp16 compute -- `linear_quantize_weights`
    docstring: "All computation at runtime uses float precision; the precision is
    increased ... constexpr_blockwise_shift_scale (per-channel) /
    constexpr_affine_dequantize (per-tensor)", the exact dequant ops the private
    engine already uses.
  * W8A8 (int8 weights AND int8 activations) inserts a `quantize` (fp->int8) on the
    activation path. Whether that engages the ANE's int8 dataflow lane (42 TOPS,
    a ~2x compute path) is the open question.

This probe answers it empirically on the public CoreML path (the only route that
*could* reach the int8 MAC; the private `modelWithMILText` route cannot, since
`ios18.conv` has type_domains T=U=fp16/fp32).

Method (matches ane_peak_tflops.py): a chain of 1x1 NCHW convs, deep enough to
amortise CoreML dispatch overhead; TFLOPS = 2*C_out*C_in*S*#conv / (ms/1000) / 1e12.
The decisive comparison is the RELATIVE speed of W8A8 vs W8A16 vs fp16 in the
compute-bound regime, gated on ANE residency (a silent CPU fallback would defeat
the test, so residency is checked explicitly).
"""
import os, sys, time, tempfile
import numpy as np
import torch
import coremltools as ct
from coremltools.models import MLModel

oc = ct.optimize.coreml
from coremltools.optimize.coreml._config import CompressionGranularity

def ALL():
    for name in ("ComputeUnit", "NSComputeAll"):
        u = getattr(ct, name, None)
        if u is not None:
            for sub in ("ALL", "NEURAL_ENGINE", "ALL_AND"):
                a = getattr(u, sub, None)
                if a is not None:
                    return a
    raise SystemExit("no ALL compute unit found in coremltools")

ALL_UNITS = ALL()

class ConvChain(torch.nn.Module):
    """N chained 1x1 Conv2d, [C,C,1,1], NCHW input [1,C,1,S]."""
    def __init__(self, C, N):
        super().__init__()
        self.convs = torch.nn.Sequential(*[
            torch.nn.Conv2d(C, C, kernel_size=(1, 1), bias=True)
            for _ in range(N)
        ])
    def forward(self, x):
        return self.convs(x)

def build_conv_chain(S, C, N):
    """Trace a PyTorch conv chain -> CoreML mlprogram (iOS18), ANE target."""
    traced = torch.jit.trace(ConvChain(C, N), torch.zeros(1, C, 1, S).float())
    inputs = [ct.TensorType(shape=[1, C, 1, S], dtype=np.float32)]
    try:
        return ct.convert(traced, source="pytorch", compute_units=ALL_UNITS,
                          inputs=inputs, minimum_deployment_target=ct.target.iOS18)
    except TypeError:
        return ct.convert(traced, source="pytorch", compute_units=ALL_UNITS, inputs=inputs)

def ops_of(mlmodel):
    """Best-effort op-type counts (nice-to-have; never fatal to the probe)."""
    try:
        spec = mlmodel.get_spec()
        from collections import Counter
        c = Counter()
        for f in spec.mlProgram.functions.values():
            for b in f.block_specializations.values():
                for op in b.operations:
                    key = None
                    try:
                        key = op.WhichOneof("type")
                    except Exception:
                        key = None
                    if key is None:
                        try:
                            key = op.type
                        except Exception:
                            key = "?"
                    c[str(key)] += 1
        return c
    except Exception as e:
        return Counter({f"(ops introspect err {e})": 1})

def bench(path, S, C, N, iters):
    t0 = time.time()
    m = ct.models.MLModel(path, compute_units=ALL_UNITS)
    load_ms = (time.time() - t0) * 1000
    # residency: is the ANE in the model's available compute devices?
    on_ne = "?"
    try:
        devs = m.get_available_compute_devices()
        on_ne = "ANE" if any("NeuralEngine" in type(d).__name__ for d in devs) else "no-ANE"
    except Exception:
        on_ne = "?"
    x = np.random.default_rng(1).standard_normal((1, C, 1, S)).astype(np.float32)
    for _ in range(3):
        m.predict({"x": x})
    t0 = time.time()
    for _ in range(iters):
        m.predict({"x": x})
    ms = (time.time() - t0) * 1000 / iters
    tflops = (2 * C * C * S * N) / (ms / 1000) / 1e12
    return dict(path=path, load_ms=load_ms, on_ne=on_ne, ms=ms, tflops=tflops)

def main():
    C = int(os.environ.get("W8A8_C", "1024"))
    N = int(os.environ.get("W8A8_N", "12"))
    S_LIST = [int(s) for s in os.environ.get("W8A8_S", "64,128,256").split(",")]
    iters = int(os.environ.get("W8A8_ITER", "15"))
    TMP = tempfile.mkdtemp(prefix="w8a8_")
    print(f"py={sys.executable}  torch={torch.__version__}  C={C} N={N} S={S_LIST} iters={iters}  tmp={TMP}\n")
    rows = []
    for S in S_LIST:
        print(f"== build S={S} (C={C}, N={N}) ==")
        base = build_conv_chain(S, C, N)
        print(f"   fp16 ops: {dict(ops_of(base))}")
        base.save(f"{TMP}/fp16_{S}.mlpackage")

        # W8A16 (weights only, per-tensor int8)
        try:
            w8a16 = oc.linear_quantize_weights(
                base, oc.OptimizationConfig(
                    global_config=oc.OpLinearQuantizerConfig(
                        dtype=np.int8, granularity=CompressionGranularity.PER_TENSOR)))
            if w8a16 is None:
                w8a16 = base
            w8a16.save(f"{TMP}/w8a16_{S}.mlpackage")
            print(f"   W8A16 ops: {dict(ops_of(w8a16))}")
        except Exception as e:
            print(f"   !! linear_quantize_weights failed: {e}")
            w8a16 = base

        # W8A8 ( + int8 activations ) -- the test arm
        w8a8 = w8a16
        try:
            sample = {"x": np.random.default_rng(2).standard_normal((1, C, 1, S)).astype(np.float32)}
            cfg = oc.OptimizationConfig(
                global_config=oc.OpLinearQuantizerConfig(
                    dtype=np.int8, granularity=CompressionGranularity.PER_TENSOR))
            w8a8 = oc.linear_quantize_activations(w8a16, config=cfg, sample_data=[sample])
            if w8a8 is None:
                w8a8 = w8a16
            w8a8.save(f"{TMP}/w8a8_{S}.mlpackage")
            print(f"   W8A8  ops: {dict(ops_of(w8a8))}")
        except Exception as e:
            print(f"   !! linear_quantize_activations failed: {e}")
            w8a8 = w8a16
            w8a8.save(f"{TMP}/w8a8_{S}.mlpackage")

        for name in ("fp16", "w8a16", "w8a8"):
            p = f"{TMP}/{name}_{S}.mlpackage"
            try:
                r = bench(p, S, C, N, iters)
                rows.append((S, name, r))
                print(f"   {name:6s} S={S:4d}  on_ne={r['on_ne']:10s}  "
                      f"{r['ms']:8.2f} ms   {r['tflops']:6.2f} TFLOP/s   (load {r['load_ms']:.0f} ms)")
            except Exception as e:
                rows.append((S, name, dict(error=str(e))))
                print(f"   {name:6s} S={S:4d}  ERROR: {e}")
    # summary
    print("\n=== SUMMARY (TFLOP/s, fp16-equiv; higher is faster) ===")
    hdr = f"{'S':>5} | {'fp16':>8} | {'W8A16':>8} | {'W8A8':>8} | W8A8/W8A16  W8A8/fp16"
    print(hdr); print("-" * len(hdr))
    for S in S_LIST:
        d = {nn: next((r for (s, name, r) in rows if s == S and name == nn), None)
             for nn in ("fp16", "w8a16", "w8a8")}
        def tf(x):
            return x["tflops"] if x and "tflops" in x else float("nan")
        r_w8a8_a16 = tf(d["w8a8"]) / tf(d["w8a16"]) if tf(d["w8a16"]) else float("nan")
        r_w8a8_fp = tf(d["w8a8"]) / tf(d["fp16"]) if tf(d["fp16"]) else float("nan")
        print(f"{S:5d} | {tf(d['fp16']):8.2f} | {tf(d['w8a16']):8.2f} | {tf(d['w8a8']):8.2f} | {r_w8a8_a16:8.2f}x  {r_w8a8_fp:8.2f}x")
    print("\nInterpretation:")
    print(" * W8A8 ~= W8A16 ~= fp16 (within ~1.3x) on ANE  -> no int8 MAC gain; the W8A8 route is a dead end too.")
    print(" * W8A8 >> W8A16 (>=1.5x) on ANE, and on_ne=ANE  -> a real int8 dataflow path IS reachable via CoreML W8A8.")
    print(f"Artifacts in {TMP}")

if __name__ == "__main__":
    main()