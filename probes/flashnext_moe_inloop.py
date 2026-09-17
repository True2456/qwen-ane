"""Instrument the actual stage_generate loop; never extrapolate a microbenchmark.

Example: FLASHNEXT_MOE=mlxresident FLASHNEXT_HEAD=mlx FLASHNEXT_PREFILL_K=16
PYTHONPATH=~/.mlx128/mlx/python python probes/flashnext_moe_inloop.py
--output /tmp/ane-moe-baseline.json --tokens 700 --warmup 100
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=700)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--swift-fixtures", type=Path)
    args = ap.parse_args()
    if args.tokens <= args.warmup:
        ap.error("tokens must exceed warmup")
    free = os.statvfs('/System/Volumes/Data')
    if free.f_bavail * free.f_frsize < 30 * 10**9:
        raise RuntimeError("Need at least 30 GB free before loading ANE assets")

    import mlx.core as mx
    from coreai.runtime import AIModel
    from runtime.flashnext_mlx_moe import ResidentMoe
    import export_flashnext_coreai as decode
    original_apply = ResidentMoe.apply

    rows, samples, residents = [], {}, {}
    counts = {}
    init = ResidentMoe.__init__
    def init_profile(self, layer, *a, **kw):
        init(self, layer, *a, **kw)
        self.profile_layer = layer
        residents[layer] = self
    ResidentMoe.__init__ = init_profile

    def timed_moe(name, original):
        def call(self, x, ids, scores):
            layer = self.profile_layer
            step = counts.get(layer, 0)
            t = time.perf_counter()
            result = original(self, x, ids, scores)
            elapsed = (time.perf_counter() - t) * 1000
            counts[layer] = step + 1
            rows.append(dict(kind=name, layer=layer, step=step, ms=elapsed))
            if args.validate and step in (args.warmup, args.tokens - 1):
                y = result[0] if name == 'moe_shared' else result
                samples[layer, step] = (name, np.array(x), np.array(ids),
                                       np.array(scores), np.array(y))
            return result
        return call
    ResidentMoe.apply = timed_moe('moe_shared', ResidentMoe.apply)
    ResidentMoe.routed_multi = timed_moe('moe_routed', ResidentMoe.routed_multi)

    route = decode.HostMoE._route
    def route_profile(self, x):
        step = counts.get(self.layer, 0)
        t = time.perf_counter()
        result = route(self, x)
        elapsed = (time.perf_counter() - t) * 1000
        rows.append(dict(kind='route', layer=self.layer, step=step, ms=elapsed))
        return result
    decode.HostMoE._route = route_profile

    load = AIModel.load.__func__
    async def load_profile(cls, path, *a, **kw):
        model = await load(cls, path, *a, **kw)
        model.profile_path = Path(path).name
        return model
    AIModel.load = classmethod(load_profile)
    load_fn = AIModel.load_function
    def function_profile(self, *a, **kw):
        fn = load_fn(self, *a, **kw)
        path = self.profile_path
        calls = 0
        async def call(*aa, **kk):
            nonlocal calls
            t = time.perf_counter()
            y = await fn(*aa, **kk)
            elapsed = (time.perf_counter() - t) * 1000
            rows.append(dict(kind='ane', asset=path, step=calls, ms=elapsed))
            calls += 1
            return y
        return call
    AIModel.load_function = function_profile
    if os.environ.get('FLASHNEXT_ANE_WORKER', '0') == '1':
        from runtime.coreai_worker import WorkerAIModel
        worker_load = WorkerAIModel.load.__func__
        async def worker_load_profile(cls, path, *a, **kw):
            model = await worker_load(cls, path, *a, **kw)
            model.profile_path = Path(path).name
            return model
        WorkerAIModel.load = classmethod(worker_load_profile)
        worker_function = WorkerAIModel.load_function
        def worker_function_profile(self, *a, **kw):
            fn = worker_function(self, *a, **kw)
            calls = 0
            async def call(*aa, **kk):
                nonlocal calls
                t = time.perf_counter()
                y = await fn(*aa, **kk)
                rows.append(dict(kind='ane_worker', asset=self.profile_path,
                                 step=calls, ms=(time.perf_counter()-t)*1000))
                calls += 1
                return y
            return call
        WorkerAIModel.load_function = worker_function_profile

    def save():
        summary = {}
        for kind in sorted({r['kind'] for r in rows}):
            vals = [r['ms'] for r in rows if r['kind'] == kind and r['step'] >= args.warmup]
            if vals:
                summary[kind] = dict(n=len(vals), mean_ms=float(np.mean(vals)),
                                     median_ms=float(np.median(vals)), p95_ms=float(np.percentile(vals,95)))
        data = dict(tokens=args.tokens, warmup=args.warmup, environment={k:v for k,v in os.environ.items()
                    if k.startswith('FLASHNEXT_')}, summary=summary, rows=rows, validation=validation,
                    residents={str(i): dict(activation_dtype=str(r.dtype),
                        scale_dtype=str(r.projections[0][1].dtype), nbytes=r.nbytes)
                        for i,r in residents.items()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data))
        print('INLOOP_SUMMARY ' + json.dumps(summary), flush=True)

    validation = []
    try:
        decode.stage_generate(seq=32, max_new=args.tokens, prompt_ids=[760])
    finally:
        save()

    # Correctness checks occur AFTER timing so CPU dequantization cannot keep
    # the GPU warm or perturb the following layer. Dequantize selected experts
    # in NumPy, not with MLX's matmul implementation under test.
    for (layer, step), (name, x, ids, scores, got) in samples.items():
        resident = residents[layer]
        x = x.astype(np.float32).reshape(-1, 2560)
        ids = ids.reshape(-1)
        scores = scores.astype(np.float32).reshape(-1)
        if args.swift_fixtures and layer in (0, 3, 24, 47):
            args.swift_fixtures.mkdir(parents=True, exist_ok=True)
            fixture = dict(x=mx.array(x), scores=mx.array(scores.reshape(1, 1, -1)),
                           expected=mx.array(got.reshape(1, 1, -1)))
            for proj, tensors in zip(('gate_proj', 'up_proj', 'down_proj'), resident.projections):
                for suffix, tensor in zip(('weight', 'scales', 'biases'), tensors):
                    fixture[proj + '.' + suffix] = tensor[mx.array(ids)]
            for suffix, tensor in zip(('gate', 'up', 'down', 'selector'), resident.shared):
                fixture['shared.' + suffix] = tensor
            reference = copy.copy(resident)
            reference.dtype = mx.float32
            reference.projections = [tuple(fixture[proj + '.' + suffix].astype(
                mx.uint32 if suffix == 'weight' else mx.float32)
                for suffix in ('weight', 'scales', 'biases'))
                for proj in ('gate_proj', 'up_proj', 'down_proj')]
            reference.shared = tuple(t.astype(mx.float32) for t in resident.shared)
            optimized, _ = original_apply(reference, x, np.arange(10), scores)
            fixture['expected_fp32'] = mx.array(optimized.reshape(1, 1, -1))
            mx.save_safetensors(str(args.swift_fixtures / f'L{layer}-step{step}.safetensors'), fixture)
        weights = []
        for p in resident.projections:
            arrays = [np.array(v[mx.array(ids)].astype(mx.float32 if j else mx.uint32))
                      for j,v in enumerate(p)]
            packed, scale, bias = arrays
            q = ((packed[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).reshape(
                *packed.shape[:-1], -1).astype(np.float32)
            weights.append((q.reshape(*q.shape[:-1], -1, 64) * scale[..., None]
                            + bias[..., None]).reshape(q.shape))
        g = weights[0] @ x[0]
        u = weights[1] @ x[0]
        act = g / (1 + np.exp(-g)) * u
        expected = ((weights[2] @ act[..., None])[..., 0] * scores[:, None]).sum(0)
        if name == 'moe_shared':
            sg, su, sd, ss = [np.array(v.astype(mx.float32)) for v in resident.shared]
            gate = x @ sg
            expected += (((gate / (1 + np.exp(-gate)) * (x @ su)) @ sd)
                         / (1 + np.exp(-(x @ ss))))[0]
        rel = float(np.linalg.norm(got.reshape(-1)-expected) / max(np.linalg.norm(expected), 1e-12))
        validation.append(dict(layer=layer, step=step, rel_l2=rel,
                               max_abs=float(np.max(np.abs(got.reshape(-1)-expected)))))
        print(f'CPU_REFERENCE L{layer} step={step} relative_L2={rel:.8g}', flush=True)
    save()
    if validation and max(v['rel_l2'] for v in validation) > 0.002:
        raise AssertionError('MoE differs from fp32 CPU reference by >0.2% relative L2')


if __name__ == '__main__':
    main()
