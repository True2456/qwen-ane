"""Numerically verified Core AI GatherMM, static or live expert weights."""
import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from coreai_torch import TorchConverter, get_decomp_table, ExternalizeSpec
from coreai_torch.composite_ops import GatherMM
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions, StorageKind


class Projection(nn.Module):
    def __init__(self, table, live, pretranspose=False):
        super().__init__()
        self.live = live
        self.pretranspose = pretranspose
        self.op = GatherMM(num_batch_axes=0 if pretranspose else 1)
        if not live:
            self.weight = nn.Parameter(table)

    def forward(self, x, ids, table=None):
        w = table if self.live else self.weight
        return self.op(x, w if self.pretranspose else w.transpose(-1, -2), rhs_indices=ids)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=["tiny", "gate", "down"], default="tiny")
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--live", action="store_true")
    p.add_argument("--device", choices=["ane", "gpu", "cpu"], default="ane")
    p.add_argument("--debug-out", type=Path)
    p.add_argument("--composite", action="store_true")
    p.add_argument("--pretranspose", action="store_true")
    p.add_argument("--storage", choices=["bytes", "io_surface", "metal"], default="bytes")
    args = p.parse_args()
    wrap = lambda a: NDArray(a, backing=StorageKind(args.storage))
    e, o, i, k = ((8, 32, 64, 2) if args.shape == "tiny" else
                  (args.experts, 1280, 2560, 10) if args.shape == "gate" else
                  (args.experts, 2560, 640, 10))
    torch.manual_seed(918)
    table = torch.randn(1, e, o, i, dtype=torch.float16) / np.sqrt(i)
    x = torch.randn(1, 1, 1, i, dtype=torch.float16) * .2
    ids = torch.arange(k, dtype=torch.int32).reshape(1, k)
    if args.composite:
        ids = ids.to(torch.uint16)
    if args.pretranspose:
        table = table[0].transpose(-1, -2).contiguous()
        x = x.reshape(1, 1, i)
        ids = ids.reshape(k)
    model = Projection(table, args.live, args.pretranspose).eval()
    inputs = (x, ids, table) if args.live else (x, ids)
    names = ["x", "ids", "table"] if args.live else ["x", "ids"]
    print(f"E={e} out={o} in={i} k={k} live={args.live}", flush=True)
    export_fn = lambda m: torch.export.export(m, args=inputs).run_decompositions(get_decomp_table())
    converter = TorchConverter()
    if args.composite:
        converter.add_pytorch_module(model, export_fn=export_fn,
            externalize_modules=[ExternalizeSpec(GatherMM, "gather_mm", ["num_batch_axes"])],
            input_names=names, output_names=["y"])
    else:
        converter.add_exported_program(export_fn(model), input_names=names, output_names=["y"])
    prog = converter.to_coreai()
    prog.optimize()
    with tempfile.TemporaryDirectory(prefix="flashnext_gmm_") as tmp:
        path = Path(tmp) / "projection.aimodel"
        prog.save_asset(path)
        kinds = {str(v): v for v in ComputeUnitKind.available_kinds()}
        t = time.perf_counter()
        spec = (SpecializationOptions.cpu_only() if args.device == "cpu" else
                SpecializationOptions.from_preferred_compute_unit_kind(
                    kinds["Neural Engine" if args.device == "ane" else "GPU"]))
        print(f"specialization={spec}; allowed={list(map(str, spec.allowed_compute_unit_kinds))}", flush=True)
        if args.debug_out:
            spec = spec.with_debug(enabled=True)
        mm = await AIModel.load(str(path), specialization_options=spec)
        if args.debug_out:
            args.debug_out.write_bytes(mm._debug_infos)
            # Metadata records multiple compiler stages; retain its ordering
            # and report the last residency, rather than counting an earlier
            # attempted ANE assignment as proof of final placement.
            for function in json.loads(mm._debug_infos):
                for op in function.get("operations", []):
                    metadata = op.get("metadatas", [])
                    residence = [m["value"]["string"]["_0"] for m in metadata
                                 if m["key"] == "residency"]
                    reasons = [m["value"]["string"]["_0"] for m in metadata
                               if m["key"] == "ane_validation_message"]
                    if residence or reasons:
                        print(f"placement {op['name']}: history={residence} reasons={reasons}", flush=True)
        fn = mm.load_function("main")
        print(f"load={time.perf_counter()-t:.3f}s", flush=True)
        table_array = wrap(table.numpy()) if args.live else None
        rng = np.random.default_rng(918)
        times, errors = [], []
        for step in range(12):
            if step == 6 and args.live:
                table *= -.5
                table_array = wrap(table.numpy())
            ids_np = rng.choice(e, k, replace=False).astype(
                np.uint16 if args.composite else np.int32).reshape(1, k)
            x_np = rng.normal(0, .2, (1, 1, 1, i)).astype(np.float16)
            if args.pretranspose:
                x_np = x_np.reshape(1, 1, i)
                ids_np = ids_np.reshape(k)
            t = time.perf_counter()
            feeds = {"x": wrap(x_np), "ids": wrap(ids_np)}
            if args.live:
                feeds["table"] = table_array
            result = await fn(feeds)
            got = np.array(result["y"].numpy(), dtype=np.float32).reshape(k, o)
            ms = (time.perf_counter()-t)*1e3
            selected = (table.numpy()[ids_np].transpose(0, 2, 1) if args.pretranspose else
                        table.numpy()[0, ids_np[0]])
            want = selected.astype(np.float32) @ x_np.ravel().astype(np.float32)
            rel = float(np.linalg.norm(got-want) / max(np.linalg.norm(want), 1e-12))
            print(f"step={step} ms={ms:.3f} rel={rel:.6g}", flush=True)
            assert np.isfinite(got).all() and rel < .02, (step, rel)
            errors.append(rel)
            if step >= 2 and step != 6:
                times.append(ms)
        print(f"PASS median={np.median(times):.3f}ms p95={np.percentile(times,95):.3f}ms "
              f"max_rel={max(errors):.6g}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
