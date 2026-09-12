"""What weight-read bandwidth does the ANE actually sustain, and what does
lower precision buy?

Decode on this port is weight-streaming-bound: a pure_step GDN layer carries
~71 M fp16 parameters and takes ~1.8 ms, i.e. ~75 GB/s. That number is only
actionable next to the ceiling. This builds single-matmul programs of a given
weight footprint and times steady-state eval, in fp16 and in palettized /
quantized form at the same shape.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreai_torch import TorchConverter, get_decomp_table  # noqa: E402
from coreai.runtime import (  # noqa: E402
    AIModel, ComputeUnitKind, SpecializationOptions,
)
from runtime.coreai_surfaces import wrap_ndarray  # noqa: E402

CIN = 512


class Proj(nn.Module):
    """One projection, or the same total weight split across ``split`` convs.

    ``elt`` inserts a sigmoid-multiply between stages, mimicking the gate /
    norm traffic in a real GDN layer, so op count can be separated from bytes.
    """

    def __init__(self, cout: int, split: int = 1, elt: bool = False):
        super().__init__()
        self.ws = nn.ModuleList(
            [nn.Conv2d(CIN, max(1, cout // split), 1, bias=False) for _ in range(split)]
        )
        self.elt = elt

    def forward(self, x):
        outs = []
        for w in self.ws:
            y = w(x)
            if self.elt:
                y = y * torch.sigmoid(y)
            outs.append(y)
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)


def _quantize(m, x, bits: int):
    import torch as _t
    from coreai_opt.quantization import (
        Quantizer, QuantizerConfig, ModuleQuantizerConfig, QuantizationSpec,
    )
    from coreai_opt.quantization.spec.granularity import (
        PerChannelGranularity, PerTensorGranularity,
    )
    import os as _os
    dtype = {4: _t.int4, 8: _t.int8}[bits]
    gran = (PerTensorGranularity() if _os.environ.get("BW_PER_TENSOR")
            else PerChannelGranularity(axis=0))
    spec = QuantizationSpec(dtype=dtype, granularity=gran)
    cfg = QuantizerConfig(module_type_configs={
        nn.Conv2d: ModuleQuantizerConfig(op_state_spec={"weight": spec})
    })
    q = Quantizer(m, cfg)
    return q.finalize(q.prepare((x,)))


def _build(cout: int, seq: int, palettize: int | None, split: int = 1, elt: bool = False):
    m = Proj(cout, split, elt).eval().half()
    x = torch.randn(1, CIN, 1, seq, dtype=torch.float16)
    if palettize and palettize < 0:
        m = _quantize(m, x, -palettize)
    elif palettize:
        from coreai_opt.palettization import (
            KMeansPalettizer, KMeansPalettizerConfig,
            ModuleKMeansPalettizerConfig, PalettizationSpec,
        )
        from coreai_opt.palettization.spec.granularity import (
            PerGroupedChannelGranularity,
        )
        spec = PalettizationSpec(
            n_bits=palettize,
            granularity=PerGroupedChannelGranularity(axis=0, group_size=1),
        )
        cfg = KMeansPalettizerConfig(
            module_type_configs={
                nn.Conv2d: ModuleKMeansPalettizerConfig(op_state_spec={"weight": spec})
            }
        )
        p = KMeansPalettizer(m, cfg)
        prepared = p.prepare((x,))
        m = p.finalize(prepared)
    prog = TorchConverter().add_pytorch_module(
        m,
        export_fn=lambda mod: torch.export.export(mod, args=(x,)).run_decompositions(get_decomp_table()),
        input_names=["x"], output_names=["y"],
    ).to_coreai()
    prog.optimize()
    out = Path(tempfile.mkdtemp(prefix="bw_")) / "m.aimodel"
    prog.save_asset(out)
    return out, x


async def run(mb_list, seq: int, palettize: int | None, reps: int,
              split: int = 1, elt: bool = False) -> None:
    ane = [k for k in ComputeUnitKind.available_kinds() if str(k) == "Neural Engine"][0]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
    tag = f"{palettize}-bit palettized" if palettize else "fp16"
    print(f"[{tag}]  in={CIN}  seq={seq}  split={split}  elementwise={elt}")
    for mb in mb_list:
        cout = int((mb * 1024 * 1024) // (CIN * 2))
        try:
            path, x = _build(cout, seq, palettize, split, elt)
        except Exception as exc:  # noqa: BLE001
            print(f"  {mb:4d} MB  build failed: {str(exc)[:90]}")
            continue
        try:
            model = await AIModel.load(str(path), specialization_options=spec)
            fn = model.load_function("main")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mb:4d} MB  load failed: {str(exc)[:90]}")
            continue
        xn = np.ascontiguousarray(x.numpy())
        from runtime.coreai_surfaces import wrap_ndarray
        feeds = {"x": wrap_ndarray(xn)}
        for _ in range(3):
            await fn(feeds)
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            await fn(feeds)
            ts.append(time.perf_counter() - t)
        ms = float(np.median(ts)) * 1e3
        stored = mb * (palettize / 16.0 if palettize else 1.0)
        print(f"  {mb:4d} MB fp16-equiv ({stored:6.1f} MB stored)  {ms:7.3f} ms  "
              f"{stored / 1024 / (ms / 1e3):7.1f} GB/s stored   "
              f"{mb / 1024 / (ms / 1e3):7.1f} GB/s fp16-equivalent")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, nargs="+", default=[16, 64, 136, 272])
    ap.add_argument("--seq", type=int, nargs="+", default=[32])
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--split", type=int, nargs="+", default=[1])
    ap.add_argument("--elementwise", action="store_true")
    a = ap.parse_args()
    for seq in a.seq:
        for split in a.split:
            asyncio.run(run(a.mb, seq, a.bits or None, a.reps, split, a.elementwise))


if __name__ == "__main__":
    main()
