#!/usr/bin/env python3
"""Can ANE do indexed expert GEMM if we spell it as Conv2d, not GatherMM?

Tiny shapes only (E=8, O=32, I=64, k=2, S=32). GPU-preferred first.
Each arm is a child process because ANE abort kills the interpreter.

Arms
  dense_baked   Conv2d(I → E*O), weights baked. Control: all-expert conv on ANE.
  live_weight   F.conv2d(x, w) with w as an input. Control: dynamic conv weights.
  onehot        host one-hot @ table, then conv. Gather as dense matmul.
  index_select  table.index_select(ids), then conv. True in-graph gather.
  espresso_fused espresso gather_nd + conv (compile only).
  mil_gather     text MIL gather (compile only).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E, O, I, K, S = 8, 32, 64, 2, 32


def _placement_lines(debug_bytes: bytes) -> list[str]:
    lines = []
    try:
        blob = json.loads(debug_bytes)
    except Exception as exc:  # noqa: BLE001
        return [f"debug json parse failed: {exc}"]
    for function in blob:
        if function.get("identifier") != "MPSGraph_optimized_op_id":
            continue
        for op in function.get("operations", []):
            md = op.get("metadatas", [])
            res = [m["value"]["string"]["_0"] for m in md if m.get("key") == "residency"]
            reasons = [
                m["value"]["string"]["_0"]
                for m in md if m.get("key") == "ane_validation_message"
            ]
            stacks = []
            for m in md:
                if m.get("key") == "call_stack":
                    stacks.append(
                        [s["string"]["_0"] for s in m["value"]["array"]["_0"]]
                    )
            src = []
            for m in md:
                if m.get("key") != "sources":
                    continue
                try:
                    for d in m["value"]["array"]["_0"]:
                        ident = d["dictionary"]["_0"]
                        nm = ident.get("name", {}).get("string", {}).get("_0")
                        ids = ident.get("identifiers", {}).get("array", {}).get("_0", [])
                        src.append((nm, [x["string"]["_0"] for x in ids]))
                except Exception:
                    pass
            if res or reasons:
                last = res[-1] if res else "?"
                lines.append(
                    f"  {op.get('name')}: last={last} history={res} "
                    f"reasons={reasons} src={src[:2]} stack={stacks[:1]}"
                )
    return lines or ["  (no residency tags)"]


def _rel(got, want) -> float:
    import numpy as np
    g = np.asarray(got, np.float32).reshape(-1)
    w = np.asarray(want, np.float32).reshape(-1)
    return float(np.linalg.norm(g - w) / max(np.linalg.norm(w), 1e-12))


async def _run_coreai(arm: str) -> None:
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from coreai_torch import TorchConverter, get_decomp_table
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    torch.manual_seed(918)
    rng = np.random.default_rng(918)
    x = torch.randn(1, I, 1, S, dtype=torch.float16) * 0.2
    table = torch.randn(E, O, I, dtype=torch.float16) / (I ** 0.5)
    ids_i64 = torch.tensor([0, 3], dtype=torch.int64)
    ids = ids_i64.to(torch.int32)
    oh = F.one_hot(ids_i64, num_classes=E).to(torch.float16)
    w_live = table.index_select(0, ids_i64).reshape(K * O, I, 1, 1)

    class DenseBaked(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(I, E * O, 1, bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(table.reshape(E * O, I, 1, 1))

        def forward(self, x):
            return self.conv(x)

    class LiveWeight(nn.Module):
        def forward(self, x, w):
            return F.conv2d(x, w)

    class OneHotConv(nn.Module):
        def forward(self, x, table, oh):
            selected = torch.matmul(oh, table.reshape(E, O * I))
            return F.conv2d(x, selected.reshape(K * O, I, 1, 1))

    class IndexSelectConv(nn.Module):
        def forward(self, x, table, ids):
            selected = table.index_select(0, ids.to(torch.int64))
            return F.conv2d(x, selected.reshape(K * O, I, 1, 1))

    if arm == "dense_baked":
        model = DenseBaked().eval().half()
        inputs, names = (x,), ["x"]
        want = model(x)
    elif arm == "live_weight":
        model = LiveWeight().eval().half()
        inputs, names = (x, w_live), ["x", "w"]
        want = model(x, w_live)
    elif arm == "onehot":
        model = OneHotConv().eval().half()
        inputs, names = (x, table, oh), ["x", "table", "oh"]
        want = model(x, table, oh)
    elif arm == "index_select":
        model = IndexSelectConv().eval().half()
        inputs, names = (x, table, ids), ["x", "table", "ids"]
        want = model(x, table, ids)
    else:
        raise SystemExit(f"unknown coreai arm {arm}")

    with torch.no_grad():
        want_np = want.detach().float().numpy()
    ep = torch.export.export(model, args=inputs).run_decompositions(get_decomp_table())
    prog = (
        TorchConverter()
        .add_exported_program(ep, input_names=list(names), output_names=["y"])
        .to_coreai()
    )
    prog.optimize()
    with tempfile.TemporaryDirectory(prefix=f"idxconv_{arm}_") as tmp:
        path = Path(tmp) / "m.aimodel"
        prog.save_asset(path)
        kinds = {str(k): k for k in ComputeUnitKind.available_kinds()}
        feeds = {n: NDArray(t.detach().cpu().numpy()) for n, t in zip(names, inputs)}
        for label, key in (("gpu", "GPU"), ("ane", "Neural Engine")):
            spec = SpecializationOptions.from_preferred_compute_unit_kind(kinds[key])
            if label == "ane":
                spec = spec.with_debug(enabled=True)
            t0 = time.perf_counter()
            mm = await AIModel.load(str(path), specialization_options=spec)
            fn = mm.load_function("main")
            out = await fn(feeds)
            got = np.asarray(out["y"].numpy(), dtype=np.float32)
            for _ in range(3):
                await fn(feeds)
            t1 = time.perf_counter()
            n = 10
            for _ in range(n):
                await fn(feeds)
            ms = (time.perf_counter() - t1) / n * 1e3
            print(
                f"{arm} {label}-preferred load={time.perf_counter()-t0:.2f}s "
                f"median={ms:.3f}ms rel={_rel(got, want_np):.3e} y{got.shape} "
                f"spec_allowed={[str(x) for x in spec.allowed_compute_unit_kinds]}",
                flush=True,
            )
            if label == "ane":
                for line in _placement_lines(mm._debug_infos):
                    print(line, flush=True)


def _espresso_fused() -> None:
    sys.path.insert(0, str(ROOT))
    from runtime.ane_lookup import AneGatherMM
    AneGatherMM(E, O, I, K, method="conv")
    print("espresso_fused compiled (unexpected)", flush=True)


def _mil_gather() -> None:
    sys.path.insert(0, str(ROOT))
    import runtime.q38_ane_engine as Ene
    mil = f"""program(1.3)
{Ene._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {I}, 1, {S}]> x, tensor<fp16, [1, {E}, 1, {I}]> table, tensor<int32, [2]> ids) {{
    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([1])];
    tensor<fp16, [1, 2, 1, {I}]> gat = gather(x=table, indices=ids, axis=ax)[name=string("gat")];
  }} -> (gat);
}}
"""
    mil_data = Ene._nsdata(mil.encode("utf-8"))
    empty = Ene._msg(Ene._cls("NSDictionary"), "dictionary")
    import ctypes
    Make = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    )
    descriptor = Make(("objc_msgSend", Ene._objc))(
        Ene._cls("_ANEInMemoryModelDescriptor"),
        Ene._sel("modelWithMILText:weights:optionsPlist:"),
        mil_data, empty, None,
    )
    print(f"mil_gather descriptor={'ok' if descriptor else 'nil'}", flush=True)
    if not descriptor:
        return
    InMem = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )
    model = InMem(("objc_msgSend", Ene._objc))(
        Ene._cls("_ANEInMemoryModel"),
        Ene._sel("inMemoryModelWithDescriptor:"),
        descriptor,
    )
    print(f"mil_gather inmem={'ok' if model else 'nil'}", flush=True)
    if not model:
        return
    local = Ene._desc(Ene._msg(model, "localModelPath"))
    if local and local != "(nil)":
        import shutil
        shutil.rmtree(local, ignore_errors=True)
        os.makedirs(os.path.join(local, "weights"), exist_ok=True)
        with open(os.path.join(local, "model.mil"), "wb") as f:
            f.write(mil.encode("utf-8"))
    err_ptr = ctypes.c_void_p(0)
    Compile = ctypes.CFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    )
    ok = Compile(("objc_msgSend", Ene._objc))(
        model, Ene._sel("compileWithQoS:options:error:"),
        21, None, ctypes.byref(err_ptr),
    )
    err = Ene._desc(err_ptr.value) if err_ptr.value else ""
    print(f"mil_gather compile={ok} err={err[:300]}", flush=True)


def _child(arm: str) -> None:
    if arm == "espresso_fused":
        _espresso_fused()
        return
    if arm == "mil_gather":
        _mil_gather()
        return
    asyncio.run(_run_coreai(arm))


def _run_all() -> int:
    py = sys.executable
    arms = [
        "dense_baked",
        "live_weight",
        "onehot",
        "index_select",
        "espresso_fused",
        "mil_gather",
    ]
    rc = 0
    for arm in arms:
        print(f"\n== {arm} ==", flush=True)
        p = subprocess.run(
            [py, str(Path(__file__).resolve()), "--arm", arm],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = (p.stdout + p.stderr).strip()
        if p.returncode < 0:
            print(f"SIGNAL {-p.returncode}\n{out[-2000:]}", flush=True)
            rc = 1
        elif p.returncode != 0:
            print(f"exit {p.returncode}\n{out[-3000:]}", flush=True)
            rc = 1
        else:
            print(out, flush=True)
    return rc


def main() -> int:
    os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=[
        "dense_baked", "live_weight", "onehot", "index_select",
        "espresso_fused", "mil_gather", "all",
    ], default="all")
    args = p.parse_args()
    if args.arm == "all":
        return _run_all()
    _child(args.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
