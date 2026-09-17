#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""export_rindi_package.py - Export a unified .rindi single model package combining Metal GPU 4-bit weights and 64 ANE precompiled layers."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import tools.ane_serve as ane_serve


def export_rindi_package(model_path: str, output_path: str, dense_bits: int = 4, group_size: int = 64):
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ane_dir = out_dir / "ane_layers"
    ane_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  RINDI UNIFIED MODEL EXPORTER")
    print(f"  Source Model:   {model_path}")
    print(f"  Target Package: {output_path}")
    print("=" * 65)

    # 1. Load Base Model
    print("\n[1/4] Loading base model...")
    t0 = time.time()
    model, tok = load(model_path)
    lm = getattr(model, "language_model", model)
    inner = lm.model
    print(f"  ✓ Base model loaded in {time.time()-t0:.2f}s")

    # 2. Quantize & Export GPU Prefill Backbone
    print("\n[2/4] Quantizing & exporting Metal GPU 4-bit prefill backbone...")
    t0 = time.time()
    gpu_inner = copy.deepcopy(inner)
    nn.quantize(gpu_inner, group_size=group_size, bits=dense_bits)
    mx.eval(gpu_inner.parameters())

    gpu_weights_path = out_dir / "gpu_backbone.safetensors"
    flat_params = dict(tree_flatten(gpu_inner.parameters()))
    mx.save_safetensors(str(gpu_weights_path), flat_params)
    print(f"  ✓ Saved GPU backbone ({gpu_weights_path.stat().st_size / 1e9:.2f} GB) in {time.time()-t0:.2f}s")

    # 3. Export ANE Prebaked Layer Blobs
    print("\n[3/4] Packaging 64 Apple Neural Engine precompiled layer programs...")
    t0 = time.time()
    cache_dir = Path(ane_serve._bake_cache_dir(model_path, dense_bits))
    if cache_dir.exists():
        blob_count = 0
        for f in cache_dir.glob("*.bin"):
            shutil.copy2(f, ane_dir / f.name)
            blob_count += 1
        print(f"  ✓ Copied {blob_count} precompiled ANE layer blobs into package in {time.time()-t0:.2f}s")
    else:
        print("  Compiling ANE layers from scratch...")
        ane_serve.attach_ane_chain(model, "mil", 32, dense_bits, str(ane_dir))

    # 4. Copy Tokenizer & Config Files
    print("\n[4/4] Writing unified package manifest & tokenizer...")
    src_dir = Path(model_path)
    for cfg_file in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"):
        src_file = src_dir / cfg_file
        if src_file.exists():
            shutil.copy2(src_file, out_dir / cfg_file)

    # Write Rindi Manifest
    manifest = {
        "format": "rindi-v1",
        "model_name": Path(model_path).name,
        "dense_bits": dense_bits,
        "group_size": group_size,
        "num_layers": len(inner.layers),
        "hidden_dim": inner.embed_tokens.weight.shape[1],
        "created_at": time.time(),
        "gpu_backbone": "gpu_backbone.safetensors",
        "ane_layers_dir": "ane_layers",
    }
    with open(out_dir / "rindi_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print("\n" + "=" * 65)
    print(f"  ✓ UNIFIED RINDI PACKAGE CREATED: {output_path}")
    print(f"  Total Package Size: {total_size / 1e9:.2f} GB")
    print("=" * 65)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export a unified .rindi single model package")
    p.add_argument("--model", default=str(Path.home() / ".lmstudio/models/Qwen/Qwen3.8-27B"), help="Source base model directory")
    p.add_argument("--output", default=str(Path.home() / ".lmstudio/models/Qwen/Qwen3.8-27B.rindi"), help="Output .rindi package path")
    p.add_argument("--bits", type=int, default=4, help="Quantization precision bits")
    args = p.parse_args()

    export_rindi_package(args.model, args.output, dense_bits=args.bits)
