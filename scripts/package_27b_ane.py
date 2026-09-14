#!/usr/bin/env python3
"""Package and upload Qwen3.8-27B-ANE to Hugging Face as a standalone ~14 GB package."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.numpy as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runtime"))

from pure_ane import Checkpoint, PureAneRuntime

DEFAULT_SOURCE = Path.home() / ".lmstudio" / "models" / "Qwen" / "Qwen3.8-27B"
DEFAULT_CACHE = Path.home() / "Library/Caches/q38-pure-ane/4edf80a2689e5c6b2ff7"
DEFAULT_MANIFEST = Path("/tmp/qwen27b_manifest.json")
DEFAULT_DEST = Path.home() / ".qwenANE" / "models" / "27b"
DEFAULT_REPO = "True2456/Qwen3.8-27B-ANE"

README_CONTENT = """---
license: apache-2.0
tags:
- apple-silicon
- ane
- apple-neural-engine
- qwen
- qwen3.8
- pure-ane
---

# Qwen3.8-27B-ANE (Apple Neural Engine Standalone Package)

High-performance, pure on-chip inference package for **Qwen3.8-27B** running natively on the Apple Neural Engine (ANE).

This package contains:
- **`model.safetensors` (5.09 GB)**: Host embeddings, RMSNorms, and LM output head.
- **`quant_cache/` (9.15 GB)**: Pre-quantized INT4 weight blobs & scales for all 64 layers (48 GDN + 16 Attention) along with `manifest.json`.
- **Tokenizers & Configs**: Complete configuration and tokenizer files.

Total package size is **~14.2 GB** (compared to 52 GB for base BF16 weights), ready to run immediately without compilation stalls or external weights.

## 🚀 Quick Start with `qwen-ane`

Install the unified CLI:
```bash
git clone https://github.com/True2456/Rindi.git
cd Rindi
pip install -e .
```

### Interactive Chat (Apple Foundation Models 'fm' Style)
```bash
qwen-ane chat -model 27b -ctx 4096
```

### OpenAI-Compatible Serving
```bash
qwen-ane serve -model 27b -port 2457 -ctx 4096
```

### Connect with Pi Coding Agent
```bash
pi -e extensions/pure27-pi.ts --provider pure27 --model Qwen3.8-27B
```
"""


def package_27b(
    source_model: Path,
    cache_dir: Path,
    manifest_path: Path,
    out_dir: Path,
    skip_verify: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    quant_dir = out_dir / "quant_cache"
    quant_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  QWEN3.8-27B-ANE STANDALONE PACKAGER")
    print(f"  Source Model:   {source_model}")
    print(f"  Quant Cache:    {cache_dir}")
    print(f"  Target Output:  {out_dir}")
    print("=" * 65)

    # 1. Copy config and tokenizer files
    print("\n[1/5] Copying tokenizer, config, and jinja templates...")
    copy_files = [
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    for fn in copy_files:
        src = source_model / fn
        if src.exists():
            shutil.copy2(src, out_dir / fn)
            print(f"  ✓ {fn}")

    # 2. Extract host tensors into model.safetensors
    st_path = out_dir / "model.safetensors"
    staging_st = Path("/tmp/qwen27b_staging/model.safetensors")
    if st_path.is_file() and st_path.stat().st_size > 4 * 1024 * 1024 * 1024:
        print(f"\n[2/5] Using existing host tensors at {st_path} ({st_path.stat().st_size / 1e9:.2f} GB)")
    elif staging_st.is_file() and staging_st.stat().st_size > 4 * 1024 * 1024 * 1024:
        print(f"\n[2/5] Linking staged host tensors -> {st_path}...")
        try:
            os.link(staging_st, st_path)
        except OSError:
            shutil.copy2(staging_st, st_path)
        print(f"  ✓ Saved host model.safetensors ({st_path.stat().st_size / 1e9:.2f} GB)")
    else:
        print("\n[2/5] Extracting host tensors from source checkpoint...")
        ckpt = Checkpoint(source_model)
        with open(manifest_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tensors = {}
        for name in meta["host_tensors"]:
            tensors[name] = ckpt.tensor(name, np.float16)
        embed_name = ckpt.embedding_name
        tensors[embed_name] = ckpt.tensor(embed_name, np.float16)
        if "lm_head.weight" not in tensors:
            tensors["lm_head.weight"] = ckpt.tensor("lm_head.weight", np.float16)
        sf.save_file(tensors, str(st_path))
        print(f"  ✓ Extracted and saved {len(tensors)} host tensors ({st_path.stat().st_size / 1e9:.2f} GB)")

    # 3. Link/copy 500 quant cache files
    print("\n[3/5] Linking 500 prebaked INT4 weight blobs into quant_cache/...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    quant_manifest = meta["quant_manifest"]

    linked = 0
    total_bytes = 0
    for key, item in quant_manifest.items():
        df_src = cache_dir / item["data_file"]
        sf_src = cache_dir / item["scales_file"]
        df_dst = quant_dir / item["data_file"]
        sf_dst = quant_dir / item["scales_file"]

        if not df_dst.exists():
            try:
                os.link(df_src, df_dst)
            except OSError:
                shutil.copy2(df_src, df_dst)
        if not sf_dst.exists():
            try:
                os.link(sf_src, sf_dst)
            except OSError:
                shutil.copy2(sf_src, sf_dst)

        linked += 2
        total_bytes += df_dst.stat().st_size + sf_dst.stat().st_size

    print(f"  ✓ Linked {linked} files into {quant_dir} ({total_bytes / 1e9:.2f} GB)")

    # 4. Write manifest.json
    print("\n[4/5] Writing quant_cache/manifest.json...")
    manifest_dst = quant_dir / "manifest.json"
    with open(manifest_dst, "w", encoding="utf-8") as f:
        json.dump({"quant_manifest": quant_manifest}, f, indent=2)
    print("  ✓ manifest.json written")

    # Write README.md
    (out_dir / "README.md").write_text(README_CONTENT)

    # 5. Verification: Load standalone package with PureAneRuntime
    if skip_verify:
        print("\n[5/5] Skipping verification (--skip-verify specified)")
    else:
        print("\n[5/5] Verifying standalone package execution...")
        env = dict(os.environ)
        env.update({
            "Q38_ANE_REUSE_COMPILED": "1",
            "Q38_ANE_FUSED_TAIL": "1",
            "Q38_ANE_CHAIN_NEXT": "1",
            "Q38_ANE_FUSE_GATE": "0",
            "Q38_ANE_HOST_PREPARE": "1",
            "Q38_ANE_BATCH_ATTN": "16",
        })
        for k, v in env.items():
            os.environ[k] = v

        test_ckpt = Checkpoint(out_dir)
        test_ckpt.configure_quant_cache(quant_dir)
        t0 = time.perf_counter()
        rt = PureAneRuntime(test_ckpt, str(ROOT), bits=4, context=4096, profile_decode=False)
        print(f"  ✓ Pure ANE runtime initialized in {time.perf_counter() - t0:.1f}s!")
        print(f"  ✓ Quant cache hits: {test_ckpt.quant_cache_stats['hits']}, misses: {test_ckpt.quant_cache_stats['misses']}")
        if test_ckpt.quant_cache_stats["misses"] > 0:
            raise RuntimeError(f"Unexpected cache misses: {test_ckpt.quant_cache_stats['misses']}")

    print("\n" + "=" * 65)
    print(f"  ✓ STANDALONE 27B PACKAGE READY: {out_dir}")
    total_pkg_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"  Total Package Size: {total_pkg_size / 1e9:.2f} GB")
    print("=" * 65)

    return out_dir


def upload_to_huggingface(pkg_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"\n🚀 Creating / Verifying Hugging Face repo: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    # 1. Configs, tokenizers, README
    print("\n📤 [1/3] Uploading configs, tokenizers, and README...")
    config_patterns = ["*.json", "*.jinja", "*.txt", "README.md"]
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(pkg_dir),
        allow_patterns=config_patterns,
        repo_type="model",
        commit_message="Add configs, tokenizer, and model card",
    )
    print("  ✓ Configs and tokenizers uploaded successfully!")

    # 2. Host model.safetensors
    st_path = pkg_dir / "model.safetensors"
    if st_path.is_file():
        print(f"\n📤 [2/3] Uploading host model.safetensors ({st_path.stat().st_size / 1e9:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(st_path),
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add host embeddings, norms, and LM head safetensors",
        )
        print("  ✓ Host model.safetensors uploaded successfully!")

    # 3. quant_cache
    quant_dir = pkg_dir / "quant_cache"
    if quant_dir.is_dir():
        print("\n📤 [3/3] Uploading quant_cache (9.15 GB, 500 prebaked INT4 blobs)...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(quant_dir),
            path_in_repo="quant_cache",
            repo_type="model",
            commit_message="Add prebaked INT4 quant_cache and manifest.json",
        )
        print("  ✓ quant_cache uploaded successfully!")

    print(f"\n🎉 Successfully uploaded {repo_id} to Hugging Face!")
    print(f"   URL: https://huggingface.co/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package and upload standalone Qwen3.8-27B-ANE")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--upload", action="store_true", help="Upload to Hugging Face after packaging")
    parser.add_argument("--skip-verify", action="store_true", help="Skip ANE runtime verification step")
    args = parser.parse_args()

    pkg = package_27b(args.source, args.cache, args.manifest, args.out_dir, skip_verify=args.skip_verify)

    if args.upload:
        upload_to_huggingface(pkg, args.repo_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
