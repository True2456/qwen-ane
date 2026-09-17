#!/usr/bin/env python3
"""Package and upload Qwen3.8-27B-ANE to Hugging Face as a standalone ~14 GB package."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
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
README_PATH = ROOT / "scripts" / "README_27B.md"


def write_mtp_sidecar(source_model: Path, out_dir: Path) -> Path | None:
    """Copy the checkpoint MTP layer next to the ANE host-tensor package.

    Keep BF16 payloads. Down-casting to F16 before INT4 quantize wrecks
    drafter acceptance (measured ~1.03 tokens/cycle vs ~2.4 from BF16).
    """
    dest = out_dir / "mtp.safetensors"
    ckpt = Checkpoint(source_model)
    names = [n for n in ckpt.weight_map if n.startswith("mtp.")]
    if not names:
        print("\n[2b/5] No mtp.* tensors in source; speculative decode disabled")
        return None
    print("\n[2b/5] Extracting MTP draft tensors (BF16)...")
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name in names:
        info = ckpt.info(name)
        raw = np.ascontiguousarray(ckpt._file(name).mmap(name))
        if info.dtype == "BF16":
            payload = np.asarray(raw, dtype="<u2").tobytes()
            dtype = "BF16"
        else:
            payload = np.asarray(raw, np.float16).tobytes()
            dtype = "F16"
        header[name] = {
            "dtype": dtype,
            "shape": list(info.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        blobs.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(encoded) % 8)) % 8
    encoded += b" " * pad
    dest.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
    print(f"  ✓ {dest.name} ({dest.stat().st_size / 1e9:.2f} GB, {len(header)} tensors)")
    return dest


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
            # Save the on-disk RMSNorm deltas. Checkpoint.tensor() would bake
            # the Qwen +1 sanitize into the package and NaN at load.
            tensors[name] = ckpt._file(name).array(name, np.float16)
        embed_name = ckpt.embedding_name
        tensors[embed_name] = ckpt._file(embed_name).array(embed_name, np.float16)
        if "lm_head.weight" not in tensors:
            tensors["lm_head.weight"] = ckpt._file("lm_head.weight").array(
                "lm_head.weight", np.float16
            )
        sf.save_file(tensors, str(st_path))
        print(f"  ✓ Extracted and saved {len(tensors)} host tensors ({st_path.stat().st_size / 1e9:.2f} GB)")

    write_mtp_sidecar(source_model, out_dir)

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

    (out_dir / "README.md").write_text(README_PATH.read_text())

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
    print(f"Creating / verifying Hugging Face repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print("\n[1/4] Uploading configs, tokenizer, and README...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(pkg_dir),
        allow_patterns=["*.json", "*.jinja", "*.txt", "README.md"],
        ignore_patterns=["quant_cache/**"],
        repo_type="model",
        commit_message="Add configs, tokenizer, and model card",
    )

    st_path = pkg_dir / "model.safetensors"
    if st_path.is_file():
        print(f"\n[2/4] Uploading model.safetensors ({st_path.stat().st_size / 1e9:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(st_path),
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add host embeddings, norms, and LM head safetensors",
        )

    mtp_path = pkg_dir / "mtp.safetensors"
    if mtp_path.is_file():
        print(f"\n[3/4] Uploading mtp.safetensors ({mtp_path.stat().st_size / 1e9:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(mtp_path),
            path_in_repo="mtp.safetensors",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add MTP draft sidecar",
        )

    quant_dir = pkg_dir / "quant_cache"
    if quant_dir.is_dir():
        n_files = sum(1 for p in quant_dir.iterdir() if p.is_file())
        nbytes = sum(p.stat().st_size for p in quant_dir.iterdir() if p.is_file())
        print(f"\n[4/4] Uploading quant_cache ({n_files} files, {nbytes / 1e9:.2f} GB)...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(quant_dir),
            path_in_repo="quant_cache",
            repo_type="model",
            commit_message="Add INT4 quant_cache blobs and manifest",
        )

    print(f"\nUploaded {repo_id}")
    print(f"URL: https://huggingface.co/{repo_id}")


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
