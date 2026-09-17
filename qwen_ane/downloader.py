"""Model discovery, downloading, and building for qwen-ane."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

from .config import get_models_dir, load_config

CANONICAL_MODELS = {
    "flash-next": "flash-next",
    "flashnext": "flash-next",
    "qwen-flash-next": "flash-next",
    "qwen3.8-flash-next": "flash-next",
    "27b": "27b",
    "qwen-27b": "27b",
    "qwen3.8-27b": "27b",
    "pure27": "27b",
}


def normalize_model_name(name: str) -> str:
    key = name.strip().lower()
    if key in CANONICAL_MODELS:
        return CANONICAL_MODELS[key]
    return key


def get_default_hf_repo(model: str) -> str | None:
    cfg = load_config()
    repos = cfg.get("hf_repos", {})
    return repos.get(model)


def find_local_model(model: str, custom_path: str | Path | None = None) -> Path | None:
    """Find local model weights on disk."""
    canon = normalize_model_name(model)
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.exists():
            return p

    models_dir = get_models_dir()
    packaged = models_dir / canon
    if packaged.exists() and any(packaged.iterdir()):
        return packaged

    home = Path.home()
    if canon == "flash-next":
        search_paths = [
            home / "models" / "Qwen3.8-Flash-Next",
            home / "models" / "Qwen3.8-Flash-Next-MLX-4bit",
            home / ".lmstudio" / "models" / "Qwen" / "Qwen3.8-Flash-Next",
        ]
    elif canon == "27b":
        search_paths = [
            home / ".lmstudio" / "models" / "Qwen" / "Qwen3.8-27B",
            home / "models" / "Qwen3.8-27B",
            home / ".lmstudio" / "models" / "Qwen" / "Qwen3.8-27B.rindi",
        ]
    else:
        search_paths = [
            home / "models" / canon,
            home / ".lmstudio" / "models" / "Qwen" / canon,
        ]

    for p in search_paths:
        if p.exists() and any(p.iterdir()):
            return p

    # Check Hugging Face hub cache
    hf_cache = home / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        pattern = f"models--*--*{canon}*"
        matches = list(hf_cache.glob(pattern))
        for m in matches:
            snapshots = m / "snapshots"
            if snapshots.exists():
                for s in snapshots.iterdir():
                    if s.is_dir() and any(s.iterdir()):
                        return s

    return None


def ensure_model(
    model: str,
    custom_path: str | Path | None = None,
    hf_repo: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Ensure the model is available locally, pulling from Hugging Face if needed."""
    canon = normalize_model_name(model)
    local = find_local_model(canon, custom_path)
    if local:
        return local

    # Model not found locally, trigger download into .qwenANE/models/<canon>
    repo_id = hf_repo or get_default_hf_repo(canon)
    if not repo_id:
        raise FileNotFoundError(
            f"Model '{model}' not found locally and no Hugging Face repository is configured. "
            f"Please specify --model-path or --hf-repo."
        )

    target_dir = get_models_dir() / canon
    target_dir.mkdir(parents=True, exist_ok=True)

    msg = f"Model '{canon}' not found locally. Pulling from Hugging Face: {repo_id} -> {target_dir}"
    if progress_callback:
        progress_callback(msg)
    else:
        print(f"\n📥 {msg}\n", flush=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to download models automatically. "
            "Install it via: pip install huggingface_hub"
        )

    downloaded = Path(snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    ))

    # If the repo bundles ane-h17 ANE packages, link them to artifacts/ane-h17
    h17_in_model = downloaded / "ane-h17"
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_h17 = repo_root / "artifacts" / "ane-h17"
    if h17_in_model.exists() and not artifacts_h17.exists():
        artifacts_h17.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(h17_in_model, artifacts_h17)
            print(f"🔗 Linked ANE MIL packages: {h17_in_model} -> {artifacts_h17}")
        except OSError:
            pass

    return downloaded


def build_from_base(
    model: str,
    source_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Build quantized ANE weights from base BF16 weights."""
    canon = normalize_model_name(model)
    src = Path(source_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source path {src} does not exist.")

    out = Path(output_dir).expanduser().resolve() if output_dir else (get_models_dir() / canon)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔨 Installing {canon} from {src} -> {out}...")
    if not (src / "tokenizer.json").is_file() and not (src / "config.json").is_file():
        raise FileNotFoundError(
            f"{src} does not look like a Qwen checkpoint (missing tokenizer.json / config.json)."
        )
    if src.resolve() == out.resolve():
        print(f"✅ Model already at {out}")
        return out
    if out.exists() or out.is_symlink():
        if out.is_symlink() or out.is_file():
            out.unlink()
        else:
            shutil.rmtree(out)
    try:
        out.symlink_to(src, target_is_directory=True)
        print(f"🔗 Linked {src} -> {out}")
    except OSError:
        shutil.copytree(src, out)
        print(f"📦 Copied {src} -> {out}")
    if canon == "27b":
        print("   First 27B serve will bake INT4 into ~/Library/Caches/q38-pure-ane.")
    print(f"✅ Model ready at {out}")
    return out
