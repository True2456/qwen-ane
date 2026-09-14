"""Server runner and supervisor for qwen-ane."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import load_config
from .downloader import ensure_model, normalize_model_name

ROOT = Path(__file__).resolve().parents[1]


def is_server_running(host: str = "127.0.0.1", port: int = 2457, timeout: float = 1.0) -> dict[str, Any] | None:
    """Check if a qwen-ane server is running on host:port, returning model info if so."""
    url = f"http://{host}:{port}/v1/models"
    req = urllib.request.Request(url, headers={"User-Agent": "qwen-ane"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return data
    except Exception:
        return None
    return None


def run_server(
    model: str = "flash-next",
    ctx: int = 131072,
    port: int = 2457,
    host: str = "127.0.0.1",
    lru: bool = True,
    model_path: str | Path | None = None,
    hf_repo: str | None = None,
    max_new: int = 2048,
    spec: int = 4,
) -> int:
    """Launch the OpenAI-compatible ANE server for the specified model."""
    canon = normalize_model_name(model)

    # 1. Check if server already running
    active = is_server_running(host, port)
    if active:
        models = [m.get("id") for m in active.get("data", [])]
        print(f"⚠️  A server is already running on http://{host}:{port}/v1 serving: {models}")
        print(f"   To connect interactively, run: qwen-ane chat --port {port}")
        return 0

    # 2. Ensure model exists locally or download it
    resolved_path = ensure_model(canon, custom_path=model_path, hf_repo=hf_repo)
    print(f"🚀 Initializing qwen-ane server...")
    print(f"   Model:   {canon} ({resolved_path})")
    print(f"   Context: {ctx} tokens")
    print(f"   Port:    {port} (http://{host}:{port}/v1)")
    print(f"   LRU/Cache: {'Enabled' if lru else 'Disabled'}")

    env = dict(os.environ)

    if canon == "flash-next":
        # Flash-Next on ANE + MLX
        env.update({
            "FLASHNEXT_SPEC": str(spec),
            "FLASHNEXT_MOE": "mlxresident",
            "FLASHNEXT_HEAD": "mlx",
            "FLASHNEXT_MIL_GDN": "1",
            "FLASHNEXT_MIL_QSA": "1",
            "FLASHNEXT_PREFILL_MIL_K": "32" if lru else "0",
            "FLASHNEXT_MODEL": str(resolved_path),
        })
        # Check python interpreter
        py = sys.executable
        server_script = ROOT / "tools" / "flashnext_server.py"
        cmd = [
            py,
            "-u",
            str(server_script),
            "--host",
            str(host),
            "--port",
            str(port),
            "--ctx",
            str(ctx),
            "--spec",
            str(spec),
            "--max-new",
            str(max_new),
        ]
        return subprocess.call(cmd, env=env, cwd=str(ROOT))

    elif canon == "27b":
        # Pure ANE 27B
        # If context is 128k, adjust down to 4096 for 27B pure ANE tile limits
        ctx_27b = min(ctx, 4096) if ctx > 4096 else ctx
        if ctx > 4096:
            print(f"   Note: 27B pure ANE uses hardware tile context of {ctx_27b} tokens.")

        env.update({
            "Q38_ANE_REUSE_COMPILED": "1",
            "Q38_ANE_FUSED_TAIL": "1",
            "Q38_ANE_CHAIN_NEXT": "1",
            "Q38_ANE_FUSE_GATE": "0",
            "Q38_ANE_HOST_PREPARE": "1",
            "Q38_ANE_BATCH_ATTN": "16",
            "Q38_MODEL": str(resolved_path),
        })
        env.pop("PYTHONPATH", None)

        server_script = ROOT / "tools" / "pure_ane_server.py"
        cmd = [
            sys.executable,
            "-u",
            "-P",
            str(server_script),
            "serve",
            "--model",
            str(resolved_path),
            "--host",
            str(host),
            "--port",
            str(port),
            "--context",
            str(ctx_27b),
            "--bits",
            "4",
            "--max-tokens",
            str(max_new),
            "--profile-decode",
        ]
        return subprocess.call(cmd, env=env, cwd=str(ROOT))

    else:
        print(f"❌ Unknown model: {model}. Choose 'flash-next' or '27b'.")
        return 1
