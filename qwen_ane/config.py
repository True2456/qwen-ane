"""Configuration and local storage management for qwen-ane (.qwenANE/)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = {
    "default_model": "flash-next",
    "default_ctx": 131072,  # 128k
    "default_port": 2457,
    "default_lru": True,
    "default_host": "127.0.0.1",
    "default_thinking": "off",
    "hf_repos": {
        "flash-next": "True2456/Qwen3.8-Flash-Next-ANE",
        "27b": "True2456/Qwen3.8-27B-ANE",
    },
}


def find_repo_root() -> Path | None:
    """Find the root repository directory if running from a git checkout."""
    cur = Path(__file__).resolve().parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists() or (p / "ane-port").exists() or (p / "runtime").exists():
            return p
    return None


def get_qwen_ane_dir() -> Path:
    """Resolve the .qwenANE storage directory.

    Priority:
      1. QWEN_ANE_HOME environment variable (explicit override)
      2. ~/.qwenANE (user's home directory root, canonical default)
      3. <repo_root>/.qwenANE (if explicitly present as non-symlinked directory)
    """
    env_home = os.environ.get("QWEN_ANE_HOME")
    if env_home:
        base = Path(env_home).expanduser().resolve()
    else:
        home_dir = Path.home() / ".qwenANE"
        repo_root = find_repo_root()
        repo_dir = repo_root / ".qwenANE" if repo_root else None

        if repo_dir and repo_dir.is_dir() and not repo_dir.is_symlink() and any(repo_dir.iterdir()):
            base = repo_dir
        else:
            base = home_dir

    base.mkdir(parents=True, exist_ok=True)
    (base / "models").mkdir(exist_ok=True)
    (base / "sessions").mkdir(exist_ok=True)
    return base


def get_config_path() -> Path:
    return get_qwen_ane_dir() / "config.json"


def load_config() -> dict[str, Any]:
    cfg_file = get_config_path()
    if not cfg_file.exists():
        cfg = dict(DEFAULT_CONFIG)
        save_config(cfg)
        return cfg
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged = dict(DEFAULT_CONFIG)
            merged.update(data)
            return merged
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(cfg: dict[str, Any]) -> None:
    cfg_file = get_config_path()
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_models_dir() -> Path:
    return get_qwen_ane_dir() / "models"


def get_sessions_dir() -> Path:
    return get_qwen_ane_dir() / "sessions"
