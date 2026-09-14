"""Apple Foundation Models (fm)-inspired interactive CLI chat for qwen-ane."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .config import get_sessions_dir, load_config
from .downloader import ensure_model, normalize_model_name
from .server import is_server_running, run_server

BANNER = """
  Qwen ANE CLI
  Apple Silicon Neural Engine Inference

  Model:    {model_name}
  Context:  {ctx_human}
  Endpoint: http://{host}:{port}/v1
  Cache:    {cache_status}
  Thinking: {thinking_status}

  Commands: /exit (quit), /clear (new chat), /think [level], /help
"""


def format_tokens(n: int) -> str:
    if n >= 1024:
        return f"{n // 1024}k" if n % 1024 == 0 else f"{n / 1024:.1f}k"
    return str(n)


class ChatSession:
    def __init__(
        self,
        model: str = "flash-next",
        host: str = "127.0.0.1",
        port: int = 2457,
        system_prompt: str | None = None,
        thinking: str = "off",
        session_id: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model = normalize_model_name(model)
        self.model_id = "Qwen3.8-Flash-Next" if self.model == "flash-next" else "Qwen3.8-27B"
        self.host = host
        self.port = port
        self.thinking = thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_id = session_id or (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        self.messages: list[dict[str, Any]] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def save(self) -> None:
        if not self.messages:
            return
        sdir = get_sessions_dir()
        path = sdir / f"{self.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "model": self.model,
                "thinking": self.thinking,
                "messages": self.messages,
                "saved_at": time.time(),
            }, f, indent=2)

    def load(self, session_id: str) -> bool:
        sdir = get_sessions_dir()
        path = sdir / f"{session_id}.json"
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.session_id = data.get("session_id", session_id)
                self.model = data.get("model", self.model)
                self.thinking = data.get("thinking", self.thinking)
                self.messages = data.get("messages", [])
                return True
        except Exception:
            return False


def stream_chat_completion(
    host: str,
    port: int,
    model_id: str,
    messages: list[dict[str, Any]],
    thinking: str = "off",
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """Streams chat completion tokens from server using SSE."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking and thinking != "off":
        payload["enable_thinking"] = True
        payload["reasoning_effort"] = thinking
    else:
        payload["enable_thinking"] = False

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                raw_chunk = line[6:]
                if raw_chunk == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw_chunk)
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        yield delta
                except json.JSONDecodeError:
                    continue


def start_chat(
    model: str = "flash-next",
    ctx: int = 131072,
    port: int = 2457,
    host: str = "127.0.0.1",
    lru: bool = True,
    thinking: str = "off",
    system_prompt: str | None = None,
    resume_id: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model_path: str | Path | None = None,
    hf_repo: str | None = None,
) -> int:
    """Launch the interactive chat REPL."""
    canon = normalize_model_name(model)
    model_id = "Qwen3.8-Flash-Next" if canon == "flash-next" else "Qwen3.8-27B"

    # Check if server is running
    server_proc = None
    if not is_server_running(host, port):
        print(f"Starting background {canon} engine on port {port}...", flush=True)
        resolved_path = ensure_model(canon, custom_path=model_path, hf_repo=hf_repo)

        env = dict(os.environ)
        root = Path(__file__).resolve().parents[1]

        if canon == "flash-next":
            env.update({
                "FLASHNEXT_SPEC": "4",
                "FLASHNEXT_MOE": "mlxresident",
                "FLASHNEXT_HEAD": "mlx",
                "FLASHNEXT_MIL_GDN": "1",
                "FLASHNEXT_MIL_QSA": "1",
                "FLASHNEXT_PREFILL_MIL_K": "32" if lru else "0",
                "FLASHNEXT_MODEL": str(resolved_path),
            })
            cmd = [
                sys.executable,
                "-u",
                str(root / "tools" / "flashnext_server.py"),
                "--host",
                str(host),
                "--port",
                str(port),
                "--ctx",
                str(ctx),
                "--max-new",
                str(max_tokens),
            ]
        else:
            ctx_27b = min(ctx, 4096)
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
            cmd = [
                sys.executable,
                "-u",
                "-P",
                str(root / "tools" / "pure_ane_server.py"),
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
            ]

        server_proc = subprocess.Popen(cmd, env=env, cwd=str(root))

        # Wait for server to become ready
        for _ in range(120):
            if is_server_running(host, port):
                break
            time.sleep(1.0)
        else:
            print(f"❌ Server timed out while starting.")
            if server_proc:
                server_proc.terminate()
            return 1

    session = ChatSession(
        model=canon,
        host=host,
        port=port,
        system_prompt=system_prompt,
        thinking=thinking,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if resume_id:
        if session.load(resume_id):
            print(f"Loaded existing session: {resume_id} ({len(session.messages)} messages)")
        else:
            print(f"Session '{resume_id}' not found. Starting fresh session.")

    ctx_human = format_tokens(ctx)
    cache_str = "Enabled (LRU prefix reuse)" if lru else "Disabled"
    print(BANNER.format(
        model_name=model_id,
        ctx_human=ctx_human,
        host=host,
        port=port,
        cache_status=cache_str,
        thinking_status=session.thinking,
    ))

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input in ("/exit", "/quit"):
                break
            elif user_input == "/clear":
                session.messages = []
                print("✨ Conversation history cleared.\n")
                continue
            elif user_input.startswith("/think"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] in ("off", "low", "medium", "xhigh"):
                    session.thinking = parts[1]
                    print(f"💡 Thinking set to: {session.thinking}\n")
                else:
                    print("Usage: /think [off|low|medium|xhigh]\n")
                continue
            elif user_input == "/help":
                print("Commands:")
                print("  /exit, /quit       Exit chat")
                print("  /clear             Clear conversation history")
                print("  /think [level]     Set thinking level (off, low, medium, xhigh)")
                print("  /help              Show this help\n")
                continue

            session.messages.append({"role": "user", "content": user_input})
            print("qwen> ", end="", flush=True)

            t0 = time.perf_counter()
            full_reply = []
            full_thought = []
            in_thinking = False

            try:
                for delta in stream_chat_completion(
                    host=session.host,
                    port=session.port,
                    model_id=session.model_id,
                    messages=session.messages,
                    thinking=session.thinking,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                ):
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        if not in_thinking:
                            print("\033[2m[thinking: ", end="", flush=True)
                            in_thinking = True
                        print(reasoning, end="", flush=True)
                        full_thought.append(reasoning)

                    content = delta.get("content")
                    if content:
                        if in_thinking:
                            print("]\033[0m\n", end="", flush=True)
                            in_thinking = False
                        print(content, end="", flush=True)
                        full_reply.append(content)

                if in_thinking:
                    print("]\033[0m", flush=True)
                print("\n")

                reply_text = "".join(full_reply)
                msg: dict[str, Any] = {"role": "assistant", "content": reply_text}
                if full_thought:
                    msg["reasoning_content"] = "".join(full_thought)
                session.messages.append(msg)
                session.save()

            except Exception as exc:
                print(f"\n❌ Error during completion: {exc}\n")

    finally:
        session.save()
        print(f"\nYou are exiting the conversation.")
        print(f"Resume conversation with: qwen-ane chat --resume {session.session_id}\n")
        if server_proc:
            print("Stopping background server...")
            server_proc.terminate()
            server_proc.wait(timeout=5)

    return 0
