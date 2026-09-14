#!/usr/bin/env python3
"""Headless OpenAI front door for Flash-Next and Qwen3.8-27B.

Both model ids are advertised. Only one backend is resident: 27B is the
framework-free pure ANE child; Flash-Next stays `tools/flashnext_server.py`
(imported never — spawned or adopted). Overlapping POSTs get HTTP 429.
Switching unloads the other engine. Flash-Next is not spawned unless
`--spawn-flashnext` is set; an already-running `:2457` is adopted instead.

    tools/ane headless --port 2456
    rindi openai --load Qwen3.8-27B
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DENSE27 = "Qwen3.8-27B"
FLASHNEXT = "Qwen3.8-Flash-Next"
_ALIAS = {
    "qwen3.8-27b": DENSE27,
    "qwen3.8-27b-pure-ane": DENSE27,
    "27b": DENSE27,
    "qwen3.8-flash-next": FLASHNEXT,
    "flash-next": FLASHNEXT,
    "flashnext": FLASHNEXT,
    "fn": FLASHNEXT,
}


def resolve_model(name: Any) -> str | None:
    if name is None or name == "":
        return None
    raw = str(name).strip()
    if raw in (DENSE27, FLASHNEXT):
        return raw
    return _ALIAS.get(raw.lower())


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_http(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last = "not started"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return
                last = f"HTTP {resp.status}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"backend not ready at {url}: {last}")


class Conflict(Exception):
    """Cannot switch because the other model is adopted, not owned."""


class Busy(Exception):
    """Another request owns the one resident backend."""


class Backend:
    def __init__(self, model: str, kind: str, url: str, proc: subprocess.Popen | None,
                 adopted: bool):
        self.model = model
        self.kind = kind
        self.url = url.rstrip("/")
        self.proc = proc
        self.adopted = adopted

    def host_port(self) -> tuple[str, int]:
        rest = self.url.split("://", 1)[-1]
        host, _, port = rest.partition(":")
        return host, int(port.split("/", 1)[0])


SpawnFn = Callable[[str, argparse.Namespace], Backend]


class Supervisor:
    """Owns at most one child (or one adopted Flash-Next server)."""

    def __init__(self, args: argparse.Namespace, spawners: dict[str, SpawnFn] | None = None):
        self.args = args
        self.lock = threading.Lock()
        self.backend: Backend | None = None
        self.loading: str | None = None
        self.spawners = spawners or {
            DENSE27: spawn_pure27,
            FLASHNEXT: spawn_flashnext,
        }

    def catalog(self) -> dict[str, Any]:
        loaded = self.backend.model if self.backend else None
        return {"object": "list", "data": [
            _model_card(DENSE27, loaded, self.loading, context=self.args.context),
            _model_card(FLASHNEXT, loaded, self.loading, context=self.args.ctx),
        ]}

    def health(self) -> dict[str, Any]:
        b = self.backend
        return {
            "status": "ok",
            "ready": b is not None and self.loading is None,
            "busy": self.lock.locked(),
            "loaded": None if b is None else b.model,
            "loading": self.loading,
            "adopted": False if b is None else b.adopted,
            "backend_url": None if b is None else b.url,
        }

    def ensure(self, model: str) -> Backend:
        if self.backend is not None and self.backend.model == model:
            self._check_alive()
            return self.backend
        self._preflight(model)
        self.loading = model
        try:
            self.unload()
            self.backend = self.spawners[model](model, self.args)
            return self.backend
        finally:
            self.loading = None

    def _preflight(self, model: str) -> None:
        b = self.backend
        if b is not None and b.adopted and b.model != model:
            raise Conflict(
                f"{b.model} is already resident at {b.url}; stop that process "
                "to load the other model"
            )
        if model == FLASHNEXT and self.spawners.get(FLASHNEXT) is spawn_flashnext:
            _flashnext_available(self.args)

    def unload(self) -> None:
        b = self.backend
        if b is None:
            return
        if b.adopted:
            raise Conflict(
                f"{b.model} is already resident at {b.url}; stop that process "
                "to load the other model"
            )
        self.backend = None
        if b.proc is not None:
            _stop(b.proc)

    def _check_alive(self) -> None:
        b = self.backend
        if b is None or b.proc is None or b.proc.poll() is None:
            return
        code = b.proc.returncode
        self.backend = None
        raise RuntimeError(f"{b.model} child exited with {code}")


def _model_card(model_id: str, loaded: str | None, loading: str | None,
                context: int) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "local",
        "loaded": loaded == model_id,
        "loading": loading == model_id,
        "context_window": context,
    }


def spawn_pure27(model: str, args: argparse.Namespace) -> Backend:
    port = free_port()
    cmd = [
        args.python27, "-u", "-P", str(ROOT / "tools" / "pure_ane_server.py"),
        "serve",
        "--model", args.q38_model,
        "--name", DENSE27,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--bits", str(args.bits),
        "--context", str(args.context),
        "--max-tokens", str(args.max_tokens),
        "--mtp-draft", str(args.mtp_draft),
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    print(f"loading {model} on :{port}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        wait_http(url + "/v1/models", args.load_timeout)
    except Exception:
        _stop(proc)
        raise
    print(f"ready {model} {url}/v1", flush=True)
    return Backend(model, "pure27", url, proc, adopted=False)


def _flashnext_available(args: argparse.Namespace) -> str | None:
    adopt = (args.flashnext_url or "").rstrip("/")
    if adopt and _reachable(adopt + "/v1/models"):
        return adopt
    if args.spawn_flashnext:
        return None
    raise Conflict(
        f"{FLASHNEXT} is not resident. Start tools/flashnext_server.py "
        f"on {adopt or 'http://127.0.0.1:2457'} or pass --spawn-flashnext"
    )


def spawn_flashnext(model: str, args: argparse.Namespace) -> Backend:
    adopt = _flashnext_available(args)
    if adopt:
        print(f"adopting {model} at {adopt}", flush=True)
        return Backend(model, "flashnext", adopt, None, adopted=True)
    port = free_port()
    cmd = [
        args.python_flashnext, "-u", str(ROOT / "tools" / "flashnext_server.py"),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx", str(args.ctx),
        "--spec", str(args.spec),
        "--prefill-k", str(args.prefill_k),
        "--max-new", str(args.max_tokens),
    ]
    print(f"loading {model} on :{port} (this takes a couple of minutes)", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        wait_http(url + "/v1/models", args.load_timeout)
    except Exception:
        _stop(proc)
        raise
    print(f"ready {model} {url}/v1", flush=True)
    return Backend(model, "flashnext", url, proc, adopted=False)


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=5)


def build_handler(sup: Supervisor):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "Rindi/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("  " + (fmt % args) + "\n")

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            if code == 429:
                self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("", "/health", "/healthz"):
                self._send(200, sup.health())
            elif path == "/v1/models":
                self._send(200, sup.catalog())
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                self._send(400, {"error": {"message": str(exc)}})
                return
            raw_model = body.get("model")
            if raw_model in (None, ""):
                wanted = sup.backend.model if sup.backend else DENSE27
            else:
                wanted = resolve_model(raw_model)
                if wanted is None:
                    self._send(400, {"error": {
                        "message": f"unknown model {raw_model!r}; "
                                   f"use {DENSE27} or {FLASHNEXT}"}})
                    return
            if path not in ("/v1/chat/completions", "/v1/completions"):
                self._send(404, {"error": {"message": "not found"}})
                return
            if path == "/v1/completions" and wanted != DENSE27:
                self._send(404, {"error": {
                    "message": "/v1/completions is only implemented on Qwen3.8-27B"}})
                return
            if not sup.lock.acquire(blocking=False):
                self._send(429, {"error": {"message": "another request is in flight"}})
                return
            try:
                backend = sup.ensure(wanted)
                self._proxy(backend, path, raw, body.get("stream"))
            except Busy as exc:
                self._send(429, {"error": {"message": str(exc)}})
            except Conflict as exc:
                self._send(409, {"error": {"message": str(exc)}})
            except (ValueError, TimeoutError) as exc:
                self._send(400 if isinstance(exc, ValueError) else 503,
                           {"error": {"message": str(exc)}})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            finally:
                sup.lock.release()

        def _proxy(self, backend: Backend, path: str, raw: bytes, stream: Any) -> None:
            host, port = backend.host_port()
            conn = HTTPConnection(host, port, timeout=self.args_timeout())
            headers = {
                "Content-Type": self.headers.get("Content-Type") or "application/json",
                "Content-Length": str(len(raw)),
                "Connection": "close",
            }
            conn.request("POST", path, body=raw, headers=headers)
            resp = conn.getresponse()
            ctype = resp.getheader("Content-Type") or "application/json"
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            if resp.status == 429:
                self.send_header("Retry-After", resp.getheader("Retry-After") or "1")
            if stream or "event-stream" in ctype:
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    conn.close()
                return
            payload = resp.read()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            conn.close()

        def args_timeout(self) -> float:
            return float(sup.args.request_timeout)

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=os.environ.get("RINDI_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("RINDI_PORT", "2456")))
    p.add_argument("--load", default="",
                   help="preload Qwen3.8-27B or Qwen3.8-Flash-Next (lazy if omitted)")
    p.add_argument("--q38-model", default=os.environ.get(
        "Q38_MODEL", str(Path.home() / ".lmstudio" / "models" / "Qwen" / "Qwen3.8-27B")))
    p.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--mtp-draft", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--ctx", type=int, default=8192, help="Flash-Next KV capacity")
    p.add_argument("--spec", type=int, default=4)
    p.add_argument("--prefill-k", type=int, default=32)
    p.add_argument("--flashnext-url", default="http://127.0.0.1:2457")
    p.add_argument("--spawn-flashnext", action="store_true",
                   help="launch Flash-Next (~78 GB). Default is adopt :2457 only")
    p.add_argument("--python27", default="/opt/homebrew/bin/python3")
    p.add_argument("--python-flashnext",
                   default=os.path.expanduser("~/.rindi/venvs/coreai/bin/python"))
    p.add_argument("--load-timeout", type=float, default=900)
    p.add_argument("--request-timeout", type=float, default=3600)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sup = Supervisor(args)
    preload = resolve_model(args.load) if args.load else None
    if args.load and preload is None:
        print(f"unknown --load {args.load!r}", file=sys.stderr)
        return 2
    if preload:
        try:
            with sup.lock:
                sup.ensure(preload)
        except Exception as exc:  # noqa: BLE001
            print(f"preload failed: {exc}", file=sys.stderr)
            return 1
    handler = build_handler(sup)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    print(f"RINDI_OPENAI_READY http://{args.host}:{args.port}/v1 "
          f"loaded={None if sup.backend is None else sup.backend.model}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            if sup.backend is not None and not sup.backend.adopted:
                _stop(sup.backend.proc) if sup.backend.proc else None
        except Conflict:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
