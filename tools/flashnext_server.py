#!/usr/bin/env python3
"""An OpenAI-compatible endpoint in front of the Flash-Next ANE path.

One model, one process, one request at a time: every request mutates the same
recurrent state, so they are serialised. Prefix reuse in the engine is what
makes that bearable for an agent loop, which resends its whole history each
turn and gets the shared part back for free.

    ~/.rindi/venvs/coreai/bin/python tools/flashnext_server.py --port 2457

Then point any OpenAI client at http://127.0.0.1:2457/v1 . The API key is not
checked. Implements /v1/models and /v1/chat/completions, streaming and not,
with tool calls parsed out of the model's own <tool_call> blocks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import eval_client  # noqa: E402

MODEL = "Qwen3.8-Flash-Next"
THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
# chat_template.jinja: only these three, default xhigh. `high` is not an alias.
_EFFORTS = {"low", "medium", "xhigh"}
_EFFORT_ALIAS = {"minimal": "low", "min": "low", "high": "xhigh", "max": "xhigh"}


def reasoning_effort(value):
    """Map a client effort string onto what the Flash-Next template accepts."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "off", "none", "false", "0"):
        return None
    s = _EFFORT_ALIAS.get(s, s)
    return s if s in _EFFORTS else None
# The generation prompt ends with an open <think>, so a reply carries the
# closing tag and not the opening one. And the template asks for calls in an
# XML-ish shape, not JSON:
#
#   <tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>
_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FUNC = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.S)
_PARAM = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)


def _coerce(value: str):
    """Parameters arrive as text; give numbers and literals their real type."""
    v = value.strip()
    if v in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[v]
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d*\.\d+([eE][-+]?\d+)?", v):
        return float(v)
    if v[:1] in "[{" and v[-1:] in "]}":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    return value.strip()


def _one_call(inner: str):
    fn = _FUNC.search(inner)
    if fn:
        name, body = fn.group(1).strip(), fn.group(2)
        args = {k.strip(): _coerce(v) for k, v in _PARAM.findall(body)}
        if not args:
            # The model does not always follow its own format; it sometimes
            # puts a JSON object where the parameter blocks should be, and
            # leaves a stray closing tag behind. Take the arguments anyway.
            loose = re.sub(r"</?(parameter|function)[^>]*>", "", body).strip()
            if loose[:1] == "{":
                try:
                    obj = json.loads(loose)
                    if isinstance(obj, dict):
                        args = obj
                except json.JSONDecodeError:
                    pass
        return name, args
    try:                                    # the JSON shape, just in case
        obj = json.loads(inner)
        return obj.get("name", ""), obj.get("arguments", {})
    except json.JSONDecodeError:
        return None


_TOOL_ALIAS = {
    "run_shell": "bash", "run_shell_command": "bash", "shell": "bash",
    "read_file": "read", "write_file": "write", "str_replace": "edit",
    "replace": "edit",
}
_ARG_ALIAS = {
    "bash": (("cmd", "command"),),
    "read": (("file_path", "path"), ("target_file", "path")),
    "write": (("file_path", "path"), ("contents", "content")),
}


def _repair_name(name: str, args: dict, tools) -> str:
    """The model does not always use the name it was given.

    It emitted `<function=cmd>` for a tool called `run_shell`, taking the
    parameter's name for the function's. If the name is not one that was
    declared, fall back to the only declared tool whose required parameters
    are all present. Pi's tools are `bash`/`read`/`write`/`edit`.
    """
    declared = {}
    for t in tools or []:
        fn = t.get("function") or t
        if fn.get("name"):
            declared[fn["name"]] = (fn.get("parameters") or {}).get("required") or []
    alias = _TOOL_ALIAS.get(name, name)
    if declared and alias in declared:
        return alias
    if not declared or name in declared:
        return name
    fits = [n for n, req in declared.items()
            if all(k in args for k in req)]
    return fits[0] if len(fits) == 1 else alias


def _alias_args(name: str, args: dict) -> dict:
    out = dict(args)
    for src, dst in _ARG_ALIAS.get(name, ()):
        if src in out and dst not in out:
            out[dst] = out.pop(src)
    return out


def split_reply(text: str, tools=None):
    """Returns (content, reasoning, tool_calls) from one raw completion."""
    reasoning = None
    end = text.find("</think>")
    if end >= 0:
        reasoning = text[:end].removeprefix("<think>").strip()
        text = text[end + len("</think>"):]
    calls = []
    for inner in _CALL.findall(text):
        got = _one_call(inner)
        if got and got[0]:
            name, args = got
            name = _repair_name(name, args, tools)
            args = _alias_args(name, args)
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
    if calls:
        text = _CALL.sub("", text)
    return text.strip(), reasoning, calls


_NONTEXT = ("image", "image_url", "input_audio", "video", "video_url", "file")


def content_text(content) -> str:
    """OpenAI content is a string or a list of parts. The Jinja template
    raises `Unexpected item type in content` on file/audio parts, and Qwen
    Code sends user text as `[{type: text, text: ...}]`."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in _NONTEXT or any(k in item for k in _NONTEXT):
            parts.append("[omitted non-text part]")
        elif "text" in item:
            parts.append(str(item.get("text") or ""))
    return "".join(parts)


def for_template(messages):
    """Rewrite OpenAI messages into what the chat template can render.

    The API carries a tool call's arguments as a JSON string and lets content
    be null or a list of parts; the template iterates the arguments as a
    mapping and concatenates content, so both have to be turned back into the
    shapes it expects. `developer` is the OpenAI name for system.
    """
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "developer":
            role = "system"
        m = {"role": role, "content": content_text(msg.get("content"))}
        if msg.get("name"):
            m["name"] = msg["name"]
        if msg.get("tool_call_id"):
            m["tool_call_id"] = msg["tool_call_id"]
        calls = []
        for call in msg.get("tool_calls") or []:
            fn = dict(call.get("function") or {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args or "{}")
                except json.JSONDecodeError:
                    fn["arguments"] = {"input": args}
            calls.append({**{k: v for k, v in call.items()
                             if k in ("id", "type")}, "function": fn})
        if calls:
            m["tool_calls"] = calls
        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            m["reasoning_content"] = rc
        out.append(m)
    return out


class Busy(Exception):
    """Another chat is using the one recurrent state."""


def client_error(exc: BaseException) -> tuple[int, str]:
    """400 for a prompt the engine rejected; 500 for a real crash."""
    if isinstance(exc, Busy):
        return 429, str(exc)
    text = str(exc)
    msg = f"{type(exc).__name__}: {exc}"
    bad = ("empty prompt", "prompt is", "context is",
           "unexpected item type", "system message", "no user query")
    lowered = text.lower()
    if any(s in lowered for s in bad):
        return 400, text
    return 500, msg


class Engine:
    """The serve subprocess, behind a lock."""

    def __init__(self, ctx: int, spec: int, prefill_k: int, max_new: int = 2048, quiet: bool = False):
        self.lock = threading.Lock()
        self.max_new = max_new
        self.client = eval_client.AneClient(ctx=ctx, spec=spec,
                                            prefill_k=prefill_k, quiet=quiet)

    def chat(self, body: dict) -> dict:
        raw = body.get("max_tokens", body.get("max_completion_tokens"))
        req = {"op": "gen", "template": True,
               "messages": for_template(body.get("messages", [])),
               "max_new": min(int(raw or 1024), self.max_new)}
        if body.get("tools"):
            req["tools"] = body["tools"]
        ctk = body.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk:
            ctk = dict(ctk)
            effort = reasoning_effort(ctk.get("reasoning_effort"))
            if effort:
                ctk["reasoning_effort"] = effort
            else:
                ctk.pop("reasoning_effort", None)
            req["chat_template_kwargs"] = ctk
        if body.get("enable_thinking") is not None:
            req["enable_thinking"] = body["enable_thinking"]
        effort = reasoning_effort(body.get("reasoning_effort"))
        if effort:
            req["reasoning_effort"] = effort
        for src, dst in (("temperature", "temperature"), ("top_p", "top_p"),
                         ("top_k", "top_k"), ("min_p", "min_p")):
            if body.get(src) is not None:
                req[dst] = body[src]
            elif dst in THINKING:
                req[dst] = THINKING[dst]
        if body.get("stop"):
            stop = body["stop"]
            req["stop"] = [stop] if isinstance(stop, str) else list(stop)
        if not self.lock.acquire(blocking=False):
            raise Busy("another request is in flight")
        try:
            return self.client._rpc(req)
        finally:
            self.lock.release()


def _completion(res: dict, stream_id: str, tools=None) -> dict:
    content, reasoning, calls = split_reply(res["text"], tools)
    message = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return {
        "id": stream_id, "object": "chat.completion",
        "created": int(time.time()), "model": MODEL,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": res.get("prompt_tokens", 0),
                  "completion_tokens": res.get("tokens", 0),
                  "total_tokens": res.get("prompt_tokens", 0)
                  + res.get("tokens", 0),
                  "prompt_tokens_details": {
                      "cached_tokens": res.get("reused", 0)}},
    }


def stream_deltas(message: dict) -> list[dict]:
    """OpenAI stream chunks from one finished message.

    The engine returns a whole turn, so this is one role chunk, then
    reasoning/content/tool_calls. Pi's completions parser keys tool
    calls by `index` when it is present.
    """
    out = [{"role": "assistant"}]
    if message.get("reasoning_content"):
        out.append({"reasoning_content": message["reasoning_content"]})
    if message.get("content"):
        out.append({"content": message["content"]})
    calls = message.get("tool_calls") or []
    if calls:
        out.append({"tool_calls": [{**c, "index": i} for i, c in enumerate(calls)]})
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine: Engine = None
    quiet: bool = False

    def log_message(self, fmt, *a):
        if not self.quiet and os.environ.get("QWEN_ANE_QUIET") != "1":
            sys.stderr.write("  %s\n" % (fmt % a))

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if code == 429:
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(200, {"object": "list", "data": [
                {"id": MODEL, "object": "model", "owned_by": "local"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, payload):
        self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
        self.wfile.flush()

    def _fail(self, exc: BaseException, stream: bool, started: bool):
        code, msg = client_error(exc)
        sys.stderr.write(f"  {code} {msg}\n")
        if code >= 500:
            traceback.print_exc(file=sys.stderr)
        if stream and started:
            try:
                self._sse({"error": {"message": msg}})
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return
        self._send(code, {"error": {"message": msg}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": {"message": str(exc)}})
            return
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        stream = bool(body.get("stream"))
        msgs = body.get("messages") or []
        n_tools = len(body.get("tools") or [])
        ctk = body.get("chat_template_kwargs")
        ctk = ctk if isinstance(ctk, dict) else {}
        think = body.get("enable_thinking", ctk.get("enable_thinking"))
        if not self.quiet and os.environ.get("QWEN_ANE_QUIET") != "1":
            sys.stderr.write(
                "  req %s stream=%s tools=%d msgs=%d max_tokens=%s think=%s bytes=%d\n"
                % (rid[:16], stream, n_tools, len(msgs),
                   body.get("max_tokens") or body.get("max_completion_tokens"),
                   think, n))
        box: dict = {}

        def run():
            try:
                box["res"] = self.engine.chat(body)
            except Exception as exc:  # noqa: BLE001
                box["exc"] = exc

        th = threading.Thread(target=run)
        th.start()
        started = False
        created = int(time.time())
        # Qwen Code aborts a stream after 4 minutes with no chunks, then
        # retries the same POST three times. Keepalives are empty deltas so
        # that guard sees activity; SSE comments are stripped before it.
        while th.is_alive():
            th.join(20.0)
            if not (stream and th.is_alive()):
                continue
            try:
                if not started:
                    self._sse_open()
                    started = True
                    self._sse({
                        "id": rid, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL,
                        "choices": [{"index": 0, "delta": {"role": "assistant"},
                                     "finish_reason": None}]})
                else:
                    self._sse({
                        "id": rid, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL,
                        "choices": [{"index": 0, "delta": {},
                                     "finish_reason": None}]})
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Escape in Pi closes the socket. Decode still owns the one
                # recurrent state; the next prompt restores the checkpoint
                # taken before this generate, so the discarded tokens never
                # become history. Until that returns, other clients 429.
                sys.stderr.write(
                    "  cancel %s client gone; decode still running\n"
                    % rid[:16])
                th.join()
                sys.stderr.write("  cancel %s discarded\n" % rid[:16])
                return
        if "exc" in box:
            self._fail(box["exc"], stream, started)
            return
        res = box["res"]
        n_gen = int(res.get("tokens") or 0)
        n_prompt = int(res.get("prompt_tokens") or 0)
        n_reused = int(res.get("reused") or 0)
        n_new = max(n_prompt - n_reused, 0)
        pf_ms = float(res.get("prefill_ms") or 0)
        wall_ms = float(res.get("ms") or 0)
        pf_s = pf_ms / 1000.0
        dec_s = max(wall_ms - pf_ms, 0.0) / 1000.0
        pf_tps = (n_new / pf_s) if pf_s > 0 else 0.0
        dec_tps = (n_gen / dec_s) if dec_s > 0 else 0.0
        if not self.quiet and os.environ.get("QWEN_ANE_QUIET") != "1":
            sys.stderr.write(
                "  done %s prompt=%d reused=%d new=%d gen=%d  "
                "wall=%.1f tok/s  prefill=%d/%.0fms %.1f tok/s  "
                "decode=%.1f tok/s\n"
                % (rid[:16], n_prompt, n_reused, n_new, n_gen,
                   float(res.get("tok_s") or 0), n_new, pf_ms, pf_tps, dec_tps))
        if not stream:
            self._send(200, _completion(res, rid, body.get("tools")))
            return
        # The engine returns a whole turn, so "streaming" is one content chunk
        # followed by the terminator. Real token streaming needs the serve loop
        # to emit deltas, which it does not yet.
        try:
            done = _completion(res, rid, body.get("tools"))
            choice = done["choices"][0]["message"]
            if not started:
                self._sse_open()
            deltas = stream_deltas(choice)
            if started:
                deltas = [d for d in deltas if d != {"role": "assistant"}]
            for delta in deltas:
                self._sse({
                    "id": rid, "object": "chat.completion.chunk",
                    "created": done["created"], "model": MODEL,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": None}]})
            self._sse({
                "id": rid, "object": "chat.completion.chunk",
                "created": done["created"], "model": MODEL,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": done["choices"][0]["finish_reason"]}],
                "usage": done["usage"]})
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2457)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--prefill-k", type=int, default=32,
                    help="prompt block width; 0 walks prompts at decode width")
    ap.add_argument("--max-new", type=int, default=2048,
                    help="cap on max_tokens; clients like Qwen Code send 64k")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress request logging to stderr")
    a = ap.parse_args()
    is_quiet = a.quiet or os.environ.get("QWEN_ANE_QUIET") == "1"
    Handler.quiet = is_quiet
    if not is_quiet:
        print(f"loading {MODEL}; this takes a couple of minutes", flush=True)
    t0 = time.perf_counter()
    Handler.engine = Engine(a.ctx, a.spec, a.prefill_k, max_new=a.max_new, quiet=is_quiet)
    if not is_quiet:
        print(f"ready in {time.perf_counter() - t0:.0f}s on "
              f"http://{a.host}:{a.port}/v1", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
