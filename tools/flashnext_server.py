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
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import eval_client  # noqa: E402

MODEL = "Qwen3.8-Flash-Next"
THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
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


def _repair_name(name: str, args: dict, tools) -> str:
    """The model does not always use the name it was given.

    It emitted `<function=cmd>` for a tool called `run_shell`, taking the
    parameter's name for the function's. If the name is not one that was
    declared, fall back to the only declared tool whose required parameters
    are all present.
    """
    declared = {}
    for t in tools or []:
        fn = t.get("function") or t
        if fn.get("name"):
            declared[fn["name"]] = (fn.get("parameters") or {}).get("required") or []
    if not declared or name in declared:
        return name
    fits = [n for n, req in declared.items()
            if all(k in args for k in req)]
    return fits[0] if len(fits) == 1 else name


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
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                "function": {"name": _repair_name(name, args, tools),
                             "arguments": json.dumps(args)},
            })
    if calls:
        text = _CALL.sub("", text)
    return text.strip(), reasoning, calls


def for_template(messages):
    """Rewrite OpenAI messages into what the chat template can render.

    The API carries a tool call's arguments as a JSON string and lets content
    be null; the template iterates the arguments as a mapping and concatenates
    content, so both have to be turned back into the shapes it expects.
    """
    out = []
    for msg in messages:
        m = {"role": msg.get("role", "user"),
             "content": msg.get("content") or ""}
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
        out.append(m)
    return out


class Engine:
    """The serve subprocess, behind a lock."""

    def __init__(self, ctx: int, spec: int, prefill_k: int):
        self.lock = threading.Lock()
        self.client = eval_client.AneClient(ctx=ctx, spec=spec,
                                            prefill_k=prefill_k, quiet=False)

    def chat(self, body: dict) -> dict:
        req = {"op": "gen", "template": True,
               "messages": for_template(body.get("messages", [])),
               "max_new": int(body.get("max_tokens") or 1024)}
        if body.get("tools"):
            req["tools"] = body["tools"]
        for src, dst in (("temperature", "temperature"), ("top_p", "top_p"),
                         ("top_k", "top_k"), ("min_p", "min_p")):
            if body.get(src) is not None:
                req[dst] = body[src]
            elif dst in THINKING:
                req[dst] = THINKING[dst]
        if body.get("stop"):
            stop = body["stop"]
            req["stop"] = [stop] if isinstance(stop, str) else list(stop)
        with self.lock:
            return self.client._rpc(req)


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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine: Engine = None

    def log_message(self, fmt, *a):
        sys.stderr.write("  %s\n" % (fmt % a))

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(200, {"object": "list", "data": [
                {"id": MODEL, "object": "model", "owned_by": "local"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

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
        try:
            res = self.engine.chat(body)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            return
        if not body.get("stream"):
            self._send(200, _completion(res, rid, body.get("tools")))
            return
        # The engine returns a whole turn, so "streaming" is one content chunk
        # followed by the terminator. Real token streaming needs the serve loop
        # to emit deltas, which it does not yet.
        done = _completion(res, rid, body.get("tools"))
        choice = done["choices"][0]["message"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for delta in ({"role": "assistant"},
                      {k: v for k, v in choice.items() if k != "role"}):
            self.wfile.write(b"data: " + json.dumps({
                "id": rid, "object": "chat.completion.chunk",
                "created": done["created"], "model": MODEL,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": None}]}).encode() + b"\n\n")
        self.wfile.write(b"data: " + json.dumps({
            "id": rid, "object": "chat.completion.chunk",
            "created": done["created"], "model": MODEL,
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": done["choices"][0]["finish_reason"]}],
            "usage": done["usage"]}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2457)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--prefill-k", type=int, default=0,
                    help="wide prefill graphs; 0 because they damage the "
                         "first tokens after a prompt, see docs/PREFILL.md")
    a = ap.parse_args()
    print(f"loading {MODEL}; this takes a couple of minutes", flush=True)
    t0 = time.perf_counter()
    Handler.engine = Engine(a.ctx, a.spec, a.prefill_k)
    print(f"ready in {time.perf_counter() - t0:.0f}s on "
          f"http://{a.host}:{a.port}/v1", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
