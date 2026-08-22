#!/usr/bin/env python3
"""bench_latency.py - measure TTFT/TPOT/E2E for the native rindi-server.

Args: --host --port --prompt_tokens --gen_tokens [--max_len]
Outputs one JSON-ish line with the standard latency table.
"""
import argparse, json, time, urllib.request, sys

def timed_chat(host, port, prompt_tokens, n, model="gpu"):
    url = f"http://{host}:{port}/v1/chat/completions"
    # build a prompt of roughly `prompt_tokens` whitespace words (tokenizer agnostic)
    words = prompt_tokens
    prompt = (" word" * words).strip()
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n, "stream": False}
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode())
    t1 = time.perf_counter()
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = resp.get("usage", {})
    ttft = usage.get("time_to_first_token_ms", None)
    e2e = (t1 - t0) * 1000.0
    ntokens = len(content.split())
    print(json.dumps({
        "pp": prompt_tokens, "tg": n,
        "resp_chars": len(content),
        "completion_tokens_est": ntokens,
        "ttft_ms": ttft if ttft is not None else "n/a",
        "e2e_ms": round(e2e, 1),
        "tok_s": round(ntokens / (e2e / 1000.0), 1) if e2e > 0 else 0.0,
    }, indent=1))
    return resp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2456)
    ap.add_argument("--pp", type=int, default=1024)
    ap.add_argument("--tg", type=int, default=64)
    a = ap.parse_args()
    timed_chat(a.host, a.port, a.pp, a.tg)