#!/usr/bin/env python3
"""llama.cpp-style pp/tg sweep for qwen-ane 27B and Flash-Next.

Measures TTFT, TPOT, ppTPS, tgTPS, E2E, throughput, peak phys_footprint,
and (if root power sampling is available) ANE/GPU/CPU watts.

    /opt/homebrew/bin/python3 -u probes/ctx_scale_bench.py
    /opt/homebrew/bin/python3 -u probes/ctx_scale_bench.py --models 27b
    /opt/homebrew/bin/python3 -u probes/ctx_scale_bench.py --models flash-next
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ctx_scale"
PP_LENGTHS = (4096, 8192, 16384, 32768)
TG = 128
# 32k prompt + 128 gen + a little headroom. Programs compile at width 32;
# this only sizes the KV / recurrent buffers.
CTX = 33792
FILLER = "The history of computing is a story of successive abstractions over silicon. "

UNIT = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
FOOT_RE = re.compile(
    r"(Footprint|phys_footprint(?:_peak)?):\s+([\d.]+)\s+(B|KB|MB|GB|TB)",
    re.I,
)
IOACC_RE = re.compile(
    r"^\s*([\d.]+)\s+(B|KB|MB|GB|TB)\b.*\b(IOAccelerator|Foundation|Malloc Large)\b",
    re.I | re.M,
)
PWR_HDR = re.compile(
    r"\*\*\* Sampled system activity \(([^)]+?) ([+-]\d{4})\) \(([\d.]+)ms"
)
PWR_LINE = re.compile(
    r"^(ANE|GPU|CPU|Combined)\s+Power:\s+([\d.]+)\s*(m?W)", re.M | re.I
)


def log(msg: str) -> None:
    print(msg, flush=True)


def bytes_of(n: float, unit: str) -> int:
    return int(float(n) * UNIT[unit.upper()])


def gb(n: int | float | None) -> float | None:
    if n is None:
        return None
    return round(float(n) / (1024**3), 2)


def parse_footprint(text: str) -> dict:
    out = {"current": None, "peak": None, "categories": {}}
    for kind, val, unit in FOOT_RE.findall(text):
        b = bytes_of(val, unit)
        key = kind.lower()
        if key == "footprint" or key == "phys_footprint":
            out["current"] = b
        elif key == "phys_footprint_peak":
            out["peak"] = b
    for val, unit, name in IOACC_RE.findall(text):
        out["categories"][name] = bytes_of(val, unit)
    return out


def sample_footprint(pid: int) -> dict:
    try:
        proc = subprocess.run(
            ["footprint", "-p", str(pid)],
            capture_output=True, text=True, timeout=20,
        )
        text = proc.stdout or proc.stderr or ""
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"error": str(exc)}
    parsed = parse_footprint(text)
    parsed["raw_head"] = "\n".join(text.splitlines()[:8])
    return parsed


class FootSampler:
    def __init__(self, pid: int, interval: float = 2.5):
        self.pid = pid
        self.interval = interval
        self.samples: list[dict] = []
        self.peak = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._th.start()

    def stop(self) -> dict:
        self._stop.set()
        self._th.join(timeout=25)
        return {
            "n": len(self.samples),
            "peak_bytes": self.peak,
            "last": self.samples[-1] if self.samples else None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = sample_footprint(self.pid)
            snap["t"] = time.time()
            cur = snap.get("current") or 0
            peak = snap.get("peak") or cur
            self.peak = max(self.peak, cur, peak)
            self.samples.append(snap)
            self._stop.wait(self.interval)


def make_prompt(tokenizer_file: Path, n: int, salt: str = "") -> str:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_file))
    # Salt the first tokens so 8k is not a prefix of 4k. Flash-Next prefix
    # reuse would otherwise make longer prompts look as fast as the short one.
    head = f"BENCH {salt} {n} {salt}. " if salt else ""
    chunk = tok.encode(head + FILLER).ids
    if not chunk:
        raise RuntimeError(f"tokenizer {tokenizer_file} encoded filler as empty")
    ids: list[int] = []
    while len(ids) < n:
        ids.extend(chunk)
    ids = ids[:n]
    text = tok.decode(ids)
    got = tok.encode(text).ids
    if len(got) == n:
        return text
    lo, hi = 0, len(text)
    best = text
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid]
        k = len(tok.encode(cand).ids)
        if k < n:
            lo = mid + 1
        else:
            best = cand
            hi = mid - 1
            if k == n:
                break
    return best


def llama_metrics(pp: int, tg: int, ttft_s: float, e2e_s: float) -> dict:
    decode_s = max(e2e_s - ttft_s, 1e-9)
    decode_n = max(tg - 1, 1)
    pp_tps = pp / max(ttft_s, 1e-9)
    tg_tps = decode_n / decode_s
    tpot_ms = 1000.0 / max(tg_tps, 1e-9)
    throughput = (pp + tg) / max(e2e_s, 1e-9)
    return {
        "TTFT_ms": round(ttft_s * 1000.0, 1),
        "TPOT_ms": round(tpot_ms, 1),
        "ppTPS": round(pp_tps, 1),
        "tgTPS": round(tg_tps, 1),
        "E2E_s": round(e2e_s, 1),
        "Throughput": round(throughput, 1),
        "prompt_tokens": pp,
        "completion_tokens": tg,
    }


def start_powermetrics(log_path: Path) -> subprocess.Popen | None:
    """Start root powermetrics without blocking model load.

    osascript buffers `do shell script` stdout until the command exits, so
    powermetrics must redirect to a file itself. The GUI password prompt, if
    any, can sit in the background while 27B bakes.
    """
    samplers = "cpu_power,gpu_power,ane_power"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    inner = [
        "/usr/bin/powermetrics", "--samplers", samplers, "-i", "500",
        "--show-extra-power-info",
    ]
    try:
        proc = subprocess.Popen(
            ["sudo", "-n", *inner],
            stdout=open(log_path, "w"), stderr=subprocess.DEVNULL,
        )
        time.sleep(0.8)
        if proc.poll() is None:
            log(f"POWER sampling via sudo -n pid={proc.pid} -> {log_path}")
            return proc
        proc.wait(timeout=2)
    except OSError:
        pass

    quoted = str(log_path)
    wrapper = log_path.parent / (log_path.stem + ".sh")
    wrapper.write_text(
        "#!/bin/bash\n"
        "exec /usr/bin/powermetrics --samplers "
        f"{samplers} -i 500 --show-extra-power-info "
        f"> {json.dumps(quoted)} 2>&1\n"
    )
    wrapper.chmod(0o755)
    # do shell script feeds sh; unquoted paths with spaces (LLM - Reap) die.
    apple = (
        f"do shell script quoted form of {json.dumps(str(wrapper))} "
        "with administrator privileges"
    )
    log("POWER: password prompt may appear; approve it so ANE/GPU/CPU rails record")
    try:
        proc = subprocess.Popen(
            ["osascript", "-e", apple],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        log(f"POWER unavailable: {exc}")
        return None
    log(f"POWER osascript pid={proc.pid} -> {log_path}")
    return proc


def wait_for_power(log_path: Path, timeout: float = 180.0) -> bool:
    log("POWER: approve the macOS password dialog if it appears")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            text = ""
        if "Sampled system activity" in text:
            log(f"POWER samples flowing ({log_path.stat().st_size} bytes)")
            return True
        time.sleep(1.0)
    log("POWER: no samples yet; rows may lack ANE/GPU/CPU watts")
    return False


def parse_power(log_path: Path, t0: float, t1: float) -> dict | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    if "Sampled system activity" not in text:
        return None
    import calendar

    blocks = []
    hits = list(PWR_HDR.finditer(text))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        ts, off = m.group(1), m.group(2)
        try:
            t = calendar.timegm(time.strptime(ts, "%a %b %d %H:%M:%S %Y"))
            sign = 1 if off[0] == "+" else -1
            t -= sign * (int(off[1:3]) * 3600 + int(off[3:5]) * 60)
        except ValueError:
            continue
        vals = {}
        for u in PWR_LINE.finditer(text[m.start():end]):
            v = float(u.group(2))
            if u.group(3).lower() == "mw":
                v /= 1000.0
            vals[u.group(1).upper()] = v
        blocks.append((t, vals))
    window = [v for t, v in blocks if t0 - 0.5 <= t <= t1 + 0.5]
    if not window:
        return {"samples": 0}

    def mean(key: str) -> float | None:
        xs = [v[key] for v in window if key in v]
        return round(sum(xs) / len(xs), 2) if xs else None

    out = {
        "samples": len(window),
        "ANE_W": mean("ANE"),
        "GPU_W": mean("GPU"),
        "CPU_W": mean("CPU"),
        "Combined_W": mean("COMBINED"),
    }
    parts = [out[k] for k in ("ANE_W", "GPU_W", "CPU_W") if out[k] is not None]
    out["sum_rails_W"] = round(sum(parts), 2) if parts else None
    return out


def wait_http(url: str, needle: str | None, timeout: float,
              server: "Server | None" = None) -> None:
    t0 = time.time()
    last = None
    log_off = 0
    next_beat = t0 + 10
    while time.time() - t0 < timeout:
        if server is not None:
            if not server.alive():
                tail = ""
                try:
                    if hasattr(server, "log_path"):
                        tail = server.log_path.read_text(errors="replace")[-2500:]
                except (OSError, AttributeError):
                    pass
                rc = getattr(getattr(server, "proc", None), "returncode", "unknown")
                raise RuntimeError(
                    f"{server.name} exited {rc}\n{tail}"
                )
            data = b""
            if hasattr(server, "log_path"):
                try:
                    data = server.log_path.read_bytes()
                except OSError:
                    data = b""
            if len(data) > log_off:
                chunk = data[log_off:].decode(errors="replace")
                log_off = len(data)
                for line in chunk.splitlines():
                    log(f"  [{server.name}] {line}")
                    if "Traceback" in line or "Error:" in line:
                        last = line
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                body = resp.read().decode()
                if needle is None or needle in body:
                    return
                last = body[:200]
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        now = time.time()
        if now >= next_beat:
            log(f"  waiting {server.name if server else url} "
                f"{now - t0:.0f}s last={last}")
            next_beat = now + 10
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {url}: {last}")


def post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {body[:800]}") from exc


def emit_row(doc: dict, row: dict, path: Path) -> None:
    doc.setdefault("rows", []).append(row)
    path.write_text(json.dumps(doc, indent=2))
    test = row.get("Test", "")
    bits = [
        f"ROW model={row.get('model')} {test}",
        f"TTFT={row.get('TTFT_ms')}ms",
        f"TPOT={row.get('TPOT_ms')}ms",
        f"ppTPS={row.get('ppTPS')}",
        f"tgTPS={row.get('tgTPS')}",
        f"E2E={row.get('E2E_s')}s",
        f"Throughput={row.get('Throughput')}",
        f"PeakMem={row.get('PeakMem')}",
    ]
    if row.get("ANE_W") is not None:
        bits.append(
            f"W=ANE {row['ANE_W']}/GPU {row['GPU_W']}/CPU {row['CPU_W']}"
        )
    if row.get("error"):
        bits.append(f"ERROR {row['error'][:200]}")
    log("  ".join(str(x) for x in bits))


def print_table(rows: list[dict], model: str) -> None:
    mine = [r for r in rows if r.get("model") == model and not r.get("error")]
    if not mine:
        return
    hdr = ("Test", "TTFT(ms)", "TPOT(ms)", "ppTPS", "tgTPS",
           "E2E(s)", "Throughput", "PeakMem")
    log("\n" + "\t".join(hdr))
    for r in mine:
        log("\t".join([
            str(r.get("Test", "")),
            str(r.get("TTFT_ms", "")),
            str(r.get("TPOT_ms", "")),
            str(r.get("ppTPS", "")),
            str(r.get("tgTPS", "")),
            str(r.get("E2E_s", "")),
            str(r.get("Throughput", "")),
            str(r.get("PeakMem", "")),
        ]))


class Server:
    def __init__(self, name: str, proc: subprocess.Popen, log_path: Path):
        self.name = name
        self.proc = proc
        self.log_path = log_path

    def pid(self) -> int:
        return int(self.proc.pid)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)


def find_pid_on_port(port: int) -> int | None:
    try:
        proc = subprocess.run(
            ["lsof", "-t", f"-i:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        out = proc.stdout.strip()
        if out:
            return int(out.split()[0])
    except Exception:
        pass
    return None


class AttachedServer:
    def __init__(self, name: str, pid: int):
        self.name = name
        self._pid = pid

    def pid(self) -> int:
        return self._pid

    def alive(self) -> bool:
        try:
            os.kill(self._pid, 0)
            return True
        except OSError:
            return False

    def stop(self) -> None:
        pass


def start_27b(port: int, model: Path, serve_log: Path, context: int = 4096) -> Server:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({
        "Q38_ANE_REUSE_COMPILED": "1",
        "Q38_ANE_FUSED_TAIL": "1",
        "Q38_ANE_CHAIN_NEXT": "1",
        "Q38_ANE_FUSE_GATE": "0",
        "Q38_ANE_HOST_PREPARE": "1",
        "Q38_ANE_BATCH_ATTN": "16",
        # chunked GDN prefill calls unrolled_mil(..., chunk_body=) which
        # is not on this tree; unroll is the working bake path.
        "Q38_ANE_GDN_PREFILL": "unroll",
        "Q38_MODEL": str(model),
    })
    py = sys.executable or "/opt/homebrew/bin/python3"
    cmd = [
        py, "-u", "-P", str(ROOT / "tools" / "pure_ane_server.py"),
        "serve", "--model", str(model), "--host", "127.0.0.1",
        "--port", str(port), "--context", str(context), "--bits", "4",
        "--max-tokens", "256", "--profile-decode",
        "--max-request-bytes", str(32 * 1024 * 1024),
    ]
    serve_log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(serve_log, "w")
    log(f"START 27b context={context} port={port}")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
    )
    return Server("27b", proc, serve_log)


def run_27b_case(port: int, prompt: str, timeout: float, tg: int) -> dict:
    data = post_json(
        f"http://127.0.0.1:{port}/v1/completions",
        {
            "prompt": prompt,
            "max_tokens": tg,
            "temperature": 0.0,
            "prefix_cache": False,
        },
        timeout=timeout,
    )
    r = data.get("pure_ane") or {}
    pp = int(r.get("prompt_tokens") or data.get("usage", {}).get("prompt_tokens") or 0)
    tg = int(r.get("completion_tokens") or data.get("usage", {}).get("completion_tokens") or 0)
    ttft = float(r.get("time_to_first_token") or 0.0)
    e2e = float(r.get("seconds") or 0.0)
    row = llama_metrics(pp, tg, ttft, e2e)
    row["decode_tokens_per_second"] = r.get("decode_tokens_per_second")
    row["finish_reason"] = r.get("finish_reason")
    row["prefix_cache_hit"] = r.get("prefix_cache_hit")
    row["profile"] = r.get("profile")
    return row


class FlashClient:
    def __init__(self, model: Path, ctx: int, serve_log: Path, tg: int = 128):
        self.tg = tg
        env = dict(os.environ)
        env.update({
            "FLASHNEXT_SPEC": "4",  # serve_loop is only entered when spec_k > 1
            "FLASHNEXT_MOE": "mlxresident",
            "FLASHNEXT_HEAD": "mlx",
            "FLASHNEXT_MIL_GDN": "1",
            "FLASHNEXT_MIL_QSA": "1",
            "FLASHNEXT_PREFILL_MIL_K": "32",
            "FLASHNEXT_MODEL": str(model),
        })
        py = str(Path.home() / ".rindi/venvs/coreai/bin/python")
        serve_log.parent.mkdir(parents=True, exist_ok=True)
        log(f"START flash-next context={ctx}")
        self.log_path = serve_log
        self.p = subprocess.Popen(
            [py, "-u", str(ROOT / "scripts/export_flashnext_coreai.py"),
             "generate", "--serve", "--serve-ctx", str(ctx),
             "--prompt-ids", "760", "--max-new", "1"],
            cwd=str(ROOT), env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.info = None
        t0 = time.time()
        last = []
        assert self.p.stdout is not None
        with open(serve_log, "w") as logf:
            for line in self.p.stdout:
                logf.write(line)
                logf.flush()
                stripped = line.rstrip()
                last.append(stripped)
                if len(last) > 40:
                    last = last[-40:]
                if stripped.startswith("{"):
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and obj.get("error"):
                        raise RuntimeError(f"flash-next: {obj['error']}")
                    if isinstance(obj, dict) and obj.get("ready"):
                        self.info = obj
                        log(f"flash-next ready in {time.time()-t0:.0f}s {self.info}")
                        return
                elif "error" in stripped.lower() or "Traceback" in stripped:
                    log(f"  [flash-next] {stripped[:300]}")
                elif time.time() - t0 < 30 or int(time.time() - t0) % 15 == 0:
                    log(f"  [flash-next] {stripped[:200]}")
                if self.p.poll() is not None and not stripped:
                    break
        tail = "\n".join(last[-20:])
        raise RuntimeError(
            f"flash-next serve process never became ready "
            f"(exit={self.p.returncode})\n{tail}"
        )

    def rpc(self, req: dict) -> dict:
        assert self.p.stdin is not None and self.p.stdout is not None
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("{"):
                out = json.loads(line)
                if "error" in out:
                    raise RuntimeError(out["error"])
                return out
        raise RuntimeError("flash-next serve process closed")

    def reset(self) -> None:
        out = self.rpc({"op": "reset"})
        log(f"flash-next reset {out}")

    def gen(self, prompt: str) -> dict:
        self.reset()
        return self.rpc({
            "op": "gen",
            "prompt": prompt,
            "max_new": self.tg,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": [],
            "stop_ids": [],
        })

    def stop(self) -> None:
        try:
            if self.p.poll() is None and self.p.stdin is not None:
                self.p.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self.p.stdin.flush()
                self.p.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.p.kill()
            try:
                self.p.wait(timeout=10)
            except Exception:
                pass


def timeouts_for(pp: int) -> float:
    # 27B prefill ~24 tok/s worst case plus decode plus a lot of slack.
    return max(900.0, pp / 8.0 + 600.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="27b,flash-next")
    ap.add_argument("--pp", default=",".join(str(x) for x in PP_LENGTHS))
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--port", type=int, default=1240)
    ap.add_argument("--model-27b", default=None, help="Path to 27B model directory")
    ap.add_argument("--model-flashnext", default=None, help="Path to Flash-Next model directory")
    ap.add_argument("--skip-power", action="store_true")
    args = ap.parse_args()
    tg = int(args.tg)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lengths = [int(x) for x in args.pp.split(",") if x.strip()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS / f"{stamp}.json"
    power_path = RESULTS / f"{stamp}.powermetrics.txt"
    doc = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": "M5 Max",
        "ctx": CTX,
        "tg": tg,
        "lengths": lengths,
        "models": models,
        "notes": (
            "Cold prefix (prefix_cache false / reset between runs). "
            "One measured run per length. 27B MTP off, Q38_ANE_GDN_PREFILL=unroll "
            "(chunked GDN is unwired on this tree). Flash-Next FLASHNEXT_SPEC=4 "
            "(serve requires spec_k>1; decode includes MTP). Prefill tile width is 32 on both. "
            "PeakMem is footprint(1) phys_footprint_peak during the run."
        ),
        "rows": [],
    }
    out_path.write_text(json.dumps(doc, indent=2))
    log(f"RESULTS {out_path}")

    try:
        from qwen_ane.downloader import find_local_model
        m27_local = find_local_model("27b")
        mfn_local = find_local_model("flash-next")
    except Exception:
        m27_local = None
        mfn_local = None

    m27 = Path(args.model_27b) if args.model_27b else (m27_local or Path.home() / ".lmstudio/models/Qwen/Qwen3.8-27B")
    mfn = Path(args.model_flashnext) if args.model_flashnext else (mfn_local or Path.home() / "models/Qwen3.8-Flash-Next")

    power_proc = None
    if not args.skip_power:
        power_proc = start_powermetrics(power_path)
        if power_proc is None:
            doc["power"] = "unavailable (sudo password required)"
        elif not wait_for_power(power_path, timeout=180):
            doc["power"] = "started but no samples yet"

    try:
        if "27b" in models:
            existing_pid = find_pid_on_port(args.port)
            if existing_pid:
                log(f"Found existing 27B server on port {args.port} (pid={existing_pid}). Reusing resident weights...")
                srv = AttachedServer("27b", existing_pid)
            else:
                slog = RESULTS / f"{stamp}.27b.serve.log"
                srv = start_27b(args.port, m27, slog, context=4096)
                wait_http(f"http://127.0.0.1:{args.port}/health",
                          '"ready":true', timeout=480, server=srv)
            try:
                idle = sample_footprint(srv.pid())
                log(f"27b idle footprint={gb(idle.get('current'))} GB "
                    f"peak={gb(idle.get('peak'))} GB pid={srv.pid()}")
                tok = m27 / "tokenizer.json"
                for pp in lengths:
                    if pp > 4096:
                        log(f"--- 27b pp {pp} / tg {tg} [SKIPPED: 27B pure ANE tile capacity is 4096 tokens; use flash-next for 4k-32k context scaling]")
                        continue
                    if not srv.alive():
                        raise RuntimeError("27b server died")
                    log(f"--- 27b pp {pp} / tg {tg}")
                    prompt = make_prompt(tok, pp, salt=f"27b-{pp}")
                    foot = FootSampler(srv.pid())
                    foot.start()
                    t0 = time.time()
                    try:
                        row = run_27b_case(args.port, prompt, timeouts_for(pp), tg)
                    except Exception as exc:  # noqa: BLE001
                        row = {"error": str(exc), "Test": f"pp {pp} / tg {tg}"}
                    t1 = time.time()
                    mem = foot.stop()
                    row["model"] = "27b"
                    row["Test"] = f"pp {pp} / tg {tg}"
                    row["PeakMem_bytes"] = mem["peak_bytes"]
                    row["PeakMem"] = (
                        f"{gb(mem['peak_bytes'])} GB" if mem["peak_bytes"] else ""
                    )
                    row["idle_footprint_GB"] = gb(idle.get("current"))
                    pwr = parse_power(power_path, t0, t1) if power_proc else None
                    if pwr:
                        row.update({
                            "ANE_W": pwr.get("ANE_W"),
                            "GPU_W": pwr.get("GPU_W"),
                            "CPU_W": pwr.get("CPU_W"),
                            "sum_rails_W": pwr.get("sum_rails_W"),
                            "power_samples": pwr.get("samples"),
                        })
                        watts = pwr.get("sum_rails_W") or pwr.get("Combined_W")
                        if watts and row.get("E2E_s"):
                            joules = watts * float(row["E2E_s"])
                            row["Joules"] = round(joules, 1)
                            tot = (row.get("prompt_tokens") or 0) + (
                                row.get("completion_tokens") or 0
                            )
                            if joules > 0:
                                row["tok_per_J"] = round(tot / joules, 2)
                    emit_row(doc, row, out_path)
                    time.sleep(2.0)
                print_table(doc["rows"], "27b")
            finally:
                srv.stop()
                log("STOP 27b")
                time.sleep(4.0)

        if "flash-next" in models:
            slog = RESULTS / f"{stamp}.flash-next.serve.log"
            client = FlashClient(mfn, CTX, slog, tg=tg)
            try:
                idle = sample_footprint(client.p.pid)
                log(f"flash-next idle footprint={gb(idle.get('current'))} GB "
                    f"peak={gb(idle.get('peak'))} GB pid={client.p.pid}")
                tok = mfn / "tokenizer.json"
                for pp in lengths:
                    if client.p.poll() is not None:
                        raise RuntimeError("flash-next server died")
                    log(f"--- flash-next pp {pp} / tg {tg}")
                    prompt = make_prompt(tok, pp, salt=f"flash-{pp}")
                    foot = FootSampler(client.p.pid)
                    foot.start()
                    t0 = time.time()
                    try:
                        res = client.gen(prompt)
                        pp_n = int(res.get("prompt_tokens") or 0)
                        tg_n = int(res.get("tokens") or 0)
                        reused = int(res.get("reused") or 0)
                        prefill_s = float(res.get("prefill_ms") or 0.0) / 1000.0
                        e2e_s = float(res.get("ms") or 0.0) / 1000.0
                        # Flash-Next reports prefill until decode starts, which
                        # is the TTFT analogue. New prompt tokens exclude reuse.
                        new_pp = max(pp_n - reused, 0)
                        row = llama_metrics(new_pp or pp_n, tg_n, prefill_s, e2e_s)
                        row["reused"] = reused
                        row["engine_tok_s"] = res.get("tok_s")
                    except Exception as exc:  # noqa: BLE001
                        row = {"error": str(exc), "Test": f"pp {pp} / tg {tg}"}
                    t1 = time.time()
                    mem = foot.stop()
                    row["model"] = "flash-next"
                    row["Test"] = f"pp {pp} / tg {tg}"
                    row["PeakMem_bytes"] = mem["peak_bytes"]
                    row["PeakMem"] = (
                        f"{gb(mem['peak_bytes'])} GB" if mem["peak_bytes"] else ""
                    )
                    row["idle_footprint_GB"] = gb(idle.get("current"))
                    pwr = parse_power(power_path, t0, t1) if power_proc else None
                    if pwr:
                        row.update({
                            "ANE_W": pwr.get("ANE_W"),
                            "GPU_W": pwr.get("GPU_W"),
                            "CPU_W": pwr.get("CPU_W"),
                            "sum_rails_W": pwr.get("sum_rails_W"),
                            "power_samples": pwr.get("samples"),
                        })
                        watts = pwr.get("sum_rails_W") or pwr.get("Combined_W")
                        if watts and row.get("E2E_s"):
                            joules = watts * float(row["E2E_s"])
                            row["Joules"] = round(joules, 1)
                            tot = (row.get("prompt_tokens") or 0) + (
                                row.get("completion_tokens") or 0
                            )
                            if joules > 0:
                                row["tok_per_J"] = round(tot / joules, 2)
                    emit_row(doc, row, out_path)
                    time.sleep(2.0)
                print_table(doc["rows"], "flash-next")
            finally:
                client.stop()
                log("STOP flash-next")
    finally:
        if power_proc is not None:
            try:
                power_proc.send_signal(signal.SIGTERM)
                power_proc.wait(timeout=5)
            except Exception:
                try:
                    power_proc.kill()
                except Exception:
                    pass
        doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        out_path.write_text(json.dumps(doc, indent=2))
        log(f"DONE {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
