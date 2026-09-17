"""Two backends behind one interface, so both arms see identical prompts.

`ane` drives the MIL path through `generate --serve`, which holds the model in
one process and answers a JSON request a line. `mlx` runs the unmodified 4-bit
model in process. Same prompts, same scoring, so the difference between the
numbers is the port and nothing else.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLX4 = "/Users/true/models/Qwen3.8-Flash-Next-MLX-4bit"


class AneClient:
    """The MIL path, held in a subprocess."""

    def __init__(self, ctx: int = 8192, spec: int = 4, prefill_k: int = 32,
                 quiet: bool = True, serve_log=None, seed: int | None = None):
        env = dict(os.environ)
        env.setdefault("FLASHNEXT_MOE", "mlxresident")
        env.setdefault("FLASHNEXT_HEAD", "mlx")
        env.setdefault("FLASHNEXT_MIL_GDN", "1")
        env.setdefault("FLASHNEXT_MIL_QSA", "1")
        env["FLASHNEXT_SPEC"] = str(spec)
        # Zero is an explicit decode-width baseline, even when the parent
        # environment or the exporter's default enables wide prefill.
        env["FLASHNEXT_PREFILL_MIL_K"] = str(prefill_k)
        if seed is not None:
            env["FLASHNEXT_SEED"] = str(seed)
        if serve_log:
            err = open(serve_log, "w")
        else:
            err = subprocess.DEVNULL if quiet else None
        py = os.environ.get("FLASHNEXT_PYTHON")
        if not py:
            cand = Path.home() / ".rindi/venvs/coreai/bin/python"
            py = str(cand) if cand.exists() else sys.executable
        self.p = subprocess.Popen(
            [py, "-u",
             str(ROOT / "scripts/export_flashnext_coreai.py"), "generate",
             "--serve", "--serve-ctx", str(ctx),
             "--prompt-ids", "760", "--max-new", "1"],
            cwd=str(ROOT), env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=err, text=True)
        # Everything before the ready line is load chatter.
        chatter = []
        for line in self.p.stdout:
            line_s = line.strip()
            if not quiet:
                sys.stderr.write(line)
                sys.stderr.flush()
            if line_s.startswith("{") and "ready" in line_s:
                self.info = json.loads(line_s)
                self.info["prefill_k"] = prefill_k
                self.info["spec"] = spec
                self.info["ple"] = env.get("FLASHNEXT_PLE", "1")
                return
            chatter.append(line_s)
        tail = "\n".join(chatter[-20:])
        raise RuntimeError(f"the serve process never became ready; output:\n{tail}")

    def _rpc(self, req: dict) -> dict:
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("{"):
                out = json.loads(line)
                if "error" in out:
                    raise RuntimeError(out["error"])
                return out
        raise RuntimeError("the serve process closed")

    def score(self, prompt: str, choices: list[str]) -> dict:
        return self._rpc({"op": "score", "prompt": prompt,
                          "choices": choices})["logprobs"]

    def gen(self, prompt: str, max_new: int, stop: list[str],
            temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
            min_p: float = 0.0, stop_ids=None) -> str:
        req = {"op": "gen", "prompt": prompt, "max_new": max_new,
               "stop": stop, "temperature": temperature,
               "top_p": top_p, "top_k": top_k, "min_p": min_p}
        # generation_config.json lists bos/im_start as eos. That ends a
        # completion-style GSM8K answer on the first token and yields "".
        if stop_ids is not None:
            req["stop_ids"] = stop_ids
        out = self._rpc(req)
        self.last = out
        return out["text"]

    def close(self) -> None:
        try:
            self._rpc_quit()
        except Exception:  # noqa: BLE001
            self.p.kill()
            self.p.wait(timeout=30)
        err = getattr(self, "_err", None)
        if err not in (None, subprocess.DEVNULL):
            err.close()

    def _rpc_quit(self) -> None:
        self.p.stdin.write(json.dumps({"op": "quit"}) + "\n")
        self.p.stdin.flush()
        self.p.wait(timeout=30)


class MlxClient:
    """The unmodified 4-bit model, for the arm to compare against."""

    def __init__(self, **_):
        mlx_lm_path = str(Path.home() / ".mlx128/mlx-lm")
        if mlx_lm_path not in sys.path:
            sys.path.insert(0, mlx_lm_path)
        import mlx.core as mx
        from mlx_lm.utils import load
        if not hasattr(mx, "unique"):
            import numpy as _np
            mx.unique = lambda a, *_a, **_k: mx.array(_np.unique(_np.array(a)))
        self.mx = mx
        self.model, self.tok = load(MLX4)
        self.info = {"ready": True, "backend": "mlx"}

    def score(self, prompt: str, choices: list[str]) -> dict:
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        ids = self.tok.encode(prompt, add_special_tokens=False)
        lg = self.model(mx.array([ids]), cache=make_prompt_cache(self.model))
        row = lg[0, -1].astype(mx.float32)
        lse = mx.logsumexp(row)
        out = {}
        for c in choices:
            cid = self.tok.encode(c, add_special_tokens=False)
            if cid:
                out[c] = float((row[cid[0]] - lse).item())
        return out

    def gen(self, prompt: str, max_new: int, stop: list[str],
            temperature: float = 0.0, **_kw) -> str:
        if temperature > 0:
            raise NotImplementedError("mlx eval arm is greedy only")
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        ids = self.tok.encode(prompt, add_special_tokens=False)
        cache = make_prompt_cache(self.model)
        lg = self.model(mx.array([ids]), cache=cache)
        out = []
        tok = int(mx.argmax(lg[0, -1]).item())
        for _ in range(max_new):
            out.append(tok)
            text = self.tok.decode(out)
            if any(s in text for s in stop if s):
                break
            lg = self.model(mx.array([[tok]]), cache=cache)
            tok = int(mx.argmax(lg[0, -1]).item())
        text = self.tok.decode(out)
        for s in stop:
            if s and s in text:
                return text[:text.index(s)]
        return text

    def close(self) -> None:
        pass


def make(backend: str, **kw):
    return AneClient(**kw) if backend == "ane" else MlxClient(**kw)
