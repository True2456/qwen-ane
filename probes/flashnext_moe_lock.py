#!/usr/bin/env python3
"""Kill-or-bless Flash-Next expert locking on the 4-bit MLX model.

Prefill uses full top-10. Decode masks the router to a frozen per-layer set.
If greedy tokens hold, that set is what we bake on ANE.

    PYTHONPATH=".:$HOME/.mlx128/mlx-lm" \
      python3 -u probes/flashnext_moe_lock.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAP = ROOT.parent
for p in (str(REAP), str(Path.home() / ".mlx128/mlx-lm")):
    if p not in sys.path:
        sys.path.insert(0, p)

from reap_stream.load_qwen4_exp import load_prebuilt  # noqa: E402


class Lock:
    def __init__(self, n_layers: int, n_experts: int, *, k_keep: int | None,
                 gdn_shared: bool, uniform: bool):
        self.n_experts = n_experts
        self.k_keep = k_keep
        self.gdn_shared = gdn_shared
        self.uniform = uniform
        self.phase = "prefill"
        self.counts = [np.zeros(n_experts, np.int64) for _ in range(n_layers)]
        self.allowed: list[np.ndarray | None] = [None] * n_layers
        self.is_linear: list[bool] = [False] * n_layers

    def record(self, layer: int, inds: np.ndarray) -> None:
        self.counts[layer] += np.bincount(inds.reshape(-1), minlength=self.n_experts)

    def freeze(self) -> None:
        for i, c in enumerate(self.counts):
            nz = np.flatnonzero(c)
            if self.k_keep is None or nz.size <= self.k_keep:
                self.allowed[i] = nz.astype(np.int32)
            else:
                self.allowed[i] = np.argsort(c)[-self.k_keep:].astype(np.int32)
        sizes = [0 if a is None else int(a.size) for a in self.allowed]
        lin = [s for s, L in zip(sizes, self.is_linear) if L]
        qsa = [s for s, L in zip(sizes, self.is_linear) if not L]
        print(f"  union sizes  min={min(sizes)}  med={int(np.median(sizes))}  "
              f"max={max(sizes)}  gdn_mean={np.mean(lin):.1f}  "
              f"qsa_mean={np.mean(qsa):.1f}", flush=True)
        self.phase = "decode"


class HookedMoe:
    """Instance wrapper so `layer.mlp(x)` hits this, not SparseMoeBlock.__call__."""

    def __init__(self, mlp, layer_i: int, is_linear: bool, box: dict):
        self.mlp = mlp
        self.i = layer_i
        self.is_linear = is_linear
        self.box = box
        self.top_k = int(mlp.top_k)
        self.n_exp = int(mlp.gate.weight.shape[0])

    def __call__(self, x):
        lock: Lock = self.box["lock"]
        mlp = self.mlp
        if lock.gdn_shared and lock.phase == "decode" and self.is_linear:
            return mx.sigmoid(mlp.shared_expert_gate(x)) * mlp.shared_expert(x)
        g = mx.softmax(mlp.gate(x), axis=-1, precise=True)
        k = self.top_k
        if lock.phase == "decode" and lock.allowed[self.i] is not None:
            allowed = lock.allowed[self.i]
            keep = mx.zeros((self.n_exp,), dtype=g.dtype)
            keep = keep.at[mx.array(allowed)].add(1)
            g = mx.where(keep > 0, g, mx.array(-1e9, dtype=g.dtype))
            k = int(min(self.top_k, allowed.size))
            if lock.uniform:
                inds = mx.array(allowed.reshape(1, 1, -1))
                if x.ndim == 3:
                    inds = mx.broadcast_to(inds, (*x.shape[:2], allowed.size))
                elif x.ndim == 2:
                    inds = mx.broadcast_to(inds.reshape(1, -1), (x.shape[0], allowed.size))
                scores = mx.ones(inds.shape, dtype=g.dtype) / float(allowed.size)
                return self._finish(x, inds, scores)
        inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(g, inds, axis=-1)
        if mlp.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if lock.phase == "prefill":
            mx.eval(inds)
            lock.record(self.i, np.array(inds))
        return self._finish(x, inds, scores)

    def _finish(self, x, inds, scores):
        mlp = self.mlp
        xs = x if mlp._awq_noop else (x / mlp._awq_scale).astype(x.dtype)
        y = (mlp.switch_mlp(xs, inds) * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(mlp.shared_expert_gate(x)) * mlp.shared_expert(x)


def install(model, box: dict) -> None:
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, HookedMoe):
            layer.mlp.box = box
            box["lock"].is_linear[i] = layer.mlp.is_linear
            continue
        layer.mlp = HookedMoe(layer.mlp, i, bool(layer.is_linear), box)
        box["lock"].is_linear[i] = bool(layer.is_linear)


def as_ids(ids) -> list[int]:
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def greedy(model, tok, ids: list[int], n_new: int, lock: Lock, eos: set[int]):
    lock.phase = "prefill"
    lock.counts = [np.zeros(lock.n_experts, np.int64) for _ in range(len(lock.counts))]
    lock.allowed = [None] * len(lock.counts)
    cache = model.make_cache()
    t0 = time.time()
    logits = model(mx.array([ids]), cache=cache)[:, -1, :]
    mx.eval(logits)
    nxt = int(mx.argmax(logits, axis=-1).item())
    print(f"  prefill S={len(ids)}  argmax={nxt}  {time.time()-t0:.2f}s", flush=True)
    lock.freeze()
    out = []
    t_dec = []
    for _ in range(n_new):
        out.append(nxt)
        if nxt in eos:
            break
        t1 = time.time()
        logits = model(mx.array([[nxt]]), cache=cache)[:, -1, :]
        mx.eval(logits)
        nxt = int(mx.argmax(logits, axis=-1).item())
        t_dec.append(time.time() - t1)
    if t_dec:
        print(f"  decode {len(t_dec)} tok  {len(t_dec)/sum(t_dec):.1f} tok/s", flush=True)
    text = tok.decode(out)
    print(f"  ids {out}\n  text {text!r}", flush=True)
    return out, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/Qwen3.8-Flash-Next-MLX-4bit"))
    ap.add_argument("--tokens", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("loading 4-bit model ...", flush=True)
    t0 = time.time()
    model = load_prebuilt(args.model, strict=False)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    n_layers = len(model.model.layers)
    n_exp = int(model.model.layers[0].mlp.gate.weight.shape[0])
    eos = {tok.eos_token_id, 248046, 248044}

    prompts = {
        "The": [760],
        "OK": as_ids(tok.apply_chat_template(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            add_generation_prompt=True, tokenize=True)),
    }
    policies = [
        ("full", dict(k_keep=n_exp, gdn_shared=False, uniform=False)),
        ("union16", dict(k_keep=16, gdn_shared=False, uniform=False)),
        ("union32", dict(k_keep=32, gdn_shared=False, uniform=False)),
        ("union32_uniform", dict(k_keep=32, gdn_shared=False, uniform=False)),
        ("gdn_shared+qsa10", dict(k_keep=n_exp, gdn_shared=True, uniform=False)),
        ("gdn_shared+union32", dict(k_keep=32, gdn_shared=True, uniform=False)),
    ]
    # fix uniform flag on union32_uniform
    policies[3] = ("union32_uniform", dict(k_keep=32, gdn_shared=False, uniform=True))

    box = {"lock": None}
    results = {}
    for pname, kw in policies:
        print(f"\n=== policy {pname} ===", flush=True)
        for label, ids in prompts.items():
            print(f"-- {label}  prompt_n={len(ids)} ids[:8]={ids[:8]}", flush=True)
            lock = Lock(n_layers, n_exp, **kw)
            box["lock"] = lock
            install(model, box)
            out, _ = greedy(model, tok, ids, args.tokens, lock, eos)
            results[(pname, label)] = out

    print("\n=== first-token / prefix compare vs full ===", flush=True)
    for label in prompts:
        ref = results[("full", label)]
        print(f"full {label}: {ref}")
        for pname, _ in policies[1:]:
            got = results[(pname, label)]
            n = min(len(ref), len(got))
            match = sum(a == b for a, b in zip(ref, got))
            print(f"  {pname:22s}  match {match}/{n}  {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
