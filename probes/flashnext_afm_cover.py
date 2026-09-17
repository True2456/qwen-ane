#!/usr/bin/env python3
"""AFM-style covering-set measurement, then optional ANE stacked-pin bench.

Records Flash-Next top-10 routes during MLX greedy generate and reports the
pin size / refresh cadence that would keep decode on a baked ANE expert set.

    PYTHONPATH=~/.mlx128/mlx-lm:~/.mlx128/mlx/python \\
      python3 -u probes/flashnext_afm_cover.py \\
      cover --max-new 600

    ... pin-bench --k 16,32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.afm_cover import covering_report, format_report  # noqa: E402

MODEL = os.environ.get(
    "FLASHNEXT_MLX4", str(Path.home() / "models/Qwen3.8-Flash-Next-MLX-4bit")
)
OUT = ROOT / "artifacts" / "coreai"
DEFAULT_PROMPT = (
    "The 2016-17 season marked a pivotal chapter in modern basketball, as "
    "analytics, load management, and three-point volume collided with old-school "
    "interior play. Coaches argued about pace, general managers hunted two-way "
    "wings, and every box score was a referendum on shot quality. Role players "
    "who could stretch the floor became more valuable than traditional backs, "
    "and the best teams treated the regular season as a long experiment in "
    "lineup combinations rather than a march of identical box scores. "
    "What followed was a debate about whether the sport had discovered a new "
    "equilibrium or simply a fashionable extreme that would later snap back."
)


def _patch_mx_unique() -> None:
    import mlx.core as mx

    if hasattr(mx, "unique"):
        return

    def _unique(a, *_a, **_k):
        return mx.array(np.unique(np.array(a)))

    mx.unique = _unique


def _decoder_layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("could not find decoder layers on loaded model")
    return layers


class RouteTrace:
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self._pending: list[tuple | None] = [None] * n_layers
        self.chunks: list[tuple[np.ndarray, np.ndarray]] = []

    def install(self, model) -> None:
        from mlx_lm.models.qwen4_exp import SparseMoeBlock

        layers = _decoder_layers(model)
        by_id = {id(layer.mlp): i for i, layer in enumerate(layers)}
        orig = SparseMoeBlock.__call__
        self._orig_moe_call = orig

        def wrapped(_moe, x):
            import mlx.core as mx

            i = by_id.get(id(_moe))
            if i is not None:
                g = mx.softmax(_moe.gate(x), axis=-1, precise=True)
                k = _moe.top_k
                inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(g, inds, axis=-1)
                if _moe.norm_topk_prob:
                    scores = scores / scores.sum(axis=-1, keepdims=True)
                self._pending[i] = (inds, scores)
            return orig(_moe, x)

        SparseMoeBlock.__call__ = wrapped

    def uninstall(self) -> None:
        if getattr(self, "_orig_moe_call", None) is None:
            return
        from mlx_lm.models.qwen4_exp import SparseMoeBlock

        SparseMoeBlock.__call__ = self._orig_moe_call
        self._orig_moe_call = None

    def flush(self) -> tuple[np.ndarray, np.ndarray]:
        import mlx.core as mx

        missing = [i for i, p in enumerate(self._pending) if p is None]
        if missing:
            raise RuntimeError(f"route trace missing layers {missing[:8]}...")
        arrays = []
        for inds, scores in self._pending:
            mx.eval(inds, scores)
            arrays.append((
                np.array(inds.astype(mx.int32)),
                np.array(scores.astype(mx.float32)),
            ))
        self._pending = [None] * self.n_layers
        # each: [B, S, K] -> [S, K]
        inds = np.stack([a[0].reshape(-1, a[0].shape[-1]) for a in arrays], axis=1)
        scores = np.stack(
            [a[1].reshape(-1, a[1].shape[-1]).astype(np.float32) for a in arrays],
            axis=1,
        )
        self.chunks.append((inds, scores))
        return inds, scores

    def stack(self) -> tuple[np.ndarray, np.ndarray]:
        inds = np.concatenate([c[0] for c in self.chunks], axis=0)
        scores = np.concatenate([c[1] for c in self.chunks], axis=0)
        return inds, scores


def cmd_self_test() -> int:
    rng = np.random.default_rng(0)
    t, l, k = 40, 8, 10
    # Sticky toy: each layer cycles a pool of 24 experts.
    pools = [rng.choice(512, size=24, replace=False) for _ in range(l)]
    inds = np.empty((t, l, k), np.int32)
    for layer in range(l):
        for tok in range(t):
            inds[tok, layer] = rng.choice(pools[layer], size=k, replace=False)
    scores = np.full((t, l, k), 1.0 / k, np.float32)
    types = np.array(["linear_attention", "qsa"] * 4)
    rep = covering_report(inds, scores, n_prompt=16, types=types)
    text = format_report(rep)
    print(text)
    assert rep["unique_end"]["all"] <= 24.1
    hits = {row["pin"]: row["decode_hit"] for row in rep["freq_pin"]}
    assert hits[24] >= 0.99, hits
    assert hits[10] < hits[24]
    print("self-test ok")
    return 0


def cmd_cover(args: argparse.Namespace) -> int:
    _patch_mx_unique()
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load

    OUT.mkdir(parents=True, exist_ok=True)
    t_load = time.perf_counter()
    model, tokenizer = load(MODEL)
    print(
        f"loaded in {time.perf_counter() - t_load:.1f}s  "
        f"active {mx.get_active_memory() / 1e9:.1f} GB",
        flush=True,
    )
    layers = _decoder_layers(model)
    types = np.array([getattr(ly, "layer_type", "") for ly in layers])
    trace = RouteTrace(len(layers))
    trace.install(model)
    try:
        return _cmd_cover_run(args, model, tokenizer, mx, layers, types, trace)
    finally:
        trace.uninstall()


def _cmd_cover_run(args, model, tokenizer, mx, layers, types, trace) -> int:
    from mlx_lm.models.cache import make_prompt_cache

    if args.prompt_ids:
        ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    else:
        text = args.prompt if args.prompt else DEFAULT_PROMPT
        if args.prompt_file:
            text = Path(args.prompt_file).read_text()
        ids = tokenizer.encode(text, add_special_tokens=False)
    if args.prompt_len and args.prompt_len > len(ids):
        rng = np.random.default_rng(0)
        extra = rng.integers(0, 32_000, size=args.prompt_len - len(ids))
        ids = ids + extra.tolist()
    prompt = list(ids)
    print(f"prompt n={len(prompt)}  max_new={args.max_new}", flush=True)

    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(mx.array([prompt]), cache=cache)
    mx.eval(logits)
    trace.flush()
    nxt = int(mx.argmax(logits[:, -1, :]).item())
    generated = [nxt]
    print(
        f"  prefill {len(prompt)} tok  {time.perf_counter() - t0:.2f}s  "
        f"first={nxt}",
        flush=True,
    )
    t_dec = time.perf_counter()
    for i in range(args.max_new - 1):
        logits = model(mx.array([[nxt]]), cache=cache)
        mx.eval(logits)
        trace.flush()
        nxt = int(mx.argmax(logits[:, -1, :]).item())
        generated.append(nxt)
        if (i + 2) % 50 == 0 or i + 2 == args.max_new:
            dt = time.perf_counter() - t_dec
            print(
                f"  decode {i + 2}/{args.max_new}  {dt:.1f}s  "
                f"{(i + 2) / dt:.2f} tok/s",
                flush=True,
            )

    inds, scores = trace.stack()
    n_prompt = len(prompt)
    if inds.shape[0] < n_prompt:
        n_prompt = inds.shape[0]
    npz = OUT / "flashnext_afm_routes.npz"
    np.savez_compressed(
        npz,
        inds=inds,
        scores=scores.astype(np.float16),
        n_prompt=np.int32(n_prompt),
        types=types.astype("U32"),
        prompt=np.asarray(prompt, np.int32),
        generated=np.asarray(generated, np.int32),
    )
    print(f"saved {npz}  inds={inds.shape}  prompt_tokens={n_prompt}", flush=True)
    try:
        print(f"generated text={tokenizer.decode(generated)!r}", flush=True)
    except Exception:
        print(f"generated ids={generated[:16]}...", flush=True)

    rep = covering_report(inds, scores, n_prompt=n_prompt, types=types)
    growth = rep.pop("unique_growth_tl")
    print(format_report(rep), flush=True)
    summary = {k: v for k, v in rep.items() if k != "unique_growth_tl"}
    (OUT / "flashnext_afm_cover.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    np.save(OUT / "flashnext_afm_union_growth.npy", growth)
    print(f"wrote {OUT / 'flashnext_afm_cover.json'}", flush=True)
    return 0


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def cmd_from_npz(path: Path) -> int:
    z = np.load(path, allow_pickle=False)
    inds = z["inds"]
    scores = z["scores"].astype(np.float32)
    n_prompt = int(z["n_prompt"])
    types = z["types"] if "types" in z.files else None
    rep = covering_report(inds, scores, n_prompt=n_prompt, types=types)
    rep.pop("unique_growth_tl", None)
    print(format_report(rep))
    return 0


def cmd_pin_bench(ks: list[int], seq: int, repeats: int) -> int:
    """Stacked-expert AneDynamicLinear: compile once, page once, eval many."""
    os.environ.setdefault("Q38_ANE_ENGINE", str(ROOT))
    from runtime.q38_ane_engine import AneDynamicLinear

    h, inner = 2560, 640
    rng = np.random.default_rng(0)
    x = rng.standard_normal((seq, h), dtype=np.float32).astype(np.float32) * 0.02
    print(f"pin-bench seq={seq}  repeats={repeats}  K={ks}", flush=True)
    for k in ks:
        o_gu = k * 2 * inner
        o_dn = h
        i_dn = k * inner
        t0 = time.perf_counter()
        gu = AneDynamicLinear.compile(h, o_gu, seq)
        dn = AneDynamicLinear.compile(i_dn, o_dn, seq)
        t_comp = time.perf_counter() - t0
        if gu is None or dn is None:
            print(f"  K={k}  compile FAILED  {t_comp:.2f}s", flush=True)
            continue
        w_gu = rng.standard_normal((o_gu, h), dtype=np.float32).astype(np.float32) * 0.02
        w_dn = rng.standard_normal((o_dn, i_dn), dtype=np.float32).astype(np.float32) * 0.02
        t0 = time.perf_counter()
        gu.write_weight(w_gu)
        dn.write_weight(w_dn)
        t_page = time.perf_counter() - t0
        y = gu.evaluate(x, as_float32=True)
        if y is None:
            print(f"  K={k}  first eval FAILED", flush=True)
            continue
        h_act = rng.standard_normal((seq, i_dn), dtype=np.float32).astype(np.float32) * 0.02
        dn.evaluate(h_act, as_float32=True)
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            gu.evaluate(x, as_float32=True)
            dn.evaluate(h_act, as_float32=True)
            ts.append((time.perf_counter() - t) * 1e3)
        ts.sort()
        med = ts[len(ts) // 2]
        bytes_fp16 = (o_gu * h + o_dn * i_dn) * 2
        print(
            f"  K={k:<3}  compile={t_comp:.2f}s  page={t_page*1e3:.1f}ms  "
            f"eval pair median={med:.2f} ms  "
            f"48L={48*med:.0f} ms/tok ({1000/(48*med):.2f} tok/s MoE-only)  "
            f"fp16={bytes_fp16/1e6:.1f} MB/layer",
            flush=True,
        )
        # page-every-token (restage) cost
        ts_p = []
        for _ in range(max(3, repeats // 3)):
            w_gu = rng.standard_normal((o_gu, h), dtype=np.float32).astype(np.float32) * 0.02
            w_dn = rng.standard_normal((o_dn, i_dn), dtype=np.float32).astype(np.float32) * 0.02
            t = time.perf_counter()
            gu.write_weight(w_gu)
            dn.write_weight(w_dn)
            gu.evaluate(x, as_float32=True)
            dn.evaluate(h_act, as_float32=True)
            ts_p.append((time.perf_counter() - t) * 1e3)
        ts_p.sort()
        print(
            f"         restage+eval median={ts_p[len(ts_p)//2]:.2f} ms/layer  "
            f"({ts_p[len(ts_p)//2]*48:.0f} ms/tok if every layer restages)",
            flush=True,
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cover")
    c.add_argument("--max-new", type=int, default=600)
    c.add_argument("--prompt", default="")
    c.add_argument("--prompt-file", default="")
    c.add_argument("--prompt-ids", default="")
    c.add_argument("--prompt-len", type=int, default=0)
    sub.add_parser("self-test")
    z = sub.add_parser("from-npz")
    z.add_argument("path", type=Path, nargs="?", default=OUT / "flashnext_afm_routes.npz")
    b = sub.add_parser("pin-bench")
    b.add_argument("--k", default="10,16,32")
    b.add_argument("--seq", type=int, default=32)
    b.add_argument("--repeats", type=int, default=11)
    args = p.parse_args()
    if args.cmd == "self-test":
        return cmd_self_test()
    if args.cmd == "from-npz":
        return cmd_from_npz(args.path)
    if args.cmd == "pin-bench":
        ks = [int(x) for x in args.k.split(",") if x.strip()]
        return cmd_pin_bench(ks, args.seq, args.repeats)
    return cmd_cover(args)


if __name__ == "__main__":
    raise SystemExit(main())
