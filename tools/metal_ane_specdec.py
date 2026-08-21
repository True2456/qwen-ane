#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""metal_ane_specdec.py - Heterogeneous Metal GPU Drafter + ANE Parallel Verifier."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import mlx_lm.models.cache as kvcache
import numpy as np

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runtime.metal_engine import MetalEngine, MetalSharedEvent
import tools.ane_serve as ane_serve


def snapshot(caches):
    """Copy the GDN recurrent states; KV caches are trimmed instead."""
    out = []
    for c in caches:
        cc = getattr(c, "cache", None)
        if cc is not None and not c.is_trimmable():
            out.append([None if x is None else mx.array(x) for x in cc])
        else:
            out.append(None)
    return out


def restore(caches, snap):
    for c, s in zip(caches, snap):
        if s is not None:
            c.cache = [None if x is None else mx.array(x) for x in s]


def main():
    p = argparse.ArgumentParser(description="Heterogeneous Metal GPU + ANE Speculative Decoding")
    p.add_argument("--model", default="/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B")
    p.add_argument("--mtp", default=None)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--draft", type=int, default=3)
    p.add_argument("--prompt", default="Explain how a transformer language model works.")
    p.add_argument("--dense-bits", type=int, default=4)
    p.add_argument("--bake-cache", action="store_true", default=True)
    p.add_argument("--gpu-prefill", action="store_true", default=True,
                   help="Run high-throughput burst prefill on Metal GPU before ANE decode")
    a = p.parse_args()

    print("=" * 60)
    print("  HETEROGENEOUS METAL GPU DRAFTER + ANE VERIFIER")
    print("=" * 60)

    # 1. Initialize Metal C Engine
    metal = MetalEngine()
    event = metal.create_shared_event()
    print(f"  [GPU Engine] {metal.device_name} (SharedEvent Active)")

    # 2. Load Model & MTP Head
    print("  [Loading Model] Qwen3.8-27B...")
    model, tok = load(a.model)
    lm = getattr(model, "language_model", model)
    inner = lm.model
    embed = inner.embed_tokens
    H = inner.embed_tokens.weight.shape[1]

    # Load MTP weights
    w, q = {}, {}
    if a.mtp:
        for f in sorted(glob.glob(a.mtp + "/*.safetensors")):
            w.update(mx.load(f))
        q = json.load(open(a.mtp + "/config.json")).get("quantization") or {}
    else:
        idx = json.load(open(a.model + "/model.safetensors.index.json"))["weight_map"]
        for sh in sorted({v for k, v in idx.items() if k.startswith("mtp.")}):
            for k, v in mx.load(a.model + "/" + sh).items():
                if k.startswith("mtp."):
                    w[k[4:]] = v
        for k in list(w):
            if "norm" in k.lower() and w[k].ndim == 1:
                w[k] = w[k].astype(mx.float32) + 1.0

    def deq(prefix, in_f):
        if prefix + ".scales" not in w:
            return w[prefix + ".weight"]
        sc, bi = w[prefix + ".scales"], w[prefix + ".biases"]
        bits = (q.get(prefix) or {}).get("bits", q.get("bits", 4))
        return mx.dequantize(w[prefix + ".weight"], sc, bi,
                             group_size=in_f // sc.shape[1], bits=bits, mode="affine")

    fcw = deq("fc", 2 * H)
    proto = next(L for L in inner.layers if hasattr(L, "self_attn"))
    mtp_layer = copy.deepcopy(proto)
    tree = {}
    for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        mod = mtp_layer
        for p_name in path.split("."):
            mod = getattr(mod, p_name)
        node, parts = tree, path.split(".")
        for p_name in parts[:-1]:
            node = node.setdefault(p_name, {})
        node[parts[-1]] = {"weight": deq(f"layers.0.{path}", mod.weight.shape[1])}
    for path in ("input_layernorm", "post_attention_layernorm",
                 "self_attn.q_norm", "self_attn.k_norm"):
        key = f"layers.0.{path}.weight"
        if key not in w:
            continue
        node, parts = tree, path.split(".")
        for p_name in parts[:-1]:
            node = node.setdefault(p_name, {})
        node[parts[-1]] = {"weight": w[key]}
    mtp_layer.update(tree)

    eps = getattr(getattr(model, "args", None), "rms_norm_eps", 1e-6)

    def rms(name):
        r = nn.RMSNorm(H, eps=eps)
        r.weight = w[name if name in w else name + ".weight"]
        return r

    pre_e, pre_h, mtp_norm = (rms("pre_fc_norm_embedding"), rms("pre_fc_norm_hidden"),
                              rms("norm"))
    mx.eval(mtp_layer.parameters(), pre_e.parameters(), pre_h.parameters(),
            mtp_norm.parameters(), fcw)
    print(f"  [MTP Drafter] Ready (Draft Depth k={a.draft})")

    def head(x):
        return lm.lm_head(inner.norm(x)) if hasattr(inner, "norm") else lm.lm_head(x)

    # 3. Prefill Prompt
    text = tok.apply_chat_template(
        [{"role": "user", "content": a.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = mx.array(tok.encode(text))
    c = kvcache.make_prompt_cache(model)
    mc = kvcache.KVCache()

    if a.gpu_prefill:
        print(f"  [GPU Burst Prefill] Processing {ids.shape[0]} prompt tokens on Metal GPU...", flush=True)
        t_prefill = time.perf_counter()
        h = inner(ids[None], cache=c)
        if ids.shape[0] > 1:
            mtp_layer(mx.concatenate([pre_e(embed(ids[None][:, 1:])),
                                      pre_h(h[:, :-1])], -1) @ fcw.T, cache=mc)
        cur = int(mx.argmax(head(h[:, -1:])[0, -1]))
        last_h = h[:, -1:]
        mx.eval(cur, last_h)
        ttft_ms = (time.perf_counter() - t_prefill) * 1e3
        prefill_tok_s = ids.shape[0] / max(ttft_ms / 1000, 1e-4)
        print(f"  [GPU Prefill Complete] {ids.shape[0]} tokens in {ttft_ms:.1f} ms = {prefill_tok_s:.1f} tok/s | Initial Token: {cur} ({repr(tok.decode([cur]))})")

        # 4. Attach Chained ANE Verifier for Sustained Speculative Decode
        print("  [ANE Verifier] Compiling & attaching 64-layer ANE chain...", flush=True)
        cr = ane_serve._bake_cache_dir(a.model, a.dense_bits) if a.bake_cache else None
        n = ane_serve.attach_ane_chain(model, "mil", 32, a.dense_bits, cr)
        ane_serve.attach_ane_lm_head(model, "mil", 32, a.dense_bits, 4)
        print(f"  ANE chained layers attached: {n}")
    else:
        # 4. Attach Chained ANE Verifier before prefill
        print("  [ANE Verifier] Compiling 64-layer ANE chain...", flush=True)
        cr = ane_serve._bake_cache_dir(a.model, a.dense_bits) if a.bake_cache else None
        n = ane_serve.attach_ane_chain(model, "mil", 32, a.dense_bits, cr)
        ane_serve.attach_ane_lm_head(model, "mil", 32, a.dense_bits, 4)
        print(f"  ANE chained layers attached: {n}")

        t_prefill = time.perf_counter()
        h = inner(ids[None], cache=c)
        if ids.shape[0] > 1:
            mtp_layer(mx.concatenate([pre_e(embed(ids[None][:, 1:])),
                                      pre_h(h[:, :-1])], -1) @ fcw.T, cache=mc)
        cur = int(mx.argmax(head(h[:, -1:])[0, -1]))
        last_h = h[:, -1:]
        mx.eval(cur, last_h)
        ttft_ms = (time.perf_counter() - t_prefill) * 1e3
        prefill_tok_s = ids.shape[0] / max(ttft_ms / 1000, 1e-4)
        print(f"  [ANE Prefill Complete] {ids.shape[0]} tokens in {ttft_ms:.1f} ms = {prefill_tok_s:.1f} tok/s | Initial Token: {cur} ({repr(tok.decode([cur]))})")

    # 5. Speculative Generation Loop (GPU Draft -> ANE Verify)
    gen, steps, accepted_total = [cur], 0, 0
    t0 = time.perf_counter()

    print("\n  Executing Speculative Pipeline...", flush=True)

    while len(gen) < a.tokens:
        # A. GPU Draft Step (Fast MTP Top-K with Metal C engine)
        msnap = snapshot([mc])
        drafts, dh, dtok = [], last_h, cur
        for _ in range(a.draft):
            d = mtp_layer(mx.concatenate([pre_e(embed(mx.array([[dtok]]))),
                                          pre_h(dh)], -1) @ fcw.T, cache=mc)
            dtok = int(mx.argmax(lm.lm_head(mtp_norm(d))[0, -1]))
            drafts.append(dtok)
            dh = d

        # Signal GPU draft completion via SharedEvent
        event.value = steps * 10 + 1

        # B. ANE Verification Step (Evaluate all candidate tokens in ONE chained pass)
        gsnap = snapshot(c)
        kv_before = [x.offset for x in c if x.is_trimmable()]
        seq = mx.array([[cur] + drafts])
        hv = inner(seq, cache=c)
        preds = [int(t) for t in mx.argmax(head(hv), axis=-1)[0]]

        # C. GPU Longest-Path Acceptance Verification
        n_ok = 0
        for i, d in enumerate(drafts):
            if preds[i] == d:
                n_ok += 1
            else:
                break

        if n_ok == a.draft:
            # Full acceptance
            gen.extend(drafts + [preds[-1]])
            cur, last_h = preds[-1], hv[:, -1:]
            accepted_total += a.draft + 1
        else:
            # Prefix acceptance + target correction
            restore(c, gsnap)
            for x, off in zip([y for y in c if y.is_trimmable()], kv_before):
                x.offset = off
            restore([mc], msnap)
            keep = drafts[:n_ok]
            fix = preds[n_ok]
            hv2 = inner(mx.array([[cur] + keep]), cache=c)
            nxt = keep + [fix]
            mtp_layer(mx.concatenate([pre_e(embed(mx.array([nxt]))),
                                      pre_h(hv2)], -1) @ fcw.T, cache=mc)
            gen.extend(nxt)
            cur, last_h = fix, hv2[:, -1:]
            accepted_total += n_ok + 1

        steps += 1
        if cur in tok.eos_token_ids:
            break

    elapsed = time.perf_counter() - t0
    tok_s = len(gen) / elapsed
    accept_per_step = accepted_total / max(steps, 1)

    print("\n" + "=" * 60)
    print(f"  {len(gen)} tokens generated in {elapsed:.2f}s = {tok_s:.2f} tok/s")
    print(f"  {steps} verification steps, {accept_per_step:.2f} accepted tokens/step (k={a.draft})")
    print(f"  Sample: {repr(tok.decode(gen[:32]))}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
