#!/usr/bin/env python3
"""Tree-based speculative decoding on Apple Neural Engine (Qwen3.8-27B).

Leverages the 64 parallel matrix lanes on the ANE to verify an 8-16 node
candidate tree in a single 3.5ms pass, maximizing accepted tokens per step.
"""

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
from mlx_lm.models import cache as kvcache

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(Path.home() / ".lmstudio/models/Qwen/Qwen3.8-27B"))
ap.add_argument("--mtp", default=None, help="separate -mtp dir; default uses in-model head")
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--top-b1", type=int, default=3, help="tree branching factor depth 1")
ap.add_argument("--top-b2", type=int, default=2, help="tree branching factor depth 2")
ap.add_argument("--ane-chain", action="store_true", help="fuse layer tails + next-layer heads on ANE")
ap.add_argument("--bake-cache", action="store_true", help="use ~/.cache/ane_bake cache")
ap.add_argument("--dense-bits", type=int, default=4)
ap.add_argument("--ane-lm-head", action="store_true", help="bake lm_head onto ANE")
ap.add_argument("--lm-head-chunks", type=int, default=4)
ap.add_argument("--engine", default=os.environ.get("Q38_ANE_ENGINE", str(_REPO_ROOT)))
ap.add_argument("--prompt", default="Explain how a transformer language model works.")
a = ap.parse_args()

print("loading target model...", flush=True)
model, tok = load(a.model)
lm = getattr(model, "language_model", model)
inner = lm.model
embed = inner.embed_tokens
H = inner.embed_tokens.weight.shape[1]

# ---- MTP head initialization -----------------------------------------------
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
    for p in path.split("."):
        mod = getattr(mod, p)
    node, parts = tree, path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = {"weight": deq(f"layers.0.{path}", mod.weight.shape[1])}
for path in ("input_layernorm", "post_attention_layernorm",
             "self_attn.q_norm", "self_attn.k_norm"):
    key = f"layers.0.{path}.weight"
    if key not in w:
        continue
    node, parts = tree, path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
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
print(f"  MTP tree drafter ready (b1={a.top_b1}, b2={a.top_b2})", flush=True)

# ---- ANE Layer Attachment --------------------------------------------------
if a.ane_chain or a.ane_lm_head:
    import ane_serve
    cr = ane_serve._bake_cache_dir(a.model, a.dense_bits) if a.bake_cache else None
    if a.ane_chain:
        n = ane_serve.attach_ane_chain(model, a.engine, 32, a.dense_bits, cr)
        print(f"  ANE chained layers: {n}", flush=True)
    if a.ane_lm_head:
        ane_serve.attach_ane_lm_head(model, a.engine, 32, a.dense_bits, a.lm_head_chunks)


def head(x):
    return lm.lm_head(inner.norm(x)) if hasattr(inner, "norm") else lm.lm_head(x)


def snapshot(caches):
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


ids = mx.array(tok.encode(tok.apply_chat_template(
    [{"role": "user", "content": a.prompt}], add_generation_prompt=True, tokenize=False)))

c = kvcache.make_prompt_cache(model)
mc = kvcache.KVCache()
h = inner(ids[None], cache=c)
if ids.shape[0] > 1:
    mtp_layer(mx.concatenate([pre_e(embed(ids[None][:, 1:])),
                              pre_h(h[:, :-1])], -1) @ fcw.T, cache=mc)
cur = int(mx.argmax(head(h[:, -1:])[0, -1]))
last_h = h[:, -1:]

gen, steps, accepted_total = [cur], 0, 0
t0 = time.perf_counter()

def make_batched_cache(caches, num_paths):
    """Broadcast cache state (e.g. conv_state and KV keys/values) across batch dimension for tree verification."""
    batched = []
    for layer_cache in caches:
        c_copy = copy.copy(layer_cache)
        if hasattr(c_copy, "keys") and c_copy.keys is not None and c_copy.keys.ndim >= 1 and c_copy.keys.shape[0] == 1:
            c_copy.keys = mx.repeat(c_copy.keys, num_paths, axis=0)
            c_copy.values = mx.repeat(c_copy.values, num_paths, axis=0)
        cc = getattr(layer_cache, "cache", None)
        if cc is not None:
            new_cc = []
            for item in cc:
                if item is not None and item.ndim >= 2 and item.shape[0] == 1:
                    new_cc.append(mx.repeat(item, num_paths, axis=0))
                else:
                    new_cc.append(item)
            c_copy.cache = new_cc
        batched.append(c_copy)
    return batched


print("\nStarting Tree Speculative Decoding...", flush=True)

while len(gen) < a.tokens:
    msnap = snapshot([mc])
    
    # 1. Draft Candidate Tree using MTP head
    # Depth 1: Token 2 candidates
    d1 = mtp_layer(mx.concatenate([pre_e(embed(mx.array([[cur]]))), pre_h(last_h)], -1) @ fcw.T, cache=mc)
    top1_ids = [int(x) for x in mx.topk(lm.lm_head(mtp_norm(d1))[0, -1], k=a.top_b1)]

    # Depth 2: Token 3 candidates
    d1_mc_snap = snapshot([mc])
    d1_kv_off = [x.offset for x in [mc] if x.is_trimmable()]

    paths = []
    for t1 in top1_ids:
        restore([mc], d1_mc_snap)
        for x, off in zip([y for y in [mc] if y.is_trimmable()], d1_kv_off):
            x.offset = off

        d2 = mtp_layer(mx.concatenate([pre_e(embed(mx.array([[t1]]))), pre_h(d1)], -1) @ fcw.T, cache=mc)
        top2_ids = [int(x) for x in mx.topk(lm.lm_head(mtp_norm(d2))[0, -1], k=a.top_b2)]
        for t2 in top2_ids:
            paths.append([t1, t2])

    # 2. Parallel Verification on ANE (Batched candidate tree in ONE forward pass)
    gsnap = snapshot(c)
    kv_before = [x.offset for x in c if x.is_trimmable()]

    # Construct single batched matrix of all candidate paths [num_paths, 1 + depth]
    num_paths = len(paths)
    batched_seq = mx.array([[cur] + path for path in paths])
    c_batched = make_batched_cache(c, num_paths)
    hv = inner(batched_seq, cache=c_batched)
    all_preds = mx.argmax(head(hv), axis=-1)

    best_accept_path = []
    best_target_fix = int(all_preds[0, 0])

    # Evaluate all candidate paths in parallel
    for path_idx, path in enumerate(paths):
        preds = [int(t) for t in all_preds[path_idx]]
        n_ok = 0
        for i, dtoken in enumerate(path):
            if preds[i] == dtoken:
                n_ok += 1
            else:
                break
        accepted_prefix = path[:n_ok]
        target_fix = preds[n_ok]

        if len(accepted_prefix) >= len(best_accept_path):
            best_accept_path = accepted_prefix
            best_target_fix = target_fix

    # 3. Commit Longest Accepted Path
    restore(c, gsnap)
    for x, off in zip([y for y in c if y.is_trimmable()], kv_before):
        x.offset = off
    restore([mc], msnap)

    full_step_tokens = best_accept_path + [best_target_fix]
    hv2 = inner(mx.array([[cur] + best_accept_path]), cache=c)
    mtp_layer(mx.concatenate([pre_e(embed(mx.array([full_step_tokens]))),
                              pre_h(hv2)], -1) @ fcw.T, cache=mc)

    gen.extend(full_step_tokens)
    cur = best_target_fix
    last_h = hv2[:, -1:]
    accepted_total += len(full_step_tokens)
    steps += 1

    if cur in tok.eos_token_ids:
        break

el = time.perf_counter() - t0
print("=" * 60)
print(f"  {len(gen)} tokens generated in {el:.2f}s = {len(gen)/el:.1f} tok/s")
print(f"  {steps} steps, {accepted_total/max(1,steps):.2f} accepted tokens/step")
print(f"  sample text: {tok.decode(gen[:60])!r}")
print("=" * 60)
