#!/usr/bin/env python3
"""MTP speculative decoding for Qwen3.8, with optional ANE MLPs.

Why this matters on the ANE: its decode dispatch costs the same whether 1 or 32
tokens are in flight (1.867 vs 2.007 ms measured), so accepted tokens per step
multiply throughput almost linearly. The GPU gains far less -- its cost scales
with tokens.

Cache handling is the hard part for this architecture. 48 of the 64 layers are
linear_attn, whose ArraysCache holds a recurrent state that cannot be trimmed --
you cannot un-apply a recurrence -- so mlx_lm's speculative path refuses the
model outright. Instead the GDN states are snapshotted before each verify.

On a partial rejection the state could be restored and the accepted prefix
replayed, but that costs a second dispatch. This uses ACCEPT-ALL-OR-ONE: if every
draft matches, keep all k+1 tokens; otherwise restore and keep the single token
the verify already proved correct. Always correct, one dispatch per step.

The bf16 checkpoint's MTP norms use the "minus one" convention; +1 is applied.
"""
import argparse, glob, json, os, sys, time
import copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models import cache as kvcache

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B")
ap.add_argument("--mtp", default=None, help="separate -mtp dir; default uses the in-model head")
ap.add_argument("--draft", type=int, default=2, help="draft depth k")
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--ane-layers", type=int, default=0, help="dense MLP layers to bake on ANE")
ap.add_argument("--dense-bits", type=int, default=8)
ap.add_argument("--ane-lm-head", action="store_true",
                help="bake lm_head onto the ANE too")
ap.add_argument("--lm-head-chunks", type=int, default=4)
ap.add_argument("--engine", default=os.environ.get(
    "Q38_ANE_ENGINE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap.add_argument("--prompt", default="Explain how a transformer language model works.")
a = ap.parse_args()

print("loading target...", flush=True)
model, tok = load(a.model)
lm = getattr(model, "language_model", model)
inner = lm.model
embed = inner.embed_tokens
H = inner.embed_tokens.weight.shape[1]

if a.ane_layers or a.ane_lm_head:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import ane_serve
    if a.ane_layers:
        n = ane_serve.attach_ane_dense(model, a.ane_layers, a.engine, 32, a.dense_bits)
        print(f"  ANE dense MLP layers: {n}", flush=True)
    if a.ane_lm_head:
        ane_serve.attach_ane_lm_head(model, a.engine, 32, a.dense_bits,
                                     a.lm_head_chunks)

# ---- MTP head -------------------------------------------------------------
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
    # "minus one" convention -- match "norm" anywhere, pre_fc_norm_embedding and
    # pre_fc_norm_hidden end in embedding/hidden and matter most
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
print(f"  MTP head ready (draft depth {a.draft})", flush=True)


def head(x):
    return lm.lm_head(inner.norm(x)) if hasattr(inner, "norm") else lm.lm_head(x)


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
while len(gen) < a.tokens:
    # ---- draft k tokens with the MTP head -------------------------------
    msnap = snapshot([mc])
    drafts, dh, dtok = [], last_h, cur
    for _ in range(a.draft):
        d = mtp_layer(mx.concatenate([pre_e(embed(mx.array([[dtok]]))),
                                      pre_h(dh)], -1) @ fcw.T, cache=mc)
        dtok = int(mx.argmax(lm.lm_head(mtp_norm(d))[0, -1]))
        drafts.append(dtok)
        dh = d

    # ---- verify all k+1 in ONE pass -------------------------------------
    gsnap = snapshot(c)
    kv_before = [x.offset for x in c if x.is_trimmable()]
    seq = mx.array([[cur] + drafts])
    hv = inner(seq, cache=c)
    preds = [int(t) for t in mx.argmax(head(hv), axis=-1)[0]]

    n_ok = 0
    for i, d in enumerate(drafts):
        if preds[i] == d:
            n_ok += 1
        else:
            break

    if n_ok == a.draft:                      # every draft matched: keep them all
        gen.extend(drafts + [preds[-1]])
        cur, last_h = preds[-1], hv[:, -1:]
        accepted_total += a.draft + 1
    else:
        # Longest-prefix acceptance. The first n_ok drafts were right, and
        # preds[n_ok] is the target's own correction, so n_ok+1 tokens are
        # proven. Advancing the state means replaying [cur] + drafts[:n_ok]:
        # the KV caches could just be rewound, but the GDN layers hold a running
        # recurrence with no trim, so the whole prefix is re-run. That second
        # pass is n_ok+1 tokens wide and decode is flat to T=64, so it costs
        # about what a single token costs.
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

el = time.perf_counter() - t0
print(f"\n  {len(gen)} tokens in {el:.2f}s = {len(gen)/el:.1f} tok/s")
print(f"  {steps} steps, {accepted_total/max(1,steps):.2f} tokens/step "
      f"(draft depth {a.draft})")
print(f"  sample: {tok.decode(gen[:60])!r}")
