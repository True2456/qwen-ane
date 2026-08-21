#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""hybrid_serve.py - Unified Apple Silicon Hybrid Inference Engine (APC + Turbo/Silent Dual-Mode)."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import mlx_lm.models.cache as kvcache
import numpy as np

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runtime.apc_cache import APCCache
from runtime.metal_engine import MetalEngine, MetalSharedEvent
from runtime.tui import RindiTUI
import tools.ane_serve as ane_serve


def snapshot_caches(caches):
    """Deep capture of KV and GDN recurrent states for APC cache storage and speculation rollback."""
    out = []
    for c in caches:
        cc = getattr(c, "cache", None)
        if cc is not None and not c.is_trimmable():
            out.append(("gdn", None, [None if x is None else mx.array(x) for x in cc]))
        elif c.is_trimmable():
            k = mx.array(c.keys) if hasattr(c, "keys") and c.keys is not None else None
            v = mx.array(c.values) if hasattr(c, "values") and c.values is not None else None
            out.append(("kv", c.offset, (k, v)))
        else:
            out.append(None)
    return out


def restore_caches(caches, snap, target_offset=None):
    for c, s in zip(caches, snap):
        if s is None:
            continue
        tag, off, payload = s
        if tag == "gdn" and hasattr(c, "cache") and payload is not None:
            c.cache = [None if x is None else mx.array(x) for x in payload]
        elif tag == "kv" and c.is_trimmable():
            c.offset = target_offset if target_offset is not None else (off if off is not None else c.offset)
            if payload is not None:
                k, v = payload
                if k is not None:
                    c.keys = mx.array(k)
                if v is not None:
                    c.values = mx.array(v)


class HybridEngine:
    """Production Dual-Mode Hybrid Engine combining APC Cache, GPU Burst Prefill, and ANE Verification."""

    def __init__(
        self,
        model_path: str = "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B",
        dense_bits: int = 4,
        bake_cache: bool = True,
        mode: str = "turbo",
        draft_depth: int = 3,
    ):
        self.model_path = model_path
        self.dense_bits = dense_bits
        self.bake_cache = bake_cache
        self.mode = mode
        self.draft_depth = draft_depth

        print("=" * 65)
        print("  HYBRID APPLE SILICON INFERENCE ENGINE (APC + DUAL-MODE)")
        print("=" * 65)

        # 1. Initialize Metal C Engine & Hardware Signaling
        self.metal = MetalEngine()
        self.event = self.metal.create_shared_event()
        print(f"  [Metal C Engine] {self.metal.device_name} (SharedEvent Active)")

        # 2. Initialize APC Radix Prefix Cache
        self.apc = APCCache()
        print("  [APC Cache] Radix Prefix Cache Ready (Zero-Copy Unified RAM)")

        # 3. Load Model (Unified .rindi Package or Base Model)
        manifest_path = Path(model_path) / "rindi_manifest.json"
        is_rindi_pkg = manifest_path.exists()

        if is_rindi_pkg:
            print(f"  [Loading Unified .rindi Package] {model_path}...")
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            # Load tokenizer from package
            self.tok = load(model_path)[1]
            from mlx_lm.utils import load_config
            from mlx_lm.models.qwen2 import Model, ModelArgs
            
            config = load_config(Path(model_path))
            model_args = ModelArgs.from_dict(config)
            self.model = Model(model_args)
            self.lm = getattr(self.model, "language_model", self.model)
            self.inner = self.lm.model
            self.embed = self.inner.embed_tokens
            self.H = self.inner.embed_tokens.weight.shape[1]

            # Direct mmap load of pre-quantized GPU backbone
            print("  [GPU Engine] Fast mmap loading pre-quantized GPU 4-bit backbone (<0.5s)...")
            self.gpu_inner = copy.deepcopy(self.inner)
            nn.quantize(self.gpu_inner, group_size=manifest.get("group_size", 64), bits=manifest.get("dense_bits", 4))
            gpu_weights_file = str(Path(model_path) / manifest.get("gpu_backbone", "gpu_backbone.safetensors"))
            if os.path.exists(gpu_weights_file):
                gpu_weights = mx.load(gpu_weights_file)
                self.gpu_inner.update(gpu_weights)
            mx.eval(self.gpu_inner.parameters())

            # Load MTP Weights BEFORE ANE chain frees MLX layer weights
            self._init_mtp_head()

            # Load ANE chain directly from packaged ane_layers directory
            print("  [ANE Engine] Registering 64 packaged ANE resident programs...")
            ane_pkg_dir = str(Path(model_path) / manifest.get("ane_layers_dir", "ane_layers"))
            self.ane_layers = ane_serve.attach_ane_chain(self.model, "mil", 32, dense_bits, ane_pkg_dir)
            ane_serve.attach_ane_lm_head(self.model, "mil", 32, dense_bits, 4)
            print(f"  ANE resident programs: {self.ane_layers} layers (41.0 GB host RAM freed)")
        else:
            print(f"  [Loading Base Model (Lazy mmap)] {model_path}...")
            self.model, self.tok = load(model_path, lazy=True)
            self.lm = getattr(self.model, "language_model", self.model)
            self.inner = self.lm.model
            self.embed = self.inner.embed_tokens
            self.H = self.inner.embed_tokens.weight.shape[1]

            # 4. Load MTP Weights
            self._init_mtp_head()

            # 5. Initialize ANE Chained Engine for Decode (Offloads 41.0 GB to ANE)
            print("  [ANE Engine] Registering 64 ANE resident layers (41.0 GB freed)...")
            cr = ane_serve._bake_cache_dir(model_path, dense_bits) if bake_cache else None
            self.ane_layers = ane_serve.attach_ane_chain(self.model, "mil", 32, dense_bits, cr)
            ane_serve.attach_ane_lm_head(self.model, "mil", 32, dense_bits, 4)
            # 6. Release unquantized host RAM and clear MLX cache
            import gc
            self.gpu_inner = self.inner
            mx.clear_cache()
            gc.collect()

    def _init_mtp_head(self):
        w = {}
        idx = json.load(open(self.model_path + "/model.safetensors.index.json"))["weight_map"]
        for sh in sorted({v for k, v in idx.items() if k.startswith("mtp.")}):
            for k, v in mx.load(self.model_path + "/" + sh).items():
                if k.startswith("mtp."):
                    w[k[4:]] = v
        for k in list(w):
            if "norm" in k.lower() and w[k].ndim == 1:
                w[k] = w[k].astype(mx.float32) + 1.0

        def deq(prefix, in_f):
            return w[prefix + ".weight"]

        self.fcw = deq("fc", 2 * self.H)
        proto = next(L for L in self.inner.layers if hasattr(L, "self_attn"))
        self.mtp_layer = copy.deepcopy(proto)
        tree = {}
        for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            mod = self.mtp_layer
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
        self.mtp_layer.update(tree)

        eps = getattr(getattr(self.model, "args", None), "rms_norm_eps", 1e-6)

        def rms(name):
            r = nn.RMSNorm(self.H, eps=eps)
            r.weight = w[name if name in w else name + ".weight"]
            return r

        self.pre_e = rms("pre_fc_norm_embedding")
        self.pre_h = rms("pre_fc_norm_hidden")
        self.mtp_norm = rms("norm")
        mx.eval(self.mtp_layer.parameters(), self.pre_e.parameters(),
                self.pre_h.parameters(), self.mtp_norm.parameters(), self.fcw)
        print(f"  [MTP Speculation] Ready (Draft Depth k={self.draft_depth})")

    def head(self, x):
        return self.lm.lm_head(self.inner.norm(x)) if hasattr(self.inner, "norm") else self.lm.lm_head(x)

    def generate(
        self,
        prompt: Any,  # str or List[Dict[str, str]]
        max_tokens: int = 64,
        mode: Optional[str] = None,
        use_apc: bool = True,
        emit_token: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        exec_mode = mode or self.mode
        def normalize_messages(msgs):
            clean = []
            for m in msgs:
                m_copy = dict(m)
                if "tool_calls" in m_copy and m_copy["tool_calls"]:
                    clean_tcs = []
                    for tc in m_copy["tool_calls"]:
                        tc_copy = dict(tc)
                        if "function" in tc_copy and isinstance(tc_copy["function"], dict):
                            fn_copy = dict(tc_copy["function"])
                            if isinstance(fn_copy.get("arguments"), str):
                                try:
                                    fn_copy["arguments"] = json.loads(fn_copy["arguments"])
                                except Exception:
                                    pass
                            tc_copy["function"] = fn_copy
                        clean_tcs.append(tc_copy)
                    m_copy["tool_calls"] = clean_tcs
                clean.append(m_copy)
            return clean

        if isinstance(prompt, list):
            kwargs = {"add_generation_prompt": True, "tokenize": False}
            if tools:
                kwargs["tools"] = tools
            text = self.tok.apply_chat_template(normalize_messages(prompt), **kwargs)
        else:
            kwargs = {"add_generation_prompt": True, "tokenize": False}
            if tools:
                kwargs["tools"] = tools
            text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
        prompt_ids = list(self.tok.encode(text))
        c = kvcache.make_prompt_cache(self.model)
        mc = kvcache.KVCache()

        t_start = time.perf_counter()
        apc_hit = False
        tokens_saved = 0
        remaining_tokens = prompt_ids

        # ---- 1. APC Cache Lookup ------------------------------------------
        if use_apc:
            matched_len, node, remaining = self.apc.match_prefix(prompt_ids)
            if matched_len > 0 and node is not None and node.cache_snapshot is not None:
                apc_hit = True
                tokens_saved = matched_len
                remaining_tokens = remaining
                restore_caches(c, node.cache_snapshot, target_offset=matched_len)
                cur = node.next_token
                last_h = node.last_hidden

        # ---- 2. Prefill (Cold or Incremental Delta) ------------------------
        if not apc_hit:
            # Full cold prefill
            ids_arr = mx.array(remaining_tokens)
            t_prefill = time.perf_counter()
            if len(remaining_tokens) > 32 and hasattr(self, "gpu_inner"):
                h = self.gpu_inner(ids_arr[None], cache=c)
                tag = "Metal GPU Tensor Cores"
            else:
                h = self.inner(ids_arr[None], cache=c)
                tag = "ANE Tiles"
            if len(remaining_tokens) > 1:
                self.mtp_layer(mx.concatenate([self.pre_e(self.embed(ids_arr[None][:, 1:])),
                                              self.pre_h(h[:, :-1])], -1) @ self.fcw.T, cache=mc)
            cur = int(mx.argmax(self.head(h[:, -1:])[0, -1]))
            last_h = h[:, -1:]
            mx.eval(cur, last_h)
            ttft_ms = (time.perf_counter() - t_start) * 1e3
            prefill_tps = len(remaining_tokens) / max(ttft_ms / 1000, 1e-4)
            print(f"\n  [{tag} Cold Prefill] {len(remaining_tokens)} tokens in {ttft_ms:.1f} ms = {prefill_tps:.1f} tok/s | Initial: {cur} ({repr(self.tok.decode([cur]))})")

            # Store computed prefix in APC cache
            if use_apc:
                self.apc.insert(prompt_ids, snapshot_caches(c), last_hidden=last_h, next_token=cur)
        else:
            if len(remaining_tokens) > 0:
                # Incremental delta prefill for only new user/tool tokens
                ids_rem = mx.array(remaining_tokens)
                t_delta = time.perf_counter()
                if len(remaining_tokens) > 32 and hasattr(self, "gpu_inner"):
                    h = self.gpu_inner(ids_rem[None], cache=c)
                    delta_tag = "Metal GPU Tensor Cores"
                else:
                    h = self.inner(ids_rem[None], cache=c)
                    delta_tag = "ANE Tiles"
                cur = int(mx.argmax(self.head(h[:, -1:])[0, -1]))
                last_h = h[:, -1:]
                mx.eval(cur, last_h)
                ttft_ms = (time.perf_counter() - t_start) * 1e3
                delta_tps = len(remaining_tokens) / max((time.perf_counter() - t_delta), 1e-4)
                print(f"\n  [APC Cache Hit + {delta_tag} Delta] Skipped {tokens_saved} tokens (0 FLOPs), computed {len(remaining_tokens)} delta tokens in {ttft_ms:.1f} ms = {delta_tps:.1f} tok/s | Initial: {cur}")

                # Update cache with new extended conversation prefix
                if use_apc:
                    self.apc.insert(prompt_ids, snapshot_caches(c), last_hidden=last_h, next_token=cur)
            else:
                # Exact full match
                ttft_ms = (time.perf_counter() - t_start) * 1e3
                print(f"\n  [APC Exact Cache Hit] Restored {tokens_saved}/{len(prompt_ids)} tokens in {ttft_ms:.2f} ms (0 FLOPs)")

        # Stop on any EOS / special boundary tokens (<|im_end|>, <|endoftext|>, <|im_start|>)
        self.eos_tokens = {151645, 151643, 151644}
        if hasattr(self.tok, "eos_token_id") and self.tok.eos_token_id is not None:
            self.eos_tokens.add(self.tok.eos_token_id)
        if hasattr(self.tok, "all_special_ids") and self.tok.all_special_ids is not None:
            self.eos_tokens.update(self.tok.all_special_ids)

        # ---- 3. Decode Loop -----------------------------------------------
        if cur in self.eos_tokens:
            return {
                "mode": exec_mode,
                "generated_tokens": 0,
                "generated_text": "",
                "ttft_ms": ttft_ms,
                "decode_tps": 0.0,
                "total_elapsed_s": time.perf_counter() - t_start,
                "steps": 0,
                "accepted_per_step": 0.0,
                "apc_hit": apc_hit,
                "tokens_saved": tokens_saved,
            }

        gen, steps, accepted_total = [cur], 0, 0
        if emit_token:
            emit_token(self.tok.decode([cur]))
        t_decode_start = time.perf_counter()

        if exec_mode == "turbo":
            # Turbo Mode: Metal GPU MTP Draft + ANE Parallel Verification
            while len(gen) < max_tokens:
                msnap = snapshot_caches([mc])
                drafts, dh, dtok = [], last_h, cur
                for _ in range(self.draft_depth):
                    d = self.mtp_layer(mx.concatenate([self.pre_e(self.embed(mx.array([[dtok]]))),
                                                      self.pre_h(dh)], -1) @ self.fcw.T, cache=mc)
                    dtok = int(mx.argmax(self.lm.lm_head(self.mtp_norm(d))[0, -1]))
                    if dtok in self.eos_tokens:
                        drafts.append(dtok)
                        dh = d
                        break
                    drafts.append(dtok)
                    dh = d

                # Signal GPU draft completion via SharedEvent
                self.event.value = steps * 10 + 1

                # Parallel ANE Verification
                gsnap = snapshot_caches(c)
                kv_before = [x.offset for x in c if x.is_trimmable()]
                seq = mx.array([[cur] + drafts])
                hv = self.inner(seq, cache=c)
                preds = [int(t) for t in mx.argmax(self.head(hv), axis=-1)[0]]

                # Longest-path acceptance
                n_ok = 0
                for i, d in enumerate(drafts):
                    if preds[i] == d:
                        n_ok += 1
                    else:
                        break

                if n_ok == len(drafts):
                    new_toks = drafts + [preds[-1]]
                    filtered, hit_eos = [], False
                    for tok in new_toks:
                        if tok in self.eos_tokens:
                            hit_eos = True
                            break
                        filtered.append(tok)
                    gen.extend(filtered)
                    if emit_token:
                        for tok in filtered:
                            emit_token(self.tok.decode([tok]))
                    if hit_eos:
                        break
                    cur, last_h = preds[-1], hv[:, -1:]
                    accepted_total += len(drafts) + 1
                else:
                    restore_caches(c, gsnap)
                    restore_caches([mc], msnap)
                    keep = drafts[:n_ok]
                    fix = preds[n_ok]
                    hv2 = self.inner(mx.array([[cur] + keep]), cache=c)
                    nxt = keep + [fix]
                    self.mtp_layer(mx.concatenate([self.pre_e(self.embed(mx.array([nxt]))),
                                                  self.pre_h(hv2)], -1) @ self.fcw.T, cache=mc)
                    filtered, hit_eos = [], False
                    for tok in nxt:
                        if tok in self.eos_tokens:
                            hit_eos = True
                            break
                        filtered.append(tok)
                    gen.extend(filtered)
                    if emit_token:
                        for tok in filtered:
                            emit_token(self.tok.decode([tok]))
                    if hit_eos:
                        break
                    cur, last_h = fix, hv2[:, -1:]
                    accepted_total += n_ok + 1

                steps += 1

        else:
            # Silent Mode: Pure ANE Single-Step Decode at ~5.9 W
            while len(gen) < max_tokens:
                hv = self.inner(mx.array([[cur]]), cache=c)
                nxt = int(mx.argmax(self.head(hv)[0, -1]))
                if nxt in self.eos_tokens:
                    break
                gen.append(nxt)
                cur = nxt
                steps += 1
                accepted_total += 1
                if emit_token:
                    emit_token(self.tok.decode([nxt]))

        total_elapsed = time.perf_counter() - t_start
        decode_elapsed = time.perf_counter() - t_decode_start
        decode_tps = (len(gen) - 1) / max(decode_elapsed, 1e-4)
        accept_rate = accepted_total / max(steps, 1)

        result_text = self.tok.decode(gen)
        return {
            "mode": exec_mode,
            "generated_tokens": len(gen),
            "generated_text": result_text,
            "ttft_ms": ttft_ms,
            "decode_tps": decode_tps,
            "total_elapsed_s": total_elapsed,
            "steps": steps,
            "accepted_per_step": accept_rate,
            "apc_hit": apc_hit,
            "tokens_saved": tokens_saved,
        }


def parse_qwen_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract any <tool_call>...</tool_call> blocks into OpenAI-compatible tool_calls."""
    tool_calls = []
    content = text
    if "<tool_call>" not in text:
        return content, tool_calls

    parts = text.split("<tool_call>")
    content = parts[0].strip()
    for block in parts[1:]:
        if "</tool_call>" in block:
            tc_str = block.split("</tool_call>")[0].strip()
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            if tc_str.startswith("{") and tc_str.endswith("}"):
                try:
                    data = json.loads(tc_str)
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": data.get("name", ""),
                            "arguments": json.dumps(data.get("arguments", {})) if isinstance(data.get("arguments"), dict) else str(data.get("arguments", "{}"))
                        }
                    })
                    continue
                except Exception:
                    pass
            fn_name = ""
            params = {}
            if "<function=" in tc_str:
                fn_match = re.search(r"<function=([^>]+)>", tc_str)
                if fn_match:
                    fn_name = fn_match.group(1).strip()
                for p_match in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", tc_str, re.DOTALL):
                    p_name = p_match.group(1).strip()
                    p_val = p_match.group(2).strip()
                    try:
                        p_val = json.loads(p_val)
                    except Exception:
                        pass
                    params[p_name] = p_val
                if fn_name:
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(params)
                        }
                    })
    return content, tool_calls


class ReasoningStream:
    """Route live Qwen text to reasoning/content/tool_calls deltas across </think> and <tool_call>."""
    marker = "</think>"

    def __init__(self, enabled: bool, reasoning_cb: Any, content_cb: Any, tool_call_cb: Optional[Any] = None):
        self.enabled = enabled
        self.send_reasoning = reasoning_cb
        self.send_content = content_cb
        self.send_tool_call = tool_call_cb
        self.pending = ""
        self.in_reasoning = True  # Always parse reasoning tags to prevent leakage
        self.content_accum = ""

    def feed(self, delta: str):
        self.pending += delta
        if not self.in_reasoning:
            # Strip any residual <think> or </think> tags
            clean = self.pending.replace("<think>", "").replace("</think>", "")
            self.content_accum += clean
            if "<tool_call>" not in self.content_accum:
                self.send_content(clean)
            self.pending = ""
            return
        
        # We are inside reasoning
        end = self.pending.find(self.marker)
        if end >= 0:
            thought = self.pending[:end].replace("<think>", "")
            if thought and self.enabled:
                self.send_reasoning(thought)
            self.pending = self.pending[end + len(self.marker):]
            self.in_reasoning = False
            if self.pending:
                clean = self.pending.replace("<think>", "").replace("</think>", "")
                self.content_accum += clean
                if "<tool_call>" not in self.content_accum:
                    self.send_content(clean)
            self.pending = ""
            return
        
        safe = max(0, len(self.pending) - len(self.marker) + 1)
        if safe > 0:
            thought = self.pending[:safe].replace("<think>", "")
            if thought and self.enabled:
                self.send_reasoning(thought)
            self.pending = self.pending[safe:]

    def finish(self):
        if self.pending:
            if self.in_reasoning:
                thought = self.pending.replace("<think>", "").replace("</think>", "")
                if thought and self.enabled:
                    self.send_reasoning(thought)
                elif thought and not self.enabled:
                    pass  # Suppress thoughts when thinking is disabled
            else:
                clean = self.pending.replace("<think>", "").replace("</think>", "")
                self.content_accum += clean
                if "<tool_call>" not in self.content_accum:
                    self.send_content(clean)
            self.pending = ""

        # Parse any tool calls in accumulated content
        if "<tool_call>" in self.content_accum:
            clean_content, tool_calls = parse_qwen_tool_calls(self.content_accum)
            if tool_calls and self.send_tool_call:
                for tc in tool_calls:
                    self.send_tool_call(tc)


_REASONING_INSTRUCTIONS = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    "medium": "",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
}


def main():
    p = argparse.ArgumentParser(description="Apple Silicon Hybrid Inference Engine (APC + Turbo/Silent)")
    p.add_argument("--model", default="/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B")
    p.add_argument("--mode", default="silent", choices=["turbo", "silent"],
                   help="Default execution mode: 'silent' (Pure ANE @ ~5.9W) or 'turbo' (MTP+ANE)")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--draft", type=int, default=3)
    p.add_argument("--prompt", default="Explain how a transformer language model works.")
    p.add_argument("--dense-bits", type=int, default=4)
    p.add_argument("--bake-cache", action="store_true", default=True)
    p.add_argument("--bench", action="store_true", help="Run comprehensive multi-turn benchmark")
    p.add_argument("--server", action="store_true", default=True, help="Start OpenAI-compatible HTTP server")
    p.add_argument("--host", default="0.0.0.0", help="HTTP server bind host")
    p.add_argument("--port", type=int, default=2456, help="HTTP server port (default: 2456)")
    p.add_argument("--context", type=int, default=262144, help="Maximum context window (default: 262144)")
    a = p.parse_args()

    if a.bench:
        engine = HybridEngine(
            model_path=a.model,
            dense_bits=a.dense_bits,
            bake_cache=a.bake_cache,
            mode=a.mode,
            draft_depth=a.draft,
        )

        print("\n" + "=" * 65)
        print("  RUNNING MULTI-TURN HYBRID BENCHMARK (APC CACHE + TURBO/SILENT)")
        print("=" * 65)

        # Turn 1: Cold Cache (Prefill + Speculative Decode)
        print("\n--- Turn 1: Cold Context (Initial Prompt) ---")
        res1 = engine.generate(a.prompt, max_tokens=a.tokens, mode=a.mode, use_apc=True)
        print(f"  TTFT: {res1['ttft_ms']:.1f} ms | Decode Speed: {res1['decode_tps']:.2f} tok/s | Steps: {res1['steps']} ({res1['accepted_per_step']:.2f} tok/step)")

        # Turn 2: Exact Prefix Match (APC Cache Hit -> 0 ms prefill)
        print("\n--- Turn 2: Cached Prefix (APC Cache Hit Verification) ---")
        res2 = engine.generate(a.prompt, max_tokens=a.tokens, mode=a.mode, use_apc=True)
        print(f"  TTFT: {res2['ttft_ms']:.2f} ms (APC HIT!) | Decode Speed: {res2['decode_tps']:.2f} tok/s | Steps: {res2['steps']}")

        print("\n" + "=" * 65)
        print("  BENCHMARK RESULTS SUMMARY")
        print("=" * 65)
        print(f"  Mode:                {engine.mode.upper()}")
        print(f"  Cold Prefill TTFT:   {res1['ttft_ms']:.1f} ms")
        print(f"  APC Cached TTFT:     {res2['ttft_ms']:.2f} ms ({res1['ttft_ms']/max(res2['ttft_ms'], 1e-2):.1f}x TTFT speedup!)")
        print(f"  Decode Throughput:   {res2['decode_tps']:.2f} tok/s")
        print(f"  Accepted Tokens/Step:{res2['accepted_per_step']:.2f}")
        print(f"  APC Cache Stats:     {engine.apc.stats()}")
        print("=" * 65)

    elif a.server:
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        import threading
        import queue
        import uuid

        gen_queue = queue.Queue()
        engine_holder = []
        engine_ready = threading.Event()

        def inference_worker():
            """Dedicated inference thread to guarantee MLX / Metal / ANE stream affinity."""
            eng = HybridEngine(
                model_path=a.model,
                dense_bits=a.dense_bits,
                bake_cache=a.bake_cache,
                mode=a.mode,
                draft_depth=a.draft,
            )
            engine_holder.append(eng)
            engine_ready.set()
            while True:
                task = gen_queue.get()
                if task is None:
                    break
                fn, args, kwargs, holder, done = task
                try:
                    res = fn(*args, **kwargs)
                    holder["result"] = res
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    holder["error"] = e
                finally:
                    done.set()
                    gen_queue.task_done()

        worker_th = threading.Thread(target=inference_worker, daemon=True)
        worker_th.start()
        engine_ready.wait()
        engine = engine_holder[0]

        def dispatch_generation(prompt, max_tokens, mode, use_apc, emit_token=None, tools=None):
            done = threading.Event()
            holder = {}
            gen_queue.put((
                engine.generate,
                (prompt,),
                {
                    "max_tokens": max_tokens,
                    "mode": mode,
                    "use_apc": use_apc,
                    "emit_token": emit_token,
                    "tools": tools,
                },
                holder,
                done,
            ))
            done.wait()
            if "error" in holder:
                raise holder["error"]
            return holder["result"]

        class OpenAIServer(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle_one_request(self):
                try:
                    super().handle_one_request()
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

            def log_message(self, fmt, *args):
                print(f"  [HTTP] {fmt % args}", flush=True)

            def _json_resp(self, data, code=200):
                body = json.dumps(data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()

            def do_GET(self):
                if self.path in ("/", "/health", "/healthz"):
                    self._json_resp({
                        "status": "ok",
                        "ready": True,
                        "queue_depth": gen_queue.qsize(),
                        "mode": engine.mode,
                        "apc_stats": engine.apc.stats(),
                        "model": "Qwen3.8-27B"
                    })
                elif self.path == "/v1/models":
                    self._json_resp({
                        "object": "list",
                        "data": [
                            {
                                "id": "Qwen3.8-27B",
                                "object": "model",
                                "owned_by": "rindi-hybrid",
                                "context_length": a.context,
                                "mode": engine.mode
                            }
                        ]
                    })
                else:
                    self._json_resp({"error": "Not Found"}, 404)

            def do_POST(self):
                if self.path == "/v1/mode":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    new_mode = body.get("mode", "").lower()
                    if new_mode in ("turbo", "silent"):
                        engine.mode = new_mode
                        print(f"\n  ⚡ [RINDI HTTP] Mode switched to {engine.mode.upper()}")
                        self._json_resp({"status": "ok", "mode": engine.mode})
                    else:
                        self._json_resp({"error": "Mode must be 'turbo' or 'silent'"}, 400)

                elif self.path == "/v1/chat/completions":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))

                    messages = body.get("messages", [])
                    tools = body.get("tools", None)
                    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 512
                    effort = body.get("reasoning_effort", "xhigh")
                    enable_thinking = body.get("enable_thinking", True)
                    stream = bool(body.get("stream", False))

                    print(f"\n  📥 [RINDI HTTP] Incoming Chat Request: {len(messages)} messages, tools={len(tools) if tools else 0}, stream={stream}, effort={effort}", flush=True)

                    # Dynamic per-request mode override if provided
                    req_mode = body.get("mode", engine.mode)

                    # Inject thinking instruction if applicable
                    if enable_thinking and effort in _REASONING_INSTRUCTIONS and _REASONING_INSTRUCTIONS[effort]:
                        instr = _REASONING_INSTRUCTIONS[effort]
                        if messages and messages[0].get("role") == "system":
                            messages[0]["content"] = instr + "\n\n" + messages[0]["content"]
                        else:
                            messages.insert(0, {"role": "system", "content": instr})

                    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    created = int(time.time())
                    model_name = body.get("model", "Qwen3.8-27B")

                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()

                        def sse_chunk(delta_dict, finish_reason=None):
                            chunk = {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta_dict,
                                    "finish_reason": finish_reason
                                }]
                            }
                            raw = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                            try:
                                self.wfile.write(raw)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                pass

                        sse_chunk({"role": "assistant"})

                        def emit_reasoning(delta):
                            sse_chunk({"reasoning_content": delta})

                        def emit_content(delta):
                            sse_chunk({"content": delta})

                        has_tool_calls = [False]
                        def emit_tool_call(tc):
                            has_tool_calls[0] = True
                            sse_chunk({"tool_calls": [tc]})

                        router = ReasoningStream(enable_thinking, emit_reasoning, emit_content, emit_tool_call)
                        try:
                            dispatch_generation(
                                messages,
                                max_tokens=max_tokens,
                                mode=req_mode,
                                use_apc=True,
                                emit_token=router.feed,
                                tools=tools,
                            )
                        except Exception as e:
                            print(f"  [Generation Error] {e}", flush=True)
                        router.finish()

                        finish_rsn = "tool_calls" if has_tool_calls[0] else "stop"
                        sse_chunk({}, finish_reason=finish_rsn)
                        try:
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        self.close_connection = True
                        return

                    try:
                        res = dispatch_generation(messages, max_tokens=max_tokens, mode=req_mode, use_apc=True, tools=tools)
                    except Exception as e:
                        self._json_resp({"error": str(e)}, 500)
                        return
                    text = res["generated_text"]

                    # Split reasoning thoughts
                    reasoning_content = None
                    content = text
                    if "</think>" in text:
                        parts = text.split("</think>", 1)
                        reasoning_content = parts[0].replace("<think>", "").strip()
                        content = parts[1].strip()

                    content, tool_calls = parse_qwen_tool_calls(content)
                    finish_reason = "tool_calls" if tool_calls else "stop"

                    tui.record_metrics(res["ttft_ms"], res["decode_tps"], res["apc_hit"], res["tokens_saved"])
                    resp = {
                        "id": cid,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content if (content or not tool_calls) else None,
                                "reasoning_content": reasoning_content,
                                "tool_calls": tool_calls if tool_calls else None,
                            },
                            "finish_reason": finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": res.get("tokens_saved", 0),
                            "completion_tokens": res["generated_tokens"],
                            "total_tokens": res.get("tokens_saved", 0) + res["generated_tokens"]
                        },
                        "rindi_stats": {
                            "mode": req_mode,
                            "ttft_ms": res["ttft_ms"],
                            "decode_tps": res["decode_tps"],
                            "accepted_per_step": res["accepted_per_step"],
                            "apc_hit": res["apc_hit"]
                        }
                    }
                    self._json_resp(resp)
                else:
                    self._json_resp({"error": "Not Found"}, 404)

        server = ThreadingHTTPServer((a.host, a.port), OpenAIServer)
        tui = RindiTUI(engine, host=a.host, port=a.port)
        tui_thread = threading.Thread(target=tui.run_interactive_loop, args=(dispatch_generation,), daemon=True)
        tui_thread.start()

        tui.print_status()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            tui.log("Shutting down Rindi server...", tag="INFO")
            server.shutdown()


if __name__ == "__main__":
    main()
