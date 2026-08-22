#!/usr/bin/env python3
"""Standalone OpenAI-compatible server for A/B testing the ANE, no oMLX patching.

    ane_serve.py --model <path> --ane-layers 4          # serve on :1239
    ane_serve.py --model <path> --ane-layers 0          # GPU baseline
    ane_serve.py --model <path> --bench --ane-layers 4  # A/B, no server

Why standalone: patching oMLX means bundle edits that App Management blocks,
that oMLX updates wipe, and that make it hard to tell whether a change came from
the ANE or from oMLX's own MoE patches. This owns the whole path.

The four things that broke the first attempt, fixed here:
  1. SSE declared HTTP/1.1 with neither Content-Length nor chunking -> clients
     saw "Connection error". Now sends Connection: close.
  2. Qwen emits tool calls as XML; OpenAI clients expect structured tool_calls.
  3. Clients send content as a list of blocks and tool results as role=tool.
  4. Qwen's chat template iterates function.arguments as a MAPPING, but OpenAI
     sends it as a JSON string -> "Can only get item pairs from a mapping".

ANE notes (measured, docs/ANE-MOE-HANDOFF.md):
  * The ANE returns ZEROS below S=32, silently. S is clamped to 32.
  * Experts are stacked so each conv is wide: gate/up along output channels,
    down along its INPUT axis, since sum_e down_e @ a_e == [down_0|..] @ [a_0;..].
  * Prefill is expert-major (one pass per expert over its tokens); token-major
    prefill measured 0.3 tok/s under a real agent, expert-major 21x faster.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import gc
import os
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import numpy as np
import mlx.core as mx

# Not an ANE limit: 500 resident programs holding 20.99 GB compile fine
# (artifacts/ane_probes/ane_program_limit.py). The 126 failure seen earlier was
# host memory pressure -- the full bf16 MLX model stays resident alongside the
# blobs. Freeing the MLX-side weights after baking is the real fix; this cap is
# only a guard rail.
FREE_MLX = True
FREED = [0]
ANE_MAX_PROGRAMS = int(os.environ.get("ANE_MAX_PROGRAMS", "127"))
# Widest output a pad+add "concat" can produce. A plain conv output reaches
# 62080, but the padded-add path stops between 9216 and 11264 channels, which
# caps how much of the next layer's projection can ride along in a chained
# program (H + ANE_MAX_CHAIN_PROJ must stay under it).
ANE_MAX_CHAIN_PROJ = 4096
ANE_MIN_SEQ = 32
TOP_K = 8


def _sweep_ane_tempdirs(max_age_s=3600):
    """Delete stale ANE compiler scratch dirs.

    Every compile_multiproc writes the full weight blobs into a fresh
    content-addressed dir under $TMPDIR and nothing ever removes them -- 267 MB
    per 27B layer, so repeated 64-layer bakes fill the disk. Only dirs older
    than max_age_s are touched, so a concurrent process is left alone.
    """
    import glob, shutil as _sh
    now, freed, n = time.time(), 0, 0
    for mil in glob.glob(os.path.join(tempfile.gettempdir(), "*_*_*", "model.mil")):
        d = os.path.dirname(mil)
        try:
            if now - os.path.getmtime(d) < max_age_s:
                continue
            freed += sum(os.path.getsize(os.path.join(r, f))
                         for r, _, fs in os.walk(d) for f in fs)
            _sh.rmtree(d, ignore_errors=True); n += 1
        except OSError:
            pass
    if n:
        print(f"  swept {n} stale ANE scratch dirs ({freed/1e9:.1f} GB)", flush=True)
    return freed


_MY_TEMPDIRS = set()


def _cleanup_my_tempdirs():
    import shutil as _sh
    for d in _MY_TEMPDIRS:
        _sh.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_my_tempdirs)

# ---------------------------------------------------------------- ANE backend
class AneMoE:
    """Stacked-expert MoE on the ANE. Shape-only programs shared by all layers."""

    def __init__(self, engine_path: str, H: int, M: int, S: int):
        if engine_path not in sys.path:
            sys.path.insert(0, engine_path)
        import runtime.q38_ane_engine as E
        from runtime.q38_ane_engine import AneDynamicLinear
        self.E, self.H, self.M = E, H, M
        self.S = max(ANE_MIN_SEQ, S)
        self.MS = TOP_K * M
        self.view = E._iosurface_view
        self.lock = threading.Lock()

        def mk(mil, ind, outd):
            E.generate_dynamic_linear_mil = lambda a, b, c: mil
            return AneDynamicLinear.compile(ind, outd, self.S)

        self.gu = mk(self._gu_mil(self.MS), H, 2 * self.MS)
        self.dn = mk(self._dn_mil(self.MS), self.MS, H)
        self.gu1 = mk(self._gu_mil(M), H, 2 * M)
        self.dn1 = mk(self._dn_mil(M), M, H)
        if None in (self.gu, self.dn, self.gu1, self.dn1):
            raise RuntimeError("ANE programs failed to compile")
        self.wg = np.empty((2 * self.MS, H), np.float16)
        self.wd = np.empty((H, self.MS), np.float16)
        self.wg1 = np.empty((2 * M, H), np.float16)
        self.wd1 = np.empty((H, M), np.float16)

    def _gu_mil(self, m):
        H, S = self.H, self.S
        return f"""program(1.3)
{self.E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {S}]> x, tensor<fp16, [1, {2*m}, 1, {H}]> wimg) {{
    tensor<int32, [4]> ws = const()[name=string("ws"), val=tensor<int32, [4]>([{2*m}, {H}, 1, 1])];
    tensor<fp16, [{2*m}, {H}, 1, 1]> w = reshape(shape=ws, x=wimg)[name=string("w")];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {2*m}, 1, {S}]> c = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("gu")];
    tensor<fp16, [1, {m}, 1, {S}]> g0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{m},1,{S}]), x=c)[name=string("g0")];
    tensor<fp16, [1, {m}, 1, {S}]> u0 = slice_by_index(begin=tensor<int32, [4]>([0,{m},0,0]), end=tensor<int32, [4]>([1,{2*m},1,{S}]), x=c)[name=string("u0")];
    tensor<fp16, [1, {m}, 1, {S}]> sg = sigmoid(x=g0)[name=string("sg")];
    tensor<fp16, [1, {m}, 1, {S}]> si = mul(x=g0, y=sg)[name=string("si")];
    tensor<fp16, [1, {m}, 1, {S}]> y = mul(x=si, y=u0)[name=string("y")];
  }} -> (y);
}}
// aneserve_gu_{m}_S{S}
"""

    def _dn_mil(self, m):
        """wimg arrives as [m, H] -- expert-major ROWS -- and is transposed
        in-graph to the conv's [H, m]. Staging the other way round writes column
        blocks with stride m, which measured 25 GB/s against 71 contiguous."""
        H, S = self.H, self.S
        return f"""program(1.3)
{self.E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {m}, 1, {S}]> x, tensor<fp16, [1, {m}, 1, {H}]> wimg) {{
    tensor<int32, [2]> w2s = const()[name=string("w2s"), val=tensor<int32, [2]>([{m}, {H}])];
    tensor<fp16, [{m}, {H}]> w2 = reshape(shape=w2s, x=wimg)[name=string("w2")];
    tensor<fp16, [{H}, {m}]> wt = transpose(perm=tensor<int32, [2]>([1, 0]), x=w2)[name=string("wt")];
    tensor<int32, [4]> ws = const()[name=string("ws"), val=tensor<int32, [4]>([{H}, {m}, 1, 1])];
    tensor<fp16, [{H}, {m}, 1, 1]> w = reshape(shape=ws, x=wt)[name=string("w")];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, {H}, 1, {S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("dn")];
  }} -> (y);
}}
// aneserve_dn_{m}_S{S}
"""

    def decode(self, G, U, D, x, sel, prob):
        # Stage directly into the weight IOSurfaces. The old path did
        # np.array(G[e]) -> staging buffer -> write_weight -> surface: three
        # copies, and np.array() on an MLX array is an eval-and-copy rather
        # than a memcpy. G/U/D here are already numpy fp16.
        H, M, MS, S = self.H, self.M, self.MS, self.S
        with self.lock:
            with self.view(self.gu._w_surf, (2*MS, H), np.float16) as w:
                for j, e in enumerate(sel):
                    e = int(e)
                    w[j*M:(j+1)*M] = G[e]
                    w[MS+j*M:MS+(j+1)*M] = U[e]
            with self.view(self.dn._w_surf, (MS, H), np.float16) as w:
                for j, e in enumerate(sel):
                    w[j*M:(j+1)*M] = D[int(e)]      # contiguous, D is [NE, M, H]
            with self.view(self.gu._x_surf, (H, S), np.float16) as d:
                d[:] = 0; d[:, 0] = x[0].astype(np.float16)
            self.gu.submit()
            with self.view(self.gu._y_surf, (MS, S), np.float16) as o:
                act = o[:, 0].astype(np.float32)
            act *= np.repeat(prob, M)
            with self.view(self.dn._x_surf, (MS, S), np.float16) as d:
                d[:] = 0; d[:, 0] = act.astype(np.float16)
            self.dn.submit()
            with self.view(self.dn._y_surf, (H, S), np.float16) as o:
                return o[:, 0].astype(np.float32).reshape(1, H)

    def prefill(self, G, U, D, x, idx, sc):
        H, M, S = self.H, self.M, self.S
        T = x.shape[0]
        out = np.zeros((T, H), np.float32)
        assign: dict[int, list] = {}
        for t in range(T):
            for j, e in enumerate(idx[t]):
                assign.setdefault(int(e), []).append((t, float(sc[t][j])))
        x16 = x.astype(np.float16)
        with self.lock:
            for e, items in assign.items():
                with self.view(self.gu1._w_surf, (2*M, H), np.float16) as w:
                    w[:M] = G[e]; w[M:] = U[e]
                with self.view(self.dn1._w_surf, (M, H), np.float16) as w:
                    w[:] = D[e]
                for c0 in range(0, len(items), S):
                    ch = items[c0:c0+S]
                    with self.view(self.gu1._x_surf, (H, S), np.float16) as d:
                        d[:] = 0
                        for k, (t, _) in enumerate(ch): d[:, k] = x16[t]
                    self.gu1.submit()
                    with self.view(self.gu1._y_surf, (M, S), np.float16) as o:
                        act = o[:, :len(ch)].copy()
                    with self.view(self.dn1._x_surf, (M, S), np.float16) as d:
                        d[:] = 0; d[:, :len(ch)] = act
                    self.dn1.submit()
                    with self.view(self.dn1._y_surf, (H, S), np.float16) as o:
                        y = o[:, :len(ch)].astype(np.float32)
                    for k, (t, p) in enumerate(ch):
                        out[t] += p * y[:, k]
        return out


SYNC_ONLY = False
HEAD_STATS = {"ms": 0.0, "n": 0}
ATTN_STATS = {"ms": 0.0, "n": 0}
LAYER_STATS = {"ms": 0.0, "n": 0}
GDNSTEP_STATS = {"ms": 0.0, "marshal_ms": 0.0, "resident_ms": 0.0, "n": 0}
FL_PHASE = {"cat": 0.0, "write": 0.0, "submit": 0.0, "read": 0.0, "mx": 0.0, "n": 0}
GDN_STATS = {"ms": 0.0, "n": 0}
PHASE = {"cast": 0.0, "write": 0.0, "submit": 0.0, "read": 0.0, "n": 0}
FUSE_GU = True
STATS = {"pos": 0, "ms": 0.0}


def attach_ane(model, n_layers: int, engine_path: str, seq: int):
    """Route the last n_layers' MoE blocks through the ANE. Returns count."""
    if n_layers == 0:
        return 0
    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers", [])
    blocks = [b.mlp for b in layers
              if getattr(getattr(b, "mlp", None), "switch_mlp", None) is not None]
    if not blocks:
        print("  no MoE blocks found (dense model?) -- staying on GPU")
        return 0
    if n_layers > 0:
        blocks = blocks[-n_layers:]

    sw = blocks[0].switch_mlp
    NE, M, H = sw.gate_proj.weight.shape
    ane = AneMoE(engine_path, H, M, seq)
    # numpy fp16 pools: ~1.6 GB per layer, but staging becomes a memcpy instead
    # of np.array() on an MLX array (an eval-and-copy). Measured: that
    # difference was ~7 ms per layer-position.
    gb = len(blocks) * NE * 3 * M * H * 2 / 1e9
    print(f"  pooling experts for {len(blocks)} layer(s): {gb:.1f} GB", flush=True)
    weights = {}
    for moe in blocks:
        sm = moe.switch_mlp
        g = np.ascontiguousarray(np.array(sm.gate_proj.weight.astype(mx.float16)))
        u = np.ascontiguousarray(np.array(sm.up_proj.weight.astype(mx.float16)))
        # down is [NE, H, M] in the model; transpose once at load to [NE, M, H]
        # so per-token staging writes contiguous rows (71 GB/s) instead of
        # column blocks with stride MS (25 GB/s). The graph transposes it back.
        d = np.ascontiguousarray(
            np.array(sm.down_proj.weight.astype(mx.float16)).transpose(0, 2, 1))
        weights[id(moe)] = (g, u, d)

    cls = type(blocks[0])
    orig = cls.__call__

    def patched(self, x, *args, **kwargs):
        w = weights.get(id(self))
        tv = bool(args[0]) if args else bool(kwargs.get("target_verify", False))
        if w is None or tv:
            return orig(self, x, *args, **kwargs)
        try:
            G, U, D = w
            shp = x.shape
            xf = np.array(x.astype(mx.float32)).reshape(-1, H)
            gl = np.array(self.gate(x).astype(mx.float32)).reshape(-1, NE)
            idx = np.argpartition(-gl, TOP_K, axis=1)[:, :TOP_K]
            sc = np.take_along_axis(gl, idx, axis=1)
            sc = np.exp(sc - sc.max(1, keepdims=True)); sc /= sc.sum(1, keepdims=True)
            t0 = time.perf_counter()
            y = (ane.prefill(G, U, D, xf, idx, sc.astype(np.float32)) if xf.shape[0] > 1
                 else ane.decode(G, U, D, xf, idx[0], sc[0].astype(np.float32)))
            STATS["ms"] += (time.perf_counter()-t0)*1e3; STATS["pos"] += xf.shape[0]
            res = mx.array(y.reshape(shp).astype(np.float32)).astype(x.dtype)
            sh = getattr(self, "shared_expert", None)
            if sh is not None:
                se = sh(x)
                g = getattr(self, "shared_expert_gate", None)
                if g is not None:
                    se = mx.sigmoid(g(x)) * se
                res = res + se
            return res
        except Exception:
            traceback.print_exc()
            return orig(self, x, *args, **kwargs)

    cls.__call__ = patched
    return len(blocks)



# ------------------------------------------------------- dense MLP on the ANE
def _bake_cache_dir(model_path, bits):
    """Where prebaked quantized blobs live for a given model+precision."""
    import hashlib, os
    key = hashlib.sha256(f"{os.path.abspath(model_path)}|{bits}".encode()).hexdigest()[:16]
    d = os.path.expanduser(f"~/.cache/ane_bake/{key}")
    os.makedirs(d, exist_ok=True)
    return d



_CACHE_MIN_FREE = 20e9   # keep this much boot-disk headroom, always
_CACHE_WARNED = []


def _quant_blob(nm, W, bits, cache=None, key=None, rows=1024):
    """Quantize one weight to ANE blobs, in row blocks, reusing a prebaked copy.

    Quantisation is 83% of a layer's bake time (0.45 s of 0.54 s for a fused
    layer; the MIL compile is only 0.09 s and is not cacheable -- ANECCompile
    re-runs from MIL every time).

    It is also where the memory went. Quantising a [34816, 5120] weight in one
    shot allocates a float32 copy plus a temporary per numpy op -- measured at
    27.5x the resulting blob size, which is what starved the ANE of the wired
    pages it needs and produced "Program load failure (0x50004)". Working in row
    blocks caps the transient at `rows * In * 4` bytes (~20 MB) and lets `W` be
    a list of pieces so the caller never builds a concatenated copy either.
    """
    out = {}
    cdir = os.path.join(cache, key) if (cache and key) else None
    if cdir and os.path.exists(os.path.join(cdir, "done")):
        for fn in os.listdir(cdir):
            if fn.endswith(".bin"):
                with open(os.path.join(cdir, fn), "rb") as fh:
                    out[fn.replace("__", nm, 1) if fn.startswith("__") else fn] = fh.read()
        if out:
            return out

    parts = W() if callable(W) else W
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    O = sum(p.shape[0] for p in parts)
    In = parts[0].shape[1]
    hi = (1 << (bits - 1)) - 1 if bits != 16 else 0

    if bits == 16:
        buf = np.empty((O, In), np.float16)
    elif bits == 4:
        buf = np.empty(O * In // 2, np.uint8)
    else:
        buf = np.empty(O * In, np.int8)
    scales = None if bits == 16 else np.empty((O, 1), np.float16)

    r0 = 0
    for part in parts:
        for a in range(0, part.shape[0], rows):
            blk = np.asarray(part[a:a+rows], np.float32)
            n = blk.shape[0]
            if bits == 16:
                buf[r0:r0+n] = blk.astype(np.float16)
            else:
                sc = np.abs(blk).max(axis=1, keepdims=True) / hi
                np.divide(blk, np.where(sc == 0, 1, sc), out=blk)
                np.rint(blk, out=blk)
                np.clip(blk, -hi - 1, hi, out=blk)
                q = blk.astype(np.int8)
                if bits == 4:
                    f = (q.reshape(-1).astype(np.uint8) & 0x0F)
                    buf[r0*In//2:(r0+n)*In//2] = (f[0::2] | (f[1::2] << 4))
                else:
                    buf[r0*In:(r0+n)*In] = q.reshape(-1)
                scales[r0:r0+n] = sc.astype(np.float16)
            r0 += n
            del blk

    out[f"{nm}.bin"] = buf.tobytes()
    if scales is not None:
        out[f"{nm}s.bin"] = scales.tobytes()
    del buf, scales

    if cdir and shutil.disk_usage(cdir if os.path.isdir(cdir)
                                  else os.path.dirname(cdir)).free < _CACHE_MIN_FREE:
        if not _CACHE_WARNED:
            print(f"  bake cache: under {_CACHE_MIN_FREE/1e9:.0f} GB free, "
                  f"not caching further blobs", flush=True)
            _CACHE_WARNED.append(1)
        cdir = None
    if cdir:
        os.makedirs(cdir, exist_ok=True)
        for fn, data in out.items():
            with open(os.path.join(cdir, fn.replace(nm, "__", 1)), "wb") as fh:
                fh.write(data)
        open(os.path.join(cdir, "done"), "wb").close()
    return out


class AneDenseMLP:
    """Baked gate/up/down for ONE dense layer, as compile-time consts.

    Dense needs no expert selection, so every weight is a const: no staging, and
    the ANE reads them at the ~150 GB/s blob rate instead of the ~44 GB/s it
    manages for dynamic weights. 64 layers at one program each is well under the
    ~127 resident-program ceiling.
    """

    def __init__(self, eng, E, view, g, u, d, S, bits, cache=None,
                 wide_prefill=False):
        """cache: directory holding prebaked quantized blobs for this layer.

        Quantization is ~0.32 s/layer and compilation ~0.27 s. Only the first is
        cacheable: ANECCompile runs from MIL every time, and preserving its
        content-addressed output directory saves just 1.1x, so the compiled
        program itself cannot be reused through this API.
        """
        self.eng, self.view = eng, view
        self.I, self.H = g.shape
        self.S = max(ANE_MIN_SEQ, S)
        blobs, decl = {}, []
        cached = {}
        if cache and os.path.exists(os.path.join(cache, "done")):
            for fn in os.listdir(cache):
                if fn.endswith(".bin"):
                    with open(os.path.join(cache, fn), "rb") as fh:
                        cached[fn] = fh.read()
        self.fuse = FUSE_GU
        if self.fuse:
            # One conv of 2I output channels instead of two of I. Decode on the
            # ANE is dispatch-bound (~0.65 ms/dispatch, flat in weight bits), so
            # collapsing 3 dispatches to 2 is worth more than any requantisation.
            # Per-output-channel scales survive the concat: each row keeps its own.
            wlist = (("gu", np.concatenate([g, u], 0), (2*self.I, self.H)),
                     ("d", d, d.shape))
        else:
            wlist = (("g", g, g.shape), ("u", u, u.shape), ("d", d, d.shape))
        for nm, W, (O, In) in wlist:
            if cached:
                blobs[f"{nm}.bin"] = cached[f"{nm}.bin"]
                if bits != 16:
                    blobs[f"{nm}s.bin"] = cached[f"{nm}s.bin"]
                decl.append(self._decl(nm, O, In, bits))
                continue
            Wf = np.asarray(W, np.float32)
            if bits == 16:
                blobs[f"{nm}.bin"] = Wf.astype(np.float16).tobytes()
                decl.append(self._decl(nm, O, In, bits))
            else:
                hi = (1 << (bits - 1)) - 1
                sc = np.abs(Wf).max(axis=1, keepdims=True) / hi
                q = np.clip(np.rint(Wf / np.where(sc == 0, 1, sc)), -hi-1, hi).astype(np.int8)
                if bits == 4:
                    f = (q.reshape(-1).astype(np.uint8) & 0x0F)
                    blobs[f"{nm}.bin"] = (f[0::2] | (f[1::2] << 4)).tobytes()
                else:
                    blobs[f"{nm}.bin"] = q.tobytes()
                blobs[f"{nm}s.bin"] = sc.astype(np.float16).tobytes()
                decl.append(self._decl(nm, O, In, bits))
        if cache and not cached:
            for fn, data in blobs.items():
                with open(os.path.join(cache, fn), "wb") as fh:
                    fh.write(data)
            open(os.path.join(cache, "done"), "wb").close()
        H, I, S_ = self.H, self.I, self.S
        if self.fuse:
            body = (
    f'    tensor<fp16, [1, {2*I}, 1, {S_}]> c = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=guw, x=x)[name=string("gu")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> gc = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{I},1,{S_}]), x=c)[name=string("g0")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> uc = slice_by_index(begin=tensor<int32, [4]>([0,{I},0,0]), end=tensor<int32, [4]>([1,{2*I},1,{S_}]), x=c)[name=string("u0")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> sg = sigmoid(x=gc)[name=string("sig")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> si = mul(x=gc, y=sg)[name=string("silu")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> ac = mul(x=si, y=uc)[name=string("act")];')
        else:
            body = (
    f'    tensor<fp16, [1, {I}, 1, {S_}]> gc = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)[name=string("gate")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> sg = sigmoid(x=gc)[name=string("sig")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> si = mul(x=gc, y=sg)[name=string("silu")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> uc = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=uw, x=x)[name=string("up")];\n'
    f'    tensor<fp16, [1, {I}, 1, {S_}]> ac = mul(x=si, y=uc)[name=string("act")];')
        mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {S_}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
{chr(10).join(decl)}
{body}
    tensor<fp16, [1, {H}, 1, {S_}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=dw, x=ac)[name=string("down")];
  }} -> (y);
}}
"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            self.prog = eng.compile_multiproc(mil, blobs, H, H, self.S)
        if self.prog is None:
            raise RuntimeError("dense MLP program failed to compile")
        eng._ensure_io(self.prog)
        self.nbytes = sum(len(b) for b in blobs.values())
        for pr in (self.prog,):
            try:
                loc = E._desc(E._msg(pr.model, "localModelPath"))
                if loc and loc != "(nil)":
                    _MY_TEMPDIRS.add(loc)
            except Exception:
                pass

        # Optional wider program for prefill. Measured on this MLP shape S=64
        # runs 13.7 TFLOP/s against S=32's 8.9 (and falls off after: 9.4 at 128,
        # 5.4 at 256), so it is worth ~1.5x on prefill. But a second program
        # holds its own copy of the weights -- 534 MB/layer instead of 267 --
        # and that exhausts ANE resources partway: a 64-layer bake stopped at
        # 17. Off by default; enable only when prefill dominates and the model
        # is small enough to afford double residency.
        self.SP, self.progp = self.S, None
        if wide_prefill:
            self.SP = 2 * self.S
            milp = mil.replace(f", 1, {self.S}]", f", 1, {self.SP}]")
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                self.progp = eng.compile_multiproc(milp, blobs, H, H, self.SP)
            if self.progp is not None:
                eng._ensure_io(self.progp)

    @staticmethod
    def _decl(nm, O, In, bits):
        if bits == 16:
            return (f'    tensor<fp16, [{O}, {In}, 1, 1]> {nm}w = const()'
                    f'[name=string("{nm}w"), val=tensor<fp16, [{O}, {In}, 1, 1]>'
                    f'(BLOBFILE(path=string("@model_path/weights/{nm}.bin"), offset=uint64(64)))];')
        dt = f"int{bits}"
        return (
f'''    tensor<{dt}, [{O}, {In}, 1, 1]> {nm}q = const()[name=string("{nm}q"), val=tensor<{dt}, [{O}, {In}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/{nm}.bin"), offset=uint64(64)))];
    tensor<fp16, [{O}, 1, 1, 1]> {nm}sc = const()[name=string("{nm}sc"), val=tensor<fp16, [{O}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/{nm}s.bin"), offset=uint64(64)))];
    tensor<fp16, [{O}, {In}, 1, 1]> {nm}w = constexpr_blockwise_shift_scale(data={nm}q, scale={nm}sc)[name=string("{nm}dq")];''')

    def __call__(self, xf):
        """xf [T, H] float32 -> [T, H] float32, chunked to the ANE width."""
        H = self.H
        T = xf.shape[0]
        # more than one token means prefill: use the wide program
        prog, S = ((self.progp, self.SP) if T > self.S and self.progp is not None
                   else (self.prog, self.S))
        out = np.zeros((T, H), np.float32)
        t = PHASE
        a = time.perf_counter()
        x16 = xf.astype(np.float16)
        t["cast"] += time.perf_counter() - a
        for c0 in range(0, T, S):
            n = min(S, T - c0)
            a = time.perf_counter()
            with self.view(prog._in_surf, (H, S), np.float16) as d:
                d[:] = 0
                d[:, :n] = x16[c0:c0+n].T
            t["write"] += time.perf_counter() - a
            a = time.perf_counter()
            self.eng.submit(prog, procedure_index=0)
            t["submit"] += time.perf_counter() - a
            a = time.perf_counter()
            with self.view(prog._out_surf, (H, S), np.float16) as o:
                out[c0:c0+n] = o[:, :n].T.astype(np.float32)
            t["read"] += time.perf_counter() - a
            t["n"] += 1
        return out



class AneSplitMLP:
    """One dense MLP layer split across ANE and GPU, running CONCURRENTLY.

    The two engines overlap almost perfectly -- 0.801 ms concurrent against
    1.544 summed -- so the ANE is free compute alongside the GPU rather than an
    alternative to it. SwiGLU splits on the intermediate axis:

        ANE:  gate[:k], up[:k], down[:, :k]   -> partial [T, H]
        GPU:  gate[k:], up[k:], down[:, k:]   -> partial [T, H]
        out = ANE partial + GPU partial

    down_proj splits along its INPUT axis, so the partials simply add. Measured
    on the 27B's MLP shape, ~55% on the ANE is the balance point and gives 1.70x
    over the GPU alone; past that the ANE becomes the bottleneck.
    """

    _pool = None

    def __init__(self, eng, E, view, g, u, d, S, bits, frac, cache=None):
        from concurrent.futures import ThreadPoolExecutor
        if AneSplitMLP._pool is None:
            AneSplitMLP._pool = ThreadPoolExecutor(max_workers=2)
        self.I, self.H = g.shape
        k = int(self.I * frac) // 64 * 64
        self.k = max(64, min(self.I - 64, k))
        self.ane = AneDenseMLP(eng, E, view, g[:self.k], u[:self.k],
                               np.ascontiguousarray(d[:, :self.k]), S, bits, cache)
        self.gpu_g = mx.array(g[self.k:])
        self.gpu_u = mx.array(u[self.k:])
        self.gpu_d = mx.array(np.ascontiguousarray(d[:, self.k:]))
        mx.eval(self.gpu_g, self.gpu_u, self.gpu_d)
        self.nbytes = self.ane.nbytes

    def __call__(self, xf, xm):
        """xf: [T,H] float32 for the ANE; xm: the same tokens as an mx.array."""
        import mlx.nn as _nn
        fut = AneSplitMLP._pool.submit(self.ane, xf)
        a = _nn.silu(xm @ self.gpu_g.T) * (xm @ self.gpu_u.T)
        gpu_part = a @ self.gpu_d.T
        mx.eval(gpu_part)
        ane_part = fut.result()
        return gpu_part + mx.array(ane_part).astype(gpu_part.dtype)



class AneLmHead:
    """lm_head on the ANE, chunked along the vocabulary.

    Qwen3.8-27B's head is [248320, 5120] -- 1.27G params, untied. That is far
    too many output channels for a single conv, so it is split into `chunks`
    programs of ~V/chunks rows each and their outputs concatenated. Each chunk
    is an independent resident model; at 8 chunks that is 8 programs on top of
    the 64 MLPs, still under the ~127 ceiling.

    Splitting along output rows is exact -- every row keeps its own scale and
    no partial sums cross a chunk boundary.
    """

    def __init__(self, eng, E, view, W, S, bits, chunks=8, cache=None):
        self.eng, self.view = eng, view
        self.V, self.H = W.shape
        self.S = max(ANE_MIN_SEQ, S)
        self.progs, self.spans = [], []
        step = -(-self.V // chunks)
        self.nbytes = 0
        for ci, v0 in enumerate(range(0, self.V, step)):
            v1 = min(v0 + step, self.V)
            Wc = np.ascontiguousarray(np.asarray(W[v0:v1], np.float32))
            O = v1 - v0
            blobs, decl = {}, []
            if bits == 16:
                blobs["h.bin"] = Wc.astype(np.float16).tobytes()
            else:
                hi = (1 << (bits - 1)) - 1
                sc = np.abs(Wc).max(axis=1, keepdims=True) / hi
                q = np.clip(np.rint(Wc / np.where(sc == 0, 1, sc)), -hi-1, hi).astype(np.int8)
                if bits == 4:
                    f = (q.reshape(-1).astype(np.uint8) & 0x0F)
                    blobs["h.bin"] = (f[0::2] | (f[1::2] << 4)).tobytes()
                else:
                    blobs["h.bin"] = q.tobytes()
                blobs["hs.bin"] = sc.astype(np.float16).tobytes()
            decl.append(AneDenseMLP._decl("h", O, self.H, bits))
            mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {self.H}, 1, {self.S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
{chr(10).join(decl)}
    tensor<fp16, [1, {O}, 1, {self.S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=hw, x=x)[name=string("head")];
  }} -> (y);
}}
"""
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                prog = eng.compile_multiproc(mil, blobs, self.H, O, self.S)
            if prog is None:
                raise RuntimeError(f"lm_head chunk {ci} ({O} rows) failed to compile")
            eng._ensure_io(prog)
            self.progs.append((prog, O))
            self.spans.append((v0, v1))
            self.nbytes += sum(len(b) for b in blobs.values())
            try:
                loc = E._desc(E._msg(prog.model, "localModelPath"))
                if loc and loc != "(nil)":
                    _MY_TEMPDIRS.add(loc)
            except Exception:
                pass

    def __call__(self, xf):
        """xf [T, H] float32 -> [T, V] float32."""
        T, S = xf.shape[0], self.S
        out = np.zeros((T, self.V), np.float32)
        x16 = xf.astype(np.float16)
        for c0 in range(0, T, S):
            n = min(S, T - c0)
            for (prog, O), (v0, v1) in zip(self.progs, self.spans):
                with self.view(prog._in_surf, (self.H, S), np.float16) as d:
                    d[:] = 0
                    d[:, :n] = x16[c0:c0+n].T
                self.eng.submit(prog, procedure_index=0)
                with self.view(prog._out_surf, (O, S), np.float16) as o:
                    out[c0:c0+n, v0:v1] = o[:, :n].T.astype(np.float32)
        return out


def attach_ane_lm_head(model, engine_path, seq, bits, chunks):
    """Bake lm_head onto the ANE. Returns the head object, or None."""
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import runtime.q38_ane_engine as E
    from runtime.q38_ane_engine import AneEngine, _iosurface_view
    eng = AneEngine()
    lm = getattr(model, "language_model", model)
    head = getattr(lm, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        print("  no lm_head found")
        return None
    W = np.array(head.weight.astype(mx.float32))
    t0 = time.time()
    h = AneLmHead(eng, E, _iosurface_view, W, seq, bits, chunks)
    print(f"  baked lm_head {h.V}x{h.H} in {chunks} chunks, "
          f"{h.nbytes/1e9:.2f} GB int{bits}, {time.time()-t0:.0f}s", flush=True)

    orig = type(head).__call__

    # Patch THIS instance only. lm_head is an nn.Linear, and overriding
    # type(head).__call__ would also hijack every other Linear in the model --
    # including the GDN in_proj_* layers, which then get vocab-sized outputs.
    def patched(self, x, *args, **kwargs):
        try:
            shp = x.shape
            xf = np.array(x.astype(mx.float32)).reshape(-1, h.H)
            t0 = time.perf_counter()
            y = h(xf)
            HEAD_STATS["ms"] += (time.perf_counter() - t0) * 1e3
            HEAD_STATS["n"] += xf.shape[0]
            return mx.array(y.reshape(*shp[:-1], h.V)).astype(x.dtype)
        except Exception:
            traceback.print_exc()
            return orig(self, x, *args, **kwargs)

    head.__class__ = type("AneLmHeadLinear", (type(head),), {"__call__": patched})
    FREED[0] += _free_mlx((head, "weight"))
    _reclaim()
    return h



class AneFusedProj:
    """One ANE conv producing several linears that share an input.

    GDN calls in_proj_qkv, in_proj_z, in_proj_b and in_proj_a on the same
    tensor, so their weights concatenate along output rows into a single
    [16480, 5120] conv -- one dispatch instead of four. Row-wise concatenation
    is exact: each row keeps its own scale and no partial sums cross a span.
    """

    def __init__(self, eng, E, view, mats, S, bits):
        self.eng, self.view = eng, view
        self.H = mats[0][1].shape[1]
        self.spans, rows = [], []
        o = 0
        for name, W in mats:
            n = W.shape[0]
            self.spans.append((name, o, o + n))
            rows.append(np.asarray(W, np.float32))
            o += n
        self.O = o
        Wc = np.ascontiguousarray(np.concatenate(rows, 0))
        self.S = max(ANE_MIN_SEQ, S)
        blobs = {}
        if bits == 16:
            blobs["p.bin"] = Wc.astype(np.float16).tobytes()
        else:
            hi = (1 << (bits - 1)) - 1
            sc = np.abs(Wc).max(axis=1, keepdims=True) / hi
            q = np.clip(np.rint(Wc / np.where(sc == 0, 1, sc)), -hi-1, hi).astype(np.int8)
            if bits == 4:
                f = (q.reshape(-1).astype(np.uint8) & 0x0F)
                blobs["p.bin"] = (f[0::2] | (f[1::2] << 4)).tobytes()
            else:
                blobs["p.bin"] = q.tobytes()
            blobs["ps.bin"] = sc.astype(np.float16).tobytes()
        decl = AneDenseMLP._decl("p", self.O, self.H, bits)
        mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {self.H}, 1, {self.S}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
{decl}
    tensor<fp16, [1, {self.O}, 1, {self.S}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=pw, x=x)[name=string("proj")];
  }} -> (y);
}}
"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            self.prog = eng.compile_multiproc(mil, blobs, self.H, self.O, self.S)
        if self.prog is None:
            _tail = "\n".join(buf.getvalue().strip().splitlines()[-6:])
            raise RuntimeError(f"fused projection [{self.O},{self.H}] failed to "
                               f"compile\n{_tail}")
        eng._ensure_io(self.prog)
        self.nbytes = sum(len(b) for b in blobs.values())
        try:
            loc = E._desc(E._msg(self.prog.model, "localModelPath"))
            if loc and loc != "(nil)":
                _MY_TEMPDIRS.add(loc)
        except Exception:
            pass

    def __call__(self, xf):
        """xf [T, H] float32 -> [T, O] float32."""
        T, S = xf.shape[0], self.S
        out = np.zeros((T, self.O), np.float32)
        x16 = xf.astype(np.float16)
        for c0 in range(0, T, S):
            n = min(S, T - c0)
            with self.view(self.prog._in_surf, (self.H, S), np.float16) as d:
                d[:] = 0
                d[:, :n] = x16[c0:c0+n].T
            self.eng.submit(self.prog, procedure_index=0)
            with self.view(self.prog._out_surf, (self.O, S), np.float16) as o:
                out[c0:c0+n] = o[:, :n].T.astype(np.float32)
        return out


def attach_ane_gdn(model, engine_path, seq, bits, max_layers=0):
    """Fuse each GDN layer's four input projections into one ANE conv.

    mlx's linear_attn calls in_proj_qkv, then _z, then _b, then _a on the same
    tensor. The qkv module runs the fused conv and stashes every span; the other
    three read their slice back out, so four GPU matmuls become one ANE dispatch.
    """
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import runtime.q38_ane_engine as E
    from runtime.q38_ane_engine import AneEngine, _iosurface_view
    eng = AneEngine()
    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers", [])
    las = [b.linear_attn for b in layers
           if getattr(getattr(b, "linear_attn", None), "in_proj_qkv", None) is not None]
    if not las:
        print("  no GDN layers found")
        return 0
    if max_layers:
        las = las[:max_layers]
    NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    t0, total, done = time.time(), 0, 0
    for li, la in enumerate(las):
        mats = [(n, np.array(getattr(la, n).weight.astype(mx.float32))) for n in NAMES]
        try:
            fp = AneFusedProj(eng, E, _iosurface_view, mats, seq, bits)
        except Exception as exc:
            print(f"  GDN stopped at {done}/{len(las)} layers "
                  f"(program budget reached)")
            break
        total += fp.nbytes
        box = {"out": None}
        qkv_mod = getattr(la, NAMES[0])

        def make_first(fp=fp, box=box, span=fp.spans[0]):
            def call(self, x, *a, **k):
                shp = x.shape
                xf = np.array(x.astype(mx.float32)).reshape(-1, fp.H)
                t = time.perf_counter()
                y = fp(xf)
                GDN_STATS["ms"] += (time.perf_counter() - t) * 1e3
                GDN_STATS["n"] += 1
                box["out"] = (y, shp, x.dtype)
                _, a0, a1 = span
                return mx.array(np.ascontiguousarray(
                    y[:, a0:a1])).reshape(*shp[:-1], a1-a0).astype(x.dtype)
            return call

        def make_rest(box=box, span=None):
            def call(self, x, *a, **k):
                y, shp, dt = box["out"]
                _, a0, a1 = span
                return mx.array(np.ascontiguousarray(
                    y[:, a0:a1])).reshape(*shp[:-1], a1-a0).astype(dt)
            return call

        qkv_mod.__class__ = type("AneFusedQKV", (type(qkv_mod),),
                                 {"__call__": make_first()})
        for name, a0, a1 in fp.spans[1:]:
            m = getattr(la, name)
            m.__class__ = type("AneFusedSlice", (type(m),),
                               {"__call__": make_rest(span=(name, a0, a1))})
        FREED[0] += _free_mlx(*[(getattr(la, nm), "weight") for nm in NAMES])
        _reclaim()
        done += 1
        if done % 16 == 0 or done == len(las):
            print(f"    fused GDN {done}/{len(las)} ({total/1e9:.1f} GB, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    print(f"  fused {done} GDN layers, {total/1e9:.2f} GB of int{bits} blobs, "
          f"{time.time()-t0:.0f}s   [free {_free_gb():.1f} GB]", flush=True)
    return done



def attach_ane_attn(model, engine_path, seq, bits, max_layers=0):
    """Fuse q/k/v of each full_attention layer into one ANE conv.

    Same trick as the GDN projections: q_proj, k_proj and v_proj all read the
    normed hidden state, so their rows concatenate into a single conv and the
    later two modules read their slice back out of the cached result.
    """
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import runtime.q38_ane_engine as E
    from runtime.q38_ane_engine import AneEngine, _iosurface_view
    eng = AneEngine()
    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers", [])
    attns = [b.self_attn for b in layers
             if getattr(getattr(b, "self_attn", None), "q_proj", None) is not None]
    if not attns:
        print("  no full_attention layers found")
        return 0
    if max_layers:
        attns = attns[:max_layers]
    NAMES = ("q_proj", "k_proj", "v_proj")
    t0, total, done = time.time(), 0, 0
    for li, at in enumerate(attns):
        mats = [(n, np.array(getattr(at, n).weight.astype(mx.float32))) for n in NAMES]
        try:
            fp = AneFusedProj(eng, E, _iosurface_view, mats, seq, bits)
        except Exception as exc:
            print(f"  attention stopped at {done}/{len(attns)} layers "
                  f"(program budget reached)")
            break
        total += fp.nbytes
        box = {"out": None}

        def make_first(fp=fp, box=box, span=fp.spans[0]):
            def call(self, x, *a, **k):
                shp = x.shape
                xf = np.array(x.astype(mx.float32)).reshape(-1, fp.H)
                t = time.perf_counter()
                y = fp(xf)
                ATTN_STATS["ms"] += (time.perf_counter() - t) * 1e3
                ATTN_STATS["n"] += 1
                box["out"] = (y, shp, x.dtype)
                _, a0, a1 = span
                return mx.array(np.ascontiguousarray(
                    y[:, a0:a1])).reshape(*shp[:-1], a1-a0).astype(x.dtype)
            return call

        def make_rest(box=box, span=None):
            def call(self, x, *a, **k):
                y, shp, dt = box["out"]
                _, a0, a1 = span
                return mx.array(np.ascontiguousarray(
                    y[:, a0:a1])).reshape(*shp[:-1], a1-a0).astype(dt)
            return call

        qm = getattr(at, NAMES[0])
        qm.__class__ = type("AneFusedQ", (type(qm),), {"__call__": make_first()})
        for name, a0, a1 in fp.spans[1:]:
            m = getattr(at, name)
            m.__class__ = type("AneFusedKV", (type(m),),
                               {"__call__": make_rest(span=(name, a0, a1))})
        FREED[0] += _free_mlx(*[(getattr(at, nm), "weight") for nm in NAMES])
        _reclaim()
        done += 1
    print(f"  fused {done} attention layers, {total/1e9:.2f} GB of int{bits} blobs, "
          f"{time.time()-t0:.0f}s   [free {_free_gb():.1f} GB]", flush=True)
    return done



class AneFusedLayer:
    """A whole layer tail in one ANE program.

        out_proj -> +residual -> RMSNorm -> gate/up -> silu -> mul -> down -> +residual

    The two inputs (the attention/GDN core output, and the residual) are
    concatenated into one surface and sliced apart in-graph, which avoids
    multi-input request plumbing.

    RMSNorm is buildable on the ANE, but not the obvious way: `reduce_mean` over
    the channel axis and `rsqrt` are both rejected by the compiler. A 1x1 conv
    with constant 1/C weights performs the cross-channel mean, and `sqrt` +
    `real_div` replace rsqrt -- `[1,1,1,S]` broadcasts against `[1,C,1,S]`.
    Measured rel 1.5e-3 against numpy.
    """

    def __init__(self, eng, E, view, out_w, pn_w, g, u, d, S, bits, eps=1e-6,
                 nxt_norm_w=None, nxt_proj_w=None, cache=None, tag=None):
        """nxt_*: fold the NEXT layer's input_layernorm and input projection in.

        The program then emits concat([out, next_proj]) and the following layer
        needs no projection program of its own -- which is what gets the whole
        model under the 126-program ceiling with room to spare.
        """
        self.eng, self.view = eng, view
        self.H, self.Dc = out_w.shape
        self.I = g.shape[0]
        self.IN = self.Dc + self.H
        self.S = max(ANE_MIN_SEQ, S)
        self.P = 0 if nxt_proj_w is None else nxt_proj_w.shape[0]
        self.OUT = self.H
        H, I, Dc, S_ = self.H, self.I, self.Dc, self.S

        blobs, decl = {}, []
        for nm, W, shp in (("o", out_w, out_w.shape),
                           ("gu", [g, u], (2 * self.I, self.H)),
                           ("dn", d, d.shape)):
            O, In = shp
            blobs.update(_quant_blob(nm, W, bits, cache,
                                     f"{tag}.{nm}" if tag else None))
            decl.append(AneDenseMLP._decl(nm, O, In, bits))
        if nxt_proj_w is not None:
            O, In = nxt_proj_w.shape
            blobs.update(_quant_blob("ip", nxt_proj_w, bits, cache,
                                     f"{tag}.ip" if tag else None))
            decl.append(AneDenseMLP._decl("ip", O, In, bits))
            blobs["il.bin"] = np.asarray(nxt_norm_w, np.float32).astype(np.float16).tobytes()
            decl.append(f'    tensor<fp16, [1, {H}, 1, 1]> ilw = const()[name=string("ilw"), '
                        f'val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/il.bin"), offset=uint64(64)))];')
        blobs["pn.bin"] = np.asarray(pn_w, np.float32).astype(np.float16).tobytes()
        blobs["on.bin"] = np.full((1, H), 1.0 / H, np.float16).tobytes()
        decl.append(f'    tensor<fp16, [1, {H}, 1, 1]> pnw = const()[name=string("pnw"), '
                    f'val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/pn.bin"), offset=uint64(64)))];')
        decl.append(f'    tensor<fp16, [1, {H}, 1, 1]> onw = const()[name=string("onw"), '
                    f'val=tensor<fp16, [1, {H}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/on.bin"), offset=uint64(64)))];')

        mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {self.IN}, 1, {S_}]> xin) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
{chr(10).join(decl)}
    tensor<fp16, [1, {Dc}, 1, {S_}]> core = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{Dc},1,{S_}]), x=xin)[name=string("core")];
    tensor<fp16, [1, {H}, 1, {S_}]> res = slice_by_index(begin=tensor<int32, [4]>([0,{Dc},0,0]), end=tensor<int32, [4]>([1,{self.IN},1,{S_}]), x=xin)[name=string("res")];
    tensor<fp16, [1, {H}, 1, {S_}]> r = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=ow, x=core)[name=string("outp")];
    tensor<fp16, [1, {H}, 1, {S_}]> h = add(x=res, y=r)[name=string("h")];
    tensor<fp16, [1, {H}, 1, {S_}]> sq = mul(x=h, y=h)[name=string("sq")];
    tensor<fp16, [1, 1, 1, {S_}]> ms = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=onw, x=sq)[name=string("ms")];
    fp16 ep = const()[name=string("ep"), val=fp16(0x1.0p-20)];
    tensor<fp16, [1, 1, 1, {S_}]> msa = add(x=ms, y=ep)[name=string("msa")];
    tensor<fp16, [1, 1, 1, {S_}]> sd = sqrt(x=msa)[name=string("sd")];
    tensor<fp16, [1, {H}, 1, {S_}]> nx = real_div(x=h, y=sd)[name=string("nx")];
    tensor<fp16, [1, {H}, 1, {S_}]> n = mul(x=nx, y=pnw)[name=string("n")];
    tensor<fp16, [1, {2*I}, 1, {S_}]> c = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=guw, x=n)[name=string("gu")];
    tensor<fp16, [1, {I}, 1, {S_}]> g0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{I},1,{S_}]), x=c)[name=string("g0")];
    tensor<fp16, [1, {I}, 1, {S_}]> u0 = slice_by_index(begin=tensor<int32, [4]>([0,{I},0,0]), end=tensor<int32, [4]>([1,{2*I},1,{S_}]), x=c)[name=string("u0")];
    tensor<fp16, [1, {I}, 1, {S_}]> sg = sigmoid(x=g0)[name=string("sg")];
    tensor<fp16, [1, {I}, 1, {S_}]> si = mul(x=g0, y=sg)[name=string("si")];
    tensor<fp16, [1, {I}, 1, {S_}]> ac = mul(x=si, y=u0)[name=string("ac")];
    tensor<fp16, [1, {H}, 1, {S_}]> m = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=dnw, x=ac)[name=string("dn")];
    tensor<fp16, [1, {H}, 1, {S_}]> {'o0' if self.P else 'y'} = add(x=h, y=m)[name=string("{'o0' if self.P else 'y'}")];
{self._tail_mil()}
  }} -> ({"y, y2" if self.P else "y"});
}}
"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            self.prog = eng.compile_multiproc(mil, blobs, self.IN, self.OUT, self.S)
        if self.prog is None:
            _tail = "\n".join(buf.getvalue().strip().splitlines()[-6:])
            raise RuntimeError(f"fused layer [{self.IN}->{self.OUT}] failed to "
                               f"compile\n{_tail}")
        eng._ensure_io(self.prog)
        self.surf2 = None
        if self.P:
            self._build_two_output_request(E)
        self.nbytes = sum(len(b) for b in blobs.values())
        try:
            loc = E._desc(E._msg(self.prog.model, "localModelPath"))
            if loc and loc != "(nil)":
                _MY_TEMPDIRS.add(loc)
        except Exception:
            pass

    def _build_two_output_request(self, E):
        """Bind a second output surface and build the request ourselves.

        The engine's _ensure_request assumes one output; rather than change it,
        this constructs the _ANERequest directly with outputs=[y, y2] and
        outputIndices=[0, 1]. Verified against a dequantized reference at
        rel 1.3e-3 / 1.6e-3 (artifacts/ane_probes/ane_two_outputs.py).
        """
        import ctypes
        E._load_iosurface()
        self.surf2 = E._create_iosurface(E._iosurface_alloc_size(self.P * self.S))
        if not self.surf2:
            raise RuntimeError("second output IOSurface allocation failed")
        surf_cls = E._cls("_ANEIOSurfaceObject")
        IS = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)

        def wrap(sf):
            return IS(("objc_msgSend", E._objc))(
                E._msg(surf_cls, "alloc"),
                E._sel("initWithIOSurface:startOffset:shouldRetain:"),
                sf, E._nsnumber_int(0), True)

        # The compiler does not preserve the order the MIL declares outputs in:
        # for this graph it emits y2@output (P channels) as symbol 0 and
        # y@output (H channels) as symbol 1. Binding by position gives each
        # surface the wrong size and inference fails with status 0x1d, so match
        # them by the channel count the model reports.
        import re
        inner = E._msg(self.prog.model, "model") or self.prog.model
        desc = E._desc(E._msg(inner, "description"))
        # Each output entry reports Channels then Name = "<sym>@output"; the
        # order of those entries is the symbol order.
        # The negative lookahead keeps each match inside a single entry --
        # without it the match runs from the input's Channels across to the
        # first output Name and reports the input width.
        chans = [int(c) for c, _, _ in re.findall(
            r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";',
            desc, re.S)]
        if sorted(chans) != sorted([self.H, self.P]):
            raise RuntimeError(f"unexpected ANE output channels {chans}, "
                               f"expected {[self.H, self.P]}")
        surf_for = {self.H: self.prog._out_surf, self.P: self.surf2}
        ordered = [wrap(surf_for[c]) for c in chans]
        self._out_order = chans

        F = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
        self.req = F(("objc_msgSend", E._objc))(
            E._msg(E._cls("_ANERequest"), "alloc"),
            E._sel("initWithInputs:inputIndices:outputs:outputIndices:"
                   "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                   "transactionHandle:"),
            E._nsarray([wrap(self.prog._in_surf)]),
            E._nsarray([E._nsnumber_int(0)]),
            E._nsarray(ordered),
            E._nsarray([E._nsnumber_int(i) for i in range(len(ordered))]),
            None, None, E._nsnumber_int(0), None, None)
        if not self.req:
            raise RuntimeError("two-output request creation failed")
        self._Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        self._E = E

    def _submit2(self):
        import ctypes
        err = ctypes.c_void_p(0)
        ok = self._Eval(("objc_msgSend", self._E._objc))(
            self.prog.model, self._E._sel("evaluateWithQoS:options:request:error:"),
            21, self.prog._compile_opts, self.req, ctypes.byref(err))
        if not ok:
            raise RuntimeError(
                f"two-output evaluate failed: "
                f"{self._E._desc(err.value) if err.value else 'unknown'}")

    def _tail_mil(self):
        """Emit either a passthrough, or the next layer's norm + projection.

        `concat` and `stack` do not compile on the ANE, and the pad+add
        substitute caps out around 9216 output channels -- too narrow for GDN's
        16480-row projection. A MIL func can return TWO tensors though, and
        `_ANERequest` takes an array of outputs, so the two halves leave through
        separate surfaces with no width limit at all.
        """
        H, S_ = self.H, self.S
        # o0 feeds both the layer output and the next layer's norm. Emitting
        # it directly as an output while other ops also consume it produced NaN
        # in that output, so the output gets its own identity copy.
        # `identity` as a program's SOLE output returns NaN, so the
        # single-output build names the residual add `y` directly. With two
        # outputs the identity copy is required instead: emitting the add as an
        # output while the norm branch also consumes it likewise gives NaN.
        if not self.P:
            return ""
        return f"""    tensor<fp16, [1, {H}, 1, {S_}]> y = identity(x=o0)[name=string("y")];
    tensor<fp16, [1, {H}, 1, {S_}]> nsq = mul(x=o0, y=o0)[name=string("nsq")];
    tensor<fp16, [1, 1, 1, {S_}]> nms = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=onw, x=nsq)[name=string("nms")];
    tensor<fp16, [1, 1, 1, {S_}]> nmsa = add(x=nms, y=ep)[name=string("nmsa")];
    tensor<fp16, [1, 1, 1, {S_}]> nsd = sqrt(x=nmsa)[name=string("nsd")];
    tensor<fp16, [1, {H}, 1, {S_}]> nnx = real_div(x=o0, y=nsd)[name=string("nnx")];
    tensor<fp16, [1, {H}, 1, {S_}]> nn = mul(x=nnx, y=ilw)[name=string("nn")];
    tensor<fp16, [1, {self.P}, 1, {S_}]> y2 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=ipw, x=nn)[name=string("y2")];"""

    def __call__(self, core, res):
        """core [T, Dc], res [T, H] float32 -> [T, OUT] float32 (H, then proj)."""
        T, S = core.shape[0], self.S
        out = np.zeros((T, self.H), np.float32)
        out2 = np.zeros((T, self.P), np.float32) if self.P else None
        _a = time.perf_counter()
        cat = np.concatenate([core, res], 1).astype(np.float16)
        FL_PHASE["cat"] += time.perf_counter() - _a
        for c0 in range(0, T, S):
            n = min(S, T - c0)
            _a = time.perf_counter()
            with self.view(self.prog._in_surf, (self.IN, S), np.float16) as d:
                d[:] = 0
                d[:, :n] = cat[c0:c0+n].T
            FL_PHASE["write"] += time.perf_counter() - _a
            _a = time.perf_counter()
            if self.P:
                self._submit2()
            else:
                self.eng.submit(self.prog, procedure_index=0)
            FL_PHASE["submit"] += time.perf_counter() - _a
            _a = time.perf_counter()
            with self.view(self.prog._out_surf, (self.H, S), np.float16) as o:
                out[c0:c0+n] = o[:, :n].T.astype(np.float32)
            if self.P:
                with self.view(self.surf2, (self.P, S), np.float16) as o2:
                    out2[c0:c0+n] = o2[:, :n].T.astype(np.float32)
            FL_PHASE["read"] += time.perf_counter() - _a
            FL_PHASE["n"] += 1
        return out if out2 is None else (out, out2)



def attach_ane_fused_layers(model, n_layers, engine_path, seq, bits, cache=None):
    """Replace each layer's tail with one fused ANE program.

    Costs the same number of programs as baking MLPs alone (one per layer) but
    additionally puts out_proj and post_attention_layernorm on the ANE. out_proj
    is turned into an identity so linear_attn/self_attn hands back its
    pre-projection core, which the fused program consumes together with the
    residual.
    """
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import runtime.q38_ane_engine as E
    from runtime.q38_ane_engine import AneEngine, _iosurface_view
    eng = AneEngine()
    lm = getattr(model, "language_model", model)
    layers = list(getattr(getattr(lm, "model", lm), "layers", []))
    eps = getattr(getattr(model, "args", None), "rms_norm_eps", 1e-6)
    cand = [b for b in layers
            if getattr(getattr(b, "mlp", None), "gate_proj", None) is not None]
    if n_layers > 0:
        cand = cand[-n_layers:]
    t0, total, done = time.time(), 0, 0
    for b in cand:
        core_mod = b.linear_attn if getattr(b, "is_linear", False) else b.self_attn
        op = getattr(core_mod, "out_proj", None) or getattr(core_mod, "o_proj", None)
        if op is None:
            continue
        try:
            fl = AneFusedLayer(
                eng, E, _iosurface_view,
                np.array(op.weight.astype(mx.float32)),
                np.array(b.post_attention_layernorm.weight.astype(mx.float32)),
                np.array(b.mlp.gate_proj.weight.astype(mx.float32)),
                np.array(b.mlp.up_proj.weight.astype(mx.float32)),
                np.array(b.mlp.down_proj.weight.astype(mx.float32)),
                seq, bits, eps, None, None, cache, f"fused{done}")
        except Exception as exc:
            print(f"  fused layer {done}: {exc}")
            break
        total += fl.nbytes

        # out_proj becomes an identity: the fused program owns that matmul now.
        op.__class__ = type("AneIdentityProj", (type(op),),
                            {"__call__": lambda self, x, *a, **k: x})

        def make(fl=fl):
            def call(self, x, mask=None, cache=None):
                core = (self.linear_attn if self.is_linear else self.self_attn)(
                    self.input_layernorm(x), mask, cache)
                shp = x.shape
                t = time.perf_counter()
                _b = time.perf_counter()
                cf = np.array(core.astype(mx.float32)).reshape(-1, fl.Dc)
                xf = np.array(x.astype(mx.float32)).reshape(-1, fl.H)
                FL_PHASE["mx"] += time.perf_counter() - _b
                y = fl(cf, xf)
                LAYER_STATS["ms"] += (time.perf_counter() - t) * 1e3
                LAYER_STATS["n"] += 1
                return mx.array(y.reshape(shp)).astype(x.dtype)
            return call

        b.__class__ = type("AneFusedDecoderLayer", (type(b),), {"__call__": make()})
        FREED[0] += _free_mlx((op, "weight"), (b.post_attention_layernorm, "weight"),
                              (b.mlp.gate_proj, "weight"), (b.mlp.up_proj, "weight"),
                              (b.mlp.down_proj, "weight"))
        _reclaim()
        done += 1
        if done % 16 == 0 or done == len(cand):
            print(f"    fused layer {done}/{len(cand)} ({total/1e9:.1f} GB, "
                  f"{time.time()-t0:.0f}s)   [free {_free_gb():.1f} GB]", flush=True)
    print(f"  fused {done} layer tails, {total/1e9:.2f} GB of int{bits} blobs, "
          f"{time.time()-t0:.0f}s", flush=True)
    return done



def _input_projs(block):
    """The modules a layer applies to its normed input, in call order."""
    if getattr(block, "is_linear", False):
        core = block.linear_attn
        names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    else:
        core = block.self_attn
        names = ("q_proj", "k_proj", "v_proj")
    return core, [(n, getattr(core, n)) for n in names
                  if getattr(core, n, None) is not None]


def attach_ane_chain(model, engine_path, seq, bits, cache=None):
    """Fuse the tail of layer N and the head of layer N+1 into one program.

    Each program computes out_proj -> +residual -> RMSNorm -> MLP -> +residual
    for its own layer, then the NEXT layer's input_layernorm and input
    projection, emitting concat([out, next_proj]). The following layer therefore
    needs no projection program at all: 64 programs cover every projection in
    the model instead of 64 + 48 + 16.

    Layer 0 has no predecessor, so its input_layernorm and projections stay on
    the GPU.
    """
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import runtime.q38_ane_engine as E
    from runtime.q38_ane_engine import AneEngine, _iosurface_view
    eng = AneEngine()
    lm = getattr(model, "language_model", model)
    layers = list(getattr(getattr(lm, "model", lm), "layers", []))
    eps = getattr(getattr(model, "args", None), "rms_norm_eps", 1e-6)
    N = len(layers)
    boxes = [{"proj": None, "shape": None, "dtype": None} for _ in range(N)]
    t0, total, done, folded = time.time(), 0, 0, 0

    for i, b in enumerate(layers):
        if getattr(getattr(b, "mlp", None), "gate_proj", None) is None:
            continue
        core_mod = b.linear_attn if getattr(b, "is_linear", False) else b.self_attn
        op = getattr(core_mod, "out_proj", None) or getattr(core_mod, "o_proj", None)
        if op is None:
            continue
        nxt_norm = nxt_proj = None
        spans = []
        if i + 1 < N:
            nb = layers[i + 1]
            _, mods = _input_projs(nb)
            if mods:
                mats, o = [], 0
                for name, m in mods:
                    W = np.array(m.weight.astype(mx.float32))
                    spans.append((name, o, o + W.shape[0]))
                    mats.append(W)
                    o += W.shape[0]
                # No width cap: the projection leaves through its own output
                # surface, so GDN's 16480 rows and attention's 14336 both fit.
                nxt_proj = np.concatenate(mats, 0)
                nxt_norm = np.array(nb.input_layernorm.weight.astype(mx.float32))
        try:
            fl = AneFusedLayer(
                eng, E, _iosurface_view,
                np.array(op.weight.astype(mx.float32)),
                np.array(b.post_attention_layernorm.weight.astype(mx.float32)),
                np.array(b.mlp.gate_proj.weight.astype(mx.float32)),
                np.array(b.mlp.up_proj.weight.astype(mx.float32)),
                np.array(b.mlp.down_proj.weight.astype(mx.float32)),
                seq, bits, eps, nxt_norm, nxt_proj, cache, f"chain{i}")
        except Exception as exc:
            if nxt_proj is not None:
                nxt_proj = nxt_norm = None
                spans = []
                try:
                    fl = AneFusedLayer(
                        eng, E, _iosurface_view,
                        np.array(op.weight.astype(mx.float32)),
                        np.array(b.post_attention_layernorm.weight.astype(mx.float32)),
                        np.array(b.mlp.gate_proj.weight.astype(mx.float32)),
                        np.array(b.mlp.up_proj.weight.astype(mx.float32)),
                        np.array(b.mlp.down_proj.weight.astype(mx.float32)),
                        seq, bits, eps)
                except Exception as exc2:
                    print(f"  chain layer {i}: {exc2}")
                    break
            else:
                print(f"  chain layer {i}: {exc}")
                break
        total += fl.nbytes
        op.__class__ = type("AneIdentityProj", (type(op),),
                            {"__call__": lambda self, x, *a, **k: x})

        def make(fl=fl, nbox=(boxes[i + 1] if i + 1 < N else None)):
            def call(self, x, mask=None, cache=None):
                core = (self.linear_attn if self.is_linear else self.self_attn)(
                    self.input_layernorm(x), mask, cache)
                shp = x.shape
                t = time.perf_counter()
                r = fl(np.array(core.astype(mx.float32)).reshape(-1, fl.Dc),
                       np.array(x.astype(mx.float32)).reshape(-1, fl.H))
                LAYER_STATS["ms"] += (time.perf_counter() - t) * 1e3
                LAYER_STATS["n"] += 1
                y, pj = r if isinstance(r, tuple) else (r, None)
                if nbox is not None and pj is not None:
                    nbox["proj"] = pj
                    nbox["shape"] = shp
                n_toks = int(np.prod(shp[:-1]))
                return mx.array(y[:n_toks].reshape(shp)).astype(x.dtype)
            return call

        b.__class__ = type("AneChainLayer", (type(b),), {"__call__": make()})
        FREED[0] += _free_mlx((op, "weight"), (b.post_attention_layernorm, "weight"),
                              (b.mlp.gate_proj, "weight"), (b.mlp.up_proj, "weight"),
                              (b.mlp.down_proj, "weight"))
        _reclaim()

        # Layer i+1 now reads its projections out of the box instead of
        # computing them; its input_layernorm was folded in above.
        if spans:
            folded += 1
            nb = layers[i + 1]
            nb.input_layernorm.__class__ = type(
                "AneIdentityNorm", (type(nb.input_layernorm),),
                {"__call__": lambda self, x, *a, **k: x})
            for name, a0, a1 in spans:
                m = getattr(nb.linear_attn if getattr(nb, "is_linear", False)
                            else nb.self_attn, name)

                def rd(box=boxes[i + 1], a0=a0, a1=a1):
                    def call(self, x, *a, **k):
                        y, shp, dt = box["proj"], box["shape"], box.get("dtype") or x.dtype
                        return mx.array(np.ascontiguousarray(
                            y[:, a0:a1])).reshape(*shp[:-1], a1 - a0).astype(dt)
                    return call

                m.__class__ = type("AneChainProj", (type(m),), {"__call__": rd()})
            # The folded projection and input norm are now read from the
            # previous tail's output surface.  Drop their MLX buffers too;
            # retaining them defeats the point of the chained ANE path and
            # can keep several gigabytes of unified memory resident.
            FREED[0] += _free_mlx(
                (nb.input_layernorm, "weight"),
                *[(getattr(nb.linear_attn if getattr(nb, "is_linear", False)
                           else nb.self_attn, name), "weight")
                  for name, _, _ in spans],
            )
            _reclaim()
        done += 1
        if done % 16 == 0 or done == N:
            print(f"    chained {done}/{N} ({total/1e9:.1f} GB, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    # Layer 0 has no predecessor to fold its head into, so it gets one program
    # of its own -- that closes the last projection still running on the GPU.
    extra = 0
    if layers:
        nb = layers[0]
        core0, mods = _input_projs(nb)
        if mods and not isinstance(getattr(mods[0][1], "__class__", None), type(None)) \
                and "AneChainProj" not in type(mods[0][1]).__name__:
            try:
                mats = [(nm, np.array(mm.weight.astype(mx.float32)))
                        for nm, mm in mods]
                fp = AneFusedProj(eng, E, _iosurface_view, mats, seq, bits)
                box0 = {"out": None}

                def first(fp=fp, box=box0, span=fp.spans[0]):
                    def call(self, x, *a, **k):
                        shp = x.shape
                        y = fp(np.array(x.astype(mx.float32)).reshape(-1, fp.H))
                        box["out"] = (y, shp, x.dtype)
                        _, a0, a1 = span
                        return mx.array(np.ascontiguousarray(
                            y[:, a0:a1])).reshape(*shp[:-1], a1-a0).astype(x.dtype)
                    return call

                def rest(box=box0, span=None):
                    def call(self, x, *a, **k):
                        y, shp, dt = box["out"]
                        _, a0, a1 = span
                        return mx.array(np.ascontiguousarray(
                            y[:, a0:a1])).reshape(*shp[:-1], a1-a0).astype(dt)
                    return call

                m0 = mods[0][1]
                m0.__class__ = type("AneL0Q", (type(m0),), {"__call__": first()})
                for nm, a0, a1 in fp.spans[1:]:
                    mm = getattr(core0, nm)
                    mm.__class__ = type("AneL0Slice", (type(mm),),
                                        {"__call__": rest(span=(nm, a0, a1))})
                total += fp.nbytes
                extra = 1
                FREED[0] += _free_mlx(*[(mm, "weight") for _, mm in mods])
                _reclaim()
            except Exception as exc:
                print(f"  layer 0 projection: {exc}")
    print(f"  chained {done} layers ({folded} with the next layer's head folded "
          f"in{', +1 for layer 0' if extra else ''}), {total/1e9:.2f} GB of "
          f"int{bits} blobs, {time.time()-t0:.0f}s", flush=True)
    return done + extra



def _free_mlx(*mods_and_names):
    """Drop MLX-side weights whose work now lives on the ANE.

    The full bf16 model stays resident next to the ANE blobs, and nothing reads
    these arrays once their block is baked. Replacing each with a 1-element
    array releases the buffer. This makes the GPU fallback path unusable for
    that module, which is the intended trade.
    """
    if not FREE_MLX:
        return 0
    freed = 0
    for mod, name in mods_and_names:
        try:
            w = getattr(mod, name, None)
            if w is None or w.size <= 1:
                continue
            freed += w.size * w.dtype.size
            setattr(mod, name, mx.zeros((1,), dtype=w.dtype))
        except Exception:
            pass
    return freed


def _rss_gb():
    """Resident set of this process, GB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    except Exception:
        return 0.0


def _free_gb():
    """Free physical memory, GB. The ANE wires its model pages, so its pool
    appears to be bounded by this -- a build that fits on an idle machine can
    fail to load when something else is resident."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "Pages free" in line:
                return int(line.split(":")[1].strip().rstrip(".")) * 16384 / 1e9
    except Exception:
        pass
    return 0.0


def _reclaim():
    """Force MLX to actually hand the freed buffers back."""
    if not FREE_MLX:
        return
    try:
        mx.eval(mx.zeros((1,)))
        mx.clear_cache()
    except Exception:
        pass
    gc.collect()



class AneGdnStep:
    """The gated-delta recurrence for ONE decode step, on the ANE.

    Weight-free apart from two constant ones-kernels, so a single compiled
    program serves every GDN layer in the model -- state arrives as data.

    Layout is what makes it expressible: state[h, dv, dk] sits at channel
    h*Dk+dk, width dv. Then the sum over Dk is a grouped conv (groups=H, Dk->1),
    broadcasting delta back over Dk is the same conv reversed (1->Dk), and
    k/q/decay/beta are width-1 columns broadcast across Dv. Decay and beta are
    formed inside the same ANE graph from raw a/b using the fp16-safe polynomial
    softplus and explicit exp/divide sigmoid. Verified against the
    numpy reference at rel 9e-4 for the recurrence outputs
    (artifacts/ane_probes/ane_gdn_step.py).

    The input width is padded to a multiple of 32 elements: row stride is
    width*2 bytes and the ANE needs 64-byte rows -- an unpadded 131 compiles and
    then fails evaluate with status 0x1d.
    """

    def __init__(self, eng, E, view, H, Dk, Dv):
        import ctypes
        self.eng, self.view, self.E = eng, view, E
        self.H, self.Dk, self.Dv = H, Dk, Dv
        self.HK = H * Dk
        self.CIN = self.HK + 2 * H
        self.W = ((Dv + 5 + 31) // 32) * 32
        HK, W, CIN = self.HK, self.W, self.CIN
        blobs = {"gsum.bin": np.ones((H, Dk, 1, 1), np.float16).tobytes(),
                 "grep.bin": np.ones((HK, 1, 1, 1), np.float16).tobytes()}

        def sl(nm, c0, c1, w0, w1):
            return (f'    tensor<fp16, [1, {c1-c0}, 1, {w1-w0}]> {nm} = '
                    f'slice_by_index(begin=tensor<int32, [4]>([0,{c0},0,{w0}]), '
                    f'end=tensor<int32, [4]>([1,{c1},1,{w1}]), x=x)'
                    f'[name=string("{nm}")];')

        mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {CIN}, 1, {W}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gh = const()[name=string("gh"), val=int32({H})];
    tensor<fp16, [{H}, {Dk}, 1, 1]> gsum = const()[name=string("gsum"), val=tensor<fp16, [{H}, {Dk}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/gsum.bin"), offset=uint64(64)))];
    tensor<fp16, [{HK}, 1, 1, 1]> grep = const()[name=string("grep"), val=tensor<fp16, [{HK}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/grep.bin"), offset=uint64(64)))];
{sl("stt", 0, HK, 0, Dv)}
{sl("kk", 0, HK, Dv, Dv+1)}
{sl("qq", 0, HK, Dv+1, Dv+2)}
{sl("aa", 0, HK, Dv+2, Dv+3)}
{sl("dtc", 0, HK, Dv+3, Dv+4)}
{sl("avec", 0, HK, Dv+4, Dv+5)}
{sl("vv", HK, HK+H, 0, Dv)}
{sl("bb", HK+H, HK+2*H, 0, 1)}
    tensor<fp16, [1, {HK}, 1, 1]> ap = add(x=aa, y=dtc)[name=string("ap")];
    tensor<fp16, [1, {HK}, 1, 1]> pos = relu(x=ap)[name=string("pos")];
    tensor<fp16, [1, {HK}, 1, 1]> ab = abs(x=ap)[name=string("ab")];
    tensor<fp16, [1, {HK}, 1, 1]> nab = mul(x=ab, y=fp16(-0x1p+0))[name=string("nab")];
    tensor<fp16, [1, {HK}, 1, 1]> tt = exp(x=nab)[name=string("tt")];
    tensor<fp16, [1, {HK}, 1, 1]> hp5 = mul(x=tt, y=fp16(-0x1.8400000000000p-6))[name=string("hp5")];
    tensor<fp16, [1, {HK}, 1, 1]> ha4 = add(x=hp5, y=fp16(0x1.9ac0000000000p-4))[name=string("ha4")];
    tensor<fp16, [1, {HK}, 1, 1]> hm4 = mul(x=ha4, y=tt)[name=string("hm4")];
    tensor<fp16, [1, {HK}, 1, 1]> ha3 = add(x=hm4, y=fp16(-0x1.ab40000000000p-3))[name=string("ha3")];
    tensor<fp16, [1, {HK}, 1, 1]> hm3 = mul(x=ha3, y=tt)[name=string("hm3")];
    tensor<fp16, [1, {HK}, 1, 1]> ha2 = add(x=hm3, y=fp16(0x1.4c40000000000p-2))[name=string("ha2")];
    tensor<fp16, [1, {HK}, 1, 1]> hm2 = mul(x=ha2, y=tt)[name=string("hm2")];
    tensor<fp16, [1, {HK}, 1, 1]> ha1 = add(x=hm2, y=fp16(-0x1.ff40000000000p-2))[name=string("ha1")];
    tensor<fp16, [1, {HK}, 1, 1]> hm1 = mul(x=ha1, y=tt)[name=string("hm1")];
    tensor<fp16, [1, {HK}, 1, 1]> ha0 = add(x=hm1, y=fp16(0x1p+0))[name=string("ha0")];
    tensor<fp16, [1, {HK}, 1, 1]> tail = mul(x=tt, y=ha0)[name=string("tail")];
    tensor<fp16, [1, {HK}, 1, 1]> soft = add(x=pos, y=tail)[name=string("soft")];
    tensor<fp16, [1, {HK}, 1, 1]> asp = mul(x=avec, y=soft)[name=string("asp")];
    tensor<fp16, [1, {HK}, 1, 1]> nasp = mul(x=asp, y=fp16(-0x1p+0))[name=string("nasp")];
    tensor<fp16, [1, {HK}, 1, 1]> dcy = exp(x=nasp)[name=string("dcy")];
    tensor<fp16, [1, {H}, 1, 1]> nb = mul(x=bb, y=fp16(-0x1p+0))[name=string("nb")];
    tensor<fp16, [1, {H}, 1, 1]> enb = exp(x=nb)[name=string("enb")];
    tensor<fp16, [1, {H}, 1, 1]> bden = add(x=enb, y=fp16(0x1p+0))[name=string("bden")];
    tensor<fp16, [1, {H}, 1, 1]> bta = real_div(x=fp16(0x1p+0), y=bden)[name=string("bta")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> s1 = mul(x=stt, y=dcy)[name=string("s1")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> sk = mul(x=s1, y=kk)[name=string("sk")];
    tensor<fp16, [1, {H}, 1, {Dv}]> kvm = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sk)[name=string("kvm")];
    tensor<fp16, [1, {H}, 1, {Dv}]> dlt = sub(x=vv, y=kvm)[name=string("dlt")];
    tensor<fp16, [1, {H}, 1, {Dv}]> dbt = mul(x=dlt, y=bta)[name=string("dbt")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> dup = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=dbt)[name=string("dup")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> upd = mul(x=dup, y=kk)[name=string("upd")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> s2 = add(x=s1, y=upd)[name=string("s2")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> sq = mul(x=s2, y=qq)[name=string("sq")];
    tensor<fp16, [1, {H}, 1, {Dv}]> y = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sq)[name=string("y")];
  }} -> (y, s2);
}}
"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            self.prog = eng.compile_multiproc(mil, blobs, CIN, H, W)
        if self.prog is None:
            raise RuntimeError("GDN step program failed to compile\n" +
                               "\n".join(buf.getvalue().strip().splitlines()[-5:]))
        E._load_iosurface()
        self.sin = E._create_iosurface(E._iosurface_alloc_size(CIN * W))
        self.sy = E._create_iosurface(E._iosurface_alloc_size(H * Dv))
        import re
        inner = E._msg(self.prog.model, "model") or self.prog.model
        desc = E._desc(E._msg(inner, "description"))
        chans = [int(c) for c, _, _ in re.findall(
            r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";',
            desc, re.S)]
        self._output_channels = chans
        self._Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        # Each live MLX cache state gets one compact IOSurface.  The compiled
        # graph is shared; only the request's state output binding differs.
        # A fixed-size LRU bounds stale prompt/request state to ~200 MB.
        from collections import OrderedDict
        self._slots = OrderedDict()
        self._max_slots = 64

    def _request_for_state(self, state_surface):
        """Bind new_state directly to a cache-owned compact IOSurface."""
        import ctypes
        E, H, HK = self.E, self.H, self.HK
        surf_for = {H: self.sy, HK: state_surface}
        request_init = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
        request = request_init(("objc_msgSend", E._objc))(
            E._msg(E._cls("_ANERequest"), "alloc"),
            E._sel("initWithInputs:inputIndices:outputs:outputIndices:"
                   "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                   "transactionHandle:"),
            E._nsarray([E._wrap_iosurface(self.sin)]),
            E._nsarray([E._nsnumber_int(0)]),
            E._nsarray([E._wrap_iosurface(surf_for[c])
                        for c in self._output_channels]),
            E._nsarray([E._nsnumber_int(i)
                        for i in range(len(self._output_channels))]),
            None, None, E._nsnumber_int(0), None, None)
        if not request:
            raise RuntimeError("GDN resident-state request creation failed")
        return request

    def _slot(self, marker, initial_state):
        """Return the ANE-resident slot represented by an MLX cache marker."""
        key = id(marker)
        slot = self._slots.pop(key, None)
        if slot is not None:
            self._slots[key] = slot
            return slot[1], slot[2]
        if initial_state is None:
            raise KeyError("GDN cache marker was evicted without an initial state")
        surface = self.E._create_iosurface(
            self.E._iosurface_alloc_size(self.HK * self.Dv))
        if not surface:
            raise RuntimeError("GDN resident-state IOSurface allocation failed")
        with self.view(surface, (self.HK, self.Dv), np.float16) as dst:
            dst[:] = initial_state.transpose(0, 2, 1).reshape(self.HK, self.Dv)
        # Retain marker itself so Python cannot reuse its id for a different
        # cache while this slot remains in the LRU.
        slot = (marker, surface, self._request_for_state(surface))
        self._slots[key] = slot
        while len(self._slots) > self._max_slots:
            self._slots.popitem(last=False)
        return slot[1], slot[2]

    def has_slot(self, marker):
        return id(marker) in self._slots

    def materialize(self, marker, release=True):
        """Recover resident state for an unsupported/batched fallback path."""
        key = id(marker)
        slot = self._slots.pop(key, None) if release else self._slots.get(key)
        if slot is None:
            raise KeyError("GDN cache marker has no resident state")
        _, surface, _ = slot
        with self.view(surface, (self.HK, self.Dv), np.float16) as src:
            return (np.array(src, np.float32)
                    .reshape(self.H, self.Dk, self.Dv)
                    .transpose(0, 2, 1))

    def resident(self, marker, initial_state, q, k, v, a, beta_logits,
                 A, dt_bias):
        """Run one step while keeping recurrent state in an IOSurface.

        ``initial_state`` ([H,Dv,Dk]) is needed only on the first decode call
        after MLX prefill. Later calls use ``marker`` to find the evolved ANE
        state and never materialise it on the host or GPU.
        """
        import ctypes
        H, Dk, Dv, HK = self.H, self.Dk, self.Dv, self.HK
        state_surface, request = self._slot(marker, initial_state)
        # The compiler accepts the recurrence with a 160-wide input but emits
        # a compact 128-wide state. NumPy performs this IOSurface-to-IOSurface
        # row copy in ~0.036 ms on M5 Max; it does not involve MLX or Metal.
        with self.view(state_surface, (HK, Dv), np.float16) as state_src, \
                self.view(self.sin, (self.CIN, self.W), np.float16) as b:
            b[:HK, :Dv] = state_src
            b[:HK, Dv] = k.reshape(-1)
            b[:HK, Dv+1] = q.reshape(-1)
            b[:HK, Dv+2] = np.repeat(a, Dk)
            b[:HK, Dv+3] = np.repeat(dt_bias, Dk)
            b[:HK, Dv+4] = np.repeat(A, Dk)
            b[HK:HK+H, :Dv] = v
            b[HK+H:HK+2*H, 0] = beta_logits
        err = ctypes.c_void_p(0)
        ok = self._Eval(("objc_msgSend", self.E._objc))(
            self.prog.model, self.E._sel("evaluateWithQoS:options:request:error:"),
            21, self.prog._compile_opts, request, ctypes.byref(err))
        if not ok:
            raise RuntimeError("GDN step evaluate failed: " + (
                self.E._desc(err.value) if err.value else "unknown"))
        with self.view(self.sy, (H, Dv), np.float16) as o:
            y = np.array(o, np.float32)
        return y



def attach_ane_gdn_step(model, engine_path, seq, bits):
    """Route the gated-delta recurrence through the ANE at decode (T=1).

    One program serves every GDN layer -- the recurrence carries no per-layer
    weights. Prefill (T>1) and the first step (state is None) fall back to
    mlx's own path, which is where the batched kernel is already efficient.
    """
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import runtime.q38_ane_engine as E
    from runtime.q38_ane_engine import AneEngine, _iosurface_view
    import mlx.nn as nn
    try:
        from mlx_lm.models import gated_delta as gd
        from mlx_lm.models import qwen3_5 as q35
    except Exception as exc:
        print(f"  gated_delta not importable: {exc}")
        return 0

    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers", [])
    la = next((b.linear_attn for b in layers
               if getattr(b, "linear_attn", None) is not None), None)
    if la is None:
        print("  no GDN layers found")
        return 0
    H = la.num_v_heads
    Dk, Dv = la.head_k_dim, la.head_v_dim
    eng = AneEngine()
    t0 = time.time()
    step = AneGdnStep(eng, E, _iosurface_view, H, Dk, Dv)
    print(f"  GDN recurrence on ANE: 1 program for all layers "
          f"(H={H} Dk={Dk} Dv={Dv}), {time.time()-t0:.0f}s", flush=True)
    try:
        loc = E._desc(E._msg(step.prog.model, "localModelPath"))
        if loc and loc != "(nil)":
            _MY_TEMPDIRS.add(loc)
    except Exception:
        pass

    orig = gd.gated_delta_update
    gate_constants = {}

    def patched(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
                use_kernel=True):
        if state is None or q.shape[0] != 1 or q.shape[1] != 1 or mask is not None:
            # If an active single-stream cache switches to an unsupported
            # shape/masked path, restore its latest ANE-owned state before
            # handing control back to MLX. This keeps fallback correctness.
            if state is not None and step.has_slot(state):
                dtype = state.dtype
                resident_state = step.materialize(state)
                state = mx.array(resident_state[None]).astype(dtype)
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
        try:
            rep = v.shape[-2] // q.shape[-2]
            qq = mx.repeat(q, rep, -2) if rep > 1 else q
            kk = mx.repeat(k, rep, -2) if rep > 1 else k
            const_key = (id(A_log), id(dt_bias))
            constants = gate_constants.get(const_key)
            if constants is None:
                constants = (
                    np.exp(np.array(A_log.astype(mx.float32))),
                    np.array(dt_bias.astype(mx.float32)),
                )
                gate_constants[const_key] = constants
            A, dt = constants
            t = time.perf_counter()
            # Import the prefill state once. Thereafter ``state`` is only the
            # stable cache marker for the evolved IOSurface-owned state.
            initial = (None if step.has_slot(state) else
                       np.array(state[0].astype(mx.float32)))
            q_host = np.array(qq[0, 0].astype(mx.float32))
            k_host = np.array(kk[0, 0].astype(mx.float32))
            v_host = np.array(v[0, 0].astype(mx.float32))
            a_host = np.array(a[0, 0].astype(mx.float32))
            b_host = np.array(b[0, 0].astype(mx.float32))
            resident_start = time.perf_counter()
            y = step.resident(state, initial, q_host, k_host, v_host,
                              a_host, b_host, A, dt)
            done = time.perf_counter()
            GDNSTEP_STATS["ms"] += (time.perf_counter() - t) * 1e3
            GDNSTEP_STATS["marshal_ms"] += (resident_start - t) * 1e3
            GDNSTEP_STATS["resident_ms"] += (done - resident_start) * 1e3
            GDNSTEP_STATS["n"] += 1
            return (mx.array(y).reshape(1, 1, *y.shape).astype(v.dtype),
                    state)
        except Exception:
            traceback.print_exc()
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)

    gd.gated_delta_update = patched
    q35.gated_delta_update = patched      # qwen3_5 imported the name directly
    return 1


def attach_ane_dense(model, n_layers, engine_path, seq, bits, cache_root=None,
                     wide_prefill=False, split_frac=0.0):
    """Bake the last n_layers' dense MLPs onto the ANE. Returns count."""
    if n_layers == 0:
        return 0
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import runtime.q38_ane_engine as E
    from runtime.q38_ane_engine import AneEngine, AneDynamicLinear, _iosurface_view
    eng = AneEngine()
    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers", [])
    mlps = [b.mlp for b in layers
            if getattr(getattr(b, "mlp", None), "gate_proj", None) is not None]
    if not mlps:
        print("  no dense MLPs found (MoE model?)")
        return 0
    if n_layers > 0:
        mlps = mlps[-n_layers:]
    gp = mlps[0].gate_proj
    hid = (gp.weight.shape[1] * (32 // getattr(gp, "bits", 16))
           if hasattr(gp, "scales") else gp.weight.shape[1])
    _ = AneDynamicLinear.compile(hid, hid, max(ANE_MIN_SEQ, seq))

    def dense_w(mod):
        """Weights as dense fp16, dequantizing a packed (AWQ/affine) layer.

        Re-quantizing a 4-bit source to int4 would stack a second quantization;
        int8 costs the same time on the ANE (1.933 vs 1.847 ms/layer) so it is
        the better target for an already-quantized checkpoint.
        """
        w = mod.weight
        if hasattr(mod, "scales"):
            bits = getattr(mod, "bits", 4)
            # infer group_size from the tensors: config.json can be stale (the
            # 27B AWQ build says 128 while its scales say 64)
            in_features = w.shape[1] * (32 // bits)
            gs = in_features // mod.scales.shape[1]
            w = mx.dequantize(w, mod.scales, mod.biases, group_size=gs,
                              bits=bits, mode=getattr(mod, "mode", "affine"))
        return np.array(w.astype(mx.float16))

    baked, total, failed = {}, 0, None
    t0 = time.perf_counter()
    for i, mlp in enumerate(mlps):
        g = dense_w(mlp.gate_proj)
        u = dense_w(mlp.up_proj)
        d = dense_w(mlp.down_proj)
        lc = None
        if cache_root:
            lc = os.path.join(cache_root, f"L{i:03d}")
            os.makedirs(lc, exist_ok=True)
        try:
            b = (AneSplitMLP(eng, E, _iosurface_view, g, u, d, seq, bits,
                             split_frac, lc) if split_frac > 0 else
                 AneDenseMLP(eng, E, _iosurface_view, g, u, d, seq, bits, lc,
                             wide_prefill))
            if split_frac <= 0:
                FREED[0] += _free_mlx((mlp.gate_proj, "weight"),
                                      (mlp.up_proj, "weight"),
                                      (mlp.down_proj, "weight"))
                _reclaim()
        except Exception as exc:
            # Baking is memory-pressure dependent and NOT deterministic: the
            # same model baked 62/64 once and 64/64 on a rerun. A partial bake
            # still runs, with the remaining layers quietly on the GPU, which
            # silently invalidates any A/B -- so say so loudly.
            failed = f"layer {i} of {len(mlps)}: {type(exc).__name__}: {exc}"
            break
        baked[id(mlp)] = b
        total += b.nbytes
        if (i + 1) % 8 == 0:
            print(f"    baked {i+1}/{len(mlps)} ({total/1e9:.1f} GB, "
                  f"{time.perf_counter()-t0:.0f}s)", flush=True)
    if not baked:
        print(f"  ANE baking FAILED with none baked: {failed}")
        return 0
    print(f"  baked {len(baked)} MLPs, {total/1e9:.2f} GB of int{bits} blobs, "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    if failed:
        print(f"\n  *** PARTIAL BAKE: {len(baked)} of {len(mlps)} layers on the ANE ***")
        print(f"  *** stopped at {failed}")
        print(f"  *** the other {len(mlps)-len(baked)} run on the GPU, so this is")
        print(f"  *** NOT a clean A/B. Free memory and rerun.\n", flush=True)

    cls = type(mlps[0])
    orig = cls.__call__

    def patched(self, x, *args, **kwargs):
        b = baked.get(id(self))
        if b is None:
            return orig(self, x, *args, **kwargs)
        if SYNC_ONLY:
            # Diagnostic: pay the host round-trip, but compute on the GPU. Any
            # slowdown against the plain baseline is pure pipeline-stall cost.
            t0 = time.perf_counter()
            _ = np.array(x.astype(mx.float32)).reshape(-1, b.H)
            STATS["ms"] += (time.perf_counter()-t0)*1e3
            STATS["pos"] += _.shape[0]
            return orig(self, x, *args, **kwargs)
        try:
            shp = x.shape
            xf = np.array(x.astype(mx.float32)).reshape(-1, b.H)
            t0 = time.perf_counter()
            if isinstance(b, AneSplitMLP):
                out = b(xf, x.reshape(-1, b.H))
                STATS["ms"] += (time.perf_counter()-t0)*1e3
                STATS["pos"] += xf.shape[0]
                return out.reshape(shp).astype(x.dtype)
            y = b(xf)
            _dt = (time.perf_counter()-t0)*1e3
            STATS["ms"] += _dt
            STATS["pos"] += xf.shape[0]
            if xf.shape[0] == 1:
                STATS["dec_ms"] = STATS.get("dec_ms", 0.0) + _dt
                STATS["dec_n"] = STATS.get("dec_n", 0) + 1
            else:
                STATS["pre_ms"] = STATS.get("pre_ms", 0.0) + _dt
                STATS["pre_tok"] = STATS.get("pre_tok", 0) + xf.shape[0]
            return mx.array(y.reshape(shp).astype(np.float32)).astype(x.dtype)
        except Exception:
            traceback.print_exc()
            return orig(self, x, *args, **kwargs)

    cls.__call__ = patched
    return len(baked)


# ------------------------------------------------------- OpenAI compatibility
TC_RE = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.S)


def split_tool_calls(text):
    calls = []
    for m in TC_RE.finditer(text):
        args = dict(PARAM_RE.findall(m.group(2)))
        calls.append({"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args)}})
    return TC_RE.sub("", text).strip(), calls


def norm_msgs(msgs):
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for b in c:
                if isinstance(b, str):
                    parts.append(b)
                elif isinstance(b, dict) and b.get("type") not in ("image", "image_url"):
                    parts.append(b.get("text") or b.get("content") or "")
            c = "\n".join(p for p in parts if p)
        c = "" if c is None else c if isinstance(c, str) else str(c)
        nm = {"role": m.get("role", "user"), "content": c}
        for k in ("tool_calls", "tool_call_id", "name"):
            if m.get(k) is not None:
                nm[k] = m[k]
        # Qwen's template iterates function.arguments as a mapping; OpenAI sends
        # a JSON string -> "Can only get item pairs from a mapping".
        for tc in nm.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                except Exception:
                    fn["arguments"] = {}
        out.append(nm)
    return out


def build(model, tok, args):
    from mlx_lm.generate import generate_step
    glock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._json({"object": "list",
                            "data": [{"id": args.name, "object": "model"}]})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            # MLX streams are per-thread. ThreadingHTTPServer hands each request
            # to a fresh worker, where the first op raises "There is no
            # Stream(cpu, 0) in current thread" -- so bind the device's streams
            # on this thread before touching any array.
            try:
                mx.set_default_device(mx.default_device())
                mx.set_default_stream(mx.new_stream(mx.default_device()))
                mx.eval(mx.zeros((1,)))
            except Exception:
                pass
            try:
                self._post()
            except Exception:
                traceback.print_exc()
                try: self._json({"error": "internal"}, 500)
                except Exception: pass

        def _post(self):
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            msgs = norm_msgs(req.get("messages", []))
            kw = {"add_generation_prompt": True, "tokenize": False}
            if req.get("tools"):
                kw["tools"] = [t.get("function", t) for t in req["tools"]]
            try:
                text = tok.apply_chat_template(msgs, **kw)
            except Exception:
                traceback.print_exc()
                text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"
            ids = mx.array(tok.encode(text))
            maxtok = int(req.get("max_tokens") or args.max_tokens)
            cid = "chatcmpl-" + uuid.uuid4().hex[:12]

            with glock:
                before = dict(STATS)
                toks, t0, ttft = [], time.perf_counter(), None
                for i, (t, _) in enumerate(generate_step(ids, model, max_tokens=maxtok)):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tid = int(t)
                    if tid in tok.eos_token_ids:
                        break
                    toks.append(tid)
                el = time.perf_counter() - t0
            content, calls = split_tool_calls(tok.decode(toks))
            dpos = STATS["pos"] - before["pos"]
            dms = STATS["ms"] - before["ms"]
            print(f"  prompt {int(ids.size)} tok | ttft {ttft*1000:.0f} ms | "
                  f"{len(toks)} tok in {el:.1f}s = {len(toks)/max(el,1e-9):.1f} tok/s"
                  + (f" | ANE {dms/max(1,dpos):.2f} ms/pos over {dpos} pos" if dpos else "")
                  + (f" | {len(calls)} tool_call(s)" if calls else ""), flush=True)

            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")   # no chunking on SSE
                self.end_headers()
                self.close_connection = True
                def sse(o):
                    self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
                    self.wfile.flush()
                sse({"id": cid, "object": "chat.completion.chunk", "model": args.name,
                     "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                if content:
                    sse({"id": cid, "object": "chat.completion.chunk", "model": args.name,
                         "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
                if calls:
                    sse({"id": cid, "object": "chat.completion.chunk", "model": args.name,
                         "choices": [{"index": 0, "delta": {"tool_calls": [
                             dict(index=j, **c) for j, c in enumerate(calls)]},
                             "finish_reason": None}]})
                sse({"id": cid, "object": "chat.completion.chunk", "model": args.name,
                     "choices": [{"index": 0, "delta": {},
                                  "finish_reason": "tool_calls" if calls else "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            else:
                msg = {"role": "assistant", "content": content or None}
                if calls:
                    msg["tool_calls"] = calls
                self._json({"id": cid, "object": "chat.completion", "model": args.name,
                            "choices": [{"index": 0, "message": msg,
                                         "finish_reason": "tool_calls" if calls else "stop"}],
                            "usage": {"prompt_tokens": int(ids.size),
                                      "completion_tokens": len(toks),
                                      "total_tokens": int(ids.size) + len(toks)}})
    return H


def _serve(args, model, tok):
    """Serve on ONE thread.

    MLX streams are per-thread, and generation issued from a
    ThreadingHTTPServer worker dies with "There is no Stream(cpu, 0) in current
    thread" -- binding a stream in the handler is not enough because the ANE
    path builds arrays deeper in the stack. Decoding is sequential anyway (one
    ANE, one KV cache), so a threaded server bought nothing.
    """
    HTTPServer((args.host, args.port), build(model, tok, args)).serve_forever()


def bench(model, tok, args):
    from mlx_lm.generate import generate_step
    prompt = ("Explain, in about one hundred words, why mixture-of-experts models "
              "are harder to run efficiently than dense models.")
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = mx.array(tok.encode(text))
    for label in ("warmup", "measure"):
        STATS["pos"] = 0; STATS["ms"] = 0.0
        for k in PHASE: PHASE[k] = 0 if k == "n" else 0.0
        for k in GDNSTEP_STATS:
            GDNSTEP_STATS[k] = 0 if k == "n" else 0.0
        if label == "measure":
            print(f"MEASURE_START {time.time():.3f}", flush=True)
        toks, t0, ttft = [], time.perf_counter(), None
        for i, (t, _) in enumerate(generate_step(ids, model, max_tokens=args.max_tokens)):
            if ttft is None:
                ttft = time.perf_counter() - t0
            toks.append(int(t))
            if len(toks) >= args.max_tokens:
                break
        el = time.perf_counter() - t0
        if label == "measure":
            print(f"MEASURE_END {time.time():.3f}", flush=True)
            print(f"\n  prompt tokens      {int(ids.size)}")
            print(f"  time to first tok  {ttft*1000:.0f} ms")
            print(f"  generated          {len(toks)} tok in {el:.2f}s "
                  f"= {len(toks)/el:.1f} tok/s")
            if STATS["pos"]:
                print(f"  ANE                {STATS['ms']:.0f} ms over {STATS['pos']} "
                      f"positions = {STATS['ms']/STATS['pos']:.2f} ms/pos")
                if STATS.get("dec_n"):
                    print(f"  ANE decode         {STATS['dec_ms']/STATS['dec_n']:.3f} ms "
                          f"per layer-token ({STATS['dec_n']} dispatches)")
                print(f"  sample             {tok.decode(toks)[:220]!r}")
                if PHASE["n"]:
                    n = PHASE["n"]
                    print(f"  ANE phases/disp    cast {PHASE['cast']/n*1e3:.3f}  "
                          f"write {PHASE['write']/n*1e3:.3f}  "
                          f"submit {PHASE['submit']/n*1e3:.3f}  "
                          f"read {PHASE['read']/n*1e3:.3f} ms  "
                          f"(host {((PHASE['cast']+PHASE['write']+PHASE['read'])/n)*1e3:.3f})")
                if STATS.get("pre_tok"):
                    print(f"  ANE prefill        {STATS['pre_ms']:.0f} ms over "
                          f"{STATS['pre_tok']} layer-tokens")
            else:
                print("  ANE                not used (GPU MLPs)")
                print(f"  sample             {tok.decode(toks)[:200]!r}")
            if FL_PHASE["n"]:
                q = FL_PHASE["n"]
                print(f"  fused phases/disp  mx {FL_PHASE['mx']/q*1e3:.3f}  "
                      f"cat {FL_PHASE['cat']/q*1e3:.3f}  write {FL_PHASE['write']/q*1e3:.3f}  "
                      f"submit {FL_PHASE['submit']/q*1e3:.3f}  read {FL_PHASE['read']/q*1e3:.3f} ms")
            if GDNSTEP_STATS["n"]:
                n = GDNSTEP_STATS["n"]
                print(f"  ANE GDN step       {GDNSTEP_STATS['ms']/n:.3f} ms per call "
                      f"({n} calls; GPU→host {GDNSTEP_STATS['marshal_ms']/n:.3f}, "
                      f"resident ANE {GDNSTEP_STATS['resident_ms']/n:.3f})")
            if LAYER_STATS["n"]:
                print(f"  ANE fused layer    {LAYER_STATS['ms']/LAYER_STATS['n']:.3f} ms "
                      f"per call ({LAYER_STATS['n']} calls)")
            if ATTN_STATS["n"]:
                print(f"  ANE attn proj      {ATTN_STATS['ms']/ATTN_STATS['n']:.3f} ms "
                      f"per call ({ATTN_STATS['n']} calls)")
            if GDN_STATS["n"]:
                print(f"  ANE GDN proj       {GDN_STATS['ms']/GDN_STATS['n']:.3f} ms "
                      f"per call ({GDN_STATS['n']} calls)")
            if HEAD_STATS["n"]:
                print(f"  ANE lm_head        {HEAD_STATS['ms']/HEAD_STATS['n']:.3f} ms "
                      f"per position ({HEAD_STATS['n']} positions)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to an MLX checkpoint")
    p.add_argument("--ane-layers", type=int, default=0,
                   help="MoE layers to run on the ANE; -1 = all, 0 = GPU baseline")
    p.add_argument("--ane-seq", type=int, default=ANE_MIN_SEQ,
                   help=f"ANE program width; clamped to >={ANE_MIN_SEQ} (zeros below)")
    p.add_argument("--engine", default=os.environ.get(
        "Q38_ANE_ENGINE",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    p.add_argument("--port", type=int, default=1239)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--name", default=None, help="model id reported to clients")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--bench", action="store_true", help="run an A/B and exit")
    p.add_argument("--dense-layers", type=int, default=0,
                   help="dense MLP layers to BAKE onto the ANE; -1 = all")
    # int8 by default: it measures the same speed as int4 on the ANE (1.933 vs
    # 1.847 ms/layer) but re-quantizing an already-4-bit checkpoint to int4
    # doubles the error (0.107 -> 0.215 vs bf16), while int8 is free (0.107).
    p.add_argument("--dense-bits", type=int, default=8, choices=[4, 8, 16])
    p.add_argument("--no-fuse-gu", action="store_true",
                   help="keep gate and up as separate convs (3 dispatches) "
                        "instead of one fused 2I-channel conv")
    p.add_argument("--eager-load", action="store_true",
                   help="materialise the whole model at load time (the old "
                        "behaviour; costs ~55 GB of RSS while baking)")
    p.add_argument("--keep-mlx-weights", action="store_true",
                   help="do not free MLX weights after baking (keeps the GPU "
                        "fallback path alive, at full model memory)")
    p.add_argument("--ane-gdn-step", action="store_true",
                   help="run the gated-delta recurrence on the ANE at decode "
                        "(one program shared by every GDN layer)")
    p.add_argument("--ane-chain", action="store_true",
                   help="fuse each layer tail with the NEXT layer's norm and "
                        "input projection; one program per layer covers "
                        "every projection in the model")
    p.add_argument("--ane-fused-layers", type=int, default=0, metavar="N",
                   help="fuse N layer tails (out_proj + residual + RMSNorm + "
                        "MLP) into one ANE program each; -1 for all")
    p.add_argument("--ane-attn", action="store_true",
                   help="fuse q/k/v of each full_attention layer onto the ANE")
    p.add_argument("--ane-gdn", action="store_true",
                   help="fuse each GDN layer's four input projections into "
                        "one ANE conv")
    p.add_argument("--ane-gdn-layers", type=int, default=0,
                   help="limit how many GDN layers are baked (0 = all)")
    p.add_argument("--ane-lm-head", action="store_true",
                   help="bake lm_head onto the ANE (chunked along vocab)")
    p.add_argument("--lm-head-chunks", type=int, default=4,
                   help="how many vocab chunks to split lm_head into")
    p.add_argument("--sync-only", action="store_true",
                   help="diagnostic: do the host round-trip but compute on the "
                        "GPU, to price the MLX pipeline flush on its own")
    p.add_argument("--split-mlp", type=float, default=0.0, metavar="FRAC",
                   help="split each MLP: FRAC of intermediate channels on the "
                        "ANE, the rest on the GPU, run concurrently (~0.55 is "
                        "the measured balance point)")
    p.add_argument("--wide-prefill", action="store_true",
                   help="second S=64 program per layer for prefill (~1.5x "
                        "prefill, but doubles ANE-resident weights)")
    p.add_argument("--bake-cache", action="store_true",
                   help="cache quantized blobs under ~/.cache/ane_bake "
                        "(~1.9x faster rebake, but ~17 GB per model+precision "
                        "and unbounded across configurations)")
    args = p.parse_args()
    globals()['SYNC_ONLY'] = args.sync_only
    globals()['FUSE_GU'] = not args.no_fuse_gu
    globals()['FREE_MLX'] = not args.keep_mlx_weights
    args.engine = os.path.expanduser(args.engine)
    args.name = args.name or os.path.basename(args.model.rstrip("/"))

    print(f"loading {args.model} ...", flush=True)
    from mlx_lm import load
    # lazy=True leaves the weights memory-mapped instead of materialising the
    # whole bf16 model up front. Baking touches each tensor once and then frees
    # it, so peak RSS tracks the ANE blobs rather than the model: without this,
    # free memory hits 0.2 GB mid-bake and the ANE cannot wire its pages
    # (Program load failure 0x50004).
    model, tok = load(args.model, lazy=not args.eager_load)
    if args.eager_load:
        mx.eval(model.parameters())

    if (args.dense_layers or args.ane_lm_head or args.ane_gdn
            or args.ane_attn or args.ane_fused_layers or args.ane_chain
            or args.ane_gdn_step):
        _sweep_ane_tempdirs()
        cr = _bake_cache_dir(args.model, args.dense_bits) if args.bake_cache else None
        if cr:
            free = shutil.disk_usage(os.path.dirname(cr)).free
            print(f"  bake cache: {cr}  ({free/1e9:.0f} GB free)", flush=True)
        n = attach_ane_dense(model, args.dense_layers, args.engine,
                             args.ane_seq, args.dense_bits, cr,
                             args.wide_prefill, args.split_mlp) if args.dense_layers else 0
        print(f"ANE dense MLP layers: {n}", flush=True)
        # Order matters: the ANE holds at most 126 resident programs (measured --
        # program 127 fails to compile). A full build wants 64 MLP + 48 GDN +
        # 16 attention + 4 lm_head = 132, so the tail gets dropped. Bake the
        # small, high-value pieces first and let GDN absorb the shortfall.
        # Bake order matters when the build runs out of whatever resource
        # caps it. ANE_ORDER=gdn_first puts the small blocks in front so the
        # large fused-layer programs absorb any shortfall instead of GDN.
        nfused = [0]
        ngdnstep = [0]

        def _bake_fused():
            if args.ane_gdn_step:
                ngdnstep[0] = attach_ane_gdn_step(
                    model, args.engine, args.ane_seq, args.dense_bits
                )
            if args.ane_chain:
                nfused[0] = attach_ane_chain(model, args.engine, args.ane_seq,
                                             args.dense_bits, cr)
                print(f"ANE chained layers: {nfused[0]}", flush=True)
            if args.ane_fused_layers:
                nfused[0] = attach_ane_fused_layers(model, args.ane_fused_layers,
                                                    args.engine, args.ane_seq,
                                                    args.dense_bits, cr)
                print(f"ANE fused layer tails: {nfused[0]}   "
                      f"[rss {_rss_gb():.1f} GB, free {_free_gb():.1f} GB]",
                      flush=True)

        def _bake_small():
            # Spend the remaining program budget on the blocks that put the most
            # weight on the ANE per program: lm_head ~160 MB, GDN ~42 MB,
            # attention ~37 MB. Attention goes last so it absorbs any shortfall.
            used = [n + nfused[0] + ngdnstep[0]]
            def room():
                return max(0, ANE_MAX_PROGRAMS - used[0])
            if args.ane_lm_head:
                if room() < args.lm_head_chunks:
                    print(f"  lm_head skipped: {room()} of {ANE_MAX_PROGRAMS} "
                          f"program slots left", flush=True)
                else:
                    attach_ane_lm_head(model, args.engine, args.ane_seq,
                                       args.dense_bits, args.lm_head_chunks)
                    used[0] += args.lm_head_chunks
            if args.ane_gdn:
                want = args.ane_gdn_layers or room()
                if want > room():
                    print(f"  GDN capped at {room()} layers", flush=True)
                    want = room()
                used[0] += attach_ane_gdn(model, args.engine, args.ane_seq,
                                          args.dense_bits, want)
            if args.ane_attn:
                if room() == 0:
                    print(f"  attention skipped: no program slots left "
                          f"({used[0]}/{ANE_MAX_PROGRAMS})", flush=True)
                else:
                    used[0] += attach_ane_attn(model, args.engine, args.ane_seq,
                                               args.dense_bits, room())
            print(f"ANE programs resident: {used[0]}/{ANE_MAX_PROGRAMS}", flush=True)

        if os.environ.get("ANE_ORDER") == "gdn_first":
            _bake_small(); _bake_fused()
        else:
            _bake_fused(); _bake_small()
        if FREED[0]:
            print(f"freed {FREED[0]/1e9:.1f} GB of MLX weights now living on the ANE",
                  flush=True)
        if args.bench:
            bench(model, tok, args); return
        print(f"serving {args.name} on http://{args.host}:{args.port}/v1", flush=True)
        _serve(args, model, tok)
        return
    n = attach_ane(model, args.ane_layers, args.engine, args.ane_seq)
    print(f"ANE MoE layers: {n}" + (" (GPU baseline)" if n == 0 else
                                    f", S={max(ANE_MIN_SEQ, args.ane_seq)}"), flush=True)

    if args.bench:
        bench(model, tok, args)
        return
    print(f"serving {args.name} on http://{args.host}:{args.port}/v1", flush=True)
    _serve(args, model, tok)


if __name__ == "__main__":
    main()
