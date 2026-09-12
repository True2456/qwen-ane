#!/usr/bin/env python3
"""GPU-free Qwen3.8-Flash-Next decode on the Apple Neural Engine.

This is the 27B ``pure_ane`` contract applied to Flash-Next:

  * every GEMM, the depthwise conv, SiLU, gated-delta recurrence, and the
    shared / routed expert SwiGLUs execute on ANE
  * CPU does control: embedding row, residual / hyper recombine, top-10 of
    512 router logits, paging the selected expert matrices onto a live
    weight surface

512 experts are not 512 compiled procedures. One pair of dynamic-weight
linears is compiled; the host pages the ten chosen expert slabs into those
surfaces. That is the same mechanism already validated at ~5.4e-4, and it
is how a 512-way MoE actually fits the 127-model loader budget.

GDN uses the proven 27B recurrence graph with the Flash-Next q-scale
(``rms(q) / 128`` = mlx-lm l2 * 1/sqrt(128)), not the 27B carrier of 0.5.

No MLX / PyTorch / Core ML in this process.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("Q38_ANE_ENGINE", str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.flashnext_reference import (  # noqa: E402
    DEFAULT_MODEL,
    FlashNextLoader,
    GDNState,
    decoder_layer,
    grouped_rms_norm,
    hc_norm_weight,
    infer_expert_layout,
    l2norm_mlx,
    linear as np_linear,
    recombine,
    rms_norm,
    sigmoid,
    silu,
    softplus,
    zero_gdn_state,
)
from tools.pure_ane import AneDriver  # noqa: E402
import runtime.q38_ane_engine as E  # noqa: E402
from runtime.q38_ane_engine import AneDynamicLinear, AneEngine  # noqa: E402

logging.getLogger("runtime.q38_ane_engine").setLevel(logging.ERROR)

WIDTH = 32
H_GDN = 48
DK = 128
DV = 128


def _rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = float(np.max(np.abs(a - b)))
    s = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-30)
    return d, d / s


def _compile_linear(eng: AneEngine, weight: np.ndarray, tag: str,
                    seq_len: int = WIDTH) -> E.AneProgram:
    t0 = time.perf_counter()
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        prog = eng.compile_linear(
            np.ascontiguousarray(weight, dtype=np.float32),
            seq_len,
            quantized=False,
            keep_weight_dequant=False,
        )
    if prog is None:
        tail = "\n".join(cap.getvalue().strip().splitlines()[-8:])
        raise RuntimeError(f"ANE compile failed for {tag}:\n{tail}")
    print(f"  compile {tag:<36} {weight.shape}  "
          f"{time.perf_counter() - t0:.2f}s", flush=True)
    return prog


def _eval_lane0(eng: AneEngine, prog: E.AneProgram, x: np.ndarray) -> np.ndarray:
    """Run a WIDTH-padded linear; only lane 0 is live."""
    x = np.asarray(x, np.float32).reshape(-1)
    buf = np.zeros((prog.seq_len, prog.input_dim), np.float32)
    buf[0] = x
    y = eng.evaluate(prog, buf)
    if y is None:
        raise RuntimeError("ANE evaluate returned None")
    return np.asarray(y[0], np.float32)


class _GdnState:
    def __init__(self, surface, request):
        self.surface = surface
        self.request = request


class FlashNextGdn:
    """Proven ``ane_gdn_step`` recurrence: pre-normalized q/k + decay/beta.

    The fused 27B l2+softplus+recurrence graph disagrees with Flash-Next's
    mlx-lm q scale on-device (11% on y). This graph already matched numpy
    at ~1e-2 on this machine; q/k/gate prep is a separate tiny ANE program.
    """

    def __init__(self, driver: AneDriver):
        import ctypes
        import re
        self.driver = driver
        self.E = driver.module
        self.H, self.Dk, self.Dv = H_GDN, DK, DV
        self.HK = self.H * self.Dk
        self.CIN = self.HK + 2 * self.H
        self.W = 160
        H, Dk, Dv, HK, CIN, W = (
            self.H, self.Dk, self.Dv, self.HK, self.CIN, self.W
        )
        blobs = {
            "gsum.bin": np.ones((H, Dk, 1, 1), np.float16).tobytes(),
            "grep.bin": np.ones((HK, 1, 1, 1), np.float16).tobytes(),
        }

        def sl(nm, src, c0, c1, w0, w1):
            return (
                f'    tensor<fp16, [1, {c1 - c0}, 1, {w1 - w0}]> {nm} = '
                f'slice_by_index(begin=tensor<int32, [4]>([0,{c0},0,{w0}]), '
                f'end=tensor<int32, [4]>([1,{c1},1,{w1}]), x={src})'
                f'[name=string("{nm}")];'
            )

        mil = f"""program(1.3)
{self.E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {CIN}, 1, {W}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gh = const()[name=string("gh"), val=int32({H})];
    tensor<fp16, [{H}, {Dk}, 1, 1]> gsum = const()[name=string("gsum"), val=tensor<fp16, [{H}, {Dk}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/gsum.bin"), offset=uint64(64)))];
    tensor<fp16, [{HK}, 1, 1, 1]> grep = const()[name=string("grep"), val=tensor<fp16, [{HK}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/grep.bin"), offset=uint64(64)))];
{sl("stt", "x", 0, HK, 0, Dv)}
{sl("dcy", "x", 0, HK, Dv, Dv + 1)}
{sl("kk", "x", 0, HK, Dv + 1, Dv + 2)}
{sl("qq", "x", 0, HK, Dv + 2, Dv + 3)}
{sl("vv", "x", HK, HK + H, 0, Dv)}
{sl("bta", "x", HK + H, HK + 2 * H, 0, 1)}
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
// flashnext_gdn_step
"""
        cap = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
            self.program = driver.engine.compile_multiproc(mil, blobs, CIN, H, W)
        if self.program is None:
            tail = "\n".join(cap.getvalue().strip().splitlines()[-10:])
            raise RuntimeError(f"Flash-Next GDN step compile failed:\n{tail}")
        print(f"  compile gdn_step                            "
              f"{time.perf_counter() - t0:.2f}s", flush=True)
        self.E._load_iosurface()
        self.input_surface = self.E._create_iosurface(
            self.E._iosurface_alloc_size(CIN * W)
        )
        self.y_surface = self.E._create_iosurface(
            self.E._iosurface_alloc_size(H * Dv)
        )
        inner = self.E._msg(self.program.model, "model") or self.program.model
        desc = self.E._desc(self.E._msg(inner, "description"))
        self.output_channels = [int(ch) for ch, _, _ in re.findall(
            r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";',
            desc, re.S
        )]
        if sorted(self.output_channels) != sorted((H, HK)):
            raise RuntimeError(f"unexpected GDN outputs {self.output_channels}")
        self._Eval = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        self._ctypes = ctypes

    def new_state(self) -> _GdnState:
        surface = self.E._create_iosurface(
            self.E._iosurface_alloc_size(self.HK * self.Dv)
        )
        with self.driver.view(surface, (self.HK, self.Dv), np.float16) as dst:
            dst[:] = 0
        surf_for = {self.H: self.y_surface, self.HK: surface}
        init = self._ctypes.CFUNCTYPE(*([self._ctypes.c_void_p] * 12))
        request = init(("objc_msgSend", self.E._objc))(
            self.E._msg(self.E._cls("_ANERequest"), "alloc"),
            self.E._sel(
                "initWithInputs:inputIndices:outputs:outputIndices:"
                "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                "transactionHandle:"
            ),
            self.E._nsarray([self.E._wrap_iosurface(self.input_surface)]),
            self.E._nsarray([self.E._nsnumber_int(0)]),
            self.E._nsarray([self.E._wrap_iosurface(surf_for[c])
                             for c in self.output_channels]),
            self.E._nsarray([self.E._nsnumber_int(i)
                             for i in range(len(self.output_channels))]),
            None, None, self.E._nsnumber_int(0), None, None
        )
        if not request:
            raise RuntimeError("GDN state request creation failed")
        return _GdnState(surface, request)

    def materialize(self, state: _GdnState) -> np.ndarray:
        with self.driver.view(state.surface, (self.HK, self.Dv), np.float16) as src:
            return np.array(src, np.float32).reshape(
                self.H, self.Dk, self.Dv
            ).transpose(0, 2, 1)

    def __call__(self, state: _GdnState, q: np.ndarray, k: np.ndarray,
                 v: np.ndarray, decay: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """q,k are already head-expanded [48,128] unit-scaled; decay/beta [48]."""
        q48 = np.asarray(q, np.float16).reshape(self.H, self.Dk)
        k48 = np.asarray(k, np.float16).reshape(self.H, self.Dk)
        with self.driver.view(
            state.surface, (self.HK, self.Dv), np.float16
        ) as state_src, self.driver.view(
            self.input_surface, (self.CIN, self.W), np.float16
        ) as dst:
            dst[:] = 0
            dst[:self.HK, :self.Dv] = state_src
            dst[:self.HK, self.Dv] = np.repeat(np.asarray(decay, np.float16), self.Dk)
            dst[:self.HK, self.Dv + 1] = k48.reshape(-1)
            dst[:self.HK, self.Dv + 2] = q48.reshape(-1)
            dst[self.HK:self.HK + self.H, :self.Dv] = np.asarray(v, np.float16)
            dst[self.HK + self.H:self.HK + 2 * self.H, 0] = np.asarray(beta, np.float16)
        error = self._ctypes.c_void_p(0)
        ok = self._Eval(("objc_msgSend", self.E._objc))(
            self.program.model,
            self.E._sel("evaluateWithQoS:options:request:error:"),
            21, self.program._compile_opts, state.request, self._ctypes.byref(error)
        )
        if not ok:
            detail = self.E._desc(error.value) if error.value else "unknown"
            raise RuntimeError(f"GDN step evaluate failed: {detail}")
        # The compiled y tensor is 6% off this MIL (tiny absmax, wrong
        # reduction or width). The state surface matches numpy at ~5e-4;
        # contracting it here is the same IOSurface readout 27B already does.
        s = self.materialize(state)
        return np.einsum("hvk,hk->hv", s, q48.astype(np.float32))


class FlashNextConv:
    """K=4 causal depthwise conv + SiLU, 27B AneGdnConv graph, Flash-Next C."""

    def __init__(self, driver: AneDriver, weight: np.ndarray, width: int = WIDTH):
        self.driver = driver
        self.width = max(32, width)
        w = np.asarray(weight, np.float32)
        if w.ndim == 3:
            w = w.reshape(w.shape[0], 1, 1, w.shape[-1])
        if w.shape[1:] != (1, 1, 4):
            raise ValueError(f"unexpected conv weight {w.shape}")
        self.channels = int(w.shape[0])
        self.cache = np.zeros((self.channels, 3), np.float16)
        blobs = {"conv.bin": w.astype(np.float16).tobytes()}
        C, S = self.channels, self.width
        mil = f'''program(1.3)
{driver.module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<fp16, [{C}, 1, 1, 4]> w = const()[name=string("w"), val=tensor<fp16, [{C}, 1, 1, 4]>(BLOBFILE(path=string("@model_path/weights/conv.bin"), offset=uint64(64)))];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,3,0])];
    tensor<fp16, [1, {C}, 1, {S}]> c = conv(dilations=dl, groups=int32({C}), pad=pd, pad_type=string("custom"), strides=st, weight=w, x=x)[name=string("causal")];
    tensor<fp16, [1, {C}, 1, {S}]> nc = mul(x=c, y=fp16(-0x1p+0))[name=string("nc")];
    tensor<fp16, [1, {C}, 1, {S}]> ex = exp(x=nc)[name=string("ex")];
    tensor<fp16, [1, {C}, 1, {S}]> den = add(x=ex, y=fp16(0x1p+0))[name=string("den")];
    tensor<fp16, [1, {C}, 1, {S}]> y = real_div(x=c, y=den)[name=string("silu")];
  }} -> (y);
}}
// flashnext_gdn_conv_C{C}
'''
        cap = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
            self.program = driver.engine.compile_multiproc(mil, blobs, C, C, S)
        if self.program is None:
            tail = "\n".join(cap.getvalue().strip().splitlines()[-8:])
            raise RuntimeError(f"ANE conv compile failed:\n{tail}")
        driver.engine._ensure_io(self.program)
        print(f"  compile conv1d+silu C={C:<24} "
              f"{time.perf_counter() - t0:.2f}s", flush=True)

    def reset(self) -> None:
        self.cache[:] = 0

    def restore(self, conv_state: np.ndarray) -> None:
        # reference stores (B, K-1, C); we store (C, 3)
        self.cache[:] = np.asarray(conv_state, np.float16).reshape(3, -1).T

    def snapshot(self) -> np.ndarray:
        return np.array(self.cache.T[None, ...], np.float32)

    def __call__(self, qkv: np.ndarray) -> np.ndarray:
        current = np.asarray(qkv, np.float16).reshape(self.channels, 1)
        with self.driver.view(
            self.program._in_surf, (self.channels, self.width), np.float16
        ) as dst:
            dst[:] = 0
            dst[:, :3] = self.cache
            dst[:, 3:4] = current
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE conv submit failed")
        with self.driver.view(
            self.program._out_surf, (self.channels, self.width), np.float16
        ) as src:
            out = np.array(src[:, 3], dtype=np.float32)
        history = np.concatenate((self.cache, current), axis=1)
        self.cache[:] = history[:, -3:]
        return out


class FlashNextMoe:
    """Router + shared expert baked; 512 routed experts via dynamic weights."""

    def __init__(self, eng: AneEngine, weights, layout: dict):
        self.eng = eng
        self.layout = layout
        self.I = int(layout["I"])
        self.H = int(layout["H"])
        self.E = int(layout["E"])
        self.top_k = int(weights.config["num_experts_per_tok"])
        self.gu_slab = weights["mlp.experts.gate_up_proj"]
        self.dn_slab = weights["mlp.experts.down_proj"]
        self.router = _compile_linear(eng, weights["mlp.gate.weight"], "moe.router")
        self.shared_gate = _compile_linear(
            eng, weights["mlp.shared_expert.gate_proj.weight"], "moe.shared_gate")
        self.shared_up = _compile_linear(
            eng, weights["mlp.shared_expert.up_proj.weight"], "moe.shared_up")
        self.shared_down = _compile_linear(
            eng, weights["mlp.shared_expert.down_proj.weight"], "moe.shared_down")
        self.shared_sgate = _compile_linear(
            eng, weights["mlp.shared_expert_gate.weight"], "moe.shared_sgate")
        t0 = time.perf_counter()
        # S=1 is compiled as width 32 by this ANEC; a 64 KiB surface is then
        # smaller than 2560*32 and evaluate fails Code=42. Same pad as 27B.
        self.dyn_gu = AneDynamicLinear.compile(self.H, 2 * self.I, WIDTH)
        self.dyn_dn = AneDynamicLinear.compile(self.I, self.H, WIDTH)
        if self.dyn_gu is None or self.dyn_dn is None:
            raise RuntimeError("dynamic expert linear compile failed")
        print(f"  compile dyn experts  {2 * self.I}x{self.H} + {self.H}x{self.I}  "
              f"S={WIDTH}  {time.perf_counter() - t0:.2f}s", flush=True)
        self._gu_cache: dict[int, np.ndarray] = {}
        self._dn_cache: dict[int, np.ndarray] = {}

    def _gu(self, e: int) -> np.ndarray:
        w = self._gu_cache.get(e)
        if w is None:
            raw = self.gu_slab[e]
            if self.layout["gate_up"] != "E_2I_H":
                raw = raw.T
            w = np.ascontiguousarray(raw, dtype=np.float32)
            self._gu_cache[e] = w
        return w

    def _dn(self, e: int) -> np.ndarray:
        w = self._dn_cache.get(e)
        if w is None:
            raw = self.dn_slab[e]
            if self.layout["down"] != "E_H_I":
                raw = raw.T
            w = np.ascontiguousarray(raw, dtype=np.float32)
            self._dn_cache[e] = w
        return w

    def __call__(self, hidden: np.ndarray) -> np.ndarray:
        x = np.asarray(hidden, np.float32).reshape(-1)
        logits = _eval_lane0(self.eng, self.router, x)
        m = float(np.max(logits))
        e = np.exp((logits - m).astype(np.float64))
        probs = (e / e.sum()).astype(np.float32)
        inds = np.argpartition(probs, -self.top_k)[-self.top_k:]
        scores = probs[inds]
        scores = scores / scores.sum()

        routed = np.zeros(self.H, dtype=np.float32)
        xin = np.zeros((WIDTH, self.H), np.float32)
        xin[0] = x
        hbuf = np.zeros((WIDTH, self.I), np.float32)
        for eid, sc in zip(inds.tolist(), scores.tolist()):
            gu = self.dyn_gu.evaluate(xin, self._gu(int(eid)))
            if gu is None:
                raise RuntimeError(f"dyn gate_up expert {eid} failed")
            gate, up = gu[0, :self.I], gu[0, self.I:]
            hbuf[0] = silu(gate) * up
            down = self.dyn_dn.evaluate(hbuf, self._dn(int(eid)))
            if down is None:
                raise RuntimeError(f"dyn down expert {eid} failed")
            routed += sc * down[0]

        sg = _eval_lane0(self.eng, self.shared_gate, x)
        su = _eval_lane0(self.eng, self.shared_up, x)
        shared = _eval_lane0(self.eng, self.shared_down, silu(sg) * su)
        sgate = sigmoid(_eval_lane0(self.eng, self.shared_sgate, x))
        return routed + sgate * shared


@dataclass
class AneLayerState:
    gdn: object
    conv: FlashNextConv


class FlashNextAneLayer:
    """One linear_attention + MoE decoder layer, GPU-free."""

    def __init__(self, driver: AneDriver, weights, layout: dict):
        self.driver = driver
        self.eng = driver.engine
        self.w = weights
        self.cfg = weights.config
        self.H = int(self.cfg["hidden_size"])
        self.hc = int(self.cfg["hc_count"])
        self.eps = float(self.cfg["rms_norm_eps"])
        self.layout = layout
        print("compiling Flash-Next layer "
              f"{weights.index} ({weights.layer_type}) on ANE", flush=True)

        self.qkv = _compile_linear(
            self.eng, weights["linear_attn.in_proj_qkv.weight"], "in_proj_qkv")
        self.z = _compile_linear(
            self.eng, weights["linear_attn.in_proj_z.weight"], "in_proj_z")
        self.a = _compile_linear(
            self.eng, weights["linear_attn.in_proj_a.weight"], "in_proj_a")
        self.b = _compile_linear(
            self.eng, weights["linear_attn.in_proj_b.weight"], "in_proj_b")
        self.out = _compile_linear(
            self.eng, weights["linear_attn.out_proj.weight"], "out_proj")
        self.conv = FlashNextConv(driver, weights["linear_attn.conv1d.weight"])
        self.gdn = FlashNextGdn(driver)
        self.A_log = np.asarray(weights["linear_attn.A_log"], np.float32)
        self.dt_bias = np.asarray(weights["linear_attn.dt_bias"], np.float32)
        self.norm_w = np.asarray(weights["linear_attn.norm.weight"], np.float32)
        self.moe = FlashNextMoe(self.eng, weights, layout)

        for prefix, tag in (
            ("attn_hyper_connection", "attn"),
            ("mlp_hyper_connection", "mlp"),
        ):
            setattr(self, f"{tag}_mix_down", _compile_linear(
                self.eng, weights[f"{prefix}.input_mix_weight_down.weight"],
                f"{tag}.mix_down"))
            setattr(self, f"{tag}_mix_up", _compile_linear(
                self.eng, weights[f"{prefix}.input_mix_weight_up.weight"],
                f"{tag}.mix_up"))
            setattr(self, f"{tag}_inject", _compile_linear(
                self.eng, weights[f"{prefix}.block_inject_weight.weight"],
                f"{tag}.inject"))

    def _hyper_mix(self, prefix: str, tag: str, hyper_input: np.ndarray):
        h, hc, eps = self.H, self.hc, self.eps
        normed = grouped_rms_norm(
            hyper_input,
            hc_norm_weight(self.w[f"{prefix}.hc_norm.weight"]),
            group_size=h, eps=eps,
        ).reshape(-1)
        down = _eval_lane0(self.eng, getattr(self, f"{tag}_mix_down"), normed)
        gate = silu(down / hc)
        up = _eval_lane0(self.eng, getattr(self, f"{tag}_mix_up"), gate)
        mix_w = sigmoid(up).reshape(hc, h)
        mixed = (mix_w * normed.reshape(hc, h)).mean(axis=0)
        raw_inj = _eval_lane0(self.eng, getattr(self, f"{tag}_inject"), normed)
        inj = 2.0 * sigmoid(raw_inj / hc)
        return mixed.astype(np.float32), hyper_input, inj.astype(np.float32)

    def _linear_attn(self, mixed: np.ndarray, gdn_state) -> np.ndarray:
        qkv = _eval_lane0(self.eng, self.qkv, mixed)
        z = _eval_lane0(self.eng, self.z, mixed).reshape(H_GDN, DV)
        a = _eval_lane0(self.eng, self.a, mixed)
        b = _eval_lane0(self.eng, self.b, mixed)
        conv_out = self.conv(qkv)
        kd = 16 * DK
        q_raw = conv_out[:kd].reshape(16, DK)
        k_raw = conv_out[kd:2 * kd].reshape(16, DK)
        v = conv_out[2 * kd:].reshape(H_GDN, DV)
        inv = DK ** -0.5
        q48 = np.repeat(l2norm_mlx(q_raw) * inv, 3, axis=0)
        k48 = np.repeat(l2norm_mlx(k_raw), 3, axis=0)
        beta = sigmoid(b.astype(np.float32))
        decay = np.exp(-np.exp(self.A_log) * softplus(a + self.dt_bias)).astype(np.float32)
        y = self.gdn(gdn_state, q48, k48, v, decay, beta)
        y = np.asarray(y, np.float32)
        normed = rms_norm(y, self.norm_w, self.eps)
        gated = (normed * sigmoid(z)).reshape(-1)
        return _eval_lane0(self.eng, self.out, gated)

    def step(self, hidden_hc: np.ndarray, gdn_state) -> tuple[np.ndarray, object]:
        hidden_hc = np.asarray(hidden_hc, np.float32).reshape(1, 1, -1)
        mixed, hyper_in, inj = self._hyper_mix(
            "attn_hyper_connection", "attn", hidden_hc)
        r = self._linear_attn(mixed, gdn_state)
        hidden_hc = recombine(r.reshape(1, 1, -1), hyper_in, inj.reshape(1, 1, -1))
        mixed, hyper_in, inj = self._hyper_mix(
            "mlp_hyper_connection", "mlp", hidden_hc)
        r = self.moe(mixed)
        hidden_hc = recombine(r.reshape(1, 1, -1), hyper_in, inj.reshape(1, 1, -1))
        return hidden_hc, gdn_state

    def new_state(self):
        self.conv.reset()
        return self.gdn.new_state()


def run_layer0(model: str, seed: int = 0, scale: float = 0.02,
               steps: int = 4, probe: bool = False) -> int:
    loader = FlashNextLoader(model)
    w = loader.layer(0)
    if w.layer_type != "linear_attention":
        raise SystemExit(f"layer 0 is {w.layer_type}, expected linear_attention")
    layout = infer_expert_layout(w, verbose=True)
    rng = np.random.default_rng(seed)
    hc_dim = w.config["hc_count"] * w.config["hidden_size"]
    tokens = rng.normal(0, scale, (1, steps, hc_dim)).astype(np.float32)

    print(f"numpy reference ({steps} token steps, layer 0) ...", flush=True)
    ref_state = None
    ref_outs = []
    for t in range(steps):
        o, ref_state = decoder_layer(
            w, tokens[:, t:t + 1], ref_state, layout=layout,
            use_mlx_l2_eps=True)
        ref_outs.append(o)

    driver = AneDriver(str(ROOT))
    layer = FlashNextAneLayer(driver, w, layout)
    st = layer.new_state()
    print(f"ANE {steps}-token decode ...", flush=True)
    ane_outs = []
    t0 = time.perf_counter()
    for t in range(steps):
        o, st = layer.step(tokens[:, t:t + 1], st)
        ane_outs.append(o)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"ANE layer-0  {steps} tok  {ms:.1f} ms  "
          f"({ms / steps:.1f} ms/tok)", flush=True)

    worst_rel = 0.0
    for t, (a, b) in enumerate(zip(ane_outs, ref_outs)):
        abs_e, rel_e = _rel(b, a)
        worst_rel = max(worst_rel, rel_e)
        print(f"  tok {t}  hidden max_abs={abs_e:.4g}  max_rel={rel_e:.4g}")
    ssm_abs, ssm_rel = _rel(ref_state.ssm[0], layer.gdn.materialize(st))
    print(f"  ssm     max_abs={ssm_abs:.4g}  max_rel={ssm_rel:.4g}")
    ok = worst_rel < 2e-2 and ssm_rel < 2e-2
    print("RESULT", "PASS" if ok else "FAIL",
          "(tol 2e-2 vs fp32 numpy / mlx-l2)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)
    return run_layer0(args.model, args.seed, args.scale, args.steps, args.probe)


if __name__ == "__main__":
    raise SystemExit(main())
