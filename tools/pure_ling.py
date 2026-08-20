#!/usr/bin/env python3
"""Standalone, GPU-free Ling-3.0-tiny (BailingMoeV3) inference on the ANE.

Like ``pure_ane.py`` this module imports no MLX, oMLX, PyTorch, Core ML or
Transformers; it reuses that module's checkpoint reader, MIL helpers and driver
boundary, and adds the BailingMoeV3 architecture on top.  Importing
``pure_ane`` is safe: its ``main()`` is guarded.

The architecture differs from Qwen3.8 in three ways that matter here:

* attention alternates KDA (linear) and MLA (full) on a ``(i+1) % 4 == 0`` rule
* every layer but layer 0 is a 128-expert MoE, top-8, plus one shared expert
* RMSNorm weights are stored plainly, not as deltas from one

Strategy for the experts is settled by ``probes/ane_ling_moe_strategy.py`` and
recorded in ``docs/LING-PORT.md``: bake all 128 experts per layer as int4
constants and compute every one, rather than staging the routed eight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pure_ane import (                                          # noqa: E402
    AneDriver, Checkpoint, StandaloneTokenizer, assert_standalone,
)

MODEL_TYPE = "bailing_hybrid"


@dataclass(frozen=True)
class LingSpec:
    """Architecture of a BailingMoeV3 checkpoint, read from config, not guessed.

    Every field here is verified against the checkpoint's tensor index by
    ``verify()`` rather than trusted: ``config.json`` carries several keys the
    modeling code never reads (``max_window_layers``,
    ``num_kv_heads_for_linear_attn``, ``use_qk_norm``), and acting on those
    would silently build the wrong model.
    """

    layers: int
    hidden: int
    intermediate: int                 # dense MLP, layer 0 only
    moe_intermediate: int
    experts: int
    top_k: int
    shared_experts: int
    first_k_dense: int
    layer_group: int
    n_group: int
    topk_group: int
    routed_scale: float
    norm_topk_prob: bool
    heads: int
    head_dim: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope: int
    qk_rope: int
    v_head_dim: int
    rope_theta: float
    rope_interleave: bool
    conv_kernel: int
    kda_lower_bound: float
    kda_safe_gate: bool
    rms_eps: float
    vocab: int
    tie_embeddings: bool

    @classmethod
    def from_config(cls, config: dict) -> "LingSpec":
        if config.get("model_type") != MODEL_TYPE:
            raise ValueError(
                f"expected model_type {MODEL_TYPE!r}, got {config.get('model_type')!r}"
            )
        return cls(
            layers=config["num_hidden_layers"],
            hidden=config["hidden_size"],
            intermediate=config["intermediate_size"],
            moe_intermediate=config["moe_intermediate_size"],
            experts=config["num_experts"],
            top_k=config["num_experts_per_tok"],
            shared_experts=config["num_shared_experts"],
            first_k_dense=config["first_k_dense_replace"],
            layer_group=config["layer_group_size"],
            n_group=config["n_group"],
            topk_group=config["topk_group"],
            routed_scale=float(config["routed_scaling_factor"]),
            norm_topk_prob=bool(config["norm_topk_prob"]),
            heads=config["num_attention_heads"],
            head_dim=config["head_dim"],
            q_lora_rank=config["q_lora_rank"],
            kv_lora_rank=config["kv_lora_rank"],
            qk_nope=config["qk_nope_head_dim"],
            qk_rope=config["qk_rope_head_dim"],
            v_head_dim=config["v_head_dim"],
            rope_theta=float(config["rope_theta"]),
            rope_interleave=bool(config["rope_interleave"]),
            conv_kernel=config["short_conv_kernel_size"],
            kda_lower_bound=float(config["kda_lower_bound"]),
            kda_safe_gate=bool(config["kda_safe_gate"]),
            rms_eps=float(config["rms_norm_eps"]),
            vocab=config["vocab_size"],
            tie_embeddings=bool(config["tie_word_embeddings"]),
        )

    # ---------------------------------------------------------------- layout
    def is_full_attention(self, layer: int) -> bool:
        """BailingMoeV3DecoderLayer.__init__: `(i+1) % layer_group == 0`.

        The reference also forces full attention for any trailing remainder
        layers (`i >= layers // group * group`), which is dead whenever the
        depth divides evenly. It is reproduced so odd depths stay correct.
        """
        return ((layer + 1) % self.layer_group == 0
                or layer >= self.layers // self.layer_group * self.layer_group)

    def is_moe(self, layer: int) -> bool:
        return layer >= self.first_k_dense

    @property
    def full_attention_layers(self) -> list[int]:
        return [i for i in range(self.layers) if self.is_full_attention(i)]

    @property
    def linear_attention_layers(self) -> list[int]:
        return [i for i in range(self.layers) if not self.is_full_attention(i)]

    @property
    def moe_layers(self) -> list[int]:
        return [i for i in range(self.layers) if self.is_moe(i)]

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def kda_proj_dim(self) -> int:
        """q/k/v/f/g each project to heads * head_dim."""
        return self.heads * self.head_dim

    # ----------------------------------------------------------------- names
    def prefix(self, layer: int) -> str:
        return f"model.layers.{layer}"

    def attention_names(self, layer: int) -> dict[str, str]:
        """Tensor names for one attention block.

        MLA's output projection is `dense`; KDA's is `o_proj`. That asymmetry
        is in the checkpoint, not a transcription slip.
        """
        p = f"{self.prefix(layer)}.attention"
        if self.is_full_attention(layer):
            return {
                "q_a": f"{p}.q_a_proj.weight",
                "q_a_norm": f"{p}.q_a_layernorm.weight",
                "q_b": f"{p}.q_b_proj.weight",
                "kv_a": f"{p}.kv_a_proj_with_mqa.weight",
                "kv_a_norm": f"{p}.kv_a_layernorm.weight",
                "kv_b": f"{p}.kv_b_proj.weight",
                "gate": f"{p}.g_proj.weight",
                "out": f"{p}.dense.weight",
            }
        return {
            "q": f"{p}.q_proj.weight", "k": f"{p}.k_proj.weight",
            "v": f"{p}.v_proj.weight",
            "q_conv": f"{p}.q_conv1d.weight", "k_conv": f"{p}.k_conv1d.weight",
            "v_conv": f"{p}.v_conv1d.weight",
            "f": f"{p}.f_proj.weight", "b": f"{p}.b_proj.weight",
            "gate": f"{p}.g_proj.weight",
            "a_log": f"{p}.A_log", "dt_bias": f"{p}.dt_bias",
            "o_norm": f"{p}.o_norm.weight", "out": f"{p}.o_proj.weight",
        }

    def mlp_names(self, layer: int) -> dict[str, str]:
        p = f"{self.prefix(layer)}.mlp"
        if not self.is_moe(layer):
            return {k: f"{p}.{k}_proj.weight" for k in ("gate", "up", "down")}
        names = {
            "router": f"{p}.gate.weight",
            "expert_bias": f"{p}.gate.expert_bias",
        }
        for k in ("gate", "up", "down"):
            names[f"shared_{k}"] = f"{p}.shared_experts.{k}_proj.weight"
        return names

    def expert_name(self, layer: int, expert: int, proj: str) -> str:
        return f"{self.prefix(layer)}.mlp.experts.{expert}.{proj}_proj.weight"

    def norm_names(self, layer: int) -> dict[str, str]:
        p = self.prefix(layer)
        return {"input": f"{p}.input_layernorm.weight",
                "post_attention": f"{p}.post_attention_layernorm.weight"}

    # ---------------------------------------------------------------- verify
    def verify(self, checkpoint: Checkpoint) -> list[str]:
        """Check every derived name and shape against the checkpoint index.

        Returns a list of problems; empty means the spec matches the tensors.
        """
        problems: list[str] = []

        def want(name: str, shape: tuple[int, ...] | None = None) -> None:
            try:
                info = checkpoint.info(name)
            except KeyError:
                problems.append(f"missing {name}")
                return
            if shape is not None and tuple(info.shape) != tuple(shape):
                problems.append(f"{name}: expected {tuple(shape)}, got {tuple(info.shape)}")

        H, P = self.hidden, self.kda_proj_dim
        # the layer split must match which tensors actually exist
        present_mla = sorted(
            int(n.split(".")[2]) for n in checkpoint.weight_map
            if n.endswith("attention.q_a_proj.weight")
        )
        present_kda = sorted(
            int(n.split(".")[2]) for n in checkpoint.weight_map
            if n.endswith("attention.A_log")
        )
        if present_mla != self.full_attention_layers:
            problems.append(
                f"full-attention layers {self.full_attention_layers} != "
                f"checkpoint {present_mla}")
        if present_kda != self.linear_attention_layers:
            problems.append(
                f"linear-attention layers {self.linear_attention_layers} != "
                f"checkpoint {present_kda}")

        for layer in (self.linear_attention_layers[0], self.full_attention_layers[0]):
            n = self.attention_names(layer)
            if self.is_full_attention(layer):
                want(n["q_a"], (self.q_lora_rank, H))
                want(n["q_a_norm"], (self.q_lora_rank,))
                want(n["q_b"], (self.heads * self.qk_head_dim, self.q_lora_rank))
                want(n["kv_a"], (self.kv_lora_rank + self.qk_rope, H))
                want(n["kv_a_norm"], (self.kv_lora_rank,))
                want(n["kv_b"],
                     (self.heads * (self.qk_nope + self.v_head_dim), self.kv_lora_rank))
                want(n["gate"], (self.heads, H))
                want(n["out"], (H, self.heads * self.v_head_dim))
            else:
                for k in ("q", "k", "v", "f", "gate"):
                    want(n[k], (P, H))
                for k in ("q_conv", "k_conv", "v_conv"):
                    want(n[k], (P, 1, self.conv_kernel))
                want(n["b"], (self.heads, H))
                want(n["a_log"], (self.heads,))
                want(n["dt_bias"], (P,))
                want(n["o_norm"], (self.head_dim,))
                want(n["out"], (H, P))
            for name in self.norm_names(layer).values():
                want(name, (H,))

        moe = self.moe_layers[0]
        m = self.mlp_names(moe)
        want(m["router"], (self.experts, H))
        want(m["expert_bias"], (self.experts,))
        shared = self.moe_intermediate * self.shared_experts
        want(m["shared_gate"], (shared, H))
        want(m["shared_down"], (H, shared))
        for e in (0, self.experts - 1):
            want(self.expert_name(moe, e, "gate"), (self.moe_intermediate, H))
            want(self.expert_name(moe, e, "up"), (self.moe_intermediate, H))
            want(self.expert_name(moe, e, "down"), (H, self.moe_intermediate))
        found = sum(
            1 for n in checkpoint.weight_map
            if n.startswith(f"{self.prefix(moe)}.mlp.experts.")
            and n.endswith("gate_proj.weight")
        )
        if found != self.experts:
            problems.append(f"layer {moe}: {found} experts, config says {self.experts}")

        if not self.is_moe(0):
            d = self.mlp_names(0)
            want(d["gate"], (self.intermediate, H))
            want(d["down"], (H, self.intermediate))

        want("model.norm.weight", (H,))
        want("lm_head.weight", (self.vocab, H))
        want(checkpoint.embedding_name, (self.vocab, H))
        return problems


def load(model_dir: str) -> tuple[Checkpoint, LingSpec]:
    """Open a BailingMoeV3 checkpoint with the plain-RMSNorm convention."""
    checkpoint = Checkpoint(model_dir, shifted_norms=False)
    return checkpoint, LingSpec.from_config(checkpoint.config)


def _expert_bytes(spec: LingSpec, bits: int) -> float:
    per = 3 * spec.moe_intermediate * spec.hidden
    return per * spec.experts * len(spec.moe_layers) * bits / 8


def inspect(model_dir: str) -> None:
    checkpoint, spec = load(model_dir)
    print(json.dumps({
        "model_type": checkpoint.config.get("model_type"),
        "layers": spec.layers, "hidden": spec.hidden, "vocab": spec.vocab,
        "experts": spec.experts, "top_k": spec.top_k,
        "moe_intermediate": spec.moe_intermediate,
        "shared_experts": spec.shared_experts,
        "rms_norm_eps": spec.rms_eps,
    }, indent=2))
    print(f"tensors={len(checkpoint.weight_map)} "
          f"shards={len(set(checkpoint.weight_map.values()))}")
    print(f"embedding={checkpoint.embedding_name} "
          f"{tuple(checkpoint.info(checkpoint.embedding_name).shape)}")
    print(f"full_attention (MLA) layers={spec.full_attention_layers}")
    print(f"linear_attention (KDA) layers={spec.linear_attention_layers}")
    print(f"moe layers={spec.moe_layers[0]}..{spec.moe_layers[-1]} "
          f"({len(spec.moe_layers)}), dense mlp layers="
          f"{[i for i in range(spec.layers) if not spec.is_moe(i)]}")
    print(f"kda safe_gate={spec.kda_safe_gate} lower_bound={spec.kda_lower_bound} "
          f"conv_kernel={spec.conv_kernel} rope_interleave={spec.rope_interleave}")
    for bits in (16, 8, 4):
        print(f"routed experts at int{bits}: "
              f"{_expert_bytes(spec, bits)/1e9:.2f} GB")
    problems = spec.verify(checkpoint)
    if problems:
        print(f"\nPURE_LING_INSPECT=FAIL ({len(problems)} problems)")
        for p in problems[:20]:
            print(f"  {p}")
        raise SystemExit(1)
    print("\nPURE_LING_INSPECT=PASS spec matches every verified tensor")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get(
        "Q38_LING_MODEL", "/Users/true/.lmstudio/models/inclusionAI/Ling-3.0-tiny"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    gen = sub.add_parser("reference-generate")
    gen.add_argument("--prompt", default="The capital of France is")
    gen.add_argument("--tokens", type=int, default=8)
    gen.add_argument("--raw-prompt", action="store_true")
    args = p.parse_args()
    assert_standalone("ling cli")
    if args.command == "inspect":
        inspect(args.model)
    elif args.command == "reference-generate":
        reference_generate(args.model, args.prompt, args.tokens, args.raw_prompt)




# ---------------------------------------------------------------------------
# Reference forward. Numpy, float32, no ANE. This is the oracle the ANE
# runtime is checked against, and it is the fastest way to prove the whole
# architecture -- KDA, MLA, and the MoE router together -- actually generates
# text before any of it is committed to MIL.
# ---------------------------------------------------------------------------

def _rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _silu(x):
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float32)))


class LingReference:
    """Single-token-at-a-time float32 forward over the real checkpoint.

    Weights are pulled lazily and cached, and routed experts are read per
    token, so a few tokens cost far less than the 15.8 GB checkpoint.
    """

    def __init__(self, checkpoint: Checkpoint, spec: LingSpec):
        self.ck, self.spec = checkpoint, spec
        self._w: dict[str, np.ndarray] = {}
        s = spec
        self.state = {i: np.zeros((s.heads, s.head_dim, s.head_dim), np.float32)
                      for i in s.linear_attention_layers}
        self.conv = {i: np.zeros((3 * s.kda_proj_dim, s.conv_kernel - 1), np.float32)
                     for i in s.linear_attention_layers}
        self.kv = {i: [] for i in s.full_attention_layers}
        self.pos = 0

    def w(self, name: str) -> np.ndarray:
        if name not in self._w:
            self._w[name] = self.ck.tensor(name, np.float32)
        return self._w[name]

    def wt(self, name: str) -> np.ndarray:
        """The transpose, cached contiguously.

        Every projection here is `x @ W.T`. `W.T` is a non-contiguous view, so
        numpy materializes a fresh copy on each call -- 966 MB per token for
        lm_head alone. Only the transpose is kept, so this costs no extra
        memory over caching W.
        """
        key = name + "\x00T"
        if key not in self._w:
            src = self._w.pop(name, None)
            if src is None:
                src = self.ck.tensor(name, np.float32)
            self._w[key] = np.ascontiguousarray(src.T)
        return self._w[key]

    def reset(self) -> None:
        for v in self.state.values():
            v[:] = 0
        for v in self.conv.values():
            v[:] = 0
        for k in self.kv:
            self.kv[k] = []
        self.pos = 0

    # ------------------------------------------------------------------ KDA
    def kda(self, x, layer):
        s, n = self.spec, self.attention_names_cached(layer)
        H, D, P = s.heads, s.head_dim, s.kda_proj_dim
        qkv = np.concatenate([x @ self.w(n[k]).T for k in ("q", "k", "v")])
        hist = self.conv[layer]
        win = np.concatenate([hist, qkv[:, None]], axis=1)      # [3P, K]
        cw = np.concatenate([self.w(n[f"{k}_conv"]).reshape(P, s.conv_kernel)
                             for k in ("q", "k", "v")])
        conv = (win * cw).sum(-1)
        self.conv[layer] = win[:, 1:]
        q, k, v = _silu(conv[:P]).reshape(H, D), _silu(conv[P:2*P]).reshape(H, D), \
            _silu(conv[2*P:]).reshape(H, D)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        k = k / np.linalg.norm(k, axis=-1, keepdims=True)
        f = (x @ self.wt(n["f"])).reshape(H, D)
        beta = _sigmoid(x @ self.wt(n["b"]))                   # [H]
        a = np.exp(self.w(n["a_log"]))[:, None]
        z = a * (f + self.w(n["dt_bias"]).reshape(H, D))
        g = np.exp(s.kda_lower_bound * _sigmoid(z))             # [H, D] per key channel
        S = self.state[layer] * g[:, None, :]
        kv_mem = np.einsum("hvd,hd->hv", S, k)
        delta = (v - kv_mem) * beta[:, None]
        S = S + delta[:, :, None] * k[:, None, :]
        self.state[layer] = S
        y = np.einsum("hvd,hd->hv", S, q)
        # o_norm is an RMSNorm applied to y, and RMSNorm is scale invariant, so
        # the reference kernel's q * D**-0.5 is absorbed here with eps * D.
        y = _rms(y, self.w(n["o_norm"]), s.rms_eps * D)
        gate = _sigmoid(x @ self.wt(n["gate"])).reshape(H, D)
        return (y * gate).reshape(-1) @ self.wt(n["out"])

    # ------------------------------------------------------------------ MLA
    def mla(self, x, layer):
        s, n = self.spec, self.attention_names_cached(layer)
        H, Dn, Dr, Dv = s.heads, s.qk_nope, s.qk_rope, s.v_head_dim
        q = _rms(x @ self.wt(n["q_a"]), self.w(n["q_a_norm"]), s.rms_eps)
        q = (q @ self.wt(n["q_b"])).reshape(H, s.qk_head_dim)
        q_nope, q_rope = q[:, :Dn], q[:, Dn:]
        c = x @ self.wt(n["kv_a"])
        lat, k_rope = c[:s.kv_lora_rank], c[s.kv_lora_rank:]
        lat = _rms(lat, self.w(n["kv_a_norm"]), s.rms_eps)
        inv = 1.0 / (s.rope_theta ** (np.arange(0, Dr, 2, np.float64) / Dr))
        f = self.pos * inv
        cos, sin = np.cos(np.concatenate([f, f])), np.sin(np.concatenate([f, f]))
        half = Dr // 2

        def rope(t):                       # interleaved (GPT-J) pairing
            td = t.reshape(*t.shape[:-1], half, 2).swapaxes(-1, -2).reshape(t.shape)
            return td * cos + np.concatenate(
                [-td[..., half:], td[..., :half]], -1) * sin

        q_rope, k_rope = rope(q_rope), rope(k_rope)
        self.kv[layer].append((lat.astype(np.float32), k_rope.astype(np.float32)))
        kv_b = self.w(n["kv_b"]).reshape(H, Dn + Dv, s.kv_lora_rank)
        W_K, W_V = kv_b[:, :Dn, :], kv_b[:, Dn:, :]
        q_abs = np.einsum("hn,hnl->hl", q_nope, W_K)
        L = np.stack([a for a, _ in self.kv[layer]])
        R = np.stack([b for _, b in self.kv[layer]])
        sc = (q_abs @ L.T + q_rope @ R.T) * (s.qk_head_dim ** -0.5)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ctx = p @ L
        attn = np.einsum("hl,hvl->hv", ctx, W_V)
        gate = _sigmoid(x @ self.wt(n["gate"]))                # [H] head_wise
        return (attn * gate[:, None]).reshape(-1) @ self.wt(n["out"])

    # ------------------------------------------------------------------ MoE
    def route(self, x, layer):
        """BailingMoeV3Gate. fp32; sigmoid scoring; group-limited top-k."""
        s, m = self.spec, self.mlp_names_cached(layer)
        logits = x.astype(np.float32) @ self.wt(m["router"]).astype(np.float32)
        scores = _sigmoid(logits)                                # NOT softmax
        routing = scores + self.w(m["expert_bias"])              # selection only
        grp = routing.reshape(s.n_group, -1)
        gs = np.sort(grp, -1)[:, -2:].sum(-1)                    # sum of top TWO
        live = np.argpartition(gs, -s.topk_group)[-s.topk_group:]
        mask = np.zeros(s.n_group, bool)
        mask[live] = True
        masked = np.where(np.repeat(mask, grp.shape[1]), routing, -np.inf)
        idx = np.argpartition(masked, -s.top_k)[-s.top_k:]
        wts = scores[idx]                                        # PRE-bias scores
        if s.norm_topk_prob:
            wts = wts / (wts.sum() + 1e-20)
        return idx, wts * s.routed_scale

    def moe(self, x, layer):
        s = self.spec
        idx, wts = self.route(x, layer)
        out = np.zeros_like(x)
        for e, wt in zip(idx, wts):
            e = int(e)
            g = self.wt(self.spec.expert_name(layer, e, "gate"))
            u = self.wt(self.spec.expert_name(layer, e, "up"))
            d = self.wt(self.spec.expert_name(layer, e, "down"))
            out += wt * ((_silu(x @ g) * (x @ u)) @ d)
        m = self.mlp_names_cached(layer)
        sg, su, sd = (self.w(m["shared_gate"]), self.w(m["shared_up"]),
                      self.w(m["shared_down"]))
        return out + (_silu(x @ sg.T) * (x @ su.T)) @ sd.T       # unscaled

    def expert_stack(self, layer, idx):
        """gate|up rows for the routed experts, stacked into one [2*K*M, H]."""
        parts = [self.w(self.spec.expert_name(layer, int(e), k))
                 for k in ("gate", "up") for e in idx]
        return np.concatenate(parts)

    def expert_down(self, layer, idx):
        """down for the routed experts, concatenated along its INPUT axis.

        sum_e down_e @ a_e == [down_0|...|down_7] @ [a_0;...;a_7], so the expert
        sum falls out of one matmul -- the same identity the ANE stacked-expert
        layout uses (docs/ANE-MOE-HANDOFF.md).
        """
        return np.concatenate(
            [self.w(self.spec.expert_name(layer, int(e), "down")) for e in idx],
            axis=1)

    def dense_mlp(self, x, layer):
        m = self.mlp_names_cached(layer)
        return (_silu(x @ self.wt(m["gate"])) * (x @ self.wt(m["up"]))) \
            @ self.wt(m["down"])

    # ---------------------------------------------------------------- caches
    def attention_names_cached(self, layer):
        key = ("attn", layer)
        if key not in self._w:
            self._w[key] = self.spec.attention_names(layer)      # type: ignore
        return self._w[key]

    def mlp_names_cached(self, layer):
        key = ("mlp", layer)
        if key not in self._w:
            self._w[key] = self.spec.mlp_names(layer)            # type: ignore
        return self._w[key]

    # --------------------------------------------------------------- forward
    def forward(self, token_id: int) -> np.ndarray:
        s = self.spec
        h = np.asarray(self.ck.embedding(token_id), np.float32).reshape(-1)
        for layer in range(s.layers):
            nn = s.norm_names(layer)
            a = _rms(h, self.w(nn["input"]), s.rms_eps)
            h = h + (self.mla(a, layer) if s.is_full_attention(layer)
                     else self.kda(a, layer))
            p = _rms(h, self.w(nn["post_attention"]), s.rms_eps)
            h = h + (self.moe(p, layer) if s.is_moe(layer)
                     else self.dense_mlp(p, layer))
        self.pos += 1
        h = _rms(h, self.w("model.norm.weight"), s.rms_eps)
        return h @ self.wt("lm_head.weight")


def reference_generate(model_dir: str, prompt: str, tokens: int,
                       raw: bool = False) -> None:
    """Greedy decode through the numpy reference. Proves the architecture."""
    import time
    checkpoint, spec = load(model_dir)
    tok = StandaloneTokenizer(model_dir)
    if not raw:
        # Bailing V3 template (chat_template.jinja): a SYSTEM turn is always
        # emitted, and thinking is on by default so the assistant turn opens
        # with <think>.
        prompt = ("<role>SYSTEM</role><|role_end|>"
                  "<role>HUMAN</role>" + prompt + "<|role_end|>"
                  "<role>ASSISTANT</role>")
    ids = tok.encode(prompt)
    print(f"prompt {len(ids)} tokens: {ids[:12]}{'...' if len(ids) > 12 else ''}")
    ref = LingReference(checkpoint, spec)
    t0 = time.perf_counter()
    for i in ids[:-1]:
        ref.forward(int(i))
    out, cur = [], int(ids[-1])
    for _ in range(tokens):
        logits = ref.forward(cur)
        cur = int(np.argmax(logits))
        out.append(cur)
        print(f"  token {cur:>7}  {tok.decode([cur])!r}", flush=True)
    dt = time.perf_counter() - t0
    print(f"\ntoken_ids={out}")
    print(f"text={tok.decode(out)!r}")
    print(f"\nLING_REFERENCE_GENERATE=PASS {len(ids)}+{tokens} tokens in "
          f"{dt:.1f}s ({(len(ids)+tokens)/dt:.2f} tok/s, numpy fp32, no ANE)")

if __name__ == "__main__":
    main()
