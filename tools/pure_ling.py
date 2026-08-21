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


# ---------------------------------------------------------------------------
# ANE runtime. Blocks are swapped in one at a time and each is checked against
# LingReference, so the model keeps generating correct text throughout.
# ---------------------------------------------------------------------------

def _bank(driver, checkpoint, sets, bits, tag, width=32):
    """Build one projection bank and drop its compiler scratch immediately.

    AneLinearProjectionBank does not call discard_compiler_files, so every bank
    leaves a materialized model directory behind -- 36 GB across one Ling bake,
    enough to fill the volume. The loaded _ANEInMemoryModel owns its compiled
    representation, so the scratch is rebuildable and safe to drop.
    """
    from pure_ane import AneLinearProjectionBank
    bank = AneLinearProjectionBank(driver, checkpoint, sets, bits, tag, width)
    driver.discard_compiler_files(bank.program)
    return bank


class AneExpertBank:
    """All experts of every MoE layer, baked as int4 constants.

    gate|up stacks along OUTPUT rows, which is exactly what
    AneLinearProjectionBank packs, so one bank holds a chunk of experts for all
    23 layers as procedures. 128 experts x 1024 rows is 131072, above the
    measured 62080 single-conv output limit, hence `chunks`.

    Row-wise quantization is what makes the stacking exact: each output row
    keeps its own scale and no partial sum crosses an expert boundary
    (docs/ANE-REFERENCE.md).
    """

    def __init__(self, driver, checkpoint, spec, bits=4, chunks=None, width=32):
        from pure_ane import AneLinearProjectionBank
        # One chunk per router group. `topk_group` of `n_group` groups are live
        # per token, so only those chunks need dispatching -- the model's own
        # group-limited routing halves the expert traffic for free.
        chunks = chunks or spec.n_group
        self.spec, self.chunks = spec, chunks
        per = spec.experts // chunks
        self.per_chunk_rows = per * 2 * spec.moe_intermediate
        self.banks = []
        for c in range(chunks):
            lo, hi = c * per, (c + 1) * per
            sets = []
            for layer in spec.moe_layers:
                # gate rows for experts lo..hi, then up rows: the activation is
                # sliced apart in the same order on read.
                sets.append(
                    [spec.expert_name(layer, e, "gate") for e in range(lo, hi)]
                    + [spec.expert_name(layer, e, "up") for e in range(lo, hi)])
            self.banks.append(_bank(driver, checkpoint, sets, bits,
                                    f"moe_gu_c{c}", width))
        self.nbytes = sum(b.nbytes for b in self.banks)

    def gate_up_batch(self, layer, X, groups=None):
        """gate|up for a whole chunk of positions: one dispatch per live group."""
        s = self.spec
        idx = s.moe_layers.index(layer)
        per = s.experts // self.chunks
        M, T = s.moe_intermediate, X.shape[0]
        out = np.zeros((T, s.experts, 2, M), np.float32)
        xh = X.T.astype(np.float16)
        for c in (range(self.chunks) if groups is None else sorted(groups)):
            r = _cols(self.banks[c].run(idx, xh)).astype(np.float32)  # [2*per*M, T]
            out[:, c*per:(c+1)*per, 0] = r[:per*M].reshape(per, M, T).transpose(2, 0, 1)
            out[:, c*per:(c+1)*per, 1] = r[per*M:].reshape(per, M, T).transpose(2, 0, 1)
        return out

    def gate_up(self, layer, x, groups=None):
        """gate|up for the routed experts only.

        `groups` is the set of chunk indices actually needed; with chunks
        aligned to the router's groups that is `topk_group` of `n_group`, so
        half the chunks are skipped entirely.
        """
        s = self.spec
        idx = s.moe_layers.index(layer)
        per = s.experts // self.chunks
        M = s.moe_intermediate
        out = np.zeros((s.experts, 2, M), np.float32)
        xh = x.astype(np.float16)
        for c in (range(self.chunks) if groups is None else sorted(groups)):
            r = self.banks[c].run(idx, xh).astype(np.float32)
            out[c * per:(c + 1) * per, 0] = r[:per * M].reshape(per, M)
            out[c * per:(c + 1) * per, 1] = r[per * M:].reshape(per, M)
        return out


class LingRuntime(LingReference):
    """LingReference with blocks moved onto the ANE.

    Subclassing the reference keeps every un-ported block exact, so the model
    generates correct text at every stage and each swap can be diffed against
    the parent implementation.
    """

    def __init__(self, checkpoint, spec, engine_path=".", bits=8,
                 ane_moe=True, ane_proj=True, ane_down=True, width=32):
        super().__init__(checkpoint, spec)
        from pure_ane import AneDriver
        self.driver = AneDriver(engine_path)
        self.bits = bits
        self.width = width
        self.experts = (AneExpertBank(self.driver, checkpoint, spec, bits,
                                     width=width)
                        if ane_moe else None)
        self.proj = (AneProjectionBanks(self.driver, checkpoint, spec, bits, width)
                     if ane_proj else None)
        self.down = (AneExpertDown(self.driver, checkpoint, spec, bits, width)
                     if ane_down else None)
        self.programs = (
            (len(self.experts.banks) if self.experts else 0)
            + (self.proj.programs if self.proj else 0)
            + (self.down.programs if self.down else 0))
        self.nbytes = ((self.experts.nbytes if self.experts else 0)
                       + (self.proj.nbytes if self.proj else 0)
                       + (self.down.nbytes if self.down else 0))

    # ------------------------------------------------------------------ KDA
    def kda(self, x, layer):
        if self.proj is None:
            return super().kda(x, layer)
        s, n = self.spec, self.attention_names_cached(layer)
        H, D, P = s.heads, s.head_dim, s.kda_proj_dim
        i = s.linear_attention_layers.index(layer)
        fused = self.proj.kda_in.run(i, x.astype(np.float16)).astype(np.float32)
        q_r, k_r, v_r = fused[:P], fused[P:2*P], fused[2*P:3*P]
        f, gate_r, beta_r = fused[3*P:4*P], fused[4*P:5*P], fused[5*P:5*P+H]

        hist = self.conv[layer]
        win = np.concatenate([hist, np.concatenate([q_r, k_r, v_r])[:, None]], 1)
        cw = np.concatenate([self.w(n[f"{k}_conv"]).reshape(P, s.conv_kernel)
                             for k in ("q", "k", "v")])
        conv = (win * cw).sum(-1)
        self.conv[layer] = win[:, 1:]
        q, k, v = (_silu(conv[:P]).reshape(H, D), _silu(conv[P:2*P]).reshape(H, D),
                   _silu(conv[2*P:]).reshape(H, D))
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        k = k / np.linalg.norm(k, axis=-1, keepdims=True)
        beta = _sigmoid(beta_r)
        a = np.exp(self.w(n["a_log"]))[:, None]
        g = np.exp(s.kda_lower_bound
                   * _sigmoid(a * (f.reshape(H, D) + self.w(n["dt_bias"]).reshape(H, D))))
        S = self.state[layer] * g[:, None, :]
        delta = (v - np.einsum("hvd,hd->hv", S, k)) * beta[:, None]
        S = S + delta[:, :, None] * k[:, None, :]
        self.state[layer] = S
        y = _rms(np.einsum("hvd,hd->hv", S, q), self.w(n["o_norm"]), s.rms_eps * D)
        y = (y * _sigmoid(gate_r).reshape(H, D)).reshape(-1)
        return self.proj.kda_out.run(i, y.astype(np.float16)).astype(np.float32)

    # ------------------------------------------------------------------ MLA
    def mla(self, x, layer):
        if self.proj is None:
            return super().mla(x, layer)
        s, n = self.spec, self.attention_names_cached(layer)
        H, Dn, Dr, Dv = s.heads, s.qk_nope, s.qk_rope, s.v_head_dim
        i = s.full_attention_layers.index(layer)
        fused = self.proj.mla_a.run(i, x.astype(np.float16)).astype(np.float32)
        qa, kva = fused[:s.q_lora_rank], fused[s.q_lora_rank:s.q_lora_rank + s.kv_lora_rank + Dr]
        gate = _sigmoid(fused[s.q_lora_rank + s.kv_lora_rank + Dr:])
        qa = _rms(qa, self.w(n["q_a_norm"]), s.rms_eps)
        q = self.proj.mla_qb.run(i, qa.astype(np.float16)).astype(np.float32)
        q = q.reshape(H, s.qk_head_dim)
        q_nope, q_rope = q[:, :Dn], q[:, Dn:]
        lat, k_rope = kva[:s.kv_lora_rank], kva[s.kv_lora_rank:]
        lat = _rms(lat, self.w(n["kv_a_norm"]), s.rms_eps)
        inv = 1.0 / (s.rope_theta ** (np.arange(0, Dr, 2, np.float64) / Dr))
        fq = self.pos * inv
        cos, sin = np.cos(np.concatenate([fq, fq])), np.sin(np.concatenate([fq, fq]))
        half = Dr // 2

        def rope(tt):
            td = tt.reshape(*tt.shape[:-1], half, 2).swapaxes(-1, -2).reshape(tt.shape)
            return td * cos + np.concatenate([-td[..., half:], td[..., :half]], -1) * sin

        q_rope, k_rope = rope(q_rope), rope(k_rope)
        self.kv[layer].append((lat.astype(np.float32), k_rope.astype(np.float32)))
        kv_b = self.w(n["kv_b"]).reshape(H, Dn + Dv, s.kv_lora_rank)
        W_K, W_V = kv_b[:, :Dn, :], kv_b[:, Dn:, :]
        q_abs = np.einsum("hn,hnl->hl", q_nope, W_K)
        L = np.stack([a for a, _ in self.kv[layer]])
        R = np.stack([b for _, b in self.kv[layer]])
        sc = (q_abs @ L.T + q_rope @ R.T) * (s.qk_head_dim ** -0.5)
        pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
        attn = np.einsum("hl,hvl->hv", pr @ L, W_V)
        y = (attn * gate[:, None]).reshape(-1)
        return self.proj.mla_out.run(i, y.astype(np.float16)).astype(np.float32)

    # ------------------------------------------------------------------ MoE
    def moe(self, x, layer):
        if self.experts is None:
            return super().moe(x, layer)
        s = self.spec
        idx, wts = self.route(x, layer)
        per = s.experts // self.experts.chunks
        gu = self.experts.gate_up(layer, x, {int(e) // per for e in idx})
        h = _silu(gu[idx, 0]) * gu[idx, 1]
        h *= wts[:, None]
        out = np.zeros_like(x)
        for j, e in enumerate(idx):
            if self.down is not None:
                out += self.down.run(layer, e, h[j]).astype(np.float32)
            else:
                out += h[j] @ self.wt(self.spec.expert_name(layer, int(e), "down"))
        if self.proj is None:
            m = self.mlp_names_cached(layer)
            return out + (_silu(x @ self.wt(m["shared_gate"]))
                          * (x @ self.wt(m["shared_up"]))) @ self.wt(m["shared_down"])
        si = s.moe_layers.index(layer)
        sgu = self.proj.shared_gu.run(si, x.astype(np.float16)).astype(np.float32)
        M = s.moe_intermediate * s.shared_experts
        sh = _silu(sgu[:M]) * sgu[M:]
        return out + self.proj.shared_dn.run(si, sh.astype(np.float16)).astype(np.float32)


class AneProjectionBanks:
    """Every dense projection in the model, as procedure banks.

    All of these are plain `x @ W.T` with a shared input dimension, so
    AneLinearProjectionBank packs them directly: one program per block class,
    one procedure per layer. No new MIL.
    """

    def __init__(self, driver, checkpoint, spec, bits=8, width=32):
        B = lambda d, c, s, b, tg: _bank(d, c, s, b, tg, width)
        self.spec = spec
        kda, mla = spec.linear_attention_layers, spec.full_attention_layers
        an = spec.attention_names

        # KDA: q|k|v|f|g|b all project from hidden, so they fuse into one conv.
        self.kda_in = B(driver, checkpoint,
                        [[an(l)[k] for k in ("q", "k", "v", "f", "gate", "b")]
                         for l in kda], bits, "kda_in")
        self.kda_out = B(driver, checkpoint, [[an(l)["out"]] for l in kda],
                         bits, "kda_out")
        # MLA: q_a|kv_a|g share the hidden input; q_b and dense do not.
        self.mla_a = B(driver, checkpoint,
                       [[an(l)[k] for k in ("q_a", "kv_a", "gate")] for l in mla],
                       bits, "mla_a")
        self.mla_qb = B(driver, checkpoint, [[an(l)["q_b"]] for l in mla],
                        bits, "mla_qb")
        self.mla_out = B(driver, checkpoint, [[an(l)["out"]] for l in mla],
                         bits, "mla_out")
        # shared expert, one procedure per MoE layer
        mn = spec.mlp_names
        self.shared_gu = B(driver, checkpoint,
                           [[mn(l)["shared_gate"], mn(l)["shared_up"]]
                            for l in spec.moe_layers], bits, "shared_gu")
        self.shared_dn = B(driver, checkpoint,
                           [[mn(l)["shared_down"]] for l in spec.moe_layers],
                           bits, "shared_dn")
        dense = [l for l in range(spec.layers) if not spec.is_moe(l)]
        self.dense_gu = B(driver, checkpoint,
                          [[mn(l)["gate"], mn(l)["up"]] for l in dense],
                          bits, "dense_gu")
        self.dense_dn = B(driver, checkpoint, [[mn(l)["down"]] for l in dense],
                          bits, "dense_dn")
        self.dense_layers = dense
        # lm_head: 157184 rows is far above the 62080 single-conv limit, so it
        # is split. It was still in numpy at 966 MB of fp32 per call.
        V = spec.vocab
        self.head_chunks = 4
        step = -(-V // self.head_chunks)
        self.head = []
        self.head_spans = []
        for c in range(self.head_chunks):
            lo, hi = c * step, min(V, (c + 1) * step)
            self.head_spans.append((lo, hi))
        self.banks = [self.kda_in, self.kda_out, self.mla_a, self.mla_qb,
                      self.mla_out, self.shared_gu, self.shared_dn,
                      self.dense_gu, self.dense_dn]
        self.nbytes = sum(b.nbytes for b in self.banks)
        self.programs = len(self.banks)


class AneVocabHead:
    """Final vocabulary projection, split under the single-conv output limit."""

    def __init__(self, driver, checkpoint, spec, bits=8, chunks=4, width=32):
        from pure_ane import _quantize_matrix, _dense_decl
        import contextlib, io
        self.driver, self.spec, self.width = driver, spec, width
        E = driver.module
        V, H = spec.vocab, spec.hidden
        step = -(-V // chunks)
        self.progs, self.spans, self.nbytes = [], [], 0
        for c in range(chunks):
            lo, hi = c * step, min(V, (c + 1) * step)
            blobs = _quantize_matrix(checkpoint, "lm_head.weight", "p", bits,
                                     row_start=lo, row_end=hi)
            O = hi - lo
            mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {H}, 1, {width}]> x) {{
    string pt=const()[name=string("pt"),val=string("valid")];
    tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])];
    tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,0,0])];
    tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])];
    int32 gr=const()[name=string("gr"),val=int32(1)];
{_dense_decl("p", O, H, bits)}
    tensor<fp16,[1,{O},1,{width}]> y=conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=pw,x=x)[name=string("mm")];
  }} -> (y);
}}
'''
            cap = io.StringIO()
            with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
                prog = driver.engine.compile_multiproc(mil, blobs, H, O, width)
            if prog is None:
                raise RuntimeError(f"vocab head chunk {c} failed:\n"
                                   f"{cap.getvalue().strip()[-200:]}")
            driver.engine._ensure_io(prog)
            driver.discard_compiler_files(prog)
            self.progs.append((prog, O))
            self.spans.append((lo, hi))
            self.nbytes += sum(len(v) for v in blobs.values())
        self.programs = len(self.progs)

    def run(self, x):
        out = np.empty(self.spec.vocab, np.float32)
        xh = np.asarray(x, np.float16)
        for (prog, O), (lo, hi) in zip(self.progs, self.spans):
            with self.driver.view(prog._in_surf, (self.spec.hidden, self.width),
                                  np.float16) as d:
                d[:] = 0; d[:, 0] = xh
            if not self.driver.engine.submit(prog, procedure_index=0):
                raise RuntimeError("vocab head submit failed")
            with self.driver.view(prog._out_surf, (O, self.width),
                                  np.float16) as o:
                out[lo:hi] = np.array(o[:hi - lo, 0], np.float32)
        return out


class AneExpertDown:
    """Per-expert `down`, banked per MoE layer.

    `down` cannot join the stacked form: gate|up concatenate along output rows,
    but down concatenates along its INPUT axis and each expert consumes a
    different slice of the activation. So it is dispatched per routed expert --
    8 per layer -- with the layer's experts as procedures.

    Measured: a bank of 64 procedures compiles, 128 does not, so each layer is
    split in two. That limit is not the 16-blob rule (the bank packs one blob)
    nor the 127-program rule; it is a distinct procedure-count ceiling.
    """

    PER_BANK = 64

    def __init__(self, driver, checkpoint, spec, bits=8, width=32):
        B = lambda d, c, s, b, tg: _bank(d, c, s, b, tg, width)
        self.spec = spec
        self.split = spec.experts // self.PER_BANK
        self.banks = {}
        for layer in spec.moe_layers:
            for c in range(self.split):
                lo = c * self.PER_BANK
                self.banks[(layer, c)] = B(
                    driver, checkpoint,
                    [[spec.expert_name(layer, e, "down")]
                     for e in range(lo, lo + self.PER_BANK)],
                    bits, f"down_l{layer}_c{c}")
        self.nbytes = sum(b.nbytes for b in self.banks.values())
        self.programs = len(self.banks)

    def run(self, layer, expert, h):
        e = int(expert)
        bank = self.banks[(layer, e // self.PER_BANK)]
        return bank.run(e % self.PER_BANK, h.astype(np.float16))


# ---------------------------------------------------------------------------
# Batched prefill. Every projection is position-independent and the ANE pads
# each dispatch to 32 lanes anyway, so a prompt costs almost the same as one
# token for the weight-heavy blocks. Only the KDA recurrence is sequential.
# ---------------------------------------------------------------------------

def _rms_rows(X, w, eps):
    return X / np.sqrt((X * X).mean(-1, keepdims=True) + eps) * w


def _cols(r):
    """Bank output as [out, lanes].

    AneLinearProjectionBank drops the lane axis when there is exactly one lane,
    which a trailing partial chunk always hits.
    """
    return r[:, None] if r.ndim == 1 else r


class LingPrefill(LingRuntime):
    """LingRuntime with a batched prompt path.

    `AneLinearProjectionBank` already accepts a [hidden, lanes] matrix; it just
    defaults to 3 active lanes. Raising that to the program width turns every
    projection into one dispatch for the whole chunk.

    What cannot batch: the KDA recurrence carries state across positions, so it
    still advances one at a time. MLA attention batches with a causal mask.
    """

    def __init__(self, *a, lanes=None, ane_absorb=True, **k):
        super().__init__(*a, **k)
        self.lanes = lanes or self.width
        self.absorb = (AneMlaAbsorb(self.driver, self.ck, self.spec, self.width)
                       if ane_absorb else None)
        if self.absorb:
            self.programs += self.absorb.programs
        self.head = AneVocabHead(self.driver, self.ck, self.spec,
                                 self.bits, 4, self.width)
        self.programs += self.head.programs
        self.nbytes += self.head.nbytes
        for bank in (self.proj.banks if self.proj else []):
            bank.active_lanes = self.lanes
        for bank in (self.experts.banks if self.experts else []):
            bank.active_lanes = self.lanes
        for bank in (self.down.banks.values() if self.down else []):
            bank.active_lanes = self.lanes

    def kda_batch(self, A, layer):
        s, n = self.spec, self.attention_names_cached(layer)
        H, D, P = s.heads, s.head_dim, s.kda_proj_dim
        T = A.shape[0]
        i = s.linear_attention_layers.index(layer)
        f = _cols(self.proj.kda_in.run(i, A.T.astype(np.float16))).astype(np.float32)
        qkv = f[:3 * P]                                    # [3P, T]
        fl, gate_r, beta_r = f[3*P:4*P], f[4*P:5*P], f[5*P:5*P+H]

        # causal depthwise conv over the chunk, seeded with the carried history
        cw = np.concatenate([self.w(n[f"{k}_conv"]).reshape(P, s.conv_kernel)
                             for k in ("q", "k", "v")])
        padded = np.concatenate([self.conv[layer], qkv], axis=1)
        conv = sum(padded[:, t:t + T] * cw[:, t:t + 1]
                   for t in range(s.conv_kernel))
        self.conv[layer] = padded[:, -(s.conv_kernel - 1):]
        q = _silu(conv[:P]).T.reshape(T, H, D)
        k = _silu(conv[P:2*P]).T.reshape(T, H, D)
        v = _silu(conv[2*P:]).T.reshape(T, H, D)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        beta = _sigmoid(beta_r).T
        a = np.exp(self.w(n["a_log"]))[None, :, None]
        g = np.exp(s.kda_lower_bound * _sigmoid(
            a * (fl.T.reshape(T, H, D) + self.w(n["dt_bias"]).reshape(H, D))))

        S = self.state[layer]
        Y = np.empty((T, H, D), np.float32)
        for t in range(T):                                  # the one serial part
            S = S * g[t][:, None, :]
            delta = (v[t] - np.einsum("hvd,hd->hv", S, k[t])) * beta[t][:, None]
            S = S + delta[:, :, None] * k[t][:, None, :]
            Y[t] = np.einsum("hvd,hd->hv", S, q[t])
        self.state[layer] = S
        Y = _rms_rows(Y, self.w(n["o_norm"]), s.rms_eps * D)
        Y = (Y * _sigmoid(gate_r).T.reshape(T, H, D)).reshape(T, -1)
        return _cols(self.proj.kda_out.run(i, Y.T.astype(np.float16))).astype(np.float32).T

    def mla_batch(self, A, layer):
        s, n = self.spec, self.attention_names_cached(layer)
        H, Dn, Dr, Dv = s.heads, s.qk_nope, s.qk_rope, s.v_head_dim
        T = A.shape[0]
        i = s.full_attention_layers.index(layer)
        f = _cols(self.proj.mla_a.run(i, A.T.astype(np.float16))).astype(np.float32).T
        qa = _rms_rows(f[:, :s.q_lora_rank], self.w(n["q_a_norm"]), s.rms_eps)
        kva = f[:, s.q_lora_rank:s.q_lora_rank + s.kv_lora_rank + Dr]
        gate = _sigmoid(f[:, s.q_lora_rank + s.kv_lora_rank + Dr:])
        q = _cols(self.proj.mla_qb.run(i, qa.T.astype(np.float16))).astype(np.float32).T
        q = q.reshape(T, H, s.qk_head_dim)
        q_nope, q_rope = q[..., :Dn], q[..., Dn:]
        lat = _rms_rows(kva[:, :s.kv_lora_rank], self.w(n["kv_a_norm"]), s.rms_eps)
        k_rope = kva[:, s.kv_lora_rank:]
        pos = np.arange(self.pos, self.pos + T, dtype=np.float64)
        inv = 1.0 / (s.rope_theta ** (np.arange(0, Dr, 2, np.float64) / Dr))
        fr = pos[:, None] * inv[None, :]
        cos = np.cos(np.concatenate([fr, fr], -1))
        sin = np.sin(np.concatenate([fr, fr], -1))
        half = Dr // 2

        def rope(x, c, sn):
            xd = x.reshape(*x.shape[:-1], half, 2).swapaxes(-1, -2).reshape(x.shape)
            return xd * c + np.concatenate([-xd[..., half:], xd[..., :half]], -1) * sn

        q_rope = rope(q_rope, cos[:, None, :], sin[:, None, :])
        k_rope = rope(k_rope, cos, sin)
        for t in range(T):
            self.kv[layer].append((lat[t].copy(), k_rope[t].copy()))
        if self.absorb is not None:
            q_abs = self.absorb.absorb(layer, q_nope)
        else:
            kv_b = self.w(n["kv_b"]).reshape(H, Dn + Dv, s.kv_lora_rank)
            q_abs = np.einsum("thn,hnl->thl", q_nope, kv_b[:, :Dn, :])
        L = np.stack([x for x, _ in self.kv[layer]])
        R = np.stack([x for _, x in self.kv[layer]])
        sc = (np.einsum("thl,sl->hts", q_abs, L)
              + np.einsum("thr,sr->hts", q_rope, R)) * (s.qk_head_dim ** -0.5)
        past = L.shape[0] - T
        mask = np.arange(L.shape[0])[None, :] <= (np.arange(T)[:, None] + past)
        sc = np.where(mask[None], sc, -np.inf)
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        ctx = np.einsum("hts,sl->thl", p, L)
        if self.absorb is not None:
            attn = self.absorb.unabsorb(layer, ctx)
        else:
            kv_b = self.w(n["kv_b"]).reshape(H, Dn + Dv, s.kv_lora_rank)
            attn = np.einsum("thl,hvl->thv", ctx, kv_b[:, Dn:, :])
        Y = (attn * gate[:, :, None]).reshape(T, -1)
        return _cols(self.proj.mla_out.run(i, Y.T.astype(np.float16))).astype(np.float32).T

    def moe_batch(self, X, layer):
        s = self.spec
        T = X.shape[0]
        per = s.experts // self.experts.chunks
        routes = [self.route(X[t], layer) for t in range(T)]
        groups = {int(e) // per for idx, _ in routes for e in idx}
        gu = self.experts.gate_up_batch(layer, X, groups)     # [T, E, 2, M]
        out = np.zeros_like(X)
        # expert-major: one dispatch per (expert, chunk of its positions)
        assign: dict[int, list[int]] = {}
        weights: dict[int, list[float]] = {}
        for t, (idx, wts) in enumerate(routes):
            for e, wt in zip(idx, wts):
                assign.setdefault(int(e), []).append(t)
                weights.setdefault(int(e), []).append(float(wt))
        M = s.moe_intermediate
        for e, ts in assign.items():
            h = (_silu(gu[ts, e, 0]) * gu[ts, e, 1]
                 * np.array(weights[e], np.float32)[:, None])
            r = _cols(self.down.run(layer, e, h.T)).astype(np.float32).T
            out[ts] += r.reshape(len(ts), -1)
        si = s.moe_layers.index(layer)
        sgu = _cols(self.proj.shared_gu.run(si, X.T.astype(np.float16))).astype(np.float32).T
        Ms = M * s.shared_experts
        sh = _silu(sgu[:, :Ms]) * sgu[:, Ms:]
        return out + _cols(self.proj.shared_dn.run(
            si, sh.T.astype(np.float16))).astype(np.float32).T

    def dense_batch(self, X, layer):
        s, pj = self.spec, self.proj
        i = pj.dense_layers.index(layer)
        gu = _cols(pj.dense_gu.run(i, X.T.astype(np.float16))).astype(np.float32).T
        I = s.intermediate
        h = _silu(gu[:, :I]) * gu[:, I:]
        return _cols(pj.dense_dn.run(i, h.T.astype(np.float16))).astype(np.float32).T

    def prefill(self, ids, chunk=None):
        """Run a prompt in chunks, returning the final logits."""
        s = self.spec
        chunk = chunk or self.lanes
        logits = None
        for a in range(0, len(ids), chunk):
            batch = ids[a:a + chunk]
            X = np.stack([np.asarray(self.ck.embedding(int(t)), np.float32)
                          for t in batch])
            for layer in range(s.layers):
                nn = s.norm_names(layer)
                A = _rms_rows(X, self.w(nn["input"]), s.rms_eps)
                X = X + (self.mla_batch(A, layer) if s.is_full_attention(layer)
                         else self.kda_batch(A, layer))
                Pn = _rms_rows(X, self.w(nn["post_attention"]), s.rms_eps)
                if s.is_moe(layer):
                    X = X + self.moe_batch(Pn, layer)
                else:
                    X = X + self.dense_batch(Pn, layer)
            self.pos += len(batch)
            last = X[-1]
        # logits are only needed for the final position: projecting every
        # chunk's last row read 966 MB of fp32 lm_head each time.
        last = _rms_rows(last[None], self.w("model.norm.weight"), s.rms_eps)[0]
        if self.head is not None:
            return self.head.run(last)
        return last @ self.wt("lm_head.weight")


class AneMlaAbsorb:
    """The absorbed MLA maps as grouped convolutions, one program per direction.

    q_abs[h] = q_nope[h] @ W_K[h]  ([128]->[512])
    out[h]   = W_V[h] @ ctx[h]     ([512]->[128])

    Both are block-diagonal over the heads, so `groups=H`. Validated in
    probes/ane_ling_mla_absorb.py at fp16 rel 2.95e-04 and 2.38e-03, with a
    head-isolation check proving the blocks really are diagonal.

    These were 12% of prefill while still in numpy: 737 ms of einsum against
    roughly 4 ms of ANE work once batched.
    """

    CONST = ('    string pt=const()[name=string("pt"),val=string("valid")];\n'
             '    tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])];\n'
             '    tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,0,0])];\n'
             '    tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])];')

    def __init__(self, driver, checkpoint, spec, width=32):
        import contextlib, io
        self.driver, self.spec, self.width = driver, spec, width
        E = driver.module
        H, Dn, Dv, L = spec.heads, spec.qk_nope, spec.v_head_dim, spec.kv_lora_rank
        self.k_progs, self.v_progs = {}, {}
        for layer in spec.full_attention_layers:
            kv_b = checkpoint.tensor(spec.attention_names(layer)["kv_b"], np.float32)
            kv_b = kv_b.reshape(H, Dn + Dv, L)
            for tag, w, ip, op, store in (
                    ("k", kv_b[:, :Dn, :].transpose(0, 2, 1), Dn, L, self.k_progs),
                    ("v", kv_b[:, Dn:, :], L, Dv, self.v_progs)):
                cin, cout = H * ip, H * op
                blobs = {"w.bin": np.ascontiguousarray(
                    w.reshape(cout, ip)).astype(np.float16).tobytes()}
                mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {cin}, 1, {width}]> x) {{
{self.CONST}
    int32 gh=const()[name=string("gh"),val=int32({H})];
    tensor<fp16, [{cout}, {ip}, 1, 1]> ww = const()[name=string("ww"), val=tensor<fp16, [{cout}, {ip}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<fp16,[1,{cout},1,{width}]> y=conv(dilations=dl,groups=gh,pad=pd,pad_type=pt,strides=st,weight=ww,x=x)[name=string("mm")];
  }} -> (y);
}}
'''
                cap = io.StringIO()
                with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
                    p = driver.engine.compile_multiproc(mil, blobs, cin, cout, width)
                if p is None:
                    raise RuntimeError(f"mla absorb {tag} layer {layer} failed:\n"
                                       f"{cap.getvalue().strip()[-200:]}")
                driver.engine._ensure_io(p)
                driver.discard_compiler_files(p)
                store[layer] = (p, cin, cout)
        self.programs = len(self.k_progs) + len(self.v_progs)

    def _run(self, prog, X):
        """X is [T, cin] -> [T, cout], chunked to the program width."""
        p, cin, cout = prog
        T = X.shape[0]
        out = np.empty((T, cout), np.float32)
        for a in range(0, T, self.width):
            blk = X[a:a + self.width]
            with self.driver.view(p._in_surf, (cin, self.width), np.float16) as d:
                d[:] = 0
                d[:, :len(blk)] = blk.T.astype(np.float16)
            if not self.driver.engine.submit(p, procedure_index=0):
                raise RuntimeError("mla absorb submit failed")
            with self.driver.view(p._out_surf, (cout, self.width), np.float16) as o:
                out[a:a + len(blk)] = np.array(o[:, :len(blk)], np.float32).T
        return out

    def absorb(self, layer, q_nope):
        s = self.spec
        T = q_nope.shape[0]
        r = self._run(self.k_progs[layer], q_nope.reshape(T, -1))
        return r.reshape(T, s.heads, s.kv_lora_rank)

    def unabsorb(self, layer, ctx):
        s = self.spec
        T = ctx.shape[0]
        r = self._run(self.v_progs[layer], ctx.reshape(T, -1))
        return r.reshape(T, s.heads, s.v_head_dim)
