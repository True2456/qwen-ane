#!/usr/bin/env python3
# Is the mixed-bit AWQ MLX build (truemod/Qwen3.8-27B-AWQ-gs64-mm) usable as a
# source for the engine's ANE per-channel int4 path?
# Compares, on real layer-20 MLP weights of Qwen3.8-27B:
#   awq_asis        - the AWQ model's own error vs bf16 (what GPU serving gets)
#   chain_clip/rtm  - AWQ dequantised -> ANE int4 (lossy requant chain)
#   direct_clip/rtn - bf16 -> ANE int4 (the right way)
import json
import struct

import numpy as np

BF16_DIR = str(Path.home() / "models/Qwen3.8-27B")
AWQ_DIR = str(Path.home() / ".lmstudio/models/truemod/Qwen3.8-27B-AWQ-gs64-mm")
LAYER = 20


def read_st(path, name):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    meta = header[name]
    begin, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + hlen + begin)
        raw = f.read(end - begin)
    dt = meta["dtype"]
    shape = meta["shape"]
    if dt == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (bits << 16).view(np.float32).reshape(shape)
    if dt == "U32":
        return np.frombuffer(raw, dtype=np.uint32).reshape(shape)
    if dt == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    raise ValueError(dt)


def bf16_source(name):
    idx = json.load(open(f"{BF16_DIR}/model.safetensors.index.json"))
    return read_st(f"{BF16_DIR}/{idx['weight_map'][name]}", name)


def awq_tensor(base):
    idx = json.load(open(f"{AWQ_DIR}/model.safetensors.index.json"))
    w = read_st(f"{AWQ_DIR}/{idx['weight_map'][base + '.weight']}", base + ".weight")
    s = read_st(f"{AWQ_DIR}/{idx['weight_map'][base + '.scales']}", base + ".scales")
    b = read_st(f"{AWQ_DIR}/{idx['weight_map'][base + '.biases']}", base + ".biases")
    O, Iw = w.shape
    I = Iw * 8  # 4-bit: 8 nibbles per uint32
    shifts = np.arange(8, dtype=np.uint32) * 4
    q = ((w[:, :, None] >> shifts[None, None, :]) & 0xF).astype(np.float32)
    q = q.reshape(O, I)
    gs = I // s.shape[1]
    scales = np.repeat(s, gs, axis=1)
    biases = np.repeat(b, gs, axis=1)
    return q * scales + biases


def ane_int4(W, clip):
    max_abs = np.abs(W).max(axis=1, keepdims=True)
    best = None
    grid = np.linspace(0.40, 1.0, 25) if clip else [1.0]
    for m in grid:
        s = np.maximum(max_abs * m / 7.0, 1e-30)
        q = np.clip(np.rint(W / s), -8, 7)
        err = ((q * s - W) ** 2).sum(axis=1, keepdims=True)
        if best is None:
            best = (err, s)
        else:
            better = err < best[0]
            best = (np.where(better, err, best[0]), np.where(better, s, best[1]))
    q = np.clip(np.rint(W / best[1]), -8, 7)
    return q * best[1]


def rel(Wq, W):
    return float(np.linalg.norm(Wq - W) / np.linalg.norm(W))


def main():
    for part in ["gate_proj", "down_proj"]:
        src = bf16_source(f"model.language_model.layers.{LAYER}.mlp.{part}.weight")
        awq = awq_tensor(f"language_model.model.layers.{LAYER}.mlp.{part}")
        print(f"\n{part} {src.shape} (bf16 source) vs AWQ {awq.shape}")
        rows = [
            ("awq_asis (GPU path)", awq),
            ("chain: awq->ane_int4_clip", ane_int4(awq, True)),
            ("chain: awq->ane_int4_rtn", ane_int4(awq, False)),
            ("direct: bf16->ane_int4_clip", ane_int4(src, True)),
            ("direct: bf16->ane_int4_rtn", ane_int4(src, False)),
        ]
        for label, Wq in rows:
            print(f"  {label:<28} rel_err_vs_bf16 = {rel(Wq, src):.4f}")


if __name__ == "__main__":
    main()
