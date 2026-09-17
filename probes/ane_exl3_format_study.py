#!/usr/bin/env python3
# Weight-format quality study: what the ANE's per-channel-only constraint costs
# vs modern formats (EXL3/GGUF/AWQ-class group-wise), measured on REAL weights.
#
# Replicates RindiAneProjection::compile_int4 exactly (RTN symmetric per-row
# int4, scale=max_abs/7) as the baseline, then tests every improvement that
# STAYS inside formats the ANE accepts (per-channel scales only):
#   clip      — per-row scale search (pure RTN improvement, no format change)
#   rot       — QuaRot-style Hadamard rotation folded around the projection
#   mixed48   — sensitive rows promoted to int8 (per-channel still)
# Bars (NOT ANE-runnable, they need group-wise/blockwise scales):
#   gw64/gw32 — the Metal path's affine group-wise int4 (what ships on GPU)
#   int8      — the ANE's high-accuracy option
import glob
import json
import sys
import time

import numpy as np

QWEN36 = str(Path.home() / ".lmstudio/models/mlx-community/Qwen3.6-35B-A3B-bf16")
QWEN38 = str(Path.home() / "models/Qwen3.8-27B")
LAYERS = [5, 20, 35]
EXPERTS = [0, 63]


def load_tensor(model_dir, name):
    """Manual safetensors read: safetensors' numpy backend rejects BF16."""
    import struct
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    path = f"{model_dir}/{idx['weight_map'][name]}"
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    meta = header[name]
    dt, shape = meta["dtype"], meta["shape"]
    begin, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + hlen + begin)
        raw = f.read(end - begin)
    if dt == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (bits << 16).view(np.float32).reshape(shape)
    if dt == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dt == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    raise ValueError(f"unsupported dtype {dt}")


# --- quantizers -------------------------------------------------------------
def ane_int4_rtn(W):
    """Exact replica of the engine's ANE int4 path."""
    max_abs = np.abs(W).max(axis=1, keepdims=True)
    scale = np.maximum(max_abs, 1e-30) / 7.0
    q = np.clip(np.rint(W / scale), -8, 7)
    return q * scale


def ane_int8_rtn(W):
    max_abs = np.abs(W).max(axis=1, keepdims=True)
    scale = np.maximum(max_abs, 1e-30) / 127.0
    q = np.clip(np.rint(W / scale), -128, 127)
    return q * scale


def sym_clip(W, bits):
    """Per-row optimal-scale search. Same format as ane_int4/int8 (per-channel
    scale only) — just picks the scale that minimises row MSE instead of RTN."""
    qmax = 2 ** (bits - 1) - 1
    max_abs = np.abs(W).max(axis=1, keepdims=True)
    best_err = None
    best_s = None
    for m in np.linspace(0.4, 1.0, 25):
        s = np.maximum(max_abs * (m / qmax), 1e-30)
        q = np.clip(np.rint(W / s), -qmax - 1, qmax)
        err = ((q * s - W) ** 2).sum(axis=1, keepdims=True)
        if best_err is None:
            best_err, best_s = err, s
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_s = np.where(better, s, best_s)
    q = np.clip(np.rint(W / best_s), -qmax - 1, qmax)
    return q * best_s


def gw_affine(W, gs):
    """Metal path format: per-gs-column group, scale=(max-min)/15, bias=min."""
    O, I = W.shape
    pad = (-I) % gs
    Wg = np.pad(W, ((0, 0), (0, pad))).reshape(O, -1, gs)
    mn = Wg.min(axis=2, keepdims=True)
    mx = Wg.max(axis=2, keepdims=True)
    sc = np.maximum(mx - mn, 1e-30) / 15.0
    q = np.clip(np.rint((Wg - mn) / sc), 0, 15)
    return (q * sc + mn).reshape(O, -1)[:, :I]


def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.kron(H, np.array([[1.0, 1.0], [1.0, -1.0]]))
    return H / np.sqrt(n)

_HB = hadamard(128)  # QuaRot-style block-diagonal rotation

def rot_apply(W):
    """Block-diagonal Hadamard over the input dim (any I % 128 == 0)."""
    O, I = W.shape
    return (W.reshape(O, -1, 128) @ _HB.T).reshape(O, I)


def act_rel_err(W, Wq, seed=0, draws=2, cols=256):
    rng = np.random.default_rng(seed)
    tot = 0.0
    for d in range(draws):
        X = rng.standard_normal((W.shape[1], cols))
        tot += np.linalg.norm((Wq - W) @ X) / np.linalg.norm(W @ X)
    return tot / draws


def rel_fro(W, Wq):
    return float(np.linalg.norm(Wq - W) / np.linalg.norm(W))


# --- study ------------------------------------------------------------------
def main():
    groups = {"moe_gate": [], "moe_up": [], "moe_down": [], "dense": []}
    t0 = time.time()

    idx36 = json.load(open(f"{QWEN36}/model.safetensors.index.json"))
    for layer in LAYERS:
        for part in ["gate", "up", "down"]:
            name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{part}_proj.weight"
            stacked = load_tensor(QWEN36, name)
            for e in EXPERTS:
                W = stacked[e]
                g = f"moe_{part}"
                run_group(groups, g, W, tag=f"L{layer}/e{e}")
        print(f"  layer {layer} done ({time.time()-t0:.0f}s)", file=sys.stderr)

    idx38 = json.load(open(f"{QWEN38}/model.safetensors.index.json"))
    for part in ["gate_proj", "down_proj"]:
        name = f"model.language_model.layers.20.mlp.{part}.weight"
        W = load_tensor(QWEN38, name)
        run_group(groups, "dense", W, tag=f"27B/L20/{part}")
    print(f"  dense done ({time.time()-t0:.0f}s)", file=sys.stderr)

    schemes = ["ane_int4_rtn", "ane_int4_clip", "rot_int4_clip",
               "mixed48_clip", "ane_int8_rtn", "metal_gw64", "gw32_ref"]
    print("\n=== Relative Frobenius error (lower is better) ===")
    hdr = f"{'scheme':<14}" + "".join(f"{g:>13}" for g in groups)
    print(hdr + "   (bpw)")
    bpw = {"ane_int4_rtn": 4.03, "ane_int4_clip": 4.03, "rot_int4_clip": 4.03,
           "mixed48_clip": "~5", "ane_int8_rtn": 8.06, "metal_gw64": 4.25,
           "gw32_ref": 4.5}
    for s in schemes:
        row = f"{s:<14}"
        for g in groups:
            vals = [r[s] for r in groups[g]]
            row += f"{np.mean(vals):>10.4f}±{np.std(vals):.3f}"
        print(row + f"   ({bpw[s]})")

    print("\n=== Activation-weighted output rel err (iid N(0,1) activations) ===")
    print("NOTE: iid activations understate outlier-driven error; treat as")
    print("lower bound. Real-calibration GPTQ/AWQ would differ.")
    print(hdr)
    for s in schemes:
        row = f"{s:<14}"
        for g in groups:
            vals = [r[s + "_act"] for r in groups[g]]
            row += f"{np.mean(vals):>10.4f}±{np.std(vals):.3f}"
        print(row)

    print(f"\nmatrices per group: " +
          ", ".join(f"{g}={len(groups[g])}" for g in groups))


def run_group(groups, g, W, tag=""):
    quants = {
        "ane_int4_rtn": ane_int4_rtn(W),
        "ane_int4_clip": sym_clip(W, 4),
        "rot_int4_clip": rot_apply(sym_clip(rot_apply(W), 4)),
        "ane_int8_rtn": ane_int8_rtn(W),
        "metal_gw64": gw_affine(W, 64),
        "gw32_ref": gw_affine(W, 32),
    }
    # mixed 4/8 by per-row clip error (rows 2x median -> int8)
    W4 = quants["ane_int4_clip"]
    W8 = quants["ane_int8_rtn"]
    e4 = ((W4 - W) ** 2).sum(axis=1)
    hot = e4 > 2.0 * np.median(e4)
    mixed = W4.copy()
    mixed[hot] = W8[hot]
    quants["mixed48_clip"] = mixed

    r = {}
    for k, Wq in quants.items():
        r[k] = rel_fro(W, Wq)
        r[k + "_act"] = act_rel_err(W, Wq, seed=len(tag) + hash(k) % 1000)
    r["_mixed_bpw"] = 4.03 * (~hot).mean() + 8.06 * hot.mean()
    groups[g].append(r)


if __name__ == "__main__":
    main()
