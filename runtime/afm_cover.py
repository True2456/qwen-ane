"""AFM-style covering-set stats for Flash-Next MoE routes.

Traces are ``inds[T, L, K]`` (top-k expert ids per token per layer) and
optional ``scores[T, L, K]`` (renormalised top-k mass). Prefill is the first
``n_prompt`` tokens; the rest are decode.
"""
from __future__ import annotations

from typing import Any

import numpy as np

N_EXPERTS = 512
K_ROUTE = 10


def _as_inds(inds: np.ndarray) -> np.ndarray:
    a = np.asarray(inds)
    if a.ndim != 3:
        raise ValueError(f"inds must be [T, L, K], got {a.shape}")
    return a.astype(np.int32, copy=False)


def union_growth(inds: np.ndarray) -> np.ndarray:
    """Cumulative unique experts per layer. Shape [T, L]."""
    inds = _as_inds(inds)
    t_n, l_n, _ = inds.shape
    out = np.zeros((t_n, l_n), np.int32)
    for layer in range(l_n):
        seen: set[int] = set()
        for t in range(t_n):
            seen.update(int(x) for x in inds[t, layer])
            out[t, layer] = len(seen)
    return out


def sliding_union_size(inds: np.ndarray, window: int) -> np.ndarray:
    """Union of the last ``window`` tokens' top-k, per layer. [T, L]."""
    inds = _as_inds(inds)
    t_n, l_n, _ = inds.shape
    out = np.zeros((t_n, l_n), np.int32)
    w = max(1, int(window))
    for layer in range(l_n):
        for t in range(t_n):
            lo = max(0, t - w + 1)
            seen: set[int] = set()
            for u in range(lo, t + 1):
                seen.update(int(x) for x in inds[u, layer])
            out[t, layer] = len(seen)
    return out


def hit_rate(inds: np.ndarray, pin: np.ndarray | list[set[int]], start: int = 0) -> np.ndarray:
    """Fraction of layers whose current top-k is a subset of the pin. [T]."""
    inds = _as_inds(inds)
    t_n, l_n, _ = inds.shape
    pins: list[set[int]]
    if isinstance(pin, np.ndarray):
        pins = [set(int(x) for x in pin[layer]) for layer in range(l_n)]
    else:
        pins = list(pin)
    hits = np.zeros(t_n, np.float64)
    for t in range(start, t_n):
        n = 0
        for layer in range(l_n):
            cur = set(int(x) for x in inds[t, layer])
            n += int(cur <= pins[layer])
        hits[t] = n / l_n
    return hits


def score_mass_in_pin(
    inds: np.ndarray,
    scores: np.ndarray,
    pin: list[set[int]],
    start: int = 0,
) -> np.ndarray:
    """Renormalised top-k mass that lands inside the pin. [T, L]."""
    inds = _as_inds(inds)
    sc = np.asarray(scores, np.float32)
    t_n, l_n, k = inds.shape
    out = np.zeros((t_n, l_n), np.float32)
    for t in range(start, t_n):
        for layer in range(l_n):
            p = pin[layer]
            mass = 0.0
            for j in range(k):
                if int(inds[t, layer, j]) in p:
                    mass += float(sc[t, layer, j])
            out[t, layer] = mass
    return out


def frequency_pin(inds: np.ndarray, size: int, end: int) -> list[set[int]]:
    """Top-``size`` experts by hit-count over tokens ``[0, end)``."""
    inds = _as_inds(inds)
    _, l_n, _ = inds.shape
    size = int(np.clip(size, 1, N_EXPERTS))
    end = max(1, min(int(end), inds.shape[0]))
    pins: list[set[int]] = []
    for layer in range(l_n):
        counts = np.bincount(inds[:end, layer].ravel(), minlength=N_EXPERTS)
        keep = np.argpartition(counts, -size)[-size:]
        pins.append(set(int(x) for x in keep))
    return pins


def union_pin(inds: np.ndarray, start: int, end: int) -> list[set[int]]:
    inds = _as_inds(inds)
    _, l_n, _ = inds.shape
    pins: list[set[int]] = []
    for layer in range(l_n):
        seen: set[int] = set()
        for t in range(start, end):
            seen.update(int(x) for x in inds[t, layer])
        pins.append(seen)
    return pins


def refresh_count(inds: np.ndarray, start: int = 0) -> dict[str, Any]:
    """Restage the pin to the current top-k whenever a layer misses.

    Returns mean restages per token (over layers) and the fraction of
    layer-steps that required a restage.
    """
    inds = _as_inds(inds)
    t_n, l_n, _ = inds.shape
    restages = 0
    steps = 0
    pins = [set(int(x) for x in inds[start, layer]) for layer in range(l_n)]
    for t in range(start + 1, t_n):
        for layer in range(l_n):
            cur = set(int(x) for x in inds[t, layer])
            steps += 1
            if not (cur <= pins[layer]):
                restages += 1
                pins[layer] = cur
    return {
        "restage_frac": restages / max(steps, 1),
        "restages": restages,
        "steps": steps,
        "mean_restage_layers_per_token": restages / max(t_n - start - 1, 1),
    }


def covering_report(
    inds: np.ndarray,
    scores: np.ndarray | None,
    n_prompt: int,
    pin_sizes: tuple[int, ...] = (10, 16, 24, 32, 48, 64, 96, 128),
    windows: tuple[int, ...] = (1, 4, 8, 16, 32, 64),
    types: np.ndarray | None = None,
) -> dict[str, Any]:
    inds = _as_inds(inds)
    t_n, l_n, k = inds.shape
    n_prompt = int(np.clip(n_prompt, 1, t_n))
    decode = slice(n_prompt, t_n)
    growth = union_growth(inds)
    adj = []
    for t in range(1, t_n):
        jacs = []
        for layer in range(l_n):
            a = set(int(x) for x in inds[t - 1, layer])
            b = set(int(x) for x in inds[t, layer])
            jacs.append(len(a & b) / max(len(a | b), 1))
        adj.append(float(np.mean(jacs)))

    gdn = qsa = None
    if types is not None:
        types = np.asarray(types)
        gdn = np.where(types == "linear_attention")[0]
        qsa = np.where(types != "linear_attention")[0]

    def _split_mean(arr_tl: np.ndarray, sl: slice) -> dict[str, float]:
        block = arr_tl[sl]
        out = {"all": float(block.mean())}
        if gdn is not None and gdn.size:
            out["gdn"] = float(block[:, gdn].mean())
        if qsa is not None and qsa.size:
            out["qsa"] = float(block[:, qsa].mean())
        return out

    freq_table = []
    prefill_union = union_pin(inds, 0, n_prompt)
    prefill_union_sizes = np.array([len(s) for s in prefill_union], np.int32)
    frozen10 = [set(int(x) for x in inds[n_prompt - 1, layer]) for layer in range(l_n)]
    rows = []
    for p in pin_sizes:
        pin = frequency_pin(inds, p, n_prompt)
        hits = hit_rate(inds, pin, start=n_prompt)
        row: dict[str, Any] = {
            "pin": p,
            "decode_hit": float(hits[n_prompt:].mean()) if t_n > n_prompt else 0.0,
            "decode_hit_last16": float(hits[-16:].mean()) if t_n > n_prompt else 0.0,
        }
        if scores is not None and t_n > n_prompt:
            mass = score_mass_in_pin(inds, scores, pin, start=n_prompt)
            row["decode_mass"] = float(mass[n_prompt:].mean())
        rows.append(row)

    win_rows = []
    for w in windows:
        sizes = sliding_union_size(inds, w)
        # hit of token t against union of [t-w, t) (previous window, not including t)
        hits_w = []
        sizes_prev = []
        for t in range(max(n_prompt, 1), t_n):
            lo = max(0, t - w)
            pin = union_pin(inds, lo, t)
            n_hit = 0
            sz = []
            for layer in range(l_n):
                cur = set(int(x) for x in inds[t, layer])
                n_hit += int(cur <= pin[layer])
                sz.append(len(pin[layer]))
            hits_w.append(n_hit / l_n)
            sizes_prev.append(float(np.mean(sz)))
        win_rows.append({
            "window": w,
            "decode_hit": float(np.mean(hits_w)) if hits_w else 0.0,
            "mean_pin_size": float(np.mean(sizes_prev)) if sizes_prev else float(sizes[-1].mean()),
        })

    report: dict[str, Any] = {
        "T": t_n,
        "L": l_n,
        "K": k,
        "n_prompt": n_prompt,
        "n_decode": max(t_n - n_prompt, 0),
        "unique_end": _split_mean(growth, slice(t_n - 1, t_n)),
        "unique_prefill_end": _split_mean(growth, slice(n_prompt - 1, n_prompt)),
        "unique_growth_tl": growth,
        "prefill_union_size": {
            "mean": float(prefill_union_sizes.mean()),
            "min": int(prefill_union_sizes.min()),
            "max": int(prefill_union_sizes.max()),
        },
        "adjacent_jaccard": {
            "all": float(np.mean(adj)) if adj else 0.0,
            "decode": float(np.mean(adj[n_prompt - 1:])) if t_n > n_prompt else 0.0,
        },
        "frozen_last_prefill_top10_hit": float(
            hit_rate(inds, frozen10, start=n_prompt)[n_prompt:].mean()
        ) if t_n > n_prompt else 0.0,
        "prefill_union_decode_hit": float(
            hit_rate(inds, prefill_union, start=n_prompt)[n_prompt:].mean()
        ) if t_n > n_prompt else 0.0,
        "freq_pin": rows,
        "sliding": win_rows,
        "refresh_top10": refresh_count(inds, start=n_prompt - 1 if n_prompt else 0),
    }
    if scores is not None and t_n > n_prompt:
        mass10 = score_mass_in_pin(inds, scores, frozen10, start=n_prompt)
        report["frozen_last_prefill_top10_mass"] = float(mass10[n_prompt:].mean())
        mass_u = score_mass_in_pin(inds, scores, prefill_union, start=n_prompt)
        report["prefill_union_decode_mass"] = float(mass_u[n_prompt:].mean())
    return report


def format_report(rep: dict[str, Any]) -> str:
    lines = [
        f"tokens={rep['T']}  prompt={rep['n_prompt']}  decode={rep['n_decode']}  "
        f"layers={rep['L']}  k={rep['K']}",
        f"unique experts at prefill end: mean={rep['unique_prefill_end']['all']:.1f}"
        + (f"  GDN={rep['unique_prefill_end'].get('gdn', 0):.1f}  "
           f"QSA={rep['unique_prefill_end'].get('qsa', 0):.1f}"
           if "gdn" in rep["unique_prefill_end"] else ""),
        f"unique experts at stream end:  mean={rep['unique_end']['all']:.1f}"
        + (f"  GDN={rep['unique_end'].get('gdn', 0):.1f}  "
           f"QSA={rep['unique_end'].get('qsa', 0):.1f}"
           if "gdn" in rep["unique_end"] else ""),
        f"prefill union size: mean={rep['prefill_union_size']['mean']:.1f}  "
        f"min={rep['prefill_union_size']['min']}  max={rep['prefill_union_size']['max']}",
        f"adjacent Jaccard: all={rep['adjacent_jaccard']['all']:.3f}  "
        f"decode={rep['adjacent_jaccard']['decode']:.3f}",
        f"frozen last-prefill top-10  decode hit={rep['frozen_last_prefill_top10_hit']:.3f}"
        + (f"  mass={rep['frozen_last_prefill_top10_mass']:.3f}"
           if "frozen_last_prefill_top10_mass" in rep else ""),
        f"prefill-union pin  decode hit={rep['prefill_union_decode_hit']:.3f}"
        + (f"  mass={rep['prefill_union_decode_mass']:.3f}"
           if "prefill_union_decode_mass" in rep else ""),
        f"restage-on-miss (pin=current top-10): "
        f"{rep['refresh_top10']['mean_restage_layers_per_token']:.1f} layers/token  "
        f"frac={rep['refresh_top10']['restage_frac']:.3f}",
        "",
        "frequency pin from prefill (decode hit = current top-10 ⊆ pin):",
    ]
    for row in rep["freq_pin"]:
        extra = f"  mass={row['decode_mass']:.3f}" if "decode_mass" in row else ""
        lines.append(
            f"  P={row['pin']:<3}  hit={row['decode_hit']:.3f}  "
            f"last16={row['decode_hit_last16']:.3f}{extra}"
        )
    lines.append("sliding previous-window union:")
    for row in rep["sliding"]:
        lines.append(
            f"  W={row['window']:<3}  hit={row['decode_hit']:.3f}  "
            f"mean pin size={row['mean_pin_size']:.1f}"
        )
    # Decision line
    hits = {row["pin"]: row["decode_hit"] for row in rep["freq_pin"]}
    good = [p for p, h in hits.items() if h >= 0.95]
    ok = [p for p, h in hits.items() if h >= 0.80]
    if good:
        lines.append(
            f"\nAFM pin: viable at P≥{min(good)} (≥95% of layer-steps fully covered)."
        )
    elif ok:
        lines.append(
            f"\nAFM pin: marginal at P≥{min(ok)} (80–95% hit). Expect restage or quality drift."
        )
    else:
        lines.append(
            "\nAFM pin: covering-set does not close. Frequency pin of 128 still "
            f"hits {hits.get(128, hits[max(hits)]):.3f} of decode top-10s. "
            "Do not bake a frozen set."
        )
    return "\n".join(lines)
