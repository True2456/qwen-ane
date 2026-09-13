"""Sampling, and speculation that stays honest under it.

Greedy verification accepts a draft only when it equals the backbone's argmax.
That is right at temperature 0 and wrong at anything else: sampling the final
token while accepting drafts on an argmax match biases the output toward
whatever the drafter proposed, so the text looks sampled but comes from the
wrong distribution.

Both drafters in this port propose a single token with no distribution of
their own -- the MTP head is used greedily and the context lookup has none --
so the usual rejection rule, accept with probability min(1, p(x)/q(x)) and
otherwise draw from the normalised positive part of p - q, collapses to its
point-mass case: accept with probability p(x), and on rejection draw from p
with x removed. That is exactly unbiased, and `tests/test_sampling.py` checks
it against the target distribution over many trials.
"""
from __future__ import annotations

import numpy as np

__all__ = ["filtered_dist", "verify_block", "SamplingParams"]


class SamplingParams(dict):
    """temperature, top_p, top_k, min_p. temperature 0 means greedy."""

    @property
    def greedy(self) -> bool:
        return float(self.get("temperature", 0)) <= 0


def filtered_dist(row, params) -> tuple[np.ndarray, np.ndarray]:
    """The distribution a token is actually drawn from. Returns (ids, probs).

    top_k, then nucleus, then min_p, which is the order serving frameworks
    apply them in, so the numbers on a model card mean the same thing here.
    """
    t = float(params.get("temperature", 0)) or 1.0
    k = int(params.get("top_k", 0)) or 64
    k = min(k, row.shape[0])
    idx = np.argpartition(row, -k)[-k:]
    logits = np.asarray(row[idx], np.float64) / max(t, 1e-6)
    order = np.argsort(-logits)
    idx, logits = idx[order], logits[order]
    pr = np.exp(logits - logits[0])
    pr /= pr.sum()
    top_p = float(params.get("top_p", 1) or 1)
    if 0 < top_p < 1:
        keep = int(np.searchsorted(np.cumsum(pr), top_p) + 1)
        idx, pr = idx[:keep], pr[:keep] / pr[:keep].sum()
    min_p = float(params.get("min_p", 0) or 0)
    if min_p > 0:
        keep = pr >= min_p * pr[0]
        idx, pr = idx[keep], pr[keep] / pr[keep].sum()
    return idx, pr


def _draw(rng, ids, pr) -> int:
    return int(ids[rng.choice(len(ids), p=pr)])


def verify_block(logits, drafts, params, rng):
    """Returns (accepted, next_token). `logits` is one row a slot."""
    if SamplingParams(params).greedy:
        preds = np.argmax(logits, axis=-1)
        m = 0
        while m < len(drafts) and int(preds[m]) == drafts[m]:
            m += 1
        return m, int(preds[m])
    m = 0
    while m < len(drafts):
        ids, pr = filtered_dist(logits[m], params)
        hit = np.flatnonzero(ids == drafts[m])
        if hit.size and rng.random() < float(pr[hit[0]]):
            m += 1
            continue
        if hit.size:
            pr = pr.copy()
            pr[hit[0]] = 0.0
            total = pr.sum()
            if total <= 0:                      # the draft held every filtered
                ids, pr = filtered_dist(logits[m], params)   # draw afresh
            else:
                pr = pr / total
        return m, _draw(rng, ids, pr)
    ids, pr = filtered_dist(logits[m], params)
    return m, _draw(rng, ids, pr)
