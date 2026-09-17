"""Speculation under sampling must not change the output distribution."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.flashnext_sampling import filtered_dist, verify_block

PARAMS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def _logits(rng, vocab=500, slots=4):
    return (rng.standard_normal((slots, vocab)) * 2.0).astype(np.float32)


def test_filters_are_applied_in_order():
    rng = np.random.default_rng(0)
    row = _logits(rng, slots=1)[0]
    ids, pr = filtered_dist(row, {"temperature": 1.0, "top_k": 20, "top_p": 1.0})
    assert len(ids) == 20 and abs(pr.sum() - 1) < 1e-9
    ids2, pr2 = filtered_dist(row, PARAMS)
    assert len(ids2) <= 20 and abs(pr2.sum() - 1) < 1e-9
    ids3, pr3 = filtered_dist(row, dict(PARAMS, min_p=0.5))
    assert np.all(pr3 >= 0.5 * pr3[0] - 1e-12)


def test_first_token_matches_the_target_distribution():
    """The whole point: drafting must not move the distribution."""
    rng = np.random.default_rng(7)
    logits = _logits(rng, slots=4)
    ids, pr = filtered_dist(logits[0], PARAMS)
    target = dict(zip(ids.tolist(), pr.tolist()))
    # A deliberately mediocre drafter: the second most likely token.
    draft = int(ids[1])
    trials, counts = 40000, {}
    r = np.random.default_rng(11)
    for _ in range(trials):
        m, nxt = verify_block(logits, [draft, draft, draft], PARAMS, r)
        first = draft if m >= 1 else nxt
        counts[first] = counts.get(first, 0) + 1
    for tok, want in target.items():
        got = counts.get(tok, 0) / trials
        assert abs(got - want) < 0.01, f"token {tok}: {got:.4f} vs {want:.4f}"


def test_greedy_is_unchanged():
    rng = np.random.default_rng(3)
    logits = _logits(rng, slots=4)
    preds = [int(v) for v in np.argmax(logits, axis=-1)]
    m, nxt = verify_block(logits, preds[:3], {"temperature": 0},
                          np.random.default_rng(0))
    assert m == 3 and nxt == preds[3]
    m, nxt = verify_block(logits, [preds[0], preds[1] ^ 1], {"temperature": 0},
                          np.random.default_rng(0))
    assert m == 1 and nxt == preds[1]


def test_pure_ane_speculative_sampler():
    from tools.pure_ane import Sampler
    
    # 1. Greedy behavior
    s_greedy = Sampler(temperature=0.0)
    logits = np.array([1.0, 5.0, 2.0, 0.5], dtype=np.float32)
    tok, _ = s_greedy.sample(logits)
    assert tok == 1
    acc, corr = s_greedy.speculative_verify(1, np.array([]), logits)
    assert acc is True and corr is None
    acc, corr = s_greedy.speculative_verify(0, np.array([]), logits)
    assert acc is False and corr == 1
    
    # 2. Speculative rejection sampling distribution match
    s_samp = Sampler(temperature=0.8, top_p=0.95, seed=123)
    target_logits = np.array([1.0, 4.0, 2.5, 0.5], dtype=np.float32)
    draft_logits = np.array([1.2, 3.8, 2.3, 0.6], dtype=np.float32)
    
    p_true = s_samp.get_probs(target_logits)
    
    # Sample many times using speculative rejection sampling
    N = 20000
    counts = np.zeros(len(target_logits))
    for _ in range(N):
        d_tok, q_probs = s_samp.sample(draft_logits)
        acc, corr = s_samp.speculative_verify(d_tok, q_probs, target_logits)
        chosen = d_tok if acc else corr
        counts[chosen] += 1
    
    p_emp = counts / N
    max_err = np.max(np.abs(p_emp - p_true))
    assert max_err < 0.015, f"Speculative sampling error {max_err} exceeded tolerance"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

