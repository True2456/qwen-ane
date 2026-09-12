"""Suffix-match drafting over the tokens already in play.

The MTP head predicts the first draft position at about 92%, and then falls
off a cliff: 40% at the second and below that at the third. It is a one-layer
module being fed its own output where it was trained on a forty-eight layer
state, so chaining it compounds the mismatch, and widening the block does not
help — K=8 accepts 24% of its drafts against 52% at K=4.

Text that repeats itself does not need the head at all. When the last few
tokens have occurred before, the token that followed them last time is a very
good guess, and it costs a dictionary lookup. That covers the case this model
is actually used for: editing code, quoting a file back, filling a structure.

The chain is still driven through the MTP head one step at a time, because the
drafter's attention cache needs a row per drafted position for a partly
accepted block to unwind. A lookup hit replaces the head's argmax and is then
fed back in, so the head continues along the matched path instead of its own.
"""
from __future__ import annotations

import os

__all__ = ["ContextLookup"]


class ContextLookup:
    """Last-occurrence index over n-grams of the tokens seen so far."""

    __slots__ = ("ids", "g", "index", "hits", "calls")

    def __init__(self, g: int | None = None):
        # Three is the shortest match that is worth trusting. Two fires
        # constantly on common bigrams and proposes noise; four rarely fires
        # on anything the head was not already going to get right.
        if g is None:
            g = int(os.environ.get("FLASHNEXT_NGRAM_G", "3"))
        self.g = max(1, int(g))
        self.ids: list[int] = []
        self.index: dict[tuple, int] = {}
        self.hits = 0
        self.calls = 0

    def extend(self, tokens) -> None:
        """Append confirmed tokens and index the n-grams they complete."""
        g = self.g
        ids = self.ids
        start = len(ids)
        ids.extend(int(t) for t in tokens)
        # An n-gram ending at position p - 1 points at p. Re-index from g
        # positions back, because the tail of the previous call only became a
        # complete n-gram once these tokens arrived.
        for p in range(max(g, start), len(ids)):
            self.index[tuple(ids[p - g:p])] = p

    def next_token(self, drafted) -> int | None:
        """The token that last followed this suffix, or None."""
        self.calls += 1
        g = self.g
        d = len(drafted)
        if d >= g:
            key = tuple(int(t) for t in drafted[d - g:])
        else:
            tail = self.ids[len(self.ids) - (g - d):] if g > d else []
            if len(tail) < g - d:
                return None
            key = tuple(tail) + tuple(int(t) for t in drafted)
        p = self.index.get(key)
        if p is None or p >= len(self.ids):
            return None
        self.hits += 1
        return self.ids[p]

    def stats(self) -> str:
        return f"{self.hits}/{self.calls}"
