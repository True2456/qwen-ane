#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""apc_cache.py - High-performance Automatic Prefix Cache (Radix Prefix Tree) for Apple Silicon."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple
import mlx.core as mx


class APCNode:
    """A node in the Radix Prefix Tree storing cached KV and GDN recurrent states."""

    def __init__(self, token_ids: List[int], parent: Optional[APCNode] = None):
        self.token_ids: List[int] = list(token_ids)
        self.parent: Optional[APCNode] = parent
        self.children: Dict[int, APCNode] = {}
        self.cache_snapshot: Optional[List[Any]] = None
        self.last_hidden: Optional[mx.array] = None
        self.next_token: Optional[int] = None
        self.last_accessed: float = time.perf_counter()
        self.access_count: int = 0

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def touch(self) -> None:
        self.last_accessed = time.perf_counter()
        self.access_count += 1


class APCCache:
    """Radix-Tree Automatic Prefix Cache for instant zero-compute prefill matching."""

    def __init__(self, max_cached_tokens: int = 131072):
        self.root = APCNode([])
        self.max_cached_tokens = max_cached_tokens
        self.total_tokens_stored = 0
        self.hits = 0
        self.misses = 0
        self.total_tokens_saved = 0

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, Optional[APCNode], List[int]]:
        """
        Finds the longest cached prefix matching the incoming token sequence.
        Returns:
            (matched_length, best_node, remaining_tokens)
        """
        curr = self.root
        matched_tokens = 0
        idx = 0
        N = len(token_ids)

        while idx < N:
            tok = token_ids[idx]
            if tok not in curr.children:
                break
            child = curr.children[tok]
            child_len = len(child.token_ids)

            # Check common prefix with child
            common_len = 0
            while common_len < child_len and idx + common_len < N and child.token_ids[common_len] == token_ids[idx + common_len]:
                common_len += 1

            if common_len == child_len:
                curr = child
                idx += child_len
                matched_tokens += child_len
                curr.touch()
            elif common_len > 0:
                # Partial match on child branch
                idx += common_len
                matched_tokens += common_len
                curr = child
                break
            else:
                break

        if matched_tokens > 0 and curr.cache_snapshot is not None:
            self.hits += 1
            self.total_tokens_saved += matched_tokens
            return matched_tokens, curr, token_ids[matched_tokens:]
        else:
            self.misses += 1
            return 0, None, token_ids

    def insert(
        self,
        token_ids: List[int],
        cache_snapshot: List[Any],
        last_hidden: Optional[mx.array] = None,
        next_token: Optional[int] = None,
    ) -> APCNode:
        """Inserts or updates a prefix path in the Radix Tree with its execution state."""
        curr = self.root
        idx = 0
        N = len(token_ids)

        while idx < N:
            tok = token_ids[idx]
            if tok not in curr.children:
                # Create new branch for remainder of tokens
                new_node = APCNode(token_ids[idx:], parent=curr)
                new_node.cache_snapshot = self._clone_snapshot(cache_snapshot)
                new_node.last_hidden = last_hidden
                new_node.next_token = next_token
                new_node.touch()
                curr.children[tok] = new_node
                self.total_tokens_stored += len(new_node.token_ids)
                return new_node

            child = curr.children[tok]
            child_len = len(child.token_ids)

            # Check common prefix with child
            common_len = 0
            while common_len < child_len and idx + common_len < N and child.token_ids[common_len] == token_ids[idx + common_len]:
                common_len += 1

            if common_len == child_len:
                curr = child
                idx += child_len
                curr.touch()
            else:
                # Split child node at common_len
                split_node = APCNode(child.token_ids[:common_len], parent=curr)
                child.token_ids = child.token_ids[common_len:]
                child.parent = split_node

                curr.children[tok] = split_node
                split_node.children[child.token_ids[0]] = child

                # Insert remainder
                idx += common_len
                if idx < N:
                    rem_node = APCNode(token_ids[idx:], parent=split_node)
                    rem_node.cache_snapshot = self._clone_snapshot(cache_snapshot)
                    rem_node.last_hidden = last_hidden
                    rem_node.next_token = next_token
                    rem_node.touch()
                    split_node.children[token_ids[idx]] = rem_node
                    self.total_tokens_stored += len(rem_node.token_ids)
                    return rem_node
                else:
                    split_node.cache_snapshot = self._clone_snapshot(cache_snapshot)
                    split_node.last_hidden = last_hidden
                    split_node.next_token = next_token
                    split_node.touch()
                    return split_node

        # Exact match node update
        curr.cache_snapshot = self._clone_snapshot(cache_snapshot)
        curr.last_hidden = last_hidden
        curr.next_token = next_token
        curr.touch()
        return curr

    def _clone_snapshot(self, snap: List[Any]) -> List[Any]:
        """Deep copy cache state arrays so subsequent rollbacks do not mutate cached entries."""
        out = []
        for s in snap:
            if s is None:
                out.append(None)
            elif isinstance(s, tuple) and len(s) == 3:
                tag, off, payload = s
                if tag == "gdn" and payload is not None:
                    cloned_raw = [None if x is None else mx.array(x) for x in payload]
                    out.append((tag, off, cloned_raw))
                elif tag == "kv" and payload is not None:
                    k, v = payload
                    cloned_k = mx.array(k) if k is not None else None
                    cloned_v = mx.array(v) if v is not None else None
                    out.append((tag, off, (cloned_k, cloned_v)))
                else:
                    out.append(s)
            elif isinstance(s, list):
                out.append([None if x is None else mx.array(x) for x in s])
            else:
                out.append(s)
        return out

    def stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100.0) if total_requests > 0 else 0.0
        return {
            "total_requests": total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": hit_rate,
            "tokens_saved": self.total_tokens_saved,
            "total_tokens_stored": self.total_tokens_stored,
        }
