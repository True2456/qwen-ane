#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""test_apc_cache.py - Unit test probe for Radix Automatic Prefix Cache."""

from __future__ import annotations

import sys
from pathlib import Path
import mlx.core as mx

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runtime.apc_cache import APCCache


def test_apc_basic_matching():
    print("Testing APC Basic Prefix Matching...")
    cache = APCCache()

    system_prompt = [100, 101, 102, 103, 104]
    mock_snapshot = [(5, [mx.ones((1, 4, 16))])]

    # 1. Initial lookup -> Miss
    matched, node, remaining = cache.match_prefix(system_prompt + [200, 201])
    assert matched == 0
    assert node is None
    assert remaining == system_prompt + [200, 201]
    print("  ✓ Initial cache miss verified")

    # 2. Insert system prompt
    cache.insert(system_prompt, mock_snapshot, last_hidden=mx.zeros((1, 1, 16)), next_token=200)

    # 3. Repeat lookup with new suffix -> Exact prefix hit
    user_query = system_prompt + [200, 201, 202]
    matched, node, remaining = cache.match_prefix(user_query)
    assert matched == 5
    assert node is not None
    assert node.next_token == 200
    assert remaining == [200, 201, 202]
    print("  ✓ Full system prompt cache hit verified (matched 5 tokens, remaining 3)")

    # 4. Insert full branch
    cache.insert(user_query, mock_snapshot, last_hidden=mx.zeros((1, 1, 16)), next_token=300)

    # 5. Lookup on deeper query
    deeper_query = user_query + [300, 301]
    matched, node, remaining = cache.match_prefix(deeper_query)
    assert matched == 8
    assert node is not None
    assert node.next_token == 300
    assert remaining == [300, 301]
    print("  ✓ Deeper multi-turn branch match verified (matched 8 tokens, remaining 2)")

    stats = cache.stats()
    print("  APC Stats:", stats)
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["tokens_saved"] == 13
    print("  ✓ All APC Radix Tree tests passed!")


if __name__ == "__main__":
    test_apc_basic_matching()
