#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""test_agentic_long_context.py - Multi-turn agentic benchmark simulating tool calling, long context, and APC cache."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tools.hybrid_serve import HybridEngine


def run_agentic_benchmark():
    print("=" * 65)
    print("  MULTI-TURN AGENTIC & LONG-CONTEXT BENCHMARK")
    print("=" * 65)

    # 1. Initialize Hybrid Engine in Turbo Mode with APC
    engine = HybridEngine(
        model_path=str(Path.home() / ".lmstudio/models/Qwen/Qwen3.8-27B"),
        mode="turbo",
        draft_depth=3,
    )

    # Simulated Long Agent System Prompt (~500 tokens of tool schemas & guidelines)
    system_prompt = (
        "You are an autonomous AI software engineering agent with access to local tools.\n"
        "Available tools:\n"
        "- run_command(command: str): Run a bash shell command\n"
        "- view_file(path: str, start_line: int, end_line: int): View file contents\n"
        "- edit_file(path: str, target: str, replacement: str): Edit source code\n"
        "- search_codebase(pattern: str, file_filter: str): Grep across files\n\n"
        "Guidelines:\n"
        "1. Always plan before modifying files.\n"
        "2. Keep thoughts concise inside <think> tags.\n"
        "3. Output valid tool calls in XML format: <tool_call><name>...</name><args>...</args></tool_call>.\n"
        "4. Verify all changes with automated unit tests before reporting completion.\n\n"
        "Current Workspace State:\n"
        "Repository: /path/to/qwen-ane\n"
        "Architecture: Apple Silicon M5 Max (16-core ANE, 40-core GPU, Unified RAM)\n"
        "Active files: runtime/metal_engine.m, runtime/apc_cache.py, tools/hybrid_serve.py\n"
    )

    # Turn 1: Initial user prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please inspect the repository and check if all Metal shaders compile cleanly."}
    ]
    print("\n--- [Turn 1: Initial Agent Task (Cold Long Context)] ---")
    res1 = engine.generate(messages, max_tokens=32, mode="turbo", use_apc=True)
    print(f"  TTFT: {res1['ttft_ms']:.1f} ms | Decode: {res1['decode_tps']:.2f} tok/s | Steps: {res1['steps']} ({res1['accepted_per_step']:.2f} tok/step)")
    print(f"  Agent Output: {repr(res1['generated_text'][:90])}...")

    # Turn 2: Agent Tool Call + Tool Result + Follow-up User Query
    messages.append({"role": "assistant", "content": res1["generated_text"]})
    messages.append({"role": "user", "content": "Tool Result: Exit code 0. Successfully compiled libmetal_engine.dylib with 0 errors. What are the next steps?"})

    print("\n--- [Turn 2: Multi-Turn Tool Result (APC Prefix Cache Hit)] ---")
    res2 = engine.generate(messages, max_tokens=32, mode="turbo", use_apc=True)
    print(f"  Tokens Saved by APC: {res2['tokens_saved']} tokens")
    print(f"  TTFT: {res2['ttft_ms']:.1f} ms (APC Prefix Match)")
    print(f"  Decode: {res2['decode_tps']:.2f} tok/s | Steps: {res2['steps']} ({res2['accepted_per_step']:.2f} tok/step)")
    print(f"  Agent Output: {repr(res2['generated_text'][:90])}...")

    # Turn 3: Second Tool Execution + Verification
    messages.append({"role": "assistant", "content": res2["generated_text"]})
    messages.append({"role": "user", "content": "Tool Result: 5/5 tests passed in 0.04s. Please summarize."})

    print("\n--- [Turn 3: Multi-Turn Final Answer (Deeper APC Hit)] ---")
    res3 = engine.generate(messages, max_tokens=32, mode="turbo", use_apc=True)
    print(f"  Tokens Saved by APC: {res3['tokens_saved']} tokens")
    print(f"  TTFT: {res3['ttft_ms']:.1f} ms (Deeper APC Match)")
    print(f"  Decode: {res3['decode_tps']:.2f} tok/s | Steps: {res3['steps']} ({res3['accepted_per_step']:.2f} tok/step)")
    print(f"  Agent Output: {repr(res3['generated_text'][:90])}...")

    print("\n" + "=" * 65)
    print("  AGENTIC MULTI-TURN SUMMARY")
    print("=" * 65)
    print(f"  Turn 1 TTFT (Cold Prefill):    {res1['ttft_ms']:.1f} ms")
    print(f"  Turn 2 TTFT (APC Hit + Delta): {res2['ttft_ms']:.1f} ms ({res1['ttft_ms']/max(res2['ttft_ms'], 1e-2):.1f}x speedup!)")
    print(f"  Turn 3 TTFT (APC Hit + Delta): {res3['ttft_ms']:.1f} ms ({res1['ttft_ms']/max(res3['ttft_ms'], 1e-2):.1f}x speedup!)")
    print(f"  Total Tokens Saved by APC:     {res2['tokens_saved'] + res3['tokens_saved']} tokens")
    print(f"  APC Radix Cache Stats:         {engine.apc.stats()}")
    print("=" * 65)


if __name__ == "__main__":
    run_agentic_benchmark()
