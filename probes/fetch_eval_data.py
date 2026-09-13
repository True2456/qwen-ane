#!/usr/bin/env python3
"""Pull MMLU and GSM8K into eval/ as JSONL, so runs are reproducible offline.

Uses the Hugging Face datasets server, which returns rows as JSON and needs
nothing but `requests`. The alternative is a parquet reader, and adding one to
the inference virtualenv for the sake of a benchmark is not worth it.

    ~/.rindi/venvs/coreai/bin/python probes/fetch_eval_data.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval"
API = "https://datasets-server.huggingface.co/rows"


def pull(dataset: str, config: str, split: str, want: int, dest: Path) -> int:
    rows = []
    while len(rows) < want:
        n = min(100, want - len(rows))
        r = requests.get(API, params={"dataset": dataset, "config": config,
                                      "split": split, "offset": len(rows),
                                      "length": n}, timeout=60)
        r.raise_for_status()
        got = r.json().get("rows", [])
        if not got:
            break
        rows.extend(x["row"] for x in got)
        print(f"  {dataset} {len(rows)}/{want}", flush=True)
        time.sleep(0.2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def main() -> int:
    n_mmlu = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
    n_gsm = int(sys.argv[2]) if len(sys.argv) > 2 else 1319
    # MMLU's dev split is the canonical source of few-shot examples, five a
    # subject, so the prompt matches what every other harness shows the model.
    print(f"mmlu dev -> {OUT / 'mmlu_dev.jsonl'}")
    pull("cais/mmlu", "all", "dev", 285, OUT / "mmlu_dev.jsonl")
    print(f"mmlu test -> {OUT / 'mmlu_test.jsonl'}")
    pull("cais/mmlu", "all", "test", n_mmlu, OUT / "mmlu_test.jsonl")
    print(f"gsm8k test -> {OUT / 'gsm8k_test.jsonl'}")
    pull("openai/gsm8k", "main", "test", n_gsm, OUT / "gsm8k_test.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
