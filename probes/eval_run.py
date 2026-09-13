#!/usr/bin/env python3
"""MMLU and GSM8K over either backend, with identical prompts on both.

    ~/.rindi/venvs/coreai/bin/python probes/eval_run.py \
        --task mmlu --backend ane --n 600 --out eval/results/mmlu_ane.json

MMLU is scored by likelihood: five shots from the dev split of the same
subject, then compare the logprob of " A", " B", " C" and " D" at the final
position. One forward pass a question, no generation, so formatting cannot
cost the model a point.

GSM8K is generative and greedy, eight shots, stopping at the next "Question:".
The shots come from the head of the test split because the datasets server
rate-limited the train split; both backends see the same eight, and those
eight are excluded from scoring.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import eval_client

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
LETTERS = ("A", "B", "C", "D")


def _jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _mc_block(row: dict, answer: str | None) -> str:
    body = row["question"].strip() + "\n"
    for letter, choice in zip(LETTERS, row["choices"]):
        body += f"{letter}. {choice}\n"
    body += "Answer:"
    return body + (f" {answer}\n\n" if answer else "")


def run_mmlu(client, n: int, shots: int) -> dict:
    dev = defaultdict(list)
    for row in _jsonl(EVAL / "mmlu_dev.jsonl"):
        dev[row["subject"]].append(row)
    test = _jsonl(EVAL / "mmlu_test.jsonl")[:n]
    per_subject = defaultdict(lambda: [0, 0])
    records, correct = [], 0
    t0 = time.perf_counter()
    for i, row in enumerate(test):
        subject = row["subject"]
        head = (f"The following are multiple choice questions (with answers) "
                f"about {subject.replace('_', ' ')}.\n\n")
        prompt = head + "".join(
            _mc_block(s, LETTERS[s["answer"]]) for s in dev[subject][:shots])
        prompt += _mc_block(row, None)
        lp = client.score(prompt, [f" {c}" for c in LETTERS])
        pred = max(lp, key=lp.get).strip()
        gold = LETTERS[row["answer"]]
        ok = pred == gold
        correct += ok
        per_subject[subject][0] += ok
        per_subject[subject][1] += 1
        records.append({"i": i, "pred": pred, "gold": gold, "ok": ok})
        if (i + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(test)}  acc {correct / (i + 1):.3f}  "
                  f"{el / (i + 1):.2f}s a question", flush=True)
    return {"task": "mmlu", "n": len(test), "shots": shots,
            "correct": correct, "accuracy": correct / max(len(test), 1),
            "scoring": "likelihood(A/B/C/D)",
            "seconds": time.perf_counter() - t0,
            "per_subject": {k: v[0] / v[1] for k, v in per_subject.items()},
            "records": records}


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str) -> str | None:
    hits = _NUM.findall(text.replace(",", ""))
    return hits[-1].rstrip(".") if hits else None


def run_gsm8k(client, n: int, shots: int) -> dict:
    rows = _jsonl(EVAL / "gsm8k_test.jsonl")
    fewshot, test = rows[:shots], rows[shots:shots + n]
    prefix = "".join(f"Question: {r['question'].strip()}\n"
                     f"Answer: {r['answer'].strip()}\n\n" for r in fewshot)
    records, correct = [], 0
    t0 = time.perf_counter()
    for i, row in enumerate(test):
        prompt = prefix + f"Question: {row['question'].strip()}\nAnswer:"
        out = client.gen(prompt, max_new=320, stop=["\nQuestion:", "\n\n"])
        pred = _last_number(out)
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        ok = pred is not None and pred == gold
        correct += ok
        records.append({"i": i, "pred": pred, "gold": gold, "ok": ok})
        if (i + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(test)}  acc {correct / (i + 1):.3f}  "
                  f"{el / (i + 1):.1f}s a question", flush=True)
    return {"task": "gsm8k", "n": len(test), "shots": shots,
            "correct": correct, "accuracy": correct / max(len(test), 1),
            "scoring": "greedy generation, last number",
            "seconds": time.perf_counter() - t0, "records": records}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mmlu", "gsm8k"], required=True)
    ap.add_argument("--backend", choices=["ane", "mlx"], default="ane")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--prefill-k", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    shots = a.shots if a.shots is not None else (5 if a.task == "mmlu" else 8)

    t0 = time.perf_counter()
    client = eval_client.make(a.backend, ctx=a.ctx, spec=a.spec,
                              prefill_k=a.prefill_k)
    print(f"  {a.backend} ready in {time.perf_counter() - t0:.0f}s "
          f"{client.info}", flush=True)
    try:
        res = (run_mmlu if a.task == "mmlu" else run_gsm8k)(client, a.n, shots)
    finally:
        client.close()
    res["backend"] = a.backend
    print(f"\n{a.task} {a.backend}: {res['correct']}/{res['n']} = "
          f"{res['accuracy']:.4f} in {res['seconds']:.0f}s", flush=True)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1))
        print(f"  wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
