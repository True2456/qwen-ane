"""PLE row gather: cache and threads must not change the table's contents."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path("/Users/true/models/Qwen3.8-Flash-Next")


def _need_checkpoint():
    if not (BASE / "model.safetensors.index.json").is_file():
        print("skip: Flash-Next checkpoint not at", BASE)
        return False
    return True


def test_lookup_matches_serial_pread():
    if not _need_checkpoint():
        return
    from runtime.flashnext_ngram import NgramRows

    os.environ["FLASHNEXT_PLE_THREADS"] = "16"
    os.environ["FLASHNEXT_PLE_CACHE"] = "1"
    fast = NgramRows(BASE)
    os.environ["FLASHNEXT_PLE_THREADS"] = "1"
    os.environ["FLASHNEXT_PLE_CACHE"] = "0"
    serial = NgramRows(BASE)
    rng = np.random.default_rng(13)
    ids = rng.integers(0, fast.total, size=(32, 16), dtype=np.int64)
    # Repeat the first block so the cache path is exercised too.
    ids = np.concatenate([ids, ids[:4]], axis=0)
    for row in ids:
        got = fast.lookup(row)
        want = serial.lookup(row)
        np.testing.assert_array_equal(got, want)
    assert fast.hits > 0
    assert serial.hits == 0
    fast.close()
    serial.close()


def test_hash_and_step_stable_on_prose_prefix():
    if not _need_checkpoint():
        return
    from tokenizers import Tokenizer
    from runtime.flashnext_ngram import CpuPLE, NgramRows

    os.environ["FLASHNEXT_PLE_THREADS"] = "16"
    os.environ["FLASHNEXT_PLE_CACHE"] = "1"
    rows = NgramRows(BASE)
    cpu = CpuPLE(rows, 1)
    tk = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    tokens = tk.encode(
        Path(__file__).resolve().parents[1].joinpath("eval/prose.txt").read_text(),
        add_special_tokens=False,
    ).ids[:32]
    hidden = np.zeros((1, 1, 10240), np.float32)
    out = []
    hashes = []
    for tok in tokens:
        hashes.append(cpu.hash_ids(tok).copy())
        out.append(cpu.step(hidden, tok).copy())
    rows.close()

    os.environ["FLASHNEXT_PLE_THREADS"] = "1"
    os.environ["FLASHNEXT_PLE_CACHE"] = "0"
    rows2 = NgramRows(BASE)
    cpu2 = CpuPLE(rows2, 1)
    hidden2 = np.zeros((1, 1, 10240), np.float32)
    for i, tok in enumerate(tokens):
        np.testing.assert_array_equal(cpu2.hash_ids(tok), hashes[i])
        got = cpu2.step(hidden2, tok)
        np.testing.assert_array_equal(got, out[i])
    rows2.close()


if __name__ == "__main__":
    test_lookup_matches_serial_pread()
    print("ok test_lookup_matches_serial_pread")
    test_hash_and_step_stable_on_prose_prefix()
    print("ok test_hash_and_step_stable_on_prose_prefix")
