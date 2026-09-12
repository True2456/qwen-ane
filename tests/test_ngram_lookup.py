import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.flashnext_ngram import ContextLookup


def test_repeat_predicts_the_continuation():
    c = ContextLookup(g=3)
    c.extend([1, 2, 3, 4, 5, 9, 1, 2, 3])
    # "1 2 3" was followed by 4 the first time, and the trailing copy is not
    # indexed because nothing follows it yet.
    assert c.next_token([]) == 4
    c.extend([4])
    assert c.next_token([]) == 5          # now the key is "2 3 4"


def test_key_spans_the_drafted_tail():
    c = ContextLookup(g=3)
    c.extend([7, 8, 9, 10, 11, 7, 8])
    assert c.next_token([9]) == 10        # key "7 8 9"
    assert c.next_token([9, 10]) == 11    # key "8 9 10"
    assert c.next_token([9, 10, 11]) == 7   # key "9 10 11"


def test_no_match_and_short_context():
    c = ContextLookup(g=3)
    c.extend([4, 4])
    assert c.next_token([]) is None
    c2 = ContextLookup(g=3)
    c2.extend([1, 2, 3, 4])
    assert c2.next_token([99, 98, 97]) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
