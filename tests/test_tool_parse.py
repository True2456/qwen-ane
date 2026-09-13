import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from flashnext_server import split_reply

RAW = """The user wants to list files. Simple task - I'll run ls.
</think>

I'll list the files.

<tool_call>
<function=run_shell>
<parameter=cmd>
ls -la
</parameter>
<parameter=limit>
20
</parameter>
</function>
</tool_call>"""


def test_reply_with_a_call():
    content, reasoning, calls = split_reply(RAW)
    assert reasoning.startswith("The user wants to list files")
    assert content == "I'll list the files."
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "run_shell"
    import json
    assert json.loads(fn["arguments"]) == {"cmd": "ls -la", "limit": 20}


def test_plain_reply():
    content, reasoning, calls = split_reply("Thinking about it.\n</think>\n\nFour")
    assert content == "Four" and reasoning == "Thinking about it." and not calls


def test_no_think_tag():
    content, reasoning, calls = split_reply("Just an answer.")
    assert content == "Just an answer." and reasoning is None and not calls



MALFORMED = """Let me run ls.
</think>

<tool_call>
<function=run_shell>
{"cmd": "ls -la"}
</parameter>
</function>
</tool_call>"""


def test_model_puts_json_where_parameters_should_be():
    import json
    _, _, calls = split_reply(MALFORMED)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "run_shell"
    assert json.loads(calls[0]["function"]["arguments"]) == {"cmd": "ls -la"}



TOOLS = [{"type": "function", "function": {
    "name": "run_shell", "description": "Run a shell command",
    "parameters": {"type": "object",
                   "properties": {"cmd": {"type": "string"}},
                   "required": ["cmd"]}}}]


def test_wrong_function_name_is_repaired():
    _, _, calls = split_reply(MALFORMED.replace("run_shell", "cmd"), TOOLS)
    assert calls[0]["function"]["name"] == "run_shell"


def test_a_declared_name_is_left_alone():
    _, _, calls = split_reply(MALFORMED, TOOLS)
    assert calls[0]["function"]["name"] == "run_shell"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
