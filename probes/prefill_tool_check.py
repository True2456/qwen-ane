"""Replay the 324-token tool regression without executing the generated tool."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_client import AneClient
from tools.flashnext_server import split_reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-k", type=int, default=32)
    args = parser.parse_args()
    request = json.loads((ROOT / "eval/prefill_tool.json").read_text())
    client = AneClient(prefill_k=args.prefill_k, quiet=False)
    try:
        print(json.dumps(client.info), flush=True)
        assert client.info["prefill_k"] == args.prefill_k, client.info
        result = client._rpc(request)
        print(json.dumps(result, indent=2), flush=True)
        assert result["prompt_tokens"] == 324, result["prompt_tokens"]
        assert "<function=run_shell>" in result["text"], result["text"]
        _, _, calls = split_reply(result["text"], request["tools"])
        assert any(call["function"]["name"] == "run_shell" and
                   json.loads(call["function"]["arguments"]).get("cmd") == "ls -la"
                   for call in calls), calls
        print("PASS: run_shell(cmd='ls -la')", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
