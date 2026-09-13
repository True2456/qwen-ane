"""The explicit baseline must survive exporter defaults and inherited settings."""
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
from eval_client import AneClient


class PrefillConfigTests(unittest.TestCase):
    def test_explicit_width_reaches_child(self):
        for inherited in (None, "32"):
            for width in (0, 4, 32):
                with self.subTest(inherited=inherited, width=width):
                    env = {} if inherited is None else {"FLASHNEXT_PREFILL_MIL_K": inherited}
                    child = SimpleNamespace(stdout=io.StringIO('{"ready": true}\n'))
                    with patch.dict(os.environ, env, clear=True), patch(
                            "eval_client.subprocess.Popen", return_value=child) as start:
                        AneClient(prefill_k=width)
                    self.assertEqual(start.call_args.kwargs["env"][
                        "FLASHNEXT_PREFILL_MIL_K"], str(width))


if __name__ == "__main__":
    unittest.main()
