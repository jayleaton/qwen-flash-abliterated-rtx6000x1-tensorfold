import json
import tempfile
import unittest
from pathlib import Path

from tensorfold_testkit import cli
from tensorfold_testkit.client import Client
from tensorfold_testkit.gates import hashes
from tensorfold_testkit.mock.server import Mock

TABLE = {
    "schema": 1,
    "tokens": 32,
    "greedy_seed": 1,
    "sampled": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    "seeds": [1, 2],
    "prompts": {
        "a": {"kind": "chat", "prompt": "Say hello.", "thinking": False},
        "b": {"kind": "completion", "prompt": "Once upon a time"},
    },
    "expected": {
        "*": {
            "a": {"greedy": "0123456789ab", "sampled": ["1123456789ab", "2123456789ab"]},
            "b": {"greedy": "3123456789ab"},
        }
    },
}


class Gates(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.table = Path(self.dir.name) / "hashes.json"
        self.table.write_text(json.dumps(TABLE))

    def tearDown(self):
        self.dir.cleanup()

    def test_every_gate_green_clean_and_red_with_its_fault(self):
        args = cli.parser().parse_args(["selftest", "--hashes", str(self.table)])
        clean, planted, plants = cli.selftest(cli.inputs(args, samples=True))
        ran = {r.gate for r in clean if not r.skipped}
        self.assertEqual(ran, set(cli.ORDER) - {"conformance"})
        self.assertTrue(all(r.passed for r in clean), "\n".join(map(str, clean)))
        self.assertTrue(all(not planted[g].passed for g in ran), "\n".join(map(str, planted.values())))
        self.assertEqual(plants, {})

    def test_a_reply_without_a_hash_fails(self):
        with Mock() as mock:
            mock.reply = lambda body, prompt, together: {"text": "x", "sha256": "", "tokens": 1, "finish": "stop"}
            result = hashes.hashes(Client(mock.url), hashes.load(self.table), None, quick=True)
        self.assertFalse(result.passed)
        self.assertIn("no token hash", result.line)

    def test_every_gate_can_be_skipped(self):
        args = cli.parser().parse_args(["gate", "--mock", "--skip", ",".join(cli.ORDER)])
        self.assertEqual(cli.chosen(args), [])


if __name__ == "__main__":
    unittest.main()
