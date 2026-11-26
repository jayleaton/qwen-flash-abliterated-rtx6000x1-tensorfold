"""The recipe's own tests, run with the kit's testkit importable:
PYTHONPATH=KIT/testkit/src python3 -m unittest discover -s tests, where KIT is the kit checkout [kit] rev names."""

import json
import unittest
from pathlib import Path

from tensorfold_testkit.gates import hashes

ROOT = Path(__file__).resolve().parents[1]


class Gates(unittest.TestCase):
    def test_the_gate_files_load(self):
        self.assertIn("*", hashes.load(ROOT / "gates" / "hashes.json")["expected"])
        self.assertIsInstance(json.loads((ROOT / "gates" / "conformance.json").read_text())["prompts"], dict)


if __name__ == "__main__":
    unittest.main()
