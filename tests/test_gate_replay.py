"""recipe gate, proved without a GPU against the stand-in: the hashes table equals the recorded run's replies, and a
quick gate passes through the kit; every gate green clean and red planted is in full_gate.py (make check-full)."""

import json
import subprocess
import unittest

from stand_in import KIT, SINGLE
from tensorfold_testkit.gates import hashes


class HashesTable(unittest.TestCase):
    def test_every_expected_hash_is_the_recorded_runs(self):
        table = hashes.load(SINGLE / "gates" / "hashes.json")
        replies = json.loads((SINGLE / "recorded.json").read_text())["replies"]
        for name, row in table["expected"]["*"].items():
            prompt = table["prompts"][name]["prompt"]
            greedy = [r["token_sha"] for r in replies if r["prompt"] == prompt and not r["temperature"]]
            sampled = [r["token_sha"] for r in replies if r["prompt"] == prompt and r["temperature"]]
            self.assertEqual((greedy, sampled), ([row["greedy"]], row["sampled"]))


class QuickGate(unittest.TestCase):
    def test_quick_gate_through_the_kit(self):
        out = subprocess.run(
            [str(KIT / "bin" / "recipe"), "gate", "--recipe", str(SINGLE), "--kit-dev", "--quick", "--mock"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("PASS hashes: 2/2 equal (greedy only)", out.stdout)


if __name__ == "__main__":
    unittest.main()
