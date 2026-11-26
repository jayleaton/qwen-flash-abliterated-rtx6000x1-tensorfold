"""Every gate on the stand-in replaying the two-card example's recorded run: green clean, red on its own planted
fault, and every conformance.py plant red."""

import json
import unittest

from stand_in import SINGLE
from tensorfold_testkit import cli


class Selftest(unittest.TestCase):
    def test_green_on_the_recording_and_red_on_each_fault(self):
        args = cli.parser().parse_args(
            [
                "selftest",
                "--hashes",
                str(SINGLE / "gates" / "hashes.json"),
                "--conformance",
                str(SINGLE / "gates" / "conformance.json"),
                "--tree",
                str(SINGLE / "engine"),
            ]
        )
        replies = json.loads((SINGLE / "recorded.json").read_text())["replies"]
        clean, planted, plants = cli.selftest(cli.inputs(args, samples=True), replies)
        self.assertEqual(
            [(r.gate, r.passed, r.skipped) for r in clean],
            [(g, True, False) for g in cli.ORDER],
            "\n".join(map(str, clean)),
        )
        self.assertEqual(
            {g for g, r in planted.items() if not r.passed}, set(cli.ORDER), "\n".join(map(str, planted.values()))
        )
        self.assertEqual(set(plants), set(cli.conformance.GATES))
        self.assertNotIn(0, plants.values())


if __name__ == "__main__":
    unittest.main()
