"""recipe bench's verdict, proved without a GPU: abba/ holds the bench clients' files of a synthetic R PF PK PK PF R
window, and decide() must replay it to the means, changes and verdicts worked out by hand below."""

import unittest

from stand_in import SINGLE
from tensorfold_testkit import bench
from tensorfold_testkit.gates import hashes

CELLS = ["decode.code.x1", "decode.chat.x1", "decode.code.x16", "decode.chat.x16", "cold.1024", "cold.2048"]
# each arm's two samples, cell by cell: PF 200 202, PK 206 208, R 180 181 on decode.code.x1, and so on
MEANS = {
    "R": [180.5, 120.5, 901, 700.5, 6000, 7000],
    "PF": [201, 141, 1002, 802, 8000, 9000],
    "PK": [207, 144, 1012, 808, 8000, 9000],
}


class Replay(unittest.TestCase):
    def setUp(self):
        dirs = sorted((SINGLE / "abba").iterdir(), key=lambda d: int(d.name.rsplit("-", 1)[1]))
        self.samples = [bench.read(d, CELLS, d.name.rsplit("-", 1)[0]) for d in dirs]
        self.expected = hashes.bench_expected(hashes.load(SINGLE / "gates" / "hashes.json"), None)

    def test_the_window_replays_to_its_numbers_and_verdict(self):
        verdict = bench.decide(self.samples, "PF", "PK", 0.015, "R", self.expected)
        self.assertEqual(verdict["means"], {arm: dict(zip(CELLS, got, strict=True)) for arm, got in MEANS.items()})
        self.assertEqual(f"{verdict['change']['decode.code.x1'] * 100:+.2f}", "+2.99")
        self.assertEqual((verdict["verdict"], verdict["problems"]), ("better", []))
        self.assertEqual(bench.decide(self.samples, "PF", "PK", 0.03, "R", self.expected)["verdict"], "no difference")

    def test_a_reply_off_the_recipes_hashes_is_named(self):
        self.samples[2].hashes["decode.code.x16"] = "0" * 12
        verdict = bench.decide(self.samples, "PF", "PK", 0.015, "R", self.expected)
        self.assertEqual(verdict["verdict"], "tokens differ")
        self.assertEqual(
            verdict["problems"], [f"PK decode.code.x16: tokens {'0' * 12}, expected {self.expected['decode.code']}"]
        )


if __name__ == "__main__":
    unittest.main()
