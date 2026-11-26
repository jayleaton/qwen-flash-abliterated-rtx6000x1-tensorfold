import json
import tempfile
import unittest
from pathlib import Path

from tensorfold_testkit import bench, receipt


def samples(a: float, b: float, control_end: float = 100.0, b_hash: str = "aa" * 6) -> list[bench.Sample]:
    def s(arm, tps, sha="aa" * 6):
        return bench.Sample(arm, {"decode.code.x1": tps, "cold.8192": 1000.0}, {"decode.code.x1": sha})

    return [s("C", 100), s("A", a), s("B", b, b_hash), s("B", b, b_hash), s("A", a), s("C", control_end)]


class Salt(unittest.TestCase):
    def test_each_salt_is_a_new_prefix_of_the_prompts_own_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "ids.json"
            prompt.write_text(json.dumps(list(range(1000, 1300))))
            runs = [json.loads(bench.salted(prompt, salt, Path(tmp)).read_text()) for salt in ("a-01", "a-02", "a-01")]
        self.assertEqual([r[bench.SALT_TOKENS :] for r in runs], [list(range(1000, 1300))] * 3)
        self.assertTrue(all(set(r[: bench.SALT_TOKENS]) <= set(range(1000, 1300)) for r in runs))
        self.assertNotEqual(runs[0][: bench.SALT_TOKENS], runs[1][: bench.SALT_TOKENS])
        self.assertEqual(runs[0], runs[2])


class Decide(unittest.TestCase):
    def test_each_verdict(self):
        cases = {
            "no difference": samples(100, 101),
            "better": samples(100, 104),
            "worse": samples(100, 96),
            "void": samples(100, 104, control_end=105),
            "tokens differ": samples(100, 104, b_hash="bb" * 6),
        }
        for word, got in cases.items():
            self.assertEqual(bench.decide(got, "A", "B", 0.03, "C")["verdict"], word)

    def test_order(self):
        self.assertEqual(bench.order("A", "B", 2, "C"), ["C", "A", "B", "B", "A", "C"])
        self.assertEqual(bench.order("A", "B", 1, None), ["A", "B"])
        with self.assertRaises(bench.BenchError):
            bench.order("A", "B", 3, None)


class Receipt(unittest.TestCase):
    def test_a_receipt_naming_a_path_or_an_address_is_refused(self):
        head = {"kit": "dev:x", "dev": True, "recipe": "r"}
        with tempfile.TemporaryDirectory() as d:
            path = receipt.write(Path(d) / "r.json", "gate", head, {"results": []})
            self.assertEqual(receipt.verify(path)["kind"], "gate")
            for leak in (
                "/".join(["", "home", "someone", "model"]),
                ".".join(["10", "0", "0", "7"]),
                "http" + "://x.org",
            ):
                with self.assertRaises(receipt.ReceiptError):
                    receipt.write(Path(d) / "r.json", "gate", head, {"results": [leak]})


if __name__ == "__main__":
    unittest.main()
