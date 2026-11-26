"""recipe soak on the stand-in: green clean, red with each planted fault (an error, a slot that never frees, tokens
that drift, memory under the floor), and the whole flow through the kit to a receipt."""

import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stand_in import KIT, SINGLE
from tensorfold_testkit import receipt, soak
from tensorfold_testkit.cli import recorded
from tensorfold_testkit.gates import hashes
from tensorfold_testkit.mock.server import Mock

WANT = {
    None: [],
    "error": "failed requests",
    "leak": "slots not drained",
    "drift": "probe hashes changed",
    "memory": "under the floor",
}


class Soak(unittest.TestCase):
    def test_green_clean_and_red_with_each_fault(self):
        table = hashes.load(SINGLE / "gates" / "hashes.json")
        cold = {size: row["path"] for size, row in table["cold"].items()}
        options = soak.Options(minutes=0.05, probe_s=60, step_s=(0.2, 0.4), poll_s=0.2, drain_s=1)

        def run(fault, out):
            with Mock("m", recorded(table, None), fault=fault, cache_words=0, tps=2000, prefill_tps=1e5) as mock:
                record = soak.run(mock.url, "m", SINGLE / "engine/tools", cold, out, options, soak.mock_meter(mock.url))
            return soak.verdict(record, soak.Floors(4.0), hashes.bench_expected(table, None))

        with tempfile.TemporaryDirectory() as scratch, ThreadPoolExecutor(len(WANT)) as pool:
            got = dict(zip(WANT, pool.map(run, WANT, [Path(scratch) / str(f) for f in WANT]), strict=True))
        self.assertEqual(got[None]["verdict"], "STABLE", got[None]["problems"])
        for fault, why in list(WANT.items())[1:]:
            self.assertEqual(got[fault]["verdict"], "UNSTABLE", fault)
            self.assertTrue(any(why in p for p in got[fault]["problems"]), (fault, got[fault]["problems"]))

    def test_repeated_cold_probes_stay_cold_with_the_prompt_cache_on(self):
        table = hashes.load(SINGLE / "gates" / "hashes.json")
        cold = {size: row["path"] for size, row in table["cold"].items()}
        options = soak.Options(minutes=0.08, probe_s=1, step_s=(0.2, 0.4), poll_s=0.2, drain_s=1)
        mock = Mock("m", recorded(table, None), cache_words=10**6, tps=2000, prefill_tps=1e5)
        with tempfile.TemporaryDirectory() as scratch, mock:
            record = soak.run(mock.url, "m", SINGLE / "engine/tools", cold, Path(scratch), options)
        verdict = soak.verdict(record, soak.Floors(None), hashes.bench_expected(table, None))
        self.assertGreaterEqual(len(record["probes"]), 2)
        self.assertEqual(verdict["verdict"], "STABLE", verdict["problems"])

    def test_recipe_soak_on_the_stand_in_writes_a_receipt(self):
        with tempfile.TemporaryDirectory() as scratch:
            out = subprocess.run(
                [
                    *(str(KIT / "bin" / "recipe"), "soak", "--recipe", str(SINGLE), "--kit-dev", "base"),
                    *(str(SINGLE / "engine"), "--mock", "--minutes", "0.02", "--out", scratch),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("STABLE: 2 probes", out.stdout)
            doc = receipt.verify(Path(scratch) / "receipt.json")
        self.assertEqual((doc["kind"], doc["soak"]["drained"], doc["soak"]["canary"]), ("soak", True, "391"))
        self.assertEqual(doc["limits"]["memory_gib"], 4.0)


if __name__ == "__main__":
    unittest.main()
