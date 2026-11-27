"""The recipe's own tests, run with the kit and its testkit importable from the repository root:
PYTHONPATH=kit:testkit/src python3 -m unittest discover -s qwen-flash-abliterated-rtx6000x1/tests,
which make check also does for every recipe this repository hosts."""

import datetime
import json
import unittest
from pathlib import Path

from recipe_kit import hardware, manifest, profiles
from recipe_kit.check import check as rules
from tensorfold_testkit.gates import hashes

ROOT = Path(__file__).resolve().parents[1]
OLD_TP1_ARGS = ["--context", "16384", "--parallel", "16"]


def plan() -> profiles.Plan:
    return profiles.load(manifest.load(ROOT))


class Gates(unittest.TestCase):
    def test_the_gate_files_load(self):
        table = hashes.load(ROOT / "gates" / "hashes.json")
        self.assertIn("*", table["expected"])
        spark = hashes.load(ROOT / "gates" / "spark.hashes.json")
        self.assertEqual(spark["prompts"], table["prompts"])
        self.assertEqual(set(spark["expected"]["*"]), set(table["prompts"]))
        self.assertIsInstance(json.loads((ROOT / "gates" / "conformance.json").read_text())["prompts"], dict)

    def test_the_control_reference_holds_the_old_one_gpu_hashes(self):
        control = json.loads((ROOT / "gates" / "control.hashes.json").read_text())
        self.assertEqual(control["control"]["rev"], (ROOT / "config" / "control.rev").read_text().strip())
        self.assertEqual(control["control"]["serve"], OLD_TP1_ARGS)
        for cell, digest in control["greedy"].items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$", cell)


class Weights(unittest.TestCase):
    def test_both_packs_are_fully_checksummed(self):
        for name, count in (("weights", 208), ("control", 13)):
            lines = (ROOT / "config" / f"{name}.sha256").read_text().splitlines()
            self.assertEqual(len(lines), count, name)
            self.assertTrue(all(len(line.split("  ")) == 2 for line in lines), name)


class Profiles_(unittest.TestCase):
    def test_the_default_profile_is_the_old_one_gpu_port(self):
        rendered = profiles.render(plan(), "default", hardware.load("rtx6000x1").env)
        self.assertEqual(rendered.args, OLD_TP1_ARGS)
        self.assertEqual(rendered.ranks, 1)
        self.assertEqual(rendered.env, {"TF_FLASHNEXT_KEPT": "device"})
        self.assertEqual(rendered.experiments, {})

    def test_the_agent_profile_widens_the_window_and_thins_the_streams(self):
        rendered = profiles.render(plan(), "agent", {})
        self.assertEqual(rendered.args[:4], ["--context", "139264", "--parallel", "3"])
        self.assertIn("--prompt-cache-gib", rendered.args)
        optimised = dict(profiles.render(plan(), "optimised", {}).env)
        del optimised["TF_FLASHNEXT_CHUNK_SOLO"]
        self.assertEqual(rendered.env, optimised)

    def test_the_ship_profile_sets_every_expected_lever(self):
        rendered = profiles.render(plan(), "optimised", {})
        self.assertEqual(rendered.args[:4], ["--context", "16384", "--parallel", "12"])
        self.assertEqual(
            rendered.env,
            {
                "TF_FLASHNEXT_ATTN_LANES": "1",
                "TF_FLASHNEXT_CHUNK_SOLO": "8192",
                "TF_FLASHNEXT_GDN_SPLIT": "1",
                "TF_FLASHNEXT_KEPT": "host",
                "TF_FLASHNEXT_READOUT_DECODE": "1",
                "TF_FLASHNEXT_ROLL_BATCH": "1",
                "TF_FLASHNEXT_ROUTE_FUSED": "1",
            },
        )

    def test_the_one_experiment_is_measured_by_its_own_arm(self):
        rendered = profiles.render(plan(), "drafts-greedy", {})
        self.assertEqual(rendered.experiments, {"TF_FLASHNEXT_DRAFTS_GREEDY": "12"})
        self.assertEqual(rendered.env, {**rendered.experiments, "TF_FLASHNEXT_KEPT": "device"})

    def test_every_switch_row_and_profile_passes_the_rules(self):
        today = datetime.date(2026, 11, 27)
        self.assertEqual(rules(plan(), set(), today), [])


if __name__ == "__main__":
    unittest.main()
