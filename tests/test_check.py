"""recipe check: the two-card example is clean, and each rule C1 to C9 turns red on its planted control."""

import datetime
import shutil
import tempfile
import unittest
from pathlib import Path

from recipe_kit import compat, hardware, manifest, profiles
from recipe_kit.check import check
from stand_in import SINGLE

TODAY = datetime.date(2026, 11, 26)
MADE_UP_ROW = '\n[[switch]]\nname = "TF_MADE_UP"\nkind = "product"\nreason = "r"\ndoc = "d"\nhardware = ["any"]\n'
ORPHAN_MODE = (
    '[[mode]]\nslot = "split"\nvalue = "fused"\napplies = "any"\nsets = { TF_EXAMPLE_ROW_SPLIT = "0" }\n'
    'receipt = "r"\nexpires = "2027-03-01"\n'
)
# rule -> (text in the clean manifest, its planted replacement, a row appended so only that rule fails)
PLANTS = {
    "C1": (
        'kind = "product"\nreason = "a prompt pass',
        'kind = "product"\nexpires = "2027-01-01"\nreason = "a prompt pass',
    ),
    "C2": ('TF_EXAMPLE_ATTN_LANES = "1" }', 'TF_EXAMPLE_ATTN_LANES = "1", TF_EXTRA = "1" }'),
    "C3": ('added = "2026-11-26"\nexpires = "2027-02-24"', 'added = "2026-11-20"\nexpires = "2026-11-25"'),
    "C4": ('added = "2026-11-26"\nexpires = "2027-02-24"', 'added = "2026-11-26"\nexpires = "2027-02-25"'),
    "C5": ('TF_EXAMPLE_ATTN_LANES = "1" }', 'TF_EXAMPLE_ATTN_LANES = "1", TF_EXAMPLE_ROLL_BATCH = "1" }'),
    "C6": ('mode = { split = "rows" }', 'mode = { split = "rows", prompt = "owner_slots" }'),
    "C7": ("[[mode]]\n", ORPHAN_MODE + "\n[[mode]]\n"),
    "C8": ('TF_EXAMPLE_ATTN_LANES = "1" }', 'TF_EXAMPLE_ATTN_LANES = "1", TF_MADE_UP = "1" }', MADE_UP_ROW),
    "C9": ('kind = "experiment"\nowner', 'kind = "experiment"\nexact = false\nowner'),
}


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "recipe"
        shutil.copytree(SINGLE, self.root, ignore=shutil.ignore_patterns("abba", "soak-sh"))
        self.names = compat.engine_names(SINGLE / "engine")

    def tearDown(self):
        self.tmp.cleanup()

    def rules(self) -> set[str]:
        recipe = manifest.load(self.root)
        hw = hardware.load(recipe.hardware)
        return {finding.rule for finding in check(profiles.load(recipe), set(hw.env), TODAY, self.names)}

    def test_the_example_is_clean(self):
        self.assertEqual(self.rules(), set())

    def test_each_rule_turns_red_on_its_plant(self):
        clean = (self.root / "recipe.toml").read_text()
        for rule, (old, new, *append) in PLANTS.items():
            with self.subTest(rule=rule):
                self.assertIn(old, clean)
                (self.root / "recipe.toml").write_text(clean.replace(old, new, 1) + "".join(append))
                self.assertEqual(self.rules(), {rule})
        (self.root / "recipe.toml").write_text(clean)


if __name__ == "__main__":
    unittest.main()
