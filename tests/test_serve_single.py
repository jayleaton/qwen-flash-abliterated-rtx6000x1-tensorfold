"""Golden: recipe serve starts each rank of every profile and arm of the two-card example with exactly the argv and
environment in serve.json, the kit adding no flag of its own; an unknown switch is refused before anything starts."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stand_in import KIT, SINGLE, launch, placeholders, single_engine

GOLDEN = json.loads((SINGLE / "serve.json").read_text())
SERVE = [str(KIT / "bin" / "recipe"), "serve", "--kit-dev", "--today", "2026-11-26"]


class ServeSingle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.engine = single_engine(self.base)
        self.model = self.base / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def started(self, target: str) -> dict:
        command = [*SERVE, "--recipe", str(SINGLE), target, str(self.engine), str(self.model)]
        return placeholders(launch(command, self.base / f"dump-{target}"), self.engine, self.model)

    def test_every_target_starts_as_the_golden(self):
        with ThreadPoolExecutor(len(GOLDEN)) as pool:
            runs = dict(zip(GOLDEN, pool.map(self.started, GOLDEN), strict=True))
        for target, ranks in runs.items():
            with self.subTest(target=target):
                self.assertEqual(ranks, GOLDEN[target])

    def test_unknown_switch_is_refused_before_anything_starts(self):
        recipe = self.base / "planted"
        shutil.copytree(SINGLE, recipe)
        text = (recipe / "recipe.toml").read_text()
        planted = 'TF_EXAMPLE_ATTN_LANES = "1", TF_EXAMPLE_NOT_A_SWITCH = "1" }'
        (recipe / "recipe.toml").write_text(text.replace('TF_EXAMPLE_ATTN_LANES = "1" }', planted, 1))
        dump = self.base / "planted-dump"
        dump.mkdir()
        env = {"PATH": os.environ["PATH"], "HOME": str(dump), "DUMP_DIR": str(dump)}
        command = [*SERVE, "--recipe", str(recipe), "base", str(self.engine), str(self.model)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("TF_EXAMPLE_NOT_A_SWITCH", result.stderr)
        self.assertEqual(list(dump.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
