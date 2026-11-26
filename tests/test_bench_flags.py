"""Golden: recipe bench starts every arm exactly as recipe serve does, adding no flag or env of its own, and its
receipt names the prompt cache each arm serves with."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from recipe_kit import bench, cli, hardware, manifest, profiles, serve
from stand_in import MULTI, SINGLE, single_engine


class BenchFlags(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.engine = single_engine(self.base)
        self.model = self.base / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def served(self, target: str) -> list[dict]:
        out = io.StringIO()
        argv = ["serve", "--recipe", str(SINGLE), "--kit-dev", "--today", "2026-11-26", "--dry-run", target]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main([*argv, str(self.engine), str(self.model)]), 0)
        return json.loads(out.getvalue())

    def test_every_arm_starts_as_it_serves(self):
        recipe = manifest.load(SINGLE)
        plan, hw, caps = profiles.load(recipe), hardware.load(recipe.hardware), {}
        binary = self.engine / recipe.engine(None).build.outputs[0]
        for target in [*plan.profiles, *plan.arms]:
            with self.subTest(target=target):
                rendered = profiles.render(plan, target, hw.env)
                spec = bench.arm(recipe, rendered, hw, self.engine, self.model, caps, recipe.port, None, None)
                started = [{"argv": r.argv, "env": {k: v for k, v in r.env.items() if k != "_"}} for r in spec.ranks]
                launches = serve.ranks(recipe, rendered, binary, self.model, [], None, recipe.port, os.environ)
                served = [{"argv": r.argv, "env": r.env} for r in launches]
                self.assertEqual(started, served)
                self.assertEqual(spec.settings["prompt_cache_gib"], "engine default")
                if target == "fast":
                    self.assertEqual(served, [{"argv": r["argv"], "env": r["env"]} for r in self.served(target)])

    def test_the_receipt_names_each_arms_prompt_cache(self):
        recipe = manifest.load(MULTI)
        plan = profiles.load(recipe)
        caches = {t: bench.prompt_cache(profiles.render(plan, t, {})) for t in ("cuda", "rtx", "metal")}
        self.assertEqual(caches, {"cuda": "24", "rtx": "8", "metal": "engine default"})


if __name__ == "__main__":
    unittest.main()
