"""recipe new: a fresh recipe from the template passes recipe check and its own test stub, for one engine or several."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from recipe_kit import cli, manifest, profiles
from stand_in import KIT


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = cli.main(list(argv))
    return code, out.getvalue()


class New(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_new_recipe_passes_check_and_its_stub(self):
        dest = self.base / "my-recipe"
        self.assertEqual(run("new", str(dest))[0], 0)
        code, out = run("check", str(dest), "--kit-dev")
        self.assertEqual((code, out.split(",")[0]), (0, "recipe check: 0 finding(s) over 1 profiles and arms"))
        self.assertEqual(manifest.load(dest).name, "my-recipe")
        env = {**os.environ, "PYTHONPATH": str(KIT / "testkit" / "src")}
        stub = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
        result = subprocess.run(stub, cwd=dest, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(run("new", str(dest))[0], 2)

    def test_several_engines_on_one_rank(self):
        dest = self.base / "two"
        self.assertEqual(run("new", str(dest), "--hardware", "gb10x1", "--engines", "cuda,metal")[0], 0)
        self.assertEqual(run("check", str(dest), "--kit-dev")[0], 0)
        recipe = manifest.load(dest)
        self.assertEqual((list(recipe.engines), recipe.engine("metal").backend), (["cuda", "metal"], "metal"))
        plan = profiles.load(recipe)
        self.assertEqual({t: profiles.engine_of(plan, t) for t in plan.profiles}, {"default": "cuda", "metal": "metal"})
        self.assertNotIn("--tp", profiles.render(plan, "metal", {}).args)


if __name__ == "__main__":
    unittest.main()
