"""recipe kit on the two-card example: a bundle passes its SHA256SUMS, fails it when a file changes, and runs the
bench end to end on the stand-in from inside the bundle."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stand_in import KIT, SINGLE
from tensorfold_testkit import receipt


class Bundle(unittest.TestCase):
    def test_a_bundle_verifies_and_runs_the_bench_on_the_stand_in(self):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            engine = scratch / "engine" / "zig-out" / "native" / "bin"
            engine.mkdir(parents=True)
            (engine / "tensorfold-native").write_text("#!/bin/sh\n")
            recipe = [str(KIT / "bin" / "recipe"), "kit", "--recipe", str(SINGLE), "--kit-dev"]
            made = subprocess.run(
                [*recipe, str(scratch / "out"), str(scratch / "engine"), "--tree", str(SINGLE / "engine")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(made.returncode, 0, made.stderr)
            root = next(p for p in (scratch / "out").iterdir() if p.is_dir())
            shutil.unpack_archive(root.with_name(root.name + ".tar.gz"), scratch / "unpacked")
            bundle = scratch / "unpacked" / root.name
            run = str(bundle / "kit" / "bin" / "recipe")
            check = subprocess.run([run, "kit", "--verify", str(bundle)], capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stdout)
            out = subprocess.run(
                [
                    run,
                    "bench",
                    "--recipe",
                    str(bundle / "recipe"),
                    "--kit-dev",
                    "base",
                    "fast",
                    str(bundle / "engine"),
                    "--mock",
                    "--cells",
                    "decode.code.x1,cold.1024",
                    "--out",
                    str(scratch / "cells"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("fast against base, bar 3.0%: no difference", out.stdout)
            doc = receipt.verify(scratch / "cells" / "receipt.json")
            self.assertEqual(doc["order"], ["base-control", "base", "fast", "fast", "base", "base-control"])
            self.assertEqual(len({a["binary"]["md5"] for a in doc["arms"].values()}), 1)
            (bundle / "recipe" / "recipe.toml").write_text("tampered\n")
            check = subprocess.run([run, "kit", "--verify", str(bundle)], capture_output=True, text=True)
            self.assertEqual(check.returncode, 1)
            self.assertIn("recipe/recipe.toml: checksum differs", check.stdout)
            self.assertEqual(json.loads((bundle / "engine" / "recipe-names.json").read_text())["pin"], None)


if __name__ == "__main__":
    unittest.main()
