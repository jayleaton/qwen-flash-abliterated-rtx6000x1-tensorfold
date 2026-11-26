import subprocess
import tempfile
import unittest
from pathlib import Path

from recipe_kit import build, engine, manifest, revision

HEAD = (
    '[recipe]\nname = "t"\nmodel = "org/m"\nmodel_type = "m"\nformat = "bf16"\nships = []\n\n'
    f'[kit]\nrev = "{"c" * 40}"\ntestkit = ">=0.1,<1"\n\n[weights]\nrev = "r"\nsha256 = "w.sha256"\n\n'
    '[hardware]\nprofile = "rtx6000x2"\n\n'
)
GIT_ID = ["-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *GIT_ID, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class Fixture:
    """An upstream repo with one commit, two patches on top of it, and a recipe whose series holds them."""

    def __init__(self, base: Path):
        self.upstream = base / "upstream"
        self.recipe = base / "recipe"
        self.upstream.mkdir()
        git(self.upstream, "init", "-q")
        (self.upstream / "engine.txt").write_text("one\n")
        git(self.upstream, "add", ".")
        git(self.upstream, "commit", "-qm", "base")
        self.rev = git(self.upstream, "rev-parse", "HEAD")
        for text in ("two\n", "three\n"):
            with (self.upstream / "engine.txt").open("a") as handle:
                handle.write(text)
            git(self.upstream, "commit", "-qam", f"add {text.strip()}")
        self.tree = git(self.upstream, "rev-parse", "HEAD^{tree}")
        (self.recipe / "patches" / "cuda").mkdir(parents=True)
        git(self.upstream, "format-patch", "-q", "-o", str(self.recipe / "patches" / "cuda"), f"{self.rev}..HEAD")
        git(self.upstream, "reset", "-q", "--hard", self.rev)
        self.write(self.rev, self.tree)

    def write(self, rev: str, tree: str, series: str = '["patches/cuda"]') -> None:
        (self.recipe / "recipe.toml").write_text(
            f'{HEAD}[engine.cuda]\nrepo = "https://github.com/example/TensorFold.git"\n'
            f'rev = "{rev}"\ntree = "{tree}"  # after the series\nseries = {series}\n'
        )

    def load(self) -> tuple[manifest.Recipe, manifest.Engine]:
        recipe = manifest.load(self.recipe)
        return recipe, recipe.engine(None)


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.fx = Fixture(self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fetch_applies_series_and_checks_tree(self):
        recipe, spec = self.fx.load()
        self.assertEqual(engine.fetch(recipe, spec, self.base / "e", str(self.fx.upstream)), self.fx.tree)
        self.assertEqual((self.base / "e" / "engine.txt").read_text(), "one\ntwo\nthree\n")

    def test_refetch_reuses_only_an_unchanged_engine(self):
        recipe, spec = self.fx.load()
        dest = self.base / "e"
        engine.fetch(recipe, spec, dest, str(self.fx.upstream))
        self.assertEqual(engine.fetch(recipe, spec, dest), self.fx.tree)
        (dest / "engine.txt").write_text("edited\n")
        with self.assertRaisesRegex(engine.EngineError, "moved off"):
            engine.fetch(recipe, spec, dest)
        (dest / "engine.txt").write_text("one\ntwo\nthree\n")
        patch = next((self.fx.recipe / "patches" / "cuda").glob("0001-*"))
        patch.write_text(patch.read_text() + "\n")
        with self.assertRaisesRegex(engine.EngineError, "not fetched from this pin"):
            engine.fetch(recipe, spec, dest)

    def test_series_stops_at_last(self):
        self.fx.write(self.fx.rev, "0" * 40, '["patches/cuda:0001"]')
        recipe, spec = self.fx.load()
        with self.assertRaisesRegex(engine.EngineError, "differs from"):
            engine.fetch(recipe, spec, self.base / "e", str(self.fx.upstream))
        self.assertEqual((self.base / "e" / "engine.txt").read_text(), "one\ntwo\n")

    def test_one_changed_byte_in_a_patch_refuses(self):
        patch = next((self.fx.recipe / "patches" / "cuda").glob("0002-*"))
        patch.write_text(patch.read_text().replace("+three", "+thre3"))
        recipe, spec = self.fx.load()
        with self.assertRaisesRegex(engine.EngineError, "differs from"):
            engine.fetch(recipe, spec, self.base / "e", str(self.fx.upstream))

    def test_recipe_pinned_to_another_kit_revision_refuses(self):
        recipe, _ = self.fx.load()
        with self.assertRaisesRegex(manifest.ManifestError, "kit\\] rev is c+"):
            revision.require(recipe)
        self.assertTrue(revision.require(recipe, dev=True).startswith("dev:"))

    def test_testkit_range(self):
        self.assertTrue(manifest.in_range("0.1.0", ">=0.1,<1"))
        self.assertFalse(manifest.in_range("1.0.0", ">=0.1,<1"))

    def test_gap_in_series_refuses(self):
        first = next((self.fx.recipe / "patches" / "cuda").glob("0001-*"))
        first.unlink()
        recipe, spec = self.fx.load()
        with self.assertRaisesRegex(engine.EngineError, "0001 missing"):
            engine.fetch(recipe, spec, self.base / "e", str(self.fx.upstream))

    def test_update_rewrites_only_the_pin(self):
        recipe, spec = self.fx.load()
        before = recipe.path.read_text()
        tree = engine.update(recipe, spec, self.fx.rev, self.base / "e", str(self.fx.upstream))
        after = recipe.path.read_text()
        self.assertEqual(tree, self.fx.tree)
        self.assertEqual(after, before)
        self.assertIn("  # after the series", after)
        moved = engine.pin(after, "cuda", "a" * 40, "b" * 40)
        self.assertEqual(manifest.load(self.fx.recipe).engines["cuda"].rev, self.fx.rev)
        self.assertIn(f'rev = "{"a" * 40}"', moved)
        self.assertIn(f'tree = "{"b" * 40}"  # after the series', moved)


class BuildTest(unittest.TestCase):
    def test_expand_replaces_vars_and_refuses_unknown(self):
        self.assertEqual(build.expand(("zig", "-Dgpu={GPU}"), {"GPU": "sm_120"}), ["zig", "-Dgpu=sm_120"])
        with self.assertRaisesRegex(build.BuildError, "NVCC"):
            build.expand(("-Dnvcc={NVCC}",), {})

    def test_build_runs_steps_and_checks_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            spec = build.Build({"OUT": "a.bin"}, (), (("touch", "{OUT}"),), ("a.bin",), ())
            self.assertEqual(build.build(spec, dest, {}), [dest / "a.bin"])
            spec = build.Build({"OUT": "a.bin"}, (), (("true",),), ("b.bin",), ())
            with self.assertRaisesRegex(build.BuildError, "wrote no b.bin"):
                build.build(spec, dest, {})


if __name__ == "__main__":
    unittest.main()
