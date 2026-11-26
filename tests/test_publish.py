"""recipe publish: the public tree is the recipe's committed files less [publish] exclude, plus the pinned kit release
and its checksum; a planted address, host name, credential or private word refuses it, and a changed byte in the
release stops kit/recipe from running it."""

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from recipe_kit import cli, publish
from test_engine import HEAD, git

KIT = Path(__file__).resolve().parents[1]
FILES = {
    "README.md": "# A recipe\n",
    "scripts/serve.sh": "#!/usr/bin/env bash\nexec true\n",
    "results/summary.md": "decode 100 tok/s\n",
    "results/raw/run-1.md": "served from " + "10." + "0.0.7\n",  # kept private by [publish] exclude
    "notes/lab.txt": "lab notes\n",
    ".check-public.allow": "an upstream name a rule mistakes for a private one\n",
}
PLANTS = {
    "docs/address.md": ("private IPv4 10/8", "the head node is " + "10." + "1.2.3\n"),
    "docs/host.md": ("machine host names", "start it on " + "dg" + "x-01\n"),
    "docs/key.md": ("credential assignments", "pass" + 'word = "' + "hunter2hunter2" + '"\n'),
    "docs/word.md": ("local list (.check-public.local)", "the " + "blue" + "fin rack\n"),
}


class Publish(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.recipe = self.base / "recipe"
        self.kit_rev = subprocess.run(
            ["git", "-C", str(KIT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        toml = HEAD.replace('rev = "' + "c" * 40 + '"', f'rev = "{self.kit_rev}"')
        toml += f'[engine.cuda]\nrepo = "https://github.com/example/TensorFold.git"\nrev = "{"a" * 40}"\n'
        toml += f'tree = "{"b" * 40}"\n\n[publish]\nexclude = ["results/raw/", "notes/*.txt"]\n'
        self.commit({**FILES, "recipe.toml": toml, ".gitignore": ".check-public.local\n"})
        (self.recipe / ".check-public.local").write_text("blue" + "fin\n")

    def tearDown(self):
        self.tmp.cleanup()

    def commit(self, files: dict[str, str]) -> None:
        for name, text in files.items():
            (self.recipe / name).parent.mkdir(parents=True, exist_ok=True)
            (self.recipe / name).write_text(text)
        if not (self.recipe / ".git").exists():
            git(self.recipe, "init", "-q")
        git(self.recipe, "add", ".")
        git(self.recipe, "commit", "-qm", "recipe")

    def publish(self, dest: Path) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["publish", "--recipe", str(self.recipe), "--kit-dev", str(dest)])
        return code, out.getvalue()

    def test_the_public_tree_and_its_kit_release(self):
        dest = self.base / "public"
        self.assertEqual(self.publish(dest)[0], 0)
        name = f"tensorfold-recipe-kit-{self.kit_rev[:12]}"
        kept = {"README.md", "scripts/serve.sh", "results/summary.md", ".check-public.allow", "recipe.toml"}
        kept.add(".gitignore")
        kit = {f"kit/{n}" for n in (f"{name}.tar.gz", "SHA256SUMS", "recipe", ".gitignore")}
        self.assertEqual({str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}, kept | kit)
        self.assertEqual((dest / "README.md").read_text(), FILES["README.md"])
        self.assertEqual(publish.release(self.kit_rev)[1], (dest / "kit" / f"{name}.tar.gz").read_bytes())
        run = subprocess.run(["bash", str(dest / "kit" / "recipe"), "--version"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(run.stdout.startswith("recipe-kit "))
        self.assertEqual((dest / "kit" / ".release" / name / "REVISION").read_text(), self.kit_rev + "\n")
        tarball = dest / "kit" / f"{name}.tar.gz"
        data = bytearray(tarball.read_bytes())
        data[len(data) // 2] ^= 1
        tarball.write_bytes(bytes(data))
        run = subprocess.run(["bash", str(dest / "kit" / "recipe"), "--version"], capture_output=True, text=True)
        self.assertEqual((run.returncode, run.stdout), (2, ""))
        self.assertIn("does not match kit/SHA256SUMS: not running it", run.stderr)

    def test_each_planted_leak_refuses_and_writes_nothing(self):
        self.commit({name: text for name, (_, text) in PLANTS.items()})
        dest = self.base / "public"
        code, out = self.publish(dest)
        self.assertEqual(code, 1)
        self.assertFalse(dest.exists())
        for name, (rule, _) in PLANTS.items():
            self.assertIn(f"== {rule}\n{name}:1:", out)
        self.assertNotIn("results/raw/run-1.md", out)


if __name__ == "__main__":
    unittest.main()
