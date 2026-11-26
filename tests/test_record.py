"""recipe fetch records the names a pinned tree reads and recipe prepare its binary's capabilities; recipe check runs
C8 from the record with no --tree, and serve takes the recorded capabilities of that very binary."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from recipe_kit import cli, manifest, record
from test_engine import HEAD, git

SERVER = """#!/usr/bin/env python3
import json
print(json.dumps({"schema": 1, "backends": ["cuda"], "families": {"m": ["bf16"]}, "serve": {"flags": {}}}))
"""
ROW = '[[switch]]\nname = "{0}"\nkind = "product"\nreason = "r"\ndoc = "d"\nhardware = ["rtx6000x2"]\n\n'


class RecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.upstream, self.recipe, self.dest = base / "upstream", base / "recipe", base / "engine"
        (self.upstream / "zig" / "src" / "platform").mkdir(parents=True)
        (self.upstream / "zig" / "src" / "platform" / "environment.zig").write_text('.{ .name = "TF_READ" },\n')
        (self.upstream / "server").write_text(SERVER)
        (self.upstream / "server").chmod(0o755)
        git(self.upstream, "init", "-q")
        git(self.upstream, "add", ".")
        git(self.upstream, "commit", "-qm", "base")
        self.rev, self.tree = git(self.upstream, "rev-parse", "HEAD"), git(self.upstream, "rev-parse", "HEAD^{tree}")
        self.recipe.mkdir()
        self.write("TF_READ")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, setting: str) -> None:
        (self.recipe / "recipe.toml").write_text(
            f'{HEAD}[engine.cuda]\nrepo = "https://github.com/example/TensorFold.git"\n'
            f'rev = "{self.rev}"\ntree = "{self.tree}"\n\n'
            '[engine.cuda.build]\nsteps = [["true"]]\noutputs = ["server"]\n\n'
            f"[profile.p]\ntopology = {{}}\ncontext = 8\nparallel = 1\nsettings = {{ {setting} = 1 }}\n\n"
            + ROW.format(setting)
        )

    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main([argv[0], "--recipe", str(self.recipe), "--kit-dev", *argv[1:]])
        return code, out.getvalue()

    def test_prepare_records_and_check_and_serve_reuse_it(self):
        self.assertEqual(self.run_cli("prepare", "--source", str(self.upstream), str(self.dest))[0], 0)
        saved = json.loads((self.recipe / record.DIR / "cuda.json").read_text())
        self.assertEqual((saved["tree"], saved["names"], saved["by"]), (self.tree, {"TF_READ": None}, "recipe prepare"))
        self.assertEqual(saved["capabilities"]["families"], {"m": ["bf16"]})
        self.assertEqual(self.run_cli("check"), (0, "recipe check: 0 finding(s) over 1 profiles and arms\n"))
        self.write("TF_UNREAD")
        code, out = self.run_cli("check")
        self.assertEqual((code, out.splitlines()[0]), (1, "C8 TF_UNREAD: the pinned engine does not read it"))
        recipe = manifest.load(self.recipe)
        spec, binary = recipe.engine(None), self.dest / "server"
        self.assertEqual(record.capabilities(recipe, spec, binary, None), saved["capabilities"])
        binary.write_text(SERVER + "\n")
        self.assertEqual(record.capabilities(recipe, spec, binary, lambda path: "live"), "live")


if __name__ == "__main__":
    unittest.main()
