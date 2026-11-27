"""The real-names scan covers the template parts and leaves what a hosted recipe owns to the recipe."""

import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "real_names.py"
# Assembled at run time so this file never holds the name it scans for.
REAL = "q" + "wen-flash"


class Scope(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)

    def repo(self, parts: dict[str, str]) -> None:
        for name, text in parts.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        self.git("init", "-q", "-b", "main")
        self.git("add", ".")
        self.git("-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-qm", "parts")

    def scan(self, *parts: str) -> subprocess.CompletedProcess:
        return subprocess.run(["python3", str(SCRIPT), *parts], cwd=self.root, capture_output=True, text=True)

    def test_a_hosted_recipe_directory_is_outside_the_scan(self):
        self.repo({"kit/code.py": "MODEL = 'example'\n", "hosted/model.toml": f'model = "{REAL}-next"\n'})
        self.assertEqual(self.scan("kit").returncode, 0)
        self.assertEqual(self.scan().returncode, 1)

    def test_a_real_name_in_a_template_part_still_fails(self):
        self.repo({"kit/code.py": f"MODEL = '{REAL}'\n"})
        self.assertEqual(self.scan("kit").returncode, 1)
        self.assertIn(f"kit/code.py:1: {REAL[:4]}", self.scan("kit").stdout)


if __name__ == "__main__":
    unittest.main()
