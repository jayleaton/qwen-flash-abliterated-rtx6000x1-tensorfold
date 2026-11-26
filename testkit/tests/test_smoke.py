import re
import tomllib
import unittest
from pathlib import Path

import tensorfold_testkit


class Smoke(unittest.TestCase):
    def test_version_matches_pyproject(self):
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
        self.assertRegex(tensorfold_testkit.__version__, re.compile(r"\d+\.\d+\.\d+"))
        self.assertEqual(tensorfold_testkit.__version__, project["version"])


if __name__ == "__main__":
    unittest.main()
