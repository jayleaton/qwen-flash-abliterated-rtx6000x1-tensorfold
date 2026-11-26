import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from recipe_kit import envfile, scrub

# Planted findings are assembled at run time, so this file passes the kit's own publish-check.
PRIVATE_IP = ".".join(["192", "168", "7", "9"])
HOME = "/" + "home/alice/"
MAIL = "ops" + "@corp.io"


class EnvFileTest(unittest.TestCase):
    def test_parse_skips_comments_and_keeps_value_verbatim(self):
        text = "# comment\n\nA=1\nB=x=y z \nA=2"
        self.assertEqual(envfile.parse(text), {"A": "2", "B": "x=y z "})

    def test_bad_key_names_the_line(self):
        with self.assertRaisesRegex(envfile.EnvFileError, "f:2"):
            envfile.parse("A=1\nlower=2\n", "f")

    def test_environment_wins_even_when_empty(self):
        entries = envfile.unset({"A": "1", "B": "2"}, {"A": ""})
        self.assertEqual(envfile.exports(entries), "export B=2\n")

    def test_exports_match_the_shell_loop_they_replace(self):
        text = "A=one two\nB=it's\n"
        loop = 'while IFS="=" read -r k v; do export "$k=$v"; done; printf "%s|%s" "$A" "$B"'
        shell = subprocess.run(["bash", "-c", loop], input=text, capture_output=True, text=True, check=True).stdout
        script = envfile.exports(envfile.parse(text)) + 'printf "%s|%s" "$A" "$B"'
        kit = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
        self.assertEqual(kit, shell)


class ScrubTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "ok.md").write_text("serve on 127.0.0.1:8090 as 58495321+someone@users.noreply.github.com\n")
        os.environ.pop("CHECK_PUBLIC_EXTRA", None)

    def tearDown(self):
        self.tmp.cleanup()

    def names(self, **kwargs) -> set[str]:
        return set(scrub.check(self.root, **kwargs).findings)

    def test_clean_tree(self):
        self.assertEqual(self.names(), set())

    def test_planted_findings(self):
        (self.root / "notes.md").write_text(f"host {PRIVATE_IP}\npath {HOME}x\nmail {MAIL}\n")
        (self.root / "run.log").write_text("x\n")
        self.assertEqual(
            self.names(),
            {"private IPv4 192.168/16", "home directories", "e-mail addresses", "files that should not be published"},
        )

    def test_local_list_with_path_skip_and_allow_list(self):
        (self.root / "a.md").write_text("on the bluebox\n")
        (self.root / "fixture.txt").write_text("bluebox\n")
        (self.root / scrub.LOCAL_LIST).write_text("# private names\nblue[[:alpha:]]ox\tfixture.*\n")
        report = scrub.check(self.root)
        self.assertEqual(report.findings, {"local list (.check-public.local)": ["a.md:1:on the bluebox"]})
        (self.root / scrub.ALLOW_LIST).write_text("^a\\.md:1:\n")
        self.assertEqual(self.names(), set())

    def test_history_finds_a_removed_secret(self):
        ident = ["-c", "user.name=T", "-c", "user.email=1+t@users.noreply.github.com", "-c", "commit.gpgsign=false"]
        (self.root / "old.md").write_text(f"{PRIVATE_IP}\n")
        subprocess.run(["git", *ident, "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", *ident, "-C", str(self.root), "commit", "-qm", "add"], check=True)
        (self.root / "old.md").unlink()
        self.assertEqual(self.names(), set())
        report = scrub.check(self.root, history=True)
        self.assertEqual(report.commits, 1)
        self.assertEqual(set(report.findings), {"private IPv4 192.168/16"})
        self.assertTrue(report.findings["private IPv4 192.168/16"][0].startswith("history old.md:1:"))


if __name__ == "__main__":
    unittest.main()
