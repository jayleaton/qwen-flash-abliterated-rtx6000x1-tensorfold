"""Fail when a real model, recipe or recipe repo name is in a tracked or untracked (not ignored) file or its path.

With pathspecs, only those parts of the tree are scanned: a repo that hosts a recipe beside the kit passes the
recipe's directory, and the scan still covers every template part the recipe does not own.
"""

import re
import subprocess
import sys
from pathlib import Path

# Each name has a bracketed letter, so this file never matches itself.
NAMES = re.compile(
    r"q[w]en|fl[a]sh.?next|g[l]m|de[e]pseek|mi[n]imax|in[k]ling|se[n]senova|\bd[s]v4\d*|\bm[m]3\d*\b|m[i]a-recipe|l[o]op9b",
    re.IGNORECASE,
)
FIX = (
    "real-names: the kit is a template and names no real model, recipe or recipe repo. Replace each name above with "
    "a made-up one (org/Example-Model, example-single, TF_EXAMPLE_*, as in tests/fixtures/); a test that needs a "
    "real recipe's bytes belongs in that recipe's own repo."
)


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files", "-z", "-co", "--exclude-standard", *sys.argv[1:]],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = []
    for name in filter(None, files.split("\0")):
        path = Path(name)
        if match := NAMES.search(name):
            found.append(f"{name}: path names {match[0]}")
        if not path.is_file() or b"\0" in (data := path.read_bytes())[:8192]:
            continue
        for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if match := NAMES.search(line):
                found.append(f"{name}:{number}: {match[0]}")
    for line in found:
        print(line)
    if found:
        print(FIX)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
