"""Fail when a tracked or untracked (not ignored) text file is longer than the line limit."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    limit = int(sys.argv[1])
    files = subprocess.run(
        ["git", "ls-files", "-z", "-co", "--exclude-standard"], capture_output=True, text=True, check=True
    ).stdout
    long = []
    for name in filter(None, files.split("\0")):
        path = Path(name)
        if not path.is_file() or b"\0" in (data := path.read_bytes())[:8192]:
            continue
        if (lines := data.count(b"\n")) > limit:
            long.append(f"{name}: {lines} lines")
    for line in long:
        print(line)
    return 1 if long else 0


if __name__ == "__main__":
    sys.exit(main())
