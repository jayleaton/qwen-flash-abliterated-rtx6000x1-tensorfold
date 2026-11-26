"""The pinned tree's tools/conformance.py against a running server: every gate, then each with its own --plant.

A plant that stays green fails the run. resumed and evict compare against a server started with
--prompt-cache-gib 0, so without a fresh URL they are not run, and the line says so.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from . import GateResult

GATES = ("drafted", "repeat", "concurrent", "resumed", "refusals", "cancel", "evict")
NEED_FRESH = ("resumed", "evict")
LINE = re.compile(r"^(PASS|FAIL|SKIP) (\S+)")


def run(tree: Path, profile: Path, url: str, fresh_url: str | None, gates: list[str], plant: str | None = None):
    """One conformance.py run: its exit code and the (word, gate) of each line it printed."""
    argv = [
        sys.executable,
        str(tree / "tools" / "conformance.py"),
        str(profile),
        "--url",
        url,
        "--gates",
        ",".join(gates),
    ]
    argv += ["--fresh-url", fresh_url] if fresh_url else []
    argv += ["--plant", plant] if plant else []
    out = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    lines = [m.groups() for m in map(LINE.match, out.stdout.splitlines()) if m]
    return out.returncode, lines, out.stdout + out.stderr


def conformance(
    tree: Path,
    profile: Path,
    url: str,
    fresh_url: str | None = None,
    gates: list[str] | None = None,
    plants: bool = True,
) -> GateResult:
    if not (tree / "tools" / "conformance.py").is_file():
        return GateResult("conformance", False, f"{tree} has no tools/conformance.py")
    skip = json.loads(Path(profile).read_text()).get("skip") or {}
    chosen = [g for g in gates or GATES if fresh_url or g not in NEED_FRESH]
    code, lines, log = run(tree, profile, url, fresh_url, chosen)
    failed = [g for word, g in lines if word == "FAIL"]
    problems = [f"FAIL {', '.join(failed)}"] if failed else []
    if code != 0 and not failed:
        problems.append(f"conformance.py exited {code}: {log.strip().splitlines()[-1:]}")
    planted, green = [g for g in chosen if g not in skip] if plants else [], []
    for gate in planted:
        code, _, _ = run(tree, profile, url, fresh_url, [gate], gate)
        (green if code == 0 else []).append(gate)
    if green:
        problems.append(f"plants that stayed green: {', '.join(green)}")
    passed = sum(word == "PASS" for word, _ in lines)
    line = f"{passed} passed of {', '.join(chosen)}"
    line += f"; {len(planted) - len(green)}/{len(planted)} plants red" if plants else ""
    if missing := [g for g in gates or GATES if g not in chosen]:
        line += f"; {', '.join(missing)} not run (no fresh URL)"
    return GateResult("conformance", not problems, line + "".join("; " + p for p in problems))
