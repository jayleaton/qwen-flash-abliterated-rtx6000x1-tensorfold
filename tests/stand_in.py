"""The synthetic recipes the tests run, and a stand-in engine that records how it was started."""

import json
import os
import shutil
import subprocess
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
SINGLE = KIT / "tests" / "fixtures" / "example-single"
MULTI = KIT / "tests" / "fixtures" / "example-multi"
# The test's own environment, what bin/recipe's shell sets (_, PWD, SHLVL) and Python's locale coercion (LC_CTYPE).
PASSED = {"PATH", "HOME", "DUMP_DIR", "_", "PWD", "SHLVL", "LC_CTYPE"}
STAND_IN = """#!/usr/bin/env python3
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1:] == ["capabilities", "--json"]:
    sys.stdout.write(open(os.path.join(here, "caps.json")).read())
    sys.exit(0)
rank = sys.argv[sys.argv.index("--rank") + 1] if "--rank" in sys.argv else "0"
with open(os.path.join(os.environ["DUMP_DIR"], f"rank{rank}.json"), "w") as out:
    json.dump({"argv": sys.argv, "env": dict(os.environ)}, out, indent=1, sort_keys=True)
others = int(sys.argv[sys.argv.index("--tp" if "--tp" in sys.argv else "--pp") + 1]) if "--rank" in sys.argv else 1
while rank == "0" and len(os.listdir(os.environ["DUMP_DIR"])) < others:
    time.sleep(0.01)
"""


def engine(root: Path, caps: dict, tree: Path | None = None) -> Path:
    """An engine directory at ROOT (a copy of TREE when given) whose server is the stand-in answering CAPS."""
    if tree is not None:
        shutil.copytree(tree, root)
    binary = root / "zig-out" / "native" / "bin" / "tensorfold-native"
    binary.parent.mkdir(parents=True)
    binary.write_text(STAND_IN)
    binary.chmod(0o755)
    (binary.parent / "caps.json").write_text(json.dumps(caps))
    (root / "zig-out" / "native" / "lib").mkdir()
    (root / "zig-out" / "native" / "lib" / "libtfgrammar.so").write_bytes(b"")
    return root


def single_engine(base: Path) -> Path:
    return engine(base / "engine", json.loads((SINGLE / "caps.json").read_text()), SINGLE / "engine")


def launch(command: list[str], dump: Path) -> dict[str, dict]:
    """Each rank's argv and env as COMMAND started them, less the test's own environment."""
    if dump.exists():
        shutil.rmtree(dump)
    dump.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(dump), "DUMP_DIR": str(dump)}
    result = subprocess.run(command, env=env, capture_output=True, text=True, cwd=dump)
    if result.returncode != 0:
        raise AssertionError(f"{command[0]} exited {result.returncode}: {result.stderr[-2000:]}")
    ranks = {path.stem: json.loads(path.read_text()) for path in sorted(dump.glob("rank*.json"))}
    for rank in ranks.values():
        rank["env"] = {k: v for k, v in rank["env"].items() if k not in PASSED}
    return ranks


def placeholders(ranks: dict[str, dict], engine_dir: Path, model: Path) -> dict[str, dict]:
    """RANKS with the engine and model directories written as {engine} and {model}, as the golden files hold them."""
    text = json.dumps(ranks).replace(str(engine_dir), "{engine}").replace(str(model), "{model}")
    return json.loads(text)
