"""What the built engine says it serves (capabilities --json), and the env names its tree reads."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .engine import STAMP
from .hardware import Hardware
from .manifest import Engine, ManifestError, Recipe
from .profiles import Rendered

KNOWN_SCHEMA = 1
SCAN = 2  # how scan_names reads a tree; a cached scan from another way is redone
NAMES = "recipe-names.json"  # in a bundle's engine directory: the names its tree reads, for want of the tree
# Rule rows in the pinned tree's environment table, and the names read with a bare getenv outside it.
RULE = re.compile(r'\.name = "([A-Z_][A-Z0-9_]*)"(?:[^\n]*?\.retire = "([0-9.]+)")?')
GETENV = re.compile(r'getenv\("([A-Z_][A-Z0-9_]*)"\)')
# The switch registry newer trees read by name (core/profile.zig: pub const Switch = enum { NAME, ... }).
REGISTRY = re.compile(r"pub const Switch = enum \{(?P<body>.*?)\n\};", re.DOTALL)


def capabilities(binary: Path) -> dict:
    try:
        out = subprocess.run([str(binary), "capabilities", "--json"], capture_output=True, text=True, timeout=30)
    except OSError as err:
        raise ManifestError(f"{binary}: {err}") from None
    if out.returncode != 0:
        raise ManifestError(f"{binary} capabilities --json exited {out.returncode}: {out.stderr.strip()[:200]}")
    try:
        caps = json.loads(out.stdout)
    except json.JSONDecodeError:
        raise ManifestError(f"{binary} capabilities --json did not print JSON") from None
    if not isinstance(caps, dict):
        raise ManifestError(f"{binary} capabilities --json is not an object")
    return caps


def check(caps: dict, recipe: Recipe, spec: Engine, hardware: Hardware, rendered: Rendered | None) -> list[str]:
    """Each way the binary cannot serve this recipe's engine as rendered (KIT-SPEC section 6.2)."""
    problems, backend = [], spec.backend
    model_type, fmt = recipe.model_type, recipe.format_of(spec)
    schema = caps.get("schema")
    if not isinstance(schema, int) or schema > KNOWN_SCHEMA:
        problems.append(f"capabilities schema {schema!r} is newer than this kit reads ({KNOWN_SCHEMA})")
    formats = caps.get("families", {}).get(model_type)
    if formats is None:
        problems.append(f"model_type {model_type} is not a family this binary serves")
    elif fmt not in formats:
        problems.append(f"family {model_type} serves {', '.join(formats)}, not {fmt}")
    if backend not in caps.get("backends", []):
        problems.append(f"backend {backend} is not in this binary's {caps.get('backends')}")
    if isinstance(hardware.arch, int) and hardware.arch not in caps.get("kernel_sms", []):
        problems.append(f"hardware {hardware.name} is sm_{hardware.arch}; the kernels are for {caps.get('kernel_sms')}")
    if rendered is not None:
        flags = caps.get("serve", {}).get("flags", {})
        for arg in rendered.every_arg:
            if arg.startswith("--") and arg not in flags:
                problems.append(f"{rendered.target} renders {arg}, which this binary's serve does not take")
    return problems


def engine_names(tree: Path) -> dict[str, str | None]:
    """scan_names, cached in a prepared tree's .git under the pin fingerprint that fetch wrote; a bundle's list."""
    if (tree / NAMES).is_file() and not (tree / "zig" / "src").is_dir():
        return json.loads((tree / NAMES).read_text())["names"]
    pin, cache = tree / ".git" / STAMP, tree / ".git" / NAMES
    stamp = pin.read_text() if pin.is_file() else None
    if stamp and cache.is_file():
        saved = json.loads(cache.read_text())
        if saved.get("pin") == stamp and saved.get("scan") == SCAN:
            return saved["names"]
    names = scan_names(tree)
    if stamp:
        cache.write_text(json.dumps({"pin": stamp, "scan": SCAN, "names": names}))
    return names


def scan_names(tree: Path) -> dict[str, str | None]:
    """Each env name the tree's rule table or switch registry registers (the table with its retire release), or
    its code reads by bare getenv."""
    platform = tree / "zig" / "src" / "platform"
    rules = [p for p in platform.glob("environment*.zig") if not p.name.endswith("_test.zig")]
    if not rules:
        raise ManifestError(f"{tree} has no zig/src/platform/environment*.zig rule table")
    names: dict[str, str | None] = {}
    for path in rules:
        for match in RULE.finditer(path.read_text()):
            names[match[1]] = match[2]
    found = subprocess.run(
        ["git", "grep", "--no-index", "-h", "-o", "-E", r'getenv\("[A-Z_][A-Z0-9_]*"\)', "--", "*.zig"],
        cwd=tree / "zig" / "src",
        capture_output=True,
        text=True,
    )
    for match in GETENV.finditer(found.stdout):
        names.setdefault(match[1], None)
    registry = tree / "zig" / "src" / "core" / "profile.zig"
    if registry.is_file() and (block := REGISTRY.search(registry.read_text())):
        for name in re.findall(r"^\s*([A-Z_][A-Z0-9_]*)\s*,", block["body"], re.MULTILINE):
            names.setdefault(name, None)
    return names
