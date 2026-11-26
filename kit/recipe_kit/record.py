"""What the kit learned about each engine pin, kept in the recipe's .recipe/engines/: the names its tree reads
(recipe fetch) and its built binary's capabilities (recipe prepare), which recipe check and recipe serve reuse. A
record holds for the tree it was made from only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .manifest import Engine, Recipe

DIR = Path(".recipe") / "engines"


def path(recipe: Recipe, spec: Engine) -> Path:
    return recipe.root / DIR / f"{spec.name}.json"


def load(recipe: Recipe, spec: Engine) -> dict:
    """The engine's record when it was made from the pinned tree; else an empty one."""
    try:
        saved = json.loads(path(recipe, spec).read_text())
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) and saved.get("tree") == spec.tree else {}


def save(recipe: Recipe, spec: Engine, **fields) -> None:
    """Merge FIELDS into the engine's record; a recipe that cannot be written to keeps none, with a warning."""
    record = {**load(recipe, spec), "engine": spec.name, "tree": spec.tree, **fields}
    target = path(recipe, spec)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, sort_keys=True) + "\n")
    except OSError as err:
        print(f"warning: no record of [engine.{spec.name}] kept: {err}", file=sys.stderr)


def sha256(binary: Path) -> str:
    digest = hashlib.sha256()
    with binary.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def capabilities(recipe: Recipe, spec: Engine, binary: Path, live) -> dict:
    """The binary's capabilities: the record's when recipe prepare recorded this very binary, else LIVE(binary)."""
    record = load(recipe, spec)
    if isinstance(record.get("capabilities"), dict) and record.get("binary") == sha256(binary):
        return record["capabilities"]
    return live(binary)
