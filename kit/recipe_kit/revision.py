"""The kit revision this command runs from, and the refusal of a recipe written for another one."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .manifest import ManifestError, Recipe, in_range

KIT_ROOT = Path(__file__).resolve().parents[2]


def current(root: Path = KIT_ROOT) -> tuple[str | None, str]:
    """The clean checkout's HEAD, or None and why there is no single revision to name."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)

    if not (root / ".git").exists() and (root / "REVISION").is_file():  # a bundle's copy of the kit
        return (root / "REVISION").read_text().strip(), ""
    head = git("rev-parse", "--verify", "HEAD")
    if head.returncode != 0 or not (root / ".git").exists():
        return None, f"{root} is not a git checkout of the kit"
    if git("status", "--porcelain").stdout.strip():
        return None, f"the kit checkout {root} has uncommitted changes"
    return head.stdout.strip(), ""


def require(recipe: Recipe, dev: bool = False) -> str:
    """The kit revision to stamp on what runs; refuse a recipe pinned to another revision unless DEV."""
    try:
        import tensorfold_testkit
    except ImportError:
        raise ManifestError(f"tensorfold_testkit is not importable; run the kit from {KIT_ROOT}/bin/recipe") from None
    if not in_range(tensorfold_testkit.__version__, recipe.testkit):
        raise ManifestError(
            f"{recipe.path}: [kit] testkit {recipe.testkit!r} excludes testkit {tensorfold_testkit.__version__}"
        )
    rev, why = current()
    if rev == recipe.kit_rev:
        return rev
    if dev:
        return f"dev:{rev or 'unknown'}"
    found = f"this kit is {rev}" if rev else why
    raise ManifestError(
        f"{recipe.path}: [kit] rev is {recipe.kit_rev}, but {found}; check out that revision, or pass --kit-dev"
    )
