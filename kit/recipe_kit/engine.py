"""Fetch an engine pin, apply the recipe's patch series and check the tree it gives."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from .manifest import HEX40, Engine, Recipe, Series

PATCH = re.compile(r"(?P<number>[0-9]{4})-.*\.patch")
# Fixed committer and dates, so the same series gives the same commits on any machine; authorship is the patches'.
AM_CONFIG = (
    "-c",
    "user.name=TensorFold recipe",
    "-c",
    "user.email=recipe@invalid",
    "-c",
    "core.autocrlf=false",
    "-c",
    "commit.gpgsign=false",
)


STAMP = "recipe-pin"


class EngineError(Exception):
    """A fetch, series or tree check that failed."""


def patches(root: Path, series: tuple[Series, ...]) -> list[Path]:
    """Every patch of every series in order; each series must be exactly 0001..LAST, one file a number."""
    found: list[Path] = []
    for layer in series:
        directory = root / layer.directory
        numbered = sorted(p for p in directory.glob("*.patch") if PATCH.fullmatch(p.name))
        if layer.last is not None:
            numbered = [p for p in numbered if p.name[:4] <= layer.last]
        for index, patch in enumerate(numbered, start=1):
            if patch.name[:4] != f"{index:04d}":
                raise EngineError(f"{layer.directory}/{index:04d} missing or not unique")
        if layer.last is not None and len(numbered) != int(layer.last):
            raise EngineError(f"{layer.directory}/{layer.last} missing")
        if not numbered:
            raise EngineError(f"{layer.directory} holds no NNNN-*.patch files")
        found.extend(numbered)
    return found


def fetch(recipe: Recipe, engine: Engine, dest: Path, source: str | None = None, check_tree: bool = True) -> str:
    """Check out the pin in DEST, apply the series and return the tree; refuse a tree other than the pinned one."""
    series = patches(recipe.root, engine.series)
    stamp = fingerprint(engine, series)
    if dest.exists():
        return reuse(engine, dest, stamp, check_tree)
    if source and Path(source).exists():
        source = str(Path(source).resolve())  # git runs in DEST, so a relative path would resolve there
    try:
        tree = checkout(engine, dest, series, source)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)  # this fetch made DEST; a half-made one would block the next try
        raise
    if check_tree and tree != engine.tree:
        raise EngineError(f"patched tree {tree} differs from [engine.{engine.name}] tree ({engine.tree})")
    (dest / ".git" / STAMP).write_text(stamp)
    return tree


def checkout(engine: Engine, dest: Path, series: list[Path], source: str | None) -> str:
    git("init", "-q", str(dest))
    git("-C", str(dest), "remote", "add", "origin", engine.repo)
    if source:
        git("-C", str(dest), "fetch", "-q", "--no-tags", source, engine.rev)
    else:
        git("-C", str(dest), "fetch", "-q", "--no-tags", "--depth=1", "origin", engine.rev)
    git("-C", str(dest), "checkout", "-q", "--detach", engine.rev)
    if series:
        git(
            "-C",
            str(dest),
            *AM_CONFIG,
            "am",
            "--quiet",
            "--committer-date-is-author-date",
            "--whitespace=nowarn",
            *map(str, series),
        )
    return git("-C", str(dest), "rev-parse", "HEAD^{tree}")


def fingerprint(engine: Engine, series: list[Path]) -> str:
    """The pin and the bytes of every patch: an engine directory made from the same fingerprint is reused."""
    digest = hashlib.sha256()
    for patch in series:
        digest.update(patch.name.encode() + b"\0" + patch.read_bytes() + b"\0")
    return f"{engine.repo} {engine.rev} {engine.tree} {digest.hexdigest()}\n"


def reuse(engine: Engine, dest: Path, stamp: str, check_tree: bool) -> str:
    """The tree of an existing engine directory that this pin and series made and nobody has edited since."""
    stamp_file = dest / ".git" / STAMP
    if not check_tree or not stamp_file.is_file() or stamp_file.read_text() != stamp:
        raise EngineError(f"{dest} exists but was not fetched from this pin and series; remove it or name another")
    tree = git("-C", str(dest), "rev-parse", "HEAD^{tree}")
    edited = subprocess.run(["git", "-C", str(dest), "diff", "--quiet", "HEAD"]).returncode != 0
    if tree != engine.tree or edited:
        raise EngineError(f"{dest} has moved off [engine.{engine.name}] tree {engine.tree}; remove it or name another")
    return tree


def latest(repo: str, branch: str = "main") -> str:
    """The commit a remote branch points at now."""
    out = git("ls-remote", repo, f"refs/heads/{branch}")
    rev = out.split()[0] if out else ""
    if not HEX40.fullmatch(rev):
        raise EngineError(f"{repo} has no branch {branch}")
    return rev


def update(recipe: Recipe, engine: Engine, rev: str, dest: Path, source: str | None = None) -> str:
    """Apply the series on REV in DEST, then write REV and the tree it gives into recipe.toml."""
    if not HEX40.fullmatch(rev):
        raise EngineError(f"{rev!r} is not a 40-hex commit")
    tree = fetch(recipe, replace(engine, rev=rev), dest, source, check_tree=False)
    recipe.path.write_text(pin(recipe.path.read_text(), engine.name, rev, tree))
    return tree


def pin(text: str, engine: str, rev: str, tree: str) -> str:
    """recipe.toml text with [engine.ENGINE]'s rev and tree replaced, everything else kept byte for byte."""
    lines = text.splitlines(keepends=True)
    header = re.compile(rf"\[\s*engine\.{re.escape(engine)}\s*\]\s*(#.*)?")
    try:
        start = next(i for i, line in enumerate(lines) if header.fullmatch(line.strip()))
    except StopIteration:
        raise EngineError(f"no [engine.{engine}] table to update") from None
    done = set()
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("["):
            break
        for key, value in (("rev", rev), ("tree", tree)):
            match = re.match(rf'(\s*{key}\s*=\s*")[0-9a-f]{{40}}(".*)', lines[i], re.DOTALL)
            if match:
                lines[i] = match[1] + value + match[2]
                done.add(key)
    if done != {"rev", "tree"}:
        raise EngineError(f'[engine.{engine}] needs rev = "..." and tree = "..." lines to update')
    return "".join(lines)


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise EngineError(f"git {' '.join(args[:4])}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()
