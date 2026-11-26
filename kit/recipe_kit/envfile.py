"""Read KEY=VALUE env files, where a variable already set in the environment wins."""

from __future__ import annotations

import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path

from .manifest import VAR


class EnvFileError(Exception):
    """A line that is not KEY=VALUE, a blank line or a # comment."""


def parse(text: str, source: str = "<env>") -> dict[str, str]:
    """Each KEY=VALUE line in order; VALUE is the rest of the line verbatim, a later KEY replaces an earlier one."""
    entries: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not VAR.fullmatch(key):
            raise EnvFileError(f"{source}:{number}: not KEY=VALUE with an UPPER_CASE key: {line[:80]}")
        entries[key] = value
    return entries


def load(paths: Iterable[Path]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for path in paths:
        entries.update(parse(path.read_text(), str(path)))
    return entries


def unset(entries: Mapping[str, str], environ: Mapping[str, str]) -> dict[str, str]:
    """The entries whose key the environment does not set, even to an empty value."""
    return {key: value for key, value in entries.items() if key not in environ}


def exports(entries: Mapping[str, str]) -> str:
    """Shell export lines for ENTRIES, each value quoted."""
    return "".join(f"export {key}={shlex.quote(value)}\n" for key, value in entries.items())
