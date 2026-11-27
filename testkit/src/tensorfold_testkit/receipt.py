"""One JSON a run, for the gates and the bench: what ran (kit, recipe, engine, binaries, settings) and what it found.

Identifiers follow upstream's public rule: no paths, addresses or accounts, so a receipt can be published as it is.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

from . import __version__

SCHEMA = 1
KINDS = {
    "gate": ("results",),
    "bench": ("arms", "order", "cells", "samples", "verdict"),
    "soak": ("arm", "soak", "verdict"),
}
HEAD = ("schema", "kind", "testkit", "kit", "dev", "recipe", "written")
PRIVATE = (
    (re.compile(r"(?<![\w.])(?:/|~/)[A-Za-z0-9._-]+/[A-Za-z0-9._/-]*"), "a filesystem path"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "an IP address"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "an e-mail address"),
    (re.compile(r"\bhttps?://"), "a URL"),
)


class ReceiptError(ValueError):
    pass


def write(path: Path, kind: str, head: dict, body: dict) -> Path:
    """Writes the receipt after checking it; refuses one that names a path, an address or an account."""
    doc = {
        "schema": SCHEMA,
        "kind": kind,
        "testkit": __version__,
        **head,
        "written": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **body,
    }
    check(doc)
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")
    return Path(path)


def verify(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as err:
        raise ReceiptError(f"{path}: {err}") from None
    check(doc)
    return doc


def check(doc: dict) -> None:
    problems = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema {doc.get('schema')!r}, this testkit reads {SCHEMA}")
    if doc.get("kind") not in KINDS:
        problems.append(f"kind {doc.get('kind')!r}")
    missing = [k for k in (*HEAD, *KINDS.get(doc.get("kind"), ())) if k not in doc]
    if missing:
        problems.append(f"missing {', '.join(missing)}")
    problems += [f"{where}: {what}" for where, what in private(doc)]
    if problems:
        raise ReceiptError("receipt: " + "; ".join(problems[:6]))


def redact(text: str) -> str:
    """text with each path, address or account in it replaced, so a log line can go into a receipt."""
    for pattern, _ in PRIVATE:
        text = pattern.sub("<redacted>", text)
    return text


def private(value, where: str = "") -> list[tuple[str, str]]:
    """Each string in the receipt that names a path, an address or an account, with where it is."""
    if isinstance(value, dict):
        return [hit for k, v in value.items() for hit in private(k, where) + private(v, f"{where}.{k}")]
    if isinstance(value, list):
        return [hit for i, v in enumerate(value) for hit in private(v, f"{where}[{i}]")]
    if isinstance(value, str):
        return [(where or "key", what) for pattern, what in PRIVATE if pattern.search(value)][:1]
    return []
