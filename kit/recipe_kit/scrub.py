"""Scan a repo for private infrastructure details and credentials before it is published."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

LOCAL_LIST = ".check-public.local"
ALLOW_LIST = ".check-public.allow"
WIDTH = 220

# Each pattern is written so that its own source text does not match it, so the kit scans itself with no exclusions.
PATTERNS = (
    ("private IPv4 192.168/16", r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    ("private IPv4 10/8", r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    ("private IPv4 172.16/12", r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
    ("CGNAT / Tailscale IPv4", r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    ("Tailscale IPv6 or names", r"\bfd7[a]:115c:|\.ts\.net\b"),
    (
        "home directories",
        r"/hom[e]/(?!(user|you|ubuntu|<[a-z]+>)/)[a-z0-9_.-]+/|/User[s]/[A-Za-z0-9_.-]+/|(^|[^\w.-])~/[a-z]",
    ),
    ("machine host names", r"\bdg[x]-?\d{1,2}\b|\bnod[e]-?\d{2}\b|\bspar[k]-?\d{2}\b"),
    ("MAC address", r"\b([0-9a-f]{2}:){5}[0-9a-f]{2}\b"),
    ("GPU or MIG UUID", r"\b(GP[U]|MI[G])-[0-9a-f]{8}-[0-9a-f]{4}"),
    ("SSH or private key", r"ssh-(ed25519|rsa|dss) AAA[A]|BEGI[N] [A-Z ]*PRIVATE KEY"),
    (
        "tokens",
        r"\bh[f]_[A-Za-z0-9]{30,}|\bg[h][pousr]_[A-Za-z0-9]{30,}|githu[b]_pat_\w{20,}|\bs[k]-[\w-]{20,}"
        r"|\bAKI[A][0-9A-Z]{16}\b|xo[x][abprs]-[A-Za-z0-9-]{10,}|tske[y]-[a-z]+-[A-Za-z0-9-]{10,}",
    ),
    ("credential assignments", r"""(api[_-]?key|token|secret|password)["']?\s*[:=]\s*["'][^"'\s]{12,}"""),
    ("tool co-author trailers", r"^co-author[e]d-by:"),
)
EMAIL = re.compile(r"[a-z0-9][a-z0-9._%+-]*@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}", re.IGNORECASE)
PUBLIC_EMAIL = re.compile(
    r"([0-9]+\+)?[A-Za-z0-9-]+@users\.noreply\.github\.com|noreply@github\.com|.*@(([a-z0-9-]+\.)*example\.(com|org|net)|invalid)",
    re.IGNORECASE,
)
UNPUBLISHABLE = re.compile(r"\.(log|out|pem|key|p12|pfx|safetensors|gguf|bin)$|(^|/)\.env$|(^|/)id_(rsa|ecdsa|ed25519)")
POSIX_CLASSES = {
    "[:space:]": r"\s",
    "[:alnum:]": "a-zA-Z0-9",
    "[:alpha:]": "a-zA-Z",
    "[:digit:]": "0-9",
    "[:upper:]": "A-Z",
    "[:lower:]": "a-z",
    "[:xdigit:]": "0-9A-Fa-f",
}


@dataclass(frozen=True)
class Rule:
    """A named pattern and the path globs it skips."""

    name: str
    regex: re.Pattern[str]
    skip: tuple[str, ...] = ()

    def applies(self, path: str) -> bool:
        return not any(fnmatch.fnmatch(path, glob) for glob in self.skip)


@dataclass
class Report:
    """Findings grouped by rule name, in the order they were found."""

    findings: dict[str, list[str]]
    scanned: int = 0
    commits: int | None = None

    def add(self, name: str, line: str) -> None:
        self.findings.setdefault(name, []).append(line[:WIDTH])

    def render(self) -> str:
        return "".join(f"== {name}\n" + "".join(f"{line}\n" for line in lines) for name, lines in self.findings.items())


def ere(pattern: str) -> str:
    """A POSIX extended regex from a list file, as a Python regex."""
    for posix, python in POSIX_CLASSES.items():
        pattern = pattern.replace(posix, python)
    return pattern


def rules(root: Path, extra: str = "") -> list[Rule]:
    """The built-in rules, then each line of .check-public.local (REGEX or REGEX<TAB>GLOB...), then EXTRA."""
    found = [Rule(name, re.compile(regex, re.IGNORECASE)) for name, regex in PATTERNS]
    for line in _list_lines(root / LOCAL_LIST):
        regex, _, skip = line.partition("\t")
        found.append(Rule(f"local list ({LOCAL_LIST})", re.compile(ere(regex), re.IGNORECASE), tuple(skip.split())))
    if extra:
        found.append(Rule("CHECK_PUBLIC_EXTRA", re.compile(ere(extra), re.IGNORECASE)))
    return found


def allowed(root: Path) -> re.Pattern[str] | None:
    """One regex from .check-public.allow: a finding line (path:line:text) matching it is dropped."""
    lines = [f"(?:{ere(line)})" for line in _list_lines(root / ALLOW_LIST)]
    return re.compile("|".join(lines)) if lines else None


def scan_text(report: Report, label: str, path: str, text: str, active: list[Rule], allow: re.Pattern[str] | None):
    for number, line in enumerate(text.splitlines(), start=1):
        where = f"{label}{path}:{number}:{line}"
        if allow and allow.search(where):
            continue
        for rule in active:
            if rule.applies(path) and rule.regex.search(line):
                report.add(rule.name, where)
        for email in EMAIL.finditer(line):
            if not PUBLIC_EMAIL.fullmatch(email[0]) and not (
                allow and allow.search(f"{label}{path}:{number}:{email[0]}")
            ):
                report.add("e-mail addresses", f"{label}{path}:{number}:{email[0]}")


def scan_tree(root: Path, active: list[Rule], allow: re.Pattern[str] | None) -> Report:
    """Every tracked and untracked (not ignored) file except the two list files."""
    report = Report({})
    for path in _git(root, "ls-files", "-z", "-co", "--exclude-standard").split("\0"):
        if not path or path in (LOCAL_LIST, ALLOW_LIST) or not (root / path).is_file():
            continue
        report.scanned += 1
        if UNPUBLISHABLE.search(path):
            report.add("files that should not be published", path)
        data = (root / path).read_bytes()
        if b"\0" not in data[:8192]:
            scan_text(report, "", path, data.decode("utf-8", "replace"), active, allow)
    return report


def scan_history(root: Path, report: Report, active: list[Rule], allow, revs: str | None, authors: set[str]) -> None:
    """Every blob, message and identity of every commit (or of the commits in REVS, such as origin/main..HEAD)."""
    selection = [revs] if revs else ["--all"]
    report.commits = len(_git(root, "rev-list", *selection).split())
    for record in _git(root, "log", "--format=%h %ae %ce", *selection).splitlines():
        sha, author, committer = record.split(" ", 2)
        bad = [email for email in (author, committer) if not PUBLIC_EMAIL.fullmatch(email) and email not in authors]
        if bad:
            report.add("history: commit identities that are not public", f"{sha} {' '.join(bad)}")
    for message in _git(root, "log", "--format=%h%x00%B%x01", *selection).split("\x01"):
        sha, _, body = message.strip("\n").partition("\0")
        if sha:
            scan_text(report, f"history message {sha}", "", body, active, allow)
    for path, text in _blobs(root, selection):
        if path not in (LOCAL_LIST, ALLOW_LIST):
            scan_text(report, "history ", path, text, active, allow)


def check(root: Path, history: bool = False, revs: str | None = None) -> Report:
    active = rules(root, os.environ.get("CHECK_PUBLIC_EXTRA", ""))
    allow = allowed(root)
    report = scan_tree(root, active, allow)
    if history:
        authors = set(os.environ.get("ALLOWED_AUTHORS", "").split())
        scan_history(root, report, active, allow, revs, authors)
    return report


def _blobs(root: Path, selection: list[str]) -> Iterator[tuple[str, str]]:
    """Each distinct text blob the selected commits reach, with the first path git lists it under."""
    paths: dict[str, str] = {}
    for line in _git(root, "rev-list", "--objects", *selection).splitlines():
        sha, _, path = line.partition(" ")
        if path and sha not in paths:
            paths[sha] = path
    if not paths:
        return
    batch = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input="".join(f"{sha}\n" for sha in paths).encode(),
        capture_output=True,
        check=True,
    ).stdout
    offset = 0
    for sha in paths:
        header_end = batch.index(b"\n", offset)
        _, kind, size = batch[offset:header_end].decode().split()
        body = batch[header_end + 1 : header_end + 1 + int(size)]
        offset = header_end + 2 + int(size)
        if kind == "blob" and b"\0" not in body[:8192]:
            yield paths[sha], body.decode("utf-8", "replace")


def _list_lines(path: Path) -> Iterable[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()} (not a git checkout?)")
    return result.stdout
