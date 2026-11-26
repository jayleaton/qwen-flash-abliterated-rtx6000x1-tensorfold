"""recipe publish: the public recipe tree, generated from the private recipe's committed files, so there is no second
repo to drift. Paths [publish] exclude names stay private; every file that goes out, the kit release included, passes
publish-check's rules and the recipe's own lists, or nothing is written. The kit goes out as the release the recipe
pins: a reproducible tarball of that revision with its sha256, which kit/recipe checks before every run."""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from . import scrub
from .bundle import KIT_PARTS
from .manifest import ManifestError, Recipe
from .revision import KIT_ROOT

KIT_DIR = "kit"
SUMS = "SHA256SUMS"
LAUNCHER = """#!/usr/bin/env bash
# Run the recipe kit release beside this file, after checking it against SHA256SUMS; from the recipe's root:
#   kit/recipe check      kit/recipe fetch DEST      kit/recipe serve TARGET ENGINE_DIR MODEL_DIR
set -euo pipefail
kit=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
release={release}
if command -v sha256sum >/dev/null; then sum=(sha256sum); else sum=(shasum -a 256); fi
(cd "$kit" && "${{sum[@]}}" -c --quiet {sums}) >&2 || {{
    echo "kit/$release.tar.gz does not match kit/{sums}: not running it" >&2
    exit 2
}}
if [[ ! -x "$kit/.release/$release/bin/recipe" ]]; then
    rm -rf "$kit/.release/.unpack" && mkdir -p "$kit/.release/.unpack"
    tar -xzf "$kit/$release.tar.gz" -C "$kit/.release/.unpack"
    mv "$kit/.release/.unpack/$release" "$kit/.release/$release" && rmdir "$kit/.release/.unpack"
fi
exec "$kit/.release/$release/bin/recipe" "$@"
"""


@dataclass
class Export:
    """What a publish wrote (or would have): its files, the ones excluded, and the scan's findings."""

    files: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    report: scrub.Report = field(default_factory=lambda: scrub.Report({}))
    release: str = ""
    sha256: str = ""


def excluded(path: str, patterns: list[str]) -> bool:
    """A pattern ending in / names a directory and everything under it; any other is a glob over the whole path."""
    return any(path.startswith(p) if p.endswith("/") else fnmatch.fnmatchcase(path, p) for p in patterns)


def export(
    root: Path, rev: str, dest: Path, exclude: list[str], active: list[scrub.Rule], allow, kit_rev: str | None
) -> Export:
    """Write ROOT's tree at REV, less EXCLUDE, and with KIT_REV the kit release, into DEST, a new directory; when the
    scan finds anything, DEST is removed again and the findings say why."""
    if dest.exists():
        raise ManifestError(f"{dest} exists: publish writes a new directory")
    out = Export()
    dest.mkdir(parents=True)
    try:
        _tree(root, rev, dest, exclude, active, allow, out)
        if kit_rev:
            _release(kit_rev, dest / KIT_DIR, active, allow, out)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)  # this call made DEST
        raise
    if out.report.findings:
        shutil.rmtree(dest, ignore_errors=True)
    return out


def _tree(root: Path, rev: str, dest: Path, exclude: list[str], active, allow, out: Export) -> None:
    archive = subprocess.Popen(["git", "-C", str(root), "archive", "--format=tar", rev], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            path = member.name
            if path == scrub.LOCAL_LIST or excluded(path, exclude):
                out.excluded.append(path)
                continue
            data = tar.extractfile(member).read()
            if path != scrub.ALLOW_LIST:
                _scan(out.report, path, data, active, allow)
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
            out.files.append(path)
    if archive.wait() != 0:
        raise ManifestError(f"git archive {rev} failed in {root}: is it a commit of the recipe's repo?")


def release(rev: str) -> tuple[str, bytes]:
    """The kit at REV from this kit's own repo (bin, kit, testkit, hardware, README) with a REVISION file, as a
    gzipped tarball that is the same bytes on every run."""
    name = f"tensorfold-recipe-kit-{rev[:12]}"
    raw = subprocess.run(
        ["git", "-C", str(KIT_ROOT), "archive", "--format=tar", f"--prefix={name}/", rev, "--", *KIT_PARTS],
        capture_output=True,
    )
    if raw.returncode != 0:
        raise ManifestError(f"[kit] rev {rev} is not a commit of the kit at {KIT_ROOT}: {raw.stderr.decode()[:200]}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw.stdout)) as source:
        members = [m for m in source.getmembers() if m.isfile() or m.isdir()]
        mtime = max((m.mtime for m in members), default=0)
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
            for member in members:
                tar.addfile(member, source.extractfile(member) if member.isfile() else None)
            stamp = (rev + "\n").encode()
            info = tarfile.TarInfo(f"{name}/REVISION")
            info.size, info.mtime, info.mode = len(stamp), mtime, 0o644
            tar.addfile(info, io.BytesIO(stamp))
    return name, buffer.getvalue()


def _release(rev: str, kit: Path, active, allow, out: Export) -> None:
    if clash := [f for f in out.files if f.startswith(f"{KIT_DIR}/")]:
        raise ManifestError(f"the recipe has its own {KIT_DIR}/ ({clash[0]}), where the kit release goes")
    name, blob = release(rev)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                _scan(out.report, f"{KIT_DIR}/{member.name}", tar.extractfile(member).read(), active, allow)
    kit.mkdir(parents=True, exist_ok=True)
    (kit / f"{name}.tar.gz").write_bytes(blob)
    out.release, out.sha256 = name, hashlib.sha256(blob).hexdigest()
    (kit / SUMS).write_text(f"{out.sha256}  {name}.tar.gz\n")
    (kit / "recipe").write_text(LAUNCHER.format(release=name, sums=SUMS))
    (kit / "recipe").chmod(0o755)
    (kit / ".gitignore").write_text(".release/\n")
    out.files += [f"{KIT_DIR}/{n}" for n in (f"{name}.tar.gz", SUMS, "recipe", ".gitignore")]


def _scan(report: scrub.Report, path: str, data: bytes, active, allow) -> None:
    report.scanned += 1
    if scrub.UNPUBLISHABLE.search(path):
        report.add("files that should not be published", path)
    if b"\0" not in data[:8192]:
        scrub.scan_text(report, "", path, data.decode("utf-8", "replace"), active, allow)


def publish(recipe: Recipe, rev: str, dest: Path, extra: str = "") -> Export:
    """The recipe's public tree at REV in DEST, refused when recipe.toml at REV is not the one on disk (its [publish]
    and [kit] are what this run reads)."""
    at_rev = subprocess.run(["git", "-C", str(recipe.root), "show", f"{rev}:recipe.toml"], capture_output=True)
    if at_rev.returncode != 0 or at_rev.stdout != recipe.path.read_bytes():
        raise ManifestError(f"{recipe.path} differs from recipe.toml at {rev}: commit it first")
    if not re.fullmatch(r"[0-9a-f]{40}", recipe.kit_rev) or set(recipe.kit_rev) == {"0"}:
        raise ManifestError(f"{recipe.path}: [kit] rev must name the kit commit the public recipe runs with")
    exclude = list(recipe.data.get("publish", {}).get("exclude", []))
    if not all(isinstance(p, str) and p and not p.startswith("/") for p in exclude):
        raise ManifestError(f"{recipe.path} [publish]: exclude lists repo-relative paths, globs or DIR/")
    active, allow = scrub.rules(recipe.root, extra), scrub.allowed(recipe.root)
    return export(recipe.root, rev, dest, exclude, active, allow, recipe.kit_rev)
