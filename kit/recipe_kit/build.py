"""Build a fetched engine tree with the steps its recipe.toml names."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .engine import git
from .manifest import Addon, Build

PLACEHOLDER = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")


class BuildError(Exception):
    """A toolchain, step or output check that failed."""


def build(spec: Build, dest: Path, environ: Mapping[str, str] = os.environ, addons: bool = True) -> list[Path]:
    """Run the build steps, then each addon's, in DEST; return every output file they must write."""
    check_zig(dest)
    values = {name: environ.get(name) or default for name, default in spec.variables.items()}
    for name in spec.executables:
        if not _executable(values[name]):
            raise BuildError(f"no executable {name}={values[name]}: set {name}")
    run_steps(spec.steps, dest, values)
    outputs = check_outputs(dest, spec.outputs)
    for addon in spec.addons if addons else ():
        checkout = addon_checkout(addon, dest, environ.get(addon.var, ""))
        if checkout is None:
            continue
        run_steps(addon.steps, dest, {**values, addon.var: str(checkout)})
        outputs += check_outputs(dest, addon.outputs)
    return outputs


def check_zig(dest: Path) -> None:
    pinned = dest / ".zig-version"
    if not pinned.exists():
        return
    want = pinned.read_text().strip()
    try:
        have = subprocess.run(["zig", "version"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        raise BuildError(f"no zig on PATH; the engine needs zig {want} (.zig-version)") from None
    if have != want:
        raise BuildError(f"zig {have} found; the engine needs zig {want} (.zig-version)")


def addon_checkout(addon: Addon, dest: Path, chosen: str) -> Path | None:
    """The addon's checkout: VAR=DIR names one, VAR=none skips it, otherwise DEST/.NAME is cloned at the tag."""
    if chosen == "none":
        return None
    if chosen:
        path = Path(chosen).resolve()
    else:
        path = dest / f".{addon.name}"
        if not (path / ".git").is_dir():
            part = path.with_name(path.name + ".part")
            git("clone", "-q", "--branch", addon.tag, "--recurse-submodules", addon.repo, str(part))
            part.rename(path)
        exclude = dest / ".git" / "info" / "exclude"
        entry = f"/.{addon.name}"
        lines = exclude.read_text().splitlines() if exclude.exists() else []
        if entry not in lines:
            exclude.parent.mkdir(parents=True, exist_ok=True)
            with exclude.open("a") as handle:
                handle.write(entry + "\n")
    got = git("-C", str(path), "rev-parse", "HEAD")
    if got != addon.commit:
        raise BuildError(f"{path} is {addon.name} {got}, not {addon.tag} ({addon.commit})")
    return path


def expand(step: tuple[str, ...], values: Mapping[str, str]) -> list[str]:
    """STEP with every {NAME} replaced by its value; an unknown name is an error."""

    def value(match: re.Match[str]) -> str:
        if match[1] not in values:
            raise BuildError(f"step {' '.join(step)} uses {{{match[1]}}}, which is not a build var")
        return values[match[1]]

    return [PLACEHOLDER.sub(value, arg) for arg in step]


def run_steps(steps: tuple[tuple[str, ...], ...], dest: Path, values: Mapping[str, str]) -> None:
    for step in steps:
        argv = expand(step, values)
        if subprocess.run(argv, cwd=dest).returncode != 0:
            raise BuildError(f"{' '.join(argv)} failed in {dest}")


def check_outputs(dest: Path, outputs: tuple[str, ...]) -> list[Path]:
    missing = [name for name in outputs if not (dest / name).is_file()]
    if missing:
        raise BuildError(f"the build wrote no {', '.join(missing)} in {dest}")
    return [dest / name for name in outputs]


def _executable(value: str) -> bool:
    path = shutil.which(value)
    return path is not None and os.access(path, os.X_OK)
