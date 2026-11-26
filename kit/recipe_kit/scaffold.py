"""recipe new: a recipe directory from templates/recipe, for a hardware profile and a list of engines."""

from __future__ import annotations

import shutil
from pathlib import Path

from . import hardware, revision
from .manifest import NAME, ManifestError

TEMPLATE = revision.KIT_ROOT / "templates" / "recipe"
ZERO = "0" * 40


def _swap(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ManifestError(f"{TEMPLATE}/recipe.toml no longer holds {old!r}: update recipe new with it")
    return text.replace(old, new)


def create(dest: Path, profile: str = "rtx6000x2", engines: list[str] | None = None) -> list[str]:
    """Write the template into DEST, a new directory named for the recipe; return what is left to fill in."""
    engines = engines or ["cuda"]
    name = dest.name
    if not NAME.fullmatch(name) or not all(NAME.fullmatch(e) for e in engines) or len(set(engines)) != len(engines):
        raise ManifestError(f"{name} {engines}: the recipe and engine names are distinct lower-case names")
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ManifestError(f"{dest} exists and is not an empty directory: recipe new writes a new one")
    hw = hardware.load(profile)
    text = (TEMPLATE / "recipe.toml").read_text()
    text = _swap(text, 'name = "modelx-rtx6000x2"', f'name = "{name}"')
    text = _swap(text, 'profile = "rtx6000x2"', f'profile = "{profile}"')
    text = _swap(text, 'GPU = "sm_120"', f'GPU = "{hw.gpu}"')
    text = _swap(
        text, "topology = { tensor = 2 }", f"topology = {{ tensor = {hw.ranks} }}" if hw.ranks > 1 else "topology = {}"
    )
    text = _swap(text, "[engine.cuda", f"[engine.{engines[0]}")
    todo = []
    rev, why = revision.current()
    if rev:
        text = _swap(text, f'rev = "{ZERO}"', f'rev = "{rev}"')
    else:
        todo.append(f"[kit] rev: {why}; set it to the kit commit you run, or pass --kit-dev")
    if len(engines) > 1:
        text = _swap(text, "[profile.default]\n", f'[profile.default]\nengine = "{engines[0]}"\n')
        for other in engines[1:]:
            text += f"\n# {other} starts as {engines[0]}'s pin and build: set what differs.\n[engine.{other}]\n"
            text += f'inherits = "{engines[0]}"\n\n[profile.{other}]\ninherits = "default"\nengine = "{other}"\n'
        todo.append(f"[engine.{', '.join(engines[1:])}]: each one's backend, pin and build")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    (dest / "recipe.toml").write_text(text)
    return [*todo, "[recipe] model, model_type, format and served_name", "[weights] rev and its sha256 file"]
