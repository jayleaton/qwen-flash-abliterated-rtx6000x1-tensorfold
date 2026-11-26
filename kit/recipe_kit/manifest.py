"""Read a recipe's recipe.toml into the parts the kit runs."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "recipe.toml"
HEX40 = re.compile(r"[0-9a-f]{40}")
GIT_URL = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?")
NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
VAR = re.compile(r"[A-Z_][A-Z0-9_]*")
VERSION_RANGE = re.compile(r"(>=|<=|==|>|<)\s*(\d+(\.\d+)*)")
SERIES = re.compile(r"(?P<dir>[A-Za-z0-9_][A-Za-z0-9_./-]*?)(:(?P<last>[0-9]{4}))?")
# Every key a table may hold; anything else is refused, so a typo cannot pass as a setting nobody reads.
KEYS = {
    "": {
        "recipe",
        "kit",
        "engine",
        "weights",
        "hardware",
        "profile",
        "arm",
        "mode",
        "switch",
        "gates",
        "bench",
        "soak",
        "publish",
    },
    "recipe": {"name", "model", "model_type", "format", "ships", "served_name", "port"},
    "kit": {"rev", "testkit"},
    "engine": {"inherits", "repo", "rev", "tree", "series", "build", "backend", "hardware", "format"},
    "build": {"vars", "executables", "steps", "outputs", "addon"},
    "addon": {"name", "var", "repo", "tag", "commit", "steps", "outputs"},
    "weights": {"rev", "sha256"},
    "hardware": {"profile"},
    "profile": {"engine", "inherits", "topology", "context", "parallel", "flags", "when_model_has", "mode", "settings"},
    "arm": {"base", "experiments"},
    "mode": {"slot", "value", "applies", "sets", "receipt", "expires"},
    "switch": {"name", "kind", "reason", "doc", "hardware", "receipt", "owner", "decides", "added", "expires", "exact"},
    "gates": {"conformance", "hashes", "checks", "mmlu200_min", "tool_chains"},
    "bench": {"cells", "rounds", "bar", "control"},
    "soak": {"minutes", "workers", "probe_min"},
    "publish": {"exclude"},
}


class ManifestError(Exception):
    """A recipe.toml that the kit refuses to run."""


@dataclass(frozen=True)
class Series:
    """One patch directory, applied 0001..last (every patch when last is None)."""

    directory: Path
    last: str | None


@dataclass(frozen=True)
class Addon:
    """A pinned checkout built into the engine tree after the main build, such as xgrammar."""

    name: str
    var: str
    repo: str
    tag: str
    commit: str
    steps: tuple[tuple[str, ...], ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class Build:
    """How to build a fetched engine tree: variables, commands and the files they must write."""

    variables: dict[str, str]
    executables: tuple[str, ...]
    steps: tuple[tuple[str, ...], ...]
    outputs: tuple[str, ...]
    addons: tuple[Addon, ...]


@dataclass(frozen=True)
class Engine:
    """One engine pin: upstream commit, patch series and the tree they must give; the backend it runs (the name,
    unless it says), and the hardware and weight format its profiles serve where they differ from the recipe's."""

    name: str
    repo: str
    rev: str
    tree: str
    series: tuple[Series, ...]
    build: Build | None
    backend: str = ""
    hardware: str | None = None
    format: str | None = None


@dataclass(frozen=True)
class Recipe:
    """A loaded recipe.toml and the directory it lives in; data keeps the tables later modules read."""

    root: Path
    name: str
    kit_rev: str
    testkit: str
    engines: dict[str, Engine]
    model: str
    model_type: str
    format: str
    ships: tuple[str, ...]
    served_name: str
    port: int
    hardware: str
    data: dict

    @property
    def path(self) -> Path:
        return self.root / MANIFEST

    def hardware_of(self, spec: Engine) -> str:
        return spec.hardware or self.hardware

    def format_of(self, spec: Engine) -> str:
        return spec.format or self.format

    def engine(self, name: str | None) -> Engine:
        if name is None:
            if len(self.engines) == 1:
                return next(iter(self.engines.values()))
            raise ManifestError(f"{self.path}: name an engine with --engine: {', '.join(self.engines)}")
        if name not in self.engines:
            raise ManifestError(f"{self.path}: no [engine.{name}]; engines: {', '.join(self.engines)}")
        return self.engines[name]


def load(root: Path) -> Recipe:
    path = root / MANIFEST
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ManifestError(f"no {path}") from None
    except tomllib.TOMLDecodeError as err:
        raise ManifestError(f"{path}: {err}") from None
    refuse_unknown(data, "", str(path))
    meta = _table(data, "recipe", path)
    refuse_unknown(meta, "recipe", f"{path} [recipe]")
    name = _text(meta, "name", f"{path} [recipe]")
    kit = _table(data, "kit", path)
    refuse_unknown(kit, "kit", f"{path} [kit]")
    kit_rev, testkit = _text(kit, "rev", f"{path} [kit]"), _text(kit, "testkit", f"{path} [kit]")
    if not HEX40.fullmatch(kit_rev):
        raise ManifestError(f"{path} [kit]: rev must be the 40-hex kit revision the recipe runs with")
    if not all(VERSION_RANGE.fullmatch(part.strip()) for part in testkit.split(",")):
        raise ManifestError(f'{path} [kit]: testkit must be a version range such as ">=0.1,<1"')
    engines = _table(data, "engine", path)
    if not engines:
        raise ManifestError(f"{path}: no [engine.<backend>] table")
    weights = _table(data, "weights", path)
    refuse_unknown(weights, "weights", f"{path} [weights]")
    for key in ("rev", "sha256"):
        _text(weights, key, f"{path} [weights]")
    hardware = _table(data, "hardware", path)
    refuse_unknown(hardware, "hardware", f"{path} [hardware]")
    for key in ("gates", "bench", "soak", "publish"):
        if key in data:
            refuse_unknown(_table(data, key, path), key, f"{path} [{key}]")
    port = meta.get("port", 8080)
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65535:
        raise ManifestError(f"{path} [recipe]: port must be a whole number from 1 to 65534")
    where = f"{path} [recipe]"
    served_name = _text(meta, "served_name", where) if "served_name" in meta else name
    return Recipe(
        root,
        name,
        kit_rev,
        testkit,
        {key: _engine(key, table, path) for key, table in _inherited(engines, path).items()},
        _text(meta, "model", where),
        _text(meta, "model_type", where),
        _text(meta, "format", where),
        _strings(meta, "ships", where),
        served_name,
        port,
        _text(hardware, "profile", f"{path} [hardware]"),
        data,
    )


def refuse_unknown(table: dict, kind: str, where: str) -> None:
    unknown = sorted(set(table) - KEYS[kind])
    if unknown:
        raise ManifestError(f"{where}: unknown key {', '.join(unknown)}; allowed: {', '.join(sorted(KEYS[kind]))}")


def in_range(version: str, spec: str) -> bool:
    """Whether VERSION meets every comma-separated bound in SPEC, such as ">=0.1,<1"."""
    have = _numbers(version)
    for part in spec.split(","):
        match = VERSION_RANGE.fullmatch(part.strip())
        if match is None:
            return False
        want, op = _numbers(match[2]), match[1]
        width = max(len(have), len(want))
        a, b = have + (0,) * (width - len(have)), want + (0,) * (width - len(want))
        if not {">=": a >= b, "<=": a <= b, "==": a == b, ">": a > b, "<": a < b}[op]:
            return False
    return True


def _numbers(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _inherited(engines: dict, path: Path) -> dict[str, dict]:
    """Each [engine.NAME] with its inherits chain merged in, parents first: tables merge key by key, any other value
    here replaces the parent's, and false drops the parent's key."""
    merged = {}
    for name in engines:
        chain, current = [], name
        while current is not None:
            if current in chain or not isinstance(engines.get(current), dict):
                why = "cycle at" if current in chain else "no engine"
                raise ManifestError(f"{path} [engine.{name}]: {why} {current}")
            chain.append(current)
            current = engines[current].get("inherits")
            if current is not None and not isinstance(current, str):
                raise ManifestError(f"{path} [engine.{name}]: inherits names another [engine.NAME]")
        table: dict = {}
        for link in reversed(chain):
            table = _merge(table, {k: v for k, v in engines[link].items() if k != "inherits"})
        merged[name] = table
    return merged


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if value is False:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _engine(name: str, table: dict, path: Path) -> Engine:
    where = f"{path} [engine.{name}]"
    refuse_unknown(table, "engine", where)
    if not NAME.fullmatch(name):
        raise ManifestError(f"{where}: engine names are lower-case letters, digits and dashes")
    repo = _text(table, "repo", where)
    if not GIT_URL.fullmatch(repo):
        raise ManifestError(f"{where}: repo must be an https://github.com/OWNER/NAME URL, not {repo!r}")
    rev, tree = _text(table, "rev", where), _text(table, "tree", where)
    if not HEX40.fullmatch(rev) or not HEX40.fullmatch(tree):
        raise ManifestError(f"{where}: rev and tree must each be 40 lower-case hex digits")
    series = []
    for entry in _strings(table, "series", where):
        match = SERIES.fullmatch(entry)
        if not match or ".." in Path(match["dir"]).parts:
            raise ManifestError(f"{where}: series entry {entry!r} is not DIR or DIR:NNNN inside the recipe")
        series.append(Series(Path(match["dir"]), match["last"]))
    build = table.get("build")
    backend = _text(table, "backend", where) if "backend" in table else name
    hardware = _text(table, "hardware", where) if "hardware" in table else None
    if not NAME.fullmatch(backend) or (hardware is not None and not NAME.fullmatch(hardware)):
        raise ManifestError(f"{where}: backend and hardware are lower-case names")
    fmt = _text(table, "format", where) if "format" in table else None
    return Engine(
        name,
        repo,
        rev,
        tree,
        tuple(series),
        _build(build, f"{where}.build") if build is not None else None,
        backend,
        hardware,
        fmt,
    )


def _build(table: dict, where: str) -> Build:
    refuse_unknown(table, "build", where)
    variables = table.get("vars", {})
    if not isinstance(variables, dict) or not all(
        VAR.fullmatch(k) and isinstance(v, str) for k, v in variables.items()
    ):
        raise ManifestError(f"{where}: vars must map UPPER_CASE names to strings")
    executables = _strings(table, "executables", where)
    unknown = [name for name in executables if name not in variables]
    if unknown:
        raise ManifestError(f"{where}: executables {unknown} are not in vars")
    addons = tuple(_addon(item, f"{where}.addon") for item in table.get("addon", []))
    names = [addon.var for addon in addons]
    if len(set(names)) != len(names) or set(names) & set(variables):
        raise ManifestError(f"{where}: each addon needs its own var, distinct from vars")
    return Build(
        dict(variables),
        executables,
        _steps(table, where, required=True),
        _strings(table, "outputs", where),
        addons,
    )


def _addon(table: dict, where: str) -> Addon:
    refuse_unknown(table, "addon", where)
    name = _text(table, "name", where)
    var, repo, commit = _text(table, "var", where), _text(table, "repo", where), _text(table, "commit", where)
    if not NAME.fullmatch(name) or not VAR.fullmatch(var):
        raise ManifestError(f"{where}: name is lower-case and var is UPPER_CASE")
    if not GIT_URL.fullmatch(repo) or not HEX40.fullmatch(commit):
        raise ManifestError(f"{where} {name}: repo must be a GitHub URL and commit 40 hex digits")
    return Addon(
        name, var, repo, _text(table, "tag", where), commit, _steps(table, where), _strings(table, "outputs", where)
    )


def _steps(table: dict, where: str, required: bool = False) -> tuple[tuple[str, ...], ...]:
    steps = table.get("steps", [])
    if not isinstance(steps, list) or not all(_is_argv(step) for step in steps) or (required and not steps):
        raise ManifestError(f"{where}: steps must be a list of non-empty argument lists")
    return tuple(tuple(step) for step in steps)


def _is_argv(step: object) -> bool:
    return isinstance(step, list) and bool(step) and all(isinstance(arg, str) for arg in step)


def _table(data: dict, key: str, path: Path) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: missing [{key}] table")
    return value


def _text(table: dict, key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: {key} must be a non-empty string")
    return value


def _strings(table: dict, key: str, where: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ManifestError(f"{where}: {key} must be a list of strings")
    return tuple(value)
