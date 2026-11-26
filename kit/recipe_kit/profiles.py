"""Profiles, arms, [[mode]] and [[switch]] rows: load them, resolve inheritance and render serve flags and env."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import NAME, VAR, ManifestError, Recipe, refuse_unknown

APPLIES = {"single", "tensor", "pipeline", "any"}
KINDS = {"product", "experiment"}
TOPOLOGY_FLAGS = {"tensor": "--tp", "pipeline": "--pp"}


@dataclass(frozen=True)
class Mode:
    """An adapter value a profile may name under `mode`, and how this pin's engine spells it."""

    slot: str
    value: str
    applies: str
    sets: dict[str, str]
    fields: dict

    @property
    def key(self) -> tuple[str, str]:
        return self.slot, self.value


@dataclass(frozen=True)
class Switch:
    """A product or experiment switch row; fields keeps every key the row was written with."""

    name: str
    kind: str
    fields: dict


@dataclass(frozen=True)
class Profile:
    """A profile as written: unset keys are None, so inheritance can tell them from values; in flags, when_model_has,
    mode and settings a false value drops the parent's key."""

    name: str
    engine: str | None = None
    inherits: str | None = None
    topology: dict | None = None
    context: int | None = None
    parallel: int | None = None
    flags: dict = field(default_factory=dict)
    when_model_has: dict = field(default_factory=dict)
    mode: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Arm:
    """A bench arm: a profile plus experiment switches."""

    name: str
    base: str
    experiments: dict[str, str]


@dataclass(frozen=True)
class Plan:
    modes: dict[tuple[str, str], Mode]
    switches: dict[str, Switch]
    profiles: dict[str, Profile]
    arms: dict[str, Arm]


@dataclass(frozen=True)
class Rendered:
    """What a target serves with: its resolved profile, serve flags and env, and where each env name came from;
    optional holds the flags passed only when the model directory has a file, by that file."""

    target: str
    profile: Profile
    args: list[str]
    env: dict[str, str]
    origin: dict[str, str]
    experiments: dict[str, str]
    optional: dict[str, list[str]] = field(default_factory=dict)

    def for_model(self, model: Path) -> list[str]:
        """The serve flags for this model directory: args, then each optional flag whose file it holds."""
        return self.args + [arg for file, args in self.optional.items() if (model / file).is_file() for arg in args]

    @property
    def every_arg(self) -> list[str]:
        return self.args + [arg for args in self.optional.values() for arg in args]

    @property
    def ranks(self) -> int:
        topology = self.profile.topology or {}
        return next(iter(topology.values()), 1)


def load(recipe: Recipe) -> Plan:
    data, where = recipe.data, str(recipe.path)
    modes: dict[tuple[str, str], Mode] = {}
    for index, row in enumerate(_rows(data, "mode", where)):
        here = f"{where} [[mode]] {index + 1}"
        refuse_unknown(row, "mode", here)
        mode = Mode(row.get("slot", ""), row.get("value", ""), row.get("applies", ""), _env(row, "sets", here), row)
        if mode.key in modes:
            raise ManifestError(f"{here}: mode {mode.slot} = {mode.value} is written twice")
        modes[mode.key] = mode
    switches: dict[str, Switch] = {}
    for index, row in enumerate(_rows(data, "switch", where)):
        here = f"{where} [[switch]] {index + 1}"
        refuse_unknown(row, "switch", here)
        name = row.get("name", "")
        if not VAR.fullmatch(name) or name in switches:
            raise ManifestError(f"{here}: name must be an UPPER_CASE engine name written once")
        switches[name] = Switch(name, row.get("kind", ""), row)
    profiles = {name: _profile(name, table, f"{where} [profile.{name}]") for name, table in _tables(data, "profile")}
    arms = {}
    for name, table in _tables(data, "arm"):
        here = f"{where} [arm.{name}]"
        refuse_unknown(table, "arm", here)
        if name in profiles or not isinstance(table.get("base"), str):
            raise ManifestError(f"{here}: an arm needs a base profile and a name no profile has")
        arms[name] = Arm(name, table["base"], _env(table, "experiments", here))
    return Plan(modes, switches, profiles, arms)


def resolve(plan: Plan, name: str) -> Profile:
    """NAME with every inherited key filled in from its parents; a cycle or a missing parent is refused."""
    chain, current = [], name
    while current is not None:
        if current in chain or current not in plan.profiles:
            raise ManifestError(f"profile {name}: {'cycle at' if current in chain else 'no profile'} {current}")
        chain.append(current)
        current = plan.profiles[current].inherits
    merged: dict = {}
    for link in reversed(chain):
        profile = plan.profiles[link]
        for key in ("engine", "topology", "context", "parallel"):
            if getattr(profile, key) is not None:
                merged[key] = getattr(profile, key)
        for key in ("flags", "when_model_has", "mode", "settings"):
            merged[key] = {**merged.get(key, {}), **getattr(profile, key)}
    for key in ("flags", "when_model_has", "mode", "settings"):
        merged[key] = {k: v for k, v in merged[key].items() if v is not False}
    merged["when_model_has"] = {k: v for k, v in merged["when_model_has"].items() if k in merged["flags"]}
    return Profile(name, **merged)


def topology_kind(profile: Profile) -> str:
    return next(iter(profile.topology), "single") if profile.topology else "single"


def engine_of(plan: Plan, target: str) -> str | None:
    """The [engine.NAME] a profile or an arm runs on (None: the recipe's only one)."""
    arm = plan.arms.get(target)
    if target not in plan.profiles and arm is None:
        raise ManifestError(f"{target} is neither a [profile.*] nor an [arm.*]")
    return resolve(plan, arm.base if arm else target).engine


def render(plan: Plan, target: str, hardware_env: dict[str, str]) -> Rendered:
    """Serve flags and env for a profile or an arm; a name set from two sources is refused."""
    arm = plan.arms.get(target)
    profile = resolve(plan, arm.base if arm else target)
    for key in ("topology", "context", "parallel"):
        if getattr(profile, key) is None:
            raise ManifestError(f"profile {profile.name}: {key} is not set by it or its parents")
    args = ["--context", str(profile.context), "--parallel", str(profile.parallel)]
    for kind, count in profile.topology.items():
        args += [TOPOLOGY_FLAGS[kind], str(count)]
    optional: dict[str, list[str]] = {}
    for flag, value in profile.flags.items():
        rendered = [f"--{flag}"] if value is True else [f"--{flag}", str(value)]
        if flag in profile.when_model_has:
            optional.setdefault(profile.when_model_has[flag], []).extend(rendered)
        else:
            args += rendered
    env: dict[str, str] = {}
    origin: dict[str, str] = {}

    def put(name: str, value: str, source: str) -> None:
        if name in env:
            raise ManifestError(f"{target}: {name} is set by both {origin[name]} and {source}")
        env[name], origin[name] = value, source

    for name, value in hardware_env.items():
        put(name, value, "hardware")
    for slot, value in profile.mode.items():
        mode = plan.modes.get((slot, value))
        if mode is None:
            raise ManifestError(f"profile {profile.name}: mode {slot} = {value} has no [[mode]] row")
        for name, setting in mode.sets.items():
            put(name, setting, f"mode {slot} = {value}")
    for name, value in profile.settings.items():
        put(name, value, f"settings of {profile.name}")
    experiments = dict(arm.experiments) if arm else {}
    for name, value in experiments.items():
        put(name, value, f"arm {target}")
    return Rendered(target, profile, args, env, origin, experiments, optional)


def parse_date(value: object, where: str) -> datetime.date:
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        raise ManifestError(f"{where}: {value!r} is not a YYYY-MM-DD date") from None


def _profile(name: str, table: dict, where: str) -> Profile:
    refuse_unknown(table, "profile", where)
    if not NAME.fullmatch(name):
        raise ManifestError(f"{where}: profile names are lower-case letters, digits and dashes")
    topology = table.get("topology")
    if topology is not None:
        if not isinstance(topology, dict) or len(topology) > 1 or not set(topology) <= set(TOPOLOGY_FLAGS):
            raise ManifestError(f"{where}: topology is {{}}, {{ tensor = N }} or {{ pipeline = N }}")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 2 for n in topology.values()):
            raise ManifestError(f"{where}: a topology degree is a whole number of at least 2")
    for key in ("context", "parallel"):
        value = table.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ManifestError(f"{where}: {key} is a positive whole number")
    flags = table.get("flags", {})
    if not isinstance(flags, dict) or not all(isinstance(v, (str, int, bool)) for v in flags.values()):
        raise ManifestError(f"{where}: flags map serve flag names (no --) to strings, numbers or booleans")
    when = table.get("when_model_has", {})
    if not isinstance(when, dict) or not all(
        v is False or (isinstance(v, str) and v and "/" not in v and v not in (".", "..")) for v in when.values()
    ):
        raise ManifestError(f"{where}: when_model_has maps serve flag names to a file name in the model directory")
    if unknown := [flag for flag, v in when.items() if v is not False and flag not in flags]:
        raise ManifestError(f"{where}: when_model_has names {', '.join(unknown)}, which its flags do not set")
    mode = table.get("mode", {})
    if not isinstance(mode, dict) or not all(isinstance(v, str) or v is False for v in mode.values()):
        raise ManifestError(f"{where}: mode maps slots to value names (false drops an inherited slot)")
    for key in ("engine", "inherits"):
        if key in table and not isinstance(table[key], str):
            raise ManifestError(f"{where}: {key} names another table")
    return Profile(
        name,
        table.get("engine"),
        table.get("inherits"),
        topology,
        table.get("context"),
        table.get("parallel"),
        dict(flags),
        dict(when),
        dict(mode),
        _env(table, "settings", where, unset=True),
    )


def _env(table: dict, key: str, where: str, unset: bool = False) -> dict:
    """KEY: UPPER_CASE names to strings (whole numbers written as strings); with unset, false drops a parent's."""
    value = table.get(key, {})
    ok = isinstance(value, dict) and all(
        VAR.fullmatch(k) and ((isinstance(v, (str, int)) and not isinstance(v, bool)) or (unset and v is False))
        for k, v in value.items()
    )
    if not ok:
        extra = " (false drops an inherited one)" if unset else ""
        raise ManifestError(f"{where}: {key} maps UPPER_CASE names to strings or whole numbers{extra}")
    return {k: v if v is False else str(v) for k, v in value.items()}


def _rows(data: dict, key: str, where: str) -> list[dict]:
    rows = data.get(key, [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ManifestError(f"{where}: {key} rows are written as [[{key}]] tables")
    return rows


def _tables(data: dict, key: str) -> list[tuple[str, dict]]:
    tables = data.get(key, {})
    if not isinstance(tables, dict) or not all(isinstance(v, dict) for v in tables.values()):
        raise ManifestError(f"[{key}.NAME] entries must be tables")
    return list(tables.items())
