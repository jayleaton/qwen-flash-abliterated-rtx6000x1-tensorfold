"""recipe check: rules C1 to C9 over a recipe's modes, switches, profiles and arms (KIT-SPEC section 4)."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from .manifest import ManifestError
from .profiles import APPLIES, KINDS, Plan, parse_date, resolve, topology_kind

# C1: the fields each row kind needs, and the ones it may not carry.
NEEDS = {
    "mode": {"slot", "value", "applies", "sets", "receipt", "expires"},
    "product": {"name", "kind", "reason", "doc", "hardware"},
    "experiment": {"name", "kind", "owner", "decides", "added", "expires"},
}
FORBIDS = {"product": {"owner", "decides", "added", "expires"}, "experiment": {"reason", "doc", "hardware"}}
MAX_EXPERIMENT_DAYS = 90


@dataclass(frozen=True)
class Finding:
    rule: str
    row: str
    message: str

    def __str__(self) -> str:
        return f"{self.rule} {self.row}: {self.message}"


def check(
    plan: Plan,
    hardware_names: set[str],
    today: datetime.date,
    engine_names: dict[str, str | None] | None = None,
    version: str | None = None,
    engine: str | None = None,
) -> list[Finding]:
    """Every finding; engine_names (name -> retire release) enables C8 against the pinned tree, for the names set
    on that engine's profiles and arms when engine is given (every name otherwise)."""
    found: list[Finding] = []
    add = lambda rule, row, message: found.append(Finding(rule, row, message))  # noqa: E731
    for key, mode in plan.modes.items():
        row = f"[[mode]] {key[0]} = {key[1]}"
        missing = NEEDS["mode"] - {k for k, v in mode.fields.items() if v not in ("", None, {})}
        if missing:
            add("C1", row, f"needs {', '.join(sorted(missing))}")
        if mode.applies and mode.applies not in APPLIES:
            add("C1", row, f"applies must be one of {', '.join(sorted(APPLIES))}")
        if "expires" in mode.fields and _date(mode.fields["expires"], row, add) < today:
            add("C3", row, f"expired on {mode.fields['expires']}")
        if mode.fields.get("exact") is False:
            add("C9", row, "is marked exact = false")
    for name, switch in plan.switches.items():
        row = f"[[switch]] {name}"
        if switch.kind not in KINDS:
            add("C1", row, f"kind must be one of {', '.join(sorted(KINDS))}")
            continue
        present = {k for k, v in switch.fields.items() if v not in ("", None, [])}
        if NEEDS[switch.kind] - present:
            add("C1", row, f"a {switch.kind} row needs {', '.join(sorted(NEEDS[switch.kind] - present))}")
        if FORBIDS[switch.kind] & set(switch.fields):
            add(
                "C1",
                row,
                f"a {switch.kind} row may not carry {', '.join(sorted(FORBIDS[switch.kind] & set(switch.fields)))}",
            )
        if switch.kind == "experiment" and {"added", "expires"} <= present:
            added, expires = _date(switch.fields["added"], row, add), _date(switch.fields["expires"], row, add)
            if expires < today:
                add("C3", row, f"expired on {expires}")
            if (expires - added).days > MAX_EXPERIMENT_DAYS:
                add("C4", row, f"runs {(expires - added).days} days; an experiment may run {MAX_EXPERIMENT_DAYS}")
        if switch.kind == "experiment" and not (switch.fields.get("owner") and switch.fields.get("decides")):
            add("C4", row, "an experiment needs an owner and what decides it")
        if switch.fields.get("exact") is False:
            add("C9", row, "is marked exact = false")
    used_modes, used_switches = set(), set()
    for name in plan.profiles:
        try:
            profile = resolve(plan, name)
        except ManifestError as err:
            add("C1", f"[profile.{name}]", str(err))
            continue
        kind = topology_kind(profile) if profile.topology is not None else None
        for slot, value in profile.mode.items():
            used_modes.add((slot, value))
            mode = plan.modes.get((slot, value))
            if mode is None:
                add("C6", f"[profile.{name}]", f"mode {slot} = {value} has no [[mode]] row")
            elif kind and mode.applies not in ("any", kind):
                add("C6", f"[profile.{name}]", f"mode {slot} = {value} applies to {mode.applies}, not {kind}")
        for setting in profile.settings:
            used_switches.add(setting)
            switch = plan.switches.get(setting)
            if switch is None:
                add("C2", f"[profile.{name}]", f"{setting} has no [[switch]] row")
            elif switch.kind == "experiment":
                add("C5", f"[profile.{name}]", f"sets experiment {setting}; only an [arm.*] may")
            if setting in hardware_names:
                add("C2", f"[profile.{name}]", f"{setting} is the hardware profile's")
    for name, arm in plan.arms.items():
        if arm.base not in plan.profiles:
            add("C1", f"[arm.{name}]", f"base {arm.base} is not a profile")
        for setting in arm.experiments:
            used_switches.add(setting)
            switch = plan.switches.get(setting)
            if switch is None:
                add("C2", f"[arm.{name}]", f"{setting} has no [[switch]] row")
            elif switch.kind != "experiment":
                add("C2", f"[arm.{name}]", f"{setting} is a {switch.kind} switch; an arm adds experiments")
    for key in plan.modes.keys() - used_modes:
        add("C7", f"[[mode]] {key[0]} = {key[1]}", "is set by no profile")
    for name in plan.switches.keys() - used_switches:
        add("C7", f"[[switch]] {name}", "is set by no profile and no arm")
    if engine_names is not None:
        names = {n for mode in plan.modes.values() for n in mode.sets} | set(plan.switches)
        if engine is not None:
            names = set_on(plan, engine)
        for name in sorted(names):
            if name not in engine_names:
                add("C8", name, "the pinned engine does not read it")
            elif engine_names[name] and version and _release(engine_names[name]) <= _release(version):
                add("C8", name, f"retired in {engine_names[name]}, at or before the pinned {version}")
    return found


def set_on(plan: Plan, engine: str) -> set[str]:
    """The env names the profiles and arms on one engine set: their settings, their modes' and the arms' switches."""
    names: set[str] = set()
    on = {name for name in plan.profiles if _engine(plan, name) == engine}
    for name in on:
        profile = resolve(plan, name)
        names |= set(profile.settings)
        modes = [plan.modes.get(key) for key in profile.mode.items()]
        names |= {n for mode in modes if mode for n in mode.sets}
    return names | {n for arm in plan.arms.values() if arm.base in on for n in arm.experiments}


def _engine(plan: Plan, name: str) -> str | None:
    try:
        return resolve(plan, name).engine
    except ManifestError:
        return None


def _date(value: object, row: str, add) -> datetime.date:
    try:
        return parse_date(value, row)
    except ManifestError as err:
        add("C1", row, str(err))
        return datetime.date.max


def _release(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())
