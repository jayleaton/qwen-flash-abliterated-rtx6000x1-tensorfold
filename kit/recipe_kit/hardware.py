"""Read a shared hardware profile, hardware/NAME.toml, in the format of hardware/README.md."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import node
from .manifest import NAME, VAR, ManifestError

HARDWARE_DIR = Path(__file__).resolve().parents[2] / "hardware"
KEYS = {
    "": {"chip", "comm", "memory", "thermal", "power", "bench"},
    "chip": {"arch", "gpu", "ranks", "unified"},
    "comm": {"transport", "env"},
    "memory": {"floor_gib", "gate_gib", "drop_caches"},
    "thermal": {"pause_c", "resume_c", "hard_c", "hard_hold_s"},
    "power": {"limit_w", "clock_max_mhz"},
    "bench": {"note"},
}
TRANSPORTS = {"pcie-p2p", "nvlink", "roce", "none"}


@dataclass(frozen=True)
class Hardware:
    """The facts a recipe's launch reads from its hardware profile."""

    name: str
    arch: int | str
    gpu: str
    ranks: int
    transport: str
    env: dict[str, str]
    unified: bool = False
    floor_gib: float | None = None
    drop_caches: bool = False
    thermal: dict[str, float] = field(default_factory=dict)
    power: dict[str, float] = field(default_factory=dict)


def load(name: str, directory: Path = HARDWARE_DIR) -> Hardware:
    path = directory / f"{name}.toml"
    if not NAME.fullmatch(name) or not path.is_file():
        raise ManifestError(f"no hardware profile {name} ({path})")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as err:
        raise ManifestError(f"{path}: {err}") from None
    for table, allowed in KEYS.items():
        section = data if not table else data.get(table, {})
        unknown = sorted(set(section) - allowed)
        if not isinstance(section, dict) or unknown:
            raise ManifestError(f"{path} [{table or 'top'}]: unknown key {', '.join(unknown)}")
    chip, comm = data.get("chip", {}), data.get("comm", {})
    arch, gpu, ranks = chip.get("arch"), chip.get("gpu"), chip.get("ranks")
    if not isinstance(arch, (int, str)) or not isinstance(gpu, str) or not isinstance(ranks, int) or ranks < 1:
        raise ManifestError(f"{path} [chip]: arch, gpu and ranks (a positive count) are required")
    transport, env = comm.get("transport"), comm.get("env", {})
    if transport not in TRANSPORTS:
        raise ManifestError(f"{path} [comm]: transport must be one of {', '.join(sorted(TRANSPORTS))}")
    if not isinstance(env, dict) or not all(VAR.fullmatch(k) and isinstance(v, str) for k, v in env.items()):
        raise ManifestError(f"{path} [comm]: env maps UPPER_CASE names to strings")
    unknown = [f"{k} = {v}" for k, v in env.items() if v.startswith("@") and v[1:] not in node.NAMES]
    if unknown:
        raise ManifestError(f"{path} [comm]: {', '.join(unknown)}: the node resolves {', '.join(node.NAMES)}")
    memory, thermal, power = data.get("memory", {}), data.get("thermal", {}), data.get("power", {})
    for table, section in (
        ("memory", {k: memory[k] for k in ("floor_gib",) if k in memory}),
        ("thermal", thermal),
        ("power", power),
    ):
        if bad := [k for k, v in section.items() if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0]:
            raise ManifestError(f"{path} [{table}]: {', '.join(bad)} must be numbers")
    if not isinstance(memory.get("drop_caches", False), bool) or not isinstance(chip.get("unified", False), bool):
        raise ManifestError(f"{path}: [chip] unified and [memory] drop_caches are true or false")
    ladder = [thermal[k] for k in ("resume_c", "pause_c", "hard_c") if k in thermal]
    if ladder != sorted(ladder):
        raise ManifestError(f"{path} [thermal]: resume_c, pause_c and hard_c must rise in that order")
    return Hardware(
        name,
        arch,
        gpu,
        ranks,
        transport,
        dict(env),
        chip.get("unified", False),
        memory.get("floor_gib"),
        memory.get("drop_caches", False),
        dict(thermal),
        dict(power),
    )
