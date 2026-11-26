"""Served gates. Each reads a server URL and returns one GateResult; each has a planted fault (see mock)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GateResult:
    gate: str
    passed: bool
    line: str
    skipped: bool = False
    seconds: float = 0.0
    rows: list = field(default_factory=list)

    def __str__(self) -> str:
        word = "SKIP" if self.skipped else "PASS" if self.passed else "FAIL"
        return f"{word} {self.gate}: {self.line} ({self.seconds:.2f} s)"


def skipped(gate: str, why: str) -> GateResult:
    return GateResult(gate, True, why, skipped=True)


def timed(run, *args, **kwargs) -> GateResult:
    start = time.perf_counter()
    result = run(*args, **kwargs)
    result.seconds = time.perf_counter() - start
    return result
