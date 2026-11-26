"""A server the testkit starts and stops itself: every rank in its own process group, ready when /health answers."""

from __future__ import annotations

import contextlib
import hashlib
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


class ServerError(OSError):
    pass


@dataclass(frozen=True)
class Rank:
    argv: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class ServerSpec:
    """What one arm serves with: its ranks (rank 0 first), the URL rank 0 answers on and the binary it runs."""

    name: str
    ranks: list[Rank]
    url: str
    binary: Path
    settings: dict = field(default_factory=dict)
    boot_timeout: float = 1200


class Server:
    def __init__(self, spec: ServerSpec, log_dir: Path):
        self.spec, self.log_dir, self.procs, self.boot_s = spec, log_dir, [], None

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        for i, rank in reversed(list(enumerate(self.spec.ranks))):
            with (self.log_dir / f"rank{i}.log").open("w") as log:
                proc = subprocess.Popen(
                    rank.argv,
                    env={**os.environ, **rank.env},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            self.procs.append(proc)
        deadline = start + self.spec.boot_timeout
        while time.perf_counter() < deadline:
            if any(p.poll() is not None for p in self.procs):
                self.__exit__()
                raise ServerError(f"{self.spec.name}: a rank exited during boot; see {self.log_dir}")
            if healthy(self.spec.url):
                self.boot_s = time.perf_counter() - start
                return self
            time.sleep(0.05 if time.perf_counter() - start < 5 else 1)
        self.__exit__()
        raise ServerError(f"{self.spec.name}: not healthy in {self.spec.boot_timeout:.0f} s; see {self.log_dir}")

    def __exit__(self, *_):
        for proc in self.procs:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + 90
        for proc in self.procs:
            try:
                proc.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        self.procs = []


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2) as reply:
            return reply.status == 200
    except OSError:
        return False


def binary_hashes(path: Path) -> dict[str, str]:
    md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 20):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def drop_page_cache() -> str:
    """Drops the page cache where this user may, so every arm loads its weights the same way; else says why not."""
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return "dropped"
    except OSError as err:
        return f"not dropped ({err.strerror.lower() if err.strerror else err})"
