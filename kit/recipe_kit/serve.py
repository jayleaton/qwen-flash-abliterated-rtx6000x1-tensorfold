"""Launch a rendered profile on one host: rank 1 and up in the background, rank 0 in the foreground."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from . import node
from .manifest import ManifestError, Recipe
from .profiles import Rendered

MASTER = "127.0.0.1"
MODEL = "{model}"  # a flag value that is the MODEL argument, such as a drafter read from the checkpoint


@dataclass(frozen=True)
class Rank:
    """One process to start: its argv, the env names the launch adds, and where its output goes."""

    rank: int
    argv: list[str]
    env: dict[str, str]
    log: Path | None


def ranks(
    recipe: Recipe,
    rendered: Rendered,
    binary: Path,
    model: Path,
    extra: list[str],
    gpus: list[str] | None,
    port: int,
    environ: Mapping[str, str],
    numa: str | None = None,
    log_dir: Path | None = None,
    resolve=node.resolve,
) -> list[Rank]:
    """Each rank's launch; a rendered name the environment already sets keeps the environment's value, and an
    "@name" value is read on this node (resolve None leaves it, for a dry run). With no gpus given, ranks take
    devices 0, 1, ... and one rank takes the environment's."""
    count = rendered.ranks
    pinned = gpus is not None or count > 1
    gpus = gpus if gpus is not None else [str(n) for n in range(count)]
    if len(gpus) != count or len(set(gpus)) != count or not all(g.isdigit() for g in gpus):
        raise ManifestError(f"{rendered.target} runs {count} rank(s): give that many distinct GPU ordinals, like 0,1")
    added = {name: value for name, value in rendered.env.items() if name not in environ}
    added = resolve(added) if resolve else added
    wrap = ["numactl", f"--cpunodebind={numa}", f"--preferred={numa}"] if numa else []
    common = [str(binary), "serve", str(model), "--name", recipe.served_name]
    common += [str(model) if arg == MODEL else arg for arg in rendered.for_model(model)] + extra
    launches = []
    for rank in range(count):
        tail = (
            ["--master", MASTER, "--rank", str(rank), "--port", str(port + rank)]
            if count > 1
            else ["--port", str(port)]
        )
        log = (log_dir or binary.parent) / f"rank{rank}.log" if rank else None
        devices = {"CUDA_VISIBLE_DEVICES": gpus[rank]} if pinned else {}
        launches.append(Rank(rank, wrap + common + tail, {**added, **devices}, log))
    return launches


def run(launches: list[Rank], environ: Mapping[str, str] = os.environ) -> int:
    """Start every rank, wait for rank 0, then stop the rest; returns rank 0's exit code."""
    background: list[subprocess.Popen] = []
    lead: subprocess.Popen | None = None

    def stop(*_: object) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        for launch in reversed(launches):
            # As a shell does when it starts a command: _ is the command's resolved path.
            env = {**environ, **launch.env, "_": shutil.which(launch.argv[0]) or launch.argv[0]}
            if launch.log is None:
                print(f"rank 0 on GPU {launch.env.get('CUDA_VISIBLE_DEVICES', '(the environment)')}", file=sys.stderr)
                lead = subprocess.Popen(launch.argv, env=env)
            else:
                with launch.log.open("w") as log:
                    proc = subprocess.Popen(launch.argv, env=env, stdout=log, stderr=subprocess.STDOUT)
                background.append(proc)
                gpu = launch.env["CUDA_VISIBLE_DEVICES"]
                print(f"rank {launch.rank} on GPU {gpu} (pid {proc.pid}, log {launch.log})", file=sys.stderr)
        return lead.wait() if lead else 0
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        for proc in [*background, *([lead] if lead else [])]:
            if proc.poll() is None:
                proc.terminate()
        for proc in [*background, *([lead] if lead else [])]:
            proc.wait()
