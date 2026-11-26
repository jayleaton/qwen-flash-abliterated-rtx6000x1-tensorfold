"""ABBA: two arms on one host, each cell from a fresh server, judged against a bar; decide() replays recorded runs.

Cells are ``decode.<set>.x<N>`` (set code or chat: upstream bench_openai's prompts at 1 stream, bench_concurrent's
at N, with every concurrent reply checked against its alone and its undrafted run) and ``cold.<size>`` (bench_cache:
one cold prompt of that many token ids, its prompt tokens a second). Each server runs exactly as its profile serves;
a cold sample stays cold by a prefix no earlier request had (salted), not by a cache turned off, and bench_cache
requires the server to report it cold and its repeated and concurrent replies to equal the first. Every other
reply's token hash is checked against the recipe's expected hashes, or else the first sample's; a salted reply's
tokens differ from sample to sample, so its hash is recorded but not compared.

The order is A B B A for each two rounds, inside a control arm that runs first and last. The window is void when the
control moves more than the bar on any cell. B is better when the first cell gains the bar and no cell loses it,
worse when any cell loses it, otherwise no different; differing tokens decide before speed does.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .server import Server, ServerError, ServerSpec, binary_hashes, drop_page_cache

CELL = re.compile(r"decode\.(code|chat)\.x([1-9][0-9]*)|cold\.([1-9][0-9]*)")
OPENAI = {"fibonacci-raw": "code", "gpu-chat-no-think": "chat"}
TOKENS = 256
SALT_TOKENS = 16


class BenchError(ValueError):
    pass


@dataclass
class Sample:
    arm: str
    metrics: dict[str, float] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    unequal: int = 0
    problems: list[str] = field(default_factory=list)
    boot_s: float | None = None
    bench_s: float | None = None
    clients_s: float | None = None
    cell_s: float | None = None
    cache: str = ""
    salt: str = ""


def parse_cells(cells: list[str]) -> list[str]:
    if bad := [c for c in cells if not CELL.fullmatch(c)]:
        raise BenchError(f"cells {bad}: each is decode.code.xN, decode.chat.xN or cold.SIZE")
    if not cells:
        raise BenchError("no bench cells")
    return list(cells)


def salted(prompt: Path, salt: str, out: Path) -> Path:
    """PROMPT's token ids behind SALT_TOKENS of its own ids that SALT picks: a prefix no earlier request had, so a
    prompt cache misses (in memory or on disk, from an earlier server too), and every id is one the tokenizer gave."""
    ids = json.loads(Path(prompt).read_text())
    digest = hashlib.sha256(salt.encode()).digest()
    picks = [ids[int.from_bytes(digest[2 * i : 2 * i + 2], "big") % len(ids)] for i in range(SALT_TOKENS)]
    path = out / f"{Path(prompt).stem}-salted.json"
    path.write_text(json.dumps(picks + ids))
    return path


def new_salt() -> str:
    return secrets.token_hex(6)


def measure(
    url: str, model: str, tools: Path, cells: list[str], cold: dict[str, Path], out: Path, salt: str = ""
) -> Sample:
    """Runs the pinned tree's bench clients for these cells against a running server; the sample they record. With
    a salt, each cold prompt is salted with it."""
    out.mkdir(parents=True, exist_ok=True)
    levels = sorted({int(m[2]) for m in map(CELL.fullmatch, cells) if m[1] and int(m[2]) > 1})
    sizes = [m[3] for m in map(CELL.fullmatch, cells) if m[3]]
    runs = []
    if any(c.endswith(".x1") for c in cells):
        runs.append(
            (
                "openai",
                [
                    "bench_openai.py",
                    url,
                    model,
                    "--tokens",
                    str(TOKENS),
                    "--reps",
                    "1",
                    "--temperatures",
                    "0",
                    "--output",
                    str(out / "openai.json"),
                ],
                600,
            )
        )
    for size in sizes:
        if size not in cold:
            raise BenchError(f"cold.{size}: the hashes table names no prompt of {size} token ids")
        runs.append(
            (
                f"cold-{size}",
                [
                    "bench_cache.py",
                    url,
                    model,
                    "--tokens",
                    "16",
                    "--repeats",
                    "1",
                    "--streams",
                    "1",
                    "--require-cold",
                    "--prompt-file",
                    str(salted(cold[size], f"{salt}/{size}", out) if salt else cold[size]),
                    "--output",
                    str(out / f"cold-{size}.json"),
                ],
                900,
            )
        )
    if levels:
        runs.append(
            (
                "concurrent",
                [
                    "bench_concurrent.py",
                    url,
                    model,
                    "--tokens",
                    str(TOKENS),
                    "--reps",
                    "1",
                    "--temperatures",
                    "0",
                    "--levels",
                    ",".join(map(str, levels)),
                    "--alone",
                    "--serial",
                    "--output",
                    str(out / "concurrent.json"),
                ],
                1800,
            )
        )
    failed, clients = [], 0.0
    for name, argv, timeout in runs:
        begin = time.perf_counter()
        with (out / f"{name}.log").open("w") as log:
            code = subprocess.run(
                [sys.executable, str(tools / argv[0]), *argv[1:]], stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        clients += time.perf_counter() - begin
        if code:
            failed.append(f"{name} exited {code}")
    sample = read(out, cells)
    sample.problems, sample.clients_s, sample.salt = sample.problems + failed, clients, salt
    if salt:
        sample.hashes = {cell: sha for cell, sha in sample.hashes.items() if not cell.startswith("cold.")}
    return sample


def read(cell: Path, cells: list[str], arm: str = "") -> Sample:
    """A cell's sample from the files upstream's bench clients wrote: tok/s a cell, token hashes, unequal replies."""
    sample = Sample(arm)
    for row in _load(cell / "openai.json") or []:
        if row.get("temperature") == 0 and row.get("prompt") in OPENAI:
            name = OPENAI[row["prompt"]]
            if row.get("decode_tps_median"):
                sample.metrics[f"decode.{name}.x1"] = row["decode_tps_median"]
            if sha := _first_sha(row.get("evidence")):
                sample.hashes[f"decode.{name}.x1"] = sha
    for c in (_load(cell / "concurrent.json") or {}).get("cells", []):
        if c.get("temperature", 0) != 0 or c.get("streams") == 1:  # one stream is bench_openai's cell
            continue
        key = f"decode.{c['prompt']}.x{c['streams']}"
        sample.metrics[key] = c.get("aggregate_tps") or 0.0
        for check in ("alone", "serial"):
            sample.unequal += (c.get(check) or {}).get("unequal", 0) + (c.get(check) or {}).get("failed", 0)
        if sha := _first_sha(c.get("reps")):
            sample.hashes[key] = sha
    for path in cell.glob("cold-*.json"):
        rows = (_load(path) or {}).get("rows") or [{}]
        size = path.stem.split("-", 1)[1]
        if rows[0].get("ttft_s"):
            sample.metrics[f"cold.{size}"] = round(rows[0]["prompt_tokens"] / rows[0]["ttft_s"])  # whole tok/s
            sample.hashes[f"cold.{size}"] = rows[0].get("token_sha256") or rows[0].get("token_sha") or ""
    sample.metrics = {c: sample.metrics[c] for c in cells if c in sample.metrics}
    sample.hashes = {c: sample.hashes[c] for c in cells if c in sample.hashes}
    sample.problems += [f"{c} not measured" for c in cells if c not in sample.metrics]
    return sample


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _first_sha(x) -> str | None:
    if isinstance(x, dict):
        if isinstance(x.get("token_sha"), str):
            return x.get("token_sha256") or x["token_sha"]
        x = list(x.values())
    for v in x if isinstance(x, list) else []:
        if found := _first_sha(v):
            return found
    return None


def order(a: str, b: str, rounds: int, control: str | None) -> list[str]:
    """A B B A for each two rounds (one round: A B), inside the control arm."""
    if rounds < 1 or (rounds > 1 and rounds % 2):
        raise BenchError(f"rounds {rounds}: 1 (a quick A B pass, not a verdict) or an even number")
    middle = [a, b] if rounds == 1 else [a, b, b, a] * (rounds // 2)
    return [control, *middle, control] if control else middle


def abba(
    arms: dict[str, ServerSpec],
    a: str,
    b: str,
    control: str | None,
    cells: list[str],
    rounds: int,
    bar: float,
    out: Path,
    tools: Path,
    model: str,
    cold: dict[str, Path],
    expected: dict[str, str],
    drop_cache: bool = True,
    log=print,
) -> dict:
    """Runs the order, one fresh server a cell, and decides; the result is a receipt's body."""
    cells, samples, start, run = parse_cells(cells), [], time.perf_counter(), new_salt()
    binaries = {name: binary_hashes(arms[name].binary) for name in dict.fromkeys(filter(None, (a, b, control)))}
    for i, name in enumerate(order(a, b, rounds, control), 1):
        cell, where = time.perf_counter(), out / f"{i:02d}-{name}"
        cache = drop_page_cache() if drop_cache else "kept"
        sample = Sample(name, problems=[f"{c} not measured" for c in cells])
        for attempt in (1, 2):  # a boot that fails is tried once more, as today's scripts do
            try:
                with Server(arms[name], where) as server:
                    begin = time.perf_counter()
                    sample = measure(arms[name].url, model, tools, cells, cold, where, f"{run}-{i:02d}")
                    sample.arm, sample.boot_s, sample.cache = name, server.boot_s, cache
                    sample.bench_s = time.perf_counter() - begin
                break
            except ServerError as err:
                sample.problems.append(f"boot {attempt}: {str(err).split(';')[0]}")
        sample.cell_s = time.perf_counter() - cell
        boot = f"boot {sample.boot_s:.2f} s, bench {sample.bench_s:.2f} s" if sample.boot_s else "no boot"
        log(f"cell {i} {name}: page cache {cache}, {boot}; " + line(sample))
        samples.append(sample)
    verdict = decide(samples, a, b, bar, control, expected, quick=rounds == 1)
    return {
        "arms": {n: {**arms[n].settings, "binary": binaries[n]} for n in binaries},
        "order": [s.arm for s in samples],
        "cells": cells,
        "bar": bar,
        "samples": [asdict(s) for s in samples],
        "verdict": verdict,
        "seconds": round(time.perf_counter() - start, 3),
    }


def line(sample: Sample) -> str:
    parts = [f"{k}={v:.1f}" for k, v in sample.metrics.items()]
    return " ".join(parts + [f"unequal={sample.unequal}"] + [f"PROBLEM {p}" for p in sample.problems])


def decide(
    samples: list[Sample],
    a: str,
    b: str,
    bar: float,
    control: str | None = None,
    expected: dict[str, str] | None = None,
    quick: bool = False,
) -> dict:
    """The verdict on B against A over recorded samples; pure, so a recorded window replays to its verdict."""
    cells = list(dict.fromkeys(k for s in samples if s.arm in (a, b) for k in s.metrics))
    by = {name: [s for s in samples if s.arm == name] for name in dict.fromkeys(s.arm for s in samples)}
    for name in (a, b):
        if not by.get(name):
            raise BenchError(f"no samples of arm {name}")
    mean = {
        name: {
            c: statistics.mean(s.metrics[c] for s in got if c in s.metrics)
            for c in cells
            if any(c in s.metrics for s in got)
        }
        for name, got in by.items()
    }
    change = {c: mean[b][c] / mean[a][c] - 1 for c in cells if mean[a].get(c) and c in mean[b]}
    tokens = token_problems(samples, expected or {})
    problems = [f"{s.arm}: {p}" for s in samples for p in s.problems] + tokens
    problems += [
        f"{s.arm}: {s.unequal} concurrent replies differ from their alone or undrafted run"
        for s in samples
        if s.unequal
    ]
    drift = {}
    if control and len(by.get(control, [])) >= 2:
        first, last = by[control][0].metrics, by[control][-1].metrics
        drift = {c: last[c] / first[c] - 1 for c in cells if first.get(c) and c in last}
    moved = {c: d for c, d in drift.items() if abs(d) > bar}
    primary = cells[0] if cells else None
    worse = {c: d for c, d in change.items() if d <= -bar}
    if problems:
        word = "tokens differ" if tokens else "unmeasured"
    elif moved:
        word = "void"
    elif worse:
        word = "worse"
    elif primary in change and change[primary] >= bar:
        word = "better"
    else:
        word = "no difference"
    return {
        "a": a,
        "b": b,
        "bar": bar,
        "verdict": word,
        "quick": quick,
        "primary": primary,
        "means": mean,
        "change": change,
        "control": control,
        "control_drift": drift,
        "problems": problems,
    }


def token_problems(samples: list[Sample], expected: dict[str, str]) -> list[str]:
    """Each hash that is not the expected one: the recipe's (by cell, or by its set for every stream count) where it
    has one, else the first sample's."""
    seen, out = {}, []
    for s in samples:
        for key, sha in s.hashes.items():
            want = seen.setdefault(key, expected.get(key) or expected.get(key.rsplit(".", 1)[0]) or sha)
            n = min(len(want), len(sha))
            if not sha or not n or sha[:n] != want[:n]:
                out.append(f"{s.arm} {key}: tokens {sha[:12] or 'missing'}, expected {want[:12]}")
    return out


def table(verdict: dict) -> str:
    """The verdict as a markdown table: each arm's mean a cell and its change against A."""
    a, means = verdict["a"], verdict["means"]
    cells = list(verdict["change"]) or list(means[a])
    rows = ["| arm | " + " | ".join(cells) + " |", "|---" * (len(cells) + 1) + "|"]
    for name, got in means.items():
        parts = [
            f"{got[c]:.1f} ({(got[c] / means[a][c] - 1) * 100:+.1f}%)" if c in got and means[a].get(c) else "-"
            for c in cells
        ]
        rows.append(f"| {name} | " + " | ".join(parts) + " |")
    drift = verdict["control_drift"]
    tail = [f"control {verdict['control']} moved " + ", ".join(f"{c} {d * 100:+.1f}%" for c, d in drift.items())]
    tail = tail if drift else []
    word = verdict["verdict"] + (" (quick: one A B pass, not a verdict)" if verdict["quick"] else "")
    change = verdict["change"].get(verdict["primary"])
    gain = f"; {verdict['primary']} {change * 100:+.1f}%" if change is not None else ""
    head = f"{verdict['b']} against {verdict['a']}, bar {verdict['bar'] * 100:.1f}%: {word}{gain}"
    return "\n".join([*rows, "", *tail, head, *verdict["problems"]])
