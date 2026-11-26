"""Soak: mixed traffic with cancels against a running server, fixed probes between, and a verdict a log replays.

Traffic is the recipes' bench/soak.py mix: each request draws a kind (code, prose, structured, tool, thinking, long,
nonstream, disconnect); about 15% of streams are cancelled at a random chunk and disconnects drop a reply mid-way.
Both are intended, not errors. A controller moves the active workers between 1 and ``workers``. Every ``probe_s``
the traffic pauses and the pinned tree's bench clients take the probe cells (soak.sh's probes): their token hashes
must not move. Free memory and temperature are sampled through the run. At the end the server must drain (nothing
running or waiting in ``GET /v1/load``), answer the canary, and have kept free memory above the hardware's floor.
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import bench

KINDS = [
    ("code", 22),
    ("prose", 18),
    ("structured", 14),
    ("tool", 14),
    ("thinking", 10),
    ("long", 10),
    ("nonstream", 7),
    ("disconnect", 5),
]
CODE = [
    "Write a Python function that merges two sorted lists, with a docstring and two doctests.",
    "Implement an LRU cache class in Python with get / put in O(1). Code only.",
    "Write a bash script that rotates log files older than 7 days into a tar.gz archive.",
]
PROSE = [
    "Write a short essay about the history of lighthouses.",
    "Describe a rainy morning in a harbor town.",
    "Explain to a ten-year-old why the sky is blue, in two paragraphs.",
]
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["title", "year", "tags"],
}
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
WORDS = [
    "amber",
    "basin",
    "cobalt",
    "drift",
    "ember",
    "fable",
    "grove",
    "harbor",
    "island",
    "jetty",
    "kelp",
    "lumen",
    "marsh",
    "north",
    "orbit",
    "pine",
    "quay",
]
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
CANARY, CANARY_ANSWER = "What is 17*23? Answer with the number only.", "391"
PROBE_CELLS = ["decode.code.x1", "decode.chat.x1"]


@dataclass
class Options:
    minutes: float = 30
    workers: int = 4
    probe_s: float = 300
    cancel_p: float = 0.15
    timeout: float = 900
    drain_s: float = 120
    seed: int = 6
    step_s: tuple[float, float] = (45, 120)
    poll_s: float = 10
    context: int = 49152


@dataclass
class Floors:
    """The hardware profile's limits: free memory a soak keeps above, the temperature that stops a run when it holds
    hold_s, and each card's power limit (a capped card's numbers do not compare: noted, not failed)."""

    memory_gib: float | None = None
    hard_c: float | None = None
    hold_s: float = 0
    power_w: float | None = None


def body_for(kind: str, model: str, rng: random.Random, context: int) -> tuple[dict, bool]:
    """A request of this kind and whether it streams."""
    t = rng.choice([0.0, 0.7])
    if kind == "code":
        b = dict(OFF, messages=[{"role": "user", "content": rng.choice(CODE)}], max_tokens=384, temperature=t)
    elif kind in ("prose", "nonstream", "disconnect"):
        b = dict(OFF, messages=[{"role": "user", "content": rng.choice(PROSE)}], max_tokens=384, temperature=t)
    elif kind == "structured":
        schema = {"type": "json_schema", "json_schema": {"name": "film", "schema": SCHEMA}}
        b = dict(OFF, messages=[{"role": "user", "content": "Describe a classic film as JSON."}], max_tokens=256)
        b.update(temperature=0, response_format=schema)
    elif kind == "tool":
        ask = f"What is the weather in {rng.choice(['Oslo', 'Lima', 'Kyiv'])}? Call the get_weather tool."
        b = dict(OFF, messages=[{"role": "user", "content": ask}], tools=[TOOL], tool_choice="auto", max_tokens=256)
        b["temperature"] = 0
    elif kind == "thinking":
        ask = "How many weekdays are there in March 2027? Think it through."
        b = {
            "messages": [{"role": "user", "content": ask}],
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
            "max_tokens": 2048,
            "temperature": t,
        }
    else:  # long: a document of up to 48K tokens that fits the window, and one question
        n = min(rng.choice([8, 16, 32, 48]) * 1024, context // 2)
        doc = " ".join(f"Entry {i}: the {rng.choice(WORDS)} met the {rng.choice(WORDS)}." for i in range(n // 11))
        ask = doc + "\n\nHow many entries are there? One number."
        b = dict(OFF, messages=[{"role": "user", "content": ask}], max_tokens=32, temperature=0)
    b["model"] = model
    return b, kind not in ("nonstream", "disconnect")


class Soak:
    def __init__(self, url: str, model: str, options: Options, meter=None, alive=None):
        self.url, self.model, self.o = url.rstrip("/"), model, options
        self.meter, self.alive = meter or (lambda: {}), alive or (lambda: True)
        self.lock, self.stop, self.pause = threading.Lock(), threading.Event(), threading.Event()
        self.active, self.busy, self.rec, self.samples, self.rank_exit = 1, 0, [], [], False

    def post(self, body: dict, timeout: float):
        request = urllib.request.Request(
            self.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        return urllib.request.urlopen(request, timeout=timeout)

    def one(self, kind: str, rng: random.Random) -> dict:
        body, streamed = body_for(kind, self.model, rng, self.o.context)
        r = {"kind": kind, "t": round(time.time() - self.t0, 1), "error": None, "check": None, "cancel": False}
        cancel_at = rng.randint(1, 20) if streamed and rng.random() < self.o.cancel_p else None
        timeout = rng.uniform(2, 5) if kind == "disconnect" else self.o.timeout
        body["stream"] = streamed
        t0, text, content, calls, chunks = time.time(), "", "", 0, 0
        try:
            with self.post(body, timeout) as resp:
                if not streamed:
                    m = json.loads(resp.read())["choices"][0]["message"]
                    content = m.get("content") or ""
                    text, calls = content + (m.get("reasoning_content") or ""), len(m.get("tool_calls") or [])
                for raw in resp if streamed else ():
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    c = json.loads(line[5:])
                    if c.get("error"):  # an error event after the 200 headers
                        r["error"] = ("stream error: " + str(c["error"]))[:200]
                        break
                    for ch in c.get("choices") or []:
                        d = ch.get("delta") or {}
                        if r.get("ttft_s") is None and (d.get("content") or d.get("reasoning_content")):
                            r["ttft_s"] = round(time.time() - t0, 3)
                        content += d.get("content") or ""
                        text += (d.get("content") or "") + (d.get("reasoning_content") or "")
                        calls += len(d.get("tool_calls") or [])
                    chunks += 1
                    if cancel_at is not None and chunks >= cancel_at:
                        r["cancel"] = True
                        break  # leaving the block closes the socket mid-stream
        except urllib.error.HTTPError as err:
            with err:
                r["error"] = f"HTTP {err.code}: " + err.read(400).decode(errors="replace")[:200]
        except TimeoutError as err:
            if kind == "disconnect":
                r["cancel"] = True
            else:
                r["error"] = f"timeout: {err}"
        except (OSError, ValueError) as err:
            r["error"] = f"{type(err).__name__}: {err}"[:200]
        r["total_s"] = round(time.time() - t0, 3)
        if r["error"] is None and not r["cancel"]:
            if kind == "structured":
                try:
                    json.loads(content.strip().removeprefix("```json").removesuffix("```"))
                except ValueError:
                    r["check"] = "structured reply not JSON"
            elif kind == "tool" and not calls:
                r["check"] = "no tool call"
            elif kind != "tool" and not text.strip():
                r["error"] = "empty reply"
        return r

    def worker(self, i: int) -> None:
        rng = random.Random(self.o.seed * 100 + i)
        names, weights = zip(*KINDS, strict=True)
        while not self.stop.is_set():
            with self.lock:
                go = i < self.active and not self.pause.is_set()
                self.busy += go
            if not go:
                time.sleep(0.05)
                continue
            try:
                r = self.one(rng.choices(names, weights)[0], rng)
            finally:
                with self.lock:
                    self.busy -= 1
            with self.lock:
                self.rec.append(r)

    def control(self) -> None:
        rng = random.Random(self.o.seed)
        while not self.stop.wait(rng.uniform(*self.o.step_s)):
            self.active = rng.randint(1, self.o.workers)

    def poll(self) -> None:
        while not self.stop.is_set():
            self.samples.append({"t": round(time.time() - self.t0, 1), **load(self.url), **self.meter()})
            if not self.alive():
                self.rank_exit = True
                self.stop.set()
            self.stop.wait(self.o.poll_s)

    def quiet(self) -> None:
        """Pauses the traffic and waits for the requests in flight to end."""
        self.pause.set()
        deadline = time.time() + self.o.timeout
        while self.busy and time.time() < deadline:
            time.sleep(0.05)

    def run(self, probe) -> dict:
        """probe(i) takes the fixed probes with no traffic running; returns the record verdict() reads."""
        self.t0 = time.time()
        threads = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(self.o.workers)]
        threads += [threading.Thread(target=f, daemon=True) for f in (self.control, self.poll)]
        for t in threads:
            t.start()
        probes, end, i = [], self.t0 + self.o.minutes * 60, 0
        while not self.stop.is_set():
            self.quiet()
            probes.append({"t": round(time.time() - self.t0, 1), **probe(i)})
            self.pause.clear()
            i += 1
            if self.stop.wait(max(0.0, min(self.o.probe_s, end - time.time()))) or time.time() >= end:
                break
        self.quiet()
        if not self.rank_exit:
            probes.append({"t": round(time.time() - self.t0, 1), **probe("end")})
        self.stop.set()
        for t in threads[: self.o.workers]:
            t.join(self.o.timeout + 30)
        drained, t1 = drain(self.url, self.o.drain_s)
        return self.record(probes, drained, round(time.time() - t1, 1))

    def record(self, probes: list[dict], drained: bool | None, drain_s: float) -> dict:
        rec, by = self.rec, {}
        for kind, _ in KINDS:
            xs = [r for r in rec if r["kind"] == kind]
            ok = sorted(r["total_s"] for r in xs if not r["error"] and not r["cancel"])
            by[kind] = {
                "n": len(xs),
                "errors": sum(bool(r["error"]) for r in xs),
                "cancelled": sum(r["cancel"] for r in xs),
                "check_fails": sum(bool(r["check"]) for r in xs),
                "total_p50": statistics.median(ok) if ok else None,
            }
        free = [s["free_gib"] for s in self.samples if s.get("free_gib") is not None]
        temps = [s["temp_c"] for s in self.samples if s.get("temp_c") is not None]
        limits = [s["power_limit_w"] for s in self.samples if s.get("power_limit_w") is not None]
        return {
            "minutes": self.o.minutes,
            "workers": self.o.workers,
            "traffic": "requests",
            "requests": len(rec),
            "errors": sum(bool(r["error"]) for r in rec),
            "cancelled": sum(r["cancel"] for r in rec),
            "check_fails": sum(bool(r["check"]) for r in rec),
            "by_kind": by,
            "error_samples": [r for r in rec if r["error"]][:10],
            "probes": probes,
            "samples": self.samples,
            "drained": drained,
            "drain_s": drain_s,
            "canary": None if self.rank_exit else canary(self.url, self.model),
            "memory_min_gib": round(min(free), 2) if free else None,
            "temp_max_c": max(temps) if temps else None,
            "power_limit_w": min(limits) if limits else None,
            "rank_exit": self.rank_exit,
            "engine_errors": [],
        }


def run(url, model, tools: Path, cold: dict[str, Path], out: Path, options: Options, meter=None, alive=None) -> dict:
    """The soak against a running server, probing with the bench clients in tools/ (and the largest cold prompt
    up to 32K token ids, as soak.sh did)."""
    sizes = sorted((int(s) for s in cold if int(s) <= 32768), reverse=True)[:1]
    cells = PROBE_CELLS + [f"cold.{s}" for s in sizes]

    run = bench.new_salt()

    def probe(i) -> dict:
        sample = bench.measure(url, model, tools, cells, cold, out / f"probe-{i}", f"{run}-probe-{i}")
        return {"hashes": sample.hashes, "metrics": sample.metrics, "problems": sample.problems}

    return Soak(url, model, options, meter, alive).run(probe)


def load(url: str) -> dict:
    """Requests running and waiting, from GET /v1/load; {} where the server has no such endpoint."""
    try:
        with urllib.request.urlopen(url + "/v1/load", timeout=10) as reply:
            doc = json.load(reply)
        return {"running": doc.get("running"), "waiting": doc.get("waiting")}
    except (OSError, ValueError):
        return {}


def drain(url: str, within: float) -> tuple[bool | None, float]:
    """Whether every slot comes free within the time given (None: the server does not say); and when it started."""
    t1 = time.time()
    while True:
        now = load(url)
        if not now:
            return None, t1
        if not now.get("running") and not now.get("waiting"):
            return True, t1
        if time.time() - t1 >= within:
            return False, t1
        time.sleep(min(2.0, max(0.05, within / 20)))


def canary(url: str, model: str) -> str:
    body = dict(OFF, model=model, messages=[{"role": "user", "content": CANARY}], max_tokens=64, temperature=0)
    request = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as reply:
            return (json.load(reply)["choices"][0]["message"].get("content") or "")[:40]
    except (OSError, ValueError, KeyError, IndexError) as err:
        return f"ERROR {type(err).__name__}"


def meter(unified: bool):
    """Free memory (host MemAvailable on unified memory, else the emptiest device's free memory) and the hottest
    GPU, read here on the node; None where this node does not say."""

    def read() -> dict:
        out: dict = {"free_gib": None, "temp_c": None, "power_limit_w": None}
        try:
            rows = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,temperature.gpu,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.splitlines()
        except (OSError, subprocess.TimeoutExpired):
            rows = []
        cells = [[_num(v) for v in row.split(",")] for row in rows]
        cells = [c for c in cells if len(c) == 4]
        temps = [c[2] for c in cells if c[2] is not None]
        limits = [c[3] for c in cells if c[3] is not None]
        out["temp_c"], out["power_limit_w"] = (max(temps) if temps else None), (min(limits) if limits else None)
        if unified:
            try:
                meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                out["free_gib"] = round(int(meminfo["MemAvailable"].split()[0]) / 1048576, 2)
            except (OSError, KeyError, ValueError):
                pass
        else:
            free = [(c[1] - c[0]) / 1024 for c in cells if None not in c[:2]]
            out["free_gib"] = round(min(free), 2) if free else None
        return out

    return read


def mock_meter(url: str):
    """The stand-in's GET /mock/memory, in place of the node's."""

    def read() -> dict:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/mock/memory", timeout=10) as reply:
                return {"free_gib": json.load(reply)["free_gib"], "temp_c": None}
        except (OSError, ValueError, KeyError):
            return {"free_gib": None, "temp_c": None}

    return read


def _num(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def verdict(record: dict, floors: Floors, expected: dict[str, str] | None = None) -> dict:
    """STABLE when nothing failed: no failed request, constant probe hashes (the expected ones where given), no
    rank exit or engine error, drained, the canary answered, free memory above the floor, below the hard limit."""
    problems, unmeasured = [], []
    if record.get("rank_exit"):
        problems.append("a rank exited or /health failed")
    if record["errors"]:
        problems.append(f"{record['errors']} failed requests")
    problems += [f"probe {p.get('t')} s: {q}" for p in record["probes"] for q in p.get("problems", [])]
    seen: dict[str, set[str]] = {}
    for p in record["probes"]:
        for cell, sha in p.get("hashes", {}).items():
            seen.setdefault(cell, set()).add(sha)
    moved = {c: sorted(v) for c, v in seen.items() if len(v) > 1}
    if moved:
        problems.append(f"probe hashes changed: {moved}")
    for cell, want in (expected or {}).items():
        got = next(iter(seen.get(cell, ())), None)
        if got and len(seen[cell]) == 1 and not (got.startswith(want) or want.startswith(got)):
            problems.append(f"{cell} hash {got[:12]}, the recipe's {want[:12]}")
    if record.get("drained") is False:
        problems.append(f"slots not drained in {record.get('drain_s')} s")
    if record.get("canary") is None:
        unmeasured.append("canary")
    elif CANARY_ANSWER not in record["canary"]:
        problems.append(f"canary answered {record['canary']!r}")
    for name, key in (("drain", "drained"), ("memory", "memory_min_gib")):
        if record.get(key) is None:
            unmeasured.append(name)
    low = record.get("memory_min_gib")
    if floors.memory_gib is not None and low is not None and low < floors.memory_gib:
        problems.append(f"free memory {low} GiB, under the floor of {floors.memory_gib} GiB")
    if floors.hard_c is not None and (held := hot_s(record, floors.hard_c)) is not None and held >= floors.hold_s:
        problems.append(f"{floors.hard_c} C or more for {held:.0f} s (the hard limit holds {floors.hold_s:.0f} s)")
    if record.get("engine_errors"):
        problems.append("engine error: " + record["engine_errors"][0][:200])
    notes = []
    capped = record.get("power_limit_w")
    if floors.power_w is not None and capped is not None and capped < floors.power_w:
        notes.append(f"a card is capped at {capped:.0f} W, under the profile's {floors.power_w:.0f} W")
    word = "UNSTABLE" if problems else "STABLE"
    hashes = {c: sorted(v) for c, v in sorted(seen.items())}
    summary = (
        f"{word}: {len(record['probes'])} probes, {record['requests']} {record.get('traffic', 'requests')}; probe "
        f"hashes {'CHANGED' if moved else 'constant'}; failed requests {record['errors']}"
    )
    return {
        "verdict": word,
        "summary": summary,
        "problems": problems,
        "unmeasured": unmeasured,
        "notes": notes,
        "hashes": hashes,
    }


def lines(verdict: dict) -> list[str]:
    """A verdict's problems, notes and what went unmeasured, a line each."""
    out = verdict["problems"] + [f"note: {n}" for n in verdict["notes"]]
    return out + ([f"not measured: {', '.join(verdict['unmeasured'])}"] if verdict["unmeasured"] else [])


def hot_s(record: dict, hard_c: float) -> float | None:
    """The longest stretch the samples stayed at or above hard_c; a record with no samples has only its maximum."""
    temps = [(s["t"], s["temp_c"]) for s in record.get("samples", []) if s.get("temp_c") is not None]
    if not temps:
        hot = record.get("temp_max_c")
        return 0.0 if hot is not None and hot >= hard_c else None
    longest, start = None, None
    for t, temp in temps:
        start = (t if start is None else start) if temp >= hard_c else None
        longest = max(longest or 0.0, t - start) if start is not None else longest
    return longest


def from_soak_sh(directory: Path) -> dict:
    """A record from soak.sh's log directory (probes.tsv, soak.tsv, errors.txt, soak.log), for verdict()."""
    names = {"fibonacci-raw": "decode.code.x1", "gpu-chat-no-think": "decode.chat.x1"}
    probes, temps = [], []
    for row in _rows(directory / "probes.tsv"):
        probe = {"t": row[0], "hashes": {}, "metrics": {}, "problems": [] if row[1] == "rc=0,0" else [row[1]]}
        for field in row[2:]:
            name, _, value = field.partition("=")
            if name == "smi":
                temps += [_num(g.split(",")[2]) for g in value.split(";") if g.count(",") == 3]
            elif ":" in value:
                tps, sha = value.rsplit(":", 1)
                cell = names.get(name) or f"cold.{int(name.removeprefix('cold').removesuffix('K')) * 1024}"
                probe["hashes"][cell], probe["metrics"][cell] = sha, _num(tps)
        probes.append(probe)
    rounds = _rows(directory / "soak.tsv")
    failed = [r for r in rounds if r[2] != "rc=0"]
    errors = (directory / "errors.txt").read_text().splitlines() if (directory / "errors.txt").is_file() else []
    log = (directory / "soak.log").read_text() if (directory / "soak.log").is_file() else ""
    temps = [t for t in temps if t is not None]
    return {
        "traffic": "load rounds",
        "requests": len(rounds),
        "errors": len(failed),
        "probes": probes,
        "drained": None,
        "canary": None,
        "memory_min_gib": None,
        "temp_max_c": max(temps) if temps else None,
        "rank_exit": "a rank exited" in log,
        "engine_errors": [e for e in errors if e.strip()],
    }


def _rows(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines() if line] if path.is_file() else []
