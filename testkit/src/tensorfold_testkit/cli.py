"""python -m tensorfold_testkit gate|selftest|mock: the served gates against a URL, or against the stand-in server."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import bench, receipt, soak
from .client import Client
from .gates import GateResult, conformance, hashes, quality, skipped, timed
from .mock.server import FAULTS, SOAK_FAULTS, Mock

ORDER = ("conformance", "hashes", "nondeterminism", "canary", "mmlu", "tools")
QUICK = ("hashes",)
SAMPLE_MMLU = [
    {
        "question": "Which planet is closest to the Sun?",
        "choices": ["Venus", "Mercury", "Mars", "Earth"],
        "answer": "B",
    },
    {"question": "What is 7 times 8?", "choices": ["54", "56", "64", "48"], "answer": "B"},
]
SAMPLE_CHAINS = [
    {
        "name": "weather",
        "messages": [{"role": "user", "content": "What is the weather in Oslo?"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {}}}},
            }
        ],
        "steps": [{"call": "get_weather", "result": "4 C and rain"}],
        "answer": "rain",
    }
]


@dataclass
class Inputs:
    table: dict | None
    profile: str | None
    conformance: Path | None
    tree: Path | None
    mmlu: list | None
    mmlu_min: float
    chains: list | None
    repeats: int
    model: str | None = None
    quick: bool = False
    plants: bool = True


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, OSError) as err:
        print(f"testkit {args.command}: {err}", file=sys.stderr)
        return 2


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="tensorfold_testkit", description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate", help="Run the gates against a server URL (or --mock).")
    gate.set_defaults(run=run_gate)
    _inputs(gate)
    gate.add_argument("--url", help="a running server, its prompt cache on")
    gate.add_argument("--fresh-url", help="the same server started with --prompt-cache-gib 0 (resumed, evict)")
    gate.add_argument("--model", help="the served model name")
    gate.add_argument("--quick", action="store_true", help="greedy hashes only: no plants, sampled pass or conformance")
    gate.add_argument("--skip", default="", help=f"comma-separated gates to skip: {', '.join(ORDER)}")
    gate.add_argument("--only", default="", help="comma-separated gates to run, and nothing else")
    gate.add_argument("--mock", action="store_true", help="run against the stand-in server replaying the table")
    gate.add_argument("--fault", choices=FAULTS, help="with --mock: plant this fault in the stand-in")
    gate.add_argument("--output", type=Path, help="write the results as JSON")
    gate.add_argument("--receipt", type=Path, help="write a receipt of the run")
    gate.add_argument("--head", default="{}", help="the receipt's head as JSON: kit, dev, recipe, engine (the kit's)")
    test = commands.add_parser("selftest", help="Every gate on the stand-in: green clean, red with its own fault.")
    test.set_defaults(run=run_selftest)
    _inputs(test)
    mock = commands.add_parser("mock", help="Serve the stand-in until interrupted.")
    mock.set_defaults(run=run_mock)
    _inputs(mock)
    mock.add_argument("--fault", choices=FAULTS + SOAK_FAULTS)
    mock.add_argument("--port", type=int, help="serve one stand-in on this port (a bench arm) instead of a pair")
    mock.add_argument("--name", default="mock", help="the served model name")
    mock.add_argument("--tps", type=float, help="pace streams at this many tokens a second")
    mock.add_argument("--prefill-tps", type=float, help="add a prompt's tokens over this rate before the reply")
    mock.add_argument("--no-cache", action="store_true", help="keep no prompts (--prompt-cache-gib 0)")
    replay = commands.add_parser("decide", help="Replay a recorded ABBA window's cells to its verdict.")
    replay.set_defaults(run=run_decide)
    replay.add_argument("cells_dir", type=Path, help="one directory a cell, named ARM-N, N the run order")
    replay.add_argument("a")
    replay.add_argument("b")
    replay.add_argument("--control", help="the control arm, first and last in the window")
    replay.add_argument("--bar", type=float, default=0.03, help="the share a cell must move to count (default 0.03)")
    replay.add_argument("--cells", default="decode.code.x1,decode.chat.x1,decode.code.x16,decode.chat.x16")
    replay.add_argument("--hashes", type=Path, help="the recipe's hashes table, for the expected tokens")
    replay.add_argument("--profile")
    soak_ = commands.add_parser("soak", help="Soak a running server (or --mock), or --replay a soak.sh log.")
    soak_.set_defaults(run=run_soak)
    _inputs(soak_)
    soak_.add_argument("--url", help="a running server")
    soak_.add_argument("--model", default="mock", help="the served model name")
    soak_.add_argument("--minutes", type=float, default=30)
    soak_.add_argument("--workers", type=int, default=4)
    soak_.add_argument("--probe-min", type=float, default=5, help="minutes between the fixed probes")
    soak_.add_argument("--floor-gib", type=float, help="free memory the run must keep")
    soak_.add_argument("--hard-c", type=float, help="the temperature that fails the run")
    soak_.add_argument("--unified", action="store_true", help="host and device share memory: read MemAvailable")
    soak_.add_argument("--mock", action="store_true", help="soak the stand-in")
    soak_.add_argument("--fault", choices=SOAK_FAULTS, help="with --mock: plant this fault")
    soak_.add_argument("--replay", type=Path, metavar="DIR", help="the verdict of soak.sh's log directory")
    check = commands.add_parser("receipt", help="Verify receipts: schema, fields and the public-identifier rule.")
    check.set_defaults(run=run_receipt)
    check.add_argument("files", type=Path, nargs="+")
    return top


def _inputs(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--hashes", type=Path, help="the recipe's gates/hashes.json")
    sub.add_argument("--profile", help="the recipe profile the server runs, for its row in the hashes table")
    sub.add_argument("--conformance", type=Path, help="a conformance profile in upstream's format")
    sub.add_argument("--tree", type=Path, help="the pinned engine tree, for its tools/conformance.py")
    sub.add_argument("--repeats", type=int, default=8, help="copies a mode in the nondeterminism gate (default 8)")
    sub.add_argument("--mmlu", type=Path, help="MMLU rows as JSONL (question, choices, answer)")
    sub.add_argument("--mmlu-min", type=float, default=0.7, help="the share the mmlu gate must answer right")
    sub.add_argument("--tool-chains", type=Path, help="tool chains as JSON")


def inputs(args, samples: bool = False) -> Inputs:
    """The gates' data from the flags; samples fills MMLU and tool chains for the stand-in when none are given."""
    mmlu = quality.load_mmlu(args.mmlu) if args.mmlu else SAMPLE_MMLU if samples else None
    chains = quality.load_chains(args.tool_chains) if args.tool_chains else SAMPLE_CHAINS if samples else None
    table = hashes.load(args.hashes) if args.hashes else None
    model, quick = getattr(args, "model", None), getattr(args, "quick", False)
    return Inputs(
        table, args.profile, args.conformance, args.tree, mmlu, args.mmlu_min, chains, args.repeats, model, quick
    )


def gates(given: Inputs, url: str, fresh_url: str | None, names, echo: bool = True) -> list[GateResult]:
    """Each named gate in ORDER against url; a gate whose inputs are missing is skipped and says why."""
    client, out = Client(url, given.model), []
    for name in [n for n in ORDER if n in names]:
        out.append(_gate(name, given, client, fresh_url))
        if echo:
            print(out[-1], flush=True)
    return out


def _gate(name: str, given: Inputs, client: Client, fresh_url: str | None) -> GateResult:
    if name == "conformance":
        if not (given.tree and given.conformance):
            return skipped(name, "needs --tree and --conformance")
        return timed(conformance.conformance, given.tree, given.conformance, client.url, fresh_url, plants=given.plants)
    if name in ("hashes", "nondeterminism") and given.table is None:
        return skipped(name, "needs --hashes")
    if name == "hashes":
        return timed(hash_gate, client, given.table, given.profile, given.quick)
    if name == "nondeterminism":
        return timed(hashes.nondeterminism, client, given.table, given.repeats)
    if name == "canary":
        return timed(quality.canary, client)
    if name == "mmlu":
        return timed(quality.mmlu, client, given.mmlu, given.mmlu_min) if given.mmlu else skipped(name, "needs --mmlu")
    return timed(quality.tools, client, given.chains) if given.chains else skipped(name, "needs --tool-chains")


def hash_gate(client, table, profile, quick) -> GateResult:
    """The hashes gate; a full run also moves every seed and must then go red, or the seed never reached sampling."""
    result = hashes.hashes(client, table, profile, quick)
    if quick or not result.passed:
        return result
    if hashes.hashes(client, table, profile, plant=True).passed:
        return GateResult("hashes", False, result.line + "; with every seed moved it stayed green", rows=result.rows)
    result.line += "; red with every seed moved"
    return result


def chosen(args) -> list[str]:
    names = [n for n in (args.only.split(",") if args.only else (QUICK if args.quick else ORDER)) if n]
    skip = [n for n in args.skip.split(",") if n]
    if unknown := [n for n in names + skip if n not in ORDER]:
        raise ValueError(f"unknown gates {unknown}: choose from {', '.join(ORDER)}")
    return [n for n in names if n not in skip]


def recorded(table: dict | None, profile: str | None) -> list[dict]:
    """The table's expected hashes as the replies the stand-in replays."""
    out = []
    for name, row in hashes.expected(table, profile).items() if table else ():
        prompt, tokens = table["prompts"][name]["prompt"], table["tokens"]
        out.append({"prompt": prompt, "temperature": 0, "max_tokens": tokens, "token_sha": row["greedy"]})
        temperature = table["sampled"]["temperature"]
        out += [
            {"prompt": prompt, "temperature": temperature, "seed": seed, "max_tokens": tokens, "token_sha": sha}
            for seed, sha in zip(table["seeds"], row.get("sampled", []), strict=False)
        ]
    for row in (table or {}).get("cold", {}).values():
        ids = json.dumps(json.loads(row["path"].read_text()))
        out.append({"prompt": ids, "temperature": 0, "max_tokens": row["tokens"], "token_sha": row["hash"]})
    return out


@contextlib.contextmanager
def stand_in(given: Inputs, fault=None, replies=None):
    """The stand-in with its prompt cache, and a second one without, for conformance's resumed and evict."""
    replies = recorded(given.table, given.profile) if replies is None else replies
    data = {"recorded": replies, "mmlu": given.mmlu or (), "chains": given.chains or (), "fault": fault}
    with Mock(**data) as warm, Mock(**data, cache_words=0) as fresh:
        yield warm, fresh


def run_gate(args) -> int:
    names, given, start = chosen(args), inputs(args, samples=args.mock), time.perf_counter()
    if args.mock:
        with stand_in(given, args.fault) as (warm, fresh):
            results = gates(given, warm.url, fresh.url, names)
    elif not args.url:
        raise ValueError("pass --url, or --mock")
    else:
        results = gates(given, args.url, args.fresh_url, names)
    failed = [r.gate for r in results if not r.passed]
    ran = sum(not r.skipped for r in results)
    print(
        f"gate: {ran - len(failed)} passed, {len(failed)} failed, {len(results) - ran} skipped "
        f"in {time.perf_counter() - start:.2f} s"
    )
    if args.output:
        args.output.write_text(json.dumps([r.__dict__ for r in results], indent=1) + "\n")
    if args.receipt:
        rows = [{k: v for k, v in r.__dict__.items() if k != "rows"} for r in results]
        receipt.write(args.receipt, "gate", json.loads(args.head), {"mock": args.mock, "results": rows})
    return 1 if failed else 0


def selftest(given: Inputs, replies=None):
    """Each gate on its own clean stand-in, again on one planted with that gate's fault, and each of conformance.py's
    own plants on its own stand-in. All at once, so the run takes as long as its longest part."""
    skip = (json.loads(given.conformance.read_text()).get("skip") or {}) if given.conformance else {}
    upstream = [g for g in conformance.GATES if g not in skip] if given.tree and given.conformance else []

    def run(job):
        kind, name = job
        alone = Inputs(**{**given.__dict__, "plants": False})
        with stand_in(alone, name if kind == "fault" else None, replies) as (warm, fresh):
            if kind == "plant":
                return conformance.run(given.tree, given.conformance, warm.url, fresh.url, [name], name)[0]
            return gates(alone, warm.url, fresh.url, [name], echo=False)[0]

    jobs = [("clean", g) for g in ORDER] + [("fault", g) for g in ORDER] + [("plant", g) for g in upstream]
    with ThreadPoolExecutor(len(jobs)) as pool:
        done = list(pool.map(run, jobs))
    n = len(ORDER)
    return done[:n], {r.gate: r for r in done[n : 2 * n]}, dict(zip(upstream, done[2 * n :], strict=True))


def run_selftest(args, replies=None) -> int:
    start = time.perf_counter()
    clean, planted, plants = selftest(inputs(args, samples=True), replies)
    problems = [f"conformance --plant {g} stayed green" for g, code in plants.items() if code == 0]
    for result in clean:
        print(result)
        if result.skipped:
            continue
        print(f"  planted {result.gate}: {planted[result.gate]}")
        if not result.passed:
            problems.append(f"{result.gate} is red on the clean stand-in")
        elif planted[result.gate].passed:
            problems.append(f"{result.gate} stayed green with its fault planted")
    if plants:
        print(f"  conformance.py --plant: {sum(code != 0 for code in plants.values())}/{len(plants)} red")
    for problem in problems:
        print("FAIL selftest: " + problem)
    ran = sum(not r.skipped for r in clean)
    print(
        f"selftest: {ran} gates and {len(plants)} conformance plants, {len(problems)} problems, "
        f"{len(clean) - ran} skipped, in {time.perf_counter() - start:.2f} s"
    )
    return 1 if problems else 0


def run_decide(args) -> int:
    cells = bench.parse_cells([c for c in args.cells.split(",") if c])
    dirs = sorted((d for d in args.cells_dir.iterdir() if d.is_dir()), key=lambda d: int(d.name.rsplit("-", 1)[1]))
    samples = [bench.read(d, cells, d.name.rsplit("-", 1)[0]) for d in dirs]
    expected = hashes.bench_expected(hashes.load(args.hashes), args.profile) if args.hashes else {}
    verdict = bench.decide(samples, args.a, args.b, args.bar, args.control, expected)
    print(bench.table(verdict))
    return 0


def run_soak(args) -> int:
    table = hashes.load(args.hashes) if args.hashes else None
    expected = hashes.bench_expected(table, args.profile) if table else {}
    floors = soak.Floors(args.floor_gib, args.hard_c)
    if args.replay:
        record = soak.from_soak_sh(args.replay)
    else:
        if not args.tree or not table:
            raise ValueError("pass --tree (its tools/ run the probes) and --hashes, or --replay DIR")
        cold = {size: row["path"] for size, row in table.get("cold", {}).items()}
        options = soak.Options(args.minutes, args.workers, args.probe_min * 60)
        out = Path(f"soak-{time.strftime('%m%d-%H%M%S')}")
        if args.mock:
            with Mock(
                args.model, recorded(table, args.profile), fault=args.fault, cache_words=0, tps=2000, prefill_tps=1e5
            ) as m:
                record = soak.run(m.url, args.model, args.tree / "tools", cold, out, options, soak.mock_meter(m.url))
        elif args.url:
            record = soak.run(args.url, args.model, args.tree / "tools", cold, out, options, soak.meter(args.unified))
        else:
            raise ValueError("pass --url, --mock or --replay")
    verdict = soak.verdict(record, floors, expected)
    print(verdict["summary"])
    for line in soak.lines(verdict):
        print("  " + line)
    return 0 if verdict["verdict"] == "STABLE" else 1


def run_receipt(args) -> int:
    for path in args.files:
        doc = receipt.verify(path)
        print(f"{path.name}: {doc['kind']} receipt, testkit {doc['testkit']}, kit {doc['kit']}: valid")
    return 0


def run_mock(args) -> int:
    given = inputs(args, samples=True)
    if args.port:
        data = {"recorded": recorded(given.table, given.profile), "mmlu": given.mmlu, "chains": given.chains}
        cache = {"cache_words": 0} if args.no_cache else {}
        mock = Mock(
            name=args.name,
            fault=args.fault,
            port=args.port,
            tps=args.tps,
            prefill_tps=args.prefill_tps,
            **data,
            **cache,
        )
        with mock, contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(3600)
        return 0
    with stand_in(given, args.fault) as (warm, fresh):
        print(f"stand-in at {warm.url}, cache-off one at {fresh.url}; interrupt to stop", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(3600)
    return 0
