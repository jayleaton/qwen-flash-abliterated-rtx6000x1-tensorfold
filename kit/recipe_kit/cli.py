"""The recipe command: prepare an engine, track upstream, render env files and scrub before publishing."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from . import (
    __version__,
    bench,
    build,
    bundle,
    compat,
    engine,
    envfile,
    hardware,
    manifest,
    profiles,
    publish,
    record,
    revision,
    scaffold,
    scrub,
    serve,
    soak,
)
from .check import check as run_rules

ERRORS = (manifest.ManifestError, engine.EngineError, build.BuildError, envfile.EnvFileError, ValueError, OSError)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except ERRORS as err:
        print(f"recipe {args.command}: {err}", file=sys.stderr)
        return 2


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="recipe", description=__doc__)
    top.add_argument("--version", action="version", version=f"recipe-kit {__version__}")
    commands = top.add_subparsers(dest="command", required=True)

    def command(name: str, run, help_text: str, engine_args: bool = True) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(run=run)
        if engine_args:
            sub.add_argument("--recipe", type=Path, default=Path("."), help="the recipe directory (default: .)")
            sub.add_argument("--engine", help="the [engine.NAME] to use when the recipe has several")
            sub.add_argument("--source", help="a local TensorFold checkout or bundle that holds the pin")
            sub.add_argument("--kit-dev", action="store_true", help="run although the recipe pins another kit revision")
            sub.add_argument("dest", type=Path, help="the engine directory; one fetched from this pin is reused")
        return sub

    new = commands.add_parser("new", help="Write a new recipe directory from the kit's template.")
    new.set_defaults(run=new_recipe)
    new.add_argument("dir", type=Path, help="the new recipe's directory; its name is the recipe's")
    new.add_argument("--hardware", default="rtx6000x2", help="the hardware/NAME.toml profile (default: rtx6000x2)")
    new.add_argument(
        "--engines", default="cuda", help="comma-separated [engine.NAME]s, the first built (default: cuda)"
    )

    command("fetch", fetch, "Fetch the pin, apply the series and check the tree hash.")
    prep = command("prepare", prepare, "Fetch, apply and check the tree as fetch does, then build the engine.")
    prep.add_argument("--no-addons", action="store_true", help="build the engine only, not its addons")
    update = command("update-engine", update_engine, "Move the pin to a new upstream commit and record its tree.")
    update.add_argument("--rev", help="the commit to move to (default: the head of --branch)")
    update.add_argument("--branch", default="main", help="the upstream branch to follow (default: main)")
    update.add_argument("--build", action="store_true", help="also build the new engine")

    rules = commands.add_parser("check", help="Check modes, switches, profiles and arms: rules C1 to C9.")
    rules.set_defaults(run=check_recipe)
    _recipe_args(rules)
    rules.add_argument("dir", type=Path, nargs="?", help="the recipe directory, as --recipe")
    rules.add_argument("--tree", type=Path, help="a fetched engine tree: also check the names it reads (C8)")
    rules.add_argument("--engine", help="the [engine.NAME] the tree is, when the recipe has several")

    launch = commands.add_parser("serve", help="Check, render and launch a profile or an arm on this host.")
    launch.set_defaults(run=serve_target)
    _recipe_args(launch)
    launch.add_argument("target", help="a [profile.NAME] or an [arm.NAME]")
    launch.add_argument("engine_dir", type=Path, help="the prepared engine directory (or a release bundle)")
    launch.add_argument("model", type=Path, help="the checkpoint directory")
    launch.add_argument("extra", nargs="*", help="more serve flags, after --")
    launch.add_argument("--tree", type=Path, help="the engine tree, when engine_dir is a release bundle")
    launch.add_argument("--gpus", default=None, help="the CUDA devices of rank 0, 1, ... (default 0,1,...)")
    launch.add_argument("--port", type=int, help="rank 0's port; rank N takes port + N (default [recipe] port)")
    launch.add_argument("--numa", help="run every rank under numactl on this NUMA node")
    launch.add_argument("--log-dir", type=Path, help="where ranks 1 and up log (default: the engine directory)")
    launch.add_argument("--no-addons", action="store_true", help="start although addon outputs are missing")
    launch.add_argument("--dry-run", action="store_true", help="print each rank's argv and added env as JSON")

    gate = commands.add_parser("gate", help="Run the recipe's gates against a running server (or --mock).")
    gate.set_defaults(run=gate_server)
    _recipe_args(gate)
    gate.add_argument("target", nargs="?", help="the profile or arm the server runs (default: the first shipped)")
    gate.add_argument("--url", help="the server, its prompt cache on (default: http://127.0.0.1:[recipe] port)")
    gate.add_argument("--fresh-url", help="the same server started with --prompt-cache-gib 0 (resumed, evict)")
    gate.add_argument("--tree", type=Path, help="the pinned engine tree, for its tools/conformance.py")
    gate.add_argument("--quick", action="store_true", help="greedy hashes only: no plants, sampled pass or conformance")
    gate.add_argument("--skip", default="", help="comma-separated gates to skip")
    gate.add_argument("--only", default="", help="comma-separated gates to run, and nothing else")
    gate.add_argument("--mmlu", type=Path, help="MMLU rows as JSONL, for the mmlu gate")
    gate.add_argument("--mock", action="store_true", help="run against the stand-in server replaying the hashes")
    gate.add_argument("--fault", help="with --mock: the fault to plant in the stand-in")
    gate.add_argument("--output", type=Path, help="write the results as JSON")
    gate.add_argument("--receipt", type=Path, help="write the run's receipt")
    gate.add_argument("--engine", help="the [engine.NAME] the server runs, for the receipt, when there are several")

    ab = commands.add_parser("bench", help="ABBA two profiles or arms on this host, inside a control arm.")
    ab.set_defaults(run=bench_arms)
    _recipe_args(ab)
    ab.add_argument("a", help="the base profile or arm")
    ab.add_argument("b", help="the profile or arm measured against it")
    ab.add_argument("engine_dir", type=Path, help="the prepared engine directory (or a bundle's engine)")
    ab.add_argument("model", type=Path, nargs="?", help="the checkpoint directory (not with --mock)")
    ab.add_argument("--tree", type=Path, help="the engine tree, for its bench clients (default: engine_dir)")
    ab.add_argument("--quick", action="store_true", help="one A B pass on the first two cells: not a verdict")
    ab.add_argument("--cells", help="comma-separated cells instead of [bench] cells")
    ab.add_argument("--rounds", type=int, help="samples an arm, even (default [bench] rounds, 2)")
    ab.add_argument("--bar", type=float, help="the share a cell must move to count (default [bench] bar, 0.03)")
    ab.add_argument("--control", help="the control arm (default: the first shipped profile)")
    ab.add_argument("--no-control", action="store_true", help="no control arm")
    ab.add_argument("--keep-cache", action="store_true", help="do not drop the page cache before each boot")
    ab.add_argument("--gpus", help="the CUDA devices of rank 0, 1, ... (default 0,1,...)")
    ab.add_argument("--port", type=int, help="rank 0's port (default [recipe] port)")
    ab.add_argument("--numa", help="run every rank under numactl on this NUMA node")
    ab.add_argument("--out", type=Path, help="where the cells go (default: bench-<a>-<b>-<time> here)")
    ab.add_argument("--receipt", type=Path, help="the receipt (default: receipt.json in --out)")
    ab.add_argument("--mock", action="store_true", help="every arm on the stand-in server: the whole flow, no GPU")
    ab.add_argument("--mock-gain", type=float, default=1.0, help="with --mock: pace B this many times faster")

    so = commands.add_parser("soak", help="Soak a profile or arm on this host at the hardware profile's floors.")
    so.set_defaults(run=soak_target)
    _recipe_args(so)
    so.add_argument("target", help="the profile or arm")
    so.add_argument("engine_dir", type=Path, help="the prepared engine directory (or a bundle's engine)")
    so.add_argument("model", type=Path, nargs="?", help="the checkpoint directory (not with --mock)")
    so.add_argument("--tree", type=Path, help="the engine tree, for its bench clients (default: engine_dir)")
    so.add_argument("--minutes", type=float, help="how long (default [soak] minutes, 30)")
    so.add_argument("--workers", type=int, help="client threads, 1 to this many at once (default [soak] workers, 4)")
    so.add_argument("--probe-min", type=float, help="minutes between the fixed probes (default [soak] probe_min, 5)")
    so.add_argument("--keep-cache", action="store_true", help="do not drop the page cache before the boot")
    so.add_argument("--gpus", help="the CUDA devices of rank 0, 1, ... (default 0,1,...)")
    so.add_argument("--port", type=int, help="rank 0's port (default [recipe] port)")
    so.add_argument("--numa", help="run every rank under numactl on this NUMA node")
    so.add_argument("--out", type=Path, help="where the probes and logs go (default: soak-<target>-<time> here)")
    so.add_argument("--receipt", type=Path, help="the receipt (default: receipt.json in --out)")
    so.add_argument("--mock", action="store_true", help="soak the stand-in server: the whole flow, no GPU")
    so.add_argument("--mock-fault", help="with --mock: plant a soak fault (error, leak, drift, memory)")

    kit = commands.add_parser("kit", help="Bundle the recipe, its engine and this kit for an offline host.")
    kit.set_defaults(run=kit_bundle)
    _recipe_args(kit)
    kit.add_argument("dest", type=Path, nargs="?", help="where the bundle directory and its .tar.gz go")
    kit.add_argument("engine_dir", type=Path, nargs="?", help="the prepared engine directory")
    kit.add_argument("--tree", type=Path, help="the engine tree (default: engine_dir)")
    kit.add_argument("--engine", help="the [engine.NAME] to bundle when the recipe has several")
    kit.add_argument("--verify", type=Path, metavar="BUNDLE", help="check a bundle against its SHA256SUMS instead")

    env = command("env", env_exports, "Print export lines for env files; a variable already set wins.", False)
    env.add_argument("files", type=Path, nargs="+", help="KEY=VALUE files, later ones replacing earlier keys")

    out = commands.add_parser("publish", help="Write the public recipe tree, with the kit release, from this one.")
    out.set_defaults(run=publish_recipe)
    _recipe_args(out)
    out.add_argument("dest", type=Path, help="a new directory for the public tree")
    out.add_argument("--rev", default="HEAD", help="the recipe commit to publish (default: HEAD)")

    check = command("publish-check", publish_check, "Scan a repo for private details and credentials.", False)
    check.add_argument("root", type=Path, nargs="?", default=Path("."), help="the repo to scan (default: .)")
    check.add_argument("--history", action="store_true", help="also scan every blob, message and identity")
    check.add_argument("--range", dest="revs", help="limit --history to these commits, such as origin/main..HEAD")
    return top


def _recipe_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--recipe", type=Path, default=Path("."), help="the recipe directory (default: .)")
    sub.add_argument("--kit-dev", action="store_true", help="run although the recipe pins another kit revision")
    sub.add_argument("--today", type=datetime.date.fromisoformat, default=datetime.date.today(), help="YYYY-MM-DD")


def _recipe(args: argparse.Namespace) -> manifest.Recipe:
    recipe = manifest.load(args.recipe.resolve())
    args.kit = revision.require(recipe, args.kit_dev)
    print(f"{recipe.name}: kit {args.kit}", file=sys.stderr)
    return recipe


def _load(args: argparse.Namespace) -> tuple[manifest.Recipe, manifest.Engine, Path]:
    recipe = _recipe(args)
    return recipe, recipe.engine(args.engine), args.dest.resolve()


def _target(recipe: manifest.Recipe, plan: profiles.Plan, target: str) -> tuple[manifest.Engine, hardware.Hardware]:
    """The engine a profile or an arm runs on, and the hardware profile that engine runs on."""
    spec = recipe.engine(profiles.engine_of(plan, target))
    return spec, hardware.load(recipe.hardware_of(spec))


def _findings(recipe, plan, today, tree: Path | None, name: str | None = None) -> list:
    """C1 to C9. C8 runs against TREE when given (engine NAME, or the engine it is the pinned tree of), else against
    each engine whose names recipe fetch recorded, for the names set on that engine's profiles and arms."""
    owned = {n for spec in recipe.engines.values() for n in hardware.load(recipe.hardware_of(spec)).env}
    found = run_rules(plan, owned, today)
    for spec, names in _engine_names(recipe, tree, name):
        on = None if len(recipe.engines) == 1 else spec.name
        found += [f for f in run_rules(plan, owned, today, names, engine=on) if f.rule == "C8"]
    return found


def _engine_names(recipe, tree: Path | None, name: str | None) -> list[tuple[manifest.Engine, dict]]:
    if tree is None:
        found = [(spec, record.load(recipe, spec).get("names")) for spec in recipe.engines.values()]
        return [(spec, names) for spec, names in found if isinstance(names, dict)]
    if name is None and len(recipe.engines) > 1:
        try:
            pinned = engine.git("-C", str(tree), "rev-parse", "HEAD^{tree}")
        except engine.EngineError:
            pinned = None
        name = next((spec.name for spec in recipe.engines.values() if spec.tree == pinned), None)
        if name is None:
            raise manifest.ManifestError(f"{tree} is no engine's pinned tree: name it with --engine")
    return [(recipe.engine(name), compat.engine_names(tree))]


def new_recipe(args: argparse.Namespace) -> int:
    todo = scaffold.create(args.dir.resolve(), args.hardware, args.engines.split(","))
    print(f"recipe new: {args.dir} from {scaffold.TEMPLATE}; still to set in recipe.toml:")
    for item in todo:
        print(f"  {item}")
    print(f"then: recipe update-engine --recipe {args.dir} ENGINE_DIR, recipe prepare, recipe check {args.dir}")
    return 0


def check_recipe(args: argparse.Namespace) -> int:
    args.recipe = args.dir or args.recipe
    recipe = _recipe(args)
    plan = profiles.load(recipe)
    tree = args.tree.resolve() if args.tree else None
    findings = _findings(recipe, plan, args.today, tree, args.engine)
    for finding in findings:
        print(finding)
    targets = len(plan.profiles) + len(plan.arms)
    if tree:
        names = ""
    else:
        missing = [spec.name for spec in recipe.engines.values() if "names" not in record.load(recipe, spec)]
        names = f", C8 not run for {', '.join(missing)} (run recipe fetch)" if missing else ""
    print(f"recipe check: {len(findings)} finding(s) over {targets} profiles and arms{names}")
    return 1 if findings else 0


def serve_target(args: argparse.Namespace) -> int:
    recipe = _recipe(args)
    plan = profiles.load(recipe)
    spec, hw = _target(recipe, plan, args.target)
    engine_dir = args.engine_dir.resolve()
    tree = (args.tree or engine_dir).resolve()
    if not (tree / "zig" / "src").is_dir() and not (tree / compat.NAMES).is_file():
        raise manifest.ManifestError(f"{tree} is not an engine tree; pass --tree with the fetched tree")
    findings = _findings(recipe, plan, args.today, tree, spec.name)
    if findings:
        raise manifest.ManifestError("recipe check fails, so nothing starts:\n  " + "\n  ".join(map(str, findings)))
    rendered = profiles.render(plan, args.target, hw.env)
    if spec.build is None or not spec.build.outputs:
        raise manifest.ManifestError(f"[engine.{spec.name}.build] outputs must name the server binary first")
    wanted = list(spec.build.outputs) + ([] if args.no_addons else [o for a in spec.build.addons for o in a.outputs])
    missing = [name for name in wanted if not (engine_dir / name).is_file()]
    if missing:
        raise manifest.ManifestError(f"{engine_dir} has no {', '.join(missing)}: run recipe prepare, or --no-addons")
    binary = engine_dir / spec.build.outputs[0]
    problems = compat.check(record.capabilities(recipe, spec, binary, compat.capabilities), recipe, spec, hw, rendered)
    if problems:
        raise manifest.ManifestError("the binary cannot serve this:\n  " + "\n  ".join(problems))
    if not (args.model / "config.json").is_file():
        raise manifest.ManifestError(f"{args.model} has no config.json")
    gpus = args.gpus.split(",") if args.gpus else None
    launches = serve.ranks(
        recipe,
        rendered,
        binary,
        args.model,
        args.extra,
        gpus,
        args.port or recipe.port,
        os.environ,
        args.numa,
        args.log_dir or engine_dir,
        None if args.dry_run else serve.node.resolve,
    )
    if args.dry_run:
        print(json.dumps([{"rank": r.rank, "argv": r.argv, "env": r.env} for r in launches], indent=1))
        return 0
    if rendered.experiments:
        print(f"{args.target} is an arm: experiments {', '.join(rendered.experiments)}", file=sys.stderr)
    return serve.run(launches)


def gate_server(args: argparse.Namespace) -> int:
    from tensorfold_testkit import cli as testkit

    recipe = _recipe(args)
    plan, gates = profiles.load(recipe), recipe.data.get("gates", {})
    target = args.target or recipe.ships[0]
    if target not in plan.profiles and target not in plan.arms:
        raise manifest.ManifestError(f"{target} is neither a [profile.*] nor an [arm.*]")
    argv = ["gate", "--profile", target, "--model", recipe.served_name, "--skip", args.skip, "--only", args.only]
    for key, flag in (("conformance", "--conformance"), ("hashes", "--hashes"), ("tool_chains", "--tool-chains")):
        if key in gates:
            argv += [flag, str(recipe.root / gates[key])]
    if "mmlu200_min" in gates:
        argv += ["--mmlu-min", str(gates["mmlu200_min"])]
    pairs = (
        ("--url", args.url or (None if args.mock else f"http://127.0.0.1:{recipe.port}")),
        ("--fresh-url", args.fresh_url),
        ("--tree", args.tree),
        ("--mmlu", args.mmlu),
        ("--fault", args.fault),
        ("--output", args.output),
    )
    argv += [str(v) for flag, value in pairs if value for v in (flag, value)]
    argv += [flag for flag, on in (("--quick", args.quick), ("--mock", args.mock)) if on]
    if args.receipt:
        head = bench.head(recipe, args.kit, recipe.engine(args.engine))
        argv += ["--receipt", str(args.receipt), "--head", json.dumps({**head, "target": target})]
    return testkit.main(argv)


def bench_arms(args: argparse.Namespace) -> int:
    from tensorfold_testkit import bench as abba
    from tensorfold_testkit import receipt

    recipe = _recipe(args)
    plan, table = profiles.load(recipe), recipe.data.get("bench", {})
    spec, hw = _target(recipe, plan, args.a)
    if not args.mock and _findings(recipe, plan, args.today, (args.tree or args.engine_dir).resolve(), spec.name):
        raise manifest.ManifestError("recipe check fails, so nothing starts: run recipe check --tree")
    cells = args.cells.split(",") if args.cells else list(table.get("cells", ["decode.code.x1", "decode.chat.x1"]))
    control = None if args.no_control or args.quick else args.control or recipe.ships[0]
    options = {
        "mock": args.mock,
        "gain": args.mock_gain,
        "port": args.port or recipe.port,
        "gpus": args.gpus.split(",") if args.gpus else None,
        "numa": args.numa,
        "cells": cells[:2] if args.quick else cells,
        "rounds": 1 if args.quick else args.rounds or table.get("rounds", 2),
        "bar": args.bar if args.bar is not None else table.get("bar", 0.03),
        "drop_cache": not (args.keep_cache or args.quick or args.mock),
    }
    for target in filter(None, (args.b, control)):
        if _target(recipe, plan, target)[1].name != hw.name:
            raise manifest.ManifestError(f"{target} runs on other hardware than {args.a}: an ABBA runs on one host")
    stamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    out = (args.out or Path(f"bench-{args.a}-{args.b}-{stamp}")).resolve()
    engine_dir = args.engine_dir.resolve()
    # the control runs under its own label, so a control that is A or B is not pooled with that arm's samples
    labels = {args.a: args.a, args.b: args.b}
    if control:
        labels[control if control not in labels else f"{control}-control"] = control
    result = bench.run(
        recipe,
        plan,
        hw,
        labels,
        (args.a, args.b, next(reversed(labels)) if control else None),
        engine_dir,
        (args.tree or args.engine_dir).resolve(),
        args.model.resolve() if args.model else None,
        out,
        options,
    )
    print(abba.table(result["verdict"]))
    head = bench.head(recipe, args.kit, spec)
    path = receipt.write(args.receipt or out / "receipt.json", "bench", head, {"mock": args.mock, **result})
    print(f"receipt: {path}", file=sys.stderr)
    return 1 if result["verdict"]["verdict"] in ("tokens differ", "unmeasured", "void") else 0


def soak_target(args: argparse.Namespace) -> int:
    from tensorfold_testkit import receipt
    from tensorfold_testkit import soak as testkit_soak

    recipe = _recipe(args)
    plan, table = profiles.load(recipe), recipe.data.get("soak", {})
    spec, hw = _target(recipe, plan, args.target)
    tree = (args.tree or args.engine_dir).resolve()
    if not args.mock and _findings(recipe, plan, args.today, tree, spec.name):
        raise manifest.ManifestError("recipe check fails, so nothing starts: run recipe check --tree")
    options = {
        "mock": args.mock,
        "fault": args.mock_fault,
        "port": args.port or recipe.port,
        "gpus": args.gpus.split(",") if args.gpus else None,
        "numa": args.numa,
        "minutes": args.minutes if args.minutes is not None else table.get("minutes", 30),
        "workers": args.workers or table.get("workers", 4),
        "probe_min": args.probe_min if args.probe_min is not None else table.get("probe_min", 5),
        "drop_cache": not (args.keep_cache or args.mock),
    }
    stamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    out = (args.out or Path(f"soak-{args.target}-{stamp}")).resolve()
    model = args.model.resolve() if args.model else None
    result = soak.run(recipe, plan, hw, args.target, args.engine_dir.resolve(), tree, model, out, options)
    verdict = result["verdict"]
    print(verdict["summary"])
    for line in testkit_soak.lines(verdict):
        print("  " + line)
    head = bench.head(recipe, args.kit, spec)
    path = receipt.write(args.receipt or out / "receipt.json", "soak", head, {"mock": args.mock, **result})
    print(f"receipt: {path}", file=sys.stderr)
    return 0 if verdict["verdict"] == "STABLE" else 1


def kit_bundle(args: argparse.Namespace) -> int:
    if args.verify:
        problems = bundle.verify(args.verify.resolve())
        for problem in problems:
            print(problem)
        print(f"recipe kit --verify: {'FAILED' if problems else 'every file matches'} {bundle.SUMS}")
        return 1 if problems else 0
    if not (args.dest and args.engine_dir):
        raise manifest.ManifestError("recipe kit DEST ENGINE_DIR (or --verify BUNDLE)")
    recipe = _recipe(args)
    spec = recipe.engine(args.engine)
    root = bundle.build(
        recipe, spec, args.engine_dir.resolve(), (args.tree or args.engine_dir).resolve(), args.dest.resolve(), args.kit
    )
    print(f"bundle {root} and {root.name}.tar.gz: {sum(1 for _ in root.rglob('*'))} entries; see {bundle.MANIFEST}")
    return 0


def fetch(args: argparse.Namespace) -> int:
    recipe, spec, dest = _load(args)
    tree = engine.fetch(recipe, spec, dest, args.source)
    _record_names(recipe, spec, dest)
    print(_summary(spec, dest, tree))
    return 0


def _record_names(recipe: manifest.Recipe, spec: manifest.Engine, dest: Path) -> None:
    try:
        record.save(recipe, spec, names=compat.engine_names(dest))
    except manifest.ManifestError as err:
        print(f"warning: no names recorded for [engine.{spec.name}]: {err}", file=sys.stderr)


def prepare(args: argparse.Namespace) -> int:
    recipe, spec, dest = _load(args)
    if spec.build is None:
        raise manifest.ManifestError(f"{recipe.path}: [engine.{spec.name}.build] is missing")
    tree = engine.fetch(recipe, spec, dest, args.source)
    _record_names(recipe, spec, dest)
    print(_summary(spec, dest, tree))
    for output in build.build(spec.build, dest, addons=not args.no_addons):
        print(f"built {output}")
    binary = dest / spec.build.outputs[0]
    caps = compat.capabilities(binary)
    record.save(recipe, spec, binary=record.sha256(binary), capabilities=caps, by="recipe prepare")
    print(f"recorded {binary.name} capabilities in {record.path(recipe, spec)}")
    return 0


def update_engine(args: argparse.Namespace) -> int:
    recipe, spec, dest = _load(args)
    rev = args.rev or engine.latest(spec.repo, args.branch)
    tree = engine.update(recipe, spec, rev, dest, args.source)
    print(
        f"[engine.{spec.name}] {spec.rev[:9]} -> {rev[:9]}, tree {spec.tree[:9]} -> {tree[:9]}; {recipe.path} updated"
    )
    if args.build:
        if spec.build is None:
            raise manifest.ManifestError(f"{recipe.path}: [engine.{spec.name}.build] is missing")
        for output in build.build(spec.build, dest):
            print(f"built {output}")
    print("run the recipe's gates and bench on the new pin before committing it", file=sys.stderr)
    return 0


def env_exports(args: argparse.Namespace) -> int:
    sys.stdout.write(envfile.exports(envfile.unset(envfile.load(args.files), os.environ)))
    return 0


def publish_recipe(args: argparse.Namespace) -> int:
    recipe = _recipe(args)
    result = publish.publish(recipe, args.rev, args.dest.resolve(), os.environ.get("CHECK_PUBLIC_EXTRA", ""))
    if result.report.findings:
        print(result.report.render())
        print(f"recipe publish: FINDINGS above ({result.report.scanned} files scanned); nothing written")
        return 1
    print(f"kit release {result.release}.tar.gz sha256 {result.sha256}")
    print(f"recipe publish: {len(result.files)} files in {args.dest}, {len(result.excluded)} kept private")
    return 0


def publish_check(args: argparse.Namespace) -> int:
    report = scrub.check(args.root.resolve(), args.history, args.revs)
    history = f", {report.commits} commit(s) of history" if report.commits is not None else ""
    if report.findings:
        print(report.render())
        print(f"publish-check: FINDINGS above ({report.scanned} files scanned{history})")
        return 1
    print(f"publish-check: clean ({report.scanned} files scanned{history})")
    return 0


def _summary(spec: manifest.Engine, dest: Path, tree: str) -> str:
    layers = " ".join(f"{s.directory}:{s.last}" if s.last else str(s.directory) for s in spec.series) or "no patches"
    return f"{spec.name} engine in {dest}: {spec.repo} {spec.rev[:9]} + {layers}, tree {tree}"
