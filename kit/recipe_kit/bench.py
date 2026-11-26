"""recipe bench: a server spec for each target from the manifest, then the testkit's ABBA, and one receipt."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tensorfold_testkit
from tensorfold_testkit import bench as abba
from tensorfold_testkit.gates import hashes
from tensorfold_testkit.mock import server as stand_in
from tensorfold_testkit.server import Rank, ServerSpec

from . import compat, profiles, serve
from .hardware import Hardware
from .manifest import Engine, ManifestError, Recipe

TESTKIT = Path(tensorfold_testkit.__file__).resolve().parents[1]
MOCK_TPS, MOCK_PREFILL_TPS = 2000.0, 25000.0


def head(recipe: Recipe, kit: str, spec: Engine) -> dict:
    """A receipt's head: the kit, the recipe and the engine pin it ran with."""
    return {
        "kit": kit,
        "dev": kit.startswith("dev:"),
        "recipe": recipe.name,
        "hardware": recipe.hardware_of(spec),
        "engine": {"name": spec.name, "rev": spec.rev, "tree": spec.tree, "series": patches(recipe, spec)},
    }


def patches(recipe: Recipe, spec: Engine) -> int | None:
    """How many patches the series applies; None where its directories are not here (a bundle)."""
    total = 0
    for layer in spec.series:
        folder = recipe.root / layer.directory
        if not folder.is_dir():
            return None
        names = sorted(p.name for p in folder.glob("[0-9][0-9][0-9][0-9]-*.patch"))
        total += len([n for n in names if layer.last is None or n[:4] <= layer.last])
    return total


def settings(rendered: profiles.Rendered) -> dict:
    return {
        "target": rendered.target,
        "profile": rendered.profile.name,
        "ranks": rendered.ranks,
        "args": rendered.args,
        "env": rendered.env,
        "experiments": rendered.experiments,
        "prompt_cache_gib": prompt_cache(rendered),
        **({"when_model_has": rendered.optional} if rendered.optional else {}),
    }


def arm(
    recipe: Recipe,
    rendered: profiles.Rendered,
    hw: Hardware,
    engine_dir: Path,
    model: Path | None,
    caps: dict[Path, dict],
    port: int,
    gpus: list[str] | None,
    numa: str | None,
) -> ServerSpec:
    """One target's launch, exactly as recipe serve renders it."""
    spec = recipe.engine(rendered.profile.engine)
    if spec.build is None or not spec.build.outputs:
        raise ManifestError(f"[engine.{spec.name}.build] outputs must name the server binary first")
    binary = engine_dir / spec.build.outputs[0]
    if not binary.is_file():
        raise ManifestError(f"{engine_dir} has no {spec.build.outputs[0]}: run recipe prepare")
    if binary not in caps:
        caps[binary] = compat.capabilities(binary)
    if problems := compat.check(caps[binary], recipe, spec, hw, rendered):
        raise ManifestError(f"{rendered.target}: the binary cannot serve this:\n  " + "\n  ".join(problems))
    if model is None or not (model / "config.json").is_file():
        raise ManifestError(f"{model} has no config.json")
    launches = serve.ranks(recipe, rendered, binary, model, [], gpus, port, os.environ, numa)
    ranks = [Rank(r.argv, {**r.env, "_": r.argv[0]}) for r in launches]
    return ServerSpec(rendered.target, ranks, f"http://127.0.0.1:{port}", binary, settings(rendered))


def prompt_cache(rendered: profiles.Rendered) -> str:
    """The prompt cache the target serves with: its --prompt-cache-gib, else the engine's default."""
    args = rendered.args
    return args[args.index("--prompt-cache-gib") + 1] if "--prompt-cache-gib" in args else "engine default"


def mock_arm(
    recipe: Recipe, rendered: profiles.Rendered, table: Path, port: int, gain: float, fault: str | None = None
) -> ServerSpec:
    """The stand-in in place of the engine, paced gain times the stand-in's speed: the whole flow with no GPU."""
    argv = [
        sys.executable,
        "-m",
        "tensorfold_testkit",
        "mock",
        "--port",
        str(port),
        *(["--no-cache"] if prompt_cache(rendered) == "0" else []),
        "--name",
        recipe.served_name,
        "--tps",
        str(MOCK_TPS * gain),
        "--prefill-tps",
        str(MOCK_PREFILL_TPS),
        "--hashes",
        str(table),
        *(["--fault", fault] if fault else []),
    ]
    binary = Path(stand_in.__file__)
    return ServerSpec(
        rendered.target,
        [Rank(argv, {"PYTHONPATH": str(TESTKIT)})],
        f"http://127.0.0.1:{port}",
        binary,
        {**settings(rendered), "mock": True},
    )


def cold_prompts(table: dict) -> dict[str, Path]:
    return {size: row["path"] for size, row in table.get("cold", {}).items()}


def run(
    recipe: Recipe,
    plan: profiles.Plan,
    hw: Hardware,
    labels: dict[str, str],
    roles: tuple[str, str, str | None],
    engine_dir: Path,
    tree: Path,
    model: Path | None,
    out: Path,
    options: dict,
) -> dict:
    """The ABBA window of roles (A, B, control: labels of the targets they run) as [bench] and the options set it."""
    a, b, control = roles
    table_path = recipe.root / recipe.data.get("gates", {}).get("hashes", "gates/hashes.json")
    table = hashes.load(table_path)
    caps: dict[Path, dict] = {}
    specs = {}
    for name, target in labels.items():
        rendered = profiles.render(plan, target, hw.env)
        if options["mock"]:
            specs[name] = mock_arm(recipe, rendered, table_path, options["port"], options["gain"] if name == b else 1)
        else:
            specs[name] = arm(
                recipe, rendered, hw, engine_dir, model, caps, options["port"], options["gpus"], options["numa"]
            )
    result = abba.abba(
        specs,
        a,
        b,
        control,
        options["cells"],
        options["rounds"],
        options["bar"],
        out,
        tree / "tools",
        recipe.served_name,
        cold_prompts(table),
        hashes.bench_expected(table, b),
        drop_cache=options["drop_cache"],
        log=lambda text: print(text, file=sys.stderr, flush=True),
    )
    seen = {
        specs[n].binary.name: {"schema": c.get("schema"), "version": c.get("version")}
        for n in specs
        for binary, c in caps.items()
        if specs[n].binary == binary
    }
    result["capabilities"] = seen or None
    return result
