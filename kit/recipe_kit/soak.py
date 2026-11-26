"""recipe soak: one fresh server for a profile or arm as recipe serve renders it, the testkit's soak, one receipt."""

from __future__ import annotations

import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tensorfold_testkit import receipt
from tensorfold_testkit import soak as testkit
from tensorfold_testkit.gates import hashes
from tensorfold_testkit.server import Server, ServerError, binary_hashes, drop_page_cache, healthy

from . import bench, profiles
from .hardware import Hardware
from .manifest import Recipe

ENGINE_ERRORS = re.compile(r"out of memory|CUDA_ERROR|panic|error:", re.IGNORECASE)


def run(
    recipe: Recipe, plan: profiles.Plan, hw: Hardware, target: str, engine_dir: Path, tree: Path, model, out, options
) -> dict:
    """The soak of target on this host at the hardware profile's floors; the result is a receipt's body."""
    start = time.perf_counter()
    rendered = profiles.render(plan, target, hw.env)
    table_path = recipe.root / recipe.data.get("gates", {}).get("hashes", "gates/hashes.json")
    table = hashes.load(table_path)
    if options["mock"]:
        spec = bench.mock_arm(recipe, rendered, table_path, options["port"], 1, options["fault"])
    else:
        spec = bench.arm(recipe, rendered, hw, engine_dir, model, {}, options["port"], options["gpus"], options["numa"])
    floors = testkit.Floors(
        hw.floor_gib, hw.thermal.get("hard_c"), hw.thermal.get("hard_hold_s", 0), hw.power.get("limit_w")
    )
    args = rendered.args
    context = int(args[args.index("--context") + 1]) if "--context" in args else testkit.Options.context
    soak_options = testkit.Options(
        minutes=options["minutes"],
        workers=options["workers"],
        probe_s=options["probe_min"] * 60,
        context=context,
    )
    cache = drop_page_cache() if options["drop_cache"] and hw.drop_caches else "kept"
    logs = out / "server"
    for attempt in (1, 2):  # a boot that fails is tried once more, as soak.sh does
        try:
            with Server(spec, logs) as server:
                meter = testkit.mock_meter(spec.url) if options["mock"] else testkit.meter(hw.unified)

                def alive(server=server) -> bool:
                    return all(p.poll() is None for p in server.procs) and healthy(spec.url)

                record = testkit.run(
                    spec.url,
                    recipe.served_name,
                    tree / "tools",
                    bench.cold_prompts(table),
                    out,
                    soak_options,
                    meter,
                    alive,
                )
                boot_s = server.boot_s
            break
        except ServerError:
            if attempt == 2:
                raise
            print(f"soak: boot {attempt} failed; trying once more", file=sys.stderr)
    record["engine_errors"] = [
        line.strip()
        for log in sorted(logs.glob("rank*.log"))
        for line in log.read_text(errors="replace").splitlines()
        if ENGINE_ERRORS.search(line)
    ][:20]
    for row in record["error_samples"]:
        row["error"] = receipt.redact(row["error"] or "")
    record["engine_errors"] = [receipt.redact(e) for e in record["engine_errors"]]
    verdict = testkit.verdict(record, floors, hashes.bench_expected(table, target))
    return {
        "arm": {**spec.settings, "binary": binary_hashes(spec.binary)},
        "limits": {**asdict(floors), "unified": hw.unified},
        "cache": cache,
        "boot_s": boot_s,
        "soak": record,
        "verdict": verdict,
        "seconds": round(time.perf_counter() - start, 3),
    }
