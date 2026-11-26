"""recipe kit: one offline bundle (the recipe's files, the built engine, the pinned tree's tools and this kit) with
a MANIFEST.txt and SHA256SUMS, so a GPU host runs gate and bench from it with no network."""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import tensorfold_testkit

from . import compat
from .manifest import Engine, ManifestError, Recipe
from .revision import KIT_ROOT

KIT_PARTS = ("bin", "kit", "testkit", "hardware", "templates", "README.md")
SUMS, MANIFEST = "SHA256SUMS", "MANIFEST.txt"


def recipe_files(recipe: Recipe) -> list[Path]:
    """recipe.toml, the files its [gates] name, the cold prompts its hashes table names and its weights checksums."""
    gates = recipe.data.get("gates", {})
    files = [recipe.path] + [recipe.root / gates[k] for k in ("conformance", "hashes", "tool_chains") if k in gates]
    if "hashes" in gates:
        table = json.loads((recipe.root / gates["hashes"]).read_text())
        files += [(recipe.root / gates["hashes"]).parent / row["prompt"] for row in table.get("cold", {}).values()]
    files += [p for p in [recipe.root / recipe.data["weights"]["sha256"]] if p.is_file()]
    if missing := [str(p) for p in files if not p.is_file()]:
        raise ManifestError(f"the recipe names files that are not there: {', '.join(missing)}")
    return [p.resolve() for p in files]


def kit_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(KIT_ROOT), "ls-files", "-z", "-co", "--exclude-standard", "--", *KIT_PARTS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [KIT_ROOT / name for name in filter(None, out.split("\0")) if (KIT_ROOT / name).is_file()]


def build(recipe: Recipe, spec: Engine, engine_dir: Path, tree: Path, dest: Path, kit: str) -> Path:
    """Writes DEST/<recipe>-<tree>/ and its .tar.gz; returns the directory."""
    if spec.build is None:
        raise ManifestError(f"[engine.{spec.name}.build] is missing")
    root = dest / f"{recipe.name}-{spec.tree[:12]}"
    if root.exists():
        raise ManifestError(f"{root} exists: remove it or choose another DEST")
    outputs = list(spec.build.outputs) + [o for a in spec.build.addons for o in a.outputs]
    if missing := [o for o in outputs[:1] if not (engine_dir / o).is_file()]:
        raise ManifestError(f"{engine_dir} has no {missing[0]}: run recipe prepare")
    plan = [(p, Path("recipe") / p.relative_to(recipe.root.resolve())) for p in recipe_files(recipe)]
    plan += [(engine_dir / o, Path("engine") / o) for o in outputs if (engine_dir / o).is_file()]
    plan += [(p, Path("engine/tools") / p.name) for p in sorted((tree / "tools").glob("*.py"))]
    plan += [(p, Path("kit") / p.relative_to(KIT_ROOT)) for p in kit_files()]
    for source, rel in plan:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, root / rel)
    (root / "engine" / compat.NAMES).write_text(json.dumps({"pin": None, "names": compat.engine_names(tree)}))
    (root / "kit" / "REVISION").write_text(kit + "\n")
    binaries = [Path("engine") / o for o in outputs if (root / "engine" / o).is_file()]
    (root / MANIFEST).write_text(manifest(recipe, spec, kit, root, binaries))
    files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file() and p.name != SUMS)
    (root / SUMS).write_text("".join(f"{sha256(root / f)}  {f}\n" for f in files))
    with tarfile.open(root.with_name(root.name + ".tar.gz"), "w:gz") as tar:
        tar.add(root, arcname=root.name)
    return root


def manifest(recipe: Recipe, spec: Engine, kit: str, root: Path, binaries: list[Path]) -> str:
    series = " ".join(f"{s.directory}:{s.last}" if s.last else str(s.directory) for s in spec.series) or "none"
    lines = [
        f"recipe {recipe.name} ({recipe.model}, {recipe.model_type} {recipe.format_of(spec)})",
        f"kit {kit}, testkit {tensorfold_testkit.__version__}",
        f"engine {spec.name}: {spec.repo} rev {spec.rev} tree {spec.tree}, series {series}",
        *(f"binary {b}: sha256 {sha256(root / b)} md5 {md5(root / b)}" for b in binaries),
        f"bundled {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "run it here: kit/bin/recipe gate --recipe recipe --tree engine --url URL",
        "             kit/bin/recipe bench A B engine MODEL --recipe recipe",
        "with no GPU: kit/bin/recipe gate --recipe recipe --mock",
        "             kit/bin/recipe bench A B engine --recipe recipe --mock",
        f"check it:    kit/bin/recipe kit --verify . (or sha256sum -c {SUMS})",
    ]
    return "\n".join(lines) + "\n"


def verify(root: Path) -> list[str]:
    """Each file whose checksum differs, that is missing, or that SHA256SUMS does not list."""
    sums = root / SUMS
    if not sums.is_file():
        return [f"{root} has no {SUMS}"]
    listed, problems = {}, []
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    for name, digest in listed.items():
        path = root / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif sha256(path) != digest:
            problems.append(f"{name}: checksum differs")
    extra = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != SUMS and "__pycache__" not in p.parts and str(p.relative_to(root)) not in listed
    )
    return problems + [f"{name}: not in {SUMS}" for name in extra]


def sha256(path: Path) -> str:
    return _digest(path, hashlib.sha256())


def md5(path: Path) -> str:
    return _digest(path, hashlib.md5(usedforsecurity=False))


def _digest(path: Path, h) -> str:
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
