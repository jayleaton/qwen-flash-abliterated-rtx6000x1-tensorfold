"""Golden: a recipe with several engines. recipe serve starts each backend's profiles with exactly the argv and
environment in serve.json; inherited engines keep their parent's build less what they change, a false value drops an
inherited flag or setting, and recipe check runs C8 for every engine from its own record."""

import contextlib
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from recipe_kit import cli, manifest
from stand_in import KIT, MULTI, engine, launch, placeholders

GOLDEN = json.loads((MULTI / "serve.json").read_text())
RECORDS = MULTI / ".recipe" / "engines"
# The SM each CUDA engine's hardware profile names, as a build with nvcc reports it.
SMS = {"cuda": [121], "upstream": [121], "rtx": [120]}
DRAFTS = " with draft.safetensors"


def backend(target: str) -> str:
    return target.split("-")[0]


class ServeMulti(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.engines = {}
        for name in ("cuda", "upstream", "rtx", "metal"):
            saved = json.loads((RECORDS / f"{name}.json").read_text())
            caps = {**saved["capabilities"], **({"kernel_sms": SMS[name]} if name in SMS else {})}
            cls.engines[name] = engine(cls.base / f"engine-{name}", caps)
            (cls.engines[name] / "recipe-names.json").write_text(json.dumps({"names": saved["names"]}))
        cls.models = {}
        for drafts in (False, True):
            model = cls.base / ("model-draft" if drafts else "model")
            model.mkdir()
            (model / "config.json").write_text("{}")
            if drafts:
                (model / "draft.safetensors").write_bytes(b"")
            cls.models[drafts] = model

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def started(self, case: str, dump: Path) -> dict:
        target, drafts = case.removesuffix(DRAFTS), case.endswith(DRAFTS)
        engine_dir, model = self.engines[backend(target)], self.models[drafts]
        command = [str(KIT / "bin" / "recipe"), "serve", "--recipe", str(MULTI), "--kit-dev"]
        return placeholders(launch([*command, target, str(engine_dir), str(model)], dump), engine_dir, model)

    def test_every_backend_starts_as_the_golden(self):
        dumps = [self.base / f"dump-{n}" for n in range(len(GOLDEN))]
        with ThreadPoolExecutor(len(GOLDEN)) as pool:
            runs = dict(zip(GOLDEN, pool.map(self.started, GOLDEN, dumps), strict=True))
        for case, ranks in runs.items():
            with self.subTest(case=case):
                self.assertEqual(ranks, GOLDEN[case])
        self.assertIn("--drafter", GOLDEN["cuda" + DRAFTS]["rank0"]["argv"])
        self.assertNotIn("--drafter", GOLDEN["cuda"]["rank0"]["argv"])
        self.assertNotIn("--drafter", GOLDEN["cuda-no-draft" + DRAFTS]["rank0"]["argv"])
        self.assertNotIn("TF_EXAMPLE_DRAFT_HOLD", GOLDEN["upstream" + DRAFTS]["rank0"]["env"])

    def test_an_inherited_engine_merges_its_parents_build(self):
        rtx = manifest.load(MULTI).engine("rtx")
        self.assertEqual(
            (rtx.backend, rtx.repo, rtx.hardware), ("cuda", "https://github.com/example/TensorFold.git", "rtx5070tix1")
        )
        self.assertEqual(rtx.build.variables, {"NVCC": "/usr/local/cuda/bin/nvcc", "SM": "120"})
        self.assertEqual(rtx.build.steps, (("zig", "build", "native", "-Dsm={SM}", "-Dnvcc={NVCC}"),))
        self.assertEqual(
            (rtx.build.executables, rtx.build.outputs), (("NVCC",), ("zig-out/native/bin/tensorfold-native",))
        )
        self.assertEqual([a.name for a in rtx.build.addons], ["xgrammar"])

    def check(self, *flags: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["check", "--recipe", str(MULTI), "--kit-dev", *flags])
        return code, out.getvalue(), err.getvalue()

    def test_one_check_runs_c8_for_every_engine_from_its_record(self):
        self.assertEqual(self.check()[:2], (0, "recipe check: 0 finding(s) over 9 profiles and arms\n"))
        tree = str(self.engines["upstream"])
        self.assertIn("is no engine's pinned tree: name it with --engine", self.check("--tree", tree)[2])
        crossed = self.check("--tree", tree, "--engine", "cuda")
        self.assertEqual(crossed[1].splitlines()[0], "C8 TF_EXAMPLE_DRAFT_HOLD: the pinned engine does not read it")

    def test_an_engine_is_checked_against_its_own_binary(self):
        script = [str(KIT / "bin" / "recipe"), "serve", "--recipe", str(MULTI), "--kit-dev", "--dry-run"]
        with self.assertRaises(AssertionError) as caught:
            launch([*script, "metal", str(self.engines["cuda"]), str(self.models[False])], self.base / "wrong")
        self.assertIn("backend metal is not in this binary's ['cuda']", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
