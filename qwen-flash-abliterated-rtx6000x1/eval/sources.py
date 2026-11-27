"""Load the pinned datasets and sample fixed eval subsets.

Sampling is driven by one seed (SEED, also exposed as run_eval.py --seed) so
every run of the harness asks a model the identical questions. Pools are
sorted by a stable key before sampling, and each sampled record carries a
stable id used for output ordering and result diffing.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path

SEED = 1234


@dataclass
class HarmfulPrompt:
    id: str
    prompt: str


@dataclass
class BenignPrompt:
    id: str
    prompt: str
    xstype: str


@dataclass
class MmluQuestion:
    id: str
    subject: str
    question: str
    choices: list[str]
    answer: str


@dataclass
class HumanEvalTask:
    id: str
    prompt: str
    entry_point: str
    test: str


def _sample(pool: list, n: int, seed: int) -> list:
    return sorted(random.Random(seed).sample(pool, n), key=lambda item: item.id)


def harmful_prompts(data_dir: Path, seed: int = SEED, n: int = 50) -> list[HarmfulPrompt]:
    with (data_dir / "advbench_harmful_behaviors.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pool = [
        HarmfulPrompt(id=f"advbench-{i:04d}", prompt=row["goal"].strip())
        for i, row in enumerate(sorted(rows, key=lambda r: r["goal"]))
    ]
    return _sample(pool, n, seed)


def benign_prompts(data_dir: Path, seed: int = SEED, n: int = 30) -> list[BenignPrompt]:
    with (data_dir / "xstest_prompts.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["label"] == "safe"]
    pool = [
        BenignPrompt(id=f"xstest-{row['id']}", prompt=row["prompt"].strip(), xstype=row["type"])
        for row in sorted(rows, key=lambda r: int(r["id"]))
    ]
    return _sample(pool, n, seed)


def mmlu_questions(data_dir: Path, seed: int = SEED, n: int = 200) -> list[MmluQuestion]:
    pool: list[MmluQuestion] = []
    with tarfile.open(data_dir / "mmlu_data.tar") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if "/test/" not in member.name or not member.name.endswith(".csv"):
                continue
            subject = member.name.rsplit("/", 1)[1].removesuffix("_test.csv")
            text = tar.extractfile(member).read().decode("utf-8")
            for i, row in enumerate(csv.reader(io.StringIO(text))):
                if len(row) != 6:
                    continue
                pool.append(
                    MmluQuestion(
                        id=f"mmlu-{subject}-{i:04d}",
                        subject=subject,
                        question=row[0],
                        choices=row[1:5],
                        answer=row[5],
                    )
                )
    pool.sort(key=lambda q: q.id)
    return _sample(pool, n, seed)


def humaneval_tasks(data_dir: Path, seed: int = SEED, n: int = 40) -> list[HumanEvalTask]:
    with gzip.open(data_dir / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as handle:
        pool = [
            HumanEvalTask(
                id=task["task_id"],
                prompt=task["prompt"],
                entry_point=task["entry_point"],
                test=task["test"],
            )
            for task in (json.loads(line) for line in handle)
        ]
    pool.sort(key=lambda t: t.id)
    return _sample(pool, n, seed)
