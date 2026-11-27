"""Fetch the pinned public datasets used by run_eval.py into eval/private/data.

Every source is pinned by URL and SHA-256. A file that already matches its
checksum is skipped, so this is safe to re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

SOURCES = {
    "advbench_harmful_behaviors.csv": (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
        "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1",
    ),
    "xstest_prompts.csv": (
        "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
        "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
    ),
    "HumanEval.jsonl.gz": (
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    ),
    "mmlu_data.tar": (
        "https://huggingface.co/datasets/cais/mmlu/resolve/main/data.tar",
        "bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "private" / "data",
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for filename, (url, sha) in SOURCES.items():
        dest = args.data_dir / filename
        if dest.exists() and sha256_file(dest) == sha:
            print(f"ok (cached) {filename}")
            continue
        print(f"downloading {filename} from {url}")
        with (
            urllib.request.urlopen(url, timeout=600) as response,
            dest.open("wb") as out,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        actual = sha256_file(dest)
        if actual != sha:
            print(f"FAIL {filename}: sha256 {actual} != {sha}", file=sys.stderr)
            return 1
        print(f"ok {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
