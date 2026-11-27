"""Recompute refusal verdicts for an existing run from its stored replies.

Use after changing the refusal markers in grading.py so that a classifier
fix does not need a GPU re-run: replies are the model's record, and the
verdict is a pure function of the reply. MMLU and HumanEval results are
not touched (their grading does not depend on the refusal markers).
Usage:

    python3 regrade.py private/runs/ctrl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import grading


def _regrade_file(path: Path) -> tuple[int, int]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        record["refusal"] = grading.is_refusal(record["reply"])
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    refused = sum(record["refusal"] for record in records)
    return len(records), refused


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1])
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())

    suite = summary["suites"]["refusal"]
    n_harmful, refusals = _regrade_file(run_dir / "items-refusal_harmful.jsonl")
    n_benign, over_refusals = _regrade_file(run_dir / "items-refusal_benign.jsonl")
    suite.update(
        {
            "n_harmful": n_harmful,
            "refusals": refusals,
            "refusal_pct": round(100.0 * refusals / n_harmful, 2) if n_harmful else 0.0,
            "n_benign": n_benign,
            "over_refusals": over_refusals,
            "over_refusal_pct": round(100.0 * over_refusals / n_benign, 2) if n_benign else 0.0,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(suite, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
