"""Render REPORT.md from one or two run summaries.

Usage:
    python3 make_report.py --ctrl private/runs/ctrl/summary.json \
        [--cand private/runs/candidate/summary.json] --out REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PLATFORM_LABEL = "GB10, TensorFold TP1"


def _delta(ctrl: float, cand: float) -> str:
    return f"{round(cand - ctrl, 2):+.2f}"


def _row(name: str, key_path: list, ctrl: dict, cand: dict | None, unit: str = "%") -> str:
    ctrl_value = ctrl
    cand_value = cand
    for key in key_path:
        ctrl_value = ctrl_value[key]
        if cand_value is not None:
            cand_value = cand_value[key]
    cells = [name, f"{ctrl_value:.2f}{unit}"]
    if cand_value is not None:
        cells += [f"{cand_value:.2f}{unit}", _delta(ctrl_value, cand_value)]
    else:
        cells += ["pending", "pending"]
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--cand", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "REPORT.md")
    args = parser.parse_args()

    ctrl = json.loads(args.ctrl.read_text())
    cand = json.loads(args.cand.read_text()) if args.cand else None
    ctrl_s = ctrl["suites"]
    cand_s = cand["suites"] if cand else None

    lines = [
        "# Qwen3.8 Flash Next abliterated vs control — eval report",
        "",
        f"Platform: {PLATFORM_LABEL}. Greedy decoding, fixed max-token budgets per suite.",
        f"Control: `{ctrl['label']}` (model `{ctrl['model']}`).",
    ]
    if cand:
        lines.append(f"Candidate: `{cand['label']}` (model `{cand['model']}`).")
    else:
        lines.append("Candidate run pending.")
    lines += [
        "",
        "| Metric | Control | Candidate | Delta |",
        "| --- | --- | --- | --- |",
        _row("AdvBench harmful refusal rate", ["refusal", "refusal_pct"], ctrl_s, cand_s),
        _row(
            "XSTest benign over-refusal rate",
            ["refusal", "over_refusal_pct"],
            ctrl_s,
            cand_s,
        ),
        _row("MMLU accuracy (200 questions)", ["mmlu", "accuracy_pct"], ctrl_s, cand_s),
        _row(
            "HumanEval pass@1 (40 tasks)",
            ["humaneval", "pass_at_1_pct"],
            ctrl_s,
            cand_s,
        ),
        "",
        (
            "Refusal classification is keyword-based (XSTest string-match convention, "
            "extended with standard substring markers); it is a heuristic applied "
            "identically to both models."
        ),
        "",
        "Details: control run "
        + f"`{ctrl['label']}`"
        + (f", candidate run `{cand['label']}`" if cand else "")
        + " under `eval/private/runs/` (git-ignored).",
        "",
    ]
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
