"""Run the refusal + quality eval against any OpenAI-compatible endpoint.

Usage:
    python3 run_eval.py --label ctrl --base-url http://127.0.0.1:8123/v1

Greedy decoding (temperature 0) and fixed per-suite max token budgets.
Prompts and model replies are written under eval/private/runs/<label>/;
only aggregate numbers go into summary.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import grading
import openai_client
import sources

EVAL_DIR = Path(__file__).resolve().parent

MAX_TOKENS = {"refusal": 256, "mmlu": 32, "humaneval": 512}


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ask(client_args, messages: dict, max_tokens: int) -> tuple[str, float]:
    start = time.monotonic()
    reply = openai_client.chat_completion(client_args.base_url, client_args.model, messages, max_tokens)
    return reply, time.monotonic() - start


def _dump(out_dir: Path, name: str, records: list[dict]) -> None:
    with (out_dir / f"items-{name}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def suite_refusal(args, data_dir: Path, out_dir: Path) -> dict:
    harmful = sources.harmful_prompts(data_dir, args.seed, args.n_harmful)
    benign = sources.benign_prompts(data_dir, args.seed, args.n_benign)
    results = {"harmful": [], "benign": []}
    for prompt in harmful:
        reply, elapsed = _ask(args, [{"role": "user", "content": prompt.prompt}], MAX_TOKENS["refusal"])
        results["harmful"].append(
            {
                "id": prompt.id,
                "prompt": prompt.prompt,
                "reply": reply,
                "refusal": grading.is_refusal(reply),
                "elapsed_s": round(elapsed, 2),
            }
        )
        print(
            f"  harmful {prompt.id} refusal={results['harmful'][-1]['refusal']}",
            flush=True,
        )
    for prompt in benign:
        reply, elapsed = _ask(args, [{"role": "user", "content": prompt.prompt}], MAX_TOKENS["refusal"])
        results["benign"].append(
            {
                "id": prompt.id,
                "prompt": prompt.prompt,
                "reply": reply,
                "refusal": grading.is_refusal(reply),
                "elapsed_s": round(elapsed, 2),
            }
        )
        print(
            f"  benign {prompt.id} refusal={results['benign'][-1]['refusal']}",
            flush=True,
        )

    _dump(out_dir, "refusal_harmful", results["harmful"])
    _dump(out_dir, "refusal_benign", results["benign"])
    n_refused = sum(item["refusal"] for item in results["harmful"])
    n_over = sum(item["refusal"] for item in results["benign"])
    return {
        "n_harmful": len(harmful),
        "refusals": n_refused,
        "refusal_pct": round(100.0 * n_refused / len(harmful), 2) if harmful else 0.0,
        "n_benign": len(benign),
        "over_refusals": n_over,
        "over_refusal_pct": round(100.0 * n_over / len(benign), 2) if benign else 0.0,
    }


def suite_mmlu(args, data_dir: Path, out_dir: Path) -> dict:
    questions = sources.mmlu_questions(data_dir, args.seed, args.n_mmlu)
    records = []
    for question in questions:
        letters = ["A", "B", "C", "D"]
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, question.choices, strict=False))
        subject = question.subject.replace("_", " ")
        message = (
            f"The following is a multiple choice question about {subject}. "
            "Reply with the letter of the correct option only.\n\n"
            f"Question: {question.question}\n{options}\nAnswer:"
        )
        reply, elapsed = _ask(args, [{"role": "user", "content": message}], MAX_TOKENS["mmlu"])
        parsed = grading.parse_mmlu_answer(reply)
        correct = parsed == question.answer
        records.append(
            {
                "id": question.id,
                "subject": question.subject,
                "question": question.question,
                "choices": question.choices,
                "gold": question.answer,
                "reply": reply,
                "parsed": parsed,
                "correct": correct,
                "elapsed_s": round(elapsed, 2),
            }
        )
        print(f"  mmlu {question.id} correct={correct}", flush=True)

    _dump(out_dir, "mmlu", records)
    n_correct = sum(item["correct"] for item in records)
    n_unparsed = sum(item["parsed"] is None for item in records)
    return {
        "n": len(records),
        "correct": n_correct,
        "unparsed": n_unparsed,
        "accuracy_pct": round(100.0 * n_correct / len(records), 2) if records else 0.0,
    }


def suite_humaneval(args, data_dir: Path, out_dir: Path) -> dict:
    tasks = sources.humaneval_tasks(data_dir, args.seed, args.n_humaneval)
    sandbox_dir = out_dir / "humaneval_sandbox"
    records = []
    for task in tasks:
        message = (
            "Complete the following Python function. Reply with the complete "
            "function definition (including the signature shown below) in a single "
            "Python code block, and nothing else.\n\n" + task.prompt
        )
        reply, elapsed = _ask(args, [{"role": "user", "content": message}], MAX_TOKENS["humaneval"])
        code = grading.extract_code(reply)
        passed, error = grading.run_humaneval_program(
            code, task.test, task.entry_point, sandbox_dir / task.id.replace("/", "_")
        )
        records.append(
            {
                "id": task.id,
                "entry_point": task.entry_point,
                "prompt": task.prompt,
                "reply": reply,
                "code": code,
                "passed": passed,
                "error": error,
                "elapsed_s": round(elapsed, 2),
            }
        )
        print(f"  humaneval {task.id} passed={passed}", flush=True)

    _dump(out_dir, "humaneval", records)
    n_passed = sum(item["passed"] for item in records)
    return {
        "n": len(records),
        "passed": n_passed,
        "pass_at_1_pct": round(100.0 * n_passed / len(records), 2) if records else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8123/v1",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="served model id; defaults to the first listed model",
    )
    parser.add_argument("--label", required=True, help="run label, e.g. ctrl or candidate")
    parser.add_argument("--seed", type=int, default=sources.SEED)
    parser.add_argument("--data-dir", type=Path, default=EVAL_DIR / "private" / "data")
    parser.add_argument("--out-root", type=Path, default=EVAL_DIR / "private" / "runs")
    parser.add_argument("--only", default="refusal,mmlu,humaneval", help="comma-separated suites to run")
    parser.add_argument("--n-harmful", type=int, default=50)
    parser.add_argument("--n-benign", type=int, default=30)
    parser.add_argument("--n-mmlu", type=int, default=200)
    parser.add_argument("--n-humaneval", type=int, default=40)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="truncate every suite to N items (smoke runs only)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"data dir {args.data_dir} missing; run fetch_data.py first")
    if args.model is None:
        args.model = openai_client.first_model(args.base_url)
        print(f"using served model id: {args.model}")
    if args.limit is not None:
        args.n_harmful = min(args.n_harmful, args.limit)
        args.n_benign = min(args.n_benign, args.limit)
        args.n_mmlu = min(args.n_mmlu, args.limit)
        args.n_humaneval = min(args.n_humaneval, args.limit)

    out_dir = args.out_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = {}
    started = time.monotonic()
    for name in [s.strip() for s in args.only.split(",") if s.strip()]:
        print(f"suite {name} starting {_utc_now()}", flush=True)
        suite_start = time.monotonic()
        if name == "refusal":
            suites["refusal"] = suite_refusal(args, args.data_dir, out_dir)
        elif name == "mmlu":
            suites["mmlu"] = suite_mmlu(args, args.data_dir, out_dir)
        elif name == "humaneval":
            suites["humaneval"] = suite_humaneval(args, args.data_dir, out_dir)
        else:
            raise SystemExit(f"unknown suite: {name}")
        suites[name]["wall_time_s"] = round(time.monotonic() - suite_start, 1)
        print(
            f"suite {name} done: {json.dumps(suites[name], ensure_ascii=False)}",
            flush=True,
        )

    summary = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "started_utc": _utc_now(),
        "wall_time_s": round(time.monotonic() - started, 1),
        "suites": suites,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
