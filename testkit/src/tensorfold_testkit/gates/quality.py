"""Answer gates: the canary, MMLU-200 and tool chains. Greedy, thinking off, one request at a time.

- canary: three fixed probes, each with a fresh nonce so no cached state answers it; each reply must hold its answer
  and not look broken (empty, U+FFFD, a loop).
- mmlu: JSONL rows {question, choices[4], answer A-D}; the share answered right must reach the bar.
- tools: a JSON list of chains {name, tools, messages, steps: [{call, result}], answer}; each step must be a call
  to that function with JSON-object arguments, answered with its result; the last reply must contain answer.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from ..client import Client
from . import GateResult

CHAT = "/v1/chat/completions"
PROBES = (
    ("capital", "What is the capital of France? Answer with one word.", 16, (r"\bParis\b",)),
    (
        "count",
        "Count from 1 to 12, separated by single spaces. Output only the numbers.",
        48,
        (
            r"\b1\D+2\D+3\D+4\D+5"
            r"\D+6\D+7\D+8\D+9\D+10\D+11\D+12\b",
        ),
    ),
    (
        "code",
        "Write a Python function add(a, b) that returns a + b. Output only the code, no explanation.",
        48,
        (r"def\s+add\s*\(", r"return"),
    ),
)
LETTERS = "ABCD"


def greedy(messages: list, max_tokens: int, **extra) -> dict:
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }


def degenerate(text: str) -> str | None:
    if not text.strip():
        return "empty reply"
    if "�" in text:
        return "U+FFFD in the reply"
    words = text.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.3:
        return "repetitive reply"
    if re.search(r"(\S+)(\s+\1){6}", text) or re.search(r"(.)\1{23,}", text):
        return "a loop in the reply"
    return None


def canary(client: Client) -> GateResult:
    nonce, bad = uuid.uuid4().hex[:8], []
    for name, prompt, tokens, expect in PROBES:
        reply = client.post(CHAT, greedy([{"role": "user", "content": f"[canary {nonce}] {prompt}"}], tokens))
        why = (
            reply.error
            or degenerate(reply.text)
            or next((f"no /{p}/" for p in expect if not re.search(p, reply.text)), None)
        )
        if why:
            bad.append(f"{name}: {why} in {reply.text[:60]!r}")
    return GateResult(
        "canary", not bad, f"{len(PROBES) - len(bad)}/{len(PROBES)} probes" + "".join("; " + b for b in bad)
    )


def mmlu_prompt(row: dict) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"], strict=True))
    return f"{row['question']}\n{options}\nAnswer with the letter only."


def load_mmlu(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or not all(len(r.get("choices", [])) == 4 and r.get("answer") in LETTERS for r in rows):
        raise ValueError(f"{path}: each row needs question, four choices and answer A-D")
    return rows


def mmlu(client: Client, rows: list[dict], minimum: float) -> GateResult:
    right = 0
    for row in rows:
        reply = client.post(CHAT, greedy([{"role": "user", "content": mmlu_prompt(row)}], 8))
        found = re.search(r"\b([ABCD])\b", reply.text)
        right += bool(found and found[1] == row["answer"])
    share = right / len(rows)
    return GateResult("mmlu", share >= minimum, f"{right}/{len(rows)} right ({share:.3f}, bar {minimum})")


def load_chains(path: Path) -> list[dict]:
    chains = json.loads(Path(path).read_text())
    if not chains or not all({"name", "tools", "messages", "steps", "answer"} <= set(c) for c in chains):
        raise ValueError(f"{path}: each chain needs name, tools, messages, steps and answer")
    return chains


def tools(client: Client, chains: list[dict]) -> GateResult:
    bad = []
    for chain in chains:
        why = run_chain(client, chain)
        if why:
            bad.append(f"{chain['name']}: {why}")
    return GateResult(
        "tools", not bad, f"{len(chains) - len(bad)}/{len(chains)} chains" + "".join("; " + b for b in bad)
    )


def run_chain(client: Client, chain: dict) -> str | None:
    messages = list(chain["messages"])
    for i, step in enumerate(chain["steps"] + [None]):
        reply = client.post(CHAT, greedy(messages, 256, tools=chain["tools"]))
        if reply.error:
            return f"step {i}: {reply.error}"
        if step is None:
            return None if chain["answer"] in reply.text else f"the answer lacks {chain['answer']!r}"
        calls = reply.tool_calls
        if reply.finish != "tool_calls" or len(calls) != 1:
            return f"step {i}: finish {reply.finish} with {len(calls)} calls, expected one call to {step['call']}"
        function = calls[0].get("function") or {}
        if function.get("name") != step["call"]:
            return f"step {i}: called {function.get('name')}, expected {step['call']}"
        try:
            if not isinstance(json.loads(function.get("arguments") or ""), dict):
                raise ValueError
        except ValueError:
            return f"step {i}: arguments are not a JSON object: {function.get('arguments')!r:.60}"
        messages += [
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0].get("id", ""), "content": step["result"]},
        ]
    return None
