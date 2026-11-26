"""Expected token hashes (a recipe's gates/hashes.json) and the gates that hold a server to them.

- hashes: each prompt's greedy reply, then each seeded sampled reply, carries the expected token hash; a reply
  without a hash fails. The requests are upstream bench_openai's, so the table holds the hashes it reports.
- nondeterminism: N copies of one greedy and one seeded request, all sent at once, return identical tokens.

The table's optional ``cold`` rows name a bench cold prompt (token ids, a path from the table's directory) and,
optionally, the unsalted prompt's greedy reply hash for the record; the bench salts each cold sample, so it holds a
cold cell to bench_cache's checks instead of this hash.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..client import Client, Reply
from . import GateResult

HEX = re.compile(r"[0-9a-f]{12}|[0-9a-f]{64}")
PATHS = {"chat": "/v1/chat/completions", "completion": "/v1/completions"}


class TableError(ValueError):
    pass


def load(path: Path) -> dict:
    try:
        table = json.loads(Path(path).read_text())
    except (OSError, ValueError) as err:
        raise TableError(f"{path}: {err}") from None
    if table.get("schema") != 1:
        raise TableError(f"{path}: schema must be 1")
    for key in ("tokens", "greedy_seed", "sampled", "seeds", "prompts", "expected"):
        if key not in table:
            raise TableError(f"{path}: no {key}")
    for name, prompt in table["prompts"].items():
        if prompt.get("kind") not in PATHS or not isinstance(prompt.get("prompt"), str):
            raise TableError(f"{path}: prompt {name} needs kind chat or completion and a prompt")
    for profile, rows in table["expected"].items():
        for name, row in rows.items():
            if name not in table["prompts"]:
                raise TableError(f"{path}: expected[{profile}] names unknown prompt {name}")
            sampled = row.get("sampled", [])
            if not all(isinstance(h, str) and HEX.fullmatch(h) for h in [row.get("greedy"), *sampled]):
                raise TableError(f"{path}: expected[{profile}][{name}] hashes must be 12 or 64 lowercase hex")
            if sampled and len(sampled) != len(table["seeds"]):
                raise TableError(f"{path}: expected[{profile}][{name}] needs one sampled hash a seed")
    for size, row in table.get("cold", {}).items():
        bad_hash = "hash" in row and not HEX.fullmatch(str(row["hash"]))
        if not size.isdigit() or bad_hash or not isinstance(row.get("prompt"), str):
            raise TableError(f"{path}: cold[{size}] needs a size in token ids and a prompt file (hash: 12 or 64 hex)")
        row["path"] = Path(path).parent / row["prompt"]
    return table


def bench_expected(table: dict, profile: str | None) -> dict[str, str]:
    """The hash each bench decode cell must carry: the greedy rows of upstream bench_openai's prompts."""
    sets = {"fibonacci-raw": "code", "gpu-chat-no-think": "chat"}
    return {f"decode.{sets[name]}": row["greedy"] for name, row in expected(table, profile).items() if name in sets}


def expected(table: dict, profile: str | None) -> dict:
    rows = table["expected"].get(profile) if profile else None
    if rows is None and "*" not in table["expected"]:
        raise TableError(f"no expected hashes for profile {profile} and no '*' row")
    return rows if rows is not None else table["expected"]["*"]


def request(table: dict, name: str, seed: int, sampled: bool) -> tuple[str, dict]:
    """bench_openai's request for one prompt: path and body."""
    prompt = table["prompts"][name]
    body = {"max_tokens": table["tokens"], "temperature": 0, "ignore_eos": True, "seed": seed}
    if sampled:
        body.update(table["sampled"])
    if prompt["kind"] == "chat":
        body["messages"] = [{"role": "user", "content": prompt["prompt"]}]
        if prompt.get("thinking") is False:
            body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["prompt"] = prompt["prompt"]
    return PATHS[prompt["kind"]], body


def hashes(client: Client, table: dict, profile: str | None, quick: bool = False, plant: bool = False) -> GateResult:
    """Greedy then seeded hashes against the table. quick: greedy only. plant: every seed moves by one."""
    want, rows, bad = expected(table, profile), [], []
    for name, row in want.items():
        cases = [("greedy", table["greedy_seed"], False, row["greedy"])]
        if not quick:
            cases += [
                ("sampled", s + plant, True, h) for s, h in zip(table["seeds"], row.get("sampled", []), strict=False)
            ]
        for mode, seed, sampled, hash_ in cases:
            reply = client.post(*request(table, name, seed, sampled), stream=True)
            verdict = reply.matches(hash_)
            rows.append({"prompt": name, "mode": mode, "seed": seed, "expected": hash_, "got": reply.token_hash()})
            if verdict is not True:
                bad.append(f"{name} {mode} seed {seed}: " + _why(reply, hash_, verdict))
    equal = len(rows) - len(bad)
    line = (
        f"{equal}/{len(rows)} equal" + (" (greedy only)" if quick else "") + ("; " + "; ".join(bad[:4]) if bad else "")
    )
    return GateResult("hashes", bool(rows) and not bad, line, rows=rows)


def nondeterminism(client: Client, table: dict, n: int = 8) -> GateResult:
    """N copies of the first prompt, greedy and seeded, sent together: every copy returns the same tokens."""
    name = next(iter(table["prompts"]))
    calls = {m: request(table, name, table["seeds"][0], m == "sampled") for m in ("greedy", "sampled")}
    replies, peak = client.together([calls[m] for m in calls for _ in range(n)])
    rows, bad = [], []
    for i, mode in enumerate(calls):
        got = [r.token_hash() for r in replies[i * n : (i + 1) * n]]
        rows.append({"mode": mode, "hashes": got})
        if None in got:
            bad.append(f"{mode}: {got.count(None)} of {n} replies carry no hash")
        elif len(set(got)) > 1:
            bad.append(f"{mode}: {len(set(got))} different outputs over {n} copies")
    if peak is not None and peak < 2:
        bad.append(f"GET /v1/load never showed two requests open (peak {peak}), so nothing ran together")
    seen = "load not reported" if peak is None else f"up to {peak} open at once"
    line = f"{2 * n} copies of {name} ({seen}); " + ("; ".join(bad) if bad else "one output a mode")
    return GateResult("nondeterminism", not bad, line, rows=rows)


def _why(reply: Reply, hash_: str, verdict: bool | None) -> str:
    if verdict is False:
        return f"got {reply.token_hash()[: len(hash_)]}, expected {hash_}"
    return reply.error or "no token hash in the reply"
