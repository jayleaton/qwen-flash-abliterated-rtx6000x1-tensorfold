"""A stand-in TensorFold server: recorded replies where it has them, deterministic ones elsewhere, and one planted
fault a gate.

It answers what the gates and upstream's conformance.py read: chat and completions (JSON or streamed), the
``tensorfold`` token hashes, usage with cached tokens, draft counters, ``GET /v1/load``, ``/metrics``, ``/health``,
``/v1/models`` and the request refusals; ``tps`` paces its streams on a fixed schedule, so upstream's bench clients
measure it. A request's tokens depend only on its prompt and sampling, never on drafts, batching or the prompt cache,
as an exact engine's do. Prompts are kept in a word-counted LRU cache.

Faults (``FAULTS``) each break one property a gate holds:
  conformance     a drafted reply's tokens differ from the plain reply's
  hashes          a recorded reply comes back with other tokens
  nondeterminism  a request that runs beside another returns one of two outputs
  canary          the probes are answered wrong
  mmlu            every question is answered with the wrong letter
  tools           tool-call arguments are not JSON

Soak faults (``SOAK_FAULTS``) each break one property the soak holds:
  error           every fifth request fails with HTTP 500
  leak            every seventh request keeps its slot after it ends, so the server never drains
  drift           the tokens change once the stand-in has served 20 requests
  memory          ``GET /mock/memory`` reports free memory under any floor
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAULTS = ("conformance", "hashes", "nondeterminism", "canary", "mmlu", "tools")
SOAK_FAULTS = ("error", "leak", "drift", "memory")
DRIFT_AFTER, FREE_GIB = 20, 64.0
CANARY = {
    "17*23": "391",
    "capital of France": "Paris",
    "Count from 1 to 12": "1 2 3 4 5 6 7 8 9 10 11 12",
    "function add(a, b)": "def add(a, b):\n    return a + b",
}
LETTERS = "ABCD"


def full_hash(sha: str) -> str:
    """A recording holds the 12-hex token hash; the stand-in's 64-hex one starts with it."""
    return sha if len(sha) == 64 else sha + hashlib.sha256(sha.encode()).hexdigest()[:52]


class Listener(ThreadingHTTPServer):
    daemon_threads, request_queue_size = True, 128


class Mock:
    def __init__(
        self,
        name="mock",
        recorded=(),
        mmlu=(),
        chains=(),
        fault=None,
        cache_words=4000,
        context=49152,
        delay=0.01,
        chunk_delay=0.0005,
        tps=None,
        prefill_tps=None,
        port=0,
    ):
        if fault not in (None, *FAULTS, *SOAK_FAULTS):
            raise ValueError(f"unknown fault {fault}: choose from {', '.join(FAULTS + SOAK_FAULTS)}")
        self.name, self.fault, self.cache_words, self.context = name, fault, cache_words, context
        # tps paces a stream at that many tokens a second, 4 a chunk; prefill_tps adds a prompt's words / prefill_tps
        self.delay, self.port, self.prefill_tps = delay, port, prefill_tps
        self.chunk_delay = 4 / tps if tps else chunk_delay
        self.recorded = {self.key(r["prompt"], r["temperature"], r.get("seed"), r["max_tokens"]): r for r in recorded}
        self.answers = {r["question"]: r["answer"] for r in mmlu}
        self.chains = list(chains)
        self.kept: OrderedDict[tuple, int] = OrderedDict()
        self.lock, self.active, self.disconnects, self.counter = threading.Lock(), 0, 0, itertools.count()
        self.open = {"chat_completions": 0, "completions": 0}  # by endpoint, as GET /v1/load reports them
        self.served = itertools.count(1)

    @staticmethod
    def key(prompt: str, temperature: float, seed, max_tokens: int) -> tuple:
        return prompt, float(temperature), seed if temperature else None, max_tokens

    def __enter__(self):
        self.http = Listener(("127.0.0.1", self.port), handler(self))
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, args=(0.01,), daemon=True).start()
        return self

    def __exit__(self, *_):
        self.http.shutdown()
        self.http.server_close()

    def refusal(self, body: dict, words: int) -> str | None:
        n = body.get("n")
        if n is not None and (type(n) is not int or not 1 <= n <= 16):
            return "n must be an integer between 1 and 16"
        if body.get("top_logprobs") is not None and not body.get("logprobs"):
            return "top_logprobs requires logprobs: true"
        penalty = body.get("presence_penalty")
        if penalty is not None and not -2 <= penalty <= 2:
            return "presence_penalty must be between -2 and 2, or null"
        if words + body.get("max_tokens", 16) > self.context:
            return f"This server's maximum context length is {self.context} tokens"
        return None

    def cached(self, words: tuple) -> int:
        """Tokens resumed from the longest kept prefix; then this prompt is kept, oldest evicted past the budget."""
        with self.lock:
            best = 0
            for kept in self.kept:
                n = next(
                    (i for i, (a, b) in enumerate(zip(kept, words, strict=False)) if a != b), min(len(kept), len(words))
                )
                best = max(best, n)
            if self.cache_words:
                self.kept[words] = len(words)
                self.kept.move_to_end(words)
                while sum(self.kept.values()) > self.cache_words and len(self.kept) > 1:
                    self.kept.popitem(last=False)
            return min(best, len(words) - 1)

    def reply(self, body: dict, prompt: str, together: bool) -> dict:
        """Text, token hash, token count, finish and tool calls for one request."""
        temperature, tokens = body.get("temperature", 1.0) or 0, body.get("max_tokens", 16)
        sampling = [body.get(k) for k in ("top_k", "top_p")] if temperature else []
        record = self.recorded.get(self.key(raw_prompt(body), temperature, body.get("seed"), tokens))
        salt = ""
        if self.fault == "conformance" and body.get("draft", True):
            salt = "drafted"
        if self.fault == "nondeterminism" and together:
            salt = f"batch{next(self.counter) % 2}"
        if self.fault == "drift" and next(self.served) > DRIFT_AFTER:
            salt = "drift"
        if record:
            sha = record["token_sha"] if self.fault != "hashes" else hashlib.sha256(prompt.encode()).hexdigest()[:12]
            digest = full_hash(sha)
            if salt:
                digest = hashlib.sha256((digest + salt).encode()).hexdigest()
            text = record.get("text") or " ".join(f"w{digest[i % 60 : i % 60 + 4]}" for i in range(tokens))
            return {"text": text, "sha256": digest, "tokens": tokens, "finish": "length"}
        digest = hashlib.sha256(
            json.dumps(
                [prompt, temperature, body.get("seed") if temperature else None, sampling, tokens, salt]
            ).encode()
        ).hexdigest()
        words = tokens if body.get("ignore_eos") else 8
        text = self.answer(body, prompt) or " ".join(f"w{digest[i % 60 : i % 60 + 4]}" for i in range(words))
        if isinstance(text, dict):
            return {"text": "", "sha256": digest, "tokens": 9, "finish": "tool_calls", "tool_calls": [text]}
        count = tokens if body.get("ignore_eos") else min(tokens, len(text.split()) + 1)
        return {"text": text, "sha256": digest, "tokens": count, "finish": "length" if count == tokens else "stop"}

    def answer(self, body: dict, prompt: str):
        """A canned right answer to a canary probe, an MMLU row or a tool-chain step, broken by its fault."""
        for cue, text in CANARY.items():
            if cue in prompt:
                return "London" if self.fault == "canary" else text
        for question, letter in self.answers.items():
            if question in prompt:
                return LETTERS[(LETTERS.index(letter) + 1) % 4] if self.fault == "mmlu" else letter
        for chain in self.chains:
            if body.get("tools") and chain["messages"][-1]["content"] in prompt:
                done = sum(m.get("role") == "tool" for m in body["messages"])
                if done == len(chain["steps"]):
                    return f"Done: {chain['answer']}"
                arguments = "{not json" if self.fault == "tools" else json.dumps({"step": done})
                return {
                    "id": f"call_{done}",
                    "type": "function",
                    "function": {"name": chain["steps"][done]["call"], "arguments": arguments},
                }
        return None


def prompt_of(body: dict) -> str:
    if "messages" in body:
        return "\n".join(f"{m.get('role')}: {m.get('content') or ''}" for m in body["messages"])
    prompt = body.get("prompt", "")
    return " ".join(map(str, prompt)) if isinstance(prompt, list) else str(prompt)


def raw_prompt(body: dict) -> str:
    """What a recording is keyed by: a completion's prompt, or a one-message chat's text."""
    messages, prompt = body.get("messages") or [], body.get("prompt", "")
    if len(messages) == 1:
        return messages[0].get("content", "")
    return json.dumps(prompt) if isinstance(prompt, list) else str(prompt)


def handler(mock: Mock):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status: int, doc, kind="application/json"):
            raw = doc if isinstance(doc, bytes) else json.dumps(doc).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"status": "ok"})
            if self.path == "/mock/memory":
                return self.send(200, {"free_gib": 0.5 if mock.fault == "memory" else FREE_GIB})
            if self.path == "/v1/models":
                caps = {"thinking_switch": True, "thinking_applicable": True}
                model = {"id": mock.name, "object": "model", "capabilities": caps}
                return self.send(200, {"object": "list", "data": [model]})
            if self.path == "/v1/load":
                with mock.lock:
                    n, open_ = mock.active, dict(mock.open)
                return self.send(200, {"running": n, "waiting": 0, "http_active": open_})
            if self.path == "/metrics":
                text = f"tensorfold:client_disconnections_total {mock.disconnects}\n"
                return self.send(200, text.encode(), "text/plain")
            self.send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                return self.send(404, {"error": {"message": "not found"}})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            prompt = prompt_of(body)
            words = tuple(prompt.split())
            refused = mock.refusal(body, len(words))
            if refused:
                return self.send(400, {"error": {"message": refused, "type": "invalid_request_error"}})
            route = self.path.removeprefix("/v1/").replace("/", "_")
            with mock.lock:
                mock.active += 1
                mock.open[route] += 1
                together = mock.active > 1
                n = next(mock.counter) if mock.fault in ("error", "leak") else 0
            if mock.fault == "error" and n % 5 == 4:
                with mock.lock:
                    mock.active -= 1
                    mock.open[route] -= 1
                return self.send(500, {"error": {"message": "planted error", "type": "server_error"}})
            try:
                time.sleep(mock.delay + (len(words) / mock.prefill_tps if mock.prefill_tps else 0))
                with mock.lock:
                    together = together or mock.active > 1
                self.answer(body, prompt, words, together)
            except (BrokenPipeError, ConnectionResetError):
                with mock.lock:
                    mock.disconnects += 1
            finally:
                with mock.lock:
                    gone = 0 if mock.fault == "leak" and n % 7 == 6 else 1
                    mock.active -= gone
                    mock.open[route] -= gone

        def answer(self, body, prompt, words, together):
            out = mock.reply(body, prompt, together)
            cached = mock.cached(words + tuple(out["text"].split()))
            usage = {
                "prompt_tokens": len(words),
                "completion_tokens": out["tokens"],
                "total_tokens": len(words) + out["tokens"],
                "prompt_tokens_details": {"cached_tokens": cached},
            }
            drafts = body.get("draft", True)
            runtime = {"token_sha": out["sha256"][:12], "token_sha256": out["sha256"]}
            spec = {"accepted": out["tokens"] // 2 if drafts else 0, "drafted": out["tokens"] - 1 if drafts else 0}
            chat = self.path == "/v1/chat/completions"
            if not body.get("stream"):
                message = {"role": "assistant", "content": out["text"] or None}
                if out.get("tool_calls"):
                    message["tool_calls"] = out["tool_calls"]
                choice = {
                    "index": 0,
                    "finish_reason": out["finish"],
                    **({"message": message} if chat else {"text": out["text"]}),
                }
                return self.send(
                    200,
                    {
                        "object": "chat.completion" if chat else "text_completion",
                        "choices": [choice],
                        "usage": usage,
                        "tensorfold": runtime,
                        "speculative": spec,
                    },
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            pieces = out["text"].split(" ")
            steps, start = max(1, out["tokens"] // 4), time.perf_counter()
            for i in range(steps):
                piece = " ".join(pieces[i * len(pieces) // steps : (i + 1) * len(pieces) // steps])
                piece = piece + " " if i + 1 < steps and piece else piece
                choice = {"index": 0, "delta": {"content": piece}} if chat else {"index": 0, "text": piece}
                self.event({"choices": [choice]})
                time.sleep(max(0.0, start + (i + 1) * mock.chunk_delay - time.perf_counter()))  # a fixed schedule
            last = {"index": 0, "finish_reason": out["finish"], **({"delta": {}} if chat else {"text": ""})}
            self.event({"choices": [last], "usage": usage, "tensorfold": runtime, "speculative": spec})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def event(self, doc):
            self.wfile.write(b"data: " + json.dumps(doc).encode() + b"\n\n")
            self.wfile.flush()

    return Handler
