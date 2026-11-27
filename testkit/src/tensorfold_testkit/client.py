"""One OpenAI-compatible endpoint; every reply reduced to the evidence a gate reads."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field

SHA = re.compile(r"[0-9a-f]{12}")
# GET /v1/load http_active endpoints that generate tokens; a gate prompt may use any of them
GENERATING = ("chat_completions", "completions", "responses", "messages")
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass
class Reply:
    status: int
    error: str | None = None
    sha: str | None = None
    sha256: str | None = None
    tokens: int | None = None
    text: str = ""
    finish: str | None = None
    tool_calls: list = field(default_factory=list)

    def token_hash(self) -> str | None:
        """The full token hash, else the short one; None for a reply that is not evidence."""
        if self.error or not self.tokens:
            return None
        return self.sha256 or self.sha

    def matches(self, expected: str) -> bool | None:
        """Whether the reply's tokens are the expected ones (12 or 64 hex); None when it carries no hash to say."""
        got = self.token_hash()
        if got is None or (len(expected) == 64 and len(got) != 64):
            return None
        return got.startswith(expected) if len(expected) == 12 else got == expected


class Client:
    def __init__(self, url: str, model: str | None = None, timeout: float = 600):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def post(self, path: str, body: dict, stream: bool = False) -> Reply:
        body = {**({"model": self.model} if self.model else {}), **body}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        request = urllib.request.Request(
            self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return read_stream(reply) if stream else reply_of(reply.status, json.load(reply))
        except urllib.error.HTTPError as err:
            raw = err.read(1 << 16)
            try:
                doc = json.loads(raw)
            except ValueError:
                doc = {"error": raw.decode(errors="replace")}
            return reply_of(err.code, doc)
        except Exception as err:  # a failed request is evidence, not a crash
            return Reply(0, f"{type(err).__name__}: {err}")

    def get(self, path: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(self.url + path, timeout=30) as reply:
                return reply.status, reply.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read(1 << 16)
        except Exception:  # an unanswered probe is evidence, not a crash
            return 0, b""

    def active(self) -> int | None:
        """Generation requests open now, from TensorFold's GET /v1/load; None when the server reports none."""
        status, raw = self.get("/v1/load")
        try:
            open_ = json.loads(raw).get("http_active", {}) if status == 200 else None
            seen = [open_.get(k) for k in GENERATING] if isinstance(open_, dict) else []
        except (ValueError, AttributeError):
            return None
        seen = [n for n in seen if type(n) is int]
        return sum(seen) if seen else None

    def together(self, calls: list[tuple[str, dict]], stream: bool = False) -> tuple[list[Reply], int | None]:
        """Sends every request at once; the replies in order and the most generation requests seen open meanwhile."""
        out: list[Reply | None] = [None] * len(calls)
        start, done, peak = threading.Barrier(len(calls)), threading.Event(), [self.active()]

        def run(i):
            start.wait(timeout=60)
            out[i] = self.post(*calls[i], stream=stream)

        def poll():
            while peak[0] is not None and not done.wait(0.005):
                seen = self.active()
                peak[0] = None if seen is None else max(peak[0], seen)

        watcher = threading.Thread(target=poll, daemon=True)
        watcher.start()
        threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(calls))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(self.timeout + 60)
        done.set()
        watcher.join(5)
        return [r or Reply(0, "request did not finish") for r in out], peak[0]


def reply_of(status: int, doc) -> Reply:
    doc = doc if isinstance(doc, dict) else {}
    if status != 200 or "error" in doc:
        err = doc.get("error")
        return Reply(status, (err.get("message") if isinstance(err, dict) else err) or f"HTTP {status}")
    choice = (doc.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") if "message" in choice else choice.get("text")
    usage = doc.get("usage") or {}
    return _evidence(status, doc.get("tensorfold"), usage, text, choice.get("finish_reason"), message.get("tool_calls"))


def read_stream(reply) -> Reply:
    runtime, usage, finish, text, ended = {}, {}, None, [], False
    for raw in reply:
        line = raw.decode().strip()
        if line == "data: [DONE]":
            ended = True
            break
        if not line.startswith("data:"):
            continue
        chunk = json.loads(line[5:])
        if "error" in chunk:
            return reply_of(reply.status, chunk)
        runtime, usage = chunk.get("tensorfold") or runtime, chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            text.append(choice.get("text") or (choice.get("delta") or {}).get("content") or "")
            finish = choice.get("finish_reason") or finish
    if not ended:
        return Reply(reply.status, "the stream ended before [DONE]")
    return _evidence(reply.status, runtime, usage, "".join(text), finish, None)


def _evidence(status, runtime, usage, text, finish, calls) -> Reply:
    runtime = runtime if isinstance(runtime, dict) else {}
    sha, sha256 = runtime.get("token_sha"), runtime.get("token_sha256")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return Reply(
        status,
        sha=sha if isinstance(sha, str) and SHA.fullmatch(sha) else None,
        sha256=sha256 if isinstance(sha256, str) and SHA256.fullmatch(sha256) else None,
        tokens=tokens if type(tokens) is int and tokens >= 0 else None,
        text=text if isinstance(text, str) else "",
        finish=finish,
        tool_calls=calls if isinstance(calls, list) else [],
    )
