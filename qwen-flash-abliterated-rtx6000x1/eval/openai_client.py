"""Minimal OpenAI-compatible chat client built on the standard library."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
    timeout_s: float = 600.0,
    retries: int = 3,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                data = json.loads(response.read())
                return data["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as error:
            if attempt == retries - 1:
                raise
            time.sleep(5.0 * (attempt + 1))
            last_error = error
    raise last_error  # pragma: no cover - unreachable


def first_model(base_url: str) -> str:
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read())
    return data["data"][0]["id"]
