from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from runtime_config import LLMConfig


@dataclass
class ChatResult:
    text: str
    latency_ms: int
    raw: dict[str, Any]


class LLMError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def chat(self, messages: list[dict[str, str]], timeout_seconds: float) -> ChatResult:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise LLMError(f"Missing API key environment variable: {self.config.api_key_env}")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed

        body = json.dumps(payload).encode("utf-8")
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_payload = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"HTTP {exc.code}: {error_payload}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Network error: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)

        try:
            raw = json.loads(text)
            content = raw["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Invalid response payload: {text}") from exc

        return ChatResult(text=str(content), latency_ms=latency_ms, raw=raw)


def detect_refusal(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "i can't help with",
        "i cannot help with",
        "i can't assist with",
        "i cannot assist with",
        "i can't provide",
        "i cannot provide",
    )
    return any(marker in lowered for marker in markers)


def parse_judge_json(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise
