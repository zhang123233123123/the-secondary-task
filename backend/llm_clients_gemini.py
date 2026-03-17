from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .llm_clients import ChatResult, LLMError
from .runtime_config import LLMConfig

try:
    import certifi
except Exception:  # noqa: BLE001
    certifi = None


class GeminiChatClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _build_endpoint(self, api_key: str) -> str:
        base_url = self.config.base_url.rstrip("/")
        if "{model}" in base_url:
            endpoint = base_url.format(model=self.config.model)
        else:
            endpoint = f"{base_url}/models/{self.config.model}:generateContent"
        separator = "&" if "?" in endpoint else "?"
        return endpoint + separator + urllib.parse.urlencode({"key": api_key})

    @staticmethod
    def _system_instruction(messages: list[dict[str, str]]) -> dict[str, Any] | None:
        system_parts = [
            {"text": item.get("content", "")}
            for item in messages
            if item.get("role") == "system" and item.get("content")
        ]
        if not system_parts:
            return None
        return {"parts": system_parts}

    @staticmethod
    def _contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for item in messages:
            role = item.get("role", "user")
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": item.get("content", "")}],
                }
            )
        return contents

    def chat(self, messages: list[dict[str, str]], timeout_seconds: float) -> ChatResult:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise LLMError(f"Missing API key environment variable: {self.config.api_key_env}")

        payload: dict[str, Any] = {
            "contents": self._contents(messages),
            "generationConfig": {
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
            },
        }
        system_instruction = self._system_instruction(messages)
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        if self.config.seed is not None:
            payload["generationConfig"]["seed"] = self.config.seed

        body = json.dumps(payload).encode("utf-8")
        endpoint = self._build_endpoint(api_key)
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        start = time.perf_counter()
        try:
            ssl_context = None
            if certifi is not None:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_payload = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"HTTP {exc.code}: {error_payload}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Network error: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)

        try:
            raw = json.loads(text)
            candidates = raw["candidates"]
            parts = candidates[0]["content"]["parts"]
            content = "".join(str(part.get("text", "")) for part in parts)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Invalid response payload: {text}") from exc

        return ChatResult(text=content.strip(), latency_ms=latency_ms, raw=raw)
