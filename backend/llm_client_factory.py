from __future__ import annotations

from typing import Protocol

from .llm_clients_gemini import GeminiChatClient
from .llm_clients import LLMError, OpenAICompatibleChatClient
from .llm_clients_transformers import TransformersChatClient
from .runtime_config import LLMConfig


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], timeout_seconds: float): ...


def build_chat_client(config: LLMConfig) -> ChatClient:
    provider = config.provider.strip().lower()
    if provider in {"deepseek", "openai_compatible", "openai-compatible", "openai"}:
        return OpenAICompatibleChatClient(config)
    if provider == "gemini":
        return GeminiChatClient(config)
    if provider == "transformers":
        return TransformersChatClient(config)
    raise LLMError(f"Unsupported LLM provider: {config.provider}")
