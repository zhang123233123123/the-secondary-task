import pytest

from backend.llm_clients import LLMError, OpenAICompatibleChatClient
from backend.runtime_config import LLMConfig


def test_chat_requires_process_environment_api_key(monkeypatch):
    monkeypatch.delenv("MISSING_LLM_TEST_KEY", raising=False)
    client = OpenAICompatibleChatClient(
        LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key_env="MISSING_LLM_TEST_KEY",
            base_url="https://api.deepseek.com/v1",
            temperature=0.0,
            top_p=1.0,
            seed=None,
        )
    )

    with pytest.raises(LLMError, match="Missing API key environment variable"):
        client.chat([{"role": "user", "content": "hello"}], timeout_seconds=1.0)
