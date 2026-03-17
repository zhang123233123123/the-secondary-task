import pytest

from backend.llm_client_factory import build_chat_client
from backend.llm_clients import LLMError, OpenAICompatibleChatClient
from backend.llm_clients_gemini import GeminiChatClient
from backend.llm_clients_transformers import TransformersChatClient
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


def test_client_factory_selects_openai_compatible():
    client = build_chat_client(
        LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com/v1",
            temperature=0.0,
            top_p=1.0,
            seed=None,
        )
    )
    assert isinstance(client, OpenAICompatibleChatClient)


def test_client_factory_selects_openai():
    client = build_chat_client(
        LLMConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
            temperature=0.0,
            top_p=1.0,
            seed=None,
        )
    )
    assert isinstance(client, OpenAICompatibleChatClient)


def test_client_factory_selects_gemini():
    client = build_chat_client(
        LLMConfig(
            provider="gemini",
            model="gemini-2.0-flash",
            api_key_env="GEMINI_API_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            temperature=0.0,
            top_p=1.0,
            seed=None,
        )
    )
    assert isinstance(client, GeminiChatClient)


def test_client_factory_selects_transformers():
    client = build_chat_client(
        LLMConfig(
            provider="transformers",
            model="google/gemma-3-12b-it",
            api_key_env="UNUSED",
            base_url="",
            temperature=0.1,
            top_p=0.9,
            seed=42,
            load_in_4bit=True,
        )
    )
    assert isinstance(client, TransformersChatClient)


def test_transformers_client_raises_without_dependencies(monkeypatch):
    client = TransformersChatClient(
        LLMConfig(
            provider="transformers",
            model="google/gemma-3-12b-it",
            api_key_env="UNUSED",
            base_url="",
            temperature=0.1,
            top_p=0.9,
            seed=42,
            load_in_4bit=True,
        )
    )
    monkeypatch.setattr(
        client,
        "_import_transformers_modules",
        lambda: (_ for _ in ()).throw(ImportError("no transformers")),
    )
    with pytest.raises(LLMError, match="Transformers backend requires"):
        client.chat([{"role": "user", "content": "hello"}], timeout_seconds=1.0)


def test_client_factory_rejects_unknown_provider():
    with pytest.raises(LLMError, match="Unsupported LLM provider"):
        build_chat_client(
            LLMConfig(
                provider="unknown",
                model="x",
                api_key_env="X",
                base_url="",
                temperature=0.1,
                top_p=0.9,
                seed=None,
            )
        )


def test_gemini_requires_process_environment_api_key(monkeypatch):
    monkeypatch.delenv("MISSING_GEMINI_TEST_KEY", raising=False)
    client = GeminiChatClient(
        LLMConfig(
            provider="gemini",
            model="gemini-2.0-flash",
            api_key_env="MISSING_GEMINI_TEST_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            temperature=0.0,
            top_p=1.0,
            seed=None,
        )
    )

    with pytest.raises(LLMError, match="Missing API key environment variable"):
        client.chat([{"role": "user", "content": "hello"}], timeout_seconds=1.0)
