import json

from pathlib import Path

from backend.llm_clients import ChatResult
from backend.prepare_orchestrator import prepare_inputs
from backend.runtime_config import load_config


def test_prepare_inputs_writes_valid_candidates(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frozen_index_path: frozen_inputs/index.json",
                "prepare_dialogue_count: 2",
                "prepare_dialogue_turns: 2",
                "llm1:",
                "  provider: deepseek",
                "  model: llm1-test",
                "  api_key_env: TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.2",
                "  top_p: 1.0",
                "llm2:",
                "  provider: deepseek",
                "  model: llm2-test",
                "  api_key_env: TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.9",
                "  top_p: 1.0",
                "llm3:",
                "  provider: deepseek",
                "  model: llm3-test",
                "  api_key_env: TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.7",
                "  top_p: 1.0",
                "llm4:",
                "  provider: deepseek",
                "  model: llm4-test",
                "  api_key_env: TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.0",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del messages, timeout_seconds
        if self.config.model == "llm1-test":
            return ChatResult(
                text=json.dumps(
                    {
                        "conditions": {"default": "d", "evil": "e", "distant": "x"},
                        "judge_system": "judge",
                        "judge_rubric": "rubric",
                        "judge_schema": {
                            "type": "object",
                            "properties": {
                                "harm": {"type": "integer"},
                                "emotion": {"type": "integer"},
                                "anthro": {"type": "integer"},
                            },
                            "required": ["harm", "emotion", "anthro"],
                        },
                    }
                ),
                latency_ms=10,
                raw={},
            )
        return ChatResult(
            text=json.dumps(
                {
                    "dialogues": [
                        {
                            "dialogue_id": "D1",
                            "domain": "creative",
                            "turns": [
                                {"role": "user", "text": "hello"},
                                {"role": "user", "text": "world"},
                            ],
                        },
                        {
                            "dialogue_id": "D2",
                            "domain": "finance",
                            "turns": [
                                {"role": "user", "text": "hi"},
                                {"role": "user", "text": "there"},
                            ],
                        },
                    ]
                }
            ),
            latency_ms=12,
            raw={},
        )

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    config = load_config(config_path)
    manifest = prepare_inputs(config=config, config_path=str(config_path), target_version="vprep")

    assert manifest["prepare_id"] == "vprep"
    assert Path(manifest["index_path"]).exists()
    assert Path(manifest["prompts_candidate"]).exists()
    assert Path(manifest["dialogues_candidate"]).exists()
