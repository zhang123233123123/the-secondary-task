import json
import re
import random

from pathlib import Path

import pytest

from backend.llm_clients import ChatResult
from backend.prepare_orchestrator import _extract_json_payload, prepare_inputs
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

    llm2_calls: list[str] = []

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del timeout_seconds
        if self.config.model == "llm1-test":
            return ChatResult(
                text=json.dumps(
                    {
                        "conditions": {"default": "d", "unhelpful": "e", "cynical": "c", "distant": "x"},
                        "judge_system": "judge",
                        "judge_rubric": "rubric",
                        "judge_schema": {
                            "type": "object",
                            "properties": {
                                "harm_1_10": {"type": "integer"},
                                "negative_emotion_1_10": {"type": "integer"},
                                "inappropriate_1_10": {"type": "integer"},
                                "empathic_language_1_10": {"type": "integer"},
                                "anthro_q1": {"type": "integer"},
                                "anthro_q2": {"type": "integer"},
                                "anthro_q3": {"type": "integer"},
                                "anthro_q4": {"type": "integer"},
                                "anthro_q5": {"type": "integer"},
                            },
                            "required": [
                                "harm_1_10",
                                "negative_emotion_1_10",
                                "inappropriate_1_10",
                                "empathic_language_1_10",
                                "anthro_q1",
                                "anthro_q2",
                                "anthro_q3",
                                "anthro_q4",
                                "anthro_q5",
                            ],
                        },
                    }
                ),
                latency_ms=10,
                raw={},
            )
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_prompt = str(msg.get("content", ""))
                break
        domain = "creative"
        marker = "domain must be '"
        if marker in user_prompt:
            start = user_prompt.index(marker) + len(marker)
            end = user_prompt.find("'", start)
            if end > start:
                domain = user_prompt[start:end]
        llm2_calls.append(domain)
        return ChatResult(
            text=json.dumps(
                {
                    "dialogue_id": f"D{len(llm2_calls)}",
                    "domain": domain,
                    "turns": [
                        {"role": "user", "text": "hello"},
                        {"role": "user", "text": "world"},
                    ],
                }
            ),
            latency_ms=12,
            raw={},
        )

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    config = load_config(config_path)
    manifest = prepare_inputs(config=config, config_path=str(config_path), target_version="vprep")

    assert manifest["prepare_id"] == "vprep"
    assert manifest["llm2_request_count"] == 2
    assert manifest["llm2_latency_ms"] == 24
    assert manifest["prepare_dialogue_turns_min"] == 1
    assert manifest["prepare_dialogue_turns_max"] == 2
    assert manifest["prepare_dialogue_turns_mode"] == "random_min_to_max"
    assert Path(manifest["index_path"]).exists()
    assert Path(manifest["prompts_candidate"]).exists()
    assert Path(manifest["dialogues_candidate"]).exists()
    assert len(llm2_calls) == 2


def test_prepare_inputs_can_skip_llm1_and_reuse_prompts_file(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts_ready.json"
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "default": "d",
                    "unhelpful": "e",
                    "cynical": "c",
                    "distant": "x",
                },
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frozen_index_path: frozen_inputs/index.json",
                f"prompts_path: {prompts_path.name}",
                "prepare_dialogue_count: 1",
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
            ]
        ),
        encoding="utf-8",
    )

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del messages, timeout_seconds
        if self.config.model == "llm1-test":
            raise AssertionError("llm1 should be skipped")
        return ChatResult(
            text=json.dumps(
                {
                    "dialogue_id": "D1",
                    "domain": "creative",
                    "turns": [
                        {"role": "user", "text": "hello"},
                        {"role": "user", "text": "world"},
                    ],
                }
            ),
            latency_ms=7,
            raw={},
        )

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    config = load_config(config_path)
    manifest = prepare_inputs(
        config=config,
        config_path=str(config_path),
        target_version="vprep_skip_llm1",
        skip_llm1=True,
    )

    assert manifest["prepare_id"] == "vprep_skip_llm1"
    assert manifest["skip_llm1"] is True
    assert manifest["llm1_model"] is None
    assert manifest["llm1_latency_ms"] is None
    prompts_candidate = Path(manifest["prompts_candidate"])
    assert json.loads(prompts_candidate.read_text(encoding="utf-8"))["judge_system"] == "judge"


def test_prepare_inputs_uses_random_turn_count_up_to_max(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts_ready.json"
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "default": "d",
                    "unhelpful": "e",
                    "cynical": "c",
                    "distant": "x",
                },
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frozen_index_path: frozen_inputs/index.json",
                f"prompts_path: {prompts_path.name}",
                "prepare_dialogue_count: 5",
                "prepare_dialogue_min_turns: 2",
                "prepare_dialogue_turns: 3",
                "llm2:",
                "  provider: deepseek",
                "  model: llm2-test",
                "  api_key_env: TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.9",
                "  top_p: 1.0",
                "  seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    requested_turns: list[int] = []

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del timeout_seconds
        if self.config.model == "deepseek-chat":
            raise AssertionError("llm1 should be skipped")
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_prompt = str(msg.get("content", ""))
                break
        match = re.search(r"EXACTLY (\d+) items", user_prompt)
        if not match:
            raise AssertionError("missing target turn count in llm2 prompt")
        turn_count = int(match.group(1))
        requested_turns.append(turn_count)
        return ChatResult(
            text=json.dumps(
                {
                    "dialogue_id": f"D{len(requested_turns)}",
                    "domain": "creative",
                    "turns": [
                        {"role": "user", "text": f"t{i}"} for i in range(1, turn_count + 1)
                    ],
                }
            ),
            latency_ms=5,
            raw={},
        )

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    config = load_config(config_path)
    manifest = prepare_inputs(
        config=config,
        config_path=str(config_path),
        target_version="vprep_random_turns",
        skip_llm1=True,
    )

    rng = random.Random(7)
    expected_turns = [rng.randint(2, 3) for _ in range(5)]
    assert requested_turns == expected_turns
    assert manifest["prepare_dialogue_turns_min"] == 2
    assert manifest["prepare_dialogue_turns_max"] == 3


def test_prepare_inputs_retries_single_dialogue_when_json_invalid(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts_ready.json"
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "default": "d",
                    "unhelpful": "e",
                    "cynical": "c",
                    "distant": "x",
                },
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frozen_index_path: frozen_inputs/index.json",
                f"prompts_path: {prompts_path.name}",
                "prepare_dialogue_count: 1",
                "prepare_dialogue_min_turns: 1",
                "prepare_dialogue_turns: 2",
                "retries: 1",
                "llm2:",
                "  provider: deepseek",
                "  model: llm2-test",
                "  api_key_env: TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.9",
                "  top_p: 1.0",
                "  seed: 1",
            ]
        ),
        encoding="utf-8",
    )

    attempts = {"count": 0}

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del messages, timeout_seconds, self
        attempts["count"] += 1
        if attempts["count"] == 1:
            return ChatResult(
                text='{"dialogue_id":"D1","domain":"creative","turns":[{"role": "user": "user", "text":"bad"}]}',
                latency_ms=3,
                raw={},
            )
        return ChatResult(
            text=json.dumps(
                {
                    "dialogue_id": "D1",
                    "domain": "creative",
                    "turns": [{"role": "user", "text": "ok"}],
                }
            ),
            latency_ms=4,
            raw={},
        )

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    monkeypatch.setattr("backend.prepare_orchestrator.time.sleep", lambda _: None)
    config = load_config(config_path)
    manifest = prepare_inputs(
        config=config,
        config_path=str(config_path),
        target_version="vprep_retry_json",
        skip_llm1=True,
    )

    assert manifest["prepare_id"] == "vprep_retry_json"
    assert attempts["count"] == 2


def test_prepare_inputs_continues_after_single_item_failure_until_target_count(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts_ready.json"
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "default": "d",
                    "unhelpful": "e",
                    "cynical": "c",
                    "distant": "x",
                },
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frozen_index_path: frozen_inputs/index.json",
                f"prompts_path: {prompts_path.name}",
                "prepare_dialogue_count: 2",
                "prepare_dialogue_min_turns: 1",
                "prepare_dialogue_turns: 1",
                "retries: 0",
                "llm2:",
                "  provider: deepseek",
                "  model: llm2-test",
                "  api_key_env: TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.9",
                "  top_p: 1.0",
                "  seed: 1",
            ]
        ),
        encoding="utf-8",
    )

    attempts = {"count": 0}

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del messages, timeout_seconds, self
        attempts["count"] += 1
        if attempts["count"] == 1:
            return ChatResult(
                text='{"dialogue_id":"BROKEN","domain":"creative","turns":[{"role": "user": "user", "text":"bad"}]}',
                latency_ms=3,
                raw={},
            )
        return ChatResult(
            text=json.dumps(
                {
                    "dialogue_id": f"D{attempts['count']}",
                    "domain": "creative",
                    "turns": [{"role": "user", "text": "ok"}],
                }
            ),
            latency_ms=4,
            raw={},
        )

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    monkeypatch.setattr("backend.prepare_orchestrator.time.sleep", lambda _: None)
    config = load_config(config_path)
    manifest = prepare_inputs(
        config=config,
        config_path=str(config_path),
        target_version="vprep_continue_after_failure",
        skip_llm1=True,
    )

    assert manifest["prepare_id"] == "vprep_continue_after_failure"
    assert manifest["llm2_request_count"] == 2
    assert manifest["llm2_failed_attempts"] == 1
    assert manifest["llm2_generation_attempts"] == 3
    assert attempts["count"] == 3


def test_extract_json_payload_keeps_comment_markers_inside_strings():
    payload = '{"url":"https://example.com/a?x=1","code":"/* keep */ // keep"}'
    parsed = _extract_json_payload(payload)
    assert parsed["url"] == "https://example.com/a?x=1"
    assert parsed["code"] == "/* keep */ // keep"


def test_prepare_inputs_validates_reused_prompts_before_llm2_calls(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts_invalid.json"
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "default": "d",
                    "unhelpful": "",
                    "cynical": "c",
                    "distant": "x",
                },
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frozen_index_path: frozen_inputs/index.json",
                f"prompts_path: {prompts_path.name}",
                "prepare_dialogue_count: 1",
                "prepare_dialogue_turns: 2",
                "llm2:",
                "  provider: deepseek",
                "  model: llm2-test",
                "  api_key_env: TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.9",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        raise AssertionError("llm2 should not be called when reused prompts are invalid")

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    config = load_config(config_path)
    with pytest.raises(ValueError, match="conditions.unhelpful"):
        prepare_inputs(
            config=config,
            config_path=str(config_path),
            target_version="vprep_invalid_prompts",
            skip_llm1=True,
        )
