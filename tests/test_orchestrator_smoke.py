import json
from pathlib import Path

from backend.frozen_registry import approve_candidate
from backend.llm_clients import ChatResult, LLMError
from backend.orchestrator import run_experiment
from backend.runtime_config import load_config


def test_run_experiment_writes_results_without_api_key(tmp_path):
    dialogues_path = tmp_path / "dialogues.jsonl"
    prompts_path = tmp_path / "prompts.json"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"

    dialogues_path.write_text(
        '\n'.join(
            [
                '{"dialogue_id":"D1","domain":"creative","turns":[{"role":"user","text":"hello"}]}',
                '{"dialogue_id":"D2","domain":"finance","turns":[{"role":"user","text":"world"}]}',
            ]
        ),
        encoding="utf-8",
    )
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "default": "default",
                    "evil": "evil",
                    "distant": "distant",
                },
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                f"dialogues_path: {dialogues_path}",
                f"prompts_path: {prompts_path}",
                f"output_dir: {output_dir}",
                "max_turns: 10",
                "resume_strategy: reconstruct",
                "flush_policy: per_turn",
                "retries: 0",
                "timeout: 5",
                "truncation_policy: sliding_window",
                "abort_on_error: false",
                "input_compatibility_mode: false",
                "max_history_messages: 20",
                "require_approved_prompts: false",
                "require_approved_dialogues: false",
                "llm3:",
                "  provider: deepseek",
                "  model: deepseek-chat",
                "  api_key_env: NON_EXISTENT_TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.7",
                "  top_p: 1.0",
                "llm4:",
                "  provider: deepseek",
                "  model: deepseek-chat",
                "  api_key_env: NON_EXISTENT_TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.0",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    result = run_experiment(config=config, config_path=str(config_path), dry_run=True)

    assert Path(result["results_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert result["summary"]["actual_rows"] == 6
    assert result["summary"]["generate_errors"] == 6
    assert result["summary"]["refusal_rate"] == 0.0
    assert result["summary"]["dry_run"] is True
    assert result["summary"]["validation_log_file"] is not None
    assert (output_dir / result["summary"]["validation_log_file"]).exists()


def test_abort_on_error_still_writes_summary(tmp_path):
    dialogues_path = tmp_path / "dialogues.jsonl"
    prompts_path = tmp_path / "prompts.json"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"

    dialogues_path.write_text(
        '{"dialogue_id":"D1","domain":"creative","turns":[{"role":"user","text":"hello"}]}\n',
        encoding="utf-8",
    )
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {"default": "d", "evil": "e", "distant": "x"},
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                f"dialogues_path: {dialogues_path}",
                f"prompts_path: {prompts_path}",
                f"output_dir: {output_dir}",
                "max_turns: 10",
                "resume_strategy: reconstruct",
                "flush_policy: per_turn",
                "retries: 0",
                "timeout: 5",
                "truncation_policy: sliding_window",
                "abort_on_error: true",
                "input_compatibility_mode: false",
                "max_history_messages: 20",
                "require_approved_prompts: false",
                "require_approved_dialogues: false",
                "llm3:",
                "  provider: deepseek",
                "  model: deepseek-chat",
                "  api_key_env: NON_EXISTENT_TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.7",
                "  top_p: 1.0",
                "llm4:",
                "  provider: deepseek",
                "  model: deepseek-chat",
                "  api_key_env: NON_EXISTENT_TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.0",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    run_id = "abort_test"
    try:
        run_experiment(config=config, config_path=str(config_path), dry_run=False, run_id=run_id)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError when abort_on_error=true")

    summary_path = output_dir / f"run_summary_{run_id}.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["aborted"] is True
    assert summary["dry_run"] is False
    assert summary["validation_log_file"] is None


def test_failed_generate_turn_not_added_to_history(tmp_path, monkeypatch):
    dialogues_path = tmp_path / "dialogues.jsonl"
    prompts_path = tmp_path / "prompts.json"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"

    dialogues_path.write_text(
        (
            '{"dialogue_id":"D1","domain":"creative","turns":['
            '{"role":"user","text":"turn one"},'
            '{"role":"user","text":"turn two"}'
            "]}\n"
        ),
        encoding="utf-8",
    )
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {"default": "default", "evil": "evil", "distant": "distant"},
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
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                f"dialogues_path: {dialogues_path}",
                f"prompts_path: {prompts_path}",
                f"output_dir: {output_dir}",
                "max_turns: 2",
                "resume_strategy: reconstruct",
                "flush_policy: per_turn",
                "retries: 0",
                "timeout: 5",
                "truncation_policy: sliding_window",
                "abort_on_error: false",
                "input_compatibility_mode: false",
                "max_history_messages: 20",
                "require_approved_prompts: false",
                "require_approved_dialogues: false",
                "llm3:",
                "  provider: deepseek",
                "  model: llm3-test",
                "  api_key_env: DEEPSEEK_API_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.7",
                "  top_p: 1.0",
                "llm4:",
                "  provider: deepseek",
                "  model: llm4-test",
                "  api_key_env: DEEPSEEK_API_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.0",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    llm3_user_messages: list[list[str]] = []
    llm3_call_count = 0

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del timeout_seconds
        nonlocal llm3_call_count
        if self.config.model == "llm3-test":
            llm3_call_count += 1
            llm3_user_messages.append([m["content"] for m in messages if m["role"] == "user"])
            if llm3_call_count == 1:
                raise LLMError("forced generate failure")
            return ChatResult(text="safe response", latency_ms=12, raw={})
        return ChatResult(text='{"harm":1,"emotion":1,"anthro":1}', latency_ms=8, raw={})

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)

    config = load_config(config_path)
    run_experiment(config=config, config_path=str(config_path), dry_run=False, run_id="history_case")

    assert llm3_user_messages[0] == ["turn one"]
    assert llm3_user_messages[1] == ["turn two"]


def test_run_experiment_requires_approved_inputs_by_default(tmp_path):
    dialogues_path = tmp_path / "dialogues.jsonl"
    prompts_path = tmp_path / "prompts.json"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"

    dialogues_path.write_text(
        '{"dialogue_id":"D1","domain":"creative","turns":[{"role":"user","text":"hello"}]}\n',
        encoding="utf-8",
    )
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {"default": "default", "evil": "evil", "distant": "distant"},
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                f"dialogues_path: {dialogues_path}",
                f"prompts_path: {prompts_path}",
                f"output_dir: {output_dir}",
                "max_turns: 10",
                "resume_strategy: reconstruct",
                "flush_policy: per_turn",
                "retries: 0",
                "timeout: 5",
                "truncation_policy: sliding_window",
                "abort_on_error: false",
                "input_compatibility_mode: false",
                "max_history_messages: 20",
                "llm3:",
                "  provider: deepseek",
                "  model: deepseek-chat",
                "  api_key_env: NON_EXISTENT_TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.7",
                "  top_p: 1.0",
                "llm4:",
                "  provider: deepseek",
                "  model: deepseek-chat",
                "  api_key_env: NON_EXISTENT_TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.0",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    try:
        run_experiment(config=config, config_path=str(config_path), dry_run=True)
    except RuntimeError as exc:
        assert "not an approved frozen version" in str(exc)
    else:
        raise AssertionError("Expected approval enforcement failure")


def test_run_experiment_accepts_approved_inputs(tmp_path):
    candidate_prompts = tmp_path / "prompts_candidate.json"
    candidate_dialogues = tmp_path / "dialogues_candidate.jsonl"
    frozen_index = tmp_path / "frozen_inputs" / "index.json"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"

    candidate_prompts.write_text(
        json.dumps(
            {
                "conditions": {"default": "default", "evil": "evil", "distant": "distant"},
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    candidate_dialogues.write_text(
        '{"dialogue_id":"D1","domain":"creative","turns":[{"role":"user","text":"hello"}]}\n',
        encoding="utf-8",
    )
    approve_candidate(
        index_path=frozen_index,
        kind="prompts",
        candidate_path=candidate_prompts,
        version="p1",
        reviewer="alice",
    )
    approve_candidate(
        index_path=frozen_index,
        kind="dialogues",
        candidate_path=candidate_dialogues,
        version="d1",
        reviewer="alice",
    )

    prompts_path = tmp_path / "frozen_inputs" / "prompts" / "prompts_p1.json"
    dialogues_path = tmp_path / "frozen_inputs" / "dialogues" / "dialogues_d1.jsonl"
    config_path.write_text(
        "\n".join(
            [
                f"dialogues_path: {dialogues_path}",
                f"prompts_path: {prompts_path}",
                f"output_dir: {output_dir}",
                f"frozen_index_path: {frozen_index}",
                "max_turns: 10",
                "resume_strategy: reconstruct",
                "flush_policy: per_turn",
                "retries: 0",
                "timeout: 5",
                "truncation_policy: sliding_window",
                "abort_on_error: false",
                "input_compatibility_mode: false",
                "max_history_messages: 20",
                "llm3:",
                "  provider: deepseek",
                "  model: deepseek-chat",
                "  api_key_env: NON_EXISTENT_TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.7",
                "  top_p: 1.0",
                "llm4:",
                "  provider: deepseek",
                "  model: deepseek-chat",
                "  api_key_env: NON_EXISTENT_TEST_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.0",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    result = run_experiment(config=config, config_path=str(config_path), dry_run=True, run_id="approved_case")
    assert result["summary"]["prompts_version"] == "p1"
    assert result["summary"]["dialogues_version"] == "d1"
    assert result["summary"]["prompts_source"] == "frozen"
    assert result["summary"]["dialogues_source"] == "frozen"


def test_judge_score_out_of_range_marked_as_judge_parse(tmp_path, monkeypatch):
    dialogues_path = tmp_path / "dialogues.jsonl"
    prompts_path = tmp_path / "prompts.json"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"

    dialogues_path.write_text(
        '{"dialogue_id":"D1","domain":"creative","turns":[{"role":"user","text":"hello"}]}\n',
        encoding="utf-8",
    )
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {"default": "default", "evil": "evil", "distant": "distant"},
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
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                f"dialogues_path: {dialogues_path}",
                f"prompts_path: {prompts_path}",
                f"output_dir: {output_dir}",
                "max_turns: 10",
                "resume_strategy: reconstruct",
                "flush_policy: per_turn",
                "retries: 0",
                "timeout: 5",
                "truncation_policy: sliding_window",
                "abort_on_error: false",
                "input_compatibility_mode: false",
                "max_history_messages: 20",
                "require_approved_prompts: false",
                "require_approved_dialogues: false",
                "llm3:",
                "  provider: deepseek",
                "  model: llm3-test",
                "  api_key_env: DEEPSEEK_API_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.7",
                "  top_p: 1.0",
                "llm4:",
                "  provider: deepseek",
                "  model: llm4-test",
                "  api_key_env: DEEPSEEK_API_KEY",
                "  base_url: https://api.deepseek.com/v1",
                "  temperature: 0.0",
                "  top_p: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del messages, timeout_seconds
        if self.config.model == "llm3-test":
            return ChatResult(text="safe response", latency_ms=11, raw={})
        return ChatResult(text='{"harm":7,"emotion":1,"anthro":1}', latency_ms=9, raw={})

    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    config = load_config(config_path)
    result = run_experiment(config=config, config_path=str(config_path), dry_run=False, run_id="judge_range")

    row = json.loads(Path(result["results_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert row["error_stage"] == "judge_parse"
    assert row["harm"] is None
    assert "out of range" in (row["error_message"] or "")
