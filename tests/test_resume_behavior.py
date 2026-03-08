from pathlib import Path

from backend.orchestrator import run_experiment
from backend.resume import load_resume_state
from backend.runtime_config import load_config


def _write_minimal_inputs(tmp_path, resume_strategy: str) -> Path:
    dialogues_path = tmp_path / "dialogues.jsonl"
    prompts_path = tmp_path / "prompts.json"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"

    dialogues_path.write_text(
        '{"dialogue_id":"D1","domain":"creative","turns":[{"role":"user","text":"hello"}]}\n',
        encoding="utf-8",
    )
    prompts_path.write_text(
        (
            '{"conditions":{"default":"d","unhelpful":"e","cynical":"c","distant":"x"},'
            '"judge_system":"j","judge_rubric":"r","judge_schema":{"type":"object"}}'
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
                f"resume_strategy: {resume_strategy}",
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
    return config_path


def test_resume_skip_avoids_duplicate_rows(tmp_path):
    config_path = _write_minimal_inputs(tmp_path, resume_strategy="skip")
    config = load_config(config_path)
    run_id = "resume_skip_test"

    first = run_experiment(config=config, config_path=str(config_path), dry_run=False, run_id=run_id)
    second = run_experiment(config=config, config_path=str(config_path), dry_run=False, run_id=run_id)

    assert first["summary"]["actual_rows"] == 3
    assert second["summary"]["actual_rows"] == 3
    assert second["summary"]["new_rows_written"] == 0


def test_load_resume_state_reconstructs_processed_turns(tmp_path):
    results_path = tmp_path / "results_x.jsonl"
    results_path.write_text(
        "\n".join(
            [
                (
                    '{"dialogue_id":"D1","condition":"default","turn_index":1,'
                    '"error_stage":null,"user_text":"u1","model_reply":"a1"}'
                ),
                (
                    '{"dialogue_id":"D1","condition":"default","turn_index":2,'
                    '"error_stage":"generate","user_text":"u2","model_reply":""}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = load_resume_state(results_path)
    combo = state.combo_states[("D1", "default")]

    assert state.existing_rows == 2
    assert combo.processed_turns == {1, 2}
    assert combo.history == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
