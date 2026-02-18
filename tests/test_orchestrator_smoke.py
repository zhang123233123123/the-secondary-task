import json
from pathlib import Path

from orchestrator import run_experiment
from runtime_config import load_config


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
    assert result["summary"]["actual_rows"] == 6
    assert result["summary"]["generate_errors"] == 6
