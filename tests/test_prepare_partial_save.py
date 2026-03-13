import json

from backend.llm_clients import ChatResult
from backend.prepare_orchestrator import prepare_inputs
from backend.runtime_config import load_config


def test_prepare_inputs_saves_partial_results_on_max_attempts(tmp_path, monkeypatch):
    """Test that partial results are saved even when max_generation_attempts is reached."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frozen_index_path: frozen_inputs/index.json",
                "prepare_dialogue_count: 10",
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
    
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(
        json.dumps(
            {
                "conditions": {"default": "d", "unhelpful": "u", "cynical": "c", "distant": "x"},
                "judge_system": "judge",
                "judge_rubric": "score 1-10",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    
    attempts = {"count": 0}
    
    def fake_chat(self, messages, timeout_seconds):  # noqa: ANN001
        del messages, timeout_seconds, self
        attempts["count"] += 1
        # Succeed only 3 times, then always fail
        if attempts["count"] <= 3:
            return ChatResult(
                text=json.dumps({
                    "dialogue_id": f"D{attempts['count']}",
                    "domain": "creative",
                    "turns": [{"role": "user", "text": f"msg{attempts['count']}"}],
                }),
                latency_ms=5,
                raw={},
            )
        # After 3 successes, always return bad JSON to trigger max_attempts
        return ChatResult(
            text='{"bad": json syntax',
            latency_ms=5,
            raw={},
        )
    
    monkeypatch.setattr("backend.llm_clients.OpenAICompatibleChatClient.chat", fake_chat)
    monkeypatch.setattr("backend.prepare_orchestrator.time.sleep", lambda _: None)
    monkeypatch.setenv("TEST_KEY", "sk-test-123")
    
    config = load_config(config_path)
    
    # Should raise error but save partial results
    try:
        manifest = prepare_inputs(
            config=config,
            config_path=str(config_path),
            target_version="vprep_partial",
            skip_llm1=True,
        )
    except ValueError as exc:
        # Expected to fail
        assert "could not reach target dialogue count" in str(exc).lower()
        
        # But manifest should exist with partial results
        candidates_dir = tmp_path / "frozen_inputs" / "candidates"
        manifest_path = candidates_dir / "prepare_manifest_vprep_partial.json"
        assert manifest_path.exists(), "Manifest should be saved even on failure"
        
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "partial"
        assert manifest["prepare_dialogue_count_actual"] == 3
        assert manifest["prepare_dialogue_count"] == 10
        assert manifest["generation_error"] is not None
        
        # Check that dialogues file exists with 3 entries
        dialogues_path = candidates_dir / "dialogues_candidate_vprep_partial.jsonl"
        assert dialogues_path.exists()
        lines = dialogues_path.read_text().strip().split("\n")
        assert len(lines) == 3
        return
    
    raise AssertionError("Expected ValueError to be raised")
