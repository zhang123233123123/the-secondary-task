import json

import pytest

from backend.input_loader import load_dialogues, load_prompts


def test_load_dialogues_supports_compat_mode(tmp_path):
    dialogues_path = tmp_path / "dialogues.jsonl"
    dialogues_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dialogue_id": "A1",
                        "domain": "creative",
                        "turns": [{"role": "user", "text": "hello"}],
                    }
                ),
                json.dumps(
                    {
                        "dialogue_id": "A2",
                        "domain": "finance",
                        "turns": ["q1", "q2"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    dialogues = load_dialogues(dialogues_path, compatibility_mode=True)

    assert len(dialogues) == 2
    assert dialogues[0].input_schema_variant == "standard"
    assert dialogues[1].input_schema_variant == "compat_string_list"
    assert dialogues[1].turns[0].role == "user"


def test_load_dialogues_rejects_duplicate_id(tmp_path):
    dialogues_path = tmp_path / "dialogues.jsonl"
    line = json.dumps(
        {
            "dialogue_id": "A1",
            "domain": "creative",
            "turns": [{"role": "user", "text": "hello"}],
        }
    )
    dialogues_path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate dialogue_id"):
        load_dialogues(dialogues_path, compatibility_mode=False)


def test_load_prompts_requires_all_conditions(tmp_path):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(
        json.dumps(
            {
                "conditions": {"default": "ok", "unhelpful": "ok"},
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conditions.cynical"):
        load_prompts(prompts_path)
