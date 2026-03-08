import json

from backend.frozen_registry import (
    apply_versions_to_config,
    approve_candidate,
    find_approved_version_for_file,
    load_frozen_index,
    resolve_frozen_file,
    set_active_versions,
)


def test_approve_and_activate_frozen_versions(tmp_path):
    index_path = tmp_path / "frozen_inputs" / "index.json"
    prompts_candidate = tmp_path / "prompts_candidate.json"
    dialogues_candidate = tmp_path / "dialogues_candidate.jsonl"

    prompts_candidate.write_text(
        json.dumps(
            {
                "conditions": {"default": "d", "unhelpful": "e", "cynical": "c", "distant": "x"},
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    dialogues_candidate.write_text(
        '{"dialogue_id":"D1","domain":"creative","turns":[{"role":"user","text":"hello"}]}\n',
        encoding="utf-8",
    )

    prompt_entry = approve_candidate(
        index_path=index_path,
        kind="prompts",
        candidate_path=prompts_candidate,
        version="v1",
        reviewer="alice",
    )
    dialogue_entry = approve_candidate(
        index_path=index_path,
        kind="dialogues",
        candidate_path=dialogues_candidate,
        version="v1",
        reviewer="alice",
    )

    active = set_active_versions(index_path=index_path, prompts_version="v1", dialogues_version="v1")
    assert active["prompts_version"] == "v1"
    assert active["dialogues_version"] == "v1"

    resolved_prompts = resolve_frozen_file(index_path=index_path, kind="prompts", version="v1")
    resolved_dialogues = resolve_frozen_file(index_path=index_path, kind="dialogues", version="v1")
    assert resolved_prompts.exists()
    assert resolved_dialogues.exists()

    assert find_approved_version_for_file(
        index_path=index_path,
        kind="prompts",
        file_path=resolved_prompts,
    ) == "v1"
    assert find_approved_version_for_file(
        index_path=index_path,
        kind="dialogues",
        file_path=resolved_dialogues,
    ) == "v1"

    index = load_frozen_index(index_path)
    assert len(index["prompts_versions"]) == 1
    assert len(index["dialogues_versions"]) == 1
    assert prompt_entry["version"] == "v1"
    assert dialogue_entry["version"] == "v1"


def test_apply_versions_to_config_updates_paths(tmp_path):
    index_path = tmp_path / "frozen_inputs" / "index.json"
    prompts_candidate = tmp_path / "prompts_candidate.json"
    dialogues_candidate = tmp_path / "dialogues_candidate.jsonl"
    config_path = tmp_path / "config.yaml"

    prompts_candidate.write_text(
        json.dumps(
            {
                "conditions": {"default": "d", "unhelpful": "e", "cynical": "c", "distant": "x"},
                "judge_system": "judge",
                "judge_rubric": "rubric",
                "judge_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    dialogues_candidate.write_text(
        '{"dialogue_id":"D1","domain":"creative","turns":[{"role":"user","text":"hello"}]}\n',
        encoding="utf-8",
    )
    config_path.write_text("prompts_path: prompts.json\ndialogues_path: dialogues.jsonl\n", encoding="utf-8")

    approve_candidate(
        index_path=index_path,
        kind="prompts",
        candidate_path=prompts_candidate,
        version="p1",
        reviewer="alice",
    )
    approve_candidate(
        index_path=index_path,
        kind="dialogues",
        candidate_path=dialogues_candidate,
        version="d1",
        reviewer="alice",
    )
    updated = apply_versions_to_config(
        config_path=config_path,
        index_path=index_path,
        prompts_version="p1",
        dialogues_version="d1",
    )
    config_text = config_path.read_text(encoding="utf-8")
    assert "require_approved_prompts: true" in config_text.lower()
    assert "require_approved_dialogues: true" in config_text.lower()
    assert "frozen_inputs/prompts/prompts_p1.json" in updated["prompts_path"]
    assert "frozen_inputs/dialogues/dialogues_d1.jsonl" in updated["dialogues_path"]
