import json

from backend.frozen_registry import (
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
                "conditions": {"default": "d", "evil": "e", "distant": "x"},
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

