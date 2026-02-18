from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ComboState:
    processed_turns: set[int] = field(default_factory=set)
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ResumeState:
    existing_rows: int
    combo_states: dict[tuple[str, str], ComboState]
    error_rows: int
    generate_errors: int
    judge_errors: int
    judge_parse_errors: int
    refusal_count: int
    truncated_count: int


def load_resume_state(results_path: str | Path) -> ResumeState:
    path = Path(results_path)
    if not path.exists():
        return ResumeState(
            existing_rows=0,
            combo_states={},
            error_rows=0,
            generate_errors=0,
            judge_errors=0,
            judge_parse_errors=0,
            refusal_count=0,
            truncated_count=0,
        )

    combo_states: dict[tuple[str, str], ComboState] = {}
    rows = 0
    error_rows = 0
    generate_errors = 0
    judge_errors = 0
    judge_parse_errors = 0
    refusal_count = 0
    truncated_count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows += 1
            try:
                row: dict[str, Any] = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            stage = row.get("error_stage")
            if stage is not None:
                error_rows += 1
                if stage == "generate":
                    generate_errors += 1
                elif stage == "judge":
                    judge_errors += 1
                elif stage == "judge_parse":
                    judge_parse_errors += 1
            if bool(row.get("refusal_detected")):
                refusal_count += 1
            if bool(row.get("context_truncated")):
                truncated_count += 1

            dialogue_id = str(row.get("dialogue_id", ""))
            condition = str(row.get("condition", ""))
            if not dialogue_id or not condition:
                continue

            key = (dialogue_id, condition)
            state = combo_states.setdefault(key, ComboState())
            turn_index = row.get("turn_index")
            if isinstance(turn_index, int) and turn_index > 0:
                state.processed_turns.add(turn_index)

            # Reconstruct in-memory history using successful generation turns only.
            if row.get("error_stage") != "generate":
                user_text = row.get("user_text")
                model_reply = row.get("model_reply")
                if isinstance(user_text, str):
                    state.history.append({"role": "user", "content": user_text})
                if isinstance(model_reply, str) and model_reply:
                    state.history.append({"role": "assistant", "content": model_reply})

    return ResumeState(
        existing_rows=rows,
        combo_states=combo_states,
        error_rows=error_rows,
        generate_errors=generate_errors,
        judge_errors=judge_errors,
        judge_parse_errors=judge_parse_errors,
        refusal_count=refusal_count,
        truncated_count=truncated_count,
    )
