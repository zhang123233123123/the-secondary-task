from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_DOMAINS = {"creative", "finance", "mental_health", "medicine"}
REQUIRED_CONDITIONS = ("default", "unhelpful", "cynical", "distant")


@dataclass
class DialogueTurn:
    role: str
    text: str


@dataclass
class Dialogue:
    dialogue_id: str
    domain: str
    turns: list[DialogueTurn]
    input_schema_variant: str


@dataclass
class PromptsBundle:
    conditions: dict[str, str]
    judge_system: str
    judge_rubric: str
    judge_schema: dict[str, Any]


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_turns(raw_turns: Any, compatibility_mode: bool) -> tuple[list[DialogueTurn], str]:
    if isinstance(raw_turns, list) and all(isinstance(item, dict) for item in raw_turns):
        turns: list[DialogueTurn] = []
        for item in raw_turns:
            role = str(item.get("role", ""))
            text = str(item.get("text", ""))
            if role == "user":
                turns.append(DialogueTurn(role="user", text=text))
        return turns, "standard"

    if compatibility_mode and isinstance(raw_turns, list) and all(
        isinstance(item, str) for item in raw_turns
    ):
        turns = [DialogueTurn(role="user", text=item) for item in raw_turns]
        return turns, "compat_string_list"

    raise ValueError("Invalid turns format")


def load_dialogues(path: str | Path, compatibility_mode: bool = False) -> list[Dialogue]:
    path_obj = Path(path)
    dialogues: list[Dialogue] = []
    seen_ids: set[str] = set()

    with path_obj.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            dialogue_id = str(raw.get("dialogue_id", "")).strip()
            domain = str(raw.get("domain", "")).strip()
            if not dialogue_id:
                raise ValueError(f"Missing dialogue_id at line {line_no}")
            if dialogue_id in seen_ids:
                raise ValueError(f"Duplicate dialogue_id: {dialogue_id}")
            if domain not in VALID_DOMAINS:
                raise ValueError(f"Invalid domain at line {line_no}: {domain}")

            turns, schema_variant = _coerce_turns(
                raw.get("turns"),
                compatibility_mode=compatibility_mode,
            )

            seen_ids.add(dialogue_id)
            dialogues.append(
                Dialogue(
                    dialogue_id=dialogue_id,
                    domain=domain,
                    turns=turns,
                    input_schema_variant=schema_variant,
                )
            )

    return dialogues


def load_prompts(path: str | Path) -> PromptsBundle:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    conditions = raw.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("prompts.json.conditions must be an object")

    normalized: dict[str, str] = {}
    for key in REQUIRED_CONDITIONS:
        value = conditions.get(key)
        if not isinstance(value, str):
            raise ValueError(f"prompts.json.conditions.{key} must be a string")
        # Allow default to be empty (no system prompt); others must be non-empty
        if key != "default" and not value.strip():
            raise ValueError(f"prompts.json.conditions.{key} must be a non-empty string")
        normalized[key] = value

    judge_system = raw.get("judge_system")
    judge_rubric = raw.get("judge_rubric")
    judge_schema = raw.get("judge_schema")
    if not isinstance(judge_system, str) or not judge_system.strip():
        raise ValueError("prompts.json.judge_system must be a non-empty string")
    if not isinstance(judge_rubric, str) or not judge_rubric.strip():
        raise ValueError("prompts.json.judge_rubric must be a non-empty string")
    if not isinstance(judge_schema, dict):
        raise ValueError("prompts.json.judge_schema must be an object")

    return PromptsBundle(
        conditions=normalized,
        judge_system=judge_system,
        judge_rubric=judge_rubric,
        judge_schema=judge_schema,
    )
