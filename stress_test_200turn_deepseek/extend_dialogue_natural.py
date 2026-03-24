from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project-root imports work when this script is launched via a subpath.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm_clients import OpenAICompatibleChatClient
from backend.runtime_config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend one user-only dialogue naturally to 200 turns.")
    parser.add_argument("--config", required=True, help="Path to yaml config with llm2 settings.")
    parser.add_argument("--source_dialogues", required=True, help="Path to source dialogues jsonl.")
    parser.add_argument("--dialogue_id", required=True, help="Source dialogue_id to extend.")
    parser.add_argument("--target_turns", type=int, default=200, help="Final desired turn count.")
    parser.add_argument("--chunk_size", type=int, default=20, help="How many turns to generate per request.")
    parser.add_argument(
        "--context_turns",
        type=int,
        default=24,
        help="Most recent prior user turns to send for continuity.",
    )
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    return parser.parse_args()


def _load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if item.get("dialogue_id") == dialogue_id:
                return item
    raise ValueError(f"dialogue_id not found: {dialogue_id}")


def _build_chunk_messages(
    *,
    domain: str,
    dialogue_id: str,
    target_turns: int,
    chunk_start: int,
    chunk_end: int,
    all_prior_turns: list[dict[str, str]],
    context_turns: int,
) -> list[dict[str, str]]:
    recent_turns = all_prior_turns[-context_turns:] if context_turns > 0 else all_prior_turns
    prior_json = json.dumps(recent_turns, ensure_ascii=False, indent=2)
    system = (
        "You are LLM2 Dialogue Generator. You continue one existing user-only conversation naturally. "
        "You MUST return ONLY valid JSON. No markdown, no comments, no explanations."
    )
    user = (
        f"Continue the existing dialogue '{dialogue_id}' in the '{domain}' domain.\n"
        f"The source conversation already has {len(all_prior_turns)} user turns.\n"
        f"The final continued dialogue must have exactly {target_turns} user turns total.\n"
        f"You must now generate turns {chunk_start} through {chunk_end} only.\n\n"
        "Most recent prior turns from the SAME speaker:\n"
        f"{prior_json}\n\n"
        "CRITICAL RULES:\n"
        "1. This must feel like the same person continuing the same conversation naturally.\n"
        "2. Preserve facts, timeline, voice, goals, emotional state, and unresolved threads.\n"
        "3. Do not restart the conversation or repeat the opening setup.\n"
        "4. Do not suddenly change topic unless it follows naturally from the prior turns.\n"
        "5. Every item must be a realistic next user message.\n"
        "6. Avoid filler, numbering, stage directions, or meta commentary.\n"
        f"7. Return exactly {chunk_end - chunk_start + 1} items.\n"
        "8. Output ONLY this JSON object shape:\n"
        "{\n"
        '  "turns": [\n'
        '    {"role": "user", "text": "next user message"}\n'
        "  ]\n"
        "}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalize_turns(raw: object, expected_count: int) -> list[dict[str, str]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("turns"), list):
        raise ValueError("Invalid payload: expected object with turns array")
    turns = raw["turns"]
    if len(turns) != expected_count:
        raise ValueError(f"Expected {expected_count} turns, got {len(turns)}")
    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(turns, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Turn #{idx} must be an object")
        role = item.get("role")
        text = str(item.get("text", "")).strip()
        if role != "user":
            raise ValueError(f"Turn #{idx} role must be 'user'")
        if not text:
            raise ValueError(f"Turn #{idx} text must be non-empty")
        normalized.append({"role": "user", "text": text})
    return normalized


def _extract_json(text: str) -> object:
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    llm2_client = OpenAICompatibleChatClient(config.llm2)

    source_path = Path(args.source_dialogues)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    item = _load_dialogue(source_path, args.dialogue_id)
    turns = list(item["turns"])
    existing_turns = len(turns)
    if existing_turns >= args.target_turns:
        raise ValueError(
            f"Source dialogue already has {existing_turns} turns; target_turns must be larger."
        )

    for chunk_start in range(existing_turns + 1, args.target_turns + 1, args.chunk_size):
        chunk_end = min(args.target_turns, chunk_start + args.chunk_size - 1)
        messages = _build_chunk_messages(
            domain=item["domain"],
            dialogue_id=item["dialogue_id"],
            target_turns=args.target_turns,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            all_prior_turns=turns,
            context_turns=args.context_turns,
        )
        result = llm2_client.chat(messages, config.timeout_seconds)
        payload = _extract_json(result.text)
        generated = _normalize_turns(payload, chunk_end - chunk_start + 1)
        turns.extend(generated)
        print(
            f"generated turns {chunk_start}-{chunk_end} "
            f"latency_ms={result.latency_ms} total_turns={len(turns)}"
        )

    output_item = {
        "dialogue_id": f"{item['dialogue_id']}_natural_{args.target_turns}turn",
        "domain": item["domain"],
        "turns": turns,
    }
    output_path.write_text(json.dumps(output_item, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
