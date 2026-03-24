from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (PROJECT_ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from extend_dialogue_natural import (
    _build_chunk_messages,
    _extract_json,
    _load_dialogue,
    _normalize_turns,
)
from backend.llm_clients import OpenAICompatibleChatClient
from backend.runtime_config import load_config


DEFAULT_DIALOGUE_IDS = [
    "creative_000",
    "finance_013",
    "mental_health_026",
    "medicine_039",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend one dialogue per domain naturally to 200 turns."
    )
    parser.add_argument("--config", required=True, help="Path to yaml config with llm2 settings.")
    parser.add_argument("--source_dialogues", required=True, help="Path to source dialogues jsonl.")
    parser.add_argument(
        "--dialogue_ids",
        nargs="*",
        default=DEFAULT_DIALOGUE_IDS,
        help="Dialogue ids to extend. Defaults to one per domain.",
    )
    parser.add_argument("--target_turns", type=int, default=200)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--context_turns", type=int, default=24)
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    return parser.parse_args()


def _extend_one(
    *,
    llm2_client: OpenAICompatibleChatClient,
    timeout_seconds: float,
    source_path: Path,
    dialogue_id: str,
    target_turns: int,
    chunk_size: int,
    context_turns: int,
) -> dict:
    item = _load_dialogue(source_path, dialogue_id)
    turns = list(item["turns"])
    existing_turns = len(turns)
    if existing_turns >= target_turns:
        raise ValueError(
            f"Source dialogue {dialogue_id} already has {existing_turns} turns; target_turns must be larger."
        )

    for chunk_start in range(existing_turns + 1, target_turns + 1, chunk_size):
        chunk_end = min(target_turns, chunk_start + chunk_size - 1)
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                messages = _build_chunk_messages(
                    domain=item["domain"],
                    dialogue_id=item["dialogue_id"],
                    target_turns=target_turns,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    all_prior_turns=turns,
                    context_turns=context_turns,
                )
                result = llm2_client.chat(messages, timeout_seconds)
                payload = _extract_json(result.text)
                generated = _normalize_turns(payload, chunk_end - chunk_start + 1)
                turns.extend(generated)
                print(
                    f"{dialogue_id}: generated turns {chunk_start}-{chunk_end} "
                    f"latency_ms={result.latency_ms} total_turns={len(turns)}"
                )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                preview = str(exc).replace("\n", " ")[:200]
                print(
                    f"{dialogue_id}: chunk {chunk_start}-{chunk_end} "
                    f"attempt {attempt}/4 failed: {preview}"
                )
                if attempt < 4:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
        if last_error is not None:
            raise last_error

    return {
        "dialogue_id": f"{item['dialogue_id']}_natural_{target_turns}turn",
        "domain": item["domain"],
        "turns": turns,
    }


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    llm2_client = OpenAICompatibleChatClient(config.llm2)

    source_path = Path(args.source_dialogues)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text("", encoding="utf-8")
    items = []
    for dialogue_id in args.dialogue_ids:
        item = _extend_one(
                llm2_client=llm2_client,
                timeout_seconds=config.timeout_seconds,
                source_path=source_path,
                dialogue_id=dialogue_id,
                target_turns=args.target_turns,
                chunk_size=args.chunk_size,
                context_turns=args.context_turns,
            )
        items.append(item)
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"saved {item['dialogue_id']} to {output_path}")
    print(f"wrote {output_path} count={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
