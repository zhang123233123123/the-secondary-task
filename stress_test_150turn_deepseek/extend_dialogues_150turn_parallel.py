"""Extend 52x100-turn dialogues to 150 turns using DeepSeek, with parallel workers."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTEND_DIR = PROJECT_ROOT / "stress_test_200turn_deepseek"

for p in (PROJECT_ROOT, EXTEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from extend_dialogue_natural import (  # noqa: E402
    _build_chunk_messages,
    _extract_json,
    _load_dialogue,
    _normalize_turns,
)
from backend.llm_clients import OpenAICompatibleChatClient  # noqa: E402
from backend.runtime_config import load_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend 52x100-turn dialogues to 150 turns with parallel workers."
    )
    parser.add_argument("--config", required=True, help="Path to yaml config with llm2 settings.")
    parser.add_argument("--source_dialogues", required=True, help="Path to source dialogues jsonl.")
    parser.add_argument(
        "--dialogue_ids",
        nargs="*",
        default=None,
        help="Dialogue ids to extend. Defaults to all ids in source file.",
    )
    parser.add_argument("--target_turns", type=int, default=150)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--context_turns", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads.")
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    return parser.parse_args()


def _extend_one(
    *,
    llm2_config,
    timeout_seconds: float,
    source_path: Path,
    dialogue_id: str,
    target_turns: int,
    chunk_size: int,
    context_turns: int,
) -> dict:
    llm2_client = OpenAICompatibleChatClient(llm2_config)
    item = _load_dialogue(source_path, dialogue_id)
    turns = list(item["turns"])
    existing_turns = len(turns)
    if existing_turns >= target_turns:
        raise ValueError(
            f"{dialogue_id}: already has {existing_turns} turns; target_turns must be larger."
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
                    f"[{dialogue_id}] turns {chunk_start}-{chunk_end} done "
                    f"(latency={result.latency_ms}ms, total={len(turns)})"
                )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                preview = str(exc).replace("\n", " ")[:200]
                print(f"[{dialogue_id}] chunk {chunk_start}-{chunk_end} attempt {attempt}/4 failed: {preview}")
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

    source_path = Path(args.source_dialogues)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all dialogue ids from source if not specified
    if args.dialogue_ids:
        dialogue_ids = args.dialogue_ids
    else:
        with source_path.open(encoding="utf-8") as fh:
            dialogue_ids = [json.loads(l)["dialogue_id"] for l in fh if l.strip()]

    print(f"Extending {len(dialogue_ids)} dialogues to {args.target_turns} turns "
          f"with {args.workers} workers -> {output_path}")

    output_path.write_text("", encoding="utf-8")
    write_lock = threading.Lock()
    success_count = 0
    failed_ids: list[str] = []

    def worker(dialogue_id: str) -> tuple[str, dict | Exception]:
        try:
            item = _extend_one(
                llm2_config=config.llm2,
                timeout_seconds=config.timeout_seconds,
                source_path=source_path,
                dialogue_id=dialogue_id,
                target_turns=args.target_turns,
                chunk_size=args.chunk_size,
                context_turns=args.context_turns,
            )
            with write_lock:
                with output_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            return dialogue_id, item
        except Exception as exc:  # noqa: BLE001
            return dialogue_id, exc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, did): did for did in dialogue_ids}
        for future in as_completed(futures):
            did, result = future.result()
            if isinstance(result, Exception):
                print(f"[FAILED] {did}: {result}")
                failed_ids.append(did)
            else:
                success_count += 1
                print(f"[SAVED] {did} -> {result['dialogue_id']} (done={success_count}/{len(dialogue_ids)})")

    print(f"\n{'='*60}")
    print(f"完成: {success_count}/{len(dialogue_ids)} 条成功写入 {output_path}")
    if failed_ids:
        print(f"失败 {len(failed_ids)} 条: {failed_ids}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
