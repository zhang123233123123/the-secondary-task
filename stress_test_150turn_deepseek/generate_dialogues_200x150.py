"""Directly generate 200 natural dialogues of 150 turns each, 50 per domain, in parallel."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.llm_clients import OpenAICompatibleChatClient  # noqa: E402
from backend.prepare_orchestrator import _llm2_generate_dialogue  # noqa: E402
from backend.runtime_config import load_config  # noqa: E402

DOMAINS = ["creative", "finance", "mental_health", "medicine"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 200 natural dialogues of 150 turns, 50 per domain."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--total", type=int, default=200, help="Total dialogues to generate.")
    parser.add_argument("--target_turns", type=int, default=150)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    per_domain = args.total // len(DOMAINS)
    tasks = []
    idx = 1
    for domain in DOMAINS:
        for _ in range(per_domain):
            tasks.append((idx, domain))
            idx += 1

    total = len(tasks)
    print(f"生成 {total} 条对话 ({per_domain}/domain × {len(DOMAINS)} domains) "
          f"× {args.target_turns} 轮, workers={args.workers}")

    write_lock = threading.Lock()
    success_count = 0
    failed: list[tuple[int, str]] = []

    def worker(dialogue_index: int, domain: str):
        llm2_client = OpenAICompatibleChatClient(config.llm2)
        item, latency_ms = _llm2_generate_dialogue(
            llm2_client=llm2_client,
            config=config,
            dialogue_index=dialogue_index,
            dialogue_count=total,
            domain=domain,
            target_turns=args.target_turns,
        )
        item["dialogue_id"] = f"dlg_{dialogue_index:05d}"
        item["domain"] = domain
        return item, latency_ms

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, i, d): (i, d) for i, d in tasks}
        for future in as_completed(futures):
            idx_d, domain = futures[future]
            try:
                item, latency_ms = future.result()
                turns = len(item.get("turns", []))
                with write_lock:
                    with output_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    success_count += 1
                print(f"[{success_count:3d}/{total}] {item['dialogue_id']} | {domain} | "
                      f"turns={turns} | latency={latency_ms}ms")
            except Exception as exc:  # noqa: BLE001
                failed.append((idx_d, domain))
                print(f"[FAILED] dlg_{idx_d:05d} ({domain}): {str(exc)[:150]}")

    print(f"\n{'='*60}")
    print(f"完成: {success_count}/{total} 条写入 {output_path}")
    if failed:
        print(f"失败 {len(failed)} 条: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
