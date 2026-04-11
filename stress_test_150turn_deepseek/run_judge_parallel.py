"""Stage 2: Judge model replies (LLM4) in parallel.

Reads Stage 1 JSONL (model_reply populated, scores null), outputs a new JSONL
with all score fields filled in. Rows with error_stage=="generate" are passed
through unchanged. Already-scored rows are also passed through (resume support).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.input_loader import compute_sha256, load_prompts  # noqa: E402
from backend.llm_clients import make_llm_client, parse_judge_json  # noqa: E402
from backend.runtime_config import load_config  # noqa: E402

JUDGE_KEYS_1_TO_10 = (
    "harm_1_10",
    "negative_emotion_1_10",
    "inappropriate_1_10",
    "empathic_language_1_10",
)
JUDGE_KEYS_1_TO_5 = (
    "anthro_q1",
    "anthro_q2",
    "anthro_q3",
    "anthro_q4",
    "anthro_q5",
)
ALL_JUDGE_KEYS = (*JUDGE_KEYS_1_TO_10, *JUDGE_KEYS_1_TO_5)


def _build_run_id() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"judge_{ts}_{uuid.uuid4().hex[:6]}"


def _timestamp_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _judge_messages(prompts, model_reply: str, *, user_text: str) -> list[dict[str, str]]:
    user_content = (
        f"User text:\n{user_text}\n\n"
        f"Rubric:\n{prompts.judge_rubric}\n\n"
        f"Model reply to evaluate:\n{model_reply}\n\n"
        f"Return only JSON with keys: {', '.join(ALL_JUDGE_KEYS)}."
    )
    return [
        {"role": "system", "content": prompts.judge_system},
        {"role": "user", "content": user_content},
    ]


def _validate_score_range(values: dict[str, int]) -> None:
    for key in JUDGE_KEYS_1_TO_10:
        v = values[key]
        if v < 1 or v > 10:
            raise ValueError(f"{key} out of range: {v}")
    for key in JUDGE_KEYS_1_TO_5:
        v = values[key]
        if v < 1 or v > 5:
            raise ValueError(f"{key} out of range: {v}")


def _call_with_retry(call, retries: int, backoff: float = 0.5):
    last_error: Exception | None = None
    for i in range(retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if i < retries:
                time.sleep(backoff * (2 ** i))
    raise last_error  # type: ignore[misc]


def _judge_row(
    row: dict[str, Any],
    *,
    prompts,
    llm4_config,
    config,
    judge_run_id: str,
) -> dict[str, Any]:
    """Call LLM4 for a single row. Returns updated row dict."""
    messages = _judge_messages(prompts, row["model_reply"], user_text=row["user_text"])

    judge_latency_ms: int | None = None
    judge_raw: str | None = None
    error_stage: str | None = row.get("error_stage")
    error_message: str | None = row.get("error_message")
    scores: dict[str, int | None] = {k: None for k in ALL_JUDGE_KEYS}

    client = make_llm_client(llm4_config)
    try:
        result = _call_with_retry(
            lambda: client.chat(messages, config.timeout_seconds),
            retries=config.retries,
        )
        judge_latency_ms = result.latency_ms
        judge_raw = result.text
        parsed = parse_judge_json(judge_raw)
        int_scores = {k: int(parsed[k]) for k in ALL_JUDGE_KEYS}
        _validate_score_range(int_scores)
        scores = int_scores  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        error_stage = "judge"
        error_message = str(exc)[:500]

    out = dict(row)
    out.update(scores)
    out["judge_latency_ms"] = judge_latency_ms
    out["judge_raw"] = judge_raw
    out["error_stage"] = error_stage
    out["error_message"] = error_message
    out["judge_run_id"] = judge_run_id
    out["judge_timestamp_utc"] = _timestamp_utc()
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: judge model replies in parallel.")
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True, help="Stage 1 JSONL with model_reply.")
    p.add_argument("--prompts", default=None, help="Path to prompts JSON. Default: from config.")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_path = Path(args.prompts) if args.prompts else Path(config.prompts_path)
    prompts = load_prompts(str(prompts_path))

    prompts_hash = compute_sha256(prompts_path)
    config_hash = compute_sha256(args.config)
    input_hash = compute_sha256(input_path)

    # Load all rows from Stage 1
    all_rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    # Separate rows: pass-through vs needs judging
    # Pass-through: already scored (harm_1_10 not None) OR generation failed
    passthrough_rows = [
        r for r in all_rows
        if r.get("error_stage") == "generate" or r.get("harm_1_10") is not None
    ]
    pending_rows = [
        r for r in all_rows
        if r.get("error_stage") != "generate" and r.get("harm_1_10") is None
    ]

    judge_run_id = _build_run_id()
    print(f"Judge run ID: {judge_run_id}")
    print(f"Input rows: {len(all_rows)} | Pass-through: {len(passthrough_rows)} | To judge: {len(pending_rows)}")
    print(f"Workers: {args.workers} | Output: {output_path}")

    # Write pass-through rows first
    output_path.write_text("", encoding="utf-8")
    with output_path.open("a", encoding="utf-8") as fh:
        for row in passthrough_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not pending_rows:
        print("Nothing to judge.")
        return 0

    write_lock = threading.Lock()
    done = 0
    error_rows: list[str] = []

    def worker(row: dict[str, Any]) -> tuple[str, dict | Exception]:
        key = f"{row.get('dialogue_id')}/{row.get('condition')}/turn{row.get('turn_index')}"
        try:
            judged = _judge_row(
                row,
                prompts=prompts,
                llm4_config=config.llm4,
                config=config,
                judge_run_id=judge_run_id,
            )
            with write_lock:
                with output_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(judged, ensure_ascii=False) + "\n")
            return key, judged
        except Exception as exc:  # noqa: BLE001
            return key, exc

    total = len(pending_rows)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, row): row for row in pending_rows}
        for future in as_completed(futures):
            key, result = future.result()
            done += 1
            if isinstance(result, Exception):
                error_rows.append(key)
                print(f"[FAILED] {key}: {result}")
            else:
                err = result.get("error_stage")
                print(
                    f"[{done}/{total}] {key}"
                    + (f" ERROR:{err}" if err else "")
                )

    print(f"\n{'='*60}")
    print(f"Judge run ID: {judge_run_id}")
    print(f"完成: {done - len(error_rows)}/{total} 行评判成功写入 {output_path}")
    if error_rows:
        print(f"失败 rows: {error_rows[:20]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
