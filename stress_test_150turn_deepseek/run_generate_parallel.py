"""Stage 1: Generate model replies (LLM3) in parallel across dialogue×condition combos.

Outputs a JSONL with model_reply populated and all score fields set to null.
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

from backend.input_loader import compute_sha256, load_dialogues, load_prompts  # noqa: E402
from backend.llm_clients import OpenAICompatibleChatClient, detect_refusal  # noqa: E402
from backend.runtime_config import load_config  # noqa: E402

CONDITIONS = ("default", "cynical", "distant")


def _build_run_id() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"gen_{ts}_{uuid.uuid4().hex[:6]}"


def _timestamp_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _truncate_history(
    messages: list[dict[str, str]],
    policy: str,
    max_history_messages: int,
    max_context_chars: int,
) -> tuple[list[dict[str, str]], bool]:
    if policy == "token_budget":
        kept = [messages[0]]
        used_chars = len(messages[0].get("content", ""))
        tail: list[dict[str, str]] = []
        for item in reversed(messages[1:]):
            item_chars = len(item.get("content", ""))
            if used_chars + item_chars > max_context_chars:
                break
            tail.append(item)
            used_chars += item_chars
        tail.reverse()
        kept.extend(tail)
        return kept, len(kept) != len(messages)
    if len(messages) <= max_history_messages:
        return messages, False
    return [messages[0], *messages[-(max_history_messages - 1):]], True


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


def _run_combo(
    *,
    dialogue,
    condition: str,
    config,
    prompts,
    llm3_config,
    max_turns: int,
    run_id: str,
    hashes: dict[str, str],
) -> list[dict[str, Any]]:
    client = OpenAICompatibleChatClient(llm3_config)
    system_prompt = prompts.conditions[condition]

    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    rows: list[dict[str, Any]] = []
    n_turns = min(len(dialogue.turns), max_turns)

    for turn_index in range(1, n_turns + 1):
        user_text = dialogue.turns[turn_index - 1].text
        history_with_turn = [*history, {"role": "user", "content": user_text}]
        prompt_messages, context_truncated = _truncate_history(
            history_with_turn,
            policy=config.truncation_policy,
            max_history_messages=config.max_history_messages,
            max_context_chars=config.max_context_chars,
        )

        model_reply = ""
        gen_latency_ms: int | None = None
        error_stage: str | None = None
        error_message: str | None = None
        refusal_detected = False

        try:
            result = _call_with_retry(
                lambda: client.chat(prompt_messages, config.timeout_seconds),
                retries=config.retries,
            )
            model_reply = result.text
            gen_latency_ms = result.latency_ms
            refusal_detected = detect_refusal(model_reply)
        except Exception as exc:  # noqa: BLE001
            error_stage = "generate"
            error_message = str(exc)[:500]

        # Update history only on success
        if error_stage is None:
            history = [*history_with_turn, {"role": "assistant", "content": model_reply}]
        else:
            history = list(history)  # keep history without this failed turn

        rows.append({
            "run_id": run_id,
            "dialogue_id": dialogue.dialogue_id,
            "domain": dialogue.domain,
            "condition": condition,
            "turn_index": turn_index,
            "timestamp_utc": _timestamp_utc(),
            "user_text": user_text,
            "model_reply": model_reply,
            # Score fields — all null at this stage
            "harm_1_10": None,
            "negative_emotion_1_10": None,
            "inappropriate_1_10": None,
            "empathic_language_1_10": None,
            "anthro_q1": None,
            "anthro_q2": None,
            "anthro_q3": None,
            "anthro_q4": None,
            "anthro_q5": None,
            "judge_latency_ms": None,
            "judge_raw": None,
            # Diagnostics
            "error_stage": error_stage,
            "error_message": error_message,
            "refusal_detected": refusal_detected,
            "context_truncated": context_truncated,
            "gen_latency_ms": gen_latency_ms,
            # Metadata
            "llm3_model": llm3_config.model,
            "llm3_provider": llm3_config.provider,
            "llm3_params": {
                "temperature": llm3_config.temperature,
                "top_p": llm3_config.top_p,
            },
            "prompts_hash": hashes["prompts"],
            "config_hash": hashes["config"],
            "dialogues_hash": hashes["dialogues"],
        })

        print(
            f"[{run_id}] {dialogue.dialogue_id}/{condition} turn {turn_index}/{n_turns}"
            + (f" ERROR:{error_stage}" if error_stage else "")
        )

    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: generate model replies in parallel.")
    p.add_argument("--config", required=True)
    p.add_argument("--dialogues", required=True, help="Path to dialogues JSONL.")
    p.add_argument("--prompts", default=None, help="Path to prompts JSON. Default: from config.")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max_turns", type=int, default=None, help="Override max_turns from config.")
    p.add_argument(
        "--conditions", nargs="*", default=list(CONDITIONS),
        help="Conditions to run. Default: default cynical distant",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    max_turns = args.max_turns or config.max_turns

    dialogues_path = Path(args.dialogues)
    prompts_path = Path(args.prompts) if args.prompts else Path(config.prompts_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dialogues = load_dialogues(str(dialogues_path), compatibility_mode=True)
    prompts = load_prompts(str(prompts_path))

    hashes = {
        "prompts": compute_sha256(prompts_path),
        "config": compute_sha256(args.config),
        "dialogues": compute_sha256(dialogues_path),
    }

    run_id = _build_run_id()
    conditions = args.conditions
    combos = [(d, c) for d in dialogues for c in conditions]
    total_combos = len(combos)
    total_rows = total_combos * max_turns

    # Resume: count already-written turns per (dialogue_id, condition)
    done_turns: dict[tuple[str, str], int] = {}
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = (r["dialogue_id"], r["condition"])
                    done_turns[key] = done_turns.get(key, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    continue
        skipped = sum(1 for (d, c) in combos if done_turns.get((d.dialogue_id, c), 0) >= max_turns)
        if skipped:
            print(f"Resume: {skipped} combos already complete, skipping.")

    combos = [(d, c) for d, c in combos if done_turns.get((d.dialogue_id, c), 0) < max_turns]
    pending_combos = len(combos)

    print(f"Run ID: {run_id}")
    print(f"Dialogues: {len(dialogues)} | Conditions: {conditions} | Max turns: {max_turns}")
    print(f"Total combos: {total_combos} | Pending: {pending_combos} | Est. rows: {pending_combos * max_turns} | Workers: {args.workers}")
    print(f"Output: {output_path}")

    if not combos:
        print("All combos already complete.")
        return 0

    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")

    write_lock = threading.Lock()
    done_combos = 0
    error_combos: list[str] = []

    def worker(dialogue, condition: str) -> tuple[str, list | Exception]:
        try:
            rows = _run_combo(
                dialogue=dialogue,
                condition=condition,
                config=config,
                prompts=prompts,
                llm3_config=config.llm3,
                max_turns=max_turns,
                run_id=run_id,
                hashes=hashes,
            )
            with write_lock:
                with output_path.open("a", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            return f"{dialogue.dialogue_id}/{condition}", rows
        except Exception as exc:  # noqa: BLE001
            return f"{dialogue.dialogue_id}/{condition}", exc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, d, c): (d.dialogue_id, c) for d, c in combos}
        for future in as_completed(futures):
            dlg_id, cond = futures[future]
            key, result = future.result()
            done_combos += 1
            if isinstance(result, Exception):
                error_combos.append(key)
                print(f"[COMBO FAILED] {key}: {result}")
            else:
                gen_errors = sum(1 for r in result if r.get("error_stage") == "generate")
                print(f"[{done_combos}/{pending_combos}] {key} done "
                      f"(rows={len(result)}, gen_errors={gen_errors})")

    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"完成: {done_combos - len(error_combos)}/{pending_combos} combos 成功写入 {output_path}")
    if error_combos:
        print(f"失败 combo: {error_combos}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
