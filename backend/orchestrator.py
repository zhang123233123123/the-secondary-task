from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .input_loader import Dialogue, PromptsBundle, compute_sha256, load_dialogues, load_prompts
from .llm_clients import (
    LLMError,
    OpenAICompatibleChatClient,
    detect_refusal,
    parse_judge_json,
)
from .output_writer import JsonlWriter, write_summary
from .report_writer import write_report
from .resume import load_resume_state
from .runs_index import write_runs_index
from .schema_validation import validate_with_simple_schema
from .runtime_config import RuntimeConfig

CONDITIONS_ORDER = ("default", "evil", "distant")


@dataclass
class RunStats:
    expected_rows: int = 0
    actual_rows: int = 0
    generate_errors: int = 0
    judge_errors: int = 0
    judge_parse_errors: int = 0
    refusal_count: int = 0
    truncated_count: int = 0


def _build_run_id() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{uuid.uuid4().hex[:6]}"


def _truncate_history(
    messages: list[dict[str, str]],
    policy: str,
    max_history_messages: int,
    max_context_chars: int,
) -> tuple[list[dict[str, str]], bool]:
    if policy == "token_budget":
        # Approximate token budget using character budget to keep implementation dependency-free.
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
    # Keep system prompt and last N-1 exchanges/messages.
    return [messages[0], *messages[-(max_history_messages - 1) :]], True


def _timestamp_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _judge_messages(prompts: PromptsBundle, model_reply: str) -> list[dict[str, str]]:
    user_content = (
        f"Rubric:\n{prompts.judge_rubric}\n\n"
        f"Model reply to evaluate:\n{model_reply}\n\n"
        "Return only JSON with keys harm, emotion, anthro."
    )
    return [
        {"role": "system", "content": prompts.judge_system},
        {"role": "user", "content": user_content},
    ]


def _call_with_retry(
    call,
    retries: int,
    backoff_seconds: float = 0.5,
):
    attempts = retries + 1
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if index == attempts - 1:
                break
            time.sleep(backoff_seconds * (2**index))
    if last_error is None:
        raise RuntimeError("Unexpected retry state")
    raise last_error


def run_experiment(
    config: RuntimeConfig,
    config_path: str,
    dry_run: bool,
    run_id: str | None = None,
) -> dict[str, Any]:
    actual_run_id = run_id or _build_run_id()

    dialogues: list[Dialogue] = load_dialogues(
        config.dialogues_path,
        compatibility_mode=config.input_compatibility_mode,
    )
    prompts = load_prompts(config.prompts_path)
    if dry_run:
        dialogues = dialogues[:5]

    hashes = {
        "prompts_hash": compute_sha256(config.prompts_path),
        "config_hash": compute_sha256(config_path),
        "dialogues_hash": compute_sha256(config.dialogues_path),
    }
    stats = RunStats()

    writer = JsonlWriter(config.output_dir, actual_run_id, config.flush_policy)
    resume_state = load_resume_state(writer.results_path)
    llm3_client = OpenAICompatibleChatClient(config.llm3)
    llm4_client = OpenAICompatibleChatClient(config.llm4)
    abort_reason: str | None = None

    try:
        for dialogue in dialogues:
            for condition in CONDITIONS_ORDER:
                system_prompt = prompts.conditions[condition]
                combo_key = (dialogue.dialogue_id, condition)
                combo_state = resume_state.combo_states.get(combo_key)
                if (
                    config.resume_strategy == "skip"
                    and combo_state is not None
                    and combo_state.processed_turns
                ):
                    continue

                history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
                processed_turns: set[int] = set()
                if config.resume_strategy == "reconstruct" and combo_state is not None:
                    history.extend(combo_state.history)
                    processed_turns = set(combo_state.processed_turns)

                max_turns = min(len(dialogue.turns), config.max_turns)
                stats.expected_rows += max_turns

                for turn_index in range(1, max_turns + 1):
                    if turn_index in processed_turns:
                        continue

                    user_text = dialogue.turns[turn_index - 1].text
                    history_with_current_turn = [*history, {"role": "user", "content": user_text}]
                    prompt_messages, context_truncated = _truncate_history(
                        history_with_current_turn,
                        policy=config.truncation_policy,
                        max_history_messages=config.max_history_messages,
                        max_context_chars=config.max_context_chars,
                    )

                    model_reply = ""
                    gen_latency_ms: int | None = None
                    judge_latency_ms: int | None = None
                    harm: int | None = None
                    emotion: int | None = None
                    anthro: int | None = None
                    error_stage: str | None = None
                    error_message: str | None = None
                    judge_raw: str | None = None
                    refusal_detected = False

                    try:
                        generate_result = _call_with_retry(
                            lambda: llm3_client.chat(prompt_messages, config.timeout_seconds),
                            retries=config.retries,
                        )
                        model_reply = generate_result.text
                        gen_latency_ms = generate_result.latency_ms
                        refusal_detected = detect_refusal(model_reply)
                    except Exception as exc:  # noqa: BLE001
                        error_stage = "generate"
                        error_message = str(exc)
                        stats.generate_errors += 1

                    if error_stage is None:
                        try:
                            judge_result = _call_with_retry(
                                lambda: llm4_client.chat(
                                    _judge_messages(prompts, model_reply),
                                    config.timeout_seconds,
                                ),
                                retries=config.retries,
                            )
                            judge_latency_ms = judge_result.latency_ms
                            judge_raw = judge_result.text
                            parsed = parse_judge_json(judge_result.text)
                            validate_with_simple_schema(parsed, prompts.judge_schema)
                            harm = int(parsed.get("harm"))
                            emotion = int(parsed.get("emotion"))
                            anthro = int(parsed.get("anthro"))
                        except ValueError as exc:
                            error_stage = "judge_parse"
                            error_message = str(exc)
                            stats.judge_parse_errors += 1
                        except (LLMError, RuntimeError) as exc:
                            error_stage = "judge"
                            error_message = str(exc)
                            stats.judge_errors += 1
                        except Exception as exc:  # noqa: BLE001
                            error_stage = "judge"
                            error_message = str(exc)
                            stats.judge_errors += 1

                    row = {
                        "run_id": actual_run_id,
                        "dialogue_id": dialogue.dialogue_id,
                        "domain": dialogue.domain,
                        "condition": condition,
                        "turn_index": turn_index,
                        "timestamp_utc": _timestamp_utc(),
                        "user_text": user_text,
                        "model_reply": model_reply,
                        "harm": harm,
                        "emotion": emotion,
                        "anthro": anthro,
                        "error_stage": error_stage,
                        "error_message": error_message,
                        "refusal_detected": refusal_detected,
                        "context_truncated": context_truncated,
                        "gen_latency_ms": gen_latency_ms,
                        "judge_latency_ms": judge_latency_ms,
                        "llm3_model": config.llm3.model,
                        "llm3_params": {
                            "temperature": config.llm3.temperature,
                            "top_p": config.llm3.top_p,
                            "seed": config.llm3.seed,
                        },
                        "resume_strategy": config.resume_strategy,
                        "input_schema_variant": dialogue.input_schema_variant,
                        "judge_raw": judge_raw,
                        **hashes,
                    }
                    writer.write(row)
                    stats.actual_rows += 1

                    if context_truncated:
                        stats.truncated_count += 1
                    if refusal_detected:
                        stats.refusal_count += 1

                    # Keep history clean from failed generate turns.
                    if error_stage != "generate":
                        history = history_with_current_turn
                        history.append({"role": "assistant", "content": model_reply})

                    if error_stage == "generate" and config.abort_on_error:
                        raise RuntimeError(error_message or "Generation failed")
                    if error_stage in {"judge", "judge_parse"} and config.abort_on_error:
                        raise RuntimeError(error_message or "Judge failed")
    except RuntimeError as exc:
        abort_reason = str(exc)

    finally:
        writer.close()

    final_resume_state = load_resume_state(writer.results_path)
    total_rows = final_resume_state.existing_rows
    error_rows = final_resume_state.error_rows
    summary = {
        "run_id": actual_run_id,
        "expected_rows": stats.expected_rows,
        "actual_rows": total_rows,
        "new_rows_written": writer.rows_written,
        "error_rows": error_rows,
        "error_rate": (error_rows / total_rows) if total_rows else 0.0,
        "generate_errors": final_resume_state.generate_errors,
        "judge_errors": final_resume_state.judge_errors,
        "judge_parse_errors": final_resume_state.judge_parse_errors,
        "new_generate_errors": stats.generate_errors,
        "new_judge_errors": stats.judge_errors,
        "new_judge_parse_errors": stats.judge_parse_errors,
        "truncated_count": final_resume_state.truncated_count,
        "refusal_count": final_resume_state.refusal_count,
        "refusal_rate": (final_resume_state.refusal_count / total_rows) if total_rows else 0.0,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
        **hashes,
    }
    summary_path = write_summary(config.output_dir, actual_run_id, summary)
    report_path = write_report(
        config.output_dir,
        actual_run_id,
        summary,
        dry_run=dry_run,
        results_path=writer.results_path,
    )
    write_runs_index(config.output_dir)
    if abort_reason is not None:
        raise RuntimeError(abort_reason)
    return {
        "run_id": actual_run_id,
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "results_path": str(writer.results_path),
        "summary": summary,
    }
